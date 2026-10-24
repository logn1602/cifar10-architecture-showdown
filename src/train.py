"""The ONE shared training loop (spec Section 4).

Every number in the showdown -- the LR sweep, E1, E2, E3 and E4 -- is produced
by :func:`train`. That is the fairness protocol in code form: if two models are
compared, the only thing that differs between their runs is the architecture
(and the per-model learning rate the sweep picked).

Design notes, in rough order of how much they matter:

* **Everything is counted in optimizer steps.** An epoch-defined run is just
  ``epochs * steps_per_epoch`` steps evaluated once per epoch. A step-defined
  run (E2, ``max_steps``) is evaluated every ``eval_every_steps`` -- by default
  the length of one *full-data* epoch (351 steps), so every point on the
  data-hunger curve gets the same number of validation passes, i.e. the same
  number of chances to pick a lucky checkpoint. Validating once per *subset*
  epoch instead would give the 5% run ~20x more selection opportunities than
  the 100% run, and quietly inflate the small-data end of the curve.

* **The LR schedule is a pure function of the step** (:func:`lr_at`), set on
  the optimizer by hand every step. No scheduler object means no scheduler
  state to save, and a resumed run lands on exactly the LR it would have had.

* **The test set is touched exactly once**, after training, on the best-val
  checkpoint (spec Section 3). Runs with ``evaluate_test=False`` (the LR sweep)
  never touch it at all.

* **Divergence stops the run and raises** :class:`DivergenceError`. Nothing is
  retried or patched -- spec Section 3: "log it, stop that run, and tell me".

* **Resume** restarts from ``last.pt`` at the last validation boundary. The
  shuffle order of every data epoch is re-derived from ``(seed, epoch)``, so a
  resumed run sees the same batches in the same order as an uninterrupted one.
  The main process's RNG state (dropout, drop path, Mixup/CutMix) is saved
  and restored too, so with ``num_workers=0`` a resumed run is bit-identical
  to an uninterrupted one -- the smoke test checks exactly that. With worker
  processes, per-worker augmentation randomness is *not* reproduced: it lives
  in other processes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:  # works for `python -m src.train` and `import src.train`
    from src.data import (
        CIFAR10_MEAN, CIFAR10_STD, CLASSES, NUM_TRAIN_TOTAL, MixupCutmix, build_dataloaders,
    )
    from src.evaluate import autocast_context, evaluate
    from src.models import MODEL_META, build_model
    from src.profile import (
        count_params, measure_inference, peak_memory_mb, reset_peak_memory, try_count_flops,
    )
    from src.utils import PROJECT_ROOT, ensure_dir, get_logger, human_time, load_json, save_json, set_seed
except ImportError:  # works for `python src/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import (  # noqa: E402
        CIFAR10_MEAN, CIFAR10_STD, CLASSES, NUM_TRAIN_TOTAL, MixupCutmix, build_dataloaders,
    )
    from src.evaluate import autocast_context, evaluate  # noqa: E402
    from src.models import MODEL_META, build_model  # noqa: E402
    from src.profile import (  # noqa: E402
        count_params, measure_inference, peak_memory_mb, reset_peak_memory, try_count_flops,
    )
    from src.utils import (  # noqa: E402
        PROJECT_ROOT, ensure_dir, get_logger, human_time, load_json, save_json, set_seed,
    )

__all__ = ["train", "DivergenceError", "lr_at", "build_param_groups", "pick_amp_dtype",
           "make_grad_scaler", "build_any_model"]

HISTORY_FIELDS = [
    "epoch", "step", "data_epoch", "train_loss", "train_acc", "val_loss", "val_acc",
    "lr", "epoch_time_sec", "train_imgs_per_sec",
]

# How often (in steps) the loss is pulled to the host to check for NaN/inf.
# Each check is a device sync; every 50 steps is free in practice and stops a
# diverged run within seconds rather than at the end of the epoch.
_NAN_CHECK_EVERY = 50


class DivergenceError(RuntimeError):
    """Training loss became NaN/inf. The run is stopped, never patched."""


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def lr_at(step: int, base_lr: float, total_steps: int, warmup_steps: int, min_lr: float) -> float:
    """Linear warmup to ``base_lr``, then cosine decay to ``min_lr``.

    ``step`` is the 0-based index of the optimizer step about to be taken.
    Warmup starts at ``base_lr / warmup_steps`` rather than 0, so step 0 is not
    a wasted step.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_param_groups(model: nn.Module, weight_decay: float,
                       no_decay_keywords: list[str] | tuple[str, ...] = ()) -> list[dict]:
    """Split parameters into decay / no-decay groups (spec Section 3).

    A parameter is exempt from weight decay when any of these hold:
      * it is 1-D -- biases and every norm layer's affine params;
      * the model names it in ``no_weight_decay()`` (ViT: pos_embed, cls_token;
        ConvNeXt: LayerScale gamma);
      * its name contains one of ``no_decay_keywords`` from base.yaml.
    The first rule alone catches nearly everything; the other two make the
    intent explicit and cover the 3-D positional embedding and cls token.
    """
    explicit: set[str] = set()
    if hasattr(model, "no_weight_decay"):
        explicit = set(model.no_weight_decay())

    decay, no_decay = [], []
    decay_names, no_decay_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        leaf = name.split(".")[-1]
        exempt = (
            p.ndim <= 1
            or name in explicit or leaf in explicit
            or any(k in name for k in no_decay_keywords)
        )
        (no_decay if exempt else decay).append(p)
        (no_decay_names if exempt else decay_names).append(name)

    return [
        {"params": decay, "weight_decay": weight_decay, "names": decay_names},
        {"params": no_decay, "weight_decay": 0.0, "names": no_decay_names},
    ]


def pick_amp_dtype(device: torch.device, amp_cfg: dict) -> torch.dtype | None:
    """bf16 where the hardware has it natively, else fp16 (+ GradScaler).

    ``torch.cuda.is_bf16_supported()`` returns True on a T4 in recent PyTorch
    because bf16 can be *emulated* there -- several times slower than fp16.
    Native bf16 needs compute capability >= 8.0 (Ampere), so that is checked
    directly. On CPU, AMP is off: CPU autocast is a debugging path here (the
    smoke test), not a training one, and fp32 keeps it simple.
    """
    if not amp_cfg.get("enabled", True) or device.type != "cuda":
        return None
    want = str(amp_cfg.get("dtype", "bf16")).lower()
    major, _ = torch.cuda.get_device_capability(device)
    if want == "bf16" and major >= 8:
        return torch.bfloat16
    return torch.float16


def make_grad_scaler(enabled: bool):
    """A CUDA ``GradScaler`` on any PyTorch from 2.0 on.

    The device-generic ``torch.amp.GradScaler("cuda", ...)`` only exists from
    PyTorch 2.3; Kaggle's image ships 2.0, where the same class lives at
    ``torch.cuda.amp.GradScaler``. Same behaviour, same ``state_dict`` format.
    The new spelling is preferred when present because newer releases
    deprecate the old one (a FutureWarning on every run).
    """
    scaler_cls = getattr(torch.amp, "GradScaler", None)
    if scaler_cls is not None:
        return scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_any_model(model_name: str, model_cfg: dict, num_classes: int = 10
                    ) -> tuple[nn.Module, dict]:
    """Build one of the six showdown models, or a timm model for E4.

    E4 names its models ``timm/<timm-name>``; everything else goes through the
    hand-written registry. Returns ``(model, info)`` where ``info`` carries
    what the loop needs to know about the model: its family (for
    ``channels_last``), the input resolution, and -- for pretrained models --
    the normalisation the weights were trained with.
    """
    if model_name.startswith("timm/"):
        import timm

        timm_name = model_name.split("/", 1)[1]
        model = timm.create_model(timm_name, pretrained=bool(model_cfg.get("pretrained", True)),
                                  num_classes=num_classes)
        pcfg = getattr(model, "pretrained_cfg", {}) or {}
        family = "convolutional" if "resnet" in timm_name else "transformer"
        return model, {
            "family": family,
            "image_size": int(model_cfg.get("image_size", 224)),
            "mean": tuple(pcfg.get("mean", (0.485, 0.456, 0.406))),
            "std": tuple(pcfg.get("std", (0.229, 0.224, 0.225))),
            "pretrained": bool(model_cfg.get("pretrained", True)),
        }

    model = build_model(model_name, model_cfg.get("model", model_cfg), num_classes=num_classes)
    return model, {"family": MODEL_META[model_name]["family"], "image_size": 32,
                   "pretrained": False}


def make_input_fn(info: dict, data_mean, data_std, device: torch.device
                  ) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """On-GPU input adapter for models that do not take 32x32 CIFAR input.

    E4 (spec 12.2 tip): upsample 32 -> 224 *after* the batch is on the GPU --
    doing it in the CPU DataLoader would make the CPU the bottleneck. The batch
    is also re-normalised from CIFAR statistics to the ImageNet statistics the
    pretrained weights expect.
    """
    if info["image_size"] == 32 and "mean" not in info:
        return None
    size = info["image_size"]
    d_mean = torch.tensor(data_mean, device=device).view(1, 3, 1, 1)
    d_std = torch.tensor(data_std, device=device).view(1, 3, 1, 1)
    t_mean = torch.tensor(info.get("mean", data_mean), device=device).view(1, 3, 1, 1)
    t_std = torch.tensor(info.get("std", data_std), device=device).view(1, 3, 1, 1)

    def fn(x: torch.Tensor) -> torch.Tensor:
        x = x * d_std + d_mean                                         # back to [0, 1]
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return (x - t_mean) / t_std                                    # (B, 3, 224, 224)

    return fn


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------


def _write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in HISTORY_FIELDS})


def _env_info(device: torch.device) -> dict:
    info = {"torch": str(torch.__version__), "python": platform.python_version(),
            "device": str(device), "gpu": None}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def _rng_state() -> dict:
    """Every global RNG the main process draws from (dropout, drop path,
    Mixup/CutMix, and augmentation when ``num_workers == 0``)."""
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu())
    if "cuda" in st and torch.cuda.is_available() and             len(st["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda"]])


def _cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def train(
    model_name: str,
    model_cfg: dict,
    train_cfg: dict,
    data_cfg: dict,
    seed: int,
    out_dir: str | Path,
    *,
    experiment: str = "debug",
    lr: float | None = None,
    epochs: int | None = None,
    max_steps: int | None = None,
    eval_every_steps: int | None = None,
    warmup_fraction: float | None = None,
    fraction: float = 1.0,
    augmentation: str = "standard",
    aug_cfg: dict | None = None,
    batch_size: int | None = None,
    save_checkpoints: bool = True,
    keep_last: bool = True,
    evaluate_test: bool = True,
    resume: bool = False,
    tb_dir: str | Path | None = None,
    num_workers: int | None = None,
    device: str | torch.device | None = None,
    ece_bins: int = 15,
    stop_after_epochs: int | None = None,
    max_eval_batches: int | None = None,
    extra_metrics: dict | None = None,
) -> dict:
    """Train one model under the fairness protocol and return its metrics.

    Args:
        model_name: Registry key (``"vit"``) or ``"timm/<name>"`` for E4.
        model_cfg: The model's YAML (``{"model": {...}}``) or its ``model:``
            block; for timm models ``{"pretrained": True, "image_size": 224}``.
        train_cfg: The ``train:`` block of base.yaml (+ any mode override).
        data_cfg: The ``data:`` block of base.yaml.
        seed: Run seed (0, 1, 2).
        out_dir: ``results/runs/<exp>/<model>/<seed>`` or similar.
        lr: Peak learning rate. Falls back to ``train_cfg["lr"]``.
        epochs: Epoch budget. Ignored when ``max_steps`` is given.
        max_steps: Step budget (E2). Overrides ``epochs``.
        eval_every_steps: Validation cadence for step-defined runs. Defaults to
            the steps in one full-data epoch (see module docstring).
        warmup_fraction: Warmup as a fraction of total steps. When None,
            epoch-defined runs use ``schedule.warmup_epochs`` and step-defined
            runs use ``schedule.warmup_fraction``.
        fraction: Training-set fraction (E2). Val and test are never subset.
        augmentation: ``"standard"`` or ``"strong"`` (E3).
        save_checkpoints: False for the LR sweep (spec 12.5: no checkpoints).
        keep_last: When False, ``last.pt`` is deleted once the run completes.
        evaluate_test: False for the LR sweep -- the test set is not touched.
        resume: Continue from ``out_dir/last.pt`` if present; return the saved
            metrics immediately if the run already completed.
        stop_after_epochs: Testing hook: stop (as if interrupted) after this
            many validation periods, without finishing the run.
        max_eval_batches: Smoke-test-only cap on validation batches.

    Returns:
        The metrics dict that is also written to ``out_dir/metrics.json``.

    Raises:
        DivergenceError: the training loss became NaN/inf.
    """
    out = ensure_dir(out_dir)
    metrics_path = out / "metrics.json"
    last_path, best_path = out / "last.pt", out / "best.pt"
    hist_path = out / "history.csv"

    if resume and metrics_path.exists():
        done = load_json(metrics_path)
        if done.get("status") == "completed":
            return done

    log = get_logger(f"train.{experiment}.{model_name}.{seed}".replace("/", "_"),
                     log_file=out / "train.log")

    # --- device, seed, precision -------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    set_seed(seed)
    amp_dtype = pick_amp_dtype(device, train_cfg.get("amp", {}))
    use_scaler = amp_dtype == torch.float16

    base_lr = float(lr if lr is not None else train_cfg.get("lr") or float("nan"))
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError(f"{model_name}: no valid learning rate (got {lr!r}); "
                         "pass lr= or run the LR sweep first")

    bs = int(batch_size or train_cfg["batch_size"])
    num_classes = int(data_cfg.get("num_classes", 10))

    # --- data ---------------------------------------------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        data_cfg, fraction=fraction, augmentation=augmentation, batch_size=bs,
        num_workers=num_workers, seed=seed, aug_cfg=aug_cfg,
        eval_batch_size=max(bs, 256),
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError(f"train set too small for batch size {bs} with drop_last")
    # Steps in one *full-data* epoch (45,000 // 128 = 351): the validation
    # cadence for step-defined runs, whatever fraction they train on.
    full_train_size = NUM_TRAIN_TOTAL - int(data_cfg.get("val_per_class", 500)) * num_classes
    full_epoch_steps = full_train_size // bs

    if max_steps is not None:
        total_steps = int(max_steps)
        eval_interval = int(eval_every_steps or full_epoch_steps)
    else:
        n_epochs = int(epochs if epochs is not None else train_cfg["epochs"])
        total_steps = n_epochs * steps_per_epoch
        eval_interval = steps_per_epoch
    eval_interval = max(1, min(eval_interval, total_steps))
    n_periods = math.ceil(total_steps / eval_interval)

    sched = train_cfg.get("schedule", {})
    min_lr = float(sched.get("min_lr", 1e-6))
    if warmup_fraction is not None:
        warmup_steps = int(round(warmup_fraction * total_steps))
    elif max_steps is None and sched.get("warmup_epochs") is not None:
        warmup_steps = int(sched["warmup_epochs"]) * steps_per_epoch
    else:
        warmup_steps = int(round(float(sched.get("warmup_fraction", 0.05)) * total_steps))
    warmup_steps = min(warmup_steps, total_steps)

    # --- model + optimizer --------------------------------------------------
    model, minfo = build_any_model(model_name, model_cfg, num_classes)
    channels_last = bool(train_cfg.get("channels_last_for_conv", True)) and \
        minfo["family"] == "convolutional"
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    input_fn = make_input_fn(minfo, data_cfg.get("mean", CIFAR10_MEAN),
                             data_cfg.get("std", CIFAR10_STD), device)

    groups = build_param_groups(model, float(train_cfg["weight_decay"]),
                                train_cfg.get("no_decay_keywords", []))
    group_names = [g.pop("names") for g in groups]
    optimizer = torch.optim.AdamW(groups, lr=base_lr, betas=tuple(train_cfg.get("betas", (0.9, 0.999))))
    scaler = make_grad_scaler(use_scaler)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    clip = train_cfg.get("grad_clip_norm")

    fwd_model: nn.Module = model
    if train_cfg.get("compile", False):
        fwd_model = torch.compile(model)  # off by default (spec 1.3)

    mixer = None
    if augmentation == "strong":
        s = aug_cfg or {}
        mixer = MixupCutmix(
            mixup_alpha=float(_cfg_get(s, "mixup.alpha", 0.8)),
            cutmix_alpha=float(_cfg_get(s, "cutmix.alpha", 1.0)),
            prob=float(s.get("mixup_cutmix_prob", 0.5)), num_classes=num_classes,
        )

    # --- state (fresh or resumed) -------------------------------------------
    state = {
        "global_step": 0, "period": 0, "best_val_acc": -1.0, "best_val_loss": None,
        "best_epoch": 0, "best_step": 0, "train_time_sec": 0.0, "wall_time_sec": 0.0,
        "history": [], "imgs_seen": 0,
    }
    best_state_mem: dict | None = None
    resumed_from = None
    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if use_scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        state.update(ckpt["state"])
        if ckpt.get("rng"):
            _set_rng_state(ckpt["rng"])
        resumed_from = state["global_step"]
        log.info("Resumed from %s at step %d / %d (period %d)",
                 last_path.name, state["global_step"], total_steps, state["period"])
    else:
        for stale in (hist_path, metrics_path, last_path, best_path, out / "diverged.json"):
            stale.unlink(missing_ok=True)

    tb = None
    if tb_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=str(ensure_dir(tb_dir)))
        except Exception as exc:  # tensorboard is a nice-to-have, never fatal
            log.warning("TensorBoard disabled: %s", exc)

    resolved = {
        "model_name": model_name, "experiment": experiment, "seed": seed, "lr": base_lr,
        "fraction": fraction, "augmentation": augmentation, "batch_size": bs,
        "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
        "eval_interval_steps": eval_interval, "warmup_steps": warmup_steps,
        "min_lr": min_lr, "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else "fp32",
        "grad_scaler": use_scaler, "channels_last": channels_last,
        "save_checkpoints": save_checkpoints, "evaluate_test": evaluate_test,
        "param_groups": {"decay": len(group_names[0]), "no_decay": len(group_names[1]),
                         "no_decay_names_sample": group_names[1][:12]},
        "model_cfg": model_cfg, "train_cfg": train_cfg, "data_cfg": data_cfg,
        "aug_cfg": aug_cfg, "env": _env_info(device),
    }
    with (out / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False, default_flow_style=None)

    log.info(
        "%s | %s seed %d | lr %.2e | %d steps (%d/epoch, eval every %d) | warmup %d | "
        "frac %.2f | aug %s | amp %s | %s",
        experiment, model_name, seed, base_lr, total_steps, steps_per_epoch, eval_interval,
        warmup_steps, fraction, augmentation, resolved["amp_dtype"], device,
    )

    # --- train --------------------------------------------------------------
    reset_peak_memory(device)
    model.train()
    gen = train_loader.generator
    step = state["global_step"]
    data_iter = None
    data_epoch_idx = -1

    def next_batch():
        """Next training batch; re-seeds the shuffle at every data-epoch start."""
        nonlocal data_iter, data_epoch_idx
        e, offset = divmod(step, steps_per_epoch)
        if data_iter is None or e != data_epoch_idx:
            # Shuffle order of data epoch `e` depends only on (seed, e).
            gen.manual_seed(int(seed) * 1_000_003 + e)
            data_iter = iter(train_loader)
            data_epoch_idx = e
            for _ in range(offset):            # only non-zero right after a resume
                next(data_iter)
        return next(data_iter)

    while step < total_steps:
        period_end = min(total_steps, (state["period"] + 1) * eval_interval)
        p_t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        correct_sum = torch.zeros((), device=device)
        n_imgs = 0
        p_steps = 0

        while step < period_end:
            x, y = next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            if input_fn is not None:
                x = input_fn(x)

            cur_lr = lr_at(step, base_lr, total_steps, warmup_steps, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            if mixer is not None:
                x, y_a, y_b, lam = mixer(x, y)
            with autocast_context(device, amp_dtype):
                logits = fwd_model(x)
                if mixer is not None:
                    loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                b = y.numel()
                loss_sum += loss.detach().float() * b
                pred = logits.argmax(1)
                if mixer is not None:  # credit each half of a mixed image
                    correct_sum += lam * (pred == y_a).sum() + (1.0 - lam) * (pred == y_b).sum()
                else:
                    correct_sum += (pred == y).sum()
            n_imgs += b
            p_steps += 1
            step += 1

            if step % _NAN_CHECK_EVERY == 0 and not torch.isfinite(loss_sum):
                break

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_secs = time.perf_counter() - p_t0
        train_loss = float(loss_sum) / max(1, n_imgs)
        if not math.isfinite(train_loss):
            state["history"].append({"epoch": state["period"] + 1, "step": step,
                                     "train_loss": train_loss, "lr": cur_lr})
            _write_history(hist_path, state["history"])
            save_json({"model": model_name, "seed": seed, "experiment": experiment,
                       "status": "diverged", "step": step, "lr": base_lr},
                      out / "diverged.json")
            log.error("DIVERGED at step %d (lr %.2e): train loss %s. Run stopped; "
                      "hyperparameters NOT changed.", step, base_lr, train_loss)
            raise DivergenceError(f"{model_name} seed {seed} diverged at step {step} "
                                  f"(lr {base_lr:.2e}, loss {train_loss})")

        val = evaluate(fwd_model, val_loader, device, amp_dtype, num_classes,
                       input_fn=input_fn, channels_last=channels_last,
                       max_batches=max_eval_batches)
        period_secs = time.perf_counter() - p_t0
        state["period"] += 1
        state["global_step"] = step
        state["train_time_sec"] += train_secs
        state["imgs_seen"] += n_imgs

        row = {
            "epoch": state["period"], "step": step,
            "data_epoch": round(step / steps_per_epoch, 4),
            "train_loss": round(train_loss, 6),
            "train_acc": round(float(correct_sum) / max(1, n_imgs), 6),
            "val_loss": round(val["loss"], 6), "val_acc": round(val["acc"], 6),
            "lr": cur_lr, "epoch_time_sec": round(period_secs, 3),
            "train_imgs_per_sec": round(n_imgs / max(train_secs, 1e-9), 1),
        }
        state["history"].append(row)
        _write_history(hist_path, state["history"])

        improved = val["acc"] > state["best_val_acc"]
        if improved:
            state.update(best_val_acc=val["acc"], best_val_loss=val["loss"],
                         best_epoch=state["period"], best_step=step)
            if save_checkpoints:
                torch.save({"model": model.state_dict(), "epoch": state["period"],
                            "step": step, "val_acc": val["acc"]}, best_path)
            else:
                best_state_mem = copy.deepcopy({k: v.detach().cpu() for k, v in
                                                model.state_dict().items()})

        state["wall_time_sec"] += period_secs
        if save_checkpoints:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if use_scaler else None,
                        "state": state, "rng": _rng_state()}, last_path)

        if tb is not None:
            for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr",
                      "train_imgs_per_sec"):
                tb.add_scalar(k, row[k], step)

        log.info("%s %3d/%d | step %6d | train %.4f / %.4f | val %.4f / %.4f%s | "
                 "lr %.2e | %.1fs | %.0f img/s",
                 "epoch" if max_steps is None else "eval", state["period"], n_periods, step,
                 row["train_loss"], row["train_acc"], row["val_loss"], row["val_acc"],
                 " *" if improved else "", cur_lr, period_secs, row["train_imgs_per_sec"])

        if stop_after_epochs is not None and state["period"] >= stop_after_epochs \
                and step < total_steps:
            log.info("stop_after_epochs=%d reached -- stopping as if interrupted",
                     stop_after_epochs)
            if tb is not None:
                tb.close()
            return {"status": "interrupted", "step": step, "period": state["period"]}

    # --- final: best checkpoint -> test, once --------------------------------
    peak_mb = peak_memory_mb(device)
    if save_checkpoints and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False)["model"])
    elif best_state_mem is not None:
        model.load_state_dict(best_state_mem)

    test_res: dict | None = None
    if evaluate_test:
        test_res = evaluate(fwd_model, test_loader, device, amp_dtype, num_classes,
                            ece_bins=ece_bins, input_fn=input_fn,
                            channels_last=channels_last, full=True,
                            max_batches=max_eval_batches)
        np.savez_compressed(out / "test_outputs.npz", targets=test_res["targets"],
                            preds=test_res["preds"], probs=test_res["probs"])

    flops, flops_error = try_count_flops(model, device, image_size=minfo["image_size"])
    if flops is None:  # a measurement, never a reason to fail a finished run
        log.warning("FLOPs not counted (%s); metrics.json records flops: null", flops_error)
    try:
        infer_ips = measure_inference(model, device, amp_dtype, image_size=32,
                                      batch_size=64 if input_fn is not None else 256,
                                      iters=10 if device.type == "cpu" else 20,
                                      channels_last=channels_last, input_fn=input_fn)
    except Exception as exc:  # never lose a finished run to a benchmark
        log.warning("inference benchmark failed: %s", exc)
        infer_ips = None

    data_epochs = total_steps / steps_per_epoch
    hist = state["history"]
    metrics = {
        "model": model_name, "seed": seed, "experiment": experiment,
        "status": "completed",
        "best_val_acc": state["best_val_acc"], "best_val_loss": state["best_val_loss"],
        "best_epoch": state["best_epoch"], "best_step": state["best_step"],
        "final_val_acc": hist[-1]["val_acc"] if hist else None,
        "test_acc": test_res["acc"] if test_res else None,
        "test_loss": test_res["loss"] if test_res else None,
        "per_class_acc": ({c: float(a) for c, a in zip(CLASSES, test_res["per_class_acc"])}
                          if test_res else {}),
        "confusion_matrix": test_res["confusion_matrix"] if test_res else None,
        "ece": test_res["ece"] if test_res else None,
        "params": count_params(model), "flops": flops, "flops_error": flops_error,
        "train_time_sec": state["train_time_sec"],
        "wall_time_sec": state["wall_time_sec"],
        "sec_per_epoch": state["wall_time_sec"] / max(data_epochs, 1e-9),
        "train_imgs_per_sec": state["imgs_seen"] / max(state["train_time_sec"], 1e-9),
        "infer_imgs_per_sec": infer_ips, "peak_mem_mb": peak_mb,
        "lr": base_lr, "train_fraction": fraction, "augmentation": augmentation,
        "epochs": data_epochs, "total_steps": total_steps, "warmup_steps": warmup_steps,
        "eval_interval_steps": eval_interval, "n_evals": state["period"],
        "batch_size": bs, "amp_dtype": resolved["amp_dtype"],
        "train_size": len(train_loader.dataset), "resumed_from_step": resumed_from,
        "test_touched": bool(test_res), "device": _env_info(device),
        "debug_eval_cap": max_eval_batches,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    save_json(metrics, metrics_path)

    if save_checkpoints and not keep_last:
        last_path.unlink(missing_ok=True)
    if tb is not None:
        tb.close()

    test_str = f"{metrics['test_acc']:.4f}" if test_res else "n/a (not touched)"
    summary = (f"[{experiment}] {model_name:<14} seed {seed} | best val "
               f"{metrics['best_val_acc']:.4f} (ep {metrics['best_epoch']}) | test {test_str} "
               f"| {human_time(state['wall_time_sec'])}")
    log.info(summary)
    print(summary, flush=True)
    return metrics


# ---------------------------------------------------------------------------
# CLI -- handy for one-off runs; experiments go through scripts/run_all.py.
# ---------------------------------------------------------------------------


def _main() -> None:
    from src.utils import load_config

    ap = argparse.ArgumentParser(description="Train one model under the fairness protocol.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--augmentation", default="standard", choices=["standard", "strong"])
    ap.add_argument("--experiment", default="debug")
    ap.add_argument("--out", help="output dir (default results/runs/<exp>/<model>/<seed>)")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config("configs/base.yaml")
    model_cfg = load_config(f"configs/models/{args.model}.yaml")
    out = args.out or (PROJECT_ROOT / "results" / "runs" / args.experiment / args.model / str(args.seed))
    train(args.model, model_cfg, cfg["train"], cfg["data"], args.seed, out,
          experiment=args.experiment, lr=args.lr, epochs=args.epochs, max_steps=args.max_steps,
          fraction=args.fraction, augmentation=args.augmentation,
          aug_cfg=cfg["augmentation"].get(args.augmentation), resume=args.resume,
          evaluate_test=not args.no_test, num_workers=args.num_workers,
          ece_bins=int(_cfg_get(cfg, "eval.ece_bins", 15)))


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    _main()

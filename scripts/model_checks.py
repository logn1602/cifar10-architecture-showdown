"""Phase 2 gate: prove every model is wired correctly before training anything.

Spec 5.7 asks for a forward-shape check, parameter and FLOP counts, and an
overfit test. Three more checks are added here because each catches a bug class
that would otherwise survive until the numbers were already wrong:

* **eval determinism** -- two eval-mode forwards on the same input must be
  bit-identical. Catches dropout or stochastic depth leaking past ``.eval()``,
  which would make "test accuracy" a random variable.
* **gradient reachability** -- every trainable parameter must receive a
  gradient. Catches a layer built but never called, which silently wastes
  capacity and distorts the parameter-count comparison the whole report rests on.
* **timm cross-check** -- spec Section 5 permits ``timm`` purely as a
  correctness reference for the two models not written from scratch for fun.

The overfit test is the real gate: a model that cannot memorise 256 images has
a bug, and no amount of training will reveal which one. Running this on CPU
before committing 8 GPU-hours is the cheapest insurance in the project.

Every check is a hard gate (exit code 1) except FLOP counting, which is a
measurement: when ``FlopCounterMode`` fails on a given PyTorch build, the
reason is printed as a warning and the count is recorded as null.

Run: ``python scripts/model_checks.py``            (~45 min on this CPU)
     ``python scripts/model_checks.py --skip-overfit``   (seconds)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_datasets  # noqa: E402
from src.models import (  # noqa: E402
    MODEL_ORDER, build_model, count_parameters, get_meta,
)
from src.profile import try_count_flops  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir, human_time, load_config, save_json, set_seed,
)

# Spec 5.7: "must reach ~100% train accuracy". 98% leaves room for one or two
# genuinely ambiguous images without waving through a broken model.
OVERFIT_THRESHOLD = 0.98

# Published / spec-stated parameter targets, for the sanity column.
PARAM_TARGETS = {
    "mlp": 3.8e6, "simple_cnn": 1.1e6, "resnet18": 11.2e6,
    "vit": 8.5e6, "mlp_mixer": 9.7e6, "convnext_mini": 8.5e6,
}
PARAM_TOLERANCE = 0.15   # +/-15% of the target


def check_eval_determinism(model: nn.Module, device: torch.device) -> tuple[bool, str]:
    """Two eval-mode forwards must be bit-identical."""
    model.eval()
    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        a, b = model(x), model(x)
    same = torch.equal(a, b)
    delta = (a - b).abs().max().item()
    return same, ("identical" if same else f"max |delta| = {delta:.3e} -- dropout active in eval?")


def check_gradients(model: nn.Module, device: torch.device) -> tuple[bool, str]:
    """Every trainable parameter must get a gradient from one backward pass."""
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    out = model(x)
    nn.functional.cross_entropy(out, y).backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    model.zero_grad(set_to_none=True)
    if missing:
        return False, f"{len(missing)} param(s) with no/non-finite grad: {missing[:4]}"
    return True, f"all {sum(1 for _ in model.parameters())} tensors reached"


def timm_crosscheck(name: str, ours: int) -> str:
    """Compare against timm where a meaningful reference exists."""
    if name != "resnet18":
        return "n/a"
    try:
        import timm
        ref = timm.create_model("resnet18", pretrained=False, num_classes=10)
        ref_n = sum(p.numel() for p in ref.parameters())
    except Exception as exc:
        return f"timm unavailable ({type(exc).__name__})"
    diff = ref_n - ours
    # timm ships the ImageNet stem: 7x7x3x64 = 9,408 weights where our CIFAR
    # stem has 3x3x3x64 = 1,728. The whole difference should be those 7,680.
    verdict = "as expected" if diff == 7680 else "UNEXPECTED"
    return f"timm {ref_n:,} vs ours {ours:,} (diff {diff:,}, {verdict}: ImageNet 7x7 stem)"


# Regularisation knobs that are switched off for the overfit test. Every one of
# these exists to *prevent* a model from fitting its training data, which is
# precisely what this test measures -- leaving them on turns a wiring check into
# a "can it memorise despite being actively stopped from doing so" check.
# ViT/Mixer/ConvNeXt all carry drop_path_rate 0.1, so this is not academic:
# with stochastic depth left on, ViT plateaus at 82%.
REGULARIZATION_KEYS = ("drop_path_rate", "drop_rate", "attn_drop_rate", "dropout")


def strip_regularization(model_cfg: dict) -> tuple[dict, list[str]]:
    """Zero every regularisation knob present in ``model_cfg``."""
    out = dict(model_cfg)
    disabled = []
    for key in REGULARIZATION_KEYS:
        if key in out and out[key]:
            disabled.append(f"{key}={out[key]}")
            out[key] = 0.0
    return out, disabled


def overfit_test(name: str, model_cfg: dict, device: torch.device, data_cfg: dict,
                 n_images: int, steps: int, batch_size: int, lr: float,
                 seed: int = 0, keep_regularization: bool = False,
                 warmup_frac: float = 0.1) -> dict:
    """Memorise ``n_images`` training images in ``steps`` optimizer steps.

    Deliberately *not* run under the fairness protocol: no augmentation (so the
    targets are fixed), no weight decay, no label smoothing, and -- unless
    ``keep_regularization`` -- no dropout or stochastic depth. All five exist to
    oppose memorisation. This is a capacity-and-wiring test, not a
    generalisation test, and the protocol is untouched for real training.
    """
    cfg, disabled = (model_cfg, []) if keep_regularization else strip_regularization(model_cfg)
    set_seed(seed)
    model = build_model(name, cfg)
    train_ds, _, _ = build_datasets(data_cfg, augmentation="none")
    subset = Subset(train_ds, list(range(n_images)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    # Linear warmup over the first `warmup_frac` of steps. Not a tuning knob
    # snuck in to make the test pass -- the protocol itself (spec Section 3)
    # warms up every real run, because transformers are famously unstable when
    # AdamW is applied at full learning rate from step 0. A gate that skips it
    # would be testing a configuration we never actually train with.
    warmup_steps = max(1, int(warmup_frac * steps))

    def lr_at(s: int) -> float:
        """Linear warmup then cosine decay -- the protocol's schedule shape.

        The decay is not cosmetic. With a constant post-warmup LR, ViT reached
        a loss of 0.119 by step 150 and then bounced back up to 0.406 by step
        199: AdamW at 1e-3 keeps taking steps too large to settle into the
        minimum it had already found. Cosine decay is what lets a model that
        has *found* the solution actually land on it, and spec Section 3
        mandates it for every real run.
        """
        if s < warmup_steps:
            return lr * (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(1, steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    t0 = time.perf_counter()
    step = 0
    losses: list[float] = []
    while step < steps:
        for x, y in loader:
            if step >= steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
    elapsed = time.perf_counter() - t0

    # Score in eval mode: dropout and stochastic depth off, BN using running stats.
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(subset, batch_size=batch_size, num_workers=0):
            x, y = x.to(device), y.to(device)
            correct += int((model(x).argmax(1) == y).sum())
            total += y.numel()

    acc = correct / total
    return {
        "accuracy": acc,
        "passed": acc >= OVERFIT_THRESHOLD,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "steps": step,
        "seconds": elapsed,
        "n_images": total,
        "regularization_disabled": disabled,
        "warmup_steps": warmup_steps,
        # Loss at a handful of checkpoints: a model that is merely slow shows a
        # steadily falling curve, where a broken one plateaus or diverges.
        "loss_curve": {str(s): round(losses[min(s, len(losses) - 1)], 4)
                       for s in (0, 10, 25, 50, 100, 150, steps - 1)},
    }


def run(args) -> int:
    cfg = load_config("configs/base.yaml")
    data_cfg = cfg["data"]
    device = torch.device(args.device)
    names = args.models or list(MODEL_ORDER)

    print("=" * 78)
    print("  PHASE 2  |  MODEL CHECKS")
    print("=" * 78)
    print(f"device={device}  overfit: {args.overfit_images} images, {args.overfit_steps} steps, "
          f"batch {args.batch_size}, lr {args.lr}\n")

    results: list[dict] = []
    failures: list[str] = []      # gates: any entry here exits non-zero
    warnings: list[str] = []      # measurements that could not be taken: reported, not gating

    for name in names:
        print("-" * 78)
        print(f"  {name}")
        print("-" * 78)
        meta = get_meta(name)
        model_cfg = load_config(f"configs/models/{name}.yaml")["model"]

        set_seed(0)
        model = build_model(name, model_cfg).to(device)
        row: dict = {"model": name, **meta, "checks": {}}

        def record(label: str, ok: bool, detail: str = "", gate: bool = True) -> None:
            row["checks"][label] = {"passed": bool(ok), "gate": gate, "detail": detail}
            tag = "  ok  " if ok else (" FAIL " if gate else " WARN ")
            print(f"  [{tag}] {label}" + (f"  --  {detail}" if detail else ""))
            if not ok:
                (failures if gate else warnings).append(f"{name}: {label} ({detail})")

        # 1. forward shape
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 32, device=device))
        record("forward (2,3,32,32) -> (2,10)", tuple(out.shape) == (2, 10), str(tuple(out.shape)))

        # 2. parameters
        n_params = count_parameters(model)
        target = PARAM_TARGETS[name]
        within = abs(n_params - target) / target <= PARAM_TOLERANCE
        row["params"] = n_params
        record("parameter count near target", within,
               f"{n_params:,} vs target ~{target:,.0f} ({100*(n_params-target)/target:+.1f}%)")

        # 3. FLOPs -- a measurement, not a gate. FlopCounterMode varies across
        # PyTorch builds (it failed for every model on one Kaggle image), and a
        # FLOP count depends only on the architecture and input size, so a count
        # from any other machine applies. A failure is logged and recorded as
        # null; it never fails this script.
        flops, flops_error = try_count_flops(model, device)
        row["flops"] = flops
        row["flops_error"] = flops_error
        record("FLOPs counted (measurement, not a gate)", flops is not None,
               f"{flops / 1e9:.3f} GFLOPs/image (forward, batch 1)" if flops is not None
               else f"not counted, recorded as null -- {flops_error}",
               gate=False)

        # 4. eval determinism
        ok, detail = check_eval_determinism(model, device)
        record("eval() is deterministic", ok, detail)

        # 5. gradient reachability
        ok, detail = check_gradients(model, device)
        record("every parameter gets a gradient", ok, detail)

        # 6. timm cross-check
        cross = timm_crosscheck(name, n_params)
        row["timm_crosscheck"] = cross
        if cross != "n/a":
            record("timm parameter cross-check", "UNEXPECTED" not in cross, cross)

        # 7. overfit
        if args.skip_overfit:
            row["overfit"] = None
            print("  [ skip ] overfit test (--skip-overfit)")
        else:
            print(f"  [ .... ] overfit test running ...", end="\r", flush=True)
            ov = overfit_test(name, model_cfg, device, data_cfg, args.overfit_images,
                              args.overfit_steps, args.batch_size, args.lr,
                              keep_regularization=args.keep_regularization,
                              warmup_frac=args.warmup_frac)
            row["overfit"] = ov
            reg = f", reg off: {', '.join(ov['regularization_disabled'])}" \
                if ov["regularization_disabled"] else ""
            curve = " ".join(f"{k}:{v}" for k, v in ov["loss_curve"].items())
            record(f"overfits {ov['n_images']} images in {ov['steps']} steps", ov["passed"],
                   f"train acc {ov['accuracy']:.1%} (threshold {OVERFIT_THRESHOLD:.0%}), "
                   f"{human_time(ov['seconds'])}{reg}")
            print(f"           loss by step -> {curve}")

        results.append(row)
        print()

    write_outputs(results, args.out_dir, device, args, failures, warnings)

    print("=" * 78)
    if warnings:
        print(f"  WARNINGS ({len(warnings)}, not gating):")
        for w in warnings:
            print(f"    - {w}")
    if failures:
        print(f"  RESULT: {len(failures)} GATING CHECK(S) FAILED")
        for f in failures:
            print(f"    - {f}")
        print("=" * 78)
        return 1
    n = sum(1 for r in results for c in r["checks"].values() if c["gate"])
    print(f"  RESULT: all {n} gating checks passed across {len(results)} models"
          + (f" ({len(warnings)} warning(s) above, not gating)" if warnings else ""))
    print("=" * 78)
    return 0


def write_outputs(results: list[dict], out_dir: str, device: torch.device, args,
                  failures: list[str], warnings: list[str]) -> None:
    out = ensure_dir(out_dir)
    tables = ensure_dir(out / "tables")
    save_json({"device": str(device), "torch": str(torch.__version__),
               "python": sys.version.split()[0],
               "gates_passed": not failures, "warnings": warnings, "results": results,
               "overfit_config": {"images": args.overfit_images, "steps": args.overfit_steps,
                                  "batch_size": args.batch_size, "lr": args.lr,
                                  "threshold": OVERFIT_THRESHOLD}},
              out / "model_checks.json")

    L = ["# Phase 2 -- Model summary\n",
         "_Generated by `scripts/model_checks.py`. Do not edit by hand._\n",
         "Six architectures, ~35 years apart, all built to land in the same parameter range so the",
         "comparison is about inductive bias rather than capacity.\n",
         "| Model | Year | Params | GFLOPs/img | Inductive bias |",
         "|---|---|---|---|---|"]
    for r in results:
        flops = f"{r['flops'] / 1e9:.3f}" if r.get("flops") is not None else "n/a"
        L.append(f"| **{r['display_name']}** | {r['year']} | {r['params']:,} | {flops} "
                 f"| {r['inductive_bias']} |")
    if any(r.get("flops") is None for r in results):
        L += ["",
              "_n/a: FLOPs could not be counted on this machine "
              f"(torch {torch.__version__}, Python {sys.version.split()[0]}) -- the reason is in "
              "`model_checks.json` under `flops_error`. FLOP counting is a measurement, not a gate: "
              "the count depends only on the architecture and the input size, so a count taken on "
              "any other machine applies unchanged._"]

    L += ["", "## Key idea\n", "| Model | Key idea |", "|---|---|"]
    for r in results:
        L.append(f"| {r['display_name']} | {r['key_idea']} |")

    if any(r.get("overfit") for r in results):
        L += ["", "## Overfit test (spec 5.7)\n",
              f"Can each model memorise {args.overfit_images} training images in "
              f"{args.overfit_steps} optimizer steps? This is a wiring-and-capacity check, so it runs",
              "without augmentation, weight decay, label smoothing, dropout or stochastic depth --",
              "all five exist to *prevent* a model fitting its training data, which is exactly what is",
              "being measured here. The LR schedule is linear warmup then cosine decay, matching the",
              "protocol (spec Section 3). Accuracy is measured in `eval()` mode.\n",
              "> **Why the schedule matters.** ViT initially failed this gate at 82%. Three things were",
              "> wrong with the *test*, not the model: stochastic depth was left on (+7.5pp when",
              "> disabled), there was no LR warmup (+7.8pp), and the post-warmup LR was held constant",
              "> rather than decayed. That last one was visible in the loss curve -- ViT reached a loss",
              "> of 0.119 by step 150 and then bounced back to 0.406 by step 199, the signature of a",
              "> learning rate too large to settle into a minimum it had already found. With cosine",
              "> decay it reaches 100%. No model hyperparameter was changed to achieve this.\n",
              "| Model | Train acc | Loss (first -> final) | Time | Verdict |",
              "|---|---|---|---|---|"]
        for r in results:
            ov = r.get("overfit")
            if not ov:
                continue
            L.append(f"| {r['display_name']} | {ov['accuracy']:.1%} | "
                     f"{ov['first_loss']:.3f} -> {ov['final_loss']:.4f} | "
                     f"{human_time(ov['seconds'])} | {'PASS' if ov['passed'] else '**FAIL**'} |")

    L += ["", "## All checks\n",
          "Every check is a gate -- one failure fails the script -- except FLOP counting, which is "
          "a measurement and only warns.\n",
          "| Model | Check | Status | Detail |", "|---|---|---|---|"]
    for r in results:
        for label, c in r["checks"].items():
            status = "PASS" if c["passed"] else ("**FAIL**" if c.get("gate", True) else "WARN (not a gate)")
            L.append(f"| {r['display_name']} | {label} | {status} | {c['detail']} |")
    L.append("")
    (tables / "model_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {tables / 'model_summary.md'}")
    print(f"wrote {out / 'model_checks.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 model checks (spec 5.7)")
    ap.add_argument("--models", nargs="*", default=None, choices=list(MODEL_ORDER))
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--overfit-images", type=int, default=256)
    ap.add_argument("--overfit-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.1,
                    help="fraction of overfit steps spent on linear LR warmup")
    ap.add_argument("--keep-regularization", action="store_true",
                    help="leave dropout/stochastic depth ON during the overfit test "
                         "(diagnostic: shows how much they slow memorisation)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":   # DataLoader guard (spec 1.3)
    main()

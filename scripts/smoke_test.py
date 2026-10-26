"""Phase 3 gate: every model through the real training loop, end to end.

Spec Section 7, Phase 3: "every model, 2 epochs on 10% data -> no errors, loss
decreasing, all output files written, resume works".

Resume is tested by *actually interrupting* each run: epoch 1 runs with
``stop_after_epochs=1`` (which returns exactly as a killed process would leave
things -- ``last.pt`` and ``history.csv`` on disk, no ``metrics.json``), then a
second call with ``resume=True`` must pick up at epoch 2 and finish. So every
model's 2-epoch run is also a resume test, at no extra cost.

Checks per model:
  * no exception, status "completed"
  * train loss at epoch 2 < epoch 1
  * every expected file exists (history.csv, best.pt, config.yaml,
    metrics.json, test_outputs.npz, TensorBoard events)
  * metrics.json carries every key spec Section 4 lists
  * the resumed run started at the right step, with the LR the schedule
    prescribes for that step (i.e. the schedule did not restart)
  * test accuracy is above chance -- a smoke test, not a benchmark, but a
    model at 10% after two epochs is broken

Plus two loop-level checks that need no data: the LR schedule's shape, and the
decay / no-decay parameter split for every model.

Run: ``python scripts/smoke_test.py``                 (all 6 models)
     ``python scripts/smoke_test.py --models mlp vit``
     ``python scripts/smoke_test.py --quick``          (CPU: caps val/test batches)
     ``python scripts/smoke_test.py --tiny``           (minutes on a CPU; see below)

``--tiny`` is for the pinned-environment rehearsal (``scripts/rehearse.ps1``),
which asks whether the code *runs* on old library versions, not whether the
models learn: 2% of the data, 2 val/test batches, shorter E2/E3 paths. Every
code path still runs and every structural check still gates; the two
learning checks (loss decreasing, accuracy above chance) need more steps than
that to be reliable, so they are reported but do not gate.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.models import MODEL_ORDER, build_model  # noqa: E402
from src.train import build_param_groups, lr_at, train  # noqa: E402
from src.utils import (  # noqa: E402
    PROJECT_ROOT, ensure_dir, human_time, load_config, load_json, resolve_path, save_json,
)

SPEC_METRIC_KEYS = [
    "model", "seed", "experiment", "best_val_acc", "best_epoch", "test_acc", "test_loss",
    "per_class_acc", "ece", "params", "flops", "train_time_sec", "sec_per_epoch",
    "train_imgs_per_sec", "infer_imgs_per_sec", "peak_mem_mb", "lr", "train_fraction",
    "augmentation",
]
EXPECTED_FILES = ["history.csv", "best.pt", "config.yaml", "metrics.json", "test_outputs.npz",
                  "train.log"]

# Smoke-test learning rate. Not the protocol LR (that is what the sweep is for),
# just a value every model is known to train at -- the Phase 2 overfit gate
# used it for all six.
SMOKE_LR = 1.0e-3


def _rel(p: Path) -> str:
    """Repo-relative path for the committed summary (no usernames in the repo)."""
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(p).name


def check_schedule() -> list[tuple[str, bool, str]]:
    """The LR schedule's shape, independent of any model."""
    out = []
    total, warm, base, lo = 1000, 50, 1e-3, 1e-6
    lrs = [lr_at(s, base, total, warm, lo) for s in range(total)]
    out.append(("warmup is linear and starts > 0",
                0 < lrs[0] < lrs[1] < lrs[warm - 1] and abs(lrs[warm - 1] - base) < 1e-12,
                f"lr[0]={lrs[0]:.2e} lr[{warm - 1}]={lrs[warm - 1]:.2e}"))
    out.append(("peak lr reached right after warmup", abs(lrs[warm] - base) < 1e-12,
                f"lr[{warm}]={lrs[warm]:.2e}"))
    mono = all(a >= b for a, b in zip(lrs[warm:], lrs[warm + 1:]))
    out.append(("cosine phase is monotone non-increasing", mono, ""))
    out.append(("final lr ~ min_lr", abs(lr_at(total, base, total, warm, lo) - lo) < 1e-12,
                f"lr[end]={lr_at(total, base, total, warm, lo):.2e}"))
    mid = lr_at(warm + (total - warm) // 2, base, total, warm, lo)
    out.append(("cosine midpoint ~ (base + min) / 2", abs(mid - (base + lo) / 2) / base < 0.01,
                f"lr[mid]={mid:.2e}"))
    return out


def check_exact_resume(cfg: dict, out_root: Path) -> tuple[bool, str]:
    """Interrupted + resumed == uninterrupted, bit for bit.

    MLP (it has both dropout and BatchNorm, so it exercises the RNG and the
    running stats), CPU, ``num_workers=0`` so augmentation also runs off the
    main-process RNG that ``last.pt`` restores. Any state the checkpoint forgot
    -- optimizer moments, an RNG, the schedule position, the data order --
    shows up here as a mismatch.
    """
    import csv

    model_cfg = load_config("configs/models/mlp.yaml")
    common = dict(experiment="smoke", lr=SMOKE_LR, epochs=2, fraction=0.1,
                  warmup_fraction=0.2, num_workers=0, evaluate_test=False,
                  max_eval_batches=4, device="cpu")
    a, b = out_root / "_resume_a", out_root / "_resume_b"
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    train("mlp", model_cfg, cfg["train"], cfg["data"], 0, a, **common)
    train("mlp", model_cfg, cfg["train"], cfg["data"], 0, b, stop_after_epochs=1, **common)
    train("mlp", model_cfg, cfg["train"], cfg["data"], 0, b, resume=True, **common)

    def hist(d: Path) -> list[tuple]:
        with (d / "history.csv").open(encoding="utf-8") as fh:
            return [(r["train_loss"], r["train_acc"], r["val_loss"], r["val_acc"], r["lr"])
                    for r in csv.DictReader(fh)]

    sa = torch.load(a / "last.pt", weights_only=False)["model"]
    sb = torch.load(b / "last.pt", weights_only=False)["model"]
    same_w = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    same_h = hist(a) == hist(b)
    detail = f"history {'identical' if same_h else 'DIFFERS'}, weights {'identical' if same_w else 'DIFFER'}"
    if not same_h:
        detail += f"\n      straight: {hist(a)}\n      resumed:  {hist(b)}"
    return same_h and same_w, detail


def check_extra_paths(cfg: dict, out_root: Path, args) -> list[tuple[str, bool, str]]:
    """Code paths the six 2-epoch runs never touch, run for a handful of steps.

    * **E3 strong augmentation**: RandAugment + RandomErasing in the loader and
      Mixup/CutMix in the loop (mixed targets, lam-weighted loss and accuracy).
    * **E4 pretrained models**: timm construction, the on-GPU 32 -> 224 upsample
      with ImageNet re-normalisation, batch 64 -- for both E4 architectures.
      Built with random weights here (no download); the real run loads the
      pretrained ones through the same ``timm.create_model`` call.
    * **Step-defined runs** (E2): ``max_steps`` with a partial final epoch.
    """
    out: list[tuple[str, bool, str]] = []
    common = dict(experiment="smoke_extra", lr=SMOKE_LR, fraction=args.fraction,
                  warmup_fraction=0.2, num_workers=args.num_workers, max_eval_batches=2,
                  keep_last=False)
    # --tiny: still a partial final epoch for E2, still past one epoch for E3.
    e3_steps, e2_steps = (8, 10) if args.tiny else (20, 40)

    def run(label: str, model: str, model_cfg: dict, **kw) -> None:
        d = out_root / "_extra" / label
        shutil.rmtree(d, ignore_errors=True)
        try:
            m = train(model, model_cfg, cfg["train"], cfg["data"], 0, d, **{**common, **kw})
            ok = (m.get("status") == "completed" and m.get("test_acc") is not None
                  and math.isfinite(load_json(d / "metrics.json")["best_val_loss"]))
            out.append((f"{label} runs end to end", ok,
                        f"{m.get('total_steps')} steps, aug={m.get('augmentation')}, "
                        f"flops={m.get('flops')}"))
        except Exception as exc:  # report, don't abort the other paths
            import traceback
            traceback.print_exc()
            out.append((f"{label} runs end to end", False, repr(exc)))

    strong = cfg["augmentation"]["strong"]
    run("E3 strong augmentation (simple_cnn)", "simple_cnn",
        load_config("configs/models/simple_cnn.yaml"), max_steps=e3_steps,
        augmentation="strong", aug_cfg=strong)
    run("E2 step-defined run (vit, partial epoch)", "vit",
        load_config("configs/models/vit.yaml"), max_steps=e2_steps)
    e4 = cfg["experiments"]["e4_pretrained"]
    for name in e4["models"]:
        # Batch 64 as in the protocol on a GPU; 8 on a CPU, where 64 images at
        # 224 px would take minutes per step and test nothing extra.
        bs = int(e4["batch_size"]) if torch.cuda.is_available() else 8
        run(f"E4 {name.split('.')[0]} @224 (random init)", f"timm/{name}",
            {"pretrained": False, "image_size": int(e4["image_size"])},
            max_steps=3, batch_size=bs, max_eval_batches=1)
    return out


def check_param_groups(name: str, model_cfg: dict, train_cfg: dict) -> tuple[bool, str]:
    """Every matrix/conv weight decays; every 1-D param and embedding does not."""
    model = build_model(name, model_cfg["model"])
    groups = build_param_groups(model, float(train_cfg["weight_decay"]),
                                train_cfg.get("no_decay_keywords", []))
    decay_names, nodecay_names = groups[0]["names"], groups[1]["names"]
    params = dict(model.named_parameters())
    bad = [n for n in decay_names if params[n].ndim <= 1]
    bad += [n for n in decay_names if "pos_embed" in n or "cls_token" in n or "gamma" in n]
    covered = len(decay_names) + len(nodecay_names) == len(params)
    ok = not bad and covered and len(decay_names) > 0
    return ok, (f"{len(decay_names)} decay / {len(nodecay_names)} no-decay"
                + (f"; WRONGLY DECAYED: {bad[:4]}" if bad else "")
                + ("" if covered else "; some params in neither group"))


def smoke_one(name: str, cfg: dict, out_root: Path, args) -> dict:
    model_cfg = load_config(f"configs/models/{name}.yaml")
    out_dir = out_root / name / "0"
    tb_dir = out_root / "tb" / name / "0"
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(tb_dir, ignore_errors=True)

    kwargs = dict(
        experiment="smoke", lr=SMOKE_LR, epochs=2, fraction=args.fraction,
        augmentation="standard", save_checkpoints=True, keep_last=True, evaluate_test=True,
        tb_dir=tb_dir, num_workers=args.num_workers, warmup_fraction=0.2, ece_bins=int(cfg["eval"]["ece_bins"]),
        max_eval_batches=args.max_eval_batches,
    )
    checks: list[tuple[str, bool, str]] = []
    notes: list[str] = []
    t0 = time.perf_counter()

    def learning(label: str, ok: bool, detail: str) -> None:
        """A check of whether the model learns: gated, except with --tiny."""
        if args.tiny:
            notes.append(f"{label}: {'yes' if ok else 'NO'} ({detail}) -- not gated with --tiny")
        else:
            checks.append((label, ok, detail))

    # 1) epoch 1, then "crash".
    first = train(name, model_cfg, cfg["train"], cfg["data"], 0, out_dir,
                  stop_after_epochs=1, **kwargs)
    checks.append(("interrupted after epoch 1", first.get("status") == "interrupted",
                   str(first)))
    checks.append(("last.pt written before interruption", (out_dir / "last.pt").exists(), ""))
    checks.append(("no metrics.json after interruption", not (out_dir / "metrics.json").exists(),
                   ""))

    # 2) resume -> epoch 2 -> finish.
    m = train(name, model_cfg, cfg["train"], cfg["data"], 0, out_dir, resume=True, **kwargs)
    secs = time.perf_counter() - t0

    checks.append(("completed", m.get("status") == "completed", m.get("status", "?")))
    checks.append(("resumed at the end of epoch 1",
                   m.get("resumed_from_step") == first.get("step"),
                   f"resumed_from_step={m.get('resumed_from_step')} expected {first.get('step')}"))

    import csv
    with (out_dir / "history.csv").open(encoding="utf-8") as fh:
        hist = list(csv.DictReader(fh))
    checks.append(("history has epochs 1 and 2", [r["epoch"] for r in hist] == ["1", "2"],
                   str([r["epoch"] for r in hist])))
    if len(hist) == 2:
        l1, l2 = float(hist[0]["train_loss"]), float(hist[1]["train_loss"])
        learning("train loss decreasing", l2 < l1, f"{l1:.4f} -> {l2:.4f}")
        # LR continuity: the last LR of epoch 2 must be what the schedule says
        # for the final step -- it would differ if the resume had restarted it.
        conf = load_json(out_dir / "metrics.json")
        expect = lr_at(conf["total_steps"] - 1, SMOKE_LR, conf["total_steps"],
                       conf["warmup_steps"], float(cfg["train"]["schedule"]["min_lr"]))
        got = float(hist[1]["lr"])
        checks.append(("LR schedule continued across resume",
                       math.isclose(got, expect, rel_tol=1e-6), f"{got:.3e} vs {expect:.3e}"))

    missing = [f for f in EXPECTED_FILES if not (out_dir / f).exists()]
    checks.append(("all output files written", not missing, f"missing: {missing}" if missing else ""))
    tb_ok = tb_dir.exists() and any(tb_dir.rglob("events.out.tfevents.*"))
    checks.append(("TensorBoard events written", tb_ok, _rel(tb_dir)))
    absent = [k for k in SPEC_METRIC_KEYS if k not in m]
    checks.append(("metrics.json has every spec key", not absent, f"missing: {absent}" if absent else ""))
    # FLOPs are deliberately not in this list: counting them is a measurement,
    # not a gate (see src.profile.try_count_flops). On a PyTorch build where
    # FlopCounterMode fails, metrics.json records null and this test warns.
    nulls = [k for k in ("test_acc", "params", "ece", "infer_imgs_per_sec") if m.get(k) is None]
    checks.append(("core metrics non-null", not nulls, f"null: {nulls}" if nulls else ""))
    learning("test acc above chance", (m.get("test_acc") or 0) > 0.15,
             f"test_acc={m.get('test_acc')}")
    warnings = []
    if m.get("flops") is None:
        warnings.append("FLOPs not counted, recorded as null (not a gate): "
                        f"{m.get('flops_error') or 'reason not recorded'}")

    return {"model": name, "checks": checks, "warnings": warnings, "notes": notes, "seconds": secs,
            "test_acc": m.get("test_acc"), "best_val_acc": m.get("best_val_acc"),
            "train_imgs_per_sec": m.get("train_imgs_per_sec"),
            "sec_per_epoch": m.get("sec_per_epoch"),
            "train_loss": [float(r["train_loss"]) for r in hist]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=list(MODEL_ORDER))
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader workers (default: data.num_workers from base.yaml)")
    ap.add_argument("--quick", action="store_true",
                    help="cap val/test at 8 batches each -- for slow CPUs only")
    ap.add_argument("--out", default="results/smoke")
    ap.add_argument("--skip-extra", action="store_true",
                    help="skip the E2/E3/E4 code-path checks")
    ap.add_argument("--extra-only", action="store_true",
                    help="run only the E2/E3/E4 code-path checks and merge them into "
                         "an existing results/smoke_test.json")
    ap.add_argument("--tiny", action="store_true",
                    help="2%% of the data and 2 val/test batches: does the code run, not does "
                         "it learn (learning checks reported, not gated) -- for the rehearsal")
    ap.add_argument("--summary", default="results/smoke_test.json",
                    help="where the JSON summary goes (default: the committed gate record)")
    args = ap.parse_args()
    args.fraction = 0.02 if args.tiny else 0.1
    args.max_eval_batches = 2 if args.tiny else (8 if args.quick else None)

    cfg = load_config("configs/base.yaml")
    out_root = ensure_dir(args.out)
    summary_path = resolve_path(args.summary)
    results = []
    all_ok = True

    print("== loop-level checks ==")
    loop_checks: list[tuple[str, bool, str]] = []
    for label, ok, detail in check_schedule():
        loop_checks.append((f"schedule: {label}", ok, detail))
    for name in args.models:
        ok, detail = check_param_groups(name, load_config(f"configs/models/{name}.yaml"),
                                        cfg["train"])
        loop_checks.append((f"param groups {name}", ok, detail))
    print("  ... exact-resume check (mlp, cpu, 0 workers)", flush=True)
    ok, detail = check_exact_resume(cfg, out_root)
    loop_checks.append(("interrupted+resumed == uninterrupted", ok, detail))
    for label, ok, detail in loop_checks:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")

    if args.extra_only:
        # Loop-level + extra paths only; per-model results already on disk are kept.
        print("\n== extra code paths (E2 / E3 / E4) ==", flush=True)
        extra = check_extra_paths(cfg, out_root, args)
        for label, ok, detail in extra:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}")
        prev = load_json(summary_path) if summary_path.exists() else {"results": []}
        kept = [r for r in prev["results"] if r["model"] not in ("loop_level", "extra_paths")]
        for name, checks in (("loop_level", loop_checks), ("extra_paths", extra)):
            kept.append({"model": name, "seconds": 0.0,
                         "checks": [{"check": c, "passed": ok, "detail": d}
                                    for c, ok, d in checks]})
        prev["results"] = kept
        prev["all_passed"] = all(c["passed"] for r in kept for c in r["checks"])
        save_json(prev, summary_path)
        n_all = sum(len(r["checks"]) for r in kept)
        n_ok = sum(c["passed"] for r in kept for c in r["checks"])
        print(f"\n{n_ok}/{n_all} checks passed overall -- "
              f"{'ALL PASS' if prev['all_passed'] else 'FAILURES ABOVE'}; "
              f"merged into {_rel(summary_path)}")
        return 0 if prev["all_passed"] else 1

    results.append({"model": "loop_level", "checks": loop_checks, "seconds": 0.0})

    for name in args.models:
        print(f"\n== {name} ==", flush=True)
        try:
            r = smoke_one(name, cfg, out_root, args)
        except Exception as exc:  # report every model, don't stop at the first
            import traceback
            traceback.print_exc()
            r = {"model": name, "checks": [("ran without error", False, repr(exc))],
                 "seconds": 0.0}
        results.append(r)
        for label, ok, detail in r["checks"]:
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not ok else
                                                               (f"  {detail}" if detail else "")))
        for w in r.get("warnings", []):
            print(f"  [WARN] {w}")
        for n in r.get("notes", []):
            print(f"  [INFO] {n}")
        print(f"  -> {human_time(r['seconds'])}", flush=True)

    if not args.skip_extra:
        print("\n== extra code paths (E2 / E3 / E4) ==", flush=True)
        extra = check_extra_paths(cfg, out_root, args)
        results.append({"model": "extra_paths", "checks": extra, "seconds": 0.0})
        for label, ok, detail in extra:
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {detail}")

    n = sum(len(r["checks"]) for r in results)
    n_ok = sum(ok for r in results for _, ok, _ in r["checks"])
    save_json({"all_passed": bool(all_ok), "quick": args.quick, "tiny": args.tiny,
               "device": "cuda" if torch.cuda.is_available() else "cpu",
               "results": [{**r, "checks": [{"check": c, "passed": ok, "detail": d}
                                            for c, ok, d in r["checks"]]} for r in results]},
              summary_path)

    print("\n" + "=" * 72)
    print(f"{'model':<15}{'loss ep1 -> ep2':>20}{'val acc':>10}{'test acc':>10}{'img/s':>9}{'time':>9}")
    for r in results:
        if r["model"] in ("loop_level", "extra_paths"):
            continue
        tl = r.get("train_loss") or []
        ls = f"{tl[0]:.3f} -> {tl[1]:.3f}" if len(tl) == 2 else "-"
        va = f"{r['best_val_acc']:.3f}" if r.get("best_val_acc") is not None else "-"
        ta = f"{r['test_acc']:.3f}" if r.get("test_acc") is not None else "-"
        ips = f"{r['train_imgs_per_sec']:.0f}" if r.get("train_imgs_per_sec") else "-"
        print(f"{r['model']:<15}{ls:>20}{va:>10}{ta:>10}{ips:>9}{human_time(r['seconds']):>9}")
    n_warn = sum(len(r.get("warnings", [])) for r in results)
    print(f"\n{n_ok}/{n} checks passed -- {'ALL PASS' if all_ok else 'FAILURES ABOVE'}"
          + (f" ({n_warn} warning(s) above, not gating)" if n_warn else ""))
    return 0 if all_ok else 1


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    sys.exit(main())

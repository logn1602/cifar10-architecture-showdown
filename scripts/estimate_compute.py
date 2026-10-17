"""Phase 0: project the compute cost of the whole plan (spec Section 9).

Section 9 says to estimate the budget "for real in Phase 0" by timing one epoch
per model on the actual hardware. This machine has no usable GPU, so there is
nothing honest to time -- instead this script projects the queue onto *named
reference hardware* from configs/compute_reference.yaml and prints what the
plan would cost there.

Phase 3's scripts/plan_budget.py does the real thing: it times ~200 steps per
model on whatever GPU the run is actually using and rewrites these numbers.
Until then every figure below carries an ESTIMATE banner.

Examples
--------
    python scripts/estimate_compute.py --device t4 --num-gpus 2 --mode 9h
    python scripts/estimate_compute.py --device cpu_i5_1135g7 --mode full
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, human_time, load_config, save_json  # noqa: E402

MODELS = ["mlp", "simple_cnn", "resnet18", "vit", "mlp_mixer", "convnext_mini"]

BANNER = (
    "!! ESTIMATES ONLY -- derived from FLOP counts and published throughput, not measured.\n"
    "!! Replace with real timings from scripts/plan_budget.py (Phase 3) before the\n"
    "!! unattended run. Treat these as order-of-magnitude, not a promise."
)


class Device:
    """Per-model cost model for one reference device."""

    def __init__(self, key: str, ref: dict, cfg: dict):
        devices = ref["devices"]
        if key not in devices:
            raise SystemExit(f"Unknown device {key!r}. Choices: {', '.join(devices)}")
        self.key = key
        self.spec = devices[key]
        self.name = self.spec["name"]
        self.ref = ref
        self.overheads = ref["overheads"]

        self.train_size = int(ref["meta"]["train_set_size"])
        self.val_size = int(ref["meta"]["val_set_size"])
        self.test_size = int(ref["meta"]["test_set_size"])
        self.batch_size = int(cfg["train"]["batch_size"])
        self.steps_per_full_epoch = self.train_size // self.batch_size  # drop_last=True
        self.infer_ips = float(self.spec["infer_imgs_per_sec"])

    def val_seconds(self) -> float:
        return self.val_size / self.infer_ips

    def train_seconds_per_epoch(self, model: str, strong: bool = False) -> float:
        """Training-only seconds for one full 45k epoch (validation removed)."""
        total = float(self.spec["sec_per_epoch"][model])
        train_only = max(total - self.val_seconds(), 0.1)
        if strong:
            train_only *= float(self.overheads["strong_augmentation"])
        return train_only

    def epoch_run(self, model: str, epochs: int, strong: bool = False) -> float:
        """Cost of an ordinary epoch-defined run on the full train split."""
        per_epoch = self.train_seconds_per_epoch(model, strong) + self.val_seconds()
        return epochs * per_epoch + float(self.overheads["per_run_seconds"])

    def step_run(self, model: str, total_steps: int, fraction: float,
                 strong: bool = False) -> float:
        """Cost of a step-matched run (E2), accounting for validation frequency.

        At a small data fraction an "epoch" is only a handful of steps, so a
        step-matched run performs far more validation passes than a full-data
        run of equal step count. At 10% that is ~200 val passes instead of ~20
        -- not free, and easy to forget when budgeting.
        """
        steps_per_epoch = max(1, int(fraction * self.train_size) // self.batch_size)
        n_epochs = math.ceil(total_steps / steps_per_epoch)
        sec_per_step = self.train_seconds_per_epoch(model, strong) / self.steps_per_full_epoch
        return (total_steps * sec_per_step
                + n_epochs * self.val_seconds()
                + float(self.overheads["per_run_seconds"]))

    def e4_run(self, model_key: str, epochs: int) -> float:
        e4 = self.ref["e4_224px"]
        ips = float(e4["train_imgs_per_sec"][self.key][model_key])
        imgs = int(e4["images_per_epoch"])
        # Validation at 224px costs roughly a third of a training epoch.
        return epochs * (imgs / ips) * 1.1 + float(self.overheads["per_run_seconds"])

    def eval_images(self, n_images: int) -> float:
        return n_images / self.infer_ips


def eval_cost(dev: Device, n_seeds: int) -> list[tuple[str, int, str, float]]:
    """E5 / E6 / E7 are forward passes only, but there are a lot of them."""
    rows = []
    # E5: 6 corruptions x 5 severities x 10,000 test images, per model per seed.
    n = 6 * 5 * dev.test_size
    rows.append(("E5 robustness (corruptions)", len(MODELS) * n_seeds,
                 f"{n:,} fwd/run",
                 len(MODELS) * n_seeds * dev.eval_images(n)))
    # E6: 3 grid sizes x 2 variants (patch shuffle, pixel shuffle) x 10,000.
    n = 3 * 2 * dev.test_size
    rows.append(("E6 patch shuffle", len(MODELS) * n_seeds,
                 f"{n:,} fwd/run",
                 len(MODELS) * n_seeds * dev.eval_images(n)))
    # E7: seed 0 only -- UMAP on 2,000, agreement matrix + mistakes on 10,000,
    # plus Grad-CAM / attention on the 16 fixed images (backward passes, slow
    # per image but only 16 of them).
    n = 2000 + dev.test_size
    rows.append(("E7 interpretability", len(MODELS), f"{n:,} fwd/run (seed 0)",
                 len(MODELS) * dev.eval_images(n) * 1.5))
    return rows


def queue_full(dev: Device, cfg: dict) -> list[dict]:
    exp = cfg["experiments"]
    sweep = exp["lr_sweep"]
    items: list[dict] = []

    sec = sum(dev.epoch_run(m, sweep["epochs"]) for m in MODELS) * len(sweep["lrs"])
    items.append({"item": "LR sweep", "runs": len(MODELS) * len(sweep["lrs"]),
                  "each": f"{sweep['epochs']} epochs", "seconds": sec})

    seeds = len(cfg["train"]["seeds"])
    sec = sum(dev.epoch_run(m, cfg["train"]["epochs"]) for m in MODELS) * seeds
    items.append({"item": "E1 main showdown", "runs": len(MODELS) * seeds,
                  "each": f"{cfg['train']['epochs']} epochs", "seconds": sec})

    # E2: 100% reuses E1; 10% gets 3 seeds, the rest 1 seed each.
    e2_steps = int(exp["e2_total_steps"])
    e2_plan = [(0.05, 1), (0.1, 3), (0.25, 1), (0.5, 1)]
    sec = sum(dev.step_run(m, e2_steps, frac) * n
              for m in MODELS for frac, n in e2_plan)
    items.append({"item": "E2 data-hunger", "runs": len(MODELS) * sum(n for _, n in e2_plan),
                  "each": f"{e2_steps:,} steps", "seconds": sec})

    sec = sum(dev.epoch_run(m, cfg["train"]["epochs"], strong=True) for m in MODELS)
    items.append({"item": "E3 strong augmentation", "runs": len(MODELS),
                  "each": f"{cfg['train']['epochs']} epochs", "seconds": sec})

    e4 = exp["e4_pretrained"]
    e4_models = ["vit_small_patch16_224", "resnet50"]
    sweep_sec = sum(dev.e4_run(m, e4["epochs"]) for m in e4_models) * len(e4["lrs"])
    main_sec = sum(dev.e4_run(m, e4["epochs"]) for m in e4_models) * seeds
    items.append({"item": "E4 pretrained (LR sweep)", "runs": len(e4_models) * len(e4["lrs"]),
                  "each": f"{e4['epochs']} epochs @224px", "seconds": sweep_sec})
    items.append({"item": "E4 pretrained (main)", "runs": len(e4_models) * seeds,
                  "each": f"{e4['epochs']} epochs @224px", "seconds": main_sec})

    for name, runs, each, sec in eval_cost(dev, seeds):
        items.append({"item": name, "runs": runs, "each": each, "seconds": sec})
    return items


def queue_9h(dev: Device, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Returns (core items, candidate upgrades) per spec 12.2 and 12.3."""
    m9 = cfg["mode_9h"]
    core: list[dict] = []

    sweep = m9["lr_sweep"]
    sec = sum(dev.epoch_run(m, sweep["epochs"]) for m in MODELS) * len(sweep["lrs"])
    core.append({"item": "LR sweep", "runs": len(MODELS) * len(sweep["lrs"]),
                 "each": f"{sweep['epochs']} epochs", "seconds": sec})

    e1 = m9["e1"]
    sec = sum(dev.epoch_run(m, e1["epochs"]) for m in MODELS) * len(e1["seeds"])
    core.append({"item": "E1 main showdown", "runs": len(MODELS) * len(e1["seeds"]),
                 "each": f"{e1['epochs']} epochs", "seconds": sec})

    for name, runs, each, sec in eval_cost(dev, 1):
        core.append({"item": name, "runs": runs, "each": each, "seconds": sec})

    e2 = m9["e2"]
    sec = sum(dev.step_run(m, e2["total_steps"], f)
              for m in MODELS for f in e2["fractions"]) * len(e2["seeds"])
    core.append({"item": "E2 data-hunger", "runs": len(MODELS) * len(e2["fractions"]),
                 "each": f"{e2['total_steps']:,} steps", "seconds": sec})

    e4 = m9["e4"]
    e4_models = ["vit_small_patch16_224", "resnet50"]
    sec = sum(dev.e4_run(m, e4["epochs"]) for m in e4_models) * len(e4["seeds"])
    core.append({"item": "E4 pretrained", "runs": len(e4_models) * len(e4["seeds"]),
                 "each": f"{e4['epochs']} epochs @224px", "seconds": sec})

    # --- upgrades, spec 12.3, in priority order ---------------------------
    e3_epochs = m9["e3"]["epochs"]
    pair = ["resnet18", "vit"]
    rest = [m for m in MODELS if m not in pair]
    upgrades = [
        {"item": "[U1] E3 for resnet18 + vit", "runs": 2, "each": f"{e3_epochs} epochs",
         "seconds": sum(dev.epoch_run(m, e3_epochs, strong=True) for m in pair)},
        {"item": "[U2] E1 seeds 1 and 2", "runs": len(MODELS) * 2,
         "each": f"{e1['epochs']} epochs",
         "seconds": sum(dev.epoch_run(m, e1["epochs"]) for m in MODELS) * 2},
        {"item": "[U3] E3 for the other 4 models", "runs": len(rest),
         "each": f"{e3_epochs} epochs",
         "seconds": sum(dev.epoch_run(m, e3_epochs, strong=True) for m in rest)},
        {"item": "[U4] E2 fractions 0.05 and 0.5", "runs": len(MODELS) * 2,
         "each": f"{e2['total_steps']:,} steps",
         "seconds": sum(dev.step_run(m, e2["total_steps"], f)
                        for m in MODELS for f in (0.05, 0.5))},
        {"item": "[U5] E2 seeds 1 and 2", "runs": len(MODELS) * len(e2["fractions"]) * 2,
         "each": f"{e2['total_steps']:,} steps",
         "seconds": sum(dev.step_run(m, e2["total_steps"], f)
                        for m in MODELS for f in e2["fractions"]) * 2},
    ]
    return core, upgrades


def wall_clock(seconds: float, num_gpus: int, efficiency: float) -> float:
    """GPU-seconds -> wall-clock seconds across `num_gpus` queue workers."""
    if num_gpus <= 1:
        return seconds
    return seconds / (num_gpus * efficiency)


def render(dev: Device, cfg: dict, args) -> tuple[str, dict]:
    eff = float(dev.overheads["multi_gpu_efficiency"])
    L: list[str] = []
    add = L.append
    payload: dict = {"device": dev.key, "device_name": dev.name, "mode": args.mode,
                     "num_gpus": args.num_gpus, "budget_hours": args.budget_hours,
                     "source": "estimate", "items": []}

    add("=" * 78)
    add(f"  COMPUTE ESTIMATE  |  mode={args.mode}  device={dev.key}  gpus={args.num_gpus}")
    add("=" * 78)
    add(f"  {dev.name}")
    if args.num_gpus > 1:
        add(f"  queue split across {args.num_gpus} GPUs at {eff:.0%} parallel efficiency")
        add("  (the workers share the same host CPU cores feeding both GPUs)")
    add("")
    add(BANNER)
    add("")

    def table(items: list[dict], title: str) -> float:
        add(f"[ {title} ]")
        add(f"  {'item':<30}{'runs':>5}{'each':>26}{'GPU-h':>9}{'wall-h':>9}")
        add(f"  {'-' * 79}")
        total = 0.0
        for it in items:
            gpu_h = it["seconds"] / 3600
            wall_h = wall_clock(it["seconds"], args.num_gpus, eff) / 3600
            total += it["seconds"]
            add(f"  {it['item'][:30]:<30}{it['runs']:>5}{it['each'][:26]:>26}"
                f"{gpu_h:>9.2f}{wall_h:>9.2f}")
            payload["items"].append({**it, "section": title, "gpu_hours": round(gpu_h, 3),
                                     "wall_hours": round(wall_h, 3)})
        add(f"  {'-' * 79}")
        add(f"  {'TOTAL':<30}{'':>5}{'':>26}{total/3600:>9.2f}"
            f"{wall_clock(total, args.num_gpus, eff)/3600:>9.2f}")
        add("")
        return total

    if args.mode == "full":
        items = queue_full(dev, cfg)
        total = table(items, "Full mode (spec Sections 3-9)")
        wall_h = wall_clock(total, args.num_gpus, eff) / 3600
        payload["total_gpu_hours"] = round(total / 3600, 2)
        payload["total_wall_hours"] = round(wall_h, 2)
        add(f"  Full mode: {human_time(total)} of GPU time "
            f"({human_time(wall_clock(total, args.num_gpus, eff))} wall-clock).")
        if total / 3600 > 60:
            add("  Spec Section 9: over ~60 GPU-hours means asking before proceeding.")
            add("  The sanctioned trims are fewer E2 seeds or fewer sweep LRs --")
            add("  NOT fewer epochs in E1.")
        add("")
        return "\n".join(L), payload

    # --- 9h mode -----------------------------------------------------------
    core, upgrades = queue_9h(dev, cfg)
    core_total = table(core, "9h mode -- core queue (spec 12.2)")
    budget_s = args.budget_hours * 3600
    core_wall = wall_clock(core_total, args.num_gpus, eff)

    add("[ Verdict ]")
    add(f"  budget              : {args.budget_hours:.1f} h "
        f"(spec 12.4 default; leaves ~1 h slack in the 9 h window)")
    add(f"  core queue          : {human_time(core_wall)} wall-clock")
    fits = core_wall <= budget_s
    add(f"  core fits?          : {'YES' if fits else 'NO'}")
    add("")

    payload["core_gpu_hours"] = round(core_total / 3600, 2)
    payload["core_wall_hours"] = round(core_wall / 3600, 2)
    payload["core_fits"] = bool(fits)
    payload["upgrades_included"] = []

    if fits:
        remaining = budget_s - core_wall
        add(f"[ Upgrades (spec 12.3) -- {human_time(remaining)} of headroom ]")
        add(f"  {'upgrade':<36}{'runs':>6}{'wall-h':>9}{'cumulative':>12}  fits?")
        add(f"  {'-' * 76}")
        cumulative = core_wall
        for up in upgrades:
            up_wall = wall_clock(up["seconds"], args.num_gpus, eff)
            ok = (cumulative + up_wall) <= budget_s
            if ok:
                cumulative += up_wall
                payload["upgrades_included"].append(up["item"])
            add(f"  {up['item']:<36}{up['runs']:>6}{up_wall/3600:>9.2f}"
                f"{cumulative/3600:>12.2f}  {'yes' if ok else 'no'}")
            payload["items"].append({**up, "section": "upgrades",
                                     "gpu_hours": round(up["seconds"]/3600, 3),
                                     "wall_hours": round(up_wall/3600, 3),
                                     "included": bool(ok)})
        add(f"  {'-' * 76}")
        add(f"  projected finish    : {human_time(cumulative)} after launch")
        payload["projected_wall_hours"] = round(cumulative / 3600, 2)
        add("")
        add("  Upgrades are added top-down only while the projection still fits. The")
        add("  unattended queue re-plans after every run with actual times and drops")
        add("  upgrades from the bottom if it slips (spec 12.5).")
    else:
        add("[ Trim ladder (spec 12.4) ]")
        add("  The core does not fit. Trims, in the mandated order -- E1 is never cut")
        add("  below 100 epochs:")
        over = core_wall - budget_s
        add(f"    over budget by {human_time(over)}")
        for desc in ("E4 epochs 5 -> 3",
                     "E2 steps 7,040 -> 5,280",
                     "LR sweep 10 -> 6 epochs"):
            add(f"    - {desc}")
        add("")
        add("  If it still does not fit after all three, the honest options are more")
        add("  GPUs (--num-gpus 2 on Kaggle) or a longer window -- not a shorter E1.")

    add("")
    add("=" * 78)
    return "\n".join(L), payload


def main() -> None:
    cfg = load_config("configs/base.yaml")
    ref = load_config("configs/compute_reference.yaml")

    ap = argparse.ArgumentParser(description="Project the compute cost of the plan")
    ap.add_argument("--device", default="t4", choices=sorted(ref["devices"]))
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--mode", default="9h", choices=["full", "9h"])
    ap.add_argument("--budget-hours", type=float, default=8.0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    dev = Device(args.device, ref, cfg)
    text, payload = render(dev, cfg, args)
    print(text)

    out_dir = ensure_dir(args.out_dir)
    tables = ensure_dir(out_dir / "tables")
    save_json(payload, out_dir / "queue_estimate.json")

    md = ["# Phase 0 -- Compute estimate\n",
          "_Generated by `scripts/estimate_compute.py`. Do not edit by hand._\n",
          "> **Estimates only.** Derived from FLOP counts and published throughput, not",
          "> measured. `scripts/plan_budget.py` (Phase 3) replaces these with real timings",
          "> taken on the GPU that will actually run the queue.\n",
          "```", text, "```", ""]
    (tables / "compute_estimate.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_dir / 'queue_estimate.json'}")
    print(f"wrote {tables / 'compute_estimate.md'}")


if __name__ == "__main__":
    main()

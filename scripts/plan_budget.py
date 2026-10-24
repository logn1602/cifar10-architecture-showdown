"""Budget planner (spec 12.4): time the real hardware, then fit the queue.

    python scripts/plan_budget.py --hours 8            # 9h mode, all visible GPUs
    python scripts/plan_budget.py --mode full          # full-mode projection (spec Section 9)
    python scripts/plan_budget.py --steps 30 --skip 5 --e4-steps 4   # quick check on a CPU

What it does:

1. **Times every model on this machine**: ~200 training steps at batch 128
   through the real DataLoader, real AMP and real optimizer (the first 20 are
   discarded as warmup -- cuDNN autotuning, worker start-up, allocator growth),
   plus one full 5,000-image validation pass. Also the strong-augmentation
   loader on its own (E3 can be input-bound), and ~50 steps of each E4 model
   at 224 px.
2. **Estimates every job** from those timings (steps x sec/step + validation
   passes + the one test pass + fixed per-run overhead).
3. **Simulates the queue** across the GPUs with the same rule the coordinator
   uses -- each free worker takes the first job whose dependencies are done --
   to get a projected wall-clock finish.
4. **Fits the budget**: if even the core overruns, walks the trim ladder
   (E4 epochs 5 -> 3, E2 steps 7,040 -> 5,280, sweep epochs 10 -> 6; E1 is
   never cut). Then adds upgrades in spec 12.3 order while the projection
   still fits.
5. Prints the queue with per-run estimates and the projected finish time, and
   saves it to ``results/queue_9h.json`` for ``run_all.py``.

Nothing is launched. Read the plan, say "go", then start ``run_all.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.data import build_dataloaders  # noqa: E402
from src.evaluate import autocast_context, evaluate  # noqa: E402
from src.experiments import (  # noqa: E402
    Job, apply_trims, build_queue, job_done, load_base_config, mode_params, results_path,
    upgrade_order,
)
from src.models import MODEL_ORDER  # noqa: E402
from src.train import build_any_model, build_param_groups, make_input_fn, pick_amp_dtype  # noqa: E402
from src.utils import PROJECT_ROOT, human_time, load_config, load_json, save_json, set_seed  # noqa: E402

FULL_EPOCH_STEPS = 351       # 45,000 // 128 with drop_last
VAL_IMAGES, TEST_IMAGES = 5_000, 10_000


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# 1. Timing
# ---------------------------------------------------------------------------


def time_training(model_name: str, model_cfg: dict, cfg: dict, device: torch.device,
                  steps: int, skip: int, num_workers: int, batch_size: int = 128,
                  augmentation: str = "standard") -> dict:
    """Seconds per optimizer step and per validation pass, measured.

    The step mirrors ``src.train.train`` exactly: same param groups, AMP dtype,
    GradScaler, grad clipping, label smoothing and channels_last.
    """
    set_seed(0)
    model, info = build_any_model(model_name, model_cfg)
    tcfg = cfg["train"]
    channels_last = bool(tcfg.get("channels_last_for_conv", True)) and \
        info["family"] == "convolutional"
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    amp_dtype = pick_amp_dtype(device, tcfg.get("amp", {}))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    groups = build_param_groups(model, float(tcfg["weight_decay"]), tcfg.get("no_decay_keywords", []))
    for g in groups:
        g.pop("names")
    opt = torch.optim.AdamW(groups, lr=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=float(tcfg["label_smoothing"]))

    train_loader, val_loader, _ = build_dataloaders(
        cfg["data"], augmentation=augmentation, batch_size=batch_size, num_workers=num_workers,
        aug_cfg=cfg["augmentation"].get(augmentation), eval_batch_size=max(batch_size, 256))
    input_fn = make_input_fn(info, cfg["data"]["mean"], cfg["data"]["std"], device)

    model.train()
    it = iter(train_loader)
    t_start = None
    done = 0
    for s in range(steps):
        if s == skip:
            _sync(device)
            t_start = time.perf_counter()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if input_fn is not None:
            x = input_fn(x)
        with autocast_context(device, amp_dtype):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip_norm"]))
        scaler.step(opt)
        scaler.update()
        if s >= skip:
            done += 1
    _sync(device)
    sec_per_step = (time.perf_counter() - t_start) / max(1, done)

    # Time the *second* val pass: the first one also starts the val loader's
    # worker processes, which a real run does once (covered by the per-run
    # overhead), not on every validation.
    evaluate(model, val_loader, device, amp_dtype, input_fn=input_fn, channels_last=channels_last)
    t = time.perf_counter()
    evaluate(model, val_loader, device, amp_dtype, input_fn=input_fn, channels_last=channels_last)
    _sync(device)
    val_sec = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return {"sec_per_step": sec_per_step, "val_sec": val_sec, "steps_timed": done,
            "imgs_per_sec": batch_size / sec_per_step, "peak_mem_mb": peak}


def time_loader(cfg: dict, augmentation: str, num_workers: int, batches: int = 60,
                skip: int = 10) -> float:
    """Seconds per 128-image batch of the input pipeline alone."""
    loader, _, _ = build_dataloaders(cfg["data"], augmentation=augmentation, batch_size=128,
                                     num_workers=num_workers,
                                     aug_cfg=cfg["augmentation"].get(augmentation))
    it = iter(loader)
    t0 = None
    for i in range(batches):
        if i == skip:
            t0 = time.perf_counter()
        next(it)
    return (time.perf_counter() - t0) / max(1, batches - skip)


# ---------------------------------------------------------------------------
# 2. Per-job estimates
# ---------------------------------------------------------------------------


def estimate_job(job: Job, t: dict, cfg: dict, n_models_eval: int) -> float:
    """Seconds this job should take on one GPU."""
    over = float(t["per_run_overhead_sec"])
    if job.kind == "eval":
        # Inference-only passes over the 10k test set, for each E1 seed-0 model.
        val_rate = {m: VAL_IMAGES / t["models"][m]["val_sec"] for m in t["models"]}
        mean_rate = sum(val_rate.values()) / max(1, len(val_rate))
        n_imgs = {"E5": 6 * 5 * TEST_IMAGES, "E6": 6 * TEST_IMAGES, "E7": 2 * 2_000}[job.exp]
        extra = {"E5": 60.0, "E6": 30.0, "E7": 300.0}[job.exp]  # corruption gen, UMAP, plots
        return n_models_eval * n_imgs / mean_rate * 1.2 + extra + over

    if job.model.startswith("timm/"):
        tm = t["e4"][job.model.split("/", 1)[1].split(".")[0]]
        spe = 45_000 // int(job.batch_size or 64)
        steps = int(job.epochs) * spe
        evals = int(job.epochs)
        sec = steps * tm["sec_per_step"] + evals * tm["val_sec"]
        if job.evaluate_test:
            sec += tm["val_sec"] * TEST_IMAGES / VAL_IMAGES
        return sec + over + 60.0   # + pretrained weight download / model build

    tm = t["models"][job.model]
    if job.max_steps is not None:
        steps = int(job.max_steps)
        evals = math.ceil(steps / FULL_EPOCH_STEPS)
    else:
        spe = int(45_000 * job.fraction) // 128
        steps = int(job.epochs) * spe
        evals = int(job.epochs)
    step = tm["sec_per_step"]
    if job.augmentation == "strong":
        # Strong aug costs CPU time; a step is as slow as the slower of the
        # model and the (heavier) input pipeline.
        step = max(step * float(t["strong_overhead"]), t["strong_loader_sec_per_batch"])
    sec = steps * step + evals * tm["val_sec"]
    if job.evaluate_test:
        sec += tm["val_sec"] * TEST_IMAGES / VAL_IMAGES
    return sec + over


def attach_estimates(jobs: list[Job], t: dict, cfg: dict, n_gpus: int) -> None:
    n_eval_models = len(MODEL_ORDER)
    # Two jobs sharing one host's CPUs run a bit slower than either alone.
    eff = float(t["multi_gpu_efficiency"]) if n_gpus > 1 else 1.0
    for j in jobs:
        j.est_sec = estimate_job(j, t, cfg, n_eval_models) / eff


# ---------------------------------------------------------------------------
# 3. Queue simulation
# ---------------------------------------------------------------------------


def simulate(jobs: list[Job], n_workers: int, skip_done: bool = True) -> tuple[float, dict]:
    """Makespan (sec) of the coordinator's policy, and each job's (start, end)."""
    finish: dict[str, float] = {}
    pending = [j for j in jobs if not (skip_done and job_done(j))]
    for j in jobs:
        if skip_done and job_done(j):
            finish[j.id] = 0.0
    ids = {j.id for j in jobs}
    workers = [0.0] * n_workers
    sched: dict[str, tuple[float, float]] = {}
    while pending:
        w = min(range(n_workers), key=lambda i: workers[i])
        now = workers[w]
        chosen = None
        for j in pending:
            deps = [d for d in j.deps if d in ids]
            if all(d in finish and finish[d] <= now for d in deps):
                chosen = j
                break
        if chosen is None:
            # Nothing ready at `now`: idle this worker until the next finish.
            future = [f for f in finish.values() if f > now]
            if not future:  # deps outside this queue -- treat as satisfied
                chosen = pending[0]
            else:
                workers[w] = min(future)
                continue
        start = now
        end = start + chosen.est_sec
        workers[w] = end
        finish[chosen.id] = end
        sched[chosen.id] = (start, end)
        pending.remove(chosen)
    return max(workers), sched


# ---------------------------------------------------------------------------
# 4. Planning
# ---------------------------------------------------------------------------


def plan_9h(cfg: dict, t: dict, n_gpus: int, budget_sec: float) -> dict:
    trims_n = 0
    trims: list[dict] = []
    while True:
        core = build_queue(cfg, "9h", (), trims)
        attach_estimates(core, t, cfg, n_gpus)
        core_span, _ = simulate(core, n_gpus)
        if core_span <= budget_sec or trims_n >= len(cfg["mode_9h"]["trim_ladder"]):
            break
        trims_n += 1
        trims = apply_trims(cfg, trims_n)
    core_fits = core_span <= budget_sec

    ups: list[str] = []
    span = core_span
    rejected = None
    if core_fits:
        for up in upgrade_order(cfg):
            q = build_queue(cfg, "9h", ups + [up], trims)
            attach_estimates(q, t, cfg, n_gpus)
            s, _ = simulate(q, n_gpus)
            if s <= budget_sec:
                ups.append(up)
                span = s
            else:
                rejected = (up, s)
                break
    return {"upgrades": ups, "trims": trims, "core_span": core_span, "span": span,
            "core_fits": core_fits, "first_rejected": rejected}


def main() -> int:
    ap = argparse.ArgumentParser(description="Time the hardware and fit the queue to a budget.")
    ap.add_argument("--mode", choices=["9h", "full"], default="9h")
    ap.add_argument("--hours", type=float, help="budget (default: mode_9h.budget_hours)")
    ap.add_argument("--gpus", type=int, help="GPUs the queue will use (default: all visible)")
    ap.add_argument("--steps", type=int, default=200, help="timed steps per model (incl. skip)")
    ap.add_argument("--skip", type=int, default=20, help="warmup steps discarded")
    ap.add_argument("--e4-steps", type=int, default=50)
    ap.add_argument("--skip-e4", action="store_true", help="use reference E4 timings instead")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--reuse-timings", action="store_true",
                    help="re-plan from results/timings.json without re-timing")
    args = ap.parse_args()

    cfg = load_base_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = args.gpus or max(1, torch.cuda.device_count())
    budget_h = float(args.hours if args.hours is not None else cfg["mode_9h"]["budget_hours"])
    budget_sec = budget_h * 3600
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = max(2, min(int(cfg["data"].get("num_workers", 4)),
                                 (os.cpu_count() or 4) // n_gpus))
    ref = load_config("configs/compute_reference.yaml")
    timings_path = results_path() / "timings.json"

    if args.reuse_timings and timings_path.exists():
        t = load_json(timings_path)
        print(f"reusing {timings_path}")
    else:
        dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(f"timing on {dev_name} | {args.steps - args.skip} timed steps/model "
              f"(+{args.skip} warmup) | loader workers {num_workers}\n")
        t = {"device": dev_name, "n_gpus_visible": torch.cuda.device_count(),
             "num_workers": num_workers, "models": {}, "e4": {},
             "per_run_overhead_sec": float(ref["overheads"]["per_run_seconds"]),
             "strong_overhead": float(ref["overheads"]["strong_augmentation"]),
             "multi_gpu_efficiency": float(ref["overheads"]["multi_gpu_efficiency"]),
             "measured_at": dt.datetime.now().isoformat(timespec="seconds")}
        for m in MODEL_ORDER:
            r = time_training(m, load_config(f"configs/models/{m}.yaml"), cfg, device,
                              args.steps, args.skip, num_workers)
            t["models"][m] = r
            print(f"  {m:<15} {r['sec_per_step'] * 1000:8.1f} ms/step  {r['imgs_per_sec']:8.0f} img/s"
                  f"  val {r['val_sec']:6.1f}s  -> {r['sec_per_step'] * FULL_EPOCH_STEPS + r['val_sec']:7.1f} s/epoch"
                  + (f"  peak {r['peak_mem_mb']:.0f} MB" if r["peak_mem_mb"] else ""), flush=True)
        t["strong_loader_sec_per_batch"] = time_loader(cfg, "strong", num_workers)
        t["standard_loader_sec_per_batch"] = time_loader(cfg, "standard", num_workers)
        print(f"  loader alone: standard {128 / t['standard_loader_sec_per_batch']:.0f} img/s, "
              f"strong {128 / t['strong_loader_sec_per_batch']:.0f} img/s")

        for name in cfg["experiments"]["e4_pretrained"]["models"]:
            key = name.split(".")[0]
            if args.skip_e4:
                ips = float(ref["e4_224px"]["train_imgs_per_sec"]["t4"].get(key, 120.0))
                t["e4"][key] = {"sec_per_step": 64 / ips, "val_sec": VAL_IMAGES / (3 * ips),
                                "source": "reference (t4)"}
                continue
            r = time_training(f"timm/{name}", {"pretrained": False, "image_size": 224}, cfg,
                              device, args.e4_steps, max(1, args.e4_steps // 5), num_workers,
                              batch_size=64)
            t["e4"][key] = r
            print(f"  {key:<24} {r['sec_per_step'] * 1000:8.1f} ms/step @224  val {r['val_sec']:.1f}s",
                  flush=True)
        save_json(t, timings_path)

    # ------------------------------------------------------------------
    now = dt.datetime.now()
    if args.mode == "full":
        jobs = build_queue(cfg, "full")
        attach_estimates(jobs, t, cfg, n_gpus)
        span, _ = simulate(jobs, n_gpus)
        by: dict[str, float] = {}
        for j in jobs:
            by[j.exp] = by.get(j.exp, 0.0) + j.est_sec
        print(f"\nFULL MODE on {n_gpus}x {t['device']}:")
        for k, v in by.items():
            print(f"  {k:<10} {v / 3600:8.1f} GPU-h")
        print(f"  {'TOTAL':<10} {sum(by.values()) / 3600:8.1f} GPU-h  "
              f"(~{span / 3600:.1f} h wall on {n_gpus} GPU(s))")
        save_json({"mode": "full", "upgrades": [], "trims": [], "budget_hours": None,
                   "n_gpus": n_gpus, "estimates": {j.id: j.est_sec for j in jobs},
                   "timings_file": str(timings_path.relative_to(PROJECT_ROOT)),
                   "projected_span_sec": span}, results_path() / "queue_full.json")
        if sum(by.values()) / 3600 > 60:
            print("\n!! Over ~60 GPU-hours (spec Section 9): ask before proceeding. Sanctioned "
                  "trims are fewer E2 seeds or fewer sweep LRs -- NOT fewer E1 epochs.")
        return 0

    res = plan_9h(cfg, t, n_gpus, budget_sec)
    jobs = build_queue(cfg, "9h", res["upgrades"], res["trims"])
    attach_estimates(jobs, t, cfg, n_gpus)
    span, sched = simulate(jobs, n_gpus)

    print(f"\n9H PLAN | {n_gpus}x {t['device']} | budget {budget_h:g} h\n")
    print(f"  {'job':<36}{'tier':<20}{'est':>10}{'start':>8}{'end':>8}")
    for j in jobs:
        st, en = sched.get(j.id, (0.0, 0.0))
        mark = "  (done)" if job_done(j) else ""
        start = f"{now + dt.timedelta(seconds=st):%H:%M}"
        end = f"{now + dt.timedelta(seconds=en):%H:%M}"
        print(f"  {j.id:<36}{j.tier:<20}{human_time(j.est_sec):>10}{start:>8}{end:>8}{mark}")

    print(f"\n  core alone:        {human_time(res['core_span'])}")
    print(f"  with upgrades:     {human_time(span)}  "
          f"(projected finish {(now + dt.timedelta(seconds=span)):%a %H:%M} if started now)")
    print(f"  upgrades included: {', '.join(res['upgrades']) or 'none'}")
    if res["first_rejected"]:
        up, s = res["first_rejected"]
        print(f"  next upgrade '{up}' would take the queue to {human_time(s)} -- over budget")
    if res["trims"]:
        print("  TRIMS APPLIED (core did not fit): "
              + ", ".join(f"{x['target']} {x['from']} -> {x['to']}" for x in res["trims"]))
    if not res["core_fits"]:
        print("\n!! EVEN THE TRIMMED CORE DOES NOT FIT THE BUDGET. E1 is never cut below 100 "
              "epochs, so either raise --hours, add a GPU, or accept that the tail of the core "
              "queue (E2, E4) may not finish.")

    plan = {
        "mode": "9h", "budget_hours": budget_h, "n_gpus": n_gpus,
        "created": now.isoformat(timespec="seconds"), "device": t["device"],
        "upgrades": res["upgrades"], "trims": res["trims"], "core_fits": res["core_fits"],
        "core_span_sec": res["core_span"], "projected_span_sec": span,
        "projected_finish": (now + dt.timedelta(seconds=span)).isoformat(timespec="minutes"),
        "timings_file": str(timings_path.relative_to(PROJECT_ROOT)),
        "estimates": {j.id: round(j.est_sec, 1) for j in jobs},
        "schedule": {k: [round(a, 1), round(b, 1)] for k, (a, b) in sched.items()},
    }
    out = results_path() / "queue_9h.json"
    save_json(plan, out)
    print(f"\nsaved {out.relative_to(PROJECT_ROOT)} -- review it, then launch:\n"
          f"  python scripts/run_all.py --mode 9h")
    return 0


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    sys.exit(main())

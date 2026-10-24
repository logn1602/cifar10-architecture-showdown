"""Run the whole queue, unattended (spec 7 "run order" and 12.5).

    python scripts/run_all.py --mode 9h        # the 9-hour day (uses results/queue_9h.json)
    python scripts/run_all.py --mode full      # full protocol (needs confirmed LRs for E1+)
    python scripts/run_all.py --mode 9h --dry-run

Re-running the same command resumes: finished runs are skipped, partial runs
continue from ``last.pt``. Launch it in its own terminal (or a committed Kaggle
run); it does not depend on any agent session staying open, and it never
prompts.

In 9h mode the plan -- which upgrades are in, which trims were applied, and a
time estimate per job -- comes from ``scripts/plan_budget.py``. Without a plan
file the core queue runs with no upgrades and no budget checks.

Internally each job runs in a child process (``--run-job``), so one crashed or
diverged run cannot take the rest of the queue down with it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import load_base_config, results_path  # noqa: E402
from src.runner import Coordinator, jobs_from_plan, plan_config, run_job  # noqa: E402
from src.utils import PROJECT_ROOT, ensure_dir, human_time, load_json, save_json  # noqa: E402



def default_plan_path(mode: str) -> Path:
    return results_path() / f"queue_{mode}.json"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the experiment queue unattended.")
    ap.add_argument("--mode", choices=["9h", "full"], default="9h")
    ap.add_argument("--plan", help="plan JSON (default results/queue_<mode>.json)")
    ap.add_argument("--budget-hours", type=float,
                    help="override the plan's budget (9h mode); upgrades are dropped to fit")
    ap.add_argument("--gpus", type=int, help="number of GPUs to use (default: all visible)")
    ap.add_argument("--num-workers", type=int,
                    help="DataLoader workers per job (default: CPUs split across GPU workers)")
    ap.add_argument("--exp", nargs="+", help="only these experiments (e.g. lr_sweep E1)")
    ap.add_argument("--models", nargs="+", help="only these models")
    ap.add_argument("--dry-run", action="store_true", help="print the queue and exit")
    ap.add_argument("--run-job", help=argparse.SUPPRESS)   # internal: child process
    return ap.parse_args(argv)


def _child(args) -> int:
    """Run exactly one job; the exit code tells the coordinator what happened."""
    plan = load_json(args.plan)
    cfg = plan_config(plan)
    jobs = {j.id: j for j in jobs_from_plan(plan, cfg)}
    if args.run_job not in jobs:
        print(f"unknown job {args.run_job!r}", flush=True)
        return 2
    try:
        return run_job(jobs[args.run_job], cfg, plan["mode"], num_workers=args.num_workers)
    except Exception:
        traceback.print_exc()
        return 1


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.run_job:
        return _child(args)

    import os

    import torch

    cfg = load_base_config()
    plan_path = Path(args.plan) if args.plan else default_plan_path(args.mode)
    if plan_path.exists():
        plan = load_json(plan_path)
        if plan.get("mode") != args.mode:
            raise SystemExit(f"{plan_path} is a {plan.get('mode')!r} plan, not {args.mode!r}")
        print(f"plan: {plan_path} (upgrades: {', '.join(plan.get('upgrades', [])) or 'none'}; "
              f"trims: {len(plan.get('trims', []))})")
    else:
        plan = {"mode": args.mode, "upgrades": [], "trims": [], "estimates": {},
                "budget_hours": None}
        print(f"no plan file at {plan_path} -- running the core queue with no upgrades "
              "and no time estimates" + (" (run scripts/plan_budget.py first for 9h mode)"
                                         if args.mode == "9h" else ""))
    if args.budget_hours is not None:
        plan["budget_hours"] = args.budget_hours

    if plan.get("cfg_overrides"):
        print("!! plan carries cfg_overrides -- this is a TEST queue, not the protocol")
        cfg = plan_config(plan)
    all_jobs = jobs_from_plan(plan, cfg)
    # Evaluation jobs have no single model -- they run over whichever E1
    # checkpoints exist -- so a --models filter keeps them.
    jobs = [j for j in all_jobs
            if (not args.exp or j.exp in args.exp)
            and (not args.models or j.model is None or j.model in args.models)]

    n_gpu = torch.cuda.device_count()
    if args.gpus is not None:
        n_gpu = min(n_gpu, args.gpus)
    gpu_ids: list[int | None] = list(range(n_gpu)) if n_gpu else [None]

    num_workers = args.num_workers
    if num_workers is None:
        # Two training jobs share the host's CPUs on a 2-GPU box; give each its
        # share rather than letting 2x4 workers fight over 4 cores.
        base = int(cfg["data"].get("num_workers", 4))
        num_workers = max(2, min(base, (os.cpu_count() or 4) // len(gpu_ids)))

    if args.dry_run:
        print(f"\n{len(jobs)} jobs, {len(gpu_ids)} worker(s), {num_workers} loader workers/job\n")
        tot = 0.0
        from src.experiments import job_done
        for j in jobs:
            tot += j.est_sec
            done = "done" if job_done(j) else ""
            print(f"  {j.id:<34} {j.tier:<20} est {human_time(j.est_sec):>10}  {done}")
        print(f"\n  total estimated: {human_time(tot)} of work "
              f"(~{human_time(tot / len(gpu_ids))} wall on {len(gpu_ids)} worker(s))")
        return 0

    # The child processes rebuild the queue from this exact file.
    active = ensure_dir(results_path() / "queue_state") / f"active_plan_{args.mode}.json"
    save_json(plan, active)

    coord = Coordinator(jobs, active, args.mode, gpu_ids, plan.get("budget_hours"),
                        num_workers, all_jobs=all_jobs,
                        log=lambda m: print(f"[queue {dt.datetime.now():%H:%M:%S}] {m}",
                                            flush=True))
    try:
        s = coord.run()
    finally:
        summary_path = results_path() / "RUN_SUMMARY.md"
        coord.write_run_summary(summary_path, plan)
        print(f"\nwrote {summary_path}")
    print(f"\nqueue finished: {s['counts']} in {human_time(s['elapsed_sec'])}")
    return 0


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    sys.exit(main())

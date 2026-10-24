"""The unattended queue: run jobs across GPUs, survive failures, keep a log.

Spec 12.5, rule by rule, and where each lives:

* **No questions, no approval gates** -- nothing here reads stdin.
* **Failures don't stop the queue** -- every job runs in its *own child
  process*. A NaN, a Python exception, a CUDA OOM or even a hard crash of the
  CUDA context kills that child only; the coordinator logs the tail of its
  output to ``results/failures/<job>.log``, marks it failed, and moves on.
  Jobs that depend on a failed job are marked skipped rather than attempted.
* **Never auto-tweak hyperparameters** -- a diverged run is recorded as
  ``diverged``; nothing is retried.
* **Re-plan as you go** -- after every job the actual/estimated time ratio is
  updated, and an *upgrade* job is only started if the projected finish, with
  that ratio applied, still fits the budget. Core jobs are never dropped.
* **Resumable** -- a job whose ``metrics.json`` (or ``done.json``) exists is
  skipped; a partial one resumes from ``last.pt``.
* **Multi-GPU aware** -- one worker thread per GPU, each launching its jobs
  with ``CUDA_VISIBLE_DEVICES`` pinned, pulling the next ready job from a shared
  queue that respects dependencies (E1 waits for its model's sweep, E5-E7 wait
  for E1).
* **Progress log + final summary** -- ``results/progress.log`` after every
  job, ``results/RUN_SUMMARY.md`` at the end.
"""

from __future__ import annotations

import collections
import datetime as dt
import importlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

try:
    from src.experiments import (
        EVAL_RUNNERS, Job, LRNotConfirmed, build_queue, job_done, load_base_config, resolve_lr,
        results_dir, results_path,
    )
    from src.utils import (
        PROJECT_ROOT, deep_merge, ensure_dir, human_time, load_config, load_json, save_json,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.experiments import (  # noqa: E402
        EVAL_RUNNERS, Job, LRNotConfirmed, build_queue, job_done, load_base_config, resolve_lr,
        results_dir, results_path,
    )
    from src.utils import (  # noqa: E402
        PROJECT_ROOT, deep_merge, ensure_dir, human_time, load_config, load_json, save_json,
    )

__all__ = ["Coordinator", "run_job", "jobs_from_plan", "plan_config", "EXIT"]

# Child-process exit codes -> job status.
EXIT = {0: "done", 3: "diverged", 4: "not_built", 5: "blocked"}
TERMINAL_BAD = {"failed", "diverged", "skipped", "not_built", "blocked"}


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def plan_config(plan: dict) -> dict:
    """base.yaml plus the plan's ``cfg_overrides``.

    Overrides exist for one purpose: shrinking every run so the *real* queue
    machinery -- child processes, dependencies, LR auto-pick, resume, failure
    isolation -- can be exercised end to end on a CPU in minutes. A real plan
    from ``plan_budget.py`` never sets them.
    """
    cfg = load_base_config()
    if plan.get("cfg_overrides"):
        cfg = deep_merge(cfg, plan["cfg_overrides"])
    return cfg


def jobs_from_plan(plan: dict, cfg: dict | None = None) -> list[Job]:
    """Rebuild the exact job list a plan describes, with its estimates."""
    cfg = cfg or plan_config(plan)
    jobs = build_queue(cfg, plan["mode"], plan.get("upgrades", []), plan.get("trims", []),
                       models=tuple(plan.get("models") or _all_models()))
    est = plan.get("estimates", {})
    for j in jobs:
        j.est_sec = float(est.get(j.id, 0.0))
    return jobs


def _all_models() -> tuple[str, ...]:
    from src.models import MODEL_ORDER
    return MODEL_ORDER


# ---------------------------------------------------------------------------
# One job (runs inside the child process)
# ---------------------------------------------------------------------------


def run_job(job: Job, cfg: dict, mode: str, num_workers: int | None = None) -> int:
    """Execute one job in this process. Returns a child exit code (see EXIT)."""
    from src.train import DivergenceError, train

    try:
        if job.kind == "eval":
            mod_name, fn_name = EVAL_RUNNERS[job.exp].split(":")
            try:
                fn = getattr(importlib.import_module(mod_name), fn_name)
            except ModuleNotFoundError as exc:
                # Only *our* module missing means "not built yet". A missing
                # third-party package inside it is a real failure.
                if exc.name is None or not mod_name.startswith(exc.name):
                    raise
                print(f"{job.id}: evaluation code not built yet ({exc})", flush=True)
                return 4
            except AttributeError as exc:
                print(f"{job.id}: evaluation code not built yet ({exc})", flush=True)
                return 4
            fn(cfg=cfg, mode=mode, out_dir=job.out_dir)
            save_json({"job": job.id, "status": "completed",
                       "finished": dt.datetime.now().isoformat(timespec="seconds")},
                      Path(PROJECT_ROOT, job.out_dir, "done.json"))
            return 0

        lr = resolve_lr(job, cfg, mode)
        if job.model.startswith("timm/"):
            model_cfg = {"pretrained": True, "image_size": job.image_size}
        else:
            model_cfg = load_config(f"configs/models/{job.model}.yaml")
        root = results_dir()
        tb_dir = Path(PROJECT_ROOT, job.out_dir.replace(f"{root}/runs", f"{root}/tb", 1))
        train(
            job.model, model_cfg, cfg["train"], cfg["data"], job.seed,
            PROJECT_ROOT / job.out_dir,
            experiment=job.exp, lr=lr, epochs=job.epochs, max_steps=job.max_steps,
            warmup_fraction=job.warmup_fraction, fraction=job.fraction,
            augmentation=job.augmentation, aug_cfg=cfg["augmentation"].get(job.augmentation),
            batch_size=job.batch_size, save_checkpoints=job.save_checkpoints,
            keep_last=job.keep_last, evaluate_test=job.evaluate_test, resume=True,
            tb_dir=tb_dir, num_workers=num_workers, ece_bins=int(cfg["eval"]["ece_bins"]),
            extra_metrics={"job_id": job.id, "mode": mode, "tier": job.tier},
        )
        return 0
    except DivergenceError as exc:
        print(f"DIVERGED: {exc}", flush=True)
        return 3
    except LRNotConfirmed as exc:
        print(f"BLOCKED: {exc}", flush=True)
        return 5


# ---------------------------------------------------------------------------
# Coordinator (parent process)
# ---------------------------------------------------------------------------


class Coordinator:
    """Run ``jobs`` on ``n_workers`` workers within ``budget_hours``."""

    def __init__(self, jobs: list[Job], plan_path: Path, mode: str, gpu_ids: list[int | None],
                 budget_hours: float | None, num_workers: int | None,
                 all_jobs: list[Job] | None = None, log=print) -> None:
        self.jobs = jobs
        self.by_id = {j.id: j for j in (all_jobs or jobs)}
        self.plan_path = plan_path
        self.mode = mode
        self.gpu_ids = gpu_ids
        self.budget_sec = budget_hours * 3600 if budget_hours else None
        self.num_workers = num_workers
        self.log = log
        self.lock = threading.Lock()
        self.status: dict[str, str] = {}
        self.reason: dict[str, str] = {}
        self.actual: dict[str, float] = {}
        self.started_at: dict[str, float] = {}
        self.ratios: list[float] = []
        self.t0 = time.time()
        self.stop = threading.Event()

        for j in jobs:
            if job_done(j):
                self.status[j.id] = "done"
                self.reason[j.id] = "already complete (skipped)"
            elif j.kind == "train" and Path(PROJECT_ROOT, j.out_dir, "diverged.json").exists():
                # Never retried: same config, same seed -> same divergence, and
                # spec Section 3 forbids patching hyperparameters to rescue it.
                self.status[j.id] = "diverged"
                self.reason[j.id] = "diverged in an earlier launch (not retried)"
            else:
                self.status[j.id] = "pending"
        self.preexisting = {j.id for j in jobs if self.status[j.id] == "done"}

    # -- budget -------------------------------------------------------------

    def ratio(self) -> float:
        """Median actual/estimated time so far, clipped to a sane range."""
        if not self.ratios:
            return 1.0
        return min(3.0, max(0.5, statistics.median(self.ratios)))

    def remaining_core_sec(self) -> float:
        r = self.ratio()
        now = time.time()
        total = 0.0
        for j in self.jobs:
            st = self.status[j.id]
            if j.tier != "core":
                continue
            if st == "pending":
                total += j.est_sec * r
            elif st == "running":
                total += max(0.0, j.est_sec * r - (now - self.started_at[j.id]))
        return total

    def fits(self, job: Job) -> bool:
        if self.budget_sec is None or job.est_sec <= 0:
            return True
        elapsed = time.time() - self.t0
        n = max(1, len(self.gpu_ids))
        need = job.est_sec * self.ratio()
        # Every other upgrade already running also occupies a worker.
        now = time.time()
        running_up = sum(max(0.0, j.est_sec * self.ratio() - (now - self.started_at[j.id]))
                         for j in self.jobs if self.status[j.id] == "running" and j.tier != "core")
        projected = elapsed + (self.remaining_core_sec() + running_up + need) / n
        return projected <= self.budget_sec and elapsed + need <= self.budget_sec

    def projected_finish(self) -> dt.datetime:
        r = self.ratio()
        now = time.time()
        work = 0.0
        for j in self.jobs:
            st = self.status[j.id]
            if st == "pending":
                work += j.est_sec * r
            elif st == "running":
                work += max(0.0, j.est_sec * r - (now - self.started_at[j.id]))
        return dt.datetime.now() + dt.timedelta(seconds=work / max(1, len(self.gpu_ids)))

    # -- scheduling ---------------------------------------------------------

    def _dep_state(self, dep_id: str) -> str:
        if dep_id in self.status:
            return self.status[dep_id]
        dep = self.by_id.get(dep_id)       # a dep outside this (filtered) queue
        return "done" if dep is not None and job_done(dep) else "blocked"

    def next_job(self) -> Job | str | None:
        """A ready job, ``"wait"`` if work remains but none is ready, or None."""
        with self.lock:
            any_pending = False
            for j in self.jobs:
                if self.status[j.id] != "pending":
                    continue
                deps = [self._dep_state(d) for d in j.deps]
                if j.soft_deps:
                    # Run once every dep has finished, whatever its outcome.
                    deps = ["done" if s in TERMINAL_BAD else s for s in deps]
                bad = [d for d, s in zip(j.deps, deps) if s in TERMINAL_BAD]
                if bad:
                    self._finish(j, "skipped", f"dependency not completed: {bad[0]} "
                                               f"({self._dep_state(bad[0])})")
                    continue
                if any(s != "done" for s in deps):
                    any_pending = True
                    continue
                if j.tier != "core" and not self.fits(j):
                    self._finish(j, "skipped", f"dropped: over budget (upgrade {j.tier}, "
                                               f"est {human_time(j.est_sec * self.ratio())})")
                    continue
                self.status[j.id] = "running"
                self.started_at[j.id] = time.time()
                return j
            return "wait" if any_pending or any(s == "running" for s in self.status.values()) \
                else None

    def _finish(self, job: Job, status: str, reason: str = "") -> None:
        self.status[job.id] = status
        if reason:
            self.reason[job.id] = reason

    # -- workers ------------------------------------------------------------

    def _launch(self, job: Job, gpu: int | None, tag: str) -> tuple[int, list[str]]:
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_all.py"),
               "--run-job", job.id, "--plan", str(self.plan_path)]
        if self.num_workers is not None:
            cmd += ["--num-workers", str(self.num_workers)]
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        tail: collections.deque[str] = collections.deque(maxlen=400)
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        for line in proc.stdout:
            tail.append(line)
            sys.stdout.write(f"{tag} {line}")
            sys.stdout.flush()
        proc.wait()
        return proc.returncode, list(tail)

    def _worker(self, idx: int) -> None:
        gpu = self.gpu_ids[idx]
        tag = f"[gpu{gpu}]" if gpu is not None else "[cpu]"
        while not self.stop.is_set():
            nxt = self.next_job()
            if nxt is None:
                return
            if nxt == "wait":
                time.sleep(10)
                continue
            job: Job = nxt
            self.log(f"{tag} START {job.id} (est {human_time(job.est_sec * self.ratio())})")
            t = time.time()
            try:
                code, tail = self._launch(job, gpu, tag)
            except Exception:
                code, tail = 1, [traceback.format_exc()]
            secs = time.time() - t
            status = EXIT.get(code, "failed")
            reason = ""
            if status != "done":
                fail_log = ensure_dir(results_path() / "failures") / f"{job.slug}.log"
                fail_log.write_text(
                    f"job: {job.id}\nstatus: {status} (exit {code})\nduration: {human_time(secs)}\n"
                    f"{'=' * 70}\n" + "".join(tail), encoding="utf-8")
                reason = f"exit {code}; see {results_dir()}/failures/{fail_log.name}"
            with self.lock:
                self._finish(job, status, reason)
                self.actual[job.id] = secs
                if status == "done" and job.est_sec > 0 and job.kind == "train":
                    self.ratios.append(secs / job.est_sec)
            self._progress(job, status, secs)

    def _progress(self, job: Job, status: str, secs: float) -> None:
        val = test = "-"
        mp = Path(PROJECT_ROOT, job.out_dir, "metrics.json")
        if status == "done" and mp.exists():
            m = load_json(mp)
            val = f"{m['best_val_acc']:.4f}" if m.get("best_val_acc") is not None else "-"
            test = f"{m['test_acc']:.4f}" if m.get("test_acc") is not None else "-"
        with self.lock:
            left = sum(1 for s in self.status.values() if s in ("pending", "running"))
            finish = self.projected_finish()
        line = (f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} | {status:<9} | {job.id:<28} | "
                f"val {val} | test {test} | {human_time(secs)} | {left} left | "
                f"projected finish {finish:%H:%M} | ratio {self.ratio():.2f}")
        self.log(line)
        with (results_path() / "progress.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def run(self) -> dict:
        ensure_dir(results_path())
        n_todo = sum(1 for s in self.status.values() if s == "pending")
        n_div = sum(1 for s in self.status.values() if s == "diverged")
        self.log(f"queue: {len(self.jobs)} jobs, {len(self.preexisting)} already complete, "
                 f"{n_div} previously diverged, "
                 f"{n_todo} to run on {len(self.gpu_ids)} worker(s)"
                 + (f", budget {human_time(self.budget_sec)}" if self.budget_sec else ""))
        with (results_path() / "progress.log").open("a", encoding="utf-8") as fh:
            fh.write(f"# {dt.datetime.now():%Y-%m-%d %H:%M:%S} queue start: mode={self.mode} "
                     f"jobs={len(self.jobs)} todo={n_todo} workers={len(self.gpu_ids)}\n")
        threads = [threading.Thread(target=self._worker, args=(i,), daemon=True)
                   for i in range(len(self.gpu_ids))]
        for th in threads:
            th.start()
        try:
            while any(th.is_alive() for th in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop.set()
            self.log("interrupted -- running jobs will finish their current child process "
                     "if not killed; re-run the same command to resume")
            raise
        return self.summary()

    # -- summary ------------------------------------------------------------

    def summary(self) -> dict:
        counts = collections.Counter(self.status.values())
        tiers = collections.defaultdict(lambda: collections.Counter())
        for j in self.jobs:
            tiers[j.tier][self.status[j.id]] += 1
        return {"counts": dict(counts), "tiers": {k: dict(v) for k, v in tiers.items()},
                "elapsed_sec": time.time() - self.t0,
                "gpu_sec": sum(self.actual.values()), "ratio": self.ratio()}

    def write_run_summary(self, path: Path, plan: dict) -> None:
        s = self.summary()
        lines = [
            "# Run summary", "",
            f"_Written by `scripts/run_all.py` at {dt.datetime.now():%Y-%m-%d %H:%M}. "
            "Do not edit by hand._", "",
            f"- **Mode:** `{self.mode}`",
            f"- **Workers:** {len(self.gpu_ids)} ({', '.join(str(g) for g in self.gpu_ids)})",
            f"- **Wall-clock this session:** {human_time(s['elapsed_sec'])}",
            f"- **GPU time this session (sum over workers):** {human_time(s['gpu_sec'])}",
            f"- **Actual / estimated time ratio:** {s['ratio']:.2f}",
            f"- **Budget:** {human_time(self.budget_sec) if self.budget_sec else 'none'}",
            f"- **Trims applied:** {', '.join(t['target'] + ' -> ' + str(t['to']) for t in plan.get('trims', [])) or 'none'}",
            "", "## Totals", "",
            "| status | jobs |", "|---|---|",
        ]
        for k, v in sorted(s["counts"].items()):
            lines.append(f"| {k} | {v} |")
        lines += ["", "## By tier (core + upgrades, spec 12.3 order)", "",
                  "| tier | done | other |", "|---|---|---|"]
        for tier, c in s["tiers"].items():
            other = ", ".join(f"{k}: {v}" for k, v in c.items() if k != "done") or "-"
            lines.append(f"| {tier} | {c.get('done', 0)} | {other} |")
        planned_ups = plan.get("upgrades", [])
        made = [u for u in planned_ups if s["tiers"].get(u, {}).get("done", 0) ==
                sum(s["tiers"].get(u, {}).values()) and s["tiers"].get(u)]
        lines += ["", f"**Upgrades planned:** {', '.join(planned_ups) or 'none'}  ",
                  f"**Upgrades fully completed:** {', '.join(made) or 'none'}", ""]

        not_done = [j for j in self.jobs if self.status[j.id] != "done"]
        if not_done:
            lines += ["## Jobs that did not complete", "", "| job | status | reason |",
                      "|---|---|---|"]
            for j in not_done:
                lines.append(f"| `{j.id}` | {self.status[j.id]} | {self.reason.get(j.id, '')} |")
            lines.append("")

        lines += ["## Every job", "", "| job | tier | status | time | note |", "|---|---|---|---|---|"]
        for j in self.jobs:
            t = human_time(self.actual[j.id]) if j.id in self.actual else "-"
            lines.append(f"| `{j.id}` | {j.tier} | {self.status[j.id]} | {t} | "
                         f"{self.reason.get(j.id, '')} |")

        caveats = ["", "## Caveats for the write-up (spec 12.7)", ""]
        if self.mode == "9h":
            caveats += [
                "- 1 seed for most results: differences under ~1 percentage point between "
                "models are **not** meaningful.",
                "- 100 epochs, not 200: Transformers and MLP-Mixer usually keep improving with "
                "longer training, so this setup slightly **favours the CNNs**.",
                "- E2 points are step-matched but short, so absolute E2 accuracies sit below E1's.",
                "- LRs were auto-picked from a 10-epoch sweep (`results/lr_choice_9h.json`), "
                "not confirmed by hand.",
            ]
        if any(self.status[j.id] == "diverged" for j in self.jobs):
            caveats.append("- At least one run **diverged**; it was stopped and not retried "
                           "(see the table above).")
        lines += caveats
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""Every run of every experiment, as a list of jobs.

The planner (``scripts/plan_budget.py``), the unattended queue
(``scripts/run_all.py``) and the per-experiment entry points
(``scripts/run_experiment.py``, ``scripts/lr_sweep.py``) all build their work
from :func:`build_queue`, so there is exactly one definition of "what E2 is" in
the repo. A job is a plain record: which model, which seed, how many steps,
where its outputs go, and which other jobs must finish first.

Two modes (spec Sections 6 and 12.2):

* ``full`` -- 200-epoch runs, 3 seeds, the 5-LR sweep. The learning rate of
  every post-sweep run comes from ``configs/models/<model>.yaml`` and must have
  been **confirmed by the owner** (spec Phase 3: "ask me to confirm the chosen
  LRs before Phase 4"). Until it is, those jobs report "blocked".
* ``9h`` -- the unattended day. Trimmed seeds/epochs, best-val LR auto-picked
  from the sweep, upgrades added in the spec 12.3 order when the budget allows.
  Warmup is 5% of total steps in every run (spec 12.2).

Job ids look like ``E2/vit/f0.1/s0`` and are stable across machines, so a
re-launched queue recognises finished work by its ``metrics.json``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from src.models import MODEL_ORDER
    from src.utils import PROJECT_ROOT, load_config, load_json, resolve_path, save_json
except ImportError:  # pragma: no cover - `python src/experiments.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.models import MODEL_ORDER  # noqa: E402
    from src.utils import PROJECT_ROOT, load_config, load_json, resolve_path, save_json  # noqa: E402

__all__ = [
    "Job", "build_queue", "mode_params", "apply_trims", "upgrade_order",
    "resolve_lr", "pick_sweep_lrs", "sweep_table", "LRNotConfirmed", "e4_key",
    "EVAL_RUNNERS", "job_done", "load_base_config", "results_dir", "results_path",
]

def upgrade_order(cfg: dict) -> tuple[str, ...]:
    """Spec 12.3 upgrade names, in priority order (base.yaml ``mode_9h.upgrades``)."""
    return tuple(u["name"] for u in cfg["mode_9h"]["upgrades"])

# Evaluation-only experiments run on finished E1 checkpoints. Resolved lazily
# (module:function) so the training queue does not depend on them importing.
EVAL_RUNNERS = {
    "E5": "src.corruptions:run_e5",
    "E6": "src.corruptions:run_e6",
    "E7": "src.interpret:run_e7",
}


class LRNotConfirmed(RuntimeError):
    """Full mode: the owner has not yet confirmed this model's swept LR."""


@dataclass
class Job:
    id: str
    exp: str                         # lr_sweep | E1 | E2 | E3 | E4_sweep | E4 | E5 | E6 | E7
    kind: str = "train"              # train | eval
    tier: str = "core"               # core | a mode_9h.upgrades name
    model: str | None = None         # registry key, or "timm/<name>" for E4
    seed: int = 0
    lr: float | None = None          # fixed LR; None -> resolved at run time
    epochs: int | None = None
    max_steps: int | None = None
    fraction: float = 1.0
    augmentation: str = "standard"
    warmup_fraction: float | None = None   # None -> protocol warmup_epochs
    batch_size: int | None = None
    save_checkpoints: bool = True
    keep_last: bool = False          # spec 12.5: keep only best.pt once finished
    evaluate_test: bool = True
    deps: list[str] = field(default_factory=list)
    # Soft deps only need to have *finished*, successfully or not. E5-E7 use
    # them: if one model's E1 run failed, robustness and interpretability
    # should still run on the other five rather than be skipped wholesale.
    soft_deps: bool = False
    out_dir: str = ""
    est_sec: float = 0.0
    image_size: int = 32

    @property
    def slug(self) -> str:
        return self.id.replace("/", "__")

    def to_dict(self) -> dict:
        return asdict(self)


def load_base_config() -> dict:
    return load_config("configs/base.yaml")


def e4_key(timm_name: str) -> str:
    """``vit_small_patch16_224.augreg_in21k_ft_in1k`` -> ``vit_small_patch16_224``."""
    return timm_name.split(".")[0]


def _fmt(x: float) -> str:
    return f"{x:g}"


# ---------------------------------------------------------------------------
# Mode parameters
# ---------------------------------------------------------------------------


def mode_params(cfg: dict, mode: str, trims: list[dict] | None = None) -> dict:
    """Flatten base.yaml into the numbers that define each experiment."""
    ex = cfg["experiments"]
    e4 = ex["e4_pretrained"]
    if mode == "full":
        p = {
            "sweep_lrs": list(ex["lr_sweep"]["lrs"]), "sweep_epochs": int(ex["lr_sweep"]["epochs"]),
            "auto_pick": False,
            "e1_epochs": int(cfg["train"]["epochs"]), "e1_seeds": list(cfg["train"]["seeds"]),
            "e2_fractions": list(ex["data_fractions"]), "e2_steps": int(ex["e2_total_steps"]),
            # Spec E2: 3 seeds for 10% and 100% (100% reuses E1), 1 for the rest.
            "e2_seeds": {float(k): list(v) for k, v in ex["e2_seeds_by_fraction"].items()},
            "e2_default_seeds": list(ex["e2_seeds_default"]),
            "e2_reuse_e1": bool(ex["e2_reuse_e1_for_full_data"]),
            "e3_epochs": int(cfg["train"]["epochs"]), "e3_models": list(MODEL_ORDER),
            "e4_lrs": list(e4["lrs"]), "e4_epochs": int(e4["epochs"]),
            "e4_seeds": list(cfg["train"]["seeds"]), "e4_sweep": True,
            "e4_warmup_epochs": int(e4["warmup_epochs"]),
            "warmup_fraction": None,          # epoch runs: 5 warmup epochs
        }
    elif mode == "9h":
        m = copy.deepcopy(cfg["mode_9h"])
        for t in trims or []:
            _set_dotted(m, t["target"], t["to"])
        p = {
            "sweep_lrs": list(m["lr_sweep"]["lrs"]), "sweep_epochs": int(m["lr_sweep"]["epochs"]),
            "auto_pick": bool(m["lr_sweep"].get("auto_pick", True)),
            "e1_epochs": int(m["e1"]["epochs"]), "e1_seeds": list(m["e1"]["seeds"]),
            "e2_fractions": list(m["e2"]["fractions"]), "e2_steps": int(m["e2"]["total_steps"]),
            "e2_seeds": {}, "e2_default_seeds": list(m["e2"]["seeds"]), "e2_reuse_e1": False,
            "e3_epochs": int(m["e3"]["epochs"]), "e3_models": [],
            "e4_lrs": [float(m["e4"]["lr"])], "e4_epochs": int(m["e4"]["epochs"]),
            "e4_seeds": list(m["e4"]["seeds"]), "e4_sweep": False, "e4_warmup_epochs": None,
            "warmup_fraction": float(m.get("warmup_fraction", 0.05)),
        }
    else:
        raise ValueError(f"mode must be 'full' or '9h', got {mode!r}")
    p["mode"] = mode
    p["e4_models"] = list(e4["models"])
    p["e4_batch_size"] = int(e4["batch_size"])
    p["e4_image_size"] = int(e4["image_size"])
    return p


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    """``"e2.total_steps"`` -> ``d["e2"]["total_steps"] = value``."""
    *head, last = dotted.split(".")
    for h in head:
        d = d[h]
    d[last] = value


def apply_trims(cfg: dict, n: int) -> list[dict]:
    """The first ``n`` rungs of the spec 12.4 trim ladder."""
    return list(cfg["mode_9h"]["trim_ladder"][:n])


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


def results_dir() -> str:
    """Repo-relative results root. ``SHOWDOWN_RESULTS`` redirects it -- used only
    by the CPU queue self-test, so its toy runs can never be mistaken for real
    ones by a later queue that skips anything with a ``metrics.json``."""
    return os.environ.get("SHOWDOWN_RESULTS", "results").replace("\\", "/").rstrip("/")


def results_path() -> Path:
    return resolve_path(results_dir())


def _run_dir(*parts: Any) -> str:
    return str(Path(results_dir(), "runs", *[str(p) for p in parts]).as_posix())


def build_queue(cfg: dict, mode: str, upgrades: list[str] | tuple[str, ...] = (),
                trims: list[dict] | None = None,
                models: list[str] | tuple[str, ...] = MODEL_ORDER) -> list[Job]:
    """Every job for ``mode``, in queue order (most important first).

    Order (spec 12.5): LR sweep -> E1 -> E7, E5, E6 -> E2 -> E4 -> upgrades.
    Full mode uses the same order with E3 placed after E2.
    """
    p = mode_params(cfg, mode, trims)
    wf = p["warmup_fraction"]
    jobs: list[Job] = []

    # 1. LR sweep -- no checkpoints, never touches test (spec 12.5 / Section 3).
    sweep_ids: dict[str, list[str]] = {}
    for m in models:
        sweep_ids[m] = []
        for lr in p["sweep_lrs"]:
            j = Job(id=f"lr_sweep/{m}/lr{_fmt(lr)}", exp="lr_sweep", model=m, lr=float(lr),
                    epochs=p["sweep_epochs"], warmup_fraction=wf, save_checkpoints=False,
                    evaluate_test=False, out_dir=_run_dir("lr_sweep", m, f"lr_{_fmt(lr)}"))
            jobs.append(j)
            sweep_ids[m].append(j.id)

    # In 9h mode the LR is auto-picked from the sweep, so a post-sweep run for
    # model m must wait for *all* of m's sweep runs to finish. The deps are
    # soft: a sweep point that diverged is a legitimate result (it is excluded
    # from the pick), not a reason to skip the model's E1/E2/E3. In full mode
    # the run waits for the owner's confirmation instead (checked at run time).
    def lr_deps(m: str) -> list[str]:
        return list(sweep_ids[m]) if mode == "9h" else []

    soft = mode == "9h"

    def e1(m: str, seed: int, tier: str) -> Job:
        return Job(id=f"E1/{m}/s{seed}", exp="E1", tier=tier, model=m, seed=seed,
                   epochs=p["e1_epochs"], warmup_fraction=wf, deps=lr_deps(m),
                   soft_deps=soft, out_dir=_run_dir("E1", m, seed))

    def e2(m: str, frac: float, seed: int, tier: str) -> Job:
        return Job(id=f"E2/{m}/f{_fmt(frac)}/s{seed}", exp="E2", tier=tier, model=m, seed=seed,
                   max_steps=p["e2_steps"], fraction=float(frac),
                   # E2 is step-defined: warmup is always 5% of its steps (spec E2).
                   warmup_fraction=wf if wf is not None else 0.05,
                   deps=lr_deps(m), soft_deps=soft,
                   out_dir=_run_dir("E2", m, f"frac_{_fmt(frac)}", seed))

    def e3(m: str, tier: str) -> Job:
        return Job(id=f"E3/{m}/s0", exp="E3", tier=tier, model=m, seed=0,
                   epochs=p["e3_epochs"], augmentation="strong", warmup_fraction=wf,
                   deps=lr_deps(m), soft_deps=soft, out_dir=_run_dir("E3", m, 0))

    # 2. E1 main showdown.
    core_e1_seeds = p["e1_seeds"]
    e1_jobs = [e1(m, s, "core") for s in core_e1_seeds for m in models]
    jobs += e1_jobs
    e1_seed0 = [j.id for j in e1_jobs if j.seed == 0]

    # 3. Evaluation-only experiments on the E1 checkpoints (cheap).
    for exp in ("E7", "E5", "E6"):
        jobs.append(Job(id=f"{exp}/eval", exp=exp, kind="eval", deps=list(e1_seed0),
                        soft_deps=True, out_dir=_run_dir(exp)))

    # 4. E2 data-hunger.
    for frac in p["e2_fractions"]:
        if frac >= 1.0 and p["e2_reuse_e1"]:
            continue  # full mode: the 100% point *is* E1 (same steps, same cadence)
        seeds = p["e2_seeds"].get(frac, p["e2_default_seeds"])
        for s in seeds:
            for m in models:
                jobs.append(e2(m, frac, s, "core"))

    # 5. E3 (full mode only in core; 9h adds it via upgrades).
    for m in p["e3_models"]:
        if m in models:
            jobs.append(e3(m, "core"))

    # 6. E4 pretrained.
    e4_sweep_ids: dict[str, list[str]] = {}
    for name in p["e4_models"]:
        key = e4_key(name)
        common = dict(model=f"timm/{name}", epochs=p["e4_epochs"],
                      batch_size=p["e4_batch_size"], image_size=p["e4_image_size"])
        warm = (wf if wf is not None
                else p["e4_warmup_epochs"] / max(1, p["e4_epochs"]))
        if p["e4_sweep"]:
            e4_sweep_ids[key] = []
            for lr in p["e4_lrs"]:
                j = Job(id=f"E4_sweep/{key}/lr{_fmt(lr)}", exp="E4_sweep", lr=float(lr),
                        warmup_fraction=warm, save_checkpoints=False, evaluate_test=False,
                        out_dir=_run_dir("E4_sweep", key, f"lr_{_fmt(lr)}"), **common)
                jobs.append(j)
                e4_sweep_ids[key].append(j.id)
        for s in p["e4_seeds"]:
            jobs.append(Job(id=f"E4/{key}/s{s}", exp="E4", seed=s,
                            lr=None if p["e4_sweep"] else float(p["e4_lrs"][0]),
                            warmup_fraction=warm, deps=e4_sweep_ids.get(key, []),
                            out_dir=_run_dir("E4", key, s), **common))

    # 7. Upgrades (9h only), in spec 12.3 order -- defined in base.yaml.
    if mode == "9h":
        for up in cfg["mode_9h"]["upgrades"]:
            name = up["name"]
            if name not in upgrades:
                continue
            ms = [m for m in (up.get("models") or models)
                  if m in models and m not in up.get("exclude_models", [])]
            if up["exp"] == "E3":
                jobs += [e3(m, name) for m in ms]
            elif up["exp"] == "E1":
                jobs += [e1(m, s, name) for s in up["seeds"] for m in ms]
            elif up["exp"] == "E2":
                fracs = up.get("fractions") or p["e2_fractions"]
                seeds = up.get("seeds") or p["e2_default_seeds"]
                jobs += [e2(m, f, s, name) for s in seeds for f in fracs for m in ms]
            else:
                raise ValueError(f"upgrade {name!r}: unsupported exp {up['exp']!r}")

    ids = [j.id for j in jobs]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise RuntimeError(f"duplicate job ids: {dupes}")
    return jobs


# ---------------------------------------------------------------------------
# Status + learning-rate resolution
# ---------------------------------------------------------------------------


def job_done(job: Job) -> bool:
    """A job is done when its completion marker exists."""
    out = resolve_path(job.out_dir)
    if job.kind == "eval":
        return (out / "done.json").exists()
    mp = out / "metrics.json"
    if not mp.exists():
        return False
    try:
        return load_json(mp).get("status") == "completed"
    except Exception:
        return False


def sweep_table(cfg: dict, mode: str, models: tuple[str, ...] = MODEL_ORDER) -> list[dict]:
    """One row per sweep run: model, lr, best/final val acc, status."""
    p = mode_params(cfg, mode)
    rows = []
    for m in models:
        for lr in p["sweep_lrs"]:
            out = resolve_path(_run_dir("lr_sweep", m, f"lr_{_fmt(lr)}"))
            row = {"model": m, "lr": float(lr), "status": "missing",
                   "best_val_acc": None, "final_val_acc": None, "best_epoch": None,
                   "epochs": p["sweep_epochs"], "train_time_sec": None}
            if (out / "metrics.json").exists():
                mt = load_json(out / "metrics.json")
                row.update(status=mt.get("status", "?"), best_val_acc=mt.get("best_val_acc"),
                           final_val_acc=mt.get("final_val_acc"), best_epoch=mt.get("best_epoch"),
                           train_time_sec=mt.get("train_time_sec"))
            elif (out / "diverged.json").exists():
                row["status"] = "diverged"
            rows.append(row)
    return rows


def pick_sweep_lrs(cfg: dict, mode: str, models: tuple[str, ...] = MODEL_ORDER,
                   require_complete: bool = True) -> dict[str, dict]:
    """Best-val LR per model. Diverged runs are excluded, never retried.

    Ties (identical best val acc) go to the *smaller* LR -- the more
    conservative choice for a longer run than the sweep itself.
    """
    rows = sweep_table(cfg, mode, models)
    picks: dict[str, dict] = {}
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        ok = [r for r in mine if r["status"] == "completed" and r["best_val_acc"] is not None]
        pending = [r for r in mine if r["status"] not in ("completed", "diverged")]
        if require_complete and pending:
            continue
        if not ok:
            continue
        best = max(ok, key=lambda r: (r["best_val_acc"], -r["lr"]))
        edge = best["lr"] in (min(r["lr"] for r in mine), max(r["lr"] for r in mine))
        picks[m] = {"lr": best["lr"], "best_val_acc": best["best_val_acc"],
                    "at_grid_edge": edge,
                    "diverged": [r["lr"] for r in mine if r["status"] == "diverged"],
                    # crashed or never run -- the pick was made without them
                    "missing": [r["lr"] for r in mine if r["status"] == "missing"]}
    return picks


def resolve_lr(job: Job, cfg: dict, mode: str) -> float:
    """The learning rate a job trains at.

    * fixed on the job (sweep points, 9h E4) -> that;
    * 9h mode -> the sweep's best-val LR, auto-picked (spec 12.2);
    * full mode -> ``lr:`` in ``configs/models/<model>.yaml``, which is only
      filled in once the owner has confirmed the sweep result.
    """
    if job.lr is not None:
        return float(job.lr)

    if job.exp == "E4":   # full mode: best of the E4 mini-sweep, by val
        key = job.out_dir.split("/")[-2]
        best = None
        for lr in cfg["experiments"]["e4_pretrained"]["lrs"]:
            mp = resolve_path(_run_dir("E4_sweep", key, f"lr_{_fmt(lr)}")) / "metrics.json"
            if mp.exists():
                acc = load_json(mp).get("best_val_acc")
                if acc is not None and (best is None or acc > best[0]):
                    best = (acc, float(lr))
        if best is None:
            raise RuntimeError(f"{job.id}: no completed E4 sweep run to pick an LR from")
        return best[1]

    if mode == "9h":
        # The queue only starts this job once every sweep point for the model
        # has *finished*, so anything still missing crashed. Pick among the
        # points that completed (unattended rule: keep going) and record which
        # ones were absent in lr_choice_9h.json for the write-up.
        choice_path = results_path() / "lr_choice_9h.json"
        picks = pick_sweep_lrs(cfg, mode, require_complete=False)
        if job.model not in picks:
            raise RuntimeError(f"{job.id}: no LR-sweep run for {job.model} completed "
                               "(all diverged or crashed) -- nothing to pick from")
        save_json({"mode": mode, "auto_picked": True, "picks": picks}, choice_path)
        return float(picks[job.model]["lr"])

    lr = load_config(f"configs/models/{job.model}.yaml").get("lr")
    if lr is None:
        raise LRNotConfirmed(
            f"{job.id}: configs/models/{job.model}.yaml has lr: null. Run the sweep, "
            "confirm the pick, and write it there (spec Phase 3 -> Phase 4 gate).")
    return float(lr)


if __name__ == "__main__":
    cfg = load_base_config()
    for mode, ups in (("9h", ()), ("9h", upgrade_order(cfg)), ("full", ())):
        q = build_queue(cfg, mode, ups)
        by = {}
        for j in q:
            by.setdefault((j.exp, j.tier), 0)
            by[(j.exp, j.tier)] += 1
        print(f"\n{mode} upgrades={list(ups) or 'none'}: {len(q)} jobs")
        for (exp, tier), n in by.items():
            print(f"  {exp:<9} {tier:<20} {n}")

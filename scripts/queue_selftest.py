"""End-to-end test of the unattended queue, on toy-sized runs (~5 min on CPU).

The 9h queue runs for eight hours with nobody watching, so the machinery
around the training loop has to be proven *before* the day, not discovered
during it. This drives the real ``scripts/run_all.py`` -- real child
processes, real dependency handling, real LR auto-pick -- with every run
shrunk to a single epoch or a few dozen steps, under an isolated results root
(``results/_queue_test``) so nothing it writes can be mistaken for a real run.

It checks that:
  * the sweep -> E1 -> E2 dependency chain runs in order;
  * a **diverging** sweep point (LR 1e30, guaranteed to blow up) is recorded as
    ``diverged``, is not retried, does not stop the queue, and is excluded
    from the LR pick;
  * the evaluation experiments (E5 robustness, E6 patch shuffle, E7
    interpretability) run on the toy E1 checkpoint once it exists, and
    complete;
  * ``progress.log``, ``RUN_SUMMARY.md`` and a failure log are written;
  * a second launch skips every finished job (resumability).

Run: ``python scripts/queue_selftest.py``
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_json, save_json  # noqa: E402

TEST_ROOT = "results/_queue_test"
MODEL = "mlp"
DIVERGE_LR = 1.0e30

PLAN = {
    "mode": "9h", "budget_hours": None, "upgrades": [], "trims": [], "estimates": {},
    "models": [MODEL],
    "cfg_overrides": {
        "mode_9h": {
            "lr_sweep": {"lrs": [1.0e-3, DIVERGE_LR], "epochs": 1},
            "e1": {"epochs": 1, "seeds": [0]},
            "e2": {"fractions": [0.1, 1.0], "total_steps": 30, "seeds": [0]},
        },
    },
}


def launch(extra_env: dict) -> tuple[int, str, float]:
    t = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--mode", "9h",
         "--plan", str(ROOT / TEST_ROOT / "plan.json"), "--models", MODEL, "--num-workers", "0"],
        cwd=str(ROOT), env={**os.environ, **extra_env, "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout + proc.stderr, time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", default="results/queue_selftest.json",
                    help="where the JSON summary goes (default: the committed gate record)")
    args = ap.parse_args()
    root = ROOT / TEST_ROOT
    shutil.rmtree(root, ignore_errors=True)
    save_json(PLAN, root / "plan.json")
    env = {"SHOWDOWN_RESULTS": TEST_ROOT}

    print("launch 1: full toy queue ...", flush=True)
    code, out, secs = launch(env)
    (root / "launch1.log").write_text(out, encoding="utf-8")
    print(f"  exit {code} in {secs:.0f}s (log: {TEST_ROOT}/launch1.log)")

    checks: list[tuple[str, bool, str]] = []
    status = dict(re.findall(r"\| `([^`]+)` \| \w+ \| (\w+) \|", (root / "RUN_SUMMARY.md")
                             .read_text(encoding="utf-8"))) if (root / "RUN_SUMMARY.md").exists() else {}
    checks.append(("queue exited 0 despite failures", code == 0, f"exit {code}"))
    checks.append(("RUN_SUMMARY.md written", bool(status), ""))
    good_lr = f"lr_sweep/{MODEL}/lr0.001"
    bad_lr = f"lr_sweep/{MODEL}/lr{DIVERGE_LR:g}"
    checks.append(("good sweep point done", status.get(good_lr) == "done", status.get(good_lr, "?")))
    checks.append(("LR 1e30 recorded as diverged", status.get(bad_lr) == "diverged",
                   status.get(bad_lr, "?")))
    checks.append(("diverged run left a failure log",
                   (root / "failures" / f"{bad_lr.replace('/', '__')}.log").exists(), ""))
    checks.append(("E1 ran after the sweep", status.get(f"E1/{MODEL}/s0") == "done",
                   status.get(f"E1/{MODEL}/s0", "?")))
    for f in ("0.1", "1"):
        jid = f"E2/{MODEL}/f{f}/s0"
        checks.append((f"{jid} done", status.get(jid) == "done", status.get(jid, "?")))
    # The evaluation experiments run on whatever E1 checkpoints exist -- here
    # the single toy MLP -- and must complete, leaving their raw outputs.
    for e, raw in (("E5", "robustness_raw.csv"), ("E6", "patch_shuffle_raw.csv"),
                   ("E7", "summary.json")):
        checks.append((f"{e} ran on the toy E1 checkpoint", status.get(f"{e}/eval") == "done",
                       status.get(f"{e}/eval", "?")))
        checks.append((f"{e} wrote {raw}", (root / "runs" / e / raw).exists(), ""))

    choice = root / "lr_choice_9h.json"
    if choice.exists():
        pick = load_json(choice)["picks"].get(MODEL, {})
        checks.append(("auto-pick chose the non-diverged LR", pick.get("lr") == 1.0e-3,
                       str(pick)))
        checks.append(("diverged LR listed in the pick record", DIVERGE_LR in pick.get("diverged", []),
                       str(pick.get("diverged"))))
    else:
        checks.append(("lr_choice_9h.json written", False, ""))

    e1 = root / "runs" / "E1" / MODEL / "0"
    if (e1 / "metrics.json").exists():
        m = load_json(e1 / "metrics.json")
        checks.append(("E1 trained at the picked LR", m.get("lr") == 1.0e-3, f"lr={m.get('lr')}"))
        checks.append(("E1 touched test once", m.get("test_touched") is True, ""))
    checks.append(("E1 kept best.pt, dropped last.pt (spec 12.5)",
                   (e1 / "best.pt").exists() and not (e1 / "last.pt").exists(), ""))
    sweep_dir = root / "runs" / "lr_sweep" / MODEL / "lr_0.001"
    checks.append(("sweep run saved no checkpoints",
                   sweep_dir.exists() and not list(sweep_dir.glob("*.pt")), ""))
    sm = load_json(sweep_dir / "metrics.json") if (sweep_dir / "metrics.json").exists() else {}
    checks.append(("sweep run never touched test", sm.get("test_touched") is False
                   and sm.get("test_acc") is None, ""))
    checks.append(("progress.log written", (root / "progress.log").exists(), ""))
    checks.append(("nothing written to the real results/runs",
                   not (ROOT / "results" / "runs" / "lr_sweep" / MODEL / f"lr_{DIVERGE_LR:g}").exists(),
                   ""))

    print("launch 2: everything should be skipped ...", flush=True)
    code2, out2, secs2 = launch(env)
    (root / "launch2.log").write_text(out2, encoding="utf-8")
    m = re.search(r"(\d+) already complete, (\d+) previously diverged, (\d+) to run", out2)
    n_done, n_div, n_todo = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (-1,) * 3
    # Everything that finished is skipped, the diverged point is NOT retried,
    # and only jobs that did not finish (the not-yet-built evals) run again.
    n_ok1 = sum(1 for v in status.values() if v == "done")
    n_retry = sum(1 for v in status.values() if v not in ("done", "diverged"))
    checks.append(("relaunch skips finished jobs", n_done == n_ok1,
                   f"{n_done} skipped as complete (expected {n_ok1}), {secs2:.0f}s"))
    checks.append(("relaunch does not retry the diverged run", n_div == 1 and n_todo == n_retry,
                   f"{n_div} diverged kept, {n_todo} to run (expected {n_retry})"))

    ok = all(c[1] for c in checks)
    print()
    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    print(f"\n{sum(c[1] for c in checks)}/{len(checks)} queue checks passed -- "
          f"{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    summary = Path(args.summary)
    save_json({"all_passed": ok, "checks": [{"check": c, "passed": p, "detail": d}
                                            for c, p, d in checks]},
              summary if summary.is_absolute() else ROOT / summary)
    return 0 if ok else 1


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    sys.exit(main())

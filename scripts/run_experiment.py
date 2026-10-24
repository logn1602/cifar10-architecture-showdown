"""Run one experiment's jobs through the same queue as ``run_all.py``.

    python scripts/run_experiment.py --exp E1 --mode full
    python scripts/run_experiment.py --exp E2 --mode 9h --models vit resnet18
    python scripts/run_experiment.py --exp E4 --mode full --dry-run

Same job definitions, same skip/resume behaviour, same failure isolation --
this is just ``run_all.py --exp <name>`` with a friendlier spelling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_all  # noqa: E402

EXPERIMENTS = ["lr_sweep", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", required=True, choices=EXPERIMENTS)
    ap.add_argument("--mode", choices=["9h", "full"], default="full")
    args, rest = ap.parse_known_args()
    exps = [args.exp, "E4_sweep"] if args.exp == "E4" else [args.exp]
    return run_all.main(["--mode", args.mode, "--exp", *exps, *rest])


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    sys.exit(main())

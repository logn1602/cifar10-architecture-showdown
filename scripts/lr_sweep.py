"""Phase 3 LR sweep: the only per-model tuning the protocol allows.

    python scripts/lr_sweep.py --mode full            # run the 5-LR x 40-epoch sweep, then report
    python scripts/lr_sweep.py --mode full --report   # report only (table + plot + picks)
    python scripts/lr_sweep.py --mode full --confirm  # write the picks into configs/models/*.yaml

Picks are by **best validation accuracy** -- the test set is never touched by a
sweep run (``evaluate_test=False``), so it cannot leak into the choice.

Full mode stops at the report: spec Phase 3 says "ask me to confirm the chosen
LRs before Phase 4", so post-sweep runs stay blocked until ``--confirm`` writes
``lr:`` into each model's YAML. In 9h mode the picks are applied automatically
by the queue (spec 12.2); this script then just reports them.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import (  # noqa: E402
    load_base_config, mode_params, pick_sweep_lrs, results_path, sweep_table,
)
from src.models import MODEL_ORDER  # noqa: E402
from src.utils import PROJECT_ROOT, ensure_dir, save_json  # noqa: E402


def write_table(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    fields = ["model", "lr", "status", "best_val_acc", "final_val_acc", "best_epoch", "epochs",
              "train_time_sec"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def plot(rows: list[dict], picks: dict, cfg: dict, path: Path, mode: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = cfg["figures"]["model_colors"]
    labels = cfg["figures"]["model_labels"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    diverged = []
    for m in MODEL_ORDER:
        diverged += [(r["lr"], colors[m]) for r in rows
                     if r["model"] == m and r["status"] == "diverged"]
        pts = sorted((r["lr"], r["best_val_acc"]) for r in rows
                     if r["model"] == m and r["best_val_acc"] is not None)
        if not pts:
            if any(r["model"] == m and r["status"] == "diverged" for r in rows):
                ax.plot([], [], "x", color=colors[m], ms=9, mew=2,
                        label=f"{labels[m]} (every LR diverged)")
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, [y * 100 for y in ys], "-o", color=colors[m], label=labels[m], lw=2, ms=5)
        if m in picks:
            ax.plot(picks[m]["lr"], picks[m]["best_val_acc"] * 100, "o", ms=12,
                    mfc="none", mec=colors[m], mew=2)
    # Diverged runs have no accuracy: mark them with an x on the axis floor,
    # drawn last so the floor is final, and pin the limits so they stay put.
    lo, hi = ax.get_ylim()
    for lr, c in diverged:
        ax.plot(lr, lo, "x", color=c, ms=9, mew=2, clip_on=False)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    lrs = sorted({r["lr"] for r in rows})
    ax.set_xticks(lrs)
    ax.set_xticklabels([f"{x:g}" for x in lrs])
    ax.minorticks_off()
    ax.set_xlabel("peak learning rate (AdamW)")
    ax.set_ylabel("best validation accuracy (%)")
    ep = rows[0]["epochs"] if rows else "?"
    ax.set_title(f"LR sweep ({mode} mode, {ep} epochs, seed 0) -- circled = pick"
                 + (", x = diverged" if diverged else ""))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]))
    plt.close(fig)


def confirm(picks: dict) -> None:
    """Write ``lr: <pick>`` into each model YAML, preserving its comments."""
    for m, p in picks.items():
        path = PROJECT_ROOT / "configs" / "models" / f"{m}.yaml"
        text = path.read_text(encoding="utf-8")
        new, n = re.subn(r"(?m)^lr:.*$", f"lr: {p['lr']:.1e}   # confirmed from the LR sweep "
                                         f"(best val {p['best_val_acc']:.4f})", text)
        if n != 1:
            raise SystemExit(f"{path}: expected exactly one top-level 'lr:' line, found {n}")
        path.write_text(new, encoding="utf-8")
        print(f"  {m:<14} lr -> {p['lr']:.1e}   ({path.relative_to(PROJECT_ROOT)})")


def report(cfg: dict, mode: str) -> dict:
    rows = sweep_table(cfg, mode)
    picks = pick_sweep_lrs(cfg, mode, require_complete=False)
    suffix = "" if mode == "full" else "_9h"
    root = results_path()
    table = root / "tables" / f"lr_sweep{suffix}.csv"
    fig = root / "figures" / f"lr_sweep{suffix}.png"
    write_table(rows, table)
    if any(r["best_val_acc"] is not None for r in rows):
        plot(rows, picks, cfg, fig, mode)
    save_json({"mode": mode, "picks": picks}, root / f"lr_choice{suffix}.json")

    lrs = mode_params(cfg, mode)["sweep_lrs"]
    print(f"\nLR sweep ({mode}) -- best val accuracy per LR")
    print(f"{'model':<15}" + "".join(f"{x:>10g}" for x in lrs) + f"{'pick':>10}")
    for m in MODEL_ORDER:
        cells = []
        for lr in lrs:
            r = next(r for r in rows if r["model"] == m and r["lr"] == lr)
            cells.append(f"{r['best_val_acc']:.4f}" if r["best_val_acc"] is not None
                         else r["status"][:8])
        pick = f"{picks[m]['lr']:g}" if m in picks else "-"
        flag = " (edge!)" if m in picks and picks[m]["at_grid_edge"] else ""
        print(f"{m:<15}" + "".join(f"{c:>10}" for c in cells) + f"{pick:>10}{flag}")
    print(f"\nwrote {table.relative_to(PROJECT_ROOT)}"
          + (f" and {fig.relative_to(PROJECT_ROOT)}" if fig.exists() else ""))
    if any(p["at_grid_edge"] for p in picks.values()):
        print("note: an '(edge!)' pick sits at the end of the LR grid -- the true optimum may "
              "lie outside it. Worth mentioning when confirming.")
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["full", "9h"], default="full")
    ap.add_argument("--report", action="store_true", help="skip running; just report")
    ap.add_argument("--confirm", action="store_true",
                    help="full mode: write the picks into configs/models/*.yaml")
    args, rest = ap.parse_known_args()
    cfg = load_base_config()

    if not args.report and not args.confirm:
        import run_all
        run_all.main(["--mode", args.mode, "--exp", "lr_sweep", *rest])

    picks = report(cfg, args.mode)
    if args.confirm:
        if args.mode != "full":
            raise SystemExit("--confirm is for full mode; 9h mode auto-picks.")
        missing = [m for m in MODEL_ORDER if m not in picks]
        if missing:
            raise SystemExit(f"sweep incomplete for {missing}; nothing written")
        print("\nwriting confirmed LRs:")
        confirm(picks)
    elif args.mode == "full" and picks:
        print("\nFull mode: confirm these before Phase 4, then run with --confirm.")
    return 0


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    sys.exit(main())

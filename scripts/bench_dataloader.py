"""Phase 0: pick `num_workers` by measuring, not by guessing.

On Windows each DataLoader worker is a *spawned* process, not a fork: it
re-imports the parent module, re-pickles the dataset, and pays full interpreter
startup. That makes workers much more expensive than on Linux, and the optimum
here is usually lower than the Linux optimum. So this must be re-measured on
Kaggle rather than copied across -- the recommendation it prints is for the
machine it ran on.

Every configuration is isolated in try/except: a crash at 8 workers (easy on a
8 GB box) must not throw away the results for 0, 2 and 4.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_dataloaders  # noqa: E402
from src.utils import ensure_dir, load_config, resolve_path  # noqa: E402

TRAIN_SET_SIZE = 45_000  # for the projected sec/epoch column

# If the top two configurations are within this relative margin, prefer the one
# with fewer workers: less RAM, faster startup, fewer ways to hang.
TIE_MARGIN = 0.05


def _available_ram_gb() -> float | None:
    try:
        return importlib.import_module("psutil").virtual_memory().available / 1024**3
    except Exception:
        return None


def time_loader(loader, n_batches: int, warmup: int) -> tuple[float, int]:
    """Iterate the loader; return (seconds for n_batches, images seen).

    Touches each batch (``.shape``, plus a mean on the first one) so nothing is
    lazily skipped, but does no real compute -- we are measuring the input
    pipeline, not the model.
    """
    it = iter(loader)
    for _ in range(warmup):
        try:
            next(it)
        except StopIteration:
            it = iter(loader)
            next(it)

    seen = 0
    t0 = time.perf_counter()
    for i in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        _ = x.shape, y.shape
        if i == 0:
            _ = float(x.float().mean())  # force the tensor to be materialised
        seen += x.shape[0]
    return time.perf_counter() - t0, seen


def run_one(data_cfg: dict, aug_cfg: dict | None, augmentation: str, workers: int,
            batch_size: int, n_batches: int, warmup: int, repeat: int) -> dict:
    row = {
        "augmentation": augmentation, "num_workers": workers, "batch_size": batch_size,
        "batches": n_batches, "repeat": repeat, "imgs_per_sec": None,
        "sec_per_batch": None, "projected_sec_per_epoch_45k": None,
        "status": "ok", "error": "",
    }
    loader = None
    try:
        loader, _, _ = build_dataloaders(
            data_cfg, fraction=1.0, augmentation=augmentation,
            batch_size=batch_size, num_workers=workers, seed=0, aug_cfg=aug_cfg,
        )
        elapsed, seen = time_loader(loader, n_batches, warmup)
        ips = seen / elapsed
        row["imgs_per_sec"] = round(ips, 1)
        row["sec_per_batch"] = round(elapsed / n_batches, 4)
        row["projected_sec_per_epoch_45k"] = round(TRAIN_SET_SIZE / ips, 1)
    except Exception as exc:  # MemoryError / OSError / RuntimeError all land here
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        # Reap worker processes before the next configuration starts, or the
        # next measurement competes with dying workers for CPU and RAM.
        del loader
        gc.collect()
    return row


def pick_best(rows: list[dict]) -> tuple[int | None, str]:
    ok = [r for r in rows if r["status"] == "ok" and r["imgs_per_sec"]]
    if not ok:
        return None, "No configuration completed successfully."
    ranked = sorted(ok, key=lambda r: -r["imgs_per_sec"])
    best = ranked[0]
    note = (f"{best['num_workers']} workers was fastest at "
            f"{best['imgs_per_sec']:,.0f} img/s.")
    if len(ranked) > 1:
        runner = ranked[1]
        rel = (best["imgs_per_sec"] - runner["imgs_per_sec"]) / best["imgs_per_sec"]
        if rel < TIE_MARGIN and runner["num_workers"] < best["num_workers"]:
            note = (f"{runner['num_workers']} and {best['num_workers']} workers were within "
                    f"{rel*100:.1f}% ({runner['imgs_per_sec']:,.0f} vs "
                    f"{best['imgs_per_sec']:,.0f} img/s), so the lower worker count wins "
                    f"on memory and startup cost.")
            best = runner
    return best["num_workers"], note


def write_outputs(rows: list[dict], best: int | None, note: str, out_dir: Path) -> None:
    fields = ["augmentation", "num_workers", "batch_size", "batches", "repeat",
              "imgs_per_sec", "sec_per_batch", "projected_sec_per_epoch_45k", "status", "error"]
    csv_path = out_dir / "dataloader_bench.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    L = ["# Phase 0 -- DataLoader throughput\n",
         "_Generated by `scripts/bench_dataloader.py`. Do not edit by hand._\n",
         "| aug | workers | img/s | s/batch | projected s/epoch (45k) | status |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        ips = f"{r['imgs_per_sec']:,.0f}" if r["imgs_per_sec"] else "-"
        spb = f"{r['sec_per_batch']:.4f}" if r["sec_per_batch"] else "-"
        spe = f"{r['projected_sec_per_epoch_45k']:,.1f}" if r["projected_sec_per_epoch_45k"] else "-"
        status = r["status"] if r["status"] == "ok" else f"**{r['status']}** ({r['error'][:80]})"
        L.append(f"| {r['augmentation']} | {r['num_workers']} | {ips} | {spb} | {spe} | {status} |")
    L += ["", f"**Recommended num_workers: {best}**", "", note, "",
          "On Windows a DataLoader worker is a spawned process (full interpreter startup and",
          "dataset re-pickling per worker), so the optimum is typically lower here than on",
          "Linux. Re-run this on Kaggle before the unattended queue rather than reusing this",
          "number.", ""]
    (out_dir / "dataloader_bench.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark DataLoader throughput vs num_workers")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--warmup-batches", type=int, default=10)
    ap.add_argument("--augmentation", default="standard",
                    choices=["none", "standard", "strong", "all"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out-dir", default="results/tables")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    out_dir = ensure_dir(args.out_dir)

    ram = _available_ram_gb()
    if ram is not None and ram < 2.0:
        print(f"!! Only {ram:.1f} GB RAM available. Results will be noisy and high worker\n"
              f"!! counts may fail outright -- closing other applications would help.\n")

    cpu_count = os.cpu_count() or 1
    augs = ["standard", "strong"] if args.augmentation == "all" else [args.augmentation]

    print(f"Benchmarking {len(augs)} augmentation(s) x {len(args.workers)} worker counts "
          f"x {args.repeats} repeat(s)")
    print(f"batch_size={args.batch_size}  batches={args.batches}  warmup={args.warmup_batches}  "
          f"cpu_count={cpu_count}\n")

    rows: list[dict] = []
    for aug in augs:
        aug_cfg = cfg.get("augmentation", {}).get(aug)
        for w in args.workers:
            if w > cpu_count:
                print(f"  skip workers={w}: exceeds os.cpu_count()={cpu_count}")
                rows.append({"augmentation": aug, "num_workers": w, "batch_size": args.batch_size,
                             "batches": args.batches, "repeat": 0, "imgs_per_sec": None,
                             "sec_per_batch": None, "projected_sec_per_epoch_45k": None,
                             "status": "skipped", "error": f"> os.cpu_count()={cpu_count}"})
                continue
            for rep in range(args.repeats):
                print(f"  aug={aug:<9} workers={w}  rep={rep} ... ", end="", flush=True)
                row = run_one(data_cfg, aug_cfg, aug, w, args.batch_size,
                              args.batches, args.warmup_batches, rep)
                rows.append(row)
                if row["status"] == "ok":
                    print(f"{row['imgs_per_sec']:>8,.0f} img/s   "
                          f"({row['projected_sec_per_epoch_45k']:,.0f} s/epoch projected)")
                else:
                    print(f"FAILED  {row['error'][:90]}")

    # The recommendation is made on the protocol's default augmentation.
    primary = [r for r in rows if r["augmentation"] == ("standard" if "standard" in augs else augs[0])]
    best, note = pick_best(primary)
    write_outputs(rows, best, note, out_dir)

    print("\n" + "-" * 70)
    print(f"RECOMMENDED num_workers: {best}")
    print(note)
    if best is not None and best != int(data_cfg.get("num_workers", 4)):
        print(f"\n  configs/base.yaml currently has data.num_workers: {data_cfg.get('num_workers')}")
        print(f"  To apply this result, set it to:   data.num_workers: {best}")
    print("-" * 70)
    print(f"wrote {out_dir / 'dataloader_bench.csv'}")
    print(f"wrote {out_dir / 'dataloader_bench.md'}")


if __name__ == "__main__":  # REQUIRED on Windows: workers re-import this module
    main()

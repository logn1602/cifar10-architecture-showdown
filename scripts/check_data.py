"""Phase 1 gate: prove the data pipeline is correct, then draw it.

Spec Phase 1 asks for two things -- "class counts balanced in every
split/subset; normalization stats correct", and a grid of augmented samples.
This script does both, and treats the checks as a gate: any hard failure exits
non-zero, because every downstream number depends on this being right.

The checks worth knowing about:

* **Determinism (check 5).** Rebuilds the split from scratch and compares
  sha256 fingerprints. Those fingerprints are also what you compare against
  Kaggle, since ``data/`` is gitignored and the split is regenerated there.
* **Nesting (check 4).** Asserts the 5% subset really is contained in the 10%
  subset, and so on. If that ever broke, the E2 data-hunger curve would be
  measuring "which images" as well as "how many".
* **Val augmentation leak (check 8).** Builds the val set twice, once asking
  for standard augmentation and once for strong, and asserts the tensors come
  out bit-identical -- i.e. that the request was correctly ignored.

Run: ``python scripts/check_data.py``   (add ``--quick`` for a 5k-image pass)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    CIFAR10_MEAN, CIFAR10_STD, CLASSES, CIFAR10Subset, MixupCutmix,
    build_datasets, build_transform, class_counts, denormalize,
    get_raw_datasets, get_split_indices, get_subset_indices,
)
from src.utils import (  # noqa: E402
    ensure_dir, load_config, resolve_path, set_seed, sha256_of_array,
)

FRACTIONS = [0.05, 0.1, 0.25, 0.5, 1.0]

# Normalized data should be near zero-mean / unit-std. The tolerances are loose
# enough to absorb the 45k-vs-50k difference but tight enough to catch a
# genuinely wrong constant or a missing Normalize.
NORM_MEAN_TOL = 0.02
NORM_STD_TOL = 0.03
# A difference this large between the spec constants and the measured 45k
# statistics is reported as a WARN, not a failure -- see check 6.
RAW_STAT_WARN_TOL = 0.01


class Checks:
    """Collects PASS / WARN / FAIL rows and decides the exit code."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    SYMBOL = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}

    def record(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"  [{self.SYMBOL[status]}] {name}" + (f"  --  {detail}" if detail else ""))

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        """A hard check: failing it exits the script non-zero."""
        self.record("PASS" if condition else "FAIL", name, detail)
        return condition

    def warn(self, name: str, condition: bool, detail: str = "") -> None:
        """A soft check: failing it is reported but does not fail the run."""
        self.record("PASS" if condition else "WARN", name, detail)

    @property
    def failed(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == "FAIL"]

    @property
    def warned(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == "WARN"]


# ---------------------------------------------------------------------------
# Part A -- validation
# ---------------------------------------------------------------------------


def channel_stats(ds, num_workers: int, batch_size: int = 512, limit: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-channel mean/std over a dataset, streamed.

    Uses sum / sum-of-squares over pixels rather than averaging per-batch means,
    so a ragged final batch cannot skew the result.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    seen = 0
    for x, _ in loader:                       # (B, 3, 32, 32)
        if limit is not None and seen >= limit:
            break
        x = x.double()
        total += x.sum(dim=(0, 2, 3))
        total_sq += (x**2).sum(dim=(0, 2, 3))
        n_pixels += x.shape[0] * x.shape[2] * x.shape[3]
        seen += x.shape[0]
    mean = (total / n_pixels).numpy()
    var = (total_sq / n_pixels).numpy() - mean**2
    return mean, np.sqrt(np.maximum(var, 0.0)), seen


def run_checks(cfg: dict, args) -> tuple[Checks, dict]:
    chk = Checks()
    facts: dict = {}

    data_cfg = cfg["data"]
    root = resolve_path(data_cfg["root"])
    nw = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 4))

    base_train, base_test = get_raw_datasets(root, download=bool(data_cfg.get("download", True)))
    split = get_split_indices(root, val_per_class=int(data_cfg["val_per_class"]),
                              seed=int(data_cfg["split_seed"]))
    train_idx, val_idx = split["train"], split["val"]
    labels = np.asarray(base_train.targets, dtype=np.int64)
    test_labels = np.asarray(base_test.targets, dtype=np.int64)

    print("\n1. Split sizes")
    chk.check("train == 45,000", train_idx.size == 45_000, f"got {train_idx.size:,}")
    chk.check("val == 5,000", val_idx.size == 5_000, f"got {val_idx.size:,}")
    chk.check("test == 10,000", len(base_test) == 10_000, f"got {len(base_test):,}")

    print("\n2. Disjointness")
    overlap = np.intersect1d(train_idx, val_idx)
    chk.check("train and val do not overlap", overlap.size == 0, f"{overlap.size} shared indices")
    union = np.union1d(train_idx, val_idx)
    chk.check("train + val cover all 50,000 train images", union.size == 50_000,
              f"covered {union.size:,}")
    # Index 0 exists in both the train archive and the test archive, so index
    # overlap says nothing here. What matters is that they are different files.
    chk.check("test comes from the official held-out archive",
              base_test.train is False and base_train.train is True,
              "base_test.train=False")

    print("\n3. Stratification")
    tr_counts, va_counts = class_counts(labels[train_idx]), class_counts(labels[val_idx])
    te_counts = class_counts(test_labels)
    facts["counts"] = {"train": tr_counts, "val": va_counts, "test": te_counts}
    chk.check("every class has exactly 4,500 train images", set(tr_counts.values()) == {4500},
              f"range {min(tr_counts.values())}-{max(tr_counts.values())}")
    chk.check("every class has exactly 500 val images", set(va_counts.values()) == {500},
              f"range {min(va_counts.values())}-{max(va_counts.values())}")
    chk.check("every class has exactly 1,000 test images", set(te_counts.values()) == {1000},
              f"range {min(te_counts.values())}-{max(te_counts.values())}")

    print("\n4. Subset fractions (stratified + nested)")
    subset_idx: dict[float, np.ndarray] = {}
    facts["subsets"] = {}
    prev: np.ndarray | None = None
    for frac in FRACTIONS:
        sub = get_subset_indices(root, frac, seed=int(data_cfg.get("subset_seed", 0)),
                                 val_per_class=int(data_cfg["val_per_class"]),
                                 split_seed=int(data_cfg["split_seed"]))
        subset_idx[frac] = sub
        counts = class_counts(labels[sub])
        expected = int(round(frac * 4500))
        ok_counts = all(abs(v - expected) <= 1 for v in counts.values())
        ok_classes = all(v > 0 for v in counts.values())
        chk.check(f"{frac:.0%} subset is stratified ({expected}/class)", ok_counts and ok_classes,
                  f"n={sub.size:,} range {min(counts.values())}-{max(counts.values())}")
        if prev is not None:
            chk.check(f"{frac:.0%} subset contains the previous fraction",
                      bool(np.isin(prev, sub).all()))
        prev = sub
        facts["subsets"][frac] = {"n": int(sub.size), "per_class": expected,
                                  "sha256": sha256_of_array(sub), "counts": counts}

    print("\n5. Determinism (these hashes must match on Kaggle)")
    rebuilt = get_split_indices(root, val_per_class=int(data_cfg["val_per_class"]),
                                seed=int(data_cfg["split_seed"]), force=True)
    train_sha, val_sha = sha256_of_array(train_idx), sha256_of_array(val_idx)
    chk.check("split rebuilds identically from scratch",
              sha256_of_array(rebuilt["train"]) == train_sha
              and sha256_of_array(rebuilt["val"]) == val_sha)
    frac_check = 0.1
    resub = get_subset_indices(root, frac_check, seed=int(data_cfg.get("subset_seed", 0)),
                               val_per_class=int(data_cfg["val_per_class"]),
                               split_seed=int(data_cfg["split_seed"]), force=True)
    chk.check(f"{frac_check:.0%} subset rebuilds identically",
              sha256_of_array(resub) == facts["subsets"][frac_check]["sha256"])
    facts["fingerprints"] = {"train_sha256": train_sha, "val_sha256": val_sha,
                             "subset_0.1_sha256": facts["subsets"][0.1]["sha256"]}
    print(f"       train  {train_sha}")
    print(f"       val    {val_sha}")

    print("\n6. Normalization")
    limit = 5_000 if args.quick else None
    norm_ds = CIFAR10Subset(base_train, train_idx, build_transform("none", train=False,
                                                                   mean=data_cfg["mean"],
                                                                   std=data_cfg["std"]))
    n_mean, n_std, n_seen = channel_stats(norm_ds, nw, limit=limit)
    chk.check("normalized train mean is ~0 per channel",
              bool(np.all(np.abs(n_mean) < NORM_MEAN_TOL)),
              f"{np.round(n_mean, 4).tolist()} (tol {NORM_MEAN_TOL}, n={n_seen:,})")
    chk.check("normalized train std is ~1 per channel",
              bool(np.all(np.abs(n_std - 1.0) < NORM_STD_TOL)),
              f"{np.round(n_std, 4).tolist()} (tol {NORM_STD_TOL})")

    raw_ds = CIFAR10Subset(base_train, train_idx, transforms.ToTensor())
    r_mean, r_std, _ = channel_stats(raw_ds, nw, limit=limit)
    d_mean = np.abs(r_mean - np.asarray(CIFAR10_MEAN))
    d_std = np.abs(r_std - np.asarray(CIFAR10_STD))
    # The spec constants are the canonical full-50k values; our 45k train split
    # excludes 5k images, so a small difference is expected and correct.
    chk.warn("measured 45k mean matches the spec constant", bool(np.all(d_mean < RAW_STAT_WARN_TOL)),
             f"measured {np.round(r_mean, 4).tolist()} vs spec {list(CIFAR10_MEAN)} "
             f"(max delta {d_mean.max():.4f})")
    chk.warn("measured 45k std matches the spec constant", bool(np.all(d_std < RAW_STAT_WARN_TOL)),
             f"measured {np.round(r_std, 4).tolist()} vs spec {list(CIFAR10_STD)} "
             f"(max delta {d_std.max():.4f})")
    facts["normalization"] = {
        "n_images": int(n_seen), "quick": bool(args.quick),
        "normalized_mean": n_mean.tolist(), "normalized_std": n_std.tolist(),
        "raw_mean_measured": r_mean.tolist(), "raw_std_measured": r_std.tolist(),
        "raw_mean_spec": list(CIFAR10_MEAN), "raw_std_spec": list(CIFAR10_STD),
    }

    print("\n7. Labels")
    chk.check("CLASSES has 10 entries", len(CLASSES) == 10)
    chk.check("CLASSES matches torchvision order", tuple(base_train.classes) == CLASSES,
              f"torchvision: {list(base_train.classes)}")

    print("\n8. Val/test are never augmented")
    _, val_std_aug, _ = build_datasets(data_cfg, augmentation="standard",
                                       aug_cfg=cfg["augmentation"]["standard"])
    _, val_strong_aug, _ = build_datasets(data_cfg, augmentation="strong",
                                          aug_cfg=cfg["augmentation"]["strong"])
    set_seed(0)
    a = torch.stack([val_std_aug[i][0] for i in range(32)])
    set_seed(0)
    b = torch.stack([val_strong_aug[i][0] for i in range(32)])
    chk.check("val tensors are identical under standard and strong requests",
              bool(torch.equal(a, b)),
              "the augmentation argument is correctly ignored for val")
    chk.check("eval pipeline has exactly 2 ops (ToTensor, Normalize)",
              len(build_transform("strong", train=False).transforms) == 2)

    return chk, facts


# ---------------------------------------------------------------------------
# Part B -- figures
# ---------------------------------------------------------------------------


def figure_augmentations(cfg: dict, out_path: Path, dpi: int) -> None:
    """Original + 4 standard draws + 4 strong draws, for 6 classes."""
    data_cfg = cfg["data"]
    root = resolve_path(data_cfg["root"])
    base_train, _ = get_raw_datasets(root)
    split = get_split_indices(root, val_per_class=int(data_cfg["val_per_class"]),
                              seed=int(data_cfg["split_seed"]))
    labels = np.asarray(base_train.targets, dtype=np.int64)
    train_idx = split["train"]

    rng = np.random.RandomState(0)
    chosen_classes = [0, 1, 3, 5, 6, 8]  # airplane, automobile, cat, dog, frog, ship
    picks = []
    for c in chosen_classes:
        candidates = train_idx[labels[train_idx] == c]
        picks.append(int(candidates[rng.randint(candidates.size)]))

    std_tf = build_transform("standard", train=True, aug_cfg=cfg["augmentation"]["standard"])
    strong_tf = build_transform("strong", train=True, aug_cfg=cfg["augmentation"]["strong"])
    n_draws = 4
    n_cols = 1 + 2 * n_draws

    fig, axes = plt.subplots(len(picks), n_cols, figsize=(1.35 * n_cols, 1.45 * len(picks)))
    set_seed(0)
    for r, idx in enumerate(picks):
        img, label = base_train[idx]
        axes[r, 0].imshow(np.asarray(img))
        axes[r, 0].set_ylabel(CLASSES[label], fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=28)
        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)
        for d in range(n_draws):
            # De-normalize for display -- the tensors are zero-mean/unit-std and
            # would otherwise render as noise.
            axes[r, 1 + d].imshow(denormalize(std_tf(img)).permute(1, 2, 0).numpy())
            axes[r, 1 + n_draws + d].imshow(denormalize(strong_tf(img)).permute(1, 2, 0).numpy())

    axes[0, 0].set_title("original", fontsize=8)
    for d in range(n_draws):
        axes[0, 1 + d].set_title(f"std {d+1}", fontsize=8)
        axes[0, 1 + n_draws + d].set_title(f"strong {d+1}", fontsize=8)

    fig.suptitle(
        "CIFAR-10 augmentation pipelines\n"
        "standard: RandomCrop(32, pad=4) -> RandomHorizontalFlip -> ToTensor -> Normalize\n"
        "strong:   standard + RandAugment(2, 9) before ToTensor, RandomErasing(p=0.25) after Normalize\n"
        "(Mixup/CutMix are batch-level and applied in the training loop, not shown here)",
        fontsize=8.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_class_distribution(facts: dict, out_path: Path, dpi: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    counts = facts["counts"]
    x = np.arange(len(CLASSES))
    width = 0.27
    for i, (name, color) in enumerate([("train", "#1f77b4"), ("val", "#ff7f0e"), ("test", "#2ca02c")]):
        vals = [counts[name][c] for c in CLASSES]
        bars = ax1.bar(x + (i - 1) * width, vals, width, label=f"{name} ({sum(vals):,})", color=color)
        ax1.bar_label(bars, fontsize=6, padding=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("images")
    ax1.set_title("Per-class counts: train / val / test")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    for frac in FRACTIONS:
        sub = facts["subsets"][frac]
        ax2.plot(x, [sub["counts"][c] for c in CLASSES], marker="o", ms=4,
                 label=f"{frac:.0%}  (n={sub['n']:,})")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("images per class (log)")
    ax2.set_title("E2 subset fractions stay stratified (and are nested)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_mixup_cutmix(cfg: dict, out_path: Path, dpi: int) -> None:
    """Visual proof that the CutMix lam-correction matches the pasted area."""
    data_cfg = cfg["data"]
    train_ds, _, _ = build_datasets(data_cfg, augmentation="standard",
                                    aug_cfg=cfg["augmentation"]["standard"])
    set_seed(0)
    batch = torch.stack([train_ds[i][0] for i in range(8)])       # (8, 3, 32, 32)
    ys = torch.tensor([train_ds[i][1] for i in range(8)])

    strong = cfg["augmentation"]["strong"]
    mixer = MixupCutmix(mixup_alpha=strong["mixup"]["alpha"],
                        cutmix_alpha=strong["cutmix"]["alpha"],
                        prob=strong["mixup_cutmix_prob"])

    n_rows = 3
    fig, axes = plt.subplots(n_rows, 8, figsize=(13, 1.75 * n_rows))
    for r in range(n_rows):
        x, y_a, y_b, lam = mixer(batch.clone(), ys)
        for c in range(8):
            axes[r, c].imshow(denormalize(x[c]).permute(1, 2, 0).numpy())
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].set_title(f"{CLASSES[int(y_a[c])][:6]}/{CLASSES[int(y_b[c])][:6]}",
                                 fontsize=6)
        axes[r, 0].set_ylabel(f"lam={lam:.2f}", fontsize=8, rotation=0, ha="right",
                              va="center", labelpad=26)
    fig.suptitle("Mixup / CutMix (batch-level, E3 only). For CutMix, the pasted box area "
                 "equals 1-lam by construction.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part C -- report
# ---------------------------------------------------------------------------


def write_report(chk: Checks, facts: dict, out_path: Path, figures: list[str]) -> None:
    L: list[str] = []
    add = L.append
    add("# Phase 1 -- Data pipeline report\n")
    add("_Generated by `scripts/check_data.py`. Every number below is computed from the "
        "actual data; nothing is hard-coded._\n")

    n_fail, n_warn = len(chk.failed), len(chk.warned)
    verdict = "**FAIL**" if n_fail else ("**PASS** (with warnings)" if n_warn else "**PASS**")
    add(f"## Verdict: {verdict}\n")
    add(f"{len(chk.rows)} checks: {len(chk.rows) - n_fail - n_warn} passed, "
        f"{n_warn} warned, {n_fail} failed.\n")

    add("| Status | Check | Detail |\n|---|---|---|")
    for status, name, detail in chk.rows:
        add(f"| {status} | {name} | {detail} |")
    add("")

    add("## Split fingerprints\n")
    add("`data/` is gitignored, so the split is **regenerated** on every machine rather than "
        "shipped. These sha256 values are how you prove Kaggle built the same split:\n")
    add("| Array | sha256 |\n|---|---|")
    for k, v in facts["fingerprints"].items():
        add(f"| `{k}` | `{v}` |")
    add("")

    add("## Per-class counts\n")
    add("| Class | Train | Val | Test |\n|---|---|---|---|")
    c = facts["counts"]
    for name in CLASSES:
        add(f"| {name} | {c['train'][name]:,} | {c['val'][name]:,} | {c['test'][name]:,} |")
    add(f"| **total** | **{sum(c['train'].values()):,}** | **{sum(c['val'].values()):,}** "
        f"| **{sum(c['test'].values()):,}** |\n")

    add("## E2 subset fractions\n")
    add("Subsets are stratified and **nested** (each is a strict superset of the one below), so "
        "the data-hunger curve varies only in how much data, not which images.\n")
    add("| Fraction | Images | Per class | sha256 (first 16) |\n|---|---|---|---|")
    for frac in FRACTIONS:
        s = facts["subsets"][frac]
        add(f"| {frac:.0%} | {s['n']:,} | {s['per_class']:,} | `{s['sha256'][:16]}` |")
    add("")

    add("## Normalization\n")
    n = facts["normalization"]
    scope = f"{n['n_images']:,} images" + (" (--quick sample)" if n["quick"] else " (full train split)")
    add(f"Measured over {scope}.\n")
    add("| Channel | Normalized mean | Normalized std | Raw mean (45k) | Spec mean | "
        "Raw std (45k) | Spec std |\n|---|---|---|---|---|---|---|")
    for i, ch in enumerate("RGB"):
        add(f"| {ch} | {n['normalized_mean'][i]:+.4f} | {n['normalized_std'][i]:.4f} | "
            f"{n['raw_mean_measured'][i]:.4f} | {n['raw_mean_spec'][i]:.4f} | "
            f"{n['raw_std_measured'][i]:.4f} | {n['raw_std_spec'][i]:.4f} |")
    add("\nThe spec constants are the canonical **full-50k** CIFAR-10 statistics. Our train split "
        "holds 45k of those images, so a small difference in the raw columns is expected and is "
        "reported as a warning rather than a failure. The normalized columns are the ones that "
        "must be ~0 and ~1.\n")

    if figures:
        add("## Figures\n")
        for f in figures:
            add(f"- `{f}`")
        add("")
    out_path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 data pipeline checks + figures")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--quick", action="store_true",
                    help="sample 5,000 images for the normalization pass instead of all 45,000")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dpi = int(cfg.get("figures", {}).get("dpi", 200))
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    tbl_dir = ensure_dir(out_dir / "tables")

    print("=" * 78)
    print("  PHASE 1  |  DATA PIPELINE CHECKS")
    print("=" * 78)

    chk, facts = run_checks(cfg, args)

    figures: list[str] = []
    if not args.no_figures:
        print("\n9. Figures")
        for name, fn in [
            ("augmentations.png", lambda p: figure_augmentations(cfg, p, dpi)),
            ("class_distribution.png", lambda p: figure_class_distribution(facts, p, dpi)),
            ("mixup_cutmix_examples.png", lambda p: figure_mixup_cutmix(cfg, p, dpi)),
        ]:
            path = fig_dir / name
            fn(path)
            figures.append(f"results/figures/{name}")
            print(f"  [  ok  ] wrote results/figures/{name}")

    report = tbl_dir / "data_report.md"
    write_report(chk, facts, report, figures)
    print(f"\nwrote {report}")

    print("\n" + "=" * 78)
    n_fail, n_warn = len(chk.failed), len(chk.warned)
    if n_fail:
        print(f"  RESULT: {n_fail} CHECK(S) FAILED")
        for _, name, detail in chk.failed:
            print(f"    - {name}: {detail}")
        print("=" * 78)
        sys.exit(1)
    print(f"  RESULT: all {len(chk.rows)} checks passed"
          + (f" ({n_warn} warning(s) -- see the report)" if n_warn else ""))
    print("=" * 78)


if __name__ == "__main__":  # REQUIRED on Windows (DataLoader workers)
    main()

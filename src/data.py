"""CIFAR-10 data pipeline: splits, transforms, subsets, loaders.

Three things in here decide whether the whole showdown is valid, so they get
most of the comments:

1. **The 45k/5k split must be bit-identical everywhere.** ``data/`` is
   gitignored, so the split is *regenerated* on Kaggle rather than shipped. If
   Kaggle's split differed from the local one, "val accuracy" would mean two
   different things. The construction below is pure numpy with explicitly
   seeded ``RandomState`` objects -- no sklearn, no ``torch.randperm``, no
   ``random.shuffle`` -- and every cached file carries a sha256 of its indices
   so the two machines can be compared directly.

2. **Val must never see train augmentation.** The usual
   ``random_split(train_set, [45000, 5000])`` idiom silently gives the val
   split whatever transform the parent dataset carries -- i.e. random crops and
   flips on your model-selection set. :class:`CIFAR10Subset` wraps one
   transform-free base dataset so train and val can share the underlying images
   while holding genuinely different pipelines.

3. **Subsets are nested.** E2 asks what happens at 5/10/25/50/100% of the data.
   If each fraction drew independently, part of the curve's shape would just be
   which images got drawn. Here the 5% set is a strict subset of the 10% set,
   and so on, so the only thing changing along the x-axis is *how much* data.

Shapes are noted inline, e.g. ``# (B, 3, 32, 32)``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

try:  # works for `python -m src.data` and `import src.data`
    from src.utils import (
        ensure_dir,
        get_logger,
        load_json,
        make_generator,
        resolve_path,
        save_json,
        seed_worker,
        sha256_of_array,
    )
except ImportError:  # works for `python src/data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import (  # noqa: E402
        ensure_dir,
        get_logger,
        load_json,
        make_generator,
        resolve_path,
        save_json,
        seed_worker,
        sha256_of_array,
    )

# Spec Section 3. These are the canonical full-50k CIFAR-10 statistics.
CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

# Official torchvision label order -- index i is label i.
CLASSES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

NUM_TRAIN_TOTAL = 50_000   # CIFAR-10 train archive, before our val split
NUM_TEST = 10_000

# Seed offsets. The split shuffle and the subset shuffle must be independent,
# otherwise the "first k of the permutation" trick in get_subset_indices would
# correlate with how the val set was carved out.
_SPLIT_SEED_BASE = 0
_SUBSET_SEED_BASE = 1000

_ALGORITHM_VERSION = 1  # bump if the split construction ever changes

logger = get_logger("data")

__all__ = [
    "CIFAR10_MEAN", "CIFAR10_STD", "CLASSES",
    "download_cifar10", "get_raw_datasets", "build_transform", "denormalize",
    "get_split_indices", "get_subset_indices", "CIFAR10Subset",
    "build_datasets", "build_dataloaders", "MixupCutmix", "class_counts",
]


# ---------------------------------------------------------------------------
# Raw datasets
# ---------------------------------------------------------------------------


def download_cifar10(root: str | Path) -> None:
    """Fetch CIFAR-10 into ``root`` if it isn't already there."""
    r = ensure_dir(root)
    CIFAR10(root=str(r), train=True, download=True)
    CIFAR10(root=str(r), train=False, download=True)


def get_raw_datasets(root: str | Path, download: bool = False) -> tuple[CIFAR10, CIFAR10]:
    """Return ``(train_50k, test_10k)`` with ``transform=None``.

    Transform-free on purpose: :class:`CIFAR10Subset` attaches the transform,
    which is what lets the train and val views of these same 50k images use
    different pipelines.

    When the extracted dataset is already on disk, ``download`` is forced off.
    torchvision would otherwise md5 the 170 MB archive on *every* call, and
    this function is called from several places per script.
    """
    r = resolve_path(root)
    if download:
        if (r / "cifar-10-batches-py" / "test_batch").is_file():
            download = False
        else:
            ensure_dir(r)
    train = CIFAR10(root=str(r), train=True, download=download, transform=None)
    test = CIFAR10(root=str(r), train=False, download=download, transform=None)
    return train, test


def _labels_of(ds: CIFAR10) -> np.ndarray:
    """Labels as an int64 array. Pinned dtype keeps hashes platform-stable."""
    return np.asarray(ds.targets, dtype=np.int64)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def build_transform(
    augmentation: str = "standard",
    train: bool = True,
    aug_cfg: dict | None = None,
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
) -> Callable:
    """Build the image pipeline.

    Args:
        augmentation: ``"none"`` | ``"standard"`` | ``"strong"``.
        train: When False the augmentation argument is **ignored** and the
            evaluation pipeline (ToTensor -> Normalize) is returned. Val and
            test must never be augmented -- that is a hard protocol rule, not a
            default, so it is enforced here rather than left to callers.
        aug_cfg: Optional ``augmentation.<name>`` block from the config; when
            given it overrides the hard-coded spec defaults below.

    Note:
        Mixup and CutMix are deliberately absent. They operate on a whole batch
        (mixing image *pairs*), so they cannot live in a per-sample transform --
        see :class:`MixupCutmix`, applied inside the training loop.
    """
    augmentation = (augmentation or "none").lower()
    if augmentation not in {"none", "standard", "strong"}:
        raise ValueError(f"Unknown augmentation {augmentation!r}; expected none/standard/strong")

    normalize = [transforms.ToTensor(), transforms.Normalize(tuple(mean), tuple(std))]

    if not train or augmentation == "none":
        # (3, 32, 32) float, roughly zero-mean/unit-std per channel.
        return transforms.Compose(normalize)

    cfg = aug_cfg or {}
    crop = cfg.get("random_crop", {"size": 32, "padding": 4})
    flip = cfg.get("random_hflip", {"p": 0.5})

    # Geometric ops first, on the PIL image.
    pre: list[Callable] = [
        transforms.RandomCrop(int(crop.get("size", 32)), padding=int(crop.get("padding", 4))),
        transforms.RandomHorizontalFlip(p=float(flip.get("p", 0.5))),
    ]

    if augmentation == "standard":
        return transforms.Compose(pre + normalize)

    # --- strong (E3 only) ---------------------------------------------------
    ra = cfg.get("randaugment", {"num_ops": 2, "magnitude": 9})
    erase = cfg.get("random_erasing", {"p": 0.25})

    # RandAugment goes BEFORE ToTensor: its policy set includes Posterize,
    # Solarize, Equalize and AutoContrast, which are defined on uint8 images.
    # Running it on normalized floats is the classic silent-corruption bug here.
    pre.append(transforms.RandAugment(num_ops=int(ra.get("num_ops", 2)),
                                      magnitude=int(ra.get("magnitude", 9))))

    # RandomErasing goes AFTER Normalize: it fills with value 0, which in
    # normalized space is the dataset mean colour. Erasing before Normalize
    # would instead paint pure black and then shift it -- a different (and
    # distribution-skewing) operation.
    post = [transforms.RandomErasing(p=float(erase.get("p", 0.25)))]

    return transforms.Compose(pre + normalize + post)


def denormalize(x: torch.Tensor,
                mean: Sequence[float] = CIFAR10_MEAN,
                std: Sequence[float] = CIFAR10_STD) -> torch.Tensor:
    """Undo Normalize for display: ``x * std + mean``, clamped to [0, 1].

    Forgetting this is why augmentation figures usually come out looking like
    television static.
    """
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _split_cache_path(root: Path, val_per_class: int, seed: int) -> Path:
    return root / "splits" / f"split_val{val_per_class}_seed{seed}.json"


def get_split_indices(
    root: str | Path,
    val_per_class: int = 500,
    seed: int = 0,
    download: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Build (or load) the fixed stratified 45,000 / 5,000 train/val split.

    Returns a dict with ``"train"`` (45,000 int64 indices), ``"val"`` (5,000)
    and ``"meta"`` (sizes, seed, sha256 fingerprints).

    The construction, exactly:

    * for class ``c`` in 0..9, take its 5,000 indices in ascending order;
    * permute them with ``np.random.RandomState(seed + c)``;
    * first ``val_per_class`` go to val, the rest to train;
    * sort each final array.

    Two details earn their keep. Using a *per-class* ``RandomState(seed + c)``
    rather than one shared generator makes each class's draw independent of how
    many classes were processed before it -- so the split can't change if the
    iteration order ever does. Sorting at the end makes the stored arrays
    canonical, so the sha256 fingerprint reflects the *set* of chosen images,
    not the order they happened to be concatenated in.
    """
    r = resolve_path(root)
    cache = _split_cache_path(r, val_per_class, seed)

    if cache.is_file() and not force:
        payload = load_json(cache)
        train = np.asarray(payload["train"], dtype=np.int64)
        val = np.asarray(payload["val"], dtype=np.int64)
        _verify_split(train, val, payload.get("meta", {}), val_per_class, cache)
        return {"train": train, "val": val, "meta": payload["meta"]}

    base_train, _ = get_raw_datasets(r, download=download)
    labels = _labels_of(base_train)  # (50000,)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for c in range(len(CLASSES)):
        idx_c = np.flatnonzero(labels == c)            # ascending, deterministic
        rng = np.random.RandomState(_SPLIT_SEED_BASE + seed + c)
        shuffled = idx_c[rng.permutation(idx_c.shape[0])]
        val_parts.append(shuffled[:val_per_class])
        train_parts.append(shuffled[val_per_class:])

    train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val = np.sort(np.concatenate(val_parts)).astype(np.int64)

    meta = {
        "val_per_class": int(val_per_class),
        "seed": int(seed),
        "n_train": int(train.size),
        "n_val": int(val.size),
        "train_sha256": sha256_of_array(train),
        "val_sha256": sha256_of_array(val),
        "created_by": "src/data.py:get_split_indices",
        "algorithm_version": _ALGORITHM_VERSION,
    }
    _verify_split(train, val, meta, val_per_class, cache)

    ensure_dir(cache.parent)
    save_json({"train": train, "val": val, "meta": meta}, cache)
    logger.info("Built split: %d train / %d val (seed %d) -> %s",
                train.size, val.size, seed, cache.name)
    return {"train": train, "val": val, "meta": meta}


def _verify_split(train: np.ndarray, val: np.ndarray, meta: dict,
                  val_per_class: int, cache: Path) -> None:
    """Fail loudly on a corrupt or stale cache rather than training on it."""
    n_expected_val = val_per_class * len(CLASSES)
    if val.size != n_expected_val or train.size != NUM_TRAIN_TOTAL - n_expected_val:
        raise RuntimeError(
            f"Split at {cache} has {train.size}/{val.size} train/val, expected "
            f"{NUM_TRAIN_TOTAL - n_expected_val}/{n_expected_val}. Delete it and rebuild."
        )
    if np.intersect1d(train, val).size:
        raise RuntimeError(f"Split at {cache} leaks: train and val overlap.")
    union = np.union1d(train, val)
    if union.size != NUM_TRAIN_TOTAL or union[0] != 0 or union[-1] != NUM_TRAIN_TOTAL - 1:
        raise RuntimeError(f"Split at {cache} does not cover all {NUM_TRAIN_TOTAL} train images.")
    for key, arr in (("train_sha256", train), ("val_sha256", val)):
        if key in meta and meta[key] != sha256_of_array(arr):
            raise RuntimeError(
                f"Split at {cache} failed its {key} check -- the file was modified. "
                f"Delete it and rebuild with force=True."
            )


# ---------------------------------------------------------------------------
# Subsets (E2 data-hunger)
# ---------------------------------------------------------------------------


def _subset_cache_path(root: Path, fraction: float, seed: int, val_per_class: int) -> Path:
    return root / "splits" / f"subset_frac{fraction:g}_val{val_per_class}_seed{seed}.json"


def get_subset_indices(
    root: str | Path,
    fraction: float,
    seed: int = 0,
    val_per_class: int = 500,
    split_seed: int = 0,
    download: bool = False,
    force: bool = False,
) -> np.ndarray:
    """Stratified, **nested** subset of the 45k train indices.

    Args:
        fraction: In ``(0, 1]``. ``1.0`` short-circuits to the full train split.
        seed: The *subset* seed. It is deliberately decoupled from the training
            seed: E2 runs 3 training seeds at 10%, and all three must see the
            **same images**, otherwise the error bars would conflate "model
            variance" with "which images we happened to draw".

    Nesting comes free from the construction: each class is permuted once, and
    ``fraction`` simply takes a longer prefix of that one fixed permutation. So
    the 5% indices are a strict subset of the 10% indices, and so on, and the
    E2 curve varies only in *how much* data -- not *which*.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    r = resolve_path(root)
    split = get_split_indices(r, val_per_class=val_per_class, seed=split_seed, download=download)
    train_idx = split["train"]                              # (45000,) sorted

    if fraction == 1.0:
        return train_idx.copy()

    cache = _subset_cache_path(r, fraction, seed, val_per_class)
    if cache.is_file() and not force:
        payload = load_json(cache)
        sub = np.asarray(payload["indices"], dtype=np.int64)
        if payload.get("meta", {}).get("sha256") != sha256_of_array(sub):
            raise RuntimeError(f"Subset cache {cache} failed its sha256 check. Delete and rebuild.")
        return sub

    base_train, _ = get_raw_datasets(r, download=download)
    labels = _labels_of(base_train)
    train_labels = labels[train_idx]                        # (45000,)

    parts: list[np.ndarray] = []
    for c in range(len(CLASSES)):
        idx_c = train_idx[train_labels == c]                # (4500,) ascending
        rng = np.random.RandomState(_SUBSET_SEED_BASE + seed + c)
        shuffled = idx_c[rng.permutation(idx_c.shape[0])]
        k = int(round(fraction * idx_c.shape[0]))
        if k < 1:
            raise ValueError(
                f"fraction={fraction} leaves {k} images for class {CLASSES[c]}; too small."
            )
        parts.append(shuffled[:k])                          # prefix -> nesting

    sub = np.sort(np.concatenate(parts)).astype(np.int64)

    meta = {
        "fraction": float(fraction),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "val_per_class": int(val_per_class),
        "n": int(sub.size),
        "sha256": sha256_of_array(sub),
        "nested": True,
        "created_by": "src/data.py:get_subset_indices",
        "algorithm_version": _ALGORITHM_VERSION,
    }
    ensure_dir(cache.parent)
    save_json({"indices": sub, "meta": meta}, cache)
    logger.info("Built %.0f%% subset: %d images -> %s", fraction * 100, sub.size, cache.name)
    return sub


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class CIFAR10Subset(Dataset):
    """A view onto a transform-free CIFAR-10 base dataset.

    Why this exists instead of ``torch.utils.data.random_split``: ``Subset``
    delegates ``__getitem__`` to the parent, so the parent's single transform
    applies to every split. Train and val would then share one pipeline -- and
    since train needs ``RandomCrop`` + ``RandomHorizontalFlip``, your val set
    would be randomly augmented too. Model selection on a noisy val set is a
    subtle, silent way to make every number in the report slightly wrong.
    """

    def __init__(self, base: CIFAR10, indices: Sequence[int] | np.ndarray,
                 transform: Callable | None = None) -> None:
        self.base = base
        self.indices = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
        self.transform = transform
        self._targets = _labels_of(base)[self.indices]

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img, label = self.base[int(self.indices[i])]   # PIL (32, 32, 3), int
        if self.transform is not None:
            img = self.transform(img)                  # (3, 32, 32) float
        return img, int(label)

    @property
    def targets(self) -> np.ndarray:
        """Labels of the selected images, as an int64 array."""
        return self._targets

    def __repr__(self) -> str:
        split = "train" if getattr(self.base, "train", True) else "test"
        return f"CIFAR10Subset(n={len(self)}, base={split}, transform={self.transform is not None})"


def class_counts(targets: Sequence[int] | np.ndarray) -> dict[str, int]:
    """Count images per class name, always listing all 10 classes."""
    t = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(t, minlength=len(CLASSES))
    return {name: int(counts[i]) for i, name in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_datasets(
    data_cfg: dict,
    fraction: float = 1.0,
    augmentation: str = "standard",
    aug_cfg: dict | None = None,
    subset_seed: int | None = None,
) -> tuple[CIFAR10Subset, CIFAR10Subset, CIFAR10Subset]:
    """Build ``(train_ds, val_ds, test_ds)``.

    ``fraction`` subsets the **train** set only. Val stays at the full 5,000 and
    test at the full 10,000 (spec E2: "Validation set stays the full 5,000") --
    otherwise the data-hunger curve would be measured against a shrinking,
    noisier yardstick at exactly the points where it matters most.
    """
    root = resolve_path(data_cfg.get("root", "data"))
    val_per_class = int(data_cfg.get("val_per_class", 500))
    split_seed = int(data_cfg.get("split_seed", 0))
    sub_seed = int(data_cfg.get("subset_seed", 0)) if subset_seed is None else int(subset_seed)
    download = bool(data_cfg.get("download", True))
    mean = data_cfg.get("mean", CIFAR10_MEAN)
    std = data_cfg.get("std", CIFAR10_STD)

    base_train, base_test = get_raw_datasets(root, download=download)
    split = get_split_indices(root, val_per_class=val_per_class, seed=split_seed)

    if fraction >= 1.0:
        train_idx = split["train"]
    else:
        train_idx = get_subset_indices(root, fraction, seed=sub_seed,
                                       val_per_class=val_per_class, split_seed=split_seed)

    train_tf = build_transform(augmentation, train=True, aug_cfg=aug_cfg, mean=mean, std=std)
    eval_tf = build_transform("none", train=False, mean=mean, std=std)

    train_ds = CIFAR10Subset(base_train, train_idx, train_tf)
    val_ds = CIFAR10Subset(base_train, split["val"], eval_tf)
    test_ds = CIFAR10Subset(base_test, np.arange(NUM_TEST, dtype=np.int64), eval_tf)
    return train_ds, val_ds, test_ds


def build_dataloaders(
    data_cfg: dict,
    fraction: float = 1.0,
    augmentation: str = "standard",
    batch_size: int = 128,
    num_workers: int | None = None,
    seed: int = 0,
    pin_memory: bool | None = None,
    aug_cfg: dict | None = None,
    subset_seed: int | None = None,
    drop_last: bool = True,
    eval_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build ``(train_loader, val_loader, test_loader)``.

    ``drop_last=True`` on train keeps steps-per-epoch an exact integer, which
    matters for E2: those runs are *step-matched* across data fractions, so a
    ragged final batch would quietly change the comparison.
    """
    train_ds, val_ds, test_ds = build_datasets(
        data_cfg, fraction=fraction, augmentation=augmentation,
        aug_cfg=aug_cfg, subset_seed=subset_seed,
    )

    nw = int(data_cfg.get("num_workers", 4)) if num_workers is None else int(num_workers)
    nw = max(0, nw)

    # pin_memory only helps when there is a GPU to copy to; on a CPU-only box it
    # is pure overhead and torch emits a warning for it.
    if pin_memory is None:
        pin_memory = bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    # torch raises if persistent_workers/prefetch_factor are set with 0 workers.
    extra: dict[str, Any] = {}
    if nw > 0:
        extra["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        extra["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))

    eval_bs = int(eval_batch_size or batch_size)

    train_loader = DataLoader(
        train_ds, batch_size=int(batch_size), shuffle=True, drop_last=drop_last,
        num_workers=nw, pin_memory=pin_memory,
        generator=make_generator(seed), worker_init_fn=seed_worker, **extra,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_bs, shuffle=False, drop_last=False,
        num_workers=nw, pin_memory=pin_memory, **extra,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_bs, shuffle=False, drop_last=False,
        num_workers=nw, pin_memory=pin_memory, **extra,
    )

    logger.info(
        "Loaders | train %d (frac %.2f, aug=%s) | val %d | test %d | bs=%d workers=%d "
        "pin=%s | %d steps/epoch",
        len(train_ds), fraction, augmentation, len(val_ds), len(test_ds),
        batch_size, nw, pin_memory, len(train_loader),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Batch-level augmentation (E3)
# ---------------------------------------------------------------------------


class MixupCutmix:
    """Mixup / CutMix, switched per batch with probability ``prob``.

    Returns hard targets plus the mixing ratio, so the caller computes::

        lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

    which composes correctly with the protocol's label smoothing.
    """

    def __init__(self, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0,
                 prob: float = 0.5, num_classes: int = 10) -> None:
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.num_classes = int(num_classes)

    def __call__(self, x: torch.Tensor, y: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """``x``: (B, 3, 32, 32), ``y``: (B,) -> (mixed_x, y_a, y_b, lam)."""
        perm = torch.randperm(x.size(0), device=x.device)
        use_cutmix = np.random.rand() < self.prob

        if use_cutmix:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            x, lam = self._cutmix(x, perm, lam)
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            x = lam * x + (1.0 - lam) * x[perm]

        return x, y, y[perm], lam

    def _cutmix(self, x: torch.Tensor, perm: torch.Tensor, lam: float
                ) -> tuple[torch.Tensor, float]:
        _, _, h, w = x.shape                               # (B, 3, 32, 32)
        cut_ratio = float(np.sqrt(1.0 - lam))
        cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
        cy, cx = int(np.random.randint(h)), int(np.random.randint(w))

        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)

        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]

        # Recompute lam from the box that actually landed. When the box is
        # clipped at an image border the pasted area is smaller than the lam we
        # sampled, so using the sampled value would mis-weight the loss. Easy to
        # miss, and it biases E3 against CutMix.
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(h * w))
        return x, lam


# ---------------------------------------------------------------------------
# Self-check:  python -m src.data
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    import argparse

    ap = argparse.ArgumentParser(description="src/data.py self-check")
    ap.add_argument("--root", default="data")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    root = resolve_path(args.root)
    print(f"data root: {root}")
    if not args.no_download:
        download_cifar10(root)

    base_train, base_test = get_raw_datasets(root)
    assert tuple(base_train.classes) == CLASSES, f"class order drift: {base_train.classes}"
    assert len(base_train) == NUM_TRAIN_TOTAL and len(base_test) == NUM_TEST

    split = get_split_indices(root)
    train_idx, val_idx = split["train"], split["val"]
    print(f"\nsplit: {train_idx.size} train / {val_idx.size} val / {len(base_test)} test")
    print(f"  train sha256 {split['meta']['train_sha256']}")
    print(f"  val   sha256 {split['meta']['val_sha256']}")

    labels = _labels_of(base_train)
    print("\nper-class counts")
    tr_c, va_c = class_counts(labels[train_idx]), class_counts(labels[val_idx])
    te_c = class_counts(_labels_of(base_test))
    print(f"  {'class':<12}{'train':>8}{'val':>7}{'test':>7}")
    for name in CLASSES:
        print(f"  {name:<12}{tr_c[name]:>8}{va_c[name]:>7}{te_c[name]:>7}")
    assert set(tr_c.values()) == {4500} and set(va_c.values()) == {500}
    assert set(te_c.values()) == {1000}
    assert np.intersect1d(train_idx, val_idx).size == 0

    print("\nsubset fractions (nested)")
    prev: np.ndarray | None = None
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        sub = get_subset_indices(root, frac)
        counts = class_counts(labels[sub])
        expected = int(round(frac * 4500))
        assert all(abs(v - expected) <= 1 for v in counts.values()), counts
        if prev is not None:
            assert np.isin(prev, sub).all(), f"{frac} is not a superset of the previous fraction"
        prev = sub
        print(f"  {frac:>5.0%}  n={sub.size:>6}  per-class={expected:>5}  "
              f"sha256={sha256_of_array(sub)[:16]}...")

    print("\ntransforms")
    for aug in ("none", "standard", "strong"):
        tf = build_transform(aug, train=True)
        n_ops = len(tf.transforms)
        img, _ = base_train[0]
        out = tf(img)
        print(f"  {aug:<9} {n_ops} ops -> {tuple(out.shape)} "
              f"mean={out.mean():+.3f} std={out.std():.3f}")
        assert out.shape == (3, 32, 32)
    assert len(build_transform("strong", train=False).transforms) == 2, "val must not be augmented"

    print("\ndata.py self-check OK")

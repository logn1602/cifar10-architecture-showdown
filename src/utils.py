"""Shared plumbing: seeding, config loading, logging, JSON I/O, hashing.

Nothing in here is experiment-specific. The one rule worth stating up front:
library functions must never mutate global RNG state as a side effect. Anything
that needs randomness takes an explicit seed and builds a local generator, so a
result never depends on *when* a function was called relative to another.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

# Repo root: this file lives at <root>/src/utils.py, so parents[1] is <root>.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

__all__ = [
    "PROJECT_ROOT",
    "set_seed",
    "seed_worker",
    "make_generator",
    "load_config",
    "deep_merge",
    "get_logger",
    "save_json",
    "load_json",
    "human_time",
    "ensure_dir",
    "resolve_path",
    "sha256_of_array",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` against :data:`PROJECT_ROOT` when it is relative.

    Lets every config hold repo-relative paths (``data``, ``results/figures``)
    that still work when a script is launched from some other directory.
    """
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(path: str | Path) -> Path:
    """``mkdir -p`` for ``path``, returning it as a :class:`Path`."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (+ CUDA) for a training run.

    Args:
        seed: The run seed. The showdown uses 0, 1, 2 (spec Section 3).
        deterministic: When True, also force deterministic cuDNN kernels. This
            costs real throughput, so it stays off by default -- the spec
            explicitly allows ``cudnn.benchmark = True``. Use it only when
            chasing a reproducibility bug.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Spec Section 3: benchmark=True is fine and is noted in the README.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by torch
    """``worker_init_fn`` that gives each DataLoader worker a derived seed.

    torch seeds each worker's torch RNG itself, but leaves numpy and the stdlib
    ``random`` module unseeded -- so two workers would draw identical
    augmentation parameters. Deriving both from ``torch.initial_seed()`` fixes
    that and keeps the whole pipeline reproducible from the single run seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A CPU :class:`torch.Generator` for DataLoader shuffling.

    Passing an explicit generator (rather than relying on the global RNG) means
    the shuffle order depends only on ``seed``, not on how much other torch
    randomness happened to be consumed before the loader was built.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested dicts merge key-by-key; every other type (including lists) is
    replaced wholesale. Lists are replaced rather than concatenated so that
    e.g. ``train.seeds: [0]`` in a 9h-mode override cleanly *shrinks* the
    default ``[0, 1, 2]`` instead of appending to it.
    """
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(*paths: str | Path, overrides: dict | None = None) -> dict:
    """Load and deep-merge one or more YAML configs, left to right.

    Later paths win over earlier ones; ``overrides`` wins over everything.

    Raises:
        FileNotFoundError: naming the exact path that is missing.
    """
    cfg: dict = {}
    for path in paths:
        p = resolve_path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config {p} must parse to a mapping, got {type(loaded).__name__}")
        cfg = deep_merge(cfg, loaded)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Idempotent: safe to call repeatedly.

    Repeat calls never stack duplicate handlers (the classic cause of
    triple-printed log lines), and ``propagate`` is off so messages don't also
    surface through the root logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                     for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logger.addHandler(sh)

    if log_file is not None:
        target = resolve_path(log_file)
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == target
            for h in logger.handlers
        )
        if not already:
            ensure_dir(target.parent)
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
            logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Encoder for the types that leak into metrics dicts.

    metrics.json is built from numpy scalars (accuracies), numpy arrays
    (per-class counts) and Paths. Without this, ``json.dump`` dies with an
    unhelpful "Object of type int64 is not JSON serializable" at the end of a
    long run -- i.e. after the expensive part.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Write ``obj`` to ``path`` as UTF-8 JSON, creating parent directories."""
    p = resolve_path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
        fh.write("\n")


def load_json(path: str | Path) -> Any:
    """Read JSON from ``path``."""
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def human_time(seconds: float) -> str:
    """Format a duration as ``"1h 23m 45s"`` / ``"4m 02s"`` / ``"42.3s"``."""
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds < 60:
        return f"{sign}{seconds:.1f}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{sign}{hours}h {minutes:02d}m {secs:02d}s"
    return f"{sign}{minutes}m {secs:02d}s"


def sha256_of_array(arr: Iterable[int] | np.ndarray) -> str:
    """Stable sha256 of an integer array.

    The dtype is pinned to int64 *before* hashing. Without that pin, the same
    index array hashes differently on Windows (where numpy's default integer is
    int32) and on Linux/Kaggle (int64) -- which would defeat the entire point
    of this function, since ``data/`` is gitignored and the 45k/5k split is
    regenerated on each machine rather than shipped. This hash is how we prove
    the two splits are identical.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()


if __name__ == "__main__":
    assert human_time(42.34) == "42.3s", human_time(42.34)
    assert human_time(125) == "2m 05s", human_time(125)
    assert human_time(3725) == "1h 02m 05s", human_time(3725)

    a = np.arange(10, dtype=np.int32)
    b = np.arange(10, dtype=np.int64)
    assert sha256_of_array(a) == sha256_of_array(b), "dtype pin failed"

    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": [1, 2]}, {"a": {"y": 3}, "b": [9]})
    assert merged == {"a": {"x": 1, "y": 3}, "b": [9]}, merged

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print("utils.py self-check OK")

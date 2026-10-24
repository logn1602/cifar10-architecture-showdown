"""Shared plumbing for the evaluation-only experiments (E5, E6, E7).

All three work on *finished* E1 runs: they load each run's best checkpoint
(already selected on validation accuracy by the training loop) and push test
images -- corrupted, scrambled, or as they are -- through it. Nothing here
trains, and nothing here makes a modelling decision.

Conventions used throughout:

* Images travel as **uint8 NHWC numpy arrays** (exactly what CIFAR-10 stores)
  until the last moment, when :class:`Predictor` moves a batch to the device,
  scales it to [0, 1] and applies the protocol's normalisation. Corruptions and
  scrambles therefore operate on real pixel values, as in CIFAR-10-C.
* Outputs go under :func:`src.experiments.results_path`, so the CPU self-tests
  can redirect everything to a scratch directory.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from src.data import CLASSES, get_raw_datasets
    from src.evaluate import autocast_context
    from src.experiments import results_path
    from src.models import MODEL_ORDER, build_model
    from src.train import pick_amp_dtype
    from src.utils import ensure_dir, load_config, load_json, resolve_path, save_json
except ImportError:  # pragma: no cover - `python src/analysis.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import CLASSES, get_raw_datasets  # noqa: E402
    from src.evaluate import autocast_context  # noqa: E402
    from src.experiments import results_path  # noqa: E402
    from src.models import MODEL_ORDER, build_model  # noqa: E402
    from src.train import pick_amp_dtype  # noqa: E402
    from src.utils import ensure_dir, load_config, load_json, resolve_path, save_json  # noqa: E402

__all__ = [
    "Run", "e1_runs", "load_run_model", "test_set", "eval_device", "Predictor",
    "stratified_indices", "fixed_image_indices", "pyplot", "model_style", "save_fig",
    "table_path", "write_csv", "read_csv", "CLASSES",
]


# ---------------------------------------------------------------------------
# Finding and loading finished runs
# ---------------------------------------------------------------------------


@dataclass
class Run:
    model: str
    seed: int
    dir: Path
    metrics: dict


def e1_runs(models: Iterable[str] = MODEL_ORDER, seeds: Iterable[int] | None = None) -> list[Run]:
    """Completed E1 runs that have a ``best.pt``, in chronological model order."""
    root = results_path() / "runs" / "E1"
    wanted = None if seeds is None else {int(s) for s in seeds}
    runs: list[Run] = []
    for m in models:
        mdir = root / m
        if not mdir.is_dir():
            continue
        for sd in sorted((p for p in mdir.iterdir() if p.is_dir() and p.name.isdigit()),
                         key=lambda p: int(p.name)):
            seed = int(sd.name)
            if wanted is not None and seed not in wanted:
                continue
            mp, ck = sd / "metrics.json", sd / "best.pt"
            if not (mp.exists() and ck.exists()):
                continue
            metrics = load_json(mp)
            if metrics.get("status") == "completed":
                runs.append(Run(m, seed, sd, metrics))
    return runs


def load_run_model(run: Run, device: torch.device) -> nn.Module:
    """Rebuild a run's model from its own resolved config and load ``best.pt``.

    The architecture comes from the run's ``config.yaml`` rather than today's
    ``configs/models/*.yaml``, so editing a config later can never silently
    pair old weights with a different architecture.
    """
    model_cfg = None
    resolved = run.dir / "config.yaml"
    if resolved.exists():
        with resolved.open(encoding="utf-8") as fh:
            mc = (yaml.safe_load(fh) or {}).get("model_cfg") or {}
        model_cfg = mc.get("model", mc) or None
    if model_cfg is None:
        model_cfg = load_config(f"configs/models/{run.model}.yaml")["model"]
    model = build_model(run.model, model_cfg)
    state = torch.load(run.dir / "best.pt", map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


def test_set(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """The official 10,000 test images as ``(uint8 (N, 32, 32, 3), int64 (N,))``."""
    _, test = get_raw_datasets(resolve_path(cfg["data"].get("root", "data")),
                               download=bool(cfg["data"].get("download", True)))
    # torchvision keeps .data as a transposed (non-contiguous) view; copy once
    # here rather than on every batch and every corruption downstream.
    return (np.ascontiguousarray(test.data, dtype=np.uint8),
            np.asarray(test.targets, dtype=np.int64))


def eval_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class Predictor:
    """uint8 NHWC images in, logits / predictions / accuracy out.

    Uses the protocol's normalisation and the same AMP policy as training
    (fp16 on a T4, fp32 on CPU), in eval mode.
    """

    def __init__(self, cfg: dict, device: torch.device, batch_size: int | None = None) -> None:
        self.device = device
        self.mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
        self.amp = pick_amp_dtype(device, cfg["train"].get("amp", {}))
        self.batch_size = int(batch_size or cfg.get("eval", {}).get("batch_size", 1000))

    def to_input(self, images: np.ndarray) -> torch.Tensor:
        """uint8 (B, 32, 32, 3) -> normalised float (B, 3, 32, 32) on the device."""
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)                 # (B, 3, 32, 32)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def logits(self, model: nn.Module, images: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            x = self.to_input(images[i:i + self.batch_size])
            with autocast_context(self.device, self.amp):
                out.append(model(x).float().cpu())
        return torch.cat(out).numpy()                                  # (N, 10)

    def predict(self, model: nn.Module, images: np.ndarray) -> np.ndarray:
        return self.logits(model, images).argmax(1)

    def accuracy(self, model: nn.Module, images: np.ndarray, labels: np.ndarray) -> float:
        return float((self.predict(model, images) == labels).mean())


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------


def stratified_indices(labels: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    """``n_total`` indices, equal per class, drawn with a fixed seed (sorted)."""
    classes = np.unique(labels)
    per = max(1, n_total // len(classes))
    rng = np.random.RandomState(seed)
    picks = [rng.choice(np.flatnonzero(labels == c), size=min(per, int((labels == c).sum())),
                        replace=False) for c in classes]
    return np.sort(np.concatenate(picks))


def fixed_image_indices(cfg: dict, labels: np.ndarray) -> np.ndarray:
    """The 16 E7 images: 2 per class for 8 classes, chosen once by seed.

    Saved to ``tables/fixed_images.json`` the first time and read back
    afterwards, so every figure in E6 and E7 -- and any later re-run -- uses the
    same images. The choice never looks at a model's predictions; picking
    images a model happens to get right would be cherry-picking.
    """
    path = table_path("fixed_images.json")
    if path.exists():
        return np.asarray(load_json(path)["indices"], dtype=np.int64)
    e7 = cfg["experiments"]["e7"]
    rng = np.random.RandomState(int(e7["selection_seed"]))
    idx = []
    for name in e7["fixed_classes"]:
        c = CLASSES.index(name)
        idx += sorted(rng.choice(np.flatnonzero(labels == c), size=int(e7["images_per_class"]),
                                 replace=False).tolist())
    save_json({"indices": idx, "classes": [CLASSES[int(labels[i])] for i in idx],
               "selection": f"{e7['images_per_class']} per class for {e7['fixed_classes']}, "
                            f"RandomState({e7['selection_seed']}); never chosen by model output"},
              path)
    return np.asarray(idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------


def pyplot():
    """matplotlib.pyplot on the headless Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def model_style(cfg: dict, model: str) -> tuple[str, str]:
    """``(colour, display label)`` -- one fixed colour per model, everywhere."""
    return cfg["figures"]["model_colors"][model], cfg["figures"]["model_labels"][model]


def save_fig(fig, name: str, cfg: dict) -> Path:
    path = ensure_dir(results_path() / "figures") / name
    fig.savefig(path, dpi=int(cfg["figures"]["dpi"]), bbox_inches="tight")
    pyplot().close(fig)
    return path


def table_path(name: str) -> Path:
    return ensure_dir(results_path() / "tables") / name


def write_csv(rows: list[dict], path: Path, fields: list[str] | None = None) -> Path:
    ensure_dir(path.parent)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return path


def read_csv(path: Path) -> list[dict[str, Any]]:
    """Rows with numeric-looking fields converted back to numbers."""
    def conv(v: str) -> Any:
        for t in (int, float):
            try:
                return t(v)
            except ValueError:
                pass
        return v
    with path.open(encoding="utf-8") as fh:
        return [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(fh)]

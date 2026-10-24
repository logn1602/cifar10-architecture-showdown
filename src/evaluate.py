"""Evaluation: accuracy, loss, per-class accuracy, confusion matrix, ECE.

One function, :func:`evaluate`, is used for every val pass during training and
for the single test pass at the end of a run, so "val accuracy" and "test
accuracy" are computed by literally the same code.

Two details worth knowing:

* **Loss is plain cross-entropy here, without label smoothing.** Training uses
  smoothing 0.1 (spec Section 3), but a smoothed loss has a floor well above
  zero and depends on the smoothing value, which makes it a poor number to
  compare across runs. Accuracy-based checkpoint selection is unaffected.
* **ECE uses equal-width confidence bins** (15 by default, ``eval.ece_bins``),
  the convention from Guo et al. 2017, so the numbers are comparable with
  published calibration results.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

__all__ = [
    "evaluate", "expected_calibration_error", "confusion_matrix",
    "per_class_accuracy", "autocast_context",
]


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    """``torch.autocast`` for ``device``, or a no-op when AMP is off."""
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15) -> float:
    """ECE: the confidence-weighted gap between confidence and accuracy.

    Predictions are bucketed by their top-class probability; within each bucket
    we compare mean confidence with actual accuracy, and average the absolute
    gaps weighted by bucket size. A perfectly calibrated model that says "80%"
    is right 80% of the time and scores 0.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # (lo, hi] -- a confidence of exactly 0 cannot occur after a softmax.
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            gap = abs(confidences[mask].mean() - correct[mask].mean())
            ece += gap * mask.sum() / n
    return float(ece)


def confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """``cm[i, j]`` = number of images of true class ``i`` predicted as ``j``."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(targets), np.asarray(preds)), 1)
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Diagonal over row sums. Classes with no samples get NaN, not 0."""
    totals = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(totals > 0, np.diag(cm) / totals, np.nan)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    num_classes: int = 10,
    ece_bins: int = 15,
    input_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    channels_last: bool = False,
    full: bool = False,
    max_batches: int | None = None,
) -> dict:
    """Run ``model`` over ``loader`` in eval mode.

    Args:
        amp_dtype: Autocast dtype, or None for fp32. Logits are cast back to
            fp32 before the softmax so ECE is not computed in half precision.
        input_fn: Optional transform applied to each batch *on the device*
            (E4 uses it to upsample 32 -> 224 and re-normalise for ImageNet).
        full: When True, also return per-class accuracy, the confusion matrix,
            ECE and the raw per-image outputs. The cheap form (loss + accuracy)
            is what runs every epoch; the full form is the one-off test pass.
        max_batches: Debug-only cap for the smoke test. Never set in a real run.

    Returns:
        ``{"loss", "acc", "n"}`` plus, when ``full``: ``"per_class_acc"``,
        ``"confusion_matrix"``, ``"ece"``, ``"targets"``, ``"preds"``,
        ``"probs"`` (float16, (N, C)).
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if input_fn is not None:
            x = input_fn(x)
        with autocast_context(device, amp_dtype):
            logits = model(x)
        logits = logits.float()                           # (B, 10)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        total += y.numel()
        if full:
            all_probs.append(logits.softmax(dim=1).cpu())
            all_targets.append(y.cpu())

    if was_training:
        model.train()

    if total == 0:
        raise RuntimeError("evaluate() saw an empty loader")

    out = {"loss": total_loss / total, "acc": correct / total, "n": total}
    if not full:
        return out

    probs = torch.cat(all_probs).numpy()                  # (N, 10)
    targets = torch.cat(all_targets).numpy()              # (N,)
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    cm = confusion_matrix(targets, preds, num_classes)
    out.update({
        "per_class_acc": per_class_accuracy(cm),
        "confusion_matrix": cm,
        "ece": expected_calibration_error(conf, preds == targets, ece_bins),
        "targets": targets,
        "preds": preds,
        "probs": probs.astype(np.float16),
    })
    return out


if __name__ == "__main__":
    # Self-check on synthetic data: python -m src.evaluate
    rng = np.random.default_rng(0)

    # Perfectly calibrated by construction: correct with probability == confidence.
    conf = rng.uniform(0.1, 1.0, 200_000)
    correct = rng.uniform(size=conf.size) < conf
    ece = expected_calibration_error(conf, correct)
    assert ece < 0.01, f"calibrated ECE should be ~0, got {ece:.4f}"

    # Always 100% confident, right half the time -> ECE 0.5.
    ece = expected_calibration_error(np.ones(1000), np.arange(1000) % 2 == 0)
    assert abs(ece - 0.5) < 1e-9, ece

    cm = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]), 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]], cm
    pca = per_class_accuracy(cm)
    assert pca.tolist() == [0.5, 1.0, 1.0], pca

    # evaluate() end to end with a model that always predicts the label.
    class Oracle(nn.Module):
        def forward(self, x):
            return F.one_hot(x[:, 0, 0, 0].long(), 10).float() * 20

    xs = torch.zeros(64, 3, 32, 32)
    ys = torch.arange(64) % 10
    xs[:, 0, 0, 0] = ys.float()
    loader = DataLoader(list(zip(xs, ys)), batch_size=16)
    res = evaluate(Oracle(), loader, torch.device("cpu"), full=True)
    assert res["acc"] == 1.0 and res["n"] == 64, res["acc"]
    assert res["probs"].shape == (64, 10)
    print("evaluate.py self-check OK")

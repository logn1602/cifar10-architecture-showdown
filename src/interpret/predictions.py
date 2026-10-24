"""Mistake galleries and the model-agreement matrix.

Both are built from the per-image test predictions each E1 run already saved
(``test_outputs.npz``, written by the training loop's single test pass), so
neither needs another pass over the test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_test_outputs", "agreement_matrix", "confident_mistakes"]


def load_test_outputs(run_dir: Path) -> dict[str, np.ndarray]:
    d = np.load(Path(run_dir) / "test_outputs.npz")
    return {"targets": d["targets"], "preds": d["preds"], "probs": d["probs"].astype(np.float32)}


def agreement_matrix(preds: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Fraction of test images on which each pair of models predicts the same class."""
    names = list(preds)
    m = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            m[i, j] = float((preds[a] == preds[b]).mean())
    return names, m


def confident_mistakes(probs: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` wrong predictions made with the highest confidence."""
    preds = probs.argmax(1)
    wrong = np.flatnonzero(preds != targets)
    conf = probs[wrong, preds[wrong]]
    return wrong[np.argsort(-conf, kind="stable")[:k]]

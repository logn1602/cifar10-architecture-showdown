"""Penultimate-layer embeddings, projected to 2-D with UMAP.

Each model's last hidden representation (the vector its linear head reads) is
computed for a fixed, class-stratified set of test images and squashed to 2-D.
A UMAP picture is easy to over-read, so every panel also carries a number that
does not depend on the projection: the silhouette score of the classes in the
*original* feature space (cosine distance; higher = classes form tighter,
better-separated clusters).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["penultimate_features", "umap_2d", "silhouette"]


@torch.no_grad()
def penultimate_features(model: nn.Module, predictor, images: np.ndarray) -> np.ndarray:
    """(N, D) features from ``forward_features`` (ViT returns a tuple: take [0])."""
    from src.evaluate import autocast_context

    model.eval()
    out = []
    for i in range(0, len(images), predictor.batch_size):
        x = predictor.to_input(images[i:i + predictor.batch_size])
        with autocast_context(predictor.device, predictor.amp):
            f = model.forward_features(x)
        if isinstance(f, tuple):
            f = f[0]
        out.append(f.float().cpu())
    return torch.cat(out).numpy()


def umap_2d(features: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    import warnings

    import umap  # umap-learn; slow to import (numba), so only when needed
    from sklearn.preprocessing import StandardScaler

    z = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist),
                        random_state=int(seed), n_components=2)
    with warnings.catch_warnings():
        # A fixed random_state makes UMAP single-threaded, and it warns about
        # that every call. Reproducible plots are the point, so: expected.
        warnings.filterwarnings("ignore", message="n_jobs value")
        return reducer.fit_transform(z)


def silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(features, labels, metric="cosine"))

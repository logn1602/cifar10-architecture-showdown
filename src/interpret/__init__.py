"""E7 -- how each architecture "sees" (spec Section 6, E7).

* ``feature_maps.py``  first-layer filters, per-stage activation maps
* ``gradcam.py``       Grad-CAM, written out by hand (+ a library cross-check)
* ``attention.py``     ViT per-head attention, attention rollout, position grid
* ``embeddings.py``    penultimate features -> UMAP, silhouette scores
* ``predictions.py``   mistake galleries, model-agreement matrix
* ``e7.py``            the orchestration that produces every figure

``run_e7`` is imported lazily: UMAP pulls in numba, which is slow to import,
and nothing but E7 itself needs it.
"""

from __future__ import annotations


def run_e7(**kwargs):
    """Entry point used by the queue (``src.interpret:run_e7``)."""
    from src.interpret.e7 import run_e7 as _run_e7
    return _run_e7(**kwargs)


__all__ = ["run_e7"]

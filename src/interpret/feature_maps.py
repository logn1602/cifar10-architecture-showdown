"""First-layer filters and per-stage feature maps.

The first layer is the one place where a learned weight can be looked at
directly as an image. CNNs on natural images famously learn oriented edges and
colour blobs there; the interesting question in this showdown is whether
ViT's and Mixer's patch embeddings -- which have no notion of "nearby pixel"
built in -- end up learning similar detectors from data alone.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["first_layer_weights", "filter_mosaic", "top_channels"]


def first_layer_weights(name: str, model: nn.Module) -> tuple[np.ndarray, str] | None:
    """``(weights (K, kh, kw, C), description)`` for the model's first layer.

    MLP has no spatial first layer worth drawing (each unit sees all 3,072
    pixels), so it returns None.
    """
    if name == "convnext_mini":
        w = model.first_depthwise.weight                     # (64, 1, 7, 7)
        desc = "first depthwise 7x7"
    elif name in ("vit", "mlp_mixer"):
        w = model.patch_embed.proj.weight                    # (384, 3, 4, 4)
        desc = "patch embedding 4x4x3"
    elif name in ("simple_cnn", "resnet18"):
        w = next(m for m in model.modules() if isinstance(m, nn.Conv2d)).weight   # (64, 3, 3, 3)
        desc = "first conv 3x3x3"
    else:
        return None
    return w.detach().float().cpu().permute(0, 2, 3, 1).numpy(), desc


def filter_mosaic(w: np.ndarray, max_filters: int = 64, pad: int = 1
                  ) -> tuple[np.ndarray, int]:
    """Tile filters into one image; each filter min-max scaled on its own.

    When there are more than ``max_filters`` (ViT/Mixer have 384), the ones
    with the largest L2 norm are shown -- the filters the layer leans on most.
    Returns ``(mosaic (H, W, C), n_total_filters)``.
    """
    n_total = w.shape[0]
    if n_total > max_filters:
        norms = np.sqrt((w.reshape(n_total, -1) ** 2).sum(1))
        w = w[np.argsort(-norms)[:max_filters]]
    k, kh, kw, c = w.shape
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    mosaic = np.ones((rows * (kh + pad) + pad, cols * (kw + pad) + pad, c), dtype=np.float32)
    for i in range(k):
        f = w[i]
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        r, q = divmod(i, cols)
        y, x = pad + r * (kh + pad), pad + q * (kw + pad)
        mosaic[y:y + kh, x:x + kw] = f
    return mosaic, n_total


@torch.no_grad()
def top_channels(fmap: torch.Tensor, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """The ``n`` most active channels of one image's map (C, H, W)."""
    act = fmap.float().mean(dim=(1, 2))
    idx = torch.argsort(act, descending=True)[:n]
    return fmap[idx].float().cpu().numpy(), idx.cpu().numpy()

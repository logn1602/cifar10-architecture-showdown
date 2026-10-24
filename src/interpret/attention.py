"""ViT attention maps, attention rollout, and the positional-embedding grid.

Three ways to ask the Transformer where it looked:

* **Last-layer cls attention, per head**: row 0 of the final block's attention
  matrix -- how much the cls token (whose final state *is* the image
  representation) reads from each of the 64 patches, separately for each head.
* **Attention rollout** (Abnar & Zuidema, 2020): multiply the head-averaged
  attention of every layer, with the residual path added in, to estimate how
  much each patch influences the cls token end to end. Implemented on the
  model itself (:meth:`ViT.attention_rollout`).
* **Positional-embedding similarity**: cosine similarity between the learned
  embeddings of every pair of the 64 positions. ViT is never told that patch 9
  sits below patch 1; if it learned 2-D structure anyway, each position's map
  lights up around its own neighbourhood.

All maps are computed in fp32, outside autocast: these are explanations, and
half-precision rounding has no business in them.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["vit_attention", "upsample_maps", "pos_similarity"]


@torch.no_grad()
def vit_attention(model, x: torch.Tensor) -> dict:
    """-> ``heads`` (B, H, g, g), ``rollout`` (B, g, g), ``logits`` (B, 10)."""
    model.eval()
    logits, attns = model(x, return_attn=True)
    last = attns[-1].float()                                  # (B, heads, 65, 65)
    b, h = last.shape[:2]
    g = model.grid_size                                       # 8
    heads = last[:, :, 0, 1:].reshape(b, h, g, g)             # cls row, patches only
    rollout = model.attention_rollout(x).float().reshape(b, g, g)
    return {"heads": heads.cpu().numpy(), "rollout": rollout.cpu().numpy(),
            "logits": logits.float().cpu().numpy()}


def upsample_maps(maps: np.ndarray, size: int = 32) -> np.ndarray:
    """(..., g, g) -> (..., size, size), bilinear, each map min-max scaled to [0, 1]."""
    lead = maps.shape[:-2]
    t = torch.from_numpy(np.asarray(maps, dtype=np.float32)).reshape(-1, 1, *maps.shape[-2:])
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    flat = up.flatten(1)
    lo, hi = flat.min(1).values[:, None, None], flat.max(1).values[:, None, None]
    up = (up - lo) / (hi - lo + 1e-7)
    return up.reshape(*lead, size, size).numpy()


@torch.no_grad()
def pos_similarity(model) -> np.ndarray:
    """(64, 64) cosine similarity between learned position embeddings."""
    return model.pos_embed_similarity().float().cpu().numpy()

"""Building blocks shared by more than one architecture.

Keeping these in one place matters for the fairness protocol: ViT, MLP-Mixer and
ConvNeXt all use stochastic depth, and if each file rolled its own slightly
different version, part of any accuracy gap between them would just be that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DropPath", "LayerNorm2d", "drop_path_rates", "init_transformer_weights"]


def drop_path_rates(depth: int, max_rate: float) -> list[float]:
    """Linearly increasing stochastic-depth rates over ``depth`` blocks.

    Block 0 gets 0.0 and the last block gets ``max_rate`` -- the standard
    schedule. Early blocks are kept intact because everything downstream depends
    on them; the deepest blocks are the most redundant and so the safest to drop.
    """
    if depth == 1:
        return [0.0]
    return [max_rate * i / (depth - 1) for i in range(depth)]


class DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch, per sample.

    Note the ``/ keep_prob`` rescale. Without it the residual branch would have
    a smaller expected magnitude at eval time (when nothing is dropped) than
    during training, and the network would be systematically mis-calibrated --
    the same inverted-dropout argument, applied to a whole block.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"drop-path probability must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.p
        # One Bernoulli draw per sample, broadcast over every other dim.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p:.3f}"


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of a ``(B, C, H, W)`` tensor.

    ``nn.LayerNorm`` normalises over the *last* dimensions, so on NCHW data it
    would normalise over (H, W) -- per-channel spatial normalisation, which is a
    different operation entirely. ConvNeXt wants per-position normalisation
    across channels, hence this channels-first variant.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)                    # (B, 1, H, W)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)     # (B, 1, H, W)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

    def extra_repr(self) -> str:
        return f"{self.num_channels}, eps={self.eps}"


def init_transformer_weights(module: nn.Module) -> None:
    """ViT / Mixer / ConvNeXt init: truncated normal (std 0.02), zero biases.

    Applied with ``model.apply(...)``. PyTorch's default ``nn.Linear`` init is
    Kaiming-uniform, which is tuned for ReLU convnets and leaves pre-norm
    transformers with activations large enough to make the first few hundred
    steps unstable. std=0.02 is what ViT, Mixer and ConvNeXt all use -- and
    since the protocol forbids per-model tuning beyond the learning rate,
    getting the published init right is how each model gets a fair shot.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, LayerNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

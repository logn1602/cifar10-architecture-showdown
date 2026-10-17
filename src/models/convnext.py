"""ConvNeXt-mini -- "a CNN rebuilt with Transformer-era ideas".

Year: 2022 (Liu et al., A ConvNet for the 2020s).
Key idea: take a plain ResNet and, one at a time, adopt the design choices that
      came with Transformers -- large depthwise kernels (7x7) for a wide
      receptive field, LayerNorm instead of BatchNorm, GELU instead of ReLU,
      an inverted bottleneck (narrow -> 4x wide -> narrow), and far fewer
      normalisation and activation layers per block. The result matches or
      beats a Swin Transformer while remaining a pure convolutional network.
Inductive bias: convolution's locality and translation equivariance, kept
      deliberately -- plus a much larger effective receptive field than a stack
      of 3x3s, approaching what attention gets globally.

This is the interesting contrast with ViT. If ConvNeXt wins, the gap was never
about attention; it was about the training recipe and the design details that
Transformer papers happened to popularise. The 7x7 depthwise convolution is
the deliberate echo of a local attention window.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import DropPath, LayerNorm2d, drop_path_rates, init_transformer_weights

META = {
    "display_name": "ConvNeXt-mini",
    "year": 2022,
    "key_idea": "A ConvNet with Transformer design choices: 7x7 depthwise, LN, GELU, inverted bottleneck.",
    "inductive_bias": "Convolution's locality + translation equivariance, with a large receptive field.",
    "family": "convolutional",
}


class ConvNeXtBlock(nn.Module):
    """dwConv7x7 -> LN -> Linear(d, 4d) -> GELU -> Linear(4d, d) -> scale -> residual.

    Two things to note.

    The 7x7 convolution is **depthwise** -- one filter per channel, no mixing
    across channels. That is what makes a kernel this large affordable: 7x7
    depthwise on 512 channels costs 25k parameters, where a dense 7x7 would
    cost 12.8M. Channel mixing is handed off to the two 1x1s, exactly as a
    Transformer block separates spatial mixing (attention) from channel mixing
    (the MLP).

    The pointwise convolutions are written as ``nn.Linear`` on channels-last
    data. A 1x1 conv and a Linear over the channel axis are the same operation;
    permuting once and using Linear is the official implementation's choice and
    is faster in practice.
    """

    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-6, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)          # applied channels-last
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
        # LayerScale: a learned per-channel gain initialised at 1e-6, so every
        # residual branch starts off contributing essentially nothing and the
        # network begins as an identity. Without it, deep ConvNeXt stacks are
        # unstable early in training.
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) \
            if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)                       # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)                # (B, H, W, C) -- channels last
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)                # back to (B, C, H, W)
        return shortcut + self.drop_path(x)


class ConvNeXtMini(nn.Module):
    """ConvNeXt for 32x32: dims [64,128,256,512], depths [2,2,6,2].

    Like ResNet, the ImageNet stem is wrong here: ConvNeXt normally starts with
    a 4x4 stride-4 "patchify" conv, which on a 32x32 image would leave an 8x8
    map immediately. The stem is a 3x3 stride-1 conv instead, so the stages
    run 32 -> 16 -> 8 -> 4.

    Args:
        num_classes: Output classes.
        dims: Channel width per stage.
        depths: Blocks per stage (most capacity in stage 3, as in the paper).
        drop_path_rate: Max stochastic-depth rate, linearly increasing over all
            12 blocks (not restarted per stage).
        layer_scale_init: LayerScale gamma init.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dims: tuple[int, ...] = (64, 128, 256, 512),
        depths: tuple[int, ...] = (2, 2, 6, 2),
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(dims) != len(depths):
            raise ValueError(f"dims {dims} and depths {depths} must be the same length")

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # One global schedule across all blocks, so block 11 gets the full rate
        # regardless of which stage it sits in.
        dpr = drop_path_rates(sum(depths), drop_path_rate)

        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        cur = 0
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            if i == 0:
                self.downsamples.append(nn.Identity())
            else:
                # Norm BEFORE the downsampling conv -- ConvNeXt's ordering.
                self.downsamples.append(nn.Sequential(
                    LayerNorm2d(dims[i - 1], eps=1e-6),
                    nn.Conv2d(dims[i - 1], dim, kernel_size=2, stride=2),
                ))
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dim, dpr[cur + j], layer_scale_init) for j in range(depth)
            ]))
            cur += depth

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)     # after GAP, channels-last
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(init_transformer_weights)
        # LayerScale gammas must survive the generic init above.
        for m in self.modules():
            if isinstance(m, ConvNeXtBlock) and m.gamma is not None:
                nn.init.constant_(m.gamma, layer_scale_init)

    @torch.jit.unused
    def no_weight_decay(self) -> set[str]:
        return {"gamma"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """-> (B, 512) penultimate features."""
        x = self.stem(x)                              # (B, 3, 32, 32) -> (B, 64, 32, 32)
        for down, stage in zip(self.downsamples, self.stages):
            x = stage(down(x))                        # ... -> (B, 512, 4, 4)
        x = x.mean(dim=(-2, -1))                      # GAP -> (B, 512)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))    # (B, 10)

    @torch.no_grad()
    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Activations after each stage -- for E7."""
        maps = []
        x = self.stem(x)
        for down, stage in zip(self.downsamples, self.stages):
            x = stage(down(x))
            maps.append(x)
        return maps

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last conv stage -- the Grad-CAM target for E7."""
        return self.stages[-1]

    @property
    def first_depthwise(self) -> nn.Module:
        """First depthwise 7x7 conv -- its filters are plotted in E7."""
        return self.stages[0][0].dwconv

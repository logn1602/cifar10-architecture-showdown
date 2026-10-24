"""Simple CNN -- "nearby pixels matter".

Year: 1998 (LeNet-5), in the VGG-style 3x3 form that became standard in 2014.
Key idea: slide a small learned filter over the image. The same weights are
      reused at every position, and stacking conv+pool builds a hierarchy --
      edges, then textures, then parts.
Inductive bias: **locality, translation equivariance, weight sharing.** A 3x3
      kernel can only see a 3x3 neighbourhood, which hard-codes the assumption
      that nearby pixels are related. Sharing those weights across all
      positions means a feature learned in one corner transfers everywhere for
      free -- and cuts the parameter count enormously. That is why this model
      beats the MLP with *one third* of its parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

META = {
    "display_name": "Simple CNN",
    "year": 1998,
    "key_idea": "Small 3x3 convolutions, pool, repeat. Weights shared across positions.",
    "inductive_bias": "Locality + translation equivariance + weight sharing.",
    "family": "convolutional",
}


class ConvBNReLU(nn.Sequential):
    """Conv3x3 -> BatchNorm -> ReLU. Conv bias is redundant before BN."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SimpleCNN(nn.Module):
    """3 stages of (Conv-BN-ReLU) x2 + MaxPool, then GAP -> Dropout -> Linear.

    Spatial progression: 32x32 -> 16x16 -> 8x8 -> 4x4, with channels
    64 -> 128 -> 256. The classic trade: halve the resolution, double the width.

    Args:
        num_classes: Output classes.
        channels: Output channels of each stage.
        dropout: Dropout before the classifier.
        blocks_per_stage: Conv-BN-ReLU units per stage, before the pool.
    """

    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.3,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            block: list[nn.Module] = []
            for _ in range(blocks_per_stage):
                block.append(ConvBNReLU(in_ch, out_ch))
                in_ch = out_ch
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            stages.append(nn.Sequential(*block))
        self.features = nn.Sequential(*stages)

        # Global average pooling, not a flatten: it collapses 4x4 spatial
        # positions into one vector per channel, so the head has 256 inputs
        # instead of 4096 and the model stays translation-tolerant.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """-> (B, 256) penultimate features."""
        x = self.features(x)             # (B, 3, 32, 32) -> (B, 256, 4, 4)
        x = self.pool(x).flatten(1)      # (B, 256)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.forward_features(x)))   # (B, 10)

    @torch.no_grad()
    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Activation maps after each stage -- used by E7 (Phase 8)."""
        maps = []
        for stage in self.features:
            x = stage(x)
            maps.append(x)               # (B, C_i, H_i, W_i)
        return maps

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last conv layer (stage 3's second Conv-BN-ReLU) -- the Grad-CAM target.

        Taken *before* the final max-pool, as Grad-CAM conventionally is, so its
        map is 8x8. ResNet-18's and ConvNeXt's last stages have no pooling of
        their own and come out at 4x4 -- a real difference in how coarse their
        maps can be, not an artefact of this choice.
        """
        return self.features[-1][-2]     # (B, 256, 8, 8)

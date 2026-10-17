"""ResNet-18 -- "go deep with skip connections".

Year: 2015 (He et al., Deep Residual Learning for Image Recognition).
Key idea: have each block learn a *residual* F(x) and output F(x) + x. The
      identity path gives gradients a route back to early layers that does not
      pass through any weights, so adding depth stops making optimisation
      harder. Before this, plain 30-layer networks trained *worse* than
      20-layer ones -- not from overfitting, but because they could not be
      optimised at all.
Inductive bias: everything the CNN has (locality, translation equivariance,
      weight sharing) plus a strong prior that each block should *refine* its
      input rather than replace it. The network can fall back to the identity
      wherever a block is not useful.

CIFAR variant: the ImageNet stem (7x7 stride-2 conv + 3x3 stride-2 maxpool)
would take a 32x32 image down to 8x8 before the first residual block -- throwing
away most of the image. So the stem here is a single 3x3 stride-1 conv with no
pooling, and downsampling happens only inside stages 2-4: 32 -> 16 -> 8 -> 4.
"""

from __future__ import annotations

import torch
import torch.nn as nn

META = {
    "display_name": "ResNet-18",
    "year": 2015,
    "key_idea": "Residual connections let gradients skip layers, making depth trainable.",
    "inductive_bias": "CNN bias + identity shortcuts; blocks refine rather than replace.",
    "family": "convolutional",
}


class BasicBlock(nn.Module):
    """Two 3x3 convs with a residual connection.

    When the block changes resolution or width, the shortcut cannot be a plain
    identity -- the shapes differ. A 1x1 stride-2 convolution projects the input
    to match (ResNet's "option B"), which is what the spec asks for.
    """

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Add BEFORE the final ReLU: this keeps the identity path completely
        # linear, which is the whole point -- a ReLU on the shortcut would
        # clip negative gradients on the way back.
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """CIFAR ResNet-18: stem + [2, 2, 2, 2] BasicBlocks over 64/128/256/512.

    Args:
        num_classes: Output classes.
        layers: BasicBlocks per stage.
        channels: Output channels per stage.
        zero_init_residual: Zero-initialise the last BN in each block.
    """

    def __init__(
        self,
        num_classes: int = 10,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        channels: tuple[int, ...] = (64, 128, 256, 512),
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        # CIFAR stem: 3x3 stride 1, NO max-pool. Output stays 32x32.
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        in_ch = channels[0]
        for i, (n_blocks, out_ch) in enumerate(zip(layers, channels)):
            stride = 1 if i == 0 else 2        # 32 -> 32 -> 16 -> 8 -> 4
            blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
            blocks += [BasicBlock(out_ch, out_ch, stride=1) for _ in range(n_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels[-1], num_classes)

        self.apply(self._init_weights)

        if zero_init_residual:
            # Zero the last BN gamma in every block, so each block starts as an
            # exact identity. The network then begins training as if it were
            # shallow and "grows" depth as the residual branches become useful.
            # Worth ~0.5pp on CIFAR and costs nothing.
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

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
        """-> (B, 512) penultimate features."""
        x = self.stem(x)                 # (B, 3, 32, 32) -> (B, 64, 32, 32)
        x = self.stages(x)               # -> (B, 512, 4, 4)
        return self.pool(x).flatten(1)   # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))   # (B, 10)

    @torch.no_grad()
    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Activations after the stem and after each stage -- for E7."""
        maps = []
        x = self.stem(x)
        maps.append(x)
        for stage in self.stages:
            x = stage(x)
            maps.append(x)
        return maps

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last conv stage -- the Grad-CAM target for E7."""
        return self.stages[-1]

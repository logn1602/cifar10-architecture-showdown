"""MLP -- "no idea images have structure".

Year: ~1986 (backprop-era fully-connected nets; the 1958 perceptron with
      hidden layers trained by gradient descent).
Key idea: flatten the image into a vector and stack dense layers. Every input
      pixel connects to every hidden unit.
Inductive bias: **none**. The model has no notion that pixel (5, 5) sits next
      to pixel (5, 6), or that an object keeps its identity when it moves. The
      3,072 inputs are an unordered bag of numbers -- you could shuffle them
      with a fixed permutation before training and the model would do exactly
      as well. Shifting an image one pixel to the left produces a completely
      new input vector, which the network has to learn about separately.

This is the control condition for the whole showdown: the floor you get with
zero architectural knowledge about images.
"""

from __future__ import annotations

import torch
import torch.nn as nn

META = {
    "display_name": "MLP",
    "year": 1986,
    "key_idea": "Flatten the image, stack fully-connected layers.",
    "inductive_bias": "None. Pixels are an unordered bag of numbers; a 1-pixel shift is a new input.",
    "family": "fully-connected",
}


class MLP(nn.Module):
    """3072 -> 1024 -> 512 -> 256 -> 10, with BatchNorm + GELU + Dropout.

    Args:
        num_classes: Output classes (10 for CIFAR-10).
        hidden_dims: Widths of the hidden layers.
        dropout: Dropout probability after each hidden block.
        in_shape: Input shape, used to compute the flattened size.
    """

    def __init__(
        self,
        num_classes: int = 10,
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.2,
        in_shape: tuple[int, int, int] = (3, 32, 32),
    ) -> None:
        super().__init__()
        in_features = int(torch.tensor(in_shape).prod())      # 3*32*32 = 3072
        self.in_shape = in_shape
        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []
        prev = in_features
        for dim in hidden_dims:
            # Linear -> BatchNorm1d -> GELU -> Dropout, per spec 5.1.
            # Bias is off: BatchNorm immediately re-centres, so the bias would
            # be an unused parameter that weight decay then pulls toward zero.
            layers += [
                nn.Linear(prev, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = dim
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # Kaiming-normal suits the GELU/ReLU family and keeps activation
            # variance stable through 3 dense layers.
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """-> (B, 256), the penultimate features used by E7's UMAP plots."""
        x = self.flatten(x)          # (B, 3, 32, 32) -> (B, 3072)
        return self.features(x)      # (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))   # (B, 10)

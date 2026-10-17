"""MLP-Mixer -- "do we even need attention?".

Year: 2021 (Tolstikhin et al., MLP-Mixer: An all-MLP Architecture for Vision).
Key idea: keep the Transformer's patch-token layout but delete the attention.
      Mix information two ways, alternately: a **token-mixing** MLP applied
      across the 64 patch positions (one shared MLP, run per channel), and a
      **channel-mixing** MLP applied within each patch (one shared MLP, run per
      token). Both are plain dense layers.
Inductive bias: the patch grid, same as ViT -- but with one telling difference.
      Attention computes its mixing weights *from the input*, so which patches
      interact changes per image. Mixer's token-mixing weights are **fixed
      after training**: patch 7 always contributes to patch 23 with the same
      weight, whatever the image. That makes it a position-dependent operator,
      and it means Mixer cannot reorder itself for a scrambled input -- which
      is precisely what E6's patch-shuffle experiment probes.

This model is the control for ViT: same tokens, same capacity, same lack of
convolutional prior, but no input-dependent routing. Any gap between them is
attributable to attention itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import DropPath, drop_path_rates, init_transformer_weights
from .vit import PatchEmbed

META = {
    "display_name": "MLP-Mixer",
    "year": 2021,
    "key_idea": "Alternate token-mixing and channel-mixing MLPs; no attention at all.",
    "inductive_bias": "Patch grid only, and mixing is fixed per position (not input-dependent).",
    "family": "mlp-mixer",
}


class FeedForward(nn.Module):
    """dim -> hidden -> dim with GELU. Used for both mixing directions."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MixerBlock(nn.Module):
    """One Mixer block: token-mixing, then channel-mixing, each residual."""

    def __init__(self, num_tokens: int = 64, dim: int = 384,
                 token_hidden: int = 192, channel_hidden: int = 1536,
                 drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = FeedForward(num_tokens, token_hidden, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = FeedForward(dim, channel_hidden, drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64, 384)
        # --- token mixing: mix ACROSS patches, independently per channel -----
        # The transpose is the whole trick. nn.Linear acts on the last dim, so
        # to mix positions we move the token axis last, apply the MLP, and move
        # it back. The same 64->192->64 MLP is shared by all 384 channels.
        y = self.norm1(x).transpose(1, 2)            # (B, 384, 64)
        y = self.token_mlp(y).transpose(1, 2)        # (B, 64, 384)
        x = x + self.drop_path(y)

        # --- channel mixing: mix WITHIN each patch, independently per token --
        x = x + self.drop_path(self.channel_mlp(self.norm2(x)))
        return x


class MLPMixer(nn.Module):
    """Mixer for 32x32: patch 4, dim 384, 8 blocks, no cls token.

    Args:
        num_classes: Output classes.
        img_size / patch_size: 32 and 4 -> 64 tokens.
        embed_dim: Channel width (384).
        depth: Number of Mixer blocks (8).
        token_hidden / channel_hidden: Hidden widths of the two MLPs.
        drop_path_rate: Max stochastic-depth rate, linearly increasing.
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 384,
        depth: int = 8,
        token_hidden: int = 192,
        channel_hidden: int = 1536,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_tokens = self.patch_embed.num_patches     # 64
        self.grid_size = self.patch_embed.grid_size   # 8
        self.embed_dim = embed_dim

        # No cls token and no positional embedding: position is implicit in
        # which row of the token-mixing MLP a patch lands on. That is also why
        # the mixing is position-dependent in a way attention is not.
        dpr = drop_path_rates(depth, drop_path_rate)
        self.blocks = nn.Sequential(*[
            MixerBlock(num_tokens, embed_dim, token_hidden, channel_hidden, drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(init_transformer_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """-> (B, 384) penultimate features (mean over tokens)."""
        x = self.patch_embed(x)          # (B, 3, 32, 32) -> (B, 64, 384)
        x = self.blocks(x)               # (B, 64, 384)
        x = self.norm(x)
        return x.mean(dim=1)             # global average pool over tokens -> (B, 384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))   # (B, 10)

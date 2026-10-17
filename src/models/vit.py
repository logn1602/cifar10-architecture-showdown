"""Vision Transformer -- "no convolutions, just attention".

Year: 2020 (Dosovitskiy et al., An Image is Worth 16x16 Words).
Key idea: cut the image into fixed patches, embed each as a token, and let
      every token attend to every other token. Unlike a convolution, the
      mixing weights are *computed from the input* -- which patches talk to
      which is decided per image, not baked into the kernel.
Inductive bias: **almost none.** The only structure given is the patch grid,
      and even that is weak: positions are learned embeddings, so at
      initialisation the model does not know patch 0 is adjacent to patch 1 any
      more than it is to patch 63. It has to *learn* 2-D locality from data --
      which is exactly why it is so data-hungry, and why E2 is the headline
      experiment. (E7 checks whether it learned: the positional-embedding
      similarity grid should develop visible 2-D structure.)

Attention is written out by hand here -- q/k/v projections, scaled dot product,
softmax, head reshaping -- rather than calling
``F.scaled_dot_product_attention``, both because that is the point of the
exercise and because E7 needs the attention matrices themselves, which the
fused kernel never materialises.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DropPath, drop_path_rates, init_transformer_weights

META = {
    "display_name": "ViT",
    "year": 2020,
    "key_idea": "Patch the image into tokens; every token attends to every other.",
    "inductive_bias": "Almost none -- only a patch grid. 2-D locality must be learned.",
    "family": "transformer",
}


class PatchEmbed(nn.Module):
    """Image -> patch tokens, via a strided convolution.

    A Conv2d with kernel == stride == patch_size is exactly a linear projection
    of each flattened non-overlapping patch; it is just the fastest way to
    express that in PyTorch. With patch 4 on 32x32 this gives an 8x8 grid = 64
    tokens.
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 384) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.grid_size = img_size // patch_size          # 8
        self.num_patches = self.grid_size**2             # 64
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # (B, 3, 32, 32) -> (B, 384, 8, 8)
        x = x.flatten(2)                 # (B, 384, 64)
        return x.transpose(1, 2)         # (B, 64, 384)


class Attention(nn.Module):
    """Multi-head self-attention, written out explicitly."""

    def __init__(self, dim: int = 384, num_heads: int = 6,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads                 # 384 / 6 = 64
        # 1/sqrt(d_k). Without it the dot products grow with head_dim and push
        # the softmax into a near-one-hot regime where gradients vanish.
        self.scale = self.head_dim**-0.5

        # One fused projection for q, k and v -- mathematically identical to
        # three separate Linear layers, but a single matmul.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, C = x.shape                                # (B, 65, 384)

        qkv = self.qkv(x)                                # (B, 65, 1152)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                 # (3, B, heads, 65, 64)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each (B, 6, 65, 64)

        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, 6, 65, 65)
        attn = attn.softmax(dim=-1)                      # rows sum to 1
        attn = self.attn_drop(attn)

        out = attn @ v                                   # (B, 6, 65, 64)
        out = out.transpose(1, 2).reshape(B, N, C)       # (B, 65, 384)
        out = self.proj_drop(self.proj(out))
        # Return the PRE-dropout weights when asked: E7 visualises them, and
        # dropout noise would be an artefact of training mode, not the model.
        return out, (attn if return_attn else None)


class Mlp(nn.Module):
    """Position-wise feed-forward: dim -> hidden -> dim, GELU in between."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer block.

    Pre-norm (LN *inside* each residual branch) rather than the original
    post-norm: it leaves a clean identity path from input to output, so deep
    stacks train without a learning-rate warmup hack. The protocol does warm up
    anyway, but post-norm at depth 7 would still be needlessly fragile.
    """

    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_ratio: float = 2.0,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, return_attn: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_w = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_w


class ViT(nn.Module):
    """ViT for 32x32: patch 4, dim 384, depth 7, 6 heads, MLP ratio 2.

    MLP ratio 2 (384 -> 768) rather than the usual 4 is what the spec asks for
    and what keeps the model at ~8.3M params, comparable to ConvNeXt-mini and
    MLP-Mixer. Matching capacity across architectures is the whole point.

    Args:
        num_classes: Output classes.
        img_size / patch_size: 32 and 4 -> 64 patch tokens.
        embed_dim / depth / num_heads: 384 / 7 / 6 (head_dim 64).
        mlp_ratio: Hidden width of the feed-forward, as a multiple of embed_dim.
        drop_path_rate: Max stochastic-depth rate, linearly increasing.
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 384,
        depth: int = 7,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches        # 64
        self.grid_size = self.patch_embed.grid_size       # 8

        # A learned token that attends to every patch; its final state is the
        # image representation. It carries no spatial position of its own,
        # which is what makes its attention row a readable "what did the model
        # look at?" map in E7.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # (1, 65, 384)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = drop_path_rates(depth, drop_path_rate)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_transformer_weights)

    @torch.jit.unused
    def no_weight_decay(self) -> set[str]:
        """Params the protocol exempts from weight decay (spec Section 3)."""
        return {"pos_embed", "cls_token"}

    def forward_features(self, x: torch.Tensor, return_attn: bool = False
                         ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """-> (cls_feature (B, 384), per-layer attention [(B, 6, 65, 65), ...])."""
        B = x.shape[0]
        x = self.patch_embed(x)                                  # (B, 64, 384)
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, 384)
        x = torch.cat((cls, x), dim=1)                           # (B, 65, 384)
        x = self.pos_drop(x + self.pos_embed)                    # (B, 65, 384)

        attns: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn_w = block(x, return_attn=return_attn)
            if return_attn and attn_w is not None:
                attns.append(attn_w)

        x = self.norm(x)                                         # (B, 65, 384)
        return x[:, 0], attns                                    # cls token (B, 384)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        feat, attns = self.forward_features(x, return_attn=return_attn)
        logits = self.head(feat)                                 # (B, 10)
        return (logits, attns) if return_attn else logits

    @torch.no_grad()
    def attention_rollout(self, x: torch.Tensor, add_residual: bool = True) -> torch.Tensor:
        """Attention rollout (Abnar & Zuidema 2020) -> (B, 64) per-patch map.

        Raw last-layer attention is a poor explanation on its own, because each
        block also passes information straight through its residual connection.
        Rollout accounts for that: add 0.5*I to each head-averaged attention
        matrix (the residual path), renormalise, and multiply the layers
        together to get end-to-end token-to-token influence. Row 0 is then how
        much each patch contributed to the cls token.
        """
        self.eval()
        _, attns = self.forward_features(x, return_attn=True)
        if not attns:
            raise RuntimeError("no attention captured")

        B, _, N, _ = attns[0].shape
        result = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1).clone()
        for attn in attns:
            a = attn.mean(dim=1)                                 # average heads -> (B, 65, 65)
            if add_residual:
                a = a + torch.eye(N, device=a.device).unsqueeze(0)
                a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result
        cls_to_patches = result[:, 0, 1:]                        # (B, 64)
        return cls_to_patches

    @torch.no_grad()
    def pos_embed_similarity(self) -> torch.Tensor:
        """Cosine similarity of every patch position's embedding with every
        other -> (64, 64). E7 reshapes this to an 8x8 grid of 8x8 maps; if the
        model learned 2-D structure, each map glows around its own position."""
        pe = self.pos_embed[0, 1:]                               # (64, 384), drop cls
        pe = F.normalize(pe, dim=-1)
        return pe @ pe.t()                                       # (64, 64)

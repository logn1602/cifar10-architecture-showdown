"""Model registry: ``build_model(name, cfg)`` for all six contestants.

The registry is the single place that maps a config string to an architecture,
so every script (training, sweeps, evaluation, interpretability) constructs
models identically. It validates constructor keyword arguments strictly --
an unknown key in a YAML file raises immediately rather than being silently
ignored, which during an unattended 8-hour queue is the difference between a
typo you notice in ten seconds and six runs of a model you did not configure.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch.nn as nn

from .convnext import META as _CONVNEXT_META, ConvNeXtMini
from .mlp import META as _MLP_META, MLP
from .mlp_mixer import META as _MIXER_META, MLPMixer
from .resnet import META as _RESNET_META, ResNet18
from .simple_cnn import META as _CNN_META, SimpleCNN
from .vit import META as _VIT_META, ViT

__all__ = [
    "MODEL_REGISTRY", "MODEL_META", "MODEL_ORDER",
    "build_model", "list_models", "get_meta", "count_parameters",
    "MLP", "SimpleCNN", "ResNet18", "ViT", "MLPMixer", "ConvNeXtMini",
]

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": MLP,
    "simple_cnn": SimpleCNN,
    "resnet18": ResNet18,
    "vit": ViT,
    "mlp_mixer": MLPMixer,
    "convnext_mini": ConvNeXtMini,
}

MODEL_META: dict[str, dict[str, Any]] = {
    "mlp": _MLP_META,
    "simple_cnn": _CNN_META,
    "resnet18": _RESNET_META,
    "vit": _VIT_META,
    "mlp_mixer": _MIXER_META,
    "convnext_mini": _CONVNEXT_META,
}

# Chronological order -- the story the report tells, oldest first.
MODEL_ORDER: tuple[str, ...] = (
    "mlp", "simple_cnn", "resnet18", "vit", "mlp_mixer", "convnext_mini",
)


def list_models() -> list[str]:
    """Registered model names, in chronological order."""
    return list(MODEL_ORDER)


def get_meta(name: str) -> dict[str, Any]:
    """Year / key idea / inductive bias for ``name`` (feeds the README)."""
    if name not in MODEL_META:
        raise KeyError(f"Unknown model {name!r}. Known: {', '.join(MODEL_ORDER)}")
    return dict(MODEL_META[name])


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in params)


def build_model(name: str, cfg: dict | None = None, num_classes: int = 10) -> nn.Module:
    """Instantiate a registered model.

    Args:
        name: Registry key, e.g. ``"resnet18"``.
        cfg: Constructor keyword arguments. A ``"name"`` key is ignored, so a
            whole ``model:`` block from YAML can be passed through unchanged.
        num_classes: Used when ``cfg`` does not set it.

    Raises:
        KeyError: Unknown model name.
        TypeError: An argument the constructor does not accept -- listing both
            the offending key and the valid ones.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known: {', '.join(MODEL_ORDER)}")

    cls = MODEL_REGISTRY[name]
    kwargs = {k: v for k, v in (cfg or {}).items() if k != "name"}
    kwargs.setdefault("num_classes", num_classes)

    valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
    unknown = set(kwargs) - valid
    if unknown:
        raise TypeError(
            f"{name}: unknown config key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return cls(**kwargs)

"""Grad-CAM (Selvaraju et al., 2017), written out by hand.

For a target class c and a conv layer with activation maps A_k:

    alpha_k = mean_ij  d(y_c) / d(A_k[i, j])     how much channel k matters for c
    CAM     = ReLU( sum_k alpha_k * A_k )        where those channels fire

The ReLU keeps only evidence *for* the class. The map comes out at the layer's
own resolution -- 8x8 for Simple CNN's last conv, 4x4 for ResNet-18's and
ConvNeXt-mini's last stage on a 32x32 input -- and is upsampled to 32x32 for
display. That coarseness is a property of the architectures at this image
size, not of the method.

The target class is the model's own prediction: the question is "what did this
model look at to decide", including when it decided wrong.

:func:`crosscheck` compares this implementation with the ``pytorch-grad-cam``
library on the same inputs; E7 records the correlation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GradCAM", "crosscheck"]


class GradCAM:
    """``GradCAM(model, layer)(x)`` -> per-image maps in [0, 1] at input resolution."""

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output: torch.Tensor) -> None:
        self.acts = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.grads = grad

    def __call__(self, x: torch.Tensor, target: torch.Tensor | None = None) -> dict:
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(x)                                    # (B, 10)
            cls = logits.argmax(1) if target is None else target.to(x.device)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, cls[:, None]).sum().backward()

        a = self.acts.detach().float()                                # (B, K, h, w)
        g = self.grads.detach().float()                               # (B, K, h, w)
        weights = g.mean(dim=(2, 3), keepdim=True)                    # alpha_k: (B, K, 1, 1)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))          # (B, 1, h, w)
        native = tuple(cam.shape[-2:])
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo = flat.min(1).values[:, None, None]
        hi = flat.max(1).values[:, None, None]
        cam = (cam - lo) / (hi - lo + 1e-7)                           # (B, 32, 32) in [0, 1]

        probs = logits.detach().float().softmax(1)
        return {"cam": cam.cpu().numpy(), "cls": cls.cpu().numpy(),
                "prob": probs.gather(1, cls[:, None])[:, 0].cpu().numpy(),
                "native_resolution": native}

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def crosscheck(model: nn.Module, layer: nn.Module, x: torch.Tensor) -> dict:
    """Mean per-image Pearson correlation between our maps and pytorch-grad-cam's.

    Both target the model's predicted class. The library resizes with OpenCV
    rather than ``F.interpolate``, so a correlation just under 1 is expected;
    anything well below would mean one of the two is computing something else.
    """
    try:
        from pytorch_grad_cam import GradCAM as LibGradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception as exc:  # optional dependency
        return {"status": "skipped", "reason": f"pytorch-grad-cam unavailable: {exc}"}

    with GradCAM(model, layer) as ours:
        res = ours(x)
    with LibGradCAM(model=model, target_layers=[layer]) as lib:
        theirs = lib(input_tensor=x, targets=[ClassifierOutputTarget(int(c)) for c in res["cls"]])
    corrs = []
    for a, b in zip(res["cam"], np.asarray(theirs)):
        if a.std() > 0 and b.std() > 0:
            corrs.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
    return {"status": "ok", "mean_corr": float(np.mean(corrs)) if corrs else None,
            "min_corr": float(np.min(corrs)) if corrs else None, "n_images": len(corrs)}

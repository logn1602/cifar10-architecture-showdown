"""Cost profiling: parameters, FLOPs, inference throughput, peak memory.

Question 1 of the showdown is "which architecture wins, *and at what cost*" --
these are the cost columns. Training throughput and sec/epoch are measured by
the training loop itself (from real epochs, not a micro-benchmark); this module
covers the numbers that need a dedicated measurement.

Note on the module name: ``src/profile.py`` shadows the stdlib ``profile``
module *only* if ``src/`` itself is put on ``sys.path``. Every entry point in
this repo imports it as ``src.profile``, so that never happens -- but do not run
files from inside ``src/`` with ``python src/x.py`` and then use cProfile.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn

__all__ = ["count_params", "count_flops", "measure_inference", "reset_peak_memory",
           "peak_memory_mb"]


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def count_flops(model: nn.Module, device: torch.device, image_size: int = 32) -> int | None:
    """Forward FLOPs for a single image, via ``FlopCounterMode``.

    Counts matmul/conv FLOPs only -- elementwise work (GELU, LayerNorm, softmax)
    is not counted, which is the convention every published FLOP number for
    these architectures follows. Returns None rather than raising: a missing
    FLOP count must never take down a finished training run.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    try:
        counter = FlopCounterMode(display=False)
    except TypeError:  # older/newer signature
        counter = FlopCounterMode()
    try:
        with counter:
            model(x)
        return int(counter.get_total_flops())
    except Exception:
        return None
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def measure_inference(
    model: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    batch_size: int = 256,
    image_size: int = 32,
    warmup_iters: int = 5,
    iters: int = 20,
    channels_last: bool = False,
    input_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    """Pure-model inference throughput in images/sec (no data loading).

    Synthetic input on the device, so this is the speed of the *architecture*,
    not of the input pipeline -- the right number for comparing models.
    ``input_fn`` (E4's upsampling) is included in the timed region because it
    is part of what inference costs for those models.
    """
    was_training = model.training
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    def step() -> None:
        inp = input_fn(x) if input_fn is not None else x
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                model(inp)
        else:
            model(inp)

    for _ in range(warmup_iters):
        step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(device)
    elapsed = time.perf_counter() - t0

    if was_training:
        model.train()
    return batch_size * iters / max(elapsed, 1e-9)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    """Peak allocated GPU memory since the last reset, in MB. 0 on CPU.

    This is memory *allocated by tensors*, not reserved by the caching
    allocator -- the number that reflects the model, not allocator slack.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

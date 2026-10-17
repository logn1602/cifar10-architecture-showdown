"""Phase 0: environment and hardware report.

Prints a human-readable report, and saves:
  * results/tables/environment.md   -- the same thing as markdown
  * results/environment.json        -- machine-readable, read by later phases
                                       (AMP dtype, device choice, worker count)

The part that matters on this project is the **GPU verdict**: it decides
whether training happens locally or on Kaggle. The verdict is derived from the
detected VRAM and torch build, never hard-coded, so running this same script on
Kaggle gives the right (different) answer.
"""

from __future__ import annotations

import argparse
import importlib
import platform
from importlib import metadata
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.utils import PROJECT_ROOT, ensure_dir, save_json  # noqa: E402

# Packages whose versions we pin in the report. Each import is guarded so one
# missing package can never abort the whole report.
TRACKED_PACKAGES = [
    ("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
    ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("seaborn", "seaborn"),
    ("scikit-learn", "sklearn"), ("timm", "timm"), ("pyyaml", "yaml"),
    ("tqdm", "tqdm"), ("tensorboard", "tensorboard"),
    ("grad-cam", "pytorch_grad_cam"), ("umap-learn", "umap"),
]

# VRAM thresholds for the verdict, in MB. ResNet-18 at batch 128 with AMP needs
# roughly 2.5-3 GB for activations alone on 32x32 input; ConvNeXt and ViT want
# more again, and E4 fine-tunes at 224px where memory use climbs sharply.
VRAM_VIABLE_MB = 6 * 1024
VRAM_MARGINAL_MB = 4 * 1024

RULE = "=" * 78
THIN = "-" * 78


def _version(dist_name: str, module_name: str) -> str:
    """Version of an installed package.

    Prefers the distribution metadata over ``module.__version__``: several
    packages here (pytorch_grad_cam, tensorboard) expose no ``__version__``
    attribute at all.
    """
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # ImportError, or a broken install
        return f"NOT INSTALLED ({type(exc).__name__})"
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "unknown"))


def _nvidia_smi() -> str | None:
    """Raw nvidia-smi output, or None when the driver/tool is absent."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout.strip() or None


def _psutil():
    try:
        return importlib.import_module("psutil")
    except Exception:
        return None


def collect() -> dict:
    """Gather every fact we report. Pure data -- no printing."""
    info: dict = {}

    info["python"] = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
    }
    # platform.release() reports "10" on Windows 11 (the kernel version never
    # changed), so the build number is carried alongside it -- build >= 22000
    # means Windows 11. Without it the report is actively misleading.
    info["platform"] = {
        "system": platform.system(), "release": platform.release(),
        "version": platform.version(), "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if platform.system() == "Windows":
        try:
            build = int(platform.version().split(".")[-1])
            info["platform"]["marketing_name"] = f"Windows {11 if build >= 22000 else 10}"
            info["platform"]["build"] = build
        except (ValueError, IndexError):
            pass

    # --- torch build -------------------------------------------------------
    info["torch"] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,       # None for a CPU-only wheel
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    cuda_available = torch.cuda.is_available()
    info["cuda_available"] = cuda_available

    gpus: list[dict] = []
    bf16 = False
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i, "name": props.name,
                "vram_mb": round(props.total_memory / 1024**2),
                "capability": f"{props.major}.{props.minor}",
                "multiprocessors": props.multi_processor_count,
            })
        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16 = False
    info["gpus"] = gpus
    info["bf16_supported"] = bf16
    info["nvidia_smi"] = _nvidia_smi()

    # --- CPU / RAM / disk --------------------------------------------------
    ps = _psutil()
    info["cpu_count"] = __import__("os").cpu_count()
    info["cpu_physical"] = ps.cpu_count(logical=False) if ps else None
    if ps:
        vm = ps.virtual_memory()
        info["ram_gb"] = round(vm.total / 1024**3, 2)
        info["ram_available_gb"] = round(vm.available / 1024**3, 2)
    else:
        info["ram_gb"] = None
        info["ram_available_gb"] = None
    usage = shutil.disk_usage(PROJECT_ROOT)
    info["disk_free_gb"] = round(usage.free / 1024**3, 2)
    info["disk_total_gb"] = round(usage.total / 1024**3, 2)

    info["packages"] = {name: _version(name, mod) for name, mod in TRACKED_PACKAGES}
    info["threads"] = {"torch_num_threads": torch.get_num_threads()}

    # --- verdict -----------------------------------------------------------
    best_vram = max((g["vram_mb"] for g in gpus), default=0)
    if not cuda_available:
        verdict, reason = "unusable", (
            "torch reports no CUDA device. "
            + ("This is a CPU-only PyTorch build (torch.version.cuda is None), which is the "
               "expected state for the local CPU workflow."
               if torch.version.cuda is None
               else "torch has a CUDA build but found no usable device -- check the driver.")
        )
    elif best_vram >= VRAM_VIABLE_MB:
        verdict, reason = "viable", f"{best_vram} MB VRAM is enough for the full protocol."
    elif best_vram >= VRAM_MARGINAL_MB:
        verdict, reason = "marginal", (
            f"{best_vram} MB VRAM will fit the small models at batch 128 but is tight for "
            "ConvNeXt/ViT and will not fit the 224px E4 fine-tuning at batch 64."
        )
    else:
        verdict, reason = "unusable", (
            f"{best_vram} MB VRAM is below the ~{VRAM_MARGINAL_MB} MB floor. ResNet-18 at "
            "batch 128 with AMP needs roughly 2.5-3 GB of activations alone on 32x32 input, "
            "and ConvNeXt/ViT need more."
        )
    info["gpu_verdict"] = verdict
    info["gpu_verdict_reason"] = reason
    info["recommended_device"] = "cuda" if verdict in {"viable", "marginal"} else "cpu"
    info["recommended_amp_dtype"] = "bf16" if (cuda_available and bf16) else (
        "fp16" if cuda_available else "none")

    return info


def benchmark_cpu_matmul(n: int = 2048, runs: int = 5, warmup: int = 2) -> float:
    """Rough CPU GFLOP/s from a dense matmul. Sized for a 8 GB box."""
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    for _ in range(warmup):
        a @ b
    t0 = time.perf_counter()
    for _ in range(runs):
        a @ b
    elapsed = (time.perf_counter() - t0) / runs
    return (2.0 * n**3) / elapsed / 1e9


def _os_name(plat: dict) -> str:
    """Human-readable OS string, including the Windows 11 build correction."""
    base = plat.get("marketing_name") or f"{plat['system']} {plat['release']}"
    build = plat.get("build")
    return f"{base} (build {build})" if build else base


def render_text(info: dict) -> str:
    """The stdout report. ASCII only -- this is a Windows console."""
    L: list[str] = []
    add = L.append

    add(RULE)
    add("  PHASE 0  |  ENVIRONMENT & HARDWARE CHECK")
    add(RULE)

    py = info["python"]
    plat = info["platform"]
    add("\n[ Python ]")
    add(f"  version      : {py['version']}")
    add(f"  executable   : {py['executable']}")
    add(f"  virtualenv   : {'yes' if py['in_venv'] else 'NO (you are using the system Python)'}")
    add(f"  platform     : {_os_name(plat)} ({plat['machine']})")

    t = info["torch"]
    add("\n[ PyTorch ]")
    add(f"  torch        : {t['version']}")
    add(f"  CUDA build   : {t['cuda_build'] or 'None (CPU-only wheel)'}")
    add(f"  cuDNN        : {t['cudnn'] or 'n/a'}")
    add(f"  threads      : {info['threads']['torch_num_threads']}")

    add("\n[ GPU ]")
    add(f"  torch.cuda.is_available() : {info['cuda_available']}")
    if info["gpus"]:
        for g in info["gpus"]:
            add(f"  [{g['index']}] {g['name']}  |  {g['vram_mb']} MB  |  "
                f"sm_{g['capability'].replace('.', '')}  |  {g['multiprocessors']} SMs")
        add(f"  bf16 supported            : {info['bf16_supported']}")
    else:
        add("  no CUDA devices visible to torch")
    if info["nvidia_smi"]:
        add("  nvidia-smi reports:")
        for line in info["nvidia_smi"].splitlines():
            add(f"    {line.strip()}")
    else:
        add("  nvidia-smi: not found (no NVIDIA driver on PATH)")

    add("\n[ CPU / memory / disk ]")
    add(f"  logical cores  : {info['cpu_count']}")
    add(f"  physical cores : {info['cpu_physical'] if info['cpu_physical'] else 'unknown (install psutil)'}")
    ram = f"{info['ram_gb']} GB total, {info['ram_available_gb']} GB available" if info["ram_gb"] else "unknown (install psutil)"
    add(f"  RAM            : {ram}")
    add(f"  disk free      : {info['disk_free_gb']} GB of {info['disk_total_gb']} GB")
    if info.get("cpu_gflops"):
        add(f"  CPU matmul     : {info['cpu_gflops']:.1f} GFLOP/s (fp32, 2048x2048)")

    add("\n[ Packages ]")
    for name, ver in info["packages"].items():
        flag = "  " if "NOT INSTALLED" not in ver else "!!"
        add(f"  {flag} {name:<16} {ver}")

    add("\n" + THIN)
    add(f"  GPU VERDICT: {info['gpu_verdict'].upper()}")
    add(THIN)
    add(f"  {info['gpu_verdict_reason']}")
    add(f"  recommended device    : {info['recommended_device']}")
    add(f"  recommended AMP dtype : {info['recommended_amp_dtype']}")

    if info["gpu_verdict"] == "unusable":
        add("")
        add("  >> No usable training GPU on this machine.")
        add("  >> Per spec 1.4 and 12.6, the plan is:")
        add("       1. build and smoke-test everything locally on CPU  (Phases 0-3)")
        add("       2. run the real queue on Kaggle with 2x T4 GPUs    (kaggle_runner.ipynb)")
        add("       3. download results.zip and run Phase 9 locally")
        add("     See docs/KAGGLE.md for the step-by-step.")
    add("")
    add(RULE)
    return "\n".join(L)


def render_markdown(info: dict) -> str:
    L: list[str] = []
    add = L.append
    add("# Phase 0 -- Environment & hardware\n")
    add("_Generated by `scripts/env_check.py`. Do not edit by hand._\n")

    add("## Verdict\n")
    add(f"| | |\n|---|---|\n"
        f"| **GPU verdict** | `{info['gpu_verdict']}` |\n"
        f"| Reason | {info['gpu_verdict_reason']} |\n"
        f"| Recommended device | `{info['recommended_device']}` |\n"
        f"| Recommended AMP dtype | `{info['recommended_amp_dtype']}` |\n")

    py, plat, t = info["python"], info["platform"], info["torch"]
    add("## System\n")
    add("| Item | Value |\n|---|---|")
    add(f"| Python | {py['version']} |")
    add(f"| Virtualenv | {'yes' if py['in_venv'] else 'NO'} |")
    add(f"| OS | {_os_name(plat)} ({plat['machine']}) |")
    add(f"| torch | {t['version']} |")
    add(f"| torch CUDA build | {t['cuda_build'] or 'None (CPU-only wheel)'} |")
    add(f"| cuDNN | {t['cudnn'] or 'n/a'} |")
    add(f"| Logical cores | {info['cpu_count']} |")
    add(f"| Physical cores | {info['cpu_physical'] or 'unknown'} |")
    add(f"| RAM | {info['ram_gb'] or 'unknown'} GB |")
    add(f"| Disk free | {info['disk_free_gb']} GB |")
    if info.get("cpu_gflops"):
        add(f"| CPU matmul | {info['cpu_gflops']:.1f} GFLOP/s |")
    add("")

    add("## GPUs\n")
    if info["gpus"]:
        add("| # | Name | VRAM (MB) | Capability | SMs |\n|---|---|---|---|---|")
        for g in info["gpus"]:
            add(f"| {g['index']} | {g['name']} | {g['vram_mb']} | {g['capability']} | {g['multiprocessors']} |")
        add(f"\nbf16 supported: `{info['bf16_supported']}`\n")
    else:
        add("torch sees no CUDA devices.\n")
        if info["nvidia_smi"]:
            add("`nvidia-smi` does report hardware:\n")
            add("```\n" + info["nvidia_smi"] + "\n```\n")

    add("## Packages\n")
    add("| Package | Version |\n|---|---|")
    for name, ver in info["packages"].items():
        add(f"| {name} | {ver} |")
    add("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 environment & hardware check")
    ap.add_argument("--json-only", action="store_true", help="suppress the stdout report")
    ap.add_argument("--out-dir", default="results", help="where to write the outputs")
    ap.add_argument("--no-benchmark", action="store_true", help="skip the CPU matmul benchmark")
    args = ap.parse_args()

    info = collect()
    if not args.no_benchmark:
        info["cpu_gflops"] = benchmark_cpu_matmul()

    out_dir = ensure_dir(args.out_dir)
    tables_dir = ensure_dir(out_dir / "tables")

    save_json(info, out_dir / "environment.json")
    (tables_dir / "environment.md").write_text(render_markdown(info), encoding="utf-8")

    if not args.json_only:
        print(render_text(info))
        print(f"\nwrote {out_dir / 'environment.json'}")
        print(f"wrote {tables_dir / 'environment.md'}")


if __name__ == "__main__":
    main()

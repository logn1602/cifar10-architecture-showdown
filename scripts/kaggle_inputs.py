"""Use what is attached to the Kaggle notebook (Add Input) before fetching anything.

    python scripts/kaggle_inputs.py cifar      # CIFAR-10: attached copy if any, else download

``cifar``
    CIFAR-10 is 170 MB from a university server, the run's only network
    dependency besides pip. If any attached input holds the official python
    version -- a ``cifar-10-batches-py/`` folder or ``cifar-10-python.tar.gz``
    -- it is copied into ``data/`` and every file is md5-checked against
    torchvision's own checksums, so the bytes (and with them the split
    fingerprint) are exactly what a download would give. An attached copy
    that fails the check is skipped with the reason printed. With no usable
    copy, CIFAR-10 is downloaded as before. Either way the source is printed.

``--input-root`` defaults to ``/kaggle/input``; the local tests point it at a
scratch directory instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, resolve_path  # noqa: E402

KAGGLE_INPUT = "/kaggle/input"


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------


def cifar_checksums() -> dict[str, str]:
    """Every file of the python version, with torchvision's md5 for it."""
    from torchvision.datasets import CIFAR10
    files = CIFAR10.train_list + CIFAR10.test_list
    files = files + [[CIFAR10.meta["filename"], CIFAR10.meta["md5"]]]
    return {name: md5 for name, md5 in files}


def cifar_problem(folder: Path) -> str | None:
    """None when ``folder`` holds every CIFAR-10 file with the right md5, else why not."""
    from torchvision.datasets.utils import check_integrity
    for name, md5 in cifar_checksums().items():
        if not (folder / name).is_file():
            return f"{name} missing"
        if not check_integrity(str(folder / name), md5):
            return f"{name} fails its md5 check"
    return None


def find_attached_cifar(input_root: Path) -> list[Path]:
    """Attached copies: extracted folders first (no unpacking needed), then archives."""
    if not input_root.is_dir():
        return []
    folders = sorted(p for p in input_root.rglob("cifar-10-batches-py") if p.is_dir())
    archives = sorted(p for p in input_root.rglob("cifar-10-python.tar.gz") if p.is_file())
    return folders + archives


def install_cifar(src: Path, data_root: Path) -> str | None:
    """Copy (or unpack) ``src`` into ``data_root``; None on success, else the problem.

    Everything lands in a staging folder first and is md5-checked there, so a
    bad copy never replaces a good one.
    """
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity, extract_archive

    stage = data_root / ".cifar_incoming"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        if src.is_dir():
            shutil.copytree(src, stage / CIFAR10.base_folder)
        else:
            if not check_integrity(str(src), CIFAR10.tgz_md5):
                return "the archive fails its md5 check"
            extract_archive(str(src), str(stage))
        problem = cifar_problem(stage / CIFAR10.base_folder)
        if problem:
            return problem
        target = data_root / CIFAR10.base_folder
        shutil.rmtree(target, ignore_errors=True)
        (stage / CIFAR10.base_folder).rename(target)
        return None
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def cmd_cifar(args: argparse.Namespace) -> int:
    from torchvision.datasets import CIFAR10

    from src.data import download_cifar10

    data_root = resolve_path(args.data_root or load_config("configs/base.yaml")["data"]["root"])
    data_root.mkdir(parents=True, exist_ok=True)
    target = data_root / CIFAR10.base_folder
    if target.is_dir() and cifar_problem(target) is None:
        print(f"CIFAR-10: already present at {target} (md5 verified); nothing to fetch.")
        return 0

    input_root = Path(args.input_root)
    candidates = find_attached_cifar(input_root)
    for src in candidates:
        problem = install_cifar(src, data_root)
        if problem is None:
            print(f"CIFAR-10: using the attached copy at {src}")
            print(f"          -> {target} (every file md5-verified against torchvision)")
            return 0
        print(f"CIFAR-10: skipping the attached copy at {src}: {problem}")

    why = "no attached copy passed the md5 check" if candidates else \
        f"no attached copy under {input_root}"
    print(f"CIFAR-10: {why}; downloading {CIFAR10.url}", flush=True)
    download_cifar10(data_root)
    problem = cifar_problem(target)
    if problem:
        print(f"CIFAR-10: the download is unusable: {problem}")
        return 1
    print(f"CIFAR-10: downloaded from {CIFAR10.url}")
    print(f"          -> {target} (every file md5-verified against torchvision)")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cifar", help="CIFAR-10 from an attached copy, else download")
    c.add_argument("--input-root", default=KAGGLE_INPUT)
    c.add_argument("--data-root", help="default: data.root from configs/base.yaml")
    c.set_defaults(func=cmd_cifar)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

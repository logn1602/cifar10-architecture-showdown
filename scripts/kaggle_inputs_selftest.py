"""CPU test of ``scripts/kaggle_inputs.py`` against fake ``/kaggle/input`` trees.

Builds each attached-input layout the notebook can meet under
``results/_inputs_test`` (gitignored), from the local CIFAR-10 files, and runs
the real script on it:

* cifar: an attached extracted folder; an attached official archive; a
  corrupted folder next to a good archive (the folder must be skipped with its
  reason, the archive used); data already in place (nothing fetched). With
  ``--download``, also an empty input root (falls back to the real download).

Run: ``python scripts/kaggle_inputs_selftest.py [--download]``
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "results" / "_inputs_test"
DATA = ROOT / "data"
BATCHES = DATA / "cifar-10-batches-py"
ARCHIVE = DATA / "cifar-10-python.tar.gz"


def run(*args: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_inputs.py"), *args],
                          cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def fresh(name: str) -> tuple[Path, Path]:
    """An empty fake input root and an empty data root for one case."""
    base = TEST_ROOT / name
    shutil.rmtree(base, ignore_errors=True)
    (base / "input").mkdir(parents=True)
    (base / "data").mkdir()
    return base / "input", base / "data"


def cifar_cases(download: bool) -> list[tuple[str, bool, str]]:
    checks = []

    def case(name: str, inp: Path, data: Path, expect_rc: int, expect: list[str],
             forbid: list[str] = ()) -> None:
        rc, out = run("cifar", "--input-root", str(inp), "--data-root", str(data))
        print(f"--- {name} ---\n{out.strip()}\n")
        ok = rc == expect_rc and all(s in out for s in expect) and not any(s in out for s in forbid)
        ok = ok and (data / "cifar-10-batches-py" / "test_batch").is_file()
        checks.append((f"cifar: {name}", ok, f"exit {rc}"))

    inp, data = fresh("folder")
    shutil.copytree(BATCHES, inp / "cifar10-python" / "cifar-10-batches-py")
    case("attached extracted folder is used", inp, data, 0,
         ["using the attached copy", "cifar10-python"], ["downloading"])

    inp, data = fresh("archive")
    (inp / "cifar10-tgz").mkdir()
    shutil.copy(ARCHIVE, inp / "cifar10-tgz" / ARCHIVE.name)
    case("attached official archive is unpacked and used", inp, data, 0,
         ["using the attached copy", ARCHIVE.name], ["downloading"])

    inp, data = fresh("corrupt")
    bad = inp / "a-broken-copy" / "cifar-10-batches-py"
    shutil.copytree(BATCHES, bad)
    with (bad / "data_batch_3").open("r+b") as fh:
        fh.seek(1000)
        fh.write(b"\x00" * 64)
    (inp / "b-good-archive").mkdir()
    shutil.copy(ARCHIVE, inp / "b-good-archive" / ARCHIVE.name)
    case("corrupted folder skipped with its reason, good archive used", inp, data, 0,
         ["skipping the attached copy", "data_batch_3 fails its md5 check",
          "using the attached copy", "b-good-archive"], ["downloading"])

    # Data already in place from the case above: nothing is fetched.
    empty = TEST_ROOT / "corrupt" / "nothing-attached"
    empty.mkdir()
    case("data already present: nothing fetched", empty, data, 0, ["already present"],
         ["downloading", "using the attached"])

    if download:
        inp, data = fresh("download")
        case("nothing attached: falls back to the download", inp, data, 0,
             ["no attached copy under", "downloaded from"])
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--download", action="store_true",
                    help="also test the fallback download (fetches 170 MB)")
    args = ap.parse_args()
    if not (BATCHES.is_dir() and ARCHIVE.is_file()):
        print(f"needs the local CIFAR-10 files in {DATA} (python -m src.data downloads them)")
        return 1
    checks = cifar_cases(args.download)
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({detail})")
    ok = all(c[1] for c in checks)
    print(f"\n{sum(c[1] for c in checks)}/{len(checks)} checks passed -- "
          f"{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""CPU test of ``scripts/kaggle_outputs.py`` on fake /kaggle/working trees.

Builds each case under ``results/_outputs_test`` (gitignored) and runs the real
script on it:

* census: an output holding the repo clone (as the notebook used to leave it)
  plus the two zips, 441 files in all -- the exact total and every top-level
  count must be printed, with the near-the-cap warning, dotfiles and nested
  folders included; and a small output with no warning.

Run: ``python scripts/kaggle_outputs_selftest.py``
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "results" / "_outputs_test"


def run(*args: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_outputs.py"), *args],
                          cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    return proc.returncode, proc.stdout + proc.stderr


def fill(folder: Path, n: int, stem: str = "f", suffix: str = ".json") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{stem}{i:04d}{suffix}").write_text("x", encoding="utf-8")


def parse_census(out: str) -> dict[str, dict]:
    """A census print as {root: {"total": n, "rows": {name: n}, "sub": {name: n}}}."""
    sections: dict[str, dict] = {}
    current = None
    for line in out.splitlines():
        head = re.match(r"^(\S.*): (\d+) files?, ", line)
        row = re.match(r"^( {2}| {4})(\S+)\s+(\d+) files?\b", line)
        if head:
            current = sections.setdefault(head.group(1), {"total": int(head.group(2)),
                                                          "rows": {}, "sub": {}})
        elif row and current is not None:
            current["rows" if row.group(1) == "  " else "sub"][row.group(2)] = int(row.group(3))
    return sections


def census_cases() -> list[tuple[str, bool, str]]:
    checks = []

    def case(name: str, output: Path, work: Path | None, expect_out: dict,
             expect_work: dict | None, expect: list[str], forbid: list[str] = ()) -> None:
        args = ["census", "--output", str(output)] + (["--work-dir", str(work)] if work else [])
        rc, out = run(*args)
        print(f"--- census: {name} ---\n{out.strip()}\n")
        problems = [f"exit {rc}"] if rc else []
        got = parse_census(out)
        for root, want in ((output, expect_out), (work, expect_work)):
            if want is None:
                continue
            have = got.get(str(root))
            if have is None:
                problems.append(f"no section for {root}")
                continue
            for key in ("total", "rows", "sub"):
                if key in want and have[key] != want[key]:
                    problems.append(f"{root.name} {key}: {have[key]}, expected {want[key]}")
        problems += [f"missing output {s!r}" for s in expect if s not in out]
        problems += [f"unexpected output {s!r}" for s in forbid if s in out]
        checks.append((f"census: {name}", not problems, "; ".join(problems) or f"exit {rc}"))

    # The layout the notebook used to leave: the clone inside /kaggle/working.
    base = TEST_ROOT / "census_old_layout"
    shutil.rmtree(base, ignore_errors=True)
    out = base / "working"
    clone = out / "cifar10-architecture-showdown"
    fill(clone / ".git" / "objects" / "pack", 3, "pack", ".idx")
    fill(clone / ".git" / "hooks", 13, "hook", ".sample")
    fill(clone / "src" / "models" / "__pycache__", 8, "m", ".pyc")
    fill(clone / "data" / "cifar-10-batches-py", 8, "data_batch_", "")
    fill(clone / "results" / "runs" / "E1" / "mlp" / "0", 6)
    fill(clone / "results" / "runs" / "lr_sweep" / "mlp" / "lr_0.001", 300)
    fill(clone / "results" / "tb", 100, "events", "")
    (clone / "README.md").write_text("x", encoding="utf-8")
    fill(out, 2, "results", ".zip")
    n_clone = 3 + 13 + 8 + 8 + 6 + 300 + 100 + 1
    case(f"the clone inside the output: {n_clone + 2} files, warned", out, clone,
         {"total": n_clone + 2, "rows": {"cifar10-architecture-showdown/": n_clone,
                                         "results0000.zip": 1, "results0001.zip": 1}},
         {"total": n_clone, "rows": {".git/": 16, "data/": 8, "README.md": 1, "results/": 406,
                                     "src/": 8},
          "sub": {"runs/": 306, "tb/": 100}},
         [f"WARNING: {n_clone + 2} files", "inside the output, counted above"])

    # A small output: exact count, no warning; a work dir outside it is "not saved".
    base = TEST_ROOT / "census_small"
    shutil.rmtree(base, ignore_errors=True)
    out, work = base / "working", base / "tmp" / "cifar10-architecture-showdown"
    fill(out, 2, "results", ".zip")
    fill(work / "results" / "runs", 5)
    case("two files in the output: no warning", out, work,
         {"total": 2, "rows": {"results0000.zip": 1, "results0001.zip": 1}},
         {"total": 5, "rows": {"results/": 5}, "sub": {"runs/": 5}},
         ["(not saved)"], forbid=["WARNING"])
    return checks


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    checks = census_cases()
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({detail})")
    ok = all(c[1] for c in checks)
    print(f"\n{sum(c[1] for c in checks)}/{len(checks)} checks passed -- "
          f"{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

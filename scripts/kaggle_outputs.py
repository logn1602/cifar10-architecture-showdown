"""What the Kaggle notebook leaves in /kaggle/working, the version's saved output.

    python scripts/kaggle_outputs.py census --output /kaggle/working --work-dir <repo clone>

``census``
    Kaggle saves /kaggle/working as the version's output, and may cap it at
    about 500 files (community reports, not documented). This prints the exact
    file count there, and per top-level entry, so every run's log shows how
    close it came. ``--work-dir`` adds the same breakdown for the repo clone
    (and ``results/`` inside it, one level deeper): what the output would hold
    if the clone lived inside /kaggle/working. It only reads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Kaggle reportedly caps a version's output at 500 files; warn well before.
OUTPUT_FILE_WARN = 400


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def count(path: Path) -> tuple[int, int]:
    """(files, bytes) under ``path``; a file or symlink counts as one. Never follows links."""
    if not path.is_dir() or path.is_symlink():
        return 1, path.lstat().st_size
    n = size = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            n += 1
            try:
                size += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return n, size


def report(root: Path, label: str, expand: tuple[str, ...] = ()) -> int:
    """Print ``root``'s file count and its top-level entries; returns the count."""
    if not root.is_dir():
        print(f"{root}: not found ({label})")
        return 0
    entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
    rows = [(e, *count(e)) for e in entries]
    total = sum(n for _, n, _ in rows)
    print(f"{root}: {total} files, {human(sum(s for _, _, s in rows))} ({label})")
    for entry, n, size in rows:
        is_dir = entry.is_dir() and not entry.is_symlink()
        print(f"  {entry.name + ('/' if is_dir else ''):<36} {n:>6} {'file ' if n == 1 else 'files'}"
              f"  {human(size):>9}")
        if is_dir and entry.name in expand:
            for sub in sorted(entry.iterdir(), key=lambda p: p.name.lower()):
                sn, ss = count(sub)
                sub_is_dir = sub.is_dir() and not sub.is_symlink()
                print(f"    {sub.name + ('/' if sub_is_dir else ''):<34} {sn:>6} "
                      f"{'file ' if sn == 1 else 'files'}  {human(ss):>9}")
    return total


def cmd_census(args: argparse.Namespace) -> int:
    output = Path(args.output)
    n = report(output, "saved as this version's output")
    if n > OUTPUT_FILE_WARN:
        print(f"WARNING: {n} files in {output}. Kaggle may cap a version's output at about "
              "500 files; past that, the output can fail to save.")
    if args.work_dir:
        work = Path(args.work_dir)
        inside = output.resolve() in work.resolve().parents
        report(work, "inside the output, counted above" if inside else "not saved",
               expand=("results",))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census", help="exact file counts in the output (and the work dir)")
    c.add_argument("--output", default="/kaggle/working")
    c.add_argument("--work-dir", help="the repo clone, for the same breakdown")
    c.set_defaults(func=cmd_census)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

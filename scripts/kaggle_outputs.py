"""What the Kaggle notebook leaves in /kaggle/working, the version's saved output.

    python scripts/kaggle_outputs.py pack --results <work dir>/results --out /kaggle/working
    python scripts/kaggle_outputs.py census --output /kaggle/working --work-dir <repo clone>

``pack``
    The notebook runs outside /kaggle/working (its work dir is under
    /kaggle/tmp, which is not saved), so this is what the version keeps.
    Exactly two files go into ``--out``:

    * ``results_light.zip``: json/csv/md/png/yaml/log/txt/npz only, no
      checkpoints -- what Phase 9 needs. Written first: it is small, so it
      survives a disk that fills up during the second zip.
    * ``results.zip``: everything, checkpoints included -- what a resume
      (``kaggle_inputs.py restore``) and a local E5/E6/E7 re-run need.

    Both leave out what is not a result: the gates' toy runs (``smoke/``, any
    ``_*`` test root) and the queue's internal state (``queue_state/``). Each
    is written under a temporary name, read back, then renamed, so a crash
    never leaves a truncated zip under the real name. After both are in
    place, everything else in ``--out`` is removed, so the output is exactly
    the two zips -- unless the results being packaged sit inside ``--out``,
    in which case nothing is removed.

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
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Kaggle reportedly caps a version's output at 500 files; warn well before.
OUTPUT_FILE_WARN = 400
ZIPS = ("results.zip", "results_light.zip")
# Not results: the gates' toy runs and the queue's internal state. Test roots
# (results/_*) never exist on Kaggle; a local pack leaves them out too.
SCRATCH_DIRS = {"smoke", "queue_state"}
# What Phase 9 reads. .npz = per-run test predictions (E7 agreement, galleries).
LIGHT_SUFFIXES = {".json", ".csv", ".md", ".png", ".yaml", ".log", ".txt", ".npz"}


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


def is_scratch(rel: Path) -> bool:
    top = rel.parts[0]
    return top in SCRATCH_DIRS or top.startswith("_")


def inside(path: Path, folder: Path) -> bool:
    path, folder = path.resolve(), folder.resolve()
    return path == folder or folder in path.parents


def write_zip(target: Path, results: Path, files: list[Path]) -> None:
    """``files`` (under ``results``) into ``target``, via a temporary name."""
    partial = target.with_name(target.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                # Checkpoints barely compress; storing them saves minutes.
                zf.write(p, p.relative_to(results).as_posix(),
                         compress_type=zipfile.ZIP_STORED if p.suffix == ".pt"
                         else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(partial) as zf:   # the central directory reads back whole
            n = len(zf.namelist())
        if n != len(files):
            raise RuntimeError(f"{partial} holds {n} entries, expected {len(files)}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def cmd_pack(args: argparse.Namespace) -> int:
    results = Path(args.results).resolve()
    out = Path(args.out).resolve()
    if not results.is_dir():
        print(f"pack: {results} does not exist -- nothing to package")
        return 1
    if inside(out, results):
        print(f"pack: --out {out} is inside {results}; the zips would contain themselves")
        return 2
    files, left_out = [], 0
    for p in sorted(results.rglob("*")):
        if p.is_file():
            if is_scratch(p.relative_to(results)):
                left_out += 1
            else:
                files.append(p)
    light = [p for p in files if p.suffix.lower() in LIGHT_SUFFIXES]
    out.mkdir(parents=True, exist_ok=True)
    write_zip(out / "results_light.zip", results, light)
    write_zip(out / "results.zip", results, files)
    for name, n, what in (("results.zip", len(files), "checkpoints included"),
                          ("results_light.zip", len(light), "no checkpoints")):
        print(f"{name:<18} {human((out / name).stat().st_size):>9}  ({n} files, {what})")
    print(f"left out: {left_out} scratch files (smoke/, queue_state/, test roots)")

    # Exactly the two zips in the output.
    if inside(results, out):
        print(f"pack: {results} is inside {out}, so nothing there is removed; the output "
              "keeps the results folder as well as the zips.")
        return 0
    for entry in sorted(out.iterdir()):
        if entry.name in ZIPS:
            continue
        n, _ = count(entry)
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        print(f"pack: removed {entry} ({n} {'file' if n == 1 else 'files'}): not part of the output")
    return 0


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
    p = sub.add_parser("pack", help="results/ -> results.zip + results_light.zip, nothing else")
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--out", default="/kaggle/working")
    p.set_defaults(func=cmd_pack)
    c = sub.add_parser("census", help="exact file counts in the output (and the work dir)")
    c.add_argument("--output", default="/kaggle/working")
    c.add_argument("--work-dir", help="the repo clone, for the same breakdown")
    c.set_defaults(func=cmd_census)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

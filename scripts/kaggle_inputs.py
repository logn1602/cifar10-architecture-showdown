"""Use what is attached to the Kaggle notebook (Add Input) before fetching anything.

    python scripts/kaggle_inputs.py cifar      # CIFAR-10: attached copy if any, else download
    python scripts/kaggle_inputs.py restore    # resume: an attached previous output -> results/

``cifar``
    CIFAR-10 is 170 MB from a university server, the run's only network
    dependency besides pip. If any attached input holds the official python
    version -- a ``cifar-10-batches-py/`` folder or ``cifar-10-python.tar.gz``
    -- it is copied into ``data/`` and every file is md5-checked against
    torchvision's own checksums, so the bytes (and with them the split
    fingerprint) are exactly what a download would give. An attached copy
    that fails the check is skipped with the reason printed. With no usable
    copy, CIFAR-10 is downloaded as before. Either way the source is printed.

``restore``
    A committed run that left work unfinished (failed runs, upgrades dropped
    for time, a crashed queue) keeps its output on its version page.
    Attached as an input, that output holds the earlier ``results/``
    packaged as ``results.zip`` and ``results_light.zip``; versions from
    before the run moved to /kaggle/tmp hold the folder itself as well. The
    source with the most finished runs is copied into ``results/``
    (``results.zip`` on a tie: the packaging cell writes it whole, while a
    folder is whatever Kaggle kept), leaving scratch and failure logs behind. The queue then skips every
    finished run through its own ``job_done()``, exactly as on a relaunch in
    place. If ``results/runs`` already holds finished runs, nothing is copied.
    Either way it reports the 9h queue against what is on disk: how much will
    be skipped, what resumes from ``last.pt``, what diverged and is not
    retried, and any restored run trained to other settings than this queue's.

``--input-root`` defaults to ``/kaggle/input``; the local tests point it at a
scratch directory instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config, load_json, resolve_path  # noqa: E402

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
# Resume: an attached previous output of this notebook
# ---------------------------------------------------------------------------

# Never restored. Scratch the packaging cell deletes anyway, local test roots,
# and failure logs: a failed job is retried, so its old log would only
# misreport this session. RUN_SUMMARY.md is rewritten by every session.
SKIP_DIRS = {"smoke", "_queue_test", "queue_state", "failures", "_rehearsal", "_report_test",
             "_eval_test", "_inputs_test"}
SKIP_FILES = {"RUN_SUMMARY.md"}
# What the packaging cell writes, best first on a tie: results.zip is whole by
# construction and has the checkpoints; results_light.zip has none.
ZIP_RANK = {"results.zip": 2, "results_light.zip": 0}
FOLDER_RANK = 1
EVAL_DONE = re.compile(r"runs/[^/]+/done\.json")


@dataclass
class Previous:
    path: Path              # a results/ folder, or a results(_light).zip
    is_zip: bool
    finished: int           # completed training runs + finished evaluations
    has_checkpoints: bool

    @property
    def rank(self) -> tuple[int, bool, int]:
        """Most finished runs, then checkpoints, then results.zip > folder > light zip."""
        kind = ZIP_RANK.get(self.path.name, 0) if self.is_zip else FOLDER_RANK
        return self.finished, self.has_checkpoints, kind

    def describe(self) -> str:
        what = "zip" if self.is_zip else "folder"
        ckpt = "with checkpoints" if self.has_checkpoints else "NO checkpoints"
        return f"{self.path} ({what}, {self.finished} finished runs, {ckpt})"


def _finished_in_folder(results: Path) -> int:
    n = sum(1 for _ in (results / "runs").glob("*/done.json"))
    for mp in (results / "runs").rglob("metrics.json"):
        try:
            n += load_json(mp).get("status") == "completed"
        except Exception:
            pass
    return n


def find_previous(input_root: Path) -> list[Previous]:
    """Every previous ``results/`` under ``input_root``: folders and the packaged zips."""
    if not input_root.is_dir():
        return []
    found = []
    for runs in sorted(input_root.rglob("runs")):
        if runs.is_dir() and runs.parent.name == "results":
            r = runs.parent
            found.append(Previous(r, False, _finished_in_folder(r),
                                  any(True for _ in runs.rglob("*.pt"))))
    for z in sorted(input_root.rglob("*.zip")):
        if z.name not in ZIP_RANK:
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                n = sum(1 for nm in names if EVAL_DONE.fullmatch(nm))
                for nm in names:
                    if nm.startswith("runs/") and nm.endswith("/metrics.json"):
                        try:
                            n += json.loads(zf.read(nm)).get("status") == "completed"
                        except Exception:
                            pass
                found.append(Previous(z, True, n, any(nm.endswith(".pt") for nm in names)))
        except zipfile.BadZipFile:
            print(f"Resume: ignoring {z}: not a readable zip")
    return found


def _skipped(rel: Path) -> bool:
    top = rel.parts[0] if rel.parts else ""
    return top in SKIP_DIRS or (len(rel.parts) == 1 and top in SKIP_FILES)


def copy_results(prev: Previous, dest: Path) -> int:
    """Copy file *contents* into ``dest``. Not permission bits: /kaggle/input is
    read-only, and read-only copies would stop the queue writing into a
    restored run (a partial run resuming, a log being appended)."""
    n = 0
    if prev.is_zip:
        with zipfile.ZipFile(prev.path) as zf:
            for info in zf.infolist():
                rel = Path(info.filename)
                # Our own archive, but never write outside dest regardless.
                if info.is_dir() or rel.is_absolute() or ".." in rel.parts or _skipped(rel):
                    continue
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
        return n
    for dirpath, dirnames, filenames in os.walk(prev.path):
        rel_dir = Path(dirpath).relative_to(prev.path)
        dirnames[:] = [d for d in dirnames if not _skipped(rel_dir / d)]
        (dest / rel_dir).mkdir(parents=True, exist_ok=True)
        for f in filenames:
            if not _skipped(rel_dir / f):
                shutil.copyfile(Path(dirpath) / f, dest / rel_dir / f)
                n += 1
    return n


def report_queue() -> None:
    """What the 9h queue will now skip -- judged by the queue's own job_done()."""
    from src.experiments import build_queue, job_done, load_base_config, upgrade_order

    cfg = load_base_config()
    jobs = build_queue(cfg, "9h", upgrade_order(cfg))
    done = [j for j in jobs if job_done(j)]
    by_exp = Counter(j.exp for j in done)
    print(f"Resume: {len(done)} of the {len(jobs)} jobs in the 9h queue (every upgrade "
          "included) are already complete; the queue will skip them.")
    if by_exp:
        print("        " + ", ".join(f"{e}: {n}" for e, n in by_exp.items()))
    train = [j for j in jobs if j.kind == "train" and not job_done(j)]
    diverged = [j.id for j in train if (resolve_path(j.out_dir) / "diverged.json").exists()]
    partial = [j.id for j in train if (resolve_path(j.out_dir) / "last.pt").exists()]
    if diverged:
        print(f"        {len(diverged)} diverged earlier and will not be retried: "
              + ", ".join(diverged))
    if partial:
        print(f"        {len(partial)} stopped part-way and will resume from last.pt: "
              + ", ".join(partial))
    # A finished run is skipped as this queue's result whatever it was trained
    # to. The planner's trims shorten runs legitimately; anything else means
    # the attached output came from another configuration. Say so loudly.
    other = []
    for j in done:
        if j.kind != "train":
            continue
        m = load_json(resolve_path(j.out_dir) / "metrics.json")
        want, got, unit = ((j.max_steps, m.get("total_steps"), "steps") if j.max_steps
                           else (j.epochs, m.get("epochs"), "epochs"))
        if got is None or abs(float(got) - float(want)) > 1e-6:
            other.append(f"{j.id} ({got} {unit}, untrimmed {want})")
    if other:
        print(f"Resume: WARNING {len(other)} finished run(s) were not trained to this queue's "
              "untrimmed settings, and are skipped as finished all the same (expected only "
              "if the earlier session's plan trimmed them): " + ", ".join(other))
    pending_eval = [j for j in jobs if j.kind == "eval" and not job_done(j)]
    e1 = [j for j in jobs if j.exp == "E1" and j.seed == 0 and job_done(j)]
    no_ckpt = [j.id for j in e1 if not (resolve_path(j.out_dir) / "best.pt").exists()]
    if pending_eval and no_ckpt:
        print(f"Resume: WARNING {', '.join(j.exp for j in pending_eval)} still have to run, "
              f"but these finished E1 runs have no best.pt to evaluate: {', '.join(no_ckpt)}")


def cmd_restore(args: argparse.Namespace) -> int:
    from src.experiments import results_path

    dest = results_path()
    already = _finished_in_folder(dest) if (dest / "runs").is_dir() else 0
    if already:
        print(f"Resume: {dest / 'runs'} already holds {already} finished runs; "
              "not restoring over them.")
        report_queue()
        return 0
    input_root = Path(args.input_root)
    found = [p for p in find_previous(input_root) if p.finished]
    if not found:
        print(f"Resume: no previous output with finished runs under {input_root}; "
              "starting from scratch.")
        return 0
    for p in found:
        print(f"Resume: found {p.describe()}")
    best = max(found, key=lambda p: p.rank)
    n = copy_results(best, dest)
    print(f"Resume: restored {n} files from {best.path} into {dest}")
    if not best.has_checkpoints:
        print("Resume: WARNING no checkpoints in this source (results_light.zip?): finished "
              "runs are skipped, but runs that stopped part-way restart from scratch and "
              "E5/E6/E7 cannot evaluate E1 runs without best.pt. Attach the full output "
              "(it contains results.zip) to avoid this.")
    report_queue()
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cifar", help="CIFAR-10 from an attached copy, else download")
    c.add_argument("--input-root", default=KAGGLE_INPUT)
    c.add_argument("--data-root", help="default: data.root from configs/base.yaml")
    c.set_defaults(func=cmd_cifar)
    r = sub.add_parser("restore", help="copy an attached previous output into results/")
    r.add_argument("--input-root", default=KAGGLE_INPUT)
    r.set_defaults(func=cmd_restore)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

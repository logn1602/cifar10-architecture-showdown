"""CPU test of ``scripts/kaggle_inputs.py`` against fake ``/kaggle/input`` trees.

Builds each attached-input layout the notebook can meet under
``results/_inputs_test`` (gitignored) and runs the real script on it:

* cifar: an attached extracted folder; an attached official archive; a
  corrupted folder next to a good archive (the folder must be skipped with its
  reason, the archive used); data already in place (nothing fetched). With
  ``--download``, also an empty input root (falls back to the real download).
  Built from the local CIFAR-10 files.
* restore: a previous output attached as its results folder, as results.zip
  and as results_light.zip; all three at once (results.zip wins the tie); a
  folder with more finished runs than the zip beside it (the folder wins); no
  usable output (a broken zip); and results/ already holding finished runs
  (nothing copied). The previous results/ holds finished, partial, diverged
  and wrongly-sized runs under real 9h-queue job ids, plus scratch that must
  stay behind. After each restore the real ``scripts/run_all.py --dry-run``
  must mark exactly the finished jobs done: the queue's own view of what it
  will skip.

Run: ``python scripts/kaggle_inputs_selftest.py [--only cifar|restore] [--download]``
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_ROOT = ROOT / "results" / "_inputs_test"
DATA = ROOT / "data"
BATCHES = DATA / "cifar-10-batches-py"
ARCHIVE = DATA / "cifar-10-python.tar.gz"


def run(*args: str, env: dict | None = None) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_inputs.py"), *args],
                          cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                          errors="replace",
                          env={**os.environ, **(env or {}), "PYTHONIOENCODING": "utf-8"})
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


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

# What the notebook's packaging cell keeps in results_light.zip.
LIGHT_SUFFIXES = {".json", ".csv", ".md", ".png", ".yaml", ".log", ".txt", ".npz"}
# Left behind by a restore (kaggle_inputs.SKIP_DIRS / SKIP_FILES).
JUNK = ["smoke/mlp/0/metrics.json", "_queue_test/runs/E1/mlp/0/metrics.json",
        "failures/E1__vit__s0.log", "queue_state/active_plan_9h.json", "RUN_SUMMARY.md"]


class Previous:
    """A previous session's results/, under real 9h-queue job ids."""

    def __init__(self) -> None:
        from src.experiments import build_queue, load_base_config, results_dir, upgrade_order

        cfg = load_base_config()
        self.jobs = {j.id: j for j in build_queue(cfg, "9h", upgrade_order(cfg))}
        self.base = results_dir()
        sweep_mlp = [i for i in self.jobs if i.startswith("lr_sweep/mlp/")]
        sweep_vit = [i for i in self.jobs if i.startswith("lr_sweep/vit/")]
        self.finished = sweep_mlp + ["E1/mlp/s0", "E7/eval"]
        self.mismatched = "E1/simple_cnn/s0"     # completed, at half the epochs
        self.partial = "E1/vit/s0"               # last.pt, no metrics.json
        self.diverged = sweep_vit[-1]
        self.extra = "E1/resnet18/s0"            # only in the "folder has more" case

    def rel(self, job_id: str) -> Path:
        return Path(self.jobs[job_id].out_dir).relative_to(self.base)

    def write(self, results: Path, extra: bool = False) -> list[Path]:
        """Write the tree; returns every file a restore must bring back (not the junk)."""
        written: list[Path] = []

        def put(rel: Path | str, text: str) -> None:
            p = results / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            written.append(Path(rel))

        def completed(job_id: str, scale: float = 1.0) -> None:
            j, d = self.jobs[job_id], self.rel(job_id)
            sized = ({"total_steps": j.max_steps} if j.max_steps
                     else {"epochs": j.epochs * scale, "total_steps": int(j.epochs * scale * 351)})
            put(d / "metrics.json", json.dumps({"status": "completed", "best_val_acc": 0.5,
                                                 "test_acc": 0.5, **sized}))
            put(d / "history.csv", "epoch,val_acc\n1,0.5\n")
            if j.save_checkpoints:
                put(d / "best.pt", "weights")

        for job_id in self.finished:
            if self.jobs[job_id].kind == "eval":
                put(self.rel(job_id) / "done.json", "{}")
                put(self.rel(job_id) / "summary.json", "{}")
            else:
                completed(job_id)
        completed(self.mismatched, scale=0.5)
        if extra:
            completed(self.extra)
        put(self.rel(self.partial) / "history.csv", "epoch,val_acc\n1,0.3\n")
        put(self.rel(self.partial) / "last.pt", "weights")
        put(self.rel(self.diverged) / "diverged.json", json.dumps({"status": "diverged"}))
        put("progress.log", "earlier session\n")
        put("tables/e1_scoreboard.md", "| model |\n")
        keep = list(written)
        for rel in JUNK:
            put(rel, json.dumps({"status": "completed"}))
        return keep

    def done_ids(self, extra: bool = False) -> set[str]:
        return set(self.finished) | {self.mismatched} | ({self.extra} if extra else set())


def make_results_zip(results: Path, out: Path) -> None:
    """results.zip exactly as the packaging cell makes it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=str(results))


def make_light_zip(results: Path, out: Path) -> None:
    """results_light.zip exactly as the packaging cell makes it: no checkpoints."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(results.rglob("*")):
            if p.is_file() and p.suffix.lower() in LIGHT_SUFFIXES:
                zf.write(p, p.relative_to(results))


def dry_run_done(results_env: str) -> tuple[int, set[str]]:
    """Job ids that the real queue's --dry-run marks done under ``results_env``."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--mode", "9h", "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "SHOWDOWN_RESULTS": results_env, "PYTHONIOENCODING": "utf-8"})
    done = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[-1] == "done":
            done.add(parts[0])
    return proc.returncode, done


def restore_cases() -> list[tuple[str, bool, str]]:
    prev = Previous()
    checks: list[tuple[str, bool, str]] = []

    def setup(name: str) -> tuple[Path, Path, str, Path]:
        """Fresh (input root, dest results root, its SHOWDOWN_RESULTS, a source results/)."""
        base = TEST_ROOT / f"restore_{name}"
        shutil.rmtree(base, ignore_errors=True)
        inp = base / "input"
        inp.mkdir(parents=True)
        dest = base / "results"
        return inp, dest, dest.relative_to(ROOT).as_posix(), base / "source" / "results"

    def case(name: str, inp: Path, dest: Path, env_root: str, expect: list[str],
             forbid: list[str] = (), done: set[str] | None = None,
             files: list[Path] = (), absent: list[Path] = ()) -> None:
        rc, out = run("restore", "--input-root", str(inp), env={"SHOWDOWN_RESULTS": env_root})
        print(f"--- restore: {name} ---\n{out.strip()}\n")
        problems = [f"exit {rc}"] if rc else []
        problems += [f"missing output {s!r}" for s in expect if s not in out]
        problems += [f"unexpected output {s!r}" for s in forbid if s in out]
        problems += [f"not restored: {p.as_posix()}" for p in files if not (dest / p).is_file()]
        problems += [f"restored but should not be: {p.as_posix()}" for p in absent
                     if (dest / p).exists()]
        if done is not None:
            drc, got = dry_run_done(env_root)
            if drc:
                problems.append(f"run_all.py --dry-run exit {drc}")
            if got != done:
                problems.append(f"dry run marks done {sorted(got)}, expected {sorted(done)}")
        checks.append((f"restore: {name}", not problems,
                       "; ".join(problems) or f"exit {rc}"
                       + (f", dry run: {len(done)} jobs done" if done is not None else "")))

    n_done = len(prev.done_ids())            # finished runs in the previous results/
    junk = [Path(Path(j).parts[0]) for j in JUNK]                 # top-level names
    report = ["already complete; the queue will skip them",
              f"1 diverged earlier and will not be retried: {prev.diverged}",
              "WARNING 1 finished run(s) were not trained to this queue's untrimmed settings",
              prev.mismatched]
    resumes = f"1 stopped part-way and will resume from last.pt: {prev.partial}"

    # 1. The previous output attached as its results folder (the repo clone in it).
    inp, dest, env_root, _ = setup("folder")
    src = inp / "showdown-runner" / "cifar10-architecture-showdown" / "results"
    files = prev.write(src)
    case("attached results folder is restored", inp, dest, env_root,
         [f"(folder, {n_done} finished runs, with checkpoints)",
          f"restored {len(files)} files from {src}", resumes] + report,
         forbid=["no best.pt"], done=prev.done_ids(), files=files, absent=junk)

    # 2. Restoring again, input still attached: results/runs is not overwritten.
    (dest / "progress.log").unlink()
    case("results/ already holds finished runs: nothing copied", inp, dest, env_root,
         [f"already holds {n_done} finished runs; not restoring over them"] + report,
         forbid=["Resume: restored"], done=prev.done_ids(), absent=[Path("progress.log")])

    # 3. Only results.zip.
    inp, dest, env_root, source = setup("zip")
    files = prev.write(source)
    make_results_zip(source, inp / "showdown-runner" / "results.zip")
    case("attached results.zip is restored", inp, dest, env_root,
         [f"(zip, {n_done} finished runs, with checkpoints)",
          f"restored {len(files)} files from", resumes] + report,
         forbid=["no best.pt"], done=prev.done_ids(), files=files, absent=junk)

    # 4. Only results_light.zip: finished runs are restored, checkpoints are not.
    inp, dest, env_root, source = setup("light")
    files = [p for p in prev.write(source) if p.suffix in LIGHT_SUFFIXES]
    make_light_zip(source, inp / "showdown-runner" / "results_light.zip")
    case("attached results_light.zip is restored, with the no-checkpoint warnings", inp, dest,
         env_root,
         [f"(zip, {n_done} finished runs, NO checkpoints)", f"restored {len(files)} files from",
          "WARNING no checkpoints in this source",
          "E5, E6 still have to run, but these finished E1 runs have no best.pt to evaluate: "
          f"E1/mlp/s0, {prev.mismatched}"] + report,
         forbid=["resume from last.pt"], done=prev.done_ids(), files=files,
         absent=junk + [prev.rel("E1/mlp/s0") / "best.pt", prev.rel(prev.partial) / "last.pt"])

    # 5. Folder, results.zip and results_light.zip together: results.zip wins the tie.
    inp, dest, env_root, _ = setup("all_three")
    out = inp / "showdown-runner"
    files = prev.write(out / "cifar10-architecture-showdown" / "results")
    make_results_zip(out / "cifar10-architecture-showdown" / "results", out / "results.zip")
    make_light_zip(out / "cifar10-architecture-showdown" / "results", out / "results_light.zip")
    case("folder + results.zip + results_light.zip: results.zip is used", inp, dest, env_root,
         [f"restored {len(files)} files from {out / 'results.zip'}"] + report,
         done=prev.done_ids(), files=files, absent=junk)

    # 6. A folder with more finished runs than the zip beside it: the folder wins.
    inp, dest, env_root, source = setup("folder_more")
    out = inp / "showdown-runner"
    prev.write(source)
    make_results_zip(source, out / "results.zip")
    folder = out / "cifar10-architecture-showdown" / "results"
    files = prev.write(folder, extra=True)
    case("a folder with more finished runs beats results.zip", inp, dest, env_root,
         [f"(folder, {n_done + 1} finished runs, with checkpoints)",
          f"(zip, {n_done} finished runs", f"restored {len(files)} files from {folder}"] + report,
         done=prev.done_ids(extra=True), files=files, absent=junk)

    # 7. Nothing usable attached: a broken zip and an unrelated dataset.
    inp, dest, env_root, _ = setup("nothing")
    (inp / "showdown-runner").mkdir()
    (inp / "showdown-runner" / "results.zip").write_text("not a zip", encoding="utf-8")
    (inp / "some-dataset" / "results").mkdir(parents=True)
    (inp / "some-dataset" / "results" / "notes.txt").write_text("x", encoding="utf-8")
    case("nothing usable attached: starts from scratch", inp, dest, env_root,
         ["not a readable zip", "no previous output with finished runs under"],
         forbid=["Resume: restored", "Resume: found"], done=set(), absent=[Path("runs")])
    return checks


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["cifar", "restore"], help="run one group of cases")
    ap.add_argument("--download", action="store_true",
                    help="also test the fallback download (fetches 170 MB)")
    args = ap.parse_args()
    checks = []
    if args.only in (None, "cifar"):
        if not (BATCHES.is_dir() and ARCHIVE.is_file()):
            print(f"needs the local CIFAR-10 files in {DATA} (python -m src.data downloads them)")
            return 1
        checks += cifar_cases(args.download)
    if args.only in (None, "restore"):
        checks += restore_cases()
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({detail})")
    ok = all(c[1] for c in checks)
    print(f"\n{sum(c[1] for c in checks)}/{len(checks)} checks passed -- "
          f"{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

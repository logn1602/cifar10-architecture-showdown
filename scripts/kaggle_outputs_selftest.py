"""CPU test of ``scripts/kaggle_outputs.py`` on fake /kaggle/working trees.

Builds each case under ``results/_outputs_test`` (gitignored) and runs the real
script on it:

* census: an output holding the repo clone (as the notebook used to leave it)
  plus the two zips, 441 files in all -- the exact total and every top-level
  count must be printed, with the near-the-cap warning, dotfiles and nested
  folders included; and a small output with no warning.
* pack: a toy results tree of exactly 600 files under real 9h-queue job ids
  (finished, partial, diverged, failed and pending runs, the plan, tables,
  figures, and scratch that must stay out), packed into an output that
  already holds junk. The output must end with only results.zip and
  results_light.zip, each holding exactly the right files and passing its
  CRC check, and the census must count 2 files. Also: results/ inside the
  output (nothing removed) and --out inside results/ (refused).
* resume: those two zips attached as an earlier version's output.
  ``kaggle_inputs.py restore`` must find them, restore results.zip byte for
  byte into a fresh results/, report the partial run as resuming from
  last.pt, and the real ``run_all.py --dry-run`` must then mark exactly the
  finished jobs done.

Run: ``python scripts/kaggle_outputs_selftest.py [--only census|pack]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
TEST_ROOT = ROOT / "results" / "_outputs_test"
TOY_FILES = 600
LIGHT_SUFFIXES = {".json", ".csv", ".md", ".png", ".yaml", ".log", ".txt", ".npz"}


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


# ---------------------------------------------------------------------------
# pack + resume
# ---------------------------------------------------------------------------


class Toy:
    """A results/ tree as a 9h run leaves it, under real queue job ids."""

    UPGRADES = ["e3_resnet18_vit", "e1_seeds_1_2"]   # the first two in spec order

    def __init__(self) -> None:
        from src.experiments import build_queue, load_base_config, results_dir

        cfg = load_base_config()
        self.jobs = {j.id: j for j in build_queue(cfg, "9h", self.UPGRADES)}
        self.base = results_dir()
        sweep_vit = [i for i in self.jobs if i.startswith("lr_sweep/vit/")]
        self.partial = "E1/vit/s0"                       # last.pt, no metrics.json
        self.diverged = sweep_vit[-1]                    # diverged.json, never retried
        self.failed = "E2/vit/f0.1/s0"                   # failure log, retried
        pending = {i for i in self.jobs if i.startswith("E4/")} | {"E6/eval"}
        self.finished = sorted(set(self.jobs) - pending
                               - {self.partial, self.diverged, self.failed})

    def rel(self, job_id: str) -> Path:
        return Path(self.jobs[job_id].out_dir).relative_to(self.base)

    def write(self, results: Path) -> dict[str, list[str]]:
        """Write exactly TOY_FILES files; returns the relative paths by kind."""
        made: dict[str, list[str]] = {"result": [], "scratch": []}

        def put(rel: Path | str, text: str = "x", kind: str = "result") -> None:
            p = results / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            made[kind].append(Path(rel).as_posix())

        def started(job_id: str) -> Path:
            d = self.rel(job_id)
            put(d / "config.yaml", "model: x\n")
            put(d / "history.csv", "epoch,val_acc\n1,0.5\n")
            put(d / "train.log", "log\n")
            put(Path("tb") / d.relative_to("runs") / "events.out.tfevents.0")
            return d

        for job_id in self.finished:
            j = self.jobs[job_id]
            if j.kind == "eval":
                d = self.rel(job_id)
                put(d / "done.json", "{}")
                put(d / "summary.json", "{}")
                put(d / "raw.csv", "a,b\n")
                continue
            d = started(job_id)
            sized = {"total_steps": j.max_steps} if j.max_steps else {"epochs": j.epochs}
            put(d / "metrics.json", json.dumps({"status": "completed", "best_val_acc": 0.5,
                                                 "test_acc": 0.5, **sized}))
            if j.save_checkpoints:
                put(d / "best.pt", "weights")
                put(d / "test_outputs.npz", "npz")
        put(started(self.partial) / "last.pt", "weights")
        put(started(self.diverged) / "diverged.json", json.dumps({"status": "diverged"}))
        started(self.failed)
        put(Path("failures") / (self.failed.replace("/", "__") + ".log"), "Traceback ...\n")
        put("queue_9h.json", json.dumps({"mode": "9h", "upgrades": self.UPGRADES, "trims": [],
                                         "estimates": {}, "budget_hours": 8.0}))
        for name in ("progress.log", "run_all.log", "RUN_SUMMARY.md", "lr_choice_9h.json",
                     "smoke_test.json", "queue_selftest.json", "model_checks.json"):
            put(name, "{}" if name.endswith(".json") else "x\n")
        for name in ("e1_scoreboard.md", "e1_scoreboard.csv", "robustness_summary.csv"):
            put(Path("tables") / name)
        # Scratch: the gates' toy runs and the queue's state. Never packaged.
        for m in ("mlp", "vit"):
            for f in ("best.pt", "last.pt", "metrics.json", "history.csv"):
                put(Path("smoke") / m / "0" / f, kind="scratch")
        put("queue_state/active_plan_9h.json", "{}", kind="scratch")
        put("_queue_test/runs/E1/mlp/0/metrics.json", json.dumps({"status": "completed"}),
            kind="scratch")
        n = len(made["result"]) + len(made["scratch"])
        if n > TOY_FILES:
            raise RuntimeError(f"toy tree already has {n} files, more than {TOY_FILES}")
        for i in range(TOY_FILES - n):                 # figures fill it up to exactly 600
            put(Path("figures") / f"fig_{i:03d}.png", "png")
        return made


def zip_names(path: Path) -> tuple[list[str], str | None]:
    """(file names, first bad member or None) -- testzip reads and CRCs every member."""
    with zipfile.ZipFile(path) as zf:
        return sorted(n for n in zf.namelist() if not n.endswith("/")), zf.testzip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_cases() -> list[tuple[str, bool, str]]:
    import kaggle_inputs_selftest as inputs_test   # its restore runner and dry-run reader

    checks: list[tuple[str, bool, str]] = []
    toy = Toy()

    # 1. The 600-file toy tree, packed into an output that already holds junk.
    base = TEST_ROOT / "pack"
    shutil.rmtree(base, ignore_errors=True)
    results = base / "tmp" / "cifar10-architecture-showdown" / "results"
    out = base / "working"
    made = toy.write(results)
    n_tree = sum(1 for p in results.rglob("*") if p.is_file())
    fill(out / "old_clone", 50)
    (out / "notes.txt").write_text("x", encoding="utf-8")
    (out / "results.zip").write_bytes(b"a stale, broken results.zip")
    (out / "results.zip.partial").write_bytes(b"left by a crashed attempt")
    rc, log = run("pack", "--results", str(results), "--out", str(out))
    print(f"--- pack: {TOY_FILES}-file toy tree ---\n{log.strip()}\n")
    want_full = sorted(made["result"])
    want_light = sorted(n for n in want_full if Path(n).suffix in LIGHT_SUFFIXES)
    left = sorted(p.name for p in out.iterdir())
    full, bad_full = zip_names(out / "results.zip") if left == ["results.zip", "results_light.zip"] \
        else ([], "n/a")
    light, bad_light = zip_names(out / "results_light.zip") if not bad_full else ([], "n/a")
    problems = [f"exit {rc}"] if rc else []
    if n_tree != TOY_FILES:
        problems.append(f"toy tree has {n_tree} files, not {TOY_FILES}")
    if left != ["results.zip", "results_light.zip"]:
        problems.append(f"output holds {left}")
    for name, have, want in (("results.zip", full, want_full),
                             ("results_light.zip", light, want_light)):
        if have != want:
            odd = sorted(set(have) ^ set(want))[:1]
            problems.append(f"{name}: {len(have)} files, expected {len(want)}"
                            + (f" (first difference: {odd[0]})" if odd else ""))
    if bad_full or bad_light:
        problems.append(f"CRC check failed: {bad_full or bad_light}")
    if any(n.endswith(".pt") for n in light):
        problems.append("a checkpoint in results_light.zip")
    for s in (f"({len(want_full)} files, checkpoints included)",
              f"({len(want_light)} files, no checkpoints)",
              f"left out: {len(made['scratch'])} scratch files",
              "old_clone (50 files): not part of the output",
              "notes.txt (1 file): not part of the output"):
        if s not in log:
            problems.append(f"missing output {s!r}")
    # The stale results.zip and .partial are not "removed": pack writes over them.
    checks.append((f"pack: {TOY_FILES}-file tree -> the output holds only the two zips",
                   not problems, "; ".join(problems) or
                   f"{len(want_full)} + {len(want_light)} files zipped, "
                   f"{len(made['scratch'])} scratch left out, 2 strays removed, "
                   "a stale zip and .partial replaced"))

    rc, log = run("census", "--output", str(out), "--work-dir", str(results.parent))
    got = parse_census(log).get(str(out), {})
    ok = rc == 0 and got.get("total") == 2 and "WARNING" not in log
    checks.append(("pack: census of that output counts 2 files, no warning", ok,
                   f"total {got.get('total')}"))

    # 2. Old layout: results/ inside the output. Zips written, nothing removed.
    base = TEST_ROOT / "pack_inside"
    shutil.rmtree(base, ignore_errors=True)
    out = base / "working"
    inner = out / "cifar10-architecture-showdown" / "results"
    fill(inner / "runs", 5)
    rc, log = run("pack", "--results", str(inner), "--out", str(out))
    left = sorted(p.name for p in out.iterdir())
    ok = (rc == 0 and left == ["cifar10-architecture-showdown", "results.zip",
                               "results_light.zip"]
          and "nothing there is removed" in log and len(list((inner / "runs").iterdir())) == 5)
    checks.append(("pack: results/ inside the output -> zips written, nothing removed", ok,
                   f"exit {rc}, output holds {left}"))

    # 3. --out inside results/: refused before anything is written.
    rc, log = run("pack", "--results", str(inner), "--out", str(inner / "out"))
    ok = rc == 2 and "would contain themselves" in log and not (inner / "out").exists()
    checks.append(("pack: --out inside results/ is refused", ok, f"exit {rc}"))

    # 4. Resume: the two zips attached as an earlier version's output.
    base = TEST_ROOT / "resume"
    shutil.rmtree(base, ignore_errors=True)
    attached = base / "input" / "showdown-runner"     # /kaggle/input/<notebook>/
    attached.mkdir(parents=True)
    for name in ("results.zip", "results_light.zip"):
        shutil.copy2(TEST_ROOT / "pack" / "working" / name, attached / name)
    dest = base / "tmp" / "cifar10-architecture-showdown" / "results"
    env_root = dest.relative_to(ROOT).as_posix()
    rc, log = inputs_test.run("restore", "--input-root", str(base / "input"),
                              env={"SHOWDOWN_RESULTS": env_root})
    print(f"--- resume from the packed output ---\n{log.strip()}\n")
    n_done = len(toy.finished)
    # restore leaves failure logs and RUN_SUMMARY.md behind (they describe the old session)
    want = sorted(n for n in want_full if not n.startswith("failures/") and n != "RUN_SUMMARY.md")
    have = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    problems = [f"exit {rc}"] if rc else []
    for s in (f"results.zip (zip, {n_done} finished runs, with checkpoints)",
              f"results_light.zip (zip, {n_done} finished runs, NO checkpoints)",
              f"restored {len(want)} files from {attached / 'results.zip'}",
              f"1 stopped part-way and will resume from last.pt: {toy.partial}",
              f"1 diverged earlier and will not be retried: {toy.diverged}"):
        if s not in log:
            problems.append(f"missing output {s!r}")
    if "WARNING" in log:
        problems.append("unexpected WARNING")
    if have != want:
        problems.append(f"restored {len(have)} files, expected {len(want)}")
    else:
        src = TEST_ROOT / "pack" / "tmp" / "cifar10-architecture-showdown" / "results"
        changed = [n for n in want if sha(src / n) != sha(dest / n)]
        if changed:
            problems.append(f"{len(changed)} restored files differ, e.g. {changed[0]}")
    drc, done = inputs_test.dry_run_done(env_root)
    if drc:
        problems.append(f"run_all.py --dry-run exit {drc}")
    if done != set(toy.finished):
        problems.append(f"dry run marks {len(done)} done, expected {n_done}: "
                        f"extra {sorted(done - set(toy.finished))[:3]}, "
                        f"missing {sorted(set(toy.finished) - done)[:3]}")
    checks.append(("resume: the packed output restores, and the queue skips exactly the "
                   "finished runs", not problems, "; ".join(problems) or
                   f"{len(have)} files restored byte for byte; dry run: {len(done)} of "
                   f"{len(toy.jobs)} jobs done, {toy.partial} resumes"))
    return checks


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["census", "pack"],
                    help="one group of cases (pack includes the resume round trip)")
    args = ap.parse_args()
    checks = []
    if args.only in (None, "census"):
        checks += census_cases()
    if args.only in (None, "pack"):
        checks += pack_cases()
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({detail})")
    ok = all(c[1] for c in checks)
    print(f"\n{sum(c[1] for c in checks)}/{len(checks)} checks passed -- "
          f"{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

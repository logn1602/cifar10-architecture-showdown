r"""Generate kaggle_runner.ipynb -- the unattended Kaggle runner -- from source.

The notebook is generated, never hand-edited. Edit the cells below, then:

    python scripts/make_kaggle_notebook.py           # rewrite the notebook, then verify it
    python scripts/make_kaggle_notebook.py --check   # verify the notebook on disk only
    python scripts/make_kaggle_notebook.py --check --python <a python3.10>   # Kaggle's parser

Every cell is a raw string (r'''...'''), so its text reaches the notebook byte
for byte: nothing is un-escaped on the way, and print("\n" + "!" * 70) stays
exactly that. The Phase 3 version of the notebook was patched through a shell
heredoc instead, which turned that \n into a real line break and split the
string literal; cell 7 then died with a SyntaxError on Kaggle, after the
pre-flight cells had already used GPU time.

Hence the verification, which runs after every write and on --check: each code
cell is extracted from the .ipynb on disk into its own file and compiled with
``python -m py_compile``, and any failure exits non-zero. A cell that does not
compile here would not run on Kaggle either. --check also fails when the
notebook on disk differs from what this script would write, i.e. when someone
edited the .ipynb directly.

A cell may not contain three consecutive single quotes (they would end its raw
string); the script refuses to write such a cell rather than mangle it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle_runner.ipynb"

# Notebook-level metadata. "kaggle.accelerator" pre-selects GPU T4 x2 and
# "isInternetEnabled" turns Internet on when the notebook is imported.
METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "accelerator": "GPU",
    "kaggle": {
        "accelerator": "nvidiaTeslaT4x2",
        "dataSources": [],
        "isInternetEnabled": True,
        "language": "python",
        "sourceType": "notebook",
    },
}

# ---------------------------------------------------------------------------
# The cells, in order: (cell type, source). Leading and trailing blank lines
# of each source are dropped when the notebook is written.
# ---------------------------------------------------------------------------

CELLS: list[tuple[str, str]] = [
    # --- cell 0 (markdown): Evolution of Vision — CIFAR-10 Architecture Showdown
    ('markdown', r'''
# Evolution of Vision — CIFAR-10 Architecture Showdown
## Kaggle runner (9-hour unattended mode)

Runs the **entire** training queue: LR sweep → E1 → E5/E6/E7 evaluation → E2 → E4 → upgrades.
Six architectures (MLP, Simple CNN, ResNet-18, ViT, MLP-Mixer, ConvNeXt-mini), all seven
experiments, trimmed seeds and epochs per spec Section 12.

---

### ⚙️ Required Kaggle settings — check these BEFORE you start

| Setting | Value | Why |
|---|---|---|
| **Accelerator** | **GPU T4 × 2** | The queue is split across both GPUs. On 1× T4 the core does **not** fit in 8 h. |
| **Internet** | **ON** | Needed for `pip install`, the E4 pretrained weights, and the CIFAR-10 download unless a copy is attached. |
| **Input** (optional) | A CIFAR-10 dataset, python version | **Add Input** → search "CIFAR-10 Python". Used instead of the 170 MB download when every file passes its md5 check; the pre-flight log prints which path it used. |
| **Input** (only to resume) | An earlier version's output | To continue a version that left runs unfinished: **Add Input** → *Notebooks* → this notebook → that version, then commit again. Every run it finished is skipped. Leave it off for a fresh run. |
| **Environment** | **Always use latest environment** | A notebook pinned to its original environment can get a 2023 image (Python 3.10, PyTorch 2.0). The code still runs there, but the latest environment has a current PyTorch. |
| **Persistence** | Files only (optional) | Not required. The run works in `/kaggle/tmp`, which is not saved; its only output is `results.zip` and `results_light.zip` in `/kaggle/working`. |

### ▶️ How to start it

Use **Save Version → Save & Run All (Commit)**, *not* the interactive "Run All".
A committed run keeps going with your browser closed; an interactive session dies when you disconnect.

**Optional, recommended first: a ~10-minute smoke run.** Set `SMOKE = True` in the CONFIG cell and
commit. Only `scripts/smoke_test.py` runs: every model through the real training loop on the T4
(fp16 + GradScaler, interrupted and resumed, the E2/E3/E4 code paths). Every later cell skips
itself. If it passes, set `SMOKE = False` and commit again for the real run.

**Check Kaggle's current per-session GPU limit before you start** (historically ~12 h, and there is a
weekly GPU quota of ~30 h). The default `BUDGET_HOURS = 8` is chosen to sit comfortably under it.

### ⏱️ Expected timeline

Roughly **6.5–8 h** on 2× T4. The planner prints a projected finish time before the long run begins —
read it, and if it exceeds your remaining session time, lower `BUDGET_HOURS`.
'''),

    # --- cell 1 (code): CONFIG -- the only cell you should normally need to edit.
    ('code', r'''
# =============================================================================
# CONFIG -- the only cell you should normally need to edit.
# =============================================================================
from pathlib import Path

# How the repo gets onto this machine.
#   "git"     -> clone from REPO_URL (the repo is public, so no credentials)
#   "dataset" -> you uploaded the repo as a Kaggle Dataset instead
SOURCE = "git"

REPO_URL = "https://github.com/logn1602/cifar10-architecture-showdown.git"
DATASET_SLUG = ""          # only used when SOURCE="dataset"; "" auto-detects

BUDGET_HOURS = 8.0         # spec 12.4 default: ~1 h of slack inside a 9 h window
MODE = "9h"

# SMOKE = True: a short GPU check before spending the 8-hour budget. Runs only
# scripts/smoke_test.py (every model through the real training loop on the T4,
# interrupted and resumed), then every later cell skips itself.
SMOKE = False

# Kaggle saves /kaggle/working as this version's output, and may cap it at ~500
# files. So the run works outside it: the clone, data/, results/ and every
# checkpoint live in WORK_DIR, which is not saved, and the packaging cell leaves
# exactly two files in OUTPUT_DIR: results.zip and results_light.zip.
OUTPUT_DIR = Path("/kaggle/working")
SCRATCH = Path("/kaggle/tmp")       # any writable path outside OUTPUT_DIR works, e.g. /tmp
WORK_DIR = SCRATCH / "cifar10-architecture-showdown"
SPEC_FILE = "CIFAR10_ARCHITECTURE_SHOWDOWN.md"   # the marker that identifies the repo root

# -----------------------------------------------------------------------------
# Shared helper. Every long-running step goes through this rather than a `!`
# magic, because a failing `!` cell does NOT stop a committed run -- it prints
# the error and the notebook sails on to the next cell. That is exactly how you
# come back after 8 hours to an empty results/ directory.
# -----------------------------------------------------------------------------
import subprocess, sys, os, time

def run(cmd, cwd=None, check=True, tee=None):
    """Run a command, streaming output live, optionally teeing to a file."""
    cwd = str(cwd or WORK_DIR)
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n  (cwd={cwd})", flush=True)
    t0 = time.time()
    sink = open(tee, "a", encoding="utf-8") if tee else None
    proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1, text=True,
                            encoding="utf-8", errors="replace")
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if sink:
                sink.write(line)
    finally:
        proc.wait()
        if sink:
            sink.close()
    dt = time.time() - t0
    print(f"\n[exit {proc.returncode} after {dt/60:.1f} min]", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {cmd}")
    return proc.returncode

PY = sys.executable
print("config loaded")
'''),

    # --- cell 2 (code): 1. Environment probe -- fail fast and loudly if the accelerator is wro
    ('code', r'''
# =============================================================================
# 1. Environment probe -- fail fast and loudly if the accelerator is wrong.
# =============================================================================
import shutil, subprocess, sys, os

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch
print(f"python            : {sys.version.split()[0]}")
print(f"torch             : {torch.__version__}  (CUDA build: {torch.version.cuda})")
print(f"cuda available    : {torch.cuda.is_available()}")
n_gpu = torch.cuda.device_count()
print(f"device count      : {n_gpu}")
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1024**3:.1f} GB  sm_{p.major}{p.minor}")
if n_gpu:
    print(f"bf16 supported    : {torch.cuda.is_bf16_supported()}")
print(f"cpu count         : {os.cpu_count()}")
try:
    import psutil
    print(f"RAM               : {psutil.virtual_memory().total/1024**3:.1f} GB")
except ImportError:
    pass
SCRATCH.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(SCRATCH).free / 1024**3
print(f"{str(SCRATCH):<18}: {free:.1f} GB free  (work dir: data, results, checkpoints)")
print(f"{str(OUTPUT_DIR):<18}: {shutil.disk_usage(OUTPUT_DIR).free / 1024**3:.1f} GB free"
      "  (output: the two zips)")

assert torch.cuda.is_available(), (
    "No GPU! Settings -> Accelerator -> GPU T4 x2, then restart the session."
)
if n_gpu < 2:
    print("\n" + "!" * 70)
    print(f"!! Only {n_gpu} GPU allocated, expected 2.")
    print("!! The queue still runs, but on 1x T4 the core does NOT fit in 8 h.")
    print("!! Set Accelerator -> GPU T4 x2 and restart, or lower the scope.")
    print("!" * 70)
if free < 15:
    print(f"\n!! Only {free:.1f} GB free in {SCRATCH} -- checkpoints + CIFAR-10 need ~10 GB. "
          "Watch for ENOSPC.")

# Kaggle can pin a notebook to the environment it was created in. A pinned
# 2023 image (Python 3.10, PyTorch 2.0) still runs this project -- the code is
# kept compatible with PyTorch 2.0 -- but the latest environment is better.
torch_version = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
if torch_version < (2, 1):
    print("\n" + "!" * 70)
    print(f"!! PyTorch {torch.__version__} on Python {sys.version.split()[0]}: this notebook")
    print("!! is probably PINNED to an old Kaggle environment. The run still works,")
    print("!! but Settings -> Environment -> 'Always use latest environment' is better.")
    print("!" * 70)
'''),

    # --- cell 3 (code): 2. Fetch the repo into the work dir (not /kaggle/working).
    ('code', r'''
# =============================================================================
# 2. Fetch the repo into the work dir under /kaggle/tmp -- not /kaggle/working,
#    which is the saved output and gets only the two zips at the end.
#
# Copy, never symlink: /kaggle/input is READ-ONLY and the code writes results,
# checkpoints and the data/ directory.
# =============================================================================
import shutil
from pathlib import Path

def find_dataset_root():
    """Locate the uploaded repo under /kaggle/input by its spec file."""
    if DATASET_SLUG:
        cand = Path("/kaggle/input") / DATASET_SLUG
        hits = [cand] if (cand / SPEC_FILE).exists() else list(cand.rglob(SPEC_FILE))
        if hits:
            return hits[0] if hits[0].is_dir() else hits[0].parent
        raise FileNotFoundError(f"{SPEC_FILE} not found under {cand}")
    hits = sorted(Path("/kaggle/input").glob(f"*/**/{SPEC_FILE}"))
    hits += sorted(Path("/kaggle/input").glob(f"*/{SPEC_FILE}"))
    roots = sorted({h.parent for h in hits})
    if not roots:
        raise FileNotFoundError(
            "No dataset under /kaggle/input contains " + SPEC_FILE + ".\n"
            "Add your repo as a Dataset (see docs/KAGGLE.md), or set SOURCE='git'."
        )
    if len(roots) > 1:
        raise RuntimeError(f"Ambiguous: {SPEC_FILE} found in {roots}. Set DATASET_SLUG.")
    return roots[0]

if (WORK_DIR / SPEC_FILE).exists():
    print(f"repo already present at {WORK_DIR} -- leaving it alone (cell is idempotent)")
elif SOURCE == "dataset":
    src = find_dataset_root()
    print(f"copying {src} -> {WORK_DIR}")
    shutil.copytree(src, WORK_DIR, dirs_exist_ok=True)
elif SOURCE == "git":
    WORK_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", REPO_URL, str(WORK_DIR)], cwd=str(WORK_DIR.parent))
else:
    raise ValueError(f"SOURCE must be 'dataset' or 'git', got {SOURCE!r}")

os.chdir(WORK_DIR)
assert (WORK_DIR / SPEC_FILE).exists(), f"{SPEC_FILE} missing -- wrong source?"
print(f"\ncwd = {os.getcwd()}\n")

for p in sorted(WORK_DIR.iterdir()):
    print(f"  {p.name}{'/' if p.is_dir() else ''}")
    if p.is_dir() and p.name in {"src", "scripts", "configs"}:
        for q in sorted(p.iterdir()):
            print(f"      {q.name}{'/' if q.is_dir() else ''}")
'''),

    # --- cell 4 (code): 3. Install the extras.
    ('code', r'''
# =============================================================================
# 3. Install the extras.
#
# NEVER `pip install torch` here. The Kaggle image already ships a CUDA build;
# a stray torch install can pull a CPU wheel and silently kill GPU training --
# you would not notice until the whole queue ran 100x too slow.
# =============================================================================
import os
os.chdir(WORK_DIR)

before = None
import torch
before = (torch.__version__, torch.version.cuda)

run([PY, "-m", "pip", "install", "-q",
     "timm", "grad-cam", "umap-learn", "pyyaml", "tqdm", "seaborn"], check=False)

# Re-verify in a FRESH interpreter: this process already has torch imported, so
# a downgraded wheel on disk would not show up here.
check = subprocess.run(
    [PY, "-c",
     "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
    capture_output=True, text=True)
print("after install:", check.stdout.strip() or check.stderr.strip())
assert "True" in check.stdout, (
    f"torch lost CUDA after the pip install (was {before}). "
    "Something pulled a CPU wheel -- restart the session and remove the offending package."
)
print("\nextras installed, torch CUDA intact")

# The versions this run actually uses. scripts/rehearse.ps1 rehearses this
# environment locally from pinned versions -- if these differ, update its pins.
import importlib
from importlib import metadata
importlib.invalidate_caches()
print("\npackage versions after the install:")
for dist in ("torch", "torchvision", "numpy", "timm", "scipy", "scikit-learn", "pandas",
             "matplotlib", "seaborn", "umap-learn", "numba", "grad-cam", "pillow",
             "opencv-python", "opencv-python-headless", "tensorboard", "psutil"):
    try:
        print(f"  {dist:<24} {metadata.version(dist)}")
    except metadata.PackageNotFoundError:
        print(f"  {dist:<24} (not installed)")
'''),

    # --- cell 5 (code): 4. Pre-flight: what exists, and does the data pipeline reproduce?
    ('code', r'''
# =============================================================================
# 4. Pre-flight: what exists, and does the data pipeline reproduce?
#
# The split sha256 printed here MUST match the one from your local
# `python scripts/check_data.py`. data/ is gitignored, so the 45k/5k split is
# regenerated here rather than shipped -- these hashes are the proof that
# "val accuracy" means the same thing on both machines.
#
# With SMOKE = True this cell fetches CIFAR-10 and runs only the GPU smoke
# test, and every cell after it skips itself.
# =============================================================================
import os, shutil
os.chdir(WORK_DIR)

# CIFAR-10: an attached Kaggle Dataset holding the official python version
# (cifar-10-batches-py/ or cifar-10-python.tar.gz) is copied into data/ and
# md5-checked file by file; with none attached, it is downloaded. Either way
# the bytes -- and so the split -- are the same. The log says which path it used.
run([PY, "scripts/kaggle_inputs.py", "cifar"])

if SMOKE:
    # Every model through the real training loop on the T4 -- fp16 autocast,
    # the GradScaler, DataLoader workers, interrupt + resume, the E2/E3/E4
    # code paths. ~2-4 min. A failure raises here and stops the run.
    try:
        run([PY, "scripts/smoke_test.py"])
    finally:
        # The work dir is not saved: put the record in the output, pass or fail.
        record = WORK_DIR / "results" / "smoke_test.json"
        if record.exists():
            shutil.copy2(record, OUTPUT_DIR / record.name)
    print("\n>>> SMOKE run passed. Nothing else runs in this version.")
    print(">>> Set SMOKE = False in the CONFIG cell for the real run.")
else:
    REQUIRED = {
        "scripts/env_check.py":        "Phase 0",
        "scripts/check_data.py":       "Phase 1",
        "scripts/model_checks.py":     "Phase 2",
        "src/data.py":                 "Phase 1",
        "scripts/plan_budget.py":      "Phase 3",
        "scripts/run_all.py":          "Phase 3",
    }
    print("Required scripts:")
    missing = []
    for rel, phase in REQUIRED.items():
        ok = (WORK_DIR / rel).exists()
        print(f"  [{'ok' if ok else ' MISSING '}] {rel:<28} ({phase})")
        if not ok:
            missing.append(rel)

    assert not missing, f"Missing {missing} -- is the clone at the right commit?"

    # Resume: if an earlier version's output is attached (Add Input ->
    # Notebooks), its results/ is copied in first, so the planner budgets only
    # what is left and the queue skips every finished run. Before the gates
    # below, so they still write this session's records over the old ones.
    run([PY, "scripts/kaggle_inputs.py", "restore"])

    run([PY, "scripts/env_check.py"])
    run([PY, "-m", "src.data"])
    run([PY, "scripts/check_data.py", "--quick"])

    # Phase 2 gate. On CPU this takes ~50 min; on a T4 it is seconds, so it is
    # cheap insurance to re-verify every model can still memorise 256 images
    # right before committing 8 GPU-hours to training them.
    run([PY, "scripts/model_checks.py"])

    # Phase 3 gates, on the real GPU (spec 12.6: "smoke test passed on the GPU").
    #  - smoke_test: every model through the real training loop, interrupted and
    #    resumed, all output files checked. ~2-4 min on a T4.
    #  - queue_selftest: the unattended queue itself -- child processes, deps,
    #    LR auto-pick, a deliberately diverging run, relaunch/skip. Toy-sized runs
    #    under results/_queue_test, so nothing it writes is mistaken for real work.
    run([PY, "scripts/smoke_test.py"])
    run([PY, "scripts/queue_selftest.py"])

    print("\n>>> Compare the split sha256 above against your local run. They must match.")
'''),

    # --- cell 6 (code): 5. Budget planner (spec 12.4) -- times the real hardware, prints the q
    ('code', r'''
# =============================================================================
# 5. Budget planner (spec 12.4) -- times the real hardware, prints the queue
#    and a projected finish time, saves results/queue_9h.json.
#
# A committed run cannot wait for a "go", so the plan is applied as printed.
# Read it in the log afterwards; RUN_SUMMARY.md records what actually ran.
# =============================================================================
import os
os.chdir(WORK_DIR)
if SMOKE:
    print("SMOKE = True -- skipped: no budget plan.")
else:
    run([PY, "scripts/plan_budget.py", "--hours", str(BUDGET_HOURS)])
'''),

    # --- cell 7 (code): 6. THE LONG RUN -- LR sweep -> E1 -> E7/E5/E6 -> E2 -> E4 -> upgrades.
    ('code', r'''
# =============================================================================
# 6. THE LONG RUN -- LR sweep -> E1 -> E7/E5/E6 -> E2 -> E4 -> upgrades.
#
# Expect 6.5-8 h. Output streams live so the committed-run log stays useful,
# and is teed to results/run_all.log.
#
# Wrapped in try/except so that a failure still reaches the packaging cell
# below: partial results are worth far more than none.
# =============================================================================
import os, traceback
os.chdir(WORK_DIR)
(WORK_DIR / "results").mkdir(parents=True, exist_ok=True)

RUN_OK = False
if SMOKE:
    print("SMOKE = True -- skipped: the queue does not run.")
else:
    try:
        run([PY, "scripts/run_all.py", "--mode", MODE],
            tee=str(WORK_DIR / "results" / "run_all.log"))
        RUN_OK = True
    except Exception:
        print("\n" + "!" * 70)
        print("!! The queue exited non-zero. Packaging whatever completed.")
        print("!" * 70)
        traceback.print_exc()
'''),

    # --- cell 8 (code): 7. Package the results -- this version's whole saved output.
    ('code', r'''
# =============================================================================
# 7. Package the results -- this version's whole saved output.
#
# Kaggle saves /kaggle/working and may cap it at ~500 files, so it gets exactly
# two, from scripts/kaggle_outputs.py pack:
#   results_light.zip  json/csv/md/png/... only, no checkpoints -- what Phase 9
#                      needs; written first, so a full disk cannot cost it
#   results.zip        everything, checkpoints included -- what a resume (Add
#                      Input) and a local E5/E6/E7 re-run need
# Anything else in /kaggle/working is removed. results/ itself, checkpoints and
# all, stays in the work dir under /kaggle/tmp, which is not saved.
# =============================================================================
import os
os.chdir(WORK_DIR)
RESULTS = WORK_DIR / "results"

if SMOKE:
    print(f"SMOKE = True -- skipped: the output is {OUTPUT_DIR / 'smoke_test.json'} alone.")
elif not RESULTS.exists() or not any(RESULTS.rglob("*")):
    print("results/ is empty -- nothing to package.")
else:
    # check=False: a failure must still reach the summary and census below.
    code = run([PY, "scripts/kaggle_outputs.py", "pack", "--results", str(RESULTS),
                "--out", str(OUTPUT_DIR)], check=False)
    if code != 0:
        print("\n" + "!" * 70)
        print(f"!! Packaging failed (exit {code}). The census below shows what was saved.")
        print("!" * 70)
'''),

    # --- cell 9 (code): 8. Final summary.
    ('code', r'''
# =============================================================================
# 8. Final summary.
# =============================================================================
import json
from pathlib import Path
RESULTS = WORK_DIR / "results"

if SMOKE:
    # Reached only if the smoke test passed: a failure stops the run in cell 4.
    s = json.loads((RESULTS / "smoke_test.json").read_text(encoding="utf-8"))
    checks = [c for r in s["results"] for c in r["checks"]]
    print(f"SMOKE run on {s['device']}: {sum(c['passed'] for c in checks)}/{len(checks)} checks "
          f"passed -- {'ALL PASS' if s['all_passed'] else 'FAILURES'}.")
    print("Set SMOKE = False in the CONFIG cell for the real run.")
else:
    for name, n_lines in [("RUN_SUMMARY.md", None), ("progress.log", 40)]:
        p = RESULTS / name
        print("=" * 78)
        print(f"  {name}")
        print("=" * 78)
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if n_lines and len(lines) > n_lines:
                print(f"... ({len(lines) - n_lines} earlier lines omitted)")
                lines = lines[-n_lines:]
            print("\n".join(lines))
        else:
            print(f"(not found -- {name} is written by scripts/run_all.py)")
        print()

    fails = RESULTS / "failures"
    if fails.exists():
        logs = sorted(fails.glob("*.log"))
        print(f"{len(logs)} failed run(s):")
        for f in logs:
            print(f"  - {f.name}")

# What this version saves: /kaggle/working, which Kaggle may cap at ~500 files.
# The exact count and each top-level folder's, then the repo clone's breakdown.
print("\n" + "=" * 78 + "\n  Output file census\n" + "=" * 78)
run([PY, "scripts/kaggle_outputs.py", "census",
     "--output", str(OUTPUT_DIR), "--work-dir", str(WORK_DIR)], check=False)
'''),

    # --- cell 10 (markdown): 📥 When you get back
    ('markdown', r'''
---

## 📥 When you get back

> After a **SMOKE** run there is nothing to download: read the pre-flight cell's log (or
> `smoke_test.json`, the output's one file), then set `SMOKE = False` and commit again.

> **Runs left unfinished** (failed, or dropped for time)? Attach this version's output (**Add
> Input** → *Notebooks* → this notebook → this version) and commit again. The pre-flight cell
> restores `results/` from it, and only the unfinished runs train.

1. **Download the archives** from the Output tab of the committed version. They are its only
   files: the run itself worked in `/kaggle/tmp`, which is not saved.
   - `results_light.zip` — tables, figures, metrics, logs. This is what Phase 9 needs.
   - `results.zip` — the same plus checkpoints, if you want to re-run E5/E6/E7 locally.

2. **Unzip into the local repo** so the directory layout matches:
   ```powershell
   Expand-Archive -Path results_light.zip -DestinationPath .\results -Force
   ```

3. **Check three things before trusting the numbers:**
   - the split sha256 from cell 4 matches your local `check_data.py` output;
   - `results/RUN_SUMMARY.md` lists which upgrades actually ran and which runs failed;
   - no run in `results/failures/` is one you needed.

4. **Run Phase 9 locally** with the coding agent: figures, `results/REPORT.md`, the README, and the
   LinkedIn drafts.

### ⚠️ What the write-up must say (spec 12.7)

9h mode trades statistical power for wall-clock. The report has to state plainly:

- **1 seed for most results** → differences under ~1 percentage point between models are **not** meaningful.
- **100 epochs, not 200** → ViT and MLP-Mixer usually keep improving with longer training, so this
  setup slightly **favours the CNNs**. Say so rather than concluding "Transformers lose".
- **E2 points are step-matched but short** (7,040 steps), so absolute E2 accuracies sit well below E1's.
- **Which upgrades actually ran**, taken from `RUN_SUMMARY.md`.
'''),

]


# ---------------------------------------------------------------------------
# Building and writing
# ---------------------------------------------------------------------------


def _source_lines(text: str) -> list[str]:
    """nbformat stores a cell's source as a list of lines, each keeping its newline."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def build() -> dict:
    cells = []
    for i, (kind, text) in enumerate(CELLS):
        if "'''" in text:
            raise ValueError(f"cell {i} contains ''' -- it cannot live in a raw string")
        text = text.strip("\n")
        if kind == "code":
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": _source_lines(text)})
        elif kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _source_lines(text)})
        else:
            raise ValueError(f"cell {i}: unknown cell type {kind!r}")
    return {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 5}


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def py_compile_cells(path: Path, python: str = sys.executable) -> list[tuple[int, bool, str]]:
    """Extract every code cell of the notebook at ``path`` and py_compile it.

    Returns ``(cell index, compiled ok, error text)`` per code cell. The cells
    are read back from the file on disk -- not from memory -- so this checks
    exactly what Kaggle will execute. ``python`` picks the interpreter whose
    parser is used, e.g. a 3.10 to match Kaggle's kernel.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = Path(tmp) / f"cell_{i:02d}.py"
            src.write_text("".join(cell["source"]), encoding="utf-8")
            proc = subprocess.run([python, "-m", "py_compile", str(src)],
                                  capture_output=True, text=True)
            results.append((i, proc.returncode == 0, (proc.stderr or proc.stdout).strip()))
    return results


def verify(path: Path, python: str = sys.executable) -> bool:
    version = subprocess.run([python, "-c", "import sys; print(sys.version.split()[0])"],
                             capture_output=True, text=True).stdout.strip() or "?"
    print(f"compiling every code cell, as extracted from the notebook file, with Python {version}:")
    ok = True
    for i, passed, err in py_compile_cells(path, python):
        print(f"  cell {i:2d}  py_compile {'OK' if passed else 'FAILED'}")
        if not passed:
            ok = False
            print("     " + err.replace("\n", "\n     "))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate and verify kaggle_runner.ipynb.")
    ap.add_argument("--check", action="store_true",
                    help="do not write; verify the notebook on disk is current and compiles")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter whose parser compiles the cells (default: this one); "
                         "e.g. a Python 3.10 to match Kaggle's kernel")
    args = ap.parse_args()

    expected = render(build())
    rel = NOTEBOOK.relative_to(ROOT)
    if args.check:
        current = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
        stale = current != expected
        print(f"{rel}: {'DIFFERS from the generator output -- regenerate it' if stale else 'matches the generator'}")
    else:
        NOTEBOOK.write_text(expected, encoding="utf-8", newline="\n")
        stale = False
        print(f"wrote {rel} ({len(CELLS)} cells)")

    compiled = verify(NOTEBOOK, args.python)
    print("ALL CODE CELLS COMPILE" if compiled else "SOME CODE CELLS DO NOT COMPILE")
    return 0 if compiled and not stale else 1


if __name__ == "__main__":
    sys.exit(main())

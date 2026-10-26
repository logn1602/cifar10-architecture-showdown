# Running the showdown on Kaggle (2× T4)

This laptop has no usable training GPU — an MX330 with 2 GB VRAM, where the largest models
(MLP-Mixer, ConvNeXt-mini) are unlikely to fit at the protocol's batch of 128; see the
[README](../README.md#hardware-why-training-happens-on-kaggle). So the plan follows spec §1.4 and
§12.6:

1. build and smoke-test everything **locally on CPU** (Phases 0–3);
2. run the real queue on **Kaggle with 2× T4** via `kaggle_runner.ipynb`;
3. download `results.zip` and do the write-up (Phase 9) locally.

> **The notebook is generated — don't edit `kaggle_runner.ipynb` by hand.** Edit the cells in
> `scripts/make_kaggle_notebook.py` and run it: it rewrites the notebook, then extracts every code
> cell and compiles it with `python -m py_compile`. `--check` verifies the committed notebook
> without writing. A hand-patched cell 7 once reached Kaggle with a SyntaxError; this is the guard.

Why 2× T4 and not 1: the Phase 0 projection puts the 9h-mode core queue at **~12 GPU-hours**.
Split across two T4s at ~90% parallel efficiency that is **~6.7 h wall-clock**, which fits an 8 h
budget with room for the first upgrade. On a single T4 the core alone overruns the window.

---

## 1. Get the repo onto Kaggle

Two options. **Cloning is the default** — the repo is public, so it needs no credentials and no
upload step.

### Option A — clone from GitHub (recommended)

Nothing to do: the notebook's CONFIG cell already points at the repo.

```python
SOURCE   = "git"
REPO_URL = "https://github.com/logn1602/cifar10-architecture-showdown.git"
```

Internet must be **On** (Settings → Internet) or the clone fails. `data/` is gitignored, so the
clone is small. Kaggle gets CIFAR-10 itself, from an attached copy or a fresh download (§2), and
regenerates the 45k/5k split from it, which is what makes the split fingerprint check in §5
meaningful.

### Option B — upload as a Kaggle Dataset

Use this if you want to run a local working copy that is ahead of what you have pushed, or if
Internet is off. Set `SOURCE = "dataset"` in the CONFIG cell, then:

Zip the repo *without* the venv, the data and the git history:

```powershell
# from the repo root
$exclude = @('.venv', 'data', '.git', 'results', '__pycache__')
$staging = "$env:TEMP\showdown-upload"
Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $staging | Out-Null

Get-ChildItem -Force | Where-Object { $exclude -notcontains $_.Name } |
    ForEach-Object { Copy-Item $_.FullName -Destination $staging -Recurse -Force }

Compress-Archive -Path "$staging\*" -DestinationPath "$env:USERPROFILE\Downloads\showdown.zip" -Force
Write-Host "wrote $env:USERPROFILE\Downloads\showdown.zip"
```

Then on kaggle.com: **Datasets → New Dataset → Upload** `showdown.zip`, give it a title, **Create**.

Leave `DATASET_SLUG = ""` in the notebook — it auto-detects your dataset by looking for
`CIFAR10_ARCHITECTURE_SHOWDOWN.md` under `/kaggle/input`. Set the slug explicitly only if you have
more than one copy attached.

In the notebook editor, attach it with **Add Input → Datasets → your dataset**.

> Excluding `data/` is deliberate: Kaggle gets CIFAR-10 itself (an attached copy or the download,
> §2), and the 45k/5k split is regenerated from scratch. The sha256 fingerprints printed by
> `check_data.py` are how you verify the regenerated split matches your local one — see §5 below.

---

## 2. Notebook settings

**Notebook → Settings** (right sidebar):

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 × 2** |
| Internet | **On** |
| Environment | **Always use latest environment** |
| Language | Python |

Internet must be on for `pip install`, the CIFAR-10 download, and the E4 pretrained weights.

### Optional: attach CIFAR-10 instead of downloading it

The download is 170 MB from cs.toronto.edu, and it can be slow: the local test took 15 minutes.
To skip it, attach a dataset that holds the official python version: **Add Input → Datasets**,
search "CIFAR-10 Python" (for example `pankrzysiu/cifar10-python`). Any dataset works if it has
`cifar-10-batches-py/` or `cifar-10-python.tar.gz` somewhere inside it.

The pre-flight cell runs `scripts/kaggle_inputs.py cifar`, in the smoke run too. It copies the
attached files into `data/` and checks each one against torchvision's md5 checksums, so they are
byte-identical to a download and the split fingerprints do not change. If a copy fails the check,
the log names the failing file and the notebook downloads instead. The log always says which path
it used:

```
CIFAR-10: using the attached copy at /kaggle/input/cifar10-python/cifar-10-batches-py
          -> /kaggle/tmp/cifar10-architecture-showdown/data/cifar-10-batches-py (every file md5-verified against torchvision)
```

Internet still has to be on, for `pip install` and the E4 weights.

A notebook pinned to its original environment can be handed a 2023 image (Python 3.10, PyTorch
2.0) — that is what the first runs got. The code is kept compatible with PyTorch 2.0 and Python
3.10–3.13, so a pinned run still works, and the environment-probe cell prints a warning when it
sees PyTorch older than 2.1; but the latest environment has a current PyTorch, so prefer it.

**Check your quota first.** Kaggle gives roughly 30 GPU-hours per week and caps a single session at
around 12 h (both have changed over time — read the current numbers on the session sidebar). The
default `BUDGET_HOURS = 8` is chosen to sit under the session cap with slack.

---

## 3. Start a committed run

**Save Version → Save & Run All (Commit)**.

This is the important step. A committed run executes on Kaggle's infrastructure with your browser
closed. The interactive **Run All** button dies when your session disconnects — which it will,
somewhere around hour two of an eight-hour job.

You can close the tab immediately. Progress is visible under the notebook's **Versions** tab.

### Before that, locally: rehearse on Kaggle's pinned versions (~20–30 min, CPU)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rehearse.ps1            # add -WithQueue for E5-E7
```

Builds a fresh Python 3.10 venv with torch 2.0.0 / torchvision 0.15.1 (CPU builds) and the
rest of Kaggle's 2023 image at pinned versions. It then runs the notebook's own `pip install`
line exactly as the notebook does, and fails if that install moves torch, torchvision or numpy. It
compiles every file with Python 3.10, resolves the GradScaler fallback on torch 2.0, and runs
`smoke_test.py --tiny` (with `-WithQueue`, `queue_selftest.py` too). The record goes to
`results/rehearsal.json`. It cannot test CUDA; that is what the smoke run below is for.

The pins are a reconstruction of the image. The notebook's install cell prints the versions
Kaggle really ends up with, so compare that list with the pins in `scripts/rehearse.ps1`
after a run and update them if they differ.

### Optional, recommended first: a smoke run (~10 min)

Set `SMOKE = True` in the CONFIG cell and commit. The notebook probes the GPU, fetches the repo,
installs the extras, runs **only** `scripts/smoke_test.py`, and every later cell skips itself. The
smoke test pushes each of the six models through the real training loop on the T4 (fp16 autocast,
the GradScaler, DataLoader workers, interrupted and resumed, plus the E2/E3/E4 code paths). It is
the cheapest way to find out the image can run the code before spending 8 GPU-hours. A failure
stops the run with the error in the pre-flight cell's log.

Once it passes, set `SMOKE = False` and commit again for the real run.

### What a run saves

Kaggle saves `/kaggle/working` as the version's output, and may cap it at about 500 files
(community reports; not documented). A full run makes far more than that, so the notebook works
somewhere else. The repo clone, `data/`, `results/` and every checkpoint live in
`/kaggle/tmp/cifar10-architecture-showdown`, which is not saved. The packaging cell leaves exactly
two files in `/kaggle/working`: `results.zip` and `results_light.zip` (§5). A smoke run leaves
only `smoke_test.json`.

Every run, smoke or real, ends with a census. The last cell prints the exact file count in
`/kaggle/working` and each top-level folder's count. It then prints the same breakdown for the
work dir, which is what the output would hold if the clone still lived inside it. It warns past
400 files.

---

## 4. Expected timeline

| Stage | Approx. wall-clock on 2× T4 |
|---|---|
| Setup, install, data checks, Phase 2 model gate, Phase 3 smoke test + queue self-test | 10–15 min |
| Budget planner | 3–5 min |
| LR sweep (18 runs × 10 epochs) | ~1.0 h |
| E1 main showdown (6 × 100 epochs) | ~2.9 h |
| E5 / E6 / E7 evaluation | ~0.1 h |
| E2 data-hunger (18 runs × 7,040 steps) | ~2.1 h |
| E4 pretrained (2 × 5 epochs @ 224px) | ~0.6 h |
| **Core total** | **~6.7 h** |
| Upgrades, budget permitting | up to ~1.3 h |

These come from `scripts/estimate_compute.py` and are **estimates**. The in-notebook planner
(`scripts/plan_budget.py`) times every model on the real T4, simulates the queue across both GPUs,
and prints the per-run plan and projected finish before the long cell starts. A committed run
cannot wait for a "go", so the plan is applied as printed: read it in the log afterwards, and
`RUN_SUMMARY.md` records which upgrades actually made it in.

---

## 5. When it finishes

Download from the committed version's **Output** tab. These two archives are the whole output
(see "What a run saves" in §3):

- **`results_light.zip`** — JSON, CSV, Markdown, PNG. This is what Phase 9 needs.
- **`results.zip`** — the same plus checkpoints, if you want to re-run E5/E6/E7 locally. It is
  also what a resume restores from.

```powershell
Expand-Archive -Path results_light.zip -DestinationPath .\results -Force
```

Then verify three things before trusting any number:

1. **The split matches.** Compare the `train_sha256` / `val_sha256` printed by the notebook's
   pre-flight cell against your local `results/tables/data_report.md`. If they differ, the Kaggle
   run used a different train/val split and the val-selected checkpoints are not comparable to
   anything local.
2. **`results/RUN_SUMMARY.md`** — which runs completed, which failed, which upgrades made it in.
3. **`results/failures/`** — empty is ideal; if not, check whether anything you need is in there.

The archive also carries Kaggle's copies of files that are committed locally, and `-Force`
overwrites them. `git status` lists what changed. Restore the FLOP-bearing ones, because the
Kaggle copies may say `n/a`:

```powershell
git checkout -- results/model_checks.json results/tables/model_summary.md
```

Then build the write-up. `make_report.py` reads only the unzipped JSON/CSV and recounts FLOPs
locally, so it needs no GPU and no checkpoints:

```powershell
python scripts/make_report.py     # -> results/REPORT.md, tables/*, figures 1-7 and 15
```

It reports any experiment or figure that is missing rather than failing, so it is safe to run on
a queue that did not finish. `python scripts/report_selftest.py` checks it end to end on
synthetic results.

### If the queue stopped before the end

Resuming needs the earlier version's output, so check its **Output** tab first. A version that
ended normally keeps its output even if runs failed or upgrades were dropped for time. So does
one whose queue crashed, because the packaging cell still runs. A version that Kaggle stopped at
its session limit may keep nothing (users report an empty Output tab after the 12-hour cutoff),
which is one more reason `BUDGET_HOURS` stays well under it. To pick up where a version stopped:

1. In the notebook editor, **Add Input → Notebooks → this notebook**, and check that the attached
   version is the one that stopped. Its output appears under `/kaggle/input/`.
2. Commit again (**Save & Run All**) with `SMOKE = False`.

The pre-flight cell runs `scripts/kaggle_inputs.py restore`. It looks for the earlier `results/`
in the attached output: `results.zip` and `results_light.zip`. Versions from before the run
moved to `/kaggle/tmp` also saved the results folder itself, and that works too. It copies the
source with the most finished runs into `results/` in the work dir, taking `results.zip` on a tie
because the packaging cell always writes it whole. Then it prints what the queue will skip:

```
Resume: found /kaggle/input/<notebook>/results.zip (zip, <n> finished runs, with checkpoints)
Resume: restored <files> files from /kaggle/input/<notebook>/results.zip into /kaggle/tmp/cifar10-architecture-showdown/results
Resume: <n> of the 113 jobs in the 9h queue (every upgrade included) are already complete; the queue will skip them.
```

From there nothing new happens: the queue skips finished runs by its usual test (`metrics.json`
with `status: completed`, or `done.json` for E5–E7), runs that stopped part-way resume from
`last.pt`, and diverged runs are not retried. The planner budgets only the work that is left.
Failure logs and scratch are not restored, so failed runs are retried. Restore from the full
output, not `results_light.zip` alone: it has no checkpoints, so partial runs would restart and
E5–E7 could not evaluate the E1 runs (the log warns if so). With nothing attached the cell prints
`starting from scratch`. `python scripts/kaggle_inputs_selftest.py` covers every case on CPU, and
`python scripts/kaggle_outputs_selftest.py` runs the whole round trip: a 600-file results tree is
packed into the two zips, restored from them, and the queue's own dry run checks what it would skip.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: No GPU!` | Accelerator set to None/CPU | Settings → Accelerator → GPU T4 × 2, then restart the session |
| `[ WARN ] FLOPs counted … not counted, recorded as null` | This PyTorch build's `FlopCounterMode` cannot count the models (it failed for all six on Kaggle's Python 3.10 image) | Nothing — it is a warning, not a gate, and the line says why. FLOPs depend only on the architecture and input size, so the local counts in `results/tables/model_summary.md` apply; the Kaggle copy of that file will show `n/a` |
| "Only 1 GPU allocated" warning | Kaggle gave a single T4 (quota or availability) | Restart the session; if it persists, the core will overrun 8 h — lower `BUDGET_HOURS` and accept fewer upgrades |
| `pip install` fails, no network | Internet toggle off | Settings → Internet → On. Note this also blocks the CIFAR-10 download |
| `CIFAR-10: skipping the attached copy at … fails its md5 check` | The attached dataset is not the official python version, or is incomplete | Nothing — the notebook downloads instead. Attach a different copy next time to skip the download |
| `torch lost CUDA after the pip install` | Something pulled a CPU torch wheel | Restart the session. Never add `torch` to the install list — Kaggle ships its own CUDA build |
| Session ends mid-queue | Hit the per-session time limit | Kaggle may keep no output from a version stopped at its limit, so keep `BUDGET_HOURS` well under it. If that version's Output tab does list files, attach it and commit again (§5, "If the queue stopped before the end"): finished runs are skipped and partial runs resume from `last.pt` |
| `OSError: No space left on device` | Checkpoints filled the work dir's disk (`/kaggle/tmp`; the probe cell prints its free space) | LR-sweep runs save no checkpoints and other runs keep only `best.pt`; if it still fills, drop `last.pt` retention or cut E1 seeds. `results_light.zip` is written before `results.zip`, so Phase 9's archive survives a disk that fills during packaging |
| Census: `WARNING: <n> files in /kaggle/working` | Something other than the two zips is in the output | The packaging cell removes strays, so packaging did not finish: read its log. The census rows say what is there |
| `PermissionError` creating `/kaggle/tmp` in the probe cell | The image does not allow new folders under `/kaggle` | Set `SCRATCH = Path("/tmp")` in the CONFIG cell. Any writable path outside `/kaggle/working` works; the probe prints its free space |
| `FileNotFoundError: ...MD not found under /kaggle/input` | Dataset not attached, or zipped with an extra top-level folder | Add Input → Datasets; or set `DATASET_SLUG` explicitly |
| Notebook "succeeded" but `results/` is empty | A `!`-magic cell failed silently | Already handled — every step runs through `subprocess` with an exit-code check. If you add cells, keep that pattern |
| Queue dies on one model, rest never run | Unhandled exception | By spec §12.5 each run is wrapped in try/except and failures are logged to `results/failures/` without stopping the queue. If the whole process died, that is a bug in `run_all.py` |

---

## 7. What this notebook deliberately does not do

- **It never installs `torch`.** See the table above.
- **It never asks a question.** Spec §12.5 suspends every approval gate during the unattended run.
- **It never auto-tweaks hyperparameters to rescue a diverging run.** A NaN is logged as a failure
  and reported honestly — silently lowering an LR would corrupt the fairness protocol that the
  entire comparison rests on.

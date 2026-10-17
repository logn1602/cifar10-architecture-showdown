# Running the showdown on Kaggle (2× T4)

This laptop has no usable training GPU — an MX330 with 2 GB VRAM, where ResNet-18 at batch 128
would OOM before the first epoch finished. So the plan follows spec §1.4 and §12.6:

1. build and smoke-test everything **locally on CPU** (Phases 0–3);
2. run the real queue on **Kaggle with 2× T4** via `kaggle_runner.ipynb`;
3. download `results.zip` and do the write-up (Phase 9) locally.

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
clone is small and CIFAR-10 is downloaded fresh on Kaggle — which is what makes the split
fingerprint check in §5 meaningful.

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

> Excluding `data/` is deliberate: Kaggle re-downloads CIFAR-10 itself, and the 45k/5k split is
> regenerated from scratch. The sha256 fingerprints printed by `check_data.py` are how you verify
> the regenerated split matches your local one — see §5 below.

---

## 2. Notebook settings

**Notebook → Settings** (right sidebar):

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 × 2** |
| Internet | **On** |
| Language | Python |

Internet must be on for `pip install`, the CIFAR-10 download, and the E4 pretrained weights.

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

---

## 4. Expected timeline

| Stage | Approx. wall-clock on 2× T4 |
|---|---|
| Setup, install, data checks | 5–10 min |
| Budget planner | 3–5 min |
| LR sweep (18 runs × 10 epochs) | ~1.0 h |
| E1 main showdown (6 × 100 epochs) | ~2.9 h |
| E5 / E6 / E7 evaluation | ~0.1 h |
| E2 data-hunger (18 runs × 7,040 steps) | ~2.1 h |
| E4 pretrained (2 × 5 epochs @ 224px) | ~0.6 h |
| **Core total** | **~6.7 h** |
| Upgrades, budget permitting | up to ~1.3 h |

These come from `scripts/estimate_compute.py` and are **estimates**. The in-notebook planner
(`scripts/plan_budget.py`, Phase 3) times the real hardware and replaces them.

---

## 5. When it finishes

Download from the committed version's **Output** tab:

- **`results_light.zip`** — JSON, CSV, Markdown, PNG. This is what Phase 9 needs.
- **`results.zip`** — the same plus checkpoints, if you want to re-run E5/E6/E7 locally.

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

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: No GPU!` | Accelerator set to None/CPU | Settings → Accelerator → GPU T4 × 2, then restart the session |
| "Only 1 GPU allocated" warning | Kaggle gave a single T4 (quota or availability) | Restart the session; if it persists, the core will overrun 8 h — lower `BUDGET_HOURS` and accept fewer upgrades |
| `pip install` fails, no network | Internet toggle off | Settings → Internet → On. Note this also blocks the CIFAR-10 download |
| `torch lost CUDA after the pip install` | Something pulled a CPU torch wheel | Restart the session. Never add `torch` to the install list — Kaggle ships its own CUDA build |
| Session ends mid-queue | Hit the per-session time limit | The queue is resumable: re-run the notebook. Runs with an existing `metrics.json` are skipped and partial runs resume from `last.pt` — but only if `/kaggle/working` persisted, so prefer lowering `BUDGET_HOURS` |
| `OSError: No space left on device` | Checkpoints filled `/kaggle/working` | LR-sweep runs save no checkpoints and other runs keep only `best.pt`; if it still fills, drop `last.pt` retention or cut E1 seeds |
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

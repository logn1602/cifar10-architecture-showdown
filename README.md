# 🖼️ Evolution of Vision — CIFAR-10 Architecture Showdown

> **Status: work in progress.** Phases 0 and 1 are complete; Phases 2–9 are not built yet.
> **There are no results in this repo.** Every accuracy number in the final write-up will be
> generated from `results/runs/` — none are hand-written, and none exist yet.

Six neural network architectures spanning roughly 35 years of computer vision, trained **from
scratch on CIFAR-10 under identical conditions**, to answer one question properly: how much of a
modern architecture's advantage is the architecture, and how much is just data?

The headline experiment starves each model of data (5% → 100% of the training set at matched
optimization steps) to test the standard claim that Transformers are not *worse* than CNNs on small
data, they are *hungrier* — because they carry almost no inductive bias about images and have to
learn from data what a convolution gets for free.

Full spec: [`CIFAR10_ARCHITECTURE_SHOWDOWN.md`](CIFAR10_ARCHITECTURE_SHOWDOWN.md).

---

## The six contestants

| Model | Year | Key idea | Inductive bias | ~Params |
|---|---|---|---|---|
| **MLP** | ~1986 | Flatten the image, stack dense layers | **None.** Every pixel is just a number; shifting the image one pixel to the left is a brand-new input | 3.8 M |
| **Simple CNN** | ~1998 | Small 3×3 convolutions, pool, repeat | Locality, translation equivariance, weight sharing | 1.1 M |
| **ResNet-18** | 2015 | Residual connections make depth trainable | CNN bias + identity shortcuts that fix vanishing gradients | 11.2 M |
| **ViT** | 2020 | No convolutions — just self-attention over 4×4 patches | **Almost none.** Only the patch grid; it must *learn* that neighbouring patches are related | ~8.3 M |
| **MLP-Mixer** | 2021 | Do we even need attention? Alternate token-mixing and channel-mixing MLPs | Patch grid only, and mixing is *fixed per position* rather than input-dependent | ~9.7 M |
| **ConvNeXt-mini** | 2022 | A CNN rebuilt with Transformer-era design choices | Convolution's bias, plus big kernels, LayerNorm, GELU, inverted bottlenecks | ~8.5 M |

MLP, Simple CNN, ViT and MLP-Mixer are written by hand (that is the point of the project).
ResNet-18 and ConvNeXt-mini are hand-written too; `timm` is used only as a parameter-count
cross-check.

---

## The fairness protocol

Every model gets **identical** treatment except the architecture itself. The only per-model tuning
allowed is the learning rate, chosen by a sweep on the **validation** set.

| Setting | Value |
|---|---|
| Train / val / test | 45,000 / 5,000 (stratified, seed 0) / official 10,000 |
| Normalization | mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)` |
| Augmentation | `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` |
| Optimizer | AdamW, β `(0.9, 0.999)`, weight decay 0.05 (not on biases, norms, pos-embeds, cls token) |
| Schedule | 5-epoch linear warmup → cosine decay to 1e-6, stepped per optimizer step |
| Loss | Cross-entropy, label smoothing 0.1 |
| Precision | Mixed (bf16 where supported, else fp16 + GradScaler) |
| Gradient clipping | Max norm 1.0 |
| Checkpoint selection | Best **val** accuracy |
| Test set | Touched **once** per run, at the very end |

The test set is never used for any decision — not for the learning rate, not for epoch count, not
for early stopping. See [`configs/base.yaml`](configs/base.yaml), which encodes this verbatim.

---

## Hardware: why training happens on Kaggle

| | |
|---|---|
| CPU | Intel Core i5-1135G7, 4C/8T — **121.7 GFLOP/s** measured (fp32, 2048² matmul) |
| RAM | 7.7 GB (often <1 GB free — see the note below) |
| GPU | **NVIDIA MX330, 2 GB VRAM** — not viable |
| Local PyTorch | 2.14.1+cpu / torchvision 0.29.1+cpu |
| DataLoader | **8,350 img/s at `num_workers=4`** (5.4 s per 45k epoch) |

An MX330 has 2 GB of VRAM. ResNet-18 at batch 128 with AMP needs roughly 2.5–3 GB for activations
alone on 32×32 input, and ViT/ConvNeXt need more — so the protocol's batch size cannot run here at
all, and changing the batch size would break the fairness protocol.

So the split is: **build and verify locally on CPU, train on Kaggle with 2× T4.**

`scripts/estimate_compute.py` puts the 9h-mode core queue at **~12 GPU-hours**, which is **~6.7 h
wall-clock on 2× T4** and does *not* fit an 8 h budget on a single T4. Full mode is a **multi-day**
job (>100 GPU-hours) — see `results/tables/compute_estimate.md`.

See [`docs/KAGGLE.md`](docs/KAGGLE.md) for the step-by-step, and
[`kaggle_runner.ipynb`](kaggle_runner.ipynb) for the runner notebook.

> **Memory note.** `num_workers=4` is the fastest setting *when RAM is free*, but on Windows each
> worker is a spawned process that gets its own copy of the 153 MB in-memory CIFAR-10 array. With
> under ~1 GB free this machine thrashes hard (observed: 7,300 page-faults/sec, 9 s of CPU progress
> per 2 min of wall-clock). If a script crawls, re-run it with `--num-workers 0` — single-process
> iteration still manages 2,873 img/s and never pages. This is a local-only concern; Kaggle has
> ~30 GB of RAM.

---

## Reproduce what exists today

```powershell
# 0. Clone
git clone https://github.com/logn1602/cifar10-architecture-showdown.git
cd cifar10-architecture-showdown

# 1. Environment (Python 3.11, CPU-only PyTorch)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Phase 0 -- environment, hardware, compute budget
python scripts/env_check.py                 # -> results/tables/environment.md
python scripts/estimate_compute.py --device t4 --num-gpus 2 --mode 9h
python scripts/bench_dataloader.py          # -> results/tables/dataloader_bench.md

# 3. Phase 1 -- data pipeline
python -m src.data                          # downloads CIFAR-10, self-checks the split
python scripts/check_data.py                # -> results/tables/data_report.md + 3 figures
```

`scripts/check_data.py` is a gate, not a report: it exits non-zero if any hard check fails.

---

## Repo layout

```
cifar10-architecture-showdown/
├── CIFAR10_ARCHITECTURE_SHOWDOWN.md   # the full spec
├── configs/
│   ├── base.yaml                      # ✅ the fairness protocol, 9h-mode sizes, figure colours
│   ├── compute_reference.yaml         # ✅ reference throughputs for the budget estimate
│   ├── models/                        # ⬜ Phase 2
│   └── experiments/                   # ⬜ Phase 3+
├── src/
│   ├── utils.py                       # ✅ seeding, config, logging, hashing
│   ├── data.py                        # ✅ splits, transforms, nested subsets, Mixup/CutMix
│   ├── models/                        # ⬜ Phase 2 -- the six architectures
│   ├── train.py  evaluate.py  profile.py   # ⬜ Phase 3
│   ├── corruptions.py                 # ⬜ Phase 7 (E5/E6)
│   └── interpret/                     # ⬜ Phase 8 (E7)
├── scripts/
│   ├── env_check.py                   # ✅ Phase 0
│   ├── estimate_compute.py            # ✅ Phase 0
│   ├── bench_dataloader.py            # ✅ Phase 0
│   ├── check_data.py                  # ✅ Phase 1
│   ├── smoke_test.py  lr_sweep.py  plan_budget.py   # ⬜ Phase 3
│   ├── run_experiment.py  run_all.py  # ⬜ Phase 3+
│   └── make_report.py                 # ⬜ Phase 9
├── kaggle_runner.ipynb                # ✅ the unattended 2×T4 runner
├── docs/KAGGLE.md                     # ✅ how to run it
└── results/                           # tables + figures (generated, never hand-edited)
```

---

## Notes on the data pipeline

Three decisions in `src/data.py` that the rest of the project depends on:

- **The 45k/5k split is regenerated, not shipped.** `data/` is gitignored, so Kaggle rebuilds the
  split itself. The construction is pure numpy with per-class `RandomState(seed + c)` — no sklearn,
  no `torch.randperm` — and every cached split carries a **sha256 fingerprint** so you can prove
  the Kaggle split is bit-identical to the local one. If it isn't, "val accuracy" means two
  different things on the two machines.
- **Val never sees train augmentation.** `CIFAR10Subset` wraps one transform-free base dataset so
  train and val hold genuinely different pipelines. The common `random_split` idiom cannot do this
  and quietly applies random crops and flips to your model-selection set.
- **E2 subsets are nested.** The 5% set is a strict subset of the 10% set, and so on, so the
  data-hunger curve varies only in *how much* data — not *which images*.

---

## Caveats (which will also appear in the final report)

9h mode trades statistical power for wall-clock time, and the write-up has to say so:

- **1 seed for most results.** Differences smaller than ~1 percentage point between models are
  **not** meaningful.
- **100 epochs, not 200.** ViT and MLP-Mixer typically keep improving with longer training, so this
  setup slightly **favours the CNNs**.
- **E2 is step-matched but short** (7,040 steps), so absolute E2 accuracies sit below E1's.
- Everything here is 32×32 images, small models, one dataset. None of it generalises to ImageNet
  without checking.

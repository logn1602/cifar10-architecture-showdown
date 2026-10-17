# 🖼️ Evolution of Vision — CIFAR-10 Architecture Showdown

> **For the coding agent:** This file is the full project spec. Read it top to bottom before writing any code.
> Build it **phase by phase**, in order. After each phase: run its checks, commit, and give me a short summary.
> **Never change the fairness protocol (Section 3) without asking me first.**
> Ask before starting any run expected to take longer than ~30 minutes.
>
> **Two modes:**
> - **Full mode** (Sections 3–9): no compute shortcuts, multi-day GPU job.
> - **⏱️ 9-Hour Unattended Mode** (Section 12): same 6 models and every experiment, trimmed seeds/epochs, runs the whole queue while I'm away. **If I say "9-hour mode", Section 12 overrides the approval gates and run sizes elsewhere in this file.**

---

## 0. TL;DR

Train 6 neural network architectures — from a 1980s-style MLP to a 2022 ConvNeXt — **from scratch on CIFAR-10 under identical conditions**, then answer:

1. Which architecture wins, and at what cost (params, FLOPs, time, memory)?
2. **How data-hungry is each one?** (Do Transformers lose on small data — and why?)
3. Do stronger augmentations help Transformers more than CNNs?
4. What happens with pretraining?
5. How robust is each model to noise, blur and scrambled images?
6. **How does each architecture "see"?** (feature maps, Grad-CAM, attention maps)

No shortcuts on compute: full-length training, multiple seeds, a learning-rate sweep for every model, and mean ± std on every headline number.

---

## 1. Environment

### 1.1 My setup
- OS: **Windows**, shell: **PowerShell**
- GPU: **unknown — detect it in Phase 0** and report what's available.
- I also have Azure for Students; Kaggle (free GPU, ~30 h/week) and Colab are fallbacks.

### 1.2 Requirements
- Python 3.11, PyTorch 2.x with CUDA (if a GPU exists)
- Use a virtual environment: `python -m venv .venv` → `.\.venv\Scripts\Activate.ps1`
- `requirements.txt` (pin exact versions after install with `pip freeze`):
  ```
  torch
  torchvision
  timm
  numpy
  pandas
  matplotlib
  seaborn
  scikit-learn
  pyyaml
  tqdm
  tensorboard
  grad-cam          # pytorch-grad-cam
  umap-learn
  ```

### 1.3 Windows gotchas (handle these)
- Every script with a DataLoader using `num_workers > 0` **must** have an `if __name__ == "__main__":` guard.
- `torch.compile` is unreliable on Windows → keep it **off by default** (config flag `compile: false`).
- Use `pathlib.Path` everywhere, never hard-coded `/` or `\` paths.
- Use `pin_memory=True`, `persistent_workers=True`, `num_workers=4` (tune in Phase 0).

### 1.4 Remote GPU option
If no local CUDA GPU: still build and smoke-test everything locally on CPU, then generate a `kaggle_runner.ipynb` (or Colab equivalent) that clones the repo and runs `scripts/run_all.py`. Results/checkpoints must be downloadable as one zip.

---

## 2. Repository structure

```
cifar10-architecture-showdown/
├── README.md                   # final write-up (Phase 9)
├── requirements.txt
├── configs/
│   ├── base.yaml               # shared training protocol (Section 3)
│   ├── models/                 # one yaml per model (Section 5)
│   │   ├── mlp.yaml
│   │   ├── simple_cnn.yaml
│   │   ├── resnet18.yaml
│   │   ├── vit.yaml
│   │   ├── mlp_mixer.yaml
│   │   └── convnext_mini.yaml
│   └── experiments/            # E1–E7 (Section 6)
├── src/
│   ├── data.py                 # datasets, splits, augmentation, subsets
│   ├── models/
│   │   ├── __init__.py         # build_model(name, cfg) registry
│   │   ├── mlp.py
│   │   ├── simple_cnn.py
│   │   ├── resnet.py
│   │   ├── vit.py
│   │   ├── mlp_mixer.py
│   │   └── convnext.py
│   ├── train.py                # the ONE shared training loop
│   ├── evaluate.py             # metrics, per-class acc, confusion matrix, ECE
│   ├── profile.py              # params, FLOPs, throughput, peak memory
│   ├── corruptions.py          # noise/blur/contrast + patch shuffle
│   ├── interpret/
│   │   ├── feature_maps.py
│   │   ├── gradcam.py
│   │   ├── attention.py        # attention maps + attention rollout
│   │   └── embeddings.py       # t-SNE / UMAP of penultimate features
│   └── utils.py                # seeding, logging, checkpointing, timers
├── scripts/
│   ├── smoke_test.py
│   ├── lr_sweep.py
│   ├── run_experiment.py       # python scripts/run_experiment.py --exp E1
│   ├── run_all.py
│   └── make_report.py          # builds results/REPORT.md + all figures
├── results/
│   ├── runs/<exp>/<model>/<seed>/   # metrics.json, history.csv, best.pt, config.yaml
│   ├── figures/
│   ├── tables/
│   └── REPORT.md
└── notebooks/
    └── explore.ipynb           # optional playground
```

---

## 3. The fairness protocol (do not change without asking)

Every model gets **identical** treatment except the architecture itself.

| Setting | Value |
|---|---|
| Train / val / test | 45,000 / 5,000 (stratified, fixed seed 0) / official 10,000 test |
| Normalization | mean `(0.4914, 0.4822, 0.4465)`, std `(0.2470, 0.2435, 0.2616)` |
| Augmentation ("standard") | `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()` |
| Optimizer | AdamW, betas (0.9, 0.999) |
| Weight decay | 0.05 (not applied to biases, norm layers, pos-embeddings, cls token) |
| LR schedule | linear warmup 5 epochs → cosine decay to 1e-6 |
| Learning rate | **chosen per model by the LR sweep (Phase 3)** — the only per-model tuning allowed |
| Epochs | **200** |
| Batch size | 128 |
| Loss | cross-entropy, label smoothing 0.1 |
| Precision | mixed precision (`torch.autocast` bf16 if supported, else fp16 + GradScaler) |
| Gradient clipping | max norm 1.0 |
| Seeds | **3 seeds (0, 1, 2)** for every headline result |
| Checkpoint selection | best **val** accuracy; test set touched **only once** per run, at the end |
| Determinism | seed python/numpy/torch/cuda; `cudnn.benchmark = True` is fine (note it in README) |

Rules:
- The test set is **never** used for any decision (LR, epochs, early stopping, nothing).
- Report test accuracy as **mean ± std over 3 seeds**.
- If any model diverges (NaN loss), log it, stop that run, and tell me — don't silently patch hyperparameters.

---

## 4. Shared training loop (`src/train.py`)

One function, used by every experiment:

```python
def train(model_name, model_cfg, train_cfg, data_cfg, seed, out_dir) -> dict:
    ...
```

It must:
- build the model via the registry, move to device, optional `channels_last` for conv models
- build param groups (decay / no-decay)
- train with AMP, grad clipping, warmup + cosine schedule (per-step, not per-epoch)
- each epoch log: train loss, train acc, val loss, val acc, LR, epoch wall time, images/sec
- save `history.csv`, `best.pt` (best val acc), `last.pt`, resolved `config.yaml`
- at the end: load `best.pt`, evaluate on test once, write `metrics.json`
- write to TensorBoard under `results/tb/<exp>/<model>/<seed>`
- support `--resume` from `last.pt` (long runs will get interrupted)
- support a `max_steps` override (used by the data-hunger experiment)
- print a clean one-line summary at the end: model, seed, best val, test acc, total time

`metrics.json` must contain at least:
```json
{
  "model": "vit", "seed": 0, "experiment": "E1",
  "best_val_acc": 0.0, "best_epoch": 0, "test_acc": 0.0, "test_loss": 0.0,
  "per_class_acc": {}, "ece": 0.0,
  "params": 0, "flops": 0, "train_time_sec": 0.0, "sec_per_epoch": 0.0,
  "train_imgs_per_sec": 0.0, "infer_imgs_per_sec": 0.0, "peak_mem_mb": 0.0,
  "lr": 0.0, "train_fraction": 1.0, "augmentation": "standard"
}
```

---

## 5. The six contestants

Implement **MLP, Simple CNN, ViT and MLP-Mixer by hand** (that's the learning). ResNet-18 and ConvNeXt-mini should also be hand-written; `timm` versions may be used only as a correctness cross-check (param counts should roughly match).

Each model file starts with a docstring: year introduced, the key idea, and the inductive bias it brings. These docstrings feed the README.

### 5.1 MLP — "no idea images have structure" (baseline)
- Flatten 3×32×32 = 3072 → 1024 → 512 → 256 → 10
- Each hidden layer: Linear → BatchNorm1d → GELU → Dropout(0.2)
- ~3.8M params
- Inductive bias: **none**. Every pixel is just a number; shifting the image by 1 pixel is a whole new input.

### 5.2 Simple CNN — "nearby pixels matter" (LeNet/VGG era)
- 3 stages, channels [64, 128, 256]
- Each stage: (Conv3×3 → BN → ReLU) × 2 → MaxPool2×2
- Head: global average pool → Dropout(0.3) → Linear(256, 10)
- ~1.1M params
- Inductive bias: **locality + translation equivariance + weight sharing**.

### 5.3 ResNet-18 — "go deep with skip connections" (2015)
- CIFAR variant: stem = Conv3×3(64), stride 1, **no max-pool**
- BasicBlocks [2, 2, 2, 2], channels [64, 128, 256, 512], downsample with stride-2 conv + 1×1 projection shortcut
- GAP → Linear(512, 10)
- ~11.2M params
- Key idea: residual connections fix vanishing gradients, enabling depth.

### 5.4 Vision Transformer (ViT) — "no convolutions, just attention" (2020)
- Patch size **4** → 8×8 = **64 patch tokens** + 1 cls token
- Patch embedding: Conv2d(3, 384, kernel=4, stride=4) (equivalent to linear on flattened patches)
- Learned positional embedding (65 × 384), cls token
- **Pre-norm** transformer blocks × **7**: LN → Multi-Head Self-Attention (6 heads, head dim 64) → residual → LN → MLP (384 → 768 → 384, GELU) → residual
- Stochastic depth (drop path) 0.1, linearly increasing over depth
- Final LN → cls token → Linear(384, 10)
- ~8–9M params
- **Write the attention by hand** (`q, k, v` projections, scaled dot-product, softmax, heads). Also store the attention weights of every layer when `return_attn=True` (needed for Phase 8).
- Inductive bias: **almost none** — only the patch grid. It has to *learn* that nearby patches are related.

### 5.5 MLP-Mixer — "do we even need attention?" (2021)
- Same patching as ViT: patch 4 → 64 tokens, hidden dim 384, **no cls token**
- 8 Mixer blocks, each:
  - **Token-mixing MLP** (mixes across the 64 patches): LN → transpose → MLP(64 → 192 → 64, GELU) → transpose → residual
  - **Channel-mixing MLP** (mixes within each patch): LN → MLP(384 → 1536 → 384, GELU) → residual
- Drop path 0.1
- Final LN → global average pool over tokens → Linear(384, 10)
- ~9–10M params
- Inductive bias: patch grid only, but mixing is **fixed per position** (unlike attention, which is input-dependent).

### 5.6 ConvNeXt-mini — "a CNN rebuilt with Transformer-era ideas" (2022)
- Stem: Conv3×3(3 → 64), stride 1, + LayerNorm (channels-first)
- 4 stages, dims [64, 128, 256, 512], depths [2, 2, 6, 2]
- Downsampling between stages: LN → Conv2×2 stride 2 (32→16→8→4)
- ConvNeXt block: depthwise Conv7×7 → LN → Linear(d → 4d) → GELU → Linear(4d → d) → layer scale (init 1e-6) → drop path → residual
- Drop path 0.1 (linearly increasing)
- GAP → LN → Linear(512, 10)
- ~8–9M params
- Key idea: borrow Transformer design choices (big kernels, LN, GELU, inverted bottleneck, fewer activations) but keep convolution's inductive bias.

### 5.7 Model checks (Phase 2)
For every model:
- forward pass on `(2, 3, 32, 32)` → output `(2, 10)`
- print param count and FLOPs (use `torch.utils.flop_counter.FlopCounterMode`)
- **overfit test:** train on 256 images for 200 steps → must reach ~100% train accuracy (proves the model can learn)
- save a `model_summary.md` table: model, year, params, FLOPs, inductive bias

---

## 6. Experiments

### E1 — Main showdown ⭐
- All 6 models, standard aug, 100% train data, 200 epochs, 3 seeds = **18 runs**
- Outputs: test acc (mean ± std), params, FLOPs, sec/epoch, total train time, train & inference throughput, peak memory, ECE, per-class accuracy, confusion matrix

### E2 — Data-hunger test ⭐ (the headline)
- All 6 models × train fractions **[5%, 10%, 25%, 50%, 100%]** (stratified subsets, fixed per fraction)
- **Equal optimization steps** across fractions: fix total steps = steps of a 200-epoch full-data run (~70,400 steps). Small subsets therefore see more epochs. Keep warmup at 5% of total steps.
- Same LR per model as E1. Validation set stays the full 5,000.
- 3 seeds for 10% and 100% (100% reuses E1); 1 seed for 5/25/50% unless I say otherwise
- Output: **accuracy vs. training-set size** line chart (log x-axis), one line per model, shaded std
- Hypothesis to test: CNN-family wins at small data; ViT/Mixer close the gap as data grows. Explain the result with **inductive bias**.

### E3 — Strong augmentation
- All 6 models, 100% data, 200 epochs, 1 seed
- Augmentation "strong": standard + `RandAugment(num_ops=2, magnitude=9)` + **Mixup (α=0.8) / CutMix (α=1.0)** switching with p=0.5 + RandomErasing(p=0.25)
- Output: gain in test acc vs. E1 per model (bar chart)
- Hypothesis: Transformers/Mixer gain more than CNNs (augmentation partly substitutes for missing inductive bias).

### E4 — Pretraining plot twist
- Fine-tune two ImageNet-pretrained models from `timm`, images upsampled to 224×224:
  - `vit_small_patch16_224.augreg_in21k_ft_in1k`
  - `resnet50.a1_in1k`
- AdamW, LR 1e-4 (sweep [3e-5, 1e-4, 3e-4] on val), wd 0.05, 10 epochs, warmup 1 epoch, batch 64, standard aug at 224 res, 3 seeds
- Output: compare against the from-scratch numbers. Expected story: pretrained ViT jumps to the high 90s → "Transformers aren't worse, they're data-hungry."

### E5 — Robustness (CIFAR-10-C style, generated locally)
- Use the **E1 best checkpoints** (all 3 seeds)
- Corruptions on the test set, 5 severities each: Gaussian noise, Gaussian blur, contrast reduction, brightness, JPEG compression, pixelate
- Output: accuracy vs. severity per corruption, plus mean corruption accuracy per model (heatmap: models × corruptions)

### E6 — Patch shuffle
- Cut each test image into a grid of **2×2, 4×4, 8×8** patches, randomly permute the patches, evaluate E1 checkpoints
- Also: **pixel shuffle within each patch** variant
- Output: accuracy vs. grid size per model + a figure of example scrambled frogs 🐸
- Question: which models rely on global shape vs. local texture?

### E7 — Interpretability (how each model "sees")
Use E1 seed-0 best checkpoints and the **same fixed set of 16 test images** (2 per class for 8 classes, chosen once and saved to `results/figures/fixed_images.png`).
- **First-layer filters** (Simple CNN, ResNet-18, ConvNeXt depthwise, ViT patch-embedding weights reshaped to 4×4×3)
- **Feature maps** at each stage for Simple CNN and ResNet-18
- **Grad-CAM** for Simple CNN, ResNet-18, ConvNeXt-mini (last conv stage)
- **ViT attention**: cls-token attention per head for the last layer, plus **attention rollout** across all layers, upsampled to 32×32 and overlaid
- **ViT positional-embedding similarity grid** (cosine similarity of each position's embedding with every other — shows if it learned 2D structure)
- **Side-by-side figure:** same images, Grad-CAM (ResNet) vs. attention rollout (ViT) ⭐
- **Embeddings:** UMAP of penultimate features for each model on 2,000 test images, colored by class
- **Mistake gallery:** 12 most confident wrong predictions per model
- **Agreement matrix:** % of test images where each pair of models gives the same prediction

---

## 7. Build phases (do these in order)

### Phase 0 — Environment & hardware check
- [ ] Create venv, install requirements, pin versions
- [ ] Print: Python, torch, CUDA availability, GPU name, VRAM, bf16 support
- [ ] Download CIFAR-10 to `data/` (add `data/` to `.gitignore`)
- [ ] Benchmark DataLoader throughput with `num_workers` in [0, 2, 4, 8]; pick the best
- [ ] `git init`, first commit
- ✅ Report: hardware summary + **compute estimate for the full plan** on this hardware (Section 9). If no GPU, propose the remote-GPU path (1.4).

### Phase 1 — Data pipeline
- [ ] `src/data.py`: fixed 45k/5k stratified split, standard + strong augmentation, stratified subset fractions (cached indices in `data/splits/`)
- [ ] Save a grid of augmented samples to `results/figures/augmentations.png`
- ✅ Check: class counts balanced in every split/subset; normalization stats correct

### Phase 2 — Models
- [ ] Implement all 6 models + registry
- [ ] Run the model checks (5.7) for each
- ✅ Output: `results/tables/model_summary.md`

### Phase 3 — Training loop + smoke test + LR sweep
- [ ] `src/train.py`, `src/evaluate.py`, `src/profile.py`
- [ ] `scripts/smoke_test.py`: every model, 2 epochs on 10% data → no errors, loss decreasing, all output files written, resume works
- [ ] `scripts/lr_sweep.py`: for each model, LRs **[2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3]**, 40 epochs, seed 0, 100% data, standard aug → pick best by **val** accuracy
- ✅ Output: `results/tables/lr_sweep.csv` + a plot; ask me to confirm the chosen LRs before Phase 4

### Phase 4 — E1 main showdown (18 runs)
- ✅ Output: scoreboard table + learning curves for all models (mean ± std band)

### Phase 5 — E2 data-hunger
- ✅ Output: the data-hunger chart + table

### Phase 6 — E3 strong augmentation + E4 pretraining
- ✅ Output: augmentation-gain bar chart + pretrained vs. from-scratch comparison table

### Phase 7 — E5 robustness + E6 patch shuffle
- ✅ Output: corruption heatmap, severity curves, patch-shuffle chart + scrambled-image examples

### Phase 8 — E7 interpretability
- ✅ Output: every figure listed in E7

### Phase 9 — Report & README
- [ ] `scripts/make_report.py` regenerates **all** tables and figures from `results/runs/` (no manual numbers anywhere)
- [ ] `results/REPORT.md`: every experiment → question, setup, result table/figure, 2–4 sentence takeaway
- [ ] `README.md`: hook, headline chart, scoreboard, "what I learned" (inductive bias explained simply), how to reproduce, hardware + total compute used
- [ ] Draft 3 LinkedIn posts in `results/linkedin_drafts.md` (Section 10)

**Run order for long jobs:** run models sequentially on one GPU; queue runs in `scripts/run_all.py` so the whole thing can run overnight and resume after interruption. Skip any run whose `metrics.json` already exists.

---

## 8. Figures checklist (all saved as PNG, 200 dpi, consistent colors per model)

Fix one color per model and use it everywhere:
MLP = grey, Simple CNN = blue, ResNet-18 = green, ViT = red, MLP-Mixer = orange, ConvNeXt-mini = purple.

1. `scoreboard.png` — bar chart, test acc mean ± std
2. `acc_vs_params.png` — scatter, bubble size = train time
3. `acc_vs_flops.png`
4. `learning_curves.png` — val acc vs. epoch, all models
5. `data_hunger.png` ⭐
6. `aug_gain.png`
7. `pretrained_vs_scratch.png`
8. `robustness_heatmap.png` + `robustness_curves.png`
9. `patch_shuffle.png` + `patch_shuffle_examples.png`
10. `first_layer_filters.png`
11. `feature_maps_resnet.png`
12. `gradcam_vs_attention.png` ⭐
13. `vit_pos_embedding_similarity.png`
14. `umap_embeddings.png`
15. `confusion_matrices.png` (2×3 grid)
16. `mistake_gallery_<model>.png`
17. `model_agreement.png`

---

## 9. Compute budget (estimate it for real in Phase 0)

Run counts:

| Item | Runs | Epochs each |
|---|---|---|
| LR sweep | 6 models × 5 LRs = 30 | 40 |
| E1 | 18 | 200 |
| E2 (excluding 100%, reused from E1) | 6 × (3 + 1 + 1 + 1) = 36 | equal steps to 200 epochs |
| E3 | 6 | 200 |
| E4 | 2 × 3 seeds (+ small LR sweep) | 10 @ 224px |
| E5–E7 | evaluation only | — |

This is a **multi-day GPU job** (think dozens of GPU-hours on a T4-class card; far less on a modern RTX/A100). In Phase 0, time one epoch per model on the actual hardware and print the real total. If it's more than ~60 GPU-hours, ask me before proceeding — options are fewer E2 seeds or fewer sweep LRs, **not** fewer epochs in E1.

---

## 10. LinkedIn drafts (Phase 9)

Write 3 drafts, casual tone, short paragraphs, one image each, numbers pulled from the actual results:

1. **The hook post** — "I rebuilt ~35 years of computer vision on one dataset. Here's what happened when I starved the Transformers of data." + `data_hunger.png`
2. **How they see** — "Same frog, two brains: how a CNN and a Transformer look at an image." + `gradcam_vs_attention.png`
3. **The weird one** — "I scrambled images like a puzzle. One model didn't care." + `patch_shuffle_examples.png`

Each ends with the repo link and one honest limitation (tiny 32×32 images, small models, one dataset).

---

## 11. Ground rules for the coding agent

- Work phase by phase; commit after each with a clear message.
- Keep all hyperparameters in YAML configs, none hard-coded.
- Every number in REPORT/README must be generated from saved results.
- Prefer clarity over cleverness — this is a learning project; comment the architecture code generously (shapes in comments, e.g. `# (B, 65, 384)`).
- If a result contradicts a hypothesis in this file, **report it honestly** — that's the interesting part.
- If something fails or looks off (NaNs, suspiciously high/low accuracy, data leakage), stop and tell me.
- At the end of each phase, give me: what was built, key numbers, anything surprising, and the next step.

---

## 12. ⏱️ 9-Hour Unattended Mode (`--mode 9h`)

> Use this when I want to start it, leave for ~9 hours (gym, errands, whatever), and come back to results.
> It keeps **all 6 models and all 7 experiments**. It trims seeds and epochs, auto-decides everything, and never stops to ask me anything during the long run.

### 12.1 How the day works

| When | Who runs it | What happens |
|---|---|---|
| **~1–1.5 h, I'm present** | the coding agent | Phases 0–3: environment, data, models + checks, training loop, smoke test **on the real GPU**, timing benchmark, budget plan (12.4) |
| **~7–8 h, I'm away** | plain Python process | `python scripts/run_all.py --mode 9h`: LR sweep → experiments → evaluation, fully automatic |
| **~30 min, I'm back** | the coding agent | Phase 9: figures, `REPORT.md`, README, LinkedIn drafts |

**Critical:** the long queue must **not** depend on an agent session staying open or on any permission prompt. It is a standalone Python process launched in its own PowerShell window (or a Kaggle background run, 12.6). The coding agent builds it; it runs on its own.

### 12.2 Run sizes: full mode vs. 9h mode

| Item | Full mode | 9h mode (core) |
|---|---|---|
| LR sweep | 5 LRs × 40 epochs | 3 LRs **[5e-4, 1e-3, 2e-3] × 10 epochs**, best val LR **auto-picked** (no confirmation) |
| E1 main | 200 epochs × 3 seeds | **100 epochs × 1 seed** (seed 0) |
| E2 data-hunger | 5 fractions, steps = 200-epoch run, 3 seeds | fractions **10%, 25%, 100%**, every point gets **exactly 7,040 steps** (= 20 full-data epochs; 10% ≈ 200 epochs, 25% ≈ 80 epochs), 1 seed. E2 does **not** reuse E1, so the plot is step-matched across fractions. |
| E3 strong aug | all 6 models, 200 epochs | upgrade only (12.3) |
| E4 pretrained | LR sweep, 10 epochs × 3 seeds | LR **1e-4 fixed, 5 epochs, 1 seed** |
| E5 / E6 / E7 | full | **full** (evaluation only, cheap; run on the E1 checkpoints) |

Everything else in Section 3 stays **identical**: split, normalization, standard augmentation, AdamW + weight decay, label smoothing, AMP, grad clipping, checkpoint selection on val, test touched once. Warmup = **5% of total steps** in every run, then cosine decay.

E4 tip: upsample 32→224 **on the GPU** (`F.interpolate` after moving the batch), not in the CPU DataLoader, or CPU becomes the bottleneck.

### 12.3 Upgrades (added only if the budget allows, in this order)
1. **E3 for ResNet-18 + ViT** (100 epochs each): the key "does augmentation help Transformers more?" comparison
2. **E1 seeds 1 and 2**: adds mean ± std error bars
3. **E3 for the other 4 models**
4. **E2 extra fractions 5% and 50%**
5. **E2 seeds 1 and 2**

### 12.4 Budget planner (`scripts/plan_budget.py`)
Built and run in Phase 3, **while I'm still present**:
- For each model, time ~200 training steps at batch 128 on the real hardware (skip the first 20 steps as warmup), plus one validation pass; also time ~50 steps of each E4 model at 224 px.
- Default budget: `--hours 8` (leaves ~1 h of slack inside the 9 h).
- Projected time = core queue first, then add upgrades from 12.3 in order while the projection still fits.
- If even the core doesn't fit, trim in this order and tell me before I leave: E4 epochs 5 → 3, then E2 steps 7,040 → 5,280, then LR sweep 10 → 6 epochs. **Never** cut E1 below 100 epochs.
- Print the full queue with per-run estimates and a projected finish time (wall clock), and save it to `results/queue_9h.json`.
- Show me the plan, wait for my "go", then I launch it.

### 12.5 Queue order and unattended rules (`scripts/run_all.py --mode 9h`)
Order (most important first, so a crash late in the day still leaves the core story):
1. LR sweep (all 6)
2. E1 (all 6)
3. E7 interpretability → E5 robustness → E6 patch shuffle (eval only, on E1 checkpoints)
4. E2 (all 6 × 3 fractions)
5. E4 pretrained
6. Upgrades from the plan, in order

Rules while I'm away:
- **No questions, no approval gates.** The "ask me" rules in Sections 7, 9 and 11 are suspended in this mode.
- **Failures don't stop the queue.** Wrap each run in try/except. On NaN/divergence or a crash, log the traceback to `results/failures/<run>.log`, mark the run failed, and move on. Never auto-tweak hyperparameters to rescue a run.
- **Re-plan as you go.** After every run, update the projection with actual times. If the projected finish exceeds the budget, drop remaining **upgrades** from the bottom of the list (never core items).
- **Resumable.** Re-running the same command skips runs with an existing `metrics.json` and resumes partial runs from `last.pt`.
- **Multi-GPU aware.** With 2+ GPUs (e.g. Kaggle 2×T4), run one worker process per GPU (`CUDA_VISIBLE_DEVICES`), each pulling the next job from the queue.
- **Light on disk.** LR-sweep runs save no checkpoints. Other runs keep only `best.pt` (+ `last.pt` until the run finishes).
- **Progress log.** Append to `results/progress.log` after each run: run name, test/val acc, duration, runs left, projected finish time.
- **Final summary.** At the end, write `results/RUN_SUMMARY.md`: completed / failed / skipped runs, which upgrades made it in, total GPU time, and any caveats to mention in the report.

### 12.6 Pre-flight checklist (before I leave)

**Local Windows GPU:**
- [ ] Smoke test passed on the GPU and the planner's projected finish is < 9 h from now
- [ ] Laptop plugged in
- [ ] Disable sleep while plugged in:
  `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`
  (restore later with e.g. `powercfg /change standby-timeout-ac 30`)
- [ ] Pause Windows Update for the day (Settings → Windows Update → Pause), so it doesn't reboot mid-run
- [ ] Launch in its **own** PowerShell window (venv activated):
  `python scripts/run_all.py --mode 9h 2>&1 | Tee-Object -FilePath results\run_all.log`
- [ ] Confirm the first LR-sweep run started and `progress.log` is updating, then leave

**No local GPU → Kaggle:**
- [ ] The coding agent generates `kaggle_runner.ipynb`: installs requirements, clones the repo, runs `plan_budget.py --hours 8`, then `run_all.py --mode 9h`, then zips `results/` into `/kaggle/working/results.zip`
- [ ] Notebook settings: Accelerator **GPU T4 ×2**, Internet **on** (pip + pretrained weights)
- [ ] Start it with **Save Version → Save & Run All (Commit)** so it runs in the background with the browser closed. Check the current per-session time limit in Kaggle first (historically ~12 h); the 8 h budget should sit under it.
- [ ] When I'm back: download `results.zip`, unzip into the repo, and run Phase 9 locally with the coding agent

### 12.7 What changes in the write-up
In 9h mode, `REPORT.md` and the README must state plainly:
- 1 seed for most results → differences smaller than ~1 percentage point between models are **not** meaningful
- 100 epochs, not 200 → Transformers and Mixer usually keep improving with longer training, so this setup slightly **favors the CNNs**
- E2 points are step-matched but short (7,040 steps), so absolute E2 accuracies are lower than E1's
- which upgrades actually ran (from `RUN_SUMMARY.md`)

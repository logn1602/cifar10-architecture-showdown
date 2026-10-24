"""E5 robustness and E6 patch shuffle: the test set, damaged on purpose.

E5 asks how gracefully each architecture degrades under the everyday damage of
CIFAR-10-C (noise, blur, contrast, brightness, JPEG, pixelation). E6 asks what
each one actually *relies on* -- the global arrangement of an image, or its
local texture -- by cutting images into a grid and shuffling it like a jigsaw.

Three rules keep the comparison fair:

* **Every model sees exactly the same damaged images.** Each condition is
  generated once for the full test set, from a fixed seed, and then fed to
  every checkpoint in turn. (A ``max_images`` cap for CPU testing selects rows
  *after* generation, so it never changes what any one image looks like.)
* **Damage is applied to raw uint8 pixels, before normalisation**, and the
  result is re-quantised to uint8 -- the same order and the same truncation as
  the CIFAR-10-C reference code, whose severity values are in base.yaml.
* **Clean accuracy is re-measured through this same pipeline** and checked
  against the training loop's own test pass. A mismatch would mean a bug in the
  evaluation path (normalisation, channel order, wrong checkpoint) and stops
  the run rather than producing a plausible-looking but wrong heatmap.

Run directly for a local test on few images:
``python -m src.corruptions --exp E5 --max-images 300``
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from src.analysis import (
        CLASSES, Predictor, e1_runs, eval_device, fixed_image_indices, load_run_model,
        model_style, pyplot, read_csv, save_fig, stratified_indices, table_path, test_set,
        write_csv,
    )
    from src.experiments import load_base_config, results_dir, results_path
    from src.models import MODEL_ORDER
    from src.utils import ensure_dir, human_time, save_json
except ImportError:  # pragma: no cover - `python src/corruptions.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.analysis import (  # noqa: E402
        CLASSES, Predictor, e1_runs, eval_device, fixed_image_indices, load_run_model,
        model_style, pyplot, read_csv, save_fig, stratified_indices, table_path, test_set,
        write_csv,
    )
    from src.experiments import load_base_config, results_dir, results_path  # noqa: E402
    from src.models import MODEL_ORDER  # noqa: E402
    from src.utils import ensure_dir, human_time, save_json  # noqa: E402

__all__ = ["corrupt", "scramble", "run_e5", "run_e6", "plot_e5", "plot_e6", "CORRUPTIONS"]

# Clean accuracy measured here vs. the training loop's own test pass. Both run
# the same weights on the same images; only kernel choice (channels_last, AMP
# rounding) differs, which moves a handful of images at most. 0.5 pp is 50
# images -- far beyond numerical noise, far below any real pipeline bug.
CLEAN_TOLERANCE = 0.005

PRETTY = {
    "gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur",
    "contrast": "Contrast", "brightness": "Brightness", "jpeg": "JPEG",
    "pixelate": "Pixelate",
}

# Images per chunk when generating damage. Fixed, because the random draws are
# consumed chunk by chunk: changing it would change the noise / permutations.
_CHUNK = 2000


# ---------------------------------------------------------------------------
# E5: corruptions (uint8 NHWC in, uint8 NHWC out)
# ---------------------------------------------------------------------------


def _f(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


def _u8(x: np.ndarray) -> np.ndarray:
    # Truncation, not rounding: the reference stores np.uint8(float * 255).
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def gaussian_noise(x: np.ndarray, c: float, rng: np.random.Generator) -> np.ndarray:
    f = _f(x)
    return _u8(f + c * rng.standard_normal(size=f.shape, dtype=np.float32))


def gaussian_blur(x: np.ndarray, c: float, rng=None) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    # skimage.filters.gaussian(..., multichannel=True) as in the reference:
    # blur the two spatial axes only, 'nearest' borders, truncate at 4 sigma.
    return _u8(gaussian_filter(_f(x), sigma=(0, c, c, 0), mode="nearest", truncate=4.0))


def contrast(x: np.ndarray, c: float, rng=None) -> np.ndarray:
    f = _f(x)
    means = f.mean(axis=(1, 2), keepdims=True)          # per image, per channel
    return _u8((f - means) * c + means)


def brightness(x: np.ndarray, c: float, rng=None) -> np.ndarray:
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
    hsv = rgb_to_hsv(_f(x))
    hsv[..., 2] = np.clip(hsv[..., 2] + c, 0.0, 1.0)
    return _u8(hsv_to_rgb(hsv))


def jpeg(x: np.ndarray, c: float, rng=None) -> np.ndarray:
    from PIL import Image
    out = np.empty_like(x)
    for i, img in enumerate(x):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=int(c))
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("RGB"))
    return out


def pixelate(x: np.ndarray, c: float, rng=None) -> np.ndarray:
    from PIL import Image
    side = x.shape[1]
    small = int(side * c)
    out = np.empty_like(x)
    for i, img in enumerate(x):
        im = Image.fromarray(img).resize((small, small), Image.BOX)
        out[i] = np.asarray(im.resize((side, side), Image.BOX))
    return out


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise, "gaussian_blur": gaussian_blur, "contrast": contrast,
    "brightness": brightness, "jpeg": jpeg, "pixelate": pixelate,
}


def corrupt(images: np.ndarray, name: str, severity: int, cfg: dict) -> np.ndarray:
    """Apply corruption ``name`` at ``severity`` (1-5) to uint8 NHWC images.

    Works through fixed-size chunks (bounded memory: the float intermediates
    of HSV conversion or blurring 10k images at once run to ~1 GB). Gaussian
    noise draws from one generator seeded by (seed, corruption, severity),
    consumed chunk after chunk, so the noise is identical for every model and
    every re-run.
    """
    ex = cfg["experiments"]
    c = ex["e5_params"][name][severity - 1]
    rng = np.random.default_rng([int(ex["e5_seed"]), list(CORRUPTIONS).index(name), severity])
    fn = CORRUPTIONS[name]
    out = np.empty_like(images)
    for i in range(0, len(images), _CHUNK):
        out[i:i + _CHUNK] = fn(images[i:i + _CHUNK], c, rng)
    return out


# ---------------------------------------------------------------------------
# E6: scrambles (uint8 NHWC in, uint8 NHWC out)
# ---------------------------------------------------------------------------


def _non_identity_perms(n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """``n`` uniformly random permutations of ``k`` items, none the identity.

    With a 2x2 grid a uniform permutation leaves the image untouched 1 time in
    24; counting those as "shuffled" would quietly inflate 2x2 accuracy.
    """
    perm = torch.argsort(torch.rand(n, k, generator=gen), dim=1)
    ident = torch.arange(k)
    while True:
        same = (perm == ident).all(dim=1)
        if not same.any():
            return perm
        perm[same] = torch.argsort(torch.rand(int(same.sum()), k, generator=gen), dim=1)


def scramble(images: np.ndarray, variant: str, grid: int, seed: int) -> np.ndarray:
    """Cut each image into ``grid x grid`` patches and shuffle.

    * ``"patch"``: permute the patches (a jigsaw) -- local texture intact,
      global layout destroyed.
    * ``"pixel"``: leave every patch where it is but shuffle the pixels inside
      it -- coarse layout intact, local texture destroyed.

    One permutation per image (and per patch, for ``"pixel"``), all drawn from
    a generator seeded by (seed, variant, grid), consumed chunk after chunk.
    """
    gen = torch.Generator().manual_seed(int(seed) * 1000 + grid * 10 + ("patch", "pixel").index(variant))
    out = np.empty_like(images)
    for i in range(0, len(images), _CHUNK):
        out[i:i + _CHUNK] = _scramble_chunk(images[i:i + _CHUNK], variant, grid, gen)
    return out


def _scramble_chunk(images: np.ndarray, variant: str, grid: int,
                    gen: torch.Generator) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(images))         # (N, 32, 32, 3)
    n, h, w, ch = x.shape
    if h % grid or w % grid:
        raise ValueError(f"{h}x{w} image does not split into a {grid}x{grid} grid")
    p = h // grid
    k = grid * grid

    # (N, g, p, g, p, C) -> (N, g*g patches, p*p pixels, C), row-major both ways
    t = x.reshape(n, grid, p, grid, p, ch).permute(0, 1, 3, 2, 4, 5).reshape(n, k, p * p, ch)
    if variant == "patch":
        perm = _non_identity_perms(n, k, gen)                       # (N, g*g)
        t = torch.gather(t, 1, perm[:, :, None, None].expand(-1, -1, p * p, ch))
    elif variant == "pixel":
        perm = torch.argsort(torch.rand(n, k, p * p, generator=gen), dim=-1)   # (N, g*g, p*p)
        t = torch.gather(t, 2, perm[..., None].expand(-1, -1, -1, ch))
    else:
        raise ValueError(f"variant must be 'patch' or 'pixel', got {variant!r}")
    out = t.reshape(n, grid, grid, p, p, ch).permute(0, 1, 3, 2, 4, 5).reshape(n, h, w, ch)
    return out.numpy()


# ---------------------------------------------------------------------------
# Shared driver
# ---------------------------------------------------------------------------


def _setup(cfg: dict, max_images: int | None, models):
    device = eval_device()
    pred = Predictor(cfg, device)
    runs = e1_runs(models or MODEL_ORDER)
    if not runs:
        raise RuntimeError(f"no completed E1 runs with best.pt under {results_path() / 'runs' / 'E1'}")
    images, labels = test_set(cfg)
    rows = (stratified_indices(labels, max_images, seed=0) if max_images
            else np.arange(len(labels)))
    loaded = [(r, load_run_model(r, device)) for r in runs]
    print(f"{len(loaded)} checkpoint(s): "
          + ", ".join(f"{r.model}/s{r.seed}" for r, _ in loaded)
          + f" | {len(rows)} test images | {device}", flush=True)
    return pred, images, labels, rows, loaded


def _clean_rows(pred, images, labels, rows, loaded, full: bool) -> list[dict]:
    out = []
    for r, m in loaded:
        acc = pred.accuracy(m, images[rows], labels[rows])
        ref = r.metrics.get("test_acc")
        if full and ref is not None and abs(acc - ref) > CLEAN_TOLERANCE:
            raise RuntimeError(
                f"{r.model}/s{r.seed}: clean accuracy through the evaluation pipeline is "
                f"{acc:.4f}, but the training loop measured {ref:.4f} on the same weights. "
                "The evaluation path is broken (normalisation? channel order? checkpoint?) "
                "-- refusing to report corrupted-image numbers built on it.")
        out.append({"model": r.model, "seed": r.seed, "acc": acc,
                    "train_loop_test_acc": ref})
    return out


# ---------------------------------------------------------------------------
# E5
# ---------------------------------------------------------------------------


def run_e5(cfg: dict | None = None, mode: str = "9h", out_dir: str = "results/runs/E5",
           max_images: int | None = None, models=None) -> dict:
    """Accuracy of every E1 checkpoint under 6 corruptions x 5 severities."""
    cfg = cfg or load_base_config()
    t0 = time.time()
    out = ensure_dir(out_dir)            # repo-relative, as the queue passes it
    pred, images, labels, rows, loaded = _setup(cfg, max_images, models)

    records = [{"model": c["model"], "seed": c["seed"], "corruption": "clean", "severity": 0,
                "acc": c["acc"]} for c in _clean_rows(pred, images, labels, rows, loaded,
                                                      full=max_images is None)]
    ex = cfg["experiments"]
    for name in ex["e5_corruptions"]:
        for sev in ex["e5_severities"]:
            t = time.time()
            damaged = corrupt(images, name, int(sev), cfg)[rows]
            for r, m in loaded:
                records.append({"model": r.model, "seed": r.seed, "corruption": name,
                                "severity": int(sev),
                                "acc": pred.accuracy(m, damaged, labels[rows])})
            print(f"  E5 {name:<15} severity {sev}  ({time.time() - t:.1f}s)", flush=True)

    raw = write_csv(records, out / "robustness_raw.csv",
                    ["model", "seed", "corruption", "severity", "acc"])
    summary = plot_e5(raw, cfg)
    example = fixed_image_indices(cfg, labels)[0]
    plot_e5_examples(images[example], cfg)
    summary.update({"n_images": int(len(rows)), "seconds": time.time() - t0,
                    "checkpoints": [f"{r.model}/s{r.seed}" for r, _ in loaded]})
    save_json(summary, out / "summary.json")
    print(f"E5 done in {human_time(time.time() - t0)}", flush=True)
    return summary


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def plot_e5(raw_csv: Path, cfg: dict) -> dict:
    """Tables + curves + heatmap, from the raw CSV alone (re-runnable in Phase 9)."""
    rows = read_csv(Path(raw_csv))
    plt = pyplot()
    corrs = [c for c in cfg["experiments"]["e5_corruptions"]]
    sevs = sorted({int(r["severity"]) for r in rows if r["corruption"] != "clean"})
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    seeds = {m: sorted({r["seed"] for r in rows if r["model"] == m}) for m in models}

    def acc(m: str, c: str, s: int) -> list[float]:
        return [r["acc"] for r in rows if r["model"] == m and r["corruption"] == c
                and int(r["severity"]) == s]

    # Per model: clean, mean over severities per corruption, overall mCA.
    table = []
    for m in models:
        clean, _ = _mean_std(acc(m, "clean", 0))
        per_c = {c: float(np.mean([np.mean(acc(m, c, s)) for s in sevs])) for c in corrs}
        # mCA per seed first, then mean +/- std across seeds.
        mca_seeds = [np.mean([r["acc"] for r in rows if r["model"] == m and r["seed"] == sd
                              and r["corruption"] != "clean"]) for sd in seeds[m]]
        mca, mca_sd = _mean_std(mca_seeds)
        worst = min(per_c, key=per_c.get)
        table.append({"model": m, "seeds": len(seeds[m]), "clean_acc": clean,
                      **{f"{c}_acc": per_c[c] for c in corrs},
                      "mean_corruption_acc": mca, "mean_corruption_acc_std": mca_sd,
                      "relative_robustness": mca / clean if clean else None,
                      "worst_corruption": worst})
    write_csv(table, table_path("robustness_summary.csv"))
    lines = ["# E5 -- robustness to CIFAR-10-C-style corruptions", "",
             "_Generated by `src/corruptions.py`; mean over severities 1-5 and over seeds. "
             "Relative robustness = mean corruption accuracy / clean accuracy._", "",
             "| model | seeds | clean | " + " | ".join(PRETTY[c] for c in corrs)
             + " | mean (mCA) | relative | worst |",
             "|---|---|---|" + "---|" * len(corrs) + "---|---|---|"]
    for t in table:
        _, label = model_style(cfg, t["model"])
        sd = f" ± {100 * t['mean_corruption_acc_std']:.1f}" if t["seeds"] > 1 else ""
        lines.append(f"| {label} | {t['seeds']} | {100 * t['clean_acc']:.1f} | "
                     + " | ".join(f"{100 * t[f'{c}_acc']:.1f}" for c in corrs)
                     + f" | **{100 * t['mean_corruption_acc']:.1f}{sd}** | "
                       f"{t['relative_robustness']:.2f} | {PRETTY[t['worst_corruption']]} |")
    table_path("robustness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Curves: one panel per corruption, severity 0 (clean) .. 5.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    for ax, c in zip(axes.flat, corrs):
        for m in models:
            color, label = model_style(cfg, m)
            xs = [0] + sevs
            stats = [_mean_std(acc(m, "clean", 0))] + [_mean_std(acc(m, c, s)) for s in sevs]
            mu = np.array([s[0] for s in stats]) * 100
            sd = np.array([s[1] for s in stats]) * 100
            ax.plot(xs, mu, "-o", color=color, label=label, lw=2, ms=4)
            if len(seeds[m]) > 1:
                ax.fill_between(xs, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
        ax.set_title(PRETTY[c])
        ax.set_xticks([0] + sevs)
        ax.set_xticklabels(["clean"] + [str(s) for s in sevs])
        ax.axhline(10, color="0.6", ls=":", lw=1)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy (%)")
    for ax in axes[-1]:
        ax.set_xlabel("severity")
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E5 -- accuracy under corruption (dotted line = chance)", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, "robustness_curves.png", cfg)

    # Heatmap: models x corruptions (+ clean and mean), mean over severities.
    cols = ["clean"] + corrs + ["mean"]
    mat = np.array([[t["clean_acc"]] + [t[f"{c}_acc"] for c in corrs]
                    + [t["mean_corruption_acc"]] for t in table]) * 100
    fig, ax = plt.subplots(figsize=(1.25 * len(cols) + 2, 0.62 * len(models) + 1.6))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=max(0, mat.min() - 5), vmax=mat.max())
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if v < (mat.min() + mat.max()) / 2 else "black",
                    fontweight="bold" if cols[j] == "mean" else None)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Clean"] + [PRETTY[c] for c in corrs] + ["Mean (mCA)"],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_style(cfg, m)[1] for m in models])
    ax.axvline(0.5, color="white", lw=2)
    ax.axvline(len(cols) - 1.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, label="test accuracy (%)", fraction=0.03)
    ax.set_title("E5 -- mean accuracy over severities 1-5")
    fig.tight_layout()
    save_fig(fig, "robustness_heatmap.png", cfg)
    return {"table": table}


def plot_e5_examples(image: np.ndarray, cfg: dict) -> None:
    """One test image under every corruption and severity -- what the models saw."""
    plt = pyplot()
    ex = cfg["experiments"]
    corrs, sevs = ex["e5_corruptions"], ex["e5_severities"]
    fig, axes = plt.subplots(len(corrs), len(sevs) + 1, figsize=(1.45 * (len(sevs) + 1),
                                                                  1.5 * len(corrs)))
    for i, c in enumerate(corrs):
        axes[i, 0].imshow(image, interpolation="nearest")
        axes[i, 0].set_ylabel(PRETTY[c], fontsize=9)
        for j, s in enumerate(sevs, start=1):
            axes[i, j].imshow(corrupt(image[None], c, int(s), cfg)[0], interpolation="nearest")
    for j, title in enumerate(["clean"] + [f"severity {s}" for s in sevs]):
        axes[0, j].set_title(title, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("E5 -- the corruptions, on one fixed test image", y=1.0)
    fig.tight_layout()
    save_fig(fig, "robustness_examples.png", cfg)


# ---------------------------------------------------------------------------
# E6
# ---------------------------------------------------------------------------


def run_e6(cfg: dict | None = None, mode: str = "9h", out_dir: str = "results/runs/E6",
           max_images: int | None = None, models=None) -> dict:
    """Accuracy of every E1 checkpoint on patch- and pixel-shuffled test images."""
    cfg = cfg or load_base_config()
    t0 = time.time()
    out = ensure_dir(out_dir)            # repo-relative, as the queue passes it
    pred, images, labels, rows, loaded = _setup(cfg, max_images, models)
    ex = cfg["experiments"]
    frogs = [i for i in fixed_image_indices(cfg, labels) if CLASSES[labels[i]] == "frog"]

    records = [{"model": c["model"], "seed": c["seed"], "variant": "clean", "grid": 1,
                "acc": c["acc"]} for c in _clean_rows(pred, images, labels, rows, loaded,
                                                      full=max_images is None)]
    # Seed-0 predictions on the example frogs, for the examples figure.
    examples: dict[str, dict] = {"clean": {"images": images[frogs], "preds": {}}}
    for r, m in loaded:
        if r.seed == 0:
            examples["clean"]["preds"][r.model] = pred.predict(m, images[frogs])

    for variant in ex["e6_variants"]:
        for grid in ex["e6_patch_grids"]:
            t = time.time()
            shuffled = scramble(images, variant, int(grid), int(ex["e6_seed"]))
            key = f"{variant}_{grid}"
            examples[key] = {"images": shuffled[frogs], "preds": {}}
            for r, m in loaded:
                records.append({"model": r.model, "seed": r.seed, "variant": variant,
                                "grid": int(grid),
                                "acc": pred.accuracy(m, shuffled[rows], labels[rows])})
                if r.seed == 0:
                    examples[key]["preds"][r.model] = pred.predict(m, shuffled[frogs])
            print(f"  E6 {variant:<5} {grid}x{grid}  ({time.time() - t:.1f}s)", flush=True)

    raw = write_csv(records, out / "patch_shuffle_raw.csv",
                    ["model", "seed", "variant", "grid", "acc"])
    np.savez_compressed(out / "examples.npz", frog_indices=np.asarray(frogs),
                        **{f"{k}__images": v["images"] for k, v in examples.items()},
                        **{f"{k}__preds__{m}": p for k, v in examples.items()
                           for m, p in v["preds"].items()})
    summary = plot_e6(raw, cfg)
    plot_e6_examples(out / "examples.npz", cfg)
    summary.update({"n_images": int(len(rows)), "seconds": time.time() - t0,
                    "checkpoints": [f"{r.model}/s{r.seed}" for r, _ in loaded]})
    save_json(summary, out / "summary.json")
    print(f"E6 done in {human_time(time.time() - t0)}", flush=True)
    return summary


def plot_e6(raw_csv: Path, cfg: dict) -> dict:
    rows = read_csv(Path(raw_csv))
    plt = pyplot()
    grids = sorted({int(r["grid"]) for r in rows if r["variant"] != "clean"})
    variants = [v for v in cfg["experiments"]["e6_variants"] if any(r["variant"] == v for r in rows)]
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]

    def stat(m: str, v: str, g: int) -> tuple[float, float]:
        return _mean_std([r["acc"] for r in rows if r["model"] == m and r["variant"] == v
                          and int(r["grid"]) == g])

    table = []
    for m in models:
        row = {"model": m, "clean_acc": stat(m, "clean", 1)[0]}
        for v in variants:
            for g in grids:
                row[f"{v}_{g}x{g}"] = stat(m, v, g)[0]
        table.append(row)
    write_csv(table, table_path("patch_shuffle.csv"))
    head = [f"{v} {g}×{g}" for v in variants for g in grids]
    lines = ["# E6 -- patch shuffle", "",
             "_Generated by `src/corruptions.py`. Test accuracy (%), mean over seeds. "
             "`patch`: patches permuted (layout destroyed, texture kept). "
             "`pixel`: pixels shuffled inside each patch (texture destroyed, layout kept)._", "",
             "| model | clean | " + " | ".join(head) + " |",
             "|---|---|" + "---|" * len(head)]
    for t in table:
        lines.append(f"| {model_style(cfg, t['model'])[1]} | {100 * t['clean_acc']:.1f} | "
                     + " | ".join(f"{100 * t[f'{v}_{g}x{g}']:.1f}" for v in variants
                                  for g in grids) + " |")
    table_path("patch_shuffle.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    titles = {"patch": "Patches permuted\n(global layout destroyed, local texture kept)",
              "pixel": "Pixels shuffled within each patch\n(local texture destroyed, layout kept)"}
    fig, axes = plt.subplots(1, len(variants), figsize=(6.2 * len(variants), 4.6), sharey=True,
                             squeeze=False)
    xs = list(range(len(grids) + 1))
    for ax, v in zip(axes[0], variants):
        for m in models:
            color, label = model_style(cfg, m)
            st = [stat(m, "clean", 1)] + [stat(m, v, g) for g in grids]
            mu = np.array([s[0] for s in st]) * 100
            sd = np.array([s[1] for s in st]) * 100
            ax.plot(xs, mu, "-o", color=color, label=label, lw=2, ms=5)
            if any(sd > 0):
                ax.fill_between(xs, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
        ax.axhline(10, color="0.6", ls=":", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(["clean"] + [f"{g}×{g}" for g in grids])
        ax.set_xlabel("grid (finer to the right)")
        ax.set_title(titles.get(v, v), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[0, -1].legend(frameon=False, fontsize=9)
    fig.suptitle("E6 -- what does each model rely on? (dotted line = chance)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "patch_shuffle.png", cfg)
    return {"table": table}


def plot_e6_examples(examples_npz: Path, cfg: dict) -> None:
    """The scrambled frogs, with which models (seed 0) still answer "frog"."""
    plt = pyplot()
    from matplotlib.lines import Line2D

    d = np.load(examples_npz)
    keys = ["clean"] + [f"{v}_{g}" for v in cfg["experiments"]["e6_variants"]
                        for g in cfg["experiments"]["e6_patch_grids"]]
    keys = [k for k in keys if f"{k}__images" in d.files]
    n_img = d["clean__images"].shape[0]
    models = [m for m in MODEL_ORDER if f"clean__preds__{m}" in d.files]
    frog = CLASSES.index("frog")

    fig_h = 2.1 * n_img + 0.9
    fig, axes = plt.subplots(n_img, len(keys), figsize=(1.75 * len(keys), fig_h), squeeze=False)
    for j, k in enumerate(keys):
        for i in range(n_img):
            ax = axes[i, j]
            ax.imshow(d[f"{k}__images"][i], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            # One dot per model under the image: filled = still says "frog".
            for q, m in enumerate(models):
                color, _ = model_style(cfg, m)
                ok = int(d[f"{k}__preds__{m}"][i]) == frog
                x = (q + 0.5) / len(models)
                ax.add_artist(Line2D([x], [-0.1], marker="o", ms=6.5, transform=ax.transAxes,
                                     color=color, mfc=color if ok else "white", mew=1.5,
                                     clip_on=False, zorder=10))
        title = "original" if k == "clean" else (
            f"{'patches' if k.startswith('patch') else 'pixels'} {k.split('_')[1]}×{k.split('_')[1]}")
        axes[0, j].set_title(title, fontsize=9)
    handles = [Line2D([0], [0], marker="o", ls="", color=model_style(cfg, m)[0],
                      label=model_style(cfg, m)[1], ms=7) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, 0.0), title="filled dot = model still says \"frog\"",
               fontsize=8, title_fontsize=8)
    fig.suptitle("E6 -- scrambled frogs", y=1.0)
    # The dots sit outside each image's axes, so tight_layout cannot see them:
    # leave explicit room between rows (h_pad) and above the legend (rect).
    fig.tight_layout(rect=(0, 0.75 / fig_h, 1, 1), h_pad=2.6)
    save_fig(fig, "patch_shuffle_examples.png", cfg)


# ---------------------------------------------------------------------------
# CLI -- local testing; the queue calls run_e5 / run_e6 directly.
# ---------------------------------------------------------------------------


def _main() -> int:
    ap = argparse.ArgumentParser(description="Run E5 (corruptions) or E6 (patch shuffle).")
    ap.add_argument("--exp", choices=["E5", "E6"], required=True)
    ap.add_argument("--max-images", type=int, help="stratified test subset (testing only)")
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    fn = run_e5 if args.exp == "E5" else run_e6
    fn(max_images=args.max_images, models=args.models,
       out_dir=f"{results_dir()}/runs/{args.exp}")
    return 0


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    sys.exit(_main())

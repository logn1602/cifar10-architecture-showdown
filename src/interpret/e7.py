"""E7 -- how each architecture "sees". Every figure listed in spec E7.

Uses the E1 seed-0 checkpoints and one fixed set of 16 test images (2 per class
for 8 classes, chosen by seed and saved -- see
:func:`src.analysis.fixed_image_indices`).

Each figure is its own step. If one fails (say, UMAP will not import), the
others are still produced, the failure is written to ``summary.json``, and the
job is then reported as failed so it cannot go unnoticed in RUN_SUMMARY.md.
The arrays behind the figures (CAMs, attention maps, UMAP coordinates) are
saved to ``e7_arrays.npz`` for the Phase 9 report.

Run directly for a local test:  ``python -m src.interpret.e7 --embed-images 500``
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

try:
    from src.analysis import (
        CLASSES, Predictor, e1_runs, eval_device, fixed_image_indices, load_run_model,
        model_style, pyplot, save_fig, stratified_indices, table_path, test_set, write_csv,
    )
    from src.experiments import load_base_config, results_dir
    from src.interpret.attention import pos_similarity, upsample_maps, vit_attention
    from src.interpret.embeddings import penultimate_features, silhouette, umap_2d
    from src.interpret.feature_maps import filter_mosaic, first_layer_weights, top_channels
    from src.interpret.gradcam import GradCAM, crosscheck
    from src.interpret.predictions import agreement_matrix, confident_mistakes, load_test_outputs
    from src.models import MODEL_ORDER
    from src.utils import ensure_dir, human_time, save_json
except ImportError:  # pragma: no cover - `python src/interpret/e7.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.analysis import (  # noqa: E402
        CLASSES, Predictor, e1_runs, eval_device, fixed_image_indices, load_run_model,
        model_style, pyplot, save_fig, stratified_indices, table_path, test_set, write_csv,
    )
    from src.experiments import load_base_config, results_dir  # noqa: E402
    from src.interpret.attention import pos_similarity, upsample_maps, vit_attention  # noqa: E402
    from src.interpret.embeddings import penultimate_features, silhouette, umap_2d  # noqa: E402
    from src.interpret.feature_maps import (  # noqa: E402
        filter_mosaic, first_layer_weights, top_channels,
    )
    from src.interpret.gradcam import GradCAM, crosscheck  # noqa: E402
    from src.interpret.predictions import (  # noqa: E402
        agreement_matrix, confident_mistakes, load_test_outputs,
    )
    from src.models import MODEL_ORDER  # noqa: E402
    from src.utils import ensure_dir, human_time, save_json  # noqa: E402

__all__ = ["run_e7"]

GRADCAM_MODELS = ("simple_cnn", "resnet18", "convnext_mini")
FEATURE_MAP_FILES = {"simple_cnn": "feature_maps_simple_cnn.png",
                     "resnet18": "feature_maps_resnet.png"}


# ---------------------------------------------------------------------------
# Small plotting helpers
# ---------------------------------------------------------------------------


def _blank(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _pred_title(ax, pred: int, prob: float, true: int, fontsize: int = 7) -> None:
    ok = pred == true
    ax.set_title(f"{CLASSES[pred]} {prob:.0%}" + ("" if ok else " ✗"), fontsize=fontsize,
                 color="#1a7f37" if ok else "#c62828")


def _overlay(ax, image: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> None:
    ax.imshow(image, interpolation="nearest")
    ax.imshow(heat, cmap="jet", alpha=alpha, vmin=0, vmax=1, interpolation="bilinear")
    _blank(ax)


def _blocks(n: int, per_row: int = 8) -> list[range]:
    return [range(s, min(n, s + per_row)) for s in range(0, n, per_row)]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _fig_fixed(images, labels, fixed, cfg) -> dict:
    plt = pyplot()
    fig, axes = plt.subplots(2, 8, figsize=(11, 3.75))
    for ax, i in zip(axes.flat, fixed):
        ax.imshow(images[i], interpolation="nearest")
        ax.set_title(f"{CLASSES[labels[i]]}  #{i}", fontsize=8)
        _blank(ax)
    fig.suptitle("E7 -- the 16 fixed test images (2 per class, chosen by seed)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fixed_images.png", cfg)
    return {}


def _fig_filters(nets, cfg) -> dict:
    plt = pyplot()
    panels = [(m, first_layer_weights(m, nets[m])) for m in MODEL_ORDER if m in nets]
    panels = [(m, p) for m, p in panels if p is not None]
    if not panels:
        return {"skipped": "no model with a spatial first layer"}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.9), squeeze=False)
    for ax, (m, (w, desc)) in zip(axes[0], panels):
        mosaic, n_total = filter_mosaic(w, max_filters=64)
        if mosaic.shape[-1] == 1:
            ax.imshow(mosaic[..., 0], cmap="gray", interpolation="nearest")
        else:
            ax.imshow(mosaic, interpolation="nearest")
        shown = min(64, n_total)
        note = f"{shown} filters" if shown == n_total else f"{shown} of {n_total}, largest norm"
        ax.set_title(f"{model_style(cfg, m)[1]}\n{desc}\n({note})", fontsize=9,
                     color=model_style(cfg, m)[0])
        _blank(ax)
    fig.suptitle("E7 -- first-layer filters (each scaled to its own range)", y=1.03)
    fig.tight_layout()
    save_fig(fig, "first_layer_filters.png", cfg)
    return {"models": [m for m, _ in panels]}


def _fig_feature_maps(nets, pred, images, labels, fixed, cfg) -> dict:
    plt = pyplot()
    frog = next((i for i in fixed if CLASSES[labels[i]] == "frog"), fixed[0])
    done = []
    for m, fname in FEATURE_MAP_FILES.items():
        if m not in nets:
            continue
        maps = nets[m].feature_maps(pred.to_input(images[[frog]]))
        n_ch = 8
        # Input on the left (spanning all rows), a label column, then channels.
        fig = plt.figure(figsize=(1.2 * n_ch + 3.6, 1.3 * len(maps) + 0.7))
        gs = fig.add_gridspec(len(maps), n_ch + 2, width_ratios=[2.6, 1.1] + [1] * n_ch,
                              wspace=0.08, hspace=0.22)
        ax_in = fig.add_subplot(gs[:, 0])
        ax_in.imshow(images[frog], interpolation="nearest")
        ax_in.set_title("input", fontsize=9)
        _blank(ax_in)
        for r, fm in enumerate(maps):
            chans, _ = top_channels(fm[0], n_ch)
            c, h, w = fm.shape[1:]
            name = ("stem" if r == 0 else f"stage {r}") if m == "resnet18" else f"stage {r + 1}"
            lab = fig.add_subplot(gs[r, 1])
            lab.axis("off")
            lab.text(0.95, 0.5, f"{name}\n{c}×{h}×{w}", ha="right", va="center", fontsize=8,
                     transform=lab.transAxes)
            for q in range(n_ch):
                ax = fig.add_subplot(gs[r, q + 2])
                ax.imshow(chans[q], cmap="viridis", interpolation="nearest")
                _blank(ax)
                if r == 0:
                    ax.set_title(f"#{q + 1}", fontsize=7)
        fig.suptitle(f"E7 -- {model_style(cfg, m)[1]} feature maps on one frog "
                     "(the 8 most active channels per stage)", y=0.98)
        save_fig(fig, fname, cfg)
        done.append(m)
    return {"models": done, "image_index": int(frog)}


def _gradcam(nets, pred, images, labels, fixed, cfg, arrays) -> dict:
    plt = pyplot()
    present = [m for m in GRADCAM_MODELS if m in nets]
    if not present:
        return {"skipped": "no conv model among the checkpoints"}
    x = pred.to_input(images[fixed])                 # fp32, no autocast: explanations
    native = {}
    for m in present:
        with GradCAM(nets[m], nets[m].gradcam_layer) as cam:
            res = cam(x)
        arrays[f"gradcam_{m}"] = res["cam"]
        arrays[f"gradcam_{m}_pred"] = res["cls"]
        arrays[f"gradcam_{m}_prob"] = res["prob"]
        native[m] = list(res["native_resolution"])
    check_model = "resnet18" if "resnet18" in nets else present[0]
    xcheck = crosscheck(nets[check_model], nets[check_model].gradcam_layer, x)
    xcheck["model"] = check_model

    blocks = _blocks(len(fixed))
    rows = 1 + len(present)
    fig, axes = plt.subplots(rows * len(blocks), 8, figsize=(11, 1.55 * rows * len(blocks)),
                             squeeze=False)
    for b, cols in enumerate(blocks):
        for q, k in enumerate(cols):
            i = fixed[k]
            ax = axes[b * rows, q]
            ax.imshow(images[i], interpolation="nearest")
            ax.set_title(CLASSES[labels[i]], fontsize=8)
            _blank(ax)
            for r, m in enumerate(present, start=1):
                ax = axes[b * rows + r, q]
                _overlay(ax, images[i], arrays[f"gradcam_{m}"][k])
                _pred_title(ax, int(arrays[f"gradcam_{m}_pred"][k]),
                            float(arrays[f"gradcam_{m}_prob"][k]), int(labels[i]))
                if q == 0:
                    ax.set_ylabel(f"{model_style(cfg, m)[1]}\n({native[m][0]}×{native[m][1]})",
                                  fontsize=8, color=model_style(cfg, m)[0])
    fig.suptitle("E7 -- Grad-CAM on the last conv stage, for each model's own prediction "
                 "(native map size in brackets)", y=1.0)
    fig.tight_layout()
    save_fig(fig, "gradcam.png", cfg)
    return {"models": present, "native_resolution": native, "crosscheck_vs_pytorch_grad_cam": xcheck}


def _vit(nets, pred, images, labels, fixed, cfg, arrays) -> dict:
    if "vit" not in nets:
        return {"skipped": "no ViT checkpoint"}
    plt = pyplot()
    vit = nets["vit"]
    att = vit_attention(vit, pred.to_input(images[fixed]))
    heads_up = upsample_maps(att["heads"])                     # (16, 6, 32, 32)
    roll_up = upsample_maps(att["rollout"])                    # (16, 32, 32)
    probs = torch.from_numpy(att["logits"]).softmax(1).numpy()
    arrays.update(vit_heads=att["heads"], vit_rollout=att["rollout"], vit_rollout_up=roll_up,
                  vit_pred=probs.argmax(1), vit_prob=probs.max(1))

    # Per-head last-layer attention for a spread of the fixed images.
    n = int(cfg["experiments"]["e7"]["attention_images"])
    pick = np.linspace(0, len(fixed) - 1, n).round().astype(int)
    n_heads = heads_up.shape[1]
    fig, axes = plt.subplots(n, n_heads + 2, figsize=(1.45 * (n_heads + 2), 1.6 * n), squeeze=False)
    for r, k in enumerate(pick):
        i = fixed[k]
        axes[r, 0].imshow(images[i], interpolation="nearest")
        axes[r, 0].set_ylabel(CLASSES[labels[i]], fontsize=8)
        _blank(axes[r, 0])
        for h in range(n_heads):
            _overlay(axes[r, h + 1], images[i], heads_up[k, h])
        _overlay(axes[r, -1], images[i], roll_up[k])
        _pred_title(axes[r, -1], int(arrays["vit_pred"][k]), float(arrays["vit_prob"][k]),
                    int(labels[i]))
    for h in range(n_heads):
        axes[0, h + 1].set_title(f"head {h + 1}", fontsize=8)
    axes[0, 0].set_title("input", fontsize=8)
    fig.suptitle("E7 -- ViT: last-layer cls attention per head, and attention rollout "
                 "(rightmost; titled with the prediction)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "vit_attention.png", cfg)

    # Positional-embedding similarity: an 8x8 grid of 8x8 maps.
    sim = pos_similarity(vit)                                  # (64, 64)
    g = vit.grid_size
    arrays["vit_pos_similarity"] = sim
    fig, axes = plt.subplots(g, g, figsize=(7.5, 7.5))
    for p in range(g * g):
        ax = axes[p // g, p % g]
        im = ax.imshow(sim[p].reshape(g, g), cmap="viridis", vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.plot(p % g, p // g, "s", ms=3.5, mfc="none", mec="white", mew=0.8)
        _blank(ax)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="cosine similarity")
    fig.suptitle("E7 -- ViT positional embeddings: similarity of each position (white box) "
                 "to every other", y=0.93)
    save_fig(fig, "vit_pos_embedding_similarity.png", cfg)

    # One number for "did it learn 2-D structure?": similarity to the 4-connected
    # neighbours vs. to positions at least 3 steps away.
    rr, cc = np.divmod(np.arange(g * g), g)
    dist = np.abs(rr[:, None] - rr[None]) + np.abs(cc[:, None] - cc[None])
    return {"pos_sim_neighbours": float(sim[dist == 1].mean()),
            "pos_sim_far": float(sim[dist >= 3].mean())}


def _fig_side_by_side(images, labels, fixed, arrays, cfg) -> dict:
    if "gradcam_resnet18" not in arrays or "vit_rollout_up" not in arrays:
        return {"skipped": "needs both the ResNet-18 and the ViT checkpoint"}
    plt = pyplot()
    blocks = _blocks(len(fixed))
    fig, axes = plt.subplots(3 * len(blocks), 8, figsize=(11, 1.5 * 3 * len(blocks)), squeeze=False)
    rc, rl = model_style(cfg, "resnet18"), model_style(cfg, "vit")
    for b, cols in enumerate(blocks):
        for q, k in enumerate(cols):
            i = fixed[k]
            ax = axes[3 * b, q]
            ax.imshow(images[i], interpolation="nearest")
            ax.set_title(CLASSES[labels[i]], fontsize=8)
            _blank(ax)
            _overlay(axes[3 * b + 1, q], images[i], arrays["gradcam_resnet18"][k])
            _pred_title(axes[3 * b + 1, q], int(arrays["gradcam_resnet18_pred"][k]),
                        float(arrays["gradcam_resnet18_prob"][k]), int(labels[i]))
            _overlay(axes[3 * b + 2, q], images[i], arrays["vit_rollout_up"][k])
            _pred_title(axes[3 * b + 2, q], int(arrays["vit_pred"][k]),
                        float(arrays["vit_prob"][k]), int(labels[i]))
        axes[3 * b + 1, 0].set_ylabel(f"{rc[1]}\nGrad-CAM", fontsize=8, color=rc[0])
        axes[3 * b + 2, 0].set_ylabel(f"{rl[1]}\nattention rollout", fontsize=8, color=rl[0])
    fig.suptitle("E7 -- same images, two brains: where a CNN and a Transformer look", y=1.0)
    fig.tight_layout()
    save_fig(fig, "gradcam_vs_attention.png", cfg)
    return {}


def _embeddings(nets, pred, images, labels, cfg, n_images, arrays) -> dict:
    plt = pyplot()
    e7 = cfg["experiments"]["e7"]
    u = e7["umap"]
    idx = stratified_indices(labels, int(n_images or e7["embed_images"]), seed=int(u["seed"]))
    y = labels[idx]
    arrays["umap_indices"] = idx
    rows = []
    models = [m for m in MODEL_ORDER if m in nets]
    for m in models:
        feats = penultimate_features(nets[m], pred, images[idx])
        coords = umap_2d(feats, u["n_neighbors"], u["min_dist"], u["seed"])
        arrays[f"umap_{m}"] = coords
        rows.append({"model": m, "feature_dim": feats.shape[1],
                     "silhouette_cosine": silhouette(feats, y)})
    write_csv(rows, table_path("embedding_separability.csv"))

    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, m, r in zip(axes.flat, models, rows):
        c = arrays[f"umap_{m}"]
        ax.scatter(c[:, 0], c[:, 1], c=[cmap(int(k)) for k in y], s=2.5, linewidths=0)
        color, label = model_style(cfg, m)
        ax.set_title(f"{label}  (silhouette {r['silhouette_cosine']:.2f})", color=color, fontsize=10)
        _blank(ax)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=cmap(k), label=CLASSES[k], ms=6)
               for k in range(len(CLASSES))]
    fig.legend(handles=handles, loc="lower center", ncol=len(CLASSES), frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.suptitle(f"E7 -- UMAP of penultimate features, {len(idx)} test images "
                 "(silhouette measured in the original feature space)", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_fig(fig, "umap_embeddings.png", cfg)
    return {"n_images": int(len(idx)),
            "silhouette": {r["model"]: round(r["silhouette_cosine"], 4) for r in rows}}


def _mistakes(rundirs, images, cfg) -> dict:
    plt = pyplot()
    k = int(cfg["experiments"]["e7"]["mistakes_per_model"])
    done = []
    for m, d in rundirs.items():
        if not (d / "test_outputs.npz").exists():
            continue
        out = load_test_outputs(d)
        idx = confident_mistakes(out["probs"], out["targets"], k)
        cols = 6
        rows_n = int(np.ceil(len(idx) / cols)) or 1
        # Square images are width-limited, so the height must leave room for
        # the two-line titles or the rows overlap.
        fig, axes = plt.subplots(rows_n, cols, figsize=(1.75 * cols, 2.45 * rows_n + 0.3),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, idx):
            p = int(out["probs"][i].argmax())
            ax.imshow(images[i], interpolation="nearest")
            ax.set_title(f"said {CLASSES[p]} {out['probs'][i, p]:.0%}\nwas {CLASSES[out['targets'][i]]}",
                         fontsize=8, color="#c62828")
            ax.axis("on")
            _blank(ax)
        color, label = model_style(cfg, m)
        fig.suptitle(f"E7 -- {label}: its {len(idx)} most confident mistakes", y=1.02, color=color)
        fig.tight_layout()
        save_fig(fig, f"mistake_gallery_{m}.png", cfg)
        done.append(m)
    return {"models": done}


def _agreement(rundirs, cfg) -> dict:
    plt = pyplot()
    outs = {m: load_test_outputs(d) for m, d in rundirs.items()
            if (d / "test_outputs.npz").exists()}
    if len(outs) < 2:
        return {"skipped": "fewer than two models with test predictions"}
    names, mat = agreement_matrix({m: o["preds"] for m, o in outs.items()})
    accs = {m: float((o["preds"] == o["targets"]).mean()) for m, o in outs.items()}
    write_csv([{"model": a, **{b: round(float(mat[i, j]), 4) for j, b in enumerate(names)}}
               for i, a in enumerate(names)], table_path("model_agreement.csv"))

    fig, ax = plt.subplots(figsize=(1.05 * len(names) + 2.2, 0.95 * len(names) + 1.2))
    im = ax.imshow(mat * 100, cmap="Blues", vmin=max(0, mat.min() * 100 - 5), vmax=100)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat[i, j] * 100
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if v > (mat.min() * 100 + 100) / 2 else "black")
    labels_ = [f"{model_style(cfg, m)[1]}\n(acc {accs[m]:.1%})" for m in names]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([model_style(cfg, m)[1] for m in names], rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels_, fontsize=8)
    fig.colorbar(im, ax=ax, label="% of test images with the same prediction", fraction=0.04)
    ax.set_title("E7 -- how often do two models give the same answer?")
    fig.tight_layout()
    save_fig(fig, "model_agreement.png", cfg)
    off = mat[~np.eye(len(names), dtype=bool)]
    return {"min_pair_agreement": float(off.min()), "max_pair_agreement": float(off.max())}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_e7(cfg: dict | None = None, mode: str = "9h", out_dir: str = "results/runs/E7",
           embed_images: int | None = None, models=None) -> dict:
    cfg = cfg or load_base_config()
    t0 = time.time()
    out = ensure_dir(out_dir)                  # repo-relative, as the queue passes it
    device = eval_device()
    pred = Predictor(cfg, device)
    runs = e1_runs(models or MODEL_ORDER, seeds=[0])
    if not runs:
        raise RuntimeError("E7: no completed E1 seed-0 runs with best.pt")
    images, labels = test_set(cfg)
    fixed = fixed_image_indices(cfg, labels)
    nets = {r.model: load_run_model(r, device) for r in runs}
    rundirs = {r.model: r.dir for r in runs}
    print(f"E7 on {', '.join(nets)} | {device}", flush=True)

    summary: dict = {"models": list(nets), "fixed_indices": [int(i) for i in fixed], "steps": {}}
    arrays: dict[str, np.ndarray] = {}

    def step(name: str, fn) -> None:
        t = time.time()
        try:
            res = fn() or {}
            summary["steps"][name] = {"status": "skipped" if "skipped" in res else "ok",
                                      "seconds": round(time.time() - t, 1), **res}
            print(f"  E7 {name:<22} {summary['steps'][name]['status']} "
                  f"({time.time() - t:.1f}s)", flush=True)
        except Exception as exc:
            traceback.print_exc()
            summary["steps"][name] = {"status": "failed", "error": repr(exc)}
            print(f"  E7 {name:<22} FAILED: {exc!r}", flush=True)

    step("fixed_images", lambda: _fig_fixed(images, labels, fixed, cfg))
    step("first_layer_filters", lambda: _fig_filters(nets, cfg))
    step("feature_maps", lambda: _fig_feature_maps(nets, pred, images, labels, fixed, cfg))
    step("gradcam", lambda: _gradcam(nets, pred, images, labels, fixed, cfg, arrays))
    step("vit_attention", lambda: _vit(nets, pred, images, labels, fixed, cfg, arrays))
    step("gradcam_vs_attention", lambda: _fig_side_by_side(images, labels, fixed, arrays, cfg))
    step("embeddings", lambda: _embeddings(nets, pred, images, labels, cfg, embed_images, arrays))
    step("mistake_galleries", lambda: _mistakes(rundirs, images, cfg))
    step("agreement", lambda: _agreement(rundirs, cfg))

    if arrays:
        np.savez_compressed(out / "e7_arrays.npz", **arrays)
    summary["seconds"] = time.time() - t0
    save_json(summary, out / "summary.json")
    failed = [k for k, v in summary["steps"].items() if v["status"] == "failed"]
    print(f"E7 done in {human_time(time.time() - t0)}"
          + (f" -- FAILED steps: {failed}" if failed else ""), flush=True)
    if failed:
        raise RuntimeError(f"E7 steps failed: {failed}; the other figures were written "
                           f"(see {out / 'summary.json'})")
    return summary


def _main() -> int:
    ap = argparse.ArgumentParser(description="Run E7 (interpretability) on the E1 checkpoints.")
    ap.add_argument("--embed-images", type=int, help="UMAP image count (default from base.yaml)")
    ap.add_argument("--models", nargs="+")
    args = ap.parse_args()
    run_e7(embed_images=args.embed_images, models=args.models, out_dir=f"{results_dir()}/runs/E7")
    return 0


if __name__ == "__main__":  # Windows spawn guard (spec 1.3)
    sys.exit(_main())

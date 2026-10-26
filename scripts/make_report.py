"""Phase 9: rebuild every table, the remaining figures and ``results/REPORT.md``.

Reads only what the runs saved -- ``metrics.json``, ``history.csv``,
``config.yaml`` and the E5/E6/E7 summaries -- so it works on the unpacked
``results_light.zip`` from Kaggle, without checkpoints or a GPU. No number in
the report is typed by hand: every figure, table cell and takeaway sentence is
computed here, and re-running the script after more runs land refreshes all of
it.

What it produces (spec Section 8 numbering):

* figures 1-7 and 15: ``scoreboard``, ``acc_vs_params``, ``acc_vs_flops``,
  ``learning_curves``, ``data_hunger``, ``aug_gain``, ``pretrained_vs_scratch``,
  ``confusion_matrices``. Figures 8-14, 16 and 17 are drawn by the E5/E6/E7
  evaluation runs themselves; this script only checks that they exist.
* tables: ``e1_scoreboard``, ``e1_per_class``, ``lr_sweep``, ``e2_data_hunger``,
  ``e3_aug_gain``, ``e4_pretrained``, ``compute_used`` (``.csv`` + ``.md``).
* ``results/REPORT.md``: question, setup, result and takeaway per experiment,
  plus the 9h-mode caveats (spec 12.7) when the run was a 9h run.
* ``tables/headline_numbers.json``: named, formatted numbers for the
  hand-written prose. In ``README.md`` and ``results/linkedin_drafts.md``,
  every ``<!-- n:KEY -->...<!-- /n -->`` span is refreshed from it, and every
  ``<!-- block:NAME -->...<!-- /block:NAME -->`` gets the table in
  ``tables/NAME.md``. The prose is written by hand; the numbers in it never are.

FLOPs are recounted locally from each run's own ``config.yaml``: the count
depends only on the architecture and input size, and ``FlopCounterMode`` failed
on one Kaggle image, so Kaggle's ``metrics.json`` may hold ``flops: null``.

Missing experiments are reported as missing, never filled in, so the script is
safe to run on a partial queue. A partly run experiment (E3 on fewer than six
models) says so in its heading and takeaway.

No takeaway calls a gap a win when one seed cannot separate it: under
``NOISE_PP`` it is a tie, up to ``NEAR_PP`` a near-tie, and tied models are
named together rather than one of them picked as the winner.

Run: ``python scripts/make_report.py``
     ``SHOWDOWN_RESULTS=results/_report_test python scripts/make_report.py``
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import CLASSES, model_style, pyplot, read_csv, save_fig, table_path, write_csv  # noqa: E402
from src.experiments import load_base_config, results_dir, results_path  # noqa: E402
from src.models import MODEL_ORDER, build_model  # noqa: E402
from src.profile import try_count_flops  # noqa: E402
from src.utils import load_config, load_json, save_json  # noqa: E402

# Presentation groups for the takeaways. "Patch models" are the two that see
# the image as a set of patches with no built-in 2-D locality.
CNN_FAMILY = ("simple_cnn", "resnet18", "convnext_mini")
PATCH_FAMILY = ("vit", "mlp_mixer")

# Spec 12.7: with one seed, gaps under ~1 percentage point are noise.
NOISE_PP = 1.0
# Gaps from NOISE_PP up to NEAR_PP are near-ties: still too close to call a win.
NEAR_PP = 1.5

# E4 pretrained model -> the from-scratch model it is compared with.
E4_COUNTERPART = {"vit_small_patch16_224": "vit", "resnet50": "resnet18"}
E4_LABEL = {"vit_small_patch16_224": "ViT-S/16 (IN-21k → IN-1k)",
            "resnet50": "ResNet-50 (IN-1k, A1)"}

# Figures drawn elsewhere (E5/E6/E7) that the report links to.
EVAL_FIGURES = {
    "E5": ["robustness_heatmap.png", "robustness_curves.png", "robustness_examples.png"],
    "E6": ["patch_shuffle.png", "patch_shuffle_examples.png"],
    "E7": ["fixed_images.png", "first_layer_filters.png", "feature_maps_resnet.png",
           "gradcam_vs_attention.png", "vit_attention.png", "vit_pos_embedding_similarity.png",
           "umap_embeddings.png", "model_agreement.png"],
}


# ---------------------------------------------------------------------------
# Loading runs
# ---------------------------------------------------------------------------


def _num(text: str) -> float:
    return float(text.split("_", 1)[1])


def load_runs(root: Path) -> dict[str, list[dict]]:
    """Every completed run under ``runs/``, grouped by experiment.

    Identity (model, seed, fraction, LR) comes from the directory layout that
    :func:`src.experiments.build_queue` defines, so a run is classified the
    same way the queue that produced it classified it.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    runs = root / "runs"
    if not runs.is_dir():
        return out
    for mp in sorted(runs.rglob("metrics.json")):
        parts = mp.parent.relative_to(runs).parts
        try:
            m = load_json(mp)
        except Exception as exc:
            print(f"  ! unreadable {mp}: {exc}")
            continue
        if m.get("status") != "completed":
            continue
        exp = parts[0]
        r = dict(m, exp=exp, dir=mp.parent)
        try:
            if exp in ("E1", "E3") and len(parts) == 3:
                r.update(model=parts[1], seed=int(parts[2]))
            elif exp == "E2" and len(parts) == 4:
                r.update(model=parts[1], fraction=_num(parts[2]), seed=int(parts[3]))
            elif exp == "E4" and len(parts) == 3:
                r.update(model=parts[1], seed=int(parts[2]), timm_name=m.get("model"))
            elif exp in ("lr_sweep", "E4_sweep") and len(parts) == 3:
                r.update(model=parts[1], lr=_num(parts[2]))
            else:
                continue
        except ValueError:
            continue
        out[exp].append(r)
    return out


def read_history(run: dict) -> list[dict]:
    path = Path(run["dir"]) / "history.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    return [r for r in rows if isinstance(r.get("val_acc"), float)]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def mean_std(xs: Iterable[float | None]) -> tuple[float, float, int]:
    """Mean, sample std (ddof=1; 0 for a single value) and count, ignoring None."""
    v = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)


def pct(x: float | None, nd: int = 1) -> str:
    return "–" if x is None or not math.isfinite(x) else f"{100 * x:.{nd}f}"


def pm(mean: float, std: float, n: int) -> str:
    """``"93.4 ± 0.2"`` over several seeds, ``"93.4"`` for one."""
    if n == 0:
        return "–"
    return f"{pct(mean)} ± {pct(std)}" if n > 1 else pct(mean)


def label(cfg: dict, model: str) -> str:
    return cfg["figures"]["model_labels"].get(model, model)


def models_in(runs: list[dict]) -> list[str]:
    present = {r["model"] for r in runs}
    return [m for m in MODEL_ORDER if m in present]


def join(words: list[str]) -> str:
    """``"A"``, ``"A and B"``, ``"A, B and C"``."""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def tie_note(spread_pp: float, noise: float = NOISE_PP) -> str:
    """``"effectively tied"`` inside the noise band, ``"a near-tie"`` up to
    ``NEAR_PP`` past it, and ``""`` for a gap one seed can call."""
    if spread_pp < noise:
        return "effectively tied"
    return "a near-tie" if spread_pp < noise + NEAR_PP - NOISE_PP else ""


def close_to_extreme(vals: dict[Any, float], highest: bool = True) -> list:
    """Keys within the near-tie band of the highest (or lowest) value, extreme first.

    Values are in points (accuracies and shares x100), so the pp bands apply.
    """
    order = sorted(vals, key=vals.get, reverse=highest)
    return [k for k in order if abs(vals[k] - vals[order[0]]) < NEAR_PP]


def ranked_group(cfg: dict, vals: dict[str, float], highest: bool = True, nd: int = 0,
                 unit: str = "%", div: float = 1.0) -> tuple[list[str], str]:
    """The model(s) at one extreme of ``vals`` and how to name them: ``"ViT (61%)"``,
    or tied models in model order, e.g. ``"Simple CNN and ResNet-18 (21% and 22%,
    effectively tied)"``. Shown values are ``vals / div`` to ``nd`` decimals."""
    group = close_to_extreme(vals, highest)
    shown = {m: f"{vals[m] / div:.{nd}f}" for m in group}
    if len(group) == 1:
        return group, f"{label(cfg, group[0])} ({shown[group[0]]}{unit})"
    spread = max(vals[m] for m in group) - min(vals[m] for m in group)
    group = sorted(group, key=MODEL_ORDER.index)
    vs = [shown[m] for m in group]
    value = f"{vs[0]}{unit} each" if len(set(vs)) == 1 else join([v + unit for v in vs])
    return group, f"{join([label(cfg, m) for m in group])} ({value}, {tie_note(spread)})"


def run_model_cfg(e1: list[dict], m: str) -> dict:
    """The model config one of ``m``'s own E1 runs was trained with (its
    ``config.yaml``), or the repo default for ``m`` when no run saved one."""
    run = next((r for r in e1 if r["model"] == m), None)
    if run is not None and (Path(run["dir"]) / "config.yaml").exists():
        with (Path(run["dir"]) / "config.yaml").open(encoding="utf-8") as fh:
            mc = (yaml.safe_load(fh) or {}).get("model_cfg") or {}
        if mc.get("model", mc):
            return mc.get("model", mc)
    return load_config(f"configs/models/{m}.yaml")["model"]


def local_flops(e1: list[dict]) -> dict[str, int | None]:
    """FLOPs per image for the six models, counted on this machine.

    Each model is rebuilt from the ``config.yaml`` of one of its own E1 runs,
    so the count belongs to the architecture that was actually trained.
    """
    flops: dict[str, int | None] = {}
    import torch
    cpu = torch.device("cpu")
    for m in MODEL_ORDER:
        value, err = try_count_flops(build_model(m, run_model_cfg(e1, m)), cpu)
        if value is None:
            value = next((r.get("flops") for r in e1 if r["model"] == m), None)
            print(f"  ! local FLOP count failed for {m} ({err}); using the run's value {value}")
        flops[m] = value
    return flops


def patch_grid(e1: list[dict]) -> dict[str, tuple[int, int]]:
    """``{model: (patch size, image size)}`` in pixels for the patch models, as trained."""
    out = {}
    for m in PATCH_FAMILY:
        mc = run_model_cfg(e1, m)
        if mc.get("patch_size"):
            out[m] = (int(mc["patch_size"]), int(mc.get("img_size", 32)))
    return out


def pretrain_data(timm_name: str | None) -> str | None:
    """Pretraining set named by a timm weight tag (``...augreg_in21k_ft_in1k``)."""
    tag = (timm_name or "").lower()
    if "in21k" in tag or "in22k" in tag:
        return "ImageNet-21k"
    return "ImageNet-1k" if "in1k" in tag else None


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_md(name: str, title: str, note: str, table: str) -> Path:
    path = table_path(name)
    path.write_text(f"# {title}\n\n_Generated by `scripts/make_report.py`. {note}_\n\n{table}\n",
                    encoding="utf-8")
    return path


def fig_link(name: str) -> str:
    return f"![{name}](figures/{name})"


# ---------------------------------------------------------------------------
# E1 -- main showdown
# ---------------------------------------------------------------------------


def e1_rows(cfg: dict, e1: list[dict], flops: dict[str, int | None]) -> list[dict]:
    rows = []
    for m in models_in(e1):
        rs = [r for r in e1 if r["model"] == m]
        acc, acc_sd, n = mean_std(r["test_acc"] for r in rs)
        ece = mean_std(r.get("ece") for r in rs)[0]
        rows.append({
            "model": m, "label": label(cfg, m), "seeds": n,
            "test_acc": acc, "test_acc_std": acc_sd,
            "best_val_acc": mean_std(r["best_val_acc"] for r in rs)[0],
            "params": rs[0].get("params"), "flops": flops.get(m),
            "lr": rs[0].get("lr"), "epochs": rs[0].get("epochs"), "ece": ece,
            "sec_per_epoch": mean_std(r.get("sec_per_epoch") for r in rs)[0],
            "train_time_h": mean_std(r.get("train_time_sec") for r in rs)[0] / 3600,
            "train_imgs_per_sec": mean_std(r.get("train_imgs_per_sec") for r in rs)[0],
            "infer_imgs_per_sec": mean_std(r.get("infer_imgs_per_sec") for r in rs)[0],
            "peak_mem_mb": mean_std(r.get("peak_mem_mb") for r in rs)[0],
            "gpu": (rs[0].get("device") or {}).get("gpu"),
        })
    return rows


def _fmt_int(x: Any) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.0f}"


def e1_tables(cfg: dict, rows: list[dict], e1: list[dict]) -> str:
    write_csv(rows, table_path("e1_scoreboard.csv"))
    ranked = sorted(rows, key=lambda r: -r["test_acc"])
    table = md_table(
        ["rank", "model", "test acc (%)", "seeds", "params", "GFLOPs/img", "LR", "ECE",
         "sec/epoch", "train time (h)", "train img/s", "infer img/s", "peak mem (MB)"],
        [[i + 1, f"**{r['label']}**", pm(r["test_acc"], r["test_acc_std"], r["seeds"]),
          r["seeds"], _fmt_int(r["params"]),
          "–" if r["flops"] is None else f"{r['flops'] / 1e9:.3f}",
          f"{r['lr']:g}" if r["lr"] else "–", f"{r['ece']:.3f}" if math.isfinite(r["ece"]) else "–",
          f"{r['sec_per_epoch']:.1f}", f"{r['train_time_h']:.2f}",
          _fmt_int(r["train_imgs_per_sec"]), _fmt_int(r["infer_imgs_per_sec"]),
          _fmt_int(r["peak_mem_mb"])] for i, r in enumerate(ranked)])
    write_md("e1_scoreboard.md", "E1 -- main showdown",
             "Test accuracy is mean ± sample std over seeds (best-val checkpoint, test touched "
             "once per run). Timings are per run on one GPU.", table)
    # The README-width version of the same table.
    write_md("e1_scoreboard_short.md", "E1 -- scoreboard (short)", "See e1_scoreboard.md.",
             md_table(["rank", "model", "test acc (%)", "params (M)", "GFLOPs/img",
                       "train time (h)"],
                      [[i + 1, f"**{r['label']}**",
                        pm(r["test_acc"], r["test_acc_std"], r["seeds"]),
                        "–" if not r["params"] else f"{r['params'] / 1e6:.1f}",
                        "–" if r["flops"] is None else f"{r['flops'] / 1e9:.2f}",
                        f"{r['train_time_h']:.2f}"] for i, r in enumerate(ranked)]))

    # Per-class accuracy, mean over seeds.
    pc_rows = []
    for m in models_in(e1):
        rs = [r for r in e1 if r["model"] == m and r.get("per_class_acc")]
        if rs:
            pc_rows.append([label(cfg, m)] + [
                pct(mean_std(r["per_class_acc"].get(c) for r in rs)[0]) for c in CLASSES])
    if pc_rows:
        write_md("e1_per_class.md", "E1 -- per-class test accuracy (%)", "Mean over seeds.",
                 md_table(["model", *CLASSES], pc_rows))
    return table


def fig_scoreboard(cfg: dict, rows: list[dict]) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        c, lab = model_style(cfg, r["model"])
        ax.bar(i, 100 * r["test_acc"], color=c, width=0.65,
               yerr=100 * r["test_acc_std"] if r["seeds"] > 1 else None, capsize=4)
        ax.text(i, 100 * (r["test_acc"] + r["test_acc_std"]) + 0.6, pct(r["test_acc"]),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [r["label"] for r in rows])
    lo = min(100 * r["test_acc"] for r in rows)
    ax.set_ylim(max(0, lo - 15), 100)
    seeds = sorted({r["seeds"] for r in rows})
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("E1: CIFAR-10 test accuracy, trained from scratch "
                 f"({'mean ± std over ' + str(seeds[-1]) + ' seeds' if seeds[-1] > 1 else '1 seed'})")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "scoreboard.png", cfg)


def fig_acc_vs_cost(cfg: dict, rows: list[dict], key: str, name: str, xlabel: str) -> bool:
    pts = [r for r in rows if r.get(key)]
    if not pts:
        return False
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    tmax = max(r["train_time_h"] for r in pts) or 1.0
    for r in pts:
        c, lab = model_style(cfg, r["model"])
        x = r[key] / (1e9 if key == "flops" else 1e6)
        size = 60 + 600 * r["train_time_h"] / tmax                    # area, points^2
        ax.scatter(x, 100 * r["test_acc"], s=size, color=c, alpha=0.75, edgecolor="black",
                   linewidth=0.6)
        ax.annotate(lab, (x, 100 * r["test_acc"]), xytext=(math.sqrt(size) / 2 + 3, 0),
                    textcoords="offset points", fontsize=9, va="center")
    ax.set_xscale("log")
    ax.margins(x=0.15, y=0.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"E1: accuracy vs. {xlabel.split(' (')[0]} (bubble area = train time)")
    ax.grid(alpha=0.3, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, name, cfg)
    return True


def fig_learning_curves(cfg: dict, e1: list[dict]) -> bool:
    """Validation accuracy and train loss per data epoch, mean over seeds."""
    curves = {}
    for m in models_in(e1):
        per_epoch: dict[float, list[tuple[float, float]]] = defaultdict(list)
        for r in (r for r in e1 if r["model"] == m):
            for h in read_history(r):
                ep = h.get("data_epoch") if isinstance(h.get("data_epoch"), float) else h["epoch"]
                per_epoch[round(float(ep), 3)].append((h["val_acc"], h.get("train_loss")))
        if per_epoch:
            eps = sorted(per_epoch)
            curves[m] = (eps, [np.mean([v for v, _ in per_epoch[e]]) for e in eps],
                         [np.mean([t for _, t in per_epoch[e] if isinstance(t, float)] or [np.nan])
                          for e in eps])
    if not curves:
        return False
    plt = pyplot()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for m, (eps, val, loss) in curves.items():
        c, lab = model_style(cfg, m)
        a1.plot(eps, 100 * np.asarray(val), color=c, label=lab, lw=1.6)
        a2.plot(eps, loss, color=c, label=lab, lw=1.6)
    a1.set(xlabel="epoch", ylabel="validation accuracy (%)", title="E1: validation accuracy")
    a2.set(xlabel="epoch", ylabel="train loss (label-smoothed CE)", title="E1: training loss")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    a1.legend(fontsize=8, loc="lower right")
    save_fig(fig, "learning_curves.png", cfg)
    return True


def fig_confusion(cfg: dict, e1: list[dict]) -> bool:
    runs = {}
    for m in models_in(e1):
        rs = sorted((r for r in e1 if r["model"] == m and r.get("confusion_matrix")),
                    key=lambda r: r["seed"])
        if rs:
            runs[m] = rs[0]
    if not runs:
        return False
    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (m, r) in zip(axes.flat, runs.items()):
        cm = np.asarray(r["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.axis("on")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(10), CLASSES, rotation=60, fontsize=7)
        ax.set_yticks(range(10), CLASSES, fontsize=7)
        ax.set_title(f"{label(cfg, m)} (seed {r['seed']}, {pct(r['test_acc'])}%)",
                     color=model_style(cfg, m)[0], fontsize=10)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
    fig.suptitle("E1: confusion matrices (row-normalised, % of each true class)")
    fig.tight_layout()
    save_fig(fig, "confusion_matrices.png", cfg)
    return True


def e1_takeaway(cfg: dict, rows: list[dict]) -> str:
    ranked = sorted(rows, key=lambda r: -r["test_acc"])
    best, second = ranked[0], ranked[1] if len(ranked) > 1 else None
    s = [f"**{best['label']}** has the highest test accuracy ({pct(best['test_acc'])}%)."]
    if second:
        gap = 100 * (best["test_acc"] - second["test_acc"])
        noise = max(NOISE_PP, 2 * 100 * max(best["test_acc_std"], second["test_acc_std"]))
        close = tie_note(gap, noise)
        if close == "effectively tied":
            s.append(f"Its lead over {second['label']} ({pct(second['test_acc'])}%) is "
                     f"{gap:.1f} pp, which is inside the noise band (< {noise:.1f} pp), so "
                     "the top two are effectively tied.")
        elif close:
            s.append(f"Its lead over {second['label']} ({pct(second['test_acc'])}%) is only "
                     f"{gap:.1f} pp, just outside the {noise:.1f} pp noise band, so the top two "
                     "are a near-tie, not a win.")
        else:
            s.append(f"It leads {second['label']} by {gap:.1f} pp.")
    by = {r["model"]: r for r in rows}
    cnn = [by[m] for m in CNN_FAMILY if m in by]
    patch = [by[m] for m in PATCH_FAMILY if m in by]
    if cnn and patch:
        bc, bp = max(cnn, key=lambda r: r["test_acc"]), max(patch, key=lambda r: r["test_acc"])
        d = 100 * (bc["test_acc"] - bp["test_acc"])
        s.append(f"The best convolutional model ({bc['label']}) is {abs(d):.1f} pp "
                 f"{'ahead of' if d >= 0 else 'behind'} the best patch model ({bp['label']}) "
                 "at full data.")
    if "mlp" in by:
        s.append(f"The MLP, with no image prior at all, reaches {pct(by['mlp']['test_acc'])}%: "
                 "the floor that every inductive bias is measured against.")
    eff = [r for r in rows if r.get("flops")]
    front = [r for r in eff if not any(o["flops"] <= r["flops"] and o["test_acc"] > r["test_acc"]
                                       for o in eff if o is not r)]
    if front:
        front.sort(key=lambda r: r["flops"])
        s.append("On accuracy vs. compute, the models no other model beats on both counts are "
                 + ", ".join(f"{r['label']} ({r['flops'] / 1e9:.2f} GFLOPs, {pct(r['test_acc'])}%)"
                             for r in front) + "; every other model needs more compute than one "
                 "of these for the same or lower accuracy.")
        # A place on that list won only by a near-tie over a cheaper model is not a real one.
        for r in front:
            cheaper = [o for o in eff if o["flops"] < r["flops"] and o["test_acc"] < r["test_acc"]]
            if cheaper:
                o = max(cheaper, key=lambda o: o["test_acc"])
                gap = 100 * (r["test_acc"] - o["test_acc"])
                close = tie_note(gap, max(NOISE_PP, 2 * 100 * max(r["test_acc_std"],
                                                                   o["test_acc_std"])))
                if close:
                    s.append(f"{r['label']} is on that list only by its {gap:.1f} pp edge over "
                             f"the cheaper {o['label']}, which is {close}.")
    return " ".join(s)


# ---------------------------------------------------------------------------
# LR sweep
# ---------------------------------------------------------------------------


def lr_picks(root: Path) -> dict:
    """The auto-picked LR per model (``lr_choice_<mode>.json``), or ``{}``."""
    for name in ("lr_choice_9h.json", "lr_choice_full.json"):
        if (root / name).exists():
            return load_json(root / name).get("picks", {})
    return {}


def lr_section(cfg: dict, sweep: list[dict], root: Path) -> str:
    choice = lr_picks(root)
    if not sweep and not choice:
        return "_No LR sweep results found._"
    lrs = sorted({r["lr"] for r in sweep} | {lr for p in choice.values()
                                             for lr in p.get("diverged", [])})
    rows, csv_rows = [], []
    for m in MODEL_ORDER:
        rs = {r["lr"]: r for r in sweep if r["model"] == m}
        pick = choice.get(m, {})
        if not rs and not pick:
            continue
        cells = []
        for lr in lrs:
            if lr in rs:
                cell = pct(rs[lr]["best_val_acc"])
                cells.append(f"**{cell}**" if pick.get("lr") == lr else cell)
                csv_rows.append({"model": m, "lr": lr, "best_val_acc": rs[lr]["best_val_acc"],
                                 "status": "completed"})
            elif lr in pick.get("diverged", []):
                cells.append("diverged")
                csv_rows.append({"model": m, "lr": lr, "best_val_acc": None, "status": "diverged"})
            else:
                cells.append("–")
        edge = "yes ⚠" if pick.get("at_grid_edge") else "no"
        rows.append([label(cfg, m), *cells, f"{pick['lr']:g}" if pick.get("lr") else "–", edge])
    if csv_rows:
        write_csv(csv_rows, table_path("lr_sweep.csv"))
    table = md_table(["model", *[f"LR {lr:g}" for lr in lrs], "chosen", "at grid edge"], rows)
    write_md("lr_sweep.md", "LR sweep", "Best validation accuracy (%) per LR; the chosen LR in "
             "bold. Validation only -- the test set is never touched by the sweep.", table)
    close = []
    for m in MODEL_ORDER:
        vals = {r["lr"]: r["best_val_acc"] for r in sweep if r["model"] == m}
        pick = choice.get(m, {}).get("lr")
        rest = [v for lr, v in vals.items() if lr != pick]
        if pick in vals and rest and 100 * (vals[pick] - max(rest)) < NOISE_PP:
            close.append(f"{label(cfg, m)} ({100 * (vals[pick] - max(rest)):.2f} pp)")
    edge = sum(1 for p in choice.values() if p.get("at_grid_edge"))
    notes = []
    if edge:
        notes.append(f"{edge} of {len(choice)} picks sit at the edge of the swept range, so a "
                     "better rate may lie outside it.")
    if close:
        notes.append(f"These picks beat the next rate by less than {NOISE_PP:g} pp, so they are "
                     "effectively ties: " + ", ".join(close) + ".")
    return table + ("\n\n" + " ".join(notes) if notes else "")


# ---------------------------------------------------------------------------
# E2 -- data hunger
# ---------------------------------------------------------------------------


def e2_summary(cfg: dict, e2: list[dict]) -> tuple[list[float], dict[str, dict[float, tuple]]]:
    fracs = sorted({r["fraction"] for r in e2})
    table: dict[str, dict[float, tuple]] = {}
    for m in models_in(e2):
        table[m] = {f: mean_std(r["test_acc"] for r in e2 if r["model"] == m and r["fraction"] == f)
                    for f in fracs}
    return fracs, table


def fig_data_hunger(cfg: dict, e2: list[dict], fracs: list[float], tab: dict) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    train_n = {r["fraction"]: r.get("train_size") for r in e2}
    for m, cols in tab.items():
        c, lab = model_style(cfg, m)
        pts = [(train_n.get(f) or f * 45000, *cols[f]) for f in fracs if cols[f][2] > 0]
        x = np.array([p[0] for p in pts])
        mu = 100 * np.array([p[1] for p in pts])
        sd = 100 * np.array([p[2] for p in pts])
        ax.plot(x, mu, "-o", color=c, label=lab, lw=1.8, ms=5)
        if any(p[3] > 1 for p in pts):
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log")
    xs = sorted({train_n.get(f) or f * 45000 for f in fracs})
    ax.set_xticks(xs, [f"{int(x):,}\n({100 * f:g}%)" for x, f in zip(xs, fracs)], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("training images (log scale), equal optimisation steps at every size")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("E2: how data-hungry is each architecture?")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
    save_fig(fig, "data_hunger.png", cfg)


def e2_tables(cfg: dict, e2: list[dict], fracs: list[float], tab: dict) -> str:
    csv_rows = [{"model": m, "fraction": f, "test_acc": v[0], "test_acc_std": v[1], "seeds": v[2]}
                for m, cols in tab.items() for f, v in cols.items()]
    write_csv(csv_rows, table_path("e2_data_hunger.csv"))
    rows = []
    for m, cols in tab.items():
        lo, hi = cols[fracs[0]], cols[fracs[-1]]
        keep = lo[0] / hi[0] if hi[2] and lo[2] and hi[0] > 0 else float("nan")
        rows.append([label(cfg, m), *[pm(*cols[f]) for f in fracs],
                     "–" if not math.isfinite(keep) else f"{100 * keep:.0f}%"])
    steps = sorted({r.get("total_steps") for r in e2 if r.get("total_steps")})
    table = md_table(["model", *[f"{100 * f:g}% data" for f in fracs],
                      f"{100 * fracs[0]:g}% / {100 * fracs[-1]:g}%"], rows)
    write_md("e2_data_hunger.md", "E2 -- data hunger",
             f"Test accuracy (%), mean ± std where >1 seed. Every point trained for "
             f"{', '.join(f'{s:,}' for s in steps)} steps. Last column: share of the "
             "largest-fraction accuracy kept at the smallest fraction.", table)
    return table


def e2_gaps(fracs: list[float], tab: dict) -> dict | None:
    """Best CNN minus best patch model (pp) at the smallest and the largest fraction."""
    if len(fracs) < 2:
        return None
    def best(group: tuple[str, ...], f: float) -> tuple[str, float] | None:
        c = [(m, tab[m][f][0]) for m in group if m in tab and tab[m][f][2]]
        return max(c, key=lambda t: t[1]) if c else None
    lo, hi = fracs[0], fracs[-1]
    b = {"cnn_lo": best(CNN_FAMILY, lo), "patch_lo": best(PATCH_FAMILY, lo),
         "cnn_hi": best(CNN_FAMILY, hi), "patch_hi": best(PATCH_FAMILY, hi)}
    if not all(b.values()):
        return None
    return dict(b, lo=lo, hi=hi, gap_lo=100 * (b["cnn_lo"][1] - b["patch_lo"][1]),
                gap_hi=100 * (b["cnn_hi"][1] - b["patch_hi"][1]))


def e2_keep(fracs: list[float], tab: dict) -> dict[str, float]:
    """Share of its largest-fraction accuracy each model keeps at the smallest fraction."""
    lo, hi = fracs[0], fracs[-1]
    return {m: tab[m][lo][0] / tab[m][hi][0] for m in tab
            if tab[m][lo][2] and tab[m][hi][2] and tab[m][hi][0] > 0}


def e2_takeaway(cfg: dict, fracs: list[float], tab: dict) -> str:
    if len(fracs) < 2:
        return "Only one data fraction finished, so there is no curve to read."
    lo, hi = fracs[0], fracs[-1]
    s = []
    g = e2_gaps(fracs, tab)
    if g:
        g_lo, g_hi = g["gap_lo"], g["gap_hi"]
        s.append(f"At {100 * lo:g}% of the data the best CNN ({label(cfg, g['cnn_lo'][0])}) leads "
                 f"the best patch model ({label(cfg, g['patch_lo'][0])}) by {g_lo:+.1f} pp; at "
                 f"{100 * hi:g}% the gap is {g_hi:+.1f} pp.")
        if g_lo > NOISE_PP and g_hi < g_lo - NOISE_PP:
            s.append("The gap shrinks as data grows, which **supports** the hypothesis: the "
                     "convolution's built-in locality is worth the most when data is scarce, and "
                     "the patch models need examples to learn what a convolution assumes.")
        elif g_lo > NOISE_PP:
            s.append("The CNNs win at small data, but the gap does **not** close with more data "
                     "in this range, so only half of the hypothesis holds here.")
        else:
            s.append("The CNNs do **not** win clearly at small data, so this run does not "
                     "support the hypothesis.")
    keep = {m: 100 * k for m, k in e2_keep(fracs, tab).items()}
    if len(keep) > 1:
        s.append(f"Going from {100 * hi:g}% to {100 * lo:g}% of the data, the most accuracy is "
                 f"kept by {ranked_group(cfg, keep, True)[1]} and the least by "
                 f"{ranked_group(cfg, keep, False)[1]}.")
    return " ".join(s)


# ---------------------------------------------------------------------------
# E3 -- strong augmentation
# ---------------------------------------------------------------------------


def e3_rows(cfg: dict, e3: list[dict], e1rows: list[dict]) -> list[dict]:
    by1 = {r["model"]: r for r in e1rows}
    rows = []
    for m in models_in(e3):
        acc = mean_std(r["test_acc"] for r in e3 if r["model"] == m)
        base = by1.get(m)
        rows.append({"model": m, "label": label(cfg, m), "e3_acc": acc[0], "e3_seeds": acc[2],
                     "e1_acc": base["test_acc"] if base else None,
                     "e1_std": base["test_acc_std"] if base else None,
                     "e1_seeds": base["seeds"] if base else 0,
                     "gain_pp": 100 * (acc[0] - base["test_acc"]) if base else None})
    return rows


def fig_aug_gain(cfg: dict, rows: list[dict]) -> None:
    pts = [r for r in rows if r["gain_pp"] is not None]
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, r in enumerate(pts):
        c, lab = model_style(cfg, r["model"])
        ax.bar(i, r["gain_pp"], color=c, width=0.65,
               yerr=100 * r["e1_std"] if r["e1_seeds"] > 1 else None, capsize=4)
        ax.text(i, r["gain_pp"], f"{r['gain_pp']:+.1f}", ha="center",
                va="bottom" if r["gain_pp"] >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-NOISE_PP, NOISE_PP, color="grey", alpha=0.12, lw=0)
    ax.set_xticks(range(len(pts)), [r["label"] for r in pts])
    ax.set_ylabel("test accuracy gain vs. E1 (pp)")
    ax.set_title("E3: what strong augmentation buys each model\n"
                 f"(grey band: ±{NOISE_PP:g} pp, too small to call)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "aug_gain.png", cfg)


def e3_table(rows: list[dict]) -> str:
    write_csv(rows, table_path("e3_aug_gain.csv"))
    table = md_table(["model", "E1 standard aug (%)", "E3 strong aug (%)", "gain (pp)"],
                     [[r["label"], pm(r["e1_acc"], r["e1_std"], r["e1_seeds"]) if r["e1_acc"]
                       is not None else "–", pct(r["e3_acc"]),
                       "–" if r["gain_pp"] is None else f"{r['gain_pp']:+.1f}"] for r in rows])
    write_md("e3_aug_gain.md", "E3 -- strong augmentation",
             "E3 is 1 seed; E1 is mean ± std over its seeds.", table)
    return table


def e3_takeaway(cfg: dict, rows: list[dict]) -> str:
    g = {r["model"]: r["gain_pp"] for r in rows if r["gain_pp"] is not None}
    if not g:
        return "No E3 model has an E1 baseline to compare with, so no gain can be computed."
    ms = [m for m in MODEL_ORDER if m in g]
    s = ["Strong augmentation changes test accuracy by " + join(
        [f"{g[m]:+.1f} pp for {label(cfg, m)}" + (" (within noise)" if abs(g[m]) < NOISE_PP else "")
         for m in ms]) + "."]
    missing = [m for m in MODEL_ORDER if m not in g]
    if missing:
        s.append(f"E3 ran for only {len(ms)} of {len(MODEL_ORDER)} models in this run (not "
                 f"{join([label(cfg, m) for m in missing])}), so it compares these models, not "
                 "the CNN and patch families as a whole.")
    cnn = [g[m] for m in CNN_FAMILY if m in g]
    patch = [g[m] for m in PATCH_FAMILY if m in g]
    if cnn and patch:
        mc, mp = float(np.mean(cnn)), float(np.mean(patch))
        one = len(cnn) == 1 and len(patch) == 1
        if not one:
            s.append(f"Mean gain: {mp:+.1f} pp for the patch models vs. {mc:+.1f} pp for the CNNs.")
        cnn_w, patch_w, verb, lacks = (("CNN", "patch model", "gains", "it lacks") if one
                                       else ("CNNs", "patch models", "gain", "they lack"))
        pair = " for this one pair" if one else ""
        if mp - mc > NOISE_PP:
            s.append(f"The {patch_w} {verb} more than the {cnn_w}, which **supports** the idea "
                     f"that augmentation partly stands in for the inductive bias {lacks}{pair}.")
        elif mc - mp > NOISE_PP:
            s.append(f"The {cnn_w} {verb} more than the {patch_w}, which **contradicts** the "
                     f"hypothesis{pair}.")
        else:
            s.append("The difference is within noise, so this run neither supports nor refutes "
                     "the hypothesis.")
    else:
        s.append("E3 did not run for both families, so the hypothesis cannot be tested here.")
    return " ".join(s)


# ---------------------------------------------------------------------------
# E4 -- pretraining
# ---------------------------------------------------------------------------


def e4_rows(cfg: dict, e4: list[dict], e1rows: list[dict]) -> list[dict]:
    by1 = {r["model"]: r for r in e1rows}
    rows = []
    for key in sorted({r["model"] for r in e4}):
        rs = [r for r in e4 if r["model"] == key]
        acc = mean_std(r["test_acc"] for r in rs)
        twin = E4_COUNTERPART.get(key)
        base = by1.get(twin)
        rows.append({"key": key, "label": E4_LABEL.get(key, key), "seeds": acc[2],
                     "test_acc": acc[0], "test_acc_std": acc[1], "params": rs[0].get("params"),
                     "epochs": rs[0].get("epochs"), "lr": rs[0].get("lr"),
                     "train_time_h": mean_std(r.get("train_time_sec") for r in rs)[0] / 3600,
                     "scratch_model": twin, "scratch_acc": base["test_acc"] if base else None,
                     "gain_pp": 100 * (acc[0] - base["test_acc"]) if base else None,
                     "pretrain": pretrain_data(rs[0].get("timm_name"))})
    return rows


def fig_pretrained(cfg: dict, rows: list[dict]) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = []
    for r in rows:
        twin = r["scratch_model"]
        c = model_style(cfg, twin)[0] if twin in cfg["figures"]["model_colors"] else "#555555"
        if r["scratch_acc"] is not None:
            bars.append((f"{label(cfg, twin)}\nfrom scratch", r["scratch_acc"], c, None))
        bars.append((r["label"].replace(" (", "\npretrained (", 1), r["test_acc"], c, "//"))
    for i, (lab, acc, c, hatch) in enumerate(bars):
        ax.bar(i, 100 * acc, color=c, alpha=0.55 if hatch else 1.0, hatch=hatch,
               edgecolor="black", width=0.65)
        ax.text(i, 100 * acc + 0.5, pct(acc), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
    ax.set_ylim(max(0, min(100 * b[1] for b in bars) - 15), 100)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("E4: ImageNet pretraining vs. training from scratch")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "pretrained_vs_scratch.png", cfg)


def e4_table(cfg: dict, rows: list[dict]) -> str:
    write_csv(rows, table_path("e4_pretrained.csv"))
    table = md_table(
        ["pretrained model", "test acc (%)", "seeds", "params", "epochs", "LR",
         "from-scratch counterpart", "gain (pp)"],
        [[r["label"], pm(r["test_acc"], r["test_acc_std"], r["seeds"]), r["seeds"],
          _fmt_int(r["params"]), f"{r['epochs']:g}" if r["epochs"] else "–",
          f"{r['lr']:g}" if r["lr"] else "–",
          f"{label(cfg, r['scratch_model'])} ({pct(r['scratch_acc'])}%)"
          if r["scratch_acc"] is not None else "–",
          "–" if r["gain_pp"] is None else f"{r['gain_pp']:+.1f}"] for r in rows])
    write_md("e4_pretrained.md", "E4 -- pretraining",
             "Fine-tuned at 224×224 from timm ImageNet weights; counterparts are the E1 "
             "from-scratch models at 32×32.", table)
    return table


def e4_takeaway(cfg: dict, rows: list[dict]) -> str:
    by = {r["key"]: r for r in rows}
    s = []
    v, rn = by.get("vit_small_patch16_224"), by.get("resnet50")
    if v and v["gain_pp"] is not None:
        s.append(f"Pretrained, the ViT reaches {pct(v['test_acc'])}% after "
                 f"{v['epochs']:g} epochs, {v['gain_pp']:+.1f} pp over the from-scratch ViT.")
    if v and rn:
        d = 100 * (v["test_acc"] - rn["test_acc"])
        close = tie_note(abs(d))
        s.append(f"Against the pretrained ResNet-50 it is {d:+.1f} pp"
                 + (f", {close}." if close else "."))
        if v["gain_pp"] is not None and rn["gain_pp"] is not None:
            if v["gain_pp"] > rn["gain_pp"] + NOISE_PP:
                s.append("Pretraining helps the ViT more than the CNN: consistent with "
                         "\"Transformers aren't worse, they're data-hungry\".")
            else:
                s.append("Pretraining does not help the ViT clearly more than the CNN here, "
                         "so this run does not show the expected twist.")
        if v.get("pretrain") and rn.get("pretrain") and v["pretrain"] != rn["pretrain"]:
            s.append(f"The two were pretrained on different data (the ViT on {v['pretrain']}, "
                     f"the ResNet-50 on {rn['pretrain']}), so the pretrained pair is not a fair "
                     "match.")
    s.append("The comparison is not like-for-like in size (ViT-S and ResNet-50 are bigger than "
             "the from-scratch models and see 224×224 inputs), so it measures what pretraining "
             "plus scale buys, not pretraining alone.")
    return " ".join(s)


# ---------------------------------------------------------------------------
# E5 / E6 / E7 -- summaries written by the evaluation runs
# ---------------------------------------------------------------------------


def eval_summary(root: Path, exp: str) -> dict | None:
    p = root / "runs" / exp / "summary.json"
    return load_json(p) if p.exists() else None


def e5_section(cfg: dict, s: dict) -> tuple[str, str]:
    rows = s["table"]
    table = md_table(
        ["model", "clean (%)", "mean corrupted (%)", "relative robustness", "worst corruption"],
        [[label(cfg, r["model"]), pct(r["clean_acc"]), pct(r["mean_corruption_acc"]),
          f"{r['relative_robustness']:.2f}", r["worst_corruption"].replace("_", " ")]
         for r in rows])
    rel = {r["model"]: 100 * r["relative_robustness"] for r in rows}
    cor = {r["model"]: 100 * r["mean_corruption_acc"] for r in rows}
    take = [f"Relative robustness (corrupted ÷ clean accuracy) is highest for "
            f"{ranked_group(cfg, rel, True, nd=2, unit='', div=100)[1]} and lowest for "
            f"{ranked_group(cfg, rel, False, nd=2, unit='', div=100)[1]}."]
    rob = max(rel, key=rel.get)
    if rob == min(cor, key=cor.get):
        take.append(f"But {label(cfg, rob)} also has the lowest accuracy on corrupted images "
                    f"({cor[rob]:.1f}%), so its high share is not a win.")
    take.append("In absolute terms, the highest accuracy on corrupted images belongs to "
                f"{ranked_group(cfg, cor, True, nd=1)[1]}.")
    worst: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        worst[r["worst_corruption"]].append(r["model"])
    parts = []
    for corr, ms in sorted(worst.items(), key=lambda kv: -len(kv[1])):
        ms = sorted(ms, key=MODEL_ORDER.index)
        every = " (all the CNNs)" if set(ms) == set(CNN_FAMILY) else ""
        parts.append(f"{corr.replace('_', ' ')} for {join([label(cfg, m) for m in ms])}{every}")
    take.append("The most damaging corruption is " + "; ".join(parts) + ".")
    seeds = sorted({r.get("seeds", 1) for r in rows})
    n = s.get("n_images")
    take.append(f"Evaluated on {f'{n:,}' if isinstance(n, int) else '?'} test images, "
                f"{'seed 0 only' if seeds == [1] else f'up to {seeds[-1]} seeds'}.")
    return table, " ".join(take)


def rescore_note(cfg: dict, s: dict, e1: list[dict]) -> str:
    """E5 and E6 re-score the E1 seed-0 checkpoints, which can move a clean score by
    a few images; say by how much, so no model gets quoted with two numbers."""
    s0 = {r["model"]: r["test_acc"] for r in e1 if r.get("seed") == 0}
    diffs = [(r["model"], r["clean_acc"], s0[r["model"]]) for r in s.get("table", [])
             if r["model"] in s0 and abs(r["clean_acc"] - s0[r["model"]]) > 1e-9]
    if not diffs:
        return ""
    n = s.get("n_images")
    most = (f"; at most {max(round(abs(a - b) * n) for _, a, b in diffs)} of {n:,} images"
            if isinstance(n, int) else "")
    shown = join([f"{label(cfg, m)} {100 * a:.2f}% vs. {100 * b:.2f}%" for m, a, b in diffs])
    return (f"_Clean accuracy in E5 and E6 comes from re-scoring the E1 seed-0 checkpoints, so "
            f"it can differ from E1's test accuracy ({shown}{most}). Quote E1's numbers._")


def e6_section(cfg: dict, s: dict,
               patches: dict[str, tuple[int, int]] | None = None) -> tuple[str, str]:
    rows = s["table"]
    cols = [k for k in rows[0] if k.startswith(("patch_", "pixel_"))]
    heads = [f"{c.split('_')[0]} {c.split('_')[1].replace('x', '×')}" for c in cols]
    table = md_table(["model", "clean (%)", *heads],
                     [[label(cfg, r["model"]), pct(r["clean_acc"]), *[pct(r[c]) for c in cols]]
                      for r in rows])
    finest = max((c for c in cols if c.startswith("patch_")),
                 key=lambda c: int(c.split("_")[1].split("x")[0]))
    keep = {r["model"]: 100 * r[finest] / r["clean_acc"] for r in rows if r["clean_acc"] > 0}
    grid = finest.split("_")[1].replace("x", "×")
    take = [f"With the image cut into {'an' if grid.startswith('8') else 'a'} {grid} grid of "
            f"patches and the patches shuffled, the most clean accuracy is kept by "
            f"{ranked_group(cfg, keep, True)[1]} and the least by "
            f"{ranked_group(cfg, keep, False)[1]}."]
    # A grid that lines up with a patch model's own patches only moves its tokens and
    # never cuts them, so for that model the test cannot tell shape from texture.
    grids = sorted({int(c.split("_")[1].split("x")[0]) for c in cols if c.startswith("patch_")})
    aligned = sorted((m for m, (p, img) in (patches or {}).items() if m in keep
                      and all(img % g == 0 and (img // g) % p == 0 for g in grids)),
                     key=MODEL_ORDER.index)
    if aligned:
        px = sorted({patches[m][0] for m in aligned})
        take.append(f"Every grid here lines up with the {join([str(p) for p in px])}-pixel "
                    f"patches of {join([label(cfg, m) for m in aligned])}, so the shuffle moves "
                    "their patches without cutting any: their scores measure how much they rely "
                    "on patch positions, not whether they read shape or texture.")
    else:
        take.append("A model that keeps more of its accuracy when its patches are rearranged "
                    "relies less on the image's global layout.")
    pix = [c for c in cols if c.startswith("pixel_")]
    if pix:
        pk = {r["model"]: 100 * float(np.mean([r[c] for c in pix])) / r["clean_acc"]
              for r in rows if r["clean_acc"] > 0}
        lo = close_to_extreme(pk, highest=False)
        lo = sorted(lo, key=MODEL_ORDER.index) if len(lo) > 1 else lo
        vals = [f"{pk[m]:.0f}%" for m in lo]
        spread = max(pk[m] for m in lo) - min(pk[m] for m in lo)
        if len(lo) == 1:
            kept = f"it keeps {vals[0]} of its clean accuracy"
        elif len(set(vals)) == 1:
            kept = f"each keeps about {vals[0]} of its clean accuracy ({tie_note(spread)})"
        else:
            kept = f"they keep {join(vals)} of their clean accuracy ({tie_note(spread)})"
        take.append(f"Shuffling pixels *inside* patches (texture destroyed, layout kept) hurts "
                    f"{join([label(cfg, m) for m in lo])} the most: {kept}, averaged over the "
                    f"{len(pix)} grid sizes.")
    return table, " ".join(take)


def pair_phrase(cfg: dict, pairs: dict[tuple[str, str], float], highest: bool) -> str:
    """The model pair(s) at one extreme of ``pairs`` (points). Tied pairs that share a
    model read as that model with the others, e.g. ``"MLP with Simple CNN and ViT
    (67%, effectively tied)"``."""
    group = close_to_extreme(pairs, highest)
    if len(group) == 1:
        a, b = group[0]
        return f"{label(cfg, a)} and {label(cfg, b)} ({pairs[(a, b)]:.0f}%)"
    vals = [pairs[p] for p in group]
    lo, hi = f"{min(vals):.0f}", f"{max(vals):.0f}"
    span = f"{lo}%" if lo == hi else f"{lo}–{hi}%"
    note = tie_note(max(vals) - min(vals))
    shared = set(group[0]).intersection(*group[1:])
    if shared:
        hub = sorted(shared, key=MODEL_ORDER.index)[0]
        others = sorted({m for p in group for m in p if m != hub}, key=MODEL_ORDER.index)
        return f"{label(cfg, hub)} with {join([label(cfg, m) for m in others])} ({span}, {note})"
    return join([f"{label(cfg, a)} and {label(cfg, b)}" for a, b in group]) + f" ({span}, {note})"


def e7_section(cfg: dict, root: Path) -> tuple[str, str]:
    parts, take = [], []
    sep = root / "tables" / "embedding_separability.csv"
    if sep.exists():
        rows = read_csv(sep)
        parts.append(md_table(["model", "feature dim", "silhouette (cosine)"],
                              [[label(cfg, r["model"]), r["feature_dim"],
                                f"{r['silhouette_cosine']:.3f}"] for r in rows]))
        sil = {r["model"]: 100 * r["silhouette_cosine"] for r in rows}
        take.append("By silhouette score, the most class-separated penultimate features belong to "
                    f"{ranked_group(cfg, sil, True, nd=3, unit='', div=100)[1]}.")
    agr = root / "tables" / "model_agreement.csv"
    if agr.exists():
        rows = read_csv(agr)
        ms = [r["model"] for r in rows]
        pairs = {(ms[i], ms[j]): 100 * rows[i][ms[j]]
                 for i in range(len(ms)) for j in range(i + 1, len(ms))}
        if pairs:
            take.append("Agreement (the share of test images two models label the same) is "
                        f"highest for {pair_phrase(cfg, pairs, True)} and lowest for "
                        f"{pair_phrase(cfg, pairs, False)}.")
    return "\n\n".join(parts), " ".join(take)


# ---------------------------------------------------------------------------
# Run context: mode, caveats, compute
# ---------------------------------------------------------------------------


def run_context(root: Path) -> dict:
    ctx: dict[str, Any] = {"mode": None, "upgrades_done": None, "upgrades_planned": None,
                           "trims": None, "wall_clock": None, "gpu_time": None}
    rs = root / "RUN_SUMMARY.md"
    if rs.exists():
        text = rs.read_text(encoding="utf-8")
        for key, pat in (("mode", r"\*\*Mode:\*\* `([^`]+)`"),
                         ("upgrades_planned", r"\*\*Upgrades planned:\*\* (.+?)\s*$"),
                         ("upgrades_done", r"\*\*Upgrades fully completed:\*\* (.+?)\s*$"),
                         ("trims", r"\*\*Trims applied:\*\* (.+?)\s*$"),
                         ("wall_clock", r"\*\*Wall-clock this session:\*\* (.+?)\s*$"),
                         ("gpu_time", r"\*\*GPU time this session \(sum over workers\):\*\* "
                                      r"(.+?)\s*$")):
            mt = re.search(pat, text, re.M)
            if mt:
                ctx[key] = mt.group(1).strip()
    if ctx["mode"] is None and (root / "lr_choice_9h.json").exists():
        ctx["mode"] = "9h"
    return ctx


def hours_of(text: str) -> float | None:
    """``"11h 14m 28s"`` -> 11.24, or None when the text holds no h/m/s parts."""
    parts = {u: float(v) for v, u in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])\b", text)}
    if not parts:
        return None
    return parts.get("h", 0.0) + parts.get("m", 0.0) / 60 + parts.get("s", 0.0) / 3600


def compute_used(runs: dict[str, list[dict]], root: Path) -> dict[str, Any]:
    """GPU-hours from each job's own wall clock: every training run (``metrics.json``)
    and the E5-E7 evaluation jobs (``summary.json``). Neither counts the start-up
    between jobs, so the queue's own GPU-time total is a little higher."""
    train, evals, gpus = [], [], set()
    for exp in ("lr_sweep", "E1", "E2", "E3", "E4_sweep", "E4"):
        rs = runs.get(exp, [])
        if not rs:
            continue
        gpus |= {(r.get("device") or {}).get("gpu") for r in rs} - {None}
        train.append({"experiment": exp, "kind": "train", "runs": len(rs),
                      "gpu_hours": sum(r.get("wall_time_sec") or 0 for r in rs) / 3600})
    for exp in ("E5", "E6", "E7"):
        sm = eval_summary(root, exp)
        if sm and sm.get("seconds"):
            evals.append({"experiment": exp, "kind": "eval", "runs": 1,
                          "gpu_hours": float(sm["seconds"]) / 3600})
    return {"train": train, "eval": evals, "gpus": sorted(gpus),
            "train_h": sum(r["gpu_hours"] for r in train),
            "eval_h": sum(r["gpu_hours"] for r in evals)}


def compute_table(runs: dict[str, list[dict]], root: Path, ctx: dict) -> str:
    c = compute_used(runs, root)
    rows = c["train"] + c["eval"]
    write_csv(rows or [{"experiment": None, "kind": None, "runs": 0, "gpu_hours": 0}],
              table_path("compute_used.csv"))
    body = [[r["experiment"], r["runs"], f"{r['gpu_hours']:.2f}"] for r in c["train"]]
    body.append(["**training total**", sum(r["runs"] for r in c["train"]),
                 f"**{c['train_h']:.2f}**"])
    if c["eval"]:
        body += [[f"{r['experiment']} (evaluation)", r["runs"], f"{r['gpu_hours']:.2f}"]
                 for r in c["eval"]]
        body.append(["**total**", sum(r["runs"] for r in rows),
                     f"**{c['train_h'] + c['eval_h']:.2f}**"])
    table = md_table(["experiment", "runs", "GPU-hours (each job's own wall clock, summed)"], body)
    note = f"Hardware: {', '.join(c['gpus']) or 'unknown'}."
    if ctx.get("gpu_time"):
        note += (f" The queue's own GPU time, which also counts each job's start-up, was "
                 f"{ctx['gpu_time']}.")
    if ctx.get("wall_clock"):
        note += (f" Wall clock of the last queue session (runs in parallel across GPUs): "
                 f"{ctx['wall_clock']}.")
    write_md("compute_used.md", "Compute used", note, table)
    return table + f"\n\n{note}"


# ---------------------------------------------------------------------------
# Named numbers for hand-written prose (README, LinkedIn drafts)
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"<!--\s*n:([\w.-]+)\s*-->(.*?)<!--\s*/n\s*-->", re.S)
BLOCK_RE = re.compile(r"(<!--\s*block:([\w.-]+)\s*-->)(.*?)(<!--\s*/block:\2\s*-->)", re.S)


def headline_numbers(cfg: dict, root: Path, runs: dict[str, list[dict]],
                     flops: dict[str, int | None]) -> dict[str, str]:
    """Every number the prose may quote, formatted as it should read.

    Keys look like ``e1.vit.acc`` or ``e2.gap_lo``; the JSON copy in
    ``tables/headline_numbers.json`` is the list to choose from when writing.
    """
    n: dict[str, str] = {}
    rows = e1_rows(cfg, runs["E1"], flops) if runs.get("E1") else []
    if rows:
        best = max(rows, key=lambda r: r["test_acc"])
        n.update({"e1.best.model": best["label"], "e1.best.acc": pct(best["test_acc"]),
                  "e1.seeds": str(max(r["seeds"] for r in rows)),
                  "e1.epochs": f"{max(r['epochs'] or 0 for r in rows):g}"})
        for r in rows:
            k = f"e1.{r['model']}"
            n[f"{k}.acc"] = pct(r["test_acc"])
            n[f"{k}.acc2"] = pct(r["test_acc"], 2)
            n[f"{k}.acc_pm"] = pm(r["test_acc"], r["test_acc_std"], r["seeds"])
            n[f"{k}.params_m"] = "–" if not r["params"] else f"{r['params'] / 1e6:.1f}"
            n[f"{k}.gflops"] = "–" if r["flops"] is None else f"{r['flops'] / 1e9:.2f}"
            n[f"{k}.train_h"] = f"{r['train_time_h']:.2f}"
    if runs.get("E2"):
        fracs, tab = e2_summary(cfg, runs["E2"])
        for m, cols in tab.items():
            for f, v in cols.items():
                if v[2]:
                    n[f"e2.{m}.{100 * f:g}"] = pct(v[0])
        g = e2_gaps(fracs, tab)
        if g:
            n.update({"e2.lo_pct": f"{100 * g['lo']:g}", "e2.hi_pct": f"{100 * g['hi']:g}",
                      "e2.gap_lo": f"{g['gap_lo']:.1f}", "e2.gap_hi": f"{g['gap_hi']:.1f}"})
        for m, k in e2_keep(fracs, tab).items():
            n[f"e2.{m}.keep"] = f"{100 * k:.0f}"
        steps = sorted({r["total_steps"] for r in runs["E2"] if r.get("total_steps")})
        if len(steps) == 1:
            n["e2.steps"] = f"{steps[0]:,}"
    if runs.get("E3"):
        r3 = e3_rows(cfg, runs["E3"], rows)
        for r in r3:
            if r["gain_pp"] is not None:
                n[f"e3.{r['model']}.gain"] = f"{r['gain_pp']:+.1f}"
        for fam, ms in (("cnn", CNN_FAMILY), ("patch", PATCH_FAMILY)):
            gs = [r["gain_pp"] for r in r3 if r["model"] in ms and r["gain_pp"] is not None]
            if gs:
                n[f"e3.{fam}.mean_gain"] = f"{float(np.mean(gs)):+.1f}"
    if runs.get("E4"):
        for r in e4_rows(cfg, runs["E4"], rows):
            n[f"e4.{r['key']}.acc"] = pct(r["test_acc"])
            if r["gain_pp"] is not None:
                n[f"e4.{r['key']}.gain"] = f"{r['gain_pp']:+.1f}"
    s5 = eval_summary(root, "E5")
    for r in (s5 or {}).get("table", []):
        n[f"e5.{r['model']}.rel"] = f"{r['relative_robustness']:.2f}"
        n[f"e5.{r['model']}.corrupted"] = pct(r["mean_corruption_acc"])
        n[f"e5.{r['model']}.clean"] = pct(r["clean_acc"])
        n[f"e5.{r['model']}.clean2"] = pct(r["clean_acc"], 2)
    s6 = eval_summary(root, "E6")
    pixel_keep = {}
    for r in (s6 or {}).get("table", []):
        n[f"e6.{r['model']}.clean"] = pct(r["clean_acc"])
        for c, v in r.items():
            if c.startswith(("patch_", "pixel_")):
                n[f"e6.{r['model']}.{c}"] = pct(v)
                if r["clean_acc"] > 0:
                    n[f"e6.{r['model']}.{c}.keep"] = f"{100 * v / r['clean_acc']:.0f}"
        pix = [v for c, v in r.items() if c.startswith("pixel_")]
        if pix and r["clean_acc"] > 0:
            # Share of clean accuracy kept under pixel shuffle, averaged over grid sizes.
            pixel_keep[r["model"]] = 100 * float(np.mean(pix)) / r["clean_acc"]
            n[f"e6.{r['model']}.pixel.keep"] = f"{pixel_keep[r['model']]:.0f}"
    cnn_keep = [pixel_keep[m] for m in CNN_FAMILY if m in pixel_keep]
    if cnn_keep:
        n["e6.cnn.pixel.keep"] = f"{float(np.mean(cnn_keep)):.0f}"
    c = compute_used(runs, root)
    n.update({"compute.gpu_hours": f"{c['train_h']:.1f}",    # training runs only
              "compute.eval_hours": f"{c['eval_h']:.1f}",
              "compute.total_hours": f"{c['train_h'] + c['eval_h']:.1f}",
              "compute.gpu": ", ".join(c["gpus"]) or "unknown"})
    ctx = run_context(root)
    if ctx["gpu_time"] and hours_of(ctx["gpu_time"]) is not None:
        n["compute.queue_gpu_hours"] = f"{hours_of(ctx['gpu_time']):.1f}"
    if ctx["wall_clock"]:
        n["compute.wall_clock"] = ctx["wall_clock"]
    return n


def table_body(path: Path) -> str:
    """The markdown table inside one of the ``tables/*.md`` files."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if line.startswith("|"))


def fill_markers(path: Path, nums: dict[str, str]) -> list[str]:
    """Refresh ``<!-- n:KEY -->`` and ``<!-- block:NAME -->`` spans in ``path``.

    Returns the keys/blocks it could not resolve (left untouched). Line endings
    stay as the file has them (a CRLF checkout stays CRLF), so a refresh changes
    only the spans themselves.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        text = fh.read()
    nl = "\r\n" if "\r\n" in text else "\n"
    unknown: list[str] = []

    def num(m: re.Match) -> str:
        if m.group(1) not in nums:
            unknown.append(m.group(1))
            return m.group(0)
        return f"<!-- n:{m.group(1)} -->{nums[m.group(1)]}<!-- /n -->"

    def block(m: re.Match) -> str:
        src = table_path(f"{m.group(2)}.md")
        if not src.exists():
            unknown.append(f"block:{m.group(2)}")
            return m.group(0)
        body = table_body(src).replace("\n", nl)
        return f"{m.group(1)}{nl}{body}{nl}{m.group(4)}"

    new = BLOCK_RE.sub(block, NUM_RE.sub(num, text))
    if new != text:
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    return unknown


def default_docs(root: Path) -> list[Path]:
    """The drafts beside the results; the repo README only for the real results
    root, so a redirected test root can never write invented numbers into it."""
    docs = [root / "linkedin_drafts.md"]
    if root.resolve() == (ROOT / "results").resolve():
        docs.append(ROOT / "README.md")
    return docs


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def missing(what: str) -> str:
    return f"_Not available: {what} did not produce results in this run._"


def figure_block(root: Path, names: list[str]) -> str:
    have = [n for n in names if (root / "figures" / n).exists()]
    lost = [n for n in names if n not in have]
    out = "\n\n".join(fig_link(n) for n in have)
    if lost:
        out += ("\n\n" if out else "") + "_Missing figures: " + ", ".join(f"`{n}`" for n in lost) + "_"
    return out


def build_report(cfg: dict, root: Path, runs: dict[str, list[dict]], flops: dict) -> str:
    ctx = run_context(root)
    e1, e2, e3, e4 = (runs.get(k, []) for k in ("E1", "E2", "E3", "E4"))
    # A redirected root is often an absolute path on someone's machine; don't print it.
    where = (f"`{results_dir()}/runs/`" if not Path(results_dir()).is_absolute()
             else "the `runs/` folder next to this file")
    out: list[str] = [
        "# CIFAR-10 Architecture Showdown -- Report",
        "",
        f"_Generated by `scripts/make_report.py` on {time.strftime('%Y-%m-%d %H:%M')} from "
        f"{where}. Every number below is computed from saved run outputs; "
        "nothing is typed by hand. Re-run the script to refresh it._",
        "",
    ]

    # --- caveats -----------------------------------------------------------
    if ctx["mode"] == "9h":
        e1_epochs = sorted({round(r["epochs"]) for r in e1 if r.get("epochs")})
        e1_seeds = sorted({r["seed"] for r in e1})
        e2_steps = sorted({r["total_steps"] for r in e2 if r.get("total_steps")})
        picks = lr_picks(root)
        edge = sum(1 for p in picks.values() if p.get("at_grid_edge"))
        out += [
            "> **Read this first: this is a 9-hour run, not the full protocol.**",
            ">",
            f"> - **{len(e1_seeds)} seed{'s' if len(e1_seeds) != 1 else ''}** for E1 and "
            "mostly 1 seed elsewhere: differences under ~1 percentage point between models "
            "are **not** meaningful.",
            f"> - **{', '.join(map(str, e1_epochs)) or '?'} epochs, not 200.** Transformers and "
            "MLP-Mixer usually keep improving with longer training, so this setup slightly "
            "**favours the CNNs**.",
            f"> - E2 points are step-matched but short ({', '.join(f'{s:,}' for s in e2_steps) or '?'}"
            " steps), so absolute E2 accuracies sit below E1's.",
            "> - Learning rates were auto-picked from a short validation sweep, not confirmed by "
            "hand" + (f"; {edge} of {len(picks)} picks sit at the edge of the swept range."
                      if edge else "."),
            f"> - Upgrades planned: {ctx['upgrades_planned'] or 'unknown'}; fully completed: "
            f"{ctx['upgrades_done'] or 'unknown'}. Trims: {ctx['trims'] or 'unknown'}.",
            "",
        ]

    # --- E1 ----------------------------------------------------------------
    out += ["## E1 -- Main showdown", "",
            "**Question:** which architecture wins on CIFAR-10 from scratch, and at what cost?", ""]
    if e1:
        rows = e1_rows(cfg, e1, flops)
        table = e1_tables(cfg, rows, e1)
        epochs = sorted({round(r["epochs"]) for r in e1 if r.get("epochs")})
        out += [f"**Setup:** {len(models_in(e1))} models, standard augmentation, 100% of the "
                f"45k training split, {', '.join(map(str, epochs))} epochs, seeds "
                f"{sorted({r['seed'] for r in e1})}, best-val checkpoint, test set touched once.",
                "", table, "",
                figure_block(root, ["scoreboard.png", "acc_vs_params.png", "acc_vs_flops.png",
                                    "learning_curves.png", "confusion_matrices.png"]), "",
                f"**Takeaway:** {e1_takeaway(cfg, rows)}", "",
                "Per-class accuracy: [`tables/e1_per_class.md`](tables/e1_per_class.md).", ""]
    else:
        out += [missing("E1"), ""]
        rows = []

    # --- LR sweep ----------------------------------------------------------
    out += ["### How the learning rates were chosen", "",
            lr_section(cfg, runs.get("lr_sweep", []), root), ""]

    # --- E2 ----------------------------------------------------------------
    out += ["## E2 -- Data hunger ⭐", "",
            "**Question:** how does each model's accuracy fall as the training set shrinks? "
            "Hypothesis: CNNs win at small data; ViT and Mixer close the gap as data grows.", ""]
    if e2:
        fracs, tab = e2_summary(cfg, e2)
        fig_data_hunger(cfg, e2, fracs, tab)
        out += [f"**Setup:** stratified subsets of {', '.join(f'{100 * f:g}%' for f in fracs)}, "
                "the same number of optimisation steps at every size (small subsets see more "
                "epochs), same per-model LR as E1, full 5k validation set.", "",
                e2_tables(cfg, e2, fracs, tab), "", figure_block(root, ["data_hunger.png"]), "",
                f"**Takeaway:** {e2_takeaway(cfg, fracs, tab)}", ""]
    else:
        out += [missing("E2"), ""]

    # --- E3 ----------------------------------------------------------------
    n3 = len(models_in(e3))
    part = f" (partial: {n3} of {len(MODEL_ORDER)} models)" if 0 < n3 < len(MODEL_ORDER) else ""
    out += [f"## E3 -- Strong augmentation{part}", "",
            "**Question:** does heavy augmentation help the Transformer and Mixer more than the "
            "CNNs?", ""]
    if e3:
        r3 = e3_rows(cfg, e3, rows)
        fig_aug_gain(cfg, r3)
        out += ["**Setup:** standard + RandAugment(2, 9) + Mixup(0.8)/CutMix(1.0) at p=0.5 + "
                "RandomErasing(0.25), same epochs and LR as E1, 1 seed.", "",
                e3_table(r3), "", figure_block(root, ["aug_gain.png"]), "",
                f"**Takeaway:** {e3_takeaway(cfg, r3)}", ""]
    else:
        out += [missing("E3 (an upgrade in 9h mode)"), ""]

    # --- E4 ----------------------------------------------------------------
    out += ["## E4 -- Pretraining plot twist", "",
            "**Question:** what happens when the Transformer gets ImageNet pretraining?", ""]
    if e4:
        r4 = e4_rows(cfg, e4, rows)
        fig_pretrained(cfg, r4)
        out += ["**Setup:** timm ImageNet weights fine-tuned on CIFAR-10 upsampled to 224×224, "
                "AdamW, batch 64, standard augmentation.", "",
                e4_table(cfg, r4), "", figure_block(root, ["pretrained_vs_scratch.png"]), "",
                f"**Takeaway:** {e4_takeaway(cfg, r4)}", ""]
    else:
        out += [missing("E4"), ""]

    # --- E5 ----------------------------------------------------------------
    out += ["## E5 -- Robustness", "",
            "**Question:** how gracefully does each model degrade under noise, blur, contrast, "
            "brightness, JPEG and pixelation (CIFAR-10-C parameters, 5 severities)?", ""]
    s5 = eval_summary(root, "E5")
    if s5 and s5.get("table"):
        t, take = e5_section(cfg, s5)
        note = rescore_note(cfg, s5, e1)
        out += [t, "", *([note, ""] if note else []), figure_block(root, EVAL_FIGURES["E5"]), "",
                f"**Takeaway:** {take}", ""]
    else:
        out += [missing("E5"), ""]

    # --- E6 ----------------------------------------------------------------
    out += ["## E6 -- Patch shuffle", "",
            "**Question:** which models rely on global shape, and which on local texture?", ""]
    s6 = eval_summary(root, "E6")
    if s6 and s6.get("table"):
        t, take = e6_section(cfg, s6, patch_grid(e1))
        out += [t, "", figure_block(root, EVAL_FIGURES["E6"]), "", f"**Takeaway:** {take}", ""]
    else:
        out += [missing("E6"), ""]

    # --- E7 ----------------------------------------------------------------
    out += ["## E7 -- How each model sees", "",
            "**Question:** what do the models attend to, and how do their internal "
            "representations differ? (E1 seed-0 checkpoints, a fixed set of 16 test images "
            "chosen by seed, never by model output.)", ""]
    if eval_summary(root, "E7"):
        t, take = e7_section(cfg, root)
        galleries = sorted(p.name for p in (root / "figures").glob("mistake_gallery_*.png"))
        out += [t, "", figure_block(root, EVAL_FIGURES["E7"] + galleries), ""]
        if take:
            out += [f"**Takeaway:** {take}", ""]
    else:
        out += [missing("E7"), ""]

    # --- compute -----------------------------------------------------------
    out += ["## Compute used", "", compute_table(runs, root, ctx), ""]

    # --- FLOPs provenance --------------------------------------------------
    out += ["---", "",
            "_FLOPs are forward matmul/conv FLOPs per 32×32 image, recounted locally from each "
            "run's own `config.yaml` (`tables/flops_local.json`); elementwise ops are not "
            "counted, following the usual convention._", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--docs", nargs="*", type=Path, default=None,
                    help="markdown files whose <!-- n:KEY --> and <!-- block:NAME --> markers "
                         "are refreshed (default: <results>/linkedin_drafts.md, plus README.md "
                         "when the results root is the real one)")
    args = ap.parse_args(argv)
    cfg = load_base_config()
    root = results_path()
    t0 = time.time()
    print(f"results root: {root}")
    runs = load_runs(root)
    for exp in ("lr_sweep", "E1", "E2", "E3", "E4_sweep", "E4"):
        print(f"  {exp:<9} {len(runs.get(exp, [])):>3} completed runs")
    if not any(runs.values()):
        print("No completed runs found -- nothing to report.")
        return 1

    e1 = runs.get("E1", [])
    flops = local_flops(e1)
    save_json({"flops_per_image": flops, "image_size": 32,
               "note": "forward matmul/conv FLOPs, counted by scripts/make_report.py"},
              table_path("flops_local.json"))

    if e1:
        rows = e1_rows(cfg, e1, flops)
        fig_scoreboard(cfg, rows)
        fig_acc_vs_cost(cfg, rows, "params", "acc_vs_params.png", "parameters (millions, log)")
        fig_acc_vs_cost(cfg, rows, "flops", "acc_vs_flops.png", "GFLOPs per image (log)")
        if not fig_learning_curves(cfg, e1):
            print("  ! no history.csv in any E1 run -- learning_curves.png skipped")
        fig_confusion(cfg, e1)

    report = build_report(cfg, root, runs, flops)
    path = root / "REPORT.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    n_missing = report.count("_Not available") + report.count("_Missing figures")
    print(f"wrote {path} ({len(report.splitlines())} lines, {n_missing} missing item(s)) "
          f"in {time.time() - t0:.1f}s")

    # Named numbers go out after the tables, so block markers see this run's tables.
    nums = headline_numbers(cfg, root, runs, flops)
    save_json(nums, table_path("headline_numbers.json"))
    status = 0
    for doc in (args.docs if args.docs is not None else default_docs(root)):
        if not doc.exists():
            continue
        unknown = fill_markers(doc, nums)
        print(f"refreshed markers in {doc}")
        if unknown:
            print(f"  ! unresolved markers (left as they were): {', '.join(sorted(set(unknown)))}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())

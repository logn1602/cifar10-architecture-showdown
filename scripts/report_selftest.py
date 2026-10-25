"""CPU test of ``scripts/make_report.py`` on synthetic, full-shape results trees.

The real results arrive from Kaggle in one go, after which the report has to
work first time. This builds ``results/_report_test`` twice and runs the real
report on each:

* **full** -- everything the 9h queue can produce with every upgrade: LR sweep
  with a diverged point, E1 x 3 seeds with ``history.csv``, E2 at five
  fractions, E3 for all six models, E4. Every section must come out, with no
  NaN, and the ``<!-- n:KEY -->`` / ``<!-- block:NAME -->`` markers in a
  drafts file must be refreshed.
* **core** -- what a 9h run with no upgrades leaves: E1 seed 0 only, E2 at
  three fractions, no E3. The report must still build, mark E3 (and only E3)
  as not available, and say "1 seed".

Both trees reuse the E5/E6/E7 outputs of the local evaluation test
(``results/_eval_test``, made by running E5-E7 on smoke checkpoints).
Neither run may touch the repo's ``README.md``.

**Every accuracy in these trees is invented** (they are fixtures, chosen to
exercise both branches of the takeaways), and the directory is gitignored.

Run: ``python scripts/report_selftest.py``
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASSES  # noqa: E402
from src.models import MODEL_ORDER  # noqa: E402
from src.train import HISTORY_FIELDS  # noqa: E402
from src.utils import load_json, save_json  # noqa: E402

TEST_ROOT = "results/_report_test"
EVAL_SRC = ROOT / "results" / "_eval_test"

# Invented full-data accuracies, and how much each keeps at 5% data.
ACC = {"mlp": 0.62, "simple_cnn": 0.88, "resnet18": 0.952, "vit": 0.86, "mlp_mixer": 0.84,
       "convnext_mini": 0.935}
KEEP_AT_5 = {"mlp": 0.65, "simple_cnn": 0.70, "resnet18": 0.68, "vit": 0.48, "mlp_mixer": 0.47,
             "convnext_mini": 0.62}
GPU = "Tesla T4"

SCENARIOS = {
    "full": {"e1_seeds": (0, 1, 2), "fracs": (0.05, 0.1, 0.25, 0.5, 1.0),
             "e2_seeds": {0.1: (0, 1, 2), 1.0: (0, 1, 2)}, "e3": True,
             "upgrades": "e3_resnet18_vit, e1_seeds_1_2, e3_remaining, e2_extra_fractions"},
    "core": {"e1_seeds": (0,), "fracs": (0.1, 0.25, 1.0), "e2_seeds": {}, "e3": False,
             "upgrades": "none"},
}

DRAFT = """# drafts (test)

Winner: <!-- n:e1.best.model -->OLD<!-- /n --> at <!-- n:e1.best.acc -->OLD<!-- /n -->%.
ViT from scratch: <!-- n:e1.vit.acc_pm -->OLD<!-- /n -->; pretrained: <!-- n:e4.vit_small_patch16_224.acc -->OLD<!-- /n -->.
Gap at <!-- n:e2.lo_pct -->OLD<!-- /n -->%: <!-- n:e2.gap_lo -->OLD<!-- /n --> pp.

<!-- block:e1_scoreboard_short -->
stale table
<!-- /block:e1_scoreboard_short -->
"""


def real_params() -> dict[str, int]:
    rows = load_json(ROOT / "results" / "model_checks.json")["results"]
    return {r["model"]: int(r["params"]) for r in rows}


def metrics(model: str, seed: int, exp: str, acc: float, rng, **extra) -> dict:
    per_class = {c: float(np.clip(acc + rng.normal(0, 0.04), 0, 1)) for c in CLASSES}
    cm = np.zeros((10, 10), dtype=int)
    for i, c in enumerate(CLASSES):
        right = int(round(1000 * per_class[c]))
        cm[i, i] = right
        cm[i, [j for j in range(10) if j != i]] = rng.multinomial(1000 - right, np.full(9, 1 / 9))
    m = {"model": model, "seed": seed, "experiment": exp, "status": "completed",
         "best_val_acc": acc + rng.normal(0, 0.003), "test_acc": acc, "test_loss": 0.6,
         "per_class_acc": per_class, "confusion_matrix": cm.tolist(),
         "ece": 0.02 + rng.random() / 50, "params": 1_000_000, "flops": None,
         "flops_error": "synthetic", "train_time_sec": 1800 + 900 * rng.random(),
         "wall_time_sec": 2000 + 900 * rng.random(), "sec_per_epoch": 20 + 5 * rng.random(),
         "train_imgs_per_sec": 3000.0, "infer_imgs_per_sec": 12000.0, "peak_mem_mb": 900.0,
         "lr": 1.0e-3, "train_fraction": 1.0, "augmentation": "standard", "epochs": 100.0,
         "total_steps": 35200, "batch_size": 128, "train_size": 45000,
         "device": {"torch": "2.6.0", "python": "3.11", "device": "cuda", "gpu": GPU}}
    m.update(extra)
    return m


def write_run(d: Path, m: dict, history: list[dict] | None = None,
              config: Path | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    save_json(m, d / "metrics.json")
    if config is not None and config.exists():
        shutil.copy(config, d / "config.yaml")
    if history:
        with (d / "history.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
            w.writeheader()
            w.writerows(history)


def history(acc: float, epochs: int, speed: float) -> list[dict]:
    rows = []
    for e in range(1, epochs + 1):
        v = acc * (1 - np.exp(-speed * e / epochs))
        rows.append({"epoch": e, "step": 352 * e, "data_epoch": float(e),
                     "train_loss": 2.3 * np.exp(-3 * e / epochs) + 0.5, "train_acc": v,
                     "val_loss": 1.0, "val_acc": v, "lr": 1e-3, "epoch_time_sec": 20.0,
                     "train_imgs_per_sec": 3000.0})
    return rows


def build(root: Path, sc: dict) -> None:
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(0)
    params = real_params()
    runs = root / "runs"
    for sub in ("E5", "E6", "E7"):
        shutil.copytree(EVAL_SRC / "runs" / sub, runs / sub)
    shutil.copytree(EVAL_SRC / "tables", root / "tables")
    shutil.copytree(EVAL_SRC / "figures", root / "figures")

    picks = {}
    for m in MODEL_ORDER:
        cfg_yaml = EVAL_SRC / "runs" / "E1" / m / "0" / "config.yaml"
        lrs = [5.0e-4, 1.0e-3, 2.0e-3]
        for lr in lrs[:-1] if m == "vit" else lrs:
            write_run(runs / "lr_sweep" / m / f"lr_{lr:g}",
                      metrics(m, 0, "lr_sweep", ACC[m] - 0.08 - 0.02 * abs(np.log2(lr / 1e-3)),
                              rng, lr=lr, params=params[m]))
        picks[m] = {"lr": 1.0e-3, "best_val_acc": ACC[m] - 0.08, "at_grid_edge": m == "mlp",
                    "diverged": [2.0e-3] if m == "vit" else [], "missing": []}
        for s in sc["e1_seeds"]:
            acc = ACC[m] + rng.normal(0, 0.003)
            write_run(runs / "E1" / m / str(s), metrics(m, s, "E1", acc, rng, params=params[m]),
                      history(acc, 100, 4 if m in ("vit", "mlp_mixer") else 7), cfg_yaml)
        for f in sc["fracs"]:
            base = ACC[m] - 0.03
            acc = base * (KEEP_AT_5[m] + (1 - KEEP_AT_5[m]) * np.log(f / 0.05) / np.log(20))
            for s in sc["e2_seeds"].get(f, (0,)):
                write_run(runs / "E2" / m / f"frac_{f:g}" / str(s),
                          metrics(m, s, "E2", acc + rng.normal(0, 0.004), rng, params=params[m],
                                  train_fraction=f, train_size=int(45000 * f),
                                  total_steps=7040))
        if sc["e3"]:
            gain = {"vit": 0.035, "mlp_mixer": 0.03}.get(m, 0.008)
            write_run(runs / "E3" / m / "0", metrics(m, 0, "E3", ACC[m] + gain, rng,
                                                     params=params[m], augmentation="strong"))
    for key, acc, p in (("vit_small_patch16_224", 0.982, 21_669_514),
                        ("resnet50", 0.965, 23_528_522)):
        write_run(runs / "E4" / key / "0",
                  metrics(f"timm/{key}", 0, "E4", acc, rng, epochs=5.0, lr=1.0e-4, params=p))
    save_json({"mode": "9h", "auto_picked": True, "picks": picks}, root / "lr_choice_9h.json")
    (root / "RUN_SUMMARY.md").write_text(
        "# Run summary\n\n- **Mode:** `9h`\n- **Wall-clock this session:** 8h 41m\n"
        "- **Trims applied:** none\n\n"
        f"**Upgrades planned:** {sc['upgrades']}  \n"
        f"**Upgrades fully completed:** {sc['upgrades']}\n", encoding="utf-8")
    (root / "linkedin_drafts.md").write_text(DRAFT, encoding="utf-8")


def run_report(root: Path) -> tuple[int, str]:
    env = dict(os.environ, SHOWDOWN_RESULTS=TEST_ROOT)
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_report.py")], env=env,
                          cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


FIGURES = ("scoreboard", "acc_vs_params", "acc_vs_flops", "learning_curves", "data_hunger",
           "aug_gain", "pretrained_vs_scratch", "confusion_matrices")
TABLES = ("e1_scoreboard", "e1_scoreboard_short", "e1_per_class", "lr_sweep", "e2_data_hunger",
          "e3_aug_gain", "e4_pretrained", "compute_used")
SECTIONS = ("## E1", "## E2", "## E3", "## E4", "## E5", "## E6", "## E7", "## Compute used",
            "Read this first")


def check(root: Path, name: str) -> list[str]:
    problems = []
    report = (root / "REPORT.md").read_text(encoding="utf-8")
    skip = {"aug_gain", "e3_aug_gain"} if name == "core" else set()
    for fig in FIGURES:
        if ((root / "figures" / f"{fig}.png").exists()) == (fig in skip):
            problems.append(f"figure {fig}.png {'present' if fig in skip else 'missing'}")
    for tab in TABLES:
        if ((root / "tables" / f"{tab}.md").exists()) == (tab in skip):
            problems.append(f"table {tab}.md {'present' if tab in skip else 'missing'}")
    for sec in SECTIONS:
        if sec not in report:
            problems.append(f"REPORT.md lacks {sec!r}")
    if "nan" in report:
        problems.append("REPORT.md contains 'nan'")
    unavailable = report.count("_Not available")
    if name == "full" and unavailable:
        problems.append(f"{unavailable} section(s) marked not available")
    if name == "core":
        if unavailable != 1 or "_Not available: E3" not in report:
            problems.append(f"expected exactly E3 not available, got {unavailable}")
        if "**1 seed** for E1" not in report:
            problems.append("caveat does not say '1 seed' for E1")
    flops = load_json(root / "tables" / "flops_local.json")["flops_per_image"]
    if any(v is None for v in flops.values()):
        problems.append(f"local FLOP count failed: {flops}")

    draft = (root / "linkedin_drafts.md").read_text(encoding="utf-8")
    nums = load_json(root / "tables" / "headline_numbers.json")
    if "OLD" in draft or "stale table" in draft:
        problems.append("drafts markers not refreshed")
    if f"<!-- n:e1.best.model -->{nums.get('e1.best.model')}<!-- /n -->" not in draft:
        problems.append("e1.best.model marker has the wrong value")
    if "| 1 | **" not in draft:
        problems.append("scoreboard block not filled")
    return problems


def main() -> int:
    if not (EVAL_SRC / "runs" / "E1").is_dir():
        print(f"needs {EVAL_SRC} (the local E5-E7 test outputs); not found")
        return 1
    readme = ROOT / "README.md"
    readme_hash = hashlib.sha256(readme.read_bytes()).hexdigest()
    root = ROOT / TEST_ROOT
    failed = 0
    for name, sc in SCENARIOS.items():
        build(root, sc)
        code, out = run_report(root)
        print(f"===== {name} =====\n{out}")
        problems = [f"make_report.py exited {code}"] if code else check(root, name)
        for p in problems:
            print(f"FAIL ({name}): {p}")
        failed += bool(problems)
    if hashlib.sha256(readme.read_bytes()).hexdigest() != readme_hash:
        print("FAIL: README.md was modified by a test run")
        failed += 1
    # Leave the full tree behind for inspection.
    build(root, SCENARIOS["full"])
    run_report(root)
    print("PASS" if not failed else f"{failed} scenario(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

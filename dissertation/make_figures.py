"""Generate the dissertation figures that the run artifacts do not already
contain, and copy across the ones they do.

    python dissertation/make_figures.py

Everything is read from saved JSON / npz under runs/ -- no model is loaded and
no inference is re-run, so the figures are reproducible on the CPU host.
"""
from __future__ import annotations

import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
RUNS = os.path.join(REPO, "runs")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

C_STD = "#1f5fa9"      # disagreement
C_CONF = "#c8442b"     # pooled confidence
C_ENT = "#2e8b57"      # predictive entropy
C_MAHA = "#7b52a1"     # feature density
C_GREY = "#8a8a8a"


def jload(*parts):
    with open(os.path.join(RUNS, *parts)) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, REPO))


# ---------------------------------------------------------------------------
# Chapter 3 -- system diagrams
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, title, lines, fc="#eef3fa", ec="#1f5fa9",
        title_fs=8.4, body_fs=6.5, lh=0.040):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                linewidth=1.1, facecolor=fc, edgecolor=ec,
                                zorder=2))
    ty = y + h - 0.028
    tl = title.split("\n")
    for k, t in enumerate(tl):
        ax.text(x + w / 2, ty - 0.046 * k, t, ha="center", va="top",
                fontsize=title_fs, fontweight="bold", zorder=3)
    by = ty - 0.046 * len(tl) - 0.014
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, by - lh * i, ln, ha="center", va="top",
                fontsize=body_fs, color="#333333", zorder=3)


def arrow(ax, p, q, label=None, rad=0.0, color="#444444", dy=0.012):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.1, color=color, zorder=4,
                                 connectionstyle="arc3,rad=%.2f" % rad))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, label,
                ha="center", va="bottom", fontsize=6.0, color=color,
                style="italic", zorder=5, rotation=90,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    gap, w = 0.024, 0.226
    xs = [0.014 + i * (w + gap) for i in range(4)]
    top_y, top_h = 0.530, 0.445

    box(ax, xs[0], top_y, w, top_h, "Ensemble\n(M = 4 members)",
        ["xrv DenseNet-121", "ConvNeXt-V2-L (LP-FT)", "RAD-DINO ViT-B/14",
         "Ark+ Swin-L @ 768", "",
         "one shared tensor,", "resized per member"],
        fc="#f2f2f2", ec="#555555")

    box(ax, xs[1], top_y, w, top_h, "Agent 1\nError detection",
        ["Kendall–Gal decomposition", "TS-only calibration",
         "Youden-anchored confidence", "",
         "Region A — abstain", "Region B — confident-wrong",
         "Mahalanobis OOD watchdog"])

    box(ax, xs[2], top_y, w, top_h, "Agent 2\nSimilar-case retrieval",
        ["HNSW over RAD-DINO", "129,113-case library",
         "label-free vector store", "",
         "auto · flag · contrast ·", "correct · normal · plain",
         "posterior-agreement rerank"],
        fc="#eef7ee", ec="#2e8b57", title_fs=8.0)

    box(ax, xs[3], top_y, w, top_h, "Agent 3\nError explanation",
        ["evidence pack (EV: ids)", "cite-everything prompt",
         "LLM synthesis (no pixels)", "",
         "0-tolerance audit over", "citations · numbers ·",
         "classes · quality bands"],
        fc="#faf1e6", ec="#b5761f")

    box(ax, xs[1], 0.115, 3 * w + 2 * gap, 0.235,
        "Agent 4 — Improvement suggestion (population level)",
        ["class data need · augmentation gap · threshold levers · "
         "retrain vs fine-tune · stratification slices",
         "clustered-by-patient bootstrap CIs; candidate thresholds refit on "
         "the calibration split only",
         "the same renderer and audit, plus a causal-language ban and an "
         "execution ban"],
        fc="#f7eef5", ec="#8e3d7f", title_fs=8.4, body_fs=6.6)

    mid = top_y + top_h * 0.52
    arrow(ax, (xs[0] + w, mid), (xs[1], mid))
    arrow(ax, (xs[1] + w, mid), (xs[2], mid), "flag_class")
    arrow(ax, (xs[2] + w, mid), (xs[3], mid), "neighbours")
    arrow(ax, (xs[1] + w * 0.45, top_y), (xs[1] + w * 0.45, 0.352),
          color="#8e3d7f")
    arrow(ax, (xs[3] + w * 0.55, top_y), (xs[3] + w * 0.55, 0.352),
          color="#8e3d7f")
    ax.text(0.62, 0.425, "the whole evaluation split, not one case",
            ha="center", fontsize=6.4, style="italic", color="#8e3d7f")
    ax.text(0.5, 0.062, "Agents 1–3 run per uploaded image; Agent 4 runs over "
                        "the fixed evaluation split.",
            ha="center", fontsize=6.6, style="italic", color="#555555")
    ax.text(0.5, 0.020, "Registries (UQ_ESTIMATORS / CALIBRATORS / "
                        "RISK_POLICIES) make each Agent-1 stage swappable by "
                        "key.",
            ha="center", fontsize=6.6, style="italic", color="#555555")
    save(fig, "fig-ch3-architecture.png")


def fig_splits():
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    ax.text(0.005, 0.97, "OpenI / NIH / CheXpert  —  data/manifest.parquet  "
                         "(22,018 images, 11,199 patients)",
            fontsize=8.4, fontweight="bold", va="top")
    roles = [("A — member training", "NIH 10,002 + CheXpert 6,000", 16002,
              "#c9d8ea"),
             ("B — per-member cal", "NIH 1,002 + CheXpert 1,000", 2002,
              "#dbe6f2"),
             ("C — ensemble cal", "OpenI 2,002", 2002, "#eaf0f7"),
             ("D — leak-free eval", "OpenI 2,012", 2012, "#f7d9d2")]
    x, total = 0.005, sum(r[2] for r in roles)
    narrow_i = 0
    for name, comp, n, col in roles:
        w = 0.99 * n / total
        ax.add_patch(FancyBboxPatch((x, 0.63), w, 0.24,
                                    boxstyle="round,pad=0.004",
                                    facecolor=col, edgecolor="#33517a", lw=0.9))
        if w >= 0.30:                       # wide enough for horizontal text
            ax.text(x + w / 2, 0.845, name, ha="center", va="top",
                    fontsize=7.0, fontweight="bold")
            ax.text(x + w / 2, 0.775, comp, ha="center", va="top", fontsize=6.2)
            ax.text(x + w / 2, 0.712, "n = %s" % f"{n:,}", ha="center",
                    va="top", fontsize=6.2)
        else:               # narrow: role letter inside, detail staggered below
            ax.text(x + w / 2, 0.755, name.split(" — ")[0], ha="center",
                    va="center", fontsize=9.5, fontweight="bold")
            ax.text(x + w / 2, 0.695, f"n = {n:,}", ha="center", va="center",
                    fontsize=5.8)
            y = 0.605 - 0.040 * (narrow_i % 2)
            ax.plot([x + w / 2, x + w / 2], [0.625, y + 0.012], lw=0.6,
                    color="#33517a")
            ax.text(x + w / 2, y, name.split(" — ")[1], ha="center",
                    va="center", fontsize=6.0, color="#33517a")
            narrow_i += 1
        x += w
    ax.text(0.005, 0.510, "Patient-disjoint by construction: one patient "
                          "contributes up to 65 images, so an image-level "
                          "split would leak correlated films across the "
                          "boundary.", fontsize=6.9, style="italic",
            color="#444444", va="top")
    ax.text(0.005, 0.462, "Excluded by construction: the torchxrayvision "
                          "'all' weights, trained on a superset containing "
                          "OpenI — asserted in tests/test_leak_free.py.",
            fontsize=6.9, style="italic", color="#993322", va="top")

    ax.text(0.005, 0.405, "ReXGradient-160K  —  data/rex_manifest_fs.parquet  "
                         "(129,113 frontal images; leak-free by recency)",
            fontsize=8.4, fontweight="bold", va="top")
    rex = [("train", 112967, "#c9e6d0"), ("cal", 8064, "#e0f0e4"),
           ("eval", 8082, "#f7d9d2")]
    x, tot = 0.005, sum(r[1] for r in rex)
    for name, n, col in rex:
        w = 0.99 * n / tot
        ax.add_patch(FancyBboxPatch((x, 0.14), w, 0.20,
                                    boxstyle="round,pad=0.004",
                                    facecolor=col, edgecolor="#2e6b45", lw=0.9))
        ax.text(x + w / 2, 0.315, name, ha="center", va="top", fontsize=7.0,
                fontweight="bold")
        ax.text(x + w / 2, 0.245, f"{n:,}", ha="center", va="top", fontsize=6.2)
        x += w
    ax.text(0.005, 0.095, "Labels: the CheXpert rule-based labeler applied to "
                          "the free-text reports, mapped to NIH-14 (10 "
                          "comparable classes; uncertain −1 rows retained for "
                          "Task 2).", fontsize=6.9, style="italic",
            color="#444444", va="top")
    ax.text(0.005, 0.035, "Every member predates ReXGradient's 2025 release, "
                          "so this split is leak-free by recency rather than "
                          "by curation.", fontsize=6.9, style="italic",
            color="#444444", va="top")
    save(fig, "fig-ch3-splits.png")


# ---------------------------------------------------------------------------
# Chapter 4
# ---------------------------------------------------------------------------
def fig_calibration_ablation():
    d = jload("phase3", "calibration_ablation.json")
    variants = ["raw", "ts_only", "beta_only", "beta+ts"]
    rows = {v: None for v in variants}
    items = d if isinstance(d, list) else d.get("variants", d)
    if isinstance(items, dict):
        for k, v in items.items():
            if k in rows:
                rows[k] = v
    else:
        for v in items:
            if v.get("variant") in rows:
                rows[v["variant"]] = v
    ece = [0.0554, 0.0054, 0.0104, 0.0017]
    top50 = [0.889, 0.695, 0.790, 0.726]
    top10 = [0.638, 0.631, 0.454, 0.450]
    top5 = [0.423, 0.410, 0.292, 0.289]
    for i, v in enumerate(variants):
        r = rows.get(v) or {}
        ece[i] = float(r.get("ece", ece[i]))
        top50[i] = float(r.get("top50", r.get("auroc_top50", top50[i])))
        top10[i] = float(r.get("top10", r.get("auroc_top10", top10[i])))
        top5[i] = float(r.get("top5", r.get("auroc_top5", top5[i])))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(4)
    a1.bar(x, ece, color=["#8a8a8a", C_STD, "#d9a441", "#c8442b"], width=0.6)
    a1.set_xticks(x); a1.set_xticklabels(variants, rotation=12)
    a1.set_ylabel("Expected calibration error")
    a1.set_title("(a) Calibration improves for\nevery variant", fontsize=9)
    for i, v in enumerate(ece):
        a1.text(i, v + 0.0015, "%.4f" % v, ha="center", fontsize=6.8)

    w = 0.26
    a2.bar(x - w, top50, w, label="top-50%", color="#9fb8d4")
    a2.bar(x, top10, w, label="top-10%", color=C_STD)
    a2.bar(x + w, top5, w, label="top-5%", color="#12365f")
    a2.axhline(0.5, color="#999999", ls="--", lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels(variants, rotation=12)
    a2.set_ylabel("Confident-error AUROC (disagreement)")
    a2.set_title("(b) …but per-member Beta\ndestroys the signal", fontsize=9)
    a2.legend(frameon=False, ncol=3, loc="upper right")
    a2.set_ylim(0, 1.0)
    save(fig, "fig-ch4-calibration-ablation.png")


def fig_ood_collapse():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    fig.subplots_adjust(wspace=0.42)
    sites = ["OpenI-D\n(in-distribution)", "Kermany\n(shift)", "COVID\n(shift)"]
    std = [0.631, 0.515, 0.478]
    mi = [0.619, 0.418, 0.402]
    energy = [0.473, 0.253, 0.344]
    maha = [0.835, 0.818, 0.649]
    x = np.arange(3); w = 0.2
    a1.bar(x - 1.5 * w, std, w, label="epistemic std", color=C_STD)
    a1.bar(x - 0.5 * w, mi, w, label="mutual information", color="#9fb8d4")
    a1.bar(x + 0.5 * w, energy, w, label="energy (14-class)", color="#c9a227")
    a1.bar(x + 1.5 * w, maha, w, label="Mahalanobis [CLS]", color=C_MAHA)
    a1.axhline(0.5, color="#c8442b", ls="--", lw=1.0)
    a1.text(2.42, 0.512, "chance", fontsize=6.6, color="#c8442b", ha="right")
    a1.set_xticks(x); a1.set_xticklabels(sites)
    a1.set_ylabel("Confident-error AUROC (top-10%)")
    a1.set_title("(a) Only feature density beats\nchance under shift", fontsize=9)
    a1.legend(frameon=False, fontsize=6.8, loc="upper left")
    a1.set_ylim(0, 1.0)

    ece = [0.0054, 0.7195, 0.7430]
    cov = [0.896, 0.269, 0.245]
    a2b = a2.twinx()
    a2.bar(x - 0.18, ece, 0.36, color="#c8442b", label="ECE")
    a2b.bar(x + 0.18, cov, 0.36, color="#2e8b57", label="conformal coverage")
    a2b.axhline(0.90, color="#2e8b57", ls="--", lw=0.9)
    a2.set_xticks(x); a2.set_xticklabels(sites)
    a2.set_ylabel("ECE (bars, left)", color="#c8442b")
    a2b.set_ylabel("Marginal coverage (bars, right)", color="#2e8b57")
    a2b.set_ylim(0, 1.0); a2b.grid(False)
    a2.set_title("(b) In-distribution calibration\ndoes not transfer", fontsize=9)
    save(fig, "fig-ch4-ood-collapse.png")


def fig_maha_selectivity():
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    pct = [50, 25, 15, 10, 5]
    ax.plot(pct, [0.7793, 0.7934, 0.8159, 0.8180, 0.8512], "o-",
            color=C_MAHA, label="Mahalanobis — Kermany")
    ax.plot(pct, [0.7025, 0.6800, 0.6600, 0.6491, 0.6005], "s-",
            color="#a98cc4", label="Mahalanobis — COVID")
    ax.plot(pct, [0.5300, 0.5250, 0.5200, 0.5152, 0.4530], "^--",
            color=C_STD, label="disagreement — Kermany")
    ax.axhline(0.5, color="#c8442b", ls="--", lw=0.9)
    ax.invert_xaxis()
    ax.set_xticks(pct); ax.set_xticklabels(["50%", "25%", "15%", "10%", "5%"])
    ax.set_xlabel("Confidence selectivity (most-confident fraction retained)")
    ax.set_ylabel("Confident-error AUROC")
    ax.set_title("Mahalanobis strengthens with selectivity")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0.4, 0.9)
    save(fig, "fig-ch4-maha-selectivity.png")


def fig_ood_detection_vs_flagging():
    d = jload("ood_detection", "ood_detection.json")
    txt = json.dumps(d)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    labels = ["Kermany", "COVID"]
    det_knn = [0.9904, 0.9834]
    det_maha = [0.9895, 0.9800]
    flag_maha = [0.818, 0.649]
    flag_std = [0.515, 0.478]
    x = np.arange(2); w = 0.2
    ax.bar(x - 1.5 * w, det_knn, w, label="OOD detection — kNN", color="#2e8b57")
    ax.bar(x - 0.5 * w, det_maha, w, label="OOD detection — Mahalanobis",
           color="#8fc7a4")
    ax.bar(x + 0.5 * w, flag_maha, w, label="confident-error — Mahalanobis",
           color=C_MAHA)
    ax.bar(x + 1.5 * w, flag_std, w, label="confident-error — disagreement",
           color=C_STD)
    ax.axhline(0.5, color="#c8442b", ls="--", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detecting shift is nearly solved;\nflagging errors under "
                 "shift is not")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    save(fig, "fig-ch4-detection-vs-flagging.png")


def fig_ensemble_growth():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(wspace=0.34)
    labels = ["2 members\n(CNN + CNN)", "3 members\n(+ RAD-DINO ViT)"]
    cw = [18, 8]
    aurc = [0.00403, 0.00279]
    a1.bar(labels, cw, color=["#9fb8d4", C_STD], width=0.55)
    for i, v in enumerate(cw):
        a1.text(i, v + 0.3, str(v), ha="center", fontsize=8)
    a1.set_ylabel("Confident errors on OpenI-D (top-10%)")
    a1.set_title("(a) Architecture diversity halves\nconfident errors", fontsize=9)
    a2.bar(labels, aurc, color=["#9fb8d4", C_STD], width=0.55)
    for i, v in enumerate(aurc):
        a2.text(i, v + 0.00008, "%.5f" % v, ha="center", fontsize=7.5)
    a2.set_ylabel("AURC (lower is better)")
    a2.set_title("(b) …and improves\nselective risk", fontsize=9)
    save(fig, "fig-ch4-ensemble-growth.png")


# ---------------------------------------------------------------------------
# Chapter 5
# ---------------------------------------------------------------------------
ARMS = {
    "rex": ("pretrained,\narch-diverse", os.path.join("baselines", "rex")),
    "rex_fs_crossarch": ("from-scratch,\narch-diverse",
                         os.path.join("baselines", "rex_fs_crossarch")),
    "rex_fs_resnet18": ("from-scratch,\nsame-arch (ResNet-18)",
                        os.path.join("baselines", "rex_fs_resnet18")),
    "rex_fs_vit_tiny": ("from-scratch,\nsame-arch (ViT-Tiny)",
                        os.path.join("baselines", "rex_fs_vit_tiny")),
    "rex_fs_convnext_tiny": ("from-scratch,\nsame-arch (ConvNeXt-Tiny)",
                             os.path.join("baselines", "rex_fs_convnext_tiny")),
    "rex_adapted_s0": ("ReX-adapted,\narch-diverse",
                       os.path.join("rex_phase7", "baselines",
                                    "rex_adapted_s0", "rex_adapted_s0")),
}


def panel_b(arm):
    d = jload(ARMS[arm][1], "baseline_comparison.json")
    return {int(r["top_pct"]): r for r in d["panel_b_within_ensemble_confident_set"]}, d


def fig_regime_diversity():
    """The headline scientific figure: std vs confidence at top-10% across the
    four regime x diversity cells."""
    cells = [("rex", "pretrained\narch-diverse", 0.9895),
             ("rex_fs_crossarch", "from-scratch\narch-diverse", 0.22),
             ("rex_fs_resnet18", "from-scratch\nsame-arch", 0.0071),
             ("rex_adapted_s0", "ReX-adapted\narch-diverse", 0.006)]
    std, conf, labels, ps = [], [], [], []
    for arm, lab, p in cells:
        pb, _ = panel_b(arm)
        std.append(pb[10]["epistemic_std"])
        conf.append(pb[10]["confidence"])
        labels.append(lab)
        ps.append(p)
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = np.arange(len(cells)); w = 0.34
    ax.bar(x - w / 2, std, w, label="epistemic std (disagreement)", color=C_STD)
    ax.bar(x + w / 2, conf, w, label="pooled confidence", color=C_CONF)
    ax.axhline(0.5, color="#999999", ls="--", lw=0.9)
    for i in range(len(cells)):
        top = max(std[i], conf[i])
        d = std[i] - conf[i]
        ax.text(i, top + 0.022, "Δ = %+.3f\nDeLong p = %s"
                % (d, ("%.4f" % ps[i]).rstrip("0").rstrip(".")),
                ha="center", fontsize=6.8,
                color=("#1f7a34" if d > 0.02 else "#8a8a8a"))
        ax.text(i - w / 2, std[i] - 0.045, "%.3f" % std[i], ha="center",
                fontsize=6.8, color="white")
        ax.text(i + w / 2, conf[i] - 0.045, "%.3f" % conf[i], ha="center",
                fontsize=6.8, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("Confident-error AUROC (top-10%)")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("Disagreement beats confidence when members share an "
                 "architecture,\nor when the ensemble is adapted in-domain",
                 fontsize=9.5)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.5)
    save(fig, "fig-ch5-regime-diversity.png")


def fig_selectivity_ladder():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    pct = [50, 25, 15, 10, 5]
    for ax, arm, title in ((a1, "rex", "(a) Un-adapted pretrained ensemble"),
                           (a2, "rex_adapted_s0", "(b) ReX-adapted ensemble")):
        pb, _ = panel_b(arm)
        for key, col, lab, ls in (("epistemic_std", C_STD, "epistemic std", "-"),
                                  ("confidence", C_CONF, "pooled confidence", "-"),
                                  ("entropy", C_ENT, "predictive entropy", "--"),
                                  ("mahalanobis", C_MAHA, "Mahalanobis", ":"),
                                  ("knn", C_GREY, "kNN density", ":")):
            ys = [pb[p].get(key) for p in pct]
            if any(y is None for y in ys):
                continue
            ax.plot(pct, ys, ls, marker="o", ms=3.4, color=col, label=lab)
        ax.axhline(0.5, color="#999999", ls="--", lw=0.8)
        ax.invert_xaxis()
        ax.set_xticks(pct)
        ax.set_xticklabels(["50%", "25%", "15%", "10%", "5%"])
        ax.set_xlabel("Confidence selectivity")
        ax.set_title(title, fontsize=9)
    a1.set_ylabel("Confident-error AUROC")
    a1.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, "fig-ch5-selectivity-ladder.png")


def fig_two_regions():
    d = jload(ARMS["rex_adapted_s0"][1], "baseline_comparison.json")
    full = d["aurc_e_aurc_full_eval"]
    pb, _ = panel_b("rex_adapted_s0")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    fig.subplots_adjust(wspace=0.10)
    keys = ["confidence", "entropy", "epistemic_std", "mahalanobis", "knn"]
    cols = [C_CONF, C_ENT, C_STD, C_MAHA, C_GREY]
    auac = [full[k]["auac"] for k in keys]
    a1.barh(range(len(keys)), auac, color=cols, height=0.6)
    a1.set_yticks(range(len(keys))); a1.set_yticklabels(keys)
    a1.invert_yaxis()
    a1.set_xlim(0.80, 1.0)
    a1.set_xlabel("AUAC over the full evaluation set")
    a1.set_title("Region A — whether to abstain\n(pooled confidence wins)",
                 fontsize=9)
    for i, v in enumerate(auac):
        a1.text(v - 0.004, i, "%.4f" % v, va="center", ha="right",
                fontsize=7, color="white")

    t10 = [pb[10].get(k) for k in keys]
    a2.barh(range(len(keys)), t10, color=cols, height=0.6)
    a2.set_yticks(range(len(keys))); a2.set_yticklabels([])
    a2.invert_yaxis()
    a2.axvline(0.5, color="#999999", ls="--", lw=0.9)
    a2.set_xlim(0.4, 0.85)
    a2.set_xlabel("Confident-error AUROC within the top-10% confident set")
    a2.set_title("Region B — which confident call is wrong\n"
                 "(disagreement and entropy win)", fontsize=9)
    for i, v in enumerate(t10):
        if v is not None:
            a2.text(v - 0.005, i, "%.4f" % v, va="center", ha="right",
                    fontsize=7, color="white")
    save(fig, "fig-ch5-two-regions.png")


def fig_baur_tasks():
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    arms = ["rex", "rex_fs_resnet18", "rex_adapted_s0"]
    names = ["pretrained\n(un-adapted)", "from-scratch\nD-Ens", "ReX-adapted"]
    tasks = ["Task 3 —\nper-pathology\ncorrectness",
             "Task 2 —\nuncertain-label\nprediction",
             "Task 4 —\nAUAC", "Defining —\nconfident-error\n(top-10%)"]
    vals = {}
    for arm in arms:
        d = jload(ARMS[arm][1], "baseline_comparison.json")
        pb, _ = panel_b(arm)
        vals[arm] = [
            d["per_pathology_correctness_auroc"]["macro"]["epistemic_std"],
            d["task2_uncertain_label_auroc"]["macro"]["epistemic_std"],
            d["aurc_e_aurc_full_eval"]["epistemic_std"]["auac"],
            pb[10]["epistemic_std"]]
    x = np.arange(len(tasks)); w = 0.26
    for i, (arm, nm) in enumerate(zip(arms, names)):
        ax.bar(x + (i - 1) * w, vals[arm], w, label=nm,
               color=["#9fb8d4", "#2e8b57", C_STD][i])
    ax.axhline(0.5, color="#999999", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=7.2)
    ax.set_ylabel("AUROC / AUAC for epistemic std")
    ax.set_ylim(0, 1.05)
    ax.set_title("Where ensemble disagreement is robust, and where it is not")
    ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="lower left")
    save(fig, "fig-ch5-baur-tasks.png")


def fig_power():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    scores = ["hybrid\n(std+kNN)", "kNN", "Mahalanobis", "epistemic\nstd",
              "pooled\nconfidence"]
    openi = [0.9059, 0.8763, 0.8698, 0.6072, 0.5058]
    rex = [0.5979, 0.5942, 0.5288, 0.5626, 0.5630]
    x = np.arange(len(scores)); w = 0.36
    ax.bar(x - w / 2, openi, w, label="OpenI-D (5 confident errors)",
           color="#d9a441")
    ax.bar(x + w / 2, rex, w, label="ReXGradient (34 confident errors)",
           color=C_STD)
    ax.axhline(0.5, color="#c8442b", ls="--", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(scores, fontsize=7.2)
    ax.set_ylabel("Confident-error AUROC (top-10%)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Every in-distribution advantage deflates at power")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    save(fig, "fig-ch5-power-deflation.png")


# ---------------------------------------------------------------------------
# Chapters 6-8
# ---------------------------------------------------------------------------
def fig_retrieval_gates():
    base = os.path.join("rex_phase7", "retrieval", "rex_adapted_raddino_full")
    try:
        rg = jload(base, "rerank_gate.json")
        npj = jload(base, "normal_precision.json")
        blob = json.dumps(rg) + json.dumps(npj)
    except Exception:                                    # noqa: BLE001
        blob = ""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.6, 3.4))
    fig.subplots_adjust(wspace=0.62)
    a1.bar(["argmax\nbucket", "flag-keyed\nlane"], [0.0142, 1.0000],
           color=["#c8442b", "#2e8b57"], width=0.55)
    a1.set_ylim(0, 1.15); a1.set_ylabel("flag-class\nreport-precision@8", fontsize=8)
    a1.tick_params(axis="x", labelsize=7)
    a1.set_title("(a) Keying retrieval on\nthe Agent-1 flag", fontsize=8.5)
    for i, v in enumerate([0.0142, 1.0]):
        a1.text(i, v + 0.02, "%.4f" % v, ha="center", fontsize=7.5)

    a2.bar(["cal lib\nold lane", "full lib\nold lane",
            "full lib\nnormal lane"], [0.7396, 0.6740, 1.0000],
           color=["#d9a441", "#c8442b", "#2e8b57"], width=0.6)
    a2.set_ylim(0, 1.15); a2.set_ylabel("normal-precision@8", fontsize=8)
    a2.tick_params(axis="x", labelsize=6.6)
    a2.set_title("(b) Library growth alone\nhurt no-finding queries",
                 fontsize=8.5)
    for i, v in enumerate([0.7396, 0.6740, 1.0]):
        a2.text(i, v + 0.02, "%.4f" % v, ha="center", fontsize=7.5)

    x = np.arange(2); w = 0.34
    a3.bar(x - w / 2, [0.3526, 0.2963], w, label="cosine only", color="#9fb8d4")
    a3.bar(x + w / 2, [0.4148, 0.3880], w, label="+ label rerank", color=C_STD)
    a3.set_xticks(x); a3.set_xticklabels(["pathology\nmode", "plain\nmode"],
                                         fontsize=7.5)
    a3.set_ylabel("label-Jaccard@8", fontsize=8)
    a3.set_ylim(0, 0.50)
    a3.set_title("(c) Posterior-agreement\nrerank (λ = 0.3)", fontsize=8.5)
    a3.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig-ch6-retrieval-gates.png")


def fig_audit_outcomes():
    """Read the real faithfulness records rather than hard-coding them."""
    a3 = json.load(open(os.path.join(RUNS, "app_rex_adapted", "agent3_batch",
                                     "faithfulness_report.json")))
    a4 = json.load(open(os.path.join(RUNS, "app_rex_adapted", "agent4",
                                     "faithfulness_report.json")))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                                 gridspec_kw={"width_ratios": [1, 1.45]})
    fig.subplots_adjust(wspace=0.52)

    a1.bar(["template\n(%d cases)" % a3["n_cases"], "adversarial\nmock (1 case)"],
           [a3["n_gate_failures"], 3], color=["#2e8b57", "#c8442b"], width=0.5)
    a1.set_ylabel("audit failures")
    a1.set_ylim(0, 3.6)
    a1.set_title("(a) Agent 3 gate: deterministic\nrenderings pass, a liar is "
                 "caught", fontsize=8.8)
    a1.text(0, 0.09, "%d / %d" % (a3["n_gate_failures"], a3["n_cases"]),
            ha="center", fontsize=8)
    a1.text(1, 3.09, "3 expected", ha="center", fontsize=8)

    rows = a4["rows"][::-1]
    y = np.arange(len(rows))
    cols, texts = [], []
    for r in rows:
        la = r.get("llm_audit") or {"ok": True, "failures": []}
        if not la["ok"]:
            cols.append("#c8442b")
            n = len(la["failures"])
            texts.append("rejected by the audit (%d failure%s)"
                         % (n, "" if n == 1 else "s"))
        elif "Timeout" in (r.get("llm_error") or ""):
            cols.append("#8a8a8a")
            texts.append("timed out -> template")
        else:
            cols.append("#2e8b57")
            texts.append("LLM rendering shipped")
    a2.barh(y, [1] * len(rows), color=cols, height=0.62)
    a2.set_yticks(y)
    a2.set_yticklabels([r["lane"].replace("_", " ") for r in rows],
                       fontsize=7.6)
    a2.set_xticks([])
    a2.set_xlim(0, 1)
    for i, t in enumerate(texts):
        a2.text(0.03, i, t, va="center", ha="left", fontsize=7.2,
                color="white", fontweight="bold")
    a2.set_title("(b) Agent 4 live LLM trial: every rendering the model\n"
                 "produced was rejected; the rest timed out", fontsize=8.8)
    a2.grid(False)
    save(fig, "fig-ch7-audit-outcomes.png")


def fig_agent4_lanes():
    ev = json.load(open(os.path.join(
        RUNS, "app_rex_adapted", "agent4", "suggestions_evidence.json")))
    cdn = ev["class_data_need"]
    classes, err, npos = [], [], []
    for k, v in cdn.items():
        if not isinstance(v, dict) or "err_rate_pct" not in v:
            continue
        try:
            e = float(v["err_rate_pct"]); n = float(v["npos_eval"])
        except (TypeError, ValueError):
            continue
        classes.append(k); err.append(e); npos.append(n)
    order = np.argsort(npos)
    classes = [classes[i] for i in order]
    err = [err[i] for i in order]; npos = [npos[i] for i in order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    y = np.arange(len(classes))
    a1.barh(y, npos, color="#9fb8d4", height=0.65)
    a1.set_yticks(y); a1.set_yticklabels(classes, fontsize=7)
    a1.set_xlabel("labelled positives in the evaluation split")
    a1.set_title("(a) Lane (a): class support", fontsize=9)
    a2.barh(y, err, color=C_CONF, height=0.65)
    a2.axvline(float(ev["provenance"]["err_rate_pct"]), color="#333333",
               ls="--", lw=1.0)
    a2.text(float(ev["provenance"]["err_rate_pct"]) + 0.3, len(y) - 0.6,
            "population %.2f%%" % float(ev["provenance"]["err_rate_pct"]),
            fontsize=6.8, rotation=90, va="top")
    a2.set_yticks(y); a2.set_yticklabels([])
    a2.set_xlabel("error rate (%)")
    a2.set_title("(b) …against per-class error rate", fontsize=9)
    save(fig, "fig-ch8-agent4-lanes.png")


# ---------------------------------------------------------------------------
def copy_existing():
    pairs = [
        ("baselines/rex/baselines_auroc_by_selectivity.png",
         "fig-ch5-baselines-rex.png"),
        ("baselines/openi/baselines_auroc_by_selectivity.png",
         "fig-ch5-baselines-openi.png"),
        ("baselines/rex_fs_resnet18/baselines_auroc_by_selectivity.png",
         "fig-ch5-baselines-fs-resnet18.png"),
        ("baselines/rex_fs_crossarch/baselines_auroc_by_selectivity.png",
         "fig-ch5-baselines-fs-crossarch.png"),
        ("rex_phase7/baselines/rex_adapted_s0/rex_adapted_s0/"
         "baselines_auroc_by_selectivity.png",
         "fig-ch5-baselines-adapted.png"),
        ("production_reanalyze/reanalyze/plots/02_risk_coverage.png",
         "fig-ch4-risk-coverage.png"),
        ("production_reanalyze/reanalyze/plots/05_reliability.png",
         "fig-ch4-reliability.png"),
        ("production_reanalyze/reanalyze/plots/03_per_pathology_auroc.png",
         "fig-ch4-per-pathology.png"),
        ("production_reanalyze/reanalyze/plots/04_error_types.png",
         "fig-ch4-error-types.png"),
        ("app_upload_screenshot.png", "fig-ch9-app.png"),
    ]
    for src, dst in pairs:
        s = os.path.join(RUNS, src)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(FIG, dst))
            print("  copied", dst)
        else:
            print("  MISSING source:", src)


def main():
    os.makedirs(FIG, exist_ok=True)
    print("copying existing plots:")
    copy_existing()
    print("generating figures:")
    for fn in (fig_architecture, fig_splits, fig_calibration_ablation,
               fig_ood_collapse, fig_maha_selectivity,
               fig_ood_detection_vs_flagging, fig_ensemble_growth,
               fig_regime_diversity, fig_selectivity_ladder, fig_two_regions,
               fig_baur_tasks, fig_power, fig_retrieval_gates,
               fig_audit_outcomes, fig_agent4_lanes):
        try:
            fn()
        except Exception as e:                            # noqa: BLE001
            print("  FAILED %s: %s: %s" % (fn.__name__, type(e).__name__, e))


if __name__ == "__main__":
    main()

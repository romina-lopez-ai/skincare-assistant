"""visualize.py — Publication-ready figures from viz_summary.json.

Figure 1 — Line chart         : 4 configs × 4 RAGAS metrics, shaded improvement region
Figure 2 — Bar chart          : grouped bars by metric, 4 configs
Figure 3 — Heatmap            : per-condition [4 RAGAS | HR@2] for FT+RAG
Figure 4 — Bar chart per cond : per-condition bars grouped by metric (FT+RAG only)

Run from repo root:
    python evaluation_metrics/rag/visualize.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_metrics.rag.config import RESULTS_DIR

VIZ_IN_PATH = RESULTS_DIR / "viz_summary.json"
FIG_DIR     = RESULTS_DIR / "figures"

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.9,
    "axes.grid":         True,
    "grid.color":        "#d0d0d0",
    "grid.alpha":        0.6,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.55,
    "legend.framealpha": 0.93,
    "legend.edgecolor":  "#cccccc",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

# ── Per-config styling (Figures 1 & 2) ───────────────────────────────────────
STYLE = {
    "finetuned_rag":    {"color": "#D4704A", "ls": (0, (6, 2)),       "marker": "o", "lw": 2.4, "ms": 8, "z": 5},
    "base_rag":         {"color": "#3A9E6E", "ls": "solid",           "marker": "s", "lw": 1.8, "ms": 7, "z": 4},
    "finetuned_no_rag": {"color": "#5B8DB8", "ls": (0, (5, 2, 1, 2)), "marker": "^", "lw": 1.8, "ms": 7, "z": 3},
    "base_no_rag":      {"color": "#999999", "ls": (0, (2, 2)),       "marker": "D", "lw": 1.4, "ms": 6, "z": 2},
}

# ── Per-metric styling (Figures 3 & 4) ───────────────────────────────────────
METRIC_COLOR = {
    "faithfulness":      "#D4704A",
    "answer_relevancy":  "#3A9E6E",
    "context_precision": "#5B8DB8",
    "context_recall":    "#9B59B6",
    "hr_at_2":           "#E8A838",
}

METRIC_LABEL = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer\nRelevancy",
    "context_precision": "Context\nPrecision",
    "context_recall":    "Context\nRecall",
    "hr_at_2":           "HR@2",
}

METRIC_SHORT = {
    "faithfulness":      "Faith.",
    "answer_relevancy":  "Ans. Rel.",
    "context_precision": "Ctx. Prec.",
    "context_recall":    "Ctx. Rec.",
    "hr_at_2":           "HR@2",
}

COND_LABEL = {"acne": "Acne", "eczema": "Eczema", "psoriasis": "Psoriasis"}

THRESHOLDS = {
    "faithfulness":      0.80,
    "answer_relevancy":  0.75,
    "context_precision": 0.70,
    "context_recall":    0.65,
}

TITLE_PREFIX = "RAG Evaluation — Ablation Study"


def load_summary() -> dict:
    return json.loads(VIZ_IN_PATH.read_text(encoding="utf-8"))


# ── Figure 1: Line chart with shaded improvement region ──────────────────────

def fig1_line_chart(data: dict) -> plt.Figure:
    metrics = data["metrics"]
    configs = data["configs"]
    x       = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    # Shaded region: total improvement (FT+RAG vs Base+no-RAG)
    ft_means   = np.array([configs["finetuned_rag"][m]["mean"] for m in metrics])
    base_means = np.array([configs["base_no_rag"][m]["mean"]   for m in metrics])
    ax.fill_between(x, base_means, ft_means,
                    color=STYLE["finetuned_rag"]["color"],
                    alpha=0.13, zorder=1, label="_nolegend_")

    for cfg, sty in STYLE.items():
        if cfg not in configs:
            continue
        means = np.array([configs[cfg][m]["mean"] for m in metrics])
        ax.plot(x, means,
                color=sty["color"], linestyle=sty["ls"],
                linewidth=sty["lw"], marker=sty["marker"],
                markersize=sty["ms"], zorder=sty["z"],
                label=configs[cfg]["label"],
                solid_capstyle="round", solid_joinstyle="round")

    # Acceptance threshold markers
    for m_idx, m in enumerate(metrics):
        ax.plot(m_idx, THRESHOLDS[m], marker="_", color="#888888",
                markersize=12, markeredgewidth=1.5, zorder=6)
    ax.plot([], [], marker="_", color="#888888", markersize=10,
            markeredgewidth=1.5, linestyle="none", label="Threshold")

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
    ax.set_ylabel("Score (0 – 1)", labelpad=8)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xlim(-0.28, len(metrics) - 0.72)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", which="major")
    ax.grid(axis="y", which="minor", alpha=0.25, linewidth=0.35)
    ax.set_axisbelow(True)

    ax.set_title(TITLE_PREFIX, fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=8.5, ncol=1,
              handlelength=2.2, borderpad=0.6, labelspacing=0.45)

    fig.tight_layout()
    return fig


# ── Figure 2: Grouped bar chart (4 configs × 4 metrics) ──────────────────────

def fig2_bar_chart(data: dict) -> plt.Figure:
    metrics  = data["metrics"]
    configs  = data["configs"]
    cfg_keys = list(STYLE.keys())
    n_cfg    = len(cfg_keys)

    bar_w   = 0.17
    x       = np.arange(len(metrics))
    offsets = np.linspace(-(n_cfg - 1) / 2, (n_cfg - 1) / 2, n_cfg) * bar_w

    fig, ax = plt.subplots(figsize=(9, 4.6))

    for i, cfg in enumerate(cfg_keys):
        if cfg not in configs:
            continue
        sty   = STYLE[cfg]
        means = np.array([configs[cfg][m]["mean"] for m in metrics])
        stds  = np.array([configs[cfg][m]["std"]  for m in metrics])

        ax.bar(x + offsets[i], means,
               width=bar_w * 0.90, color=sty["color"], alpha=0.82,
               label=configs[cfg]["label"], zorder=3)
        #ax.errorbar(x + offsets[i], means, yerr=stds,
        #            fmt="none", color=sty["color"],
        #            elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)

    # Threshold lines per metric group
    for m_idx, m in enumerate(metrics):
        thr = THRESHOLDS[m]
        ax.hlines(thr,
                  xmin=m_idx - (n_cfg / 2) * bar_w - 0.02,
                  xmax=m_idx + (n_cfg / 2) * bar_w + 0.02,
                  colors="#555555", linewidths=1.4, linestyles="--", zorder=5)
    ax.hlines([], [], [], colors="#555555", linewidths=1.4,
              linestyles="--", label="Threshold")

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
    ax.set_ylabel("Score (0 – 1)", labelpad=8)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis="y", which="major", zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(TITLE_PREFIX, fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2,
              handlelength=1.8, columnspacing=1.0)

    fig.tight_layout()
    return fig


# ── Figure 3: Heatmap per condition ──────────────────────────────────────────

def fig3_heatmap(data: dict) -> plt.Figure:
    metrics     = data["metrics"]
    conditions  = list(data["per_condition"].keys())
    cond_labels = [COND_LABEL.get(c, c.capitalize()) for c in conditions]

    col_keys   = metrics + ["hr_at_2"]
    col_labels = [METRIC_LABEL[m] for m in metrics] + ["HR@2"]

    matrix = np.array([
        [data["per_condition"][cond].get(col, 0.0) for col in col_keys]
        for cond in conditions
    ])

    # Temporarily disable grid for heatmap
    with plt.rc_context({"axes.grid": False}):
        fig, ax = plt.subplots(figsize=(9.5, 2.8))
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(len(conditions)):
        for j in range(len(col_keys)):
            val   = matrix[i, j]
            color = "white" if val > 0.58 else "#1a1a1a"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11.5, color=color, fontweight="bold")

    # Separator between RAGAS and HR@2
    sep = len(metrics) - 0.5
    ax.axvline(sep, color="white",   linewidth=4.0)
    ax.axvline(sep, color="#666666", linewidth=0.9, linestyle="--", alpha=0.5)

    # Labels at BOTTOM
    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(cond_labels, fontsize=11)
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True,
                   axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Score", fontsize=9.5)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("RAG Evaluation — Per-condition", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


# ── Figure 4: Bar chart per condition ────────────────────────────────────────

COND_COLOR = {
    "acne":      "#D4704A",
    "eczema":    "#3A9E6E",
    "psoriasis": "#5B8DB8",
}

def fig4_bar_condition(data: dict) -> plt.Figure:
    conditions = list(data["per_condition"].keys())
    all_cols   = data["metrics"] + ["hr_at_2"]
    n_cond     = len(conditions)
    n_cols     = len(all_cols)

    bar_w   = 0.22
    x       = np.arange(n_cols)
    offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * bar_w

    fig, ax = plt.subplots(figsize=(9, 4.6))

    for i, cond in enumerate(conditions):
        vals = [data["per_condition"][cond].get(col, 0.0) for col in all_cols]
        ax.bar(x + offsets[i], vals,
               width=bar_w * 0.88,
               color=COND_COLOR[cond],
               alpha=0.82,
               label=COND_LABEL[cond],
               zorder=3)

    # Separator before HR@2 (last column)
    ax.axvline(n_cols - 1 - 0.5, color="#bbbbbb", linewidth=0.9,
               linestyle="--", alpha=0.6, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[c] for c in all_cols], fontsize=10)
    ax.set_ylabel("Score (0 – 1)", labelpad=8)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(axis="y", which="major", zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("RAG Evaluation — Per-condition", fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=9, ncol=1,
              handlelength=1.5, borderpad=0.6)

    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not VIZ_IN_PATH.exists():
        print(f"[ERROR] viz_summary.json not found: {VIZ_IN_PATH}")
        print("  Run first: python evaluation_metrics/rag/generate_viz_data.py")
        sys.exit(1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_summary()

    figures = [
        ("fig1_line_chart",      fig1_line_chart(data)),
        ("fig2_bar_chart",       fig2_bar_chart(data)),
        ("fig3_heatmap",         fig3_heatmap(data)),
        ("fig4_bar_per_cond",    fig4_bar_condition(data)),
    ]

    for name, fig in figures:
        for ext in ("png", "pdf"):
            out = FIG_DIR / f"{name}.{ext}"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

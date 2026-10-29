#!/usr/bin/env python3
"""Generate evaluation charts from HalluGuard quick demo results.

Creates a figure with 3 subplots:
  1. Hallucination Rate comparison (bar)
  2. Mean Faithfulness Score comparison (bar)
  3. Mitigation Strategy Distribution (pie)

Saves PNG (200 dpi) and PDF to experiments/results/.
"""

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ────────────────────────────────────────────────────────────
RESULTS_JSON = os.path.join(_SCRIPT_DIR, "results", "quick_demo_results.json")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results")
OUT_PNG = os.path.join(OUT_DIR, "halluguard_results.png")
OUT_PDF = os.path.join(OUT_DIR, "halluguard_results.pdf")

# ── Colors ───────────────────────────────────────────────────────────
RED = "#E74C3C"
GREEN = "#27AE60"
PIE_COLORS = {
    "none": "#27AE60",
    "rewrite": "#3498DB",
    "rerank": "#E67E22",
    "self_reflection": "#9B59B6",
    "query_rewrite": "#3498DB",
}
PIE_LABELS = {
    "none": "No mitigation",
    "rewrite": "Query rewrite",
    "query_rewrite": "Query rewrite",
    "rerank": "Context rerank",
    "self_reflection": "Self-reflection",
}


def main():
    # ── Load results ─────────────────────────────────────────────────
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)

    metrics = data["metrics"]
    bl = metrics["baseline"]
    hg = metrics["halluguard"]
    tiers = metrics["mitigation_tier_counts"]
    n_samples = metrics["num_samples"]

    # ── Figure setup ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"HalluGuard Evaluation Results — HaluEval ({n_samples} samples)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # ── Chart 1: Hallucination Rate ──────────────────────────────────
    ax1 = axes[0]
    labels1 = ["Baseline RAG", "HalluGuard"]
    values1 = [bl["hallucination_rate_pct"], hg["hallucination_rate_pct"]]
    colors1 = [RED, GREEN]
    bars1 = ax1.bar(labels1, values1, color=colors1, width=0.5, edgecolor="white")
    for bar, val in zip(bars1, values1):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    ax1.set_title("Hallucination Rate (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Hallucination Rate (%)")
    ax1.set_ylim(0, max(values1) * 1.25)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # ── Chart 2: Mean Faithfulness Score ─────────────────────────────
    ax2 = axes[1]
    labels2 = ["Baseline RAG", "HalluGuard"]
    values2 = [bl["mean_faithfulness"], hg["mean_faithfulness"]]
    colors2 = [RED, GREEN]
    bars2 = ax2.bar(labels2, values2, color=colors2, width=0.5, edgecolor="white")
    for bar, val in zip(bars2, values2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    ax2.set_title("Mean Faithfulness Score", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Faithfulness Score")
    ax2.set_ylim(0, 1.0)

    # ── Chart 3: Mitigation Strategy Pie ─────────────────────────────
    ax3 = axes[2]
    pie_labels = []
    pie_sizes = []
    pie_colors = []
    for tier, count in sorted(tiers.items()):
        pie_labels.append(PIE_LABELS.get(tier, tier))
        pie_sizes.append(count)
        pie_colors.append(PIE_COLORS.get(tier, "#95A5A6"))

    wedges, texts, autotexts = ax3.pie(
        pie_sizes,
        labels=pie_labels,
        colors=pie_colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * sum(pie_sizes)))})",
        startangle=140,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax3.set_title("Mitigation Strategy Used", fontsize=13, fontweight="bold")

    # ── Save ─────────────────────────────────────────────────────────
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"✅ PNG saved: {OUT_PNG}")
    print(f"✅ PDF saved: {OUT_PDF}")
    plt.show()


if __name__ == "__main__":
    main()

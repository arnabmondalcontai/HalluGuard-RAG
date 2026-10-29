#!/usr/bin/env python3
"""Print a formatted metrics table from HalluGuard quick demo results.

Reads experiments/results/quick_demo_results.json and renders:
  - Main metrics comparison table (Baseline vs HalluGuard)
  - Mitigation tier breakdown
  - Top 3 improvement cases
"""

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

RESULTS_JSON = os.path.join(_SCRIPT_DIR, "results", "quick_demo_results.json")

# ── Try tabulate, fall back to manual formatting ─────────────────────
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def _fmt_table(headers, rows, title=None):
    """Render a table with tabulate or manual formatting."""
    if HAS_TABULATE:
        txt = tabulate(rows, headers=headers, tablefmt="pretty", stralign="right")
    else:
        # Manual column-width formatting
        col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]
        sep = "+" + "+".join("-" * w for w in col_widths) + "+"
        hdr = "|" + "|".join(str(headers[i]).center(col_widths[i]) for i in range(len(headers))) + "|"
        lines = [sep, hdr, sep]
        for row in rows:
            lines.append("|" + "|".join(str(row[i]).rjust(col_widths[i] - 1) + " " for i in range(len(row))) + "|")
        lines.append(sep)
        txt = "\n".join(lines)
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    print(txt)


def main():
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)

    metrics = data["metrics"]
    bl = metrics["baseline"]
    hg = metrics["halluguard"]
    tiers = metrics["mitigation_tier_counts"]
    n_samples = metrics["num_samples"]
    best = data.get("best_improvements", [])

    # ── Derived values ───────────────────────────────────────────────
    halluc_reduction = bl["hallucination_rate_pct"] - hg["hallucination_rate_pct"]
    faith_improvement = hg["mean_faithfulness"] - bl["mean_faithfulness"]

    # ── Main metrics table ───────────────────────────────────────────
    headers = ["Metric", "Baseline RAG", "HalluGuard"]
    rows = [
        ["Hallucination Rate", f"{bl['hallucination_rate_pct']:.2f}%", f"{hg['hallucination_rate_pct']:.2f}%"],
        ["Mean Faithfulness", f"{bl['mean_faithfulness']:.4f}", f"{hg['mean_faithfulness']:.4f}"],
        ["AUROC", f"{bl['auroc']:.4f}", f"{hg['auroc']:.4f}"],
        ["Mean Latency (ms)", f"{bl['latency_mean_ms']:.2f}", f"{hg['latency_mean_ms']:.2f}"],
        ["p95 Latency (ms)", f"{bl['latency_p95_ms']:.2f}", f"{hg['latency_p95_ms']:.2f}"],
        ["", "", ""],
        ["Hallucination Reduction", "—", f"-{halluc_reduction:.1f} points"],
        ["Faithfulness Improvement", "—", f"+{faith_improvement:.4f}"],
    ]
    _fmt_table(headers, rows, title=f"HALLUGUARD EVALUATION RESULTS — HaluEval ({n_samples} samples)")

    # ── Mitigation tier breakdown ────────────────────────────────────
    tier_labels = {
        "none": "No mitigation",
        "rewrite": "Query rewriting",
        "query_rewrite": "Query rewriting",
        "rerank": "Context reranking",
        "self_reflection": "Self-reflection",
    }
    tier_headers = ["Strategy", "Samples", "Percentage"]
    tier_rows = []
    for tier, count in sorted(tiers.items()):
        pct = count / n_samples * 100
        tier_rows.append([tier_labels.get(tier, tier), str(count), f"{pct:.1f}%"])
    _fmt_table(tier_headers, tier_rows, title="MITIGATION TIER BREAKDOWN")

    # ── Top 3 improvement cases ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TOP 3 IMPROVEMENT CASES")
    print(f"{'=' * 60}")
    for rank, case in enumerate(best[:3], start=1):
        improvement = case.get("improvement", 0)
        print(f"\n{'─' * 55}")
        print(f"  Case #{rank}  (improvement: {improvement:+.4f})")
        print(f"{'─' * 55}")
        print(f"  ❓ Question:     {case.get('question', 'N/A')}")
        bl_resp = case.get("baseline_response", "")
        hg_resp = case.get("halluguard_response", "")
        print(f"  🔴 Baseline:     {bl_resp[:180]}{'…' if len(bl_resp) > 180 else ''}")
        print(f"     Score:        {case.get('baseline_score', 'N/A'):.4f}")
        print(f"  🟢 HalluGuard:   {hg_resp[:180]}{'…' if len(hg_resp) > 180 else ''}")
        print(f"     Score:        {case.get('halluguard_score', 'N/A'):.4f}")
        print(f"  🛠️  Strategy:     {case.get('strategy_used', 'N/A')}")
    print(f"\n{'─' * 55}\n")


if __name__ == "__main__":
    main()

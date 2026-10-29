#!/usr/bin/env python3
"""HalluGuard Quick Demo — Baseline vs. Full Pipeline on HaluEval QA.

Loads 500 samples from pminervini/HaluEval qa_samples, runs both a baseline
RAG pipeline (NLI-only scoring, no mitigation) and the full HalluGuard
pipeline (EnsembleDetector + ConfidenceAdaptiveRouter), then computes and
prints comparative metrics.

Includes checkpoint/resume support (every 50 samples), rate-limit handling
for Groq API 429 errors (60s wait + retry), and ETA progress display.

Usage:
    cd halluguard/
    python -m experiments.quick_demo
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the project root (halluguard/) is on sys.path so that `src.*`
# imports resolve correctly regardless of how the script is invoked.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SAMPLES = 500
HALLUCINATION_THRESHOLD = 0.5
GROQ_SLEEP_SECONDS = 1  # Rate-limit pause between Groq API calls
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "experiments", "results")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "checkpoint_500.json")
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N samples
RESULTS_FILE_500 = os.path.join(RESULTS_DIR, "quick_demo_500_results.json")
RATE_LIMIT_WAIT = 60  # Seconds to wait on Groq 429


# ===================================================================
# 1. Data Loading
# ===================================================================

def load_samples(n: int = NUM_SAMPLES) -> List[Dict[str, Any]]:
    """Load *n* samples from the HaluEval QA split.

    Each raw HaluEval row contains a question, knowledge paragraph,
    an answer, and a hallucination label ('yes' or 'no').
    """
    from datasets import load_dataset

    print(f"\n📥 Loading HaluEval qa_samples from HuggingFace …")
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    samples: List[Dict[str, Any]] = []
    for item in ds:
        knowledge = item.get("knowledge", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        hallucination = item.get("hallucination", "")

        # 'yes' means it has hallucination (label 1), 'no' means it is faithful (label 0)
        label = 1 if hallucination == "yes" else 0

        samples.append({
            "query": question,
            "context": [knowledge] if knowledge else [],
            "response": answer,
            "label": label,
        })

        # Stop once we have enough
        if len(samples) >= n:
            break

    n_faithful = sum(1 for s in samples if s["label"] == 0)
    n_hallucinated = sum(1 for s in samples if s["label"] == 1)
    print(f"   ✅ Loaded {len(samples)} samples  "
          f"({n_faithful} faithful, {n_hallucinated} hallucinated)\n")
    return samples



# ===================================================================
# 2. Pipeline Helpers
# ===================================================================

class MockRetriever:
    """Mock retriever that returns the pre-defined context chunks for the current sample."""

    def __init__(self, context_chunks: List[str]):
        self.context_chunks = context_chunks

    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        from src.rag_pipeline.retriever import Document
        return [
            Document(page_content=chunk, metadata={"source": "mock"})
            for chunk in self.context_chunks
        ]


def _init_components():
    """Lazily initialise all shared pipeline components once."""
    from src.detection.nli_detector import NLIDetector
    from src.detection.embedding_detector import EmbeddingDetector
    from src.detection.llm_judge import LLMJudge
    from src.detection.ensemble import EnsembleDetector
    from src.rag_pipeline.generator import LLMGenerator
    from src.mitigation.router import ConfidenceAdaptiveRouter

    print("🔧 Loading models — one time only …")

    # --- Detectors (singleton-cached, loaded once) ---
    nli_detector = NLIDetector()
    emb_detector = EmbeddingDetector()
    llm_judge = LLMJudge()
    ensemble_detector = EnsembleDetector(
        nli_detector=nli_detector,
        embedding_detector=emb_detector,
        llm_judge=llm_judge,
    )

    # --- Generator (shared across baseline & full pipeline) ---
    generator = LLMGenerator()

    # --- Router (for full pipeline) ---
    router = ConfidenceAdaptiveRouter(
        generator=generator,
        detector=ensemble_detector,
    )

    # --- Warmup: trigger model loading before timed loop ---
    print("   ⏳ Warming up models …")
    _ = ensemble_detector.score("test", "test response", ["test context"])
    print("   ✅ All components ready and warmed up.\n")

    return nli_detector, ensemble_detector, generator, router


def run_baseline(
    sample: Dict[str, Any],
    nli_detector,
    generator,
) -> Dict[str, Any]:
    """Baseline: generate a response with the LLM then score with NLI only.

    Since HaluEval already provides a pre-generated response we score that
    directly — no retrieval step is needed (context is provided in the sample).
    If the response field is empty we generate one via the LLM.
    """
    start = time.perf_counter()

    query = sample["query"]
    context_chunks = sample["context"]
    response = sample.get("response", "")

    # Fall back to LLM generation if no pre-generated response
    if not response.strip():
        response = generator.generate(query=query, context_chunks=context_chunks)
        time.sleep(GROQ_SLEEP_SECONDS)

    # Score with NLI detector
    hallucination_score = nli_detector.score(response=response, context_chunks=context_chunks)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "hallucination_score": hallucination_score,
        "response": response,
        "latency_ms": round(elapsed_ms, 2),
    }


def run_halluguard(
    sample: Dict[str, Any],
    ensemble_detector,
    generator,
    router,
) -> Dict[str, Any]:
    """Full HalluGuard pipeline: EnsembleDetector + ConfidenceAdaptiveRouter.

    1. Score the (pre-existing) response with the ensemble detector.
    2. If hallucination detected, route through the adaptive router.
    3. Return the final response, score, and strategy used.
    """
    start = time.perf_counter()

    query = sample["query"]
    context_chunks = sample["context"]
    response = sample.get("response", "")

    if not response.strip():
        response = generator.generate(query=query, context_chunks=context_chunks)
        time.sleep(GROQ_SLEEP_SECONDS)

    # ---- Ensemble detection ----
    detection_result = ensemble_detector.score(
        query=query,
        response=response,
        context_chunks=context_chunks,
    )
    time.sleep(GROQ_SLEEP_SECONDS)  # LLM-judge inside ensemble uses Groq

    hallucination_score = detection_result.get("final_score", 0.0)
    is_hallucinated = hallucination_score > HALLUCINATION_THRESHOLD

    final_response = response
    mitigation_applied = "none"
    final_score = hallucination_score

    # ---- Mitigation (if needed) ----
    if is_hallucinated:
        try:
            # Set MockRetriever dynamically for the current sample's context
            router.retriever = MockRetriever(context_chunks)

            router_result = router.route(
                query=query,
                response=response,
                context_chunks=context_chunks,
                hallucination_score=hallucination_score,
                detection_result=detection_result,
            )
            time.sleep(GROQ_SLEEP_SECONDS)

            final_response = router_result.get("final_response", response)
            mitigation_applied = router_result.get("strategy_used", "none")
            final_score = router_result.get("final_score", hallucination_score)
        except Exception as exc:
            logger.warning("Router mitigation failed: %s", exc)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "final_score": final_score,
        "initial_ensemble_score": hallucination_score,
        "mitigation_applied": mitigation_applied,
        "final_response": final_response,
        "latency_ms": round(elapsed_ms, 2),
    }


# ===================================================================
# 3. Metrics
# ===================================================================

def compute_metrics(
    samples: List[Dict[str, Any]],
    baseline_results: List[Dict[str, Any]],
    halluguard_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute all comparative metrics."""
    from sklearn.metrics import roc_auc_score

    ground_truth = [s["label"] for s in samples]

    # --- Hallucination rates ---
    baseline_scores = [r["hallucination_score"] for r in baseline_results]
    halluguard_scores = [r["final_score"] for r in halluguard_results]

    baseline_halluc_rate = sum(
        1 for s in baseline_scores if s > HALLUCINATION_THRESHOLD
    ) / len(baseline_scores) * 100

    halluguard_halluc_rate = sum(
        1 for s in halluguard_scores if s > HALLUCINATION_THRESHOLD
    ) / len(halluguard_scores) * 100

    # --- Mean faithfulness (1 - mean_hallucination_score) ---
    baseline_faithfulness = 1.0 - float(np.mean(baseline_scores))
    halluguard_faithfulness = 1.0 - float(np.mean(halluguard_scores))

    # --- AUROC using HaluEval labels as ground truth ---
    try:
        baseline_auroc = float(roc_auc_score(ground_truth, baseline_scores))
    except ValueError:
        baseline_auroc = 0.5

    try:
        halluguard_auroc = float(roc_auc_score(ground_truth, halluguard_scores))
    except ValueError:
        halluguard_auroc = 0.5

    # --- Per-mitigation-tier counts ---
    tier_counts = Counter(r["mitigation_applied"] for r in halluguard_results)

    # --- Latency ---
    baseline_latencies = [r["latency_ms"] for r in baseline_results]
    halluguard_latencies = [r["latency_ms"] for r in halluguard_results]

    return {
        "num_samples": len(samples),
        "baseline": {
            "hallucination_rate_pct": round(baseline_halluc_rate, 2),
            "mean_faithfulness": round(baseline_faithfulness, 4),
            "auroc": round(baseline_auroc, 4),
            "latency_p95_ms": round(float(np.percentile(baseline_latencies, 95)), 2),
            "latency_mean_ms": round(float(np.mean(baseline_latencies)), 2),
        },
        "halluguard": {
            "hallucination_rate_pct": round(halluguard_halluc_rate, 2),
            "mean_faithfulness": round(halluguard_faithfulness, 4),
            "auroc": round(halluguard_auroc, 4),
            "latency_p95_ms": round(float(np.percentile(halluguard_latencies, 95)), 2),
            "latency_mean_ms": round(float(np.mean(halluguard_latencies)), 2),
        },
        "mitigation_tier_counts": dict(tier_counts),
    }


# ===================================================================
# 4. Reporting
# ===================================================================

def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print the comparative metrics table."""
    b = metrics["baseline"]
    h = metrics["halluguard"]
    tiers = metrics["mitigation_tier_counts"]

    header = "=" * 64
    print(f"\n{header}")
    print("          HalluGuard Quick Demo — Results Summary")
    print(f"{header}\n")

    print(f"  {'Metric':<35} {'Baseline':>12}  {'HalluGuard':>12}")
    print(f"  {'-'*35} {'-'*12}  {'-'*12}")
    print(f"  {'Hallucination Rate (%)':<35} {b['hallucination_rate_pct']:>11.2f}%  {h['hallucination_rate_pct']:>11.2f}%")
    print(f"  {'Mean Faithfulness':<35} {b['mean_faithfulness']:>12.4f}  {h['mean_faithfulness']:>12.4f}")
    print(f"  {'AUROC (vs HaluEval labels)':<35} {b['auroc']:>12.4f}  {h['auroc']:>12.4f}")
    print(f"  {'p95 Latency (ms)':<35} {b['latency_p95_ms']:>12.2f}  {h['latency_p95_ms']:>12.2f}")
    print(f"  {'Mean Latency (ms)':<35} {b['latency_mean_ms']:>12.2f}  {h['latency_mean_ms']:>12.2f}")

    print(f"\n  📊 Per-Mitigation-Tier Counts:")
    for tier, count in sorted(tiers.items()):
        print(f"     • {tier:<25} {count:>4} samples")

    print()


def find_best_improvements(
    samples: List[Dict[str, Any]],
    baseline_results: List[Dict[str, Any]],
    halluguard_results: List[Dict[str, Any]],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Find the top_n samples with the biggest score improvement."""
    improvements = []
    for i, (sample, bl, hg) in enumerate(
        zip(samples, baseline_results, halluguard_results)
    ):
        improvement = bl["hallucination_score"] - hg["final_score"]
        improvements.append({
            "index": i,
            "improvement": improvement,
            "question": sample["query"],
            "context_snippet": (sample["context"][0][:200] + "…")
                if sample["context"] and len(sample["context"][0]) > 200
                else (sample["context"][0] if sample["context"] else "(no context)"),
            "baseline_response": bl["response"],
            "halluguard_response": hg["final_response"],
            "baseline_score": bl["hallucination_score"],
            "halluguard_score": hg["final_score"],
            "strategy_used": hg["mitigation_applied"],
        })

    improvements.sort(key=lambda x: x["improvement"], reverse=True)
    return improvements[:top_n]


def print_best_cases(cases: List[Dict[str, Any]]) -> None:
    """Pretty-print the best improvement case studies."""
    print("=" * 64)
    print("          🏆 Top 3 Improvement Cases")
    print("=" * 64)

    for rank, case in enumerate(cases, start=1):
        print(f"\n{'─' * 60}")
        print(f"  Case #{rank}  (improvement: {case['improvement']:+.4f})")
        print(f"{'─' * 60}")
        print(f"  ❓ Question:       {case['question']}")
        print(f"  📚 Context:        {case['context_snippet']}")
        print(f"  🔴 Baseline:       {case['baseline_response'][:200]}{'…' if len(case['baseline_response']) > 200 else ''}")
        print(f"     Score:          {case['baseline_score']:.4f}")
        print(f"  🟢 HalluGuard:     {case['halluguard_response'][:200]}{'…' if len(case['halluguard_response']) > 200 else ''}")
        print(f"     Score:          {case['halluguard_score']:.4f}")
        print(f"  🛠️  Strategy:       {case['strategy_used']}")

    print(f"\n{'─' * 60}\n")


# ===================================================================
# 5. Checkpoint Helpers
# ===================================================================

def _load_checkpoint() -> dict | None:
    """Load checkpoint if it exists and return saved state."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            print(f"🔄 Resuming from sample {checkpoint['completed']}")
            return checkpoint
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Corrupt checkpoint, starting fresh: %s", exc)
    return None


def _save_checkpoint(
    completed: int,
    baseline_results: List[Dict[str, Any]],
    halluguard_results: List[Dict[str, Any]],
) -> None:
    """Save current progress to checkpoint file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint = {
        "completed": completed,
        "baseline_results": baseline_results,
        "halluguard_results": halluguard_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Checkpoint saved at sample %d", completed)


def _run_with_rate_limit_retry(func, *args, sample_idx: int = 0, **kwargs):
    """Run a function, catching Groq 429 errors with a 60s wait + retry."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            exc_str = str(exc)
            # Check for Groq 429 rate-limit error
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "Rate limit" in exc_str:
                print(f"Rate limited. Waiting {RATE_LIMIT_WAIT}s... (sample {sample_idx})")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            else:
                raise  # Re-raise non-rate-limit errors
    # If all retries exhausted, raise
    raise RuntimeError(f"Rate limit retries exhausted for sample {sample_idx}")


# ===================================================================
# 6. Main
# ===================================================================

def main() -> None:
    # --- Load data ---
    samples = load_samples(NUM_SAMPLES)

    # --- Init components (includes warmup — not timed) ---
    nli_detector, ensemble_detector, generator, router = _init_components()

    # --- Check for checkpoint ---
    checkpoint = _load_checkpoint()
    if checkpoint:
        start_idx = checkpoint["completed"]
        baseline_results = checkpoint["baseline_results"]
        halluguard_results = checkpoint["halluguard_results"]
    else:
        start_idx = 0
        baseline_results = []
        halluguard_results = []

    # --- Start timing AFTER warmup ---
    start_wall = time.time()
    remaining = len(samples) - start_idx

    # --- Run pipelines ---
    print(f"🚀 Running evaluation ({start_idx}/{len(samples)} already done) …\n")
    for i in range(start_idx, len(samples)):
        sample = samples[i]
        sample_start = time.perf_counter()

        # 2a. Baseline (NLI-only, no mitigation)
        try:
            bl = _run_with_rate_limit_retry(
                run_baseline, sample, nli_detector, generator,
                sample_idx=i,
            )
        except Exception as exc:
            logger.error("Baseline failed for sample %d: %s", i, exc)
            bl = {
                "hallucination_score": 0.5,
                "response": sample.get("response", ""),
                "latency_ms": 0.0,
            }
        baseline_results.append(bl)

        # 2b. Full HalluGuard pipeline
        try:
            hg = _run_with_rate_limit_retry(
                run_halluguard, sample, ensemble_detector, generator, router,
                sample_idx=i,
            )
        except Exception as exc:
            logger.error("HalluGuard failed for sample %d: %s", i, exc)
            hg = {
                "final_score": 0.5,
                "initial_ensemble_score": 0.5,
                "mitigation_applied": "error",
                "final_response": sample.get("response", ""),
                "latency_ms": 0.0,
            }
        halluguard_results.append(hg)

        # --- Progress display every 10 samples ---
        completed = i + 1
        if completed % 10 == 0 or completed == len(samples):
            elapsed = time.time() - start_wall
            done_this_run = completed - start_idx
            if done_this_run > 0:
                per_sample_s = elapsed / done_this_run
                remaining_samples = len(samples) - completed
                eta_s = per_sample_s * remaining_samples
                eta_min = eta_s / 60
                if eta_min >= 60:
                    eta_str = f"{eta_min / 60:.1f}h"
                else:
                    eta_str = f"{eta_min:.1f}m"
            else:
                eta_str = "?"

            hg_scores_so_far = [r["final_score"] for r in halluguard_results]
            rate = sum(1 for s in hg_scores_so_far if s > HALLUCINATION_THRESHOLD) / len(hg_scores_so_far) * 100
            print(f"Sample {completed}/{len(samples)} | Hallucination rate so far: {rate:.1f}% | ETA: {eta_str}")

        # --- Checkpoint every CHECKPOINT_INTERVAL samples ---
        if completed % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(completed, baseline_results, halluguard_results)
            print(f"   💾 Checkpoint saved at sample {completed}")

    # --- Compute & print metrics ---
    metrics = compute_metrics(samples, baseline_results, halluguard_results)
    print_metrics(metrics)

    # --- Best improvement cases ---
    best_cases = find_best_improvements(samples, baseline_results, halluguard_results, top_n=3)
    print_best_cases(best_cases)

    # --- Assemble full results payload ---
    wall_time_s = time.time() - start_wall
    results_payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(samples),
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
            "wall_time_seconds": round(wall_time_s, 2),
        },
        "metrics": metrics,
        "best_improvements": best_cases,
        "per_sample": [
            {
                "index": i,
                "query": samples[i]["query"],
                "label": samples[i]["label"],
                "baseline_score": baseline_results[i]["hallucination_score"],
                "halluguard_score": halluguard_results[i]["final_score"],
                "mitigation_applied": halluguard_results[i]["mitigation_applied"],
                "baseline_latency_ms": baseline_results[i]["latency_ms"],
                "halluguard_latency_ms": halluguard_results[i]["latency_ms"],
            }
            for i in range(len(samples))
        ],
    }

    # --- Save 500-sample results JSON ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE_500, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"💾 Results saved to: {RESULTS_FILE_500}")

    # --- Also save to the standard results file for backward compat ---
    output_path = os.path.join(RESULTS_DIR, "quick_demo_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=False, default=str)

    # --- Clean up checkpoint on successful completion ---
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🧹 Checkpoint file removed (run complete)")

    print(f"⏱️  Total wall time: {wall_time_s:.1f}s\n")


if __name__ == "__main__":
    main()

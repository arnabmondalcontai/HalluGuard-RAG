#!/usr/bin/env python3
"""HalluGuard Latency Benchmarking Report.

Measures actual latency improvement after the parallel detector fix.
Loads 10 HaluEval samples, initialises all models ONCE, runs a warmup,
then times each sample to produce cold-start, warm mean/p95, per-module
breakdown, and before/after comparison metrics.

Usage:
    cd halluguard/
    python experiments/latency_report.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SAMPLES = 10
BEFORE_PARALLEL_MS = 20_108  # baseline before parallel fix
GROQ_SLEEP_SECONDS = 1       # rate-limit pause between Groq API calls
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "experiments", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "latency_report.json")


# ===================================================================
# 1. Data Loading
# ===================================================================

def load_samples(n: int = NUM_SAMPLES) -> List[Dict[str, Any]]:
    """Load *n* samples from HaluEval QA split."""
    from datasets import load_dataset

    print(f"\n📥 Loading first {n} HaluEval qa_samples …")
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    samples: List[Dict[str, Any]] = []
    for item in ds:
        knowledge = item.get("knowledge", "")
        question = item.get("question", "")
        answer = item.get("answer", "")

        samples.append({
            "query": question,
            "context": [knowledge] if knowledge else [],
            "response": answer,
        })
        if len(samples) >= n:
            break

    print(f"   ✅ Loaded {len(samples)} samples\n")
    return samples


# ===================================================================
# 2. Mock Retriever (uses pre-loaded context)
# ===================================================================

class MockRetriever:
    """Returns pre-defined context chunks for the current sample."""

    def __init__(self, context_chunks: List[str]):
        self.context_chunks = context_chunks

    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        from src.rag_pipeline.retriever import Document
        return [
            Document(page_content=chunk, metadata={"source": "mock"})
            for chunk in self.context_chunks
        ]


# ===================================================================
# 3. Model Initialisation (cold start measurement)
# ===================================================================

def init_all_components():
    """Initialise all models ONCE. Returns components + cold-start time."""
    from src.detection.nli_detector import NLIDetector
    from src.detection.embedding_detector import EmbeddingDetector
    from src.detection.llm_judge import LLMJudge
    from src.detection.ensemble import EnsembleDetector
    from src.rag_pipeline.generator import LLMGenerator
    from src.rag_pipeline.pipeline import RAGPipeline
    from src.mitigation.router import ConfidenceAdaptiveRouter

    print("🔧 Loading all models (measuring cold start) …")
    cold_start = time.perf_counter()

    nli_detector = NLIDetector()
    emb_detector = EmbeddingDetector()
    llm_judge = LLMJudge()
    ensemble_detector = EnsembleDetector(
        nli_detector=nli_detector,
        embedding_detector=emb_detector,
        llm_judge=llm_judge,
    )
    generator = LLMGenerator()
    router = ConfidenceAdaptiveRouter(
        generator=generator,
        detector=ensemble_detector,
    )
    pipeline = RAGPipeline(
        generator=generator,
        detector=ensemble_detector,
        router=router,
    )

    cold_start_ms = (time.perf_counter() - cold_start) * 1000
    print(f"   ✅ Cold start: {cold_start_ms:.0f} ms\n")

    return {
        "nli_detector": nli_detector,
        "emb_detector": emb_detector,
        "llm_judge": llm_judge,
        "ensemble_detector": ensemble_detector,
        "generator": generator,
        "router": router,
        "pipeline": pipeline,
        "cold_start_ms": cold_start_ms,
    }


# ===================================================================
# 4. Warmup
# ===================================================================

def warmup(components: dict):
    """Run a warmup call to ensure JIT / first-call overhead is excluded."""
    print("⏳ Running warmup call …")
    ensemble = components["ensemble_detector"]
    _ = ensemble.score(
        query="warmup question",
        response="warmup response",
        context_chunks=["warmup context"],
    )
    time.sleep(GROQ_SLEEP_SECONDS)
    print("   ✅ Warmup done\n")


# ===================================================================
# 5. Timed Benchmark
# ===================================================================

def benchmark_sample(
    sample: Dict[str, Any],
    components: dict,
) -> Dict[str, Any]:
    """Time a single sample through the full HalluGuard pipeline.

    Returns per-module timing breakdown and total latency.
    """
    query = sample["query"]
    context_chunks = sample["context"]
    response = sample["response"]
    ensemble = components["ensemble_detector"]
    generator = components["generator"]
    router = components["router"]

    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    # --- Retriever (mock — negligible, but measured for completeness) ---
    t0 = time.perf_counter()
    retriever = MockRetriever(context_chunks)
    _ = retriever.retrieve(query)
    timings["retriever_ms"] = (time.perf_counter() - t0) * 1000

    # --- Detector (ensemble — main cost) ---
    t0 = time.perf_counter()
    detection_result = ensemble.score(
        query=query,
        response=response,
        context_chunks=context_chunks,
    )
    time.sleep(GROQ_SLEEP_SECONDS)  # respect Groq rate limit
    timings["detector_ms"] = (time.perf_counter() - t0) * 1000

    hallucination_score = detection_result.get("final_score", 0.0)

    # --- Router (only if hallucination detected) ---
    t0 = time.perf_counter()
    mitigation_applied = "none"
    if hallucination_score > 0.5:
        try:
            router.retriever = retriever
            router_result = router.route(
                query=query,
                response=response,
                context_chunks=context_chunks,
                hallucination_score=hallucination_score,
                detection_result=detection_result,
            )
            time.sleep(GROQ_SLEEP_SECONDS)
            mitigation_applied = router_result.get("strategy_used", "none")
        except Exception as exc:
            logger.warning("Router failed: %s", exc)
    timings["router_ms"] = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    return {
        "total_ms": round(total_ms, 2),
        "retriever_ms": round(timings["retriever_ms"], 2),
        "detector_ms": round(timings["detector_ms"], 2),
        "router_ms": round(timings["router_ms"], 2),
        "hallucination_score": round(hallucination_score, 4),
        "mitigation_applied": mitigation_applied,
    }


# ===================================================================
# 6. Report Generation
# ===================================================================

def generate_report(
    cold_start_ms: float,
    sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate metrics and build the report."""
    latencies = [r["total_ms"] for r in sample_results]
    retriever_times = [r["retriever_ms"] for r in sample_results]
    detector_times = [r["detector_ms"] for r in sample_results]
    router_times = [r["router_ms"] for r in sample_results]

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    after_parallel_ms = mean_latency
    improvement_pct = (
        (BEFORE_PARALLEL_MS - after_parallel_ms) / BEFORE_PARALLEL_MS * 100
    )

    report = {
        "cold_start_ms": round(cold_start_ms, 2),
        "num_samples": len(sample_results),
        "warm_inference_mean_ms": round(mean_latency, 2),
        "warm_inference_p95_ms": round(p95_latency, 2),
        "per_module_breakdown": {
            "retriever_ms_mean": round(float(np.mean(retriever_times)), 2),
            "retriever_ms_p95": round(float(np.percentile(retriever_times, 95)), 2),
            "detector_ms_mean": round(float(np.mean(detector_times)), 2),
            "detector_ms_p95": round(float(np.percentile(detector_times, 95)), 2),
            "router_ms_mean": round(float(np.mean(router_times)), 2),
            "router_ms_p95": round(float(np.percentile(router_times, 95)), 2),
        },
        "comparison": {
            "before_parallel_fix_ms": BEFORE_PARALLEL_MS,
            "after_parallel_fix_ms": round(after_parallel_ms, 2),
            "improvement_pct": round(improvement_pct, 2),
        },
        "per_sample": sample_results,
    }
    return report


def print_report(report: Dict[str, Any]):
    """Print a clean table of all metrics."""
    sep = "+" + "-" * 42 + "+" + "-" * 18 + "+"
    print("\n" + "=" * 62)
    print("  📊 HalluGuard Latency Benchmarking Report")
    print("=" * 62)

    print(sep)
    print(f"| {'Metric':<40} | {'Value':>16} |")
    print(sep)
    print(f"| {'Cold start (model loading)':<40} | {report['cold_start_ms']:>12.0f} ms |")
    print(f"| {'Warm inference — mean':<40} | {report['warm_inference_mean_ms']:>12.2f} ms |")
    print(f"| {'Warm inference — p95':<40} | {report['warm_inference_p95_ms']:>12.2f} ms |")
    print(sep)

    bd = report["per_module_breakdown"]
    print(f"| {'Retriever mean':<40} | {bd['retriever_ms_mean']:>12.2f} ms |")
    print(f"| {'Retriever p95':<40} | {bd['retriever_ms_p95']:>12.2f} ms |")
    print(f"| {'Detector mean':<40} | {bd['detector_ms_mean']:>12.2f} ms |")
    print(f"| {'Detector p95':<40} | {bd['detector_ms_p95']:>12.2f} ms |")
    print(f"| {'Router mean':<40} | {bd['router_ms_mean']:>12.2f} ms |")
    print(f"| {'Router p95':<40} | {bd['router_ms_p95']:>12.2f} ms |")
    print(sep)

    comp = report["comparison"]
    print(f"| {'Before parallel fix':<40} | {comp['before_parallel_fix_ms']:>12,} ms |")
    print(f"| {'After parallel fix':<40} | {comp['after_parallel_fix_ms']:>12.2f} ms |")
    print(f"| {'Improvement':<40} | {comp['improvement_pct']:>11.2f}  % |")
    print(sep)

    print(f"\n  Per-sample latencies (ms):")
    for i, s in enumerate(report["per_sample"]):
        print(
            f"    Sample {i+1:>2}: {s['total_ms']:>10.2f} ms  "
            f"(retriever={s['retriever_ms']:.1f}, "
            f"detector={s['detector_ms']:.1f}, "
            f"router={s['router_ms']:.1f})  "
            f"score={s['hallucination_score']:.3f}  "
            f"mitigation={s['mitigation_applied']}"
        )
    print()


# ===================================================================
# 7. Main
# ===================================================================

def main():
    # Load samples
    samples = load_samples(NUM_SAMPLES)

    # Init all components (measures cold start)
    components = init_all_components()
    cold_start_ms = components["cold_start_ms"]

    # Warmup (not timed)
    warmup(components)

    # Benchmark each sample
    print(f"🏃 Benchmarking {NUM_SAMPLES} samples …\n")
    sample_results = []
    for i, sample in enumerate(samples):
        result = benchmark_sample(sample, components)
        sample_results.append(result)
        print(
            f"  Sample {i+1:>2}/{NUM_SAMPLES}: "
            f"{result['total_ms']:>10.2f} ms  "
            f"(score={result['hallucination_score']:.3f})"
        )

    # Generate report
    report = generate_report(cold_start_ms, sample_results)

    # Save to JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved to {RESULTS_FILE}")

    # Print clean table
    print_report(report)


if __name__ == "__main__":
    main()

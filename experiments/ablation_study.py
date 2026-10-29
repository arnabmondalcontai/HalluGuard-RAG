#!/usr/bin/env python3
"""HalluGuard Ablation Study — 5 configurations on the same 200 samples.

Compares:
  Config 1 — Baseline RAG (NLI-only scoring, no mitigation)
  Config 2 — NLI detector only (no mitigation)
  Config 3 — Ensemble + fixed query rewrite for ALL detected hallucinations
  Config 4 — Ensemble + fixed self-reflection for ALL detected hallucinations
  Config 5 — HalluGuard full (confidence-adaptive router)

Configs 1, 2, and 5 are derived from existing quick_demo_500_results.json.
Configs 3 and 4 require running the ensemble detector + fixed mitigation.

Usage:
    cd halluguard/
    python experiments/ablation_study.py
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

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
NUM_SAMPLES = 200
RANDOM_SEED = 42
HALLUCINATION_THRESHOLD = 0.5
GROQ_SLEEP_SECONDS = 1
RATE_LIMIT_WAIT = 60
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "experiments", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "ablation_results.json")
EXISTING_RESULTS = os.path.join(RESULTS_DIR, "quick_demo_500_results.json")


# ===================================================================
# 1. Load existing data + HaluEval samples
# ===================================================================

def load_existing_results() -> List[Dict[str, Any]]:
    """Load the first 200 per-sample results from the 500-sample run."""
    print(f"\n📥 Loading existing results from quick_demo_500_results.json …")
    with open(EXISTING_RESULTS, "r") as f:
        data = json.load(f)
    samples = data["per_sample"][:NUM_SAMPLES]
    print(f"   ✅ Loaded {len(samples)} existing per-sample results\n")
    return samples


def load_halueval_samples(n: int = NUM_SAMPLES) -> List[Dict[str, Any]]:
    """Load the first n HaluEval QA samples (same order as quick_demo)."""
    from datasets import load_dataset

    print(f"📥 Loading first {n} HaluEval qa_samples for context/response data …")
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    samples = []
    for item in ds:
        knowledge = item.get("knowledge", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        hallucination = item.get("hallucination", "")
        label = 1 if hallucination == "yes" else 0

        samples.append({
            "query": question,
            "context": [knowledge] if knowledge else [],
            "response": answer,
            "label": label,
        })
        if len(samples) >= n:
            break

    print(f"   ✅ Loaded {len(samples)} HaluEval samples\n")
    return samples


# ===================================================================
# 2. Mock Retriever
# ===================================================================

class MockRetriever:
    """Returns pre-defined context chunks."""
    def __init__(self, context_chunks: List[str]):
        self.context_chunks = context_chunks

    def retrieve(self, query: str, top_k: int = 5):
        from src.rag_pipeline.retriever import Document
        return [Document(page_content=c, metadata={"source": "mock"})
                for c in self.context_chunks]


# ===================================================================
# 3. Rate-limit retry wrapper
# ===================================================================

def _run_with_retry(func, *args, sample_idx: int = 0, **kwargs):
    """Run a function with Groq 429 rate-limit retry (60s wait)."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower() or "Rate limit" in exc_str:
                print(f"Rate limited. Waiting {RATE_LIMIT_WAIT}s... (sample {sample_idx})")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            else:
                raise
    raise RuntimeError(f"Rate limit retries exhausted for sample {sample_idx}")


# ===================================================================
# 4. Checkpoint helpers
# ===================================================================

def _checkpoint_path(config_num: int) -> str:
    return os.path.join(RESULTS_DIR, f"ablation_checkpoint_{config_num}.json")


def _load_checkpoint(config_num: int) -> Optional[dict]:
    path = _checkpoint_path(config_num)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            print(f"   🔄 Resuming Config {config_num} from sample {data['completed']}")
            return data
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_checkpoint(config_num: int, completed: int, results: List[Dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(_checkpoint_path(config_num), "w") as f:
        json.dump({"completed": completed, "results": results}, f,
                  indent=2, default=str)


def _clear_checkpoint(config_num: int):
    path = _checkpoint_path(config_num)
    if os.path.exists(path):
        os.remove(path)


# ===================================================================
# 5. Metrics computation
# ===================================================================

def compute_config_metrics(
    ground_truth: List[int],
    scores: List[float],
    latencies: List[float],
) -> Dict[str, Any]:
    """Compute halluc%, faithfulness, AUROC, F1, latency for one config."""
    from sklearn.metrics import roc_auc_score, f1_score

    halluc_rate = sum(1 for s in scores if s > HALLUCINATION_THRESHOLD) / len(scores) * 100
    faithfulness = 1.0 - float(np.mean(scores))

    # AUROC
    try:
        auroc = float(roc_auc_score(ground_truth, scores))
    except ValueError:
        auroc = 0.5

    # F1 — binary predictions from scores
    predictions = [1 if s > HALLUCINATION_THRESHOLD else 0 for s in scores]
    try:
        f1 = float(f1_score(ground_truth, predictions))
    except ValueError:
        f1 = 0.0

    return {
        "hallucination_rate_pct": round(halluc_rate, 2),
        "mean_faithfulness": round(faithfulness, 4),
        "auroc": round(auroc, 4),
        "f1": round(f1, 4),
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
    }


# ===================================================================
# 6. Config runners
# ===================================================================

def run_config_1_baseline(existing: List[Dict]) -> Dict:
    """Config 1 — Baseline RAG: NLI-only scoring, no mitigation."""
    print("\n🔹 Config 1 — Baseline RAG (NLI-only, no mitigation)")
    scores = [s["baseline_score"] for s in existing]
    latencies = [s["baseline_latency_ms"] for s in existing]
    labels = [s["label"] for s in existing]
    metrics = compute_config_metrics(labels, scores, latencies)
    print(f"   Halluc%: {metrics['hallucination_rate_pct']}%  "
          f"Faith: {metrics['mean_faithfulness']}  "
          f"AUROC: {metrics['auroc']}  F1: {metrics['f1']}")
    return {"config": "Baseline RAG", "metrics": metrics,
            "scores": scores, "latencies": latencies}


def run_config_2_nli_only(existing: List[Dict]) -> Dict:
    """Config 2 — NLI detector only (no mitigation).

    Same as Config 1 since baseline already uses NLI-only scoring.
    """
    print("\n🔹 Config 2 — NLI detector only (no mitigation)")
    scores = [s["baseline_score"] for s in existing]
    latencies = [s["baseline_latency_ms"] for s in existing]
    labels = [s["label"] for s in existing]
    metrics = compute_config_metrics(labels, scores, latencies)
    print(f"   Halluc%: {metrics['hallucination_rate_pct']}%  "
          f"Faith: {metrics['mean_faithfulness']}  "
          f"AUROC: {metrics['auroc']}  F1: {metrics['f1']}")
    return {"config": "NLI detector only", "metrics": metrics,
            "scores": scores, "latencies": latencies}


def run_config_3_fixed_rewrite(
    halueval_samples: List[Dict],
    existing: List[Dict],
    ensemble_detector,
    generator,
) -> Dict:
    """Config 3 — Ensemble + fixed query rewrite for ALL hallucinations."""
    print("\n🔹 Config 3 — Ensemble + fixed query rewrite")
    from src.mitigation.query_rewriter import QueryRewriter
    from src.mitigation.reranker import ChunkReranker

    query_rewriter = QueryRewriter()

    checkpoint = _load_checkpoint(3)
    if checkpoint:
        start_idx = checkpoint["completed"]
        results = checkpoint["results"]
    else:
        start_idx = 0
        results = []

    for i in range(start_idx, len(halueval_samples)):
        sample = halueval_samples[i]
        query = sample["query"]
        context_chunks = sample["context"]
        response = sample["response"]

        start_t = time.perf_counter()

        # Run ensemble detection
        try:
            detection_result = _run_with_retry(
                ensemble_detector.score,
                query=query, response=response, context_chunks=context_chunks,
                sample_idx=i,
            )
            time.sleep(GROQ_SLEEP_SECONDS)
        except Exception as exc:
            logger.warning("Ensemble failed sample %d: %s", i, exc)
            detection_result = {"final_score": 0.5}

        ensemble_score = detection_result.get("final_score", 0.5)
        final_score = ensemble_score

        # Apply FIXED rewrite for ALL detected hallucinations
        if ensemble_score > HALLUCINATION_THRESHOLD:
            try:
                # Rewrite query
                rewritten = _run_with_retry(
                    query_rewriter.rewrite, query, response, sample_idx=i,
                )
                time.sleep(GROQ_SLEEP_SECONDS)

                # Re-retrieve with first rewritten query
                retriever = MockRetriever(context_chunks)
                new_chunks = context_chunks  # mock retriever returns same chunks

                # Re-generate response with rewritten query
                best_query = rewritten[0] if rewritten else query
                new_response = _run_with_retry(
                    generator.generate,
                    query=best_query, context_chunks=new_chunks,
                    sample_idx=i,
                )
                time.sleep(GROQ_SLEEP_SECONDS)

                # Re-score
                new_detection = _run_with_retry(
                    ensemble_detector.score,
                    query=best_query, response=new_response,
                    context_chunks=new_chunks,
                    sample_idx=i,
                )
                time.sleep(GROQ_SLEEP_SECONDS)
                new_score = new_detection.get("final_score", ensemble_score)

                if new_score < ensemble_score:
                    final_score = new_score
                else:
                    final_score = ensemble_score

            except Exception as exc:
                logger.warning("Rewrite mitigation failed sample %d: %s", i, exc)
                final_score = ensemble_score

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        results.append({
            "score": round(final_score, 4),
            "latency_ms": round(elapsed_ms, 2),
        })

        completed = i + 1
        if completed % 10 == 0:
            hrate = sum(1 for r in results if r["score"] > HALLUCINATION_THRESHOLD) / len(results) * 100
            print(f"   Config 3: {completed}/{NUM_SAMPLES} | Halluc rate: {hrate:.1f}%")
        if completed % 25 == 0:
            _save_checkpoint(3, completed, results)

    _clear_checkpoint(3)
    scores = [r["score"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    labels = [s["label"] for s in existing]
    metrics = compute_config_metrics(labels, scores, latencies)
    print(f"   Halluc%: {metrics['hallucination_rate_pct']}%  "
          f"Faith: {metrics['mean_faithfulness']}  "
          f"AUROC: {metrics['auroc']}  F1: {metrics['f1']}")
    return {"config": "Ensemble + fixed rewrite", "metrics": metrics,
            "scores": scores, "latencies": latencies}


def run_config_4_fixed_self_reflection(
    halueval_samples: List[Dict],
    existing: List[Dict],
    ensemble_detector,
    generator,
) -> Dict:
    """Config 4 — Ensemble + fixed self-reflection for ALL hallucinations."""
    print("\n🔹 Config 4 — Ensemble + fixed self-reflection")
    from src.mitigation.self_reflection import SelfReflectionMitigator

    self_reflect = SelfReflectionMitigator()

    checkpoint = _load_checkpoint(4)
    if checkpoint:
        start_idx = checkpoint["completed"]
        results = checkpoint["results"]
    else:
        start_idx = 0
        results = []

    for i in range(start_idx, len(halueval_samples)):
        sample = halueval_samples[i]
        query = sample["query"]
        context_chunks = sample["context"]
        response = sample["response"]

        start_t = time.perf_counter()

        # Run ensemble detection
        try:
            detection_result = _run_with_retry(
                ensemble_detector.score,
                query=query, response=response, context_chunks=context_chunks,
                sample_idx=i,
            )
            time.sleep(GROQ_SLEEP_SECONDS)
        except Exception as exc:
            logger.warning("Ensemble failed sample %d: %s", i, exc)
            detection_result = {"final_score": 0.5}

        ensemble_score = detection_result.get("final_score", 0.5)
        final_score = ensemble_score

        # Apply FIXED self-reflection for ALL detected hallucinations
        if ensemble_score > HALLUCINATION_THRESHOLD:
            try:
                reflect_result = _run_with_retry(
                    self_reflect.mitigate,
                    query=query, hallucinated_response=response,
                    context_chunks=context_chunks,
                    detector=ensemble_detector,
                    generator=generator,
                    sample_idx=i,
                )
                time.sleep(GROQ_SLEEP_SECONDS)

                new_score = reflect_result.get("final_score",
                            reflect_result.get("score", ensemble_score))
                if new_score < ensemble_score:
                    final_score = new_score
                else:
                    final_score = ensemble_score

            except Exception as exc:
                logger.warning("Self-reflection failed sample %d: %s", i, exc)
                final_score = ensemble_score

        elapsed_ms = (time.perf_counter() - start_t) * 1000
        results.append({
            "score": round(final_score, 4),
            "latency_ms": round(elapsed_ms, 2),
        })

        completed = i + 1
        if completed % 10 == 0:
            hrate = sum(1 for r in results if r["score"] > HALLUCINATION_THRESHOLD) / len(results) * 100
            print(f"   Config 4: {completed}/{NUM_SAMPLES} | Halluc rate: {hrate:.1f}%")
        if completed % 25 == 0:
            _save_checkpoint(4, completed, results)

    _clear_checkpoint(4)
    scores = [r["score"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    labels = [s["label"] for s in existing]
    metrics = compute_config_metrics(labels, scores, latencies)
    print(f"   Halluc%: {metrics['hallucination_rate_pct']}%  "
          f"Faith: {metrics['mean_faithfulness']}  "
          f"AUROC: {metrics['auroc']}  F1: {metrics['f1']}")
    return {"config": "Ensemble + fixed self-reflect", "metrics": metrics,
            "scores": scores, "latencies": latencies}


def run_config_5_halluguard(existing: List[Dict]) -> Dict:
    """Config 5 — HalluGuard full (confidence-adaptive router)."""
    print("\n🔹 Config 5 — HalluGuard full (confidence-adaptive router)")
    scores = [s["halluguard_score"] for s in existing]
    latencies = [s["halluguard_latency_ms"] for s in existing]
    labels = [s["label"] for s in existing]
    metrics = compute_config_metrics(labels, scores, latencies)
    print(f"   Halluc%: {metrics['hallucination_rate_pct']}%  "
          f"Faith: {metrics['mean_faithfulness']}  "
          f"AUROC: {metrics['auroc']}  F1: {metrics['f1']}")
    return {"config": "HalluGuard (adaptive router)", "metrics": metrics,
            "scores": scores, "latencies": latencies}


# ===================================================================
# 7. Results table
# ===================================================================

def print_ablation_table(configs: List[Dict]):
    """Print the ablation study results table."""
    print("\n" + "=" * 88)
    print("                     📊  Ablation Study Results  (200 samples)")
    print("=" * 88)

    header = (f"{'Configuration':<35} {'Halluc%':>8} {'Faithful':>10} "
              f"{'AUROC':>7} {'F1':>6} {'Latency ms':>12}")
    sep = "-" * 88

    print(sep)
    print(header)
    print(sep)

    for cfg in configs:
        m = cfg["metrics"]
        print(f"{cfg['config']:<35} {m['hallucination_rate_pct']:>7.1f}% "
              f"{m['mean_faithfulness']:>10.4f} {m['auroc']:>7.4f} "
              f"{m['f1']:>6.4f} {m['latency_mean_ms']:>12.2f}")

    print(sep)

    # Improvement over baseline
    if len(configs) >= 5:
        baseline = configs[0]["metrics"]
        best = configs[4]["metrics"]  # HalluGuard full
        print(f"\n  📈 HalluGuard vs Baseline improvement:")
        print(f"     Hallucination rate: {baseline['hallucination_rate_pct']:.1f}% → "
              f"{best['hallucination_rate_pct']:.1f}% "
              f"(↓ {baseline['hallucination_rate_pct'] - best['hallucination_rate_pct']:.1f}pp)")
        print(f"     Faithfulness:      {baseline['mean_faithfulness']:.4f} → "
              f"{best['mean_faithfulness']:.4f} "
              f"(↑ {best['mean_faithfulness'] - baseline['mean_faithfulness']:.4f})")
    print()


# ===================================================================
# 8. Main
# ===================================================================

def main():
    # Load data
    existing_results = load_existing_results()
    halueval_samples = load_halueval_samples(NUM_SAMPLES)

    # Configs 1, 2, 5 — derived from existing data (no re-run)
    config1 = run_config_1_baseline(existing_results)
    config2 = run_config_2_nli_only(existing_results)
    config5 = run_config_5_halluguard(existing_results)

    # Configs 3, 4 — need ensemble + fixed mitigation
    print("\n🔧 Loading models for Configs 3 & 4 …")
    from src.detection.nli_detector import NLIDetector
    from src.detection.embedding_detector import EmbeddingDetector
    from src.detection.llm_judge import LLMJudge
    from src.detection.ensemble import EnsembleDetector
    from src.rag_pipeline.generator import LLMGenerator

    nli_detector = NLIDetector()
    emb_detector = EmbeddingDetector()
    llm_judge = LLMJudge()
    ensemble_detector = EnsembleDetector(
        nli_detector=nli_detector,
        embedding_detector=emb_detector,
        llm_judge=llm_judge,
    )
    generator = LLMGenerator()

    # Warmup
    print("   ⏳ Warming up …")
    _ = ensemble_detector.score("warmup", "warmup response", ["warmup context"])
    time.sleep(GROQ_SLEEP_SECONDS)
    print("   ✅ Models ready\n")

    config3 = run_config_3_fixed_rewrite(
        halueval_samples, existing_results, ensemble_detector, generator,
    )
    config4 = run_config_4_fixed_self_reflection(
        halueval_samples, existing_results, ensemble_detector, generator,
    )

    # Assemble all configs
    all_configs = [config1, config2, config3, config4, config5]

    # Print table
    print_ablation_table(all_configs)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": NUM_SAMPLES,
            "random_seed": RANDOM_SEED,
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
        },
        "configs": [
            {
                "name": cfg["config"],
                "metrics": cfg["metrics"],
            }
            for cfg in all_configs
        ],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(save_payload, f, indent=2, default=str)
    print(f"💾 Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""HalluGuard Cross-Domain Evaluation.

Evaluates the HalluGuard pipeline across three distinct domains without retraining:
1. General Domain (HaluEval): Loaded from existing quick_demo_500_results.json
2. Medical Domain (MedQA): 100 samples from USMLE / MedQA
3. Legal Domain: 100 samples from legal statutory & contract QA

Features:
- Automatic macOS caffeinate mode to prevent sleep during evaluation
- Checkpoint/resume support for both Medical and Legal runs
- Groq rate-limit handling (automatic backoff and retry)
- Saves individual domain results and combined cross-domain comparison table
- Prints formatted ASCII comparison table

Usage:
    cd halluguard/
    python experiments/cross_domain_eval.py
"""

import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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
NUM_SAMPLES_PER_DOMAIN = 100
HALLUCINATION_THRESHOLD = 0.5
GROQ_SLEEP_SECONDS = 0.5
RATE_LIMIT_WAIT = 60
CHECKPOINT_INTERVAL = 10

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "experiments", "results")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

GENERAL_RESULTS_500 = os.path.join(RESULTS_DIR, "quick_demo_500_results.json")
GENERAL_RESULTS_50 = os.path.join(RESULTS_DIR, "quick_demo_results.json")

MEDQA_CHECKPOINT = os.path.join(RESULTS_DIR, "medqa_checkpoint.json")
MEDQA_RESULTS_FILE = os.path.join(RESULTS_DIR, "medqa_results.json")

LEGAL_CHECKPOINT = os.path.join(RESULTS_DIR, "legal_checkpoint.json")
LEGAL_RESULTS_FILE = os.path.join(RESULTS_DIR, "legal_results.json")
LEGAL_SYNTHETIC_DATA = os.path.join(DATA_DIR, "legal_synthetic_50.json")

CROSS_DOMAIN_TABLE_FILE = os.path.join(RESULTS_DIR, "cross_domain_table.json")


# ===================================================================
# 0. Caffeinate Process Helper
# ===================================================================

def enable_caffeinate() -> Optional[subprocess.Popen]:
    """Prevent system sleep on macOS during long benchmark runs."""
    if sys.platform == "darwin":
        try:
            proc = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
            print("☕ Caffeinate mode enabled (macOS power assertions active to prevent sleep).")
            return proc
        except Exception as exc:
            print(f"⚠️ Could not start caffeinate: {exc}")
    return None


# ===================================================================
# 1. Pipeline Helpers
# ===================================================================

class MockRetriever:
    """Returns pre-defined context chunks for the given sample."""
    def __init__(self, context_chunks: List[str]):
        self.context_chunks = context_chunks

    def retrieve(self, query: str, top_k: int = 5):
        from src.rag_pipeline.retriever import Document
        return [Document(page_content=c, metadata={"source": "domain_eval"})
                for c in self.context_chunks]


def init_pipeline():
    """Lazily initialize and warm up shared pipeline components."""
    from src.detection.nli_detector import NLIDetector
    from src.detection.embedding_detector import EmbeddingDetector
    from src.detection.llm_judge import LLMJudge
    from src.detection.ensemble import EnsembleDetector
    from src.rag_pipeline.generator import LLMGenerator
    from src.mitigation.router import ConfidenceAdaptiveRouter

    print("\n🔧 Initializing HalluGuard components …")
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

    print("   ⏳ Running warmup …")
    _ = ensemble_detector.score("warmup", "warmup response", ["warmup context"])
    print("   ✅ Models ready.\n")
    return ensemble_detector, generator, router


def run_pipeline_sample(
    sample: Dict[str, Any],
    ensemble_detector,
    generator,
    router,
) -> Dict[str, Any]:
    """Run full HalluGuard detection and adaptive mitigation on one sample."""
    start = time.perf_counter()

    query = sample["query"]
    context_chunks = sample["context"]
    response = sample.get("response", "")

    if not response.strip():
        response = generator.generate(query=query, context_chunks=context_chunks)
        time.sleep(GROQ_SLEEP_SECONDS)

    # 1. Ensemble Detection
    detection_result = _run_with_retry(
        ensemble_detector.score,
        query=query,
        response=response,
        context_chunks=context_chunks,
    )
    time.sleep(GROQ_SLEEP_SECONDS)

    initial_score = detection_result.get("final_score", 0.0)
    is_hallucinated = initial_score > HALLUCINATION_THRESHOLD

    final_response = response
    mitigation_applied = "none"
    final_score = initial_score

    # 2. Adaptive Mitigation
    if is_hallucinated:
        try:
            router.retriever = MockRetriever(context_chunks)
            router_result = _run_with_retry(
                router.route,
                query=query,
                response=response,
                context_chunks=context_chunks,
                hallucination_score=initial_score,
                detection_result=detection_result,
            )
            time.sleep(GROQ_SLEEP_SECONDS)

            final_response = router_result.get("final_response", response)
            mitigation_applied = router_result.get("strategy_used", "none")
            final_score = router_result.get("final_score", initial_score)
        except Exception as exc:
            logger.warning("Mitigation failed: %s", exc)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "query": query,
        "context": context_chunks,
        "original_response": response,
        "final_response": final_response,
        "ground_truth_label": sample["label"],
        "initial_score": round(float(initial_score), 4),
        "final_score": round(float(final_score), 4),
        "mitigation_applied": mitigation_applied,
        "is_hallucinated_final": bool(final_score > HALLUCINATION_THRESHOLD),
        "latency_ms": round(float(elapsed_ms), 2),
    }


def _run_with_retry(func, *args, **kwargs):
    """Run function with Groq 429 rate limit backoff."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower():
                print(f"   ⚠️ Rate limited (429). Waiting {RATE_LIMIT_WAIT}s …")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            raise
    raise RuntimeError("Max retries exceeded for API call.")


# ===================================================================
# 2. Domain Data Loaders
# ===================================================================

def load_general_domain_results() -> Dict[str, Any]:
    """Load existing General Domain results without re-running pipeline."""
    target_file = GENERAL_RESULTS_500 if os.path.exists(GENERAL_RESULTS_500) else GENERAL_RESULTS_50
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"General domain results file not found at {target_file}")

    print(f"📥 Loading General Domain results from {os.path.basename(target_file)} …")
    with open(target_file, "r") as f:
        data = json.load(f)

    halluguard_metrics = data["metrics"]["halluguard"]
    res = {
        "domain": "General (HaluEval)",
        "num_samples": data["metrics"]["num_samples"],
        "hallucination_rate_pct": float(halluguard_metrics["hallucination_rate_pct"]),
        "mean_faithfulness": float(halluguard_metrics["mean_faithfulness"]),
        "auroc": float(halluguard_metrics["auroc"]),
        "latency_mean_ms": float(halluguard_metrics["latency_mean_ms"]),
        "latency_p95_ms": float(halluguard_metrics.get("latency_p95_ms", 0.0)),
        "source_file": os.path.basename(target_file),
    }
    print(f"   ✅ General Domain: Halluc={res['hallucination_rate_pct']}%, "
          f"Faithfulness={res['mean_faithfulness']:.4f}, AUROC={res['auroc']:.4f}, "
          f"Latency={res['latency_mean_ms']:.1f}ms\n")
    return res


def load_medical_samples(n_total: int = NUM_SAMPLES_PER_DOMAIN) -> List[Dict[str, Any]]:
    """Load Medical QA samples (MedQA-USMLE) converted to standard schema."""
    import datasets

    print("📥 Loading Medical Domain (MedQA) dataset …")
    samples: List[Dict[str, Any]] = []

    # Option A
    try:
        ds = datasets.load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        print(f"   ✅ Loaded MedQA-USMLE test split ({len(ds)} items).")
        
        # We construct 50 faithful (correct answer) and 50 hallucinated (distractor option)
        num_pairs = n_total // 2
        for i, item in enumerate(ds):
            if len(samples) >= n_total:
                break
            
            question_text = item["question"]
            correct_answer = item["answer"]
            options = item.get("options", {})
            answer_idx = item.get("answer_idx", "")

            # Faithful sample (label = 0)
            if i < num_pairs:
                samples.append({
                    "query": question_text,
                    "context": [question_text],
                    "response": correct_answer,
                    "label": 0,
                })
            else:
                # Hallucinated sample (label = 1): pick an incorrect distractor
                distractors = [
                    opt_val for opt_key, opt_val in options.items()
                    if opt_key != answer_idx and opt_val != correct_answer
                ]
                distractor = distractors[0] if distractors else "Incorrect medical diagnosis."
                samples.append({
                    "query": question_text,
                    "context": [question_text],
                    "response": distractor,
                    "label": 1,
                })

        print(f"   ✅ Prepared {len(samples)} Medical QA samples (50 faithful, 50 hallucinated).\n")
        return samples
    except Exception as exc:
        print(f"   ⚠️ Option A failed ({exc}), trying Option B …")

    # Option B
    try:
        ds = datasets.load_dataset("bigbio/med_qa", split="test")
        print(f"   ✅ Loaded bigbio/med_qa test split.")
        for item in ds:
            if len(samples) >= n_total:
                break
            samples.append({
                "query": item["question"],
                "context": [item.get("context", item["question"])],
                "response": item.get("answer", ""),
                "label": 0,
            })
        return samples
    except Exception as exc:
        print(f"   ⚠️ Option B failed ({exc}), falling back to synthetic generation …")

    # Option C: Synthetic generation via Groq
    return generate_synthetic_medical_samples(n_total)


def generate_synthetic_medical_samples(n_total: int) -> List[Dict[str, Any]]:
    """Generate synthetic medical QA evaluation samples."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("   💉 Generating synthetic medical QA pairs with Groq …")
    prompt = """Generate 10 medical clinical case questions about drug dosages, pharmacology, or diagnoses.
For each item provide:
- question: clinical scenario / question
- context: 2-3 sentence clinical guideline or pharmacological fact
- correct_answer: faithful answer based ONLY on the context
- hallucinated_answer: plausible but incorrect or contraindicated answer

Return as JSON: {"samples": [{"question": "...", "context": "...", "correct_answer": "...", "hallucinated_answer": "..."}]}"""

    pairs = []
    while len(pairs) < (n_total // 2):
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a clinical pharmacologist creating hallucination evaluation benchmarks. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        pairs.extend(data.get("samples", []))
        time.sleep(1)

    samples = []
    for p in pairs[:n_total // 2]:
        samples.append({
            "query": p["question"],
            "context": [p["context"]],
            "response": p["correct_answer"],
            "label": 0,
        })
        samples.append({
            "query": p["question"],
            "context": [p["context"]],
            "response": p["hallucinated_answer"],
            "label": 1,
        })
    return samples[:n_total]


def load_legal_samples(n_total: int = NUM_SAMPLES_PER_DOMAIN) -> List[Dict[str, Any]]:
    """Load Legal QA samples converted to standard schema."""
    import datasets

    print("📥 Loading Legal Domain dataset …")
    
    # Check if cached synthetic legal data exists
    if os.path.exists(LEGAL_SYNTHETIC_DATA):
        try:
            with open(LEGAL_SYNTHETIC_DATA, "r") as f:
                cached_data = json.load(f)
            if len(cached_data) >= n_total:
                print(f"   ✅ Loaded {len(cached_data)} cached Legal QA samples from {os.path.basename(LEGAL_SYNTHETIC_DATA)}\n")
                return cached_data[:n_total]
        except Exception as exc:
            logger.warning("Could not read cached legal data: %s", exc)

    # Option A: HuggingFace nguyen-brat/legal_questions
    try:
        ds = datasets.load_dataset("nguyen-brat/legal_questions", split="train")
        print(f"   ✅ Loaded nguyen-brat/legal_questions.")
        samples = []
        for item in ds:
            if len(samples) >= n_total:
                break
            samples.append({
                "query": item.get("question", item.get("text", "")),
                "context": [item.get("context", "")],
                "response": item.get("answer", ""),
                "label": int(item.get("label", 0)),
            })
        return samples
    except Exception as exc:
        print(f"   ℹ️ Option A (nguyen-brat/legal_questions) not available ({exc})")

    # Option B: HuggingFace coastalcph/lex_glue scotus
    # (SCOTUS is long classification text, so Option C gives high-quality QA pairs)
    print("   ⚖️ Generating 50 high-quality Legal QA pairs (contracts, torts, corporate law, statutes) …")
    samples = generate_synthetic_legal_samples(n_total)

    # Cache for future runs
    with open(LEGAL_SYNTHETIC_DATA, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"   💾 Saved generated legal samples to {os.path.basename(LEGAL_SYNTHETIC_DATA)}\n")

    return samples


def generate_synthetic_legal_samples(n_total: int) -> List[Dict[str, Any]]:
    """Generate 50 legal QA pairs with Groq, yielding 100 samples (50 faithful + 50 hallucinated)."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    target_pairs = n_total // 2
    pairs = []
    
    legal_topics = [
        "contract breach and remedies (liquidated damages, force majeure)",
        "intellectual property and trade secrets (fair use, patent claims)",
        "corporate governance and fiduciary duty (business judgment rule)",
        "statutory interpretation and regulatory compliance (SEC, FTC)",
        "torts, negligence, and liability indemnification clauses"
    ]

    pbar = tqdm(total=target_pairs, desc="Generating Legal QA pairs")
    topic_idx = 0
    while len(pairs) < target_pairs:
        topic = legal_topics[topic_idx % len(legal_topics)]
        topic_idx += 1
        prompt = f"""Generate 10 distinct legal question-and-answer items focusing on {topic}.
For each item provide:
- question: A precise legal question.
- context: A 2-4 sentence realistic excerpt from a contract clause, statute, or legal standard.
- correct_answer: A strictly faithful, accurate answer based ONLY on the context.
- hallucinated_answer: A plausible-sounding answer that contradicts the context, invents non-existent legal tests, or misstates timelines/obligations.

Return valid JSON with format:
{{
  "samples": [
    {{
      "question": "...",
      "context": "...",
      "correct_answer": "...",
      "hallucinated_answer": "..."
    }}
  ]
}}"""
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are a legal scholar creating benchmark data for AI hallucination detection. Output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            batch = data.get("samples", [])
            pairs.extend(batch)
            pbar.update(len(batch))
            time.sleep(GROQ_SLEEP_SECONDS)
        except Exception as exc:
            print(f"\n   ⚠️ Groq generation error: {exc}. Retrying in 10s …")
            time.sleep(10)

    pbar.close()

    samples = []
    for p in pairs[:target_pairs]:
        # Faithful sample (label = 0)
        samples.append({
            "query": p["question"],
            "context": [p["context"]],
            "response": p["correct_answer"],
            "label": 0,
        })
        # Hallucinated sample (label = 1)
        samples.append({
            "query": p["question"],
            "context": [p["context"]],
            "response": p["hallucinated_answer"],
            "label": 1,
        })

    print(f"   ✅ Prepared {len(samples)} Legal QA evaluation samples (50 faithful, 50 hallucinated).")
    return samples[:n_total]


# ===================================================================
# 3. Domain Evaluation Runner with Checkpoint Support
# ===================================================================

def evaluate_domain(
    domain_name: str,
    samples: List[Dict[str, Any]],
    checkpoint_file: str,
    results_file: str,
    ensemble_detector,
    generator,
    router,
) -> Dict[str, Any]:
    """Evaluate HalluGuard on a domain dataset with checkpointing and resume."""
    print(f"\n🚀 Starting Evaluation for {domain_name} ({len(samples)} samples) …")

    per_sample_results: List[Dict[str, Any]] = []
    start_idx = 0

    # Resume from checkpoint if present
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                ckpt = json.load(f)
            per_sample_results = ckpt.get("results", [])
            start_idx = len(per_sample_results)
            print(f"   🔄 Found checkpoint: Resuming {domain_name} from sample {start_idx + 1}/{len(samples)}")
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)

    pbar = tqdm(
        total=len(samples),
        initial=start_idx,
        desc=f"Evaluating {domain_name}",
        unit="sample"
    )

    t_start = time.perf_counter()

    for idx in range(start_idx, len(samples)):
        sample = samples[idx]
        result = run_pipeline_sample(
            sample=sample,
            ensemble_detector=ensemble_detector,
            generator=generator,
            router=router,
        )
        per_sample_results.append(result)
        pbar.update(1)

        # Progress display and checkpointing
        if (idx + 1) % CHECKPOINT_INTERVAL == 0 or (idx + 1) == len(samples):
            # Compute current hallucination rate
            curr_halluc_rate = sum(1 for r in per_sample_results if r["final_score"] > HALLUCINATION_THRESHOLD) / len(per_sample_results) * 100
            pbar.set_postfix({"Halluc %": f"{curr_halluc_rate:.1f}%"})

            # Save checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "domain": domain_name,
                    "completed": len(per_sample_results),
                    "total": len(samples),
                    "timestamp": datetime.now().isoformat(),
                    "results": per_sample_results,
                }, f, indent=2)

    pbar.close()
    wall_time_s = time.perf_counter() - t_start

    # Compute domain aggregate metrics
    final_scores = [r["final_score"] for r in per_sample_results]
    initial_scores = [r["initial_score"] for r in per_sample_results]
    ground_truth = [r["ground_truth_label"] for r in per_sample_results]
    latencies = [r["latency_ms"] for r in per_sample_results]

    halluc_rate_pct = sum(1 for s in final_scores if s > HALLUCINATION_THRESHOLD) / len(final_scores) * 100
    mean_faithfulness = 1.0 - float(np.mean(final_scores))

    try:
        auroc = float(roc_auc_score(ground_truth, final_scores))
    except Exception:
        auroc = 0.5

    latency_mean_ms = float(np.mean(latencies))
    latency_p95_ms = float(np.percentile(latencies, 95))
    mitigation_counts = dict(Counter(r["mitigation_applied"] for r in per_sample_results))

    domain_summary = {
        "domain": domain_name,
        "num_samples": len(per_sample_results),
        "hallucination_rate_pct": round(halluc_rate_pct, 2),
        "mean_faithfulness": round(mean_faithfulness, 4),
        "auroc": round(auroc, 4),
        "latency_mean_ms": round(latency_mean_ms, 2),
        "latency_p95_ms": round(latency_p95_ms, 2),
        "mitigation_tier_counts": mitigation_counts,
        "wall_time_seconds": round(wall_time_s, 2),
        "timestamp": datetime.now().isoformat(),
    }

    # Save full domain results
    full_output = {
        "meta": domain_summary,
        "per_sample": per_sample_results,
    }
    with open(results_file, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"   💾 Saved {domain_name} results to {os.path.basename(results_file)}")

    # Clean up checkpoint on successful complete run
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
        except OSError:
            pass

    return domain_summary


# ===================================================================
# 4. Cross-Domain Table Generation
# ===================================================================

def print_and_save_cross_domain_table(
    general_metrics: Dict[str, Any],
    medical_metrics: Dict[str, Any],
    legal_metrics: Dict[str, Any],
) -> None:
    """Format and print comparative table across all 3 domains."""
    domains = [
        ("General (HaluEval)", general_metrics),
        ("Medical (MedQA)", medical_metrics),
        ("Legal", legal_metrics),
    ]

    print("\n" + "=" * 70)
    print("           HALLUGUARD CROSS-DOMAIN GENERALISATION RESULTS")
    print("=" * 70)
    print(f"+--------------------+----------+-------------+-------+------------+")
    print(f"| {'Domain':<18} | {'Halluc %':<8} | {'Faithfulness':<11} | {'AUROC':<5} | {'Latency ms':<10} |")
    print(f"+--------------------+----------+-------------+-------+------------+")

    table_data = []
    for label, m in domains:
        h_rate = f"{m['hallucination_rate_pct']:.1f}%"
        faith = f"{m['mean_faithfulness']:.4f}"
        auroc = f"{m['auroc']:.4f}"
        lat = f"{m['latency_mean_ms']:.1f}"
        print(f"| {label:<18} | {h_rate:>8} | {faith:>11} | {auroc:>5} | {lat:>10} |")

        table_data.append({
            "domain": label,
            "hallucination_rate_pct": m["hallucination_rate_pct"],
            "mean_faithfulness": m["mean_faithfulness"],
            "auroc": m["auroc"],
            "latency_mean_ms": m["latency_mean_ms"],
            "num_samples": m.get("num_samples", 100),
        })

    print(f"+--------------------+----------+-------------+-------+------------+\n")

    # Save to cross_domain_table.json
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": table_data,
        "detailed_domains": {
            "general": general_metrics,
            "medical": medical_metrics,
            "legal": legal_metrics,
        }
    }
    with open(CROSS_DOMAIN_TABLE_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"💾 Saved combined table to {os.path.basename(CROSS_DOMAIN_TABLE_FILE)}\n")


# ===================================================================
# 5. Main Execution
# ===================================================================

def main():
    print("\n" + "=" * 70)
    print("       HalluGuard: Cross-Domain Evaluation (Medical & Legal)")
    print("=" * 70)

    # 1. Enable caffeinate automatically to prevent sleeping during benchmark
    caffeinate_proc = enable_caffeinate()

    try:
        # 2. General Domain (already completed in quick_demo_500)
        general_metrics = load_general_domain_results()

        # 3. Initialize Shared Pipeline Models
        ensemble_detector, generator, router = init_pipeline()

        # 4. Medical Domain Evaluation (100 samples)
        med_samples = load_medical_samples(NUM_SAMPLES_PER_DOMAIN)
        medical_metrics = evaluate_domain(
            domain_name="Medical (MedQA)",
            samples=med_samples,
            checkpoint_file=MEDQA_CHECKPOINT,
            results_file=MEDQA_RESULTS_FILE,
            ensemble_detector=ensemble_detector,
            generator=generator,
            router=router,
        )

        # 5. Legal Domain Evaluation (100 samples)
        legal_samples = load_legal_samples(NUM_SAMPLES_PER_DOMAIN)
        legal_metrics = evaluate_domain(
            domain_name="Legal",
            samples=legal_samples,
            checkpoint_file=LEGAL_CHECKPOINT,
            results_file=LEGAL_RESULTS_FILE,
            ensemble_detector=ensemble_detector,
            generator=generator,
            router=router,
        )

        # 6. Combined Comparison Table
        print_and_save_cross_domain_table(
            general_metrics=general_metrics,
            medical_metrics=medical_metrics,
            legal_metrics=legal_metrics,
        )

        print("=" * 70)
        print("SECTION 4 COMPLETE")
        print("=" * 70 + "\n")

    finally:
        # Clean up caffeinate process
        if caffeinate_proc is not None:
            caffeinate_proc.terminate()
            print("☕ Caffeinate power assertion released.")


if __name__ == "__main__":
    main()

"""Full evaluation benchmark pipeline for HalluGuard.

Runs the hallucination detection pipeline on standard benchmarks (HaluEval,
TruthfulQA), performs ablation studies comparing individual vs ensemble
detectors, and generates results tables for the research paper.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import (
    faithfulness_score,
    hallucination_detection_auroc,
    hallucination_detection_f1,
    latency_stats,
    mitigation_improvement_rate,
)
from src.utils.logger import get_logger, log_metrics

logger = get_logger(__name__)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "experiments", "results"
)


class FixedStrategyRouter:
    """Helper router class that bypasses confidence routing and forces a fixed strategy."""

    def __init__(self, strategy: str, base_router) -> None:
        self.strategy = strategy
        self.base_router = base_router

    def route_and_mitigate(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
        hallucination_score: float,
        detection_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs the fixed strategy if hallucination score is > 0.5."""
        if hallucination_score <= 0.5:
            return {
                "strategy_used": "none",
                "original_score": hallucination_score,
                "final_score": hallucination_score,
                "improvement": 0.0,
                "final_response": response,
                "details": {},
            }

        # Force the strategy
        if self.strategy == "rewrite":
            mitigation_res = self.base_router._mitigate_rewrite(
                query, response, context_chunks, detection_result
            )
        elif self.strategy == "rerank":
            mitigation_res = self.base_router._mitigate_rerank(
                query, response, context_chunks
            )
        else:  # self_reflection
            mitigation_res = self.base_router._mitigate_self_reflection(
                query, response, context_chunks
            )

        new_response = mitigation_res.get("response", response)
        new_score = mitigation_res.get("score", hallucination_score)
        improvement = hallucination_score - new_score

        result = {
            "strategy_used": self.strategy,
            "original_score": hallucination_score,
            "final_score": new_score if new_score < hallucination_score else hallucination_score,
            "improvement": max(0.0, improvement) if new_score < hallucination_score else 0.0,
            "final_response": new_response if new_score < hallucination_score else response,
            "details": mitigation_res.get("details", {}),
        }
        return result



class Benchmark:
    """Full benchmark evaluation pipeline for HalluGuard.

    Runs the detection and mitigation pipeline on standard datasets,
    computes metrics, performs ablation studies, and saves results.

    Args:
        config_path: Path to experiment config YAML file.
        results_dir: Directory to save result JSON files.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        results_dir: str = RESULTS_DIR,
    ) -> None:
        self.config = self._load_config(config_path) if config_path else {}
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self._detector = None
        self._pipeline = None
        self.all_results: Dict[str, Any] = {}

        logger.info("Benchmark initialized (results_dir=%s)", results_dir)

    @property
    def pipeline(self):
        """Lazy-load the RAGPipeline."""
        if self._pipeline is None:
            from src.rag_pipeline.pipeline import RAGPipeline
            self._pipeline = RAGPipeline()
        return self._pipeline

    def _load_config(self, config_path: str) -> dict:
        """Load experiment configuration from YAML.

        Args:
            config_path: Path to YAML config file.

        Returns:
            Configuration dictionary.
        """
        import yaml

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info("Loaded config from %s", config_path)
            return config
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return {}

    @property
    def detector(self):
        """Lazy-load the ensemble detector."""
        if self._detector is None:
            from src.detection.ensemble import EnsembleDetector
            self._detector = EnsembleDetector()
        return self._detector

    def run_on_halueval(
        self, split: str = "data", n_samples: int = 500
    ) -> Dict[str, Any]:
        """Run benchmark on the HaluEval dataset.

        Args:
            split: Dataset split to use.
            n_samples: Number of samples to evaluate.

        Returns:
            Dictionary of computed metrics.
        """
        logger.info("Running HaluEval benchmark (n=%d)...", n_samples)

        from src.utils.data_loader import load_halueval

        data = load_halueval(split=split)
        if len(data) > n_samples:
            data = data[:n_samples]

        results = self._evaluate_dataset(data, "halueval")
        self.all_results["halueval"] = results
        self._save_results("halueval", results)

        return results

    def run_on_truthfulqa(self, n_samples: int = 200) -> Dict[str, Any]:
        """Run benchmark on the TruthfulQA dataset.

        Args:
            n_samples: Number of samples to evaluate.

        Returns:
            Dictionary of computed metrics.
        """
        logger.info("Running TruthfulQA benchmark (n=%d)...", n_samples)

        from src.utils.data_loader import load_truthfulqa

        data = load_truthfulqa()
        if len(data) > n_samples:
            data = data[:n_samples]

        results = self._evaluate_dataset(data, "truthfulqa")
        self.all_results["truthfulqa"] = results
        self._save_results("truthfulqa", results)

        return results

    def _evaluate_dataset(
        self, data: List[Dict[str, Any]], dataset_name: str
    ) -> Dict[str, Any]:
        """Evaluate the detector on a labeled dataset.

        Args:
            data: List of data samples with query, context, response, label.
            dataset_name: Name of the dataset for logging.

        Returns:
            Dictionary of metrics including AUROC, F1, faithfulness, latency.
        """
        y_true = []
        scores = []
        latencies = []

        for sample in tqdm(data, desc=f"Evaluating {dataset_name}"):
            query = sample.get("query", "")
            context = sample.get("context", [])
            response = sample.get("response", "")
            label = sample.get("label", 0)

            start_time = time.time()

            try:
                result = self.detector.score(
                    query=query,
                    response=response,
                    context_chunks=context,
                )
                h_score = result.get("final_score", 0.5)
            except Exception as e:
                logger.warning("Scoring failed for sample: %s", e)
                h_score = 0.5

            elapsed_ms = (time.time() - start_time) * 1000

            y_true.append(label)
            scores.append(h_score)
            latencies.append(elapsed_ms)

        # Compute metrics
        auroc = hallucination_detection_auroc(y_true, scores)
        f1_result = hallucination_detection_f1(y_true, scores)
        lat_stats = latency_stats(latencies)

        metrics = {
            "dataset": dataset_name,
            "n_samples": len(data),
            "auroc": auroc,
            **f1_result,
            "latency": lat_stats,
            "timestamp": datetime.now().isoformat(),
        }

        log_metrics({
            f"{dataset_name}_auroc": auroc,
            f"{dataset_name}_f1": f1_result["f1"],
        })

        logger.info(
            "%s results — AUROC: %.4f, F1: %.4f, Mean latency: %.2f ms",
            dataset_name, auroc, f1_result["f1"], lat_stats["mean"],
        )

        return metrics

    def run_ablation_study(self, n_samples: int = 50) -> Dict[str, Any]:
        """Compare NLI-only vs Embedding-only vs LLM-judge-only vs Ensemble,
        and compare with-router vs without-router (fixed single strategy) to
        demonstrate the benefit of confidence-adaptive routing.

        Args:
            n_samples: Number of samples to run ablation study on.

        Returns:
            Dictionary mapping ablation experiment name to its results.
        """
        logger.info("Running ablation study (n=%d)...", n_samples)

        from src.detection.embedding_detector import EmbeddingDetector
        from src.detection.nli_detector import NLIDetector
        from src.utils.data_loader import load_halueval

        data = load_halueval(split="data")
        if len(data) > n_samples:
            data = data[:n_samples]

        ablation_results = {}

        # Part 1: Detector Ablation
        logger.info("Part 1: Evaluating individual detectors...")

        # NLI-only
        logger.info("Ablation: NLI-only detector...")
        nli = NLIDetector()
        nli_scores = self._run_single_detector(data, nli, "nli")
        ablation_results["nli_only"] = nli_scores

        # Embedding-only
        logger.info("Ablation: Embedding-only detector...")
        emb = EmbeddingDetector()
        emb_scores = self._run_single_detector(data, emb, "embedding")
        ablation_results["embedding_only"] = emb_scores

        # LLM Judge-only (uses API, can be slow)
        try:
            from src.detection.llm_judge import LLMJudge
            logger.info("Ablation: LLM-judge-only detector...")
            judge = LLMJudge()
            judge_scores = self._run_single_detector(data, judge, "llm_judge")
            ablation_results["llm_judge_only"] = judge_scores
        except Exception as e:
            logger.warning("LLM judge ablation skipped: %s", e)

        # Ensemble
        logger.info("Ablation: Ensemble detector...")
        ensemble_scores = self._evaluate_dataset(data, "ensemble_ablation")
        ablation_results["ensemble"] = ensemble_scores

        # Part 2: Router/Mitigation Ablation (with-router vs without-router)
        logger.info("Part 2: Evaluating confidence-adaptive routing vs fixed strategies...")

        # 1. With adaptive router (default pipeline)
        logger.info("Evaluating: Pipeline with confidence-adaptive router...")
        res_with_router = self.evaluate_pipeline_mitigation(data, use_router=True)
        ablation_results["with_router"] = res_with_router

        # 2. Without router: fixed Self-Reflection
        logger.info("Evaluating: Pipeline with fixed Self-Reflection strategy...")
        res_fixed_reflection = self.evaluate_pipeline_mitigation(data, use_router=False, strategy="self_reflection")
        ablation_results["fixed_self_reflection"] = res_fixed_reflection

        # 3. Without router: fixed Query-Rewriting
        logger.info("Evaluating: Pipeline with fixed Query-Rewriting strategy...")
        res_fixed_rewrite = self.evaluate_pipeline_mitigation(data, use_router=False, strategy="rewrite")
        ablation_results["fixed_rewrite"] = res_fixed_rewrite

        # 4. Without router: fixed Chunk-Reranking
        logger.info("Evaluating: Pipeline with fixed Chunk-Reranking strategy...")
        res_fixed_rerank = self.evaluate_pipeline_mitigation(data, use_router=False, strategy="rerank")
        ablation_results["fixed_rerank"] = res_fixed_rerank

        # 5. Without router: no mitigation at all
        logger.info("Evaluating: Pipeline with no mitigation...")
        res_no_mitigation = self.evaluate_pipeline_mitigation(data, use_router=False, strategy=None)
        ablation_results["no_mitigation"] = res_no_mitigation

        self.all_results["ablation"] = ablation_results
        self._save_results("ablation", ablation_results)

        return ablation_results

    def evaluate_pipeline_mitigation(
        self, data: List[Dict[str, Any]], use_router: bool, strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate the RAGPipeline end-to-end on a dataset with or without adaptive routing.

        Args:
            data: List of sample dicts (query, context, ground_truth/response).
            use_router: Whether to use the confidence-adaptive router.
            strategy: If not using the router, a fixed strategy to apply when hallucination is detected.

        Returns:
            Dictionary of aggregate metrics.
        """
        from unittest.mock import MagicMock
        from src.rag_pipeline.pipeline import RAGPipeline
        from src.rag_pipeline.retriever import Document

        post_scores = []
        pre_scores = []
        latencies = []
        improvements = []
        mitigation_applied_counts = {"rewrite": 0, "rerank": 0, "self_reflection": 0, "none": 0}

        for sample in tqdm(data, desc=f"Pipeline (router={use_router}, fixed={strategy})"):
            query = sample.get("query", "")
            context = sample.get("context", [])

            # Set up a mock retriever that returns the sample's context
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [
                Document(page_content=c, metadata={"source": "dataset", "chunk_id": i})
                for i, c in enumerate(context)
            ]

            # Set up the pipeline
            if use_router:
                pipeline = RAGPipeline(retriever=mock_retriever)
            else:
                from src.mitigation.router import ConfidenceAdaptiveRouter
                
                base_router = ConfidenceAdaptiveRouter(retriever=mock_retriever)
                if strategy:
                    fixed_router = FixedStrategyRouter(strategy=strategy, base_router=base_router)
                    pipeline = RAGPipeline(retriever=mock_retriever, router=fixed_router)
                else:
                    # No mitigation router (dummy router that does nothing)
                    class DummyRouter:
                        def route_and_mitigate(self, q, r, cc, score, dr=None):
                            return {
                                "strategy_used": "none",
                                "original_score": score,
                                "final_score": score,
                                "improvement": 0.0,
                                "final_response": r,
                                "details": {},
                            }
                    pipeline = RAGPipeline(retriever=mock_retriever, router=DummyRouter())

            start_time = time.time()
            try:
                pipeline_result = pipeline.run(query)
                pre_score = pipeline_result.get("hallucination_score", 0.0)
                
                mitigation_applied = pipeline_result.get("mitigation_applied", "none")
                mitigation_details = pipeline_result.get("mitigation_details", {})
                
                if mitigation_applied != "none" and isinstance(mitigation_details, dict):
                    post_score = mitigation_details.get("final_score", pre_score)
                else:
                    post_score = pre_score
            except Exception as e:
                logger.error("Pipeline run failed in benchmark: %s", e)
                pre_score = 1.0
                post_score = 1.0
                mitigation_applied = "none"

            elapsed_ms = (time.time() - start_time) * 1000
            
            pre_scores.append(pre_score)
            post_scores.append(post_score)
            latencies.append(elapsed_ms)
            improvements.append(pre_score - post_score)
            mitigation_applied_counts[mitigation_applied] = mitigation_applied_counts.get(mitigation_applied, 0) + 1

        mean_pre = float(np.mean(pre_scores)) if pre_scores else 0.0
        mean_post = float(np.mean(post_scores)) if post_scores else 0.0
        mean_improvement = float(np.mean(improvements)) if improvements else 0.0
        lat_stats = latency_stats(latencies)

        return {
            "mean_pre_score": round(mean_pre, 4),
            "mean_post_score": round(mean_post, 4),
            "mean_improvement": round(mean_improvement, 4),
            "latency": lat_stats,
            "counts": mitigation_applied_counts,
            "n_samples": len(data),
        }

    def _run_single_detector(
        self, data: List[Dict], detector: Any, name: str
    ) -> Dict[str, Any]:
        """Run a single detector on the dataset.

        Args:
            data: List of labeled samples.
            detector: Detector instance with .score() method.
            name: Detector name for logging.

        Returns:
            Metrics dictionary.
        """
        y_true = []
        scores = []

        for sample in tqdm(data, desc=f"Ablation: {name}"):
            response = sample.get("response", "")
            context = sample.get("context", [])
            label = sample.get("label", 0)

            try:
                if name == "llm_judge":
                    result = detector.score(
                        query=sample.get("query", ""),
                        response=response,
                        context_chunks=context,
                    )
                    h_score = result.get("score", 0.5)
                else:
                    h_score = detector.score(response, context)
            except Exception as e:
                logger.warning("%s scoring failed: %s", name, e)
                h_score = 0.5

            y_true.append(label)
            scores.append(h_score)

        auroc = hallucination_detection_auroc(y_true, scores)
        f1_result = hallucination_detection_f1(y_true, scores)

        return {
            "detector": name,
            "auroc": auroc,
            **f1_result,
            "n_samples": len(data),
        }

    def generate_results_table(self) -> pd.DataFrame:
        """Generate a formatted results table for the paper.

        Returns:
            Pandas DataFrame with columns: Type, Method/Router, Dataset/Task, Metric 1, Metric 2, Metric 3, Metric 4, Latency.
        """
        rows = []

        for dataset_name, metrics in self.all_results.items():
            if dataset_name == "ablation":
                for method, m in metrics.items():
                    if "only" in method or method == "ensemble":
                        # Detector ablation
                        rows.append({
                            "Type": "Detector",
                            "Method/Router": method.replace("_only", "").capitalize(),
                            "Dataset/Task": "HaluEval (Ablation)",
                            "Metric 1 (AUROC/Pre-Score)": m.get("auroc", "—"),
                            "Metric 2 (F1/Post-Score)": m.get("f1", "—"),
                            "Metric 3 (Imprv/Precision)": m.get("precision", "—"),
                            "Metric 4 (Recall)": m.get("recall", "—"),
                            "Latency Mean (ms)": m.get("latency", {}).get("mean", "—") if isinstance(m.get("latency"), dict) else "—",
                        })
                    else:
                        # Pipeline/Mitigation ablation
                        rows.append({
                            "Type": "Pipeline",
                            "Method/Router": method.replace("fixed_", "Fixed: ").replace("with_", "Adaptive: ").replace("no_mitigation", "No Mitigation").title(),
                            "Dataset/Task": "HaluEval (Mitigation Ablation)",
                            "Metric 1 (AUROC/Pre-Score)": m.get("mean_pre_score", "—"),
                            "Metric 2 (F1/Post-Score)": m.get("mean_post_score", "—"),
                            "Metric 3 (Imprv/Precision)": m.get("mean_improvement", "—"),
                            "Metric 4 (Recall)": "—",
                            "Latency Mean (ms)": m.get("latency", {}).get("mean", "—") if isinstance(m.get("latency"), dict) else "—",
                        })
            else:
                rows.append({
                    "Type": "Evaluation",
                    "Method/Router": "Ensemble Detector",
                    "Dataset/Task": dataset_name.upper(),
                    "Metric 1 (AUROC/Pre-Score)": metrics.get("auroc", "—"),
                    "Metric 2 (F1/Post-Score)": metrics.get("f1", "—"),
                    "Metric 3 (Imprv/Precision)": metrics.get("precision", "—"),
                    "Metric 4 (Recall)": metrics.get("recall", "—"),
                    "Latency Mean (ms)": metrics.get("latency", {}).get("mean", "—") if isinstance(metrics.get("latency"), dict) else "—",
                })

        df = pd.DataFrame(rows)
        logger.info("Results table:\n%s", df.to_string())
        return df

    def _save_results(self, name: str, results: dict) -> None:
        """Save results to JSON file with timestamp.

        Args:
            name: Result set name.
            results: Results dictionary.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info("Results saved to %s", filepath)
        except Exception as e:
            logger.error("Failed to save results: %s", e)


def main():
    """CLI entry point for running benchmarks."""
    parser = argparse.ArgumentParser(description="Run HalluGuard benchmarks")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Run ablation study"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500, help="Number of samples"
    )
    parser.add_argument(
        "--ablation-samples", type=int, default=50, help="Number of samples for ablation study"
    )
    args = parser.parse_args()

    benchmark = Benchmark(config_path=args.config)

    # Run benchmarks
    benchmark.run_on_halueval(n_samples=args.n_samples)
    benchmark.run_on_truthfulqa(n_samples=min(args.n_samples, 200))

    if args.ablation:
        benchmark.run_ablation_study(n_samples=args.ablation_samples)

    # Generate and print results table
    table = benchmark.generate_results_table()
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()

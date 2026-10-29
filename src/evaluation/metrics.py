"""Evaluation metrics for hallucination detection and RAG quality assessment.

Provides functions for computing faithfulness, AUROC, F1, latency statistics,
and mitigation improvement rates for benchmarking the HalluGuard pipeline.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def faithfulness_score(
    responses: List[str],
    contexts: List[List[str]],
    detector: Any,
) -> float:
    """Compute mean faithfulness score across a set of response-context pairs.

    Args:
        responses: List of LLM-generated responses.
        contexts: List of context chunk lists (one per response).
        detector: Detector instance with a .score() method.

    Returns:
        Mean faithfulness score (1 - mean hallucination score).

    Raises:
        ValueError: If responses and contexts have different lengths.
    """
    if len(responses) != len(contexts):
        raise ValueError(
            f"responses ({len(responses)}) and contexts ({len(contexts)}) must have same length."
        )

    if not responses:
        return 0.0

    from tqdm import tqdm

    hallucination_scores = []
    for resp, ctx in tqdm(zip(responses, contexts), total=len(responses), desc="Computing faithfulness"):
        try:
            if hasattr(detector, 'score') and callable(detector.score):
                # Handle both ensemble (needs query) and simple detectors
                try:
                    result = detector.score(query="", response=resp, context_chunks=ctx)
                    h_score = result.get("final_score", result) if isinstance(result, dict) else float(result)
                except TypeError:
                    h_score = detector.score(resp, ctx)
            else:
                h_score = 0.5
            hallucination_scores.append(h_score)
        except Exception as e:
            logger.warning("Faithfulness scoring failed for a sample: %s", e)
            hallucination_scores.append(0.5)

    mean_hallucination = float(np.mean(hallucination_scores))
    faithfulness = 1.0 - mean_hallucination

    logger.info("Mean faithfulness: %.4f (n=%d)", faithfulness, len(responses))
    return round(faithfulness, 4)


def answer_relevance(
    queries: List[str],
    responses: List[str],
) -> float:
    """Compute answer relevance score using embedding similarity.

    Measures how relevant the response is to the original query
    using cosine similarity of their sentence embeddings.

    Args:
        queries: List of original queries.
        responses: List of generated responses.

    Returns:
        Mean answer relevance score (0.0 to 1.0).

    Raises:
        ValueError: If queries and responses have different lengths.
    """
    if len(queries) != len(responses):
        raise ValueError("queries and responses must have same length.")

    if not queries:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embeddings = model.encode(queries, show_progress_bar=False)
        response_embeddings = model.encode(responses, show_progress_bar=False)

        # Compute pairwise cosine similarity
        similarities = []
        for q_emb, r_emb in zip(query_embeddings, response_embeddings):
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            r_norm = r_emb / (np.linalg.norm(r_emb) + 1e-9)
            sim = float(np.dot(q_norm, r_norm))
            similarities.append(max(0.0, sim))

        mean_relevance = float(np.mean(similarities))
        logger.info("Mean answer relevance: %.4f", mean_relevance)
        return round(mean_relevance, 4)

    except Exception as e:
        logger.error("Answer relevance computation failed: %s", e)
        return 0.0


def context_recall(
    responses: List[str],
    ground_truths: List[str],
) -> float:
    """Compute context recall by measuring overlap with ground truth.

    Uses token-level F1 overlap between generated responses and ground truth
    answers to approximate how much relevant information was recalled.

    Args:
        responses: List of generated responses.
        ground_truths: List of ground truth answer strings.

    Returns:
        Mean context recall score (0.0 to 1.0).
    """
    if len(responses) != len(ground_truths):
        raise ValueError("responses and ground_truths must have same length.")

    if not responses:
        return 0.0

    recalls = []
    for resp, gt in zip(responses, ground_truths):
        resp_tokens = set(resp.lower().split())
        gt_tokens = set(gt.lower().split())

        if not gt_tokens:
            recalls.append(0.0)
            continue

        overlap = resp_tokens & gt_tokens
        recall = len(overlap) / len(gt_tokens)
        recalls.append(recall)

    mean_recall = float(np.mean(recalls))
    logger.info("Mean context recall: %.4f", mean_recall)
    return round(mean_recall, 4)


def hallucination_detection_auroc(
    y_true: List[int],
    scores: List[float],
) -> float:
    """Compute AUROC for hallucination detection.

    Args:
        y_true: Ground truth labels (0=faithful, 1=hallucinated).
        scores: Predicted hallucination scores (0.0 to 1.0).

    Returns:
        Area Under the ROC Curve.

    Raises:
        ValueError: If inputs have different lengths or contain invalid values.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have same length.")

    if not y_true:
        return 0.0

    # Check that we have both classes
    unique_labels = set(y_true)
    if len(unique_labels) < 2:
        logger.warning("Only one class present in y_true. AUROC is undefined.")
        return 0.5

    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true, scores)
        logger.info("AUROC: %.4f", auroc)
        return round(float(auroc), 4)
    except Exception as e:
        logger.error("AUROC computation failed: %s", e)
        return 0.0


def hallucination_detection_f1(
    y_true: List[int],
    scores: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute F1 score and related metrics for hallucination detection.

    Args:
        y_true: Ground truth labels (0=faithful, 1=hallucinated).
        scores: Predicted hallucination scores (0.0 to 1.0).
        threshold: Score threshold for binary classification.

    Returns:
        Dictionary with precision, recall, f1, accuracy, and threshold.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have same length.")

    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "threshold": threshold}

    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        y_pred = [1 if s > threshold else 0 for s in scores]

        result = {
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "threshold": threshold,
        }

        logger.info("Detection F1: %.4f (P=%.4f, R=%.4f)", result["f1"], result["precision"], result["recall"])
        return result

    except Exception as e:
        logger.error("F1 computation failed: %s", e)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "threshold": threshold}


def latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute latency statistics.

    Args:
        latencies: List of latency values in milliseconds.

    Returns:
        Dictionary with mean, median (p50), p95, p99, min, max.
    """
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(latencies)
    result = {
        "mean": round(float(np.mean(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }

    logger.info("Latency — mean: %.2f ms, p50: %.2f ms, p95: %.2f ms", result["mean"], result["p50"], result["p95"])
    return result


def mitigation_improvement_rate(
    pre_scores: List[float],
    post_scores: List[float],
) -> float:
    """Compute the mean improvement rate after mitigation.

    Args:
        pre_scores: Hallucination scores before mitigation.
        post_scores: Hallucination scores after mitigation.

    Returns:
        Mean improvement rate (positive = improvement, negative = degradation).

    Raises:
        ValueError: If input lists have different lengths.
    """
    if len(pre_scores) != len(post_scores):
        raise ValueError("pre_scores and post_scores must have same length.")

    if not pre_scores:
        return 0.0

    improvements = [pre - post for pre, post in zip(pre_scores, post_scores)]
    mean_improvement = float(np.mean(improvements))

    logger.info("Mean mitigation improvement: %.4f", mean_improvement)
    return round(mean_improvement, 4)

"""Weighted ensemble hallucination detector combining NLI, embedding, and LLM judge.

Aggregates scores from three independent detection methods into a single
hallucination score using configurable weights, providing robust detection
that leverages the complementary strengths of each method.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleDetector:
    """Weighted ensemble of NLI, embedding, and LLM-judge hallucination detectors.

    Combines three complementary detection signals:
    - NLI (weight=0.5): Logical entailment between context and response sentences
    - Embedding (weight=0.2): Semantic similarity via dense embeddings
    - LLM Judge (weight=0.3): LLM-based faithfulness assessment

    Args:
        nli_detector: NLIDetector instance (lazy-loaded if None).
        embedding_detector: EmbeddingDetector instance (lazy-loaded if None).
        llm_judge: LLMJudge instance (lazy-loaded if None).
        weights: Dictionary of detector weights (must sum to ~1.0).
        threshold: Hallucination detection threshold (default: 0.5).
    """

    THRESHOLD = 0.5

    def __init__(
        self,
        nli_detector=None,
        embedding_detector=None,
        llm_judge=None,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ) -> None:
        self._nli_detector = nli_detector
        self._embedding_detector = embedding_detector
        self._llm_judge = llm_judge
        self.weights = weights or {"nli": 0.5, "embedding": 0.2, "llm_judge": 0.3}
        self.THRESHOLD = threshold

        # Validate weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Detector weights sum to %.3f, not 1.0. Normalizing.", total
            )
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(
            "EnsembleDetector initialized (weights=%s, threshold=%.2f)",
            self.weights, self.THRESHOLD,
        )

    @property
    def nli_detector(self):
        """Lazy-load NLI detector."""
        if self._nli_detector is None:
            from src.detection.nli_detector import NLIDetector
            self._nli_detector = NLIDetector()
        return self._nli_detector

    @property
    def embedding_detector(self):
        """Lazy-load embedding detector."""
        if self._embedding_detector is None:
            from src.detection.embedding_detector import EmbeddingDetector
            self._embedding_detector = EmbeddingDetector()
        return self._embedding_detector

    @property
    def llm_judge(self):
        """Lazy-load LLM judge."""
        if self._llm_judge is None:
            from src.detection.llm_judge import LLMJudge
            self._llm_judge = LLMJudge()
        return self._llm_judge

    def score(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
    ) -> Dict[str, Any]:
        """Run all three detectors in parallel and compute weighted ensemble score.

        Uses ThreadPoolExecutor to run NLI, Embedding, and LLM Judge
        simultaneously, reducing total detection time from
        NLI + Embedding + LLM Judge  →  max(NLI, Embedding, LLM Judge).

        Args:
            query: The original user query.
            response: The response to evaluate for hallucination.
            context_chunks: Context passages the response should be grounded in.

        Returns:
            Dictionary containing:
                - final_score (float): Weighted ensemble hallucination score 0.0-1.0
                - nli_score (float): NLI detector score
                - embedding_score (float): Embedding detector score
                - llm_judge_score (float): LLM judge score
                - is_hallucinated (bool): True if final_score > threshold
                - threshold (float): Current threshold value
                - problematic_claims (list): Claims flagged by LLM judge
                - reasoning (str): LLM judge reasoning
        """
        with ThreadPoolExecutor(max_workers=3) as executor:

            # Submit all three detectors simultaneously
            future_nli = executor.submit(
                self.nli_detector.score,
                response,
                context_chunks,
            )
            future_emb = executor.submit(
                self.embedding_detector.score,
                response,
                context_chunks,
            )
            future_llm = executor.submit(
                self.llm_judge.score,
                query,
                response,
                context_chunks,
            )

            # Collect results with error handling
            try:
                nli_score = future_nli.result(timeout=30)
            except Exception as e:
                logger.error(f"NLI detector failed: {e}")
                nli_score = 0.5

            try:
                emb_score = future_emb.result(timeout=30)
            except Exception as e:
                logger.error(f"Embedding detector failed: {e}")
                emb_score = 0.5

            try:
                llm_result = future_llm.result(timeout=30)
                llm_score = llm_result.get("score", 0.5) if isinstance(
                    llm_result, dict) else 0.5
                problematic_claims = llm_result.get(
                    "problematic_claims", []) if isinstance(
                    llm_result, dict) else []
                reasoning = llm_result.get(
                    "reasoning", "") if isinstance(
                    llm_result, dict) else ""
            except Exception as e:
                logger.error(f"LLM judge failed: {e}")
                llm_score = 0.5
                problematic_claims = []
                reasoning = ""

        # Weighted ensemble fusion
        final_score = (
            nli_score * self.weights["nli"]
            + emb_score * self.weights["embedding"]
            + llm_score * self.weights["llm_judge"]
        )

        is_hallucinated = final_score > self.THRESHOLD

        logger.info(
            "Ensemble score: %.3f (NLI=%.3f, Emb=%.3f, Judge=%.3f) → %s",
            final_score, nli_score, emb_score, llm_score,
            "HALLUCINATED" if is_hallucinated else "FAITHFUL",
        )

        return {
            "final_score": round(final_score, 4),
            "nli_score": round(nli_score, 4),
            "embedding_score": round(emb_score, 4),
            "llm_judge_score": round(llm_score, 4),
            "is_hallucinated": is_hallucinated,
            "threshold": self.THRESHOLD,
            "problematic_claims": problematic_claims,
            "reasoning": reasoning,
        }

    def _run_nli(self, response: str, context_chunks: List[str]) -> float:
        """Run NLI detector with error fallback.

        Args:
            response: Response text.
            context_chunks: Context passages.

        Returns:
            NLI hallucination score, or 0.5 on failure.
        """
        try:
            return self.nli_detector.score(response, context_chunks)
        except Exception as e:
            logger.error("NLI detector failed: %s. Using fallback score 0.5.", e)
            return 0.5

    def _run_embedding(self, response: str, context_chunks: List[str]) -> float:
        """Run embedding detector with error fallback.

        Args:
            response: Response text.
            context_chunks: Context passages.

        Returns:
            Embedding hallucination score, or 0.5 on failure.
        """
        try:
            return self.embedding_detector.score(response, context_chunks)
        except Exception as e:
            logger.error("Embedding detector failed: %s. Using fallback score 0.5.", e)
            return 0.5

    def _run_llm_judge(
        self, query: str, response: str, context_chunks: List[str]
    ) -> Dict[str, Any]:
        """Run LLM judge with error fallback.

        Args:
            query: Original query.
            response: Response text.
            context_chunks: Context passages.

        Returns:
            LLM judge result dict, or fallback on failure.
        """
        try:
            return self.llm_judge.score(query, response, context_chunks)
        except Exception as e:
            logger.error("LLM judge failed: %s. Using fallback score 0.5.", e)
            return {
                "score": 0.5,
                "faithfulness_score": 3,
                "reasoning": f"Judge failed: {e}",
                "problematic_claims": [],
            }

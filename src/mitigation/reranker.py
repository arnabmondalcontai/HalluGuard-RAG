"""Cross-encoder chunk reranker for improved context quality.

Reranks retrieved context chunks by relevance to the query using
a cross-encoder model, ensuring the generator receives the most
pertinent information for faithful response generation.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkReranker:
    """Cross-encoder reranker for improving context chunk relevance.

    Uses a cross-encoder model to score each chunk for relevance to
    the query, returning only the top-k most relevant chunks. This
    improves context quality before re-generation during mitigation.

    Args:
        model_name: HuggingFace cross-encoder model for reranking.
        device: Device for inference ("cpu", "cuda", or None for auto-detect).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

        logger.info("ChunkReranker initialized (model=%s)", model_name)

    @property
    def model(self):
        """Lazy-load and cache the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model: %s...", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker model loaded.")
        return self._model

    def rerank(
        self, query: str, chunks: List[str], top_k: int = 3
    ) -> List[str]:
        """Rerank context chunks by relevance to query.

        Args:
            query: The search query to rank against.
            chunks: List of context chunk texts to rerank.
            top_k: Number of top-ranked chunks to return.

        Returns:
            List of top_k most relevant chunks, ordered by relevance.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty.")

        if len(chunks) <= top_k:
            logger.debug("Fewer chunks (%d) than top_k (%d). Returning all.", len(chunks), top_k)
            return chunks

        # Create query-chunk pairs for cross-encoder scoring
        pairs = [(query, chunk) for chunk in chunks]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Sort by score descending
            scored_chunks = list(zip(chunks, scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            top_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]

            logger.info(
                "Reranked %d chunks → top %d (scores: %.3f to %.3f)",
                len(chunks), top_k,
                scored_chunks[0][1], scored_chunks[top_k - 1][1],
            )

            return top_chunks

        except Exception as e:
            logger.error("Reranking failed: %s. Returning first %d chunks.", e, top_k)
            return chunks[:top_k]

    def rerank_with_scores(
        self, query: str, chunks: List[str], top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Rerank chunks and return with their relevance scores.

        Args:
            query: The search query.
            chunks: Context chunks to rerank.
            top_k: Number of results to return.

        Returns:
            List of (chunk_text, relevance_score) tuples, sorted by score.
        """
        if not chunks:
            return []

        pairs = [(query, chunk) for chunk in chunks]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            scored_chunks = sorted(
                zip(chunks, scores), key=lambda x: x[1], reverse=True
            )
            return scored_chunks[:top_k]

        except Exception as e:
            logger.error("Reranking with scores failed: %s", e)
            return [(chunk, 0.0) for chunk in chunks[:top_k]]

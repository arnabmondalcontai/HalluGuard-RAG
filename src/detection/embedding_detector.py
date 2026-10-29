"""Embedding-based hallucination detector using cosine similarity.

Measures semantic similarity between response and context embeddings
to detect hallucinated content that drifts from the provided evidence.
Provides both response-level and sentence-level scoring.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingDetector:
    """Embedding-based hallucination detector using cosine similarity.

    Embeds the response and all context chunks, then computes the maximum
    cosine similarity. Low similarity indicates the response has drifted
    from the provided context (potential hallucination).

    Args:
        model_name: Sentence-transformer model for embeddings.
        device: Device for inference ("cpu", "cuda", or None for auto-detect).
    """

    _model = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device

        if EmbeddingDetector._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model for first time...")
            EmbeddingDetector._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded and cached")

        self.model = EmbeddingDetector._model
        self._model = EmbeddingDetector._model

        logger.info("EmbeddingDetector initialized (model=%s)", model_name)

    def score(self, response: str, context_chunks: List[str]) -> float:
        """Compute hallucination score based on embedding similarity.

        Score of 0.0 = response is semantically identical to context (faithful).
        Score of 1.0 = response is completely unrelated (hallucinated).

        Args:
            response: The LLM-generated response to evaluate.
            context_chunks: List of context passages for comparison.

        Returns:
            Hallucination score between 0.0 and 1.0.

        Raises:
            ValueError: If response is empty.
        """
        if not response or not response.strip():
            raise ValueError("Response cannot be empty.")

        if not context_chunks:
            logger.warning("No context provided. Returning max hallucination score.")
            return 1.0

        # Encode response and context
        response_embedding = self.model.encode([response], show_progress_bar=False)
        context_embeddings = self.model.encode(context_chunks, show_progress_bar=False)

        # Compute cosine similarities
        similarities = self._cosine_similarity(response_embedding, context_embeddings)

        # Max similarity across all chunks
        max_similarity = float(np.max(similarities))

        # Hallucination score = 1 - max_similarity
        hallucination_score = max(0.0, min(1.0, 1.0 - max_similarity))

        logger.debug(
            "Embedding score: %.3f (max similarity: %.3f, %d chunks)",
            hallucination_score, max_similarity, len(context_chunks),
        )

        return round(hallucination_score, 4)

    def sentence_level_score(
        self, response: str, context_chunks: List[str]
    ) -> List[Dict[str, object]]:
        """Compute per-sentence hallucination scores with highlighting info.

        Args:
            response: The LLM-generated response to evaluate.
            context_chunks: Context passages for comparison.

        Returns:
            List of dicts per sentence with keys:
                - sentence (str): The sentence text
                - similarity (float): Max cosine similarity to any context chunk
                - hallucination_score (float): 1 - similarity
                - is_hallucinated (bool): True if score > 0.5
        """
        if not response or not context_chunks:
            return []

        # Tokenize response into sentences
        sentences = self._tokenize_sentences(response)
        if not sentences:
            return []

        # Encode all sentences and context
        sentence_embeddings = self.model.encode(sentences, show_progress_bar=False)
        context_embeddings = self.model.encode(context_chunks, show_progress_bar=False)

        results = []
        for i, (sent, sent_emb) in enumerate(zip(sentences, sentence_embeddings)):
            similarities = self._cosine_similarity(
                sent_emb.reshape(1, -1), context_embeddings
            )
            max_sim = float(np.max(similarities))
            h_score = max(0.0, min(1.0, 1.0 - max_sim))

            results.append({
                "sentence": sent,
                "similarity": round(max_sim, 4),
                "hallucination_score": round(h_score, 4),
                "is_hallucinated": h_score > 0.5,
            })

        return results

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between embedding arrays.

        Args:
            a: First embedding array of shape (n, d).
            b: Second embedding array of shape (m, d).

        Returns:
            Similarity matrix of shape (n, m).
        """
        a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
        return np.dot(a_norm, b_norm.T)

    @staticmethod
    def _tokenize_sentences(text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text to split.

        Returns:
            List of sentence strings.
        """
        try:
            import nltk
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

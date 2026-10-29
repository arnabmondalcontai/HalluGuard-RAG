"""NLI-based hallucination detector using DeBERTa v3 cross-encoder.

Scores response faithfulness by running Natural Language Inference (NLI)
between each response sentence (hypothesis) and the concatenated context
(premise). High entailment probability indicates faithfulness; low
entailment indicates potential hallucination.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NLIDetector:
    """NLI-based hallucination detector using a cross-encoder model.

    For each sentence in the response, computes the probability that it
    is entailed by the provided context. The hallucination score is
    1 - mean(entailment_probabilities).

    Args:
        model_name: HuggingFace cross-encoder model for NLI.
        batch_size: Number of sentence pairs to process in each batch.
        device: Device for inference ("cpu", "cuda", or None for auto-detect).
    """

    _model = None
    _tokenizer = None

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        if NLIDetector._model is None:
            logger.info("Loading NLI model for first time...")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            NLIDetector._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            NLIDetector._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )
            # Determine device and set to eval
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            NLIDetector._model = NLIDetector._model.to(dev)
            NLIDetector._model.eval()
            logger.info("NLI model loaded and cached")
        
        self.model = NLIDetector._model
        self.tokenizer = NLIDetector._tokenizer
        self._model = NLIDetector._model
        self._tokenizer = NLIDetector._tokenizer

        logger.info("NLIDetector initialized (model=%s, batch_size=%d)", model_name, batch_size)

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK sentence tokenizer.

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
        except Exception as e:
            logger.warning("NLTK tokenization failed: %s. Falling back to split.", e)
            # Fallback: split on common sentence terminators
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def score(self, response: str, context_chunks: List[str]) -> float:
        """Compute hallucination score for a response against context.

        Score of 0.0 = fully faithful (all sentences entailed by context).
        Score of 1.0 = fully hallucinated (no sentences entailed).

        Args:
            response: The LLM-generated response to evaluate.
            context_chunks: List of context passages the response should be grounded in.

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

        sentences = self._tokenize_sentences(response)
        if not sentences:
            return 1.0

        # Concatenate all context as the NLI premise
        premise = " ".join(context_chunks)

        # Compute entailment probability for each sentence
        entailment_probs = self._batch_nli(premise, sentences)

        # Hallucination score = 1 - mean(entailment)
        mean_entailment = float(np.mean(entailment_probs))
        hallucination_score = 1.0 - mean_entailment

        logger.debug(
            "NLI score: %.3f (mean entailment: %.3f, %d sentences)",
            hallucination_score, mean_entailment, len(sentences),
        )

        return round(hallucination_score, 4)

    def score_detailed(
        self, response: str, context_chunks: List[str]
    ) -> Dict[str, object]:
        """Compute detailed per-sentence hallucination analysis.

        Args:
            response: The LLM-generated response to evaluate.
            context_chunks: Context passages for grounding.

        Returns:
            Dictionary with overall score and per-sentence breakdown.
        """
        if not response or not response.strip():
            return {"score": 1.0, "sentences": [], "mean_entailment": 0.0}

        if not context_chunks:
            return {"score": 1.0, "sentences": [], "mean_entailment": 0.0}

        sentences = self._tokenize_sentences(response)
        premise = " ".join(context_chunks)
        entailment_probs = self._batch_nli(premise, sentences)

        sentence_details = []
        for sent, prob in zip(sentences, entailment_probs):
            sentence_details.append({
                "sentence": sent,
                "entailment_prob": round(float(prob), 4),
                "is_hallucinated": float(prob) < 0.5,
            })

        mean_entailment = float(np.mean(entailment_probs))
        return {
            "score": round(1.0 - mean_entailment, 4),
            "mean_entailment": round(mean_entailment, 4),
            "sentences": sentence_details,
        }

    def _batch_nli(self, premise: str, hypotheses: List[str]) -> List[float]:
        """Run batched NLI inference to get entailment probabilities.

        Args:
            premise: The context text (NLI premise).
            hypotheses: List of sentences to check (NLI hypotheses).

        Returns:
            List of entailment probabilities for each hypothesis.
        """
        import torch

        model = self.model  # Trigger lazy loading
        entailment_probs = []

        # Process in batches
        for i in range(0, len(hypotheses), self.batch_size):
            batch = hypotheses[i : i + self.batch_size]

            # Tokenize batch
            inputs = self._tokenizer(
                [premise] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Get entailment probabilities
            # NLI labels: 0=contradiction, 1=neutral, 2=entailment
            probs = torch.softmax(logits, dim=-1)
            entailment_idx = 2 if logits.shape[-1] == 3 else 0
            batch_probs = probs[:, entailment_idx].cpu().numpy().tolist()
            entailment_probs.extend(batch_probs)

        return entailment_probs

"""Claim-level explanation module for hallucination detection.

Explains why specific sentences/claims in a generated response are flagged
as hallucinations by comparing each claim against the retrieved context chunks.
"""

import logging
from typing import Any, Dict, List, Optional

from src.detection.nli_detector import NLIDetector
from src.rag_pipeline.generator import LLMGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainabilityModule:
    """Explains why individual claims are flagged as hallucinated in plain language.

    Attributes:
        generator: LLMGenerator instance for generating explanations.
        nli_detector: NLIDetector instance for sentence-level faithfulness scoring.
        THRESHOLD: Float score threshold above which a sentence is considered hallucinated.
        logger: Logger instance.
    """

    def __init__(
        self,
        generator: Optional[LLMGenerator] = None,
        nli_detector: Optional[NLIDetector] = None,
    ) -> None:
        self.generator = generator
        self.nli_detector = nli_detector
        self.THRESHOLD = 0.5
        self.logger = logging.getLogger(__name__)

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK sentence tokenizer with fallback."""
        if not text or not text.strip():
            return []
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
            self.logger.warning("Sentence tokenization failed: %s. Falling back to split.", e)
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _find_most_relevant_chunk(
        self, sentence: str, context_chunks: List[str]
    ) -> str:
        """Find the context chunk with highest entailment (lowest hallucination score)."""
        if not context_chunks:
            return ""
        if len(context_chunks) == 1:
            return context_chunks[0]

        best_chunk = context_chunks[0]
        lowest_score = float("inf")

        for chunk in context_chunks:
            try:
                score = self.nli_detector.score(sentence, [chunk])
                if score < lowest_score:
                    lowest_score = score
                    best_chunk = chunk
            except Exception as e:
                self.logger.debug("Error scoring single chunk against sentence: %s", e)

        return best_chunk

    def explain(self, response: str, context_chunks: List[str]) -> List[Dict[str, Any]]:
        """Score each sentence, generate explanation for hallucinated ones.

        Args:
            response: Generated text to analyse
            context_chunks: Retrieved context chunks

        Returns:
            List of dicts with keys:
                sentence: str
                score: float (0=faithful, 1=hallucinated)
                is_hallucinated: bool
                explanation: str (empty if faithful)
                supporting_chunk: str (most relevant chunk)
        """
        try:
            sentences = self._tokenize_sentences(response)
            if not sentences:
                return []

            results: List[Dict[str, Any]] = []
            for sentence in sentences:
                try:
                    if self.nli_detector is not None:
                        score = float(self.nli_detector.score(sentence, context_chunks))
                        supporting_chunk = self._find_most_relevant_chunk(sentence, context_chunks)
                    else:
                        score = 0.0
                        supporting_chunk = context_chunks[0] if context_chunks else ""

                    is_hallucinated = score > self.THRESHOLD
                    explanation = ""

                    if is_hallucinated and self.generator is not None:
                        prompt = (
                            f"A claim was made: '{sentence}'\n"
                            f"The retrieved context states: '{supporting_chunk}'\n"
                            f"In exactly one sentence explain why this claim is not "
                            f"supported by the context. Be specific about what differs."
                        )
                        try:
                            explanation = self.generator.generate(
                                query=prompt,
                                context_chunks=[],
                            )
                            if not explanation:
                                explanation = "Claim is not supported by the retrieved context."
                        except Exception as gen_err:
                            self.logger.error("Explanation generation failed: %s", gen_err)
                            explanation = "Claim contradicts or is unsupported by the context."

                    results.append({
                        "sentence": sentence,
                        "score": round(score, 4),
                        "is_hallucinated": is_hallucinated,
                        "explanation": explanation,
                        "supporting_chunk": supporting_chunk,
                    })

                except Exception as sent_err:
                    self.logger.error("Error explaining sentence '%s': %s", sentence, sent_err)
                    results.append({
                        "sentence": sentence,
                        "score": 0.5,
                        "is_hallucinated": False,
                        "explanation": "",
                        "supporting_chunk": context_chunks[0] if context_chunks else "",
                    })

            return results

        except Exception as e:
            self.logger.error("ExplainabilityModule.explain failed: %s", e)
            return []

    def highlight_response(self, response: str, results: List[Dict[str, Any]]) -> str:
        """Mark hallucinated sentences in the response text.

        Returns response with format:
        Faithful sentences unchanged.
        Hallucinated: [HALLUCINATED: sentence | Reason: explanation]
        """
        if not response or not results:
            return response or ""

        highlighted = response
        for item in results:
            if item.get("is_hallucinated", False):
                sentence = item.get("sentence", "")
                explanation = item.get("explanation", "")
                hallucination_tag = f"[HALLUCINATED: {sentence} | Reason: {explanation}]"
                if sentence in highlighted:
                    highlighted = highlighted.replace(sentence, hallucination_tag, 1)

        return highlighted

    def summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summary of explanation results.

        Returns:
            total_sentences: int
            hallucinated_count: int
            faithful_count: int
            hallucination_rate: float
            explanations: List[str] (only hallucinated sentences)
        """
        total_sentences = len(results)
        hallucinated_count = sum(1 for r in results if r.get("is_hallucinated", False))
        faithful_count = total_sentences - hallucinated_count
        hallucination_rate = (
            round(hallucinated_count / total_sentences, 4) if total_sentences > 0 else 0.0
        )
        explanations = [
            r.get("explanation", "")
            for r in results
            if r.get("is_hallucinated", False) and r.get("explanation")
        ]

        return {
            "total_sentences": total_sentences,
            "hallucinated_count": hallucinated_count,
            "faithful_count": faithful_count,
            "hallucination_rate": hallucination_rate,
            "explanations": explanations,
        }

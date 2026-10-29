"""Query rewriting module for improved retrieval on hallucination detection.

When hallucination is detected, generates alternative query phrasings that
are more specific and targeted, enabling the retriever to find better
context for re-generation.
"""

import json
import logging
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

REWRITE_PROMPT = """You are a search query optimizer. The original query did not retrieve good enough context, leading to a potentially hallucinated response.

## Original Query:
{original_query}

## Failed Response (contains hallucinations):
{failed_response}

## Problematic Claims:
{problematic_claims}

## Task:
Generate exactly 3 alternative phrasings of the original query that are:
1. More specific and targeted
2. Use different keywords that might match relevant documents
3. Break down complex questions into simpler sub-questions

Respond with ONLY a JSON array of 3 strings:
["query 1", "query 2", "query 3"]"""


class QueryRewriter:
    """Query rewriting for improved retrieval after hallucination detection.

    Uses an LLM to generate alternative query phrasings that are more specific
    and targeted, helping the retriever find better context for re-generation.

    Args:
        model_name: Groq model identifier.
        temperature: Sampling temperature (slightly higher for diversity).
        num_rewrites: Number of alternative queries to generate.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        temperature: float = 0.3,
        num_rewrites: int = 3,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.num_rewrites = num_rewrites
        self._generator = None

        logger.info("QueryRewriter initialized (model=%s)", model_name)

    @property
    def generator(self):
        """Lazy-load the LLM generator."""
        if self._generator is None:
            from src.rag_pipeline.generator import LLMGenerator
            self._generator = LLMGenerator(
                model_name=self.model_name,
                temperature=self.temperature,
            )
        return self._generator

    def rewrite(
        self,
        original_query: str,
        failed_response: str,
        problematic_claims: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate alternative query phrasings for improved retrieval.

        Args:
            original_query: The original user query that led to hallucination.
            failed_response: The hallucinated response text.
            problematic_claims: Specific claims identified as unsupported.

        Returns:
            List of rewritten query strings (up to num_rewrites).

        Raises:
            RuntimeError: If LLM generation completely fails.
        """
        claims_text = (
            "\n".join(f"- {claim}" for claim in problematic_claims)
            if problematic_claims
            else "Not specified"
        )

        prompt = REWRITE_PROMPT.format(
            original_query=original_query,
            failed_response=failed_response[:500],  # Truncate long responses
            problematic_claims=claims_text,
        )

        try:
            raw_result = self.generator.generate(
                query=prompt,
                context_chunks=[],
                system_prompt="You output ONLY valid JSON arrays. No explanation.",
            )

            queries = self._parse_queries(raw_result)

            if queries:
                logger.info("Generated %d rewritten queries.", len(queries))
                return queries[: self.num_rewrites]

        except Exception as e:
            logger.error("Query rewriting failed: %s", e)

        # Fallback: generate simple variations
        logger.warning("Using fallback query variations.")
        return self._fallback_rewrites(original_query)

    def _parse_queries(self, raw_response: str) -> List[str]:
        """Parse JSON array of queries from LLM response.

        Args:
            raw_response: Raw LLM output containing JSON array.

        Returns:
            List of query strings, or empty list on failure.
        """
        text = raw_response.strip()

        # Strip markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find array boundaries
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(q).strip() for q in result if q and str(q).strip()]
        except json.JSONDecodeError:
            pass

        return []

    def _fallback_rewrites(self, query: str) -> List[str]:
        """Generate simple fallback query variations.

        Args:
            query: Original query to vary.

        Returns:
            List of 3 query variations.
        """
        return [
            f"What is {query}?",
            f"Explain {query} in detail",
            f"Provide specific information about {query}",
        ]

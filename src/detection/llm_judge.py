"""LLM-as-judge hallucination scorer using Groq API.

Uses a structured few-shot prompt to ask the LLM to judge the faithfulness
of a response against its source context. Returns a normalized score,
reasoning, and list of problematic claims.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

JUDGE_PROMPT_TEMPLATE = """You are an expert fact-checker evaluating whether an AI assistant's response is faithful to the provided context.

## Context (Source of Truth):
{context}

## Question:
{query}

## Response to Evaluate:
{response}

## Instructions:
1. Compare every claim in the response against the context.
2. A claim is "faithful" if it is directly supported by or can be reasonably inferred from the context.
3. A claim is "hallucinated" if it contains information not present in or contradicted by the context.

## Few-Shot Examples:

### Example 1 (Faithful):
Context: "The Eiffel Tower was built in 1889 and is 330 meters tall."
Response: "The Eiffel Tower, constructed in 1889, stands at 330 meters."
Judgment: {{"faithfulness_score": 5, "reasoning": "All claims directly supported by context.", "problematic_claims": []}}

### Example 2 (Partially Hallucinated):
Context: "Python was created by Guido van Rossum in 1991."
Response: "Python was created by Guido van Rossum in 1991. It is the most popular programming language in the world."
Judgment: {{"faithfulness_score": 3, "reasoning": "The creation fact is supported, but the popularity claim is not in the context.", "problematic_claims": ["It is the most popular programming language in the world."]}}

### Example 3 (Fully Hallucinated):
Context: "Machine learning uses statistical methods to learn from data."
Response: "Deep learning was invented by Geoffrey Hinton in 2006 at Google Brain."
Judgment: {{"faithfulness_score": 1, "reasoning": "None of the claims are supported by the context. The response discusses different topics.", "problematic_claims": ["Deep learning was invented by Geoffrey Hinton in 2006 at Google Brain."]}}

## Your Judgment:
Rate the faithfulness on a scale of 1-5:
- 1: Completely hallucinated — no claims supported
- 2: Mostly hallucinated — few claims supported
- 3: Mixed — some claims supported, some not
- 4: Mostly faithful — minor unsupported details
- 5: Fully faithful — all claims supported

Respond with ONLY valid JSON in this exact format:
{{"faithfulness_score": <1-5>, "reasoning": "<explanation>", "problematic_claims": ["<claim1>", "<claim2>"]}}"""


class LLMJudge:
    """LLM-as-judge hallucination scorer.

    Uses a structured few-shot prompt to evaluate response faithfulness
    by asking an LLM to act as a fact-checker and identify unsupported claims.

    Args:
        model_name: Groq model identifier for the judge LLM.
        temperature: Sampling temperature (low for consistent judgments).
        max_retries: Number of retry attempts for API and parsing failures.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self._generator = None

        logger.info("LLMJudge initialized (model=%s)", model_name)

    @property
    def generator(self):
        """Lazy-load the LLM generator for judge calls."""
        if self._generator is None:
            from src.rag_pipeline.generator import LLMGenerator
            self._generator = LLMGenerator(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=512,
            )
        return self._generator

    def score(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
    ) -> Dict[str, Any]:
        """Score a response's faithfulness using LLM-as-judge.

        Args:
            query: The original user query.
            response: The response to evaluate.
            context_chunks: Context passages the response should be grounded in.

        Returns:
            Dictionary containing:
                - score (float): Normalized hallucination score 0.0-1.0
                - faithfulness_score (int): Raw 1-5 score from the judge
                - reasoning (str): Judge's explanation
                - problematic_claims (List[str]): Claims not supported by context
        """
        if not response or not response.strip():
            return {
                "score": 1.0,
                "faithfulness_score": 1,
                "reasoning": "Empty response.",
                "problematic_claims": [],
            }

        if not context_chunks:
            return {
                "score": 1.0,
                "faithfulness_score": 1,
                "reasoning": "No context provided — cannot verify faithfulness.",
                "problematic_claims": [],
            }

        context_text = "\n\n".join(
            f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
        )

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            context=context_text,
            query=query,
            response=response,
        )

        # Try to get and parse a valid judgment
        for attempt in range(self.max_retries + 1):
            try:
                raw_result = self.generator.generate(
                    query=prompt,
                    context_chunks=[],
                    system_prompt="You are a precise JSON-outputting fact-checker. Output ONLY valid JSON.",
                )

                parsed = self._parse_judgment(raw_result)
                if parsed:
                    return parsed

                logger.warning(
                    "Failed to parse judgment (attempt %d/%d).",
                    attempt + 1, self.max_retries + 1,
                )

            except Exception as e:
                logger.warning(
                    "LLM judge call failed (attempt %d/%d): %s",
                    attempt + 1, self.max_retries + 1, str(e),
                )

        # Fallback: return neutral score
        logger.error("All LLM judge attempts failed. Returning neutral score.")
        return {
            "score": 0.5,
            "faithfulness_score": 3,
            "reasoning": "Judge evaluation failed — returning neutral score.",
            "problematic_claims": [],
        }

    def _parse_judgment(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Parse the JSON judgment from the LLM response.

        Args:
            raw_response: Raw LLM output containing JSON.

        Returns:
            Parsed judgment dict or None if parsing fails.
        """
        text = raw_response.strip()

        # Strip markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        try:
            data = json.loads(text)

            faithfulness_score = int(data.get("faithfulness_score", 3))
            faithfulness_score = max(1, min(5, faithfulness_score))

            # Normalize to 0.0-1.0 hallucination score
            # faithfulness 5 → hallucination 0.0, faithfulness 1 → hallucination 1.0
            normalized_score = (5 - faithfulness_score) / 4.0

            return {
                "score": round(normalized_score, 4),
                "faithfulness_score": faithfulness_score,
                "reasoning": data.get("reasoning", ""),
                "problematic_claims": data.get("problematic_claims", []),
            }

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug("JSON parse error: %s for text: %s", e, text[:200])
            return None

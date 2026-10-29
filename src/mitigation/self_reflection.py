"""Self-reflection mitigation loop for iterative hallucination correction.

Implements an iterative self-correction process where the LLM identifies
unsupported claims in its own response, rewrites them to be faithful
to the context, and re-checks until the hallucination score drops
below the threshold or maximum iterations are reached.
"""

import logging
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

REFLECTION_PROMPT = """You are a careful assistant revising your own response to remove hallucinations.

## Context (Source of Truth):
{context}

## Original Question:
{query}

## Your Previous Response (may contain hallucinations):
{response}

## Task:
1. Identify any claims in your response that are NOT directly supported by the context.
2. Remove or correct those unsupported claims.
3. Keep all claims that ARE supported by the context.
4. If you cannot answer from the context, say "I cannot find this in the provided documents."
5. Always cite which part of the context supports your answer.

## Revised Response:"""


class SelfReflectionMitigator:
    """Iterative self-reflection loop for hallucination mitigation.

    The LLM identifies unsupported claims in its response, rewrites
    them to be faithful to context, then re-checks with the detector.
    Continues until the score drops below the threshold or max iterations.

    Args:
        max_iterations: Maximum self-reflection iterations (default: 3).
        threshold: Score below which the response is considered faithful.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        threshold: float = 0.5,
    ) -> None:
        self.max_iterations = max_iterations
        self.threshold = threshold

        logger.info(
            "SelfReflectionMitigator initialized (max_iter=%d, threshold=%.2f)",
            max_iterations, threshold,
        )

    def mitigate(
        self,
        query: str,
        hallucinated_response: str,
        context_chunks: List[str],
        detector: Any,
        generator: Any,
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run iterative self-reflection to correct hallucinations.

        Loop:
        1. Ask LLM to identify unsupported claims and rewrite response
        2. Re-score with detector
        3. Stop if score < threshold OR max iterations reached

        Args:
            query: The original user query.
            hallucinated_response: The response identified as hallucinated.
            context_chunks: Context passages for grounding.
            detector: EnsembleDetector instance for re-scoring.
            generator: LLMGenerator instance for rewriting.
            max_iterations: Override default max iterations.

        Returns:
            Dictionary containing:
                - final_response (str): The corrected response
                - iterations (int): Number of iterations performed
                - final_score (float): Final hallucination score
                - improvement (float): Score improvement (initial - final)
                - iteration_history (List[dict]): Per-iteration details
        """
        max_iter = max_iterations or self.max_iterations
        current_response = hallucinated_response
        iteration_history = []

        # Get initial score
        try:
            initial_result = detector.score(
                query=query,
                response=current_response,
                context_chunks=context_chunks,
            )
            initial_score = initial_result.get("final_score", 1.0)
        except Exception as e:
            logger.error("Initial detection failed: %s", e)
            initial_score = 1.0

        current_score = initial_score
        context_text = "\n\n".join(
            f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
        )

        for iteration in range(max_iter):
            logger.info(
                "Self-reflection iteration %d/%d (current score: %.3f)",
                iteration + 1, max_iter, current_score,
            )

            # Step 1: Ask LLM to revise response
            try:
                reflection_prompt = REFLECTION_PROMPT.format(
                    context=context_text,
                    query=query,
                    response=current_response,
                )

                revised_response = generator.generate(
                    query=reflection_prompt,
                    context_chunks=[],
                    system_prompt=(
                        "You are revising your previous response to remove any "
                        "claims not supported by the provided context. Be precise "
                        "and only state what the context supports."
                    ),
                )

                if not revised_response or not revised_response.strip():
                    logger.warning("Empty revision in iteration %d.", iteration + 1)
                    break

            except Exception as e:
                logger.error("Revision generation failed: %s", e)
                break

            # Step 2: Re-score the revised response
            try:
                new_result = detector.score(
                    query=query,
                    response=revised_response,
                    context_chunks=context_chunks,
                )
                new_score = new_result.get("final_score", current_score)
            except Exception as e:
                logger.error("Re-scoring failed: %s", e)
                new_score = current_score

            # Track iteration
            iteration_history.append({
                "iteration": iteration + 1,
                "response": revised_response,
                "score": new_score,
                "improvement": current_score - new_score,
            })

            # Update if improved
            if new_score < current_score:
                current_response = revised_response
                current_score = new_score

            # Step 3: Check if below threshold
            if current_score < self.threshold:
                logger.info(
                    "Score dropped below threshold (%.3f < %.3f) after %d iterations.",
                    current_score, self.threshold, iteration + 1,
                )
                break

        improvement = initial_score - current_score
        result = {
            "final_response": current_response,
            "iterations": len(iteration_history),
            "initial_score": round(initial_score, 4),
            "final_score": round(current_score, 4),
            "improvement": round(improvement, 4),
            "iteration_history": iteration_history,
        }

        logger.info(
            "Self-reflection complete: %d iterations, score %.3f → %.3f (improvement: %.3f)",
            len(iteration_history), initial_score, current_score, improvement,
        )

        return result

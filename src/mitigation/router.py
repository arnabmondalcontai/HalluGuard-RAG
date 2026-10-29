"""Confidence-adaptive routing module for hallucination mitigation.

Routes mitigation to different strategies depending on the hallucination score
returned by the EnsembleDetector. This allows the system to balance latency and
effectiveness by selecting light-weight strategies for low-confidence
hallucinations and more computationally expensive strategies (like self-reflection)
only for high-confidence hallucinations.
"""

import time
from typing import Any, Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceAdaptiveRouter:
    """Confidence-adaptive router for RAG hallucination mitigation.

    Routes hallucination mitigation to a strategy based on the hallucination score
    returned by the EnsembleDetector. This allows the system to balance latency and
    effectiveness by selecting light-weight strategies for low-confidence
    hallucinations and more computationally expensive strategies (like self-reflection)
    only for high-confidence hallucinations.
    """

    def __init__(
        self,
        retriever=None,
        generator=None,
        detector=None,
        query_rewriter=None,
        reranker=None,
        self_reflection=None,
    ) -> None:
        """Initialize the router with its dependencies.

        Args:
            retriever: HybridRetriever instance.
            generator: LLMGenerator instance.
            detector: EnsembleDetector instance.
            query_rewriter: QueryRewriter instance.
            reranker: ChunkReranker instance.
            self_reflection: SelfReflectionMitigator instance.
        """
        self.retriever = retriever
        self.generator = generator
        self.detector = detector
        self._query_rewriter = query_rewriter
        self._reranker = reranker
        self._self_reflection = self_reflection

        logger.info("ConfidenceAdaptiveRouter initialized.")

    @property
    def query_rewriter(self):
        """Lazy-load QueryRewriter."""
        if self._query_rewriter is None:
            from src.mitigation.query_rewriter import QueryRewriter
            self._query_rewriter = QueryRewriter()
        return self._query_rewriter

    @property
    def reranker(self):
        """Lazy-load ChunkReranker."""
        if self._reranker is None:
            from src.mitigation.reranker import ChunkReranker
            self._reranker = ChunkReranker()
        return self._reranker

    @property
    def self_reflection(self):
        """Lazy-load SelfReflectionMitigator."""
        if self._self_reflection is None:
            from src.mitigation.self_reflection import SelfReflectionMitigator
            self._self_reflection = SelfReflectionMitigator()
        return self._self_reflection

    def route_and_mitigate(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
        hallucination_score: float,
        detection_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Alias for route()."""
        return self.route(
            query=query,
            response=response,
            context_chunks=context_chunks,
            hallucination_score=hallucination_score,
            detection_result=detection_result,
        )

    def route(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
        hallucination_score: float,
        detection_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route to the appropriate mitigation tier based on confidence and execute it.

        Args:
            query: The original user query.
            response: The hallucinated response to mitigate.
            context_chunks: Originally retrieved context chunks.
            hallucination_score: Hallucination score from EnsembleDetector (0.0 to 1.0).
            detection_result: Full detection breakdown dict.

        Returns:
            Dictionary containing:
                - strategy_used (str): "none" | "rewrite" | "rerank" | "self_reflection"
                - original_score (float): Score before mitigation
                - final_score (float): Score after mitigation
                - improvement (float): original_score - final_score
                - final_response (str): The mitigated (or original) response text
                - details (dict): Strategy-specific execution logs
        """
        logger.info(
            "ConfidenceAdaptiveRouter evaluating response with score %.3f",
            hallucination_score,
        )

        result = {
            "strategy_used": "none",
            "original_score": hallucination_score,
            "final_score": hallucination_score,
            "improvement": 0.0,
            "final_response": response,
            "details": {},
        }

        # No mitigation needed if score is below threshold
        if hallucination_score <= 0.5:
            logger.info("Hallucination score %.3f <= 0.5. No mitigation applied.", hallucination_score)
            return result

        # Determine strategy based on confidence tiers
        if 0.5 < hallucination_score < 0.65:
            strategy = "rewrite"
            logger.info("Score %.3f in (0.5, 0.65). Routing to query rewriting mitigation.", hallucination_score)
            mitigation_res = self._mitigate_rewrite(query, response, context_chunks, detection_result)

        elif 0.65 <= hallucination_score < 0.80:
            strategy = "rerank"
            logger.info("Score %.3f in [0.65, 0.80]. Routing to chunk reranking mitigation.", hallucination_score)
            mitigation_res = self._mitigate_rerank(query, response, context_chunks)

        else:  # hallucination_score >= 0.80
            strategy = "self_reflection"
            logger.info("Score %.3f >= 0.80. Routing to self-reflection mitigation.", hallucination_score)
            mitigation_res = self._mitigate_self_reflection(query, response, context_chunks)

        # Evaluate if the mitigation improved the score
        new_response = mitigation_res["response"]
        new_score = mitigation_res["score"]
        improvement = hallucination_score - new_score

        result["strategy_used"] = strategy
        result["details"] = mitigation_res.get("details", {})

        # We always return the response that has the lower hallucination score
        if new_score < hallucination_score:
            logger.info(
                "Mitigation strategy '%s' succeeded. Score improved: %.3f → %.3f (improvement: %.3f)",
                strategy, hallucination_score, new_score, improvement,
            )
            result["final_response"] = new_response
            result["final_score"] = new_score
            result["improvement"] = improvement
        else:
            logger.warning(
                "Mitigation strategy '%s' did not improve score. Keeping original response. (Mitigated score: %.3f)",
                strategy, new_score,
            )
            result["final_response"] = response
            result["final_score"] = hallucination_score
            result["improvement"] = 0.0

        return result

    def _mitigate_rewrite(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
        detection_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Low confidence mitigation: rewrite query + re-retrieve + re-generate.

        Args:
            query: The original query.
            response: The hallucinated response.
            context_chunks: Original context chunks.
            detection_result: Detailed detector result.

        Returns:
            Dict containing 'response', 'score', 'details'.
        """
        problematic_claims = None
        if detection_result:
            problematic_claims = detection_result.get("problematic_claims", None)

        try:
            # 1. Rewrite queries
            rewritten_queries = self.query_rewriter.rewrite(
                original_query=query,
                failed_response=response,
                problematic_claims=problematic_claims,
            )
            logger.info("Rewritten queries: %s", rewritten_queries)

            # 2. Re-retrieve with rewritten queries
            all_chunks = list(context_chunks)
            for q in rewritten_queries:
                try:
                    retrieved_docs = self.retriever.retrieve(q)
                    for doc in retrieved_docs:
                        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                        if content not in all_chunks:
                            all_chunks.append(content)
                except Exception as e:
                    logger.error("Failed to retrieve for query '%s': %s", q, e)

            # 3. Rerank to limit to original context size to keep context length reasonable
            original_size = len(context_chunks) or 5
            try:
                final_chunks = self.reranker.rerank(query, all_chunks, top_k=original_size)
            except Exception as e:
                logger.error("Reranking failed in rewrite mitigation: %s", e)
                final_chunks = all_chunks[:original_size]

            # 4. Re-generate response
            new_response = self.generator.generate(query=query, context_chunks=final_chunks)

            # 5. Score new response
            new_detection = self.detector.score(
                query=query,
                response=new_response,
                context_chunks=final_chunks,
            )
            new_score = new_detection.get("final_score", 1.0)

            # Start timing for latency tracking
            start_time = time.perf_counter()

            return {
                "response": new_response,
                "score": new_score,
                "details": {
                    "rewritten_queries": rewritten_queries,
                    "retrieved_chunk_count": len(all_chunks),
                    "final_chunk_count": len(final_chunks),
                    "new_score": new_score,
                },
            }

        except Exception as e:
            logger.error("Rewrite mitigation failed: %s", e, exc_info=True)
            return {"response": response, "score": 1.0, "details": {"error": str(e)}}

    def _mitigate_rerank(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
    ) -> Dict[str, Any]:
        """Medium confidence mitigation: rerank existing chunks + re-generate.

        Args:
            query: The original query.
            response: The hallucinated response.
            context_chunks: Original context chunks.

        Returns:
            Dict containing 'response', 'score', 'details'.
        """
        try:
            # 1. Rerank existing chunks and take top 3
            top_k = min(3, len(context_chunks))
            if not context_chunks:
                logger.warning("No context chunks to rerank.")
                return {"response": response, "score": 1.0, "details": {"error": "No context chunks"}}

            reranked_chunks = self.reranker.rerank(query, context_chunks, top_k=top_k)

            # 2. Re-generate response
            new_response = self.generator.generate(query=query, context_chunks=reranked_chunks)

            # 3. Score new response
            new_detection = self.detector.score(
                query=query,
                response=new_response,
                context_chunks=reranked_chunks,
            )
            new_score = new_detection.get("final_score", 1.0)

            return {
                "response": new_response,
                "score": new_score,
                "details": {
                    "reranked_chunks": reranked_chunks,
                    "new_score": new_score,
                },
            }

        except Exception as e:
            logger.error("Rerank mitigation failed: %s", e, exc_info=True)
            return {"response": response, "score": 1.0, "details": {"error": str(e)}}

    def _mitigate_self_reflection(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
    ) -> Dict[str, Any]:
        """High confidence mitigation: SelfReflectionMitigator.

        Args:
            query: The original query.
            response: The hallucinated response.
            context_chunks: Original context chunks.

        Returns:
            Dict containing 'response', 'score', 'details'.
        """
        try:
            mitigation_res = self.self_reflection.mitigate(
                query=query,
                hallucinated_response=response,
                context_chunks=context_chunks,
                detector=self.detector,
                generator=self.generator,
            )
            return {
                "response": mitigation_res["final_response"],
                "score": mitigation_res["final_score"],
                "details": mitigation_res,
            }
        except Exception as e:
            logger.error("Self-reflection mitigation failed: %s", e, exc_info=True)
            return {"response": response, "score": 1.0, "details": {"error": str(e)}}
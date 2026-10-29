"""Full RAG pipeline orchestration with hallucination detection and mitigation.

Composes the HybridRetriever, LLMGenerator, and HallucinationDetector
into a unified pipeline that retrieves context, generates responses,
detects hallucinations, and automatically applies mitigation when needed.
"""

import time
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline with hallucination detection and mitigation.

    Orchestrates the full workflow: query → retrieve → generate → detect →
    mitigate (if needed) → return structured results.

    Args:
        retriever: HybridRetriever instance for document retrieval.
        generator: LLMGenerator instance for response generation.
        detector: EnsembleDetector instance for hallucination scoring.
        mitigator: SelfReflectionMitigator instance for hallucination correction.
        hallucination_threshold: Score above which mitigation is triggered (default: 0.5).
        top_k: Number of context chunks to retrieve (default: 5).
    """

    def __init__(
        self,
        retriever=None,
        generator=None,
        detector=None,
        mitigator=None,
        router=None,
        hallucination_threshold: float = 0.5,
        top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._detector = detector
        self._mitigator = mitigator
        self._router = router
        self.hallucination_threshold = hallucination_threshold
        self.top_k = top_k

        logger.info(
            "RAGPipeline initialized (threshold=%.2f, top_k=%d)",
            hallucination_threshold, top_k,
        )

    @property
    def retriever(self):
        """Lazy-load the HybridRetriever."""
        if self._retriever is None:
            from src.rag_pipeline.retriever import HybridRetriever
            self._retriever = HybridRetriever()
        return self._retriever

    @property
    def generator(self):
        """Lazy-load the LLMGenerator."""
        if self._generator is None:
            from src.rag_pipeline.generator import LLMGenerator
            self._generator = LLMGenerator()
        return self._generator

    @property
    def detector(self):
        """Lazy-load the EnsembleDetector."""
        if self._detector is None:
            from src.detection.ensemble import EnsembleDetector
            self._detector = EnsembleDetector()
        return self._detector

    @property
    def mitigator(self):
        """Lazy-load the SelfReflectionMitigator."""
        if self._mitigator is None:
            from src.mitigation.self_reflection import SelfReflectionMitigator
            self._mitigator = SelfReflectionMitigator()
        return self._mitigator

    @property
    def router(self):
        """Lazy-load the ConfidenceAdaptiveRouter."""
        if self._router is None:
            from src.mitigation.router import ConfidenceAdaptiveRouter
            self._router = ConfidenceAdaptiveRouter(
                retriever=self.retriever,
                generator=self.generator,
                detector=self.detector,
                self_reflection=self.mitigator,
            )
        return self._router

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the full RAG pipeline with hallucination detection and mitigation.

        Workflow:
        1. Retrieve relevant context chunks
        2. Generate response grounded in context
        3. Score response for hallucination
        4. If hallucination detected (score > threshold), apply mitigation
        5. Return structured results

        Args:
            query: The user's question or instruction.

        Returns:
            Dictionary containing:
                - query (str): Original query
                - retrieved_chunks (List[str]): Context chunks used
                - raw_response (str): Initial LLM response
                - hallucination_score (float): 0.0=faithful, 1.0=hallucinated
                - hallucination_detected (bool): Whether score exceeds threshold
                - final_response (str): Mitigated response if needed
                - mitigation_applied (str): "none" | "rewrite" | "self_reflection"
                - latency_ms (float): Total pipeline execution time
        """
        start_time = time.time()

        result = {
            "query": query,
            "retrieved_chunks": [],
            "raw_response": "",
            "hallucination_score": 0.0,
            "hallucination_detected": False,
            "final_response": "",
            "mitigation_applied": "none",
            "latency_ms": 0.0,
        }

        try:
            # Step 1: Retrieve context
            logger.info("Step 1/4: Retrieving context for query...")
            documents = self.retriever.retrieve(query, top_k=self.top_k)
            context_chunks = [doc.page_content for doc in documents]
            result["retrieved_chunks"] = context_chunks

            if not context_chunks:
                logger.warning("No context chunks retrieved. Generating without context.")

            # Step 2: Generate response
            logger.info("Step 2/4: Generating response...")
            raw_response = self.generator.generate(
                query=query, context_chunks=context_chunks
            )
            result["raw_response"] = raw_response
            result["final_response"] = raw_response

            # Step 3: Detect hallucination
            logger.info("Step 3/4: Detecting hallucination...")
            detection_result = self.detector.score(
                query=query,
                response=raw_response,
                context_chunks=context_chunks,
            )

            hallucination_score = detection_result.get("final_score", 0.0)
            result["hallucination_score"] = hallucination_score
            result["hallucination_detected"] = (
                hallucination_score > self.hallucination_threshold
            )
            result["detection_breakdown"] = detection_result

            # Step 4: Mitigate if hallucination detected
            if result["hallucination_detected"]:
                logger.info(
                    "Step 4/4: Hallucination detected (score=%.3f). Applying mitigation via router...",
                    hallucination_score,
                )
                router_result = self.router.route_and_mitigate(
                    query=query,
                    response=raw_response,
                    context_chunks=context_chunks,
                    hallucination_score=hallucination_score,
                    detection_result=detection_result,
                )
                result["final_response"] = router_result["final_response"]
                result["mitigation_applied"] = router_result["strategy_used"]
                result["mitigation_details"] = router_result
            else:
                logger.info(
                    "Step 4/4: No hallucination detected (score=%.3f). Skipping mitigation.",
                    hallucination_score,
                )

        except Exception as e:
            logger.error("Pipeline error: %s", str(e), exc_info=True)
            result["error"] = str(e)
            if not result["final_response"]:
                result["final_response"] = (
                    "An error occurred while processing your query. "
                    "Please try again."
                )

        # Track latency
        result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        logger.info("Pipeline completed in %.2f ms.", result["latency_ms"])

        return result

    def run_detection_only(
        self,
        query: str,
        response: str,
        context_chunks: List[str],
    ) -> Dict[str, Any]:
        """Run only the hallucination detection step (no retrieval or generation).

        Useful for the /detect API endpoint and benchmarking.

        Args:
            query: The original query.
            response: The response to evaluate.
            context_chunks: The context chunks the response should be grounded in.

        Returns:
            Dictionary with hallucination_score, is_hallucinated, and breakdown.
        """
        start_time = time.time()

        try:
            detection_result = self.detector.score(
                query=query,
                response=response,
                context_chunks=context_chunks,
            )

            return {
                "hallucination_score": detection_result.get("final_score", 0.0),
                "is_hallucinated": detection_result.get("is_hallucinated", False),
                "breakdown": detection_result,
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

        except Exception as e:
            logger.error("Detection error: %s", str(e), exc_info=True)
            return {
                "hallucination_score": 0.5,
                "is_hallucinated": False,
                "breakdown": {"error": str(e)},
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }

    def add_documents(
        self,
        docs: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """Add documents to the retriever's index.

        Args:
            docs: List of document text strings.
            metadatas: Optional metadata dicts for each document.
        """
        self.retriever.add_documents(docs, metadatas)

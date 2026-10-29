"""Tests for the RAG pipeline orchestration.

Validates pipeline output schema, mitigation triggering, and latency tracking.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestRAGPipeline:
    """Test suite for the RAGPipeline."""

    def _make_pipeline_with_mocks(
        self, hallucination_score=0.3, mitigation_score=0.2
    ):
        """Create a pipeline with fully mocked components."""
        from src.rag_pipeline.pipeline import RAGPipeline
        from src.rag_pipeline.retriever import Document

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            Document(
                page_content="Test context chunk.",
                metadata={"source": "test", "chunk_id": 0, "score": 0.9},
            )
        ]

        # Mock generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Test response based on context."

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.score.return_value = {
            "final_score": hallucination_score,
            "nli_score": hallucination_score,
            "embedding_score": hallucination_score,
            "llm_judge_score": hallucination_score,
            "is_hallucinated": hallucination_score > 0.5,
            "threshold": 0.5,
            "reasoning": "",
            "problematic_claims": [],
        }

        # Mock mitigator
        mock_mitigator = MagicMock()
        mock_mitigator.mitigate.return_value = {
            "final_response": "Corrected response.",
            "iterations": 2,
            "initial_score": hallucination_score,
            "final_score": mitigation_score,
            "improvement": hallucination_score - mitigation_score,
        }

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
            detector=mock_detector,
            mitigator=mock_mitigator,
        )

        return pipeline, mock_mitigator

    def test_pipeline_returns_required_keys(self):
        """run() output must have all required keys."""
        pipeline, _ = self._make_pipeline_with_mocks(hallucination_score=0.3)

        result = pipeline.run("What is Python?")

        required_keys = {
            "query",
            "retrieved_chunks",
            "raw_response",
            "hallucination_score",
            "hallucination_detected",
            "final_response",
            "mitigation_applied",
            "latency_ms",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_pipeline_output_types(self):
        """Verify output value types are correct."""
        pipeline, _ = self._make_pipeline_with_mocks()
        result = pipeline.run("Test query")

        assert isinstance(result["query"], str)
        assert isinstance(result["retrieved_chunks"], list)
        assert isinstance(result["raw_response"], str)
        assert isinstance(result["hallucination_score"], float)
        assert isinstance(result["hallucination_detected"], bool)
        assert isinstance(result["final_response"], str)
        assert isinstance(result["mitigation_applied"], str)
        assert isinstance(result["latency_ms"], float)

    def test_router_triggers_on_high_score(self):
        """Mock detector to return 0.9, verify SelfReflectionMitigator is called."""
        from src.rag_pipeline.pipeline import RAGPipeline
        from src.mitigation.router import ConfidenceAdaptiveRouter

        # Mocks
        mock_retriever = MagicMock()
        mock_generator = MagicMock()
        mock_detector = MagicMock()
        mock_detector.score.return_value = {
            "final_score": 0.9,
            "is_hallucinated": True,
        }

        mock_self_reflection = MagicMock()
        mock_self_reflection.mitigate.return_value = {
            "final_response": "Corrected response via reflection.",
            "final_score": 0.2,
            "improvement": 0.7,
        }

        mock_rewriter = MagicMock()
        mock_reranker = MagicMock()

        router = ConfidenceAdaptiveRouter(
            retriever=mock_retriever,
            generator=mock_generator,
            detector=mock_detector,
            query_rewriter=mock_rewriter,
            reranker=mock_reranker,
            self_reflection=mock_self_reflection,
        )

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
            detector=mock_detector,
            router=router,
        )

        result = pipeline.run("What is Python?")

        assert result["hallucination_detected"] is True
        assert result["mitigation_applied"] == "self_reflection"
        mock_self_reflection.mitigate.assert_called_once()
        mock_rewriter.rewrite.assert_not_called()
        mock_reranker.rerank.assert_not_called()

    def test_router_tier_selection(self):
        """Verify scores of 0.55, 0.7, 0.85 route to the correct strategy each."""
        from src.mitigation.router import ConfidenceAdaptiveRouter

        # Mock dependencies
        mock_retriever = MagicMock()
        mock_generator = MagicMock()
        mock_detector = MagicMock()
        mock_rewriter = MagicMock()
        mock_reranker = MagicMock()
        mock_self_reflection = MagicMock()

        # Set up default return values for mocks
        mock_rewriter.rewrite.return_value = ["q1", "q2", "q3"]
        mock_retriever.retrieve.return_value = []
        mock_reranker.rerank.return_value = ["c1", "c2"]
        mock_generator.generate.return_value = "Mitigated response."
        mock_detector.score.return_value = {"final_score": 0.3}
        mock_self_reflection.mitigate.return_value = {
            "final_response": "Reflection response.",
            "final_score": 0.2,
            "improvement": 0.65,
        }

        router = ConfidenceAdaptiveRouter(
            retriever=mock_retriever,
            generator=mock_generator,
            detector=mock_detector,
            query_rewriter=mock_rewriter,
            reranker=mock_reranker,
            self_reflection=mock_self_reflection,
        )

        # 1. Test score 0.55 -> rewrite
        res_rewrite = router.route_and_mitigate(
            query="test query",
            response="hallucinated response",
            context_chunks=["chunk1"],
            hallucination_score=0.55,
        )
        assert res_rewrite["strategy_used"] == "rewrite"
        mock_rewriter.rewrite.assert_called_once()
        mock_self_reflection.mitigate.assert_not_called()

        # Reset mocks
        mock_rewriter.rewrite.reset_mock()
        mock_self_reflection.mitigate.reset_mock()
        mock_reranker.rerank.reset_mock()

        # 2. Test score 0.7 -> rerank
        res_rerank = router.route_and_mitigate(
            query="test query",
            response="hallucinated response",
            context_chunks=["chunk1"],
            hallucination_score=0.7,
        )
        assert res_rerank["strategy_used"] == "rerank"
        mock_reranker.rerank.assert_called_once()
        mock_rewriter.rewrite.assert_not_called()
        mock_self_reflection.mitigate.assert_not_called()

        # Reset mocks
        mock_reranker.rerank.reset_mock()

        # 3. Test score 0.85 -> self_reflection
        res_reflection = router.route_and_mitigate(
            query="test query",
            response="hallucinated response",
            context_chunks=["chunk1"],
            hallucination_score=0.85,
        )
        assert res_reflection["strategy_used"] == "self_reflection"
        mock_self_reflection.mitigate.assert_called_once()
        mock_rewriter.rewrite.assert_not_called()

    def test_latency_tracked(self):
        """Verify latency_ms > 0 in output."""
        pipeline, _ = self._make_pipeline_with_mocks()

        result = pipeline.run("Test query")

        assert result["latency_ms"] > 0

    def test_detection_only_mode(self):
        """Test run_detection_only returns expected schema."""
        pipeline, _ = self._make_pipeline_with_mocks(hallucination_score=0.4)

        result = pipeline.run_detection_only(
            query="test",
            response="test response",
            context_chunks=["test context"],
        )

        assert "hallucination_score" in result
        assert "is_hallucinated" in result
        assert "breakdown" in result
        assert "latency_ms" in result

    def test_pipeline_handles_errors_gracefully(self):
        """Pipeline should not crash on component failures."""
        from src.rag_pipeline.pipeline import RAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("Retrieval failed")

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            generator=MagicMock(),
            detector=MagicMock(),
            mitigator=MagicMock(),
        )

        result = pipeline.run("test query")

        # Should return a result dict even on error
        assert "error" in result
        assert result["latency_ms"] > 0

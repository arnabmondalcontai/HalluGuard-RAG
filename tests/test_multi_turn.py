"""Tests for Multi-Turn Hallucination Tracking Pipeline.

Tests consistency tracking across conversation turns, contradiction detection,
history resetting, and composite score computation.
"""

from unittest.mock import MagicMock
import pytest

from src.rag_pipeline.multi_turn_pipeline import MultiTurnRAGPipeline


def test_single_turn_no_history():
    """First turn should have turn_consistency_score = 1.0, and no conflicts possible on turn 1."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ["Context chunk about Albert Einstein."]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Albert Einstein won the Nobel Prize in 1921."

    mock_detector = MagicMock()
    mock_detector.score.return_value = {"final_score": 0.1, "is_hallucinated": False}

    pipeline = MultiTurnRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        detector=mock_detector,
    )

    result = pipeline.run_turn("Who won the Nobel Prize in 1921?")

    assert result["turn_id"] == 1
    assert result["turn_consistency_score"] == 1.0
    assert result["entity_conflicts"] == []
    assert result["hallucination_score"] == 0.1
    assert result["conversation_length"] == 1


def test_consistent_turns_high_score():
    """Turn 1 and Turn 2 maintain consistent entity mentions (Einstein, 1921)."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ["Context chunk."]

    mock_generator = MagicMock()
    # Turn 1
    mock_generator.generate.side_effect = [
        "Einstein won the Nobel Prize in 1921.",
        "Einstein was awarded the Nobel Prize in 1921 for his work.",
    ]

    mock_detector = MagicMock()
    mock_detector.score.return_value = {"final_score": 0.1, "is_hallucinated": False}

    pipeline = MultiTurnRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        detector=mock_detector,
    )

    turn1 = pipeline.run_turn("Tell me about Einstein's prize.")
    turn2 = pipeline.run_turn("Can you confirm the year?")

    assert turn2["turn_id"] == 2
    assert turn2["turn_consistency_score"] >= 0.8
    assert turn2["entity_conflicts"] == []


def test_contradicting_entity_lowers_score():
    """Turn 1: 'Einstein won Nobel Prize in 1921', Turn 2: 'Einstein won Nobel Prize in 1925'."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ["Context chunk."]

    mock_generator = MagicMock()
    mock_generator.generate.side_effect = [
        "Einstein won Nobel Prize in 1921",
        "Einstein won Nobel Prize in 1925",
    ]

    mock_detector = MagicMock()
    mock_detector.score.return_value = {"final_score": 0.1, "is_hallucinated": False}

    pipeline = MultiTurnRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        detector=mock_detector,
    )

    turn1 = pipeline.run_turn("When did Einstein win the Nobel Prize?")
    turn2 = pipeline.run_turn("Are you sure about the year?")

    assert turn2["turn_id"] == 2
    assert turn2["turn_consistency_score"] < 0.5
    assert len(turn2["entity_conflicts"]) > 0


def test_reset_clears_everything():
    """Run 3 turns then call reset(); verify history, memory and counter are cleared."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ["Context."]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Einstein won in 1921."

    mock_detector = MagicMock()
    mock_detector.score.return_value = {"final_score": 0.1, "is_hallucinated": False}

    pipeline = MultiTurnRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        detector=mock_detector,
    )

    pipeline.run_turn("Turn 1")
    pipeline.run_turn("Turn 2")
    pipeline.run_turn("Turn 3")

    assert len(pipeline.conversation_history) == 3
    assert pipeline.turn_counter == 3
    assert len(pipeline.entity_memory) > 0

    pipeline.reset()

    assert pipeline.conversation_history == []
    assert pipeline.entity_memory == {}
    assert pipeline.turn_counter == 0

    summary = pipeline.get_conversation_summary()
    assert summary["total_turns"] == 0
    assert summary["mean_hallucination_score"] == 0.0


def test_combined_score_above_threshold():
    """Mock ensemble_score = 0.4, mock turn_consistency_score = 0.0 -> combined = 0.7*0.4 + 0.3*1.0 = 0.58 > 0.5."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ["Context."]

    mock_generator = MagicMock()
    mock_generator.generate.side_effect = [
        "Einstein won in 1921.",
        "Einstein won in 1925.",
    ]

    mock_detector = MagicMock()
    mock_detector.score.return_value = {"final_score": 0.4, "is_hallucinated": False}

    mock_router = MagicMock()
    mock_router.route_and_mitigate.return_value = {
        "final_response": "Corrected: Einstein won in 1921.",
        "strategy_used": "self_reflection",
    }

    pipeline = MultiTurnRAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        detector=mock_detector,
        router=mock_router,
    )

    # Turn 1
    pipeline.run_turn("Turn 1")

    # Turn 2: override _check_entity_consistency to return 0.0 consistency
    pipeline._check_entity_consistency = MagicMock(return_value=(0.0, ["Contradiction detected"]))

    turn2 = pipeline.run_turn("Turn 2")

    # expected final_score: 0.7 * 0.4 + 0.3 * (1 - 0.0) = 0.28 + 0.30 = 0.58
    assert pytest.approx(turn2["hallucination_score"], 0.01) == 0.58
    assert turn2["is_hallucinated"] is True
    assert turn2["hallucination_detected"] is True
    mock_router.route_and_mitigate.assert_called_once()

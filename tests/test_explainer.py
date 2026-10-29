"""Tests for the claim-level explanation module.

Validates per-sentence scoring, hallucination explanation generation,
highlight formatting, and summary statistics.
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.detection.explainer import ExplainabilityModule


@pytest.fixture
def mock_explainer():
    """Create an ExplainabilityModule with mocked generator and NLI detector."""
    mock_generator = MagicMock()
    mock_nli = MagicMock()

    # Default generator behavior: return explanation for any prompt
    def mock_generate(query, context_chunks=None, **kwargs):
        return "The claim contradicts the context which explicitly states Paris is the capital."

    mock_generator.generate.side_effect = mock_generate

    # Default NLI score behavior based on content
    def mock_score(sentence, context_chunks):
        premise = " ".join(context_chunks)
        # If sentence contains contradiction keywords, score high (hallucinated)
        if "quantum" in sentence or "London" in sentence or "Mars" in sentence:
            return 0.85
        return 0.15

    mock_nli.score.side_effect = mock_score

    return ExplainabilityModule(generator=mock_generator, nli_detector=mock_nli)


def test_faithful_sentence_no_explanation(mock_explainer):
    """Clearly supported response: all is_hallucinated=False, explanations empty."""
    context = ["The Eiffel Tower was built in 1889 and is located in Paris, France."]
    response = "The Eiffel Tower was built in 1889. It is in Paris."

    results = mock_explainer.explain(response, context)

    assert len(results) >= 1
    for item in results:
        assert item["is_hallucinated"] is False
        assert item["explanation"] == ""
        assert item["score"] <= mock_explainer.THRESHOLD


def test_hallucinated_sentence_gets_explanation(mock_explainer):
    """Response contradicting context: at least one is_hallucinated=True with non-empty explanation."""
    context = ["The Eiffel Tower was built in 1889 and is located in Paris, France."]
    response = (
        "The Eiffel Tower is located in Paris. "
        "It is powered entirely by quantum computing hardware on Mars."
    )

    results = mock_explainer.explain(response, context)

    hallucinated_items = [r for r in results if r["is_hallucinated"]]
    assert len(hallucinated_items) >= 1

    for item in hallucinated_items:
        assert item["is_hallucinated"] is True
        assert len(item["explanation"]) > 0
        assert item["score"] > mock_explainer.THRESHOLD
        assert item["supporting_chunk"] != ""


def test_highlight_marks_hallucinations(mock_explainer):
    """Known hallucinated response should have [HALLUCINATED: in highlighted text."""
    context = ["The Eiffel Tower was built in 1889 in Paris."]
    response = "The Eiffel Tower was built in 1889. It is located in London on Mars."

    results = mock_explainer.explain(response, context)
    highlighted = mock_explainer.highlight_response(response, results)

    assert "[HALLUCINATED:" in highlighted
    assert "| Reason:" in highlighted
    assert "The Eiffel Tower was built in 1889." in highlighted


def test_summary_counts_correct(mock_explainer):
    """Summary counts and rates match individual sentence results."""
    context = ["Paris is the capital of France. Python was created by Guido in 1991."]
    response = "Paris is the capital of France. Python runs on quantum chips."

    results = mock_explainer.explain(response, context)
    summary = mock_explainer.summary(results)

    total = summary["total_sentences"]
    hallucinated = summary["hallucinated_count"]
    faithful = summary["faithful_count"]
    rate = summary["hallucination_rate"]

    assert total == len(results)
    assert total == hallucinated + faithful
    assert rate == round(hallucinated / total, 4)
    assert len(summary["explanations"]) == hallucinated

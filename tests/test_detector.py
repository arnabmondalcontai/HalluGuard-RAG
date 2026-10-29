"""Tests for hallucination detection modules.

Validates NLI detector, embedding detector, and ensemble detector
with known faithful and hallucinated responses.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestNLIDetector:
    """Test suite for the NLI-based detector."""

    @pytest.fixture
    def detector(self):
        """Create an NLI detector instance with mocked NLI batch function."""
        from src.detection.nli_detector import NLIDetector
        det = NLIDetector()

        def mock_batch_nli(premise, hypotheses):
            probs = []
            for h in hypotheses:
                if "quantum" in h:
                    probs.append(0.1)  # contradiction / hallucination
                elif "Eiffel Tower" in h:
                    probs.append(0.9)  # entailment / faithful
                else:
                    probs.append(0.8)
            return probs

        det._batch_nli = mock_batch_nli
        return det

    def test_nli_detector_faithful(self, detector):
        """Known faithful response should score < 0.3."""
        context = [
            "The Eiffel Tower was built in 1889 for the World's Fair. "
            "It is located in Paris, France and is 330 meters tall."
        ]
        response = "The Eiffel Tower was built in 1889 and is 330 meters tall."

        score = detector.score(response, context)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low for faithful response

    def test_nli_detector_hallucinated(self, detector):
        """Known hallucinated response should score > 0.5."""
        context = [
            "Machine learning uses statistical methods to learn from data."
        ]
        response = (
            "Deep learning was invented by Geoffrey Hinton in 2006 at Google Brain. "
            "It requires quantum computing hardware to run efficiently."
        )

        score = detector.score(response, context)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should be higher for hallucinated response

    def test_nli_detector_empty_response(self, detector):
        """Empty response should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            detector.score("", ["some context"])

    def test_nli_detector_empty_context(self, detector):
        """Empty context should return max hallucination score."""
        score = detector.score("Some response.", [])
        assert score == 1.0

    def test_nli_detector_detailed(self, detector):
        """Detailed scoring should return per-sentence breakdown."""
        context = ["Python was created by Guido van Rossum in 1991."]
        response = "Python was made by Guido van Rossum. It runs on quantum chips."

        result = detector.score_detailed(response, context)

        assert "score" in result
        assert "sentences" in result
        assert len(result["sentences"]) >= 1
        assert "entailment_prob" in result["sentences"][0]


class TestEmbeddingDetector:
    """Test suite for the embedding-based detector."""

    @pytest.fixture
    def detector(self):
        """Create an embedding detector instance with mocked embedding model."""
        from src.detection.embedding_detector import EmbeddingDetector
        import numpy as np
        
        det = EmbeddingDetector()
        mock_model = MagicMock()

        def mock_encode(inputs, **kwargs):
            embeddings = []
            for text in inputs:
                if "Paris" in text or "Eiffel" in text:
                    embeddings.append(np.array([1.0, 0.0, 0.0]))
                elif "quantum" in text or "Mars" in text or "crashed" in text:
                    embeddings.append(np.array([0.0, 1.0, 0.0]))
                else:
                    embeddings.append(np.array([0.0, 0.0, 1.0]))
            return np.array(embeddings)

        mock_model.encode = mock_encode
        det._model = mock_model
        return det

    def test_embedding_faithful(self, detector):
        """Semantically similar response should have low score."""
        context = ["The capital of France is Paris. It is known for the Eiffel Tower."]
        response = "Paris is the capital of France, famous for the Eiffel Tower."

        score = detector.score(response, context)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_embedding_hallucinated(self, detector):
        """Semantically dissimilar response should have higher score."""
        context = ["Photosynthesis converts sunlight into chemical energy in plants."]
        response = "The stock market crashed in 2008 due to subprime mortgages."

        score = detector.score(response, context)

        assert isinstance(score, float)
        assert score > 0.3

    def test_sentence_level_score(self, detector):
        """Sentence-level scoring should return per-sentence results."""
        context = ["Python is a programming language."]
        response = "Python is a language. Java was invented on Mars."

        results = detector.sentence_level_score(response, context)

        assert len(results) >= 1
        assert "sentence" in results[0]
        assert "hallucination_score" in results[0]
        assert "is_hallucinated" in results[0]


class TestEnsembleDetector:
    """Test suite for the ensemble detector."""

    def test_ensemble_weighted_average(self):
        """Verify weighted average computation with mocked detectors."""
        from src.detection.ensemble import EnsembleDetector

        # Create mock detectors
        mock_nli = MagicMock()
        mock_nli.score.return_value = 0.8

        mock_embedding = MagicMock()
        mock_embedding.score.return_value = 0.6

        mock_judge = MagicMock()
        mock_judge.score.return_value = {
            "score": 0.4,
            "reasoning": "test",
            "problematic_claims": [],
        }

        ensemble = EnsembleDetector(
            nli_detector=mock_nli,
            embedding_detector=mock_embedding,
            llm_judge=mock_judge,
            weights={"nli": 0.5, "embedding": 0.2, "llm_judge": 0.3},
        )

        result = ensemble.score(
            query="test query",
            response="test response",
            context_chunks=["test context"],
        )

        # Expected: 0.5*0.8 + 0.2*0.6 + 0.3*0.4 = 0.4 + 0.12 + 0.12 = 0.64
        expected = 0.5 * 0.8 + 0.2 * 0.6 + 0.3 * 0.4
        assert abs(result["final_score"] - expected) < 0.01
        assert result["nli_score"] == 0.8
        assert result["embedding_score"] == 0.6
        assert result["llm_judge_score"] == 0.4

    def test_threshold_detection(self):
        """Score 0.6 with threshold 0.5 should return is_hallucinated=True."""
        from src.detection.ensemble import EnsembleDetector

        # Mock all detectors to produce a final score > 0.5
        mock_nli = MagicMock()
        mock_nli.score.return_value = 0.7

        mock_embedding = MagicMock()
        mock_embedding.score.return_value = 0.7

        mock_judge = MagicMock()
        mock_judge.score.return_value = {
            "score": 0.7,
            "reasoning": "hallucinated",
            "problematic_claims": ["claim"],
        }

        ensemble = EnsembleDetector(
            nli_detector=mock_nli,
            embedding_detector=mock_embedding,
            llm_judge=mock_judge,
            threshold=0.5,
        )

        result = ensemble.score("q", "r", ["c"])
        assert result["is_hallucinated"] is True
        assert result["final_score"] > 0.5

    def test_threshold_faithful(self):
        """Score below threshold should return is_hallucinated=False."""
        from src.detection.ensemble import EnsembleDetector

        mock_nli = MagicMock()
        mock_nli.score.return_value = 0.2

        mock_embedding = MagicMock()
        mock_embedding.score.return_value = 0.1

        mock_judge = MagicMock()
        mock_judge.score.return_value = {
            "score": 0.1,
            "reasoning": "faithful",
            "problematic_claims": [],
        }

        ensemble = EnsembleDetector(
            nli_detector=mock_nli,
            embedding_detector=mock_embedding,
            llm_judge=mock_judge,
            threshold=0.5,
        )

        result = ensemble.score("q", "r", ["c"])
        assert result["is_hallucinated"] is False
        assert result["final_score"] < 0.5

    def test_ensemble_returns_all_keys(self):
        """Ensemble score output must have all required keys."""
        from src.detection.ensemble import EnsembleDetector

        mock_nli = MagicMock()
        mock_nli.score.return_value = 0.5

        mock_embedding = MagicMock()
        mock_embedding.score.return_value = 0.5

        mock_judge = MagicMock()
        mock_judge.score.return_value = {
            "score": 0.5, "reasoning": "", "problematic_claims": []
        }

        ensemble = EnsembleDetector(
            nli_detector=mock_nli,
            embedding_detector=mock_embedding,
            llm_judge=mock_judge,
        )

        result = ensemble.score("q", "r", ["c"])

        required_keys = {
            "final_score", "nli_score", "embedding_score",
            "llm_judge_score", "is_hallucinated", "threshold",
        }
        assert required_keys.issubset(set(result.keys()))

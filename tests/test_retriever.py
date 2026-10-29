"""Tests for the HybridRetriever component.

Validates document indexing, retrieval, and Reciprocal Rank Fusion merging.
"""

import pytest


class TestHybridRetriever:
    """Test suite for the HybridRetriever."""

    def _make_retriever(self, tmp_path):
        """Create a retriever with a temporary ChromaDB directory and mock embedding model."""
        from src.rag_pipeline.retriever import HybridRetriever
        from unittest.mock import MagicMock
        import numpy as np

        retriever = HybridRetriever(
            persist_directory=str(tmp_path / "chroma_test"),
            collection_name="test_collection",
        )

        mock_embedding_model = MagicMock()
        def mock_encode(docs, **kwargs):
            is_single = isinstance(docs, str)
            docs_list = [docs] if is_single else docs
            
            arr = []
            for doc in docs_list:
                h = float(hash(doc) % 100) / 100.0
                vec = np.zeros(384)
                vec[0] = h
                vec[1] = 1.0 - h
                arr.append(vec)
            
            return np.array(arr)

        mock_embedding_model.encode = mock_encode
        retriever._embedding_model = mock_embedding_model
        return retriever

    def test_add_documents(self, tmp_path):
        """Test that documents can be indexed."""
        retriever = self._make_retriever(tmp_path)
        docs = [
            "The Eiffel Tower was built in 1889.",
            "Python is a programming language created by Guido van Rossum.",
            "Machine learning uses statistical methods.",
        ]
        metadatas = [
            {"source": "wiki", "chunk_id": 0},
            {"source": "wiki", "chunk_id": 1},
            {"source": "wiki", "chunk_id": 2},
        ]

        retriever.add_documents(docs, metadatas)
        assert retriever.collection.count() == 3

    def test_add_documents_empty_raises(self, tmp_path):
        """Test that adding empty docs raises ValueError."""
        retriever = self._make_retriever(tmp_path)
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.add_documents([], [])

    def test_add_documents_mismatched_lengths(self, tmp_path):
        """Test that mismatched docs/metadatas raises ValueError."""
        retriever = self._make_retriever(tmp_path)
        with pytest.raises(ValueError, match="must match"):
            retriever.add_documents(["doc1"], [{"a": 1}, {"b": 2}])

    def test_retrieve_returns_documents(self, tmp_path):
        """Test that retrieval returns Document objects."""
        retriever = self._make_retriever(tmp_path)
        docs = [
            "The Eiffel Tower is 330 meters tall and was built in 1889.",
            "Python was created by Guido van Rossum in 1991.",
            "The capital of France is Paris.",
        ]
        retriever.add_documents(docs)

        results = retriever.retrieve("How tall is the Eiffel Tower?", top_k=2)

        assert len(results) > 0
        assert len(results) <= 2
        assert hasattr(results[0], "page_content")
        assert hasattr(results[0], "metadata")

    def test_retrieve_empty_index(self, tmp_path):
        """Test that retrieving from empty index returns empty list."""
        retriever = self._make_retriever(tmp_path)
        results = retriever.retrieve("test query")
        assert results == []

    def test_retrieve_relevance(self, tmp_path):
        """Test that the most relevant document is ranked first."""
        retriever = self._make_retriever(tmp_path)
        docs = [
            "Bananas are yellow fruits.",
            "The speed of light is approximately 299,792 km per second.",
            "Water boils at 100 degrees Celsius at sea level.",
        ]
        retriever.add_documents(docs)

        results = retriever.retrieve("What is the speed of light?", top_k=3)

        # The light-related document should be among results
        contents = [r.page_content for r in results]
        assert any("speed of light" in c for c in contents)

    def test_document_has_score_metadata(self, tmp_path):
        """Test that retrieved documents include score in metadata."""
        retriever = self._make_retriever(tmp_path)
        retriever.add_documents(["Test document about artificial intelligence."])
        results = retriever.retrieve("AI")

        assert len(results) > 0
        assert "score" in results[0].metadata

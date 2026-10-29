"""Hybrid retriever combining dense and sparse retrieval with Reciprocal Rank Fusion.

Implements a HybridRetriever that merges results from a dense retriever
(sentence-transformers + ChromaDB) and a sparse retriever (BM25) using
Reciprocal Rank Fusion (RRF) for robust document retrieval.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Represents a retrieved document chunk.

    Attributes:
        page_content: The text content of the document chunk.
        metadata: Dictionary containing source, chunk_id, score, and other info.
    """

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Hybrid retriever combining dense (embedding) and sparse (BM25) retrieval.

    Uses sentence-transformers for dense embeddings stored in ChromaDB,
    and BM25 for sparse term-matching retrieval. Results are merged using
    Reciprocal Rank Fusion (RRF).

    Args:
        dense_model_name: Sentence-transformer model for dense embeddings.
        persist_directory: Directory to persist ChromaDB data.
        collection_name: Name of the ChromaDB collection.
        bm25_weight: Weight for BM25 scores in RRF fusion (default: 0.3).
        dense_weight: Weight for dense scores in RRF fusion (default: 0.7).
        rrf_k: RRF constant to prevent division by zero (default: 60).
    """

    def __init__(
        self,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        collection_name: str = "halluguard_docs",
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        self.dense_model_name = dense_model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        # Lazy-loaded components
        self._embedding_model = None
        self._chroma_client = None
        self._collection = None
        self._bm25 = None
        self._corpus: List[str] = []
        self._corpus_metadatas: List[Dict[str, Any]] = []

        logger.info(
            "HybridRetriever initialized (dense=%s, bm25_weight=%.2f, dense_weight=%.2f)",
            dense_model_name, bm25_weight, dense_weight,
        )

    @property
    def embedding_model(self):
        """Lazy-load the sentence transformer embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.dense_model_name)
            self._embedding_model = SentenceTransformer(self.dense_model_name)
        return self._embedding_model

    @property
    def collection(self):
        """Lazy-load or create the ChromaDB collection."""
        if self._collection is None:
            import chromadb

            os.makedirs(self.persist_directory, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_directory
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d documents).",
                self.collection_name, self._collection.count(),
            )
        return self._collection

    def add_documents(
        self,
        docs: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index new documents into both dense and sparse retrieval stores.

        Args:
            docs: List of document text strings to index.
            metadatas: Optional list of metadata dicts for each document.
                Each dict should contain at minimum a 'source' key.

        Raises:
            ValueError: If docs is empty or metadatas length doesn't match.
        """
        if not docs:
            raise ValueError("docs list cannot be empty.")

        if metadatas and len(metadatas) != len(docs):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match docs length ({len(docs)})."
            )

        if metadatas is None:
            metadatas = [{"source": "unknown", "chunk_id": i} for i in range(len(docs))]

        # Generate unique IDs
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(docs))]

        # Dense indexing: encode and add to ChromaDB
        logger.info("Encoding %d documents for dense retrieval...", len(docs))
        embeddings = self.embedding_model.encode(
            docs, show_progress_bar=True, batch_size=32
        ).tolist()

        self.collection.add(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        # Sparse indexing: update BM25
        self._corpus.extend(docs)
        self._corpus_metadatas.extend(metadatas)
        self._rebuild_bm25()

        logger.info(
            "Indexed %d documents. Total: %d.", len(docs), self.collection.count()
        )

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current corpus."""
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve top-k relevant documents using hybrid dense + sparse retrieval.

        Gets results from both dense (ChromaDB) and sparse (BM25) retrievers,
        then merges and deduplicates using Reciprocal Rank Fusion.

        Args:
            query: The search query string.
            top_k: Number of documents to return (default: 5).

        Returns:
            Ranked list of Document objects with page_content and metadata
            (including source, chunk_id, and score).

        Raises:
            RuntimeError: If no documents have been indexed.
        """
        if self.collection.count() == 0:
            logger.warning("No documents indexed. Returning empty results.")
            return []

        # Dense retrieval from ChromaDB
        dense_results = self._dense_retrieve(query, top_k=top_k * 2)

        # Sparse retrieval from BM25
        sparse_results = self._sparse_retrieve(query, top_k=top_k * 2)

        # Merge with Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)

        logger.info(
            "Retrieved %d documents for query: '%s...'",
            len(fused), query[:50],
        )
        return fused

    def _dense_retrieve(self, query: str, top_k: int) -> List[Document]:
        """Perform dense retrieval using ChromaDB.

        Args:
            query: Search query string.
            top_k: Number of results to retrieve.

        Returns:
            List of Document objects with similarity scores.
        """
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # ChromaDB returns cosine distance; convert to similarity
                similarity = 1.0 - distance
                metadata["score"] = similarity
                metadata["retrieval_method"] = "dense"
                documents.append(Document(page_content=doc_text, metadata=metadata))

        return documents

    def _sparse_retrieve(self, query: str, top_k: int) -> List[Document]:
        """Perform sparse BM25 retrieval.

        Args:
            query: Search query string.
            top_k: Number of results to retrieve.

        Returns:
            List of Document objects with BM25 scores.
        """
        if self._bm25 is None or not self._corpus:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        documents = []
        for idx in top_indices:
            if scores[idx] > 0:
                metadata = (
                    self._corpus_metadatas[idx].copy()
                    if idx < len(self._corpus_metadatas)
                    else {}
                )
                metadata["score"] = float(scores[idx])
                metadata["retrieval_method"] = "sparse"
                documents.append(
                    Document(
                        page_content=self._corpus[idx],
                        metadata=metadata,
                    )
                )

        return documents

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
        top_k: int,
    ) -> List[Document]:
        """Merge results from dense and sparse retrievers using RRF.

        RRF score = sum(weight / (k + rank)) across retrieval methods.

        Args:
            dense_results: Documents from dense retrieval.
            sparse_results: Documents from sparse retrieval.
            top_k: Number of final results to return.

        Returns:
            Deduplicated and re-ranked list of Documents.
        """
        # Build a mapping: doc_content -> (document, rrf_score)
        doc_scores: Dict[str, tuple] = {}

        for rank, doc in enumerate(dense_results):
            content = doc.page_content
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            if content in doc_scores:
                existing_doc, existing_score = doc_scores[content]
                doc_scores[content] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[content] = (doc, rrf_score)

        for rank, doc in enumerate(sparse_results):
            content = doc.page_content
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            if content in doc_scores:
                existing_doc, existing_score = doc_scores[content]
                doc_scores[content] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[content] = (doc, rrf_score)

        # Sort by RRF score descending
        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x[1], reverse=True
        )

        # Return top_k with updated scores
        results = []
        for doc, rrf_score in sorted_docs[:top_k]:
            doc.metadata["rrf_score"] = rrf_score
            doc.metadata["score"] = rrf_score
            results.append(doc)

        return results

import os
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import settings
from src.core.embeddings import get_embeddings

COLLECTION_NAME = "technical_docs"

class VectorStoreManager:
    """
    Manages persistent ChromaDB vector store operations.
    """
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        os.makedirs(self.persist_directory, exist_ok=True)
        self._embeddings: Optional[Any] = None
        self._vector_store: Optional[Chroma] = None

    @property
    def embeddings(self):
        # Defer model initialization until the vector store is first used.
        # Importing API routes and running unit tests should not download a model.
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
        return self._vector_store

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add langchain Documents with metadata to the vector store."""
        if not documents:
            return []
        ids = [doc.metadata.get("chunk_id") for doc in documents if "chunk_id" in doc.metadata]
        if ids and len(ids) == len(documents):
            return self.vector_store.add_documents(documents=documents, ids=ids)
        return self.vector_store.add_documents(documents=documents)

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform standard vector similarity search."""
        top_k = k or settings.TOP_K_RETRIEVAL
        return self.vector_store.similarity_search(query=query, k=top_k)

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search returning documents and distance scores."""
        top_k = k or settings.TOP_K_RETRIEVAL
        return self.vector_store.similarity_search_with_score(query=query, k=top_k)

    def mmr_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform Maximal Marginal Relevance search for diverse results."""
        top_k = k or settings.TOP_K_RETRIEVAL
        return self.vector_store.max_marginal_relevance_search(query=query, k=top_k)

    def count(self) -> int:
        """Return total number of vectors in collection."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def get_document_catalog(self) -> List[Dict[str, Any]]:
        """Return distinct source documents with chunk counts and titles."""
        try:
            col = self.vector_store._collection
            metadata_results = col.get(include=["metadatas"])
            metadatas = metadata_results.get("metadatas", [])
            
            summary: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                if not meta:
                    continue
                source = meta.get("source", "unknown")
                if source not in summary:
                    summary[source] = {
                        "source": source,
                        "title": meta.get("title", source),
                        "chunk_count": 0,
                        "last_ingested": meta.get("ingested_at", "N/A"),
                    }
                summary[source]["chunk_count"] += 1

            return list(summary.values())
        except Exception:
            return []

# Singleton instance
vector_store_manager = VectorStoreManager()

"""
VectorStore — Wrapper ChromaDB pour la recherche sémantique (RAG).

Gère 3 collections :
    - daily_summaries : résumés journaliers indexés par embedding
    - user_notes      : notes et rappels de l'utilisateur
    - knowledge_base  : informations statiques (quartiers, contacts…)

Utilise sentence-transformers pour les embeddings multilingues.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from configuration import CONFIGURATION

logger = logging.getLogger(__name__)

_conf = CONFIGURATION.memory


class VectorStore:
    """Wrapper ChromaDB embarqué (pas de serveur)."""

    def __init__(self, persist_directory: str | None = None):
        self._persist_dir = persist_directory or _conf.chromadb_path
        self._client = None
        self._embedding_fn = None
        self._collections: dict = {}

    def _ensure_initialized(self) -> None:
        """Initialise le client ChromaDB et la fonction d'embedding (lazy)."""
        if self._client is not None:
            return

        import chromadb
        from chromadb.utils import embedding_functions

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_conf.embedding_model,
        )
        logger.info(
            "ChromaDB initialisé (path=%s, model=%s)",
            self._persist_dir,
            _conf.embedding_model,
        )

    def _get_collection(self, name: str):
        """Récupère ou crée une collection ChromaDB."""
        self._ensure_initialized()
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    # ─── Indexation ────────────────────────────────────────────────────────

    def index_document(
        self,
        collection_name: str,
        doc_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """Indexe un document dans une collection.

        Args:
            collection_name: 'daily_summaries' | 'user_notes' | 'knowledge_base'
            doc_id: Identifiant unique du document.
            content: Texte à indexer.
            metadata: Métadonnées (user_id, date, category…).
        """
        collection = self._get_collection(collection_name)
        collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata or {}],
        )
        logger.debug("Document indexé dans '%s' : id=%s", collection_name, doc_id)

    # ─── Recherche ─────────────────────────────────────────────────────────

    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 3,
        where: dict | None = None,
    ) -> List[dict]:
        """Recherche sémantique dans une collection.

        Returns:
            Liste de dicts : [{"id": str, "content": str, "metadata": dict, "distance": float}]
        """
        collection = self._get_collection(collection_name)

        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            logger.warning("Erreur recherche ChromaDB (%s) : %s", collection_name, e)
            return []

        docs = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                docs.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        return docs

    def search_all_collections(
        self,
        query: str,
        user_id: str,
        n_results: int = 5,
    ) -> List[dict]:
        """Recherche dans toutes les collections pour un utilisateur donné."""
        all_results = []
        for coll_name in ["daily_summaries", "user_notes"]:
            results = self.search(
                collection_name=coll_name,
                query=query,
                n_results=n_results,
                where={"user_id": user_id},
            )
            for r in results:
                r["collection"] = coll_name
            all_results.extend(results)

        # Trier par distance (plus faible = plus pertinent)
        all_results.sort(key=lambda x: x.get("distance", 1.0))
        return all_results[:n_results]

    # ─── Suppression ───────────────────────────────────────────────────────

    def delete_document(self, collection_name: str, doc_id: str) -> None:
        """Supprime un document d'une collection."""
        collection = self._get_collection(collection_name)
        collection.delete(ids=[doc_id])


# Singleton global
vector_store = VectorStore()

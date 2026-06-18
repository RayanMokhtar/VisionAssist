"""
Mémoire long terme — Interface RAG via ChromaDB.

Fournit :
    - Recherche sémantique dans les résumés journaliers et notes
    - Indexation de nouveaux documents (résumés, notes)
    - Contexte enrichi pour le prompt du LLM
"""

from __future__ import annotations

import logging
from typing import List

from langage.database.vector_store import vector_store

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Interface de mémoire long terme basée sur la recherche sémantique (RAG).

    Utilise ChromaDB pour indexer et rechercher dans :
        - Les résumés journaliers
        - Les notes utilisateur
    """

    def __init__(self):
        self._store = vector_store

    # ─── Recherche ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        user_id: str,
        n_results: int = 3,
    ) -> List[dict]:
        """Recherche sémantique dans toutes les collections pour un utilisateur.

        Returns:
            Liste de résultats : [{"content": str, "collection": str, "metadata": dict}]
        """
        results = self._store.search_all_collections(
            query=query,
            user_id=user_id,
            n_results=n_results,
        )
        logger.debug(
            "RAG search '%s' pour user=%s → %d résultats",
            query[:50], user_id, len(results),
        )
        return results

    def search_summaries(
        self,
        query: str,
        user_id: str,
        n_results: int = 3,
    ) -> List[dict]:
        """Recherche uniquement dans les résumés journaliers."""
        return self._store.search(
            collection_name="daily_summaries",
            query=query,
            n_results=n_results,
            where={"user_id": user_id},
        )

    def search_notes(
        self,
        query: str,
        user_id: str,
        n_results: int = 3,
    ) -> List[dict]:
        """Recherche uniquement dans les notes utilisateur."""
        return self._store.search(
            collection_name="user_notes",
            query=query,
            n_results=n_results,
            where={"user_id": user_id},
        )

    # ─── Indexation ────────────────────────────────────────────────────────

    def index_daily_summary(
        self,
        doc_id: str,
        content: str,
        user_id: str,
        date_str: str,
    ) -> None:
        """Indexe un résumé journalier dans ChromaDB."""
        self._store.index_document(
            collection_name="daily_summaries",
            doc_id=doc_id,
            content=content,
            metadata={"user_id": user_id, "date": date_str, "type": "daily_summary"},
        )
        logger.info("Résumé indexé dans ChromaDB : user=%s date=%s", user_id, date_str)

    def index_user_note(
        self,
        doc_id: str,
        content: str,
        user_id: str,
        category: str = "general",
    ) -> None:
        """Indexe une note utilisateur dans ChromaDB."""
        self._store.index_document(
            collection_name="user_notes",
            doc_id=doc_id,
            content=content,
            metadata={"user_id": user_id, "category": category, "type": "note"},
        )

    # ─── Formatage pour le contexte ────────────────────────────────────────

    def format_search_results(self, results: List[dict]) -> str:
        """Formate les résultats de recherche pour injection dans le prompt."""
        if not results:
            return "Aucune information pertinente trouvée dans la mémoire."

        lines = []
        for i, r in enumerate(results, 1):
            source = r.get("collection", "inconnu")
            meta = r.get("metadata", {})
            date = meta.get("date", "")
            prefix = f"[{source}]" + (f" ({date})" if date else "")
            lines.append(f"{i}. {prefix} {r['content'][:300]}")

        return "\n".join(lines)


# Singleton global
long_term_memory = LongTermMemory()

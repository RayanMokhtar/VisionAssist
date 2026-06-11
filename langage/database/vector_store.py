"""
VectorStore — Wrapper FAISS pour la recherche sémantique (RAG).

Architecture :
    - Embeddings : sentence-transformers/all-MiniLM-L6-v2 (384-dim, léger, multilingue)
    - Index       : FAISS IndexFlatIP (produit scalaire sur vecteurs L2-normalisés = cosine)
    - Persistance : fichiers .faiss + .meta.json dans persist_directory

Gère 3 collections :
    - daily_summaries : résumés journaliers indexés par embedding
    - user_notes      : notes et rappels de l'utilisateur
    - knowledge_base  : informations statiques (quartiers, contacts…)

L'interface publique (index_document / search / search_all_collections / delete_document)
est identique à l'ancien wrapper ChromaDB — aucun changement requis dans le reste du code.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constantes ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# ─── FaissCollection ─────────────────────────────────────────────────────────

class FaissCollection:
    """
    Une collection = un index FAISS + sa table de métadonnées.

    Structure sur disque :
        <persist_dir>/<name>.faiss      → index binaire FAISS
        <persist_dir>/<name>.meta.json  → [{id, content, metadata}, ...]

    Stratégie de recherche :
        IndexFlatIP (Inner Product) sur vecteurs L2-normalisés
        ⟹ équivalent à la similarité cosinus, simple et exact.
    """

    def __init__(self, name: str, persist_dir: str, embedding_fn) -> None:
        import faiss  # import tardif pour éviter le coût si non utilisé

        self.name = name
        self._faiss = faiss
        self._persist_dir = persist_dir
        self._embedding_fn = embedding_fn
        self._lock = threading.Lock()

        self._index_path = os.path.join(persist_dir, f"{name}.faiss")
        self._meta_path = os.path.join(persist_dir, f"{name}.meta.json")

        # Charger ou créer l'index
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._meta: List[Dict] = json.load(f)
            logger.info(
                "Collection FAISS '%s' chargée (%d documents).",
                name, len(self._meta),
            )
        else:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._meta = []
            logger.info("Collection FAISS '%s' créée (vide).", name)

    # ── Persistance ──────────────────────────────────────────────────────

    def _save(self) -> None:
        """Sauvegarde l'index FAISS + métadonnées sur disque."""
        self._faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    # ── Encodage ─────────────────────────────────────────────────────────

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode une liste de textes → matrice float32 L2-normalisée.
        FAISS IndexFlatIP sur vecteurs normalisés = cosine similarity.
        """
        embeddings = self._embedding_fn(texts)
        arr = np.array(embeddings, dtype=np.float32)
        # Normalisation L2 ligne par ligne
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # éviter division par zéro
        return arr / norms

    # ── API publique ──────────────────────────────────────────────────────

    def count(self) -> int:
        """Nombre de documents dans la collection."""
        return len(self._meta)

    def upsert(self, doc_id: str, content: str, metadata: Optional[dict] = None) -> None:
        """
        Insère ou met à jour un document.
        Si doc_id existe déjà, l'ancienne entrée est supprimée puis réindexée.
        """
        with self._lock:
            # Supprimer l'ancienne entrée si elle existe
            self._delete_by_id(doc_id)

            # Encoder et ajouter
            vec = self._encode([content])   # shape (1, 384)
            self._index.add(vec)

            self._meta.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata or {},
                "_faiss_pos": self._index.ntotal - 1,  # position dans l'index
            })
            self._save()

        logger.debug("Document '%s' indexé dans '%s'.", doc_id, self.name)

    def delete(self, doc_id: str) -> None:
        """Supprime un document par son ID."""
        with self._lock:
            self._delete_by_id(doc_id)
            self._save()

    def _delete_by_id(self, doc_id: str) -> None:
        """
        Suppression logique : retire l'entrée de _meta.
        FAISS IndexFlatIP ne supporte pas la suppression native,
        on reconstruit l'index si nécessaire (rare en usage normal).
        """
        original_count = len(self._meta)
        self._meta = [m for m in self._meta if m["id"] != doc_id]

        if len(self._meta) < original_count:
            # Reconstruire l'index depuis les vecteurs restants
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Reconstruit l'index FAISS depuis les métadonnées restantes."""
        import faiss as _faiss
        new_index = _faiss.IndexFlatIP(EMBEDDING_DIM)
        if self._meta:
            contents = [m["content"] for m in self._meta]
            vecs = self._encode(contents)
            new_index.add(vecs)
            # Mettre à jour les positions
            for i, m in enumerate(self._meta):
                m["_faiss_pos"] = i
        self._index = new_index

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> List[dict]:
        """
        Recherche sémantique.

        Args:
            query_text : question/texte de recherche
            n_results  : nombre max de résultats
            where      : filtre sur les métadonnées, ex: {"user_id": "u1"}

        Returns:
            [{id, content, metadata, distance}, ...] trié par pertinence décroissante
        """
        if self._index.ntotal == 0:
            return []

        # Pré-filtrage par métadonnées (avant la recherche vectorielle pour réduire le bruit)
        if where:
            filtered_meta = [
                (i, m) for i, m in enumerate(self._meta)
                if all(m["metadata"].get(k) == v for k, v in where.items())
            ]
            if not filtered_meta:
                return []

            # Construire un sous-index FAISS temporaire avec seulement les docs filtrés
            import faiss as _faiss
            sub_index = _faiss.IndexFlatIP(EMBEDDING_DIM)
            filtered_positions = [pos for pos, _ in filtered_meta]
            filtered_docs = [m for _, m in filtered_meta]

            # Récupérer les vecteurs correspondants depuis l'index principal
            # FAISS ne stocke pas les vecteurs par défaut avec IndexFlat* → on réencode
            sub_contents = [m["content"] for m in filtered_docs]
            sub_vecs = self._encode(sub_contents)
            sub_index.add(sub_vecs)

            k = min(n_results, len(filtered_docs))
            query_vec = self._encode([query_text])
            scores, indices = sub_index.search(query_vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc = filtered_docs[idx]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "distance": float(1.0 - score),  # convertir score IP → distance cosine
                })
            return results

        else:
            # Recherche globale
            k = min(n_results, self._index.ntotal)
            query_vec = self._encode([query_text])
            scores, indices = self._index.search(query_vec, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._meta):
                    continue
                doc = self._meta[idx]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "distance": float(1.0 - score),
                })
            return results


# ─── VectorStore ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Wrapper FAISS embarqué (pas de serveur, pas de dépendance externe lourde).

    Remplace ChromaDB avec la même interface publique :
        index_document / search / search_all_collections / delete_document

    Lazy-init : le modèle d'embedding et FAISS ne sont chargés qu'au premier appel.
    """

    def __init__(self, persist_directory: str | None = None):
        self._persist_dir = persist_directory or "./faiss_db"
        self._embedding_model = None
        self._collections: Dict[str, FaissCollection] = {}
        self._init_lock = threading.Lock()
        self._initialized = False

    # ── Initialisation lazy ───────────────────────────────────────────────

    def _ensure_initialized(self) -> None:
        """Charge sentence-transformers une seule fois (thread-safe)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers requis : pip install sentence-transformers"
                ) from e

            try:
                import faiss  # noqa: F401 — vérification de présence
            except ImportError as e:
                raise ImportError(
                    "faiss-cpu requis : pip install faiss-cpu"
                ) from e

            os.makedirs(self._persist_dir, exist_ok=True)

            logger.info(
                "Chargement du modèle d'embedding FAISS : %s", EMBEDDING_MODEL
            )
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self._initialized = True
            logger.info("VectorStore FAISS initialisé (path=%s).", self._persist_dir)

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode une liste de textes. Retourne un tableau numpy float32."""
        return self._embedding_model.encode(texts, convert_to_numpy=True)

    def _get_collection(self, name: str) -> FaissCollection:
        """Récupère ou crée une FaissCollection."""
        self._ensure_initialized()
        if name not in self._collections:
            self._collections[name] = FaissCollection(
                name=name,
                persist_dir=self._persist_dir,
                embedding_fn=self._embed,
            )
        return self._collections[name]

    # ── Indexation ────────────────────────────────────────────────────────

    def index_document(
        self,
        collection_name: str,
        doc_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Indexe (ou met à jour) un document dans une collection FAISS.

        Args:
            collection_name : 'daily_summaries' | 'user_notes' | 'knowledge_base'
            doc_id          : identifiant unique (upsert si déjà présent)
            content         : texte à encoder et indexer
            metadata        : dict libre (user_id, date, category…)
        """
        collection = self._get_collection(collection_name)
        collection.upsert(doc_id=doc_id, content=content, metadata=metadata)
        logger.debug(
            "Document indexé dans '%s' : id=%s", collection_name, doc_id
        )

    # ── Recherche ─────────────────────────────────────────────────────────

    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 3,
        where: dict | None = None,
    ) -> List[dict]:
        """
        Recherche sémantique dans une collection.

        Returns:
            [{id, content, metadata, distance}, ...] — distance cosine ∈ [0, 2],
            0 = identique, 1 = orthogonal, 2 = opposé.
        """
        collection = self._get_collection(collection_name)
        if collection.count() == 0:
            return []

        return collection.query(
            query_text=query,
            n_results=min(n_results, collection.count()),
            where=where,
        )

    def search_all_collections(
        self,
        query: str,
        user_id: str,
        n_results: int = 5,
    ) -> List[dict]:
        """Recherche dans toutes les collections pertinentes pour un utilisateur."""
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

    # ── Suppression ────────────────────────────────────────────────────────

    def delete_document(self, collection_name: str, doc_id: str) -> None:
        """Supprime un document d'une collection."""
        collection = self._get_collection(collection_name)
        collection.delete(doc_id)
        logger.debug(
            "Document supprimé de '%s' : id=%s", collection_name, doc_id
        )

    # ── Utilitaires ───────────────────────────────────────────────────────

    def collection_count(self, collection_name: str) -> int:
        """Retourne le nombre de documents dans une collection."""
        return self._get_collection(collection_name).count()

    def search_by_session(
        self,
        query: str,
        session_id: str,
        n_results: int = 4,
        score_threshold: float = 0.60,
    ) -> list[dict]:
        """
        Recherche sémantique FAISS filtrée par session_id.

        Interroge 'user_notes' et 'daily_summaries' en filtrant sur
        metadata["session_id"]. Trie par pertinence et applique un seuil
        de distance cosinus pour ne garder que les résultats pertinents.

        Args:
            query          : question de l'utilisateur (texte libre)
            session_id     : ID de session (issu de la base, déjà connu)
            n_results      : nombre max de passages retournés
            score_threshold: distance cosinus max acceptée (0=parfait, 1=orthogonal)

        Returns:
            [{id, content, metadata, distance, collection}, ...]
        """
        all_hits: list[dict] = []
        for coll in ("user_notes", "daily_summaries"):
            hits = self.search(
                collection_name=coll,
                query=query,
                n_results=n_results,
                where={"session_id": session_id},
            )
            for h in hits:
                h["collection"] = coll
            all_hits.extend(hits)

        all_hits.sort(key=lambda x: x.get("distance", 1.0))
        return [h for h in all_hits if h.get("distance", 1.0) <= score_threshold][:n_results]


# Singleton global
vector_store = VectorStore()

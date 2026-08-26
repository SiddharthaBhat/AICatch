"""ChromaDB Vector Database Wrapper for AICatch.

Handles initialization, vector insertion, and similarity querying for
visual (CLIP) and facial embeddings.
"""

import chromadb
from chromadb.config import Settings
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from config import CHROMA_PERSIST_DIR

LOGGER = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.init_collections()

    def init_collections(self) -> None:
        """Initialize CLIP visual collection and Face embeddings collection."""
        # Visual collection: cosine distance search for CLIP embeddings (512-D for ViT-B/32)
        self.visual_coll = self.client.get_or_create_collection(
            name="clip_visual_vectors",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Face collection: cosine distance search for face embeddings (usually 512-D)
        self.face_coll = self.client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

        # Person Re-ID collection: cosine distance search for 576-D Re-ID embeddings
        self.reid_coll = self.client.get_or_create_collection(
            name="person_reid_vectors",
            metadata={"hnsw:space": "cosine"}
        )

    # ==============================================================================
    # Insert / Add Methods
    # ==============================================================================

    def add_visual_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None
    ) -> None:
        """Insert a batch of visual crop (CLIP) embeddings into ChromaDB."""
        if not ids:
            return
        self.visual_coll.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )


    def add_reid_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None
    ) -> None:
        """Insert a batch of Person Re-ID crop embeddings into ChromaDB."""
        if not ids:
            return
        self.reid_coll.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query_reid_embeddings(
        self,
        query_embedding: List[float],
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the Person Re-ID collection for matching tracklets."""
        where = self._build_where_filter(filters)
        results = self.reid_coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"]
        )
        return self._format_results(results)

    def add_face_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None
    ) -> None:
        """Insert a batch of face crop embeddings into ChromaDB."""
        if not ids:
            return
        self.face_coll.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    # ==============================================================================
    # Query / Search Methods
    # ==============================================================================

    def query_visual_embeddings(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the visual collection for matching CLIP embeddings."""
        where = self._build_where_filter(filters)
        results = self.visual_coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"]
        )
        return self._format_results(results)

    def query_face_embeddings(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the face collection for matching facial embeddings."""
        where = self._build_where_filter(filters)
        results = self.face_coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"]
        )
        return self._format_results(results)

    # ==============================================================================
    # Helpers
    # ==============================================================================

    def _build_where_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Transforms a key-value dictionary into a valid ChromaDB where-filter query."""
        if not filters:
            return None
        
        # Keep it simple for single filter keys, format complex filters if needed
        # ChromaDB syntax for multiple conditions: {"$and": [{"camera_id": "cam1"}, {"job_id": "job1"}]}
        clean_filters = {k: v for k, v in filters.items() if v is not None}
        if not clean_filters:
            return None
            
        if len(clean_filters) == 1:
            return clean_filters
            
        return {"$and": [{k: v} for k, v in clean_filters.items()]}

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formats raw ChromaDB dictionary outputs into clean, standardized matching maps."""
        formatted = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for item_id, meta, dist in zip(ids, metadatas, distances):
            # Cosine similarity = 1.0 - Cosine Distance
            similarity = round(1.0 - float(dist), 4)
            formatted.append({
                "id": item_id,
                "metadata": meta,
                "distance": float(dist),
                "similarity": similarity
            })
            
        # Sort desc by similarity
        formatted.sort(key=lambda x: x["similarity"], reverse=True)
        return formatted

    def delete_by_job(self, job_id: str) -> None:
        """Deletes all vector references associated with a specific job ID."""
        self.visual_coll.delete(where={"job_id": job_id})
        self.face_coll.delete(where={"job_id": job_id})
        LOGGER.info(f"Deleted vectors for job: {job_id}")

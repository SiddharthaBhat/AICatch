"""ChromaDB Vector Database Wrapper for AICatch.

Handles initialization, vector insertion, and similarity querying for
visual (CLIP) and facial embeddings.
"""

import chromadb
from chromadb.config import Settings
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_VERSION

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
        # A vector collection is a model space.  Never place CLIP and SigLIP2
        # vectors together: same-dimensional vectors are still incompatible.
        self.visual_coll = self.client.get_or_create_collection(
            name=f"visual_{EMBEDDING_MODEL_NAME}_{EMBEDDING_VERSION}".replace("-", "_"),
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL_NAME, "embedding_version": EMBEDDING_VERSION}
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

        # Scene / Keyframe collection: cosine distance search for full-frame CLIP vectors
        self.scene_coll = self.client.get_or_create_collection(
            name=f"scene_{EMBEDDING_MODEL_NAME}_{EMBEDDING_VERSION}".replace("-", "_"),
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL_NAME, "embedding_version": EMBEDDING_VERSION}
        )

    # ==============================================================================
    # Insert / Add Methods
    # ==============================================================================

    def _validate_and_normalize_embeddings(
        self,
        embeddings: List[List[float]],
        expected_dim: Optional[int] = None
    ) -> List[List[float]]:
        """Validates that vectors are finite, non-zero, and match expected dimension."""
        if not embeddings:
            return []
        validated = []
        for i, vec in enumerate(embeddings):
            if not isinstance(vec, (list, tuple, np.ndarray)) or len(vec) == 0:
                raise ValueError(f"Embedding at index {i} is empty or invalid type: {type(vec)}")
            if expected_dim is not None and len(vec) != expected_dim:
                raise ValueError(f"Embedding at index {i} dimension mismatch: expected {expected_dim}, got {len(vec)}")
            # Check for finite values and non-zero norm
            vec_list = [float(v) for v in vec]
            norm_sq = 0.0
            for val in vec_list:
                if not np.isfinite(val):
                    raise ValueError(f"Embedding at index {i} contains non-finite value: {val}")
                norm_sq += val * val
            if norm_sq <= 1e-12:
                raise ValueError(f"Embedding at index {i} has zero or near-zero magnitude")
            validated.append(vec_list)
        return validated

    def add_visual_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None
    ) -> int:
        """Insert a batch of visual crop embeddings into ChromaDB with validation and verification."""
        if not ids:
            return 0
        if len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError(f"Count mismatch: ids={len(ids)}, embeddings={len(embeddings)}, metadatas={len(metadatas)}")
        
        # Validate vector values
        validated_embeddings = self._validate_and_normalize_embeddings(embeddings)
        dim = len(validated_embeddings[0])

        # Ensure metadata has active model/version/dim
        normalized_metas = []
        for meta in metadatas:
            m = dict(meta)
            m.setdefault("embedding_model", EMBEDDING_MODEL_NAME)
            m.setdefault("embedding_version", EMBEDDING_VERSION)
            m.setdefault("embedding_dimension", dim)
            if m.get("embedding_model") != EMBEDDING_MODEL_NAME or m.get("embedding_version") != EMBEDDING_VERSION:
                raise ValueError(f"Visual embedding metadata ({m.get('embedding_model')}:{m.get('embedding_version')}) does not match the active vector space ({EMBEDDING_MODEL_NAME}:{EMBEDDING_VERSION})")
            normalized_metas.append(m)

        self.visual_coll.upsert(
            ids=ids,
            embeddings=validated_embeddings,
            metadatas=normalized_metas,
            documents=documents
        )
        LOGGER.info("VISUAL_VECTOR_INSERT attempted=%d collection=%s total=%d", len(ids), self.visual_coll.name, self.visual_coll.count())
        return len(ids)

    def verify_vectors_written(
        self,
        collection_type: str,
        ids: List[str],
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """Performs actual write/read retrieval verification for indexed vectors."""
        if not ids:
            return {"verified": True, "count": 0, "sample_verified": 0}
        
        coll = self.visual_coll if collection_type == "visual" else (self.reid_coll if collection_type == "reid" else self.face_coll)
        check_ids = ids[:sample_size] if len(ids) > sample_size else ids
        
        retrieved = coll.get(ids=check_ids, include=["metadatas", "embeddings"])
        retrieved_ids = set(retrieved.get("ids", []))
        
        missing = [cid for cid in check_ids if cid not in retrieved_ids]
        if missing:
            err_msg = f"Vector read-back verification failed in {coll.name}: missing IDs {missing}"
            LOGGER.error(err_msg)
            return {"verified": False, "missing_ids": missing, "error": err_msg}
            
        LOGGER.info("VECTOR_WRITE_VERIFIED collection=%s sampled=%d/%d total_coll=%d", coll.name, len(check_ids), len(ids), coll.count())
        return {
            "verified": True,
            "collection": coll.name,
            "sampled_count": len(check_ids),
            "total_ids": len(ids),
            "collection_total": coll.count()
        }

    def add_scene_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None
    ) -> int:
        """Insert a batch of full-frame scene keyframe embeddings into ChromaDB with validation."""
        if not ids:
            return 0
        if len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError(f"Count mismatch: ids={len(ids)}, embeddings={len(embeddings)}, metadatas={len(metadatas)}")
        
        validated_embeddings = self._validate_and_normalize_embeddings(embeddings)
        dim = len(validated_embeddings[0])

        normalized_metas = []
        for meta in metadatas:
            m = dict(meta)
            m.setdefault("embedding_model", EMBEDDING_MODEL_NAME)
            m.setdefault("embedding_version", EMBEDDING_VERSION)
            m.setdefault("embedding_dim", dim)
            m.setdefault("embedding_type", "scene")
            normalized_metas.append(m)

        self.scene_coll.upsert(
            ids=ids,
            embeddings=validated_embeddings,
            metadatas=normalized_metas,
            documents=documents
        )
        LOGGER.info(
            "SCENE_VECTOR_INSERT attempted=%d collection=%s dim=%d total_coll=%d",
            len(ids), self.scene_coll.name, dim, self.scene_coll.count()
        )
        return len(ids)

    def query_scene_embeddings(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the scene keyframe collection for matching full-frame visual vectors."""
        where = self._build_where_filter(filters)
        results = self.scene_coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"]
        )
        return self._format_results(results)

    def describe_scene_index(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return aggregate index facts for scene keyframes."""
        active_filters = dict(filters or {})
        active_filters.update({"embedding_model": EMBEDDING_MODEL_NAME, "embedding_version": EMBEDDING_VERSION})
        records = self.scene_coll.get(where=self._build_where_filter(active_filters), include=["metadatas", "embeddings"])
        metadatas, embeddings = records.get("metadatas", []), records.get("embeddings", [])
        by_camera: Dict[str, int] = {}
        for meta in metadatas:
            camera = meta.get("camera_id", "unknown")
            by_camera[camera] = by_camera.get(camera, 0) + 1
        return {
            "collection": self.scene_coll.name,
            "total": len(metadatas),
            "by_camera": by_camera,
            "embedding_dimensions": sorted({len(vector) for vector in embeddings if vector is not None}),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_version": EMBEDDING_VERSION
        }

    def add_reid_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None
    ) -> int:
        """Insert a batch of Person Re-ID crop embeddings into ChromaDB with validation."""
        if not ids:
            return 0
        if len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError(f"Count mismatch: ids={len(ids)}, embeddings={len(embeddings)}, metadatas={len(metadatas)}")
        
        validated_embeddings = self._validate_and_normalize_embeddings(embeddings)
        self.reid_coll.upsert(
            ids=ids,
            embeddings=validated_embeddings,
            metadatas=metadatas,
            documents=documents
        )
        LOGGER.info("REID_VECTOR_INSERT attempted=%d collection=%s total=%d", len(ids), self.reid_coll.name, self.reid_coll.count())
        return len(ids)

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
    ) -> int:
        """Insert a batch of face crop embeddings into ChromaDB with validation."""
        if not ids:
            return 0
        if len(ids) != len(embeddings) or len(ids) != len(metadatas):
            raise ValueError(f"Count mismatch: ids={len(ids)}, embeddings={len(embeddings)}, metadatas={len(metadatas)}")
            
        validated_embeddings = self._validate_and_normalize_embeddings(embeddings)
        self.face_coll.upsert(
            ids=ids,
            embeddings=validated_embeddings,
            metadatas=metadatas,
            documents=documents
        )
        LOGGER.info("FACE_VECTOR_INSERT attempted=%d collection=%s total=%d", len(ids), self.face_coll.name, self.face_coll.count())
        return len(ids)

    def delete_face_embeddings(self, ids: List[str]) -> int:
        """Remove specific face embedding IDs from ChromaDB."""
        if not ids:
            return 0
        try:
            self.face_coll.delete(ids=ids)
            LOGGER.info(f"Deleted {len(ids)} face embeddings from {self.face_coll.name}")
            return len(ids)
        except Exception as e:
            LOGGER.warning(f"Failed to delete face embeddings {ids}: {e}")
            return 0

    # ==============================================================================
    # Query / Search Methods
    # ==============================================================================

    def query_visual_embeddings(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search only the active model/version visual space."""
        filters = dict(filters or {})
        filters.update({"embedding_model": EMBEDDING_MODEL_NAME, "embedding_version": EMBEDDING_VERSION})
        where = self._build_where_filter(filters)
        results = self.visual_coll.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"]
        )
        return self._format_results(results)

    def describe_visual_index(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return safe aggregate index facts for developer search diagnostics."""
        active_filters = dict(filters or {})
        active_filters.update({"embedding_model": EMBEDDING_MODEL_NAME, "embedding_version": EMBEDDING_VERSION})
        records = self.visual_coll.get(where=self._build_where_filter(active_filters), include=["metadatas", "embeddings"])
        metadatas, embeddings = records.get("metadatas", []), records.get("embeddings", [])
        by_type: Dict[str, int] = {}
        by_camera: Dict[str, int] = {}
        for meta in metadatas:
            kind, camera = meta.get("entity_type", "unknown"), meta.get("camera_id", "unknown")
            by_type[kind] = by_type.get(kind, 0) + 1
            by_camera[camera] = by_camera.get(camera, 0) + 1
        return {"collection": self.visual_coll.name, "total": len(metadatas), "by_object_type": by_type,
                "by_camera": by_camera, "embedding_dimensions": sorted({len(vector) for vector in embeddings if vector is not None}),
                "embedding_model": EMBEDDING_MODEL_NAME, "embedding_version": EMBEDDING_VERSION}

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

    def delete_by_job(self, job_id: str) -> Dict[str, int]:
        """Deletes all vector references associated with a specific job ID across all collections with verification."""
        deleted_counts = {"visual": 0, "reid": 0, "face": 0, "scene": 0}
        
        for coll_name, coll in [("visual", self.visual_coll), ("reid", self.reid_coll), ("face", self.face_coll), ("scene", self.scene_coll)]:
            try:
                # Direct collection where-delete (ChromaDB native)
                try:
                    coll.delete(where={"job_id": job_id})
                except Exception:
                    pass
                try:
                    coll.delete(where={"investigation_id": job_id})
                except Exception:
                    pass

                # Also find and delete any remaining IDs in batches until fully purged
                matched_ids = set()
                while True:
                    batch_ids = []
                    try:
                        res1 = coll.get(where={"job_id": job_id}, limit=500)
                        if res1 and res1.get("ids"):
                            batch_ids.extend(res1["ids"])
                    except Exception:
                        pass
                        
                    try:
                        res2 = coll.get(where={"investigation_id": job_id}, limit=500)
                        if res2 and res2.get("ids"):
                            batch_ids.extend(res2["ids"])
                    except Exception:
                        pass

                    if not batch_ids:
                        break

                    matched_ids.update(batch_ids)
                    try:
                        coll.delete(ids=list(set(batch_ids)))
                    except Exception as del_err:
                        LOGGER.warning(f"Batch delete error in {coll_name} for {job_id}: {del_err}")
                        break

                deleted_counts[coll_name] = len(matched_ids)
                if matched_ids:
                    LOGGER.info(f"Deleted {len(matched_ids)} {coll_name} vectors for job: {job_id}")
            except Exception as e:
                LOGGER.warning(f"Error cleaning {coll_name} vectors for job {job_id}: {e}")
                
        return deleted_counts

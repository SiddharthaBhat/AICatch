"""Unified Search Abstraction Layer for AICatch.

Combines vector query results from ChromaDB with relational metadata
from SQLite to provide a structured API for the frontend and matcher.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from storage.media_storage import MediaStorage

LOGGER = logging.getLogger(__name__)

class VisualIndexEmptyError(RuntimeError):
    """The selected investigation has no vectors in its active visual space."""
    def __init__(self, diagnostics: Dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__("This investigation has no visual embeddings. Re-index with the active visual model.")

def parse_query_text(query: Optional[str]) -> Dict[str, Any]:
    """Helper to parse a text query into target entity type, requested colors, and action verbs."""
    if not query:
        return {"entity_type": None, "colors": [], "action": None, "is_disallowed": False}
    words = query.lower().split()
    
    # 1. Color categories mapping
    colors_list = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown", "black", "white", "gray"]
    detected_colors = [c for c in colors_list if c in words]
    
    # 2. Action Type matching
    action = None
    if any(w in ["run", "running", "sprint", "fast"] for w in words):
        action = "running"
    elif any(w in ["walk", "walking", "stroll", "move", "moving"] for w in words):
        action = "walking"
    elif any(w in ["stand", "standing", "still", "stationary", "stop", "stopped"] for w in words):
        action = "standing"
    elif any(w in ["sit", "sitting", "lie", "lying", "lay", "couch", "chair"] for w in words):
        action = "sitting"
        
    # 3. Entity Type matching (with expanded attribute keywords)
    person_keywords = ["person", "people", "man", "woman", "lady", "boy", "girl", "guy", "subject", "someone", "individual",
                       "shirt", "pants", "clothing", "jacket", "top", "coat", "dress", "jeans", "hat", "cap", "wear", "wearing"]
    vehicle_keywords = ["car", "vehicle", "truck", "bus", "suv", "sedan", "automobile", "bike", "motorcycle",
                        "wheel", "window", "door", "tire", "trunk", "hood", "plate"]
    
    entity_type = None
    if any(w in person_keywords for w in words):
        entity_type = "person"
    elif any(w in vehicle_keywords for w in words):
        entity_type = "vehicle"
    elif action is not None:
        # Actions like running/walking/sitting apply exclusively to person entities in the dataset
        entity_type = "person"
        
    # 4. Disallowed / Non-indexed YOLO classes (Section 13)
    disallowed_objects = ["cow", "dog", "cat", "horse", "sheep", "elephant", "bear", "zebra", "giraffe", "bird", "airplane", "boat", "train", "bed", "table", "laptop", "tv", "cell phone", "bottle", "cup", "fork", "knife", "spoon", "banana", "apple", "sandwich", "pizza", "donut", "cake"]
    is_disallowed = any(w in disallowed_objects for w in words)
        
    return {
        "entity_type": entity_type,
        "colors": detected_colors,
        "action": action,
        "is_disallowed": is_disallowed
    }

def cluster_and_diversify_occurrences(
    results: List[Dict[str, Any]],
    temporal_window_sec: float = 12.0,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """Groups adjacent keyframe/observation matches on the same camera into temporal occurrence episodes,
    and applies camera-fair interleaving to prevent single-camera monopolization while preserving all raw observations
    for forensic drill-down."""
    if not results:
        return []

    # 1. Group results by camera
    by_camera: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        cam_id = item.get("camera_id") or "unknown_cam"
        by_camera.setdefault(cam_id, []).append(item)

    camera_clusters: Dict[str, List[Dict[str, Any]]] = {}

    # 2. For each camera, sort by timestamp and form temporal clusters
    for cam_id, cam_items in by_camera.items():
        sorted_items = sorted(cam_items, key=lambda x: float(x.get("timestamp", 0.0)))
        clusters: List[List[Dict[str, Any]]] = []
        current_cluster: List[Dict[str, Any]] = []

        for item in sorted_items:
            t = float(item.get("timestamp", 0.0))
            if not current_cluster:
                current_cluster.append(item)
            else:
                prev_t = float(current_cluster[-1].get("timestamp", 0.0))
                # Group if adjacent within temporal window on same camera
                if abs(t - prev_t) <= temporal_window_sec:
                    current_cluster.append(item)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [item]
        if current_cluster:
            clusters.append(current_cluster)

        # 3. For each cluster, pick representative item (highest similarity/score) and attach occurrence metadata
        rep_occurrences: List[Dict[str, Any]] = []
        for c_idx, cluster in enumerate(clusters):
            sorted_by_score = sorted(
                cluster,
                key=lambda x: (x.get("rrf_score") or x.get("similarity") or 0.0),
                reverse=True
            )
            rep_item = dict(sorted_by_score[0])

            timestamps = [float(x.get("timestamp", 0.0)) for x in cluster]
            start_t = min(timestamps)
            end_t = max(timestamps)

            occ_id = f"occ_{cam_id}_{int(start_t)}_{int(end_t)}_{c_idx}"
            rep_item["occurrence_id"] = occ_id
            rep_item["occurrence_start"] = round(start_t, 2)
            rep_item["occurrence_end"] = round(end_t, 2)
            rep_item["occurrence_duration"] = round(end_t - start_t, 2)
            rep_item["occurrence_count"] = len(cluster)
            rep_item["occurrence_observations"] = cluster

            rep_occurrences.append(rep_item)

        rep_occurrences.sort(
            key=lambda x: (x.get("rrf_score") or x.get("similarity") or 0.0),
            reverse=True
        )
        camera_clusters[cam_id] = rep_occurrences

    # 4. Multi-camera fair interleaving (Round-Robin with score priority)
    diversified: List[Dict[str, Any]] = []
    camera_queues = [list(occ_list) for occ_list in camera_clusters.values() if occ_list]

    while camera_queues and len(diversified) < top_k:
        camera_queues.sort(
            key=lambda q: (q[0].get("rrf_score") or q[0].get("similarity") or 0.0),
            reverse=True
        )
        best_queue = camera_queues[0]
        diversified.append(best_queue.pop(0))
        if not best_queue:
            camera_queues.pop(0)

    return diversified[:top_k]


class SearchLayer:
    def __init__(self, sqlite_db: SQLiteDatabase, vector_db: VectorDatabase, media_storage: MediaStorage):
        self.sqlite = sqlite_db
        self.vector = vector_db
        self.media = media_storage

    def search_by_text(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        query_text: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Performs a hybrid visual text search using both semantic CLIP similarities and structured color/entity metadata."""
        # Parse query for structured constraints
        parsed = parse_query_text(query_text)
        
        # Check if query is for a disallowed/non-indexed entity class (Section 13)
        if parsed.get("is_disallowed"):
            return []
            
        # Build search filters dynamically
        search_filters = dict(filters or {})
        if parsed["entity_type"] is not None and "entity_type" not in search_filters:
            search_filters["entity_type"] = parsed["entity_type"]
            
        # Stage 1: Candidate Retrieval (fetch top max 100 for re-ranking)
        candidate_k = max(100, top_k * 4)
        vector_matches = self.vector.query_visual_embeddings(
            query_embedding=query_embedding,
            top_k=candidate_k,
            filters=search_filters
        )
        
        # Stage 2: Inflate, Score and Re-rank
        from config import TEXT_MATCH_THRESHOLD, FINAL_SCORE_MIN_TEXT
        re_ranked = self._inflate_metadata_and_re_rank(
            vector_matches,
            query_colors=parsed["colors"],
            is_face=False,
            query_action=parsed["action"],
            threshold=TEXT_MATCH_THRESHOLD,
            final_score_threshold=FINAL_SCORE_MIN_TEXT,
            expected_entity_type=parsed["entity_type"]
        )
        return re_ranked[:top_k]

    def search_face_internal(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 100
    ) -> Dict[str, float]:
        """Queries ChromaDB face collection and returns a map of track_id -> similarity_score."""
        vector_matches = self.vector.query_face_embeddings(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        face_scores = {}
        for match in vector_matches:
            meta = match["metadata"]
            track_uuid = meta.get("track_id")
            if track_uuid:
                similarity = float(match["similarity"])
                if track_uuid not in face_scores or similarity > face_scores[track_uuid]:
                    face_scores[track_uuid] = similarity
        return face_scores

    def search_by_image(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        entity_type: Optional[str] = None,
        face_embedding: Optional[List[float]] = None,
        face_status: Optional[str] = None,
        strict_gate: bool = True,
        reid_embedding: Optional[List[float]] = None,
        enable_rrf: bool = False,
        rrf_k: int = 60,
        clip_weight: float = 0.7,
        reid_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Run hybrid object-crop and scene-keyframe image retrieval across cameras with full multi-signal breakdown."""
        from config import IMAGE_MATCH_THRESHOLD, IMAGE_VISUAL_WEIGHT, IMAGE_FACE_WEIGHT, FACE_MATCH_THRESHOLD, FINAL_SCORE_MIN_IMAGE
        
        search_filters = dict(filters or {})
        job_id = search_filters.get("job_id")
        cam_id = search_filters.get("camera_id")
        
        # Determine cameras in scope
        cameras_in_scope = []
        if job_id:
            cameras_in_scope = self.sqlite.get_cameras_by_job(job_id)
        allowed_camera_ids = {c["id"] for c in cameras_in_scope} if cameras_in_scope else None
        
        obj_filters = dict(search_filters)
        if entity_type is not None and "entity_type" not in obj_filters:
            if entity_type != "object":
                obj_filters["entity_type"] = entity_type

        is_identity_query = (entity_type == "person")

        index_info = self.vector.describe_visual_index(obj_filters)
        scene_info = self.vector.describe_scene_index(search_filters) if (hasattr(self.vector, "describe_scene_index") and not is_identity_query) else {"total": 0}
        region_info = self.vector.describe_region_index(search_filters) if (hasattr(self.vector, "describe_region_index") and not is_identity_query) else {"total": 0}
        
        diagnostics: Dict[str, Any] = {
            "collection": index_info["collection"],
            "indexed_vectors": index_info["total"],
            "indexed_scene_vectors": scene_info.get("total", 0),
            "indexed_region_vectors": region_info.get("total", 0),
            "indexed_by_object_type": index_info["by_object_type"],
            "indexed_by_camera": index_info["by_camera"],
            "embedding_model": index_info["embedding_model"],
            "embedding_version": index_info["embedding_version"],
            "query_embedding_dimension": len(query_embedding),
            "query_embedding_norm": round(sum(value * value for value in query_embedding) ** 0.5, 6),
            "filters": search_filters,
        }
        self.last_image_search_diagnostics = diagnostics
        
        if is_identity_query:
            if not index_info["total"]:
                diagnostics["search_state"] = "NO_DATA"
                raise VisualIndexEmptyError(diagnostics)
        else:
            if not index_info["total"] and not scene_info.get("total", 0) and not region_info.get("total", 0):
                diagnostics["search_state"] = "NO_DATA"
                raise VisualIndexEmptyError(diagnostics)

        # Stage 1: Candidate Retrieval (Object Crops - CLIP & Re-ID)
        candidate_k = max(150, top_k * 6)
        vector_matches: List[Dict[str, Any]] = []
        seen_ids = set()

        if entity_type not in ["scene", "region"]:
            # Global candidate retrieval across investigation scope (CLIP visual)
            global_vis = self.vector.query_visual_embeddings(
                query_embedding=query_embedding,
                top_k=candidate_k,
                filters=obj_filters
            )
            for cr in global_vis:
                c_id = cr.get("metadata", {}).get("camera_id")
                if allowed_camera_ids and c_id and c_id not in allowed_camera_ids:
                    continue
                if cr["id"] not in seen_ids:
                    seen_ids.add(cr["id"])
                    vector_matches.append(cr)

        # Re-ID candidate retrieval and candidate expansion for person queries
        reid_scores_by_track: Dict[str, float] = {}
        reid_ranks_by_track: Dict[str, int] = {}
        if enable_rrf and reid_embedding is not None:
            try:
                seen_r_ids = set()
                reid_matches: List[Dict[str, Any]] = []

                # Global Re-ID candidate retrieval across investigation scope
                global_reid = self.vector.query_reid_embeddings(
                    query_embedding=reid_embedding,
                    top_k=candidate_k,
                    filters=obj_filters
                )
                for cr in global_reid:
                    c_id = cr.get("metadata", {}).get("camera_id")
                    if allowed_camera_ids and c_id and c_id not in allowed_camera_ids:
                        continue
                    if cr["id"] not in seen_r_ids:
                        seen_r_ids.add(cr["id"])
                        reid_matches.append(cr)

                    # Bi-directional candidate expansion: include Re-ID candidates directly in vector_matches
                    if cr["id"] not in seen_ids:
                        seen_ids.add(cr["id"])
                        vector_matches.append(cr)

                # Candidate Completeness: ensure Re-ID evidence is present for visual candidate tracks
                import numpy as np
                q_reid_arr = np.array(reid_embedding, dtype=np.float32)
                q_reid_norm = float(np.linalg.norm(q_reid_arr))
                if q_reid_norm > 0:
                    candidate_tracks = set()
                    for m in vector_matches:
                        t = m.get("metadata", {}).get("track_id")
                        if t:
                            candidate_tracks.add(t)

                    for tid in candidate_tracks:
                        try:
                            t_reid = self.vector.reid_coll.get(where={"track_id": tid}, include=["embeddings", "metadatas"])
                            if t_reid and t_reid.get("embeddings") is not None and len(t_reid["embeddings"]) > 0:
                                best_sim = -1.0
                                best_id = None
                                best_meta = None
                                for vid, vemb, vmeta in zip(t_reid["ids"], t_reid["embeddings"], t_reid["metadatas"]):
                                    c_id = vmeta.get("camera_id")
                                    if allowed_camera_ids and c_id and c_id not in allowed_camera_ids:
                                        continue
                                    arr = np.array(vemb, dtype=np.float32)
                                    arr_norm = float(np.linalg.norm(arr))
                                    if arr_norm > 0:
                                        cos = float(np.dot(q_reid_arr, arr) / (q_reid_norm * arr_norm))
                                        if cos > best_sim:
                                            best_sim = cos
                                            best_id = vid
                                            best_meta = vmeta
                                if best_id and best_id not in seen_r_ids:
                                    seen_r_ids.add(best_id)
                                    reid_matches.append({
                                        "id": best_id,
                                        "similarity": best_sim,
                                        "metadata": best_meta
                                    })
                        except Exception as track_reid_err:
                            LOGGER.debug(f"Candidate completeness lookup for {tid}: {track_reid_err}")

                reid_matches.sort(key=lambda x: float(x["similarity"]), reverse=True)
                for r_rank, r_match in enumerate(reid_matches, start=1):
                    r_meta = r_match.get("metadata", {})
                    t_id = r_meta.get("track_id")
                    if t_id and t_id not in reid_ranks_by_track:
                        reid_ranks_by_track[t_id] = r_rank
                        reid_scores_by_track[t_id] = float(r_match["similarity"])
            except Exception as re_err:
                LOGGER.debug(f"Re-ID collection query note: {re_err}")

        vector_matches.sort(key=lambda x: float(x["similarity"]), reverse=True)
        diagnostics["raw_candidate_count"] = len(vector_matches)
        diagnostics["raw_top_scores"] = [round(float(match["similarity"]), 4) for match in vector_matches[:10]]

        # Stage 2: Inflate and Re-rank Object Matches
        obj_results, below_threshold = self._inflate_metadata_and_re_rank_with_below_gate(
            vector_matches,
            query_colors=None,
            is_face=False,
            threshold=IMAGE_MATCH_THRESHOLD,
            final_score_threshold=FINAL_SCORE_MIN_IMAGE if strict_gate else 0.0,
            expected_entity_type=entity_type,
            reid_scores_by_track=reid_scores_by_track if reid_scores_by_track else None
        )

        # Apply Reciprocal Rank Fusion (RRF) on object results if enabled
        if enable_rrf and reid_ranks_by_track:
            effective_reid_weight = 0.6 if entity_type == "person" else reid_weight
            effective_clip_weight = 0.4 if entity_type == "person" else clip_weight
            for c_rank, match in enumerate(obj_results, start=1):
                t_id = match.get("track_id")
                clip_rrf = effective_clip_weight / (rrf_k + c_rank)
                r_rank = reid_ranks_by_track.get(t_id, candidate_k + 1)
                reid_rrf = effective_reid_weight / (rrf_k + r_rank)
                rrf_score = round(clip_rrf + reid_rrf, 5)

                reid_sim = reid_scores_by_track.get(t_id)
                match["rrf_score"] = rrf_score
                match["visual_similarity"] = match.get("semantic_score", match.get("similarity"))
                match["reid_similarity"] = round(reid_sim, 4) if reid_sim is not None else None
                match["composite_score"] = match.get("similarity")
                if "signal_breakdown" in match:
                    match["signal_breakdown"]["reid"] = round(reid_sim, 4) if reid_sim is not None else None
                    match["signal_breakdown"]["rrf"] = rrf_score
            obj_results = sorted(obj_results, key=lambda x: x.get("rrf_score", x["similarity"]), reverse=True)
        else:
            for match in obj_results:
                match["visual_similarity"] = match.get("semantic_score", match.get("similarity"))
                match["reid_similarity"] = None
                match["composite_score"] = match.get("similarity")

        for match in obj_results:
            match["match_type"] = "object"
            match["source_type"] = "object"
            match["diagnostic_label"] = "Object match"

        # Stage 3: Candidate Retrieval for Scene Keyframes (Full Frames)
        scene_results: List[Dict[str, Any]] = []
        if hasattr(self.vector, "scene_coll") and entity_type not in ["person", "region"]:
            scene_filters = {}
            if job_id:
                scene_filters["job_id"] = job_id
            if cam_id:
                scene_filters["camera_id"] = cam_id
                
            raw_scene_matches: List[Dict[str, Any]] = []
            if cameras_in_scope and not cam_id and len(cameras_in_scope) > 1:
                per_cam_k = max(20, candidate_k // len(cameras_in_scope))
                seen_s_ids = set()
                for cam in cameras_in_scope:
                    cam_sf = dict(scene_filters)
                    cam_sf["camera_id"] = cam["id"]
                    cam_sres = self.vector.query_scene_embeddings(
                        query_embedding=query_embedding,
                        top_k=per_cam_k,
                        filters=cam_sf
                    )
                    for sr in cam_sres:
                        if sr["id"] not in seen_s_ids:
                            seen_s_ids.add(sr["id"])
                            raw_scene_matches.append(sr)
            else:
                raw_scene_matches = self.vector.query_scene_embeddings(
                    query_embedding=query_embedding,
                    top_k=candidate_k,
                    filters=scene_filters if scene_filters else None
                )

            # Inflate scene candidates
            camera_map = {c["id"]: c for c in cameras_in_scope} if cameras_in_scope else {}
            
            # Sort scene matches by similarity desc
            raw_scene_matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            seen_scene_windows: Dict[str, List[float]] = {}  # camera_id -> list of timestamps
            
            for s_match in raw_scene_matches:
                s_sim = float(s_match["similarity"])
                s_meta = s_match["metadata"]
                
                # Check match threshold
                if strict_gate and s_sim < IMAGE_MATCH_THRESHOLD:
                    low_cand = {
                        "similarity": round(s_sim, 4),
                        "semantic_score": round(s_sim, 4),
                        "entity_type": "scene",
                        "match_type": "scene",
                        "job_id": s_meta.get("job_id"),
                        "camera_id": s_meta.get("camera_id"),
                        "timestamp": float(s_meta.get("timestamp", 0.0)),
                        "crop_path": s_meta.get("source_frame_path"),
                        "is_low_confidence": True,
                        "status": "BELOW_THRESHOLD",
                        "diagnostic_label": "Below Relevance Threshold"
                    }
                    below_threshold.append(low_cand)
                    continue

                c_id = s_meta.get("camera_id", "")
                t_stamp = float(s_meta.get("timestamp", 0.0))
                
                # Deduplicate temporal keyframes within 2.0s on the same camera
                if c_id not in seen_scene_windows:
                    seen_scene_windows[c_id] = []
                if any(abs(t_stamp - past_t) < 2.0 for past_t in seen_scene_windows[c_id]):
                    continue
                seen_scene_windows[c_id].append(t_stamp)

                # Camera metadata lookup
                if c_id not in camera_map and s_meta.get("job_id"):
                    job_cams = self.sqlite.get_cameras_by_job(s_meta["job_id"])
                    for jc in job_cams:
                        camera_map[jc["id"]] = jc

                cam_info = camera_map.get(c_id, {})
                video_source = cam_info.get("source_path") or s_meta.get("source_video", "")
                camera_name = cam_info.get("name") or c_id

                scene_item = {
                    "similarity": round(s_sim, 4),
                    "semantic_score": round(s_sim, 4),
                    "visual_similarity": round(s_sim, 4),
                    "reid_similarity": None,
                    "composite_score": round(s_sim, 4),
                    "attribute_score": 1.0,
                    "quality_score": 1.0,
                    "entity_type": "scene",
                    "source_type": "scene",
                    "match_type": "scene",
                    "diagnostic_label": "Scene keyframe match",
                    "job_id": s_meta.get("job_id"),
                    "camera_id": c_id,
                    "camera_name": camera_name,
                    "timestamp": t_stamp,
                    "frame_idx": s_meta.get("frame_idx", 0),
                    "track_id": None,
                    "confidence": 0.95,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "crop_path": s_meta.get("source_frame_path"),
                    "video_source": video_source,
                    "plate_text": None,
                    "video_clip_path": None,
                    "signal_breakdown": {
                        "visual": round(s_sim, 4),
                        "reid": None,
                        "face": None,
                        "color": 1.0,
                        "quality": 1.0,
                        "temporal": 0.0,
                        "camera": "PASS",
                        "ocr": None,
                        "composite": round(s_sim, 4)
                    }
                }
                scene_results.append(scene_item)

        # Stage 3b: Candidate Retrieval for Spatial Regions / Tiles
        region_results: List[Dict[str, Any]] = []
        if hasattr(self.vector, "region_coll") and entity_type not in ["person", "scene"]:
            region_filters = {}
            if job_id:
                region_filters["job_id"] = job_id
            if cam_id:
                region_filters["camera_id"] = cam_id

            raw_region_matches: List[Dict[str, Any]] = []
            if cameras_in_scope and not cam_id and len(cameras_in_scope) > 1:
                per_cam_rk = max(30, candidate_k // len(cameras_in_scope))
                seen_r_ids = set()
                for cam in cameras_in_scope:
                    cam_rf = dict(region_filters)
                    cam_rf["camera_id"] = cam["id"]
                    cam_rres = self.vector.query_region_embeddings(
                        query_embedding=query_embedding,
                        top_k=per_cam_rk,
                        filters=cam_rf
                    )
                    for rr in cam_rres:
                        if rr["id"] not in seen_r_ids:
                            seen_r_ids.add(rr["id"])
                            raw_region_matches.append(rr)
            else:
                raw_region_matches = self.vector.query_region_embeddings(
                    query_embedding=query_embedding,
                    top_k=candidate_k,
                    filters=region_filters if region_filters else None
                )

            # Sort region matches by similarity desc
            raw_region_matches.sort(key=lambda x: x["similarity"], reverse=True)

            camera_map = {c["id"]: c for c in cameras_in_scope} if cameras_in_scope else {}
            seen_region_windows: Dict[str, List[Tuple[float, int]]] = {}  # camera_id -> list of (timestamp, tile_idx)

            for r_match in raw_region_matches:
                r_sim = float(r_match["similarity"])
                r_meta = r_match["metadata"]

                # Parse bbox
                bbox = [0.0, 0.0, 0.0, 0.0]
                if "bbox_x1" in r_meta and "bbox_y1" in r_meta:
                    bbox = [
                        float(r_meta["bbox_x1"]),
                        float(r_meta["bbox_y1"]),
                        float(r_meta["bbox_x2"]),
                        float(r_meta["bbox_y2"]),
                    ]
                elif "bbox" in r_meta and isinstance(r_meta["bbox"], str):
                    try:
                        bbox = [float(v) for v in r_meta["bbox"].split(",")]
                    except Exception:
                        bbox = [0.0, 0.0, 0.0, 0.0]

                # Check match threshold
                if strict_gate and r_sim < IMAGE_MATCH_THRESHOLD:
                    low_cand = {
                        "similarity": round(r_sim, 4),
                        "semantic_score": round(r_sim, 4),
                        "entity_type": "region",
                        "match_type": "region",
                        "job_id": r_meta.get("job_id"),
                        "camera_id": r_meta.get("camera_id"),
                        "timestamp": float(r_meta.get("timestamp", 0.0)),
                        "crop_path": r_meta.get("crop_path"),
                        "bbox": bbox,
                        "is_low_confidence": True,
                        "status": "BELOW_THRESHOLD",
                        "diagnostic_label": "Below Relevance Threshold"
                    }
                    below_threshold.append(low_cand)
                    continue

                c_id = r_meta.get("camera_id", "")
                t_stamp = float(r_meta.get("timestamp", 0.0))
                tile_idx = int(r_meta.get("tile_idx", 0))

                # Deduplicate temporal tiles within 2.0s on the same camera and same tile_idx
                if c_id not in seen_region_windows:
                    seen_region_windows[c_id] = []
                if any(abs(t_stamp - past_t) < 2.0 and tile_idx == past_tile for past_t, past_tile in seen_region_windows[c_id]):
                    continue
                seen_region_windows[c_id].append((t_stamp, tile_idx))

                # Camera metadata lookup
                if c_id not in camera_map and r_meta.get("job_id"):
                    job_cams = self.sqlite.get_cameras_by_job(r_meta["job_id"])
                    for jc in job_cams:
                        camera_map[jc["id"]] = jc

                cam_info = camera_map.get(c_id, {})
                video_source = cam_info.get("source_path") or r_meta.get("source_video", "")
                camera_name = cam_info.get("name") or c_id

                region_item = {
                    "similarity": round(r_sim, 4),
                    "semantic_score": round(r_sim, 4),
                    "visual_similarity": round(r_sim, 4),
                    "reid_similarity": None,
                    "composite_score": round(r_sim, 4),
                    "attribute_score": 1.0,
                    "quality_score": 1.0,
                    "entity_type": "region",
                    "source_type": "region",
                    "match_type": "region",
                    "diagnostic_label": "Region tile match",
                    "job_id": r_meta.get("job_id"),
                    "camera_id": c_id,
                    "camera_name": camera_name,
                    "timestamp": t_stamp,
                    "frame_idx": r_meta.get("frame_idx", 0),
                    "track_id": None,
                    "confidence": 0.95,
                    "bbox": bbox,
                    "crop_path": r_meta.get("crop_path"),
                    "source_frame_path": r_meta.get("source_frame_path"),
                    "video_source": video_source,
                    "plate_text": None,
                    "video_clip_path": None,
                    "signal_breakdown": {
                        "visual": round(r_sim, 4),
                        "reid": None,
                        "face": None,
                        "color": 1.0,
                        "quality": 1.0,
                        "temporal": 0.0,
                        "camera": "PASS",
                        "ocr": None,
                        "composite": round(r_sim, 4)
                    }
                }
                region_results.append(region_item)

        # Stage 4: Candidate Fusion & Re-ranking
        all_results = obj_results + region_results + scene_results
        if enable_rrf and reid_ranks_by_track:
            all_results.sort(key=lambda x: x.get("rrf_score", x["similarity"]), reverse=True)
        else:
            all_results.sort(key=lambda x: x["similarity"], reverse=True)

        diagnostics["post_gate_count"] = len(all_results)
        diagnostics["object_matches_count"] = len(obj_results)
        diagnostics["region_matches_count"] = len(region_results)
        diagnostics["scene_matches_count"] = len(scene_results)
        diagnostics["below_threshold_count"] = len(below_threshold)
        diagnostics["gate_threshold"] = FINAL_SCORE_MIN_IMAGE if strict_gate else 0.0
        
        # Determine Search State
        if not all_results and below_threshold:
            diagnostics["search_state"] = "BELOW_THRESHOLD"
        elif all_results:
            diagnostics["search_state"] = "RESULTS"
        else:
            diagnostics["search_state"] = "NO_CANDIDATES"
            
        LOGGER.info("IMAGE_SEARCH_TRACE state=%s objects=%d regions=%d scenes=%d %s",
                    diagnostics["search_state"], len(obj_results), len(region_results), len(scene_results), diagnostics)

        self.last_below_threshold_results = below_threshold[:top_k]
        diversified_results = cluster_and_diversify_occurrences(
            all_results,
            temporal_window_sec=12.0,
            top_k=top_k
        )
        return diversified_results


    def search_by_face(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Performs a facial recognition search matching face crop FaceNet vectors."""
        vector_matches = self.vector.query_face_embeddings(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        from config import FACE_MATCH_THRESHOLD
        return self._inflate_metadata_and_re_rank(
            vector_matches,
            query_colors=None,
            is_face=True,
            threshold=FACE_MATCH_THRESHOLD
        )[:top_k]

    def search_by_plate(
        self,
        query_text: str,
        camera_filter: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Performs an exact or wildcard search on license plate OCR strings from SQLite."""
        db_matches = self.sqlite.search_plates(query_text, camera_filter, job_id)
        inflated = []
        
        for match in db_matches:
            # Reconstruct bbox from comma-separated string if present
            bbox = []
            
            # Fetch detections associated with this track to get exact detail
            detections = self.sqlite.get_detections_by_track(match["id"])
            if detections:
                det = detections[0] # Take first detection keyframe
                bbox = [det["bbox_x1"], det["bbox_y1"], det["bbox_x2"], det["bbox_y2"]]
                crop_path = det["crop_path"]
                timestamp = det["timestamp"]
            else:
                crop_path = match.get("crop_path", "")
                timestamp = match.get("det_time", 0.0)
                bbox = [0.0, 0.0, 0.0, 0.0]

            camera_rec = self.sqlite.get_cameras_by_job(match["job_id"])
            video_source = camera_rec[0]["source_path"] if camera_rec else ""
            
            clip_path = self.media.get_clip_path(match["job_id"], match["camera_id"], match["track_id"])
            
            inflated.append({
                "similarity": 1.0,  # Exact search holds absolute similarity
                "entity_type": match["entity_type"],
                "job_id": match["job_id"],
                "camera_id": match["camera_id"],
                "timestamp": timestamp,
                "track_id": match["id"],
                "bbox": bbox,
                "crop_path": crop_path,
                "video_source": video_source,
                "plate_text": match["plate_text"],
                "video_clip_path": str(clip_path) if clip_path.exists() else None
            })
            
        return inflated

    def search_saved_person(
        self,
        saved_person_id: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Performs facial re-identification using a pre-saved person's face profile."""
        import sqlite3
        with self.sqlite.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM saved_persons WHERE id = ?", (saved_person_id,))
            person = cursor.fetchone()
            
        if not person:
            raise ValueError(f"Saved person with ID {saved_person_id} does not exist.")
            
        # Get face embedding ID
        face_emb_id = person["face_embedding_id"]
        
        # Look up embedding vector from ChromaDB face collection
        stored_emb = self.vector.face_coll.get(ids=[face_emb_id], include=["embeddings"])
        embeddings = stored_emb.get("embeddings")
        if embeddings is None or len(embeddings) == 0 or len(embeddings[0]) == 0:
            raise RuntimeError(f"Face embedding for saved person {saved_person_id} not found in vector store.")
            
        # Run visual query against all face vectors
        return self.search_by_face(embeddings[0], filters, top_k)

    # ==============================================================================
    # Internal inflation and scoring helper
    # ==============================================================================

    def _inflate_metadata_and_re_rank(
        self,
        vector_matches: List[Dict[str, Any]],
        query_colors: Optional[List[str]] = None,
        is_face: bool = False,
        query_action: Optional[str] = None,
        threshold: float = 0.0,
        final_score_threshold: float = 0.0,
        expected_entity_type: Optional[str] = None,
        reid_scores_by_track: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Combines ChromaDB vector distances with SQLite metadata and performs transparent hybrid scoring & track-level deduplication."""
        if is_face:
            best_matches_by_track = {}
            
            for match in vector_matches:
                meta = match["metadata"]
                track_uuid = meta.get("track_id")
                if not track_uuid:
                    continue
                    
                similarity = float(match["similarity"])
                # Enforce match threshold (Section 8 & 9)
                if similarity < threshold:
                    continue
                    
                # Query SQLite for full track and detection info
                track = self.sqlite.get_track(track_uuid)
                if not track:
                    continue
                    
                # Query camera video source
                camera_recs = self.sqlite.get_cameras_by_job(track["job_id"])
                camera_map = {c["id"]: c for c in camera_recs}
                video_source = camera_map[track["camera_id"]]["source_path"] if track["camera_id"] in camera_map else ""
                
                # Retrieve face metadata from collection
                face_quality = meta.get("face_quality", 0.0)
                face_crop_path = meta.get("face_crop_path") or ""
                
                det_id = meta.get("detection_id")
                detections = self.sqlite.get_detections_by_track(track_uuid)
                
                # Default to first detection if specific detection_id is missing or not found
                target_det = None
                if det_id:
                    for d in detections:
                        if d["id"] == det_id:
                            target_det = d
                            break
                if not target_det and detections:
                    target_det = detections[0]
                    
                if not target_det:
                    continue
                    
                bbox = [
                    target_det["bbox_x1"],
                    target_det["bbox_y1"],
                    target_det["bbox_x2"],
                    target_det["bbox_y2"]
                ]
                
                # Fetch generated context clip path
                clip_path = self.media.get_clip_path(track["job_id"], track["camera_id"], track["track_id"])
                
                result_item = {
                    "similarity": round(similarity, 4),
                    "semantic_score": round(similarity, 4),
                    "attribute_score": 1.0,
                    "quality_score": round(face_quality, 4),
                    "entity_type": "face",
                    "job_id": track["job_id"],
                    "camera_id": track["camera_id"],
                    "timestamp": target_det["timestamp"],
                    "track_id": track_uuid,
                    "confidence": round(float(target_det.get("confidence", 0.90)), 2),
                    "bbox": bbox,
                    "crop_path": face_crop_path or target_det["crop_path"],
                    "video_source": video_source,
                    "plate_text": track["plate_text"],
                    "video_clip_path": str(clip_path) if clip_path.exists() else None,
                    "face_quality": face_quality,
                    "model_name": meta.get("model_name", "InceptionResnetV1_vggface2"),
                    "model_version": meta.get("model_version", "1.0")
                }
                
                # Group by track_uuid, keep the highest face similarity (Section 11)
                if track_uuid not in best_matches_by_track or similarity > best_matches_by_track[track_uuid]["similarity"]:
                    best_matches_by_track[track_uuid] = result_item
                    
            # Sort desc by similarity
            sorted_results = sorted(best_matches_by_track.values(), key=lambda x: x["similarity"], reverse=True)
            return sorted_results
            
        from config import SEARCH_WEIGHT_SEMANTIC, SEARCH_WEIGHT_ATTRIBUTE, SEARCH_WEIGHT_QUALITY
        
        inflated_results = []
        best_matches_by_track = {}
        
        for match in vector_matches:
            meta = match["metadata"]
            track_uuid = meta.get("track_id")
            if not track_uuid:
                continue
                
            # --- Scoring & Re-ranking Algorithm ---
            # 1. Semantic Similarity Score (from ChromaDB)
            semantic_score = float(match["similarity"])
            reid_sim = reid_scores_by_track.get(track_uuid) if reid_scores_by_track else None

            # For person search with Re-ID, allow strong Re-ID similarity to satisfy threshold
            effective_sim = max(semantic_score, reid_sim) if reid_sim is not None else semantic_score
            
            # Enforce verified match threshold to prevent low-similarity false positives (Section 13)
            if effective_sim < threshold:
                continue
                
            # Query SQLite for full track and detection info
            track = self.sqlite.get_track(track_uuid)
            if not track:
                continue
            
            # Hard rejection: wrong entity type (safety net beyond ChromaDB filter)
            if expected_entity_type:
                if expected_entity_type == "person" and track["entity_type"] != "person":
                    continue
                elif expected_entity_type == "vehicle" and track["entity_type"] != "vehicle":
                    continue
                elif expected_entity_type == "object" and track["entity_type"] == "person":
                    continue
                
            # Query camera video source
            camera_recs = self.sqlite.get_cameras_by_job(track["job_id"])
            camera_map = {c["id"]: c for c in camera_recs}
            video_source = camera_map[track["camera_id"]]["source_path"] if track["camera_id"] in camera_map else ""
            
            # Fetch specific crop and coordinates from detections
            det_id = meta.get("detection_id")
            detections = self.sqlite.get_detections_by_track(track_uuid)
            
            # Default to first detection if specific detection_id is missing or not found
            target_det = None
            if det_id:
                for d in detections:
                    if d["id"] == det_id:
                        target_det = d
                        break
            if not target_det and detections:
                target_det = detections[0]
                
            if not target_det:
                continue
                
            # 2. Action / Motion Verification (Section 10)
            action_multiplier = 1.0
            if query_action and target_det["entity_type"] == "person":
                if len(detections) >= 2:
                    sorted_dets = sorted(detections, key=lambda d: d["timestamp"])
                    d1, d2 = sorted_dets[0], sorted_dets[-1]
                    x1 = (d1["bbox_x1"] + d1["bbox_x2"]) / 2.0
                    y1 = (d1["bbox_y1"] + d1["bbox_y2"]) / 2.0
                    x2 = (d2["bbox_x1"] + d2["bbox_x2"]) / 2.0
                    y2 = (d2["bbox_y1"] + d2["bbox_y2"]) / 2.0
                    
                    dt = max(0.1, d2["timestamp"] - d1["timestamp"])
                    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    speed = distance / dt
                    
                    aspects = [
                        (d["bbox_x2"] - d["bbox_x1"]) / max(1.0, d["bbox_y2"] - d["bbox_y1"])
                        for d in detections
                    ]
                    avg_aspect = sum(aspects) / len(aspects)
                else:
                    speed = 0.0
                    avg_aspect = 0.4
                    
                # Action Classifier Rules
                if query_action == "running":
                    if speed < 12.0:
                        action_multiplier = 0.0
                elif query_action == "walking":
                    if speed < 3.0 or speed >= 15.0:
                        action_multiplier = 0.0
                elif query_action == "standing":
                    if speed >= 4.0 or avg_aspect >= 0.65:
                        action_multiplier = 0.0
                elif query_action == "sitting":
                    if speed >= 4.0 or avg_aspect < 0.65:
                        action_multiplier = 0.0
                        
            # Filter out tracks that do not match spatial-temporal pattern rules
            if action_multiplier == 0.0:
                continue
                
            bbox = [
                target_det["bbox_x1"],
                target_det["bbox_y1"],
                target_det["bbox_x2"],
                target_det["bbox_y2"]
            ]
            
            # Fetch generated context clip path
            t_id_val = track.get("track_id") or track.get("id") or track_uuid
            clip_path = self.media.get_clip_path(track["job_id"], track["camera_id"], t_id_val)
            
            # 3. Attribute / Color Agreement Score
            if query_colors:
                candidate_colors = []
                if target_det["entity_type"] == "person":
                    candidate_colors = [
                        target_det.get("dominant_upper_color"),
                        target_det.get("secondary_upper_color"),
                        target_det.get("dominant_lower_color"),
                        target_det.get("secondary_lower_color")
                    ]
                else:
                    candidate_colors = [
                        target_det.get("dominant_upper_color"),
                        target_det.get("secondary_upper_color")
                    ]
                candidate_colors = [c for c in candidate_colors if c]
                
                has_color_match = False
                for qc in query_colors:
                    if qc in candidate_colors:
                        has_color_match = True
                        break
                        
                conf = target_det.get("color_confidence", 0.0) or 0.0
                if conf < 0.05:
                    # Grayscale / IR / Monochrome footage: chromatic color is absent.
                    # Rely purely on the visual information actually present in the source frame.
                    attribute_score = 1.0
                    weight_semantic = 0.90
                    weight_attr = 0.0
                    weight_quality = 0.05
                elif has_color_match:
                    attribute_score = 0.5 + 0.5 * conf  # 0.50–1.00: reward matching color
                    weight_semantic = SEARCH_WEIGHT_SEMANTIC
                    weight_attr = SEARCH_WEIGHT_ATTRIBUTE
                    weight_quality = SEARCH_WEIGHT_QUALITY
                else:
                    # Confident color mismatch on color footage → heavy penalty (near-zero)
                    # Low confidence → mild penalty (uncertain, don't punish too hard)
                    attribute_score = max(0.0, 0.20 * (1.0 - conf))  # 0.00–0.20
                    weight_semantic = SEARCH_WEIGHT_SEMANTIC
                    weight_attr = SEARCH_WEIGHT_ATTRIBUTE
                    weight_quality = SEARCH_WEIGHT_QUALITY
            else:
                attribute_score = 1.0 # Neutral if no colors queried
                weight_semantic = 0.90
                weight_attr = 0.0
                weight_quality = 0.05
                
            # 4. Quality Score
            quality_score = target_det.get("quality_score", 0.0) or 0.0
            
            # 5. Temporal / Track Consistency Boost
            # Tracks with more detections are more likely to be true positives than transient single-frame noise.
            track_length = len(detections)
            temporal_boost = min(0.05, track_length * 0.002)
            
            # Combined Transparent Score
            final_score = (
                semantic_score * weight_semantic +
                attribute_score * weight_attr +
                quality_score * weight_quality +
                temporal_boost
            )
            
            result_item = {
                "similarity": round(final_score, 4),
                "semantic_score": round(semantic_score, 4),
                "attribute_score": round(attribute_score, 4),
                "quality_score": round(quality_score, 4),
                "entity_type": "face" if is_face else track["entity_type"],
                "job_id": track["job_id"],
                "camera_id": track["camera_id"],
                "timestamp": target_det["timestamp"],
                "track_id": track_uuid,
                "confidence": round(float(target_det.get("confidence", 0.85)), 2),
                "bbox": bbox,
                "crop_path": target_det["crop_path"],
                "video_source": video_source,
                "plate_text": track["plate_text"],
                "video_clip_path": str(clip_path) if clip_path.exists() else None,
                "signal_breakdown": {
                    "visual": round(semantic_score, 4),
                    "reid": None,
                    "face": None,
                    "color": round(attribute_score, 4),
                    "quality": round(quality_score, 4),
                    "temporal": round(temporal_boost, 4),
                    "camera": "PASS",
                    "ocr": track.get("plate_text"),
                    "composite": round(final_score, 4)
                },
                "dominant_upper_color": target_det.get("dominant_upper_color"),
                "secondary_upper_color": target_det.get("secondary_upper_color"),
                "dominant_lower_color": target_det.get("dominant_lower_color"),
                "secondary_lower_color": target_det.get("secondary_lower_color"),
                "color_confidence": target_det.get("color_confidence")
            }
            
            # Post-rerank relevance gate: reject weak composite matches
            if final_score_threshold > 0 and final_score < final_score_threshold:
                continue
            
            if track_uuid not in best_matches_by_track or final_score > best_matches_by_track[track_uuid]["similarity"]:
                best_matches_by_track[track_uuid] = result_item
                
        sorted_results = sorted(best_matches_by_track.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_results

    def _inflate_metadata_and_re_rank_with_below_gate(
        self,
        vector_matches: List[Dict[str, Any]],
        query_colors: Optional[List[str]] = None,
        is_face: bool = False,
        query_action: Optional[str] = None,
        threshold: float = 0.0,
        final_score_threshold: float = 0.0,
        expected_entity_type: Optional[str] = None,
        reid_scores_by_track: Optional[Dict[str, float]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Separates candidates into passed results and below-threshold (low confidence) results."""
        all_candidates = self._inflate_metadata_and_re_rank(
            vector_matches=vector_matches,
            query_colors=query_colors,
            is_face=is_face,
            query_action=query_action,
            threshold=0.0,
            final_score_threshold=0.0,
            expected_entity_type=expected_entity_type,
            reid_scores_by_track=reid_scores_by_track
        )
        passed = []
        below = []
        for cand in all_candidates:
            # Check if candidate satisfies visual similarity or Re-ID similarity threshold
            t_uuid = cand.get("track_id")
            t_reid_sim = reid_scores_by_track.get(t_uuid) if (reid_scores_by_track and t_uuid) else cand.get("reid_similarity")
            effective_sim = max(cand.get("semantic_score", 0.0), t_reid_sim or 0.0)

            if cand["similarity"] >= final_score_threshold and effective_sim >= threshold:
                passed.append(cand)
            elif t_reid_sim is not None and t_reid_sim >= threshold:
                passed.append(cand)
            else:
                low_cand = dict(cand)
                low_cand["is_low_confidence"] = True
                low_cand["status"] = "BELOW_THRESHOLD"
                low_cand["diagnostic_label"] = "Below Relevance Threshold"
                below.append(low_cand)

        return passed, below


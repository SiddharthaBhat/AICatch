"""Unified Search Abstraction Layer for AICatch.

Combines vector query results from ChromaDB with relational metadata
from SQLite to provide a structured API for the frontend and matcher.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from storage.media_storage import MediaStorage

LOGGER = logging.getLogger(__name__)

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
        face_status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Performs a visual query-by-image search matching image crop CLIP vectors, with optional internal face-assisted re-ranking."""
        search_filters = dict(filters or {})
        if entity_type is not None and "entity_type" not in search_filters:
            search_filters["entity_type"] = entity_type
            
        # Stage 1: Candidate Retrieval
        candidate_k = max(100, top_k * 4)
        vector_matches = self.vector.query_visual_embeddings(
            query_embedding=query_embedding,
            top_k=candidate_k,
            filters=search_filters
        )
        
        # Stage 2: Inflate and Re-rank (Visual matches only first)
        from config import IMAGE_MATCH_THRESHOLD, IMAGE_VISUAL_WEIGHT, IMAGE_FACE_WEIGHT, FACE_MATCH_THRESHOLD, FINAL_SCORE_MIN_IMAGE
        re_ranked = self._inflate_metadata_and_re_rank(
            vector_matches,
            query_colors=None,
            is_face=False,
            threshold=IMAGE_MATCH_THRESHOLD,
            final_score_threshold=FINAL_SCORE_MIN_IMAGE,
            expected_entity_type=entity_type
        )
        
        # Stage 3: Optional Face-assisted Hybrid Re-ranking
        # Only use face signal if face was detected and is of GOOD or PARTIAL quality
        use_face = face_embedding is not None and face_status in ["GOOD_FACE", "PARTIAL_FACE"]
        
        if use_face and re_ranked:
            # 1. Fetch face similarity scores for the query face
            face_scores = self.search_face_internal(face_embedding, filters, top_k=100)
            
            # 2. Identify which retrieved tracks have indexed face crops
            track_uuids = {m["track_id"] for m in re_ranked if m.get("track_id")}
            face_tracks = set()
            if track_uuids:
                try:
                    # Retrieve all matching track_id face crops to check index presence
                    face_res = self.vector.face_coll.get(
                        where={"track_id": {"$in": list(track_uuids)}} if len(track_uuids) > 1 else {"track_id": list(track_uuids)[0]},
                        include=["metadatas"]
                    )
                    for meta in face_res.get("metadatas", []):
                        t_id = meta.get("track_id")
                        if t_id:
                            face_tracks.add(t_id)
                except Exception as db_err:
                    LOGGER.debug(f"Failed to query face presence: {db_err}")
            
            # 3. Calculate final hybrid scores
            for match in re_ranked:
                track_uuid = match.get("track_id")
                S_vis = match["similarity"]
                
                if track_uuid in face_tracks:
                    # Track has face crops: apply weighted visual + face similarity
                    S_face = face_scores.get(track_uuid, 0.0)
                    
                    # Section 11: Weak/mismatched face shouldn't override strong visual.
                    # We only combine if face similarity is above threshold, otherwise face similarity is 0.0 (penalizing mismatch).
                    if S_face < FACE_MATCH_THRESHOLD:
                        S_face = 0.0
                        
                    final_score = IMAGE_VISUAL_WEIGHT * S_vis + IMAGE_FACE_WEIGHT * S_face
                    match["similarity"] = round(final_score, 4)
                    match["face_score"] = round(S_face, 4)
                    match["match_type"] = "face_assisted"
                else:
                    # Neutral fallback: track has no indexed face crops
                    match["similarity"] = S_vis
                    match["face_score"] = 0.0
                    match["match_type"] = "visual_only"
            
            # Re-sort descending by final hybrid similarity score
            re_ranked = sorted(re_ranked, key=lambda x: x["similarity"], reverse=True)
        else:
            # Fallback/Visual-only indicators
            for match in re_ranked:
                match["match_type"] = "visual_only"
                match["face_score"] = 0.0
                
        return re_ranked[:top_k]

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
        expected_entity_type: Optional[str] = None
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
            
            # Enforce verified match threshold to prevent low-similarity false positives (Section 13)
            if semantic_score < threshold:
                continue
                
            # Query SQLite for full track and detection info
            track = self.sqlite.get_track(track_uuid)
            if not track:
                continue
            
            # Hard rejection: wrong entity type (safety net beyond ChromaDB filter)
            if expected_entity_type and track["entity_type"] != expected_entity_type:
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
            clip_path = self.media.get_clip_path(track["job_id"], track["camera_id"], track["track_id"])
            
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
                if has_color_match:
                    attribute_score = 0.5 + 0.5 * conf  # 0.50–1.00: reward matching color
                else:
                    # Confident color mismatch → heavy penalty (near-zero)
                    # Low confidence → mild penalty (uncertain, don't punish too hard)
                    attribute_score = max(0.0, 0.20 * (1.0 - conf))  # 0.00–0.20
            else:
                attribute_score = 1.0 # Neutral if no colors queried
                
            # 4. Quality Score
            quality_score = target_det.get("quality_score", 0.0) or 0.0
            
            # 5. Temporal / Track Consistency Boost
            # Tracks with more detections are more likely to be true positives than transient single-frame noise.
            track_length = len(detections)
            temporal_boost = min(0.05, track_length * 0.002)
            
            # Combined Transparent Score
            final_score = (
                semantic_score * SEARCH_WEIGHT_SEMANTIC +
                attribute_score * SEARCH_WEIGHT_ATTRIBUTE +
                quality_score * SEARCH_WEIGHT_QUALITY +
                temporal_boost
            )
            
            # Post-rerank relevance gate: reject weak composite matches
            if final_score_threshold > 0 and final_score < final_score_threshold:
                continue
            
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
                
                # Expose colors for transparency in UI
                "dominant_upper_color": target_det.get("dominant_upper_color"),
                "secondary_upper_color": target_det.get("secondary_upper_color"),
                "dominant_lower_color": target_det.get("dominant_lower_color"),
                "secondary_lower_color": target_det.get("secondary_lower_color"),
                "color_confidence": target_det.get("color_confidence")
            }
            
            # Group by track_uuid, keep the highest-scoring detection instance
            if track_uuid not in best_matches_by_track or final_score > best_matches_by_track[track_uuid]["similarity"]:
                best_matches_by_track[track_uuid] = result_item
                
        # Sort desc by final_score
        sorted_results = sorted(best_matches_by_track.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_results

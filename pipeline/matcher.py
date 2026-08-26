"""Multi-Session Active Query Matcher for AICatch.

Maintains active alert query rules and runs low-latency vectorized matches
against incoming inference results during video indexing.
"""

import logging
import threading
from typing import Dict, Any, List, Optional, Set
import numpy as np

LOGGER = logging.getLogger(__name__)

class ActiveQuery:
    """Represents a registered search rule for live matching."""
    def __init__(
        self,
        query_id: str,
        session_id: str,
        qtype: str,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        threshold: float = 0.70,
        job_id: Optional[str] = None
    ):
        self.query_id = query_id
        self.job_id = job_id
        self.session_id = session_id
        self.qtype = qtype.lower().strip()  # "text" | "image" | "face" | "plate"
        self.query_text = query_text.upper().strip() if query_text else None
        self.threshold = threshold
        
        # Parsed entity and colors for text alerts
        self.target_entity_type = None
        self.target_colors = []
        if self.qtype == "text" and query_text:
            from storage.search import parse_query_text
            parsed = parse_query_text(query_text)
            self.target_entity_type = parsed["entity_type"]
            self.target_colors = parsed["colors"]
            
        # NumPy representation of query vector
        self.vector: Optional[np.ndarray] = None
        if query_embedding is not None:
            self.vector = np.array(query_embedding, dtype=np.float32)
            # Ensure vector is L2 normalized
            norm = np.linalg.norm(self.vector)
            if norm > 0:
                self.vector = self.vector / norm

class ActiveQueryMatcher:
    def __init__(self):
        # query_id -> ActiveQuery
        self.queries: Dict[str, ActiveQuery] = {}
        
        # Track duplicate alerts: {(query_id, track_uuid)}
        self.alerted_tracks: Set[tuple] = set()
        
        # Lock for thread safety
        self.lock = threading.Lock()

    def register_query(
        self,
        query_id: str,
        session_id: str,
        qtype: str,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        threshold: float = 0.70,
        job_id: Optional[str] = None
    ) -> None:
        """Register or update an active query rule."""
        query = ActiveQuery(
            query_id=query_id,
            session_id=session_id,
            qtype=qtype,
            query_text=query_text,
            query_embedding=query_embedding,
            threshold=threshold,
            job_id=job_id
        )
        with self.lock:
            self.queries[query_id] = query
            LOGGER.info(f"Registered active query {query_id} (Type: {qtype}, Session: {session_id}, Job: {job_id}).")

    def remove_query(self, query_id: str) -> None:
        """Deregister an active query rule."""
        with self.lock:
            if query_id in self.queries:
                self.queries.pop(query_id)
                
                # Cleanup alerts history for this query
                self.alerted_tracks = {item for item in self.alerted_tracks if item[0] != query_id}
                LOGGER.info(f"Removed active query {query_id}.")

    def evaluate_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluates a single crop's inference vectors and text against all active queries in parallel."""
        matched_queries = []
        track_uuid = result["track_uuid"]
        entity_type = result["entity_type"] # "person" or "vehicle"
        
        with self.lock:
            if not self.queries:
                return []
                
            for qid, query in list(self.queries.items()):
                # Scope check: do not evaluate queries from other jobs/investigations
                if query.job_id and query.job_id != result.get("job_id"):
                    continue
                # Avoid duplicate alerts for the same track
                if (qid, track_uuid) in self.alerted_tracks:
                    continue
                
                # 1. License Plate queries (Only apply to vehicles, match text string)
                if query.qtype == "plate":
                    if entity_type == "vehicle" and result.get("plate_text"):
                        # Wildcard or substring match
                        curr_plate = result["plate_text"].upper()
                        if query.query_text in curr_plate:
                            self.alerted_tracks.add((qid, track_uuid))
                            matched_queries.append({
                                "query_id": qid,
                                "session_id": query.session_id,
                                "similarity": 1.0
                            })
                            
                # 2. Face queries (Only apply to face embeddings on person crops)
                elif query.qtype == "face":
                    if entity_type == "person" and result.get("face_embedding") is not None:
                        if query.vector is not None:
                            # Calculate cosine similarity using NumPy dot product
                            incoming = np.array(result["face_embedding"], dtype=np.float32)
                            incoming_norm = np.linalg.norm(incoming)
                            if incoming_norm > 0:
                                incoming = incoming / incoming_norm
                                similarity = float(np.dot(query.vector, incoming))
                                if similarity >= query.threshold:
                                    self.alerted_tracks.add((qid, track_uuid))
                                    matched_queries.append({
                                        "query_id": qid,
                                        "session_id": query.session_id,
                                        "similarity": similarity
                                    })
                                    
                # 3. Multimodal Text / Visual Image queries (Uses standard CLIP embedding)
                elif query.qtype in ["text", "image"]:
                    if result.get("clip_embedding") is not None and query.vector is not None:
                        # Soft filter on entity type: if alert specifies person but detection is vehicle, skip
                        if query.qtype == "text" and query.target_entity_type and entity_type != query.target_entity_type:
                            continue
                            
                        incoming = np.array(result["clip_embedding"], dtype=np.float32)
                        incoming_norm = np.linalg.norm(incoming)
                        if incoming_norm > 0:
                            incoming = incoming / incoming_norm
                            semantic_score = float(np.dot(query.vector, incoming))
                            
                            # Calculate combined score for text queries
                            if query.qtype == "text":
                                from config import SEARCH_WEIGHT_SEMANTIC, SEARCH_WEIGHT_ATTRIBUTE, SEARCH_WEIGHT_QUALITY
                                
                                # Compute color matching
                                if query.target_colors:
                                    candidate_colors = []
                                    if entity_type == "person":
                                        candidate_colors = [
                                            result.get("dominant_upper_color"),
                                            result.get("secondary_upper_color"),
                                            result.get("dominant_lower_color"),
                                            result.get("secondary_lower_color")
                                        ]
                                    else:
                                        candidate_colors = [
                                            result.get("dominant_upper_color"),
                                            result.get("secondary_upper_color")
                                        ]
                                    candidate_colors = [c for c in candidate_colors if c]
                                    
                                    has_color_match = False
                                    for qc in query.target_colors:
                                        if qc in candidate_colors:
                                            has_color_match = True
                                            break
                                            
                                    conf = result.get("color_confidence", 0.0) or 0.0
                                    if has_color_match:
                                        attribute_score = 0.5 + 0.5 * conf
                                    else:
                                        attribute_score = 0.5 * (1.0 - conf)
                                else:
                                    attribute_score = 1.0
                                    
                                quality_score = result.get("quality_score", 0.0) or 0.0
                                
                                final_score = (
                                    semantic_score * SEARCH_WEIGHT_SEMANTIC +
                                    attribute_score * SEARCH_WEIGHT_ATTRIBUTE +
                                    quality_score * SEARCH_WEIGHT_QUALITY
                                )
                            else:
                                # For image alerts, just use semantic similarity
                                final_score = semantic_score
                                
                            if final_score >= query.threshold:
                                self.alerted_tracks.add((qid, track_uuid))
                                matched_queries.append({
                                    "query_id": qid,
                                    "session_id": query.session_id,
                                    "similarity": final_score
                                })
                                
        return matched_queries

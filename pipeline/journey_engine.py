"""CCTV Multi-Camera Journey Reconstruction Engine for AICatch.

Reconstructs chronological target journeys across camera feeds based on
visual re-ID embeddings, timestamps, temporal plausibility, and transition topologies.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase

LOGGER = logging.getLogger(__name__)

class JourneyEngine:
    def __init__(self, sqlite_db: SQLiteDatabase, vector_db: VectorDatabase):
        self.sqlite = sqlite_db
        self.vector = vector_db
        
        # Format: (camera_A, camera_B) -> (min_sec, max_sec)
        self.camera_topology: Dict[tuple, tuple] = {}

    def set_topology(self, camera_A: str, camera_B: str, min_sec: float, max_sec: float) -> None:
        """Configures expected transit times between two camera locations."""
        self.camera_topology[(camera_A, camera_B)] = (min_sec, max_sec)
        self.camera_topology[(camera_B, camera_A)] = (min_sec, max_sec)

    def get_transit_limits(self, camera_A: str, camera_B: str) -> tuple:
        """Returns the (min_sec, max_sec) travel limits for a transition."""
        if camera_A == camera_B:
            return (0.0, 30.0) # Within same camera
        return self.camera_topology.get((camera_A, camera_B), (2.0, 300.0))  # Default 2s to 5 mins

    def reconstruct_journey(
        self,
        target_track_uuid: str,
        similarity_threshold: float = 0.65,
        time_window_sec: float = 600.0
    ) -> Dict[str, Any]:
        """Reconstructs the chronological journey of a target track across cameras.
        
        Distinguishes between CONFIRMED (high confidence multi-signal matches)
        and POSSIBLE TRANSITIONS (weaker temporal or appearance evidence).
        """
        # 1. Fetch target track metadata
        target_track = self.sqlite.get_track(target_track_uuid)
        if not target_track:
            LOGGER.error(f"Target track {target_track_uuid} not found for journey reconstruction.")
            return {"target": None, "confirmed_journey": [], "possible_transitions": []}

        # Retrieve target track detections
        target_dets = self.sqlite.get_detections_by_track(target_track_uuid)
        target_crop = target_dets[0]["crop_path"] if target_dets else ""

        target_info = {
            "track_uuid": target_track_uuid,
            "track_id": target_track["track_id"],
            "camera_id": target_track["camera_id"],
            "start_time": target_track["start_time"],
            "end_time": target_track["end_time"],
            "crop_path": target_crop,
            "entity_type": target_track.get("entity_type", "person")
        }

        # 2. Get the visual embedding of the target track
        stored = self.vector.visual_coll.get(where={"track_id": target_track_uuid}, include=["embeddings"])
        embeddings = stored.get("embeddings")
        if embeddings is None or len(embeddings) == 0 or len(embeddings[0]) == 0:
            LOGGER.warning(f"No vector embedding found for track {target_track_uuid}.")
            return {
                "target": target_info,
                "confirmed_journey": [{
                    "track_uuid": target_track_uuid,
                    "camera_id": target_track["camera_id"],
                    "timestamp": target_track["start_time"],
                    "duration": target_track["end_time"] - target_track["start_time"],
                    "similarity": 1.0,
                    "confidence": 1.0,
                    "confidence_tier": "HIGH",
                    "reason": "Initial target observation",
                    "crop_path": target_crop
                }],
                "possible_transitions": []
            }
            
        target_emb = embeddings[0]
        
        # 3. Query similar visual embeddings in ChromaDB within the same job
        filters = {"job_id": target_track["job_id"]}
        matches = self.vector.query_visual_embeddings(
            query_embedding=target_emb,
            top_k=50,
            filters=filters
        )

        camera_recs = self.sqlite.get_cameras_by_job(target_track["job_id"])
        camera_map = {c["id"]: c for c in camera_recs}
        
        # 4. Filter and score candidate events
        candidate_events = []
        seen_tracks = {target_track_uuid}
        
        # Start node
        start_node = {
            "track_uuid": target_track_uuid,
            "camera_id": target_track["camera_id"],
            "camera_name": camera_map.get(target_track["camera_id"], {}).get("name", target_track["camera_id"]),
            "timestamp": target_track["start_time"],
            "duration": target_track["end_time"] - target_track["start_time"],
            "similarity": 1.0,
            "confidence": 1.0,
            "confidence_tier": "HIGH",
            "reason": "Origin of selected subject",
            "crop_path": target_crop
        }
        candidate_events.append(start_node)

        for match in matches:
            meta = match["metadata"]
            match_uuid = meta.get("track_id")
            if not match_uuid or match_uuid in seen_tracks:
                continue
                
            similarity = match["similarity"]
            if similarity < similarity_threshold:
                continue
                
            track = self.sqlite.get_track(match_uuid)
            if not track:
                continue
                
            seen_tracks.add(match_uuid)
            dets = self.sqlite.get_detections_by_track(match_uuid)
            crop = dets[0]["crop_path"] if dets else ""

            candidate_events.append({
                "track_uuid": match_uuid,
                "camera_id": track["camera_id"],
                "camera_name": camera_map.get(track["camera_id"], {}).get("name", track["camera_id"]),
                "timestamp": track["start_time"],
                "duration": track["end_time"] - track["start_time"],
                "similarity": round(similarity, 3),
                "crop_path": crop
            })

        # Sort all candidates chronologically by start timestamp
        candidate_events.sort(key=lambda x: x["timestamp"])
        
        confirmed_journey = [start_node]
        possible_transitions = []

        for i in range(1, len(candidate_events)):
            prev_event = confirmed_journey[-1]
            curr_event = candidate_events[i]
            
            prev_end = prev_event["timestamp"] + prev_event["duration"]
            transit_time = curr_event["timestamp"] - prev_end
            
            # Temporal overlap check
            if transit_time < -3.0:
                # Incompatible overlapping timeline on different cameras
                continue

            min_sec, max_sec = self.get_transit_limits(prev_event["camera_id"], curr_event["camera_id"])
            
            # Plausibility scoring
            time_penalty = 1.0
            tier = "HIGH"
            reason = ""

            if prev_event["camera_id"] != curr_event["camera_id"]:
                if transit_time < min_sec:
                    tier = "WEAK"
                    time_penalty = 0.5
                    reason = f"Transition too rapid ({transit_time:.1f}s, min required {min_sec}s)"
                elif transit_time > max_sec:
                    tier = "MEDIUM"
                    time_penalty = max(0.4, 1.0 - ((transit_time - max_sec) / time_window_sec))
                    reason = f"Appearance match with transit gap ({transit_time:.1f}s)"
                else:
                    tier = "HIGH" if curr_event["similarity"] >= 0.78 else "MEDIUM"
                    reason = f"Plausible transit ({transit_time:.1f}s) between cameras with {int(curr_event['similarity']*100)}% appearance match"
            else:
                # Same camera re-entry
                tier = "HIGH" if curr_event["similarity"] >= 0.75 else "MEDIUM"
                reason = f"Re-appearance in {curr_event['camera_name']} after {transit_time:.1f}s"

            composite_conf = curr_event["similarity"] * time_penalty
            curr_event["confidence"] = round(composite_conf, 3)
            curr_event["confidence_tier"] = tier
            curr_event["reason"] = reason
            curr_event["transit_time_sec"] = max(0.0, transit_time)

            if tier == "HIGH" and composite_conf >= 0.72:
                confirmed_journey.append(curr_event)
            elif composite_conf >= 0.45:
                possible_transitions.append(curr_event)

        return {
            "target": target_info,
            "confirmed_journey": confirmed_journey,
            "possible_transitions": possible_transitions
        }

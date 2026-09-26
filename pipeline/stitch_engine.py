"""Cross-Camera Evidence Stitching & Reconstruction Engine for AICatch.

Constructs verifiable, chronological forensic journeys across multi-camera CCTV feeds
from search observations, track histories, and Re-ID candidate matches.

Guarantees:
- Deterministic, explainable multi-signal continuity scoring (Re-ID, Visual, Temporal, Topology).
- No fake identity probabilities or fabricated observations.
- Preserves full forensic provenance (job, camera, timestamp, crop path, bbox, video).
- Gracefully identifies and flags insufficient cross-camera evidence.
"""

import logging
import uuid
import time
from typing import Dict, Any, List, Optional, Union, Tuple

from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from storage.media_storage import MediaStorage
from pipeline.stitch_models import (
    StitchObservation,
    StitchSegment,
    StitchTransition,
    StitchContinuitySignals,
    StitchProvenance,
    StitchResult
)
from pipeline.stitch_validator import StitchValidator

LOGGER = logging.getLogger(__name__)


class StitchEngine:
    """Core Cross-Camera Evidence Stitching & Journey Reconstruction Service."""

    def __init__(
        self,
        sqlite_db: SQLiteDatabase,
        vector_db: VectorDatabase,
        media_storage: Optional[MediaStorage] = None
    ):
        self.sqlite = sqlite_db
        self.vector = vector_db
        self.media = media_storage or MediaStorage()
        self.camera_graphs: Dict[str, Dict[str, List[str]]] = {}  # job_id -> camera_graph

    def set_job_camera_topology(self, job_id: str, camera_graph: Dict[str, List[str]]) -> None:
        """Configures optional directional camera adjacency graph for a job."""
        self.camera_graphs[job_id] = camera_graph

    def reconstruct_stitch(
        self,
        job_id: str,
        observations: Optional[List[Union[Dict[str, Any], StitchObservation]]] = None,
        search_results: Optional[List[Dict[str, Any]]] = None,
        target_track_uuid: Optional[str] = None,
        entity_type: str = "person",
        camera_graph: Optional[Dict[str, List[str]]] = None,
        min_similarity: float = 0.45,
        temporal_window_sec: float = 12.0,
        max_transit_sec: float = 600.0
    ) -> StitchResult:
        """Executes full evidence reconstruction from provided observations or target track ID."""
        stitch_id = f"stitch_{uuid.uuid4().hex[:12]}"
        entity_type_clean = entity_type.lower() if entity_type else "person"
        active_graph = camera_graph or self.camera_graphs.get(job_id)

        # 1. Resolve raw observations from inputs
        normalized_obs: List[StitchObservation] = []

        if observations:
            for idx, item in enumerate(observations):
                if isinstance(item, StitchObservation):
                    normalized_obs.append(item)
                elif isinstance(item, dict):
                    normalized_obs.append(StitchObservation.from_search_result(item, default_job_id=job_id, rank=idx))

        elif search_results:
            for idx, res in enumerate(search_results):
                normalized_obs.append(StitchObservation.from_search_result(res, default_job_id=job_id, rank=idx))

        elif target_track_uuid:
            normalized_obs = self._expand_candidates_for_track(
                job_id=job_id,
                target_track_uuid=target_track_uuid,
                entity_type=entity_type_clean,
                min_similarity=min_similarity
            )

        # 2. Strict cross-investigation isolation check
        valid_obs = StitchValidator.validate_observations_for_job(normalized_obs, expected_job_id=job_id)

        # Resolve camera display names
        cameras = self.sqlite.get_cameras_by_job(job_id) if hasattr(self.sqlite, "get_cameras_by_job") else []
        camera_name_map = {c["id"]: c.get("name", c["id"]) for c in cameras}

        for o in valid_obs:
            if not o.camera_name:
                o.camera_name = camera_name_map.get(o.camera_id, o.camera_id)

        if not valid_obs:
            return StitchResult(
                stitch_id=stitch_id,
                investigation_id=job_id,
                job_id=job_id,
                entity_type=entity_type_clean,
                status="insufficient_evidence",
                reason="No valid observations found for this investigation case."
            )

        # 3. Collapse intra-camera observations into distinct temporal segments
        segments = self._group_observations_into_segments(
            observations=valid_obs,
            job_id=job_id,
            temporal_window_sec=temporal_window_sec,
            camera_name_map=camera_name_map
        )

        if not segments:
            return StitchResult(
                stitch_id=stitch_id,
                investigation_id=job_id,
                job_id=job_id,
                entity_type=entity_type_clean,
                status="insufficient_evidence",
                reason="Unable to form valid temporal occurrence segments from candidate evidence."
            )

        # 4. Sort segments chronologically by start timestamp
        segments.sort(key=lambda s: s.start_time)

        # 5. Build cross-camera transitions and evaluate plausibility
        confirmed_segments: List[StitchSegment] = [segments[0]]
        transitions: List[StitchTransition] = []

        for i in range(1, len(segments)):
            prev_seg = confirmed_segments[-1]
            curr_seg = segments[i]

            is_plausible, signals = StitchValidator.validate_transition_plausibility(
                prev_seg=prev_seg,
                curr_seg=curr_seg,
                camera_graph=active_graph,
                max_transit_sec=max_transit_sec
            )

            if not is_plausible:
                # Physically impossible (e.g. negative transit between different cameras)
                LOGGER.info("STITCH_PRUNE: Pruned transition from %s to %s (%s)", prev_seg.camera_id, curr_seg.camera_id, signals.reason)
                continue

            # Update segment continuity signals
            curr_seg.continuity = signals

            transit_time = max(0.0, curr_seg.start_time - prev_seg.end_time)
            trans = StitchTransition(
                transition_id=f"trans_{uuid.uuid4().hex[:8]}",
                from_camera=prev_seg.camera_id,
                to_camera=curr_seg.camera_id,
                from_camera_name=prev_seg.camera_name,
                to_camera_name=curr_seg.camera_name,
                from_segment_id=prev_seg.segment_id,
                to_segment_id=curr_seg.segment_id,
                transit_time_sec=transit_time,
                score=signals.score,
                signals=signals
            )
            transitions.append(trans)
            confirmed_segments.append(curr_seg)

        # 6. Evaluate evidence sufficiency
        status, reason = StitchValidator.evaluate_evidence_sufficiency(
            segments=confirmed_segments,
            min_camera_diversity=2,
            min_continuity_score=min_similarity
        )

        unique_cameras = set(s.camera_id for s in confirmed_segments)
        total_duration = (confirmed_segments[-1].end_time - confirmed_segments[0].start_time) if confirmed_segments else 0.0

        # Build target summary if target track was provided or take initial segment
        target_summary = None
        if confirmed_segments:
            first_seg = confirmed_segments[0]
            target_summary = {
                "origin_camera_id": first_seg.camera_id,
                "origin_camera_name": first_seg.camera_name,
                "start_time": first_seg.start_time,
                "end_time": first_seg.end_time,
                "representative_crop": first_seg.representative_crop_path,
                "entity_type": entity_type_clean
            }

        return StitchResult(
            stitch_id=stitch_id,
            investigation_id=job_id,
            job_id=job_id,
            entity_type=entity_type_clean,
            status=status,
            reason=reason,
            target_summary=target_summary,
            segments=confirmed_segments,
            transitions=transitions,
            camera_diversity=len(unique_cameras),
            total_duration_sec=max(0.0, total_duration)
        )

    def _group_observations_into_segments(
        self,
        observations: List[StitchObservation],
        job_id: str,
        temporal_window_sec: float,
        camera_name_map: Dict[str, str]
    ) -> List[StitchSegment]:
        """Groups raw observations by camera and temporal proximity into occurrence segments."""
        # Partition observations by camera_id
        obs_by_camera: Dict[str, List[StitchObservation]] = {}
        for o in observations:
            obs_by_camera.setdefault(o.camera_id, []).append(o)

        segments: List[StitchSegment] = []

        for cam_id, cam_obs in obs_by_camera.items():
            # Sort chronologically by start timestamp
            cam_obs.sort(key=lambda x: x.timestamp)

            current_cluster: List[StitchObservation] = []
            cluster_start = 0.0
            cluster_end = 0.0

            for obs in cam_obs:
                obs_start = obs.timestamp
                obs_end = obs.end_timestamp if (obs.end_timestamp is not None and obs.end_timestamp >= obs_start) else obs_start

                if not current_cluster:
                    current_cluster = [obs]
                    cluster_start = obs_start
                    cluster_end = obs_end
                else:
                    # Check if within temporal grouping window
                    if obs_start - cluster_end <= temporal_window_sec:
                        current_cluster.append(obs)
                        cluster_end = max(cluster_end, obs_end)
                    else:
                        # Finalize previous segment
                        segments.append(self._create_segment(current_cluster, cluster_start, cluster_end, job_id, cam_id, camera_name_map))
                        current_cluster = [obs]
                        cluster_start = obs_start
                        cluster_end = obs_end

            if current_cluster:
                segments.append(self._create_segment(current_cluster, cluster_start, cluster_end, job_id, cam_id, camera_name_map))

        return segments

    def _create_segment(
        self,
        cluster: List[StitchObservation],
        start_time: float,
        end_time: float,
        job_id: str,
        cam_id: str,
        camera_name_map: Dict[str, str]
    ) -> StitchSegment:
        """Constructs a single StitchSegment from a list of clustered observations."""
        segment_id = f"seg_{uuid.uuid4().hex[:8]}"
        duration = max(0.5, end_time - start_time)

        # Select representative observation (highest combined score or first)
        rep_obs = max(cluster, key=lambda x: (x.combined_score or x.visual_similarity or 0.0))
        
        # Aggregate similarity signals
        valid_vis = [o.visual_similarity for o in cluster if o.visual_similarity is not None]
        valid_reid = [o.reid_similarity for o in cluster if o.reid_similarity is not None]
        
        avg_vis = float(sum(valid_vis) / len(valid_vis)) if valid_vis else rep_obs.visual_similarity
        avg_reid = float(sum(valid_reid) / len(valid_reid)) if valid_reid else rep_obs.reid_similarity

        score = rep_obs.combined_score or avg_vis or 1.0

        continuity = StitchContinuitySignals(
            score=score,
            reid_similarity=avg_reid,
            visual_similarity=avg_vis,
            temporal_score=1.0,
            transition_score=1.0,
            topology_available=False,
            reason=f"Occurrence of {len(cluster)} observation(s) on {cam_id}"
        )

        provenance = StitchProvenance(
            job_id=job_id,
            camera_id=cam_id,
            camera_name=camera_name_map.get(cam_id, cam_id),
            source_type=rep_obs.source_type,
            video_path=rep_obs.source_video_path,
            evidence_path=rep_obs.crop_path,
            crop_path=rep_obs.crop_path,
            bbox=rep_obs.bbox,
            timestamp=start_time,
            track_id=rep_obs.track_id,
            detection_id=rep_obs.detection_id
        )

        return StitchSegment(
            segment_id=segment_id,
            camera_id=cam_id,
            camera_name=camera_name_map.get(cam_id, cam_id),
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            occurrence_id=rep_obs.occurrence_id,
            representative_observation_id=rep_obs.observation_id,
            representative_crop_path=rep_obs.crop_path,
            observations_count=len(cluster),
            observations=cluster,
            continuity=continuity,
            provenance=provenance
        )

    def _expand_candidates_for_track(
        self,
        job_id: str,
        target_track_uuid: str,
        entity_type: str,
        min_similarity: float
    ) -> List[StitchObservation]:
        """Derives candidate observations from SQLite and ChromaDB for a target track ID."""
        observations: List[StitchObservation] = []
        target_track = self.sqlite.get_track(target_track_uuid)
        if not target_track:
            LOGGER.warning("Target track %s not found in SQLite for stitch expansion", target_track_uuid)
            return observations

        # Origin observation
        target_dets = self.sqlite.get_detections_by_track(target_track_uuid)
        origin_crop = target_dets[0]["crop_path"] if target_dets else ""
        origin_bbox = [target_dets[0]["bbox_x1"], target_dets[0]["bbox_y1"], target_dets[0]["bbox_x2"], target_dets[0]["bbox_y2"]] if target_dets else None

        origin_obs = StitchObservation(
            observation_id=f"orig_{target_track_uuid}",
            job_id=job_id,
            camera_id=target_track["camera_id"],
            timestamp=target_track["start_time"],
            end_timestamp=target_track["end_time"],
            source_type="object",
            entity_type=entity_type,
            track_id=target_track_uuid,
            bbox=origin_bbox,
            crop_path=origin_crop,
            visual_similarity=1.0,
            reid_similarity=1.0 if entity_type == "person" else None,
            combined_score=1.0
        )
        observations.append(origin_obs)

        # Retrieve embeddings for the target track
        vis_res = self.vector.visual_coll.get(where={"track_id": target_track_uuid}, include=["embeddings"])
        vis_embs = vis_res.get("embeddings") if vis_res else None

        if vis_embs is None or len(vis_embs) == 0:
            return observations

        target_emb = vis_embs[0]
        filters = {"job_id": job_id}

        # Query visual candidates
        vis_matches = self.vector.query_visual_embeddings(
            query_embedding=target_emb,
            top_k=40,
            filters=filters
        )

        # Optional Re-ID candidate retrieval for person
        reid_map = {}
        if entity_type == "person":
            reid_res = self.vector.reid_coll.get(where={"track_id": target_track_uuid}, include=["embeddings"])
            reid_embs = reid_res.get("embeddings") if reid_res else None
            if reid_embs is not None and len(reid_embs) > 0:

                try:
                    reid_matches = self.vector.query_reid_embeddings(
                        query_embedding=reid_embs[0],
                        top_k=40,
                        filters=filters
                    )
                    for rm in reid_matches:
                        rm_tid = rm.get("metadata", {}).get("track_id")
                        if rm_tid:
                            reid_map[rm_tid] = max(reid_map.get(rm_tid, 0.0), rm.get("similarity", 0.0))
                except Exception as re_err:
                    LOGGER.debug("Re-ID query during stitch expansion: %s", re_err)

        seen_tracks = {target_track_uuid}

        for match in vis_matches:
            m_tid = match.get("metadata", {}).get("track_id")
            if not m_tid or m_tid in seen_tracks:
                continue

            vis_sim = match.get("similarity", 0.0)
            reid_sim = reid_map.get(m_tid)

            # Combined score
            if reid_sim is not None:
                comb_score = 0.60 * reid_sim + 0.40 * vis_sim
            else:
                comb_score = vis_sim

            if comb_score < min_similarity:
                continue

            trk = self.sqlite.get_track(m_tid)
            if not trk:
                continue

            seen_tracks.add(m_tid)
            dets = self.sqlite.get_detections_by_track(m_tid)
            crop = dets[0]["crop_path"] if dets else ""
            bbox = [dets[0]["bbox_x1"], dets[0]["bbox_y1"], dets[0]["bbox_x2"], dets[0]["bbox_y2"]] if dets else None

            observations.append(StitchObservation(
                observation_id=f"match_{m_tid}",
                job_id=job_id,
                camera_id=trk["camera_id"],
                timestamp=trk["start_time"],
                end_timestamp=trk["end_time"],
                source_type="object",
                entity_type=entity_type,
                track_id=m_tid,
                bbox=bbox,
                crop_path=crop,
                visual_similarity=vis_sim,
                reid_similarity=reid_sim,
                combined_score=comb_score
            ))

        return observations

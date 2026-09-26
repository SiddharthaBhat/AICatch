"""Data models for AICatch Cross-Camera Evidence Stitching & Reconstruction Subsystem.

Provides immutable, strongly-typed representations for:
- Normalized forensic observations
- Segment-level provenance and occurrence boundaries
- Cross-camera continuity signals (Re-ID, Visual, Temporal, Topology)
- Cross-camera transition links
- Unified Stitch Result payloads
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import uuid


@dataclass
class StitchProvenance:
    """Forensic provenance contract guaranteeing traceability of evidence."""
    job_id: str
    camera_id: str
    camera_name: Optional[str] = None
    source_type: str = "object"  # 'object' | 'region' | 'scene'
    video_path: Optional[str] = None
    evidence_path: Optional[str] = None
    crop_path: Optional[str] = None
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    timestamp: float = 0.0
    track_id: Optional[Union[str, int]] = None
    detection_id: Optional[Union[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StitchContinuitySignals:
    """Continuity and appearance signals between observations and across cameras.
    
    All similarity and confidence metrics use raw cosine similarity and bounded
    geometric/temporal plausibility. No deceptive identity probabilities.
    """
    score: float = 1.0
    reid_similarity: Optional[float] = None
    visual_similarity: Optional[float] = None
    temporal_score: float = 1.0
    transition_score: float = 1.0
    topology_available: bool = False
    reason: str = "Origin observation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "reid_similarity": round(self.reid_similarity, 4) if self.reid_similarity is not None else None,
            "visual_similarity": round(self.visual_similarity, 4) if self.visual_similarity is not None else None,
            "temporal_score": round(self.temporal_score, 4),
            "transition_score": round(self.transition_score, 4),
            "topology_available": self.topology_available,
            "reason": self.reason
        }


@dataclass
class StitchObservation:
    """Normalized internal observation derived from search results or track detections."""
    observation_id: str
    job_id: str
    camera_id: str
    camera_name: Optional[str] = None
    timestamp: float = 0.0
    end_timestamp: Optional[float] = None
    source_type: str = "object"  # 'object' | 'region' | 'scene'
    entity_type: str = "person"  # 'person' | 'object'
    track_id: Optional[Union[str, int]] = None
    detection_id: Optional[Union[str, int]] = None
    bbox: Optional[List[float]] = None
    crop_path: Optional[str] = None
    source_video_path: Optional[str] = None
    visual_similarity: Optional[float] = None
    reid_similarity: Optional[float] = None
    combined_score: Optional[float] = None
    occurrence_id: Optional[str] = None
    original_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_search_result(cls, r: Dict[str, Any], default_job_id: str = "", rank: int = 0) -> "StitchObservation":
        obs_id = str(r.get("id") or r.get("track_uuid") or r.get("track_id") or r.get("occurrence_id") or f"obs_{uuid.uuid4().hex[:8]}")
        job_id = str(r.get("job_id") or r.get("investigation_id") or default_job_id)
        cam_id = str(r.get("camera_id") or "unknown_cam")
        t_start = float(r.get("timestamp") or r.get("start_time") or r.get("occurrence_start") or 0.0)
        t_end = float(r.get("end_timestamp") or r.get("end_time") or r.get("occurrence_end") or t_start)
        
        # Bbox extraction
        bbox = r.get("bbox")
        if not bbox and all(k in r for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
            bbox = [float(r["bbox_x1"]), float(r["bbox_y1"]), float(r["bbox_x2"]), float(r["bbox_y2"])]

        # Breakdown signal extraction
        breakdown = r.get("signal_breakdown", {})
        reid_sim = r.get("reid_similarity")
        if reid_sim is None and "reid" in breakdown:
            reid_sim = breakdown.get("reid")
        
        vis_sim = r.get("similarity")
        if vis_sim is None and "visual" in breakdown:
            vis_sim = breakdown.get("visual")

        return cls(
            observation_id=obs_id,
            job_id=job_id,
            camera_id=cam_id,
            camera_name=r.get("camera_name"),
            timestamp=t_start,
            end_timestamp=t_end,
            source_type=str(r.get("source_type") or r.get("match_type") or "object"),
            entity_type=str(r.get("entity_type") or "person"),
            track_id=r.get("track_id") or r.get("track_uuid"),
            detection_id=r.get("detection_id"),
            bbox=bbox,
            crop_path=r.get("crop_path"),
            source_video_path=r.get("source_video_path") or r.get("source_path"),
            visual_similarity=float(vis_sim) if vis_sim is not None else None,
            reid_similarity=float(reid_sim) if reid_sim is not None else None,
            combined_score=float(r.get("confidence") or r.get("similarity") or 0.0),
            occurrence_id=r.get("occurrence_id"),
            original_rank=rank,
            metadata={k: v for k, v in r.items() if k not in (
                "id", "job_id", "camera_id", "timestamp", "bbox", "crop_path", "similarity", "reid_similarity"
            )}
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.visual_similarity is not None:
            d["visual_similarity"] = round(self.visual_similarity, 4)
        if self.reid_similarity is not None:
            d["reid_similarity"] = round(self.reid_similarity, 4)
        if self.combined_score is not None:
            d["combined_score"] = round(self.combined_score, 4)
        return d


@dataclass
class StitchSegment:
    """A continuous temporal occurrence on a single camera within the reconstructed journey."""
    segment_id: str
    camera_id: str
    camera_name: str
    start_time: float
    end_time: float
    duration: float
    occurrence_id: Optional[str]
    representative_observation_id: str
    representative_crop_path: Optional[str]
    observations_count: int
    observations: List[StitchObservation]
    continuity: StitchContinuitySignals
    provenance: StitchProvenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "occurrence_id": self.occurrence_id,
            "representative_observation_id": self.representative_observation_id,
            "representative_crop_path": self.representative_crop_path,
            "observations_count": self.observations_count,
            "observations": [o.to_dict() for o in self.observations],
            "continuity": self.continuity.to_dict(),
            "provenance": self.provenance.to_dict()
        }


@dataclass
class StitchTransition:
    """A cross-camera transition linking two consecutive chronological segments."""
    transition_id: str
    from_camera: str
    to_camera: str
    from_camera_name: str
    to_camera_name: str
    from_segment_id: str
    to_segment_id: str
    transit_time_sec: float
    score: float
    signals: StitchContinuitySignals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_camera": self.from_camera,
            "to_camera": self.to_camera,
            "from_camera_name": self.from_camera_name,
            "to_camera_name": self.to_camera_name,
            "from_segment_id": self.from_segment_id,
            "to_segment_id": self.to_segment_id,
            "transit_time_sec": round(self.transit_time_sec, 2),
            "score": round(self.score, 4),
            "signals": self.signals.to_dict()
        }


@dataclass
class StitchResult:
    """Final unified result of the Cross-Camera Evidence Stitching subsystem."""
    stitch_id: str
    investigation_id: str
    job_id: str
    entity_type: str  # 'person' | 'object'
    status: str  # 'complete' | 'partial' | 'insufficient_evidence'
    reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    target_summary: Optional[Dict[str, Any]] = None
    segments: List[StitchSegment] = field(default_factory=list)
    transitions: List[StitchTransition] = field(default_factory=list)
    camera_diversity: int = 0
    total_duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stitch_id": self.stitch_id,
            "investigation_id": self.investigation_id,
            "job_id": self.job_id,
            "entity_type": self.entity_type,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
            "target_summary": self.target_summary,
            "segments": [s.to_dict() for s in self.segments],
            "transitions": [t.to_dict() for t in self.transitions],
            "camera_diversity": self.camera_diversity,
            "total_duration_sec": round(self.total_duration_sec, 2)
        }

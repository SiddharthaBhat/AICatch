"""Validation and Forensic Integrity Rules for AICatch Evidence Stitching.

Enforces:
1. Input provenance validity & cross-investigation isolation.
2. Chronological consistency and physical transit plausibility (no negative transit times).
3. Evidence sufficiency verification (rejects fake / ungrounded transitions).
4. Camera topology validation without inventing synthetic graph edges.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import logging

from pipeline.stitch_models import StitchObservation, StitchSegment, StitchContinuitySignals

LOGGER = logging.getLogger(__name__)


class StitchValidator:
    """Enforces strict forensic plausibility and evidence integrity checks on stitching candidates."""

    @staticmethod
    def validate_observations_for_job(observations: List[StitchObservation], expected_job_id: str) -> List[StitchObservation]:
        """Filters observations ensuring they strictly belong to the active investigation case."""
        valid: List[StitchObservation] = []
        for obs in observations:
            if not obs.camera_id:
                LOGGER.warning("STITCH_VALIDATOR_REJECT: Observation %s missing camera_id", obs.observation_id)
                continue
            if expected_job_id and obs.job_id and obs.job_id != expected_job_id:
                LOGGER.warning(
                    "STITCH_VALIDATOR_ISOLATION_REJECT: Observation %s from job %s rejected for target job %s",
                    obs.observation_id, obs.job_id, expected_job_id
                )
                continue
            valid.append(obs)
        return valid

    @staticmethod
    def validate_transition_plausibility(
        prev_seg: StitchSegment,
        curr_seg: StitchSegment,
        camera_graph: Optional[Dict[str, List[str]]] = None,
        min_transit_sec: float = 0.5,
        max_transit_sec: float = 600.0
    ) -> Tuple[bool, StitchContinuitySignals]:
        """Evaluates whether a cross-camera transition between two segments is physically and temporally plausible."""
        same_camera = (prev_seg.camera_id == curr_seg.camera_id)
        transit_time = curr_seg.start_time - prev_seg.end_time

        # 1. Strict Physical Impossibility Check:
        # A subject cannot appear on Camera B before leaving Camera A (negative transit time between different cameras)
        if not same_camera and transit_time < -1.0:
            LOGGER.info(
                "STITCH_REJECT_IMPOSSIBLE_TRANSIT: %s (end %.1fs) -> %s (start %.1fs) negative transit (%.1fs)",
                prev_seg.camera_id, prev_seg.end_time, curr_seg.camera_id, curr_seg.start_time, transit_time
            )
            return False, StitchContinuitySignals(
                score=0.0,
                temporal_score=0.0,
                transition_score=0.0,
                reason=f"Physically impossible negative transit time ({transit_time:.1f}s) across distinct cameras"
            )

        # 2. Topology Check (Optional graph)
        topology_available = bool(camera_graph and len(camera_graph) > 0)
        topology_compatible = True
        if topology_available and not same_camera:
            neighbors = camera_graph.get(prev_seg.camera_id, [])
            if curr_seg.camera_id not in neighbors:
                topology_compatible = False

        # 3. Temporal Plausibility Scoring
        # Effective transit time clamped to 0
        eff_transit = max(0.0, transit_time)
        temporal_score = 1.0
        reason = ""

        if not same_camera:
            if eff_transit < min_transit_sec:
                # Highly rapid transit between different cameras
                temporal_score = 0.50
                reason = f"Rapid cross-camera transit ({eff_transit:.1f}s)"
            elif eff_transit > max_transit_sec:
                # Long temporal gap
                gap_over = eff_transit - max_transit_sec
                temporal_score = max(0.20, 1.0 - (gap_over / max_transit_sec))
                reason = f"Cross-camera transit with time gap ({eff_transit:.1f}s)"
            else:
                # Plausible transit window
                temporal_score = 1.0
                reason = f"Plausible transit ({eff_transit:.1f}s) from {prev_seg.camera_name} to {curr_seg.camera_name}"
        else:
            if eff_transit > 60.0:
                temporal_score = max(0.40, 1.0 - (eff_transit / 600.0))
                reason = f"Re-appearance on same camera after {eff_transit:.1f}s gap"
            else:
                temporal_score = 1.0
                reason = f"Continuous occurrence on {prev_seg.camera_name}"

        # 4. Appearance Signals
        vis_sim = curr_seg.continuity.visual_similarity
        reid_sim = curr_seg.continuity.reid_similarity

        # Composite score
        transition_weight = 1.0 if topology_compatible else 0.50
        
        if reid_sim is not None and vis_sim is not None:
            appearance_score = 0.60 * reid_sim + 0.40 * vis_sim
        elif reid_sim is not None:
            appearance_score = reid_sim
        elif vis_sim is not None:
            appearance_score = vis_sim
        else:
            appearance_score = curr_seg.continuity.score

        composite_score = appearance_score * temporal_score * transition_weight

        signals = StitchContinuitySignals(
            score=round(composite_score, 4),
            reid_similarity=round(reid_sim, 4) if reid_sim is not None else None,
            visual_similarity=round(vis_sim, 4) if vis_sim is not None else None,
            temporal_score=round(temporal_score, 4),
            transition_score=round(transition_weight, 4),
            topology_available=topology_available,
            reason=reason
        )

        return True, signals

    @staticmethod
    def evaluate_evidence_sufficiency(
        segments: List[StitchSegment],
        min_camera_diversity: int = 2,
        min_continuity_score: float = 0.45
    ) -> Tuple[str, Optional[str]]:
        """Determines if the stitched journey represents sufficient cross-camera evidence or marks insufficient_evidence."""
        if not segments:
            return "insufficient_evidence", "Zero candidate observations provided"

        cameras = set(s.camera_id for s in segments)
        if len(cameras) < min_camera_diversity:
            return "insufficient_evidence", f"Single camera observation ({list(cameras)[0]}) — no cross-camera transitions detected"

        # Check if cross-camera segments meet minimum continuity threshold
        valid_transitions = [
            s for s in segments[1:]
            if s.continuity.score >= min_continuity_score
        ]

        if not valid_transitions:
            return "insufficient_evidence", "Cross-camera candidate matches fail minimum appearance/temporal continuity thresholds"

        if len(valid_transitions) < len(segments) - 1:
            return "partial", "Some cross-camera transitions exhibit weak continuity evidence"

        return "complete", None

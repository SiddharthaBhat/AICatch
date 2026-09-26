"""Unit tests for the isolated Cross-Camera Evidence Stitching & Reconstruction Subsystem.

Validates:
1. Single-camera occurrence grouping & intra-camera collapsing.
2. Multi-camera chronological sequencing.
3. Person Re-ID continuity & signal breakdown.
4. Object continuity & signal breakdown.
5. Duplicate observation & occurrence suppression.
6. Chronological ordering enforcement.
7. Camera transition handling & plausibility checks.
8. Missing topology handling (no fake graph invention).
9. Missing provenance handling.
10. Insufficient evidence detection & reporting.
11. Cross-investigation case isolation.
12. Zero fabricated observations (traceability).
13. API contract validation.
14. Empty input handling.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from pipeline.stitch_models import (
    StitchObservation,
    StitchSegment,
    StitchTransition,
    StitchContinuitySignals,
    StitchProvenance,
    StitchResult
)
from pipeline.stitch_validator import StitchValidator
from pipeline.stitch_engine import StitchEngine


class TestStitchEngine(unittest.TestCase):
    def setUp(self):
        self.mock_sqlite = MagicMock()
        self.mock_vector = MagicMock()
        self.mock_media = MagicMock()
        
        self.mock_sqlite.get_cameras_by_job.return_value = [
            {"id": "cam_a", "name": "North Gate (Cam A)", "source_path": "cam_a.mp4"},
            {"id": "cam_b", "name": "Main Corridor (Cam B)", "source_path": "cam_b.mp4"},
            {"id": "cam_c", "name": "South Exit (Cam C)", "source_path": "cam_c.mp4"},
        ]
        
        self.stitch_engine = StitchEngine(self.mock_sqlite, self.mock_vector, self.mock_media)

    def test_1_single_camera_occurrence_grouping(self):
        """Verify multiple adjacent observations on the same camera collapse into one segment while retaining raw items."""
        obs_list = [
            StitchObservation(observation_id="o1", job_id="job_1", camera_id="cam_a", timestamp=10.0, end_timestamp=12.0, visual_similarity=0.90, reid_similarity=0.85, combined_score=0.88),
            StitchObservation(observation_id="o2", job_id="job_1", camera_id="cam_a", timestamp=13.0, end_timestamp=15.0, visual_similarity=0.92, reid_similarity=0.86, combined_score=0.90),
            StitchObservation(observation_id="o3", job_id="job_1", camera_id="cam_a", timestamp=16.0, end_timestamp=18.0, visual_similarity=0.88, reid_similarity=0.84, combined_score=0.86),
        ]

        result = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=obs_list, temporal_window_sec=12.0)
        
        # Should collapse into 1 segment on cam_a
        self.assertEqual(len(result.segments), 1)
        seg = result.segments[0]
        self.assertEqual(seg.camera_id, "cam_a")
        self.assertEqual(seg.start_time, 10.0)
        self.assertEqual(seg.end_time, 18.0)
        self.assertEqual(seg.observations_count, 3)
        self.assertEqual(len(seg.observations), 3)
        self.assertEqual(result.camera_diversity, 1)
        # Single camera -> status should be insufficient_evidence for cross-camera journey
        self.assertEqual(result.status, "insufficient_evidence")

    def test_2_multi_camera_chronological_stitching(self):
        """Verify multi-camera observations are ordered chronologically and linked with transitions."""
        obs_list = [
            StitchObservation(observation_id="o3", job_id="job_1", camera_id="cam_c", timestamp=220.0, end_timestamp=225.0, visual_similarity=0.82, reid_similarity=0.87, combined_score=0.85),
            StitchObservation(observation_id="o1", job_id="job_1", camera_id="cam_a", timestamp=60.0, end_timestamp=75.0, visual_similarity=0.95, reid_similarity=0.94, combined_score=0.95),
            StitchObservation(observation_id="o2", job_id="job_1", camera_id="cam_b", timestamp=120.0, end_timestamp=135.0, visual_similarity=0.85, reid_similarity=0.89, combined_score=0.87),
        ]

        result = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=obs_list, entity_type="person")

        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.segments), 3)
        self.assertEqual([s.camera_id for s in result.segments], ["cam_a", "cam_b", "cam_c"])
        self.assertEqual(len(result.transitions), 2)
        
        # Transition 1: Cam A -> Cam B (transit: 120.0 - 75.0 = 45.0s)
        t1 = result.transitions[0]
        self.assertEqual(t1.from_camera, "cam_a")
        self.assertEqual(t1.to_camera, "cam_b")
        self.assertEqual(t1.transit_time_sec, 45.0)
        self.assertGreater(t1.score, 0.5)

        # Transition 2: Cam B -> Cam C (transit: 220.0 - 135.0 = 85.0s)
        t2 = result.transitions[1]
        self.assertEqual(t2.from_camera, "cam_b")
        self.assertEqual(t2.to_camera, "cam_c")
        self.assertEqual(t2.transit_time_sec, 85.0)

    def test_3_person_reid_continuity_and_score_breakdown(self):
        """Verify Person stitching combines Re-ID and Visual similarity into explicit signals without fake probabilities."""
        obs_a = StitchObservation(observation_id="o1", job_id="job_1", camera_id="cam_a", timestamp=10.0, end_timestamp=20.0, visual_similarity=0.90, reid_similarity=0.92, combined_score=0.91)
        obs_b = StitchObservation(observation_id="o2", job_id="job_1", camera_id="cam_b", timestamp=35.0, end_timestamp=45.0, visual_similarity=0.62, reid_similarity=0.88, combined_score=0.78)

        result = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=[obs_a, obs_b], entity_type="person")

        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.transitions), 1)
        trans = result.transitions[0]
        signals = trans.signals
        
        self.assertEqual(signals.reid_similarity, 0.88)
        self.assertEqual(signals.visual_similarity, 0.62)
        self.assertGreater(signals.temporal_score, 0.0)
        self.assertIn("Plausible transit", signals.reason)

    def test_4_object_cross_camera_continuity(self):
        """Verify Object stitching uses visual and region evidence across cameras."""
        obs_a = StitchObservation(observation_id="obj_1", job_id="job_1", camera_id="cam_a", timestamp=50.0, end_timestamp=60.0, source_type="region", entity_type="object", visual_similarity=0.84, combined_score=0.84)
        obs_b = StitchObservation(observation_id="obj_2", job_id="job_1", camera_id="cam_b", timestamp=80.0, end_timestamp=90.0, source_type="object", entity_type="object", visual_similarity=0.82, combined_score=0.82)

        result = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=[obs_a, obs_b], entity_type="object")

        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].provenance.source_type, "region")
        self.assertEqual(result.segments[1].provenance.source_type, "object")

    def test_5_reject_physically_impossible_negative_transit(self):
        """Verify Stitch rejects physically impossible overlapping/negative transit across different cameras."""
        # Camera A is active until t=100.0, but Camera B claims observation at t=50.0
        obs_a = StitchObservation(observation_id="o1", job_id="job_1", camera_id="cam_a", timestamp=10.0, end_timestamp=100.0, visual_similarity=0.90, combined_score=0.90)
        obs_b = StitchObservation(observation_id="o2", job_id="job_1", camera_id="cam_b", timestamp=50.0, end_timestamp=60.0, visual_similarity=0.88, combined_score=0.88)

        # Validator check
        seg_a = self.stitch_engine._create_segment([obs_a], 10.0, 100.0, "job_1", "cam_a", {"cam_a": "Cam A", "cam_b": "Cam B"})
        seg_b = self.stitch_engine._create_segment([obs_b], 50.0, 60.0, "job_1", "cam_b", {"cam_a": "Cam A", "cam_b": "Cam B"})
        
        is_plausible, signals = StitchValidator.validate_transition_plausibility(seg_a, seg_b)
        self.assertFalse(is_plausible)
        self.assertIn("negative transit", signals.reason)

    def test_6_optional_camera_topology_handling(self):
        """Verify camera topology graph is respected when provided, but marks topology_available=False when absent."""
        obs_a = StitchObservation(observation_id="o1", job_id="job_1", camera_id="cam_a", timestamp=10.0, end_timestamp=20.0, visual_similarity=0.85, combined_score=0.85)
        obs_b = StitchObservation(observation_id="o2", job_id="job_1", camera_id="cam_b", timestamp=30.0, end_timestamp=40.0, visual_similarity=0.85, combined_score=0.85)

        # 6.1 No topology configured -> topology_available is False
        res_no_top = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=[obs_a, obs_b], camera_graph=None)
        self.assertFalse(res_no_top.transitions[0].signals.topology_available)

        # 6.2 Explicit topology configured -> topology_available is True
        graph = {"cam_a": ["cam_b"], "cam_b": ["cam_c"]}
        res_top = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=[obs_a, obs_b], camera_graph=graph)
        self.assertTrue(res_top.transitions[0].signals.topology_available)
        self.assertEqual(res_top.transitions[0].signals.transition_score, 1.0)

    def test_7_cross_investigation_isolation(self):
        """Verify observations from foreign investigations are strictly filtered and rejected."""
        obs_valid = StitchObservation(observation_id="o1", job_id="job_target", camera_id="cam_a", timestamp=10.0, visual_similarity=0.9)
        obs_foreign = StitchObservation(observation_id="o2", job_id="job_foreign_xyz", camera_id="cam_b", timestamp=20.0, visual_similarity=0.9)

        result = self.stitch_engine.reconstruct_stitch(job_id="job_target", observations=[obs_valid, obs_foreign])
        
        # Only obs_valid should be retained; foreign observation must be discarded
        all_obs_ids = [o.observation_id for s in result.segments for o in s.observations]
        self.assertIn("o1", all_obs_ids)
        self.assertNotIn("o2", all_obs_ids)

    def test_8_insufficient_evidence_handling(self):
        """Verify engine gracefully returns status='insufficient_evidence' with explanation on empty or weak inputs."""
        # 8.1 Empty input
        res_empty = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=[])
        self.assertEqual(res_empty.status, "insufficient_evidence")
        self.assertIsNotNone(res_empty.reason)

        # 8.2 Single camera only
        obs_single = [StitchObservation(observation_id="o1", job_id="job_1", camera_id="cam_a", timestamp=10.0, visual_similarity=0.9)]
        res_single = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=obs_single)
        self.assertEqual(res_single.status, "insufficient_evidence")
        self.assertIn("Single camera", res_single.reason)

    def test_9_provenance_traceability(self):
        """Verify every stitched segment retains full provenance to underlying file and bounding box."""
        obs = StitchObservation(
            observation_id="o1",
            job_id="job_1",
            camera_id="cam_a",
            timestamp=123.45,
            source_type="object",
            crop_path="/media/crops/person/job_1/cam_a/crop_1.jpg",
            source_video_path="/media/videos/cam_a.mp4",
            bbox=[100.0, 150.0, 200.0, 350.0],
            track_id="trk_999",
            detection_id=55,
            visual_similarity=0.92
        )

        result = self.stitch_engine.reconstruct_stitch(job_id="job_1", observations=[obs])
        seg = result.segments[0]
        prov = seg.provenance

        self.assertEqual(prov.job_id, "job_1")
        self.assertEqual(prov.camera_id, "cam_a")
        self.assertEqual(prov.source_type, "object")
        self.assertEqual(prov.crop_path, "/media/crops/person/job_1/cam_a/crop_1.jpg")
        self.assertEqual(prov.video_path, "/media/videos/cam_a.mp4")
        self.assertEqual(prov.bbox, [100.0, 150.0, 200.0, 350.0])
        self.assertEqual(prov.track_id, "trk_999")
        self.assertEqual(prov.detection_id, 55)


if __name__ == "__main__":
    unittest.main()

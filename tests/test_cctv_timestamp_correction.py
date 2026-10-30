import unittest
import tempfile
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from storage.sqlite_db import SQLiteDatabase
from pipeline.stitch_models import StitchObservation, StitchSegment, StitchTransition, StitchResult
from pipeline.stitch_engine import StitchEngine
from pipeline.timestamp_detector import (
    compute_effective_timestamps,
    observation_to_absolute_time,
    TimestampProvenance
)
from main import app, sqlite_db as main_sqlite


class TestCCTVTimestampCorrection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_correction.db"
        self.db = SQLiteDatabase(self.db_path)
        self.client = TestClient(app)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_effective_timestamps_manual_start(self):
        # 1. Manual recording start time mode
        manual_iso = "2026-10-11T10:00:00Z"
        eff_iso, eff_epoch = compute_effective_timestamps(
            mode="manual_start",
            detected_start_time=None,
            manual_start_time=manual_iso,
            clock_offset_sec=0.0
        )
        self.assertIsNotNone(eff_epoch)
        self.assertEqual(eff_iso, "2026-10-11T10:00:00+00:00")
        
        # Observation at t=15.5s
        obs_iso, obs_epoch = observation_to_absolute_time(eff_epoch, 15.5)
        self.assertEqual(obs_epoch, eff_epoch + 15.5)
        self.assertEqual(obs_iso, "2026-10-11T10:00:15.500000+00:00")

    def test_compute_effective_timestamps_clock_offset(self):
        # 2. Clock offset correction: CCTV clock says 10:00:00, offset is +15 minutes (900s)
        detected_iso = "2026-10-11T10:00:00Z"
        eff_iso, eff_epoch = compute_effective_timestamps(
            mode="clock_offset",
            detected_start_time=detected_iso,
            manual_start_time=None,
            clock_offset_sec=900.0  # +15 minutes
        )
        self.assertIsNotNone(eff_epoch)
        self.assertEqual(eff_iso, "2026-10-11T10:15:00+00:00")

        # Negative offset: CCTV clock is 10 minutes fast -> offset -600s
        eff_iso_neg, eff_epoch_neg = compute_effective_timestamps(
            mode="clock_offset",
            detected_start_time=detected_iso,
            manual_start_time=None,
            clock_offset_sec=-600.0
        )
        self.assertEqual(eff_iso_neg, "2026-10-11T09:50:00+00:00")

    def test_database_persistence_and_restart(self):
        # 3. Settings survive restart (reopening SQLiteDatabase on same file)
        job_id = "job_persist_test"
        cam_id = "cam_persist_test"
        self.db.add_job(job_id, "Persist Test", "FAST")
        self.db.add_camera(cam_id, job_id, "North Gate", "file", "test.mp4")

        # Set detected values
        self.db.update_camera_detected_timestamp(
            camera_id=cam_id,
            detected_start_time="2026-10-11T10:00:00Z",
            detected_raw_text="2026-10-11 10:00:00",
            detected_source="ocr_overlay",
            detected_status="verified",
            detected_confidence=0.95,
            detected_timezone="UTC"
        )

        # Apply clock offset correction
        eff_iso, eff_epoch = compute_effective_timestamps(
            mode="clock_offset",
            detected_start_time="2026-10-11T10:00:00Z",
            manual_start_time=None,
            clock_offset_sec=300.0
        )
        self.db.update_camera_timestamp_config(
            camera_id=cam_id,
            mode="clock_offset",
            manual_start_time=None,
            clock_offset_sec=300.0,
            effective_start_time=eff_iso,
            effective_start_epoch=eff_epoch
        )

        # Simulate restart: create brand new SQLiteDatabase connection
        restarted_db = SQLiteDatabase(self.db_path)
        cams = restarted_db.get_cameras_by_job(job_id)
        self.assertEqual(len(cams), 1)
        cam = cams[0]
        self.assertEqual(cam["timestamp_config_mode"], "clock_offset")
        self.assertEqual(cam["clock_offset_sec"], 300.0)
        self.assertEqual(cam["detected_raw_text"], "2026-10-11 10:00:00")
        self.assertEqual(cam["detected_source"], "ocr_overlay")
        self.assertEqual(cam["effective_start_time"], "2026-10-11T10:05:00+00:00")
        self.assertEqual(cam["effective_start_epoch"], eff_epoch)

    def test_camera_vs_job_scope_updates(self):
        # 4. Scope separation: camera scope affects 1 camera; job scope affects all cameras in job
        job_id = "job_scope_test"
        self.db.add_job(job_id, "Scope Test", "FAST")
        self.db.add_camera("cam_1", job_id, "Cam 1", "file", "c1.mp4")
        self.db.add_camera("cam_2", job_id, "Cam 2", "file", "c2.mp4")

        # Update only cam_1
        self.db.update_camera_timestamp_config(
            camera_id="cam_1",
            mode="manual_start",
            manual_start_time="2026-10-11T08:00:00Z",
            clock_offset_sec=0.0,
            effective_start_time="2026-10-11T08:00:00+00:00",
            effective_start_epoch=1791705600.0
        )

        cams = {c["id"]: c for c in self.db.get_cameras_by_job(job_id)}
        self.assertEqual(cams["cam_1"]["timestamp_config_mode"], "manual_start")
        self.assertIn(cams["cam_2"]["timestamp_config_mode"], ("auto", "automatic"))

        # Update entire job scope
        self.db.update_job_cameras_timestamp_config(
            job_id=job_id,
            mode="clock_offset",
            manual_start_time=None,
            clock_offset_sec=120.0,
            effective_start_time="2026-10-11T08:02:00+00:00",
            effective_start_epoch=1791705720.0
        )

        cams_after = {c["id"]: c for c in self.db.get_cameras_by_job(job_id)}
        self.assertEqual(cams_after["cam_1"]["timestamp_config_mode"], "clock_offset")
        self.assertEqual(cams_after["cam_1"]["clock_offset_sec"], 120.0)
        self.assertEqual(cams_after["cam_2"]["timestamp_config_mode"], "clock_offset")
        self.assertEqual(cams_after["cam_2"]["clock_offset_sec"], 120.0)

    def test_video_relative_seconds_preserved(self):
        # 5. Backward compatibility: observation.timestamp remains video-relative seconds
        obs = StitchObservation(
            observation_id="obs_seek_test",
            job_id="job_seek",
            camera_id="cam_seek",
            timestamp=45.2,
            end_timestamp=47.8
        )
        # Invariant check
        self.assertEqual(obs.timestamp, 45.2)
        self.assertEqual(obs.video_timestamp_seconds, 45.2)

        # When dict is produced
        d = obs.to_dict()
        self.assertEqual(d["timestamp"], 45.2)
        self.assertEqual(d["video_timestamp_seconds"], 45.2)

    def test_stitch_cross_camera_chronology_synchronization(self):
        # 6. Stitch engine synchronizes cross-camera segments using absolute time when available
        # Camera A started at 10:00:00 (epoch 10000.0). Person seen at video second 30 (epoch 10030.0).
        # Camera B started at 10:02:00 (epoch 10120.0). Person seen at video second 5 (epoch 10125.0).
        # Notice that in video seconds, Cam B (5s) < Cam A (30s).
        # But in absolute real time, Cam A (10030.0) -> Cam B (10125.0).
        engine = StitchEngine(self.db, None, None)

        cam_a_rec = {
            "id": "cam_a",
            "name": "Camera Entrance",
            "effective_start_epoch": 10000.0,
            "timestamp_config_mode": "manual_start",
            "detected_status": "verified"
        }
        cam_b_rec = {
            "id": "cam_b",
            "name": "Camera Corridor",
            "effective_start_epoch": 10120.0,
            "timestamp_config_mode": "manual_start",
            "detected_status": "verified"
        }

        # Mock sqlite.get_cameras_by_job
        self.db.add_job("job_stitch_sync", "Stitch Sync", "FAST")
        self.db.add_camera("cam_a", "job_stitch_sync", "Entrance", "file", "a.mp4")
        self.db.add_camera("cam_b", "job_stitch_sync", "Corridor", "file", "b.mp4")
        self.db.update_camera_timestamp_config(
            "cam_a", "manual_start", None, 0.0,
            datetime.fromtimestamp(10000.0, tz=timezone.utc).isoformat(), 10000.0
        )
        self.db.update_camera_timestamp_config(
            "cam_b", "manual_start", None, 0.0,
            datetime.fromtimestamp(10120.0, tz=timezone.utc).isoformat(), 10120.0
        )

        obs_a = StitchObservation(
            observation_id="obs_a",
            job_id="job_stitch_sync",
            camera_id="cam_a",
            timestamp=30.0,
            end_timestamp=35.0,
            visual_similarity=0.9
        )
        obs_b = StitchObservation(
            observation_id="obs_b",
            job_id="job_stitch_sync",
            camera_id="cam_b",
            timestamp=5.0,
            end_timestamp=10.0,
            visual_similarity=0.88
        )

        # Pass in reverse order to test sorting
        res = engine.reconstruct_stitch(
            job_id="job_stitch_sync",
            observations=[obs_b, obs_a]
        )

        self.assertEqual(res.status, "complete")
        self.assertEqual(res.chronology_status, "absolute_synchronized")
        self.assertEqual(len(res.segments), 2)
        # Cam A should be first because absolute start (10030.0) < Cam B absolute start (10125.0)
        self.assertEqual(res.segments[0].camera_id, "cam_a")
        self.assertEqual(res.segments[1].camera_id, "cam_b")
        # Video timestamps remain video-relative
        self.assertEqual(res.segments[0].start_time, 30.0)
        self.assertEqual(res.segments[1].start_time, 5.0)
        # Transitions have absolute transit time
        self.assertEqual(len(res.transitions), 1)
        trans = res.transitions[0]
        self.assertEqual(trans.chronological_confidence, "verified")
        self.assertIsNotNone(trans.absolute_transit_time_sec)
        # Transit time = 10125.0 - 10035.0 = 90.0s
        self.assertEqual(trans.absolute_transit_time_sec, 90.0)

    def test_stitch_uncalibrated_graceful_fallback(self):
        # 7. Uncalibrated cameras fall back safely without inventing timestamps
        self.db.add_job("job_stitch_uncal", "Stitch Uncal", "FAST")
        self.db.add_camera("cam_u1", "job_stitch_uncal", "Uncal 1", "file", "u1.mp4")
        self.db.add_camera("cam_u2", "job_stitch_uncal", "Uncal 2", "file", "u2.mp4")
        # Neither camera has effective_start_epoch configured

        engine = StitchEngine(self.db, None, None)
        obs_1 = StitchObservation(
            observation_id="o1",
            job_id="job_stitch_uncal",
            camera_id="cam_u1",
            timestamp=10.0,
            end_timestamp=15.0,
            visual_similarity=0.9
        )
        obs_2 = StitchObservation(
            observation_id="o2",
            job_id="job_stitch_uncal",
            camera_id="cam_u2",
            timestamp=25.0,
            end_timestamp=30.0,
            visual_similarity=0.85
        )

        res = engine.reconstruct_stitch(
            job_id="job_stitch_uncal",
            observations=[obs_1, obs_2]
        )
        self.assertEqual(res.status, "complete")
        self.assertEqual(res.chronology_status, "video_relative_only")
        self.assertEqual(len(res.transitions), 1)
        self.assertEqual(res.transitions[0].chronological_confidence, "uncalibrated")
        self.assertIsNone(res.transitions[0].absolute_transit_time_sec)

    def test_api_timestamp_endpoints(self):
        # 8. Test REST endpoints on main FastAPI app
        import uuid
        job_id = f"job_api_ts_{uuid.uuid4().hex[:8]}"
        cam_id = f"cam_api_ts_{uuid.uuid4().hex[:8]}"
        main_sqlite.add_job(job_id, "API TS Test", "FAST")
        main_sqlite.add_camera(cam_id, job_id, "Lobby", "file", "lobby.mp4")
        main_sqlite.update_camera_detected_timestamp(
            camera_id=cam_id,
            detected_start_time="2026-10-11T12:00:00Z",
            detected_raw_text="2026-10-11 12:00:00",
            detected_source="ocr_overlay",
            detected_status="verified",
            detected_confidence=0.92,
            detected_timezone="UTC"
        )

        # GET config
        resp = self.client.get(f"/api/jobs/{job_id}/cameras/{cam_id}/timestamp-config")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["detected_raw_text"], "2026-10-11 12:00:00")
        self.assertIn(data["timestamp_config_mode"], ("auto", "automatic"))

        # POST preview with +10 minutes offset (600s)
        prev_resp = self.client.post(
            f"/api/jobs/{job_id}/cameras/{cam_id}/timestamp-preview",
            json={
                "mode": "clock_offset",
                "manual_start_time": None,
                "clock_offset_sec": 600.0
            }
        )
        self.assertEqual(prev_resp.status_code, 200)
        p_data = prev_resp.json()
        self.assertEqual(p_data["effective_start_time"], "2026-10-11T12:10:00+00:00")
        self.assertEqual(len(p_data["samples"]), 3)

        # POST save config
        save_resp = self.client.post(
            f"/api/jobs/{job_id}/cameras/{cam_id}/timestamp-config",
            json={
                "mode": "clock_offset",
                "manual_start_time": None,
                "clock_offset_sec": 600.0,
                "scope": "camera"
            }
        )
        self.assertEqual(save_resp.status_code, 200)
        s_data = save_resp.json()
        self.assertEqual(s_data["status"], "success")
        self.assertEqual(s_data["effective_start_time"], "2026-10-11T12:10:00+00:00")

        # Verify get_job_cameras returns the calibrated timestamp fields
        cams_resp = self.client.get(f"/api/jobs/{job_id}/cameras")
        self.assertEqual(cams_resp.status_code, 200)
        cams = cams_resp.json()
        target_cam = next((c for c in cams if c["id"] == cam_id), None)
        self.assertIsNotNone(target_cam)
        self.assertEqual(target_cam["timestamp_config_mode"], "clock_offset")
        self.assertEqual(target_cam["effective_start_time"], "2026-10-11T12:10:00+00:00")


if __name__ == "__main__":
    unittest.main()

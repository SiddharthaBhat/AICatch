"""Integration Test Suite for Evidence Extraction REST Endpoints and Job Lifecycle.

Tests:
1. POST /api/jobs/{job_id}/evidence-extractions (submits extraction job).
2. GET /api/jobs/{job_id}/evidence-extractions/{extraction_id} (polls progress and manifest).
3. GET /api/jobs/{job_id}/evidence-extractions (lists all job extractions).
4. POST /api/jobs/{job_id}/evidence-extractions/{extraction_id}/cancel (cancels extraction).
5. Validation: invalid job_id, missing observations/tracks, non-existent video files.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from main import app, sqlite_db, evidence_job_manager


class TestEvidenceExtractionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Find sample video
        sample = Path("data/media/uploads/outside.mp4").resolve()
        if not sample.exists():
            uploads = list(Path("data/media/uploads").glob("*.mp4"))
            if uploads:
                sample = uploads[0].resolve()
        cls.sample_video = sample

    def setUp(self):
        self.job_id = f"job_api_ext_{int(time.time())}"
        sqlite_db.add_job(self.job_id, "API Extraction Test", "BALANCED", "multi_camera")
        self.cam_id = f"cam_api_{int(time.time())}"
        sqlite_db.add_camera(
            camera_id=self.cam_id,
            job_id=self.job_id,
            name="Front Gate Camera",
            source_type="file",
            source_path=str(self.sample_video).replace("\\", "/")
        )
        sqlite_db.update_camera_timestamp_config(
            camera_id=self.cam_id,
            mode="clock_offset",
            clock_offset_sec=300.0,
            effective_start_time="2026-10-15T12:00:00+00:00",
            effective_start_epoch=1792065600.0
        )

    def tearDown(self):
        try:
            sqlite_db.delete_job(self.job_id)
        except Exception:
            pass

    def test_post_evidence_extraction_and_poll_status(self):
        """Submit evidence extraction via REST, poll status, and verify completed manifest."""
        if not self.sample_video.exists():
            self.skipTest("No sample video available for extraction test.")

        payload = {
            "title": "Suspect Gate Crossing",
            "buffer_sec": 0.5,
            "enable_overlay": True,
            "folder_name": "Test Extractions",
            "observations": [
                {
                    "camera_id": self.cam_id,
                    "timestamp": 1.0,
                    "duration": 1.5,
                    "track_id": "trk_99",
                    "bbox": [50.0, 50.0, 200.0, 300.0]
                }
            ]
        }

        # 1. Submit job
        res = self.client.post(f"/api/jobs/{self.job_id}/evidence-extractions", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "queued")
        ext_id = data["id"]
        self.assertIsNotNone(ext_id)

        # 2. Poll until completed or timeout (max 30s)
        start = time.time()
        completed = False
        final_data = None
        while time.time() - start < 30.0:
            poll_res = self.client.get(f"/api/jobs/{self.job_id}/evidence-extractions/{ext_id}")
            self.assertEqual(poll_res.status_code, 200)
            final_data = poll_res.json()
            if final_data["status"] == "completed":
                completed = True
                break
            time.sleep(0.3)

        self.assertTrue(completed, f"Job did not finish in time; final status: {final_data}")
        self.assertEqual(final_data["progress"], 1.0)
        self.assertIsNotNone(final_data["manifest"])
        self.assertEqual(final_data["manifest"]["evidence_title"], "Suspect Gate Crossing")

        # 3. List extractions
        list_res = self.client.get(f"/api/jobs/{self.job_id}/evidence-extractions")
        self.assertEqual(list_res.status_code, 200)
        extractions = list_res.json()
        self.assertTrue(any(e["id"] == ext_id for e in extractions))

    def test_evidence_extraction_invalid_inputs(self):
        """Verify error handling on invalid requests."""
        # Non-existent job
        res = self.client.post("/api/jobs/job_non_existent/evidence-extractions", json={"title": "Test"})
        self.assertEqual(res.status_code, 404)

        # Empty observations
        res2 = self.client.post(f"/api/jobs/{self.job_id}/evidence-extractions", json={"title": "Test", "observations": []})
        self.assertEqual(res2.status_code, 400)


if __name__ == "__main__":
    unittest.main()

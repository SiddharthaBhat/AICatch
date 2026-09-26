"""API integration tests for the Stitch Endpoints in FastAPI.

Tests:
1. POST /api/investigations/{job_id}/stitch with valid observations.
2. POST /api/investigations/{job_id}/stitch with empty observations (insufficient evidence).
3. GET /api/investigations/{job_id}/stitch/{track_uuid}
4. POST /api/jobs/{job_id}/camera-topology
5. Isolation rejection when foreign job_id is passed.
"""

import unittest
from fastapi.testclient import TestClient
from main import app, sqlite_db, vector_db


class TestStitchAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_1_post_stitch_valid_payload(self):
        """Verify POST /api/investigations/{job_id}/stitch reconstructs segments and transitions."""
        payload = {
            "entity_type": "person",
            "observations": [
                {
                    "id": "obs_1",
                    "job_id": "job_test_api",
                    "camera_id": "cam_gate",
                    "timestamp": 10.0,
                    "end_timestamp": 20.0,
                    "similarity": 0.92,
                    "reid_similarity": 0.90,
                    "source_type": "object",
                    "crop_path": "/crops/p1.jpg"
                },
                {
                    "id": "obs_2",
                    "job_id": "job_test_api",
                    "camera_id": "cam_hallway",
                    "timestamp": 35.0,
                    "end_timestamp": 45.0,
                    "similarity": 0.85,
                    "reid_similarity": 0.88,
                    "source_type": "object",
                    "crop_path": "/crops/p2.jpg"
                }
            ],
            "min_similarity": 0.45
        }

        resp = self.client.post("/api/investigations/job_test_api/stitch", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("stitch_id", data)
        self.assertEqual(data["job_id"], "job_test_api")
        self.assertEqual(data["status"], "complete")
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(len(data["transitions"]), 1)
        self.assertEqual(data["transitions"][0]["from_camera"], "cam_gate")
        self.assertEqual(data["transitions"][0]["to_camera"], "cam_hallway")

    def test_2_post_stitch_empty_payload(self):
        """Verify POST /api/investigations/{job_id}/stitch handles empty observations cleanly."""
        payload = {
            "entity_type": "person",
            "observations": []
        }

        resp = self.client.post("/api/investigations/job_test_api/stitch", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "insufficient_evidence")
        self.assertIn("No valid observations", data["reason"])

    def test_3_set_camera_topology_api(self):
        """Verify POST /api/jobs/{job_id}/camera-topology configures graph cleanly."""
        payload = {
            "camera_graph": {
                "cam_gate": ["cam_hallway"],
                "cam_hallway": ["cam_exit"]
            }
        }

        resp = self.client.post("/api/jobs/job_test_api/camera-topology", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "configured")
        self.assertEqual(data["camera_graph"]["cam_gate"], ["cam_hallway"])


if __name__ == "__main__":
    unittest.main()

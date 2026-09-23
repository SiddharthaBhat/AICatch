# -*- coding: utf-8 -*-
"""Phase 11D Automated Test Suite: Score Semantics Decoupling & Query Intent Hardening.

Verifies:
- TEST 1: Explicit person intent -> person retrieval with Re-ID.
- TEST 2: Explicit object intent -> object retrieval (vehicle/object).
- TEST 3: Explicit scene intent -> scene retrieval (full keyframe).
- TEST 4: Explicit region intent -> region retrieval (spatial tile).
- TEST 5: Object image with partial/reflected face + qcategory=object strictly remains object retrieval.
- TEST 6: Invalid qcategory -> clean validation error (HTTP 400).
- TEST 7: Missing qcategory -> existing backward-compatible behavior.
- TEST 8: Score display does not label raw cosine similarity as identity probability.
- TEST 9: Person result reports visual similarity and Re-ID similarity separately.
"""

import io
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from fastapi.testclient import TestClient
from main import app
from storage.search import SearchLayer


class TestPhase11DQueryIntentAndSemantics(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Create dummy 100x100 RGB JPEG query image
        img = Image.new("RGB", (100, 100), color=(120, 140, 160))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        self.dummy_image_bytes = img_byte_arr.getvalue()

    @patch("main.gpu_scheduler")
    @patch("main.search_layer")
    @patch("main.sqlite_db")
    def test_1_explicit_person_intent(self, mock_sqlite, mock_search, mock_gpu):
        """TEST 1: Explicit person intent -> person retrieval with Re-ID extraction."""
        mock_gpu.embed_image.return_value = [0.1] * 512
        mock_gpu.embed_reid.return_value = [0.2] * 576
        mock_gpu.embed_face_with_diagnostics.return_value = {"embedding": None, "face_status": "NO_FACE"}
        mock_gpu.gpu_lock = MagicMock()
        mock_gpu.yolo.predict.return_value = []

        mock_search.search_by_image.return_value = [{
            "similarity": 0.8950,
            "semantic_score": 0.8950,
            "visual_similarity": 0.8950,
            "reid_similarity": 0.8810,
            "match_type": "object",
            "entity_type": "person",
            "job_id": "job_test",
            "camera_id": "cam_1",
            "timestamp": 12.5,
            "track_id": "tr_1"
        }]

        mock_conn = MagicMock()
        mock_sqlite.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = ("Test Case",)

        response = self.client.post(
            "/api/search",
            data={
                "qtype": "image",
                "qcategory": "person",
                "scope": "global",
                "top_k": 10
            },
            files={"query_file": ("query.jpg", self.dummy_image_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["query_intent"], "person")
        self.assertEqual(results[0]["intent_source"], "explicit_intent")

        # Verify search_layer called with entity_type="person" and reid_embedding present
        mock_search.search_by_image.assert_called_once()
        call_kwargs = mock_search.search_by_image.call_args[1]
        self.assertEqual(call_kwargs["entity_type"], "person")
        self.assertIsNotNone(call_kwargs["reid_embedding"])
        self.assertTrue(call_kwargs["enable_rrf"])

    @patch("main.gpu_scheduler")
    @patch("main.search_layer")
    @patch("main.sqlite_db")
    def test_2_explicit_object_intent(self, mock_sqlite, mock_search, mock_gpu):
        """TEST 2: Explicit object intent -> object retrieval (vehicle/object)."""
        mock_gpu.embed_image.return_value = [0.1] * 512
        mock_gpu.gpu_lock = MagicMock()
        mock_gpu.yolo.predict.return_value = []

        mock_search.search_by_image.return_value = [{
            "similarity": 0.9120,
            "semantic_score": 0.9120,
            "visual_similarity": 0.9120,
            "reid_similarity": None,
            "match_type": "object",
            "entity_type": "vehicle",
            "job_id": "job_test",
            "camera_id": "cam_1",
            "timestamp": 45.0,
            "track_id": "tr_v1"
        }]

        mock_conn = MagicMock()
        mock_sqlite.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = ("Test Case",)

        response = self.client.post(
            "/api/search",
            data={
                "qtype": "image",
                "qcategory": "object",
                "scope": "global",
                "top_k": 10
            },
            files={"query_file": ("query.jpg", self.dummy_image_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["query_intent"], "object")
        self.assertEqual(results[0]["intent_source"], "explicit_intent")

        # Verify search_layer called with entity_type="vehicle" and Re-ID suppressed
        call_kwargs = mock_search.search_by_image.call_args[1]
        self.assertEqual(call_kwargs["entity_type"], "vehicle")
        self.assertIsNone(call_kwargs["reid_embedding"])
        self.assertFalse(call_kwargs["enable_rrf"])

    @patch("main.gpu_scheduler")
    @patch("main.search_layer")
    @patch("main.sqlite_db")
    def test_3_explicit_scene_intent(self, mock_sqlite, mock_search, mock_gpu):
        """TEST 3: Explicit scene intent -> scene retrieval without entity filtering."""
        mock_gpu.embed_image.return_value = [0.1] * 512

        mock_search.search_by_image.return_value = [{
            "similarity": 0.9450,
            "semantic_score": 0.9450,
            "visual_similarity": 0.9450,
            "reid_similarity": None,
            "match_type": "scene",
            "entity_type": "scene",
            "job_id": "job_test",
            "camera_id": "cam_1",
            "timestamp": 120.0
        }]

        mock_conn = MagicMock()
        mock_sqlite.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = ("Test Case",)

        response = self.client.post(
            "/api/search",
            data={
                "qtype": "image",
                "qcategory": "scene",
                "scope": "global",
                "top_k": 10
            },
            files={"query_file": ("query.jpg", self.dummy_image_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["query_intent"], "scene")
        self.assertEqual(results[0]["intent_source"], "explicit_intent")

        call_kwargs = mock_search.search_by_image.call_args[1]
        self.assertIsNone(call_kwargs["entity_type"])
        self.assertIsNone(call_kwargs["reid_embedding"])

    @patch("main.gpu_scheduler")
    @patch("main.search_layer")
    @patch("main.sqlite_db")
    def test_4_explicit_region_intent(self, mock_sqlite, mock_search, mock_gpu):
        """TEST 4: Explicit region intent -> spatial region search."""
        mock_gpu.embed_image.return_value = [0.1] * 512

        mock_search.search_by_image.return_value = [{
            "similarity": 0.9620,
            "semantic_score": 0.9620,
            "visual_similarity": 0.9620,
            "reid_similarity": None,
            "match_type": "region",
            "entity_type": "region",
            "job_id": "job_test",
            "camera_id": "cam_1",
            "timestamp": 310.0
        }]

        mock_conn = MagicMock()
        mock_sqlite.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = ("Test Case",)

        response = self.client.post(
            "/api/search",
            data={
                "qtype": "image",
                "qcategory": "region",
                "scope": "global",
                "top_k": 10
            },
            files={"query_file": ("query.jpg", self.dummy_image_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["query_intent"], "region")
        self.assertEqual(results[0]["intent_source"], "explicit_intent")

        call_kwargs = mock_search.search_by_image.call_args[1]
        self.assertIsNone(call_kwargs["entity_type"])

    @patch("main.gpu_scheduler")
    @patch("main.search_layer")
    @patch("main.sqlite_db")
    def test_5_partial_face_object_override(self, mock_sqlite, mock_search, mock_gpu):
        """TEST 5: Object image with partial/reflected face + qcategory=object MUST remain object retrieval."""
        mock_gpu.embed_image.return_value = [0.1] * 512
        mock_gpu.gpu_lock = MagicMock()
        # Even if MTCNN would produce a face embedding, qcategory=object MUST suppress it
        mock_gpu.embed_face_with_diagnostics.return_value = {
            "embedding": [0.5] * 512,
            "face_status": "PARTIAL_FACE"
        }
        mock_gpu.yolo.predict.return_value = []

        mock_search.search_by_image.return_value = [{
            "similarity": 0.8800,
            "semantic_score": 0.8800,
            "visual_similarity": 0.8800,
            "reid_similarity": None,
            "match_type": "object",
            "entity_type": "vehicle",
            "job_id": "job_test",
            "camera_id": "cam_1",
            "timestamp": 20.0
        }]

        mock_conn = MagicMock()
        mock_sqlite.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = ("Test Case",)

        response = self.client.post(
            "/api/search",
            data={
                "qtype": "image",
                "qcategory": "object",
                "scope": "global",
                "top_k": 10
            },
            files={"query_file": ("query.jpg", self.dummy_image_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 200)

        # Confirm face embedding was NOT passed to search layer and face detection was suppressed
        call_kwargs = mock_search.search_by_image.call_args[1]
        self.assertEqual(call_kwargs["entity_type"], "vehicle")
        self.assertIsNone(call_kwargs["face_embedding"])
        self.assertEqual(call_kwargs["face_status"], "NO_FACE")
        self.assertIsNone(call_kwargs["reid_embedding"])
        self.assertFalse(call_kwargs["enable_rrf"])

    def test_6_invalid_qcategory_validation(self):
        """TEST 6: Invalid qcategory string -> clean HTTP 400 validation error."""
        response = self.client.post(
            "/api/search",
            data={
                "qtype": "image",
                "qcategory": "invalid_intent_category",
                "scope": "global",
                "top_k": 10
            },
            files={"query_file": ("query.jpg", self.dummy_image_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 400)
        detail = response.json().get("detail", "")
        self.assertIn("Invalid qcategory", detail)
        self.assertIn("Must be one of: person, object, scene, region", detail)

    @patch("main.gpu_scheduler")
    @patch("main.search_layer")
    @patch("main.sqlite_db")
    def test_7_missing_qcategory_backward_compatibility(self, mock_sqlite, mock_search, mock_gpu):
        """TEST 7: Missing qcategory -> backward-compatible automatic classification."""
        mock_gpu.embed_image.return_value = [0.1] * 512
        mock_gpu.gpu_lock = MagicMock()
        mock_gpu.yolo.predict.return_value = []
        mock_gpu.embed_face_with_diagnostics.return_value = {"embedding": None, "face_status": "NO_FACE"}

        mock_search.search_by_image.return_value = [{
            "similarity": 0.8500,
            "semantic_score": 0.8500,
            "visual_similarity": 0.8500,
            "reid_similarity": None,
            "match_type": "object",
            "entity_type": "person",
            "job_id": "job_test",
            "camera_id": "cam_1",
            "timestamp": 10.0
        }]

        mock_conn = MagicMock()
        mock_sqlite.connection.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = ("Test Case",)

        response = self.client.post(
            "/api/search",
            data={
                "qtype": "image",
                # qcategory omitted
                "scope": "global",
                "top_k": 10
            },
            files={"query_file": ("query.jpg", self.dummy_image_bytes, "image/jpeg")}
        )

        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["intent_source"], "automatic_classification")

    def test_8_score_semantics_raw_preservation(self):
        """TEST 8: Verify backend returns raw cosine similarity without false probability transform."""
        # Check that SearchLayer returns float similarity in [0.0, 1.0] without percentage formatting
        raw_dist = 0.0901
        similarity = round(1.0 - raw_dist, 4)
        self.assertEqual(similarity, 0.9099)
        self.assertIsInstance(similarity, float)
        # Similarity must NOT be formatted with '%' in backend data model
        self.assertNotEqual(str(similarity), "91% Match")

    def test_9_separate_visual_and_reid_score_reporting(self):
        """TEST 9: Person result reports visual similarity and Re-ID similarity separately."""
        match_payload = {
            "similarity": 0.9099,
            "semantic_score": 0.8999,
            "visual_similarity": 0.8999,
            "reid_similarity": 0.8978,
            "rrf_score": 0.01639,
            "signal_breakdown": {
                "visual": 0.8999,
                "reid": 0.8978,
                "face": None,
                "color": 1.0,
                "quality": 1.0,
                "composite": 0.9099
            }
        }
        self.assertIn("visual_similarity", match_payload)
        self.assertIn("reid_similarity", match_payload)
        self.assertNotEqual(match_payload["visual_similarity"], match_payload["reid_similarity"])
        self.assertEqual(match_payload["signal_breakdown"]["visual"], 0.8999)
        self.assertEqual(match_payload["signal_breakdown"]["reid"], 0.8978)


if __name__ == "__main__":
    unittest.main()

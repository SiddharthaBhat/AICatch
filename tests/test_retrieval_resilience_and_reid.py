"""Unit tests for retrieval resilience, Person Re-ID candidate expansion,

temporal occurrence clustering, and multi-camera diversity.
"""
import unittest
from unittest.mock import MagicMock, patch
from storage.vector_db import VectorDatabase
from storage.search import SearchLayer, cluster_and_diversify_occurrences


class TestRetrievalResilienceAndReID(unittest.TestCase):
    def setUp(self):
        self.mock_sqlite = MagicMock()
        self.mock_vector = MagicMock()
        self.mock_media = MagicMock()
        self.search_layer = SearchLayer(self.mock_sqlite, self.mock_vector, self.mock_media)

    def test_1_chroma_region_query_resilient_fallback(self):
        """Verify region query falls back to in-memory filter when primary ChromaDB where-query raises Exception."""
        vdb = VectorDatabase.__new__(VectorDatabase)
        vdb.region_coll = MagicMock()
        vdb._format_results = VectorDatabase._format_results.__get__(vdb, VectorDatabase)
        vdb._build_where_filter = VectorDatabase._build_where_filter.__get__(vdb, VectorDatabase)

        # Primary query raises ChromaDB indexing error
        def mock_query(query_embeddings, n_results, where=None, include=None):
            if where and "$and" in where:
                raise RuntimeError("ChromaDB index failure simulation on large collection")
            # Fallback sub-query succeeds
            return {
                "ids": [["reg_1", "reg_2"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [[
                    {"job_id": "job_1", "camera_id": "cam_a", "bbox": "0,0,10,10"},
                    {"job_id": "job_1", "camera_id": "cam_b", "bbox": "10,10,20,20"}
                ]]
            }

        vdb.region_coll.query.side_effect = mock_query
        vdb.region_coll.count.return_value = 2

        results = vdb.query_region_embeddings(
            query_embedding=[0.1] * 512,
            top_k=5,
            filters={"job_id": "job_1", "camera_id": "cam_a"}
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "reg_1")
        self.assertEqual(results[0]["metadata"]["camera_id"], "cam_a")

    def test_2_person_reid_candidate_expansion(self):
        """Verify Person search includes Re-ID matches in candidate pool and applies 0.6 Re-ID / 0.4 Visual RRF."""
        self.mock_vector.describe_visual_index.return_value = {
            "collection": "visual_clip_vit_b32_v1",
            "total": 100,
            "by_object_type": {"person": 100},
            "by_camera": {"cam_1": 50, "cam_2": 50},
            "embedding_dimensions": [512],
            "embedding_model": "clip_vit_b32",
            "embedding_version": "v1"
        }
        self.mock_vector.describe_scene_index.return_value = {"total": 0}
        self.mock_vector.describe_region_index.return_value = {"total": 0}

        # Visual query returns track_1 on cam_1
        self.mock_vector.query_visual_embeddings.return_value = [
            {"id": "vis_1", "similarity": 0.65, "metadata": {"track_id": "track_1", "camera_id": "cam_1", "job_id": "job_1"}}
        ]

        # Re-ID query returns track_2 on cam_2 (which had daylight-shift and low CLIP score)
        self.mock_vector.query_reid_embeddings.return_value = [
            {"id": "reid_2", "similarity": 0.88, "metadata": {"track_id": "track_2", "camera_id": "cam_2", "job_id": "job_1"}}
        ]

        # Setup SQLite mock metadata returns
        def mock_get_track(track_id):
            return {
                "id": track_id,
                "job_id": "job_1",
                "camera_id": "cam_1" if track_id == "track_1" else "cam_2",
                "entity_type": "person",
                "plate_text": None
            }
        self.mock_sqlite.get_track.side_effect = mock_get_track

        def mock_get_detections(track_id):
            return [{
                "id": f"det_{track_id}",
                "track_id": track_id,
                "timestamp": 15.0 if track_id == "track_1" else 25.0,
                "confidence": 0.95,
                "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 50.0, "bbox_y2": 100.0,
                "crop_path": f"/crops/{track_id}.jpg"
            }]
        self.mock_sqlite.get_detections_by_track.side_effect = mock_get_detections
        self.mock_sqlite.get_cameras_by_job.return_value = [
            {"id": "cam_1", "name": "Cam 1", "source_path": "cam1.mp4"},
            {"id": "cam_2", "name": "Cam 2", "source_path": "cam2.mp4"}
        ]
        self.mock_media.get_clip_path.return_value = MagicMock(exists=lambda: False)

        results = self.search_layer.search_by_image(
            query_embedding=[0.1] * 512,
            filters={"job_id": "job_1"},
            top_k=10,
            entity_type="person",
            reid_embedding=[0.2] * 512,
            enable_rrf=True,
            strict_gate=False
        )

        # Both track_1 (from visual) and track_2 (from Re-ID expansion) must be present
        track_ids = [r.get("track_id") for r in results]
        self.assertIn("track_1", track_ids)
        self.assertIn("track_2", track_ids)

        # Candidate from Re-ID expansion should have reid_similarity and rrf_score populated
        cand_2 = next(r for r in results if r.get("track_id") == "track_2")
        self.assertEqual(cand_2["reid_similarity"], 0.88)
        self.assertIn("rrf_score", cand_2)

    def test_3_temporal_occurrence_clustering(self):
        """Verify adjacent frames within 12.0s window on same camera form single occurrence."""
        raw_matches = [
            {"camera_id": "cam_a", "timestamp": 10.0, "similarity": 0.80, "track_id": "t1", "crop_path": "c1.jpg"},
            {"camera_id": "cam_a", "timestamp": 12.5, "similarity": 0.92, "track_id": "t1", "crop_path": "c2.jpg"},
            {"camera_id": "cam_a", "timestamp": 16.0, "similarity": 0.85, "track_id": "t1", "crop_path": "c3.jpg"},
            {"camera_id": "cam_a", "timestamp": 60.0, "similarity": 0.78, "track_id": "t1", "crop_path": "c4.jpg"}, # >12s later -> separate
        ]

        occurrences = cluster_and_diversify_occurrences(raw_matches, temporal_window_sec=12.0, top_k=5)
        self.assertEqual(len(occurrences), 2)

        # First occurrence (t=10 to t=16)
        occ1 = next(o for o in occurrences if o["occurrence_start"] == 10.0)
        self.assertEqual(occ1["occurrence_end"], 16.0)
        self.assertEqual(occ1["occurrence_count"], 3)
        self.assertEqual(occ1["similarity"], 0.92) # highest similarity picked as representative
        self.assertEqual(len(occ1["occurrence_observations"]), 3)

        # Second occurrence (t=60)
        occ2 = next(o for o in occurrences if o["occurrence_start"] == 60.0)
        self.assertEqual(occ2["occurrence_end"], 60.0)
        self.assertEqual(occ2["occurrence_count"], 1)

    def test_4_multi_camera_fair_diversity(self):
        """Verify multi-camera fair allocation prevents single camera from starving others."""
        raw_matches = [
            # Cam 1 has 3 separate occurrences with slightly higher scores
            {"camera_id": "cam_1", "timestamp": 10.0, "similarity": 0.95, "track_id": "t1"},
            {"camera_id": "cam_1", "timestamp": 50.0, "similarity": 0.90, "track_id": "t1"},
            {"camera_id": "cam_1", "timestamp": 90.0, "similarity": 0.88, "track_id": "t1"},
            # Cam 2 has 1 occurrence with solid score
            {"camera_id": "cam_2", "timestamp": 20.0, "similarity": 0.89, "track_id": "t2"},
            # Cam 3 has 1 occurrence
            {"camera_id": "cam_3", "timestamp": 30.0, "similarity": 0.85, "track_id": "t3"},
        ]

        occurrences = cluster_and_diversify_occurrences(raw_matches, temporal_window_sec=12.0, top_k=3)
        cameras_in_top3 = [o["camera_id"] for o in occurrences]

        # Top 3 should include cam_1 and cam_2
        self.assertIn("cam_1", cameras_in_top3)
        self.assertIn("cam_2", cameras_in_top3)

    def test_5_describe_reid_index(self):
        """Verify describe_reid_index reports collection metadata correctly."""
        vdb = VectorDatabase.__new__(VectorDatabase)
        vdb.reid_coll = MagicMock()
        vdb.reid_coll.name = "person_reid_vectors"
        vdb.reid_coll.count.return_value = 42
        vdb.reid_coll.get.return_value = {
            "metadatas": [
                {"camera_id": "cam_1"},
                {"camera_id": "cam_1"},
                {"camera_id": "cam_2"}
            ]
        }
        vdb._build_where_filter = VectorDatabase._build_where_filter.__get__(vdb, VectorDatabase)

        facts = vdb.describe_reid_index()
        self.assertEqual(facts["collection"], "person_reid_vectors")
        self.assertEqual(facts["total"], 42)
        self.assertEqual(facts["by_camera"], {"cam_1": 2, "cam_2": 1})
        self.assertEqual(facts["embedding_model"], "osnet_x1_0")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for static and small object retrieval, multi-scale spatial tiling,

source_type provenance, and intent routing.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from pipeline.scene_indexer import SceneIndexer
from storage.vector_db import VectorDatabase
from storage.search import SearchLayer, cluster_and_diversify_occurrences


class TestStaticAndSmallObjectRetrieval(unittest.TestCase):
    def setUp(self):
        self.mock_sqlite = MagicMock()
        self.mock_vector = MagicMock()
        self.mock_media = MagicMock()
        self.search_layer = SearchLayer(self.mock_sqlite, self.mock_vector, self.mock_media)

    def test_1_multi_scale_tile_generation(self):
        """Verify generate_frame_tiles produces bounded multi-scale tiles (macro ~45% + salient micro ~22%)."""
        # Create a synthetic frame 640x360 with a small textured object in one quadrant
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Add texture in small 60x60 region
        frame[100:160, 200:260] = np.random.randint(50, 255, (60, 60, 3), dtype=np.uint8)

        tiles = SceneIndexer.generate_frame_tiles(frame, tile_ratio=0.45, stride_ratio=0.25, include_micro_scale=True)
        self.assertGreater(len(tiles), 0)
        # Bounded tile count
        self.assertLessEqual(len(tiles), 30)

        # Check macro tile sizes
        macro_tiles = [t for t in tiles if (t[1][2] - t[1][0]) >= int(640 * 0.40)]
        self.assertGreater(len(macro_tiles), 0)

        # Check micro tile sizes
        micro_tiles = [t for t in tiles if (t[1][2] - t[1][0]) <= int(640 * 0.25)]
        self.assertGreater(len(micro_tiles), 0)

        # Verify all tiles have valid bboxes within image dimensions
        for t_idx, bbox, crop in tiles:
            bx1, by1, bx2, by2 = bbox
            self.assertGreaterEqual(bx1, 0)
            self.assertGreaterEqual(by1, 0)
            self.assertLessEqual(bx2, 640)
            self.assertLessEqual(by2, 360)
            self.assertGreater(bx2, bx1)
            self.assertGreater(by2, by1)
            self.assertEqual(crop.shape[:2], (by2 - by1, bx2 - bx1))

    def test_2_object_query_searches_regions_and_scenes(self):
        """Verify object query routes across object tracks, spatial regions, and scene keyframes."""
        self.mock_vector.describe_visual_index.return_value = {
            "collection": "visual_clip_vit_b32_v1", "total": 10,
            "by_object_type": {"vehicle": 10}, "by_camera": {"cam_1": 10},
            "embedding_dimensions": [512], "embedding_model": "clip_vit_b32", "embedding_version": "v1"
        }
        self.mock_vector.describe_scene_index.return_value = {"total": 5}
        self.mock_vector.describe_region_index.return_value = {"total": 20}

        # 1. Visual object candidate (detected vehicle track)
        self.mock_vector.query_visual_embeddings.return_value = [
            {"id": "vis_obj_1", "similarity": 0.85, "metadata": {"track_id": "t_veh_1", "camera_id": "cam_1", "job_id": "job_1"}}
        ]
        # 2. Region candidate (static bag in tile)
        self.mock_vector.query_region_embeddings.return_value = [
            {"id": "reg_1", "similarity": 0.82, "metadata": {
                "job_id": "job_1", "camera_id": "cam_2", "timestamp": 45.0, "frame_idx": 100,
                "bbox": "50,50,150,150", "crop_path": "/crops/reg_1.jpg", "source_frame_path": "/kf/1.jpg"
            }}
        ]
        # 3. Scene candidate (wide frame match)
        self.mock_vector.query_scene_embeddings.return_value = [
            {"id": "scene_1", "similarity": 0.79, "metadata": {
                "job_id": "job_1", "camera_id": "cam_3", "timestamp": 80.0, "frame_idx": 200,
                "source_frame_path": "/kf/2.jpg"
            }}
        ]

        def mock_get_track(track_id):
            return {"id": track_id, "job_id": "job_1", "camera_id": "cam_1", "entity_type": "vehicle", "plate_text": None}
        self.mock_sqlite.get_track.side_effect = mock_get_track
        self.mock_sqlite.get_detections_by_track.return_value = [{
            "id": "det_veh_1", "track_id": "t_veh_1", "timestamp": 12.0, "confidence": 0.90,
            "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 100.0, "bbox_y2": 80.0, "crop_path": "/crops/veh.jpg"
        }]
        self.mock_sqlite.get_cameras_by_job.return_value = [
            {"id": "cam_1", "name": "Entrance", "source_path": "cam1.mp4"},
            {"id": "cam_2", "name": "Lobby", "source_path": "cam2.mp4"},
            {"id": "cam_3", "name": "Parking", "source_path": "cam3.mp4"}
        ]
        self.mock_media.get_clip_path.return_value = MagicMock(exists=lambda: False)

        results = self.search_layer.search_by_image(
            query_embedding=[0.1] * 512,
            filters={"job_id": "job_1"},
            top_k=10,
            entity_type="object",
            strict_gate=False
        )

        # Should contain object, region, and scene results
        source_types = [r.get("source_type") for r in results]
        self.assertIn("object", source_types)
        self.assertIn("region", source_types)
        self.assertIn("scene", source_types)

    def test_3_source_type_and_provenance_integrity(self):
        """Verify source_type, bbox, crop_path, and full provenance are preserved on all results."""
        self.mock_vector.describe_visual_index.return_value = {"total": 0, "collection": "c", "by_object_type": {}, "by_camera": {}, "embedding_model": "m", "embedding_version": "v"}
        self.mock_vector.describe_scene_index.return_value = {"total": 0}
        self.mock_vector.describe_region_index.return_value = {"total": 1}

        self.mock_vector.query_visual_embeddings.return_value = []
        self.mock_vector.query_region_embeddings.return_value = [
            {"id": "reg_box", "similarity": 0.88, "metadata": {
                "job_id": "job_static_1", "camera_id": "cam_hallway", "timestamp": 123.45, "frame_idx": 500,
                "bbox_x1": 20.0, "bbox_y1": 30.0, "bbox_x2": 120.0, "bbox_y2": 150.0,
                "crop_path": "/crops/reg_box.jpg", "source_frame_path": "/kf/hallway.jpg"
            }}
        ]
        self.mock_sqlite.get_cameras_by_job.return_value = [{"id": "cam_hallway", "name": "Hallway", "source_path": "hallway.mp4"}]

        results = self.search_layer.search_by_image(
            query_embedding=[0.1] * 512,
            filters={"job_id": "job_static_1"},
            top_k=5,
            entity_type="region",
            strict_gate=False
        )

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["source_type"], "region")
        self.assertEqual(r["match_type"], "region")
        self.assertEqual(r["job_id"], "job_static_1")
        self.assertEqual(r["camera_id"], "cam_hallway")
        self.assertEqual(r["timestamp"], 123.45)
        self.assertEqual(r["frame_idx"], 500)
        self.assertEqual(r["bbox"], [20.0, 30.0, 120.0, 150.0])
        self.assertEqual(r["crop_path"], "/crops/reg_box.jpg")
        self.assertEqual(r["source_frame_path"], "/kf/hallway.jpg")

    def test_4_person_query_retains_person_focus(self):
        """Verify person queries remain strictly identity-focused on person visual crops and Re-ID."""
        self.mock_vector.describe_visual_index.return_value = {
            "collection": "visual_clip_vit_b32_v1", "total": 10,
            "by_object_type": {"person": 10}, "by_camera": {"cam_1": 10},
            "embedding_dimensions": [512], "embedding_model": "clip_vit_b32", "embedding_version": "v1"
        }
        self.mock_vector.describe_scene_index.return_value = {"total": 5}
        self.mock_vector.describe_region_index.return_value = {"total": 20}

        self.mock_vector.query_visual_embeddings.return_value = [
            {"id": "vis_person_1", "similarity": 0.90, "metadata": {"track_id": "t_p1", "camera_id": "cam_1", "job_id": "job_1"}}
        ]
        self.mock_sqlite.get_track.return_value = {"id": "t_p1", "job_id": "job_1", "camera_id": "cam_1", "entity_type": "person", "plate_text": None}
        self.mock_sqlite.get_detections_by_track.return_value = [{
            "id": "det_p1", "track_id": "t_p1", "timestamp": 10.0, "confidence": 0.95,
            "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 50.0, "bbox_y2": 150.0, "crop_path": "/crops/p1.jpg"
        }]
        self.mock_sqlite.get_cameras_by_job.return_value = [{"id": "cam_1", "name": "Cam 1", "source_path": "cam1.mp4"}]
        self.mock_media.get_clip_path.return_value = MagicMock(exists=lambda: False)

        results = self.search_layer.search_by_image(
            query_embedding=[0.1] * 512,
            filters={"job_id": "job_1"},
            top_k=5,
            entity_type="person",
            strict_gate=False
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_type"], "object")
        self.assertEqual(results[0]["entity_type"], "person")
        # Scene and region query methods should NOT be called for person queries
        self.mock_vector.query_scene_embeddings.assert_not_called()
        self.mock_vector.query_region_embeddings.assert_not_called()


if __name__ == "__main__":
    unittest.main()

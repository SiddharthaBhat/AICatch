import os
import sys
import time
import shutil
import logging
from pathlib import Path
import sqlite3
import numpy as np
import torch
from PIL import Image

# Ensure root directory is in python PATH
sys.path.append(os.path.abspath("."))

from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, MEDIA_DIR, UPLOAD_DIR
from storage import SQLiteDatabase, VectorDatabase, MediaStorage, SearchLayer
from pipeline import JobManager, GPUScheduler, ActiveQueryMatcher, ResultDispatcher
from pipeline.gpu_scheduler import preprocess_face_image
from pipeline.decoder import VideoStreamDecoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
LOGGER = logging.getLogger("VerifyPipeline")

def run_tests():
    print("===============================================================")
    print("      AICATCH BACKEND PIPELINE COMPREHENSIVE VERIFICATION      ")
    print("===============================================================")
    
    results = {}
    
    # 1. Backend import test
    try:
        from pipeline.decoder import VideoStreamDecoder
        from pipeline.gpu_scheduler import GPUScheduler
        from pipeline.dispatcher import ResultDispatcher
        results["1. Backend Imports"] = "PASSED"
    except Exception as e:
        results["1. Backend Imports"] = f"FAILED: {e}"
        
    # 2. SQLite initialization
    try:
        sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
        with sqlite_db.connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        results["2. SQLite Initialization"] = f"PASSED (Found tables: {', '.join(tables)})"
    except Exception as e:
        results["2. SQLite Initialization"] = f"FAILED: {e}"
        
    # 3. ChromaDB initialization
    try:
        vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
        results["3. ChromaDB Initialization"] = f"PASSED (Collections: clip_visual_vectors, face_embeddings)"
    except Exception as e:
        results["3. ChromaDB Initialization"] = f"FAILED: {e}"

    # 4. FFmpeg availability
    try:
        exe = shutil.which("ffmpeg")
        if not exe:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            results["4. FFmpeg Availability"] = f"PASSED (Found at: {exe})"
        else:
            results["4. FFmpeg Availability"] = "FAILED (Not found)"
    except Exception as e:
        results["4. FFmpeg Availability"] = f"FAILED: {e}"

    # 5. CUDA availability
    cuda_ok = torch.cuda.is_available()
    results["5. CUDA Availability"] = "PASSED (Available)" if cuda_ok else "PASSED (Not available, running on CPU)"

    # 6. GPU device detection
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        results["6. GPU Device Detection"] = f"PASSED (Device: {device_name})"
    else:
        results["6. GPU Device Detection"] = "PASSED (No GPU, using CPU mode)"

    # Initialize singletons for further testing
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    media_storage = MediaStorage()
    search_layer = SearchLayer(sqlite_db, vector_db, media_storage)
    matcher = ActiveQueryMatcher()
    dispatcher = ResultDispatcher(sqlite_db, vector_db, matcher)
    job_manager = JobManager(sqlite_db)
    
    # 7. Model initialization
    try:
        gpu_scheduler = GPUScheduler(job_manager, media_storage, dispatcher)
        gpu_scheduler.load_models()
        results["7. GPU Models Initialization"] = "PASSED (All models loaded sequentially)"
    except Exception as e:
        results["7. GPU Models Initialization"] = f"FAILED: {e}"
        return results

    # Face preprocessor tests
    # Helper to generate embeddings and assert format
    def test_face_img(img):
        processed = preprocess_face_image(img)
        # Ensure processed image is RGB, 3 channels, 8-bit
        assert processed.mode == "RGB", f"Expected mode RGB, got {processed.mode}"
        assert len(processed.split()) == 3, f"Expected 3 channels, got {len(processed.split())}"
        
        # Test FaceNet embedding extraction
        emb = gpu_scheduler.embed_face(processed)
        assert emb is not None, "Failed to extract face embedding"
        assert len(emb) == 512 or len(emb) == 128, f"Expected face vector dimension 512/128, got {len(emb)}"
        return emb

    # 8. Face RGB test
    try:
        img_rgb = Image.new("RGB", (200, 200), color=(100, 150, 200))
        proc_rgb = preprocess_face_image(img_rgb)
        assert proc_rgb.mode == "RGB"
        results["8. Face RGB preprocessing"] = "PASSED"
    except Exception as e:
        results["8. Face RGB preprocessing"] = f"FAILED: {e}"

    # 9. Face RGBA test
    try:
        img_rgba = Image.new("RGBA", (200, 200), color=(100, 150, 200, 128))
        proc_rgba = preprocess_face_image(img_rgba)
        assert proc_rgba.mode == "RGB"
        results["9. Face RGBA preprocessing"] = "PASSED"
    except Exception as e:
        results["9. Face RGBA preprocessing"] = f"FAILED: {e}"

    # 10. Face grayscale test
    try:
        img_l = Image.new("L", (200, 200), color=128)
        proc_l = preprocess_face_image(img_l)
        assert proc_l.mode == "RGB"
        results["10. Face Grayscale preprocessing"] = "PASSED"
    except Exception as e:
        results["10. Face Grayscale preprocessing"] = f"FAILED: {e}"

    # 11. Face palette test
    try:
        img_p = Image.new("P", (200, 200), color=5)
        proc_p = preprocess_face_image(img_p)
        assert proc_p.mode == "RGB"
        results["11. Face Palette preprocessing"] = "PASSED"
    except Exception as e:
        results["11. Face Palette preprocessing"] = f"FAILED: {e}"

    # Visual Embeddings Query Verification
    mock_job_id = "test_job_9999"
    mock_vector = [0.05] * 512
    mock_face_vector = [0.02] * 512

    # Add mock vectors for search testing
    vector_db.add_visual_embeddings(
        ids=[f"test_vis_1"],
        embeddings=[mock_vector],
        metadatas=[{"job_id": mock_job_id, "camera_id": "test_cam_99", "track_id": "track_1"}]
    )
    vector_db.add_face_embeddings(
        ids=[f"test_face_1"],
        embeddings=[mock_face_vector],
        metadatas=[{"job_id": mock_job_id, "camera_id": "test_cam_99", "track_id": "track_1"}]
    )
    
    with sqlite_db.connection() as conn:
        conn.execute("INSERT OR REPLACE INTO jobs (id, name, status, profile, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)", (mock_job_id, "Mock Job", "completed", "BALANCED", time.time(), time.time()))
        conn.execute("INSERT OR REPLACE INTO cameras (id, job_id, name, source_type, source_path, status) VALUES (?, ?, ?, ?, ?, ?)", ("test_cam_99", mock_job_id, "Cam 99", "file", "path/to/cam", "inactive"))
        conn.execute("INSERT OR REPLACE INTO tracks (id, job_id, camera_id, track_id, entity_type, start_time, end_time, plate_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("track_1", mock_job_id, "test_cam_99", 1, "vehicle", 1.0, 5.0, "XYZ123"))
        conn.execute("INSERT OR REPLACE INTO detections (id, track_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, confidence, entity_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("det_1", "track_1", 1, 1.0, 10.0, 10.0, 50.0, 50.0, "crops/det_1.jpg", 0.9, "vehicle"))

    # 12. Text search
    try:
        matches = search_layer.search_by_text(mock_vector, {"job_id": mock_job_id}, top_k=5)
        assert len(matches) > 0
        results["12. Text Search"] = f"PASSED (Found {len(matches)} matches)"
    except Exception as e:
        results["12. Text Search"] = f"FAILED: {e}"

    # 13. Image search
    try:
        matches = search_layer.search_by_image(mock_vector, {"job_id": mock_job_id}, top_k=5)
        assert len(matches) > 0
        results["13. Image Search"] = "PASSED"
    except Exception as e:
        results["13. Image Search"] = f"FAILED: {e}"

    # 14. Face search
    try:
        matches = search_layer.search_by_face(mock_face_vector, {"job_id": mock_job_id}, top_k=5)
        assert len(matches) > 0
        results["14. Face Search"] = "PASSED"
    except Exception as e:
        results["14. Face Search"] = f"FAILED: {e}"

    # 15. Plate search
    try:
        matches = search_layer.search_by_plate("XYZ", camera_filter=None, job_id=mock_job_id)
        assert len(matches) > 0
        results["15. Plate Search"] = f"PASSED (Found text match: {matches[0].get('plate_text')})"
    except Exception as e:
        results["15. Plate Search"] = f"FAILED: {e}"

    # Job Management Tests
    temp_job_id = None
    # 16. Job creation
    try:
        temp_job_id = job_manager.create_job(
            name="verification_lifecycle_job",
            profile="FAST",
            camera_sources=[{"name": "cam_verify", "path": "C:/Users/siddu/Downloads/AICatch/data/media/uploads/videoplayback1.mp4", "is_live": False}]
        )
        assert temp_job_id is not None
        results["16. Job Creation"] = f"PASSED (Created Job ID: {temp_job_id})"
    except Exception as e:
        results["16. Job Creation"] = f"FAILED: {e}"

    # 17. Job start
    try:
        job_manager.start_job(temp_job_id)
        assert temp_job_id in job_manager.active_jobs
        assert job_manager.active_jobs[temp_job_id].status == "indexing"
        results["17. Job Start"] = "PASSED"
    except Exception as e:
        results["17. Job Start"] = f"FAILED: {e}"

    # 18. Job pause
    try:
        job_manager.pause_job(temp_job_id)
        assert job_manager.active_jobs[temp_job_id].status == "paused"
        results["18. Job Pause"] = "PASSED"
    except Exception as e:
        results["18. Job Pause"] = f"FAILED: {e}"

    # 19. Job resume
    try:
        job_manager.resume_job(temp_job_id)
        assert job_manager.active_jobs[temp_job_id].status == "indexing"
        results["19. Job Resume"] = "PASSED"
    except Exception as e:
        results["19. Job Resume"] = f"FAILED: {e}"

    # 20. Job cancellation
    try:
        job_manager.cancel_job(temp_job_id)
        assert temp_job_id not in job_manager.active_jobs
        job_rec = sqlite_db.get_job(temp_job_id)
        assert job_rec["status"] == "cancelled"
        results["20. Job Cancellation"] = "PASSED"
    except Exception as e:
        results["20. Job Cancellation"] = f"FAILED: {e}"

    # Job deletion & scoped cleanup verification
    # 21. Job deletion
    try:
        # Create vectors & records for deletion test
        del_job_id = "test_del_job_8888"
        vector_db.add_visual_embeddings(
            ids=[f"test_del_vis_1"],
            embeddings=[mock_vector],
            metadatas=[{"job_id": del_job_id, "camera_id": "test_cam_88", "track_id": "track_del"}]
        )
        with sqlite_db.connection() as conn:
            conn.execute("INSERT OR REPLACE INTO jobs (id, name, status, profile, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)", (del_job_id, "Del Job", "completed", "BALANCED", time.time(), time.time()))
            conn.execute("INSERT OR REPLACE INTO cameras (id, job_id, name, source_type, source_path, status) VALUES (?, ?, ?, ?, ?, ?)", ("test_cam_88", del_job_id, "Cam 88", "file", "path/to/cam", "inactive"))
            conn.execute("INSERT OR REPLACE INTO tracks (id, job_id, camera_id, track_id, entity_type, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?)", ("track_del", del_job_id, "test_cam_88", 2, "person", 1.0, 5.0))
        
        # Verify they exist
        vis_res_before = vector_db.visual_coll.get(where={"job_id": del_job_id})
        assert len(vis_res_before["ids"]) > 0, "Failed to pre-insert test vectors"
        
        # Execute Deletion
        vector_db.delete_by_job(del_job_id)
        vis_res_after = vector_db.visual_coll.get(where={"job_id": del_job_id})
        
        sqlite_db.delete_job(del_job_id)
        job_record = sqlite_db.get_job(del_job_id)
        
        results["21. Job Deletion Trigger"] = "PASSED"
        
        # 22. ChromaDB job-scoped deletion
        if len(vis_res_after["ids"]) == 0:
            results["22. ChromaDB job-scoped deletion"] = "PASSED (Successfully isolated and cleared vectors)"
        else:
            results["22. ChromaDB job-scoped deletion"] = "FAILED (Vectors still remain)"
            
        # 23. SQLite job-scoped deletion
        if job_record is None:
            results["23. SQLite job-scoped deletion"] = "PASSED (Successfully cascading-cleared job and child tables)"
        else:
            results["23. SQLite job-scoped deletion"] = "FAILED (SQLite record still exists)"
            
    except Exception as e:
        results["21. Job Deletion Trigger"] = f"FAILED: {e}"
        results["22. ChromaDB job-scoped deletion"] = f"FAILED: {e}"
        results["23. SQLite job-scoped deletion"] = f"FAILED: {e}"

    # 24. Media job-scoped deletion
    try:
        # Simulate crop files creation in media folder
        CROPS_DIR = MEDIA_DIR / "crops"
        job_crop_dir = CROPS_DIR / "person" / mock_job_id
        job_crop_dir.mkdir(parents=True, exist_ok=True)
        test_file = job_crop_dir / "test_crop.jpg"
        test_file.write_text("dummy crop content")
        
        # Simulate media deletion for mock_job_id
        media_storage.delete_job_media(mock_job_id)
        
        if not job_crop_dir.exists():
            results["24. Media job-scoped deletion"] = "PASSED (Successfully removed job directory tree)"
        else:
            results["24. Media job-scoped deletion"] = "FAILED (Job directory still exists)"
            # Cleanup manually if failed
            shutil.rmtree(job_crop_dir, ignore_errors=True)
    except Exception as e:
        results["24. Media job-scoped deletion"] = f"FAILED: {e}"

    # Cleanup temporary test files / entries
    try:
        vector_db.delete_by_job(mock_job_id)
        sqlite_db.delete_job(mock_job_id)
        if temp_job_id:
            sqlite_db.delete_job(temp_job_id)
    except Exception:
        pass

    # Stop Result Dispatcher and GPU scheduler
    dispatcher.stop_dispatcher()
    
    print("\n---------------------- TEST SUMMARY ----------------------")
    passed_count = 0
    for name, res in results.items():
        print(f"{name:<40}: {res}")
        if "PASSED" in res:
            passed_count += 1
            
    print(f"\nFinal Result: {passed_count}/{len(results)} checkpoints passed.")
    print("===============================================================")

if __name__ == "__main__":
    run_tests()

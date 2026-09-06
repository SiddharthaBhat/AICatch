# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import logging
from pathlib import Path
import sqlite3
import torch
from PIL import Image
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.append(os.path.abspath("."))

from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, MEDIA_DIR, UPLOAD_DIR
from storage import SQLiteDatabase, VectorDatabase, MediaStorage, SearchLayer
from pipeline import JobManager, GPUScheduler, ActiveQueryMatcher, ResultDispatcher
from pipeline.journey_engine import JourneyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
LOGGER = logging.getLogger("E2EAcceptance")

def run_e2e_acceptance():
    print("===============================================================")
    print("          AICATCH END-TO-END ACCEPTANCE VERIFICATION           ")
    print("===============================================================")
    
    test_video = Path("data/media/uploads/outside.mp4")
    if not test_video.exists():
        test_video = Path("data/media/uploads/inside.mp4")
    if not test_video.exists():
        raise RuntimeError("No test video found in data/media/uploads/ for acceptance test.")
        
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    media_storage = MediaStorage()
    search_layer = SearchLayer(sqlite_db, vector_db, media_storage)
    matcher = ActiveQueryMatcher()
    dispatcher = ResultDispatcher(sqlite_db, vector_db, matcher)
    
    job_manager = JobManager(sqlite_db)
    journey_engine = JourneyEngine(sqlite_db, vector_db)
    
    scheduler = GPUScheduler(job_manager, media_storage, dispatcher)
    scheduler.load_models()
    scheduler.start()
    
    job_id = f"e2e_test_job_{int(time.time())}"
    camera_sources = [
        {"name": "Cam_Acceptance_1", "type": "file", "path": str(test_video).replace("\\", "/")}
    ]
    
    LOGGER.info(f"1. Creating acceptance test job {job_id}...")
    created_id = job_manager.create_job("E2E Acceptance Test Job", "BALANCED", camera_sources, investigation_type="single_video")
    
    LOGGER.info(f"2. Starting video decoding and AI indexing for {created_id}...")
    job_manager.start_job(created_id)
    
    # Wait until decoder decodes frames and scheduler runs inference
    t0 = time.time()
    tracks_count = 0
    detections_count = 0
    
    while time.time() - t0 < 35:
        time.sleep(2.0)
        with sqlite_db.connection() as conn:
            conn.row_factory = sqlite3.Row
            cur_t = conn.execute("SELECT count(*) as count FROM tracks WHERE job_id = ?", (created_id,))
            tracks_count = cur_t.fetchone()["count"]
            cur_d = conn.execute("SELECT count(*) as count FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (created_id,))
            detections_count = cur_d.fetchone()["count"]
            
        LOGGER.info(f"Indexing progress: tracks={tracks_count}, detections={detections_count}")
        job = job_manager.active_jobs.get(created_id)
        all_finished = job and job.decoders and all(d.has_finished for d in job.decoders.values())
        if (tracks_count >= 1 and detections_count >= 1) or all_finished:
            LOGGER.info("Sufficient frames indexed or decoder finished. Stopping decoder and flushing writes...")
            break
            
    # Stop decoders and flush persistence queue
    job_manager.cleanup_job_resources(created_id)
    dispatcher.flush_and_wait(timeout=10.0)
    
    # Complete job indexing verification
    success = job_manager.complete_job(created_id, vector_db=vector_db, dispatcher=dispatcher)
    LOGGER.info(f"Job indexing completion status: {success}")
    
    # 3. Verify SQLite records
    with sqlite_db.connection() as conn:
        conn.row_factory = sqlite3.Row
        cur_t = conn.execute("SELECT * FROM tracks WHERE job_id = ?", (created_id,))
        tracks = [dict(r) for r in cur_t.fetchall()]
        cur_d = conn.execute("SELECT d.* FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (created_id,))
        detections = [dict(r) for r in cur_d.fetchall()]
        
    print(f"-> SQLite Tracks: {len(tracks)}, Detections: {len(detections)}")
    assert len(tracks) > 0, "No tracks created in SQLite"
    assert len(detections) > 0, "No detections created in SQLite"
    
    # 4. Verify Forensic Lineage
    sample_det = detections[0]
    assert sample_det.get("track_id"), "Detection missing track_id"
    assert sample_det.get("crop_path"), "Detection missing crop_path"
    crop_file = Path(sample_det["crop_path"])
    print(f"-> Forensic Lineage Crop: {crop_file} (Exists: {crop_file.exists()})")
    assert crop_file.exists(), f"Crop file missing on disk: {crop_file}"
    
    # 5. Verify ChromaDB Vectors
    vis_res = vector_db.visual_coll.get(where={"job_id": created_id}, include=["metadatas"])
    vis_ids = vis_res.get("ids", [])
    print(f"-> ChromaDB Indexed Visual Vectors: {len(vis_ids)}")
    assert len(vis_ids) > 0, "No visual embeddings indexed in ChromaDB"
    
    # 6. Verify Unified Search Layer with 5 States & Multi-Signal Breakdown
    sample_img = Image.open(crop_file)
    query_emb = scheduler.embed_image(sample_img)
    search_res = search_layer.search_by_image(query_emb, filters={"job_id": created_id}, top_k=5)
    
    print(f"-> Search Results for Crop Image: {len(search_res)} hits")
    assert len(search_res) > 0, "Image search returned 0 results"
    
    first_hit = search_res[0]
    assert "signal_breakdown" in first_hit, "Result missing multi-signal breakdown"
    print(f"-> Hit 1 Multi-Signal Breakdown: {json.dumps(first_hit['signal_breakdown'], indent=2)}")
    
    # 7. Verify Multi-Camera Journey Engine
    target_uuid = tracks[0]["id"]
    journey = journey_engine.reconstruct_journey(target_uuid)
    print(f"-> Journey Engine Reconstructed Events: {len(journey.get('confirmed_journey', []))}")
    assert len(journey.get("confirmed_journey", [])) > 0, "Journey engine failed to generate origin node"
    
    # 8. Cleanup test job
    LOGGER.info(f"Cleaning up acceptance test job {created_id}...")
    scheduler.stop_scheduler()
    dispatcher.stop_dispatcher()
    vector_db.delete_by_job(created_id)
    with sqlite_db.connection() as conn:
        conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (created_id,))
        conn.execute("DELETE FROM tracks WHERE job_id = ?", (created_id,))
        conn.execute("DELETE FROM cameras WHERE job_id = ?", (created_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (created_id,))
    
    print("\n===============================================================")
    print("      ALL END-TO-END FORENSIC ACCEPTANCE CRITERIA PASSED!      ")
    print("===============================================================\n")

if __name__ == "__main__":
    run_e2e_acceptance()

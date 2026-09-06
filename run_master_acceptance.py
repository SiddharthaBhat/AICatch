# -*- coding: utf-8 -*-
import sys
import time
import os
import json
import logging
from pathlib import Path
import sqlite3
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath("."))

from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, MEDIA_DIR, UPLOAD_DIR
from storage import SQLiteDatabase, VectorDatabase, MediaStorage, SearchLayer
from pipeline import JobManager, GPUScheduler, ActiveQueryMatcher, ResultDispatcher
from pipeline.journey_engine import JourneyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
LOGGER = logging.getLogger("MasterAcceptance")

def run_acceptance():
    print("\n===============================================================", flush=True)
    print("      AICATCH REAL CCTV MASTER ACCEPTANCE WORKFLOW             ", flush=True)
    print("===============================================================\n", flush=True)
    
    test_video = Path("data/media/uploads/inside.mp4")
    if not test_video.exists():
        test_video = Path("data/media/uploads/outside.mp4")
    assert test_video.exists(), "No CCTV test video found in data/media/uploads/"
    
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
    
    # 1. Create and Start Ingestion Job
    job_name = f"Acceptance_Verification_{int(time.time())}"
    camera_sources = [
        {"name": "CCTV_Main_Entrance", "type": "file", "path": str(test_video).replace("\\", "/")}
    ]
    job_id = job_manager.create_job(job_name, "BALANCED", camera_sources, investigation_type="single_video")
    print(f"[STAGE 1] Job created: {job_id}", flush=True)
    
    job_manager.start_job(job_id)
    print(f"[STAGE 2] Ingestion and AI indexing started for {job_id}...", flush=True)
    
    # Wait for frames to be processed and indexed
    t0 = time.time()
    tracks_count = 0
    detections_count = 0
    
    while time.time() - t0 < 35:
        time.sleep(2.0)
        with sqlite_db.connection() as conn:
            conn.row_factory = sqlite3.Row
            cur_t = conn.execute("SELECT count(*) as count FROM tracks WHERE job_id = ?", (job_id,))
            tracks_count = cur_t.fetchone()["count"]
            cur_d = conn.execute("SELECT count(*) as count FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (job_id,))
            detections_count = cur_d.fetchone()["count"]
        print(f"    -> Progress: {time.time()-t0:.1f}s elapsed | tracks={tracks_count}, detections={detections_count}", flush=True)
        if tracks_count >= 2 and detections_count >= 4:
            print("    -> Sufficient representative frames indexed for acceptance.", flush=True)
            break
            
    # Stop decoders and flush persistence queue
    job_manager.cleanup_job_resources(job_id)
    dispatcher.flush_and_wait(timeout=5.0)
    job_manager.complete_job(job_id, vector_db=vector_db, dispatcher=dispatcher)
    
    # 2. Verify Detections and Tracks
    with sqlite_db.connection() as conn:
        conn.row_factory = sqlite3.Row
        cur_t = conn.execute("SELECT * FROM tracks WHERE job_id = ?", (job_id,))
        tracks = [dict(r) for r in cur_t.fetchall()]
        cur_d = conn.execute("SELECT d.* FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (job_id,))
        detections = [dict(r) for r in cur_d.fetchall()]
        
    print(f"\n[STAGE 3] Verified SQLite Records: {len(tracks)} tracks, {len(detections)} detections.", flush=True)
    assert len(tracks) > 0, "Acceptance Failure: 0 tracks in SQLite"
    assert len(detections) > 0, "Acceptance Failure: 0 detections in SQLite"
    
    # 3. Verify Forensic Crops on Disk
    sample_det = detections[0]
    crop_path = Path(sample_det["crop_path"])
    print(f"[STAGE 4] Verified Crop File: {crop_path} (Exists: {crop_path.exists()}, Size: {crop_path.stat().st_size if crop_path.exists() else 0} bytes)", flush=True)
    assert crop_path.exists() and crop_path.stat().st_size > 0, f"Crop missing or empty: {crop_path}"
    
    # 4. Verify ChromaDB Vectors and Read-back
    vis_res = vector_db.visual_coll.get(where={"job_id": job_id}, include=["metadatas", "embeddings"])
    vis_ids = vis_res.get("ids", [])
    print(f"[STAGE 5] Verified ChromaDB Visual Vectors: {len(vis_ids)} indexed.", flush=True)
    assert len(vis_ids) > 0, "Acceptance Failure: 0 visual embeddings in ChromaDB"
    sample_vec_meta = vis_res["metadatas"][0]
    print(f"    -> Read-back Sample Vector Meta: {sample_vec_meta}", flush=True)
    assert sample_vec_meta["job_id"] == job_id, "Vector metadata job_id mismatch"
    
    # 5. Verify Text Search
    text_emb = scheduler.embed_text("person walking")
    text_search_res = search_layer.search_by_text(text_emb, filters={"job_id": job_id}, top_k=5)
    print(f"\n[STAGE 6] Verified Text Search ('person walking'): {len(text_search_res)} results returned.", flush=True)
    assert len(text_search_res) > 0, "Text search returned 0 results"
    print(f"    -> Top Hit Score: {text_search_res[0].get('score')}, Multi-Signal Breakdown: {text_search_res[0].get('signal_breakdown')}", flush=True)
    
    # 6. Verify Image Search (5-State Engine)
    crop_img = Image.open(crop_path)
    img_emb = scheduler.embed_image(crop_img)
    img_search_res = search_layer.search_by_image(img_emb, filters={"job_id": job_id}, top_k=5)
    print(f"\n[STAGE 7] Verified Image Search: {len(img_search_res)} results returned.", flush=True)
    assert len(img_search_res) > 0, "Image search returned 0 results"
    print(f"    -> Top Hit ID: {img_search_res[0].get('id')}, Score: {img_search_res[0].get('score')}", flush=True)
    
    # 7. Verify Multi-Camera Journey Engine
    target_track_uuid = tracks[0]["id"]
    journey = journey_engine.reconstruct_journey(target_track_uuid)
    confirmed_nodes = journey.get("confirmed_journey", [])
    print(f"\n[STAGE 8] Verified Journey Engine: {len(confirmed_nodes)} nodes reconstructed.", flush=True)
    assert len(confirmed_nodes) > 0, "Journey engine failed to produce origin node"
    
    # 8. Clean up
    scheduler.stop_scheduler()
    dispatcher.stop_dispatcher()
    vector_db.delete_by_job(job_id)
    with sqlite_db.connection() as conn:
        conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
        conn.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM cameras WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        
    print("\n===============================================================", flush=True)
    print("   MASTER ACCEPTANCE WORKFLOW COMPLETED AND VERIFIED 100%     ", flush=True)
    print("===============================================================\n", flush=True)

if __name__ == "__main__":
    run_acceptance()

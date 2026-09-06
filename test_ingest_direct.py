# -*- coding: utf-8 -*-
import sys
import time
import os
import sqlite3
import cv2
import torch
from pathlib import Path

sys.path.append(os.path.abspath("."))

from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR
from storage import SQLiteDatabase, VectorDatabase, MediaStorage, SearchLayer
from pipeline import JobManager, GPUScheduler, ActiveQueryMatcher, ResultDispatcher

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def test_ingest():
    print("=== STARTING DIRECT INGEST TEST ===", flush=True)
    
    # 0. Check CUDA
    t0 = time.time()
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    print(f"[STEP 0] PyTorch: {torch.__version__}, CUDA: {cuda_avail}, Device: {dev_name} (Elapsed: {time.time()-t0:.3f}s)", flush=True)
    
    # 1. Initialize databases
    t1 = time.time()
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    media_storage = MediaStorage()
    matcher = ActiveQueryMatcher()
    dispatcher = ResultDispatcher(sqlite_db, vector_db, matcher)
    job_manager = JobManager(sqlite_db)
    scheduler = GPUScheduler(job_manager, media_storage, dispatcher)
    print(f"[STEP 1] Databases & Pipeline instances created (Elapsed: {time.time()-t1:.3f}s)", flush=True)
    
    # 2. Load models
    t2 = time.time()
    print("[STEP 2] START: Loading AI models on GPU/CPU...", flush=True)
    scheduler.load_models()
    print(f"[STEP 2] END: AI models loaded (Elapsed: {time.time()-t2:.3f}s)", flush=True)
    
    # 3. Create job
    t3 = time.time()
    video_source = "data/media/uploads/inside.mp4"
    if not os.path.exists(video_source):
        video_source = "data/media/uploads/outside.mp4"
    assert os.path.exists(video_source), f"Missing video: {video_source}"
    
    job_id = job_manager.create_job("Direct Ingest Test", "BALANCED", [{"name": "cam1", "type": "file", "path": video_source}])
    print(f"[STEP 3] Created job {job_id} for {video_source} (Elapsed: {time.time()-t3:.3f}s)", flush=True)
    
    # 4. Start job
    t4 = time.time()
    job_manager.start_job(job_id)
    decoders = job_manager.get_active_decoders()
    print(f"[STEP 4] Started job, active decoders: {len(decoders)} (Elapsed: {time.time()-t4:.3f}s)", flush=True)
    
    # 5. Wait for decoder queue
    t5 = time.time()
    for _ in range(30):
        time.sleep(0.2)
        if decoders and not decoders[0].frame_queue.empty():
            break
    qsz = decoders[0].frame_queue.qsize() if decoders else 0
    print(f"[STEP 5] Decoder queue ready, size: {qsz} (Elapsed: {time.time()-t5:.3f}s)", flush=True)
    
    # 6. Process packets
    t6 = time.time()
    processed_count = 0
    while decoders and not decoders[0].frame_queue.empty() and processed_count < 3:
        packet = decoders[0].frame_queue.get()
        print(f"    Processing frame packet #{processed_count+1}: idx={packet[0]}, ts={packet[1]:.2f}, shape={packet[2].shape}", flush=True)
        scheduler._process_batch([(decoders[0].camera_id, False, packet)])
        processed_count += 1
    print(f"[STEP 6] Processed {processed_count} packets (Elapsed: {time.time()-t6:.3f}s)", flush=True)
    
    # 7. Flush dispatcher
    t7 = time.time()
    print(f"[STEP 7] Flushing {dispatcher.db_write_queue.qsize()} items from db_write_queue...", flush=True)
    dispatcher.flush_and_wait(timeout=5.0)
    print(f"[STEP 7] Flush completed (Elapsed: {time.time()-t7:.3f}s)", flush=True)
    
    # 8. Verify SQLite records
    with sqlite_db.connection() as conn:
        conn.row_factory = sqlite3.Row
        cur_t = conn.execute("SELECT * FROM tracks WHERE job_id = ?", (job_id,))
        tracks = [dict(r) for r in cur_t.fetchall()]
        cur_d = conn.execute("SELECT d.* FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (job_id,))
        detections = [dict(r) for r in cur_d.fetchall()]
        
    print(f"[STEP 8] Verified SQLite: {len(tracks)} tracks, {len(detections)} detections", flush=True)
    
    # 9. Verify Chroma vectors
    vis_res = vector_db.visual_coll.get(where={"job_id": job_id}, include=["metadatas", "embeddings"])
    vis_ids = vis_res.get("ids", [])
    print(f"[STEP 9] Verified Chroma: {len(vis_ids)} visual vectors indexed", flush=True)
    
    # 10. Verify Crop on disk & Search
    if detections:
        sample_det = detections[0]
        crop_path = Path(sample_det["crop_path"])
        print(f"[STEP 10] Verified crop file: {crop_path} (exists: {crop_path.exists()})", flush=True)
        
        search_layer = SearchLayer(sqlite_db, vector_db, media_storage)
        text_emb = scheduler.embed_text("person")
        hits = search_layer.search_by_text(query_text="person", query_embedding=text_emb, job_id=job_id)
        top_sim = hits[0].get('similarity', 0.0) if hits else 0.0
        print(f"[STEP 11] Text search query 'person': {len(hits)} hits, score: {top_sim:.3f}", flush=True)
        
    # Cleanup
    print("[STEP 12] Cleaning up test job resources...", flush=True)
    job_manager.cleanup_job_resources(job_id)
    vector_db.delete_by_job(job_id)
    with sqlite_db.connection() as conn:
        conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
        conn.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM cameras WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    dispatcher.stop_dispatcher()
    print("=== DIRECT INGEST TEST COMPLETED 100% SUCCESSFULLY ===", flush=True)

if __name__ == "__main__":
    test_ingest()

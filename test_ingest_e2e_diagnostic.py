import time
import os
import sys
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, MEDIA_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_VERSION
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from storage.media_storage import MediaStorage
from storage.search import SearchLayer
from pipeline.matcher import ActiveQueryMatcher
from pipeline.dispatcher import ResultDispatcher
from pipeline.job_manager import JobManager
from pipeline.gpu_scheduler import GPUScheduler

print("=== STARTING FULL END-TO-END INGEST & SEARCH DIAGNOSTIC ===", flush=True)

# 1. Initialize DBs
sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
media_storage = MediaStorage()
matcher = ActiveQueryMatcher()
dispatcher = ResultDispatcher(sqlite_db, vector_db, matcher)
job_manager = JobManager(sqlite_db)
gpu_scheduler = GPUScheduler(job_manager, media_storage, dispatcher)
search_layer = SearchLayer(sqlite_db, vector_db, media_storage)

print(f"[STEP 1] Initialized DBs & Pipeline components.", flush=True)
print(f"[*] ChromaDB visual collection name: {vector_db.visual_coll.name}", flush=True)
print(f"[*] ChromaDB visual collection count before test: {vector_db.visual_coll.count()}", flush=True)

# 2. Load Models
t_load = time.time()
print(f"[STEP 2] Loading AI Models...", flush=True)
gpu_scheduler.load_models()
print(f"[STEP 2] AI Models loaded in {time.time()-t_load:.3f}s. YOLO={gpu_scheduler.yolo is not None}, CLIP={gpu_scheduler.clip_model is not None}, ReID={gpu_scheduler.reid_model is not None}", flush=True)

# 3. Create Ingest Job for inside.mp4
video_path = "data/media/uploads/inside.mp4"
if not os.path.exists(video_path):
    video_path = "data/media/uploads/outside.mp4"
assert os.path.exists(video_path), f"Video file not found: {video_path}"

job_id = job_manager.create_job(
    name="CLIP_Diagnostics_Job",
    profile="BALANCED",
    camera_sources=[{"name": "cam_diag_1", "type": "file", "path": video_path}]
)
print(f"[STEP 3] Created job_id={job_id} for video={video_path}", flush=True)

# 4. Start Ingest Job
job_manager.start_job(job_id)
decoders = job_manager.get_active_decoders()
print(f"[STEP 4] Started job {job_id}, active decoders: {len(decoders)}", flush=True)

# 5. Process first 5 sampled frame packets
packets_processed = 0
start_proc = time.time()
while packets_processed < 5 and (time.time() - start_proc < 30):
    for dec in decoders:
        try:
            packet = dec.frame_queue.get(timeout=1.0)
            print(f"\n--- [FEED FRAME #{packets_processed+1}] idx={packet[0]}, ts={packet[1]:.2f}s ---", flush=True)
            gpu_scheduler._process_batch([(dec.camera_id, False, packet)])
            packets_processed += 1
        except Exception as q_e:
            time.sleep(0.1)

print(f"\n[STEP 5] Processed {packets_processed} frame packets. Flushing dispatcher queue...", flush=True)
dispatcher.flush_and_wait(timeout=10.0)

# 6. Complete Job
print(f"[STEP 6] Running complete_job for {job_id}...", flush=True)
success = job_manager.complete_job(job_id, vector_db=vector_db, dispatcher=dispatcher)
print(f"[STEP 6] complete_job result: {success}", flush=True)

# 7. Check SQLite & ChromaDB state
with sqlite_db.connection() as conn:
    conn.row_factory = sqlite3.Row
    j_row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    tracks_cnt = conn.execute("SELECT count(*) FROM tracks WHERE job_id = ?", (job_id,)).fetchone()[0]
    dets_cnt = conn.execute("SELECT count(*) FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (job_id,)).fetchone()[0]

vis_res_job = vector_db.visual_coll.get(where={"job_id": job_id}, include=["metadatas", "embeddings"])
vis_res_inv = vector_db.visual_coll.get(where={"investigation_id": job_id}, include=["metadatas", "embeddings"])
reid_res = vector_db.reid_coll.get(where={"job_id": job_id})
face_res = vector_db.face_coll.get(where={"job_id": job_id})

print("\n=== STEP 7: DIRECT DATABASE AUDIT ===")
print(f"SQLite Job Status: {j_row['status'] if j_row else 'NOT FOUND'}")
print(f"SQLite Tracks: {tracks_cnt}")
print(f"SQLite Detections: {dets_cnt}")
print(f"ChromaDB visual vectors by job_id: {len(vis_res_job.get('ids', []))}")
print(f"ChromaDB visual vectors by investigation_id: {len(vis_res_inv.get('ids', []))}")
print(f"ChromaDB Re-ID vectors: {len(reid_res.get('ids', []))}")
print(f"ChromaDB Face vectors: {len(face_res.get('ids', []))}")

if vis_res_job.get("ids"):
    sample_id = vis_res_job["ids"][0]
    sample_meta = vis_res_job["metadatas"][0]
    sample_emb = vis_res_job["embeddings"][0]
    norm = sum(x*x for x in sample_emb)**0.5
    print(f"Sample Vector ID: {sample_id}")
    print(f"Sample Vector Dimension: {len(sample_emb)}")
    print(f"Sample Vector Norm: {norm:.6f}")
    print(f"Sample Vector Metadata: {sample_meta}")

# 8. Test Search
print("\n=== STEP 8: TEST SEARCH ON THIS INVESTIGATION ===")
text_query = "person walking"
query_emb = gpu_scheduler.embed_text(text_query)
print(f"Text Query: '{text_query}' -> Query embedding dim={len(query_emb)}, norm={sum(x*x for x in query_emb)**0.5:.6f}")

try:
    results = search_layer.search_by_text(query_emb, filters={"job_id": job_id}, top_k=5, query_text=text_query)
    print(f"[SEARCH SUCCESS] Found {len(results)} matches for '{text_query}' in job {job_id}!")
    for r in results[:3]:
        print(f"  - Match: track={r.get('track_id')} | sim={r.get('similarity')} | score={r.get('score')} | crop={r.get('crop_path')}")
except Exception as s_err:
    print(f"[SEARCH FAILED]: {type(s_err).__name__}: {s_err}")

# Cleanup test job decoders
job_manager.cleanup_job_resources(job_id)
print("\n=== DIAGNOSTIC COMPLETE ===", flush=True)

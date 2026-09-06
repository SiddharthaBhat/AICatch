import sqlite3
import json
import shutil
from pathlib import Path
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from storage.media_storage import MediaStorage
from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, MEDIA_DIR, UPLOAD_DIR
from main import delete_job

sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
media = MediaStorage()

print("=== REPRODUCING BUG 1 (DELETE CASCADE & LEFTOVERS) ===")
# 1. Create a dummy test job with full entities
job_id = "test_del_job_001"
with sqlite_db.connection() as conn:
    conn.execute("INSERT OR REPLACE INTO jobs (id, name, status, created_at, updated_at, profile) VALUES (?, ?, ?, ?, ?, ?)",
                 (job_id, "Delete Test", "INDEXED", 1000.0, 1000.0, "BALANCED"))
    conn.execute("INSERT OR REPLACE INTO cameras (id, job_id, name, source_type, source_path, status) VALUES (?, ?, ?, ?, ?, ?)",
                 ("cam_del_01", job_id, "Cam 1", "file", "test.mp4", "inactive"))
    conn.execute("INSERT OR REPLACE INTO tracks (id, camera_id, job_id, track_id, entity_type, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 ("track_del_01", "cam_del_01", job_id, 1, "person", 0.0, 1.0))
    conn.execute("INSERT OR REPLACE INTO detections (id, track_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, confidence, entity_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 ("det_del_01", "track_del_01", 1, 0.0, 10, 10, 50, 50, "data/media/crops/person/test_del_job_001/cam_del_01/track_1_frame_1.jpg", 0.9, "person"))

# 2. Add dummy Chroma vector
dummy_emb = [0.1] * 512
vector_db.visual_coll.add(ids=["det_del_01"], embeddings=[dummy_emb], metadatas=[{"job_id": job_id, "investigation_id": job_id, "camera_id": "cam_del_01"}])

# 3. Add dummy crop file on disk
crop_file = MEDIA_DIR / "crops" / "person" / job_id / "cam_del_01" / "track_1_frame_1.jpg"
crop_file.parent.mkdir(parents=True, exist_ok=True)
crop_file.write_text("dummy image data")

print(f"Created dummy job {job_id} with SQLite rows, Chroma vector, and disk crop at {crop_file}")

# 4. Call delete_job
print("Calling delete_job(job_id)...")
res = delete_job(job_id)
print(f"delete_job returned: {res}")

# 5. Check what is left behind
with sqlite_db.connection() as conn:
    j_rem = conn.execute("SELECT count(*) FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    t_rem = conn.execute("SELECT count(*) FROM tracks WHERE job_id = ?", (job_id,)).fetchone()[0]
    d_rem = conn.execute("SELECT count(*) FROM detections WHERE id = 'det_del_01'").fetchone()[0]
    c_rem = conn.execute("SELECT count(*) FROM cameras WHERE job_id = ?", (job_id,)).fetchone()[0]

chroma_rem = len(vector_db.visual_coll.get(where={"job_id": job_id})["ids"])
file_rem = crop_file.exists()

print(f"Post-Delete Verification:")
print(f"  Jobs remaining: {j_rem}")
print(f"  Tracks remaining: {t_rem}")
print(f"  Detections remaining: {d_rem}")
print(f"  Cameras remaining: {c_rem}")
print(f"  Chroma vectors remaining: {chroma_rem}")
print(f"  Disk file remaining: {file_rem}")

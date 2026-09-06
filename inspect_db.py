import sqlite3
import json
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR

sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
vector_db = VectorDatabase(CHROMA_PERSIST_DIR)

print("=== CURRENT SQLITE JOBS ===")
with sqlite_db.connection() as conn:
    conn.row_factory = sqlite3.Row
    jobs = conn.execute("SELECT id, name, status, created_at, error_msg FROM jobs ORDER BY created_at DESC").fetchall()
    print(f"Total jobs in SQLite: {len(jobs)}")
    for j in jobs:
        t_cnt = conn.execute("SELECT count(*) FROM tracks WHERE job_id = ?", (j["id"],)).fetchone()[0]
        d_cnt = conn.execute("SELECT count(*) FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (j["id"],)).fetchone()[0]
        
        # Check Chroma vectors for this job
        vis_cnt = len(vector_db.visual_coll.get(where={"job_id": j["id"]})["ids"])
        reid_cnt = len(vector_db.reid_coll.get(where={"job_id": j["id"]})["ids"])
        face_cnt = len(vector_db.face_coll.get(where={"job_id": j["id"]})["ids"])
        
        print(f"Job: {j['id']} | Name: {j['name'][:30]} | Status: {j['status']} | Tracks: {t_cnt} | Dets: {d_cnt} | Chroma: vis={vis_cnt}, reid={reid_cnt}, face={face_cnt}")

print("\n=== CHROMADB TOTAL COUNTS ===")
print(f"Visual collection total: {vector_db.visual_coll.count()}")
print(f"ReID collection total: {vector_db.reid_coll.count()}")
print(f"Face collection total: {vector_db.face_coll.count()}")

"""AICatch Database & Vector Store Reconciliation Tool.

Cross-checks SQLite investigation metadata, ChromaDB collections, and
the media filesystem to identify and fix drift, false INDEXED states, and orphans.
"""
import sys
import json
import sqlite3
import shutil
from pathlib import Path

from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, MEDIA_DIR
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from storage.media_storage import MediaStorage
from main import delete_job

def run_reconciliation(auto_fix: bool = False, purge_failed: bool = False):
    print("=" * 70)
    print("       AICATCH INVESTIGATION RECONCILIATION & INTEGRITY AUDIT")
    print("=" * 70)
    
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    media_storage = MediaStorage()
    
    all_jobs = sqlite_db.get_jobs()
    print(f"[*] Total investigations in SQLite: {len(all_jobs)}")
    print(f"[*] ChromaDB visual collection total: {vector_db.visual_coll.count()}")
    print(f"[*] ChromaDB Re-ID collection total: {vector_db.reid_coll.count()}")
    print(f"[*] ChromaDB Face collection total: {vector_db.face_coll.count()}")
    print("-" * 70)
    
    anomalies = []
    healthy = []
    known_job_ids = set()
    
    with sqlite_db.connection() as conn:
        conn.row_factory = sqlite3.Row
        for j in all_jobs:
            jid = j["id"]
            known_job_ids.add(jid)
            status = j["status"]
            name = j["name"]
            
            cur_t = conn.execute("SELECT count(*) as count FROM tracks WHERE job_id = ?", (jid,))
            track_count = cur_t.fetchone()["count"]
            
            cur_d = conn.execute("SELECT count(*) as count FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (jid,))
            det_count = cur_d.fetchone()["count"]
            
            vis_ids = vector_db.visual_coll.get(where={"job_id": jid})["ids"]
            reid_ids = vector_db.reid_coll.get(where={"job_id": jid})["ids"]
            face_ids = vector_db.face_coll.get(where={"job_id": jid})["ids"]
            
            vis_count = len(vis_ids)
            reid_count = len(reid_ids)
            face_count = len(face_ids)
            
            job_issues = []
            if status.upper() == "INDEXED" and vis_count == 0:
                job_issues.append("STATUS_MISMATCH: Marked INDEXED but ChromaDB has 0 visual vectors")
            if status.lower() == "failed" and track_count == 0 and det_count == 0:
                job_issues.append("ORPHANED_FAILED_RUN: Failed test/pipeline run with 0 metadata")
            if status.upper() == "DELETE_FAILED":
                job_issues.append(f"DELETE_FAILED: Prior deletion error ({j.get('error_msg')})")
                
            entry = {
                "id": jid,
                "name": name,
                "status": status,
                "tracks": track_count,
                "detections": det_count,
                "chroma": {"vis": vis_count, "reid": reid_count, "face": face_count},
                "issues": job_issues
            }
            
            if job_issues:
                anomalies.append(entry)
                print(f"[!] ANOMALY: {jid} ({name[:20]}) -> Status: {status} | Tracks: {track_count} | Dets: {det_count} | Vectors: {vis_count}")
                for issue in job_issues:
                    print(f"      - {issue}")
            else:
                healthy.append(entry)
                print(f"[+] HEALTHY: {jid} ({name[:20]}) -> Status: {status} | Tracks: {track_count} | Dets: {det_count} | Vectors: {vis_count}")

    print("-" * 70)
    print(f"Summary: {len(healthy)} healthy, {len(anomalies)} anomalies detected.")

    if auto_fix:
        print("\n[*] Applying Auto-Fixes...")
        # 1. Update statuses of mismatched jobs
        with sqlite_db.connection() as conn:
            for a in anomalies:
                jid = a["id"]
                if "STATUS_MISMATCH: Marked INDEXED but ChromaDB has 0 visual vectors" in a["issues"]:
                    conn.execute(
                        "UPDATE jobs SET status = 'failed', error_msg = 'Reconciled: 0 visual vectors in ChromaDB' WHERE id = ?",
                        (jid,)
                    )
                    print(f"  -> Fixed {jid}: status updated to 'failed'.")
                    
        # 2. Purge failed jobs outside of active transaction
        if purge_failed:
            for a in anomalies:
                jid = a["id"]
                if "ORPHANED_FAILED_RUN: Failed test/pipeline run with 0 metadata" in a["issues"]:
                    try:
                        delete_job(jid)
                        print(f"  -> Purged empty failed test job {jid}.")
                    except Exception as de:
                        print(f"  -> Failed to purge {jid}: {de}")

        print("[*] Auto-Fix complete.")


if __name__ == "__main__":
    auto_fix = "--fix" in sys.argv
    purge_failed = "--purge-failed" in sys.argv
    run_reconciliation(auto_fix=auto_fix, purge_failed=purge_failed)

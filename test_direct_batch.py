import cv2, time
from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR
from storage import SQLiteDatabase, VectorDatabase, MediaStorage
from pipeline import JobManager, GPUScheduler, ActiveQueryMatcher, ResultDispatcher

sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
media_storage = MediaStorage()
matcher = ActiveQueryMatcher()
dispatcher = ResultDispatcher(sqlite_db, vector_db, matcher)
job_manager = JobManager(sqlite_db)
scheduler = GPUScheduler(job_manager, media_storage, dispatcher)
scheduler.load_models()

job_id = job_manager.create_job("Direct Test Job", "BALANCED", [{"name": "cam1", "type": "file", "path": "data/media/uploads/inside.mp4"}])
job_manager.start_job(job_id)

cap = cv2.VideoCapture("data/media/uploads/inside.mp4")
ret, frame = cap.read()
cap.release()

cam_id = list(job_manager.active_jobs[job_id].decoders.keys())[0]
print(f"Running _process_batch for cam_id={cam_id}...")
scheduler._process_batch([(cam_id, False, (1, 0.0, frame))])
print(f"db_write_queue size={dispatcher.db_write_queue.qsize()}")

dispatcher.flush_and_wait(timeout=5.0)

with sqlite_db.connection() as conn:
    tracks = conn.execute("SELECT count(*) FROM tracks WHERE job_id = ?", (job_id,)).fetchone()[0]
    dets = conn.execute("SELECT count(*) FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (job_id,)).fetchone()[0]
print(f"SQLite tracks={tracks}, dets={dets}")

vis = vector_db.visual_coll.get(where={"job_id": job_id})
print(f"Chroma visual vectors={len(vis.get('ids', []))}")

job_manager.cleanup_job_resources(job_id)
with sqlite_db.connection() as conn:
    conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
    conn.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
    conn.execute("DELETE FROM cameras WHERE job_id = ?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
vector_db.delete_by_job(job_id)
dispatcher.stop_dispatcher()
print("Direct batch test SUCCESS!")

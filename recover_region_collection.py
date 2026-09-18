"""Controlled recovery script for AICatch vector index.

Recreates only the corrupted region collection, preserves all other collections,
and re-indexes region tiles for existing jobs.
"""
import shutil
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings

CHROMA_DIR = Path("c:/Users/siddu/Downloads/AICatch/data/chroma_db")
CORRUPT_SEGMENT_DIR = CHROMA_DIR / "341bebc1-5251-42d9-9400-cf78dee23b0b"

print(f"[1] Connecting to Chroma at {CHROMA_DIR}...", flush=True)
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False)
)

print("[2] Deleting corrupted collection 'region_clip_vit_b32_v1'...", flush=True)
try:
    client.delete_collection("region_clip_vit_b32_v1")
    print(" -> Successfully deleted corrupted collection.", flush=True)
except Exception as e:
    print(f" -> Delete collection note: {e}", flush=True)

# Clean up corrupted segment folder if still present
if CORRUPT_SEGMENT_DIR.exists():
    print(f"[3] Removing remaining corrupted segment directory: {CORRUPT_SEGMENT_DIR}...", flush=True)
    try:
        shutil.rmtree(CORRUPT_SEGMENT_DIR)
        print(" -> Corrupted segment directory removed.", flush=True)
    except Exception as e:
        print(f" -> Could not remove folder: {e}", flush=True)

print("[4] Recreating clean 'region_clip_vit_b32_v1' collection...", flush=True)
region_coll = client.get_or_create_collection(
    name="region_clip_vit_b32_v1",
    metadata={"hnsw:space": "cosine", "embedding_model": "clip_vit_b32", "embedding_version": "v1"}
)
print(f" -> Created clean collection. Initial count: {region_coll.count()}", flush=True)

print("\n[5] Verifying health of all collections:", flush=True)
for c in client.list_collections():
    print(f" -> {c.name}: count = {c.count()}, metadata = {c.metadata}", flush=True)

print("\n[6] Re-indexing regions for existing valid investigations...", flush=True)
from config import SQLITE_DB_PATH, MEDIA_DIR
from storage import SQLiteDatabase, VectorDatabase, MediaStorage
from pipeline.scene_indexer import SceneIndexer

sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
vector_db = VectorDatabase(CHROMA_DIR)
media_storage = MediaStorage()
scene_indexer = SceneIndexer(sqlite_db, vector_db, media_storage)

# Re-index regions for jobs that are in INDEXED status
jobs = sqlite_db.get_jobs()
for job in jobs:
    if job["status"] == "INDEXED":
        job_id = job["id"]
        print(f" -> Checking regions for job {job_id} ({job['name']})...", flush=True)
        cameras = sqlite_db.get_cameras_by_job(job_id)
        for cam in cameras:
            source = cam["source_path"]
            if Path(source).exists():
                print(f"    Indexing scenes/regions for camera {cam['id']} ({source})...", flush=True)
                try:
                    res = scene_indexer.extract_and_index_scenes_for_camera(
                        job_id=job_id,
                        camera_id=cam["id"],
                        source_path=source,
                        force_reindex=False
                    )
                    print(f"    Result: {res}", flush=True)
                except Exception as ex:
                    print(f"    Failed indexing camera {cam['id']}: {ex}", flush=True)
            else:
                print(f"    Source video {source} does not exist, skipping.", flush=True)

print(f"\n[7] Final region_clip_vit_b32_v1 count: {vector_db.region_coll.count()}", flush=True)
print("Vector store controlled recovery completed successfully!", flush=True)

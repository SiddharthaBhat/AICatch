"""CLI entrypoint: rebuild the current CLIP visual index for one investigation."""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CHROMA_PERSIST_DIR, SQLITE_DB_PATH
from pipeline.visual_reindex import rebuild_visual_embeddings
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild an investigation's active visual-vector index.")
    parser.add_argument("job_id", help="Investigation/job ID to rebuild")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(rebuild_visual_embeddings(
        SQLiteDatabase(SQLITE_DB_PATH), VectorDatabase(CHROMA_PERSIST_DIR), args.job_id, batch_size=args.batch_size,
    ))

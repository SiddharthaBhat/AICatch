"""Rebuild only derived visual vectors for an existing investigation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from config import EMBEDDING_MODEL_NAME, EMBEDDING_VERSION
from db_engine import get_image_embeddings
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase

LOGGER = logging.getLogger(__name__)


def rebuild_visual_embeddings(sqlite_db: SQLiteDatabase, vector_db: VectorDatabase, job_id: str, batch_size: int = 16) -> dict[str, Any]:
    """Re-embed valid object crops for one investigation into the active collection."""
    if not sqlite_db.get_job(job_id):
        raise ValueError(f"Unknown investigation {job_id}")
    before = vector_db.describe_visual_index({"job_id": job_id})["total"]
    vector_db.visual_coll.delete(where={"job_id": job_id})
    with sqlite_db.connection() as conn:
        rows = conn.execute(
            """SELECT d.id, d.crop_path, d.timestamp, d.entity_type, t.id, t.camera_id
                 FROM detections d JOIN tracks t ON t.id = d.track_id
                 WHERE t.job_id = ? ORDER BY d.frame_idx""", (job_id,)
        ).fetchall()
    valid = [(row, Path(row[1])) for row in rows if row[1] and Path(row[1]).is_file()]
    stats: dict[str, Any] = {"investigation_id": job_id, "collection": vector_db.visual_coll.name,
        "detections": len(rows), "object_crops": len(valid), "missing_crops": len(rows) - len(valid),
        "before_count": before, "insert_attempts": 0, "successful_inserts": 0,
        "embeddings_generated": 0, "sample_dimension": 0, "sample_norm": 0.0}
    for offset in range(0, len(valid), batch_size):
        chunk = valid[offset:offset + batch_size]
        images, kept = [], []
        for row, path in chunk:
            try:
                images.append(Image.open(path).convert("RGB"))
                kept.append(row)
            except Exception as exc:
                LOGGER.warning("Skipping unreadable crop %s: %s", path, exc)
        if not images:
            continue
        vectors = get_image_embeddings(images, batch_size=batch_size)
        ids, metas = [], []
        for row, vector in zip(kept, vectors):
            det_id, _, timestamp, entity_type, track_uuid, camera_id = row
            ids.append(det_id)
            metas.append({"job_id": job_id, "investigation_id": job_id, "camera_id": camera_id,
                "video_id": camera_id, "track_id": track_uuid, "detection_id": det_id,
                "timestamp": float(timestamp), "entity_type": entity_type, "object_type": entity_type,
                "embedding_model": EMBEDDING_MODEL_NAME, "embedding_version": EMBEDDING_VERSION,
                "quality_score": 0.0, "color_confidence": 0.0, "dominant_upper_color": "",
                "secondary_upper_color": "", "dominant_lower_color": "", "secondary_lower_color": ""})
        stats["insert_attempts"] += len(ids)
        stats["successful_inserts"] += vector_db.add_visual_embeddings(ids, vectors, metas)
        stats["embeddings_generated"] += len(vectors)
        if vectors and not stats["sample_dimension"]:
            stats["sample_dimension"] = len(vectors[0])
            stats["sample_norm"] = round(sum(x * x for x in vectors[0]) ** 0.5, 6)
    stats["after_count"] = vector_db.describe_visual_index({"job_id": job_id})["total"]
    LOGGER.info("VISUAL_REINDEX_COMPLETE %s", stats)
    return stats

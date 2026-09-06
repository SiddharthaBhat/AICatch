"""Thread-Safe Result Dispatcher for AICatch.

Splits incoming inference results into a real-time matching/alerting path
and an asynchronous database/media persistence path.
"""

import threading
import queue
import logging
import time
from typing import Dict, Any, List, Optional, Set, Callable

from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from pipeline.matcher import ActiveQueryMatcher
from utils.monitor import PerformanceMonitorInstance

LOGGER = logging.getLogger(__name__)

class ResultDispatcher:
    def __init__(
        self,
        sqlite_db: SQLiteDatabase,
        vector_db: VectorDatabase,
        matcher: ActiveQueryMatcher
    ):
        self.sqlite = sqlite_db
        self.vector = vector_db
        self.matcher = matcher
        
        # Bounded queue for database writes (persistence path)
        self.db_write_queue = queue.Queue(maxsize=2000)
        
        # Lock and control flags
        self.stopped = False
        self.write_thread = threading.Thread(target=self._db_write_loop, daemon=True)
        self.write_thread.start()
        
        # Pub/Sub alert queues for WebSocket/SSE broadcasters
        self.alert_listeners: List[Callable[[Dict[str, Any]], None]] = []
        self.listeners_lock = threading.Lock()
        
        # Per-job Ingestion Diagnostics Tracking
        self.job_stats: Dict[str, Dict[str, Any]] = {}
        self.stats_lock = threading.RLock()

    def get_or_create_job_stats(self, job_id: str) -> Dict[str, Any]:
        """Returns or initializes diagnostic metrics for a job."""
        with self.stats_lock:
            if job_id not in self.job_stats:
                self.job_stats[job_id] = {
                    "job_id": job_id,
                    "frames_processed": 0,
                    "frame_failures": 0,
                    "detections": 0,
                    "tracks": 0,
                    "crops": 0,
                    "embedding_attempts": 0,
                    "successful_embeddings": 0,
                    "failed_embeddings": 0,
                    "insertion_attempts": 0,
                    "successful_insertions": 0,
                    "failed_insertions": 0,
                    "verified_vectors": 0,
                    "embedding_model": None,
                    "embedding_version": None,
                    "embedding_dim": None,
                    "start_time": time.time(),
                    "end_time": None,
                    "duration_sec": 0.0,
                    "fps": 0.0,
                }
            return self.job_stats[job_id]

    def record_job_metric(self, job_id: str, **kwargs) -> None:
        """Increment or set diagnostic metrics for a job."""
        with self.stats_lock:
            stats = self.get_or_create_job_stats(job_id)
            for k, v in kwargs.items():
                if k in stats:
                    if isinstance(stats[k], int) and isinstance(v, int):
                        stats[k] += v
                    else:
                        stats[k] = v

    def add_alert_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for real-time alert delivery."""
        with self.listeners_lock:
            self.alert_listeners.append(listener)

    def remove_alert_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """Unregister a callback."""
        with self.listeners_lock:
            if listener in self.alert_listeners:
                self.alert_listeners.remove(listener)

    def dispatch(self, result: Dict[str, Any]) -> None:
        """Splits an inference result into synchronous matching and async writing with stage instrumentation."""
        if self.stopped:
            return
            
        t_start = time.time()
        job_id = result.get("job_id")
        track_uuid = result.get("track_uuid", "unknown")
        
        # Stage 1: Diagnostic Metrics Recording
        t0 = time.time()
        if job_id:
            try:
                self.record_job_metric(
                    job_id,
                    detections=1,
                    crops=1,
                    embedding_attempts=1 if result.get("should_embed") else 0,
                    successful_embeddings=1 if (result.get("should_embed") and result.get("clip_embedding") is not None) else 0,
                    failed_embeddings=1 if (result.get("should_embed") and result.get("clip_embedding") is None) else 0
                )
            except Exception as e:
                LOGGER.error(f"[DISPATCH: METRICS_RECORD] Error for track {track_uuid}: {e}", exc_info=True)
        t_metrics = time.time() - t0

        # Stage 2: Real-Time Active Query Matching
        t1 = time.time()
        try:
            matches = self.matcher.evaluate_result(result)
            if matches:
                for match in matches:
                    alert_payload = {
                        "query_id": match["query_id"],
                        "session_id": match["session_id"],
                        "similarity": match["similarity"],
                        "camera_id": result["camera_id"],
                        "timestamp": result["timestamp"],
                        "track_id": result["track_uuid"],
                        "entity_type": result["entity_type"],
                        "crop_path": result["crop_path"],
                        "plate_text": result["plate_text"],
                        "bbox": result["bbox"],
                        "job_id": result["job_id"]
                    }
                    self._broadcast_alert(alert_payload)
        except Exception as me:
            LOGGER.error(f"[DISPATCH: MATCHER_EVALUATE] Error for track {track_uuid}: {me}", exc_info=True)
        t_matcher = time.time() - t1

        # Stage 3: Async Persistence Queue Enqueue
        t2 = time.time()
        try:
            self.db_write_queue.put(result, block=True, timeout=5.0)
        except queue.Full:
            LOGGER.critical(f"[DISPATCH: QUEUE_PUT] Failed - write queue is full! Backpressure triggered for {track_uuid}.")
        except Exception as qe:
            LOGGER.error(f"[DISPATCH: QUEUE_PUT] Error enqueueing track {track_uuid}: {qe}", exc_info=True)
        t_queue = time.time() - t2
        
        t_total = time.time() - t_start
        LOGGER.debug(
            f"[DISPATCH: TRACE] track={track_uuid} total={t_total*1000:.2f}ms (metrics={t_metrics*1000:.2f}ms, matcher={t_matcher*1000:.2f}ms, queue={t_queue*1000:.2f}ms)"
        )

    def _broadcast_alert(self, alert: Dict[str, Any]) -> None:
        """Pushes alert payloads to all registered WebSocket/SSE queues."""
        with self.listeners_lock:
            for listener in self.alert_listeners:
                try:
                    listener(alert)
                except Exception as le:
                    LOGGER.debug(f"Failed to push alert to listener: {le}")

    def _db_write_loop(self) -> None:
        """Background thread worker pulling items and performing batch writes to SQL and ChromaDB."""
        batch_size = 64
        accumulated_results: List[Dict[str, Any]] = []
        last_flush_time = time.time()
        
        while not self.stopped:
            try:
                try:
                    result = self.db_write_queue.get(timeout=0.2)
                    accumulated_results.append(result)
                except queue.Empty:
                    pass
                
                now = time.time()
                time_since_flush = now - last_flush_time
                
                if accumulated_results and (len(accumulated_results) >= batch_size or time_since_flush >= 1.0):
                    start_db = time.time()
                    self._flush_batch(accumulated_results)
                    PerformanceMonitorInstance.record_latency("db", time.time() - start_db)
                    
                    accumulated_results.clear()
                    last_flush_time = now
                    
            except Exception as e:
                LOGGER.error(f"Error in database writer loop: {e}", exc_info=True)
                time.sleep(1.0)
        
        # Flush remaining on shutdown
        if accumulated_results:
            try:
                self._flush_batch(accumulated_results)
            except Exception as fe:
                LOGGER.error(f"Error flushing final batch on shutdown: {fe}")

    def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Executes actual SQLite schema transactions and ChromaDB vector updates in batches."""
        if not batch:
            return
            
        if not hasattr(self, "_first_flush_logged"):
            LOGGER.info(f"MILESTONE: FIRST RESULT STORED at {time.time()}")
            self._first_flush_logged = True
            
        # 1. Explicitly initialize all accumulation structures
        detections_to_insert = []
        vis_ids = []
        vis_embs = []
        vis_metas = []
        reid_ids = []
        reid_embs = []
        reid_metas = []
        face_ids = []
        face_embs = []
        face_metas = []
        
        job_counts: Dict[str, Dict[str, int]] = {}
        
        # SQLite track updates on a single connection transaction block
        with self.sqlite.connection() as conn:
            for result in batch:
                job_id = result["job_id"]
                camera_id = result["camera_id"]
                track_id = result["track_id"]
                track_uuid = result["track_uuid"]
                entity_type = result["entity_type"]
                timestamp = result["timestamp"]
                frame_idx = result["frame_idx"]
                bbox = result["bbox"]
                
                if job_id not in job_counts:
                    job_counts[job_id] = {"vis": 0, "reid": 0, "face": 0, "dets": 0}
                job_counts[job_id]["dets"] += 1
                
                # Step A: Register or update track in SQLite
                conn.execute(
                    "INSERT OR IGNORE INTO tracks (id, camera_id, job_id, track_id, entity_type, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (track_uuid, camera_id, job_id, track_id, entity_type, timestamp, timestamp)
                )
                
                plate_text = result.get("plate_text")
                if plate_text:
                    conn.execute(
                        "UPDATE tracks SET end_time = ?, plate_text = ? WHERE id = ?",
                        (timestamp, plate_text, track_uuid)
                    )
                else:
                    conn.execute(
                        "UPDATE tracks SET end_time = ? WHERE id = ?",
                        (timestamp, track_uuid)
                    )
                
                # Step B: Queue detection
                det_uuid = f"{track_uuid}_det_{frame_idx}"
                
                detections_to_insert.append((
                    det_uuid,
                    track_uuid,
                    frame_idx,
                    timestamp,
                    float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                    result["crop_path"],
                    result["confidence"],
                    entity_type,
                    result.get("quality_score", 0.0),
                    result.get("dominant_upper_color"),
                    result.get("secondary_upper_color"),
                    result.get("dominant_lower_color"),
                    result.get("secondary_lower_color"),
                    result.get("color_confidence", 0.0),
                    result.get("embedding_model"),
                    result.get("embedding_version")
                ))
                
                # Step C: Accumulate visual (CLIP) embedding ONLY on keyframes (should_embed = True)
                if result.get("clip_embedding") is not None and result.get("should_embed", False):
                    vis_ids.append(det_uuid)
                    vis_embs.append(result["clip_embedding"])
                    vis_metas.append({
                        "job_id": job_id,
                        "investigation_id": job_id,
                        "camera_id": camera_id,
                        "video_id": camera_id,
                        "track_id": track_uuid,
                        "detection_id": det_uuid,
                        "timestamp": timestamp,
                        "entity_type": entity_type,
                        "object_type": entity_type,
                        "dominant_upper_color": result.get("dominant_upper_color") or "",
                        "secondary_upper_color": result.get("secondary_upper_color") or "",
                        "dominant_lower_color": result.get("dominant_lower_color") or "",
                        "secondary_lower_color": result.get("secondary_lower_color") or "",
                        "color_confidence": result.get("color_confidence", 0.0),
                        "quality_score": result.get("quality_score", 0.0),
                        "embedding_model": result.get("embedding_model"),
                        "embedding_version": result.get("embedding_version"),
                    })
                    job_counts[job_id]["vis"] += 1

                # Step C2: Accumulate Person Re-ID embedding
                if result.get("reid_embedding") is not None and result.get("should_embed", False):
                    reid_ids.append(f"reid_{det_uuid}")
                    reid_embs.append(result["reid_embedding"])
                    reid_metas.append({
                        "job_id": job_id,
                        "camera_id": camera_id,
                        "track_id": track_uuid,
                        "detection_id": det_uuid,
                        "timestamp": timestamp,
                        "entity_type": entity_type,
                        "reid_model": result.get("reid_model", "mobilenet_v3_reid_576"),
                        "reid_version": result.get("reid_version", "v1")
                    })
                    job_counts[job_id]["reid"] += 1

                # Accumulate face embedding (if available)
                if result.get("face_embedding") is not None and result.get("face_crop_path") is not None:
                    face_emb_id = f"face_{track_uuid}_{frame_idx}"
                    face_ids.append(face_emb_id)
                    face_embs.append(result["face_embedding"])
                    face_metas.append({
                        "job_id": job_id,
                        "camera_id": camera_id,
                        "track_id": track_uuid,
                        "detection_id": det_uuid,
                        "timestamp": timestamp,
                        "face_crop_path": result["face_crop_path"],
                        "face_quality": result.get("face_quality", 0.0),
                        "model_name": "InceptionResnetV1_vggface2",
                        "model_version": "1.0"
                    })
                    job_counts[job_id]["face"] += 1
                    
            # Bulk save detections to SQLite (directly execute on conn)
            start_sqlite = time.time()
            print("    [STAGE: SQLITE PERSIST] START writing tracks & detections...", flush=True)
            if detections_to_insert:
                conn.executemany(
                    "INSERT OR REPLACE INTO detections (id, track_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, confidence, entity_type, quality_score, dominant_upper_color, secondary_upper_color, dominant_lower_color, secondary_lower_color, color_confidence, embedding_model, embedding_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    detections_to_insert
                )
            elapsed_sqlite = time.time() - start_sqlite
            print(f"    [STAGE: SQLITE PERSIST] END writing {len(detections_to_insert)} detections (Elapsed: {elapsed_sqlite:.3f}s)", flush=True)

        start_chroma = time.time()
        print("    [STAGE: CHROMA INSERT] START inserting embeddings...", flush=True)
        # Bulk save Person Re-ID embeddings to ChromaDB with write verification
        if reid_ids:
            try:
                self.vector.add_reid_embeddings(ids=reid_ids, embeddings=reid_embs, metadatas=reid_metas)
                self.vector.verify_vectors_written("reid", reid_ids, sample_size=5)
            except Exception as re_err:
                LOGGER.error(f"Failed to persist Re-ID embeddings to ChromaDB: {re_err}")
                for jid in job_counts:
                    self.record_job_metric(jid, failed_insertions=len(reid_ids))
                raise

        # Bulk save visual embeddings to ChromaDB with write verification
        if vis_ids:
            try:
                inserted = self.vector.add_visual_embeddings(ids=vis_ids, embeddings=vis_embs, metadatas=vis_metas)
                first = vis_embs[0]
                self.vector.verify_vectors_written("visual", vis_ids, sample_size=10)
                
                for jid, counts in job_counts.items():
                    self.record_job_metric(
                        jid,
                        insertion_attempts=counts["vis"],
                        successful_insertions=counts["vis"],
                        verified_vectors=counts["vis"],
                        embedding_dim=len(first),
                        embedding_model=vis_metas[0].get("embedding_model"),
                        embedding_version=vis_metas[0].get("embedding_version")
                    )
                LOGGER.info(
                    "VISUAL_INDEX_TRACE attempts=%d inserted=%d dim=%d norm=%.6f collection=%s",
                    len(vis_ids), inserted, len(first), sum(value * value for value in first) ** 0.5, self.vector.visual_coll.name,
                )
            except Exception as ve_err:
                LOGGER.exception("VISUAL_INDEX_TRACE insertion failed attempts=%d: %s", len(vis_ids), ve_err)
                for jid in job_counts:
                    self.record_job_metric(jid, failed_insertions=len(vis_ids))
                raise
            
        # Bulk save face embeddings to ChromaDB
        if face_ids:
            try:
                self.vector.add_face_embeddings(ids=face_ids, embeddings=face_embs, metadatas=face_metas)
                self.vector.verify_vectors_written("face", face_ids, sample_size=5)
            except Exception as fe_err:
                LOGGER.error(f"Failed to persist Face embeddings to ChromaDB: {fe_err}")
                raise
            
        elapsed_chroma = time.time() - start_chroma
        print(f"    [STAGE: CHROMA INSERT] END inserting {len(vis_ids)} visual, {len(reid_ids)} reid, {len(face_ids)} face vectors (Elapsed: {elapsed_chroma:.3f}s)", flush=True)
        LOGGER.info(f"Asynchronously flushed {len(batch)} inference metadata records to DBs.")


    def flush_and_wait(self, timeout: float = 10.0) -> None:
        """Blocks until the database write queue is completely drained."""
        start = time.time()
        while not self.db_write_queue.empty():
            if time.time() - start > timeout:
                LOGGER.warning("Timeout waiting for ResultDispatcher queue to drain.")
                break
            time.sleep(0.05)
        # Allow one additional cycle for any final batch flush
        time.sleep(0.2)

    def stop_dispatcher(self) -> None:
        """Shuts down background database writer."""
        self.stopped = True
        self.flush_and_wait(timeout=3.0)
        if self.write_thread.is_alive():
            self.write_thread.join(timeout=3.0)
        LOGGER.info("ResultDispatcher stopped.")

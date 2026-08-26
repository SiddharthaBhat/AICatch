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
        """Splits an inference result into synchronous matching and async writing."""
        if self.stopped:
            return
            
        # 1. Real-Time Alert Path (Does not block on DB)
        try:
            # Check for matches in Active Query Matcher
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
                    # Notify WS/SSE listeners
                    self._broadcast_alert(alert_payload)
        except Exception as me:
            LOGGER.error(f"Error in active query evaluation: {me}", exc_info=True)

        # 2. Persistence Path (Queue for async DB writer)
        try:
            # Bounded queue backpressure: if queue is full, log warning and block
            # (In production, we queue with block=True to protect SQLite lock states)
            self.db_write_queue.put(result, block=True, timeout=5.0)
        except queue.Full:
            LOGGER.critical("ResultDispatcher database write queue is full! Backpressure failed.")

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
        # Accumulation lists for batch writes
        batch_size = 64
        accumulated_results: List[Dict[str, Any]] = []
        last_flush_time = time.time()
        
        while not self.stopped:
            try:
                # Try to pull a result
                try:
                    # Timeout of 0.2s allows progressive flushes even during low traffic
                    result = self.db_write_queue.get(timeout=0.2)
                    accumulated_results.append(result)
                except queue.Empty:
                    pass
                
                # Check if we should flush batch
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

    def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Executes actual SQLite schema transactions and ChromaDB vector updates in batches."""
        if not hasattr(self, "_first_flush_logged"):
            LOGGER.info(f"MILESTONE: FIRST RESULT STORED at {time.time()}")
            self._first_flush_logged = True
            
        # 1. Group records for SQLite insertions
        detections_to_insert = []
        
        # ChromaDB bulk write accumulation
        vis_ids = []
        vis_embs = []
        vis_metas = []
        
        face_ids = []
        face_embs = []
        face_metas = []
        
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
                
                # Step A: Register or update track in SQLite (directly execute on conn)
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
                        "camera_id": camera_id,
                        "track_id": track_uuid,
                        "detection_id": det_uuid,
                        "timestamp": timestamp,
                        "entity_type": entity_type,
                        "dominant_upper_color": result.get("dominant_upper_color") or "",
                        "secondary_upper_color": result.get("secondary_upper_color") or "",
                        "dominant_lower_color": result.get("dominant_lower_color") or "",
                        "secondary_lower_color": result.get("secondary_lower_color") or "",
                        "color_confidence": result.get("color_confidence", 0.0),
                        "quality_score": result.get("quality_score", 0.0)
                    })
                    

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
                    
            # Bulk save detections to SQLite (directly execute on conn)
            if detections_to_insert:
                conn.executemany(
                    "INSERT INTO detections (id, track_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, confidence, entity_type, quality_score, dominant_upper_color, secondary_upper_color, dominant_lower_color, secondary_lower_color, color_confidence, embedding_model, embedding_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    detections_to_insert
                )
                

        # Bulk save Person Re-ID embeddings to ChromaDB
        if reid_ids:
            self.vector.add_reid_embeddings(ids=reid_ids, embeddings=reid_embs, metadatas=reid_metas)

        # Bulk save visual embeddings to ChromaDB
        if vis_ids:
            self.vector.add_visual_embeddings(ids=vis_ids, embeddings=vis_embs, metadatas=vis_metas)
            
        # Bulk save face embeddings to ChromaDB
        if face_ids:
            self.vector.add_face_embeddings(ids=face_ids, embeddings=face_embs, metadatas=face_metas)
            
        LOGGER.info(f"Asynchronously flushed {len(batch)} inference metadata records to DBs.")

    def stop_dispatcher(self) -> None:
        """Shuts down background database writer."""
        self.stopped = True
        if self.write_thread.is_alive():
            self.write_thread.join(timeout=3.0)
        LOGGER.info("ResultDispatcher stopped.")

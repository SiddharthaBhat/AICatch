"""SQLite Relational Database Manager for AICatch.

Handles connection pooling, WAL mode settings, relational schemas,
and thread-safe write/read queries for metadata indexing.
"""

import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager
from pathlib import Path
import time

from config import SQLITE_DB_PATH

LOGGER = logging.getLogger(__name__)

class SQLiteDatabase:
    def __init__(self, db_path: Path = SQLITE_DB_PATH):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def connection(self):
        """Context manager for thread-safe SQLite transactions with WAL enabled."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False
        )
        # Enable WAL mode and foreign keys
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            LOGGER.error(f"SQLite transaction failed: {e}")
            raise e
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize all schema tables and indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            # 1. Jobs Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                profile TEXT NOT NULL,
                duration_sec REAL DEFAULT 0.0,
                processed_sec REAL DEFAULT 0.0,
                fps REAL DEFAULT 0.0,
                cpu_usage REAL DEFAULT 0.0,
                gpu_usage REAL DEFAULT 0.0,
                vram_usage REAL DEFAULT 0.0,
                error_msg TEXT
            )""")

            # 2. Cameras Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                status TEXT NOT NULL,
                resolution TEXT,
                fps REAL,
                checksum TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )""")

            # 3. Frames Table (for downsampled thumbnails)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                thumbnail_path TEXT,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            )""")

            # 4. Tracks Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                plate_text TEXT,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )""")

            # 5. Detections Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                track_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                bbox_x1 REAL NOT NULL,
                bbox_y1 REAL NOT NULL,
                bbox_x2 REAL NOT NULL,
                bbox_y2 REAL NOT NULL,
                crop_path TEXT NOT NULL,
                confidence REAL NOT NULL,
                entity_type TEXT NOT NULL,
                quality_score REAL,
                dominant_upper_color TEXT,
                secondary_upper_color TEXT,
                dominant_lower_color TEXT,
                secondary_lower_color TEXT,
                color_confidence REAL,
                embedding_model TEXT,
                embedding_version TEXT,
                FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE
            )""")

            # 6. Saved Persons Table (for named face recognition)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_persons (
                id TEXT PRIMARY KEY,
                job_id TEXT,
                name TEXT NOT NULL,
                face_image_path TEXT NOT NULL,
                face_embedding_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )""")

            # 7. Active Alert Queries Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS active_queries (
                id TEXT PRIMARY KEY,
                job_id TEXT,
                session_id TEXT NOT NULL,
                type TEXT NOT NULL,
                query_text TEXT,
                query_image_path TEXT,
                query_embedding_id TEXT,
                similarity_threshold REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )""")

            # 8. Journey Events Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS journey_events (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                duration REAL NOT NULL,
                transition_confidence REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY(track_id) REFERENCES tracks(id) ON DELETE CASCADE,
                FOREIGN KEY(camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            )""")

            # 9. Performance Stats Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS perf_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                processing_fps REAL NOT NULL,
                queue_depth INTEGER NOT NULL,
                cpu_util REAL NOT NULL,
                gpu_util REAL NOT NULL,
                vram_used REAL NOT NULL,
                yolo_lat REAL NOT NULL,
                clip_lat REAL NOT NULL,
                ocr_lat REAL NOT NULL,
                face_lat REAL NOT NULL,
                db_lat REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )""")

            # 10. Folders Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY,
                job_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            )""")

            # 11. Folder Items Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS folder_items (
                id TEXT PRIMARY KEY,
                folder_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY(folder_id) REFERENCES folders(id) ON DELETE CASCADE
            )""")

            # 12. Notes Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                parent_type TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )""")

            # Migration check for existing databases
            cursor = conn.execute("PRAGMA table_info(detections)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            new_cols = [
                ("quality_score", "REAL"),
                ("dominant_upper_color", "TEXT"),
                ("secondary_upper_color", "TEXT"),
                ("dominant_lower_color", "TEXT"),
                ("secondary_lower_color", "TEXT"),
                ("color_confidence", "REAL"),
                ("embedding_model", "TEXT"),
                ("embedding_version", "TEXT")
            ]
            
            for col_name, col_type in new_cols:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE detections ADD COLUMN {col_name} {col_type}")
                    LOGGER.info(f"Database Migration: Added column '{col_name}' to 'detections' table.")

            # Migration check for tracks table (label column)
            cursor_t = conn.execute("PRAGMA table_info(tracks)")
            existing_cols_t = {row[1] for row in cursor_t.fetchall()}
            if "label" not in existing_cols_t:
                conn.execute("ALTER TABLE tracks ADD COLUMN label TEXT")
                LOGGER.info("Database Migration: Added column 'label' to 'tracks' table.")

            # Migration check for jobs table (investigation_type & diagnostics columns)
            cursor_j = conn.execute("PRAGMA table_info(jobs)")
            existing_cols_j = {row[1] for row in cursor_j.fetchall()}
            if "investigation_type" not in existing_cols_j:
                conn.execute("ALTER TABLE jobs ADD COLUMN investigation_type TEXT DEFAULT 'single_video'")
                LOGGER.info("Database Migration: Added column 'investigation_type' to 'jobs' table.")
            if "diagnostics" not in existing_cols_j:
                conn.execute("ALTER TABLE jobs ADD COLUMN diagnostics TEXT")
                LOGGER.info("Database Migration: Added column 'diagnostics' to 'jobs' table.")


            # Migration check for folders table (job_id column)
            cursor_f = conn.execute("PRAGMA table_info(folders)")
            existing_cols_f = {row[1] for row in cursor_f.fetchall()}
            if "job_id" not in existing_cols_f:
                conn.execute("ALTER TABLE folders ADD COLUMN job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE")
                LOGGER.info("Database Migration: Added column 'job_id' to 'folders' table.")

            # Migration check for saved_persons table (job_id column)
            cursor_sp = conn.execute("PRAGMA table_info(saved_persons)")
            existing_cols_sp = {row[1] for row in cursor_sp.fetchall()}
            if "job_id" not in existing_cols_sp:
                conn.execute("ALTER TABLE saved_persons ADD COLUMN job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE")
                LOGGER.info("Database Migration: Added column 'job_id' to 'saved_persons' table.")

            # Migration check for active_queries table (job_id column)
            cursor_aq = conn.execute("PRAGMA table_info(active_queries)")
            existing_cols_aq = {row[1] for row in cursor_aq.fetchall()}
            if "job_id" not in existing_cols_aq:
                conn.execute("ALTER TABLE active_queries ADD COLUMN job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE")
                LOGGER.info("Database Migration: Added column 'job_id' to 'active_queries' table.")

            # Migration check for cameras table (checksum column)
            cursor_c = conn.execute("PRAGMA table_info(cameras)")
            existing_cols_c = {row[1] for row in cursor_c.fetchall()}
            if "checksum" not in existing_cols_c:
                conn.execute("ALTER TABLE cameras ADD COLUMN checksum TEXT")
                LOGGER.info("Database Migration: Added column 'checksum' to 'cameras' table.")

            # Automatic repair: Associate orphaned folders to jobs based on items
            cursor_f_repair = conn.execute("SELECT id FROM folders WHERE job_id IS NULL")
            unassigned_folders = [row[0] for row in cursor_f_repair.fetchall()]
            for fid in unassigned_folders:
                cur_t = conn.execute("""
                    SELECT DISTINCT t.job_id 
                    FROM folder_items fi 
                    JOIN tracks t ON fi.item_id = t.id 
                    WHERE fi.folder_id = ? AND fi.item_type = 'track'
                """, (fid,))
                jobs_found = {r[0] for r in cur_t.fetchall()}
                
                cur_d = conn.execute("""
                    SELECT DISTINCT t.job_id 
                    FROM folder_items fi 
                    JOIN detections d ON fi.item_id = d.id 
                    JOIN tracks t ON d.track_id = t.id 
                    WHERE fi.folder_id = ? AND fi.item_type = 'detection'
                """, (fid,))
                jobs_found.update({r[0] for r in cur_d.fetchall()})
                
                if len(jobs_found) == 1:
                    resolved_job = list(jobs_found)[0]
                    conn.execute("UPDATE folders SET job_id = ? WHERE id = ?", (resolved_job, fid))
                    LOGGER.info(f"Database Migration: Repaired folder '{fid}' by assigning to job '{resolved_job}' based on items.")
                elif len(jobs_found) > 1:
                    LOGGER.warning(f"Database Migration: Folder '{fid}' contains mixed data from multiple jobs {jobs_found}. Left unassigned.")

            # Performance Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_camera ON tracks(camera_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_job ON tracks(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_camera ON frames(camera_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journey_job ON journey_events(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_job ON perf_stats(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_folders_job ON folders(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_saved_persons_job ON saved_persons(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_active_queries_job ON active_queries(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cameras_checksum ON cameras(checksum)")
            
            # Object-Level Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_upper_color ON detections(dominant_upper_color)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_entity_type ON detections(entity_type)")

    # ==============================================================================
    # Write Methods
    # ==============================================================================

    def add_job(self, job_id: str, name: str, profile: str, investigation_type: str = "single_video") -> None:
        """Insert a new job metadata record."""
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO jobs (id, name, status, created_at, updated_at, profile, investigation_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, name, "queued", now, now, profile, investigation_type)
            )

    def delete_job(self, job_id: str) -> None:
        """Deletes a job and its associated data from SQLite using cascading deletes."""
        with self.connection() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def update_job_status(self, job_id: str, status: str, error_msg: Optional[str] = None, diagnostics: Optional[str] = None) -> None:
        """Update job execution state."""
        now = time.time()
        with self.connection() as conn:
            if diagnostics is not None:
                conn.execute(
                    "UPDATE jobs SET status = ?, error_msg = ?, diagnostics = ?, updated_at = ? WHERE id = ?",
                    (status, error_msg, diagnostics, now, job_id)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = ?, error_msg = ?, updated_at = ? WHERE id = ?",
                    (status, error_msg, now, job_id)
                )


    def update_job_progress(self, job_id: str, processed_sec: float, duration_sec: float, fps: float) -> None:
        """Update indexing performance metrics."""
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "UPDATE jobs SET processed_sec = ?, duration_sec = ?, fps = ?, updated_at = ? WHERE id = ?",
                (processed_sec, duration_sec, fps, now, job_id)
            )

    def add_camera(self, camera_id: str, job_id: str, name: str, source_type: str, source_path: str, checksum: Optional[str] = None) -> None:
        """Register a camera source associated with a job."""
        initial_status = "inactive" if source_path else "not_configured"
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO cameras (id, job_id, name, source_type, source_path, status, checksum) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (camera_id, job_id, name, source_type, source_path, initial_status, checksum)
            )

    def update_camera_source(self, camera_id: str, source_path: str, name: Optional[str] = None, checksum: Optional[str] = None) -> None:
        """Update camera source path, name, and checksum."""
        status = "inactive" if source_path else "not_configured"
        with self.connection() as conn:
            if name:
                conn.execute(
                    "UPDATE cameras SET source_path = ?, name = ?, checksum = ?, status = ? WHERE id = ?",
                    (source_path, name, checksum, status, camera_id)
                )
            else:
                conn.execute(
                    "UPDATE cameras SET source_path = ?, checksum = ?, status = ? WHERE id = ?",
                    (source_path, checksum, status, camera_id)
                )

    def update_camera_metadata(self, camera_id: str, status: str, resolution: str, fps: float) -> None:
        """Update camera parameters after opening source."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE cameras SET status = ?, resolution = ?, fps = ? WHERE id = ?",
                (status, resolution, fps, camera_id)
            )

    def add_frame(self, frame_id: str, camera_id: str, frame_idx: int, timestamp: float, thumbnail_path: str) -> None:
        """Record processed frame mapping."""
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO frames (id, camera_id, frame_idx, timestamp, thumbnail_path) VALUES (?, ?, ?, ?, ?)",
                (frame_id, camera_id, frame_idx, timestamp, thumbnail_path)
            )

    def add_track(self, track_uuid: str, camera_id: str, job_id: str, track_id: int, entity_type: str, start_time: float) -> None:
        """Register a new tracked entity path."""
        with self.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tracks (id, camera_id, job_id, track_id, entity_type, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (track_uuid, camera_id, job_id, track_id, entity_type, start_time, start_time)
            )

    def update_track_endpoint(self, track_uuid: str, end_time: float, plate_text: Optional[str] = None) -> None:
        """Extend track duration and assign OCR text if available."""
        with self.connection() as conn:
            if plate_text:
                conn.execute(
                    "UPDATE tracks SET end_time = ?, plate_text = ? WHERE id = ?",
                    (end_time, plate_text, track_uuid)
                )
            else:
                conn.execute(
                    "UPDATE tracks SET end_time = ? WHERE id = ?",
                    (end_time, track_uuid)
                )

    def add_detections_batch(self, detections: List[Tuple[str, str, int, float, float, float, float, float, str, float, str]]) -> None:
        """Batch save detections to reduce filesystem/transaction load."""
        with self.connection() as conn:
            conn.executemany(
                "INSERT INTO detections (id, track_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, confidence, entity_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                detections
            )

    def add_saved_person(self, person_id: str, name: str, face_image_path: str, face_embedding_id: str, job_id: Optional[str] = None) -> None:
        """Save a face profile for custom searches."""
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO saved_persons (id, job_id, name, face_image_path, face_embedding_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (person_id, job_id, name, face_image_path, face_embedding_id, now)
            )

    def add_active_query(self, query_id: str, session_id: str, qtype: str, query_text: Optional[str], query_image_path: Optional[str], query_embedding_id: Optional[str], threshold: float, job_id: Optional[str] = None) -> None:
        """Register active query for same-pass alerting."""
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO active_queries (id, job_id, session_id, type, query_text, query_image_path, query_embedding_id, similarity_threshold, is_active, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (query_id, job_id, session_id, qtype, query_text, query_image_path, query_embedding_id, threshold, now)
            )

    def deactivate_query(self, query_id: str) -> None:
        """Remove alert configuration."""
        with self.connection() as conn:
            conn.execute("UPDATE active_queries SET is_active = 0 WHERE id = ?", (query_id,))

    def add_journey_event(self, event_id: str, job_id: str, track_id: str, camera_id: str, timestamp: float, duration: float, confidence: float) -> None:
        """Record dynamic trajectory sequence event."""
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO journey_events (id, job_id, track_id, camera_id, timestamp, duration, transition_confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, job_id, track_id, camera_id, timestamp, duration, confidence)
            )

    def add_perf_stat(self, job_id: str, fps: float, q_depth: int, cpu: float, gpu: float, vram: float, yolo_l: float, clip_l: float, ocr_l: float, face_l: float, db_l: float) -> None:
        """Record dynamic hardware telemetry stats."""
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO perf_stats (job_id, timestamp, processing_fps, queue_depth, cpu_util, gpu_util, vram_used, yolo_lat, clip_lat, ocr_lat, face_lat, db_lat) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, now, fps, q_depth, cpu, gpu, vram, yolo_l, clip_l, ocr_l, face_l, db_l)
            )

    # ==============================================================================
    # Read Methods
    # ==============================================================================

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch job record details."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_jobs(self) -> List[Dict[str, Any]]:
        """List all registered jobs."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_cameras_by_job(self, job_id: str) -> List[Dict[str, Any]]:
        """Get list of cameras associated with a job."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM cameras WHERE job_id = ?", (job_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get a single camera by ID."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
            row = cursor.fetchone()
            return dict(row) if row else None


    def get_track(self, track_uuid: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific entity path."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_uuid,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_tracks_by_ids(self, track_uuids: List[str]) -> List[Dict[str, Any]]:
        """Fetch tracks by primary keys."""
        if not track_uuids:
            return []
        placeholders = ",".join("?" for _ in track_uuids)
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"SELECT * FROM tracks WHERE id IN ({placeholders})", track_uuids)
            return [dict(row) for row in cursor.fetchall()]

    def get_detections_by_track(self, track_uuid: str) -> List[Dict[str, Any]]:
        """Get all chronological detections in a track."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM detections WHERE track_id = ? ORDER BY frame_idx ASC", (track_uuid,))
            return [dict(row) for row in cursor.fetchall()]

    def get_saved_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific saved person profile by ID."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM saved_persons WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_saved_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Delete a saved person profile and return their metadata for media cleanup."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM saved_persons WHERE id = ?", (person_id,))
            row = cursor.fetchone()
            if not row:
                return None
            record = dict(row)
            conn.execute("DELETE FROM saved_persons WHERE id = ?", (person_id,))
            return record

    def delete_active_query(self, query_id: str) -> bool:
        """Permanently delete an active query alert rule."""
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM active_queries WHERE id = ?", (query_id,))
            return cursor.rowcount > 0

    def get_saved_persons(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve saved identity profiles, optionally filtered by job_id."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            if job_id:
                cursor = conn.execute("SELECT * FROM saved_persons WHERE job_id = ? ORDER BY created_at DESC", (job_id,))
            else:
                cursor = conn.execute("SELECT * FROM saved_persons ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_active_queries(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve active alert rules, optionally filtered by job_id."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            if job_id:
                cursor = conn.execute("SELECT * FROM active_queries WHERE is_active = 1 AND job_id = ?", (job_id,))
            else:
                cursor = conn.execute("SELECT * FROM active_queries WHERE is_active = 1")
            return [dict(row) for row in cursor.fetchall()]

    def get_journey_events(self, job_id: str) -> List[Dict[str, Any]]:
        """Fetch reconstructed transition events for a job."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM journey_events WHERE job_id = ? ORDER BY timestamp ASC", (job_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_perf_stats(self, job_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch telemetry sequence logs for performance analysis."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM perf_stats WHERE job_id = ? ORDER BY timestamp DESC LIMIT ?", (job_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def search_plates(self, query_text: str, camera_filter: Optional[str] = None, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform exact/wildcard search on license plates."""
        # Clean query
        clean = f"%{query_text.upper().strip()}%"
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT t.*, d.crop_path, d.timestamp as det_time, d.frame_idx FROM tracks t JOIN detections d ON d.track_id = t.id WHERE t.plate_text LIKE ?"
            params = [clean]
            
            if camera_filter:
                query += " AND t.camera_id = ?"
                params.append(camera_filter)
                
            if job_id:
                query += " AND t.job_id = ?"
                params.append(job_id)
                
            query += " GROUP BY t.id"
            cursor = conn.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    # ==============================================================================
    # Folders, Folder Items, Notes, Labels Methods
    # ==============================================================================
    def update_track_label(self, track_uuid: str, label: str) -> None:
        """Assign or update a user-defined label for a track."""
        with self.connection() as conn:
            conn.execute("UPDATE tracks SET label = ? WHERE id = ?", (label, track_uuid))

    def create_folder(self, folder_id: str, name: str, description: Optional[str] = None, job_id: Optional[str] = None) -> None:
        """Create a new investigation folder."""
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO folders (id, job_id, name, description, created_at) VALUES (?, ?, ?, ?, ?)",
                (folder_id, job_id, name, description, now)
            )

    def delete_folder(self, folder_id: str) -> None:
        """Delete an investigation folder (cascades items and notes)."""
        with self.connection() as conn:
            conn.execute("DELETE FROM notes WHERE parent_type = 'folder' AND parent_id = ?", (folder_id,))
            conn.execute("DELETE FROM folder_items WHERE folder_id = ?", (folder_id,))
            conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))

    def get_folders(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all investigation folders, optionally filtered by job_id."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            if job_id:
                cursor = conn.execute("SELECT * FROM folders WHERE job_id = ? ORDER BY created_at DESC", (job_id,))
            else:
                cursor = conn.execute("SELECT * FROM folders ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def add_folder_item(self, item_id: str, folder_id: str, item_type: str, target_id: str, metadata: Optional[str] = None) -> None:
        """Add an observation, crop, clip, or event to a folder."""
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO folder_items (id, folder_id, item_type, item_id, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, folder_id, item_type, target_id, metadata, now)
            )

    def delete_folder_item(self, folder_id: str, target_id: str) -> None:
        """Remove a specific item from a folder."""
        with self.connection() as conn:
            conn.execute("DELETE FROM folder_items WHERE folder_id = ? AND (item_id = ? OR id = ?)", (folder_id, target_id, target_id))

    def get_folder_items(self, folder_id: str) -> List[Dict[str, Any]]:
        """Get all items inside a folder."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM folder_items WHERE folder_id = ? ORDER BY created_at ASC", (folder_id,))
            return [dict(row) for row in cursor.fetchall()]

    def add_note(self, note_id: str, parent_type: str, parent_id: str, content: str) -> None:
        """Save a note against a folder or track."""
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO notes (id, parent_type, parent_id, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (note_id, parent_type, parent_id, content, now)
            )

    def delete_note(self, note_id: str) -> None:
        """Remove a note."""
        with self.connection() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    def get_notes(self, parent_type: str, parent_id: str) -> List[Dict[str, Any]]:
        """Get all notes for a specific folder or track."""
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM notes WHERE parent_type = ? AND parent_id = ? ORDER BY created_at DESC", (parent_type, parent_id))
            return [dict(row) for row in cursor.fetchall()]

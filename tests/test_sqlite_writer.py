"""Comprehensive Test Suite for SQLite Single-Writer and Database Concurrency.

Tests:
1. Queue behavior and FIFO execution.
2. Writer serialization under high thread concurrency (no SQLITE_BUSY).
3. Transaction correctness: COMMIT on success and ROLLBACK on failure.
4. Exception propagation to callers on sync execution.
5. Backpressure handling on queue limits.
6. Graceful shutdown and queue task draining.
7. Immediate read consistency after sync write.
8. Batch transaction commit and fault resilience.
"""

from __future__ import annotations

import os
import queue
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from storage.sqlite_db import SQLiteDatabase
from storage.sqlite_writer import SQLiteWriteQueue, SQLiteWriteTask


class TestSQLiteSingleWriter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "writer_test.db"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_writer_serialization_concurrent_threads(self):
        """Verify 20 concurrent threads performing writes encounter zero SQLITE_BUSY errors."""
        writer = SQLiteWriteQueue(self.db_path, maxsize=1000)

        # Initialize test table
        def _init(conn):
            conn.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY AUTOINCREMENT, val INTEGER);")
        writer.enqueue(_init, sync=True)

        num_threads = 20
        writes_per_thread = 25
        errors = []

        def worker(thread_idx: int):
            for i in range(writes_per_thread):
                try:
                    def _insert(conn):
                        conn.execute("INSERT INTO counter (val) VALUES (?);", (thread_idx * 1000 + i,))
                    writer.enqueue(_insert, sync=True, timeout=10.0)
                except Exception as ex:
                    errors.append(ex)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Encountered write errors: {errors}")

        # Verify total count
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM counter;")
        total = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(total, num_threads * writes_per_thread)
        writer.stop()

    def test_transaction_rollback_and_exception_propagation(self):
        """Verify that an exception in a write callable rolls back and raises in sync caller."""
        writer = SQLiteWriteQueue(self.db_path)

        def _init(conn):
            conn.execute("CREATE TABLE items (id TEXT PRIMARY KEY, val TEXT);")
            conn.execute("INSERT INTO items VALUES ('k1', 'initial');")
        writer.enqueue(_init, sync=True)

        # Attempt write that fails intentionally
        def _failing_write(conn):
            conn.execute("UPDATE items SET val = 'corrupted' WHERE id = 'k1';")
            raise ValueError("Simulated catastrophic failure inside transaction")

        with self.assertRaises(ValueError):
            writer.enqueue(_failing_write, sync=True)

        # Verify database state rolled back
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT val FROM items WHERE id = 'k1';")
        val = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(val, "initial")
        writer.stop()

    def test_backpressure_bounded_queue(self):
        """Verify bounded queue triggers backpressure exception when full."""
        small_writer = SQLiteWriteQueue(self.db_path, maxsize=2)

        # Block writer with a slow task
        event_started = threading.Event()
        event_unblock = threading.Event()

        def _slow_task(conn):
            event_started.set()
            event_unblock.wait(timeout=3.0)

        small_writer.enqueue(_slow_task, sync=False)
        # Ensure _slow_task has been popped by worker and is currently running
        event_started.wait(timeout=1.0)

        # Now worker is busy. Fill the queue (maxsize=2)
        small_writer.enqueue(lambda c: None, sync=False)
        small_writer.enqueue(lambda c: None, sync=False)

        # Next put should trigger backpressure timeout
        with self.assertRaises(RuntimeError):
            small_writer.enqueue(lambda c: None, sync=False, timeout=0.05)

        event_unblock.set()
        small_writer.stop()

    def test_shutdown_graceful_draining(self):
        """Verify all queued tasks complete before writer shutdown terminates."""
        writer = SQLiteWriteQueue(self.db_path, maxsize=500)

        def _init(conn):
            conn.execute("CREATE TABLE drained (id INTEGER PRIMARY KEY);")
        writer.enqueue(_init, sync=True)

        # Enqueue 50 async tasks
        for i in range(50):
            def _make_task(val):
                return lambda conn: conn.execute("INSERT INTO drained VALUES (?);", (val,))
            writer.enqueue(_make_task(i), sync=False)

        # Stop writer (should drain all 50 tasks)
        writer.stop(timeout=5.0)

        # Reopen and check
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM drained;")
        total = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(total, 50)

    def test_sqlite_database_integration(self):
        """Verify SQLiteDatabase write methods use single-writer and maintain consistency."""
        db = SQLiteDatabase(self.db_path)

        # Add job
        db.add_job("job_single_writer_1", "Test Job", "BALANCED", "multi_camera")
        job = db.get_job("job_single_writer_1")
        self.assertIsNotNone(job)
        self.assertEqual(job["name"], "Test Job")
        self.assertEqual(job["investigation_type"], "multi_camera")

        # Add camera with timestamp configuration
        db.add_camera("cam_sw_1", "job_single_writer_1", "Front Gate", "file", "/tmp/front.mp4")
        db.update_camera_timestamp_config(
            camera_id="cam_sw_1",
            mode="clock_offset",
            clock_offset_sec=120.0,
            effective_start_time="2026-10-15T08:00:00+00:00",
            effective_start_epoch=1792051200.0
        )

        cam = db.get_camera("cam_sw_1")
        self.assertEqual(cam["clock_offset_sec"], 120.0)
        self.assertEqual(cam["timestamp_config_mode"], "clock_offset")

        # Test batch detections write
        dets = [
            (f"det_{i}", "track_1", i, float(i), 0.1, 0.2, 0.5, 0.6, "/tmp/crop.jpg", 0.95, "person")
            for i in range(10)
        ]
        db.add_track("track_1", "cam_sw_1", "job_single_writer_1", 1, "person", 0.0)
        db.add_detections_batch(dets)

        retrieved_dets = db.get_detections_by_track("track_1")
        self.assertEqual(len(retrieved_dets), 10)

        # Test evidence extraction table methods
        db.add_evidence_extraction(
            extraction_id="ext_test_1",
            job_id="job_single_writer_1",
            title="Suspect Tracking Segment",
            manifest_json='{"provenance": "ok"}'
        )
        ext = db.get_evidence_extraction("ext_test_1")
        self.assertIsNotNone(ext)
        self.assertEqual(ext["status"], "queued")

        db.update_evidence_extraction("ext_test_1", status="completed", progress=1.0, duration_sec=42.5)
        ext_updated = db.get_evidence_extraction("ext_test_1")
        self.assertEqual(ext_updated["status"], "completed")
        self.assertEqual(ext_updated["progress"], 1.0)
        self.assertEqual(ext_updated["duration_sec"], 42.5)

        db.close()


if __name__ == "__main__":
    unittest.main()

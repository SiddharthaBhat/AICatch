"""SQLite Single-Writer Engine for AICatch.

Guarantees:
1. Exactly one dedicated background writer thread owning a write connection.
2. Serialized writes to eliminate SQLITE_BUSY / database locked contention.
3. Bounded queue with backpressure handling.
4. Automatic transaction demarcation (BEGIN IMMEDIATE / COMMIT / ROLLBACK).
5. Synchronous and asynchronous write execution modes.
6. Batching of pending compatible writes to optimize disk flush throughput.
7. Graceful draining on shutdown with timeout protection.
8. Complete error propagation to callers (no silent write loss).
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
import traceback
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

LOGGER = logging.getLogger(__name__)


class SQLiteWriteTask:
    """Encapsulates a unit of database write work."""

    def __init__(
        self,
        func: Callable[[sqlite3.Connection], Any],
        future: Optional[Future] = None,
        is_batch: bool = False,
        task_name: str = "write"
    ):
        self.func = func
        self.future = future
        self.is_batch = is_batch
        self.task_name = task_name
        self.created_at = time.time()


class SQLiteWriteQueue:
    """Thread-safe single-writer queue for SQLite databases.

    Serializes all mutation queries through a designated writer thread to prevent
    concurrency collisions in SQLite WAL mode.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        maxsize: int = 2000,
        busy_timeout: float = 30.0,
        enable_wal: bool = True
    ):
        self.db_path = Path(db_path)
        self.busy_timeout = float(busy_timeout)
        self.enable_wal = enable_wal
        self.maxsize = maxsize

        self.task_queue: queue.Queue[Optional[SQLiteWriteTask]] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._is_running = False
        self._failed_tasks_count = 0
        self._successful_tasks_count = 0
        self._metrics_lock = threading.Lock()

        # Start designated single writer thread
        self._thread = threading.Thread(
            target=self._worker,
            name="SQLiteSingleWriter",
            daemon=True
        )
        self._thread.start()
        self._is_running = True
        LOGGER.info("SQLiteSingleWriter initialized for %s (queue maxsize=%d)", self.db_path, maxsize)

    def _open_writer_connection(self) -> sqlite3.Connection:
        """Opens and configures the dedicated writer connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout,
            check_same_thread=False,
            isolation_level=None  # Explicit transaction control
        )
        if self.enable_wal:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={int(self.busy_timeout * 1000)};")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _worker(self) -> None:
        """Worker thread body executing write tasks serially with batch draining."""
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._open_writer_connection()
        except Exception as exc:
            LOGGER.critical("SQLiteSingleWriter failed to open writer connection: %s", exc, exc_info=True)
            self._is_running = False
            return

        while not self._stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                self.task_queue.task_done()
                break

            self._execute_task_or_batch(conn, task)

        # Drain any remaining tasks before terminating connection
        self._drain_remaining(conn)

        try:
            conn.close()
        except Exception as close_err:
            LOGGER.debug("SQLite writer connection close note: %s", close_err)
        LOGGER.info("SQLiteSingleWriter terminated cleanly for %s", self.db_path)

    def _execute_task_or_batch(self, conn: sqlite3.Connection, first_task: SQLiteWriteTask) -> None:
        """Executes the task, batching additional pending tasks if available."""
        batch: List[SQLiteWriteTask] = [first_task]

        # Greedily pull up to 64 additional pending tasks for batch transaction commit
        while len(batch) < 64:
            try:
                next_task = self.task_queue.get_nowait()
                if next_task is None:
                    # Put back sentinel for graceful shutdown loop
                    self.task_queue.put(None)
                    break
                batch.append(next_task)
            except queue.Empty:
                break

        if len(batch) == 1:
            self._execute_single(conn, batch[0])
            self.task_queue.task_done()
        else:
            self._execute_batch(conn, batch)
            for _ in batch:
                self.task_queue.task_done()

    def _execute_single(self, conn: sqlite3.Connection, task: SQLiteWriteTask) -> None:
        """Executes a single write task within an explicit transaction."""
        try:
            conn.execute("BEGIN IMMEDIATE;")
            res = task.func(conn)
            conn.execute("COMMIT;")
            with self._metrics_lock:
                self._successful_tasks_count += 1
            if task.future and not task.future.done():
                task.future.set_result(res)
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            with self._metrics_lock:
                self._failed_tasks_count += 1
            LOGGER.error("SQLite write task '%s' failed: %s", task.task_name, exc)
            LOGGER.debug(traceback.format_exc())
            if task.future and not task.future.done():
                task.future.set_exception(exc)

    def _execute_batch(self, conn: sqlite3.Connection, batch: List[SQLiteWriteTask]) -> None:
        """Executes multiple tasks inside one combined transaction for high throughput."""
        try:
            conn.execute("BEGIN IMMEDIATE;")
            results = []
            for task in batch:
                res = task.func(conn)
                results.append((task, res, None))
            conn.execute("COMMIT;")
            with self._metrics_lock:
                self._successful_tasks_count += len(batch)
            for task, res, _ in results:
                if task.future and not task.future.done():
                    task.future.set_result(res)
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            LOGGER.warning("Batch transaction failed (%s); retrying tasks individually...", exc)
            # Fallback: Retry each task individually so that one faulty write does not discard good ones
            for task in batch:
                self._execute_single(conn, task)

    def _drain_remaining(self, conn: sqlite3.Connection) -> None:
        """Drains all remaining tasks in the queue on shutdown."""
        while True:
            try:
                task = self.task_queue.get_nowait()
            except queue.Empty:
                break
            if task is None:
                self.task_queue.task_done()
                continue
            self._execute_single(conn, task)
            self.task_queue.task_done()

    def enqueue(
        self,
        func: Callable[[sqlite3.Connection], Any],
        *,
        sync: bool = False,
        timeout: Optional[float] = None,
        task_name: str = "write"
    ) -> Any:
        """Submits a write callable to the single-writer queue.

        Args:
            func: Function taking `sqlite3.Connection` as sole argument.
            sync: If True, blocks until completion and returns the result (or raises).
            timeout: Maximum wait time for sync result or queue insertion.
            task_name: Descriptive name for logging and telemetry.

        Returns:
            The return value of `func(conn)` if `sync=True`, else None.
        """
        if self._stop_event.is_set():
            raise RuntimeError(f"SQLiteWriteQueue for {self.db_path} has been stopped.")

        future: Optional[Future] = Future() if sync else None
        task = SQLiteWriteTask(func=func, future=future, task_name=task_name)

        # Backpressure: bounded put with timeout
        q_timeout = timeout if not sync else 10.0
        try:
            self.task_queue.put(task, block=True, timeout=q_timeout)
        except queue.Full:
            LOGGER.critical("SQLiteWriteQueue is full (%d items)! Backpressure triggered for %s", self.maxsize, task_name)
            raise RuntimeError(f"SQLiteWriteQueue backpressure: queue full ({self.maxsize} tasks).")

        if sync and future:
            return future.result(timeout=timeout)
        return None

    def execute_write(
        self,
        func: Callable[[sqlite3.Connection], Any],
        sync: bool = True,
        timeout: Optional[float] = 30.0,
        task_name: str = "write"
    ) -> Any:
        """Convenience alias for enqueue."""
        return self.enqueue(func, sync=sync, timeout=timeout, task_name=task_name)

    def get_stats(self) -> Dict[str, Any]:
        """Telemetry snapshot of queue depth and task outcomes."""
        with self._metrics_lock:
            return {
                "db_path": str(self.db_path),
                "queue_depth": self.task_queue.qsize(),
                "max_size": self.maxsize,
                "successful_tasks": self._successful_tasks_count,
                "failed_tasks": self._failed_tasks_count,
                "is_running": self._is_running,
            }

    def stop(self, timeout: float = 10.0) -> None:
        """Signals the writer to finish queued tasks and shutdown."""
        self._stop_event.set()
        try:
            self.task_queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=timeout)
        self._is_running = False
        if self._thread.is_alive():
            LOGGER.warning("SQLite writer thread did not exit within %fs timeout.", timeout)

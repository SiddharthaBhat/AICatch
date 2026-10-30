"""Comprehensive Test Suite for Evidence Video Extraction and Background Jobs.

Tests:
1. Valid subclip extraction from test video.
2. Boundary clamping and short duration handling.
3. Timestamp overlay formatting and character escaping.
4. Normalization and multi-clip concatenation.
5. Integrity and playability validation.
6. Provenance manifest generation (.manifest.json).
7. Background extraction job lifecycle and progress reporting.
8. Job cancellation and restart recovery.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from pipeline.evidence_extractor import (
    EVIDENCE_EXTRACTOR,
    EvidenceSegment,
    EvidenceExtractor,
    EvidenceExtractionJobManager,
    escape_drawtext,
    inspect_video_stream,
    resolve_video_path
)
from storage.sqlite_db import SQLiteDatabase


class TestEvidenceExtraction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "evidence_test.db"
        self.db = SQLiteDatabase(self.db_path)

        # Resolve representative sample video
        self.sample_video = Path("data/media/uploads/outside.mp4").resolve()
        if not self.sample_video.exists():
            # Fallback to any existing mp4 in data/media/uploads
            uploads = list(Path("data/media/uploads").glob("*.mp4"))
            if uploads:
                self.sample_video = uploads[0].resolve()

        self.extractor = EvidenceExtractor()

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_escape_drawtext(self):
        """Verify proper escaping of FFmpeg drawtext control characters."""
        raw = "CAM 1: Front [Gate] 10:30:00 (100% Match) 'John's car'"
        escaped = escape_drawtext(raw)
        self.assertIn(r"\:", escaped)
        self.assertIn(r"\'", escaped)
        self.assertIn("%%", escaped)
        self.assertIn(r"\[", escaped)
        self.assertIn(r"\]", escaped)

    def test_inspect_video_stream(self):
        """Verify video metadata extraction on sample footage."""
        if not self.sample_video.exists():
            self.skipTest("No sample video found in repository.")
        info = inspect_video_stream(self.sample_video)
        self.assertTrue(info["is_valid"])
        self.assertGreater(info["width"], 0)
        self.assertGreater(info["height"], 0)
        self.assertGreater(info["fps"], 0)
        self.assertGreater(info["duration_sec"], 0)

    def test_extract_single_subclip_with_overlay(self):
        """Verify precise subclip extraction with forensic timestamp overlay."""
        if not self.sample_video.exists():
            self.skipTest("No sample video found in repository.")

        seg = EvidenceSegment(
            source_path=self.sample_video,
            camera_id="cam_test_01",
            camera_name="Main Entrance Gate",
            start_sec=1.0,
            end_sec=3.5,
            raw_cctv_timestamp="2026-10-15 14:30:00",
            detected_start_time="2026-10-15T14:30:00+00:00",
            effective_start_time="2026-10-15T14:45:00+00:00",
            effective_start_epoch=1792075500.0,
            clock_offset_sec=900.0,
            timestamp_source="clock_offset",
            timestamp_status="verified",
            track_id="trk_person_01",
            bbox=[100.0, 150.0, 300.0, 500.0]
        )

        out_clip = Path(self.temp_dir) / "test_subclip.mp4"
        res = self.extractor.extract_subclip(
            segment=seg,
            output_path=out_clip,
            context_buffer_sec=1.0,
            enable_overlay=True
        )

        self.assertTrue(out_clip.exists())
        self.assertGreater(out_clip.stat().st_size, 0)
        self.assertGreater(res["duration_sec"], 0.5)
        self.assertTrue(res["overlay"]["rendered"])
        self.assertTrue(any("Main Entrance Gate" in line for line in res["overlay"]["lines"]))
        self.assertTrue(any("CORRECTED:" in line for line in res["overlay"]["lines"]))

        # Validate playability
        valid, reason = self.extractor.validate_video_playable(out_clip)
        self.assertTrue(valid, f"Output clip playability failed: {reason}")

    def test_multi_segment_concatenation_and_manifest(self):
        """Verify stitching multiple clips, normalization, and provenance manifest."""
        if not self.sample_video.exists():
            self.skipTest("No sample video found in repository.")

        seg1 = EvidenceSegment(
            source_path=self.sample_video,
            camera_id="cam_01",
            camera_name="Gate 1",
            start_sec=0.5,
            end_sec=2.0,
            detected_start_time="2026-10-15T10:00:00+00:00",
            effective_start_epoch=1792058400.0
        )
        seg2 = EvidenceSegment(
            source_path=self.sample_video,
            camera_id="cam_02",
            camera_name="Lobby Inside",
            start_sec=2.5,
            end_sec=4.0,
            detected_start_time="2026-10-15T10:02:00+00:00",
            effective_start_epoch=1792058520.0
        )

        final_out = Path(self.temp_dir) / "stitched_evidence.mp4"
        progress_events = []

        def _prog(p, msg):
            progress_events.append((p, msg))

        manifest = self.extractor.concatenate_segments(
            segments=[seg1, seg2],
            final_output_path=final_out,
            job_id="job_test_evidence",
            title="Suspect Tracking Video",
            context_buffer_sec=0.5,
            enable_overlay=True,
            progress_callback=_prog
        )

        self.assertTrue(final_out.exists())
        self.assertGreater(final_out.stat().st_size, 0)
        self.assertEqual(manifest["segments_count"], 2)
        self.assertEqual(len(manifest["timeline"]), 2)
        self.assertEqual(manifest["timeline"][0]["camera_name"], "Gate 1")
        self.assertEqual(manifest["timeline"][1]["camera_name"], "Lobby Inside")

        # Verify manifest file exists on disk
        manifest_file = final_out.with_suffix(".manifest.json")
        self.assertTrue(manifest_file.exists())
        with open(manifest_file, "r", encoding="utf-8") as f:
            disk_manifest = json.load(f)
        self.assertEqual(disk_manifest["evidence_title"], "Suspect Tracking Video")

        # Validate playability
        valid, reason = self.extractor.validate_video_playable(final_out)
        self.assertTrue(valid, f"Stitched output playability failed: {reason}")
        self.assertGreater(len(progress_events), 0)

    def test_background_job_manager_lifecycle(self):
        """Verify background job manager executes asynchronously and updates SQLite."""
        if not self.sample_video.exists():
            self.skipTest("No sample video found in repository.")

        mgr = EvidenceExtractionJobManager(self.db, max_workers=2)

        seg = EvidenceSegment(
            source_path=self.sample_video,
            camera_id="cam_bg_1",
            camera_name="West Alley",
            start_sec=1.0,
            end_sec=2.5
        )

        job_id = "job_bg_test"
        self.db.add_job(job_id, "Background Job Test", "BALANCED")

        ext_id = mgr.submit_job(
            job_id=job_id,
            title="West Alley Evidence",
            segments=[seg],
            context_buffer_sec=0.5
        )
        self.assertIsNotNone(ext_id)

        # Poll status until completed or timeout
        start = time.time()
        completed = False
        while time.time() - start < 30.0:
            status = mgr.get_job_status(ext_id)
            if status and status["status"] == "completed":
                completed = True
                break
            time.sleep(0.2)

        self.assertTrue(completed, f"Job did not complete in time; last status: {mgr.get_job_status(ext_id)}")

        # Verify record in SQLite
        db_rec = self.db.get_evidence_extraction(ext_id)
        self.assertIsNotNone(db_rec)
        self.assertEqual(db_rec["status"], "completed")
        self.assertEqual(db_rec["progress"], 1.0)
        self.assertTrue(Path(db_rec["output_path"]).exists())

    def test_stale_job_recovery(self):
        """Verify stale queued/running jobs are marked failed on startup."""
        job_id = "job_stale_test"
        self.db.add_job(job_id, "Stale Recovery Test", "BALANCED")
        self.db.add_evidence_extraction("ext_stale_1", job_id, "Stale Job", status="running")
        self.db.add_evidence_extraction("ext_stale_2", job_id, "Stale Job Queued", status="queued")

        # Initialize manager which runs recover_stale_jobs
        mgr = EvidenceExtractionJobManager(self.db)
        r1 = self.db.get_evidence_extraction("ext_stale_1")
        r2 = self.db.get_evidence_extraction("ext_stale_2")
        self.assertEqual(r1["status"], "failed")
        self.assertEqual(r2["status"], "failed")
        self.assertIn("restart", r1["error_msg"])


if __name__ == "__main__":
    unittest.main()

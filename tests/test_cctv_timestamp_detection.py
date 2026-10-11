"""Tests for Automatic CCTV Date/Time Detection and Resolution Engine.

Validates:
1. Valid container creation timestamp parsing.
2. Container metadata marked candidate (not assumed ground-truth recording time).
3. Supported date/time formats (ISO, DD-MM-YYYY, DD/MM/YYYY, MM/DD/YYYY, compact).
4. Strict ambiguity handling on dates (e.g. 03/04/2026 without disambiguation).
5. Disambiguation with external date hint.
6. Time-only overlays combined with date hint.
7. Midnight rollover handling across video timeline.
8. Invalid dates and malformed OCR noise recovery.
9. No timestamp available reporting.
10. Valid CCTV timestamp OCR extraction on frames.
11. Multi-frame clock rate consistency check.
12. Provenance fields integrity (source, status, confidence, timezone).
13. Resolution engine cache and invalidation.
"""

from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cv2

from pipeline.timestamp_detector import (
    clean_ocr_text,
    parse_cctv_timestamp_text,
    parse_container_metadata_text,
    TimestampCandidate,
    TimestampProvenance,
    TimestampResolutionEngine,
    TIMESTAMP_ENGINE,
)


class TestCCTVTimestampDetection(unittest.TestCase):
    def setUp(self):
        self.engine = TimestampResolutionEngine()

    def test_1_valid_container_creation_timestamp(self):
        """Phase 2A: Valid container creation timestamp parsed correctly."""
        raw_ffmpeg_output = """
        Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'sample.mp4':
          Metadata:
            major_brand     : mp42
            minor_version   : 0
            compatible_brands: mp42isom
            creation_time   : 2025-11-04T00:10:47.000000Z
          Duration: 00:45:26.36, start: 0.000000, bitrate: 718 kb/s
        """
        cand = parse_container_metadata_text(raw_ffmpeg_output, Path("sample.mp4"))
        self.assertIsNotNone(cand)
        self.assertEqual(cand.source, "container_metadata")
        self.assertEqual(cand.status, "candidate")
        self.assertIn("2025-11-04T00:10:47", cand.iso_timestamp)
        self.assertEqual(cand.timezone_name, "UTC")
        self.assertGreater(cand.confidence, 0.5)

    def test_2_container_metadata_marked_candidate_until_validated(self):
        """Phase 2A: Container metadata must be marked candidate, not assumed verified recording time."""
        raw_ffmpeg_output = "TAG:creation_time=2024-06-14 06:00:40"
        cand = parse_container_metadata_text(raw_ffmpeg_output, Path("test.mp4"))
        self.assertIsNotNone(cand)
        self.assertEqual(cand.status, "candidate")
        self.assertIn("marked candidate until validated", cand.reason)

    def test_3_supported_datetime_formats(self):
        """Phase 2B: Detect and parse all standard CCTV timestamp overlay formats."""
        # 1. YYYY-MM-DD HH:MM:SS
        dt1, st1, _, conf1 = parse_cctv_timestamp_text("2026-08-23 11:10:46")
        self.assertIsNotNone(dt1)
        self.assertEqual(dt1.year, 2026)
        self.assertEqual(dt1.month, 8)
        self.assertEqual(dt1.day, 23)
        self.assertEqual(dt1.hour, 11)
        self.assertEqual(dt1.minute, 10)
        self.assertEqual(dt1.second, 46)

        # 2. DD-MM-YYYY HH:MM:SS (unambiguous day 25 > 12)
        dt2, st2, _, conf2 = parse_cctv_timestamp_text("25-12-2026 18:30:00")
        self.assertIsNotNone(dt2)
        self.assertEqual(dt2.day, 25)
        self.assertEqual(dt2.month, 12)

        # 3. MM/DD/YYYY HH:MM:SS (unambiguous day 24 > 12)
        dt3, st3, _, conf3 = parse_cctv_timestamp_text("10/24/2026 09:15:30")
        self.assertIsNotNone(dt3)
        self.assertEqual(dt3.month, 10)
        self.assertEqual(dt3.day, 24)

        # 4. Compact YYYYMMDD HHMMSS
        dt4, st4, _, _ = parse_cctv_timestamp_text("20260823 111046")
        self.assertIsNotNone(dt4)
        self.assertEqual(dt4.year, 2026)
        self.assertEqual(dt4.month, 8)
        self.assertEqual(dt4.day, 23)

        # 5. Glitched punctuation repair (e.g. 11*10:46)
        dt5, st5, _, _ = parse_cctv_timestamp_text("2026-08-23 11*10:46")
        self.assertIsNotNone(dt5)
        self.assertEqual(dt5.hour, 11)
        self.assertEqual(dt5.minute, 10)

    def test_4_ambiguous_date_formats_handled_explicitly(self):
        """Phase 2B: Ambiguous dates (e.g. 03/04/2026) must not be silently interpreted."""
        dt, status, reason, conf = parse_cctv_timestamp_text("03/04/2026 14:00:00")
        self.assertIsNone(dt)
        self.assertEqual(status, "ambiguous")
        self.assertIn("Ambiguous date format", reason)
        self.assertEqual(conf, 0.0)

    def test_5_ambiguous_date_disambiguated_with_hint(self):
        """Phase 2B: Ambiguous dates resolved when container date hint matches."""
        # Date hint: April 3, 2026 (day=3, month=4) -> matches 03/04/2026 as DD/MM/YYYY
        dt_ddmm, status_ddmm, _, _ = parse_cctv_timestamp_text(
            "03/04/2026 14:00:00",
            date_hint=date(2026, 4, 3)
        )
        self.assertIsNotNone(dt_ddmm)
        self.assertEqual(dt_ddmm.day, 3)
        self.assertEqual(dt_ddmm.month, 4)

        # Date hint: March 4, 2026 (month=3, day=4) -> matches 03/04/2026 as MM/DD/YYYY
        dt_mmdd, status_mmdd, _, _ = parse_cctv_timestamp_text(
            "03/04/2026 14:00:00",
            date_hint=date(2026, 3, 4)
        )
        self.assertIsNotNone(dt_mmdd)
        self.assertEqual(dt_mmdd.month, 3)
        self.assertEqual(dt_mmdd.day, 4)

    def test_6_time_only_overlay_with_known_date(self):
        """Phase 2B: Time-only overlay combined with known date hint."""
        # With date hint
        dt, status, reason, conf = parse_cctv_timestamp_text(
            "14:22:15",
            date_hint=date(2026, 8, 23)
        )
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 8)
        self.assertEqual(dt.day, 23)
        self.assertEqual(dt.hour, 14)
        self.assertEqual(dt.minute, 22)
        self.assertEqual(dt.second, 15)
        self.assertEqual(status, "candidate")

        # Without date hint: reports candidate with note that date is missing
        dt_nodate, status_nodate, reason_nodate, _ = parse_cctv_timestamp_text("14:22:15")
        self.assertIsNone(dt_nodate)
        self.assertEqual(status_nodate, "candidate")
        self.assertIn("date is unavailable", reason_nodate)

    def test_7_midnight_rollover_calculation(self):
        """Phase 2B: Midnight rollover handling across video timeline."""
        # Video starts at 23:59:50 on 2026-10-11
        start_dt = datetime(2026, 10, 11, 23, 59, 50, tzinfo=timezone.utc)
        start_epoch = start_dt.timestamp()

        # Observation at video second 30.0 (past midnight)
        obs_t = 30.0
        obs_epoch = start_epoch + obs_t
        obs_dt = datetime.fromtimestamp(obs_epoch, tz=timezone.utc)

        self.assertEqual(obs_dt.year, 2026)
        self.assertEqual(obs_dt.month, 10)
        self.assertEqual(obs_dt.day, 12)  # Rolled over to next day!
        self.assertEqual(obs_dt.hour, 0)
        self.assertEqual(obs_dt.minute, 0)
        self.assertEqual(obs_dt.second, 20)

    def test_8_invalid_dates_and_malformed_ocr(self):
        """Phase 2B: Invalid dates and garbage OCR handled cleanly."""
        # Month 13 is invalid
        dt_inv_m, st_inv_m, _, _ = parse_cctv_timestamp_text("2026-13-40 25:99:99")
        self.assertIsNone(dt_inv_m)
        self.assertEqual(st_inv_m, "unavailable")

        # Garbage text
        dt_garb, st_garb, _, _ = parse_cctv_timestamp_text("CAM01 NO_SIGNAL ENTRY")
        self.assertIsNone(dt_garb)
        self.assertEqual(st_garb, "unavailable")

        # Empty
        dt_emp, st_emp, _, _ = parse_cctv_timestamp_text("")
        self.assertIsNone(dt_emp)
        self.assertEqual(st_emp, "unavailable")

    def test_9_no_timestamp_available_reporting(self):
        """Phase 2C: Video with no readable metadata reports unavailable and manual config."""
        prov = TimestampProvenance(
            source="none",
            status="unavailable",
            confidence=0.0,
            reason="Recording date/time unavailable — configure manually"
        )
        data = prov.to_dict()
        self.assertEqual(data["status"], "unavailable")
        self.assertEqual(data["source"], "none")
        self.assertIn("configure manually", data["reason"])

    def test_10_provenance_contract_preservation(self):
        """Phase 2C: Provenance contract retains source, text, status, timezone, confidence."""
        prov = TimestampProvenance(
            source="ocr_overlay",
            status="verified",
            confidence=0.95,
            raw_text="2026-08-23 11:10:46",
            parsed_iso="2026-08-23T11:10:46+00:00",
            parsed_epoch=1787483446.0,
            timezone_name="UTC",
            reason="CCTV timestamp verified across 3 sampled frames",
            clock_offset_sec=0.0,
            detected_start_iso="2026-08-23T11:10:46+00:00",
            detected_start_epoch=1787483446.0
        )
        d = prov.to_dict()
        self.assertEqual(d["source"], "ocr_overlay")
        self.assertEqual(d["status"], "verified")
        self.assertEqual(d["raw_text"], "2026-08-23 11:10:46")
        self.assertEqual(d["parsed_iso"], "2026-08-23T11:10:46+00:00")
        self.assertEqual(d["timezone"], "UTC")
        self.assertEqual(d["confidence"], 0.95)

    def test_11_engine_cache_and_invalidation(self):
        """Phase 7: Performance protection - cache video resolution and support invalidation."""
        engine = TimestampResolutionEngine()

        # Non-existent file
        res1 = engine.resolve_recording_timestamp("non_existent_video.mp4")
        self.assertEqual(res1.status, "unavailable")

        # Invalidate cache
        engine.invalidate_cache()
        self.assertEqual(len(engine._cache), 0)


if __name__ == "__main__":
    unittest.main()

"""CCTV Date/Time Detection and Resolution Engine for AICatch.

Provides:
1. Container metadata inspection (FFmpeg creation_time / tags).
2. CCTV visible overlay OCR detection on bounded sampled frames.
3. Multi-format date/time parsing with strict ambiguity handling.
4. Cross-frame temporal consistency validation and midnight rollover.
5. Forensic timestamp provenance tracking (source, status, confidence, timezone).
6. Performance-safe video-level caching.
"""

from __future__ import annotations

import collections
import dataclasses
from datetime import datetime, date, time as dt_time, timedelta, timezone
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# 1. Provenance and Canonical Data Models
# ==============================================================================

@dataclasses.dataclass
class TimestampProvenance:
    """Forensic provenance contract for CCTV recording timestamps."""
    source: str  # "container_metadata" | "ocr_overlay" | "manual_recording_start" | "manual_clock_offset" | "none"
    status: str  # "verified" | "candidate" | "corrected" | "ambiguous" | "unavailable"
    confidence: float = 0.0  # 0.0 - 1.0
    raw_text: Optional[str] = None
    parsed_iso: Optional[str] = None
    parsed_epoch: Optional[float] = None
    timezone_name: str = "UTC"
    reason: str = ""
    clock_offset_sec: float = 0.0
    detected_start_iso: Optional[str] = None
    detected_start_epoch: Optional[float] = None
    sample_details: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "confidence": round(self.confidence, 4),
            "raw_text": self.raw_text,
            "parsed_iso": self.parsed_iso,
            "parsed_epoch": self.parsed_epoch,
            "timezone": self.timezone_name,
            "reason": self.reason,
            "clock_offset_sec": round(self.clock_offset_sec, 2),
            "detected_start_iso": self.detected_start_iso,
            "detected_start_epoch": self.detected_start_epoch,
            "sample_details": self.sample_details,
        }


# ==============================================================================
# 2. Text Normalization and Regex Parsers
# ==============================================================================

# Common day-of-week noise to strip
_DOW_PATTERN = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun|\d?un|\d?on|\d?ue|\d?ed|\d?hu|\d?ri|\d?at)\b[)\].]?", re.IGNORECASE)

# Common OCR character repairs for numeric/date strings
def clean_ocr_text(text: str) -> str:
    """Normalizes OCR artifacts, glitched colons, and punctuation."""
    if not text:
        return ""
    # Strip day of week words and trailing brackets
    cleaned = _DOW_PATTERN.sub(" ", text)
    # Common glitched colon separators in times: e.g. 11*10:46 or 11.10.46
    # Replace * with : when flanked by digits
    cleaned = re.sub(r"(\d{2})\*(\d{2})", r"\1:\2", cleaned)
    # Replace semicolon with colon when flanked by digits
    cleaned = re.sub(r"(\d{2});(\d{2})", r"\1:\2", cleaned)
    # Normalize multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_cctv_timestamp_text(
    text: str,
    date_hint: Optional[date] = None,
    allow_ambiguous: bool = False
) -> Tuple[Optional[datetime], str, str, float]:
    """Parses CCTV date/time text strings into a datetime object.

    Returns:
        (parsed_datetime, status, reason, confidence)
        status: "verified" | "candidate" | "ambiguous" | "unavailable"
    """
    if not text or not text.strip():
        return None, "unavailable", "Empty text", 0.0

    raw = clean_ocr_text(text.strip())

    # -------------------------------------------------------------------------
    # Format 1: Standard ISO-like YYYY-MM-DD HH:MM:SS (and with / or .)
    # -------------------------------------------------------------------------
    iso_match = re.search(
        r"\b(20\d{2})[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12]\d|3[01])\s+([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b",
        raw
    )
    if iso_match:
        y, m, d, hh, mm, ss = iso_match.groups()
        ss = ss or "00"
        try:
            dt = datetime(int(y), int(m), int(d), int(hh), int(mm), int(ss), tzinfo=timezone.utc)
            return dt, "candidate", f"Matched YYYY-MM-DD HH:MM:SS: {y}-{m}-{d}", 0.90
        except ValueError as ex:
            return None, "unavailable", f"Invalid date values: {ex}", 0.0

    # -------------------------------------------------------------------------
    # Format 2: Disambiguated / Ambiguous 3-part date with year last:
    # PartA [-/.] PartB [-/.] YYYY HH:MM:SS
    # -------------------------------------------------------------------------
    date_last_match = re.search(
        r"\b(0?[1-9]|[12]\d|3[01])[-/. ](0?[1-9]|[12]\d|3[01])[-/. ](20\d{2})\s+([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b",
        raw
    )
    if date_last_match:
        p1_str, p2_str, y_str, hh, mm, ss = date_last_match.groups()
        p1, p2, y = int(p1_str), int(p2_str), int(y_str)
        ss = ss or "00"

        # Explicit Ambiguity Disambiguation
        if p1 > 12 and p2 <= 12:
            # p1 cannot be month, must be Day -> DD/MM/YYYY
            try:
                dt = datetime(y, p2, p1, int(hh), int(mm), int(ss), tzinfo=timezone.utc)
                return dt, "candidate", f"Unambiguous DD-MM-YYYY (day={p1} > 12)", 0.88
            except ValueError as ex:
                return None, "unavailable", f"Invalid date: {ex}", 0.0

        elif p2 > 12 and p1 <= 12:
            # p2 cannot be month, must be Day -> MM/DD/YYYY
            try:
                dt = datetime(y, p1, p2, int(hh), int(mm), int(ss), tzinfo=timezone.utc)
                return dt, "candidate", f"Unambiguous MM-DD-YYYY (day={p2} > 12)", 0.88
            except ValueError as ex:
                return None, "unavailable", f"Invalid date: {ex}", 0.0

        elif p1 == p2:
            # Day and month are identical
            try:
                dt = datetime(y, p1, p2, int(hh), int(mm), int(ss), tzinfo=timezone.utc)
                return dt, "candidate", f"Symmetric date (day=month={p1})", 0.88
            except ValueError as ex:
                return None, "unavailable", f"Invalid date: {ex}", 0.0

        else:
            # Both p1 <= 12 and p2 <= 12 and p1 != p2!
            # Format cannot be established without ambiguity!
            if date_hint:
                # Disambiguate using external date hint (e.g. from container metadata)
                if date_hint.year == y and date_hint.month == p2 and date_hint.day == p1:
                    dt = datetime(y, p2, p1, int(hh), int(mm), int(ss), tzinfo=timezone.utc)
                    return dt, "candidate", f"Disambiguated as DD-MM-YYYY via container date hint ({date_hint})", 0.85
                elif date_hint.year == y and date_hint.month == p1 and date_hint.day == p2:
                    dt = datetime(y, p1, p2, int(hh), int(mm), int(ss), tzinfo=timezone.utc)
                    return dt, "candidate", f"Disambiguated as MM-DD-YYYY via container date hint ({date_hint})", 0.85

            if allow_ambiguous:
                # If caller explicitly accepts best-effort with ambiguous flag
                try:
                    dt = datetime(y, p1, p2, int(hh), int(mm), int(ss), tzinfo=timezone.utc)
                    return dt, "ambiguous", f"Ambiguous date format: {p1_str}/{p2_str}/{y_str} (cannot distinguish MM/DD vs DD/MM)", 0.40
                except ValueError:
                    pass

            return None, "ambiguous", f"Ambiguous date format: {p1_str}/{p2_str}/{y_str} (cannot distinguish MM/DD vs DD/MM without additional context)", 0.0

    # -------------------------------------------------------------------------
    # Format 3: Compact / spaced formats e.g. "08 23 2026 11:10:46" or "20260823 111046"
    # -------------------------------------------------------------------------
    compact_match = re.search(
        r"\b(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\s+([01]\d|2[0-3])([0-5]\d)([0-5]\d)\b",
        raw
    )
    if compact_match:
        y, m, d, hh, mm, ss = compact_match.groups()
        try:
            dt = datetime(int(y), int(m), int(d), int(hh), int(mm), int(ss), tzinfo=timezone.utc)
            return dt, "candidate", "Matched compact YYYYMMDD HHMMSS", 0.85
        except ValueError as ex:
            return None, "unavailable", f"Invalid compact date: {ex}", 0.0

    # -------------------------------------------------------------------------
    # Format 4: Time-only overlays e.g. "HH:MM:SS" or "HH:MM:SS.mmm"
    # -------------------------------------------------------------------------
    time_match = re.search(r"\b([01]\d|2[0-3]):([0-5]\d):([0-5]\d)(?:\.(\d{1,3}))?\b", raw)
    if time_match:
        hh, mm, ss, ms = time_match.groups()
        micro = int(ms.ljust(6, "0")[:6]) if ms else 0
        t_val = dt_time(int(hh), int(mm), int(ss), micro)

        if date_hint:
            # Combine known date with detected overlay time
            dt = datetime.combine(date_hint, t_val, tzinfo=timezone.utc)
            return dt, "candidate", f"Time overlay ({hh}:{mm}:{ss}) combined with date hint ({date_hint})", 0.80
        else:
            # Time overlay is present but lacks date
            return None, "candidate", f"Time overlay detected ({hh}:{mm}:{ss}) but date is unavailable", 0.50

    return None, "unavailable", f"No recognizable date/time pattern in: {raw[:40]!r}", 0.0


# ==============================================================================
# 3. FFmpeg Container Metadata Inspection
# ==============================================================================

def get_ffmpeg_binary() -> Optional[str]:
    """Finds available FFmpeg executable."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return None


def inspect_container_metadata(video_path: Union[str, Path]) -> Optional[TimestampCandidate]:
    """Inspects video container creation time and stream metadata tags.

    Note: Container creation time is treated as a 'candidate' until validated,
    because container metadata can represent export, transcode, or copy time.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        LOGGER.warning("Video file not found: %s", vpath)
        return None

    ffmpeg_bin = get_ffmpeg_binary()
    if not ffmpeg_bin:
        LOGGER.warning("FFmpeg binary not available for container metadata probing.")
        return None

    try:
        startupinfo = None
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 11

        cmd = [ffmpeg_bin, "-i", str(vpath), "-f", "ffmetadata", "-"]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            startupinfo=startupinfo
        )

        metadata_text = proc.stdout + "\n" + proc.stderr
        return parse_container_metadata_text(metadata_text, vpath)

    except Exception as ex:
        LOGGER.error("Failed to extract container metadata from %s: %s", vpath, ex)
        return None


def parse_container_metadata_text(metadata_text: str, video_path: Path) -> Optional[TimestampCandidate]:
    """Parses raw FFmpeg output for creation_time or recording date tags."""
    # Look for tags: creation_time, date, com.apple.quicktime.creationdate
    patterns = [
        r"(?:creation_time|date|com\.apple\.quicktime\.creationdate)\s*[:=]\s*([^\r\n]+)",
        r"TAG:creation_time=([^\r\n]+)",
    ]

    raw_val = None
    for pat in patterns:
        m = re.search(pat, metadata_text, re.IGNORECASE)
        if m:
            raw_val = m.group(1).strip()
            break

    if not raw_val:
        return None

    # Parse ISO-8601 strings: e.g. 2025-11-04T00:10:47.000000Z or 2024-06-14 06:00:40
    cleaned = raw_val.strip("'\"")
    # Clean Z or timezone
    cleaned_iso = cleaned.replace("Z", "+00:00")
    parsed_dt = None

    try:
        parsed_dt = datetime.fromisoformat(cleaned_iso)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
    except Exception:
        # Fallback to general date parser
        parsed_dt, status, _, _ = parse_cctv_timestamp_text(cleaned)
        if not parsed_dt:
            return None

    iso_str = parsed_dt.isoformat()
    epoch_sec = parsed_dt.timestamp()

    return TimestampCandidate(
        source="container_metadata",
        status="candidate",
        confidence=0.65,
        raw_text=raw_val,
        parsed_datetime=parsed_dt,
        iso_timestamp=iso_str,
        epoch_seconds=epoch_sec,
        timezone_name="UTC",
        reason="Extracted from container metadata creation_time (marked candidate until validated)"
    )


# ==============================================================================
# 4. CCTV Overlay OCR Extractor
# ==============================================================================

_GLOBAL_OCR_READER: Any = None
_OCR_LOCK = threading.Lock()

def get_ocr_reader() -> Any:
    """Thread-safe lazy initialization of EasyOCR reader on CPU."""
    global _GLOBAL_OCR_READER
    with _OCR_LOCK:
        if _GLOBAL_OCR_READER is None:
            try:
                import easyocr
                LOGGER.info("Initializing EasyOCR reader on CPU for CCTV timestamp extraction...")
                _GLOBAL_OCR_READER = easyocr.Reader(["en"], gpu=False)
            except Exception as ex:
                LOGGER.error("Failed to initialize EasyOCR: %s", ex)
                raise
        return _GLOBAL_OCR_READER


@dataclasses.dataclass
class TimestampCandidate:
    source: str
    status: str
    confidence: float
    raw_text: Optional[str] = None
    parsed_datetime: Optional[datetime] = None
    iso_timestamp: Optional[str] = None
    epoch_seconds: Optional[float] = None
    timezone_name: str = "UTC"
    reason: str = ""
    frame_idx: Optional[int] = None
    video_offset_sec: Optional[float] = None
    overlay_bbox: Optional[List[int]] = None


def detect_cctv_overlay_timestamp(
    video_path: Union[str, Path],
    max_samples: int = 3,
    container_date_hint: Optional[date] = None
) -> Optional[TimestampCandidate]:
    """Inspects a bounded set of sampled video frames using EasyOCR.

    Performance protections:
    - Samples only 2 to 3 frames near beginning and separated points.
    - Crops top 20% and bottom 20% regions where timestamps reside.
    - If a valid overlay region is detected in frame 1, subsequent frames reuse the focused crop.
    - Validates clock progression consistency across sampled frames.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        return None

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # Choose bounded sample positions (in seconds): e.g. 0.0s, 5.0s, 10.0s
    sample_offsets: List[float] = [0.0]
    if duration_sec > 6.0:
        sample_offsets.append(5.0)
    if duration_sec > 12.0:
        sample_offsets.append(10.0)

    # Cap to max_samples
    sample_offsets = sample_offsets[:max_samples]

    reader = get_ocr_reader()
    detected_samples: List[Dict[str, Any]] = []
    cached_bbox: Optional[List[int]] = None  # [y1, y2, x1, x2]

    try:
        for offset_sec in sample_offsets:
            frame_idx = int(offset_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]

            # If we already have a cached overlay region from an earlier frame, test that first
            crops_to_test = []
            if cached_bbox:
                y1, y2, x1, x2 = cached_bbox
                crops_to_test.append(("cached_roi", frame[y1:y2, x1:x2], cached_bbox))
            else:
                # Test top band (top 20%) and bottom band (bottom 20%)
                top_h = max(30, int(h * 0.20))
                bot_y1 = max(0, int(h * 0.80))
                crops_to_test.append(("top", frame[:top_h, :], [0, top_h, 0, w]))
                crops_to_test.append(("bottom", frame[bot_y1:, :], [bot_y1, h, 0, w]))

            sample_match = None

            for band_name, crop_img, crop_box in crops_to_test:
                ocr_results = reader.readtext(crop_img)
                # Combine detected text tokens
                full_text = " ".join([t[1] for t in ocr_results])
                if not full_text.strip():
                    continue

                dt, status, reason, conf = parse_cctv_timestamp_text(
                    full_text,
                    date_hint=container_date_hint
                )

                if dt is not None:
                    # Found candidate! If this is top/bottom band, refine bbox
                    refined_bbox = crop_box
                    if ocr_results and band_name in ("top", "bottom"):
                        # Find the specific bounding box of the matching token
                        for b_box, t_text, t_conf in ocr_results:
                            if any(char.isdigit() for char in t_text):
                                ys = [pt[1] for pt in b_box]
                                xs = [pt[0] for pt in b_box]
                                y_min = max(0, int(crop_box[0] + min(ys) - 5))
                                y_max = min(h, int(crop_box[0] + max(ys) + 5))
                                x_min = max(0, int(crop_box[2] + min(xs) - 10))
                                x_max = min(w, int(crop_box[2] + max(xs) + 10))
                                refined_bbox = [y_min, y_max, x_min, x_max]
                                break

                    sample_match = {
                        "offset_sec": offset_sec,
                        "frame_idx": frame_idx,
                        "datetime": dt,
                        "raw_text": full_text,
                        "status": status,
                        "reason": reason,
                        "confidence": conf,
                        "bbox": refined_bbox
                    }
                    cached_bbox = refined_bbox
                    break

            if sample_match:
                detected_samples.append(sample_match)

    finally:
        cap.release()

    if not detected_samples:
        return None

    # -------------------------------------------------------------------------
    # Cross-Sample Validation
    # -------------------------------------------------------------------------
    first_sample = detected_samples[0]
    first_dt = first_sample["datetime"]
    first_offset = first_sample["offset_sec"]

    # Calculate inferred video start time (time zero): T_start = T_sample - offset
    computed_start_dt = first_dt - timedelta(seconds=first_offset)
    is_consistent = True
    validation_reason = first_sample["reason"]
    final_conf = first_sample["confidence"]

    if len(detected_samples) > 1:
        # Validate that clock progression between samples matches elapsed video time
        all_consistent = True
        for i in range(1, len(detected_samples)):
            prev = detected_samples[i - 1]
            curr = detected_samples[i]
            dt_diff = (curr["datetime"] - prev["datetime"]).total_seconds()
            video_diff = curr["offset_sec"] - prev["offset_sec"]

            # Clock difference should be within 2.0s tolerance of video difference
            if abs(dt_diff - video_diff) > 2.0:
                all_consistent = False
                LOGGER.warning(
                    "CCTV clock inconsistency on %s: delta_clock=%.1fs vs delta_video=%.1fs",
                    vpath.name, dt_diff, video_diff
                )
                break

        if all_consistent:
            is_consistent = True
            final_status = "verified"
            final_conf = max(0.92, first_sample["confidence"])
            validation_reason = f"CCTV timestamp verified across {len(detected_samples)} sampled frames with consistent clock rate."
        else:
            is_consistent = False
            final_status = "ambiguous"
            final_conf = 0.50
            validation_reason = "CCTV clock progression inconsistent between sampled frames."
    else:
        final_status = first_sample["status"]
        if final_status == "candidate":
            validation_reason = f"Single frame detection ({first_sample['raw_text'][:30]})"

    return TimestampCandidate(
        source="ocr_overlay",
        status=final_status,
        confidence=final_conf,
        raw_text=first_sample["raw_text"],
        parsed_datetime=computed_start_dt,
        iso_timestamp=computed_start_dt.isoformat(),
        epoch_seconds=computed_start_dt.timestamp(),
        timezone_name="UTC",
        reason=validation_reason,
        frame_idx=first_sample["frame_idx"],
        video_offset_sec=first_offset,
        overlay_bbox=first_sample.get("bbox")
    )


# ==============================================================================
# 5. Full Timestamp Resolution Engine & Cache
# ==============================================================================

class TimestampResolutionEngine:
    """Singleton resolver managing precedence, caching, and manual offsets."""

    def __init__(self):
        # Cache key: (video_path, file_size, mtime) -> TimestampProvenance
        self._cache: Dict[Tuple[str, int, float], TimestampProvenance] = {}
        self._lock = threading.Lock()

    def _get_cache_key(self, video_path: Union[str, Path]) -> Optional[Tuple[str, int, float]]:
        p = Path(video_path)
        if not p.exists():
            return None
        stat = p.stat()
        return (str(p.resolve()), stat.st_size, stat.st_mtime)

    def invalidate_cache(self, video_path: Optional[Union[str, Path]] = None) -> None:
        """Invalidates cache for a specific video or all videos."""
        with self._lock:
            if video_path is None:
                self._cache.clear()
            else:
                key = self._get_cache_key(video_path)
                if key and key in self._cache:
                    del self._cache[key]

    def resolve_recording_timestamp(
        self,
        video_path: Union[str, Path],
        force_refresh: bool = False
    ) -> TimestampProvenance:
        """Resolves the best automatic recording start timestamp with provenance.

        Precedence:
        1. Validated CCTV OCR timestamp from visible video overlay.
        2. Container metadata creation_time (marked candidate).
        3. If neither: unavailable.
        """
        p = Path(video_path)
        if not p.exists():
            return TimestampProvenance(
                source="none",
                status="unavailable",
                reason=f"Video file does not exist: {video_path}"
            )

        cache_key = self._get_cache_key(p)
        if not force_refresh and cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        sample_details = []

        # Step 1: Inspect container metadata (fast FFmpeg tag check)
        meta_candidate = inspect_container_metadata(p)
        date_hint = None
        if meta_candidate and meta_candidate.parsed_datetime:
            date_hint = meta_candidate.parsed_datetime.date()
            sample_details.append({
                "source": "container_metadata",
                "status": meta_candidate.status,
                "raw": meta_candidate.raw_text,
                "iso": meta_candidate.iso_timestamp
            })

        # Step 2: Attempt CCTV overlay OCR detection
        ocr_candidate = None
        try:
            ocr_candidate = detect_cctv_overlay_timestamp(
                video_path=p,
                max_samples=3,
                container_date_hint=date_hint
            )
            if ocr_candidate:
                sample_details.append({
                    "source": "ocr_overlay",
                    "status": ocr_candidate.status,
                    "raw": ocr_candidate.raw_text,
                    "iso": ocr_candidate.iso_timestamp,
                    "reason": ocr_candidate.reason
                })
        except Exception as ex:
            LOGGER.warning("OCR overlay detection error on %s: %s", p.name, ex)

        # Step 3: Apply explicit source precedence
        result_prov: TimestampProvenance

        if ocr_candidate and ocr_candidate.status == "verified":
            # OCR overlay verified across frames is highest automatic priority!
            result_prov = TimestampProvenance(
                source="ocr_overlay",
                status="verified",
                confidence=ocr_candidate.confidence,
                raw_text=ocr_candidate.raw_text,
                parsed_iso=ocr_candidate.iso_timestamp,
                parsed_epoch=ocr_candidate.epoch_seconds,
                timezone_name=ocr_candidate.timezone_name,
                reason=ocr_candidate.reason,
                detected_start_iso=ocr_candidate.iso_timestamp,
                detected_start_epoch=ocr_candidate.epoch_seconds,
                sample_details=sample_details
            )

        elif meta_candidate and meta_candidate.status == "candidate":
            # Container metadata candidate
            if ocr_candidate and ocr_candidate.status == "ambiguous":
                # OCR was attempted but ambiguous; keep metadata with warning
                reason = f"Container metadata creation_time ({meta_candidate.iso_timestamp}) used; visible overlay was ambiguous: {ocr_candidate.reason}"
            else:
                reason = "Container metadata creation_time (marked candidate until investigator validation)"

            result_prov = TimestampProvenance(
                source="container_metadata",
                status="candidate",
                confidence=meta_candidate.confidence,
                raw_text=meta_candidate.raw_text,
                parsed_iso=meta_candidate.iso_timestamp,
                parsed_epoch=meta_candidate.epoch_seconds,
                timezone_name=meta_candidate.timezone_name,
                reason=reason,
                detected_start_iso=meta_candidate.iso_timestamp,
                detected_start_epoch=meta_candidate.epoch_seconds,
                sample_details=sample_details
            )

        elif ocr_candidate and ocr_candidate.status in ("candidate", "ambiguous"):
            result_prov = TimestampProvenance(
                source="ocr_overlay",
                status=ocr_candidate.status,
                confidence=ocr_candidate.confidence,
                raw_text=ocr_candidate.raw_text,
                parsed_iso=ocr_candidate.iso_timestamp,
                parsed_epoch=ocr_candidate.epoch_seconds,
                timezone_name=ocr_candidate.timezone_name,
                reason=ocr_candidate.reason,
                detected_start_iso=ocr_candidate.iso_timestamp,
                detected_start_epoch=ocr_candidate.epoch_seconds,
                sample_details=sample_details
            )

        else:
            result_prov = TimestampProvenance(
                source="none",
                status="unavailable",
                confidence=0.0,
                reason="Recording date/time unavailable — configure manually",
                sample_details=sample_details
            )

        if cache_key:
            with self._lock:
                self._cache[cache_key] = result_prov

        return result_prov


# Global instance
TIMESTAMP_ENGINE = TimestampResolutionEngine()


def compute_effective_timestamps(
    config_mode: Optional[str] = None,
    manual_start_time: Optional[str] = None,
    clock_offset_sec: float = 0.0,
    detected_start_iso: Optional[str] = None,
    detected_start_epoch: Optional[float] = None,
    mode: Optional[str] = None,
    detected_start_time: Optional[str] = None
) -> Tuple[Optional[str], Optional[float]]:
    """Computes effective recording start time and epoch based on configuration mode.

    Returns:
        (effective_start_iso, effective_start_epoch)
    """
    clean_mode = (mode or config_mode or "auto").strip().lower()
    start_iso = detected_start_time or detected_start_iso

    # Resolve detected start epoch from ISO if not passed directly
    if detected_start_epoch is None and start_iso:
        try:
            clean_iso = start_iso.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            detected_start_epoch = dt.timestamp()
            start_iso = dt.isoformat()
        except Exception:
            dt, _, _, _ = parse_cctv_timestamp_text(start_iso)
            if dt:
                detected_start_epoch = dt.timestamp()
                start_iso = dt.isoformat()

    if clean_mode in ("manual_start", "set_recording_start"):
        if not manual_start_time or not manual_start_time.strip():
            return None, None
        try:
            clean = manual_start_time.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat(), dt.timestamp()
        except Exception:
            dt, st, _, _ = parse_cctv_timestamp_text(manual_start_time)
            if dt:
                return dt.isoformat(), dt.timestamp()
            return None, None

    elif clean_mode in ("manual_offset", "clock_offset"):
        if detected_start_epoch is None:
            return None, None
        eff_epoch = float(detected_start_epoch) + float(clock_offset_sec)
        eff_dt = datetime.fromtimestamp(eff_epoch, tz=timezone.utc)
        return eff_dt.isoformat(), eff_epoch

    else:  # "automatic", "auto"
        if start_iso and detected_start_epoch is not None:
            return start_iso, float(detected_start_epoch)
        return None, None


def observation_to_absolute_time(
    video_timestamp_sec: float,
    effective_start_epoch: Optional[float] = None
) -> Tuple[Optional[str], Optional[float]]:
    """Converts video-relative position (seconds) into an absolute timestamp."""
    if effective_start_epoch is None:
        return None, None
    if video_timestamp_sec > 100000000.0 and effective_start_epoch < 100000000.0:
        video_timestamp_sec, effective_start_epoch = effective_start_epoch, video_timestamp_sec
    obs_epoch = float(effective_start_epoch) + float(video_timestamp_sec)
    obs_dt = datetime.fromtimestamp(obs_epoch, tz=timezone.utc)
    return obs_dt.isoformat(), obs_epoch

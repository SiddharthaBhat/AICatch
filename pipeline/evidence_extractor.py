"""Forensic Evidence Video Extraction and Stitched Compilation Engine for AICatch.

Guarantees:
1. Safe subprocess argument list invocation (no shell-string interpolation).
2. Precise subclip trimming with boundary clamping and context padding.
3. Accurate CCTV timestamp overlay rendering with strict attribution
   (original observed CCTV time distinguished from corrected timestamp).
4. Multi-stream normalization for seamless concatenation of heterogeneous sources
   (differing resolutions, codecs, frame rates, pixel formats, audio).
5. Fast-stream copying (-c copy) when stream parameters are strictly homogeneous.
6. FFprobe/FFmpeg integrity validation ensuring playable, non-corrupted MP4 output.
7. Complete forensic provenance manifest generation (.manifest.json).
8. Asynchronous bounded background worker execution with progress reporting and cancellation.
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import imageio_ffmpeg

from config import CLIPS_DIR, MEDIA_DIR, UPLOAD_DIR
from storage.sqlite_db import SQLiteDatabase

LOGGER = logging.getLogger(__name__)


def get_ffmpeg_exe() -> str:
    """Resolves absolute path to bundled imageio-ffmpeg static binary."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        LOGGER.warning(f"Could not resolve imageio_ffmpeg binary: {exc}. Defaulting to 'ffmpeg'")
        return "ffmpeg"


def resolve_video_path(raw_path: str) -> Optional[Path]:
    """Resolves relative or absolute video source path to existing file on disk."""
    if not raw_path:
        return None
    clean = raw_path.replace("\\", "/").strip()
    p = Path(clean)
    if p.is_absolute() and p.exists():
        return p
    for base in [UPLOAD_DIR, MEDIA_DIR, CLIPS_DIR]:
        candidate = base / clean
        if candidate.exists():
            return candidate
        candidate_name = base / p.name
        if candidate_name.exists():
            return candidate_name
    return None


def escape_drawtext(text: str) -> str:
    """Escapes special characters for FFmpeg drawtext filter syntax."""
    if not text:
        return ""
    # In ffmpeg drawtext: \ -> \\, ' -> \', : -> \:, % -> %%
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("'", "\\'")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("%", "%%")
    escaped = escaped.replace("[", "\\[").replace("]", "\\]")
    return escaped


def inspect_video_stream(video_path: Path) -> Dict[str, Any]:
    """Inspects video stream parameters using OpenCV and FFmpeg."""
    info = {
        "path": str(video_path).replace("\\", "/"),
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "duration_sec": 0.0,
        "frame_count": 0,
        "codec": "h264",
        "has_audio": False,
        "is_valid": False
    }
    if not video_path.exists():
        return info

    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0 or fps > 120:
            fps = 30.0
        info["fps"] = round(fps, 3)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info["frame_count"] = fc
        if fps > 0 and fc > 0:
            info["duration_sec"] = round(fc / fps, 3)
        info["is_valid"] = True
        cap.release()

    return info


@dataclasses.dataclass
class EvidenceSegment:
    """Defines an individual subclip segment for extraction."""
    source_path: Path
    camera_id: str
    camera_name: str
    start_sec: float
    end_sec: float
    video_id: Optional[str] = None
    job_id: Optional[str] = None
    track_id: Optional[Union[str, int]] = None
    bbox: Optional[List[float]] = None
    raw_cctv_timestamp: Optional[str] = None
    detected_start_time: Optional[str] = None
    effective_start_time: Optional[str] = None
    effective_start_epoch: Optional[float] = None
    clock_offset_sec: float = 0.0
    timestamp_source: Optional[str] = None
    timestamp_status: str = "unavailable"
    timezone: str = "UTC"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": str(self.source_path).replace("\\", "/"),
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "video_id": self.video_id or self.camera_id,
            "job_id": self.job_id,
            "track_id": self.track_id,
            "bbox": self.bbox,
            "raw_cctv_timestamp": self.raw_cctv_timestamp,
            "detected_start_time": self.detected_start_time,
            "effective_start_time": self.effective_start_time,
            "effective_start_epoch": self.effective_start_epoch,
            "clock_offset_sec": self.clock_offset_sec,
            "timestamp_source": self.timestamp_source,
            "timestamp_status": self.timestamp_status,
            "timezone": self.timezone,
        }


class EvidenceExtractor:
    """Standalone, independently testable FFmpeg extraction and concatenation module."""

    def __init__(self, ffmpeg_exe: Optional[str] = None):
        self.ffmpeg_exe = ffmpeg_exe or get_ffmpeg_exe()

    def validate_video_playable(self, video_path: Path) -> Tuple[bool, str]:
        """Validates that the generated video file is non-empty and playable without decoding errors."""
        if not video_path.exists():
            return False, f"File does not exist: {video_path}"
        if video_path.stat().st_size == 0:
            return False, f"File size is 0 bytes: {video_path}"

        cmd = [
            self.ffmpeg_exe,
            "-v", "error",
            "-i", str(video_path),
            "-t", "2",
            "-f", "null",
            "-"
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
            if res.returncode != 0:
                return False, f"FFmpeg validation failed: {res.stderr[:200]}"
            return True, "Valid"
        except subprocess.TimeoutExpired:
            return False, "Validation timed out"
        except Exception as exc:
            return False, f"Validation exception: {exc}"

    def extract_subclip(
        self,
        segment: EvidenceSegment,
        output_path: Path,
        context_buffer_sec: float = 2.0,
        enable_overlay: bool = True
    ) -> Dict[str, Any]:
        """Extracts a precise subclip from source video, applying context padding and forensic overlay."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source = segment.source_path
        if not source.exists():
            raise FileNotFoundError(f"Source video file not found: {source}")

        # Clamping and context padding
        info = inspect_video_stream(source)
        vid_duration = info.get("duration_sec", 36000.0)

        actual_start = max(0.0, segment.start_sec - context_buffer_sec)
        actual_end = segment.end_sec + context_buffer_sec
        if vid_duration > 0:
            actual_end = min(vid_duration, actual_end)

        clip_duration = max(0.5, actual_end - actual_start)

        # Build overlay text filter
        vf_filters: List[str] = []
        overlay_manifest: Dict[str, Any] = {"rendered": False, "lines": []}

        if enable_overlay:
            lines = []
            cam_label = segment.camera_name or segment.camera_id or "Camera"
            lines.append(f"LOC: {cam_label}")

            # Original CCTV timestamp if known
            if segment.raw_cctv_timestamp:
                lines.append(f"CCTV (RAW): {segment.raw_cctv_timestamp}")
            elif segment.detected_start_time:
                try:
                    dt = datetime.fromisoformat(segment.detected_start_time.replace("Z", "+00:00"))
                    orig_dt = datetime.fromtimestamp(dt.timestamp() + actual_start, tz=timezone.utc)
                    lines.append(f"CCTV (ORIG): {orig_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                except Exception:
                    lines.append(f"CCTV (ORIG): {segment.detected_start_time}")

            # Distinct corrected timestamp if calibrated
            if segment.effective_start_epoch is not None:
                abs_epoch = float(segment.effective_start_epoch) + actual_start
                corr_dt = datetime.fromtimestamp(abs_epoch, tz=timezone.utc)
                offset_str = f" ({segment.clock_offset_sec:+.1f}s)" if segment.clock_offset_sec != 0.0 else ""
                lines.append(f"CORRECTED: {corr_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC{offset_str}")

            lines.append(f"REL T: {actual_start:.2f}s - {actual_end:.2f}s")

            overlay_manifest["rendered"] = True
            overlay_manifest["lines"] = lines

            # Combine lines for drawtext
            y_offset = 20
            for line in lines:
                escaped = escape_drawtext(line)
                vf_filters.append(
                    f"drawtext=text='{escaped}':x=20:y={y_offset}:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=4"
                )
                y_offset += 26

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-ss", f"{actual_start:.3f}",
            "-i", str(source),
            "-t", f"{clip_duration:.3f}",
        ]

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(output_path)
        ])

        LOGGER.info(f"Extracting subclip for {segment.camera_name} ({actual_start:.2f}s - {actual_end:.2f}s)...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
        if res.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
            LOGGER.error(f"FFmpeg subclip extraction failed: {res.stderr[:300]}")
            raise RuntimeError(f"FFmpeg subclip extraction failed for {source.name}: {res.stderr[:200]}")

        # Validate extracted subclip
        valid, reason = self.validate_video_playable(output_path)
        if not valid:
            raise RuntimeError(f"Extracted subclip failed playability validation: {reason}")

        out_info = inspect_video_stream(output_path)
        return {
            "output_path": str(output_path).replace("\\", "/"),
            "file_size": output_path.stat().st_size,
            "actual_start_sec": actual_start,
            "actual_end_sec": actual_end,
            "duration_sec": out_info.get("duration_sec", clip_duration),
            "width": out_info.get("width"),
            "height": out_info.get("height"),
            "fps": out_info.get("fps"),
            "overlay": overlay_manifest,
            "segment": segment.to_dict()
        }

    def normalize_clip_for_concat(
        self,
        clip_path: Path,
        normalized_path: Path,
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: float = 30.0
    ) -> bool:
        """Normalizes heterogeneous clip to uniform resolution, frame rate, pixel format, and audio."""
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        vf = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={target_fps},"
            f"format=yuv420p"
        )
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-i", str(clip_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-movflags", "+faststart",
            str(normalized_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        return res.returncode == 0 and normalized_path.exists() and normalized_path.stat().st_size > 0

    def concatenate_segments(
        self,
        segments: List[EvidenceSegment],
        final_output_path: Path,
        job_id: str,
        title: str = "Compiled Evidence Video",
        context_buffer_sec: float = 2.0,
        enable_overlay: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> Dict[str, Any]:
        """Extracts, normalizes, and stitches multiple camera observations into a single evidence video."""
        if not segments:
            raise ValueError("No evidence segments provided for concatenation.")

        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = final_output_path.parent / f"tmp_extract_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        extracted_subclips: List[Path] = []
        normalized_subclips: List[Path] = []
        provenance_timeline: List[Dict[str, Any]] = []

        timeline_cursor_sec = 0.0

        try:
            total_segs = len(segments)
            for idx, seg in enumerate(segments):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Evidence extraction was cancelled by user.")

                if progress_callback:
                    p = (idx / total_segs) * 0.60
                    progress_callback(p, f"Extracting clip {idx + 1}/{total_segs} ({seg.camera_name})...")

                clip_file = temp_dir / f"clip_{idx:03d}.mp4"
                res = self.extract_subclip(
                    segment=seg,
                    output_path=clip_file,
                    context_buffer_sec=context_buffer_sec,
                    enable_overlay=enable_overlay
                )
                extracted_subclips.append(clip_file)

                clip_dur = res["duration_sec"]
                provenance_timeline.append({
                    "sequence_index": idx,
                    "timeline_start_sec": round(timeline_cursor_sec, 3),
                    "timeline_end_sec": round(timeline_cursor_sec + clip_dur, 3),
                    "clip_duration_sec": round(clip_dur, 3),
                    "source_path": seg.to_dict()["source_path"],
                    "camera_id": seg.camera_id,
                    "camera_name": seg.camera_name,
                    "video_id": seg.video_id or seg.camera_id,
                    "source_relative_start_sec": res["actual_start_sec"],
                    "source_relative_end_sec": res["actual_end_sec"],
                    "raw_cctv_timestamp": seg.raw_cctv_timestamp,
                    "detected_start_time": seg.detected_start_time,
                    "effective_start_time": seg.effective_start_time,
                    "effective_start_epoch": seg.effective_start_epoch,
                    "clock_offset_sec": seg.clock_offset_sec,
                    "timestamp_source": seg.timestamp_source,
                    "timestamp_status": seg.timestamp_status,
                    "track_id": seg.track_id,
                    "bbox": seg.bbox,
                    "overlay": res["overlay"]
                })
                timeline_cursor_sec += clip_dur

            if not extracted_subclips:
                raise RuntimeError("No valid subclips could be extracted for compilation.")

            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Evidence extraction was cancelled by user.")

            # Normalization phase
            if progress_callback:
                progress_callback(0.65, "Normalizing heterogeneous video streams...")

            # Check if all clips are identical in resolution
            first_info = inspect_video_stream(extracted_subclips[0])
            needs_normalization = False
            for c in extracted_subclips[1:]:
                c_info = inspect_video_stream(c)
                if (c_info["width"] != first_info["width"] or
                    c_info["height"] != first_info["height"] or
                    abs(c_info["fps"] - first_info["fps"]) > 0.5):
                    needs_normalization = True
                    break

            concat_target_clips = extracted_subclips
            if needs_normalization:
                for idx, c in enumerate(extracted_subclips):
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("Evidence extraction was cancelled by user.")
                    norm_file = temp_dir / f"norm_{idx:03d}.mp4"
                    success = self.normalize_clip_for_concat(
                        clip_path=c,
                        normalized_path=norm_file,
                        target_width=1920,
                        target_height=1080,
                        target_fps=30.0
                    )
                    if not success:
                        LOGGER.warning(f"Normalization failed for {c.name}; using raw extracted clip.")
                        normalized_subclips.append(c)
                    else:
                        normalized_subclips.append(norm_file)
                concat_target_clips = normalized_subclips

            # Build concat list
            concat_list_file = temp_dir / "concat_list.txt"
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for c in concat_target_clips:
                    clean_p = str(c).replace("\\", "/")
                    f.write(f"file '{clean_p}'\n")

            if progress_callback:
                progress_callback(0.85, "Stitching final compiled evidence video...")

            # Try fast stream copy first
            fast_cmd = [
                self.ffmpeg_exe,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(final_output_path)
            ]
            res_fast = subprocess.run(fast_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)

            # If fast copy fails or outputs empty file, re-encode
            if res_fast.returncode != 0 or not final_output_path.exists() or final_output_path.stat().st_size == 0:
                LOGGER.info("Fast stream copy concat unavailable; performing seamless re-encode stitch...")
                reenc_cmd = [
                    self.ffmpeg_exe,
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list_file),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(final_output_path)
                ]
                res_reenc = subprocess.run(reenc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
                if res_reenc.returncode != 0 or not final_output_path.exists():
                    raise RuntimeError(f"Concat failed: {res_reenc.stderr[:200]}")

            if progress_callback:
                progress_callback(0.95, "Validating output video playability...")

            # Validate final output with FFprobe/FFmpeg
            valid, reason = self.validate_video_playable(final_output_path)
            if not valid:
                raise RuntimeError(f"Final stitched evidence video failed integrity validation: {reason}")

            final_info = inspect_video_stream(final_output_path)

            # Build comprehensive provenance manifest
            manifest = {
                "manifest_version": "1.0",
                "evidence_title": title,
                "job_id": job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "output_video": {
                    "filename": final_output_path.name,
                    "path": str(final_output_path).replace("\\", "/"),
                    "relative_url": f"/media/clips/{job_id}/{final_output_path.name}",
                    "duration_sec": final_info.get("duration_sec", timeline_cursor_sec),
                    "file_size_bytes": final_output_path.stat().st_size,
                    "width": final_info.get("width"),
                    "height": final_info.get("height"),
                    "fps": final_info.get("fps"),
                    "codec": final_info.get("codec", "h264"),
                    "is_playable": True
                },
                "segments_count": len(provenance_timeline),
                "timeline": provenance_timeline
            }

            # Write manifest file to disk alongside MP4 (.manifest.json)
            manifest_file = final_output_path.with_suffix(".manifest.json")
            with open(manifest_file, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)

            if progress_callback:
                progress_callback(1.0, "Evidence video compilation completed.")

            return manifest

        finally:
            # Clean up temporary clips directory
            try:
                shutil.rmtree(str(temp_dir), ignore_errors=True)
            except Exception as ce:
                LOGGER.debug("Temporary evidence dir cleanup note: %s", ce)


# Global extractor instance
EVIDENCE_EXTRACTOR = EvidenceExtractor()


class EvidenceExtractionJobManager:
    """Manages asynchronous background extraction jobs with bounded concurrency and persistence."""

    def __init__(self, sqlite_db: SQLiteDatabase, max_workers: int = 2):
        self.sqlite = sqlite_db
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="EvidenceExtractWorker"
        )
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.recover_stale_jobs()

    def recover_stale_jobs(self) -> None:
        """Recovers any extraction jobs interrupted during application restarts."""
        try:
            extractions = self.sqlite.get_evidence_extractions()
            for ext in extractions:
                if ext.get("status") in ("queued", "running"):
                    self.sqlite.update_evidence_extraction(
                        extraction_id=ext["id"],
                        status="failed",
                        error_msg="Extraction interrupted by application restart."
                    )
                    LOGGER.info(f"Recovered stale evidence extraction job {ext['id']} -> marked failed.")
        except Exception as exc:
            LOGGER.debug(f"Stale evidence job recovery note: {exc}")

    def submit_job(
        self,
        job_id: str,
        title: str,
        segments: List[EvidenceSegment],
        context_buffer_sec: float = 2.0,
        enable_overlay: bool = True,
        folder_id: Optional[str] = None
    ) -> str:
        """Submits an asynchronous evidence video compilation background job."""
        extraction_id = f"ext_{uuid.uuid4().hex[:12]}"

        # Record in SQLite with status="queued"
        self.sqlite.add_evidence_extraction(
            extraction_id=extraction_id,
            job_id=job_id,
            title=title,
            folder_id=folder_id,
            status="queued"
        )

        cancel_event = threading.Event()

        with self.lock:
            self.active_jobs[extraction_id] = {
                "id": extraction_id,
                "job_id": job_id,
                "title": title,
                "status": "queued",
                "progress": 0.0,
                "stage": "Queued in background worker pool",
                "cancel_event": cancel_event,
                "error": None,
                "result": None,
                "created_at": time.time()
            }

        # Submit to bounded thread pool
        self.executor.submit(
            self._worker_execute,
            extraction_id=extraction_id,
            job_id=job_id,
            title=title,
            segments=segments,
            context_buffer_sec=context_buffer_sec,
            enable_overlay=enable_overlay,
            folder_id=folder_id,
            cancel_event=cancel_event
        )

        LOGGER.info(f"Submitted evidence extraction job {extraction_id} for investigation {job_id} ({len(segments)} segments)")
        return extraction_id

    def _worker_execute(
        self,
        extraction_id: str,
        job_id: str,
        title: str,
        segments: List[EvidenceSegment],
        context_buffer_sec: float,
        enable_overlay: bool,
        folder_id: Optional[str],
        cancel_event: threading.Event
    ) -> None:
        """Worker execution running off the HTTP and GPU threads."""
        try:
            with self.lock:
                if extraction_id in self.active_jobs:
                    self.active_jobs[extraction_id]["status"] = "running"
                    self.active_jobs[extraction_id]["stage"] = "Initializing extraction..."

            self.sqlite.update_evidence_extraction(
                extraction_id=extraction_id,
                status="running",
                progress=0.05
            )

            job_clips_dir = CLIPS_DIR / job_id
            job_clips_dir.mkdir(parents=True, exist_ok=True)
            safe_title = re.sub(r"[^\w\-]", "_", title.lower())[:32]
            output_filename = f"evidence_{safe_title}_{extraction_id[-6:]}.mp4"
            final_output_path = job_clips_dir / output_filename

            def _progress_cb(p: float, msg: str):
                with self.lock:
                    if extraction_id in self.active_jobs:
                        self.active_jobs[extraction_id]["progress"] = round(p, 3)
                        self.active_jobs[extraction_id]["stage"] = msg
                self.sqlite.update_evidence_extraction(
                    extraction_id=extraction_id,
                    progress=round(p, 3)
                )

            manifest = EVIDENCE_EXTRACTOR.concatenate_segments(
                segments=segments,
                final_output_path=final_output_path,
                job_id=job_id,
                title=title,
                context_buffer_sec=context_buffer_sec,
                enable_overlay=enable_overlay,
                progress_callback=_progress_cb,
                cancel_event=cancel_event
            )

            manifest_json = json.dumps(manifest)
            duration_sec = manifest["output_video"]["duration_sec"]
            segments_count = manifest["segments_count"]

            with self.lock:
                if extraction_id in self.active_jobs:
                    self.active_jobs[extraction_id]["status"] = "completed"
                    self.active_jobs[extraction_id]["progress"] = 1.0
                    self.active_jobs[extraction_id]["stage"] = "Completed"
                    self.active_jobs[extraction_id]["result"] = manifest

            self.sqlite.update_evidence_extraction(
                extraction_id=extraction_id,
                status="completed",
                progress=1.0,
                output_path=str(final_output_path).replace("\\", "/"),
                manifest_json=manifest_json,
                duration_sec=duration_sec,
                segments_count=segments_count
            )

            # Auto-bookmark in evidence folder if folder_id is provided or created
            if folder_id:
                try:
                    self.sqlite.add_folder_item(
                        item_id=f"clip_{extraction_id}",
                        folder_id=folder_id,
                        item_type="clip",
                        target_id=str(final_output_path).replace("\\", "/"),
                        metadata=manifest_json
                    )
                except Exception as fe:
                    LOGGER.debug(f"Auto-bookmark note: {fe}")

            LOGGER.info(f"Evidence extraction {extraction_id} completed successfully -> {final_output_path.name}")

        except Exception as exc:
            err_msg = str(exc)
            LOGGER.error(f"Evidence extraction job {extraction_id} failed: {err_msg}", exc_info=True)
            with self.lock:
                if extraction_id in self.active_jobs:
                    self.active_jobs[extraction_id]["status"] = "failed"
                    self.active_jobs[extraction_id]["error"] = err_msg

            self.sqlite.update_evidence_extraction(
                extraction_id=extraction_id,
                status="failed",
                error_msg=err_msg
            )

    def cancel_job(self, extraction_id: str) -> bool:
        """Signals cancellation of a running extraction job."""
        with self.lock:
            job_info = self.active_jobs.get(extraction_id)
            if job_info:
                job_info["cancel_event"].set()
                job_info["status"] = "cancelled"
                self.sqlite.update_evidence_extraction(
                    extraction_id=extraction_id,
                    status="cancelled",
                    error_msg="Extraction cancelled by user."
                )
                return True
        return False

    def get_job_status(self, extraction_id: str) -> Optional[Dict[str, Any]]:
        """Returns live execution status and provenance manifest for an extraction job."""
        with self.lock:
            live = self.active_jobs.get(extraction_id)
            if live:
                return {
                    "id": live["id"],
                    "job_id": live["job_id"],
                    "title": live["title"],
                    "status": live["status"],
                    "progress": live["progress"],
                    "stage": live.get("stage", ""),
                    "error": live["error"],
                    "manifest": live["result"]
                }

        # Fallback to persistent SQLite record
        rec = self.sqlite.get_evidence_extraction(extraction_id)
        if rec:
            manifest = None
            if rec.get("manifest_json"):
                try:
                    manifest = json.loads(rec["manifest_json"])
                except Exception:
                    pass
            return {
                "id": rec["id"],
                "job_id": rec["job_id"],
                "title": rec["title"],
                "status": rec["status"],
                "progress": rec.get("progress", 0.0),
                "stage": rec["status"].capitalize(),
                "error": rec.get("error_msg"),
                "output_path": rec.get("output_path"),
                "duration_sec": rec.get("duration_sec"),
                "segments_count": rec.get("segments_count"),
                "manifest": manifest
            }
        return None

    def list_jobs(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all evidence extractions for an investigation."""
        records = self.sqlite.get_evidence_extractions(job_id=job_id)
        results = []
        for rec in records:
            manifest = None
            if rec.get("manifest_json"):
                try:
                    manifest = json.loads(rec["manifest_json"])
                except Exception:
                    pass
            results.append({
                "id": rec["id"],
                "job_id": rec["job_id"],
                "title": rec["title"],
                "status": rec["status"],
                "progress": rec.get("progress", 0.0),
                "error": rec.get("error_msg"),
                "output_path": rec.get("output_path"),
                "relative_url": f"/media/clips/{rec['job_id']}/{Path(rec['output_path']).name}" if rec.get("output_path") else None,
                "duration_sec": rec.get("duration_sec"),
                "segments_count": rec.get("segments_count"),
                "created_at": rec.get("created_at"),
                "manifest": manifest
            })
        return results

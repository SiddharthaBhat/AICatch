"""Video indexing, detector crops, and multi-camera journey stitching."""
from __future__ import annotations

import hashlib
import logging
import uuid
import os
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Any, Callable

# Keep first-run caches/settings inside the app workspace instead of user-profile
# directories, which are often read-only in managed deployments.
_RUNTIME_DIR = Path(__file__).resolve().parent / "temp"
_YOLO_CONFIG_DIR = _RUNTIME_DIR / "ultralytics"
_MPL_CONFIG_DIR = _RUNTIME_DIR / "matplotlib"
_YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import easyocr
import imageio_ffmpeg

from config import (
    CLIP_DIR, INDEX_DETECTOR_IMAGE_SIZE, INDEX_MAX_SUBJECTS_PER_SAMPLE,
    INDEX_SAMPLE_RATE_SEC, OUTPUT_DIR, SUPPORTED_DETECTION_CLASSES,
    THUMBNAIL_DIR, YOLO_MODEL_PATH, ensure_directories,
)
from db_engine import add_frames_data, get_image_embeddings

LOGGER = logging.getLogger(__name__)
_detector: YOLO | None = None
_ocr_reader: easyocr.Reader | None = None

def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        LOGGER.info("Loading EasyOCR reader...")
        _ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return _ocr_reader


def _sharpness(crop: Any) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    try:
        return float(cv2.Laplacian(crop, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _process_ocr(crop: Any, label: str) -> tuple[str, list[str], float]:
    primary_plate = ""
    plate_candidates = []
    confidence = 0.0
    if label in {"car", "motorcycle", "bus", "truck"}:
        try:
            reader = _get_ocr_reader()
            ocr_results = reader.readtext(
                crop,
                decoder='beamsearch',
                beamWidth=5,
                detail=1
            )
            best_conf = 0.0
            for bbox, text, conf in ocr_results:
                if conf > 0.20:
                    cleaned = "".join(c for c in text if c.isalnum()).upper()
                    if cleaned:
                        plate_candidates.append(cleaned)
                        if conf > best_conf:
                            best_conf = conf
                            primary_plate = cleaned
            plate_candidates = list(dict.fromkeys(plate_candidates))
            confidence = best_conf
        except Exception as ocr_err:
            LOGGER.error("OCR extraction failed: %s", ocr_err)
    return primary_plate, plate_candidates, confidence


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _get_detector() -> YOLO:
    global _detector
    if _detector is None:
        if Path(YOLO_MODEL_PATH).name not in {"yolov8n.pt", "yolo11n.pt"}:
            raise ValueError("Indexing requires a lightweight YOLO nano model (yolov8n.pt or yolo11n.pt)")
        device = _device()
        _detector = YOLO(YOLO_MODEL_PATH)
        _detector.to(device)
        if device == "cuda":
            _detector.model.half()
            torch.backends.cudnn.benchmark = True
    return _detector


def _open_capture(video_source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {video_source}")
    return capture


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:80]


def _motion_frame(frame: Any) -> Any:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(gray, (320, 180)), (5, 5), 0)


def _has_motion(previous: Any | None, current: Any, threshold: int = 20) -> bool:
    """Reject near-identical sampled frames before expensive model inference."""
    if previous is None:
        return True
    difference = cv2.absdiff(previous, current)
    _, changed = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(changed) >= int(changed.size * 0.005)


def process_video_stream(
    video_source: str,
    camera_id: str,
    sample_rate_sec: float = 2.5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Sample a source, detect subjects, and persist cropped CLIP embeddings."""
    if sample_rate_sec <= 0:
        raise ValueError("sample_rate_sec must be greater than zero")
    ensure_directories()
    
    # Retrieve metadata using a quick capture opening and release
    capture = _open_capture(video_source)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    
    subjects, sampled, skipped_static = 0, 0, 0
    detector = _get_detector()
    video_id = _safe_name(camera_id)
    thumbnail_folder = THUMBNAIL_DIR / video_id
    thumbnail_folder.mkdir(parents=True, exist_ok=True)
    
    crop_queue: list[dict[str, Any]] = []
    db_queue: list[dict[str, Any]] = []
    
    track_buffer: dict[int, list[dict[str, Any]]] = {}
    track_start_times: dict[int, float] = {}

    def process_crop_queue() -> None:
        if not crop_queue:
            return
        images = [record["image"] for record in crop_queue]
        embeddings = get_image_embeddings(images, batch_size=32)
        for record, embedding in zip(crop_queue, embeddings):
            record["embedding"] = embedding
            record.pop("image", None)
            db_queue.append(record)
        crop_queue.clear()
        
        if len(db_queue) >= 32:
            flush_db_queue()

    def flush_db_queue() -> None:
        nonlocal subjects
        if not db_queue:
            return
        add_frames_data(db_queue)
        subjects += len(db_queue)
        db_queue.clear()

    def queue_candidate(crop, bbox, timestamp, frame_idx, label, record_id):
        primary_plate, plate_candidates, confidence = _process_ocr(crop, label)
        
        # Save thumbnail immediately to disk to save memory
        cv2.imwrite(str(thumbnail_folder / f"{record_id}.jpg"), crop)
        
        crop_queue.append({
            "id": record_id,
            "camera_id": camera_id,
            "video_source": video_source,
            "timestamp": timestamp,
            "frame_idx": frame_idx,
            "bounding_box": bbox,
            "image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            "primary_plate": primary_plate,
            "plate_candidates": plate_candidates,
            "confidence": confidence
        })
        if len(crop_queue) >= 32:
            process_crop_queue()

    device = _device()
    # Replace manual OpenCV frame loops with Ultralytics Native Tracking:
    results = detector.track(
        source=video_source,
        stream=True,
        vid_stride=8,       # Native C++/FFmpeg frame stride (~3-4 FPS)
        imgsz=320,          # Ultra-fast tensor inference
        persist=True,
        conf=0.25,
        device=device,
        half=(device == "cuda"),
        verbose=False
    )

    last_reported_pct = -5.0
    
    for idx, r in enumerate(results):
        frame_idx = idx * 8
        sampled += 1
        names = r.names
        height, width = r.orig_img.shape[:2]
        current_frame_tids = set()
        
        if r.boxes is not None:
            for box in r.boxes:
                tid = int(box.id[0]) if box.id is not None else None
                if tid is not None:
                    current_frame_tids.add(tid)
                    
            for box in r.boxes:
                label = names[int(box.cls[0])]
                if label not in SUPPORTED_DETECTION_CLASSES:
                    continue
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Apply 25% context padding expansion
                w = x2 - x1
                h = y2 - y1
                x1_pad = max(0, int(x1 - 0.25 * w))
                y1_pad = max(0, int(y1 - 0.25 * h))
                x2_pad = min(width, int(x2 + 0.25 * w))
                y2_pad = min(height, int(y2 + 0.25 * h))
                crop = r.orig_img[y1_pad:y2_pad, x1_pad:x2_pad]
                if crop.size == 0:
                    continue
                
                bbox_padded = [x1_pad, y1_pad, x2_pad, y2_pad]
                tid = int(box.id[0]) if box.id is not None else None
                timestamp = frame_idx / fps
                if tid is not None:
                    try:
                        sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
                    except Exception:
                        sharpness = 0.0
                    
                    if tid not in track_buffer:
                        track_buffer[tid] = []
                        track_start_times[tid] = timestamp
                    
                    track_buffer[tid].append({
                        "crop": crop,
                        "bbox": bbox_padded,
                        "timestamp": timestamp,
                        "frame_idx": frame_idx,
                        "label": label,
                        "sharpness": sharpness
                    })
                    
                    if timestamp - track_start_times[tid] > 10.0:
                        best_candidate = max(track_buffer[tid], key=lambda x: x["sharpness"])
                        record_id = f"{video_id}_{tid}_{best_candidate['frame_idx']}_{uuid.uuid4().hex[:8]}"
                        queue_candidate(
                            best_candidate["crop"],
                            best_candidate["bbox"],
                            best_candidate["timestamp"],
                            best_candidate["frame_idx"],
                            best_candidate["label"],
                            record_id
                        )
                        track_buffer[tid] = [track_buffer[tid][-1]]
                        track_start_times[tid] = timestamp
                else:
                    record_id = f"{video_id}_untracked_{frame_idx}_{uuid.uuid4().hex[:8]}"
                    queue_candidate(crop, bbox_padded, timestamp, frame_idx, label, record_id)
        
        ended_tids = [tid for tid in track_buffer if tid not in current_frame_tids]
        for tid in ended_tids:
            if track_buffer[tid]:
                best_candidate = max(track_buffer[tid], key=lambda x: x["sharpness"])
                record_id = f"{video_id}_{tid}_{best_candidate['frame_idx']}_{uuid.uuid4().hex[:8]}"
                queue_candidate(
                    best_candidate["crop"],
                    best_candidate["bbox"],
                    best_candidate["timestamp"],
                    best_candidate["frame_idx"],
                    best_candidate["label"],
                    record_id
                )
            del track_buffer[tid]
            if tid in track_start_times:
                del track_start_times[tid]
                
        percentage = (frame_idx / total_frames) * 100 if total_frames else 0.0
        if progress_callback and (percentage - last_reported_pct >= 5.0 or frame_idx >= total_frames):
            progress_callback(min(frame_idx, total_frames), total_frames)
            last_reported_pct = percentage
            
    for tid in list(track_buffer.keys()):
        if track_buffer[tid]:
            best_candidate = max(track_buffer[tid], key=lambda x: x["sharpness"])
            record_id = f"{video_id}_{tid}_{best_candidate['frame_idx']}_{uuid.uuid4().hex[:8]}"
            queue_candidate(
                best_candidate["crop"],
                best_candidate["bbox"],
                best_candidate["timestamp"],
                best_candidate["frame_idx"],
                best_candidate["label"],
                record_id
            )
        del track_buffer[tid]
        
    process_crop_queue()
    flush_db_queue()
    
    return {"camera_id": camera_id, "frames_sampled": sampled, "frames_skipped_static": skipped_static,
            "subjects_indexed": subjects, "fps": fps, "device": _device()}


def _write_clip(source: str, timestamp: float, destination: Path, seconds: float = 5) -> bool:
    """Create a five-second H.264 fragment using FFmpeg, falling back to OpenCV."""
    start = max(0.0, timestamp - seconds / 2)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if ffmpeg:
        command = [ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", source, "-t", str(seconds), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode == 0 and destination.exists():
            return True
    cap = _open_capture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    written = 0
    try:
        while written < int(fps * seconds):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release(); writer.release()
    return written > 0


def create_context_clip(match: dict[str, Any], seconds: float = 10) -> str:
    """Extract a full-frame video clip centred on a search observation."""
    ensure_directories()
    metadata = match["metadata"]
    safe_id = _safe_name(str(match["id"]))
    destination = OUTPUT_DIR / "match_clips" / f"{safe_id}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and not _write_clip(
        str(metadata["video_source"]), float(metadata["timestamp"]), destination, seconds
    ):
        raise RuntimeError("Could not extract a playable clip from this video source")
    return str(destination)


def stitch_reid_clips(match_results: list[dict[str, Any]], output_path: str = "stitched_output.mp4") -> dict[str, Any]:
    """Chronologically extract detected moments and join them into one master video."""
    if not match_results:
        raise ValueError("Choose search matches before stitching a journey")
    ensure_directories()
    output = Path(output_path)
    if not output.is_absolute():
        output = OUTPUT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(match_results, key=lambda item: float(item["metadata"]["timestamp"]))
    segments: list[Path] = []
    journey: list[str] = []
    for index, match in enumerate(ordered, start=1):
        metadata = match["metadata"]
        timestamp = float(metadata["timestamp"])
        segment = CLIP_DIR / f"segment_{index:03d}.mp4"
        if _write_clip(metadata["video_source"], timestamp, segment):
            segments.append(segment)
            journey.append(f"{index}. Target detected on {metadata['camera_id']} at {timestamp:.2f}s.")
    if not segments:
        raise RuntimeError("No clips could be extracted from the selected sources")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if ffmpeg:
        manifest = CLIP_DIR / "concat.txt"
        manifest.write_text("\n".join(f"file '{path.resolve().as_posix()}'" for path in segments), encoding="utf-8")
        result = subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(output)], capture_output=True, text=True, timeout=180, check=False)
        if result.returncode == 0 and output.exists():
            return {"output_path": str(output), "journey_log": journey, "segments": len(segments)}
    # OpenCV fallback supports varying sources by resizing every frame to first clip dimensions.
    first = _open_capture(str(segments[0])); fps = first.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(first.get(cv2.CAP_PROP_FRAME_WIDTH)), int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))); first.release()
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    try:
        for segment in segments:
            cap = _open_capture(str(segment))
            while True:
                ok, frame = cap.read()
                if not ok: break
                writer.write(cv2.resize(frame, size) if frame.shape[1::-1] != size else frame)
            cap.release()
    finally:
        writer.release()
    return {"output_path": str(output), "journey_log": journey, "segments": len(segments)}

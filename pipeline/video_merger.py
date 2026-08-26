"""AICatch Video Clip and Journey Concatenation Engine.

Extracts frame-accurate video segments around observation timestamps, adds
context padding (e.g. ±3 seconds), and concatenates them into single MP4 files
for download, playback, and evidence archives.
"""

import os
import sys
import time
import uuid
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import imageio_ffmpeg
from config import CLIPS_DIR, MEDIA_DIR, UPLOAD_DIR

LOGGER = logging.getLogger(__name__)

def get_ffmpeg_path() -> str:
    """Returns the absolute path to the bundled FFmpeg static executable."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        LOGGER.warning(f"Could not resolve imageio_ffmpeg binary, falling back to system ffmpeg: {e}")
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
        if (base / clean).exists():
            return base / clean
        if (base / p.name).exists():
            return base / p.name
    return None

class VideoMerger:
    def __init__(self):
        self.ffmpeg_exe = get_ffmpeg_path()
        LOGGER.info(f"Initialized VideoMerger with FFmpeg executable: {self.ffmpeg_exe}")

    def extract_subclip(
        self,
        source_path: Path,
        start_sec: float,
        end_sec: float,
        output_path: Path
    ) -> bool:
        """Extracts a precise subclip from start_sec to end_sec."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            duration = max(0.5, end_sec - start_sec)
            
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-ss", f"{start_sec:.3f}",
                "-i", str(source_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(output_path)
            ]
            
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            if res.returncode != 0:
                LOGGER.error(f"FFmpeg subclip failed for {source_path}: {res.stderr[:200]}")
                return False
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception as e:
            LOGGER.error(f"Error extracting subclip from {source_path}: {e}", exc_info=True)
            return False

    def merge_observation_clips(
        self,
        job_id: str,
        observations: List[Dict[str, Any]],
        context_buffer_sec: float = 3.0,
        output_filename: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Extracts and merges subclips for selected search observations chronologically."""
        if not observations:
            return None

        sorted_obs = sorted(observations, key=lambda x: x.get("timestamp", 0.0))
        
        job_clips_dir = CLIPS_DIR / job_id
        job_clips_dir.mkdir(parents=True, exist_ok=True)
        
        temp_dir = job_clips_dir / f"temp_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        subclip_paths: List[Path] = []
        segments_info: List[Dict[str, Any]] = []

        try:
            for idx, obs in enumerate(sorted_obs):
                source_path_str = obs.get("source_path") or obs.get("video_source")
                source_path = resolve_video_path(source_path_str)
                
                if not source_path:
                    LOGGER.warning(f"Skipping observation {idx}: could not resolve source path {source_path_str}")
                    continue

                ts = float(obs.get("timestamp", 0.0))
                start_sec = max(0.0, ts - context_buffer_sec)
                end_sec = ts + context_buffer_sec + float(obs.get("duration", 2.0))
                
                clip_file = temp_dir / f"clip_{idx:03d}.mp4"
                cam_name = obs.get("camera_name") or obs.get("camera_id") or "Camera"
                
                success = self.extract_subclip(
                    source_path=source_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    output_path=clip_file
                )
                
                if success:
                    subclip_paths.append(clip_file)
                    segments_info.append({
                        "camera_name": cam_name,
                        "timestamp": ts,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "track_id": obs.get("track_id")
                    })

            if not subclip_paths:
                LOGGER.error("No valid subclips could be extracted for merge.")
                return None

            concat_list_file = temp_dir / "concat_list.txt"
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for cp in subclip_paths:
                    clean_path = str(cp).replace("\\", "/")
                    f.write(f"file '{clean_path}'\n")

            final_name = output_filename or f"merged_clips_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4"
            final_output_path = job_clips_dir / final_name

            concat_cmd = [
                self.ffmpeg_exe,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(final_output_path)
            ]

            res = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            if res.returncode != 0 or not final_output_path.exists():
                reencode_cmd = [
                    self.ffmpeg_exe,
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list_file),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "22",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(final_output_path)
                ]
                res2 = subprocess.run(reencode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
                if res2.returncode != 0 or not final_output_path.exists():
                    LOGGER.error(f"FFmpeg re-encode concat failed: {res2.stderr[:200]}")
                    return None

            return {
                "filename": final_name,
                "path": str(final_output_path).replace("\\", "/"),
                "relative_path": f"/media/clips/{job_id}/{final_name}",
                "segments_count": len(subclip_paths),
                "segments": segments_info
            }

        finally:
            try:
                for cp in subclip_paths:
                    if cp.exists():
                        cp.unlink()
                if (temp_dir / "concat_list.txt").exists():
                    (temp_dir / "concat_list.txt").unlink()
                if temp_dir.exists():
                    temp_dir.rmdir()
            except Exception as ce:
                LOGGER.debug(f"Temporary clip cleanup note: {ce}")

# Global instance
video_merger = VideoMerger()

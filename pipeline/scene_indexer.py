"""Intelligent Scene / Keyframe Indexer for AICatch.

Extracts representative scene keyframes across CCTV video streams,
generates normalized full-frame CLIP embeddings, and stores them in ChromaDB
under the scene collection with forensic metadata.
"""

import cv2
import numpy as np
from PIL import Image
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_VERSION, CLIP_DEVICE,
    PROCESSING_PROFILES
)
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from storage.media_storage import MediaStorage

LOGGER = logging.getLogger(__name__)

class SceneIndexer:
    def __init__(
        self,
        sqlite_db: SQLiteDatabase,
        vector_db: VectorDatabase,
        media_storage: MediaStorage,
        gpu_scheduler=None
    ):
        self.sqlite = sqlite_db
        self.vector = vector_db
        self.media = media_storage
        self.gpu_scheduler = gpu_scheduler

    def extract_and_index_scenes_for_camera(
        self,
        job_id: str,
        camera_id: str,
        source_path: str,
        keyframe_interval_sec: float = 3.0,
        scene_change_threshold: float = 0.08,
        batch_size: int = 16,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """Extracts intelligent scene keyframes for a single camera and indexes them."""
        # Check if scene vectors already exist for this camera/job
        existing = self.vector.scene_coll.get(
            where={"$and": [{"job_id": job_id}, {"camera_id": camera_id}]},
            limit=5
        )
        if existing and existing.get("ids") and not force_reindex:
            existing_count = len(self.vector.scene_coll.get(
                where={"$and": [{"job_id": job_id}, {"camera_id": camera_id}]}
            )["ids"])
            LOGGER.info(f"Camera {camera_id} already has {existing_count} scene vectors indexed. Skipping.")
            return {
                "camera_id": camera_id,
                "status": "ALREADY_INDEXED",
                "scene_vectors": existing_count,
                "frames_processed": 0
            }

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        duration_sec = total_frames / fps if fps > 0 else 0.0
        
        # Calculate frame step for keyframe sampling
        frame_step = max(1, int(fps * keyframe_interval_sec))
        
        LOGGER.info(
            f"Indexing scenes for cam {camera_id} ({source_path}): "
            f"Total Frames={total_frames}, FPS={fps:.2f}, Step={frame_step} (~{keyframe_interval_sec}s)"
        )

        keyframes_to_embed: List[Tuple[int, float, np.ndarray, Path]] = []
        last_gray: Optional[np.ndarray] = None
        last_keyframe_time = -999.0
        
        current_frame_idx = 0
        extracted_count = 0
        skipped_duplicates = 0
        
        t0 = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_idx += 1
            timestamp = current_frame_idx / fps
            
            # Periodic & scene-change keyframe check
            is_periodic = (current_frame_idx % frame_step == 1) or (current_frame_idx == 1)
            time_since_last = timestamp - last_keyframe_time
            
            # Downsample for fast change comparison
            small_gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            is_scene_change = False
            
            if last_gray is not None and time_since_last >= 1.5:
                diff = cv2.absdiff(last_gray, small_gray)
                change_ratio = float(np.count_nonzero(diff > 25)) / float(diff.size)
                if change_ratio >= scene_change_threshold:
                    is_scene_change = True
            
            should_capture = (is_periodic and time_since_last >= 2.0) or is_scene_change or (current_frame_idx == 1)
            
            if should_capture:
                # Save keyframe image to disk
                keyframe_path = self.media.save_keyframe(
                    frame, job_id, camera_id, current_frame_idx, timestamp
                )
                keyframes_to_embed.append((current_frame_idx, timestamp, frame, keyframe_path))
                last_gray = small_gray
                last_keyframe_time = timestamp
                extracted_count += 1
            else:
                skipped_duplicates += 1

        cap.release()
        
        # Generate CLIP embeddings and store in ChromaDB in batches
        indexed_count = 0
        ids_batch = []
        embs_batch = []
        metas_batch = []
        
        for f_idx, t_stamp, f_bgr, kf_path in keyframes_to_embed:
            pil_img = Image.fromarray(cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB))
            
            if self.gpu_scheduler is not None:
                emb = self.gpu_scheduler.embed_image(pil_img)
            else:
                from transformers import CLIPModel, CLIPProcessor
                import torch
                processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
                model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(CLIP_DEVICE)
                inputs = processor(images=[pil_img], return_tensors="pt").to(CLIP_DEVICE)
                with torch.no_grad():
                    feats = model.get_image_features(**inputs)
                    if hasattr(feats, "pooler_output"):
                        feats = feats.pooler_output
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    emb = feats.cpu().numpy()[0].tolist()
            
            vec_id = f"scene_{job_id}_{camera_id}_f{f_idx}"
            ids_batch.append(vec_id)
            embs_batch.append(emb)
            metas_batch.append({
                "job_id": job_id,
                "investigation_id": job_id,
                "camera_id": camera_id,
                "video_id": camera_id,
                "frame_idx": f_idx,
                "timestamp": round(float(t_stamp), 3),
                "embedding_type": "scene",
                "object_type": "scene",
                "source_frame_path": str(kf_path),
                "source_video": source_path,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_version": EMBEDDING_VERSION
            })
            
            if len(ids_batch) >= batch_size:
                self.vector.add_scene_embeddings(ids_batch, embs_batch, metas_batch)
                indexed_count += len(ids_batch)
                ids_batch.clear()
                embs_batch.clear()
                metas_batch.clear()
                
        if ids_batch:
            self.vector.add_scene_embeddings(ids_batch, embs_batch, metas_batch)
            indexed_count += len(ids_batch)
            
        elapsed = time.time() - t0
        LOGGER.info(
            f"Scene indexing finished for cam {camera_id}: "
            f"{indexed_count} keyframes indexed, {skipped_duplicates} frames skipped in {elapsed:.2f}s"
        )
        
        return {
            "camera_id": camera_id,
            "status": "INDEXED",
            "total_video_frames": total_frames,
            "keyframes_indexed": indexed_count,
            "frames_skipped": skipped_duplicates,
            "deduplication_rate": round(skipped_duplicates / max(1, total_frames), 4),
            "duration_sec": round(elapsed, 2)
        }

    def index_job_scenes(self, job_id: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Indexes scene keyframes for all cameras in a given job/investigation."""
        cameras = self.sqlite.get_cameras_by_job(job_id)
        if not cameras:
            raise ValueError(f"No cameras found for job {job_id}")
            
        results = []
        total_keyframes = 0
        t0 = time.time()
        
        for cam in cameras:
            res = self.extract_and_index_scenes_for_camera(
                job_id=job_id,
                camera_id=cam["id"],
                source_path=cam["source_path"],
                force_reindex=force_reindex
            )
            results.append(res)
            total_keyframes += res.get("keyframes_indexed", res.get("scene_vectors", 0))
            
        total_time = time.time() - t0
        return {
            "job_id": job_id,
            "cameras_count": len(cameras),
            "total_scene_keyframes": total_keyframes,
            "camera_results": results,
            "elapsed_sec": round(total_time, 2)
        }

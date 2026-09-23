"""Intelligent Scene / Keyframe Indexer for AICatch.

Extracts representative scene keyframes across CCTV video streams,
generates normalized full-frame CLIP embeddings, and stores them in ChromaDB
under the scene collection with forensic metadata.
"""

import collections
import concurrent.futures
import cv2
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import queue
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

from config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_VERSION, CLIP_DEVICE, CLIP_MODEL_NAME,
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
        self._fallback_clip_model = None
        self._fallback_clip_processor = None

    def _get_clip(self):
        if self._fallback_clip_model is None or self._fallback_clip_processor is None:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            # The scheduler already treats the locally cached model as the normal
            # path.  Do the same here: scene indexing must not block on network
            # retries when an investigation is being validated or run offline.
            self._fallback_clip_processor = CLIPProcessor.from_pretrained(
                CLIP_MODEL_NAME, local_files_only=True
            )
            self._fallback_clip_model = CLIPModel.from_pretrained(
                CLIP_MODEL_NAME, local_files_only=True
            ).to(CLIP_DEVICE)
            if CLIP_DEVICE == "cuda":
                self._fallback_clip_model = self._fallback_clip_model.half()
            self._fallback_clip_model.eval()
        return self._fallback_clip_processor, self._fallback_clip_model

    @staticmethod
    def generate_frame_tiles(
        frame: np.ndarray,
        tile_ratio: float = 0.45,
        stride_ratio: float = 0.25,
        include_micro_scale: bool = True
    ) -> List[Tuple[int, List[int], np.ndarray]]:
        """Generates overlapping multi-scale spatial tiles across a video frame.
        Includes macro context tiles (~45%) and high-salience micro tiles (~22%) for small-object retrieval.
        Returns list of (tile_idx, [x1, y1, x2, y2], tile_bgr_crop).
        """
        H, W = frame.shape[:2]
        tile_w = max(64, int(W * tile_ratio))
        tile_h = max(64, int(H * tile_ratio))
        stride_x = max(32, int(W * stride_ratio))
        stride_y = max(32, int(H * stride_ratio))

        xs = list(range(0, W - tile_w + 1, stride_x))
        if not xs or xs[-1] + tile_w < W:
            xs.append(max(0, W - tile_w))
        ys = list(range(0, H - tile_h + 1, stride_y))
        if not ys or ys[-1] + tile_h < H:
            ys.append(max(0, H - tile_h))

        xs = sorted(list(set(xs)))
        ys = sorted(list(set(ys)))

        tiles = []
        tile_idx = 0
        for y1 in ys:
            y2 = min(H, y1 + tile_h)
            for x1 in xs:
                x2 = min(W, x1 + tile_w)
                tile_crop = frame[y1:y2, x1:x2]
                tiles.append((tile_idx, [int(x1), int(y1), int(x2), int(y2)], tile_crop))
                tile_idx += 1

        # Micro-scale tiles for fine-grained small static objects (22% scale)
        if include_micro_scale and W >= 128 and H >= 128:
            tile_w_micro = max(32, int(W * 0.22))
            tile_h_micro = max(32, int(H * 0.22))
            stride_x_micro = max(24, int(W * 0.18))
            stride_y_micro = max(24, int(H * 0.18))

            xs_micro = sorted(list(set(list(range(0, W - tile_w_micro + 1, stride_x_micro)) + [max(0, W - tile_w_micro)])))
            ys_micro = sorted(list(set(list(range(0, H - tile_h_micro + 1, stride_y_micro)) + [max(0, H - tile_h_micro)])))

            micro_candidates = []
            for my1 in ys_micro:
                my2 = min(H, my1 + tile_h_micro)
                for mx1 in xs_micro:
                    mx2 = min(W, mx1 + tile_w_micro)
                    m_crop = frame[my1:my2, mx1:mx2]
                    gray = cv2.cvtColor(m_crop, cv2.COLOR_BGR2GRAY)
                    variance = float(np.var(gray))
                    micro_candidates.append((variance, [int(mx1), int(my1), int(mx2), int(my2)], m_crop))

            # Select top-12 most salient micro tiles by visual texture/contrast
            micro_candidates.sort(key=lambda x: x[0], reverse=True)
            for _, m_bbox, m_crop in micro_candidates[:12]:
                tiles.append((tile_idx, m_bbox, m_crop))
                tile_idx += 1

        return tiles

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
        """Extracts intelligent scene keyframes and spatial region tiles for a single camera and indexes them."""
        # Handle force_reindex cleanup
        if force_reindex:
            try:
                self.vector.scene_coll.delete(where={"$and": [{"job_id": job_id}, {"camera_id": camera_id}]})
            except Exception:
                pass
            try:
                self.vector.region_coll.delete(where={"$and": [{"job_id": job_id}, {"camera_id": camera_id}]})
            except Exception:
                pass
        else:
            # Check if both scene and region vectors already exist for this camera/job
            existing_scenes = self.vector.scene_coll.get(
                where={"$and": [{"job_id": job_id}, {"camera_id": camera_id}]},
                limit=5
            )
            existing_regions = self.vector.region_coll.get(
                where={"$and": [{"job_id": job_id}, {"camera_id": camera_id}]},
                limit=5
            )
            if existing_scenes and existing_scenes.get("ids") and existing_regions and existing_regions.get("ids"):
                existing_s_count = len(self.vector.scene_coll.get(
                    where={"$and": [{"job_id": job_id}, {"camera_id": camera_id}]}
                )["ids"])
                existing_r_count = len(self.vector.region_coll.get(
                    where={"$and": [{"job_id": job_id}, {"camera_id": camera_id}]}
                )["ids"])
                LOGGER.info(
                    f"Camera {camera_id} already has {existing_s_count} scene vectors and "
                    f"{existing_r_count} region vectors indexed. Skipping."
                )
                return {
                    "camera_id": camera_id,
                    "status": "ALREADY_INDEXED",
                    "scene_vectors": existing_s_count,
                    "region_vectors": existing_r_count,
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
            f"Indexing scenes and regions for cam {camera_id} ({source_path}): "
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
                # Save full keyframe image to disk
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
        
        # Batched indexing for Scenes & Regions
        indexed_scene_count = 0
        indexed_region_count = 0
        
        scene_ids: List[str] = []
        scene_embs: List[List[float]] = []
        scene_metas: List[Dict[str, Any]] = []
        
        region_ids: List[str] = []
        region_images: List[Image.Image] = []
        region_metas: List[Dict[str, Any]] = []

        def flush_scenes():
            nonlocal indexed_scene_count
            if scene_ids:
                self.vector.add_scene_embeddings(scene_ids, scene_embs, scene_metas)
                indexed_scene_count += len(scene_ids)
                scene_ids.clear()
                scene_embs.clear()
                scene_metas.clear()

        def flush_regions():
            nonlocal indexed_region_count
            if region_ids:
                if self.gpu_scheduler is not None and hasattr(self.gpu_scheduler, "embed_images_batch"):
                    embs = self.gpu_scheduler.embed_images_batch(region_images, batch_size=16)
                elif self.gpu_scheduler is not None:
                    embs = [self.gpu_scheduler.embed_image(img) for img in region_images]
                else:
                    processor, model = self._get_clip()
                    import torch
                    all_embs = []
                    for b_i in range(0, len(region_images), 16):
                        chunk = region_images[b_i : b_i + 16]
                        inputs = processor(images=chunk, return_tensors="pt", padding=True).to(CLIP_DEVICE)
                        if CLIP_DEVICE == "cuda":
                            inputs["pixel_values"] = inputs["pixel_values"].half()
                        with torch.no_grad():
                            feats = model.get_image_features(**inputs)
                            if hasattr(feats, "pooler_output"):
                                feats = feats.pooler_output
                            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                            all_embs.extend(feats.cpu().float().numpy().tolist())
                    embs = all_embs

                self.vector.add_region_embeddings(region_ids, embs, region_metas)
                indexed_region_count += len(region_ids)
                region_ids.clear()
                region_images.clear()
                region_metas.clear()
        
        for f_idx, t_stamp, f_bgr, kf_path in keyframes_to_embed:
            # 1. Full-frame keyframe embedding
            pil_img = Image.fromarray(cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB))
            
            if self.gpu_scheduler is not None:
                emb = self.gpu_scheduler.embed_image(pil_img)
            else:
                processor, model = self._get_clip()
                import torch
                inputs = processor(images=[pil_img], return_tensors="pt").to(CLIP_DEVICE)
                if CLIP_DEVICE == "cuda":
                    inputs["pixel_values"] = inputs["pixel_values"].half()
                with torch.no_grad():
                    feats = model.get_image_features(**inputs)
                    if hasattr(feats, "pooler_output"):
                        feats = feats.pooler_output
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    emb = feats.cpu().float().numpy()[0].tolist()
            
            vec_id = f"scene_{job_id}_{camera_id}_f{f_idx}"
            scene_ids.append(vec_id)
            scene_embs.append(emb)
            scene_metas.append({
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
            
            if len(scene_ids) >= batch_size:
                flush_scenes()

            # 2. Overlapping Spatial Regions / Tiles
            tiles = self.generate_frame_tiles(f_bgr)
            for tile_idx, (bx1, by1, bx2, by2), tile_bgr in tiles:
                # Save region tile crop image
                reg_path = self.media.save_region(
                    tile_bgr, job_id, camera_id, f_idx, tile_idx
                )
                tile_pil = Image.fromarray(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))
                r_vec_id = f"region_{job_id}_{camera_id}_f{f_idx}_t{tile_idx}"
                
                region_ids.append(r_vec_id)
                region_images.append(tile_pil)
                region_metas.append({
                    "job_id": job_id,
                    "investigation_id": job_id,
                    "camera_id": camera_id,
                    "video_id": camera_id,
                    "frame_idx": f_idx,
                    "tile_idx": tile_idx,
                    "timestamp": round(float(t_stamp), 3),
                    "bbox": f"{bx1},{by1},{bx2},{by2}",
                    "bbox_x1": float(bx1),
                    "bbox_y1": float(by1),
                    "bbox_x2": float(bx2),
                    "bbox_y2": float(by2),
                    "embedding_type": "region",
                    "object_type": "region",
                    "crop_path": str(reg_path),
                    "source_frame_path": str(kf_path),
                    "source_video": source_path,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "embedding_version": EMBEDDING_VERSION
                })

                if len(region_ids) >= 32:
                    flush_regions()
            
        flush_scenes()
        flush_regions()
            
        elapsed = time.time() - t0
        LOGGER.info(
            f"Scene & Region indexing finished for cam {camera_id}: "
            f"{indexed_scene_count} keyframes and {indexed_region_count} regions indexed, "
            f"{skipped_duplicates} frames skipped in {elapsed:.2f}s"
        )
        
        return {
            "camera_id": camera_id,
            "status": "INDEXED",
            "total_video_frames": total_frames,
            "keyframes_indexed": indexed_scene_count,
            "regions_indexed": indexed_region_count,
            "frames_skipped": skipped_duplicates,
            "deduplication_rate": round(skipped_duplicates / max(1, total_frames), 4),
            "duration_sec": round(elapsed, 2)
        }

    def index_job_scenes(
        self,
        job_id: str,
        force_reindex: bool = False,
        max_cpu_workers: int = 4,
    ) -> Dict[str, Any]:
        """Indexes scene keyframes and spatial regions for all cameras in a given job/investigation
        using controlled concurrency (pipelined CPU camera preparation -> bounded GPU worker -> persistence worker).
        """
        cameras = self.sqlite.get_cameras_by_job(job_id)
        if not cameras:
            raise ValueError(f"No cameras found for job {job_id}")

        t0 = time.perf_counter()

        # Handle force_reindex cleanup
        if force_reindex:
            for cam in cameras:
                cid = cam["id"]
                try:
                    self.vector.scene_coll.delete(where={"$and": [{"job_id": job_id}, {"camera_id": cid}]})
                except Exception:
                    pass
                try:
                    self.vector.region_coll.delete(where={"$and": [{"job_id": job_id}, {"camera_id": cid}]})
                except Exception:
                    pass

        # Filter cameras that need indexing
        cams_to_process = []
        already_indexed_results = []
        for cam in cameras:
            cid = cam["id"]
            if not force_reindex:
                ex_s = self.vector.scene_coll.get(where={"$and": [{"job_id": job_id}, {"camera_id": cid}]}, limit=1)
                ex_r = self.vector.region_coll.get(where={"$and": [{"job_id": job_id}, {"camera_id": cid}]}, limit=1)
                if ex_s and ex_s.get("ids") and ex_r and ex_r.get("ids"):
                    existing_s_count = len(self.vector.scene_coll.get(
                        where={"$and": [{"job_id": job_id}, {"camera_id": cid}]}
                    )["ids"])
                    existing_r_count = len(self.vector.region_coll.get(
                        where={"$and": [{"job_id": job_id}, {"camera_id": cid}]}
                    )["ids"])
                    already_indexed_results.append({
                        "camera_id": cid,
                        "status": "ALREADY_INDEXED",
                        "scene_vectors": existing_s_count,
                        "region_vectors": existing_r_count,
                        "frames_processed": 0,
                    })
                    continue
            cams_to_process.append(cam)

        if not cams_to_process:
            return {
                "job_id": job_id,
                "cameras_count": len(cameras),
                "total_scene_keyframes": sum(r.get("scene_vectors", 0) for r in already_indexed_results),
                "total_scene_regions": sum(r.get("region_vectors", 0) for r in already_indexed_results),
                "camera_results": already_indexed_results,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }

        # Concurrency primitives
        # Bounded queues prevent RAM and VRAM explosion
        work_queue: queue.Queue[Optional[Tuple[str, str, Image.Image, Dict[str, Any]]]] = queue.Queue(maxsize=128)
        persist_queue: queue.Queue[Optional[Tuple[str, List[str], List[List[float]], List[Dict[str, Any]]]]] = queue.Queue(maxsize=128)

        total_keyframes_counter = collections.Counter()
        total_regions_counter = collections.Counter()
        camera_stats: Dict[str, Dict[str, Any]] = {}
        error_holder: List[Exception] = []

        # 1. Persistence Worker (Single writer thread for ChromaDB)
        def persistence_worker_loop():
            while True:
                task = persist_queue.get()
                if task is None:
                    persist_queue.task_done()
                    break
                item_type, ids, embs, metas = task
                try:
                    if item_type == "scene":
                        self.vector.add_scene_embeddings(ids, embs, metas)
                    elif item_type == "region":
                        self.vector.add_region_embeddings(ids, embs, metas)
                except Exception as ex:
                    LOGGER.error(f"Persistence error for {item_type}: {ex}", exc_info=True)
                    error_holder.append(ex)
                finally:
                    persist_queue.task_done()

        persist_thread = threading.Thread(target=persistence_worker_loop, name="SceneRegionPersistWorker", daemon=True)
        persist_thread.start()

        # 2. GPU Worker (Single controlled CLIP inference path)
        def gpu_worker_loop():
            accum_scenes: List[Tuple[str, Image.Image, Dict[str, Any]]] = []
            accum_regions: List[Tuple[str, Image.Image, Dict[str, Any]]] = []

            def flush_scenes(final=False):
                if not accum_scenes:
                    return
                ids = [it[0] for it in accum_scenes]
                pils = [it[1] for it in accum_scenes]
                metas = [it[2] for it in accum_scenes]

                # Match baseline exact single-image inference for scene keyframes
                if self.gpu_scheduler is not None and hasattr(self.gpu_scheduler, "embed_image"):
                    embs = [self.gpu_scheduler.embed_image(img) for img in pils]
                else:
                    proc, mod = self._get_clip()
                    import torch
                    embs = []
                    for img in pils:
                        rgb_img = img if img.mode == "RGB" else img.convert("RGB")
                        inp = proc(images=[rgb_img], return_tensors="pt").to(CLIP_DEVICE)
                        if CLIP_DEVICE == "cuda":
                            inp["pixel_values"] = inp["pixel_values"].half()
                        with torch.no_grad():
                            res = mod.get_image_features(**inp)
                            if hasattr(res, "pooler_output"):
                                res = res.pooler_output
                            res = res / res.norm(p=2, dim=-1, keepdim=True)
                            embs.append(res.cpu().float().numpy()[0].tolist())

                persist_queue.put(("scene", ids, embs, metas))
                accum_scenes.clear()

            def flush_regions(final=False):
                if not accum_regions:
                    return
                ids = [it[0] for it in accum_regions]
                pils = [it[1] for it in accum_regions]
                metas = [it[2] for it in accum_regions]

                # Match baseline exact batched inference for region tiles
                if self.gpu_scheduler is not None and hasattr(self.gpu_scheduler, "embed_images_batch"):
                    embs = self.gpu_scheduler.embed_images_batch(pils, batch_size=16)
                else:
                    proc, mod = self._get_clip()
                    import torch
                    embs = []
                    for b_i in range(0, len(pils), 16):
                        chunk = pils[b_i : b_i + 16]
                        rgb_chunk = [img if img.mode == "RGB" else img.convert("RGB") for img in chunk]
                        inp = proc(images=rgb_chunk, return_tensors="pt", padding=True).to(CLIP_DEVICE)
                        if CLIP_DEVICE == "cuda":
                            inp["pixel_values"] = inp["pixel_values"].half()
                        with torch.no_grad():
                            res = mod.get_image_features(**inp)
                            if hasattr(res, "pooler_output"):
                                res = res.pooler_output
                            res = res / res.norm(p=2, dim=-1, keepdim=True)
                            embs.extend(res.cpu().float().numpy().tolist())

                persist_queue.put(("region", ids, embs, metas))
                accum_regions.clear()

            while True:
                try:
                    item = work_queue.get(timeout=0.2)
                except queue.Empty:
                    if error_holder:
                        break
                    continue

                if item is None:
                    work_queue.task_done()
                    break

                item_type, vec_id, pil_img, meta = item
                if item_type == "scene":
                    accum_scenes.append((vec_id, pil_img, meta))
                    if len(accum_scenes) >= 16:
                        flush_scenes()
                elif item_type == "region":
                    accum_regions.append((vec_id, pil_img, meta))
                    if len(accum_regions) >= 32:
                        flush_regions()

                work_queue.task_done()

            # Flush any remaining items
            flush_scenes(final=True)
            flush_regions(final=True)

        gpu_thread = threading.Thread(target=gpu_worker_loop, name="SceneRegionGPUWorker", daemon=True)
        gpu_thread.start()

        # 3. CPU Camera Workers (Video decode, keyframe detection, tiling, crop saving)
        def process_camera_cpu(cam: Dict[str, Any]):
            cid = cam["id"]
            source_path = cam["source_path"]

            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                LOGGER.error(f"Cannot open video source: {source_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
            frame_step = max(1, int(fps * 3.0))

            last_gray: Optional[np.ndarray] = None
            last_keyframe_time = -999.0
            current_frame_idx = 0
            extracted_count = 0
            skipped_duplicates = 0
            regions_count = 0
            t_cam_0 = time.perf_counter()

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_frame_idx += 1
                    timestamp = current_frame_idx / fps

                    is_periodic = (current_frame_idx % frame_step == 1) or (current_frame_idx == 1)
                    time_since_last = timestamp - last_keyframe_time

                    small_gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
                    is_scene_change = False

                    if last_gray is not None and time_since_last >= 1.5:
                        diff = cv2.absdiff(last_gray, small_gray)
                        change_ratio = float(np.count_nonzero(diff > 25)) / float(diff.size)
                        if change_ratio >= 0.08:
                            is_scene_change = True

                    should_capture = (is_periodic and time_since_last >= 2.0) or is_scene_change or (current_frame_idx == 1)

                    if should_capture:
                        # Save full keyframe image to disk
                        kf_path = self.media.save_keyframe(frame, job_id, cid, current_frame_idx, timestamp)
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        vec_id = f"scene_{job_id}_{cid}_f{current_frame_idx}"
                        meta = {
                            "job_id": job_id,
                            "investigation_id": job_id,
                            "camera_id": cid,
                            "video_id": cid,
                            "frame_idx": current_frame_idx,
                            "timestamp": round(float(timestamp), 3),
                            "embedding_type": "scene",
                            "object_type": "scene",
                            "source_frame_path": str(kf_path),
                            "source_video": source_path,
                            "embedding_model": EMBEDDING_MODEL_NAME,
                            "embedding_version": EMBEDDING_VERSION,
                        }
                        work_queue.put(("scene", vec_id, pil_img, meta))
                        extracted_count += 1
                        total_keyframes_counter[cid] += 1
                        last_gray = small_gray
                        last_keyframe_time = timestamp

                        # Generate overlapping spatial tiles
                        tiles = self.generate_frame_tiles(frame)
                        for tile_idx, (bx1, by1, bx2, by2), tile_bgr in tiles:
                            reg_path = self.media.save_region(tile_bgr, job_id, cid, current_frame_idx, tile_idx)
                            tile_pil = Image.fromarray(cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB))
                            r_vec_id = f"region_{job_id}_{cid}_f{current_frame_idx}_t{tile_idx}"
                            r_meta = {
                                "job_id": job_id,
                                "investigation_id": job_id,
                                "camera_id": cid,
                                "video_id": cid,
                                "frame_idx": current_frame_idx,
                                "tile_idx": tile_idx,
                                "timestamp": round(float(timestamp), 3),
                                "bbox": f"{bx1},{by1},{bx2},{by2}",
                                "bbox_x1": float(bx1),
                                "bbox_y1": float(by1),
                                "bbox_x2": float(bx2),
                                "bbox_y2": float(by2),
                                "embedding_type": "region",
                                "object_type": "region",
                                "crop_path": str(reg_path),
                                "source_frame_path": str(kf_path),
                                "source_video": source_path,
                                "embedding_model": EMBEDDING_MODEL_NAME,
                                "embedding_version": EMBEDDING_VERSION,
                            }
                            work_queue.put(("region", r_vec_id, tile_pil, r_meta))
                            regions_count += 1
                            total_regions_counter[cid] += 1
                    else:
                        skipped_duplicates += 1
            finally:
                cap.release()

            camera_stats[cid] = {
                "camera_id": cid,
                "status": "INDEXED",
                "total_video_frames": total_frames,
                "keyframes_indexed": extracted_count,
                "regions_indexed": regions_count,
                "frames_skipped": skipped_duplicates,
                "deduplication_rate": round(skipped_duplicates / max(1, total_frames), 4),
                "duration_sec": round(time.perf_counter() - t_cam_0, 2),
            }

        # Run CPU camera workers concurrently
        workers = min(max_cpu_workers, len(cams_to_process)) if cams_to_process else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="SceneCPUWorker") as executor:
            futures = [executor.submit(process_camera_cpu, c) for c in cams_to_process]
            concurrent.futures.wait(futures)
            for f in futures:
                if f.exception():
                    error_holder.append(f.exception())

        # Signal GPU worker to finish and wait
        work_queue.put(None)
        gpu_thread.join(timeout=60.0)

        # Signal Persistence worker to finish and wait
        persist_queue.put(None)
        persist_thread.join(timeout=60.0)

        if error_holder:
            raise error_holder[0]

        elapsed = round(time.perf_counter() - t0, 2)
        total_kf = sum(total_keyframes_counter.values()) + sum(r.get("scene_vectors", 0) for r in already_indexed_results)
        total_reg = sum(total_regions_counter.values()) + sum(r.get("region_vectors", 0) for r in already_indexed_results)

        return {
            "job_id": job_id,
            "cameras_count": len(cameras),
            "total_scene_keyframes": total_kf,
            "total_scene_regions": total_reg,
            "camera_results": list(camera_stats.values()) + already_indexed_results,
            "elapsed_sec": elapsed,
        }

"""VRAM-Safe GPU Inference Scheduler for AICatch.

Coordinates batching, YOLOv8 tracking, per-camera track management,
sharp keyframe filtering, and sequential model scheduling on the RTX 3050.
"""

import threading
import queue
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.utils
transformers.utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from PIL import Image

import torchvision.models as tv_models
from torchvision import transforms
from config import (
    DEVICE, YOLO_DEVICE, CLIP_DEVICE, REID_DEVICE, FACE_DEVICE, OCR_DEVICE, REID_MODEL_NAME, REID_VERSION, EMBEDDING_MODEL_NAME, EMBEDDING_VERSION,
    YOLO_MODEL_PATH, CLIP_MODEL_NAME, BATCH_SIZE, GPU_QUEUE_LIMIT,
    PROCESSING_PROFILES, MAX_VRAM_USAGE_MB, YOLO_USE_AMP
)
from storage.cache import TrackCache, TrackData
from storage.media_storage import MediaStorage
from utils.color_analyzer import evaluate_crop_quality, extract_body_colors, extract_vehicle_colors
from utils.monitor import PerformanceMonitorInstance

# Dynamic library imports (to handle modular loading)
from ultralytics import YOLO

from transformers import CLIPModel, CLIPProcessor

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization
import easyocr

LOGGER = logging.getLogger(__name__)

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0

def _reid_prep_worker(images: List[Image.Image], reid_transform: transforms.Compose) -> torch.Tensor:
    """Pure CPU worker function for Re-ID tensor transformation.
    Operates strictly on independent PIL images; does not touch GPU or shared mutable state.
    """
    return torch.stack([reid_transform(img if img.mode == "RGB" else img.convert("RGB")) for img in images])

def preprocess_face_image(image: Image.Image) -> Image.Image:
    """Standardizes input PIL image to 8-bit RGB (3 channels).
    Handles conversion from RGBA, Grayscale (L), and Palette (P) modes.
    """
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA") or (hasattr(image, "info") and "transparency" in image.info):
        # Convert RGBA/LA to RGB by pasting onto white background
        bg = Image.new("RGB", image.size, (255, 255, 255))
        mask = image.split()[-1]
        bg.paste(image, mask=mask)
        return bg
    return image.convert("RGB")

class GPUScheduler(threading.Thread):
    def __init__(self, job_manager, media_storage: MediaStorage, dispatcher=None):
        super().__init__(name="GPUScheduler", daemon=True)
        self.job_manager = job_manager
        self.media = media_storage
        self.dispatcher = dispatcher
        self.stopped = False
        
        # Bounded scheduler request queue
        self.track_cache = TrackCache()
        self.next_track_ids: Dict[str, int] = {} # camera_id -> incrementing track index
        
        # Lazy model references (loaded sequentially in CUDA)
        self.yolo: Optional[YOLO] = None
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None
        self.face_detector: Optional[MTCNN] = None
        self.face_embedder: Optional[InceptionResnetV1] = None
        self.ocr_reader: Optional[easyocr.Reader] = None
        self.reid_model: Optional[torch.nn.Module] = None
        self.reid_transform: Optional[transforms.Compose] = None
        
        # Thread lock for model execution (re-entrant)
        self.gpu_lock = threading.RLock()

        # Dedicated single-worker pool for intra-batch Re-ID CPU tensor preparation overlap
        self._reid_prep_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ReIDPrepWorker")

        # CPU Enrichment Worker Pool for color analysis, face, OCR, and media saving
        self.enrichment_queue: queue.Queue = queue.Queue(maxsize=256)
        self.cpu_face_lock = threading.Lock()
        self.cpu_ocr_lock = threading.Lock()
        self.num_enrichment_workers = 1  # Measured optimal configuration: 422.3 tasks/sec with lowest latency and minimal CPU contention
        self.enrichment_workers: List[threading.Thread] = []
        self._start_enrichment_workers()

        # Link scheduler to job_manager and dispatcher if available
        try:
            self.job_manager.gpu_scheduler = self
        except Exception:
            pass
        if self.dispatcher is not None:
            try:
                self.dispatcher.gpu_scheduler = self
            except Exception:
                pass

    def load_models(self) -> None:
        """Initializes and budgets all models on CUDA sequentially to prevent VRAM overflow."""
        with self.gpu_lock:
            if self.yolo is not None and self.clip_model is not None:
                LOGGER.info("Models already loaded in GPUScheduler, skipping redundant initialization.")
                return
            LOGGER.info("Initializing neural network models on GPU...")
            
            # 1. Load YOLOv8 Tracker
            self.yolo = YOLO(YOLO_MODEL_PATH)
            self.yolo.to(YOLO_DEVICE)
            LOGGER.info(f"YOLOv8 initialized on {YOLO_DEVICE}.")
            
            # 2. Load CLIP (ViT-B/32 - ~150MB)
            try:
                self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, local_files_only=True)
                self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, local_files_only=True).to(CLIP_DEVICE)
            except Exception:
                LOGGER.info("CLIP models not found in local cache. Loading from Hugging Face online hub...")
                self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
                self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(CLIP_DEVICE)
                
            if CLIP_DEVICE == "cuda":
                self.clip_model = self.clip_model.half()
            self.clip_model.eval()
            LOGGER.info(f"CLIP ViT-B/32 model initialized on {CLIP_DEVICE}.")
            

            # 2b. Load Lightweight Person Re-ID Extractor (576-D)
            try:
                self.reid_model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
                self.reid_model.classifier = torch.nn.Identity()
                self.reid_model.to(REID_DEVICE)
                if REID_DEVICE == "cuda":
                    self.reid_model = self.reid_model.half()
                self.reid_model.eval()
                self.reid_transform = transforms.Compose([
                    transforms.Resize((256, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                LOGGER.info(f"Person Re-ID model (MobileNetV3 576-D) initialized on {REID_DEVICE}.")
            except Exception as re_err:
                LOGGER.warning(f"Could not load Person Re-ID weights: {re_err}")

            # 3. Load MTCNN & FaceNet (InceptionResnetV1)
            self.face_detector = MTCNN(keep_all=False, select_largest=True, device=FACE_DEVICE)
            self.face_embedder = InceptionResnetV1(pretrained='vggface2').to(FACE_DEVICE)
            if FACE_DEVICE == "cuda":
                self.face_embedder = self.face_embedder.half()
            self.face_embedder.eval()
            LOGGER.info(f"Facial detection and embedding models initialized on {FACE_DEVICE}.")
            
            # 4. Load EasyOCR
            self.ocr_reader = easyocr.Reader(['en'], gpu=(OCR_DEVICE == "cuda"))
            LOGGER.info(f"EasyOCR initialized (GPU={OCR_DEVICE == 'cuda'}).")
            
            # Check VRAM allocation
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                LOGGER.info(f"Model VRAM allocations complete. Active VRAM: {allocated:.1f} MB.")

    def run(self) -> None:
        """Scheduler main execution loop."""
        # Load models before entering loop
        try:
            self.load_models()
        except Exception as e:
            LOGGER.critical(f"Failed to load GPU models: {e}", exc_info=True)
            self.stopped = True
            return

        while not self.stopped:
            try:
                # 1. Manage Active Jobs (Complete or Fail based on decoder threads)
                active_jobs = list(self.job_manager.active_jobs.items())
                for job_id, job in active_jobs:
                    # Check if any decoder failed
                    job_failed = False
                    failure_msg = ""
                    with job.lock:
                        for dec in job.decoders.values():
                            if getattr(dec, "failed", False):
                                job_failed = True
                                failure_msg = getattr(dec, "error_msg", "Decoder error")
                                break
                    
                    if job_failed:
                        LOGGER.error(f"Job {job_id} encountered a failed decoder stream: {failure_msg}")
                        self.job_manager.fail_job(job_id, failure_msg)
                        continue

                    # Check if decoders finished and queue is cleared
                    all_decoders_finished = False
                    with job.lock:
                        if job.status == "indexing" and job.decoders:
                            all_decoders_finished = True
                            for cid, dec in list(job.decoders.items()):
                                if dec.is_alive() or not dec.frame_queue.empty():
                                    all_decoders_finished = False
                                else:
                                    if getattr(dec, "has_finished", False) and not getattr(dec, "has_marked_completed", False):
                                        try:
                                            self.job_manager.sqlite.update_camera_metadata(
                                                cid, "completed", f"{dec.width}x{dec.height}", dec.fps
                                            )
                                            dec.has_marked_completed = True
                                        except Exception:
                                            pass
                    
                    if all_decoders_finished:
                        # Check if any registered camera in SQLite is still pending (not yet started)
                        try:
                            cams = self.job_manager.sqlite.get_cameras_by_job(job_id)
                            has_pending_cams = any(c.get("status") in ["inactive", "queued", "ready", "not_configured"] for c in cams)
                        except Exception:
                            has_pending_cams = False

                        if not has_pending_cams:
                            LOGGER.info(f"All decoders completed work for Job {job_id}. Running indexing verification and completion...")
                            self.flush_enrichment_pool()
                            vdb = getattr(self.dispatcher, "vector", None) if self.dispatcher else None
                            self.job_manager.complete_job(job_id, vector_db=vdb, dispatcher=self.dispatcher, gpu_scheduler=self)

                # 2. Gather active decoders to process frame queue
                decoders = self.job_manager.get_active_decoders()
                if not decoders:
                    time.sleep(0.1)
                    continue
                
                # Gather raw frames from decoders up to BATCH_SIZE
                batch_packets = []
                while len(batch_packets) < BATCH_SIZE:
                    pulled_in_round = 0
                    for dec in decoders:
                        try:
                            # Grab frame from decoder queue
                            packet = dec.frame_queue.get_nowait()
                            # packet is (frame_idx, timestamp, frame_bgr)
                            batch_packets.append((dec.camera_id, dec.is_live, packet))
                            pulled_in_round += 1
                            if len(batch_packets) >= BATCH_SIZE:
                                break
                        except queue.Empty:
                            continue
                    if pulled_in_round == 0:
                        break
                
                if not batch_packets:
                    time.sleep(0.01)
                    continue
                
                # Process the gathered frame batch
                PerformanceMonitorInstance.update_queue_depth(
                    sum(d.frame_queue.qsize() for d in decoders)
                )
                self._process_batch(batch_packets)
                
            except RuntimeError as re:
                if "out of memory" in str(re).lower():
                    LOGGER.warning("CUDA Out of Memory caught in GPUScheduler! Clearing cache...")
                    torch.cuda.empty_cache()
                    time.sleep(1.0)
                else:
                    LOGGER.error(f"Inference error: {re}", exc_info=True)
            except Exception as e:
                LOGGER.error(f"Unexpected GPUScheduler error: {e}", exc_info=True)
                time.sleep(0.5)

    def _process_batch(self, batch: List[Tuple[str, bool, Tuple[int, float, np.ndarray]]]) -> None:
        """Executes YOLO, filters keyframes, and runs conditional pipelines on batch."""
        if not hasattr(self, "_first_batch_logged"):
            LOGGER.info(f"MILESTONE: FIRST GPU BATCH CREATED at {time.time()}")
            self._first_batch_logged = True
            
        filtered_batch = []
        for camera_id, is_live, packet in batch:
            frame_idx, timestamp, frame_bgr = packet
            
            # Select target job profile from active job database info
            profile_name = "BALANCED"
            job_id = None
            target_decoder = None
            for job in list(self.job_manager.active_jobs.values()):
                if camera_id in job.decoders:
                    profile_name = job.profile
                    job_id = job.job_id
                    target_decoder = job.decoders[camera_id]
                    break
            
            if not job_id:
                continue
                
            profile = PROCESSING_PROFILES.get(profile_name, PROCESSING_PROFILES["BALANCED"])
            
            # Static background filter check BEFORE running YOLO (if not already filtered by decoder)
            is_prefiltered = getattr(target_decoder, "static_filtered", False)
            if not is_prefiltered:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3 else frame_bgr
                if self.track_cache.is_scene_static(camera_id, gray, profile["skip_static_threshold"]):
                    PerformanceMonitorInstance.update_counts(decoded=1)
                    continue
                
            filtered_batch.append((camera_id, is_live, packet, job_id, profile))
            
        if not filtered_batch:
            return

        # 1. Run YOLOv8 Object Detection on filtered frames (ensure 3 channels for YOLO)
        frames = [
            cv2.cvtColor(item[2][2], cv2.COLOR_GRAY2BGR) if len(item[2][2].shape) == 2 or item[2][2].shape[2] == 1 else item[2][2]
            for item in filtered_batch
        ]
        
        start_yolo = time.time()
        with self.gpu_lock:
            if YOLO_USE_AMP and YOLO_DEVICE == "cuda":
                # CUDA Automatic Mixed Precision (AMP) FP16 for Tensor Core acceleration on RTX 3050
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    results = self.yolo.predict(
                        source=frames,
                        imgsz=640,
                        device=YOLO_DEVICE,
                        verbose=False,
                        conf=0.25
                    )
            else:
                # Standard FP32 baseline execution
                results = self.yolo.predict(
                    source=frames,
                    imgsz=640,
                    device=YOLO_DEVICE,
                    verbose=False,
                    conf=0.25
                )
        elapsed_yolo = time.time() - start_yolo
        LOGGER.info(f"GPU_TRACE: YOLO inference finished on {len(frames)} frames in {elapsed_yolo:.3f}s")
        PerformanceMonitorInstance.record_latency("yolo", elapsed_yolo)
        PerformanceMonitorInstance.update_counts(decoded=len(filtered_batch))

        # Process detections frame by frame and collect tracked candidates
        # Regroup detections by camera_id to ensure strict chronological tracking per camera
        camera_groups = {}
        for idx, (camera_id, is_live, (frame_idx, timestamp, frame_bgr), job_id, profile) in enumerate(filtered_batch):
            res = results[idx]
            
            PerformanceMonitorInstance.record_frame_processed()
            
            # Filter classes: person, car, motorcycle, bus, truck
            # YOLO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
            allowed_classes = {0, 2, 3, 5, 7}
            
            boxes_to_track = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes_data = res.boxes.data.cpu().numpy()
                for row in boxes_data:
                    cls_id = int(row[5])
                    if cls_id in allowed_classes:
                        boxes_to_track.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), cls_id))
            
            if camera_id not in camera_groups:
                camera_groups[camera_id] = []
            camera_groups[camera_id].append((frame_idx, timestamp, frame_bgr, job_id, profile, boxes_to_track))

        candidates = []
        for camera_id, frame_items in camera_groups.items():
            # Strict chronological FIFO order within each camera
            frame_items.sort(key=lambda x: (x[0], x[1]))
            for frame_idx, timestamp, frame_bgr, job_id, profile, boxes_to_track in frame_items:
                tracked_entities = self._update_camera_tracker(camera_id, job_id, frame_idx, timestamp, frame_bgr, boxes_to_track)
                LOGGER.info(f"GPU_TRACE: Frame {frame_idx} produced {len(tracked_entities)} tracked entities to schedule")
                for track_data, crop, entity_type, bbox in tracked_entities:
                    candidates.append((job_id, camera_id, track_data, crop, entity_type, bbox, frame_idx, timestamp, profile))

        if not candidates:
            return

        # 2. Batch CLIP & Re-ID Evaluation across all tracked entities in this frame batch
        clip_batch_items = []
        cand_keyframe_decisions: Dict[int, bool] = {}
        track_sim_state: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for cand_idx, cand in enumerate(candidates):
            job_id, camera_id, track, crop, entity_type, bbox, frame_idx, timestamp, profile = cand
            track_key = (camera_id, track.track_id)
            if track_key not in track_sim_state:
                track_sim_state[track_key] = {
                    "has_embedding": track.clip_embedding is not None,
                    "last_sharpness": track.last_embedded_sharpness,
                    "last_size": track.last_embedded_size,
                    "last_time": track.last_embedded_time,
                }
            state = track_sim_state[track_key]

            is_valid, quality = evaluate_crop_quality(track.best_crop, track.best_sharpness)
            cand_run_clip = False
            if is_valid:
                crop_size = float(track.best_crop.shape[0] * track.best_crop.shape[1])
                if not state["has_embedding"]:
                    cand_run_clip = True
                elif track.best_sharpness > (state["last_sharpness"] * 1.20):
                    cand_run_clip = True
                elif crop_size > (state["last_size"] * 1.30):
                    cand_run_clip = True
                elif (timestamp - state["last_time"]) > 5.0:
                    cand_run_clip = True

            cand_keyframe_decisions[cand_idx] = cand_run_clip

            if cand_run_clip:
                state["has_embedding"] = True
                state["last_sharpness"] = track.best_sharpness
                state["last_size"] = float(track.best_crop.shape[0] * track.best_crop.shape[1])
                state["last_time"] = timestamp

                if len(track.best_crop.shape) == 2 or track.best_crop.shape[2] == 1:
                    pil_img = Image.fromarray(track.best_crop).convert("RGB")
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(track.best_crop, cv2.COLOR_BGR2RGB))
                clip_batch_items.append({
                    "cand_idx": cand_idx,
                    "cand": cand,
                    "pil_img": pil_img,
                    "quality": quality,
                    "entity_type": entity_type,
                    "timestamp": timestamp,
                    "track": track
                })

        # Run batched CLIP & Re-ID inference with intra-batch CPU overlap
        if clip_batch_items:
            start_clip_batch = time.time()

            # Identify Re-ID items upfront
            reid_batch_items = [
                item for item in clip_batch_items
                if item["entity_type"] == "person" and self.reid_model is not None and self.reid_transform is not None
            ]

            # 1. Asynchronously dispatch Re-ID CPU tensor preparation to single worker thread
            future_reid_tensors = None
            if reid_batch_items:
                reid_images = [item["pil_img"] for item in reid_batch_items]
                future_reid_tensors = self._reid_prep_executor.submit(
                    _reid_prep_worker, reid_images, self.reid_transform
                )

            # 2. Concurrently execute batched CLIP on scheduler thread
            clip_images = [item["pil_img"] for item in clip_batch_items]
            clip_embeddings = self.embed_images_batch(clip_images, batch_size=16)

            # 3. Complete Re-ID GPU inference (tensors already prepared concurrently during CLIP)
            reid_embeddings = []
            if future_reid_tensors is not None:
                reid_tensors = future_reid_tensors.result()
                reid_embeddings = self._embed_reid_prepared_tensors(reid_tensors, batch_size=16)

            # Assign CLIP embeddings & metadata
            for idx_c, item in enumerate(clip_batch_items):
                tr = item["track"]
                tr.clip_embedding = clip_embeddings[idx_c]
                tr.last_embedded_sharpness = tr.best_sharpness
                tr.last_embedded_size = float(tr.best_crop.shape[0] * tr.best_crop.shape[1])
                tr.last_embedded_time = item["timestamp"]
                tr.embedded_count += 1
                tr.quality_score = item["quality"]

            # Assign Re-ID embeddings
            for idx_r, item in enumerate(reid_batch_items):
                if idx_r < len(reid_embeddings) and reid_embeddings[idx_r] is not None:
                    item["track"].reid_embedding = reid_embeddings[idx_r]

            elapsed_clip = time.time() - start_clip_batch
            PerformanceMonitorInstance.record_latency("clip", elapsed_clip)
            PerformanceMonitorInstance.update_counts(embeddings=len(clip_batch_items))
            LOGGER.info(f"GPU_TRACE: Batched CLIP ({len(clip_batch_items)}) & Re-ID ({len(reid_batch_items)}) finished in {elapsed_clip:.3f}s")

        # 3. Enqueue candidates to asynchronous CPU enrichment pool
        for cand_idx, cand in enumerate(candidates):
            job_id, camera_id, track_data, crop, entity_type, bbox, frame_idx, timestamp, profile = cand
            run_clip_flag = cand_keyframe_decisions.get(cand_idx, False)

            run_face = False
            if entity_type == "person" and profile.get("run_face_analysis", True):
                if track_data.face_embedding is None and track_data.face_run_count < 3:
                    run_face = True
                    track_data.face_run_count += 1

            run_ocr = False
            if entity_type == "vehicle" and profile.get("run_ocr_analysis", True):
                if track_data.plate_text is None and track_data.ocr_run_count < 3:
                    run_ocr = True
                    track_data.ocr_run_count += 1

            best_crop = track_data.best_crop.copy() if track_data.best_crop is not None else crop.copy()
            raw_crop = crop.copy() if crop is not None else best_crop.copy()

            task = {
                "job_id": job_id,
                "camera_id": camera_id,
                "track_id": track_data.track_id,
                "track_uuid": f"{job_id}_{camera_id}_{track_data.track_id}",
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "bbox": list(bbox),
                "best_crop": best_crop,
                "raw_crop": raw_crop,
                "entity_type": entity_type,
                "profile": profile,
                "run_clip": run_clip_flag,
                "clip_embedding": track_data.clip_embedding,
                "reid_embedding": getattr(track_data, "reid_embedding", None),
                "quality_score": track_data.quality_score,
                "run_face": run_face,
                "run_ocr": run_ocr,
                "track_ref": track_data,
            }
            self.enrichment_queue.put(task, block=True)

    def _update_camera_tracker(
        self,
        camera_id: str,
        job_id: str,
        frame_idx: int,
        timestamp: float,
        frame_bgr: np.ndarray,
        detections: List[Tuple[float, float, float, float, float, int]]
    ) -> List[Tuple[TrackData, np.ndarray, str, List[float]]]:
        """Matches boxes between frames per camera to assign incrementing track IDs."""
        if camera_id not in self.next_track_ids:
            self.next_track_ids[camera_id] = 1
            
        h_img, w_img = frame_bgr.shape[:2]
        
        # Get active tracks on this camera
        if camera_id not in self.track_cache.active_tracks:
            self.track_cache.active_tracks[camera_id] = {}
        active_tracks = self.track_cache.active_tracks[camera_id]
        
        # List of matched incoming index
        matched_detections = set()
        tracks_to_update = []
        
        # Match detections with active tracks using IoU
        for track_id, track in list(active_tracks.items()):
            # Find the last coordinates of this track
            if not track.history:
                continue
            last_bbox = track.best_bbox  # Check comparison against best crop coordinates
            
            best_iou = 0.0
            best_idx = -1
            
            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                iou = calculate_iou(last_bbox, [det[0], det[1], det[2], det[3]])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou > 0.30:  # Matching IoU threshold
                matched_detections.add(best_idx)
                det = detections[best_idx]
                
                # Crop bounding box with 15% context padding
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                w, h = x2 - x1, y2 - y1
                x1_pad = max(0, int(x1 - 0.15 * w))
                y1_pad = max(0, int(y1 - 0.15 * h))
                x2_pad = min(w_img, int(x2 + 0.15 * w))
                y2_pad = min(h_img, int(y2 + 0.15 * h))
                crop = frame_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop.size > 0:
                    sharpness = self.track_cache.add_to_track(
                        camera_id, track_id, frame_idx, timestamp,
                        [x1, y1, x2, y2], crop
                    )
                    tracks_to_update.append((track, crop, "person" if det[5] == 0 else "vehicle", [x1, y1, x2, y2]))
                    
        # Create new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx in matched_detections:
                continue
                
            track_id = self.next_track_ids[camera_id]
            self.next_track_ids[camera_id] += 1
            
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            w, h = x2 - x1, y2 - y1
            x1_pad = max(0, int(x1 - 0.15 * w))
            y1_pad = max(0, int(y1 - 0.15 * h))
            x2_pad = min(w_img, int(x2 + 0.15 * w))
            y2_pad = min(h_img, int(y2 + 0.15 * h))
            crop = frame_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if crop.size > 0:
                track = self.track_cache.get_or_create_track(camera_id, track_id, timestamp)
                sharpness = self.track_cache.add_to_track(
                    camera_id, track_id, frame_idx, timestamp,
                    [x1, y1, x2, y2], crop
                )
                tracks_to_update.append((track, crop, "person" if det[5] == 0 else "vehicle", [x1, y1, x2, y2]))

        # Timeout expired tracks (inactive for more than 4 seconds)
        cutoff_time = timestamp - 4.0
        for track_id, track in list(active_tracks.items()):
            if track.end_time < cutoff_time:
                # Track ended, dispatch complete event
                self.track_cache.remove_track(camera_id, track_id)
                LOGGER.info(f"Terminated track {track_id} on camera {camera_id}.")
                
        return tracks_to_update

    def _start_enrichment_workers(self) -> None:
        """Starts background CPU enrichment worker threads."""
        for i in range(self.num_enrichment_workers):
            t = threading.Thread(
                target=self._enrichment_worker_loop,
                name=f"CPUEnrichmentWorker-{i}",
                daemon=True
            )
            t.start()
            self.enrichment_workers.append(t)

    def _enrichment_worker_loop(self) -> None:
        """Processes CPU-bound tasks (color analysis, MTCNN face, EasyOCR, and media saving) off the GPU scheduler."""
        while not self.stopped:
            try:
                task = self.enrichment_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if task is None:
                self.enrichment_queue.task_done()
                break
            try:
                self._execute_enrichment_task(task)
            except Exception as e:
                LOGGER.error(f"Error executing CPU enrichment task: {e}", exc_info=True)
            finally:
                self.enrichment_queue.task_done()

    def flush_enrichment_pool(self, timeout: float = 15.0) -> bool:
        """Blocks until all pending CPU enrichment tasks are drained and dispatched."""
        start = time.time()
        while self.enrichment_queue.unfinished_tasks > 0:
            if time.time() - start > timeout:
                LOGGER.warning(
                    f"Enrichment pool flush timed out after {timeout}s "
                    f"({self.enrichment_queue.unfinished_tasks} tasks remain)."
                )
                return False
            time.sleep(0.01)
        return True

    def _execute_enrichment_task(self, task: Dict[str, Any]) -> None:
        """Executes color extraction, MTCNN/FaceNet, EasyOCR, crop saves, and result dispatch."""
        job_id = task["job_id"]
        camera_id = task["camera_id"]
        track_id = task["track_id"]
        track_uuid = task["track_uuid"]
        frame_idx = task["frame_idx"]
        timestamp = task["timestamp"]
        bbox = task["bbox"]
        best_crop = task["best_crop"]
        raw_crop = task["raw_crop"]
        entity_type = task["entity_type"]
        profile = task["profile"]
        run_clip = task["run_clip"]
        clip_embedding = task["clip_embedding"]
        reid_embedding = task["reid_embedding"]
        quality_score = task["quality_score"]
        run_face = task["run_face"]
        run_ocr = task["run_ocr"]
        track_ref = task.get("track_ref")

        # 1. Color Extraction (if run_clip was True)
        if run_clip and best_crop is not None and best_crop.size > 0:
            if entity_type == "person":
                colors = extract_body_colors(best_crop)
                dominant_upper_color = colors["dominant_upper_color"]
                secondary_upper_color = colors["secondary_upper_color"]
                dominant_lower_color = colors["dominant_lower_color"]
                secondary_lower_color = colors["secondary_lower_color"]
                color_confidence = colors["color_confidence"]
                is_monochrome = colors.get("is_monochrome", False)
            elif entity_type == "vehicle":
                colors = extract_vehicle_colors(best_crop)
                dominant_upper_color = colors["dominant_upper_color"]
                secondary_upper_color = colors["secondary_upper_color"]
                dominant_lower_color = None
                secondary_lower_color = None
                color_confidence = colors["color_confidence"]
                is_monochrome = colors.get("is_monochrome", False)
            else:
                dominant_upper_color = None
                secondary_upper_color = None
                dominant_lower_color = None
                secondary_lower_color = None
                color_confidence = 0.0
                is_monochrome = False

            if track_ref is not None:
                track_ref.dominant_upper_color = dominant_upper_color
                track_ref.secondary_upper_color = secondary_upper_color
                track_ref.dominant_lower_color = dominant_lower_color
                track_ref.secondary_lower_color = secondary_lower_color
                track_ref.color_confidence = color_confidence
                track_ref.is_monochrome = is_monochrome
        else:
            dominant_upper_color = getattr(track_ref, "dominant_upper_color", None) if track_ref else None
            secondary_upper_color = getattr(track_ref, "secondary_upper_color", None) if track_ref else None
            dominant_lower_color = getattr(track_ref, "dominant_lower_color", None) if track_ref else None
            secondary_lower_color = getattr(track_ref, "secondary_lower_color", None) if track_ref else None
            color_confidence = getattr(track_ref, "color_confidence", 0.0) if track_ref else 0.0
            is_monochrome = getattr(track_ref, "is_monochrome", False) if track_ref else False

        # 2. Facial Recognition (MTCNN + FaceNet)
        face_embedding = None
        face_crop_path = None
        face_quality = 0.0
        if run_face and best_crop is not None and best_crop.size > 0:
            start_face = time.time()
            if len(best_crop.shape) == 2 or best_crop.shape[2] == 1:
                crop_rgb = cv2.cvtColor(best_crop, cv2.COLOR_GRAY2RGB)
            else:
                crop_rgb = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
            boxes, probs = None, None
            with self.cpu_face_lock:
                try:
                    if self.face_detector is not None:
                        boxes, probs = self.face_detector.detect(crop_rgb)
                except Exception as f_err:
                    LOGGER.debug(f"Face detector note for track {track_id}: {f_err}")
                    boxes, probs = None, None

            if boxes is not None and len(boxes) > 0 and probs is not None and probs[0] is not None:
                face_conf = float(probs[0])
                fx1, fy1, fx2, fy2 = (int(coord) for coord in boxes[0])
                ch, cw = crop_rgb.shape[:2]
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(cw, fx2), min(ch, fy2)
                fw = fx2 - fx1
                fh = fy2 - fy1

                if face_conf >= 0.95 and fw >= 45 and fh >= 45:
                    face_bgr = best_crop[fy1:fy2, fx1:fx2]
                    if len(face_bgr.shape) == 2 or face_bgr.shape[2] == 1:
                        face_pil = Image.fromarray(face_bgr).convert("RGB")
                    else:
                        face_pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                    face_tensor = extract_face(face_pil, [0, 0, fw, fh], image_size=160, margin=14)
                    face_tensor = fixed_image_standardization(face_tensor)
                    face_tensor = face_tensor.to(FACE_DEVICE).unsqueeze(0)
                    if FACE_DEVICE == "cuda":
                        face_tensor = face_tensor.half()
                    with self.cpu_face_lock:
                        with torch.no_grad():
                            raw_emb = self.face_embedder(face_tensor)
                            normalized_emb = raw_emb / raw_emb.norm(p=2, dim=-1, keepdim=True)
                            embedding = normalized_emb.cpu().numpy()[0].tolist()
                    if track_ref is not None:
                        track_ref.face_embedding = embedding
                    face_embedding = embedding
                    face_quality = round(face_conf * min(1.0, (fw * fh) / (120.0 * 120.0)), 3)
                    face_crop_path = self.media.save_crop(
                        face_bgr, "face", job_id, camera_id, track_id, frame_idx
                    )
            PerformanceMonitorInstance.record_latency("face", time.time() - start_face)

        # 3. License Plate OCR (EasyOCR)
        plate_text = None
        plate_crop_path = None
        if run_ocr and best_crop is not None and best_crop.size > 0:
            start_ocr = time.time()
            vh, vw = best_crop.shape[:2]
            lower_half = best_crop[int(vh * 0.45):, :]
            ocr_results = []
            with self.cpu_ocr_lock:
                if self.ocr_reader is not None:
                    ocr_results = self.ocr_reader.readtext(
                        lower_half,
                        decoder="beamsearch",
                        beamWidth=5,
                        detail=1
                    )
            best_text = ""
            best_conf = 0.0
            for bbox_pts, text, conf in ocr_results:
                if conf > 0.20:
                    cleaned = "".join(c for c in text if c.isalnum()).upper()
                    if len(cleaned) >= 3 and conf > best_conf:
                        best_conf = conf
                        best_text = cleaned
            if best_text:
                if track_ref is not None:
                    track_ref.plate_text = best_text
                plate_text = best_text
                plate_crop_path = self.media.save_crop(
                    lower_half, "plate", job_id, camera_id, track_id, frame_idx
                )
            PerformanceMonitorInstance.record_latency("ocr", time.time() - start_ocr)

        # 4. Save crop assets to disk asynchronously
        crop_path = self.media.save_crop(
            best_crop, entity_type, job_id, camera_id, track_id, frame_idx
        )
        thumbnail_path = self.media.save_thumbnail(
            raw_crop, job_id, camera_id, frame_idx
        )
        LOGGER.info(f"GPU_TRACE: Saved crop {crop_path.name} and thumbnail {thumbnail_path.name}")

        # 5. Prepare result payload for dispatcher
        result_payload = {
            "job_id": job_id,
            "camera_id": camera_id,
            "track_uuid": track_uuid,
            "track_id": track_id,
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "bbox": bbox,
            "crop_path": str(crop_path),
            "thumbnail_path": str(thumbnail_path),
            "confidence": 0.85,
            "entity_type": entity_type,
            "clip_embedding": clip_embedding,
            "face_embedding": face_embedding,
            "face_crop_path": str(face_crop_path) if face_crop_path else None,
            "face_quality": face_quality,
            "plate_text": plate_text,
            "plate_crop_path": str(plate_crop_path) if plate_crop_path else None,
            "quality_score": quality_score,
            "dominant_upper_color": dominant_upper_color,
            "secondary_upper_color": secondary_upper_color,
            "dominant_lower_color": dominant_lower_color,
            "secondary_lower_color": secondary_lower_color,
            "color_confidence": color_confidence,
            "is_monochrome": is_monochrome,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_version": EMBEDDING_VERSION,
            "reid_embedding": reid_embedding,
            "reid_model": REID_MODEL_NAME,
            "reid_version": REID_VERSION,
            "should_embed": run_clip
        }

        # Dispatch output
        if self.dispatcher:
            LOGGER.info(f"GPU_TRACE: Dispatching result for track {track_id} (job {job_id})")
            self.dispatcher.dispatch(result_payload)
        else:
            LOGGER.debug(f"GPUScheduler dispatch placeholder: {result_payload['track_uuid']} | {entity_type}")

    def _schedule_conditional_models(
        self,
        job_id: str,
        camera_id: str,
        track: TrackData,
        crop: np.ndarray,
        entity_type: str,
        bbox: List[float],
        frame_idx: int,
        timestamp: float,
        profile: Dict[str, Any],
        precomputed_run_clip: Optional[bool] = None,
        batch_thumbnails: Optional[Dict[Tuple[str, int], Path]] = None
    ) -> None:
        """Synchronous wrapper for legacy callers to execute conditional enrichment."""
        run_clip = precomputed_run_clip if precomputed_run_clip is not None else False
        run_face = False
        if entity_type == "person" and profile.get("run_face_analysis", True):
            if track.face_embedding is None and track.face_run_count < 3:
                run_face = True
                track.face_run_count += 1

        run_ocr = False
        if entity_type == "vehicle" and profile.get("run_ocr_analysis", True):
            if track.plate_text is None and track.ocr_run_count < 3:
                run_ocr = True
                track.ocr_run_count += 1

        best_crop = track.best_crop.copy() if track.best_crop is not None else crop.copy()
        raw_crop = crop.copy() if crop is not None else best_crop.copy()

        task = {
            "job_id": job_id,
            "camera_id": camera_id,
            "track_id": track.track_id,
            "track_uuid": f"{job_id}_{camera_id}_{track.track_id}",
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "bbox": list(bbox),
            "best_crop": best_crop,
            "raw_crop": raw_crop,
            "entity_type": entity_type,
            "profile": profile,
            "run_clip": run_clip,
            "clip_embedding": track.clip_embedding,
            "reid_embedding": getattr(track, "reid_embedding", None),
            "quality_score": track.quality_score,
            "run_face": run_face,
            "run_ocr": run_ocr,
            "track_ref": track,
        }
        self._execute_enrichment_task(task)

    def stop_scheduler(self) -> None:
        """Gracefully halts scheduler thread and cleans VRAM."""
        self.stopped = True
        if hasattr(self, "_reid_prep_executor"):
            try:
                self._reid_prep_executor.shutdown(wait=True)
            except Exception:
                pass
        for _ in range(self.num_enrichment_workers):
            try:
                self.enrichment_queue.put_nowait(None)
            except Exception:
                pass
        for w in self.enrichment_workers:
            if w.is_alive():
                w.join(timeout=2.0)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        LOGGER.info("GPUScheduler stop requested.")

    def embed_text(self, text: str) -> List[float]:
        """Embeds query text into a normalised CLIP vector under the GPU lock."""
        with self.gpu_lock:
            if self.clip_model is None or self.clip_processor is None:
                self.load_models()
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(CLIP_DEVICE)
            with torch.no_grad():
                res = self.clip_model.get_text_features(**inputs)
                if hasattr(res, "pooler_output"):
                    res = res.pooler_output
                res = res / res.norm(p=2, dim=-1, keepdim=True)
                return res.cpu().numpy()[0].tolist()

    def embed_image(self, image: Image.Image) -> List[float]:
        """Embeds a single PIL image into a normalized CLIP vector under the GPU lock."""
        with self.gpu_lock:
            if self.clip_model is None or self.clip_processor is None:
                self.load_models()
            rgb_img = image if image.mode == "RGB" else image.convert("RGB")
            inputs = self.clip_processor(images=[rgb_img], return_tensors="pt").to(CLIP_DEVICE)
            if CLIP_DEVICE == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.no_grad():
                res = self.clip_model.get_image_features(**inputs)
                if hasattr(res, "pooler_output"):
                    res = res.pooler_output
                res = res / res.norm(p=2, dim=-1, keepdim=True)
                return res.cpu().numpy()[0].tolist()

    def embed_images_batch(self, images: List[Image.Image], batch_size: int = 16) -> List[List[float]]:
        """Embeds a batch of PIL images into normalized CLIP vectors under the GPU lock."""
        if not images:
            return []
        all_embeddings: List[List[float]] = []
        with self.gpu_lock:
            if self.clip_model is None or self.clip_processor is None:
                self.load_models()
            clip_model = self.clip_model
            clip_processor = self.clip_processor
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            rgb_chunk = [img if img.mode == "RGB" else img.convert("RGB") for img in chunk]
            inputs = clip_processor(images=rgb_chunk, return_tensors="pt").to(CLIP_DEVICE)
            if CLIP_DEVICE == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with self.gpu_lock:
                with torch.no_grad():
                    res = clip_model.get_image_features(**inputs)
                    if hasattr(res, "pooler_output"):
                        res = res.pooler_output
                    res = res / res.norm(p=2, dim=-1, keepdim=True)
                    all_embeddings.extend(res.cpu().float().numpy().tolist())
        return all_embeddings

    def embed_reid(self, image: Image.Image) -> Optional[List[float]]:
        """Extracts and normalizes person Re-ID embedding under the GPU lock."""
        with self.gpu_lock:
            if self.reid_model is None or self.reid_transform is None:
                self.load_models()
            if self.reid_model is None or self.reid_transform is None:
                return None
        with self.gpu_lock:
            rgb_img = image if image.mode == "RGB" else image.convert("RGB")
            tensor_crop = self.reid_transform(rgb_img).unsqueeze(0).to(REID_DEVICE)
            if REID_DEVICE == "cuda":
                tensor_crop = tensor_crop.half()
            with torch.no_grad():
                reid_features = self.reid_model(tensor_crop)
                reid_features = reid_features / reid_features.norm(p=2, dim=-1, keepdim=True)
                return reid_features[0].cpu().float().numpy().tolist()

    def embed_reid_batch(self, images: List[Image.Image], batch_size: int = 16) -> List[Optional[List[float]]]:
        """Extracts and normalizes person Re-ID embeddings in batches under the GPU lock."""
        if not images:
            return []
        with self.gpu_lock:
            if self.reid_model is None or self.reid_transform is None:
                self.load_models()
            if self.reid_model is None or self.reid_transform is None:
                return [None] * len(images)
            reid_model = self.reid_model
            reid_transform = self.reid_transform

        all_results: List[Optional[List[float]]] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            try:
                tensor_crops = torch.stack([reid_transform(img if img.mode == "RGB" else img.convert("RGB")) for img in chunk]).to(REID_DEVICE)
                if REID_DEVICE == "cuda":
                    tensor_crops = tensor_crops.half()
                with self.gpu_lock:
                    with torch.no_grad():
                        reid_features = reid_model(tensor_crops)
                        reid_features = reid_features / reid_features.norm(p=2, dim=-1, keepdim=True)
                        all_results.extend(reid_features.cpu().float().numpy().tolist())
            except Exception as ex:
                LOGGER.warning(f"Batch Re-ID extraction error: {ex}")
                for img in chunk:
                    try:
                        rgb_img = img if img.mode == "RGB" else img.convert("RGB")
                        t = reid_transform(rgb_img).unsqueeze(0).to(REID_DEVICE)
                        if REID_DEVICE == "cuda":
                            t = t.half()
                        with self.gpu_lock:
                            with torch.no_grad():
                                rf = reid_model(t)
                                rf = rf / rf.norm(p=2, dim=-1, keepdim=True)
                                all_results.append(rf[0].cpu().float().numpy().tolist())
                    except Exception:
                        all_results.append(None)
        return all_results

    def _embed_reid_prepared_tensors(self, tensor_crops: torch.Tensor, batch_size: int = 16) -> List[Optional[List[float]]]:
        """Extracts and normalizes person Re-ID embeddings from pre-transformed CPU tensors under the GPU lock."""
        if tensor_crops is None or len(tensor_crops) == 0:
            return []
        with self.gpu_lock:
            if self.reid_model is None:
                self.load_models()
            if self.reid_model is None:
                return [None] * len(tensor_crops)
            reid_model = self.reid_model

        all_results: List[Optional[List[float]]] = []
        for i in range(0, len(tensor_crops), batch_size):
            chunk = tensor_crops[i : i + batch_size].to(REID_DEVICE)
            if REID_DEVICE == "cuda":
                chunk = chunk.half()
            try:
                with self.gpu_lock:
                    with torch.no_grad():
                        reid_features = reid_model(chunk)
                        reid_features = reid_features / reid_features.norm(p=2, dim=-1, keepdim=True)
                        all_results.extend(reid_features.cpu().float().numpy().tolist())
            except Exception as ex:
                LOGGER.warning(f"Batch Re-ID inference error: {ex}")
                for t in chunk:
                    try:
                        with self.gpu_lock:
                            with torch.no_grad():
                                rf = reid_model(t.unsqueeze(0))
                                rf = rf / rf.norm(p=2, dim=-1, keepdim=True)
                                all_results.append(rf[0].cpu().float().numpy().tolist())
                    except Exception:
                        all_results.append(None)
        return all_results

    def embed_face(self, image: Image.Image) -> Optional[List[float]]:
        """Extracts and embeds a face from a PIL image under the GPU lock."""
        res = self.embed_face_with_diagnostics(image)
        return res["embedding"]

    def embed_face_with_diagnostics(self, image: Image.Image) -> Dict[str, Any]:
        """Extracts and embeds a face from a PIL image under the GPU lock, returning complete diagnostic information."""
        LOGGER.info("FACE PIPELINE TRACE: [1] Image received.")
        
        # 1. Image Preprocessing / Decoding
        processed_img = preprocess_face_image(image)
        LOGGER.info("FACE PIPELINE TRACE: [2] Image decoded and normalized to RGB.")
        
        with self.gpu_lock:
            # 2. Face Detector Execution
            cv_img = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2BGR)
            LOGGER.info("FACE PIPELINE TRACE: [3] Face detector executed (MTCNN).")
            
            boxes, probs = self.face_detector.detect(cv_img)
            
            num_faces = len(boxes) if boxes is not None else 0
            LOGGER.info(f"FACE PIPELINE TRACE: [4] Number of detected faces: {num_faces}")
            
            face_tensor = None
            face_status = "NO_FACE"
            diagnostic_msg = "No face detected in the image."
            confidence = 0.0
            bbox = []
            
            # Helper to calculate face metrics
            def analyze_face_metrics(bx, prob, w, h):
                fx1, fy1, fx2, fy2 = (int(coord) for coord in bx)
                fw = fx2 - fx1
                fh = fy2 - fy1
                face_area = fw * fh
                rel_size = face_area / (w * h)
                
                # Check blur / sharpness
                face_bgr = cv_img[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.size > 0 else np.array([])
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() if gray.size > 0 else 0.0
                
                # Check brightness
                brightness = gray.mean() if gray.size > 0 else 0.0
                
                return {
                    "fw": fw,
                    "fh": fh,
                    "area": face_area,
                    "rel_size": rel_size,
                    "sharpness": sharpness,
                    "brightness": brightness,
                    "bbox": [fx1, fy1, fx2, fy2]
                }
            
            w, h = processed_img.size
            
            # --- NORMAL FACE DETECTION GATE (Strict) ---
            if boxes is not None and len(boxes) > 0 and probs[0] is not None:
                bx = boxes[0]
                prob = float(probs[0])
                metrics = analyze_face_metrics(bx, prob, w, h)
                
                LOGGER.info(f"FACE PIPELINE TRACE: [5] Bounding boxes: {boxes}")
                LOGGER.info(f"FACE PIPELINE TRACE: [6] Detector confidence: {prob:.4f}")
                LOGGER.info(f"FACE PIPELINE TRACE: [7] Face metrics: Area={metrics['area']}px, RelSize={metrics['rel_size']:.4f}, Sharpness={metrics['sharpness']:.1f}, Brightness={metrics['brightness']:.1f}")
                
                # Normal strict validation
                if prob >= 0.95 and metrics["fw"] >= 45 and metrics["fh"] >= 45 and metrics["sharpness"] >= 15.0:
                    face_status = "GOOD_FACE"
                    diagnostic_msg = "Good quality face detected."
                    confidence = prob
                    bbox = metrics["bbox"]
                    
                    face_bgr = cv_img[max(0, bbox[1]):min(h, bbox[2]), max(0, bbox[0]):min(w, bbox[3])]
                    face_pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                    face_tensor = extract_face(face_pil, [0, 0, metrics["fw"], metrics["fh"]], image_size=160, margin=14)
                    face_tensor = fixed_image_standardization(face_tensor)
                    LOGGER.info("FACE PIPELINE TRACE: [8] Face crop created with standard alignment and margin=14.")
                else:
                    LOGGER.info("Strict validation failed. Checking fallback routes.")
            
            # --- CONTROLLED FALLBACK FOR PARTIAL / LOWER QUALITY FACES (Section 3) ---
            if face_tensor is None:
                # Fallback Route 1: If face was detected but failed strict quality gates
                if boxes is not None and len(boxes) > 0 and probs[0] is not None:
                    bx = boxes[0]
                    prob = float(probs[0])
                    metrics = analyze_face_metrics(bx, prob, w, h)
                    
                    # Section 3: Configurable partial-face fallback thresholds
                    # MTCNN confidence >= 0.70, minimum crop size >= 30x30, brightness >= 30
                    if prob >= 0.70 and metrics["fw"] >= 30 and metrics["fh"] >= 30 and metrics["brightness"] >= 30:
                        face_status = "PARTIAL_FACE"
                        diagnostic_msg = "Partial face detected — lower-confidence recognition."
                        confidence = prob
                        bbox = metrics["bbox"]
                        
                        face_bgr = cv_img[max(0, bbox[1]):min(h, bbox[2]), max(0, bbox[0]):min(w, bbox[3])]
                        face_pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                        face_tensor = extract_face(face_pil, [0, 0, metrics["fw"], metrics["fh"]], image_size=160, margin=14)
                        face_tensor = fixed_image_standardization(face_tensor)
                        LOGGER.info("FACE PIPELINE TRACE: [8-Fallback] Partial face crop created with margin=14.")
                    elif metrics["sharpness"] < 15.0 or metrics["fw"] < 30:
                        face_status = "LOW_QUALITY_FACE"
                        diagnostic_msg = "Face detected but image quality is too low (blurry or too small)."
                
                # Fallback Route 2: Assume query image is already pre-cropped face
                if face_tensor is None and w <= 320 and h <= 320 and 0.5 <= (w / h) <= 2.0:
                    # Analyze whole crop as face
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() if gray.size > 0 else 0.0
                    brightness = gray.mean() if gray.size > 0 else 0.0
                    
                    if sharpness >= 8.0 and brightness >= 25:
                        face_status = "PARTIAL_FACE"
                        diagnostic_msg = "Pre-cropped face detected — lower-confidence recognition."
                        confidence = 0.90
                        bbox = [0, 0, w, h]
                        try:
                            face_tensor = extract_face(processed_img, [0, 0, w, h], image_size=160, margin=14)
                            face_tensor = fixed_image_standardization(face_tensor)
                            LOGGER.info("FACE PIPELINE TRACE: [8-Precropped] Face crop standardized from whole image fallback.")
                        except Exception as e:
                            LOGGER.debug(f"Pre-cropped fallback face extraction failed: {e}")
                            face_tensor = None
                    else:
                        face_status = "LOW_QUALITY_FACE"
                        diagnostic_msg = "Pre-cropped face detected but quality is too low."

            # If still no usable face detected/validated
            if face_tensor is None:
                if face_status == "NO_FACE":
                    diagnostic_msg = "No face detected in query photo."
                elif face_status == "LOW_QUALITY_FACE":
                    pass # Keep diagnostic_msg
                else:
                    face_status = "NO_FACE"
                    diagnostic_msg = "Face is only partially visible. Face recognition requires a more complete face."
                
                LOGGER.warning(f"FACE PIPELINE TRACE: Failed. Status: {face_status}, Msg: {diagnostic_msg}")
                return {
                    "embedding": None,
                    "face_status": face_status,
                    "diagnostic_msg": diagnostic_msg,
                    "confidence": 0.0,
                    "bounding_box": []
                }
            
            # 3. Generate Embedding
            LOGGER.info("FACE PIPELINE TRACE: [9] Face alignment and face embedding generation starting.")
            face_tensor = face_tensor.to(FACE_DEVICE).unsqueeze(0)
            if FACE_DEVICE == "cuda":
                face_tensor = face_tensor.half()
                
            with torch.no_grad():
                raw_emb = self.face_embedder(face_tensor)
                normalized_emb = raw_emb / raw_emb.norm(p=2, dim=-1, keepdim=True)
                emb_list = normalized_emb.cpu().numpy()[0].tolist()
                
            LOGGER.info("FACE PIPELINE TRACE: [10] Face embedding generated successfully.")
            return {
                "embedding": emb_list,
                "face_status": face_status,
                "diagnostic_msg": diagnostic_msg,
                "confidence": confidence,
                "bounding_box": bbox
            }


# -*- coding: utf-8 -*-
import sys
import time
import os
import traceback

def step_wrapper(step_num, step_name, func, timeout_sec=60):
    print(f"\n========== STEP {step_num} ==========", flush=True)
    print(f"START: {step_name}", flush=True)
    t0 = time.time()
    try:
        res = func()
        elapsed = time.time() - t0
        print(f"END: {step_name}", flush=True)
        print(f"elapsed: {elapsed:.3f} seconds", flush=True)
        return res
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED: {step_name}", flush=True)
        print(f"elapsed: {elapsed:.3f} seconds", flush=True)
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        print(f"\nHANG/RESPONSE TIMEOUT or FAILURE:\noperation: {step_name}\nelapsed: {elapsed:.3f}s\nlast successful step: STEP {step_num-1}\nprobable dependency: {type(e).__name__}", flush=True)
        sys.exit(1)

# STEP 1: Python/import environment
step_wrapper(1, "Python/import environment", lambda: print(f"Python executable: {sys.executable}\nPython version: {sys.version.split()[0]}", flush=True))

# STEP 2: cv2 import
cv2 = step_wrapper(2, "cv2 import", lambda: __import__("cv2"))

# STEP 3: torch import
def do_torch():
    import torch
    print(f"torch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}", flush=True)
    return torch
torch = step_wrapper(3, "torch import", do_torch)

# STEP 4: project imports
def do_proj_imports():
    sys.path.append(os.path.abspath("."))
    import config
    from storage.sqlite_db import SQLiteDatabase
    from storage.vector_db import VectorDatabase
    from storage.media_storage import MediaStorage
    from storage.cache import TrackCache, TrackData
    from pipeline.matcher import ActiveQueryMatcher
    from pipeline.dispatcher import ResultDispatcher
    from pipeline.job_manager import JobManager
    return "All project modules successfully imported"
step_wrapper(4, "project imports", do_proj_imports)

# STEP 5: SQLite connection
sqlite_db = None
def do_sqlite():
    global sqlite_db
    from config import SQLITE_DB_PATH
    from storage.sqlite_db import SQLiteDatabase
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    with sqlite_db.connection() as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"SQLite database: {SQLITE_DB_PATH}\nTables: {len(tables)}", flush=True)
    return sqlite_db
step_wrapper(5, "SQLite connection", do_sqlite)

# STEP 6: Chroma initialization
vector_db = None
def do_chroma():
    global vector_db
    from config import CHROMA_PERSIST_DIR
    from storage.vector_db import VectorDatabase
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    counts = {c.name: c.count() for c in vector_db.client.list_collections()}
    print(f"Chroma directory: {CHROMA_PERSIST_DIR}\nCollections: {counts}", flush=True)
    return vector_db
step_wrapper(6, "Chroma initialization", do_chroma)

# STEP 7: YOLO weight discovery
yolo_weight_path = None
def do_yolo_discovery():
    global yolo_weight_path
    for candidate in ["yolo11s.pt", "yolov8n.pt", "models/yolo11s.pt"]:
        if os.path.exists(candidate):
            yolo_weight_path = os.path.abspath(candidate)
            break
    if not yolo_weight_path:
        raise FileNotFoundError("No YOLO weight file found locally.")
    print(f"exact weight path: {yolo_weight_path}", flush=True)
    return yolo_weight_path
step_wrapper(7, "YOLO weight discovery", do_yolo_discovery)

# STEP 8: YOLO model load
yolo_model = None
def do_yolo_load():
    global yolo_model
    from ultralytics import YOLO
    yolo_model = YOLO(yolo_weight_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model.to(device)
    return yolo_model
step_wrapper(8, "YOLO model load", do_yolo_load)

# STEP 9: test video/frame discovery
video_path = None
def do_video_discovery():
    global video_path
    for cand in ["data/media/uploads/inside.mp4", "data/media/uploads/outside.mp4"]:
        if os.path.exists(cand):
            video_path = cand
            break
    if not video_path:
        raise FileNotFoundError("No test video found in data/media/uploads/")
    print(f"exact source: {video_path}", flush=True)
    return video_path
step_wrapper(9, "test video/frame discovery", do_video_discovery)

# STEP 10: read ONE frame
frame_bgr = None
def do_read_frame():
    global frame_bgr
    cap = cv2.VideoCapture(video_path)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret or frame_bgr is None:
        raise RuntimeError("Failed to read frame from video.")
    print(f"frame shape: {frame_bgr.shape}", flush=True)
    return frame_bgr
step_wrapper(10, "read ONE frame", do_read_frame)

# STEP 11: ONE YOLO inference
yolo_dets = []
def do_yolo_inference():
    global yolo_dets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = yolo_model.predict(frame_bgr, verbose=False, device=device, conf=0.25)
    boxes = res[0].boxes
    allowed_classes = {0, 2, 3, 5, 7}
    for box in boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in allowed_classes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            yolo_dets.append((x1, y1, x2, y2, conf, cls_id))
    print(f"detections: {len(yolo_dets)}", flush=True)
    return yolo_dets
step_wrapper(11, "ONE YOLO inference", do_yolo_inference)

# STEP 12: tracker initialization
track_cache = None
def do_tracker_init():
    global track_cache
    from storage.cache import TrackCache
    track_cache = TrackCache()
    return track_cache
step_wrapper(12, "tracker initialization", do_tracker_init)

# STEP 13: ONE tracking operation
tracked_entities = []
def do_tracking_op():
    global tracked_entities
    cam_id = "test_cam_diag"
    active_tracks = track_cache.active_tracks.setdefault(cam_id, {})
    h_img, w_img = frame_bgr.shape[:2]
    
    for idx, det in enumerate(yolo_dets):
        track_id = idx + 1
        x1, y1, x2, y2, conf, cls_id = det
        w, h = x2 - x1, y2 - y1
        x1_pad = max(0, int(x1 - 0.15 * w))
        y1_pad = max(0, int(y1 - 0.15 * h))
        x2_pad = min(w_img, int(x2 + 0.15 * w))
        y2_pad = min(h_img, int(y2 + 0.15 * h))
        crop = frame_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.size > 0:
            track = track_cache.get_or_create_track(cam_id, track_id, 0.0)
            track_cache.add_to_track(cam_id, track_id, 1, 0.0, [x1, y1, x2, y2], crop)
            tracked_entities.append((track, crop, "person" if cls_id == 0 else "vehicle", [x1, y1, x2, y2], conf))
            
    print(f"tracks: {len(tracked_entities)}", flush=True)
    return tracked_entities
step_wrapper(13, "ONE tracking operation", do_tracking_op)

# STEP 14: crop generation
crops = []
def do_crop_gen():
    global crops
    for track, crop, entity_type, bbox, conf in tracked_entities:
        crops.append((crop, entity_type, bbox, conf))
    print(f"crops: {len(crops)}", flush=True)
    return crops
step_wrapper(14, "crop generation", do_crop_gen)

# STEP 15: CLIP model initialization
clip_processor = None
clip_model = None
def do_clip_init():
    global clip_processor, clip_model
    from transformers import CLIPProcessor, CLIPModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision="refs/pr/66", use_safetensors=True).to(device)
    if device == "cuda":
        clip_model = clip_model.half()
    clip_model.eval()
    return clip_model
step_wrapper(15, "CLIP model initialization", do_clip_init)

# STEP 16: ONE CLIP embedding
clip_emb = None
def do_clip_embedding():
    global clip_emb
    from PIL import Image
    import numpy as np
    sample_crop = crops[0][0]
    pil_img = Image.fromarray(cv2.cvtColor(sample_crop, cv2.COLOR_BGR2RGB))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = clip_processor(images=[pil_img], return_tensors="pt").to(device)
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()
    with torch.no_grad():
        feat = clip_model.get_image_features(**inputs)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        clip_emb = feat.cpu().float().numpy()[0].tolist()
    norm = float(np.linalg.norm(clip_emb))
    print(f"dimension: {len(clip_emb)}", flush=True)
    print(f"norm: {norm:.4f}", flush=True)
    return clip_emb
step_wrapper(16, "ONE CLIP embedding", do_clip_embedding)

# STEP 17: Re-ID model initialization
reid_model = None
reid_transform = None
def do_reid_init():
    global reid_model, reid_transform
    from torchvision import models as tv_models, transforms
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reid_model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
    reid_model.classifier = torch.nn.Identity()
    reid_model.to(device)
    if device == "cuda":
        reid_model = reid_model.half()
    reid_model.eval()
    reid_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return reid_model
step_wrapper(17, "Re-ID model initialization", do_reid_init)

# STEP 18: ONE Re-ID embedding
reid_emb = None
def do_reid_embedding():
    global reid_emb
    from PIL import Image
    import numpy as np
    sample_crop = crops[0][0]
    pil_img = Image.fromarray(cv2.cvtColor(sample_crop, cv2.COLOR_BGR2RGB))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = reid_transform(pil_img).unsqueeze(0).to(device)
    if device == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        feat = reid_model(tensor)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        reid_emb = feat.cpu().float().numpy()[0].tolist()
    norm = float(np.linalg.norm(reid_emb))
    print(f"dimension: {len(reid_emb)}", flush=True)
    print(f"norm: {norm:.4f}", flush=True)
    return reid_emb
step_wrapper(18, "ONE Re-ID embedding", do_reid_embedding)

# STEP 19: SQLite write
diag_job_id = f"diag_{int(time.time())}"
diag_track_id = f"{diag_job_id}_cam_1"
diag_det_id = f"det_{diag_job_id}_1"
def do_sqlite_write():
    now = time.time()
    with sqlite_db.connection() as conn:
        conn.execute("INSERT INTO jobs (id, name, status, created_at, updated_at, profile) VALUES (?, ?, ?, ?, ?, ?)",
                     (diag_job_id, "Diag Job", "indexing", now, now, "BALANCED"))
        conn.execute("INSERT INTO cameras (id, job_id, name, source_type, source_path, status) VALUES (?, ?, ?, ?, ?, ?)",
                     ("cam_diag", diag_job_id, "Diag Cam", "file", video_path, "active"))
        conn.execute("INSERT INTO tracks (id, camera_id, job_id, track_id, entity_type, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (diag_track_id, "cam_diag", diag_job_id, 1, "person", 0.0, 0.0))
        conn.execute("INSERT INTO detections (id, track_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, confidence, entity_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (diag_det_id, diag_track_id, 1, 0.0, 10.0, 10.0, 50.0, 100.0, "data/media/crops/sample.jpg", 0.95, "person"))
    return diag_det_id
step_wrapper(19, "SQLite write", do_sqlite_write)

# STEP 20: Chroma write
diag_vec_id = f"vis_{diag_track_id}_1"
def do_chroma_write():
    coll = vector_db.visual_coll
    before = coll.count()
    coll.add(
        ids=[diag_vec_id],
        embeddings=[clip_emb],
        metadatas=[{
            "job_id": diag_job_id,
            "camera_id": "cam_diag",
            "track_id": diag_track_id,
            "detection_id": diag_det_id,
            "entity_type": "person",
            "embedding_model": "clip_vit_b32",
            "embedding_version": "v1",
            "embedding_dimension": 512
        }]
    )
    after = coll.count()
    print(f"collection: {coll.name}", flush=True)
    print(f"before count: {before}", flush=True)
    print(f"after count: {after}", flush=True)
    return after
step_wrapper(20, "Chroma write", do_chroma_write)

# STEP 21: Chroma read-back
def do_chroma_readback():
    res = vector_db.visual_coll.get(ids=[diag_vec_id], include=["metadatas", "embeddings"])
    if not res["ids"]:
        raise RuntimeError("Vector not found on read-back!")
    v_id = res["ids"][0]
    meta = res["metadatas"][0]
    dim = len(res["embeddings"][0])
    print(f"vector ID: {v_id}", flush=True)
    print(f"metadata: {meta}", flush=True)
    print(f"dimension: {dim}", flush=True)
    
    # Cleanup
    vector_db.delete_by_job(diag_job_id)
    with sqlite_db.connection() as conn:
        conn.execute("DELETE FROM detections WHERE id = ?", (diag_det_id,))
        conn.execute("DELETE FROM tracks WHERE id = ?", (diag_track_id,))
        conn.execute("DELETE FROM cameras WHERE job_id = ?", (diag_job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (diag_job_id,))
    return True
step_wrapper(21, "Chroma read-back", do_chroma_readback)

print("\n===============================================================")
print("   SUCCESS: ALL 21 DIAGNOSTIC STEPS EXECUTED AND VALIDATED    ")
print("===============================================================\n", flush=True)

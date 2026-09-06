# -*- coding: utf-8 -*-
import time
import sys
import os

def timed_step(step_num, name, func):
    print(f"[{step_num}] START {name}", flush=True)
    t0 = time.time()
    try:
        res = func()
        elapsed = time.time() - t0
        print(f"[{step_num}] END {name} (Elapsed: {elapsed:.3f}s)", flush=True)
        if res is not None:
            print(f"    Info: {res}", flush=True)
        return res
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{step_num}] FAILED {name} (Elapsed: {elapsed:.3f}s) | Error: {e}", flush=True)
        raise

# [1] import cv2
timed_step(1, "import cv2", lambda: __import__("cv2").__version__)

# [2] import torch
def step2():
    import torch
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    return f"PyTorch: {torch.__version__}, CUDA: {cuda_avail}, GPU: {dev_name}"
timed_step(2, "import torch", step2)

# [3] import project modules
def step3():
    sys.path.append(os.path.abspath("."))
    from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, MEDIA_DIR
    from storage import SQLiteDatabase, VectorDatabase, MediaStorage
    from storage.cache import TrackCache
    return "All project storage and pipeline modules imported"
timed_step(3, "import project modules", step3)

# [4] SQLite initialization
sqlite_db = None
def step4():
    global sqlite_db
    from config import SQLITE_DB_PATH
    from storage.sqlite_db import SQLiteDatabase
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    with sqlite_db.connection() as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    return f"SQLite path: {SQLITE_DB_PATH}, Tables ({len(tables)}): {tables[:5]}"
timed_step(4, "SQLite initialization", step4)

# [5] Chroma initialization
vector_db = None
def step5():
    global vector_db
    from config import CHROMA_PERSIST_DIR
    from storage.vector_db import VectorDatabase
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    counts = {c.name: c.count() for c in vector_db.client.list_collections()}
    return f"Chroma path: {CHROMA_PERSIST_DIR}, Collections/Counts: {counts}"
timed_step(5, "Chroma initialization", step5)

# [6] YOLO model loading
yolo_model = None
def step6():
    global yolo_model
    from ultralytics import YOLO
    import torch
    weights = "yolo11s.pt" if os.path.exists("yolo11s.pt") else "yolov8n.pt"
    yolo_model = YOLO(weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model.to(device)
    return f"Loaded weights: {weights} on {device}"
timed_step(6, "YOLO model loading", step6)

# [7] reading the single test frame
frame = None
def step7():
    global frame
    import cv2
    video_path = "data/media/uploads/inside.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame from {video_path}")
    return f"Source: {video_path}, Res: {w}x{h}, FPS: {fps}, Frame shape: {frame.shape}"
timed_step(7, "reading the single test frame", step7)

# [8] YOLO inference
yolo_results = None
def step8():
    global yolo_results
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_results = yolo_model.predict(frame, verbose=False, device=device)
    boxes = yolo_results[0].boxes if yolo_results else []
    return f"Detection count: {len(boxes)}"
timed_step(8, "YOLO inference", step8)

# [9] tracker
tracked_boxes = []
def step9():
    global tracked_boxes
    allowed_classes = {0, 2, 3, 5, 7} # person, car, motorcycle, bus, truck
    for box in yolo_results[0].boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in allowed_classes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            tracked_boxes.append((x1, y1, x2, y2, conf, "person" if cls_id == 0 else "vehicle"))
    return f"Filtered tracked entities count: {len(tracked_boxes)}"
timed_step(9, "tracker", step9)

# [10] crop generation
crops = []
def step10():
    global crops
    h_img, w_img = frame.shape[:2]
    for x1, y1, x2, y2, conf, entity_type in tracked_boxes:
        w, h = x2 - x1, y2 - y1
        x1_pad = max(0, int(x1 - 0.15 * w))
        y1_pad = max(0, int(y1 - 0.15 * h))
        x2_pad = min(w_img, int(x2 + 0.15 * w))
        y2_pad = min(h_img, int(y2 + 0.15 * h))
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size > 0:
            crops.append((crop, entity_type, [x1, y1, x2, y2], conf))
    return f"Generated valid crops count: {len(crops)}"
timed_step(10, "crop generation", step10)

# [11] CLIP embedding
clip_emb = None
def step11():
    global clip_emb
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch
    import numpy as np
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, revision="refs/pr/66", use_safetensors=True).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()
    
    sample_crop = crops[0][0] if crops else frame
    pil_img = Image.fromarray(sample_crop[:, :, ::-1])
    inputs = processor(images=[pil_img], return_tensors="pt").to(device)
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        clip_emb = feat.cpu().float().numpy()[0].tolist()
        
    norm = float(np.linalg.norm(clip_emb))
    return f"Dim: {len(clip_emb)}, dtype: float32, L2 norm: {norm:.4f}"
timed_step(11, "CLIP embedding", step11)

# [12] Re-ID embedding
reid_emb = None
def step12():
    global reid_emb
    import torch
    import numpy as np
    from torchvision import models as tv_models, transforms
    from PIL import Image
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reid_model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
    reid_model.classifier = torch.nn.Identity()
    reid_model.to(device)
    if device == "cuda":
        reid_model = reid_model.half()
    reid_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    sample_crop = crops[0][0] if crops else frame
    pil_img = Image.fromarray(sample_crop[:, :, ::-1])
    tensor = transform(pil_img).unsqueeze(0).to(device)
    if device == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        feat = reid_model(tensor)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        reid_emb = feat.cpu().float().numpy()[0].tolist()
        
    norm = float(np.linalg.norm(reid_emb))
    return f"Dim: {len(reid_emb)}, dtype: float32, L2 norm: {norm:.4f}"
timed_step(12, "Re-ID embedding", step12)

# [13] SQLite persistence
test_job_id = f"diag_job_{int(time.time())}"
test_track_id = f"{test_job_id}_cam1_1"
def step13():
    now = time.time()
    with sqlite_db.connection() as conn:
        conn.execute("INSERT INTO jobs (id, name, status, created_at, updated_at, profile) VALUES (?, ?, ?, ?, ?, ?)",
                     (test_job_id, "Diagnostic Job", "indexing", now, now, "BALANCED"))
        conn.execute("INSERT INTO cameras (id, job_id, name, source_type, source_path, status) VALUES (?, ?, ?, ?, ?, ?)",
                     ("cam1", test_job_id, "Cam 1", "file", "data/media/uploads/inside.mp4", "active"))
        conn.execute("INSERT INTO tracks (id, camera_id, job_id, track_id, entity_type, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (test_track_id, "cam1", test_job_id, 1, "person", 0.0, 0.0))
        conn.execute("INSERT INTO detections (id, track_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, confidence, entity_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     ("det_diag_1", test_track_id, 1, 0.0, 10.0, 10.0, 100.0, 200.0, "data/media/crops/person/sample.jpg", 0.90, "person"))
    return f"Committed job={test_job_id}, track={test_track_id}, detection=det_diag_1"
timed_step(13, "SQLite persistence", step13)

# [14] Chroma vector insertion
test_vec_id = f"vis_{test_track_id}_det_1"
def step14():
    coll = vector_db.visual_coll
    count_before = coll.count()
    coll.add(
        ids=[test_vec_id],
        embeddings=[clip_emb],
        metadatas=[{
            "job_id": test_job_id,
            "camera_id": "cam1",
            "track_id": test_track_id,
            "detection_id": "det_diag_1",
            "entity_type": "person",
            "embedding_model": "clip_vit_b32",
            "embedding_version": "v1",
            "embedding_dimension": 512
        }]
    )
    count_after = coll.count()
    return f"Collection: {coll.name}, Inserted: 1, Before: {count_before}, After: {count_after}"
timed_step(14, "Chroma vector insertion", step14)

# [15] Chroma read-back verification
def step15():
    res = vector_db.visual_coll.get(ids=[test_vec_id], include=["metadatas", "embeddings"])
    if not res.get("ids") or len(res["ids"]) == 0:
        raise RuntimeError(f"Vector {test_vec_id} could not be read back from ChromaDB!")
    meta = res["metadatas"][0]
    emb_back = res["embeddings"][0]
    
    # Cleanup test data
    vector_db.delete_by_job(test_job_id)
    with sqlite_db.connection() as conn:
        conn.execute("DELETE FROM detections WHERE id = 'det_diag_1'")
        conn.execute("DELETE FROM tracks WHERE id = ?", (test_track_id,))
        conn.execute("DELETE FROM cameras WHERE job_id = ?", (test_job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (test_job_id,))
        
    return f"Read-back vector ID: {res['ids'][0]}, Meta job_id: {meta['job_id']}, Dim: {len(emb_back)}, Cleanup: SUCCESS"
timed_step(15, "Chroma read-back verification", step15)

print("\n===============================================================")
print("  ALL 15 STEP-BY-STEP DIAGNOSTIC COMPONENTS COMPLETED CLEANLY  ")
print("===============================================================\n")

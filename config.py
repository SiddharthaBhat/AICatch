"""System configuration settings for AICatch.

Optimized for AMD Ryzen 7 7700, 16GB RAM, RTX 3050 (4GB VRAM) on Windows.
"""

from pathlib import Path
from typing import Dict, Any

# ==============================================================================
# 1. Path Definitions
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
DATA_DIR = BASE_DIR / "data"

# SQLite and Vector DB Paths
SQLITE_DB_PATH = DATA_DIR / "aicatch.db"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# Media Storage Directories
MEDIA_DIR = DATA_DIR / "media"
UPLOAD_DIR = MEDIA_DIR / "uploads"
CLIPS_DIR = MEDIA_DIR / "clips"
THUMBNAILS_DIR = MEDIA_DIR / "thumbnails"
CROPS_DIR = MEDIA_DIR / "crops"

# Sub-crop Directories
PERSON_CROPS_DIR = CROPS_DIR / "person"
VEHICLE_CROPS_DIR = CROPS_DIR / "vehicle"
FACE_CROPS_DIR = CROPS_DIR / "face"
PLATE_CROPS_DIR = CROPS_DIR / "plate"

# Ensure all structural folders exist
for folder in [
    TEMP_DIR, DATA_DIR, MEDIA_DIR, UPLOAD_DIR, CLIPS_DIR, THUMBNAILS_DIR,
    CROPS_DIR, PERSON_CROPS_DIR, VEHICLE_CROPS_DIR, FACE_CROPS_DIR, PLATE_CROPS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. Hardware and Model Settings
# ==============================================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configurable CPU/GPU placement targets (RTX 3050 VRAM optimization)
YOLO_DEVICE = DEVICE
CLIP_DEVICE = "cuda"  # Run visual embeddings on GPU (CLIP is fast on GPU, low VRAM)
REID_DEVICE = "cuda"  # Person Re-ID feature extraction on GPU
FACE_DEVICE = "cpu"   # Run face models on CPU
OCR_DEVICE = "cpu"    # Run EasyOCR on CPU

# Model Names / Versioning (Lightweight choices to avoid CUDA OOM)
YOLO_MODEL_PATH = str(BASE_DIR / "yolo11s.pt")  # High-accuracy YOLO11s model
YOLO_MODEL_NAME = "YOLO11s"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # 150MB FP32 CLIP model (VRAM safe)
EMBEDDING_MODEL_NAME = "clip_vit_b32"
EMBEDDING_VERSION = "v2"
REID_MODEL_NAME = "mobilenet_v3_reid_576"
REID_VERSION = "v1"
FACE_DETECTOR_MODEL = "yolov8n-face.pt"  # Face detector or similar lightweight detector
FACE_EMBEDDER_MODEL = "facenet"  # InceptionResnetV1 pretrained on VGGFace2
OCR_MODEL_NAME = "easyocr"

# ==============================================================================
# 3. GPU Scheduler and Memory Settings (RTX 3050 4GB Optimization)
# ==============================================================================
MAX_VRAM_USAGE_MB = 3000  # Cap target active VRAM usage at 3.0 GB to avoid crash
BATCH_SIZE = 8            # Dynamic GPU Worker batch size
GPU_QUEUE_LIMIT = 64      # Bounded frame queue to prevent CPU RAM overflow

# Coexistence threshold for triggering empty cache
VRAM_DANGER_THRESHOLD_MB = 3600

# ==============================================================================
# 4. Processing Profiles
# ==============================================================================
# Fast, Balanced, and Forensic configurations.
# NOTE: Do NOT claim exact accuracy values; these control ingestion rate and granularity.
PROCESSING_PROFILES: Dict[str, Dict[str, Any]] = {
    "FAST": {
        "sampling_rate_sec": 4.0,           # Process one frame every 4.0s
        "keyframe_min_sharpness": 150.0,    # Strict sharpness filter
        "skip_static_threshold": 0.015,     # High threshold to skip static content
        "run_face_analysis": False,          # Skip heavy models to maximize speed
        "run_ocr_analysis": False,           # Skip OCR in FAST profile
    },
    "BALANCED": {
        "sampling_rate_sec": 2.0,           # Process one frame every 2.0s
        "keyframe_min_sharpness": 100.0,    # Moderate sharpness filter
        "skip_static_threshold": 0.008,     # Balanced static frame bypass
        "run_face_analysis": True,
        "run_ocr_analysis": True,
    },
    "FORENSIC": {
        "sampling_rate_sec": 0.5,           # Process one frame every 0.5s (high recall)
        "keyframe_min_sharpness": 50.0,     # Loose sharpness check to capture all angles
        "skip_static_threshold": 0.002,     # Analyze even minor movements
        "run_face_analysis": True,
        "run_ocr_analysis": True,
    }
}

# ==============================================================================
# 5. Pipeline & RTSP Settings
# ==============================================================================
MAX_RTSP_RECONNECT_ATTEMPTS = 5
RTSP_RECONNECT_BACKOFF_FACTOR = 2.0  # Exponential factor

# OCR / Plate settings
PLATE_CONFIDENCE_THRESHOLD = 0.50

# Matcher settings
DEFAULT_ALERT_SIMILARITY_THRESHOLD = 0.72

# ==============================================================================
# 6. Legacy Compatibility Layer (for old app.py Streamlit dashboard compatibility)
# ==============================================================================
OUTPUT_DIR = BASE_DIR / "outputs"
THUMBNAIL_DIR = THUMBNAILS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ensure_directories() -> None:
    """Legacy compatibility helper. Folders are created on import."""
    pass


# ==============================================================================
# 7. Search and Re-ranking Settings
# ==============================================================================
SEARCH_WEIGHT_SEMANTIC = 0.75
SEARCH_WEIGHT_ATTRIBUTE = 0.15
SEARCH_WEIGHT_QUALITY = 0.10

# Face recognition matching similarity threshold
FACE_MATCH_THRESHOLD = 0.72

# Text search matching similarity threshold (CLIP text-image cosine)
TEXT_MATCH_THRESHOLD = 0.26

# Image search matching similarity threshold (CLIP image-image cosine)
IMAGE_MATCH_THRESHOLD = 0.55

# Unified Image Search weighting
IMAGE_VISUAL_WEIGHT = 0.70
IMAGE_FACE_WEIGHT = 0.30

# Post-rerank final score gates (applied AFTER weighted combination)
# Results below these thresholds are rejected even if they passed the raw similarity gate
FINAL_SCORE_MIN_TEXT = 0.30
FINAL_SCORE_MIN_IMAGE = 0.50

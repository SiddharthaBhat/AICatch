# AICatch — Forensic Multi-Camera CCTV Investigation Platform

AICatch is an end-to-end computer vision and vector retrieval platform designed for forensic video analysis across multi-camera CCTV environments. It indexes footage into structured SQLite metadata and high-dimensional visual embeddings, enabling fast natural language, reference image, and cross-camera target tracking with strict video timestamp provenance.

---

## Key Capabilities

- **Multi-Camera Investigation Workspace**: Process, manage, and search synchronized multi-channel CCTV footage scoped by investigation case.
- **Hybrid Visual & Scene Retrieval**:
  - **Object Crop Indexing**: Detects and extracts person and vehicle bounding boxes using YOLO and indexes visual features via normalized CLIP ViT-B/32 (512-D) and lightweight Re-ID (MobileNetV3 576-D).
  - **Scene Keyframe Indexing**: Extracts adaptive scene keyframes with perceptual deduplication (HSV histogram + spatial delta) to index stationary, environmental, and non-detected visual targets.
- **Cross-Camera Journey Tracking**: Re-identifies subjects across non-overlapping camera feeds and reconstructs chronological transit timelines.
- **Forensic Video Viewer**: Synchronized in-browser video playback that jumps directly to exact event timestamps with bounding box overlays.
- **Evidence Management**: Organize, tag, and bookmark visual observations and notes into forensic evidence folders.
- **Resource-Budgeted Sequential GPU Scheduler**: Prevents CUDA out-of-memory (OOM) errors through thread-locked inference queuing and dynamic VRAM budgeting tailored for desktop GPUs (e.g., RTX 3050 4GB).

---

## System Architecture

`
                       CCTV Video Streams / Files
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │    Sequential GPU Scheduler     │
                  │  (YOLOv8 + CLIP + ReID + MTCNN) │
                  └────────────────┬────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
       ┌───────────────────┐               ┌───────────────────┐
       │ SQLite Relational │               │ ChromaDB Vector   │
       │     Database      │               │     Database      │
       │ (Tracks, Frames,  │               │ (Object & Scene   │
       │   Evidence Logs)  │               │    Embeddings)    │
       └─────────┬─────────┘               └─────────┬─────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    FastAPI Backend Server     │
                   │ (Stratified Hybrid Retrieval) │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   React / TypeScript UI       │
                   │ (Forensic Workspace & Viewer) │
                   └───────────────────────────────┘
`

---

## Technology Stack

### Backend & AI Pipeline
- **Framework**: FastAPI, Uvicorn, Pydantic
- **Object Detection & Tracking**: YOLOv8 (ultralytics)
- **Visual Embeddings**: CLIP ViT-B/32 (	ransformers, 	orch)
- **Person Re-Identification**: MobileNetV3 (576-D feature extractor)
- **Facial & Text Detection**: MTCNN / InceptionResnetV1 (acenet-pytorch), EasyOCR
- **Vector Database**: ChromaDB (cosine similarity space)
- **Relational Storage**: SQLite with WAL mode (sqlite3)

### Frontend
- **Framework**: React 18, TypeScript, Vite
- **Styling & UI**: Modern Dark Forensic Theme, Lucide React Icons
- **Video Playback**: HTML5 Video API with screen-space bounding box interpolation

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (CUDA-enabled PyTorch recommended for GPU acceleration)
- Node.js 18+ and npm
- FFmpeg (for video decoding and thumbnail extraction)

### 1. Backend Setup
`ash
# Clone the repository
git clone https://github.com/SiddharthaBhat/AICatch.git
cd AICatch

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt  # Or install project dependencies

# Initialize configuration
cp .env.example .env
`

### 2. Frontend Setup
`ash
cd frontend
npm install
npm run build   # Production build
npm run dev     # Development server (http://localhost:5173)
`

### 3. Running the Application
`ash
# From repository root:
python main.py
`
The FastAPI backend server will start on http://localhost:8000.

---

## Verification & Testing

The repository contains automated diagnostic and integration test suites:

`ash
# Run end-to-end search verification
python -m pytest tests/  # Or run diagnostic scripts in test/
`

---

## Project Structure

`
AICatch/
├── backend/ (Root service modules)
│   ├── main.py                     # FastAPI REST & SSE endpoints
│   ├── config.py                   # System paths, model registry, & hardware thresholds
│   ├── pipeline/
│   │   ├── gpu_scheduler.py        # Sequential GPU worker & model budgeter
│   │   ├── scene_indexer.py        # Keyframe extraction & perceptual deduplication
│   │   ├── dispatcher.py           # Result distribution & batch indexing
│   │   └── journey_engine.py       # Cross-camera Re-ID transition analysis
│   ├── storage/
│   │   ├── sqlite_db.py            # SQLite schema, queries, & WAL management
│   │   ├── vector_db.py            # ChromaDB visual & scene vector collections
│   │   ├── search.py               # Stratified hybrid search & calibration
│   │   └── media_storage.py        # Crops, keyframes, and video clip persistence
│   └── utils/
│       ├── monitor.py              # Real-time VRAM, GPU, and CPU telemetry
│       └── color_analyzer.py       # Dominant upper/lower body color extraction
├── frontend/                       # React + TypeScript Web Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── SearchWorkspace.tsx # Multi-camera image and text search
│   │   │   ├── ForensicVideoViewer.tsx # Video playback with frame bboxes
│   │   │   ├── JourneyView.tsx     # Chronological camera transit flow
│   │   │   └── Evidence.tsx        # Evidence folder & note management
│   │   └── services/api.ts         # REST client for backend communication
├── .gitignore
├── .env.example
└── README.md
`

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

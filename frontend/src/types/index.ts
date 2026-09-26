export interface Job {
  id: string;
  name: string;
  status: 'queued' | 'starting' | 'indexing' | 'paused' | 'completed' | 'failed' | 'cancelled';
  created_at: number;
  updated_at: number;
  profile: 'FAST' | 'BALANCED' | 'FORENSIC';
  duration_sec?: number;
  processed_sec?: number;
  fps?: number;
  cpu_usage?: number;
  gpu_usage?: number;
  vram_usage?: number;
  error_msg?: string;
  progress?: number;
  investigation_type?: 'single_video' | 'multi_camera';
}

export interface Camera {
  id: string;
  job_id: string;
  name: string;
  source_type: 'live' | 'file';
  source_path: string;
  status: 'active' | 'inactive' | 'ready' | 'indexing' | 'completed' | 'failed' | 'not_configured' | 'uploading' | string;
  resolution?: string;
  fps?: number;
  progress?: number;
  processed_sec?: number;
  duration_sec?: number;
  tracks_count?: number;
}

export interface Track {
  id: string;
  camera_id: string;
  job_id: string;
  track_id: number;
  entity_type: 'person' | 'vehicle';
  start_time: number;
  end_time: number;
  plate_text?: string;
  label?: string;
}

export interface Detection {
  id: string;
  track_id: string;
  frame_idx: number;
  timestamp: number;
  bbox_x1: number;
  bbox_y1: number;
  bbox_x2: number;
  bbox_y2: number;
  crop_path: string;
  confidence: number;
  entity_type: 'person' | 'vehicle';
}

export interface SavedPerson {
  id: string;
  name: string;
  face_image_path: string;
  face_embedding_id: string;
  created_at: number;
}

export interface ActiveQuery {
  id: string;
  session_id: string;
  type: 'text' | 'image' | 'face' | 'plate' | 'saved_person';
  query_text?: string;
  query_image_path?: string;
  query_embedding_id?: string;
  similarity_threshold: number;
  is_active: number;
  created_at: number;
  match_count?: number;
}

export interface Alert {
  query_id: string;
  session_id: string;
  similarity: number;
  camera_id: string;
  timestamp: number;
  track_id: string;
  entity_type: 'person' | 'vehicle';
  crop_path: string;
  plate_text?: string;
  bbox: number[];
  job_id?: string;
}

export interface HardwareMetrics {
  processing_fps: number;
  queue_depth: number;
  cpu_util: number;
  gpu_util: number;
  vram_used_mb: number;
  ram_used_gb: number;
  latency_yolo: number;
  latency_clip: number;
  latency_ocr: number;
  latency_face: number;
  latency_db: number;
  frames_decoded: number;
  frames_analyzed: number;
  detections_count: number;
  embeddings_count: number;
}

export interface MetricStreamPayload {
  timestamp: number;
  progress: number;
  status: string;
  hardware: HardwareMetrics;
}

// ==========================================
// Cross-Camera Evidence Stitching Types
// ==========================================

export interface StitchProvenance {
  job_id: string;
  camera_id: string;
  camera_name?: string;
  source_type: string;
  video_path?: string;
  evidence_path?: string;
  crop_path?: string;
  bbox?: number[];
  timestamp: number;
  track_id?: string | number;
  detection_id?: string | number;
}

export interface StitchContinuitySignals {
  score: number;
  reid_similarity?: number | null;
  visual_similarity?: number | null;
  temporal_score: number;
  transition_score: number;
  topology_available: boolean;
  reason: string;
}

export interface StitchObservation {
  observation_id: string;
  job_id: string;
  camera_id: string;
  camera_name?: string;
  timestamp: number;
  end_timestamp?: number;
  source_type: string;
  entity_type: string;
  track_id?: string | number;
  detection_id?: string | number;
  bbox?: number[];
  crop_path?: string;
  source_video_path?: string;
  visual_similarity?: number | null;
  reid_similarity?: number | null;
  combined_score?: number | null;
  occurrence_id?: string;
  original_rank?: number;
  metadata?: Record<string, any>;
}

export interface StitchSegment {
  segment_id: string;
  camera_id: string;
  camera_name: string;
  start_time: number;
  end_time: number;
  duration: number;
  occurrence_id?: string | null;
  representative_observation_id: string;
  representative_crop_path?: string | null;
  observations_count: number;
  observations: StitchObservation[];
  continuity: StitchContinuitySignals;
  provenance: StitchProvenance;
}

export interface StitchTransition {
  transition_id: string;
  from_camera: string;
  to_camera: string;
  from_camera_name: string;
  to_camera_name: string;
  from_segment_id: string;
  to_segment_id: string;
  transit_time_sec: number;
  score: number;
  signals: StitchContinuitySignals;
}

export interface StitchResult {
  stitch_id: string;
  investigation_id: string;
  job_id: string;
  entity_type: 'person' | 'object';
  status: 'complete' | 'partial' | 'insufficient_evidence';
  reason?: string | null;
  created_at: number;
  target_summary?: Record<string, any> | null;
  segments: StitchSegment[];
  transitions: StitchTransition[];
  camera_diversity: number;
  total_duration_sec: number;
}


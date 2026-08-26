import React, { useState, useEffect, useRef, useCallback } from "react";
import { 
  X, Play, Pause, RotateCcw, RotateCw, ChevronLeft, ChevronRight, 
  Volume2, VolumeX, Maximize, FolderPlus, FileText, Search, 
  AlertCircle, EyeOff, Bug
} from "lucide-react";
import { apiService, getMediaUrl } from "../services/api";
import type { Camera, Detection } from "../types";

export interface ObservationTarget {
  jobId: string;
  cameraId: string;
  cameraName?: string;
  timestamp: number;
  trackId?: string | number;
  trackUuid?: string;
  cropPath?: string;
  entityType?: string;
  confidence?: number;
  similarity?: number;
  label?: string;
  bbox?: number[]; // [x1, y1, x2, y2]
}

interface ForensicVideoViewerProps {
  isOpen: boolean;
  onClose: () => void;
  observation: ObservationTarget | null;
  cameras: Camera[];
  caseName?: string;
  onSaveToEvidence: (obs: ObservationTarget) => void;
  onGetDetails: (obs: ObservationTarget) => void;
  onFindSimilar: (obs: ObservationTarget) => void;
}

const MAX_TRACK_TIME_DELTA = 0.40; // Enforce maximum temporal tolerance (seconds)

export const ForensicVideoViewer: React.FC<ForensicVideoViewerProps> = ({
  isOpen,
  onClose,
  observation,
  cameras,
  caseName = "Investigation",
  onSaveToEvidence,
  onGetDetails,
  onFindSimilar
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const videoWrapperRef = useRef<HTMLDivElement | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [videoError, setVideoError] = useState<string | null>(null);

  // Track detections strictly belonging to the selected trackUuid
  const [trackDetections, setTrackDetections] = useState<Detection[]>([]);
  
  // Screen-space calculated bounding box style
  const [bboxStyle, setBboxStyle] = useState<{
    left: number;
    top: number;
    width: number;
    height: number;
    rawCoords: [number, number, number, number];
    timeDelta: number;
    matchedTime: number;
  } | null>(null);

  // Diagnostic toggle
  const [showDebug, setShowDebug] = useState(false);

  // Resolve camera source
  const matchedCamera = cameras.find(c => c.id === observation?.cameraId);
  const cameraDisplayName = observation?.cameraName || matchedCamera?.name || observation?.cameraId || "Camera Source";
  const videoSourceUrl = matchedCamera?.source_path ? getMediaUrl(matchedCamera.source_path) : "";

  // Selected track identifier
  const targetTrackUuid = observation?.trackUuid || (typeof observation?.trackId === "string" ? observation.trackId : undefined);
  const shortTrackId = targetTrackUuid ? targetTrackUuid.split("_").pop() : (observation?.trackId || "Target");

  // Load track history strictly for the selected track_uuid
  useEffect(() => {
    if (targetTrackUuid) {
      apiService.getTrackDetections(targetTrackUuid)
        .then(dets => {
          // Double verify all detections strictly belong to targetTrackUuid
          const strictlyMatched = dets.filter(d => d.track_id === targetTrackUuid);
          const sorted = [...strictlyMatched].sort((a, b) => a.timestamp - b.timestamp);
          setTrackDetections(sorted);
        })
        .catch(err => {
          console.warn("Could not load track detections:", err);
          setTrackDetections([]);
        });
    } else {
      setTrackDetections([]);
    }
  }, [targetTrackUuid]);

  // Compute exact letterboxed coordinate mapping
  const calculateScreenBbox = useCallback((rawBbox: [number, number, number, number]) => {
    const video = videoRef.current;
    const wrapper = videoWrapperRef.current;
    if (!video || !wrapper || video.videoWidth === 0 || video.videoHeight === 0) return null;

    const [x1, y1, x2, y2] = rawBbox;
    const vWidth = video.videoWidth;
    const vHeight = video.videoHeight;
    const cWidth = wrapper.clientWidth;
    const cHeight = wrapper.clientHeight;

    const videoRatio = vWidth / vHeight;
    const containerRatio = cWidth / cHeight;

    let renderWidth = cWidth;
    let renderHeight = cHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (containerRatio > videoRatio) {
      // Pillarboxing: black bars on left and right
      renderHeight = cHeight;
      renderWidth = renderHeight * videoRatio;
      offsetX = (cWidth - renderWidth) / 2;
      offsetY = 0;
    } else {
      // Letterboxing: black bars on top and bottom
      renderWidth = cWidth;
      renderHeight = renderWidth / videoRatio;
      offsetX = 0;
      offsetY = (cHeight - renderHeight) / 2;
    }

    const scaleX = renderWidth / vWidth;
    const scaleY = renderHeight / vHeight;

    const boxLeft = offsetX + (Math.min(x1, x2) * scaleX);
    const boxTop = offsetY + (Math.min(y1, y2) * scaleY);
    const boxWidth = Math.abs(x2 - x1) * scaleX;
    const boxHeight = Math.abs(y2 - y1) * scaleY;

    return {
      left: Math.round(boxLeft),
      top: Math.round(boxTop),
      width: Math.max(8, Math.round(boxWidth)),
      height: Math.max(8, Math.round(boxHeight))
    };
  }, []);

  // Update bounding box strictly matching selected track at timestamp t
  const updateBoundingBoxForTime = useCallback((t: number) => {
    if (trackDetections.length === 0) {
      // If we have an initial bbox on the observation and t is near observation.timestamp
      if (observation?.bbox && observation.bbox.length === 4 && Math.abs(observation.timestamp - t) <= MAX_TRACK_TIME_DELTA) {
        const raw = observation.bbox as [number, number, number, number];
        const screen = calculateScreenBbox(raw);
        if (screen) {
          setBboxStyle({
            ...screen,
            rawCoords: raw,
            timeDelta: Math.abs(observation.timestamp - t),
            matchedTime: observation.timestamp
          });
          return;
        }
      }
      setBboxStyle(null);
      return;
    }

    // 1. Find adjacent detections in the selected track history
    let prevDet: Detection | null = null;
    let nextDet: Detection | null = null;

    for (let i = 0; i < trackDetections.length; i++) {
      const d = trackDetections[i];
      if (d.timestamp <= t) {
        prevDet = d;
      }
      if (d.timestamp >= t && !nextDet) {
        nextDet = d;
        break;
      }
    }

    // 2. Linear interpolation between bounding boxes if within consecutive frames
    if (prevDet && nextDet && prevDet.id !== nextDet.id) {
      const dt = nextDet.timestamp - prevDet.timestamp;
      if (dt > 0 && dt <= 1.5) {
        const alpha = Math.max(0, Math.min(1, (t - prevDet.timestamp) / dt));
        const x1 = (1 - alpha) * prevDet.bbox_x1 + alpha * nextDet.bbox_x1;
        const y1 = (1 - alpha) * prevDet.bbox_y1 + alpha * nextDet.bbox_y1;
        const x2 = (1 - alpha) * prevDet.bbox_x2 + alpha * nextDet.bbox_x2;
        const y2 = (1 - alpha) * prevDet.bbox_y2 + alpha * nextDet.bbox_y2;
        const rawCoords: [number, number, number, number] = [x1, y1, x2, y2];
        const screen = calculateScreenBbox(rawCoords);
        if (screen) {
          setBboxStyle({
            ...screen,
            rawCoords,
            timeDelta: 0,
            matchedTime: t
          });
          return;
        }
      }
    }

    // 3. Fallback to nearest detection strictly within MAX_TRACK_TIME_DELTA
    let bestDet: Detection | null = null;
    let minDelta = Infinity;

    for (const d of trackDetections) {
      const delta = Math.abs(d.timestamp - t);
      if (delta < minDelta) {
        minDelta = delta;
        bestDet = d;
      }
    }

    if (bestDet && minDelta <= MAX_TRACK_TIME_DELTA) {
      const rawCoords: [number, number, number, number] = [bestDet.bbox_x1, bestDet.bbox_y1, bestDet.bbox_x2, bestDet.bbox_y2];
      const screen = calculateScreenBbox(rawCoords);
      if (screen) {
        setBboxStyle({
          ...screen,
          rawCoords,
          timeDelta: minDelta,
          matchedTime: bestDet.timestamp
        });
        return;
      }
    }

    // If target track is not present at this frame, hide the box!
    setBboxStyle(null);
  }, [trackDetections, observation, calculateScreenBbox]);

  // Initial seek & setup when observation changes
  useEffect(() => {
    if (observation && videoRef.current) {
      setVideoError(null);
      const targetTime = Math.max(0, observation.timestamp);

      const applySeek = () => {
        if (videoRef.current) {
          videoRef.current.currentTime = targetTime;
          videoRef.current.pause();
          setIsPlaying(false);
          setCurrentTime(targetTime);
          updateBoundingBoxForTime(targetTime);
        }
      };

      if (videoRef.current.readyState >= 1) {
        applySeek();
      } else {
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            setDuration(videoRef.current.duration || 0);
            applySeek();
          }
        };
      }
    }
  }, [observation, videoSourceUrl, updateBoundingBoxForTime]);

  // Continuous tracking on playback time update
  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const t = videoRef.current.currentTime;
    setCurrentTime(t);
    updateBoundingBoxForTime(t);
  };

  // Window resize re-alignment
  useEffect(() => {
    const handleResize = () => {
      if (videoRef.current) {
        updateBoundingBoxForTime(videoRef.current.currentTime);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [updateBoundingBoxForTime]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const jumpSeconds = (delta: number) => {
    if (!videoRef.current) return;
    const newTime = Math.max(0, Math.min(duration || 9999, videoRef.current.currentTime + delta));
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    updateBoundingBoxForTime(newTime);
  };

  const jumpToDetection = (direction: "prev" | "next") => {
    if (!videoRef.current || trackDetections.length === 0) return;
    const t = videoRef.current.currentTime;

    if (direction === "prev") {
      const prevDets = trackDetections.filter(d => d.timestamp < t - 0.2);
      if (prevDets.length > 0) {
        const target = prevDets[prevDets.length - 1];
        videoRef.current.currentTime = target.timestamp;
        setCurrentTime(target.timestamp);
        updateBoundingBoxForTime(target.timestamp);
      }
    } else {
      const nextDets = trackDetections.filter(d => d.timestamp > t + 0.2);
      if (nextDets.length > 0) {
        const target = nextDets[0];
        videoRef.current.currentTime = target.timestamp;
        setCurrentTime(target.timestamp);
        updateBoundingBoxForTime(target.timestamp);
      }
    }
  };

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current || duration <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    const newTime = Math.max(0, Math.min(duration, pos * duration));
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
    updateBoundingBoxForTime(newTime);
  };

  const formatTimestamp = (secs: number) => {
    if (isNaN(secs) || secs < 0) return "00:00:00.000";
    const hrs = Math.floor(secs / 3600);
    const mins = Math.floor((secs % 3600) / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 1000);
    return `${hrs.toString().padStart(2, "0")}:${mins.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms.toString().padStart(3, "0")}`;
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      containerRef.current.requestFullscreen().catch(() => {});
    }
  };

  const cycleSpeed = () => {
    const speeds = [0.5, 1, 2];
    const nextIdx = (speeds.indexOf(playbackRate) + 1) % speeds.length;
    const nextSpeed = speeds[nextIdx];
    setPlaybackRate(nextSpeed);
    if (videoRef.current) {
      videoRef.current.playbackRate = nextSpeed;
    }
  };

  if (!isOpen || !observation) return null;

  return (
    <div 
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.85)",
        backdropFilter: "blur(6px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 99999,
        padding: "24px"
      }}
      onClick={onClose}
    >
      <div 
        ref={containerRef}
        className="card"
        style={{
          maxWidth: "1100px",
          width: "100%",
          maxHeight: "92vh",
          display: "flex",
          flexDirection: "column",
          padding: "24px",
          backgroundColor: "var(--bg-card)",
          border: "1px solid var(--border-color)",
          borderRadius: "var(--radius-main)",
          overflowY: "auto",
          gap: "16px"
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Top Bar */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "12px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "11px", fontWeight: "700", color: "var(--color-accent)", letterSpacing: "0.5px" }}>
                FORENSIC VIDEO VIEWER
              </span>
              <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>· Case: {caseName}</span>
            </div>
            <h2 style={{ fontSize: "17px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
              {cameraDisplayName}
            </h2>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ fontSize: "12px", fontFamily: "monospace", color: "var(--color-accent)", backgroundColor: "var(--bg-input)", padding: "4px 8px", borderRadius: "4px", border: "1px solid var(--border-color)" }}>
              Observation Time: {formatTimestamp(observation.timestamp)}
            </span>
            <button 
              type="button"
              className="btn btn-secondary"
              style={{ padding: "4px 8px", fontSize: "11px", display: "flex", alignItems: "center", gap: "4px" }}
              onClick={() => setShowDebug(!showDebug)}
              title="Toggle Target Tracking Diagnostics"
            >
              <Bug size={13} /> {showDebug ? "Hide Diagnostics" : "Diagnostics"}
            </button>
            <button 
              type="button"
              onClick={onClose}
              style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", padding: "4px" }}
              title="Close Viewer"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Video Area & Exact Coordinate Bounding Box Overlay */}
        <div 
          ref={videoWrapperRef}
          style={{ 
            width: "100%", 
            height: "440px", 
            backgroundColor: "#000", 
            borderRadius: "8px", 
            overflow: "hidden", 
            position: "relative",
            display: "flex",
            alignItems: "center",
            justifyContent: "center"
          }}
        >
          {videoSourceUrl ? (
            <video 
              ref={videoRef}
              src={videoSourceUrl}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={() => {
                if (videoRef.current) {
                  setDuration(videoRef.current.duration || 0);
                  videoRef.current.currentTime = observation.timestamp;
                  videoRef.current.pause();
                  setIsPlaying(false);
                  updateBoundingBoxForTime(observation.timestamp);
                }
              }}
              onError={() => {
                setVideoError("Unable to stream source video from local storage.");
              }}
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
              playsInline
            />
          ) : (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px", color: "var(--text-muted)" }}>
              <AlertCircle size={36} style={{ color: "var(--color-warning)" }} />
              <span>Video source path not registered for this camera.</span>
            </div>
          )}

          {/* Synchronized Selected Target Bounding Box Overlay */}
          {bboxStyle ? (
            <div 
              style={{
                position: "absolute",
                left: `${bboxStyle.left}px`,
                top: `${bboxStyle.top}px`,
                width: `${bboxStyle.width}px`,
                height: `${bboxStyle.height}px`,
                border: "2.5px solid #06B6D4",
                boxShadow: "0 0 12px rgba(6, 182, 212, 0.6), inset 0 0 8px rgba(6, 182, 212, 0.2)",
                pointerEvents: "none",
                borderRadius: "3px",
                transition: "all 0.06s linear",
                zIndex: 10
              }}
            >
              {/* Target Label Tag */}
              <div 
                style={{
                  position: "absolute",
                  top: "-22px",
                  left: "-2px",
                  backgroundColor: "rgba(6, 182, 212, 0.95)",
                  color: "#0B0F19",
                  fontSize: "10px",
                  fontWeight: "800",
                  letterSpacing: "0.4px",
                  padding: "1px 6px",
                  borderRadius: "3px 3px 0 0",
                  whiteSpace: "nowrap",
                  textTransform: "uppercase"
                }}
              >
                TARGET #{shortTrackId} · {observation.entityType || "PERSON"}
              </div>
            </div>
          ) : (
            /* Target Not Visible Badge when outside track detection window */
            <div 
              style={{
                position: "absolute",
                top: "12px",
                left: "12px",
                backgroundColor: "rgba(17, 24, 39, 0.85)",
                border: "1px solid var(--border-color)",
                color: "var(--text-secondary)",
                fontSize: "11px",
                fontWeight: "600",
                padding: "4px 10px",
                borderRadius: "6px",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                zIndex: 5
              }}
            >
              <EyeOff size={13} style={{ color: "var(--text-muted)" }} />
              <span>Target not visible in current frame</span>
            </div>
          )}

          {/* Diagnostic Overlay */}
          {showDebug && (
            <div 
              style={{
                position: "absolute",
                bottom: "12px",
                left: "12px",
                backgroundColor: "rgba(11, 15, 25, 0.92)",
                border: "1px solid var(--border-color)",
                borderRadius: "6px",
                padding: "10px 12px",
                fontSize: "10px",
                fontFamily: "monospace",
                color: "#E2E8F0",
                zIndex: 20,
                lineHeight: "1.5",
                pointerEvents: "none"
              }}
            >
              <div style={{ color: "var(--color-accent)", fontWeight: "700", marginBottom: "2px" }}>TARGET TRACK DIAGNOSTICS</div>
              <div>Selected Track UUID: {targetTrackUuid || "None"}</div>
              <div>Current Playback Time: {currentTime.toFixed(3)}s</div>
              <div>Track Keyframes Indexed: {trackDetections.length}</div>
              <div>Target Visible: <strong style={{ color: bboxStyle ? "var(--color-success)" : "var(--color-warning)" }}>{bboxStyle ? "YES" : "NO"}</strong></div>
              {bboxStyle && (
                <>
                  <div>Matched Obs Time: {bboxStyle.matchedTime.toFixed(3)}s (delta: {bboxStyle.timeDelta.toFixed(3)}s)</div>
                  <div>Raw Video BBox: [{bboxStyle.rawCoords.map(c => Math.round(c)).join(", ")}]</div>
                  <div>Screen Box: [L:{bboxStyle.left}, T:{bboxStyle.top}, W:{bboxStyle.width}, H:{bboxStyle.height}]</div>
                </>
              )}
            </div>
          )}

          {videoError && (
            <div style={{ position: "absolute", inset: 0, backgroundColor: "rgba(0,0,0,0.85)", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "10px", color: "var(--color-danger)" }}>
              <AlertCircle size={32} />
              <span style={{ fontSize: "14px", fontWeight: "500" }}>{videoError}</span>
            </div>
          )}
        </div>

        {/* Interactive Forensic Timeline */}
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <div 
            style={{ 
              width: "100%", 
              height: "22px", 
              backgroundColor: "var(--bg-input)", 
              borderRadius: "4px", 
              position: "relative",
              cursor: "pointer",
              border: "1px solid var(--border-color)",
              overflow: "hidden"
            }}
            onClick={handleTimelineClick}
          >
            {/* Progress fill */}
            <div 
              style={{ 
                width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%`, 
                height: "100%", 
                backgroundColor: "rgba(6, 182, 212, 0.25)" 
              }} 
            />

            {/* Matched Search Observation Marker */}
            {duration > 0 && (
              <div 
                style={{
                  position: "absolute",
                  top: 0,
                  bottom: 0,
                  left: `${(observation.timestamp / duration) * 100}%`,
                  width: "4px",
                  backgroundColor: "var(--color-accent)",
                  boxShadow: "0 0 8px #06B6D4",
                  zIndex: 4
                }}
                title={`Matched Search Observation: ${formatTimestamp(observation.timestamp)}`}
              />
            )}

            {/* Selected track detection keyframe markers */}
            {duration > 0 && trackDetections.map((d, idx) => (
              <div 
                key={idx}
                style={{
                  position: "absolute",
                  top: "4px",
                  bottom: "4px",
                  left: `${(d.timestamp / duration) * 100}%`,
                  width: "2px",
                  backgroundColor: "rgba(16, 185, 129, 0.8)",
                  zIndex: 2
                }}
                title={`Track #${shortTrackId} Keyframe #${idx + 1}: ${formatTimestamp(d.timestamp)}`}
              />
            ))}

            {/* Current Playback Needle */}
            {duration > 0 && (
              <div 
                style={{
                  position: "absolute",
                  top: 0,
                  bottom: 0,
                  left: `${(currentTime / duration) * 100}%`,
                  width: "2px",
                  backgroundColor: "#FFFFFF",
                  zIndex: 5
                }}
              />
            )}
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", fontFamily: "monospace", color: "var(--text-secondary)" }}>
            <span>Current: <strong style={{ color: "var(--text-primary)" }}>{formatTimestamp(currentTime)}</strong></span>
            <span>Duration: <strong style={{ color: "var(--text-primary)" }}>{formatTimestamp(duration)}</strong></span>
          </div>
        </div>

        {/* Video Control Bar */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px", padding: "10px 14px", backgroundColor: "var(--bg-input)", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
          {/* Jump / Detection Steps */}
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <button 
              type="button" 
              className="btn btn-secondary" 
              style={{ fontSize: "11px", padding: "6px 10px" }}
              onClick={() => jumpToDetection("prev")}
              disabled={trackDetections.length === 0}
              title="Jump to Previous Target Detection"
            >
              <ChevronLeft size={14} /> Prev Target Det
            </button>

            <button 
              type="button" 
              className="btn btn-secondary" 
              style={{ fontSize: "11px", padding: "6px 10px" }}
              onClick={() => jumpSeconds(-10)}
            >
              <RotateCcw size={13} style={{ marginRight: "4px" }} /> -10s
            </button>

            <button 
              type="button" 
              className="btn btn-secondary" 
              style={{ fontSize: "11px", padding: "6px 10px" }}
              onClick={() => jumpSeconds(-5)}
            >
              -5s
            </button>

            {/* Play / Pause */}
            <button 
              type="button" 
              className="btn btn-primary" 
              style={{ padding: "8px 16px", fontSize: "13px", fontWeight: "600" }}
              onClick={togglePlay}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />} {isPlaying ? "Pause" : "Play"}
            </button>

            <button 
              type="button" 
              className="btn btn-secondary" 
              style={{ fontSize: "11px", padding: "6px 10px" }}
              onClick={() => jumpSeconds(5)}
            >
              +5s
            </button>

            <button 
              type="button" 
              className="btn btn-secondary" 
              style={{ fontSize: "11px", padding: "6px 10px" }}
              onClick={() => jumpSeconds(10)}
            >
              <RotateCw size={13} style={{ marginRight: "4px" }} /> +10s
            </button>

            <button 
              type="button" 
              className="btn btn-secondary" 
              style={{ fontSize: "11px", padding: "6px 10px" }}
              onClick={() => jumpToDetection("next")}
              disabled={trackDetections.length === 0}
              title="Jump to Next Target Detection"
            >
              Next Target Det <ChevronRight size={14} />
            </button>
          </div>

          {/* Right Controls: Speed, Audio, Fullscreen */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button 
              type="button" 
              className="btn btn-secondary"
              style={{ fontSize: "11px", padding: "6px 10px" }}
              onClick={cycleSpeed}
            >
              {playbackRate}x
            </button>

            <button 
              type="button" 
              className="btn btn-secondary"
              style={{ padding: "6px 10px" }}
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.muted = !isMuted;
                  setIsMuted(!isMuted);
                }
              }}
            >
              {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
            </button>

            <button 
              type="button" 
              className="btn btn-secondary"
              style={{ padding: "6px 10px" }}
              onClick={toggleFullscreen}
            >
              <Maximize size={14} />
            </button>
          </div>
        </div>

        {/* Observation Metadata & Primary Actions */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", paddingTop: "6px" }}>
          {/* Metadata Block */}
          <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "12px", padding: "14px", borderRadius: "8px", backgroundColor: "var(--bg-input)", border: "1px solid var(--border-color)" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Target Track:</span>
              <strong style={{ color: "var(--color-accent)" }}>Track #{shortTrackId}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Object Type:</span>
              <strong style={{ color: "var(--text-primary)", textTransform: "capitalize" }}>{observation.entityType || "Person"}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Camera Source:</span>
              <strong style={{ color: "var(--text-primary)" }}>{cameraDisplayName}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Detection Confidence:</span>
              <strong style={{ color: "var(--color-success)" }}>{observation.confidence ? `${(observation.confidence * 100).toFixed(0)}%` : "0.85"}</strong>
            </div>
            {observation.similarity !== undefined && (
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Search Similarity Score:</span>
                <strong style={{ color: "var(--color-accent)" }}>{(observation.similarity * 100).toFixed(0)}%</strong>
              </div>
            )}
          </div>

          {/* Primary Action Buttons */}
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", justifyContent: "center" }}>
            <button 
              type="button" 
              className="btn btn-primary"
              style={{ padding: "10px", fontSize: "13px", fontWeight: "600", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}
              onClick={() => onSaveToEvidence(observation)}
            >
              <FolderPlus size={15} /> Save Observation to Evidence
            </button>

            <button 
              type="button" 
              className="btn btn-secondary"
              style={{ padding: "10px", fontSize: "13px", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}
              onClick={() => onGetDetails(observation)}
            >
              <FileText size={15} /> Get Details (Subject Summary)
            </button>

            <button 
              type="button" 
              className="btn btn-secondary"
              style={{ padding: "10px", fontSize: "13px", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}
              onClick={() => onFindSimilar(observation)}
            >
              <Search size={15} /> Find Similar Observations
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

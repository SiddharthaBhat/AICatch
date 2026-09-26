import React, { useState, useEffect } from "react";
import { 
  Layers, Search, AlertTriangle, FileText, RefreshCw, Eye, Film, Play,
  ChevronDown, ChevronUp, Download, ArrowRight, ShieldCheck, Camera as CameraIcon
} from "lucide-react";
import type { Camera, StitchResult, StitchSegment } from "../types";


import { apiService, getMediaUrl } from "../services/api";

interface JourneyViewProps {
  jobId: string;
  cameras: Camera[];
  targetSubject: {
    trackUuid: string;
    trackId?: string | number;
    cameraId?: string;
    cameraName?: string;
    timestamp?: number;
    cropPath?: string;
    entityType?: string;
    label?: string;
  } | null;
  onSelectTarget: (subj: any) => void;
  onOpenVideo: (obs: any) => void;
  onGenerateSummary: () => void;
  onGoToSearch: () => void;
}

export const JourneyView: React.FC<JourneyViewProps> = ({
  jobId,
  cameras,
  targetSubject,
  onOpenVideo,
  onGenerateSummary,
  onGoToSearch
}) => {
  const [loading, setLoading] = useState(false);
  const [stitchResult, setStitchResult] = useState<StitchResult | null>(null);
  const [minContinuity, setMinContinuity] = useState(0.45);
  const [maxGapSec, setMaxGapSec] = useState(600.0);
  const [expandedSegments, setExpandedSegments] = useState<Record<string, boolean>>({});

  const fetchStitchJourney = async () => {
    if (!targetSubject) return;
    setLoading(true);
    try {
      // Execute unified Stitch Engine
      const res = await apiService.reconstructStitch(jobId, {
        target_entity: targetSubject.entityType || "person",
        track_uuid: targetSubject.trackUuid,
        min_continuity_score: minContinuity,
        max_gap_seconds: maxGapSec
      });
      setStitchResult(res);
    } catch (err) {
      console.error("Failed to reconstruct stitch journey:", err);
      // Construct honest fallback result from targetSubject
      const fallbackSegment: StitchSegment = {
        segment_id: `seg_${targetSubject.trackUuid}_init`,
        camera_id: targetSubject.cameraId || "unknown",
        camera_name: targetSubject.cameraName || "Origin Camera",
        start_time: targetSubject.timestamp || 0.0,
        end_time: (targetSubject.timestamp || 0.0) + 2.0,
        duration: 2.0,
        occurrence_id: null,
        representative_observation_id: targetSubject.trackUuid,
        representative_crop_path: targetSubject.cropPath,
        observations_count: 1,
        observations: [{
          observation_id: targetSubject.trackUuid,
          job_id: jobId,
          camera_id: targetSubject.cameraId || "unknown",
          camera_name: targetSubject.cameraName,
          timestamp: targetSubject.timestamp || 0.0,
          end_timestamp: (targetSubject.timestamp || 0.0) + 2.0,
          source_type: "object",
          entity_type: targetSubject.entityType || "person",
          track_id: targetSubject.trackId,
          crop_path: targetSubject.cropPath,
          combined_score: 1.0
        }],
        continuity: {
          score: 1.0,
          reid_similarity: null,
          visual_similarity: null,
          temporal_score: 1.0,
          transition_score: 1.0,
          topology_available: false,
          reason: "Origin observation of selected subject"
        },
        provenance: {
          job_id: jobId,
          camera_id: targetSubject.cameraId || "unknown",
          camera_name: targetSubject.cameraName,
          source_type: "object",
          crop_path: targetSubject.cropPath,
          timestamp: targetSubject.timestamp || 0.0,
          track_id: targetSubject.trackId
        }
      };

      setStitchResult({
        stitch_id: `stitch_fallback_${Date.now()}`,
        investigation_id: jobId,
        job_id: jobId,
        entity_type: (targetSubject.entityType as any) || "person",
        status: "insufficient_evidence",
        reason: "Network error or single observation provided; cross-camera transitions could not be computed.",
        created_at: Date.now() / 1000,
        target_summary: { track_uuid: targetSubject.trackUuid },
        segments: [fallbackSegment],
        transitions: [],
        camera_diversity: 1,
        total_duration_sec: 2.0
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (targetSubject) {
      fetchStitchJourney();
    }
  }, [targetSubject, minContinuity, maxGapSec]);

  const toggleSegmentExpanded = (segId: string) => {
    setExpandedSegments(prev => ({
      ...prev,
      [segId]: !prev[segId]
    }));
  };

  const formatSec = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 10);
    return `${mins.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms}s`;
  };

  const exportStitchJson = () => {
    if (!stitchResult) return;
    const blob = new Blob([JSON.stringify(stitchResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `stitch_reconstruction_${jobId}_${targetSubject?.trackUuid || "subject"}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!targetSubject) {
    return (
      <div className="card" style={{ padding: "48px 24px", textAlign: "center", maxWidth: "600px", margin: "20px auto", display: "flex", flexDirection: "column", alignItems: "center", gap: "14px" }}>
        <Layers size={40} style={{ color: "var(--text-muted)" }} />
        <div>
          <h3 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "4px" }}>
            No Target Subject Selected
          </h3>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", maxWidth: "420px", lineHeight: "1.5" }}>
            Cross-camera evidence stitching requires selecting a target observation or track candidate from Search to trace its chronological multi-camera journey.
          </p>
        </div>
        <button 
          type="button" 
          className="btn btn-primary"
          onClick={onGoToSearch}
          style={{ display: "flex", alignItems: "center", gap: "6px" }}
        >
          <Search size={14} /> Search for Target Subject
        </button>
      </div>
    );
  }

  const segments = stitchResult?.segments || [];
  const transitions = stitchResult?.transitions || [];
  const isInsufficient = stitchResult?.status === "insufficient_evidence";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }} className="animate-fade-in">
      {/* Target Subject Top Banner */}
      <div className="card" style={{ padding: "18px 22px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "16px", border: "1px solid var(--color-accent)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {targetSubject.cropPath && (
            <img 
              src={getMediaUrl(targetSubject.cropPath)} 
              alt="Target Subject" 
              style={{ width: "56px", height: "56px", objectFit: "cover", borderRadius: "8px", border: "1px solid var(--border-color)" }}
            />
          )}
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "10px", fontWeight: "700", color: "var(--color-accent)", letterSpacing: "0.5px" }}>
                CROSS-CAMERA STITCH TARGET
              </span>
              <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                · Track #{targetSubject.trackId || 1}
              </span>
              <span style={{ 
                fontSize: "10px", 
                fontWeight: "600",
                padding: "2px 6px",
                borderRadius: "4px",
                backgroundColor: isInsufficient ? "rgba(245, 158, 11, 0.15)" : "rgba(16, 185, 129, 0.15)",
                color: isInsufficient ? "var(--color-warning)" : "var(--color-success)"
              }}>
                {stitchResult?.status?.toUpperCase() || "PROCESSING"}
              </span>
            </div>
            <h2 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
              {targetSubject.label || `Subject Observation (${targetSubject.cameraName})`}
            </h2>
            <div style={{ fontSize: "11px", color: "var(--text-secondary)" }}>
              First Sighted: {targetSubject.cameraName} @ {formatSec(targetSubject.timestamp || 0)}
              {stitchResult && ` · ${segments.length} Camera Segments · ${transitions.length} Cross-Camera Transitions`}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px" }}>
            <span style={{ color: "var(--text-muted)" }}>Min Continuity:</span>
            <select 
              className="form-select" 
              value={minContinuity}
              onChange={(e) => setMinContinuity(parseFloat(e.target.value))}
              style={{ padding: "4px 8px", fontSize: "11px" }}
            >
              <option value="0.65">Strict (0.65)</option>
              <option value="0.45">Standard (0.45)</option>
              <option value="0.35">Permissive (0.35)</option>
            </select>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px" }}>
            <span style={{ color: "var(--text-muted)" }}>Max Gap:</span>
            <select 
              className="form-select" 
              value={maxGapSec}
              onChange={(e) => setMaxGapSec(parseFloat(e.target.value))}
              style={{ padding: "4px 8px", fontSize: "11px" }}
            >
              <option value="300">5 min</option>
              <option value="600">10 min</option>
              <option value="1800">30 min</option>
              <option value="3600">1 hour</option>
            </select>
          </div>

          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={fetchStitchJourney}
            disabled={loading}
            style={{ fontSize: "11px", padding: "6px 10px", display: "flex", alignItems: "center", gap: "4px" }}
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} /> Re-Stitch
          </button>

          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={exportStitchJson}
            disabled={!stitchResult}
            style={{ fontSize: "11px", padding: "6px 10px", display: "flex", alignItems: "center", gap: "4px" }}
            title="Export Stitch forensic JSON dossier"
          >
            <Download size={12} /> Export JSON
          </button>

          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={async () => {
              if (!targetSubject) return;
              try {
                const res = await apiService.mergeJourneyVideo(jobId, targetSubject.trackUuid, minContinuity, 3.0);
                alert(`Successfully merged cross-camera journey video: ${res.filename} (${res.segments_count} camera transitions). Saved to Evidence.`);
              } catch (err: any) {
                alert(err.message || "Failed to merge journey video");
              }
            }}
            style={{ fontSize: "11px", padding: "6px 10px", display: "flex", alignItems: "center", gap: "4px" }}
          >
            <Film size={12} /> Merge Video
          </button>

          <button 
            type="button" 
            className="btn btn-primary"
            onClick={onGenerateSummary}
            style={{ fontSize: "11px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "6px", fontWeight: "600" }}
          >
            <FileText size={13} /> Subject Dossier
          </button>
        </div>
      </div>

      {/* Forensic Transparency: Insufficient Evidence Alert Banner */}
      {isInsufficient && (
        <div style={{
          backgroundColor: "rgba(245, 158, 11, 0.08)",
          border: "1px solid var(--color-warning)",
          borderRadius: "8px",
          padding: "16px 20px",
          display: "flex",
          alignItems: "flex-start",
          gap: "12px"
        }}>
          <AlertTriangle size={20} style={{ color: "var(--color-warning)", flexShrink: 0, marginTop: "2px" }} />
          <div style={{ fontSize: "13px" }}>
            <strong style={{ color: "var(--text-primary)", display: "block", marginBottom: "3px" }}>
              Forensic Evidence Notice: Single Camera Observation
            </strong>
            <span style={{ color: "var(--text-secondary)", lineHeight: "1.5" }}>
              {stitchResult?.reason || "Observations for this subject are currently confined to a single camera stream. No verified cross-camera transitions met the continuity threshold. AICatch presents only verified evidence without fabricating synthetic transitions."}
            </span>
          </div>
        </div>
      )}

      {/* Unified Chronological Stitch Timeline */}
      <div className="card" style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
          <div>
            <h3 style={{ fontSize: "15px", fontWeight: "600", color: "var(--text-primary)" }}>
              Reconstructed Multi-Camera Journey Timeline ({segments.length} Occurrence Segments)
            </h3>
            <p style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
              Multi-signal continuity evaluated using Re-ID cosine similarity, visual features, temporal transit windows, and physical movement plausibility.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span className="badge badge-completed" style={{ fontSize: "11px" }}>
              <ShieldCheck size={12} style={{ marginRight: "4px" }} /> Multi-Signal Provenance
            </span>
            <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
              Diversity: {stitchResult?.camera_diversity || 1} Cam{(stitchResult?.camera_diversity || 1) > 1 ? "s" : ""}
            </span>
          </div>
        </div>

        {/* Timeline Sequence */}
        <div style={{ display: "flex", flexDirection: "column", gap: "0", position: "relative" }}>
          {segments.map((seg, idx) => {
            const camName = seg.camera_name || cameras.find(c => c.id === seg.camera_id)?.name || seg.camera_id;
            const isExpanded = !!expandedSegments[seg.segment_id];
            const transition = transitions[idx]; // Transition to next segment if exists

            return (
              <React.Fragment key={seg.segment_id || idx}>
                {/* 1. Camera Occurrence Segment Card */}
                <div 
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    padding: "16px",
                    borderRadius: "8px",
                    backgroundColor: "var(--bg-input)",
                    border: "1px solid var(--border-color)",
                    gap: "12px",
                    transition: "all 0.15s ease"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
                    {/* Left: Crop + Meta */}
                    <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
                      {seg.representative_crop_path ? (
                        <img 
                          src={getMediaUrl(seg.representative_crop_path)} 
                          alt="Segment Crop" 
                          style={{ width: "48px", height: "48px", objectFit: "cover", borderRadius: "6px", border: "1px solid var(--border-color)" }}
                        />
                      ) : (
                        <div style={{ width: "48px", height: "48px", backgroundColor: "#000", borderRadius: "6px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                          <CameraIcon size={20} style={{ color: "var(--text-muted)" }} />
                        </div>
                      )}

                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                          <span style={{ 
                            fontSize: "10px", 
                            fontWeight: "700", 
                            color: "var(--color-accent)",
                            backgroundColor: "rgba(6, 182, 212, 0.1)",
                            padding: "2px 6px",
                            borderRadius: "4px"
                          }}>
                            SEGMENT {idx + 1}
                          </span>
                          <strong style={{ fontSize: "14px", color: "var(--text-primary)" }}>
                            {camName}
                          </strong>
                          <span style={{ fontSize: "12px", fontFamily: "monospace", color: "var(--color-accent)" }}>
                            {formatSec(seg.start_time)} – {formatSec(seg.end_time)}
                          </span>
                          <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                            ({seg.duration.toFixed(1)}s duration)
                          </span>
                        </div>

                        {/* Signal Breakdown Badges */}
                        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "4px", flexWrap: "wrap" }}>
                          <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>
                            Score: <strong style={{ color: "var(--color-accent)" }}>{seg.continuity.score.toFixed(3)}</strong>
                          </span>
                          {seg.continuity.reid_similarity !== null && seg.continuity.reid_similarity !== undefined && (
                            <span style={{ fontSize: "10px", color: "var(--color-success)" }}>
                              Re-ID Sim: <strong>{seg.continuity.reid_similarity.toFixed(3)}</strong>
                            </span>
                          )}
                          {seg.continuity.visual_similarity !== null && seg.continuity.visual_similarity !== undefined && (
                            <span style={{ fontSize: "10px", color: "#38bdf8" }}>
                              CLIP Sim: <strong>{seg.continuity.visual_similarity.toFixed(3)}</strong>
                            </span>
                          )}
                          <span style={{ fontSize: "10px", color: "var(--text-secondary)" }}>
                            · {seg.observations_count} detection{seg.observations_count > 1 ? "s" : ""} collapsed
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Right: Actions */}
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <button 
                        type="button"
                        className="btn btn-secondary"
                        style={{ fontSize: "11px", padding: "6px 10px", display: "flex", alignItems: "center", gap: "4px" }}
                        onClick={() => toggleSegmentExpanded(seg.segment_id)}
                      >
                        {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                        {isExpanded ? "Hide Details" : "Inspect Detections"}
                      </button>

                      <button 
                        type="button"
                        className="btn btn-primary"
                        style={{ fontSize: "11px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "4px" }}
                        onClick={() => onOpenVideo({
                          jobId,
                          cameraId: seg.camera_id,
                          cameraName: camName,
                          timestamp: seg.start_time,
                          trackUuid: seg.representative_observation_id,
                          cropPath: seg.representative_crop_path
                        })}
                      >
                        <Eye size={12} /> View Video
                      </button>
                    </div>
                  </div>

                  {/* Expanded Observation Drill-Down */}
                  {isExpanded && seg.observations && seg.observations.length > 0 && (
                    <div style={{
                      marginTop: "8px",
                      paddingTop: "12px",
                      borderTop: "1px solid var(--border-color)",
                      display: "flex",
                      flexDirection: "column",
                      gap: "8px"
                    }}>
                      <span style={{ fontSize: "11px", fontWeight: "600", color: "var(--text-secondary)" }}>
                        Segment Observation Log ({seg.observations.length} items):
                      </span>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "8px" }}>
                        {seg.observations.map((obs, obsIdx) => (
                          <div 
                            key={obs.observation_id || obsIdx}
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                              padding: "8px 10px",
                              backgroundColor: "var(--bg-card)",
                              borderRadius: "6px",
                              border: "1px solid var(--border-color)"
                            }}
                          >
                            {obs.crop_path && (
                              <img 
                                src={getMediaUrl(obs.crop_path)} 
                                alt="Obs crop" 
                                style={{ width: "32px", height: "32px", objectFit: "cover", borderRadius: "4px" }}
                              />
                            )}
                            <div style={{ fontSize: "11px", flex: 1, minWidth: 0 }}>
                              <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <strong style={{ color: "var(--text-primary)" }}>{formatSec(obs.timestamp)}</strong>
                                <span style={{ color: "var(--text-muted)", fontSize: "10px" }}>{obs.source_type}</span>
                              </div>
                              <div style={{ color: "var(--text-secondary)", fontSize: "10px" }}>
                                {obs.reid_similarity !== null && obs.reid_similarity !== undefined ? `Re-ID: ${obs.reid_similarity.toFixed(3)} ` : ""}
                                {obs.visual_similarity !== null && obs.visual_similarity !== undefined ? `CLIP: ${obs.visual_similarity.toFixed(3)}` : ""}
                              </div>
                            </div>
                            <button 
                              type="button"
                              className="btn btn-secondary"
                              style={{ fontSize: "10px", padding: "3px 6px" }}
                              onClick={() => onOpenVideo({
                                jobId,
                                cameraId: obs.camera_id,
                                cameraName: camName,
                                timestamp: obs.timestamp,
                                trackUuid: obs.observation_id,
                                cropPath: obs.crop_path,
                                bbox: obs.bbox
                              })}
                              title="Play video at this detection"
                            >
                              <Play size={10} /> Play
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* 2. Inter-Camera Transition Connector */}
                {transition && (
                  <div style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "10px 0",
                    position: "relative"
                  }}>
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      padding: "6px 14px",
                      borderRadius: "20px",
                      backgroundColor: "rgba(6, 182, 212, 0.08)",
                      border: "1px dashed var(--color-accent)",
                      fontSize: "11px"
                    }}>
                      <ArrowRight size={13} style={{ color: "var(--color-accent)" }} />
                      <span style={{ color: "var(--text-secondary)" }}>
                        Cross-Camera Transit: <strong>{transition.from_camera_name}</strong> → <strong>{transition.to_camera_name}</strong>
                      </span>
                      <span style={{ 
                        fontFamily: "monospace", 
                        fontWeight: "600",
                        color: transition.transit_time_sec >= 0 ? "var(--color-success)" : "var(--color-danger)" 
                      }}>
                        +{transition.transit_time_sec.toFixed(1)}s
                      </span>
                      <span style={{ color: "var(--text-muted)", fontSize: "10px" }}>
                        (Transition Score: {transition.score.toFixed(3)})
                      </span>
                    </div>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
};

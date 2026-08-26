import React, { useState, useEffect } from "react";
import { 
  Layers, Search, CheckCircle2, AlertTriangle, FileText, RefreshCw, Eye, Film
} from "lucide-react";
import type { Camera } from "../types";
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
  const [confirmedJourney, setConfirmedJourney] = useState<any[]>([]);
  const [possibleTransitions, setPossibleTransitions] = useState<any[]>([]);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.70);

  const fetchJourney = async () => {
    if (!targetSubject) return;
    setLoading(true);
    try {
      const data: any = await apiService.getJourneyTimeline(jobId, targetSubject.trackUuid, similarityThreshold);
      if (data && typeof data === "object" && !Array.isArray(data)) {
        setConfirmedJourney(data.confirmed_journey || []);
        setPossibleTransitions(data.possible_transitions || []);
      } else if (Array.isArray(data)) {
        // Fallback for legacy list response
        setConfirmedJourney(data);
        setPossibleTransitions([]);
      }
    } catch (err) {
      console.error("Failed to reconstruct journey:", err);
      // Construct honest initial point from target
      setConfirmedJourney([{
        track_uuid: targetSubject.trackUuid,
        camera_id: targetSubject.cameraId,
        camera_name: targetSubject.cameraName,
        timestamp: targetSubject.timestamp || 0,
        duration: 2.0,
        confidence: 1.0,
        confidence_tier: "HIGH",
        reason: "Origin of selected subject",
        crop_path: targetSubject.cropPath
      }]);
      setPossibleTransitions([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (targetSubject) {
      fetchJourney();
    }
  }, [targetSubject, similarityThreshold]);

  const formatSec = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${mins.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
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
            Journey reconstruction requires selecting a target observation from Search to trace cross-camera transitions.
          </p>
        </div>
        <button 
          type="button" 
          className="btn btn-primary"
          onClick={onGoToSearch}
          style={{ display: "flex", alignItems: "center", gap: "6px" }}
        >
          <Search size={14} /> Search for Target
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }} className="animate-fade-in">
      {/* Target Subject Top Banner */}
      <div className="card" style={{ padding: "18px 22px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "16px", border: "1px solid var(--color-accent)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {targetSubject.cropPath && (
            <img 
              src={getMediaUrl(targetSubject.cropPath)} 
              alt="Target Subject" 
              style={{ width: "52px", height: "52px", objectFit: "cover", borderRadius: "8px", border: "1px solid var(--border-color)" }}
            />
          )}
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "10px", fontWeight: "700", color: "var(--color-accent)", letterSpacing: "0.5px" }}>
                TARGET SUBJECT
              </span>
              <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                · Track #{targetSubject.trackId || 1}
              </span>
            </div>
            <h2 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
              {targetSubject.label || `Subject Observation (${targetSubject.cameraName})`}
            </h2>
            <div style={{ fontSize: "11px", color: "var(--text-secondary)" }}>
              First Sighted: {targetSubject.cameraName} @ {formatSec(targetSubject.timestamp || 0)}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px" }}>
            <span style={{ color: "var(--text-muted)" }}>Re-ID Sensitivity:</span>
            <select 
              className="form-select" 
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              style={{ padding: "4px 8px", fontSize: "11px" }}
            >
              <option value="0.75">Strict (0.75)</option>
              <option value="0.70">Standard (0.70)</option>
              <option value="0.65">Permissive (0.65)</option>
            </select>
          </div>

          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={fetchJourney}
            disabled={loading}
            style={{ fontSize: "11px", padding: "6px 10px", display: "flex", alignItems: "center", gap: "4px" }}
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} /> Re-Evaluate
          </button>

          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={async () => {
              if (!targetSubject) return;
              try {
                const res = await apiService.mergeJourneyVideo(jobId, targetSubject.trackUuid, similarityThreshold, 3.0);
                alert(`Successfully merged cross-camera journey video: ${res.filename} (${res.segments_count} camera transitions). Saved to Evidence.`);
              } catch (err: any) {
                alert(err.message || "Failed to merge journey video");
              }
            }}
            style={{ fontSize: "12px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "6px" }}
          >
            <Film size={14} /> Merge Journey Video
          </button>

          <button 
            type="button" 
            className="btn btn-primary"
            onClick={onGenerateSummary}
            style={{ fontSize: "12px", padding: "6px 14px", display: "flex", alignItems: "center", gap: "6px", fontWeight: "600" }}
          >
            <FileText size={14} /> Generate Subject Summary
          </button>
        </div>
      </div>

      {/* Confirmed Journey Timeline */}
      <div className="card" style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h3 style={{ fontSize: "15px", fontWeight: "600", color: "var(--text-primary)" }}>
              Confirmed Chronological Journey ({confirmedJourney.length} Transitions)
            </h3>
            <p style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
              High-confidence transitions validated through multi-signal visual similarity and temporal plausibility.
            </p>
          </div>

          <span className="badge badge-completed" style={{ fontSize: "11px" }}>
            <CheckCircle2 size={12} style={{ marginRight: "4px" }} /> High Confidence
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "14px", position: "relative", paddingLeft: "10px" }}>
          {confirmedJourney.map((evt, idx) => {
            const camName = evt.camera_name || cameras.find(c => c.id === evt.camera_id)?.name || evt.camera_id;
            return (
              <div 
                key={idx}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "14px 16px",
                  borderRadius: "8px",
                  backgroundColor: "var(--bg-input)",
                  border: "1px solid var(--border-color)"
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
                  {evt.crop_path ? (
                    <img 
                      src={getMediaUrl(evt.crop_path)} 
                      alt="Crop" 
                      style={{ width: "44px", height: "44px", objectFit: "cover", borderRadius: "6px", border: "1px solid var(--border-color)" }}
                    />
                  ) : (
                    <div style={{ width: "44px", height: "44px", backgroundColor: "#000", borderRadius: "6px" }} />
                  )}

                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span style={{ fontSize: "10px", fontWeight: "700", color: "var(--color-accent)" }}>
                        STEP {idx + 1}
                      </span>
                      <strong style={{ fontSize: "13px", color: "var(--text-primary)" }}>
                        {camName}
                      </strong>
                      <span style={{ fontSize: "12px", fontFamily: "monospace", color: "var(--color-accent)" }}>
                        @ {formatSec(evt.timestamp)}
                      </span>
                    </div>

                    <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                      {evt.reason || `Appearance correlation: ${(evt.confidence * 100).toFixed(0)}%`}
                    </div>
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  {evt.transit_time_sec !== undefined && evt.transit_time_sec > 0 && (
                    <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                      Transit: +{evt.transit_time_sec.toFixed(0)}s
                    </span>
                  )}
                  <button 
                    type="button"
                    className="btn btn-secondary"
                    style={{ fontSize: "11px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "4px" }}
                    onClick={() => onOpenVideo({
                      jobId,
                      cameraId: evt.camera_id,
                      cameraName: camName,
                      timestamp: evt.timestamp,
                      trackUuid: evt.track_uuid,
                      cropPath: evt.crop_path
                    })}
                  >
                    <Eye size={13} /> View Video
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Possible Transitions (Weaker Evidence) */}
      {possibleTransitions.length > 0 && (
        <div className="card" style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "16px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <AlertTriangle size={15} style={{ color: "var(--color-warning)" }} />
              <h3 style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)" }}>
                Possible Transitions ({possibleTransitions.length})
              </h3>
            </div>
            <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>
              Candidate matches with appearance similarity but weaker temporal or spatial corroboration.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "12px" }}>
            {possibleTransitions.map((cand, idx) => {
              const camName = cand.camera_name || cameras.find(c => c.id === cand.camera_id)?.name || cand.camera_id;
              return (
                <div 
                  key={idx}
                  style={{
                    padding: "12px 14px",
                    borderRadius: "8px",
                    backgroundColor: "var(--bg-input)",
                    border: "1px solid var(--border-color)",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center"
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    {cand.crop_path && (
                      <img 
                        src={getMediaUrl(cand.crop_path)} 
                        alt="Crop" 
                        style={{ width: "36px", height: "36px", objectFit: "cover", borderRadius: "4px" }}
                      />
                    )}
                    <div>
                      <div style={{ fontSize: "12px", fontWeight: "600", color: "var(--text-primary)" }}>
                        {camName} @ {formatSec(cand.timestamp)}
                      </div>
                      <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>
                        {cand.reason || `Similarity: ${(cand.similarity * 100).toFixed(0)}%`}
                      </div>
                    </div>
                  </div>

                  <button 
                    type="button"
                    className="btn btn-secondary"
                    style={{ fontSize: "10px", padding: "4px 8px" }}
                    onClick={() => onOpenVideo({
                      jobId,
                      cameraId: cand.camera_id,
                      cameraName: camName,
                      timestamp: cand.timestamp,
                      trackUuid: cand.track_uuid,
                      cropPath: cand.crop_path
                    })}
                  >
                    Inspect
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

import React, { useState } from "react";
import { X, Copy, Check, User, Play } from "lucide-react";
import { getMediaUrl } from "../services/api";

export interface SubjectObservation {
  camera_name: string;
  camera_id?: string;
  timestamp: number;
  confidence: number;
  crop_path?: string;
  entity_type: string;
  track_id?: string;
  notes?: string;
}

interface SubjectSummaryModalProps {
  isOpen: boolean;
  onClose: () => void;
  subjectLabel: string;
  observations: SubjectObservation[];
  caseName: string;
  onOpenVideo?: (obs: SubjectObservation) => void;
}

export const SubjectSummaryModal: React.FC<SubjectSummaryModalProps> = ({
  isOpen,
  onClose,
  subjectLabel,
  observations,
  caseName,
  onOpenVideo
}) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  // Sort observations chronologically
  const sorted = [...observations].sort((a, b) => a.timestamp - b.timestamp);
  const firstSeen = sorted.length > 0 ? sorted[0] : null;
  const lastSeen = sorted.length > 0 ? sorted[sorted.length - 1] : null;

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // Generate factual narrative
  const generateNarrative = () => {
    if (sorted.length === 0) return "No confirmed observations recorded for this subject.";
    
    if (sorted.length === 1) {
      return `The subject (${subjectLabel}) was observed at camera location "${sorted[0].camera_name}" at time index ${formatTime(sorted[0].timestamp)} (detection confidence: ${(sorted[0].confidence * 100).toFixed(0)}%). No subsequent camera transitions were recorded in this investigation.`;
    }

    const uniqueCams = Array.from(new Set(sorted.map(s => s.camera_name)));
    let text = `In investigation "${caseName}", subject (${subjectLabel}) was first observed at "${sorted[0].camera_name}" at time index ${formatTime(sorted[0].timestamp)}.\n\n`;
    
    for (let i = 1; i < sorted.length; i++) {
      const prev = sorted[i - 1];
      const curr = sorted[i];
      const transitSec = Math.max(0, curr.timestamp - prev.timestamp);
      
      if (curr.camera_name === prev.camera_name) {
        text += `• Maintained continuous presence in "${curr.camera_name}" until ${formatTime(curr.timestamp)}.\n`;
      } else {
        text += `• Transitioned from "${prev.camera_name}" to "${curr.camera_name}" (estimated transition duration: +${transitSec.toFixed(0)}s, recorded at ${formatTime(curr.timestamp)} with ${(curr.confidence * 100).toFixed(0)}% confidence).\n`;
      }
    }

    text += `\nThe subject was last seen at "${sorted[sorted.length - 1].camera_name}" at time index ${formatTime(sorted[sorted.length - 1].timestamp)}. Total recorded span across ${uniqueCams.length} camera location(s): ${(lastSeen!.timestamp - firstSeen!.timestamp).toFixed(0)} seconds.`;
    return text;
  };

  const narrativeText = generateNarrative();

  const handleCopy = () => {
    navigator.clipboard.writeText(narrativeText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div 
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
        padding: "20px"
      }}
      onClick={onClose}
    >
      <div 
        className="card"
        style={{
          maxWidth: "760px",
          width: "100%",
          maxHeight: "85vh",
          display: "flex",
          flexDirection: "column",
          padding: "28px",
          backgroundColor: "var(--bg-card)",
          border: "1px solid var(--border-color)",
          borderRadius: "var(--radius-main)",
          overflowY: "auto"
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
          <div>
            <span style={{ fontSize: "11px", fontWeight: "700", color: "var(--color-accent)", letterSpacing: "0.5px" }}>
              FORENSIC SUBJECT REPORT
            </span>
            <h2 style={{ fontSize: "18px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
              Subject Summary: {subjectLabel}
            </h2>
            <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
              Case: {caseName} · {sorted.length} recorded observation(s)
            </span>
          </div>

          <button 
            type="button"
            onClick={onClose}
            style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", padding: "4px" }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Key Metrics Banner */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginBottom: "20px" }}>
          <div style={{ padding: "12px", borderRadius: "8px", backgroundColor: "var(--bg-input)", border: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "10px", color: "var(--text-muted)", textTransform: "uppercase" }}>First Seen</span>
            <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
              {firstSeen ? `${formatTime(firstSeen.timestamp)} (${firstSeen.camera_name})` : "N/A"}
            </div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", backgroundColor: "var(--bg-input)", border: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "10px", color: "var(--text-muted)", textTransform: "uppercase" }}>Last Seen</span>
            <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
              {lastSeen ? `${formatTime(lastSeen.timestamp)} (${lastSeen.camera_name})` : "N/A"}
            </div>
          </div>

          <div style={{ padding: "12px", borderRadius: "8px", backgroundColor: "var(--bg-input)", border: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "10px", color: "var(--text-muted)", textTransform: "uppercase" }}>Total Locations</span>
            <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--color-accent)", marginTop: "2px" }}>
              {Array.from(new Set(sorted.map(s => s.camera_name))).length} Camera(s)
            </div>
          </div>
        </div>

        {/* Generated Narrative */}
        <div style={{ marginBottom: "24px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <span style={{ fontSize: "12px", fontWeight: "600", color: "var(--text-primary)" }}>
              Chronological Subject Narrative
            </span>
            <button 
              type="button"
              className="btn btn-secondary"
              onClick={handleCopy}
              style={{ fontSize: "11px", padding: "4px 10px", display: "flex", alignItems: "center", gap: "4px" }}
            >
              {copied ? <Check size={12} style={{ color: "var(--color-success)" }} /> : <Copy size={12} />}
              {copied ? "Copied" : "Copy Narrative"}
            </button>
          </div>

          <div 
            style={{ 
              padding: "14px", 
              borderRadius: "8px", 
              backgroundColor: "var(--bg-input)", 
              border: "1px solid var(--border-color)", 
              fontSize: "12px", 
              lineHeight: "1.6", 
              color: "var(--text-primary)",
              whiteSpace: "pre-line"
            }}
          >
            {narrativeText}
          </div>
        </div>

        {/* Chronological Observation Cards List */}
        <div>
          <span style={{ fontSize: "12px", fontWeight: "600", color: "var(--text-primary)", display: "block", marginBottom: "10px" }}>
            Observation Timeline ({sorted.length})
          </span>

          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {sorted.map((obs, idx) => (
              <div 
                key={idx}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "10px 14px",
                  borderRadius: "8px",
                  backgroundColor: "var(--bg-input)",
                  border: "1px solid var(--border-color)"
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  {obs.crop_path ? (
                    <img 
                      src={getMediaUrl(obs.crop_path)} 
                      alt="obs crop" 
                      style={{ width: "38px", height: "38px", borderRadius: "6px", objectFit: "cover", backgroundColor: "#000" }} 
                    />
                  ) : (
                    <div style={{ width: "38px", height: "38px", borderRadius: "6px", backgroundColor: "#000", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
                      <User size={16} />
                    </div>
                  )}

                  <div>
                    <div style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>
                      {obs.camera_name}
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--text-secondary)" }}>
                      Time: {formatTime(obs.timestamp)} · Confidence: {(obs.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  {onOpenVideo && (
                    <button
                      type="button"
                      className="btn btn-primary"
                      style={{ fontSize: "11px", padding: "4px 10px", display: "flex", alignItems: "center", gap: "4px" }}
                      onClick={() => onOpenVideo(obs)}
                    >
                      <Play size={11} /> Open Video
                    </button>
                  )}
                  <span style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "monospace" }}>
                    #{idx + 1}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

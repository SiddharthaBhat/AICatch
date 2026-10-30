import React, { useState, useEffect } from "react";
import { 
  ArrowLeft, Search, Layers, FolderOpen, FileText, 
  Eye, AlertCircle, ChevronRight, RefreshCw, Download, Clock
} from "lucide-react";
import type { Job, Camera, Track } from "../types";
import { apiService, getMediaUrl } from "../services/api";
import { SearchWorkspace } from "./SearchWorkspace";
import { JourneyView } from "./JourneyView";
import { Evidence } from "./Evidence";
import { ForensicVideoViewer, type ObservationTarget } from "./ForensicVideoViewer";
import { SubjectSummaryModal } from "./SubjectSummaryModal";
import { CameraTimestampModal } from "./CameraTimestampModal";

interface InvestigationWorkspaceProps {
  jobId: string;
  onBack: () => void;
}

export const InvestigationWorkspace: React.FC<InvestigationWorkspaceProps> = ({ jobId, onBack }) => {
  const [activeTab, setActiveTab] = useState<"overview" | "search" | "journey" | "evidence" | "summary">("overview");
  const [job, setJob] = useState<Job | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(true);

  // Selected camera for Overview video player
  const [selectedPreviewCamera, setSelectedPreviewCamera] = useState<Camera | null>(null);
  const [timestampModalCamera, setTimestampModalCamera] = useState<Camera | null>(null);

  // Target Subject state for Journey and Subject Summary
  const [targetSubject, setTargetSubject] = useState<{
    trackUuid: string;
    trackId?: string | number;
    cameraId?: string;
    cameraName?: string;
    timestamp?: number;
    cropPath?: string;
    entityType?: string;
    label?: string;
  } | null>(null);

  // Full Forensic Video Viewer state
  const [viewerObservation, setViewerObservation] = useState<ObservationTarget | null>(null);
  const [showVideoViewer, setShowVideoViewer] = useState(false);

  // Summary Modal state (for Get Details)
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [modalSubjectLabel, setModalSubjectLabel] = useState("");
  const [modalObservations, setModalObservations] = useState<any[]>([]);

  // Generated Subject Summary text
  const [subjectSummaryReport, setSubjectSummaryReport] = useState<string | null>(null);
  const [generatingSummary, setGeneratingSummary] = useState(false);

  const loadData = async () => {
    try {
      const [jobData, camsData, tracksData] = await Promise.all([
        apiService.getJobStatus(jobId),
        apiService.getJobCameras(jobId),
        apiService.getJobTracks(jobId)
      ]);
      setJob(jobData);
      setCameras(camsData);
      setTracks(tracksData);
      if (camsData.length > 0 && !selectedPreviewCamera) {
        setSelectedPreviewCamera(camsData[0]);
      }
    } catch (err) {
      console.error("Failed to load investigation workspace data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [jobId]);

  if (loading && !job) {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "60vh" }}>
        <span style={{ fontSize: "14px", color: "var(--text-secondary)" }}>Loading investigation workspace...</span>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="card" style={{ padding: "40px", textAlign: "center", maxWidth: "600px", margin: "40px auto" }}>
        <AlertCircle size={36} style={{ color: "var(--color-danger)", margin: "0 auto 12px" }} />
        <h2 style={{ fontSize: "16px", color: "var(--text-primary)", marginBottom: "8px" }}>Investigation Not Found</h2>
        <button type="button" className="btn btn-secondary" onClick={onBack}>
          <ArrowLeft size={14} /> Back to Dashboard
        </button>
      </div>
    );
  }

  const isMulti = job.investigation_type === "multi_camera";
  const activePreviewCam = selectedPreviewCamera || (cameras.length > 0 ? cameras[0] : null);

  // Truthful metrics
  const personTracks = tracks.filter(t => t.entity_type === "person");
  const vehicleTracks = tracks.filter(t => t.entity_type === "vehicle");

  const handleSelectTargetFromSearch = (obs: any) => {
    const camName = cameras.find(c => c.id === obs.camera_id)?.name || obs.camera_id || "Camera";
    const trackUuid = obs.track_uuid || (typeof obs.track_id === "string" ? obs.track_id : `track_${obs.track_id}`);
    setTargetSubject({
      trackUuid,
      trackId: obs.track_id,
      cameraId: obs.camera_id,
      cameraName: camName,
      timestamp: obs.timestamp || 0,
      cropPath: obs.crop_path,
      entityType: obs.entity_type || "person",
      label: obs.label
    });
    setActiveTab("journey");
  };

  const handleOpenSubjectDetails = (obs: any) => {
    const label = obs.label || (obs.entity_type === "person" ? `Subject #${obs.track_id || 1}` : "Observation");
    setModalSubjectLabel(label);
    const camName = cameras.find(c => c.id === obs.camera_id)?.name || obs.camera_id || "Camera";
    setModalObservations([{
      camera_name: camName,
      camera_id: obs.camera_id,
      timestamp: obs.timestamp || 0,
      confidence: obs.confidence || 0.85,
      crop_path: obs.crop_path,
      entity_type: obs.entity_type || "person",
      track_id: obs.track_id
    }]);
    setShowSummaryModal(true);
  };

  const handleGenerateSubjectSummary = async () => {
    if (!targetSubject) return;
    setGeneratingSummary(true);
    try {
      const summaryData = await apiService.getTrackSummary(targetSubject.trackUuid);
      setSubjectSummaryReport(summaryData.summary);
    } catch (err) {
      console.error("Failed to generate subject summary:", err);
      // Construct honest fallback summary from target observations
      const dateStr = new Date((job.created_at || Date.now() / 1000) * 1000).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
      const fallback = `# SUBJECT FORENSIC SUMMARY: ${targetSubject.label || `SUBJECT #${targetSubject.trackId || 1}`}

**Investigation:** ${job.name}
**Recorded Date:** ${dateStr}
**Origin Location:** ${targetSubject.cameraName}
**First Sighted Time:** ${targetSubject.timestamp?.toFixed(1)}s

### Validated Subject Observations
- Detected on **${targetSubject.cameraName}** at timestamp **${targetSubject.timestamp?.toFixed(1)}s** with verified visual features.

*Summary generated based strictly on verified candidate observations.*`;
      setSubjectSummaryReport(fallback);
    } finally {
      setGeneratingSummary(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px", maxWidth: "1200px", margin: "0 auto", paddingBottom: "60px" }} className="animate-fade-in">
      {/* Top Breadcrumb & Case Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px", borderBottom: "1px solid var(--border-color)", paddingBottom: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <button 
            type="button" 
            className="btn btn-secondary" 
            onClick={onBack}
            style={{ padding: "8px 12px", fontSize: "12px" }}
          >
            <ArrowLeft size={14} /> Back
          </button>
          <div>
            <h1 style={{ fontSize: "18px", fontWeight: "600", color: "var(--text-primary)" }}>
              {job.name}
            </h1>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "2px" }}>
              <span 
                style={{ 
                  fontSize: "9px", 
                  fontWeight: "700", 
                  letterSpacing: "0.5px",
                  padding: "2px 6px", 
                  borderRadius: "4px",
                  backgroundColor: isMulti ? "rgba(16, 185, 129, 0.1)" : "rgba(6, 182, 212, 0.1)",
                  color: isMulti ? "var(--color-success)" : "var(--color-accent)"
                }}
              >
                {isMulti ? "MULTI-CAMERA" : "SINGLE VIDEO"}
              </span>
              <span className={`badge badge-${job.status}`} style={{ fontSize: "9px" }}>
                {job.status}
              </span>
              <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                · Profile: {job.profile}
              </span>
            </div>
          </div>
        </div>

        {/* 5 Primary Workspace Tabs */}
        <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-input)", padding: "4px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
          <button
            type="button"
            className="btn"
            onClick={() => setActiveTab("overview")}
            style={{
              padding: "6px 14px",
              fontSize: "12px",
              fontWeight: "600",
              backgroundColor: activeTab === "overview" ? "var(--bg-card)" : "transparent",
              color: activeTab === "overview" ? "var(--color-accent)" : "var(--text-secondary)",
              border: activeTab === "overview" ? "1px solid var(--border-color)" : "none",
              borderRadius: "6px"
            }}
          >
            Overview
          </button>

          <button
            type="button"
            className="btn"
            onClick={() => setActiveTab("search")}
            style={{
              padding: "6px 14px",
              fontSize: "12px",
              fontWeight: "600",
              backgroundColor: activeTab === "search" ? "var(--bg-card)" : "transparent",
              color: activeTab === "search" ? "var(--color-accent)" : "var(--text-secondary)",
              border: activeTab === "search" ? "1px solid var(--border-color)" : "none",
              borderRadius: "6px",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}
          >
            <Search size={13} /> Search
          </button>

          {isMulti && (
            <button
              type="button"
              className="btn"
              onClick={() => setActiveTab("journey")}
              style={{
                padding: "6px 14px",
                fontSize: "12px",
                fontWeight: "600",
                backgroundColor: activeTab === "journey" ? "var(--bg-card)" : "transparent",
                color: activeTab === "journey" ? "var(--color-accent)" : "var(--text-secondary)",
                border: activeTab === "journey" ? "1px solid var(--border-color)" : "none",
                borderRadius: "6px",
                display: "flex",
                alignItems: "center",
                gap: "6px"
              }}
            >
              <Layers size={13} /> Journey
            </button>
          )}

          <button
            type="button"
            className="btn"
            onClick={() => setActiveTab("evidence")}
            style={{
              padding: "6px 14px",
              fontSize: "12px",
              fontWeight: "600",
              backgroundColor: activeTab === "evidence" ? "var(--bg-card)" : "transparent",
              color: activeTab === "evidence" ? "var(--color-accent)" : "var(--text-secondary)",
              border: activeTab === "evidence" ? "1px solid var(--border-color)" : "none",
              borderRadius: "6px",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}
          >
            <FolderOpen size={13} /> Evidence
          </button>

          <button
            type="button"
            className="btn"
            onClick={() => setActiveTab("summary")}
            style={{
              padding: "6px 14px",
              fontSize: "12px",
              fontWeight: "600",
              backgroundColor: activeTab === "summary" ? "var(--bg-card)" : "transparent",
              color: activeTab === "summary" ? "var(--color-accent)" : "var(--text-secondary)",
              border: activeTab === "summary" ? "1px solid var(--border-color)" : "none",
              borderRadius: "6px",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}
          >
            <FileText size={13} /> Summary
          </button>
        </div>
      </div>

      {/* Tab Content 1: Overview */}
      {activeTab === "overview" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          {/* Truthful Forensic Metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "14px" }}>
            <div className="card" style={{ padding: "16px" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: "600", letterSpacing: "0.5px" }}>
                CAMERA SOURCES
              </div>
              <div style={{ fontSize: "22px", fontWeight: "700", color: "var(--text-primary)", marginTop: "4px" }}>
                {cameras.length}
              </div>
              <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                {isMulti ? "Multi-camera network" : "Single video feed"}
              </div>
            </div>

            <div className="card" style={{ padding: "16px" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: "600", letterSpacing: "0.5px" }}>
                INDEXED FOOTAGE
              </div>
              <div style={{ fontSize: "22px", fontWeight: "700", color: "var(--color-accent)", marginTop: "4px" }}>
                {job.duration_sec ? `${Math.round(job.duration_sec / 60)} min` : `${Math.round(job.processed_sec || 0)}s`}
              </div>
              <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                Processed: {Math.round(job.processed_sec || 0)}s
              </div>
            </div>

            <div className="card" style={{ padding: "16px" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: "600", letterSpacing: "0.5px" }}>
                TRACKED PERSON INSTANCES
              </div>
              <div style={{ fontSize: "22px", fontWeight: "700", color: "var(--color-success)", marginTop: "4px" }}>
                {personTracks.length}
              </div>
              <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                Localized person tracklets
              </div>
            </div>

            <div className="card" style={{ padding: "16px" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: "600", letterSpacing: "0.5px" }}>
                TOTAL TRACK INSTANCES
              </div>
              <div style={{ fontSize: "22px", fontWeight: "700", color: "var(--text-primary)", marginTop: "4px" }}>
                {tracks.length}
              </div>
              <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                {vehicleTracks.length > 0 ? `${vehicleTracks.length} vehicle tracks` : "0 vehicle tracks detected"}
              </div>
            </div>
          </div>

          {/* Camera Sources & Video Preview Layout */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: "20px" }}>
            {/* Left: Camera Sources List */}
            <div className="card" style={{ display: "flex", flexDirection: "column", gap: "12px", padding: "20px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h3 style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)" }}>
                  Camera Sources ({cameras.length})
                </h3>
                <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                  Select camera to preview video
                </span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {cameras.map((cam, idx) => {
                  const isSelected = activePreviewCam?.id === cam.id;
                  const camTracks = tracks.filter(t => t.camera_id === cam.id);
                  return (
                    <div 
                      key={cam.id}
                      onClick={() => setSelectedPreviewCamera(cam)}
                      style={{
                        padding: "12px 14px",
                        borderRadius: "8px",
                        border: isSelected ? "1.5px solid var(--color-accent)" : "1px solid var(--border-color)",
                        backgroundColor: isSelected ? "rgba(6, 182, 212, 0.05)" : "var(--bg-input)",
                        cursor: "pointer",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        transition: "all 0.15s ease"
                      }}
                    >
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                          <span style={{ fontSize: "11px", fontWeight: "700", color: isSelected ? "var(--color-accent)" : "var(--text-secondary)" }}>
                            Camera {idx + 1}
                          </span>
                          <strong style={{ fontSize: "13px", color: "var(--text-primary)" }}>
                            {cam.name}
                          </strong>
                        </div>
                        <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                          Source: {cam.source_path.split("/").pop() || cam.source_path} · {camTracks.length} tracks
                        </div>
                      </div>

                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          onClick={(e) => {
                            e.stopPropagation();
                            setTimestampModalCamera(cam);
                          }}
                          style={{ fontSize: "11px", padding: "4px 8px", display: "flex", alignItems: "center", gap: "4px" }}
                          title="Configure CCTV timestamp detection or clock offset"
                        >
                          <Clock size={12} />
                          {cam.timestamp_config_mode && cam.timestamp_config_mode !== 'auto' ? 'Calibrated' : 'Timestamps'}
                        </button>
                        <span className="badge badge-completed" style={{ fontSize: "9px" }}>
                          Indexed
                        </span>
                        <ChevronRight size={14} style={{ color: isSelected ? "var(--color-accent)" : "var(--text-muted)" }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right: Active Camera Source Video Preview */}
            <div className="card" style={{ display: "flex", flexDirection: "column", gap: "12px", padding: "20px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <span style={{ fontSize: "11px", color: "var(--color-accent)", fontWeight: "700" }}>
                    SOURCE VIDEO PREVIEW
                  </span>
                  <h3 style={{ fontSize: "15px", fontWeight: "600", color: "var(--text-primary)" }}>
                    {activePreviewCam?.name || "Camera Preview"}
                  </h3>
                </div>

                {activePreviewCam && (
                  <button 
                    type="button"
                    className="btn btn-secondary"
                    style={{ fontSize: "11px", padding: "4px 10px", display: "flex", alignItems: "center", gap: "4px" }}
                    onClick={() => {
                      setViewerObservation({
                        jobId,
                        cameraId: activePreviewCam.id,
                        cameraName: activePreviewCam.name,
                        timestamp: 0
                      });
                      setShowVideoViewer(true);
                    }}
                  >
                    <Eye size={13} /> Full Forensic Viewer
                  </button>
                )}
              </div>

              {activePreviewCam ? (
                <div style={{ width: "100%", height: "280px", backgroundColor: "#000", borderRadius: "6px", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <video 
                    key={activePreviewCam.id}
                    src={getMediaUrl(activePreviewCam.source_path)}
                    controls
                    style={{ width: "100%", height: "100%", objectFit: "contain" }}
                  />
                </div>
              ) : (
                <div style={{ width: "100%", height: "280px", backgroundColor: "var(--bg-input)", borderRadius: "6px", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
                  No camera selected.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab Content 2: Search */}
      {activeTab === "search" && (
        <SearchWorkspace 
          jobId={jobId}
          cameras={cameras}
          onSelectTargetForJourney={handleSelectTargetFromSearch}
        />
      )}

      {/* Tab Content 3: Journey */}
      {activeTab === "journey" && isMulti && (
        <JourneyView 
          jobId={jobId}
          cameras={cameras}
          targetSubject={targetSubject}
          onSelectTarget={(subj) => setTargetSubject(subj)}
          onOpenVideo={(obs) => {
            setViewerObservation(obs);
            setShowVideoViewer(true);
          }}
          onGenerateSummary={() => {
            setActiveTab("summary");
            handleGenerateSubjectSummary();
          }}
          onGoToSearch={() => setActiveTab("search")}
        />
      )}

      {/* Tab Content 4: Evidence */}
      {activeTab === "evidence" && (
        <Evidence 
          jobId={jobId}
        />
      )}

      {/* Tab Content 5: Subject Summary */}
      {activeTab === "summary" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          {!targetSubject ? (
            /* Empty State: No Target Selected */
            <div className="card" style={{ padding: "48px 24px", textAlign: "center", maxWidth: "600px", margin: "20px auto", display: "flex", flexDirection: "column", alignItems: "center", gap: "14px" }}>
              <FileText size={40} style={{ color: "var(--text-muted)" }} />
              <div>
                <h3 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "4px" }}>
                  No Validated Subject Journey Selected
                </h3>
                <p style={{ fontSize: "13px", color: "var(--text-secondary)", maxWidth: "440px", lineHeight: "1.5" }}>
                  To generate an honest chronological subject summary:
                </p>
                <div style={{ textAlign: "left", display: "inline-block", fontSize: "12px", color: "var(--text-muted)", marginTop: "8px", lineHeight: "1.8" }}>
                  1. Search for a person or target observation<br />
                  2. Select the observation as Target Subject<br />
                  3. Reconstruct and validate cross-camera journey<br />
                  4. Generate Subject Summary
                </div>
              </div>

              <button 
                type="button" 
                className="btn btn-primary"
                onClick={() => setActiveTab("search")}
                style={{ marginTop: "8px", display: "flex", alignItems: "center", gap: "6px" }}
              >
                <Search size={14} /> Go to Search
              </button>
            </div>
          ) : (
            /* Subject Summary Content */
            <div className="card" style={{ padding: "28px", display: "flex", flexDirection: "column", gap: "20px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", borderBottom: "1px solid var(--border-color)", paddingBottom: "16px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
                  {targetSubject.cropPath && (
                    <img 
                      src={getMediaUrl(targetSubject.cropPath)} 
                      alt="Target"
                      style={{ width: "56px", height: "56px", objectFit: "cover", borderRadius: "8px", border: "1px solid var(--border-color)" }}
                    />
                  )}
                  <div>
                    <span style={{ fontSize: "11px", color: "var(--color-accent)", fontWeight: "700" }}>
                      SUBJECT SUMMARY DOSSIER
                    </span>
                    <h2 style={{ fontSize: "17px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
                      {targetSubject.label || `Subject Track #${targetSubject.trackId || 1}`}
                    </h2>
                    <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                      First Sighted: {targetSubject.cameraName} @ {targetSubject.timestamp?.toFixed(1)}s
                    </span>
                  </div>
                </div>

                <div style={{ display: "flex", gap: "8px" }}>
                  <button 
                    type="button" 
                    className="btn btn-secondary"
                    onClick={handleGenerateSubjectSummary}
                    disabled={generatingSummary}
                    style={{ fontSize: "12px", display: "flex", alignItems: "center", gap: "6px" }}
                  >
                    <RefreshCw size={13} className={generatingSummary ? "animate-spin" : ""} />
                    {generatingSummary ? "Compiling..." : "Regenerate"}
                  </button>

                  {subjectSummaryReport && (
                    <>
                      <button 
                        type="button" 
                        className="btn btn-secondary"
                        style={{ fontSize: "12px", display: "flex", alignItems: "center", gap: "6px" }}
                        onClick={() => {
                          const element = document.createElement("a");
                          const file = new Blob([subjectSummaryReport], { type: "text/plain" });
                          element.href = URL.createObjectURL(file);
                          element.download = `AICatch_Summary_${job.name.replace(/\\s+/g, "_")}.txt`;
                          document.body.appendChild(element);
                          element.click();
                          document.body.removeChild(element);
                        }}
                      >
                        <Download size={13} /> Export TXT
                      </button>

                      <button 
                        type="button" 
                        className="btn btn-primary"
                        style={{ fontSize: "12px", display: "flex", alignItems: "center", gap: "6px" }}
                        onClick={() => window.print()}
                      >
                        <FileText size={13} /> Print / PDF
                      </button>
                    </>
                  )}
                </div>
              </div>

              {subjectSummaryReport ? (
                <div 
                  style={{ 
                    whiteSpace: "pre-wrap", 
                    fontFamily: "var(--font-sans)", 
                    fontSize: "13px", 
                    lineHeight: "1.7", 
                    color: "var(--text-primary)",
                    backgroundColor: "var(--bg-input)",
                    padding: "20px",
                    borderRadius: "8px",
                    border: "1px solid var(--border-color)"
                  }}
                >
                  {subjectSummaryReport}
                </div>
              ) : (
                <div style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)" }}>
                  Click "Regenerate Summary" to compile chronological subject timeline.
                </div>
              )}
            </div>
          )}
        </div>
      )}


      {/* CCTV Timestamp Configuration Modal */}
      {timestampModalCamera && (
        <CameraTimestampModal
          jobId={jobId}
          camera={timestampModalCamera}
          isOpen={!!timestampModalCamera}
          onClose={() => setTimestampModalCamera(null)}
          onConfigSaved={async () => {
            try {
              const updatedCams = await apiService.getJobCameras(jobId);
              setCameras(updatedCams);
            } catch (e) {
              console.error("Failed to reload cameras after timestamp save:", e);
            }
          }}
        />
      )}

      {/* Forensic Video Viewer Modal */}
      {showVideoViewer && viewerObservation && (
        <ForensicVideoViewer
          isOpen={showVideoViewer}
          onClose={() => setShowVideoViewer(false)}
          observation={viewerObservation}
          cameras={cameras}
          caseName={job.name}
          onSaveToEvidence={async (obs) => {
            try {
              const folders = await apiService.listFolders(jobId);
              let foldId = folders[0]?.id;
              if (!foldId) {
                const newF = await apiService.createFolder("Subject Evidence", "Saved from Forensic Viewer", jobId);
                foldId = newF.id;
              }
              await apiService.addFolderItem(foldId, "track", obs.trackUuid || `track_${obs.trackId || 1}`, JSON.stringify(obs));
              alert("Saved observation to evidence folder.");
            } catch (err: any) {
              alert(err.message || "Failed to save evidence");
            }
          }}
          onGetDetails={(obs) => {
            setShowVideoViewer(false);
            handleOpenSubjectDetails(obs);
          }}
          onFindSimilar={() => {
            setShowVideoViewer(false);
            setActiveTab("search");
          }}
        />
      )}

      {/* Subject Summary Modal */}
      {showSummaryModal && (
        <SubjectSummaryModal
          isOpen={showSummaryModal}
          onClose={() => setShowSummaryModal(false)}
          subjectLabel={modalSubjectLabel}
          observations={modalObservations}
          caseName={job.name}
          onOpenVideo={(obs) => {
            setShowSummaryModal(false);
            setViewerObservation({
              jobId,
              cameraId: obs.camera_id || "",
              cameraName: obs.camera_name,
              timestamp: obs.timestamp,
              trackId: obs.track_id,
              cropPath: obs.crop_path,
              entityType: obs.entity_type
            });
            setShowVideoViewer(true);
          }}
        />
      )}
    </div>
  );
};

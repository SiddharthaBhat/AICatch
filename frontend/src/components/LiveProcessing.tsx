import React, { useEffect, useState, useRef } from "react";
import { 
  ArrowLeft, Play, Pause, Square, 
  AlertCircle, ArrowRight, Loader2,
  Plus, Upload, Video, CheckCircle2, RefreshCw
} from "lucide-react";
import type { Job, Camera, Track } from "../types";
import { apiService } from "../services/api";

interface LiveProcessingProps {
  jobId: string;
  onBack: () => void;
  onOpenWorkspace: (jobId: string) => void;
}

export const LiveProcessing: React.FC<LiveProcessingProps> = ({
  jobId,
  onBack,
  onOpenWorkspace
}) => {
  const [job, setJob] = useState<Job | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [startingCams, setStartingCams] = useState<Record<string, boolean>>({});
  const [uploadingCams, setUploadingCams] = useState<Record<string, { uploading: boolean; error?: string }>>({});
  const [addingCamera, setAddingCamera] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  const fetchJobDetails = async () => {
    try {
      const [jobData, camsData, tracksData] = await Promise.all([
        apiService.getJobStatus(jobId),
        apiService.getJobCameras(jobId),
        apiService.getJobTracks(jobId)
      ]);
      setJob(jobData);
      setCameras(camsData);
      setTracks(tracksData);
    } catch (err: any) {
      console.error("Error polling job details:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobDetails();
    if (job?.status === "completed" || job?.status === "failed" || job?.status === "cancelled") {
      return;
    }
    const interval = setInterval(fetchJobDetails, 2000);
    return () => clearInterval(interval);
  }, [jobId, job?.status]);

  const handleStartAll = async () => {
    setActionLoading(true);
    try {
      await apiService.startAllCameras(jobId);
      setStatusMsg({ type: "success", text: "Started indexing all eligible cameras." });
      await fetchJobDetails();
    } catch (err: any) {
      setStatusMsg({ type: "error", text: err.message || "Failed to start remaining cameras." });
    } finally {
      setActionLoading(false);
    }
  };

  const handleStartCamera = async (cameraId: string) => {
    setStartingCams(prev => ({ ...prev, [cameraId]: true }));
    try {
      await apiService.startCamera(jobId, cameraId);
      setStatusMsg({ type: "success", text: "Camera indexing initiated." });
      await fetchJobDetails();
    } catch (err: any) {
      setStatusMsg({ type: "error", text: err.message || "Failed to start camera." });
    } finally {
      setStartingCams(prev => ({ ...prev, [cameraId]: false }));
    }
  };

  const handleAddCamera = async () => {
    setAddingCamera(true);
    try {
      const nextIdx = cameras.length + 1;
      const defaultName = nextIdx === 3 ? "Parking Area" : nextIdx === 4 ? "Back Exit" : `Camera ${nextIdx}`;
      await apiService.addJobCamera(jobId, defaultName);
      setStatusMsg({ type: "success", text: `Added ${defaultName} to investigation.` });
      await fetchJobDetails();
    } catch (err: any) {
      setStatusMsg({ type: "error", text: err.message || "Failed to add camera feed." });
    } finally {
      setAddingCamera(false);
    }
  };

  const handleCameraFileUpload = async (cameraId: string, file: File) => {
    setUploadingCams(prev => ({ ...prev, [cameraId]: { uploading: true } }));
    try {
      const uploadRes = await apiService.uploadVideoFile(file);
      await apiService.updateJobCamera(jobId, cameraId, { path: uploadRes.path });
      setStatusMsg({ type: "success", text: `Uploaded video for camera successfully.` });
      await fetchJobDetails();
    } catch (err: any) {
      setUploadingCams(prev => ({ ...prev, [cameraId]: { uploading: false, error: err.message || "Upload failed." } }));
      setStatusMsg({ type: "error", text: err.message || "Failed to upload video for camera." });
    } finally {
      setUploadingCams(prev => ({ ...prev, [cameraId]: { uploading: false } }));
    }
  };

  const handlePause = async () => {
    setActionLoading(true);
    try {
      await apiService.pauseJob(jobId);
      setStatusMsg({ type: "success", text: "Indexing paused." });
      await fetchJobDetails();
    } catch (err: any) {
      setStatusMsg({ type: "error", text: err.message || "Failed to pause indexing." });
    } finally {
      setActionLoading(false);
    }
  };

  const handleResume = async () => {
    setActionLoading(true);
    try {
      await apiService.resumeJob(jobId);
      setStatusMsg({ type: "success", text: "Indexing resumed." });
      await fetchJobDetails();
    } catch (err: any) {
      setStatusMsg({ type: "error", text: err.message || "Failed to resume indexing." });
    } finally {
      setActionLoading(false);
    }
  };

  const handleCancel = async () => {
    if (window.confirm("Are you sure you want to stop indexing this investigation?")) {
      setActionLoading(true);
      try {
        await apiService.cancelJob(jobId);
        setStatusMsg({ type: "success", text: "Indexing cancelled." });
        await fetchJobDetails();
      } catch (err: any) {
        setStatusMsg({ type: "error", text: err.message || "Failed to cancel indexing." });
      } finally {
        setActionLoading(false);
      }
    }
  };

  if (loading && !job) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60vh", gap: "12px" }}>
        <Loader2 size={32} className="animate-spin" style={{ color: "var(--color-accent)" }} />
        <span style={{ fontSize: "14px", color: "var(--text-secondary)" }}>Loading indexing state...</span>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="card" style={{ padding: "40px", textAlign: "center", maxWidth: "600px", margin: "40px auto" }}>
        <AlertCircle size={36} style={{ color: "var(--color-danger)", margin: "0 auto 12px" }} />
        <h2 style={{ fontSize: "16px", color: "var(--text-primary)", marginBottom: "8px" }}>Investigation Not Found</h2>
        <button type="button" className="btn btn-secondary" onClick={onBack}>
          <ArrowLeft size={14} /> Back to Investigations
        </button>
      </div>
    );
  }

  const isMulti = job.investigation_type === "multi_camera";
  const isComplete = job.status === "completed";
  let rawProgress = typeof job.progress === "number" ? job.progress : 0;
  if (rawProgress > 0 && rawProgress <= 1.0) {
    rawProgress = rawProgress * 100;
  }
  const overallProgress = isComplete ? 100 : Math.min(99, Math.round(rawProgress));
  const isPaused = job.status === "paused";
  const isIndexing = job.status === "indexing" || job.status === "starting";

  const hasReadyCameras = cameras.some(c => (c.status as string) === "ready" || (c.status as string) === "inactive");
  const anyIndexingOrComplete = cameras.some(c => (c.status as string) === "indexing" || (c.status as string) === "active" || (c.status as string) === "completed");
  const startAllLabel = anyIndexingOrComplete ? "Start Remaining" : "Start Indexing All";

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", paddingBottom: "60px" }} className="animate-fade-in">
      {/* Top Breadcrumb & Title */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <button 
            type="button" 
            className="btn btn-secondary" 
            onClick={onBack}
            style={{ padding: "8px 12px", fontSize: "12px" }}
          >
            <ArrowLeft size={14} /> Back
          </button>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "2px" }}>
              <span 
                style={{ 
                  fontSize: "10px", 
                  fontWeight: "700", 
                  letterSpacing: "0.5px",
                  padding: "2px 6px", 
                  borderRadius: "4px",
                  backgroundColor: isMulti ? "rgba(16, 185, 129, 0.1)" : "rgba(6, 182, 212, 0.1)",
                  color: isMulti ? "var(--color-success)" : "var(--color-accent)"
                }}
              >
                {isMulti ? "MULTI-CAMERA INDEXING" : "SINGLE VIDEO INDEXING"}
              </span>
              <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>· Case ID: {job.id}</span>
            </div>
            <h1 style={{ fontSize: "22px", fontWeight: "600", color: "var(--text-primary)" }}>
              {job.name}
            </h1>
          </div>
        </div>

        {/* Global Action Buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {hasReadyCameras && (
            <button 
              type="button" 
              className="btn btn-primary"
              onClick={handleStartAll}
              disabled={actionLoading}
              style={{ fontSize: "12px", padding: "8px 14px", display: "flex", alignItems: "center", gap: "6px", fontWeight: "600" }}
            >
              {actionLoading ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />} {startAllLabel}
            </button>
          )}

          {isIndexing && (
            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={handlePause}
              disabled={actionLoading}
              style={{ fontSize: "12px", padding: "8px 14px", display: "flex", alignItems: "center", gap: "6px" }}
            >
              <Pause size={14} /> Pause All
            </button>
          )}

          {isPaused && (
            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={handleResume}
              disabled={actionLoading}
              style={{ fontSize: "12px", padding: "8px 14px", display: "flex", alignItems: "center", gap: "6px" }}
            >
              <Play size={14} /> Resume Indexing
            </button>
          )}

          {!isComplete && (
            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={handleCancel}
              disabled={actionLoading}
              style={{ fontSize: "12px", padding: "8px 14px", color: "var(--color-danger)" }}
            >
              <Square size={14} /> Cancel
            </button>
          )}

          <button 
            type="button" 
            className="btn btn-primary"
            onClick={() => onOpenWorkspace(job.id)}
            style={{ fontSize: "12px", padding: "8px 16px", display: "flex", alignItems: "center", gap: "6px", fontWeight: "600" }}
          >
            Open Investigation <ArrowRight size={14} />
          </button>
        </div>
      </div>

      {statusMsg && (
        <div 
          style={{ 
            padding: "10px 16px", 
            borderRadius: "8px", 
            marginBottom: "20px",
            backgroundColor: statusMsg.type === "success" ? "rgba(16, 185, 129, 0.1)" : "rgba(244, 63, 94, 0.1)",
            border: `1px solid ${statusMsg.type === "success" ? "var(--color-success)" : "var(--color-danger)"}`,
            color: statusMsg.type === "success" ? "var(--color-success)" : "var(--color-danger)",
            fontSize: "13px"
          }}
        >
          {statusMsg.text}
        </div>
      )}

      {/* Overall Progress Banner */}
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: "14px", padding: "20px", marginBottom: "24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)" }}>
              Overall Progress
            </span>
            <span 
              className={`badge badge-${job.status}`}
              style={{ fontSize: "11px" }}
            >
              {isComplete ? "✓ Complete" : isIndexing ? "Indexing" : job.status}
            </span>
          </div>
          <span style={{ fontSize: "16px", fontWeight: "700", color: "var(--color-accent)" }}>
            {overallProgress}%
          </span>
        </div>

        {/* Progress Bar */}
        <div style={{ width: "100%", height: "8px", backgroundColor: "var(--bg-input)", borderRadius: "4px", overflow: "hidden" }}>
          <div 
            style={{ 
              width: `${overallProgress}%`, 
              height: "100%", 
              backgroundColor: isComplete ? "var(--color-success)" : "var(--color-accent)",
              transition: "width 0.4s ease"
            }} 
          />
        </div>

        {/* Aggregate Stats */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "24px", paddingTop: "8px", fontSize: "12px", color: "var(--text-secondary)" }}>
          <span>Cameras: <strong style={{ color: "var(--text-primary)" }}>{cameras.length}</strong></span>
          <span>Tracks: <strong style={{ color: "var(--text-primary)" }}>{tracks.length}</strong></span>
          {job.fps !== undefined && job.fps > 0 && (
            <span>Decoding: <strong style={{ color: "var(--color-accent)" }}>{job.fps.toFixed(1)} FPS</strong></span>
          )}
          {job.gpu_usage !== undefined && job.gpu_usage > 0 && (
            <span>GPU Util: <strong style={{ color: "var(--text-primary)" }}>{job.gpu_usage}%</strong></span>
          )}
        </div>
      </div>

      {/* Per-Camera Cards Grid */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
          <h2 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)" }}>
            Camera Sources Status ({cameras.length})
          </h2>

          {isMulti && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleAddCamera}
              disabled={addingCamera || actionLoading}
              style={{ fontSize: "12px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "6px" }}
            >
              {addingCamera ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />}
              Add Camera
            </button>
          )}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: "18px" }}>
          {cameras.map((cam, idx) => {
            const camNumberStr = `CAMERA ${(idx + 1).toString().padStart(2, "0")}`;
            const camTracks = tracks.filter(t => t.camera_id === cam.id);
            const isCamUploading = uploadingCams[cam.id]?.uploading;
            
            // Derive camera-specific state
            const rawCamStatus = (cam.status as string) || "not_configured";
            const isCamNotConfigured = rawCamStatus === "not_configured" || (!cam.source_path && !isCamUploading);
            const isCamComplete = rawCamStatus === "completed";
            const isCamIndexing = rawCamStatus === "indexing" || rawCamStatus === "active";
            const isCamReady = (rawCamStatus === "ready" || rawCamStatus === "inactive") && !isCamNotConfigured && !isCamUploading;
            const isCamFailed = rawCamStatus === "failed";

            let camProgressVal = typeof cam.progress === "number" ? cam.progress : 0;
            if (camProgressVal > 0 && camProgressVal <= 1.0) {
              camProgressVal = camProgressVal * 100;
            }
            const camProgress = isCamComplete ? 100 : Math.min(99, Math.round(camProgressVal));

            let camStatusText = isCamUploading
              ? "UPLOADING"
              : isCamNotConfigured
              ? "NOT CONFIGURED"
              : isCamComplete
              ? "COMPLETED"
              : isCamIndexing
              ? "INDEXING"
              : isCamFailed
              ? "FAILED"
              : "READY";

            let camStatusColor = isCamUploading
              ? "var(--color-accent)"
              : isCamNotConfigured
              ? "var(--text-muted)"
              : isCamComplete 
              ? "var(--color-success)" 
              : isCamIndexing 
              ? "var(--color-accent)" 
              : isCamFailed 
              ? "var(--color-danger)" 
              : "var(--color-warning, #f59e0b)";

            let camStatusBg = isCamUploading
              ? "rgba(6, 182, 212, 0.1)"
              : isCamNotConfigured
              ? "rgba(148, 163, 184, 0.1)"
              : isCamComplete 
              ? "rgba(16, 185, 129, 0.1)" 
              : isCamIndexing 
              ? "rgba(6, 182, 212, 0.1)" 
              : isCamFailed 
              ? "rgba(244, 63, 94, 0.1)" 
              : "rgba(245, 158, 11, 0.1)";

            const fileName = cam.source_path ? cam.source_path.split("/").pop()?.split("\\").pop() : null;

            return (
              <div 
                key={cam.id}
                className="card"
                style={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  padding: "18px",
                  border: "1px solid var(--border-color)",
                  backgroundColor: "var(--bg-card)",
                  borderRadius: "var(--radius-main)"
                }}
              >
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <span style={{ fontSize: "10px", fontWeight: "700", letterSpacing: "0.5px", color: "var(--color-accent)" }}>
                      {camNumberStr}
                    </span>
                    <span 
                      style={{ 
                        fontSize: "10px", 
                        fontWeight: "700", 
                        padding: "2px 6px", 
                        borderRadius: "4px",
                        backgroundColor: camStatusBg,
                        color: camStatusColor,
                        textTransform: "uppercase",
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "4px"
                      }}
                    >
                      {isCamComplete && <CheckCircle2 size={11} />}
                      {camStatusText}
                    </span>
                  </div>

                  <h3 style={{ fontSize: "15px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "4px" }}>
                    {cam.name}
                  </h3>

                  {fileName ? (
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: "12px", display: "flex", alignItems: "center", gap: "5px" }}>
                      <Video size={12} style={{ flexShrink: 0, color: "var(--color-accent)" }} />
                      <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{fileName}</span>
                    </div>
                  ) : (
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "12px" }}>
                      No video footage assigned
                    </div>
                  )}

                  {/* Individual Camera Progress (for Indexing or Completed) */}
                  {(isCamIndexing || isCamComplete) && (
                    <div style={{ width: "100%", height: "6px", backgroundColor: "var(--bg-input)", borderRadius: "3px", overflow: "hidden", marginBottom: "12px" }}>
                      <div 
                        style={{ 
                          width: `${camProgress}%`, 
                          height: "100%", 
                          backgroundColor: isCamComplete ? "var(--color-success)" : "var(--color-accent)",
                          transition: "width 0.4s ease"
                        }} 
                      />
                    </div>
                  )}

                  {/* Upload Dropzone for NOT_CONFIGURED cameras */}
                  {isCamNotConfigured && !isCamUploading && (
                    <div 
                      onClick={() => fileInputRefs.current[cam.id]?.click()}
                      style={{
                        border: "1.5px dashed var(--border-color)",
                        borderRadius: "var(--radius-main)",
                        padding: "12px",
                        backgroundColor: "var(--bg-input)",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: "8px",
                        color: "var(--text-secondary)",
                        marginBottom: "12px",
                        transition: "border-color 0.2s ease"
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--color-accent)")}
                      onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border-color)")}
                    >
                      <input 
                        type="file"
                        accept="video/*"
                        ref={el => { fileInputRefs.current[cam.id] = el; }}
                        style={{ display: "none" }}
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            handleCameraFileUpload(cam.id, e.target.files[0]);
                          }
                        }}
                      />
                      <Upload size={14} style={{ color: "var(--color-accent)" }} />
                      <span style={{ fontSize: "11px", fontWeight: "500" }}>Upload Video Footage</span>
                    </div>
                  )}

                  {/* Uploading State Indicator */}
                  {isCamUploading && (
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "10px 12px", borderRadius: "6px", backgroundColor: "rgba(6, 182, 212, 0.08)", marginBottom: "12px" }}>
                      <Loader2 size={14} className="animate-spin" style={{ color: "var(--color-accent)" }} />
                      <span style={{ fontSize: "11px", color: "var(--color-accent)", fontWeight: "500" }}>Uploading video to server...</span>
                    </div>
                  )}

                  {/* Per-camera start control for READY cameras */}
                  {isCamReady && (
                    <div style={{ marginTop: "4px", marginBottom: "12px" }}>
                      <button
                        type="button"
                        className="btn btn-primary"
                        onClick={() => handleStartCamera(cam.id)}
                        disabled={actionLoading || startingCams[cam.id]}
                        style={{ fontSize: "11px", padding: "6px 12px", width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px", fontWeight: "600" }}
                      >
                        {startingCams[cam.id] ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                        Start Indexing
                      </button>
                    </div>
                  )}

                  {/* Failed state: Retry button */}
                  {isCamFailed && (
                    <div style={{ marginTop: "4px", marginBottom: "12px" }}>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={() => handleStartCamera(cam.id)}
                        disabled={actionLoading || startingCams[cam.id]}
                        style={{ fontSize: "11px", padding: "6px 12px", width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px", color: "var(--color-danger)" }}
                      >
                        {startingCams[cam.id] ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                        Retry Indexing
                      </button>
                    </div>
                  )}
                </div>

                {/* Footer Metrics */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: "10px", borderTop: "1px solid var(--border-color)", fontSize: "11px", color: "var(--text-secondary)" }}>
                  <span>Tracks: <strong style={{ color: "var(--text-primary)" }}>{cam.tracks_count !== undefined ? cam.tracks_count : camTracks.length}</strong></span>
                  <span>Progress: <strong style={{ color: "var(--text-primary)" }}>{camProgress}%</strong></span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

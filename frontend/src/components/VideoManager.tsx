import React, { useState } from "react";
import { Film, Radio, Plus, Trash2, ShieldAlert, Check } from "lucide-react";
import { apiService } from "../services/api";

interface VideoManagerProps {
  onJobStarted: (jobId: string) => void;
}

export const VideoManager: React.FC<VideoManagerProps> = ({ onJobStarted }) => {
  const [jobName, setJobName] = useState("");
  const [profile, setProfile] = useState<"FAST" | "BALANCED" | "FORENSIC">("BALANCED");
  
  // Camera inputs list
  const [cameras, setCameras] = useState<{ name: string; path: string; is_live: boolean }[]>([]);
  const [uploadProgress, setUploadProgress] = useState<string>("");
  
  // Single camera input state
  const [camName, setCamName] = useState("");
  const [camPath, setCamPath] = useState("");
  const [isLive, setIsLive] = useState(false);
  
  const [statusMsg, setStatusMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [uploadingVideo, setUploadingVideo] = useState(false);

  const handleLocalVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);
      setUploadingVideo(true);
      setStatusMsg(null);
      
      const uploadedCameras: { name: string; path: string; is_live: boolean }[] = [];
      
      try {
        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          setUploadProgress(`Uploading file ${i + 1} of ${files.length}: ${file.name}...`);
          const res = await apiService.uploadVideoFile(file);
          const name = res.filename.substring(0, res.filename.lastIndexOf(".")) || res.filename;
          uploadedCameras.push({
            name: name,
            path: res.path,
            is_live: false
          });
        }
        
        setCameras(prev => [...prev, ...uploadedCameras]);
        setUploadProgress("");
        setStatusMsg({ 
          type: "success", 
          text: `Successfully uploaded and added ${files.length} video(s) to the ingestion list.` 
        });
      } catch (err: any) {
        console.error(err);
        setStatusMsg({ 
          type: "error", 
          text: err.message || "Failed to upload one or more video files." 
        });
      } finally {
        setUploadingVideo(false);
        setUploadProgress("");
      }
    }
  };


  const addCameraToList = (e: React.FormEvent) => {
    e.preventDefault();
    if (!camName.trim() || !camPath.trim()) {
      setStatusMsg({ type: "error", text: "Please provide a camera label and a valid video path or RTSP URL." });
      return;
    }
    
    setCameras([...cameras, { name: camName.trim(), path: camPath.trim(), is_live: isLive }]);
    setCamName("");
    setCamPath("");
    setIsLive(false);
    setStatusMsg(null);
  };

  const removeCameraFromList = (index: number) => {
    setCameras(cameras.filter((_, i) => i !== index));
  };

  const handleStartIndexing = async () => {
    if (!jobName.trim()) {
      setStatusMsg({ type: "error", text: "Please enter a descriptive name for this analysis job." });
      return;
    }
    if (cameras.length === 0) {
      setStatusMsg({ type: "error", text: "Please register at least one video source or live camera stream." });
      return;
    }

    setSubmitting(true);
    setStatusMsg(null);

    try {
      // 1. Register job
      const jobId = await apiService.createJob(jobName.trim(), profile, cameras);
      
      // 2. Launch job asynchronously on backend
      await apiService.startJob(jobId);
      
      setStatusMsg({ type: "success", text: `Successfully launched Job ${jobName}. Process streams indexed asynchronously.` });
      setCameras([]);
      setJobName("");
      onJobStarted(jobId);
      
    } catch (e: any) {
      console.error(e);
      setStatusMsg({ type: "error", text: e.message || "Failed to launch video analysis job." });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "20px", width: "100%" }}>
      {/* Educational Banner */}
      <div 
        style={{ 
          padding: "16px 20px", 
          borderRadius: "var(--radius-main)", 
          backgroundColor: "rgba(0, 240, 210, 0.04)", 
          border: "1px solid rgba(0, 240, 210, 0.15)",
          color: "var(--text-primary)",
          fontSize: "13px",
          display: "flex",
          flexDirection: "column",
          gap: "6px"
        }}
      >
        <div style={{ fontWeight: "700", color: "var(--color-accent)", display: "flex", alignItems: "center", gap: "6px" }}>
          💡 Spatio-Temporal Multi-Camera Journey Stitching
        </div>
        <p style={{ color: "var(--text-secondary)", lineHeight: "1.5" }}>
          To track a target moving across different cameras (e.g. entrance lobby, street crossings, loading bay exit), batch upload all video files or register multiple live feeds under the <strong>same job</strong>. The pipeline indexes the streams in parallel, allowing the <strong>Journey Stitcher</strong> to automatically reconstruct transition sequences and travel times.
        </p>
      </div>

      <div className="grid-2">
        {/* Column 1: Add video streams */}
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div className="card-title">
            <Film size={18} style={{ color: "var(--color-accent)" }} />
            Add Video Source
          </div>

          <form onSubmit={addCameraToList} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div className="form-group">
              <label className="form-label">Camera / Video Label</label>
              <input 
                type="text" 
                className="form-input" 
                placeholder="e.g. Camera 1, Front Lobby, Parking South" 
                value={camName}
                onChange={(e) => setCamName(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Video Path or RTSP URL</label>
              <input 
                type="text" 
                className="form-input" 
                placeholder="e.g. C:/CCTV/footage.mp4 or rtsp://192.168.1.100:554/stream1" 
                value={camPath}
                onChange={(e) => setCamPath(e.target.value)}
              />
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px", margin: "4px 0" }}>
              <input 
                type="checkbox" 
                id="isLiveCheckbox" 
                checked={isLive}
                onChange={(e) => setIsLive(e.target.checked)}
                style={{ cursor: "pointer", width: "16px", height: "16px" }}
              />
              <label htmlFor="isLiveCheckbox" style={{ fontSize: "14px", color: "var(--text-secondary)", cursor: "pointer" }}>
                This is a live RTSP stream (enable frame dropping on backpressure)
              </label>
            </div>

            <button type="submit" className="btn btn-secondary" style={{ marginTop: "8px" }}>
              <Plus size={16} /> Register Stream
            </button>

            <div style={{ borderTop: "1px solid var(--border-color)", paddingTop: "12px", marginTop: "8px", display: "flex", flexDirection: "column", gap: "8px" }}>
              <label className="form-label">Or Batch Upload Local Video Files</label>
              <input 
                type="file" 
                multiple
                accept="video/*" 
                onChange={handleLocalVideoUpload}
                style={{ display: "none" }}
                id="localVideoFileInput"
                disabled={uploadingVideo}
              />
              <label 
                htmlFor="localVideoFileInput" 
                className="btn btn-secondary"
                style={{ display: "inline-flex", cursor: "pointer", fontSize: "13px", padding: "10px 14px", justifyContent: "center", gap: "8px", opacity: uploadingVideo ? 0.5 : 1 }}
              >
                {uploadingVideo ? "Uploading to Server..." : "Select Local Video Files (.mp4, .avi)"}
              </label>
              {uploadProgress && (
                <div style={{ fontSize: "12px", color: "var(--color-accent)", marginTop: "4px", fontStyle: "italic" }}>
                  {uploadProgress}
                </div>
              )}
            </div>
          </form>
        </div>

        {/* Column 2: Job parameters & Cameras overview */}
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div className="card-title">
            <Radio size={18} style={{ color: "var(--color-info)" }} />
            Analytics Job Settings
          </div>

          <div className="form-group">
            <label className="form-label">Job Name</label>
            <input 
              type="text" 
              className="form-input" 
              placeholder="e.g. Weekly Ingress Audit, Incident Investigation"
              value={jobName}
              onChange={(e) => setJobName(e.target.value)}
            />
          </div>

          {/* Processing profile tabs */}
          <div className="form-group">
            <label className="form-label">Processing Profile (RTX 3050 Optimized)</label>
            <div style={{ display: "flex", gap: "8px", marginTop: "4px" }}>
              {["FAST", "BALANCED", "FORENSIC"].map((p) => {
                const active = profile === p;
                return (
                  <button
                    key={p}
                    className="btn"
                    style={{
                      flexGrow: 1,
                      backgroundColor: active ? "rgba(0, 240, 210, 0.08)" : "var(--bg-input)",
                      border: active ? "1px solid var(--color-accent)" : "1px solid var(--border-color)",
                      color: active ? "var(--color-accent)" : "var(--text-secondary)",
                      padding: "10px 0"
                    }}
                    onClick={() => setProfile(p as any)}
                  >
                    {p}
                  </button>
                );
              })}
            </div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "4px" }}>
              {profile === "FAST" && "FAST: Decodes keyframes every 4s, skips face/OCR models to maximize throughput."}
              {profile === "BALANCED" && "BALANCED: Decodes frames every 2s. Runs YOLO, sharpness-filtered face models, and OCR."}
              {profile === "FORENSIC" && "FORENSIC: High-precision. Decodes frames every 0.5s. Runs complete deep analysis."}
            </div>
          </div>

          {/* List of registered cameras */}
          <div>
            <div className="form-label" style={{ marginBottom: "8px" }}>
              Sources to Analyze ({cameras.length})
            </div>
            {cameras.length === 0 ? (
              <div style={{ padding: "20px", border: "1px dashed var(--border-color)", borderRadius: "var(--radius-main)", textAlign: "center", color: "var(--text-muted)", fontSize: "13px" }}>
                No video sources registered yet. Add sources using the left panel.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "180px", overflowY: "auto" }}>
                {cameras.map((cam, idx) => (
                  <div 
                    key={idx} 
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "8px 12px",
                      backgroundColor: "var(--bg-input)",
                      border: "1px solid var(--border-color)",
                      borderRadius: "var(--radius-main)"
                    }}
                  >
                    <div style={{ display: "flex", flexDirection: "column", gap: "2px", overflow: "hidden" }}>
                      <span style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)" }}>{cam.name}</span>
                      <span style={{ fontSize: "11px", color: "var(--text-secondary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {cam.is_live ? "RTSP: " : "File: "}{cam.path}
                      </span>
                    </div>
                    <button 
                      onClick={() => removeCameraFromList(idx)} 
                      style={{ background: "transparent", border: "none", color: "var(--color-danger)", cursor: "pointer", padding: "4px" }}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {statusMsg && (
            <div 
              style={{ 
                display: "flex", 
                gap: "8px", 
                alignItems: "center", 
                padding: "12px", 
                borderRadius: "var(--radius-main)",
                backgroundColor: statusMsg.type === "error" ? "rgba(255, 77, 109, 0.08)" : "rgba(0, 230, 118, 0.08)",
                border: statusMsg.type === "error" ? "1px solid var(--color-danger)" : "1px solid var(--color-success)",
                color: statusMsg.type === "error" ? "var(--color-danger)" : "var(--color-success)",
                fontSize: "13px"
              }}
            >
              {statusMsg.type === "error" ? <ShieldAlert size={16} /> : <Check size={16} />}
              {statusMsg.text}
            </div>
          )}

          <button 
            className="btn btn-primary" 
            onClick={handleStartIndexing}
            disabled={submitting || cameras.length === 0}
            style={{ width: "100%", padding: "14px" }}
          >
            {submitting ? "Spawning Ingestion Threads..." : "🚀 Launch Job & Start Indexing"}
          </button>
        </div>
      </div>
    </div>
  );
};

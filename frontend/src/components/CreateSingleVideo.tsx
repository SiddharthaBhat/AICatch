import React, { useState } from "react";
import { ArrowLeft, Film, Upload, Video, Loader2, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";
import { apiService } from "../services/api";

interface CreateSingleVideoProps {
  onBack: () => void;
  onJobCreated: (jobId: string) => void;
}

type FileUploadStatus = "IDLE" | "SELECTED" | "UPLOADING" | "UPLOADED" | "FAILED";

export const CreateSingleVideo: React.FC<CreateSingleVideoProps> = ({ onBack, onJobCreated }) => {
  const [caseName, setCaseName] = useState("");
  const [cameraName, setCameraName] = useState("Front Entrance");
  const [videoDate, setVideoDate] = useState(new Date().toISOString().split("T")[0]);
  const [videoTime, setVideoTime] = useState("10:00");
  const [profile, setProfile] = useState<"FAST" | "BALANCED" | "FORENSIC">("BALANCED");
  
  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<FileUploadStatus>("IDLE");
  const [uploadedPath, setUploadedPath] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  // Submitting state
  const [submitting, setSubmitting] = useState(false);
  const [generalError, setGeneralError] = useState<string | null>(null);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const uploadFileToServer = async (file: File) => {
    setUploadStatus("UPLOADING");
    setUploadError(null);
    try {
      const res = await apiService.uploadVideoFile(file);
      setUploadedPath(res.path);
      setUploadStatus("UPLOADED");
    } catch (err: any) {
      console.error("Upload error:", err);
      setUploadError(err.message || "Failed to upload video file to backend");
      setUploadStatus("FAILED");
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setUploadStatus("SELECTED");
      
      // Auto-populate case name from filename if empty
      if (!caseName.trim()) {
        const baseName = file.name.substring(0, file.name.lastIndexOf(".")) || file.name;
        setCaseName(baseName.replace(/[_-]/g, " "));
      }

      // Immediately initiate upload
      uploadFileToServer(file);
    }
  };

  const handleRetryUpload = () => {
    if (selectedFile) {
      uploadFileToServer(selectedFile);
    }
  };

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!caseName.trim()) {
      setGeneralError("Please enter an investigation / case name.");
      return;
    }
    if (!uploadedPath || uploadStatus !== "UPLOADED") {
      setGeneralError("Please select a video file and wait for upload to complete.");
      return;
    }

    setSubmitting(true);
    setGeneralError(null);
    try {
      const camLabel = cameraName.trim() || "Camera 1";
      const cameras = [{ name: camLabel, path: uploadedPath, is_live: false }];
      const jobId = await apiService.createJob(caseName.trim(), profile, cameras, "single_video");
      await apiService.startJob(jobId);
      onJobCreated(jobId);
    } catch (err: any) {
      setGeneralError(err.message || "Failed to initialize single video investigation.");
      setSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: "760px", margin: "0 auto", paddingBottom: "60px" }} className="animate-fade-in">
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "24px" }}>
        <button 
          type="button" 
          className="btn btn-secondary" 
          onClick={onBack}
          style={{ padding: "8px 12px", fontSize: "12px" }}
        >
          <ArrowLeft size={14} /> Back
        </button>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: "600", color: "var(--text-primary)" }}>
            Create Single Video Investigation
          </h1>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Configure case parameters and upload video footage for forensic indexing and search.
          </p>
        </div>
      </div>

      <form onSubmit={handleStart} style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
        {/* Main Details Card */}
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "24px" }}>
          <div className="card-title" style={{ fontSize: "14px", color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "8px" }}>
            <Film size={16} style={{ color: "var(--color-accent)" }} />
            Investigation Configuration
          </div>

          <div className="form-group">
            <label className="form-label">Investigation / Case Name *</label>
            <input 
              type="text"
              className="form-input"
              placeholder="e.g. Front Entrance Incident"
              value={caseName}
              onChange={(e) => setCaseName(e.target.value)}
              required
            />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            <div className="form-group">
              <label className="form-label">Camera Name *</label>
              <input 
                type="text"
                className="form-input"
                placeholder="e.g. Front Entrance, Lobby Cam"
                value={cameraName}
                onChange={(e) => setCameraName(e.target.value)}
                required
              />
              <span style={{ fontSize: "10px", color: "var(--text-muted)", marginTop: "4px" }}>
                Human-readable location label used on observation cards.
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Processing Profile</label>
              <select 
                className="form-select"
                value={profile}
                onChange={(e) => setProfile(e.target.value as any)}
              >
                <option value="FAST">FAST (Keyframes Only, Low Latency)</option>
                <option value="BALANCED">BALANCED (Standard Forensic Indexing)</option>
                <option value="FORENSIC">FORENSIC (Dense Sampling, High Precision)</option>
              </select>
            </div>
          </div>

          {/* Date & Start Time - Clean single layout */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            <div className="form-group">
              <label className="form-label">Investigation Date</label>
              <input 
                type="date"
                className="form-input"
                value={videoDate}
                onChange={(e) => setVideoDate(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Video Start Time</label>
              <input 
                type="time"
                className="form-input"
                value={videoTime}
                onChange={(e) => setVideoTime(e.target.value)}
              />
            </div>
          </div>

          {/* Video File Dropzone */}
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">Video Footage File *</label>
            
            <div 
              style={{
                border: "1.5px dashed var(--border-color)",
                borderRadius: "var(--radius-main)",
                padding: "24px",
                textAlign: "center",
                backgroundColor: "var(--bg-input)",
                position: "relative",
                cursor: "pointer",
                transition: "border 0.2s ease"
              }}
            >
              <input 
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  opacity: 0,
                  cursor: "pointer"
                }}
              />

              {!selectedFile ? (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px", color: "var(--text-secondary)" }}>
                  <Upload size={28} style={{ color: "var(--color-accent)" }} />
                  <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>
                    Click or drag video file to upload
                  </span>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                    Supports MP4, AVI, MOV, MKV, WebM
                  </span>
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <div style={{ padding: "8px", borderRadius: "6px", backgroundColor: "rgba(6, 182, 212, 0.1)", color: "var(--color-accent)" }}>
                      <Video size={20} />
                    </div>
                    <div style={{ textAlign: "left" }}>
                      <div style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>
                        {selectedFile.name}
                      </div>
                      <div style={{ fontSize: "11px", color: "var(--text-secondary)" }}>
                        Size: {formatFileSize(selectedFile.size)}
                      </div>
                    </div>
                  </div>

                  {/* Status Indicator */}
                  <div>
                    {uploadStatus === "UPLOADING" && (
                      <span style={{ fontSize: "11px", color: "var(--color-accent)", display: "flex", alignItems: "center", gap: "6px" }}>
                        <Loader2 size={13} className="animate-spin" /> Uploading...
                      </span>
                    )}

                    {uploadStatus === "UPLOADED" && (
                      <span style={{ fontSize: "11px", color: "var(--color-success)", fontWeight: "600", display: "flex", alignItems: "center", gap: "4px" }}>
                        <CheckCircle2 size={14} /> Ready for Indexing
                      </span>
                    )}

                    {uploadStatus === "FAILED" && (
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        <span style={{ fontSize: "11px", color: "var(--color-danger)", display: "flex", alignItems: "center", gap: "4px" }}>
                          <AlertCircle size={14} /> Upload Failed
                        </span>
                        <button 
                          type="button" 
                          className="btn btn-secondary" 
                          style={{ padding: "4px 8px", fontSize: "10px", display: "flex", alignItems: "center", gap: "4px" }}
                          onClick={(e) => { e.stopPropagation(); handleRetryUpload(); }}
                        >
                          <RefreshCw size={11} /> Retry
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {uploadError && (
              <div style={{ fontSize: "11px", color: "var(--color-danger)", marginTop: "6px", display: "flex", alignItems: "center", gap: "4px" }}>
                <AlertCircle size={12} /> {uploadError}
              </div>
            )}
          </div>
        </div>

        {generalError && (
          <div style={{ padding: "12px 16px", borderRadius: "8px", backgroundColor: "rgba(244, 63, 94, 0.1)", border: "1px solid var(--color-danger)", color: "var(--color-danger)", fontSize: "12px", display: "flex", alignItems: "center", gap: "8px" }}>
            <AlertCircle size={15} /> {generalError}
          </div>
        )}

        {/* Submit Action */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "12px" }}>
          <button 
            type="button" 
            className="btn btn-secondary" 
            onClick={onBack}
            disabled={submitting}
          >
            Cancel
          </button>
          
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={submitting || uploadStatus !== "UPLOADED" || !caseName.trim()}
            style={{ padding: "10px 24px", fontSize: "13px", fontWeight: "600", display: "flex", alignItems: "center", gap: "8px" }}
          >
            {submitting ? <Loader2 size={15} className="animate-spin" /> : <Film size={15} />}
            {submitting ? "Initializing Investigation..." : "Start Indexing"}
          </button>
        </div>
      </form>
    </div>
  );
};

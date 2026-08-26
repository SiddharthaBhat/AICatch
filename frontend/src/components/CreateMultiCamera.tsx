import React, { useState } from "react";
import { ArrowLeft, Layers, Plus, Trash2, Video, Loader2, CheckCircle2, AlertCircle, Upload } from "lucide-react";
import { apiService } from "../services/api";

interface CameraDraft {
  id: string;
  name: string;
  file: File | null;
  path: string | null;
  time: string;
  dateOverride?: string;
  uploadStatus: "IDLE" | "SELECTED" | "UPLOADING" | "UPLOADED" | "FAILED";
  error?: string;
}

interface CreateMultiCameraProps {
  onBack: () => void;
  onJobCreated: (jobId: string) => void;
}

export const CreateMultiCamera: React.FC<CreateMultiCameraProps> = ({ onBack, onJobCreated }) => {
  const [caseName, setCaseName] = useState("");
  const [investigationDate, setInvestigationDate] = useState(new Date().toISOString().split("T")[0]);
  const [syncDateTime, setSyncDateTime] = useState(true);
  const [profile, setProfile] = useState<"FAST" | "BALANCED" | "FORENSIC">("BALANCED");
  
  const [cameras, setCameras] = useState<CameraDraft[]>([
    {
      id: "cam_1",
      name: "Front Entrance",
      file: null,
      path: null,
      time: "10:00",
      uploadStatus: "IDLE"
    },
    {
      id: "cam_2",
      name: "Balcony",
      file: null,
      path: null,
      time: "10:02",
      uploadStatus: "IDLE"
    }
  ]);

  const [submitting, setSubmitting] = useState(false);
  const [generalError, setGeneralError] = useState<string | null>(null);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const uploadCameraFile = async (camId: string, file: File) => {
    setCameras(prev => prev.map(c => c.id === camId ? { ...c, uploadStatus: "UPLOADING", error: undefined } : c));
    try {
      const res = await apiService.uploadVideoFile(file);
      setCameras(prev => prev.map(c => c.id === camId ? { ...c, path: res.path, uploadStatus: "UPLOADED" } : c));
    } catch (err: any) {
      console.error(`Upload error for camera ${camId}:`, err);
      setCameras(prev => prev.map(c => c.id === camId ? { ...c, uploadStatus: "FAILED", error: err.message || "Upload failed" } : c));
    }
  };

  const handleFileChange = (camId: string, e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setCameras(prev => prev.map(c => c.id === camId ? { ...c, file, uploadStatus: "SELECTED" } : c));
      uploadCameraFile(camId, file);
    }
  };

  const handleRetryCameraUpload = (camId: string) => {
    const cam = cameras.find(c => c.id === camId);
    if (cam && cam.file) {
      uploadCameraFile(camId, cam.file);
    }
  };

  const handleAddCamera = () => {
    const nextIdx = cameras.length + 1;
    const nextTimeMin = (nextIdx - 1) * 2;
    const nextTimeStr = `10:${nextTimeMin.toString().padStart(2, "0")}`;
    const newCam: CameraDraft = {
      id: `cam_${Date.now()}_${nextIdx}`,
      name: nextIdx === 3 ? "Parking Area" : nextIdx === 4 ? "Back Exit" : `Camera ${nextIdx}`,
      file: null,
      path: null,
      time: nextTimeStr,
      uploadStatus: "IDLE"
    };
    setCameras(prev => [...prev, newCam]);
  };

  const handleRemoveCamera = (camId: string) => {
    if (cameras.length <= 1) return;
    setCameras(prev => prev.filter(c => c.id !== camId));
  };

  const handleCameraNameChange = (camId: string, name: string) => {
    setCameras(prev => prev.map(c => c.id === camId ? { ...c, name } : c));
  };

  const handleCameraTimeChange = (camId: string, time: string) => {
    setCameras(prev => prev.map(c => c.id === camId ? { ...c, time } : c));
  };

  const handleCameraDateOverride = (camId: string, dateOverride: string) => {
    setCameras(prev => prev.map(c => c.id === camId ? { ...c, dateOverride } : c));
  };

  const allUploaded = cameras.length >= 2 && cameras.every(c => c.uploadStatus === "UPLOADED" && c.path);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!caseName.trim()) {
      setGeneralError("Please enter an investigation / case name.");
      return;
    }
    if (cameras.length < 2) {
      setGeneralError("A Multi-Camera Stitch investigation requires at least 2 camera sources.");
      return;
    }
    if (!allUploaded) {
      setGeneralError("Please ensure all camera sources have uploaded video files.");
      return;
    }

    setSubmitting(true);
    setGeneralError(null);
    try {
      const cameraPayload = cameras.map((c, idx) => ({
        name: c.name.trim() || `Camera ${idx + 1}`,
        path: c.path!,
        is_live: false
      }));

      const jobId = await apiService.createJob(caseName.trim(), profile, cameraPayload, "multi_camera");
      await apiService.startJob(jobId);
      onJobCreated(jobId);
    } catch (err: any) {
      setGeneralError(err.message || "Failed to initialize multi-camera investigation.");
      setSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: "860px", margin: "0 auto", paddingBottom: "60px" }} className="animate-fade-in">
      {/* Top Header */}
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
            Create Multi-Camera Stitch Investigation
          </h1>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Configure multiple named camera locations for automated spatio-temporal tracking across feeds.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
        {/* Investigation-Level Card */}
        <div className="card" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "24px" }}>
          <div className="card-title" style={{ fontSize: "14px", color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "8px" }}>
            <Layers size={16} style={{ color: "var(--color-success)" }} />
            Investigation Configuration
          </div>

          <div className="form-group">
            <label className="form-label">Investigation / Case Name *</label>
            <input 
              type="text"
              className="form-input"
              placeholder="e.g. Mall Multi-Camera Incident"
              value={caseName}
              onChange={(e) => setCaseName(e.target.value)}
              required
            />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            <div className="form-group">
              <label className="form-label">Investigation Date</label>
              <input 
                type="date"
                className="form-input"
                value={investigationDate}
                onChange={(e) => setInvestigationDate(e.target.value)}
              />
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

          {/* Synchronization Option */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px", paddingTop: "4px" }}>
            <input 
              type="checkbox"
              id="syncDateTime"
              checked={syncDateTime}
              onChange={(e) => setSyncDateTime(e.target.checked)}
              style={{ accentColor: "var(--color-accent)", cursor: "pointer", width: "16px", height: "16px" }}
            />
            <label htmlFor="syncDateTime" style={{ fontSize: "12px", color: "var(--text-primary)", cursor: "pointer", userSelect: "none" }}>
              Cameras share the same date basis ({investigationDate})
            </label>
          </div>
        </div>

        {/* Camera Sources List */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
            <h2 style={{ fontSize: "15px", fontWeight: "600", color: "var(--text-primary)" }}>
              Camera Sources ({cameras.length})
            </h2>

            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={handleAddCamera}
              style={{ fontSize: "12px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "6px" }}
            >
              <Plus size={14} /> Add Camera
            </button>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {cameras.map((cam, idx) => {
              const camNum = `CAMERA ${(idx + 1).toString().padStart(2, "0")}`;
              return (
                <div 
                  key={cam.id} 
                  className="card"
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "14px",
                    padding: "20px",
                    border: "1px solid var(--border-color)",
                    backgroundColor: "var(--bg-card)"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                      <span style={{ fontSize: "11px", fontWeight: "700", color: "var(--color-accent)", letterSpacing: "0.5px" }}>
                        {camNum}
                      </span>
                      <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>
                        {cam.name || `Camera ${idx + 1}`}
                      </span>
                    </div>

                    {cameras.length > 2 && (
                      <button 
                        type="button" 
                        className="btn btn-secondary"
                        onClick={() => handleRemoveCamera(cam.id)}
                        style={{ padding: "4px 8px", color: "var(--color-danger)", border: "none" }}
                        title="Remove camera"
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>

                  {/* Camera Inputs */}
                  <div style={{ display: "grid", gridTemplateColumns: syncDateTime ? "2fr 1fr" : "2fr 1fr 1fr", gap: "12px" }}>
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label" style={{ fontSize: "11px" }}>Camera Location Name *</label>
                      <input 
                        type="text"
                        className="form-input"
                        placeholder="e.g. Front Entrance, Balcony, Gate"
                        value={cam.name}
                        onChange={(e) => handleCameraNameChange(cam.id, e.target.value)}
                        required
                        style={{ fontSize: "12px" }}
                      />
                    </div>

                    {!syncDateTime && (
                      <div className="form-group" style={{ marginBottom: 0 }}>
                        <label className="form-label" style={{ fontSize: "11px" }}>Override Date</label>
                        <input 
                          type="date"
                          className="form-input"
                          value={cam.dateOverride || investigationDate}
                          onChange={(e) => handleCameraDateOverride(cam.id, e.target.value)}
                          style={{ fontSize: "12px" }}
                        />
                      </div>
                    )}

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label" style={{ fontSize: "11px" }}>Start Time</label>
                      <input 
                        type="time"
                        className="form-input"
                        value={cam.time}
                        onChange={(e) => handleCameraTimeChange(cam.id, e.target.value)}
                        style={{ fontSize: "12px" }}
                      />
                    </div>
                  </div>

                  {/* Video Upload Dropzone */}
                  <div 
                    style={{
                      border: "1.5px dashed var(--border-color)",
                      borderRadius: "var(--radius-main)",
                      padding: "16px",
                      backgroundColor: "var(--bg-input)",
                      position: "relative",
                      cursor: "pointer"
                    }}
                  >
                    <input 
                      type="file"
                      accept="video/*"
                      onChange={(e) => handleFileChange(cam.id, e)}
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        opacity: 0,
                        cursor: "pointer"
                      }}
                    />

                    {!cam.file ? (
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", color: "var(--text-secondary)" }}>
                        <Upload size={16} style={{ color: "var(--color-accent)" }} />
                        <span style={{ fontSize: "12px" }}>
                          Click or drag video file for {cam.name || `Camera ${idx + 1}`}
                        </span>
                      </div>
                    ) : (
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                          <Video size={18} style={{ color: "var(--color-accent)" }} />
                          <div>
                            <div style={{ fontSize: "12px", fontWeight: "600", color: "var(--text-primary)" }}>
                              {cam.file.name}
                            </div>
                            <div style={{ fontSize: "10px", color: "var(--text-secondary)" }}>
                              {formatFileSize(cam.file.size)}
                            </div>
                          </div>
                        </div>

                        <div>
                          {cam.uploadStatus === "UPLOADING" && (
                            <span style={{ fontSize: "11px", color: "var(--color-accent)", display: "flex", alignItems: "center", gap: "4px" }}>
                              <Loader2 size={12} className="animate-spin" /> Uploading...
                            </span>
                          )}

                          {cam.uploadStatus === "UPLOADED" && (
                            <span style={{ fontSize: "11px", color: "var(--color-success)", fontWeight: "600", display: "flex", alignItems: "center", gap: "4px" }}>
                              <CheckCircle2 size={13} /> Uploaded ✓
                            </span>
                          )}

                          {cam.uploadStatus === "FAILED" && (
                            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                              <span style={{ fontSize: "11px", color: "var(--color-danger)", display: "flex", alignItems: "center", gap: "4px" }}>
                                <AlertCircle size={13} /> Failed
                              </span>
                              <button 
                                type="button"
                                className="btn btn-secondary"
                                style={{ padding: "2px 6px", fontSize: "10px" }}
                                onClick={(e) => { e.stopPropagation(); handleRetryCameraUpload(cam.id); }}
                              >
                                Retry
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {cam.error && (
                    <div style={{ fontSize: "11px", color: "var(--color-danger)", display: "flex", alignItems: "center", gap: "4px" }}>
                      <AlertCircle size={12} /> {cam.error}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {generalError && (
          <div style={{ padding: "12px 16px", borderRadius: "8px", backgroundColor: "rgba(244, 63, 94, 0.1)", border: "1px solid var(--color-danger)", color: "var(--color-danger)", fontSize: "12px", display: "flex", alignItems: "center", gap: "8px" }}>
            <AlertCircle size={15} /> {generalError}
          </div>
        )}

        {/* Submit Actions */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "12px", paddingTop: "8px" }}>
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
            disabled={submitting || !allUploaded || !caseName.trim()}
            style={{ padding: "10px 24px", fontSize: "13px", fontWeight: "600", display: "flex", alignItems: "center", gap: "8px" }}
          >
            {submitting ? <Loader2 size={15} className="animate-spin" /> : <Layers size={15} />}
            {submitting ? "Initializing Multi-Camera Stitch..." : "Start Multi-Camera Indexing"}
          </button>
        </div>
      </form>
    </div>
  );
};

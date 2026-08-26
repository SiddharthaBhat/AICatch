import React, { useState } from "react";
import { Video, Layers, ArrowRight, Clock, Plus, Film, AlertCircle, CheckCircle2, Trash2, AlertTriangle } from "lucide-react";
import type { Job } from "../types";
import { apiService } from "../services/api";

interface DashboardProps {
  jobs: Job[];
  onStartSingleVideo: () => void;
  onStartMultiCamera: () => void;
  onOpenInvestigation: (jobId: string) => void;
  onRefreshJobs?: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({
  jobs,
  onStartSingleVideo,
  onStartMultiCamera,
  onOpenInvestigation,
  onRefreshJobs
}) => {
  const [jobToDelete, setJobToDelete] = useState<Job | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const getStatusBadge = (status: Job["status"]) => {
    switch (status) {
      case "completed":
        return <span className="badge badge-completed"><CheckCircle2 size={12} style={{ marginRight: "4px" }} /> Ready</span>;
      case "indexing":
        return <span className="badge badge-indexing">Indexing</span>;
      case "failed":
        return <span className="badge badge-failed"><AlertCircle size={12} style={{ marginRight: "4px" }} /> Failed</span>;
      case "paused":
        return <span className="badge badge-inactive">Paused</span>;
      default:
        return <span className="badge badge-queued">{status}</span>;
    }
  };

  const formatDate = (timestamp: number) => {
    if (!timestamp) return "Recent";
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  };

  const handleDeleteConfirm = async () => {
    if (!jobToDelete) return;
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await apiService.deleteJob(jobToDelete.id);
      setJobToDelete(null);
      if (onRefreshJobs) {
        onRefreshJobs();
      }
    } catch (err: any) {
      console.error("Failed to delete job:", err);
      setDeleteError(err.message || "Failed to delete investigation.");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px", maxWidth: "1200px", margin: "0 auto", paddingBottom: "40px" }}>
      {/* Top Welcome Header */}
      <div>
        <h1 style={{ fontSize: "24px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "6px" }}>
          What do you want to investigate?
        </h1>
        <p style={{ fontSize: "14px", color: "var(--text-secondary)" }}>
          Select an investigation mode to begin indexing video footage and searching observations.
        </p>
      </div>

      {/* Two Large Primary Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "24px" }}>
        {/* Single Video Card */}
        <div 
          className="card" 
          style={{ 
            display: "flex", 
            flexDirection: "column", 
            justifyContent: "space-between", 
            padding: "28px", 
            border: "1px solid var(--border-color)",
            backgroundColor: "var(--bg-card)",
            borderRadius: "var(--radius-main)",
            cursor: "pointer",
            transition: "all 0.2s ease"
          }}
          onClick={onStartSingleVideo}
        >
          <div>
            <div style={{ display: "inline-flex", padding: "12px", borderRadius: "10px", backgroundColor: "rgba(6, 182, 212, 0.1)", color: "var(--color-accent)", marginBottom: "16px" }}>
              <Film size={28} />
            </div>
            <h2 style={{ fontSize: "18px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "8px" }}>
              SINGLE VIDEO
            </h2>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.5", marginBottom: "20px" }}>
              Search and investigate a single video or footage stream without cross-camera stitching. Fast standalone forensic indexing.
            </p>
          </div>
          <button 
            type="button" 
            className="btn btn-primary" 
            style={{ width: "100%", padding: "12px 16px", fontWeight: "600", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}
            onClick={(e) => { e.stopPropagation(); onStartSingleVideo(); }}
          >
            <Plus size={16} /> Start Single Video
          </button>
        </div>

        {/* Multi-Camera Stitch Card */}
        <div 
          className="card" 
          style={{ 
            display: "flex", 
            flexDirection: "column", 
            justifyContent: "space-between", 
            padding: "28px", 
            border: "1px solid var(--border-color)",
            backgroundColor: "var(--bg-card)",
            borderRadius: "var(--radius-main)",
            cursor: "pointer",
            transition: "all 0.2s ease"
          }}
          onClick={onStartMultiCamera}
        >
          <div>
            <div style={{ display: "inline-flex", padding: "12px", borderRadius: "10px", backgroundColor: "rgba(16, 185, 129, 0.1)", color: "var(--color-success)", marginBottom: "16px" }}>
              <Layers size={28} />
            </div>
            <h2 style={{ fontSize: "18px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "8px" }}>
              MULTI-CAMERA STITCH
            </h2>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: "1.5", marginBottom: "20px" }}>
              Track a subject across multiple named camera locations with automated spatio-temporal stitching and journey reconstruction.
            </p>
          </div>
          <button 
            type="button" 
            className="btn" 
            style={{ 
              width: "100%", 
              padding: "12px 16px", 
              fontWeight: "600", 
              display: "flex", 
              alignItems: "center", 
              justifyContent: "center", 
              gap: "8px",
              backgroundColor: "var(--color-accent)",
              color: "#000"
            }}
            onClick={(e) => { e.stopPropagation(); onStartMultiCamera(); }}
          >
            <Plus size={16} /> Start Multi-Camera
          </button>
        </div>
      </div>

      {/* Recent Investigations Section */}
      <div style={{ marginTop: "12px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <h2 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)" }}>
            Recent Investigations
          </h2>
          <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
            {jobs.length} case{jobs.length === 1 ? "" : "s"} registered
          </span>
        </div>

        {jobs.length === 0 ? (
          <div 
            className="card" 
            style={{ 
              textAlign: "center", 
              padding: "48px 24px", 
              color: "var(--text-secondary)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "12px"
            }}
          >
            <Video size={40} style={{ color: "var(--text-muted)", strokeWidth: 1.5 }} />
            <div style={{ fontSize: "15px", fontWeight: "500", color: "var(--text-primary)" }}>
              No investigations yet.
            </div>
            <p style={{ fontSize: "13px", color: "var(--text-muted)", maxWidth: "380px" }}>
              Get started by launching a Single Video or Multi-Camera investigation above.
            </p>
            <div style={{ display: "flex", gap: "12px", marginTop: "8px" }}>
              <button type="button" className="btn btn-secondary" onClick={onStartSingleVideo} style={{ fontSize: "12px", padding: "8px 14px" }}>
                + Single Video
              </button>
              <button type="button" className="btn btn-primary" onClick={onStartMultiCamera} style={{ fontSize: "12px", padding: "8px 14px" }}>
                + Multi-Camera
              </button>
            </div>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "16px" }}>
            {jobs.map((job) => {
              const isMulti = job.investigation_type === "multi_camera";
              return (
                <div
                  key={job.id}
                  className="card"
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    padding: "20px",
                    cursor: "pointer",
                    border: "1px solid var(--border-color)",
                    backgroundColor: "var(--bg-card)",
                    transition: "all 0.2s ease"
                  }}
                  onClick={() => onOpenInvestigation(job.id)}
                >
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                      <span 
                        style={{ 
                          fontSize: "10px", 
                          fontWeight: "700", 
                          letterSpacing: "0.5px",
                          padding: "3px 8px", 
                          borderRadius: "4px",
                          backgroundColor: isMulti ? "rgba(16, 185, 129, 0.1)" : "rgba(6, 182, 212, 0.1)",
                          color: isMulti ? "var(--color-success)" : "var(--color-accent)"
                        }}
                      >
                        {isMulti ? "MULTI-CAMERA" : "SINGLE VIDEO"}
                      </span>
                      
                      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                        {getStatusBadge(job.status)}
                        <button
                          type="button"
                          className="btn btn-secondary"
                          style={{ padding: "4px", color: "var(--text-muted)", border: "none", backgroundColor: "transparent" }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setJobToDelete(job);
                          }}
                          title="Delete Investigation"
                        >
                          <Trash2 size={14} className="hover:text-red-400" />
                        </button>
                      </div>
                    </div>

                    <h3 style={{ fontSize: "15px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "6px" }}>
                      {job.name}
                    </h3>

                    <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "14px", display: "flex", alignItems: "center", gap: "6px" }}>
                      <Clock size={13} style={{ color: "var(--text-muted)" }} />
                      Created: {formatDate(job.created_at)}
                    </div>
                  </div>

                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: "12px", borderTop: "1px solid var(--border-color)" }}>
                    <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                      {(job as any).sources || (isMulti ? "Multiple cameras" : "1 Camera")}
                    </span>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ fontSize: "12px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "4px" }}
                      onClick={(e) => { e.stopPropagation(); onOpenInvestigation(job.id); }}
                    >
                      Open <ArrowRight size={13} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Delete Investigation Confirmation Modal */}
      {jobToDelete && (
        <div 
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.8)",
            backdropFilter: "blur(4px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 99999,
            padding: "20px"
          }}
          onClick={() => !isDeleting && setJobToDelete(null)}
        >
          <div 
            className="card"
            style={{
              maxWidth: "480px",
              width: "100%",
              padding: "24px",
              display: "flex",
              flexDirection: "column",
              gap: "16px",
              backgroundColor: "var(--bg-card)",
              border: "1px solid var(--border-color)"
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <div style={{ padding: "10px", borderRadius: "8px", backgroundColor: "rgba(244, 63, 94, 0.1)", color: "var(--color-danger)" }}>
                <AlertTriangle size={24} />
              </div>
              <div>
                <h3 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)" }}>
                  Delete Investigation?
                </h3>
                <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginTop: "2px" }}>
                  Case: <strong>{jobToDelete.name}</strong>
                </p>
              </div>
            </div>

            <div style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.6", backgroundColor: "var(--bg-input)", padding: "14px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <p style={{ marginBottom: "6px", fontWeight: "600", color: "var(--text-primary)" }}>
                This action will permanently delete:
              </p>
              <ul style={{ paddingLeft: "18px", margin: 0, display: "flex", flexDirection: "column", gap: "4px" }}>
                <li>Investigation metadata and camera sources</li>
                <li>All indexed observations and track instances</li>
                <li>ChromaDB vector embeddings and facial features</li>
                <li>Saved evidence items, clips, and bookmarks</li>
                <li>Associated case notes and reconstructed journeys</li>
              </ul>
            </div>

            {deleteError && (
              <div style={{ fontSize: "12px", color: "var(--color-danger)" }}>
                {deleteError}
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", marginTop: "4px" }}>
              <button 
                type="button"
                className="btn btn-secondary"
                onClick={() => setJobToDelete(null)}
                disabled={isDeleting}
              >
                Cancel
              </button>

              <button 
                type="button"
                className="btn"
                style={{ backgroundColor: "var(--color-danger)", color: "#FFF", fontWeight: "600" }}
                onClick={handleDeleteConfirm}
                disabled={isDeleting}
              >
                {isDeleting ? "Deleting..." : "Delete Investigation"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

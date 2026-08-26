import React, { useState } from "react";
import { Folder, Clock, Film, Layers, CheckCircle2, AlertCircle, Trash2, AlertTriangle, ArrowRight } from "lucide-react";
import type { Job } from "../types";
import { apiService } from "../services/api";

interface InvestigationsListProps {
  jobs: Job[];
  onOpenJob?: (jobId: string) => void;
  onOpenInvestigation?: (jobId: string) => void;
  onCreateSingle?: () => void;
  onCreateMulti?: () => void;
  onStartSingleVideo?: () => void;
  onStartMultiCamera?: () => void;
  onRefreshJobs?: () => void;
  onRefresh?: () => Promise<void> | void;
}

export const InvestigationsList: React.FC<InvestigationsListProps> = ({
  jobs,
  onOpenJob,
  onOpenInvestigation,
  onCreateSingle,
  onCreateMulti,
  onStartSingleVideo,
  onStartMultiCamera,
  onRefreshJobs,
  onRefresh
}) => {
  const handleOpen = (id: string) => {
    if (onOpenJob) onOpenJob(id);
    else if (onOpenInvestigation) onOpenInvestigation(id);
  };
  const handleCreateSingle = () => {
    if (onCreateSingle) onCreateSingle();
    else if (onStartSingleVideo) onStartSingleVideo();
  };
  const handleCreateMulti = () => {
    if (onCreateMulti) onCreateMulti();
    else if (onStartMultiCamera) onStartMultiCamera();
  };
  const handleRefresh = () => {
    if (onRefreshJobs) onRefreshJobs();
    if (onRefresh) onRefresh();
  };
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
      handleRefresh();
    } catch (err: any) {
      console.error("Failed to delete job:", err);
      setDeleteError(err.message || "Failed to delete investigation.");
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px", maxWidth: "1200px", margin: "0 auto", paddingBottom: "40px" }} className="animate-fade-in">
      {/* Top Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: "600", color: "var(--text-primary)" }}>
            All Investigations
          </h1>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Manage active and completed forensic video cases.
          </p>
        </div>

        <div style={{ display: "flex", gap: "10px" }}>
          <button type="button" className="btn btn-secondary" onClick={handleCreateSingle} style={{ fontSize: "12px" }}>
            <Film size={14} /> + Single Video
          </button>
          <button type="button" className="btn btn-primary" onClick={handleCreateMulti} style={{ fontSize: "12px" }}>
            <Layers size={14} /> + Multi-Camera
          </button>
        </div>
      </div>

      {jobs.length === 0 ? (
        <div className="card" style={{ padding: "48px", textAlign: "center", color: "var(--text-secondary)" }}>
          <Folder size={40} style={{ color: "var(--text-muted)", margin: "0 auto 12px" }} />
          <h3 style={{ fontSize: "16px", color: "var(--text-primary)", marginBottom: "4px" }}>No investigations found</h3>
          <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>Create a new case using the buttons above.</p>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: "16px" }}>
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
                onClick={() => handleOpen(job.id)}
              >
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "12px" }}>
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

                  <h3 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "6px" }}>
                    {job.name}
                  </h3>

                  <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "16px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <Clock size={13} style={{ color: "var(--text-muted)" }} />
                    Created: {formatDate(job.created_at)}
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: "14px", borderTop: "1px solid var(--border-color)" }}>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                    Profile: {job.profile}
                  </span>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{ fontSize: "12px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "4px" }}
                    onClick={(e) => { e.stopPropagation(); handleOpen(job.id); }}
                  >
                    Open <ArrowRight size={13} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Delete Confirmation Modal */}
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

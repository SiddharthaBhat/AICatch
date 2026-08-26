import React, { useState, useEffect } from "react";
import { Folder, Plus, Trash2, X, Download, Play, User } from "lucide-react";
import { apiService, getMediaUrl } from "../services/api";
import { ForensicVideoViewer } from "./ForensicVideoViewer";
import type { ObservationTarget } from "./ForensicVideoViewer";
import type { Camera } from "../types";

interface EvidenceProps {
  jobId?: string | null;
}

export const Evidence: React.FC<EvidenceProps> = ({ jobId }) => {
  const [folders, setFolders] = useState<any[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<any | null>(null);
  const [folderItems, setFolderItems] = useState<any[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  
  // Folder Creation Form
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [loading, setLoading] = useState(true);

  // Folder notes state
  const [notes, setNotes] = useState<any[]>([]);
  const [newNoteContent, setNewNoteContent] = useState("");

  // Folder summary states
  const [folderSummary, setFolderSummary] = useState<string>("");
  const [loadingSummary, setLoadingSummary] = useState<boolean>(false);

  // Forensic Video Viewer Modal
  const [viewerObservation, setViewerObservation] = useState<ObservationTarget | null>(null);
  const [showVideoViewer, setShowVideoViewer] = useState(false);

  const loadFolders = async () => {
    try {
      const data = await apiService.listFolders(jobId || undefined);
      setFolders(data);
      if (jobId) {
        const cams = await apiService.getJobCameras(jobId);
        setCameras(cams);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadFolders();
    setSelectedFolder(null);
    setFolderItems([]);
    setNotes([]);
    setFolderSummary("");
  }, [jobId]);

  const handleCreateFolder = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !jobId) return;
    try {
      await apiService.createFolder(name.trim(), description.trim(), jobId);
      setName("");
      setDescription("");
      setShowCreate(false);
      loadFolders();
    } catch (e) {
      alert("Failed to create folder.");
    }
  };

  const handleSelectFolder = async (folder: any) => {
    setSelectedFolder(folder);
    try {
      const items = await apiService.listFolderItems(folder.id);
      const parsedItems = items.map((it: any) => {
        let meta = {};
        try {
          meta = JSON.parse(it.metadata_json || "{}");
        } catch (e) {
          meta = {};
        }
        return { ...it, parsedMeta: meta };
      });
      setFolderItems(parsedItems);

      const folderNotes = await apiService.listNotes("folder", folder.id);
      setNotes(folderNotes);
      setFolderSummary("");
    } catch (e) {
      console.error("Failed to load folder details", e);
    }
  };

  const handleDeleteFolder = async (folderId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this folder?")) return;
    try {
      await apiService.deleteFolder(folderId);
      if (selectedFolder && selectedFolder.id === folderId) {
        setSelectedFolder(null);
        setFolderItems([]);
      }
      loadFolders();
    } catch (e) {
      alert("Failed to delete folder");
    }
  };

  const handleRemoveItem = async (itemId: string) => {
    if (!selectedFolder) return;
    try {
      await apiService.removeFolderItem(selectedFolder.id, itemId);
      setFolderItems(prev => prev.filter(i => i.item_id !== itemId));
    } catch (e) {
      alert("Failed to remove item from folder");
    }
  };

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFolder || !newNoteContent.trim()) return;
    try {
      const newNote = await apiService.addNote("folder", selectedFolder.id, newNoteContent.trim());
      setNotes(prev => [newNote, ...prev]);
      setNewNoteContent("");
    } catch (e) {
      alert("Failed to add note.");
    }
  };

  const handleDeleteNote = async (noteId: string) => {
    if (!selectedFolder) return;
    try {
      await apiService.deleteNote(noteId);
      setNotes(prev => prev.filter(n => n.id !== noteId));
    } catch (e) {
      alert("Failed to delete note.");
    }
  };

  const handleExportFolder = () => {
    if (!selectedFolder) return;
    const exportPayload = {
      folder: selectedFolder,
      items: folderItems,
      notes: notes
    };
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `aicatch_folder_export_${selectedFolder.name.toLowerCase().replace(/\s+/g, "_")}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleGenerateFolderSummary = async () => {
    if (!selectedFolder) return;
    setLoadingSummary(true);
    try {
      const res = await apiService.getFolderSummary(selectedFolder.id);
      setFolderSummary(res.summary);
    } catch (e) {
      alert("Failed to generate folder summary.");
    } finally {
      setLoadingSummary(false);
    }
  };

  const handleOpenEvidenceVideo = (data: any) => {
    setViewerObservation({
      jobId: data.job_id || jobId || "",
      cameraId: data.camera_id || "",
      cameraName: data.camera_name || data.camera_id || "Camera",
      timestamp: data.timestamp || 0,
      trackId: data.track_id,
      trackUuid: data.track_uuid || (typeof data.track_id === "string" ? data.track_id : undefined),
      cropPath: data.crop_path,
      entityType: data.entity_type,
      confidence: data.confidence,
      similarity: data.similarity,
      label: data.label
    });
    setShowVideoViewer(true);
  };

  return (
    <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* 1. Header with create option */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "16px", fontWeight: "600", color: "var(--text-primary)" }}>
            Forensic Evidence Inquests
          </h2>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
            Organize observations, timestamps, clips and logs into case folder archives.
          </p>
        </div>
        {!selectedFolder && (
          <button className="btn btn-primary" onClick={() => setShowCreate(!showCreate)}>
            <Plus size={16} /> New Folder
          </button>
        )}
      </div>

      {/* Create New Folder form */}
      {showCreate && !selectedFolder && (
        <form onSubmit={handleCreateFolder} className="card" style={{ display: "flex", flexDirection: "column", gap: "12px", padding: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "13px", fontWeight: "700" }}>Create Investigation Folder</span>
            <button type="button" onClick={() => setShowCreate(false)} style={{ background: "none", border: "none", color: "var(--text-secondary)", cursor: "pointer" }}>
              <X size={16} />
            </button>
          </div>
          <div className="form-group">
            <label className="form-label">Folder Name</label>
            <input type="text" className="form-input" placeholder="e.g. Person of Interest" value={name} onChange={(e) => setName(e.target.value)} required />
          </div>
          <div className="form-group">
            <label className="form-label">Description / Scope</label>
            <textarea className="form-textarea" rows={2} placeholder="Briefly describe..." value={description} onChange={(e) => setDescription(e.target.value)} />
          </div>
          <div style={{ display: "flex", gap: "10px", justifyContent: "flex-end" }}>
            <button type="button" className="btn btn-secondary" onClick={() => setShowCreate(false)}>Cancel</button>
            <button type="submit" className="btn btn-primary">Create Folder</button>
          </div>
        </form>
      )}

      {/* Main Layout Grid */}
      <div style={{ display: "grid", gridTemplateColumns: selectedFolder ? "30% 70%" : "100%", gap: "20px" }}>
        
        {/* Folders List */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {loading ? (
            <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>Loading folders...</div>
          ) : folders.length === 0 ? (
            <div className="card" style={{ padding: "30px", textAlign: "center", color: "var(--text-secondary)", fontSize: "13px" }}>
              No folders found. Click "New Folder" or save search results to create one.
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: selectedFolder ? "1fr" : "repeat(auto-fill, minmax(280px, 1fr))", gap: "12px" }}>
              {folders.map(f => (
                <div 
                  key={f.id} 
                  className={`card ${selectedFolder && selectedFolder.id === f.id ? "active-folder" : ""}`}
                  style={{ 
                    cursor: "pointer", 
                    padding: "14px", 
                    display: "flex", 
                    flexDirection: "column", 
                    justifyContent: "space-between",
                    border: selectedFolder && selectedFolder.id === f.id ? "1px solid var(--color-accent)" : "1px solid var(--border-color)",
                    backgroundColor: "var(--bg-card)"
                  }}
                  onClick={() => handleSelectFolder(f)}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                      <Folder size={18} style={{ color: "var(--color-accent)" }} />
                      <div>
                        <div style={{ fontWeight: "700", fontSize: "13px", color: "var(--text-primary)" }}>{f.name}</div>
                        {f.description && <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>{f.description}</div>}
                      </div>
                    </div>
                    <button 
                      className="btn btn-secondary" 
                      style={{ padding: "4px", color: "var(--color-danger)", border: "none" }}
                      onClick={(e) => handleDeleteFolder(f.id, e)}
                      title="Delete folder"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: "12px", fontSize: "10px", color: "var(--text-muted)" }}>
                    <span>{f.item_count || 0} observations</span>
                    <span>{new Date(f.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Selected Folder Workspace */}
        {selectedFolder && (
          <div className="card" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "14px" }}>
              <div>
                <h3 style={{ fontSize: "16px", fontWeight: "700", color: "var(--color-accent)" }}>{selectedFolder.name} Workspace</h3>
                {selectedFolder.description && (
                  <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>{selectedFolder.description}</p>
                )}
              </div>
              <div style={{ display: "flex", gap: "10px" }}>
                <button className="btn btn-primary" style={{ backgroundColor: "#06B6D4", border: "none", color: "#000", fontSize: "12px", padding: "6px 12px" }} onClick={handleGenerateFolderSummary} disabled={loadingSummary}>
                  {loadingSummary ? "Compiling..." : "Generate Summary"}
                </button>
                <button className="btn btn-secondary" onClick={handleExportFolder}>
                  <Download size={14} style={{ marginRight: "4px" }} /> Export Case
                </button>
                <button className="btn btn-secondary" style={{ padding: "6px", color: "var(--text-secondary)" }} onClick={() => setSelectedFolder(null)}>
                  <X size={16} />
                </button>
              </div>
            </div>

            {/* Folder Layout: Observations vs Notes */}
            <div style={{ display: "grid", gridTemplateColumns: "60% 40%", gap: "20px" }}>
              
              {/* Evidence Items list */}
              <div>
                <h4 style={{ fontSize: "13px", fontWeight: "700", marginBottom: "12px", textTransform: "uppercase", color: "var(--text-muted)" }}>
                  Captured Observations ({folderItems.length})
                </h4>
                
                {folderItems.length === 0 ? (
                  <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)", border: "1px dashed var(--border-color)", borderRadius: "var(--radius-main)", fontSize: "12px" }}>
                    No observations added yet. Go to Search workspace to bookmark results to this case folder.
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "10px", maxHeight: "400px", overflowY: "auto", paddingRight: "6px" }}>
                    {folderItems.map(item => {
                      const data = item.parsedMeta || {};
                      return (
                        <div key={item.id} style={{ display: "flex", gap: "12px", padding: "10px", background: "var(--bg-input)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-main)", alignItems: "center" }}>
                          {data.crop_path ? (
                            <img 
                              src={getMediaUrl(data.crop_path)} 
                              alt="crop" 
                              style={{ width: "45px", height: "45px", borderRadius: "4px", objectFit: "cover", backgroundColor: "#000" }} 
                            />
                          ) : (
                            <div style={{ width: "45px", height: "45px", borderRadius: "4px", backgroundColor: "#000", display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <User size={20} style={{ color: "var(--text-muted)" }} />
                            </div>
                          )}

                          <div style={{ flexGrow: 1 }}>
                            <div style={{ fontWeight: "700", fontSize: "12px", color: "var(--text-primary)" }}>
                              {data.label ? `🏷️ ${data.label}` : `${data.entity_type || "OBSERVATION"}`.toUpperCase()} #{data.track_id?.toString().split('_').pop() || 1}
                            </div>
                            <div style={{ fontSize: "10px", color: "var(--text-secondary)", marginTop: "2px" }}>
                              Camera: {data.camera_name || data.camera_id} | Time: {typeof data.timestamp === "number" ? `${data.timestamp.toFixed(1)}s` : "N/A"}
                            </div>
                          </div>

                          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                            <button
                              type="button"
                              className="btn btn-primary"
                              style={{ fontSize: "11px", padding: "4px 8px", display: "flex", alignItems: "center", gap: "4px" }}
                              onClick={() => handleOpenEvidenceVideo(data)}
                              title="Open original video at this timestamp"
                            >
                              <Play size={11} /> Open Video
                            </button>
                            
                            <button 
                              className="btn btn-secondary" 
                              style={{ padding: "6px", color: "var(--color-danger)", border: "none" }}
                              onClick={() => handleRemoveItem(item.item_id)}
                              title="Remove from folder"
                            >
                              <Trash2 size={12} />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Notes Feed */}
              <div style={{ borderLeft: "1px solid var(--border-color)", paddingLeft: "20px" }}>
                <h4 style={{ fontSize: "13px", fontWeight: "700", marginBottom: "12px", textTransform: "uppercase", color: "var(--text-muted)" }}>
                  Investigation Notes ({notes.length})
                </h4>

                {/* Add Note Form */}
                <form onSubmit={handleAddNote} style={{ display: "flex", flexDirection: "column", gap: "8px", marginBottom: "16px" }}>
                  <textarea 
                    className="form-textarea" 
                    rows={2} 
                    placeholder="Enter observation note..." 
                    value={newNoteContent} 
                    onChange={(e) => setNewNoteContent(e.target.value)}
                    style={{ fontSize: "12px" }}
                  />
                  <button type="submit" className="btn btn-secondary" style={{ alignSelf: "flex-end", fontSize: "11px", padding: "4px 10px" }} disabled={!newNoteContent.trim()}>
                    Add Note
                  </button>
                </form>

                {/* Notes List */}
                <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "280px", overflowY: "auto" }}>
                  {notes.length === 0 ? (
                    <div style={{ color: "var(--text-muted)", fontSize: "11px", fontStyle: "italic" }}>No notes added yet.</div>
                  ) : (
                    notes.map(n => (
                      <div key={n.id} style={{ background: "var(--bg-input)", border: "1px solid var(--border-color)", borderRadius: "6px", padding: "8px", fontSize: "11px", display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div style={{ whiteSpace: "pre-wrap", flexGrow: 1 }}>{n.content}</div>
                        <button 
                          onClick={() => handleDeleteNote(n.id)}
                          style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", marginLeft: "6px" }}
                          title="Delete note"
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Folder AI Summary Box */}
            {folderSummary && (
              <div style={{ marginTop: "14px", padding: "14px", backgroundColor: "var(--bg-input)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-main)" }}>
                <span style={{ fontSize: "12px", fontWeight: "700", color: "var(--color-accent)", display: "block", marginBottom: "6px" }}>
                  Case Evidence Summary
                </span>
                <div style={{ fontSize: "12px", lineHeight: "1.6", color: "var(--text-primary)", whiteSpace: "pre-line" }}>
                  {folderSummary}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Forensic Video Viewer Modal */}
      <ForensicVideoViewer 
        isOpen={showVideoViewer}
        onClose={() => setShowVideoViewer(false)}
        observation={viewerObservation}
        cameras={cameras}
        onSaveToEvidence={() => {}}
        onGetDetails={() => {}}
        onFindSimilar={() => {}}
      />
    </div>
  );
};

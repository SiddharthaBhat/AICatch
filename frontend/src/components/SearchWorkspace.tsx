import React, { useState, useEffect } from "react";
import {
  Search, Image as ImageIcon, Play,
  FolderPlus, AlertTriangle,
  X, Sliders, FileText, Loader2, Layers
} from "lucide-react";
import { apiService, getMediaUrl } from "../services/api";
import { SubjectSummaryModal } from "./SubjectSummaryModal";
import { ForensicVideoViewer } from "./ForensicVideoViewer";
import type { ObservationTarget } from "./ForensicVideoViewer";
import type { Camera } from "../types";

interface DetectionCropImageProps {
  cropPath?: string;
  alt?: string;
  style?: React.CSSProperties;
}

const DetectionCropImage: React.FC<DetectionCropImageProps> = ({
  cropPath,
  alt = "Detection crop",
  style = { width: "100%", height: "100%", objectFit: "contain" }
}) => {
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    setLoadError(false);
  }, [cropPath]);

  // If crop_path is missing, empty, or failed to load, display neutral missing-evidence state (never a dummy avatar)
  if (!cropPath || loadError) {
    return (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text-muted)",
          fontSize: "11px",
          textAlign: "center",
          padding: "8px",
          backgroundColor: "#0d1520"
        }}
      >
        <span>No detection crop available</span>
      </div>
    );
  }

  return (
    <img
      src={getMediaUrl(cropPath)}
      alt={alt}
      style={style}
      onError={() => setLoadError(true)}
    />
  );
};

interface SearchWorkspaceProps {
  jobId: string | null;
  cameras?: Camera[];
  caseName?: string;
  onSelectTrack?: (trackUuid: string, camera_id: string, timestamp: number) => void;
  onSelectTargetForJourney?: (obs: any) => void;
}

export const SearchWorkspace: React.FC<SearchWorkspaceProps> = ({
  jobId,
  cameras = [],
  caseName = "Investigation",
  onSelectTargetForJourney
}) => {
  const [tab, setTab] = useState<"text" | "image" | "plate">("text");

  // Query Inputs
  const [queryText, setQueryText] = useState("");
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryFilePreview, setQueryFilePreview] = useState<string | null>(null);
  const [queryCategory, setQueryCategory] = useState<"auto" | "person" | "object" | "scene" | "region">("auto");

  // Advanced Filters
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [filterCamera, setFilterCamera] = useState("");
  const [filterObjectType, setFilterObjectType] = useState("");
  const [filterSimilarity, setFilterSimilarity] = useState(0.28);
  const [filterConfidence] = useState(0.50);

  // State
  const [results, setResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Detail panel state
  const [selectedResult, setSelectedResult] = useState<any | null>(null);
  const [selectedResultDetections, setSelectedResultDetections] = useState<any[]>([]);

  // Forensic Video Viewer Modal
  const [viewerObservation, setViewerObservation] = useState<ObservationTarget | null>(null);
  const [showVideoViewer, setShowVideoViewer] = useState(false);

  // Subject Summary Modal
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [modalSubjectLabel, setModalSubjectLabel] = useState("Person");
  const [modalObservations, setModalObservations] = useState<any[]>([]);

  // Modals / Dialog states
  const [showFolderModal, setShowFolderModal] = useState(false);
  const [folders, setFolders] = useState<any[]>([]);
  const [selectedFolderId, setSelectedFolderId] = useState("");
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null);

  const getCameraDisplayName = (camId?: string) => {
    if (!camId) return "Camera";
    const found = cameras.find(c => c.id === camId);
    return found ? found.name : camId;
  };

  const loadFolders = async () => {
    try {
      const data = await apiService.listFolders(jobId || undefined);
      setFolders(data);
      if (data.length > 0) {
        setSelectedFolderId(data[0].id);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadFolders();
    setResults([]);
    setSelectedResult(null);
    setQueryText("");
    setQueryFile(null);
    setQueryFilePreview(null);
    setErrorMsg(null);
    setSearched(false);
  }, [jobId]);

  useEffect(() => {
    if (selectedResult) {
      const fetchDetections = async () => {
        try {
          const trackUuid = selectedResult.track_uuid || selectedResult.track_id;
          if (trackUuid) {
            const dets = await apiService.getTrackDetections(trackUuid);
            setSelectedResultDetections(dets);
          }
        } catch (e) {
          console.error("Failed to load detections for track:", e);
        }
      };
      fetchDetections();
    } else {
      setSelectedResultDetections([]);
    }
  }, [selectedResult]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setQueryFile(file);
      setQueryFilePreview(URL.createObjectURL(file));
    }
  };

  const handleSearchSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setSearching(true);
    setErrorMsg(null);
    setSearched(true);
    setResults([]);
    setSelectedResult(null);

    try {
      let searchParams: any = { qtype: tab, jobId: jobId || undefined, topK: 40 };

      if (tab === "text") {
        if (!queryText.trim()) throw new Error("Please enter a description to search.");
        searchParams.queryText = queryText.trim();
      } else if (tab === "image") {
        if (!queryFile) throw new Error("Please select an image file.");
        searchParams.queryFile = queryFile;
        if (queryCategory !== "auto") {
          searchParams.qcategory = queryCategory;
        }
      } else if (tab === "plate") {
        if (!queryText.trim()) throw new Error("Please enter a license plate string.");
        searchParams.queryText = queryText.trim();
      }

      if (filterCamera) searchParams.cameraId = filterCamera;
      if (filterObjectType) searchParams.objectType = filterObjectType;

      const matches = await apiService.executeForensicSearch(searchParams);

      // Filter strictly by accepted threshold
      const filtered = matches.filter(match => {
        if (match.similarity !== undefined && match.similarity < filterSimilarity) return false;
        if (match.confidence !== undefined && match.confidence < filterConfidence) return false;
        if (filterObjectType && match.entity_type !== filterObjectType) return false;
        return true;
      });

      // Group visual search matches by track_id to avoid identical frame duplicates
      const groupedMatches: any[] = [];
      const seenTracks = new Set<string>();
      for (const match of filtered) {
        const tId = match.track_uuid || match.track_id;
        if (!tId) {
          groupedMatches.push(match);
          continue;
        }
        if (!seenTracks.has(tId)) {
          seenTracks.add(tId);
          groupedMatches.push(match);
        }
      }

      setResults(groupedMatches);
    } catch (err: any) {
      setErrorMsg(err.message || "Search failed");
    } finally {
      setSearching(false);
    }
  };

  const handleOpenVideoViewer = (result: any) => {
    setViewerObservation({
      jobId: jobId || result.job_id || "",
      cameraId: result.camera_id,
      cameraName: getCameraDisplayName(result.camera_id),
      timestamp: result.timestamp || 0,
      trackId: result.track_id,
      trackUuid: result.track_uuid || (typeof result.track_id === "string" ? result.track_id : undefined),
      cropPath: result.crop_path,
      entityType: result.entity_type,
      confidence: result.confidence,
      similarity: result.similarity,
      label: result.label,
      bbox: result.bbox
    });
    setShowVideoViewer(true);
  };

  const handleOpenGetDetails = (result: any) => {
    const label = result.label || (result.entity_type === "person" ? `Subject #${result.track_id || 1}` : "Observation");
    setModalSubjectLabel(label);

    const obsList = (selectedResultDetections.length > 0 ? selectedResultDetections : [result]).map(d => ({
      camera_name: getCameraDisplayName(d.camera_id || result.camera_id),
      camera_id: d.camera_id || result.camera_id,
      timestamp: d.timestamp || result.timestamp || 0,
      confidence: d.confidence || result.confidence || 0.85,
      crop_path: d.crop_path || result.crop_path,
      entity_type: d.entity_type || result.entity_type || "person",
      track_id: result.track_id
    }));

    setModalObservations(obsList);
    setShowSummaryModal(true);
  };

  const handleSaveToEvidence = async (result: any) => {
    if (!selectedFolderId && folders.length === 0) {
      try {
        const newFold = await apiService.createFolder("Person of Interest", "Saved search observations", jobId!);
        await apiService.addFolderItem(newFold.id, "crop", result.crop_path, JSON.stringify(result));
        setSaveSuccessMsg("Saved to Evidence folder: Person of Interest");
        setTimeout(() => setSaveSuccessMsg(null), 3000);
        loadFolders();
        return;
      } catch (err: any) {
        alert(err.message || "Failed to save evidence");
        return;
      }
    }

    setShowFolderModal(true);
  };

  const confirmAddToFolder = async () => {
    if (!selectedFolderId || !selectedResult) return;
    try {
      await apiService.addFolderItem(selectedFolderId, "crop", selectedResult.crop_path, JSON.stringify(selectedResult));
      setShowFolderModal(false);
      setSaveSuccessMsg("Saved to Evidence.");
      setTimeout(() => setSaveSuccessMsg(null), 3000);
    } catch (err: any) {
      alert(err.message || "Failed to add item to folder");
    }
  };

  const handleFindSimilar = async (result: any) => {
    if (!result.crop_path) return;
    try {
      const cropUrl = getMediaUrl(result.crop_path);
      const resBlob = await fetch(cropUrl);
      const blob = await resBlob.blob();
      const file = new File([blob], "query_crop.jpg", { type: "image/jpeg" });

      setTab("image");
      setFilterSimilarity(0.55);
      setQueryFile(file);
      setQueryFilePreview(cropUrl);

      const searchParams = {
        qtype: "image" as const,
        queryFile: file,
        jobId: jobId || undefined,
        topK: 40
      };

      setSearching(true);
      setSearched(true);
      const matches = await apiService.executeForensicSearch(searchParams);
      const accepted = matches.filter(m => (m.similarity || 0) >= 0.55);

      // Group visual search matches by track to avoid identical frame duplicates
      const groupedSimilar: any[] = [];
      const seenTracks = new Set<string>();
      for (const match of accepted) {
        const rawId = match.track_uuid || match.track_id;
        if (!rawId) {
          groupedSimilar.push(match);
          continue;
        }
        const isGlobalUuid = typeof rawId === "string" && (rawId.includes("-") || rawId.startsWith("tr_"));
        const dedupeKey = isGlobalUuid ? rawId : `${match.camera_id || "cam"}_${rawId}`;
        if (!seenTracks.has(dedupeKey)) {
          seenTracks.add(dedupeKey);
          groupedSimilar.push(match);
        }
      }
      setResults(groupedSimilar);
    } catch (err: any) {
      setErrorMsg(err.message || "Find similar search failed");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div style={{ display: "flex", gap: "24px", height: "100%", position: "relative" }}>
      {/* Left Search Sidebar & Filters */}
      <div
        style={{
          width: "320px",
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          gap: "16px",
          overflowY: "auto"
        }}
      >
        <div className="card" style={{ padding: "18px" }}>
          {/* Mode Switcher */}
          <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-input)", padding: "4px", borderRadius: "8px", border: "1px solid var(--border-color)", marginBottom: "16px" }}>
            <button
              type="button"
              className="btn"
              onClick={() => { setTab("text"); setFilterSimilarity(0.28); setErrorMsg(null); }}
              style={{
                flex: 1,
                padding: "6px 8px",
                fontSize: "11px",
                fontWeight: "600",
                backgroundColor: tab === "text" ? "var(--bg-card)" : "transparent",
                color: tab === "text" ? "var(--color-accent)" : "var(--text-secondary)",
                border: tab === "text" ? "1px solid var(--border-color)" : "none",
                borderRadius: "6px"
              }}
            >
              Text
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => { setTab("image"); setFilterSimilarity(0.55); setErrorMsg(null); }}
              style={{
                flex: 1,
                padding: "6px 8px",
                fontSize: "11px",
                fontWeight: "600",
                backgroundColor: tab === "image" ? "var(--bg-card)" : "transparent",
                color: tab === "image" ? "var(--color-accent)" : "var(--text-secondary)",
                border: tab === "image" ? "1px solid var(--border-color)" : "none",
                borderRadius: "6px"
              }}
            >
              Image
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => { setTab("plate"); setErrorMsg(null); }}
              style={{
                flex: 1,
                padding: "6px 8px",
                fontSize: "11px",
                fontWeight: "600",
                backgroundColor: tab === "plate" ? "var(--bg-card)" : "transparent",
                color: tab === "plate" ? "var(--color-accent)" : "var(--text-secondary)",
                border: tab === "plate" ? "1px solid var(--border-color)" : "none",
                borderRadius: "6px"
              }}
            >
              Plate
            </button>
          </div>

          {/* Query Inputs */}
          <form onSubmit={handleSearchSubmit} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {tab === "text" && (
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label" style={{ fontSize: "11px" }}>Describe what you're looking for</label>
                <textarea
                  className="form-textarea"
                  rows={3}
                  placeholder="e.g. person in red shirt, white sedan, person with backpack"
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  style={{ fontSize: "12px" }}
                />
              </div>
            )}

            {tab === "image" && (
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label" style={{ fontSize: "11px" }}>Target Image</label>
                <div
                  style={{
                    border: "1px dashed var(--border-color)",
                    borderRadius: "8px",
                    padding: "16px 10px",
                    textAlign: "center",
                    backgroundColor: "var(--bg-input)",
                    cursor: "pointer",
                    position: "relative"
                  }}
                >
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    style={{ opacity: 0, position: "absolute", top: 0, left: 0, width: "100%", height: "100%", cursor: "pointer" }}
                  />
                  {queryFilePreview ? (
                    <img src={queryFilePreview} alt="preview" style={{ maxHeight: "90px", maxWidth: "100%", objectFit: "contain", borderRadius: "4px" }} />
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px", color: "var(--text-secondary)" }}>
                      <ImageIcon size={20} />
                      <span style={{ fontSize: "11px" }}>Drop image or browse</span>
                    </div>
                  )}
                </div>

                <div style={{ marginTop: "10px" }}>
                  <label className="form-label" style={{ fontSize: "11px", marginBottom: "4px", display: "block" }}>Target Intent</label>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "4px" }}>
                    {(["auto", "person", "object", "scene", "region"] as const).map((cat) => (
                      <button
                        key={cat}
                        type="button"
                        onClick={() => setQueryCategory(cat)}
                        style={{
                          fontSize: "10px",
                          padding: "5px 2px",
                          borderRadius: "4px",
                          border: queryCategory === cat ? "1px solid var(--color-accent)" : "1px solid var(--border-color)",
                          backgroundColor: queryCategory === cat ? "rgba(99, 102, 241, 0.2)" : "var(--bg-card)",
                          color: queryCategory === cat ? "var(--color-accent)" : "var(--text-secondary)",
                          fontWeight: queryCategory === cat ? "700" : "500",
                          textTransform: "capitalize",
                          cursor: "pointer"
                        }}
                      >
                        {cat}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {tab === "plate" && (
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label" style={{ fontSize: "11px" }}>License Plate String</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. XYZ123"
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  style={{ fontSize: "12px" }}
                />
              </div>
            )}

            {/* Collapsible Advanced Filters */}
            <div style={{ borderTop: "1px solid var(--border-color)", paddingTop: "10px" }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                style={{ width: "100%", fontSize: "11px", padding: "6px", display: "flex", alignItems: "center", justifyContent: "space-between" }}
              >
                <span><Sliders size={12} style={{ display: "inline", marginRight: "6px" }} /> Advanced Filters</span>
                <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>{showAdvancedFilters ? "▲" : "▼"}</span>
              </button>

              {showAdvancedFilters && (
                <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "10px" }}>
                  {cameras.length > 0 && (
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label" style={{ fontSize: "10px" }}>Camera Filter</label>
                      <select
                        className="form-select"
                        value={filterCamera}
                        onChange={(e) => setFilterCamera(e.target.value)}
                        style={{ fontSize: "11px", padding: "6px 8px" }}
                      >
                        <option value="">All Cameras</option>
                        {cameras.map(c => (
                          <option key={c.id} value={c.id}>{c.name}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label" style={{ fontSize: "10px" }}>Object Type</label>
                    <select
                      className="form-select"
                      value={filterObjectType}
                      onChange={(e) => setFilterObjectType(e.target.value)}
                      style={{ fontSize: "11px", padding: "6px 8px" }}
                    >
                      <option value="">All Objects</option>
                      <option value="person">Persons Only</option>
                      <option value="vehicle">Vehicles Only</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {errorMsg && (
              <div style={{ fontSize: "11px", color: "var(--color-danger)", padding: "4px 0" }}>
                {errorMsg}
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              disabled={searching}
              style={{ width: "100%", padding: "10px", fontSize: "13px", fontWeight: "600", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
            >
              {searching ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />} Search
            </button>
          </form>
        </div>
      </div>

      {/* Center Search Results Grid */}
      <div style={{ flexGrow: 1, overflowY: "auto", paddingRight: selectedResult ? "380px" : "0" }}>
        {saveSuccessMsg && (
          <div style={{ padding: "8px 14px", borderRadius: "6px", backgroundColor: "rgba(16, 185, 129, 0.1)", border: "1px solid var(--color-success)", color: "var(--color-success)", fontSize: "12px", marginBottom: "14px" }}>
            {saveSuccessMsg}
          </div>
        )}

        {searching ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "40vh", gap: "10px" }}>
            <Loader2 size={28} className="animate-spin" style={{ color: "var(--color-accent)" }} />
            <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>Searching observations...</span>
          </div>
        ) : searched && results.length === 0 ? (
          <div className="card" style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
            <AlertTriangle size={32} style={{ margin: "0 auto 10px", color: "var(--text-muted)" }} />
            <h3 style={{ fontSize: "15px", color: "var(--text-primary)", marginBottom: "4px" }}>
              {errorMsg ? "Image search needs attention" : "No reliable matches found"}
            </h3>
            <p style={{ fontSize: "12px", color: "var(--text-muted)" }}>
              {errorMsg || "Try adjusting your search query description or select a different image."}
            </p>
          </div>
        ) : results.length > 0 ? (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "12px" }}>
              <span style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)" }}>
                Ranked Match Results ({results.length})
              </span>
              <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                Showing {results.length} validated observation(s)
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "16px" }}>
              {results.map((res, idx) => {
                const isSelected = selectedResult && (selectedResult.track_id === res.track_id || selectedResult.crop_path === res.crop_path);
                const camName = getCameraDisplayName(res.camera_id);

                return (
                  <div
                    key={idx}
                    className="card"
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      padding: "12px",
                      border: isSelected ? "1px solid var(--color-accent)" : "1px solid var(--border-color)",
                      backgroundColor: "var(--bg-card)",
                      borderRadius: "var(--radius-main)",
                      cursor: "pointer",
                      gap: "8px",
                      transition: "all 0.2s ease"
                    }}
                    onClick={() => { setSelectedResult(res); handleOpenVideoViewer(res); }}
                  >
                    {/* Real Person / Object Crop Thumbnail */}
                    <div
                      style={{ width: "100%", height: "140px", backgroundColor: "#000", borderRadius: "6px", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}
                      onClick={(e) => { e.stopPropagation(); handleOpenVideoViewer(res); }}
                      title="Click to open original video at this timestamp"
                    >
                      <DetectionCropImage cropPath={res.crop_path} alt="Detection crop" />

                      <div style={{ position: "absolute", top: "6px", left: "6px", display: "flex", gap: "4px" }}>
                        <span style={{ fontSize: "9px", fontWeight: "700", padding: "2px 6px", borderRadius: "4px", backgroundColor: "rgba(0,0,0,0.7)", color: "var(--color-accent)", textTransform: "uppercase" }}>
                          {res.entity_type || "OBJECT"}
                        </span>
                        {(res.source_type || res.match_type) && (
                          <span style={{
                            fontSize: "9px",
                            fontWeight: "700",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            backgroundColor: (res.source_type === "scene" || res.match_type === "scene") ? "rgba(59, 130, 246, 0.85)" : ((res.source_type === "region" || res.match_type === "region") ? "rgba(245, 158, 11, 0.85)" : "rgba(16, 185, 129, 0.85)"),
                            color: "#ffffff",
                            textTransform: "uppercase"
                          }}>
                            {(res.source_type === "scene" || res.match_type === "scene") ? "Scene Match" : ((res.source_type === "region" || res.match_type === "region") ? "Region Match" : "Object Match")}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Metadata */}
                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {camName}
                        </span>
                        <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                          {typeof res.similarity === "number" && (
                            <span style={{ fontSize: "11px", color: "var(--color-success)", fontWeight: "700" }}>
                              Sim: {res.similarity.toFixed(2)}
                            </span>
                          )}
                          {typeof res.reid_similarity === "number" && (
                            <span style={{ fontSize: "10px", color: "var(--color-accent)", fontWeight: "600" }}>
                              Re-ID: {res.reid_similarity.toFixed(2)}
                            </span>
                          )}
                        </div>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                        <span>Time: {typeof res.timestamp === "number" ? `${res.timestamp.toFixed(1)}s` : "N/A"}</span>
                        {(res.label || res.track_id || res.track_uuid) && (
                          <span style={{ color: "var(--text-muted)", fontSize: "10px", fontFamily: "monospace" }}>
                            {res.label || (typeof res.track_id === "number" || (typeof res.track_id === "string" && !res.track_id.includes("-")) ? `Track #${res.track_id}` : `Track ${String(res.track_uuid || res.track_id).slice(0, 8)}`)}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Card Actions */}
                    <div style={{ display: "flex", gap: "6px", marginTop: "4px" }}>
                      <button
                        type="button"
                        className="btn btn-primary"
                        style={{ flex: 1, fontSize: "11px", padding: "6px 8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "4px" }}
                        onClick={(e) => { e.stopPropagation(); handleOpenVideoViewer(res); }}
                      >
                        <Play size={11} /> Jump to Video
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ fontSize: "11px", padding: "6px 8px" }}
                        onClick={(e) => { e.stopPropagation(); handleOpenGetDetails(res); }}
                      >
                        Details
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary"
                        style={{ fontSize: "11px", padding: "6px 8px" }}
                        onClick={(e) => { e.stopPropagation(); handleSaveToEvidence(res); }}
                      >
                        Save
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "40vh", color: "var(--text-muted)", gap: "8px" }}>
            <Search size={36} style={{ strokeWidth: 1.5 }} />
            <span style={{ fontSize: "14px" }}>Enter a query on the left to search observations in this case.</span>
          </div>
        )}
      </div>

      {/* Right Side Observation Detail Panel (Drawer) */}
      {selectedResult && (
        <div
          className="card"
          style={{
            position: "absolute",
            top: 0,
            right: 0,
            width: "360px",
            height: "100%",
            backgroundColor: "var(--bg-card)",
            border: "1px solid var(--border-color)",
            borderRadius: "var(--radius-main)",
            padding: "20px",
            display: "flex",
            flexDirection: "column",
            gap: "14px",
            overflowY: "auto",
            zIndex: 10
          }}
        >
          {/* Drawer Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
            <span style={{ fontSize: "12px", fontWeight: "700", color: "var(--color-accent)", textTransform: "uppercase" }}>
              Observation Detail
            </span>
            <button
              type="button"
              onClick={() => setSelectedResult(null)}
              style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
            >
              <X size={18} />
            </button>
          </div>

          {/* Large Observation Image */}
          <div
            style={{ width: "100%", height: "200px", backgroundColor: "#000", borderRadius: "8px", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }}
            onClick={() => handleOpenVideoViewer(selectedResult)}
            title="Click to open video viewer"
          >
            <DetectionCropImage cropPath={selectedResult.crop_path} alt="Observation detail crop" />
          </div>

          {/* Meta Info */}
          <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Camera:</span>
              <strong style={{ color: "var(--text-primary)" }}>{getCameraDisplayName(selectedResult.camera_id)}</strong>
            </div>
            {(selectedResult.source_type || selectedResult.match_type) && (
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Source Type:</span>
                <strong style={{
                  color: (selectedResult.source_type === "scene" || selectedResult.match_type === "scene") ? "#3b82f6" : ((selectedResult.source_type === "region" || selectedResult.match_type === "region") ? "#f59e0b" : "var(--color-success)"),
                  textTransform: "capitalize"
                }}>
                  {(selectedResult.source_type === "scene" || selectedResult.match_type === "scene") ? "Scene Keyframe" : ((selectedResult.source_type === "region" || selectedResult.match_type === "region") ? "Spatial Region Tile" : "Object Detection Crop")}
                </strong>
              </div>
            )}
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Object Type:</span>
              <strong style={{ color: "var(--text-primary)", textTransform: "capitalize" }}>{selectedResult.entity_type || "person"}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Timestamp:</span>
              <strong style={{ color: "var(--text-primary)" }}>{typeof selectedResult.timestamp === "number" ? `${selectedResult.timestamp.toFixed(1)}s` : "N/A"}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Visual Similarity:</span>
              <strong style={{ color: "var(--color-accent)" }}>{typeof selectedResult.similarity === "number" ? selectedResult.similarity.toFixed(4) : "N/A"}</strong>
            </div>
            {typeof selectedResult.reid_similarity === "number" && (
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Re-ID Similarity:</span>
                <strong style={{ color: "var(--color-success)" }}>{selectedResult.reid_similarity.toFixed(4)}</strong>
              </div>
            )}
            {typeof selectedResult.face_similarity === "number" && (
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Face Similarity:</span>
                <strong style={{ color: "#38bdf8" }}>{selectedResult.face_similarity.toFixed(4)}</strong>
              </div>
            )}
            {selectedResult.query_intent && (
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Query Intent:</span>
                <strong style={{ color: "var(--text-primary)", textTransform: "capitalize" }}>
                  {selectedResult.query_intent} {selectedResult.intent_source === "explicit_intent" ? "(Explicit)" : "(Auto)"}
                </strong>
              </div>
            )}
            {selectedResult.diagnostic_label && (
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Diagnostic:</span>
                <span style={{ color: "var(--text-secondary)", fontSize: "11px" }}>{selectedResult.diagnostic_label}</span>
              </div>
            )}
          </div>

          {/* Primary Action Buttons */}
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "10px" }}>
            {/* Primary Highlighted Open Video Action */}
            <button
              type="button"
              className="btn btn-primary"
              style={{ fontSize: "13px", padding: "10px", display: "flex", alignItems: "center", justifyContent: "center", gap: "8px", fontWeight: "600" }}
              onClick={() => handleOpenVideoViewer(selectedResult)}
            >
              <Play size={15} /> Open Original Video
            </button>

            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: "12px", padding: "8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
              onClick={() => handleFindSimilar(selectedResult)}
            >
              <Search size={13} /> Find Similar Observations
            </button>

            {onSelectTargetForJourney && (
              <button
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: "12px", padding: "8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
                onClick={() => onSelectTargetForJourney(selectedResult)}
              >
                <Layers size={13} /> Trace Cross-Camera Journey
              </button>
            )}

            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: "12px", padding: "8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
              onClick={() => handleOpenGetDetails(selectedResult)}
            >
              <FileText size={13} /> Get Details (Subject Summary)
            </button>

            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: "12px", padding: "8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
              onClick={() => handleSaveToEvidence(selectedResult)}
            >
              <FolderPlus size={13} /> Save to Evidence
            </button>
          </div>
        </div>
      )}

      {/* Forensic Video Viewer Modal */}
      <ForensicVideoViewer
        isOpen={showVideoViewer}
        onClose={() => setShowVideoViewer(false)}
        observation={viewerObservation}
        cameras={cameras}
        caseName={caseName}
        onSaveToEvidence={(obs) => handleSaveToEvidence(obs)}
        onGetDetails={(obs) => handleOpenGetDetails(obs)}
        onFindSimilar={(obs) => handleFindSimilar(obs)}
      />

      {/* Subject Summary Modal */}
      <SubjectSummaryModal
        isOpen={showSummaryModal}
        onClose={() => setShowSummaryModal(false)}
        subjectLabel={modalSubjectLabel}
        observations={modalObservations}
        caseName={caseName}
        onOpenVideo={(obs) => {
          setShowSummaryModal(false);
          handleOpenVideoViewer(obs);
        }}
      />

      {/* Save to Folder Modal */}
      {showFolderModal && (
        <div style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.7)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 99999 }}>
          <div className="card" style={{ maxWidth: "400px", width: "100%", padding: "24px" }}>
            <h3 style={{ fontSize: "15px", color: "var(--text-primary)", marginBottom: "12px" }}>Save to Evidence Folder</h3>

            <div className="form-group">
              <label className="form-label">Select Folder</label>
              <select className="form-select" value={selectedFolderId} onChange={(e) => setSelectedFolderId(e.target.value)}>
                {folders.map(f => (
                  <option key={f.id} value={f.id}>{f.name}</option>
                ))}
              </select>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", marginTop: "16px" }}>
              <button type="button" className="btn btn-secondary" onClick={() => setShowFolderModal(false)}>Cancel</button>
              <button type="button" className="btn btn-primary" onClick={confirmAddToFolder}>Save Observation</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

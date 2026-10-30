import React, { useState, useEffect, useRef } from "react";
import { 
  ArrowLeft, Search, FolderOpen, FileText, 
  RotateCcw, RotateCw, Play, Pause, Maximize, Volume2, VolumeX,
  Eye, CheckCircle2, CheckSquare, Square,
  Film, X, FolderPlus, RefreshCw, AlertCircle, Clock
} from "lucide-react";
import type { Job, Camera } from "../types";
import { apiService, getMediaUrl } from "../services/api";
import { Evidence } from "./Evidence";
import { SubjectSummaryModal } from "./SubjectSummaryModal";
import { CameraTimestampModal } from "./CameraTimestampModal";

interface SingleVideoWorkspaceProps {
  jobId: string;
  onBack: () => void;
}

export const SingleVideoWorkspace: React.FC<SingleVideoWorkspaceProps> = ({ jobId, onBack }) => {
  const [activeTab, setActiveTab] = useState<"workspace" | "evidence" | "summary">("workspace");
  const [job, setJob] = useState<Job | null>(null);
  const [camera, setCamera] = useState<Camera | null>(null);
  const [showTimestampModal, setShowTimestampModal] = useState<boolean>(false);
  const [loading, setLoading] = useState(true);

  // Video Player state
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);

  // Search state
  const [searchTab, setSearchTab] = useState<"text" | "image" | "plate">("text");
  const [queryText, setQueryText] = useState("");
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryFilePreview, setQueryFilePreview] = useState<string | null>(null);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.55);
  const [groupByTrack, setGroupByTrack] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Selection & Clip Merge state
  const [selectedResultIds, setSelectedResultIds] = useState<Set<string>>(new Set());
  const [mergingClips, setMergingClips] = useState(false);
  const [mergedClipSuccess, setMergedClipSuccess] = useState<string | null>(null);

  // Drawer & Summary Modal state
  const [drawerResult, setDrawerResult] = useState<any | null>(null);
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [modalObservations, setModalObservations] = useState<any[]>([]);

  // Summary Report state
  const [summaryReport, setSummaryReport] = useState<string | null>(null);
  const [generatingSummary, setGeneratingSummary] = useState(false);

  const loadData = async () => {
    try {
      const [jobData, camsData] = await Promise.all([
        apiService.getJobStatus(jobId),
        apiService.getJobCameras(jobId)
      ]);
      setJob(jobData);
      if (camsData.length > 0) setCamera(camsData[0]);
    } catch (err) {
      console.error("Failed to load single video workspace:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [jobId]);

  // Video Controls
  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const jumpSeconds = (delta: number) => {
    if (!videoRef.current) return;
    const newTime = Math.max(0, Math.min(duration || 9999, videoRef.current.currentTime + delta));
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const seekToTimestamp = (timestamp: number, _trackUuid?: string) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
      setCurrentTime(timestamp);
    }
  };

  const formatTimestamp = (secs: number) => {
    if (isNaN(secs) || secs < 0) return "00:00:00";
    const hrs = Math.floor(secs / 3600);
    const mins = Math.floor((secs % 3600) / 60);
    const s = Math.floor(secs % 60);
    if (hrs > 0) {
      return `${hrs.toString().padStart(2, "0")}:${mins.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
    }
    return `${mins.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // Search Execution
  const handleExecuteSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setSearching(true);
    setSearchError(null);
    setSearched(true);
    setSelectedResultIds(new Set());

    try {
      let qtype: "text" | "image" | "plate" = searchTab;
      let text = queryText.trim();
      let file: File | undefined = queryFile || undefined;

      const data = await apiService.executeForensicSearch({
        qtype,
        queryText: text || undefined,
        queryFile: file,
        jobId,
        topK: 60
      });

      // Filter by confidence / threshold
      const filtered = data.filter((d: any) => (d.similarity || d.confidence || 0) >= similarityThreshold);
      setSearchResults(filtered);
    } catch (err: any) {
      console.error("Search failed:", err);
      setSearchError(err.message || "Search failed");
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  // Selection toggle
  const toggleResultSelection = (id: string) => {
    const next = new Set(selectedResultIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelectedResultIds(next);
  };

  const toggleSelectAll = () => {
    if (selectedResultIds.size === searchResults.length) {
      setSelectedResultIds(new Set());
    } else {
      setSelectedResultIds(new Set(searchResults.map(r => r.id || `${r.track_id}_${r.timestamp}`)));
    }
  };

  // Merge Selected Clips
  const handleMergeSelectedClips = async () => {
    const selectedObs = searchResults.filter(r => {
      const id = r.id || `${r.track_id}_${r.timestamp}`;
      return selectedResultIds.has(id);
    });

    if (selectedObs.length === 0) {
      alert("Please select at least one observation result to merge.");
      return;
    }

    setMergingClips(true);
    setMergedClipSuccess(null);
    try {
      const res = await apiService.mergeSelectedClips(jobId, selectedObs, 3.0);
      setMergedClipSuccess(`Successfully generated merged video: ${res.filename} (${res.segments_count} clips). Saved to Evidence.`);
      setTimeout(() => setMergedClipSuccess(null), 6000);
    } catch (err: any) {
      alert(err.message || "Failed to merge selected clips");
    } finally {
      setMergingClips(false);
    }
  };

  // Generate Summary Report
  const handleGenerateSummary = async () => {
    setGeneratingSummary(true);
    try {
      const summaryData = await apiService.getCaseSummary(jobId);
      setSummaryReport(summaryData.summary);
    } catch (err: any) {
      console.error("Failed to generate case summary:", err);
      setSummaryReport("Could not generate summary from backend.");
    } finally {
      setGeneratingSummary(false);
    }
  };

  if (loading && !job) {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "60vh" }}>
        <span style={{ fontSize: "14px", color: "var(--text-secondary)" }}>Loading single video workspace...</span>
      </div>
    );
  }

  if (!job || !camera) {
    return (
      <div className="card" style={{ padding: "40px", textAlign: "center", maxWidth: "600px", margin: "40px auto" }}>
        <AlertCircle size={36} style={{ color: "var(--color-danger)", margin: "0 auto 12px" }} />
        <h2 style={{ fontSize: "16px", color: "var(--text-primary)", marginBottom: "8px" }}>Investigation Footage Not Found</h2>
        <button type="button" className="btn btn-secondary" onClick={onBack}>
          <ArrowLeft size={14} /> Back to Dashboard
        </button>
      </div>
    );
  }

  // Displayed search results (optionally grouped by track)
  const displayResults = groupByTrack ? (() => {
    const trackMap = new Map<string, any>();
    for (const r of searchResults) {
      const key = r.track_uuid || r.track_id || "single";
      if (!trackMap.has(key)) {
        trackMap.set(key, { ...r, observationCount: 1, occurrences: [r] });
      } else {
        const entry = trackMap.get(key);
        entry.observationCount += 1;
        entry.occurrences.push(r);
      }
    }
    return Array.from(trackMap.values());
  })() : searchResults;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px", maxWidth: "1500px", margin: "0 auto", paddingBottom: "60px" }} className="animate-fade-in">
      {/* Top Header Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px", borderBottom: "1px solid var(--border-color)", paddingBottom: "14px" }}>
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
              <span style={{ fontSize: "9px", fontWeight: "700", padding: "2px 6px", borderRadius: "4px", backgroundColor: "rgba(6, 182, 212, 0.1)", color: "var(--color-accent)" }}>
                SINGLE VIDEO
              </span>
              <span className={`badge badge-${job.status}`} style={{ fontSize: "9px" }}>
                {job.status}
              </span>
              <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                · Camera: <strong>{camera.name}</strong> ({camera.source_path.split("/").pop()})
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowTimestampModal(true)}
                  style={{ fontSize: "10px", padding: "2px 6px", display: "inline-flex", alignItems: "center", gap: "3px", marginLeft: "6px" }}
                  title="Configure CCTV timestamp detection and manual correction"
                >
                  <Clock size={10} />
                  {camera.timestamp_config_mode && camera.timestamp_config_mode !== 'auto' ? 'Calibrated' : 'Timestamps'}
                </button>
              </span>
            </div>
          </div>
        </div>

        {/* Primary Tabs */}
        <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-input)", padding: "4px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
          <button
            type="button"
            className="btn"
            onClick={() => setActiveTab("workspace")}
            style={{
              padding: "6px 14px",
              fontSize: "12px",
              fontWeight: "600",
              backgroundColor: activeTab === "workspace" ? "var(--bg-card)" : "transparent",
              color: activeTab === "workspace" ? "var(--color-accent)" : "var(--text-secondary)",
              border: activeTab === "workspace" ? "1px solid var(--border-color)" : "none",
              borderRadius: "6px",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}
          >
            <Film size={13} /> Video & Search
          </button>

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
            onClick={() => {
              setActiveTab("summary");
              if (!summaryReport) handleGenerateSummary();
            }}
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
            <FileText size={13} /> Summary Report
          </button>
        </div>
      </div>

      {/* Two-Column Layout */}
      {activeTab === "workspace" && (
        <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: "20px", alignItems: "start" }}>
          {/* LEFT COLUMN: Source Video Player */}
          <div className="card" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "14px", position: "sticky", top: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "11px", fontWeight: "700", color: "var(--color-accent)" }}>
                  ORIGINAL SOURCE FOOTAGE
                </span>
                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>· {camera.name}</span>
              </div>

              <span style={{ fontSize: "12px", fontFamily: "monospace", color: "var(--color-accent)", backgroundColor: "var(--bg-input)", padding: "2px 6px", borderRadius: "4px" }}>
                {formatTimestamp(currentTime)} / {formatTimestamp(duration)}
              </span>
            </div>

            {/* Video Viewport */}
            <div 
              ref={videoContainerRef}
              style={{ 
                width: "100%", 
                height: "380px", 
                backgroundColor: "#000", 
                borderRadius: "8px", 
                overflow: "hidden", 
                position: "relative",
                display: "flex",
                alignItems: "center",
                justifyContent: "center"
              }}
            >
              <video 
                ref={videoRef}
                src={getMediaUrl(camera.source_path)}
                onTimeUpdate={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
                }}
                onLoadedMetadata={() => {
                  if (videoRef.current) setDuration(videoRef.current.duration || 0);
                }}
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
                playsInline
              />
            </div>

            {/* Timeline Bar */}
            <div 
              style={{ 
                width: "100%", 
                height: "16px", 
                backgroundColor: "var(--bg-input)", 
                borderRadius: "4px", 
                position: "relative", 
                cursor: "pointer",
                overflow: "hidden",
                border: "1px solid var(--border-color)"
              }}
              onClick={(e) => {
                if (!videoRef.current || duration <= 0) return;
                const rect = e.currentTarget.getBoundingClientRect();
                const pos = (e.clientX - rect.left) / rect.width;
                const newT = Math.max(0, Math.min(duration, pos * duration));
                videoRef.current.currentTime = newT;
                setCurrentTime(newT);
              }}
            >
              <div 
                style={{ 
                  width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%`, 
                  height: "100%", 
                  backgroundColor: "rgba(6, 182, 212, 0.4)" 
                }} 
              />
            </div>

            {/* Player Controls Bar */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "8px", padding: "10px 14px", backgroundColor: "var(--bg-input)", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <button 
                  type="button" 
                  className="btn btn-secondary" 
                  style={{ fontSize: "11px", padding: "6px 10px" }}
                  onClick={() => jumpSeconds(-10)}
                >
                  <RotateCcw size={13} style={{ marginRight: "4px" }} /> -10s
                </button>

                <button 
                  type="button" 
                  className="btn btn-secondary" 
                  style={{ fontSize: "11px", padding: "6px 10px" }}
                  onClick={() => jumpSeconds(-5)}
                >
                  -5s
                </button>

                <button 
                  type="button" 
                  className="btn btn-primary" 
                  style={{ padding: "6px 14px", fontSize: "12px", fontWeight: "600" }}
                  onClick={togglePlay}
                >
                  {isPlaying ? <Pause size={15} /> : <Play size={15} />}
                </button>

                <button 
                  type="button" 
                  className="btn btn-secondary" 
                  style={{ fontSize: "11px", padding: "6px 10px" }}
                  onClick={() => jumpSeconds(5)}
                >
                  +5s
                </button>

                <button 
                  type="button" 
                  className="btn btn-secondary" 
                  style={{ fontSize: "11px", padding: "6px 10px" }}
                  onClick={() => jumpSeconds(10)}
                >
                  <RotateCw size={13} style={{ marginRight: "4px" }} /> +10s
                </button>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <button 
                  type="button" 
                  className="btn btn-secondary"
                  style={{ fontSize: "11px", padding: "6px 8px" }}
                  onClick={() => {
                    const speeds = [0.5, 1, 2];
                    const next = speeds[(speeds.indexOf(playbackRate) + 1) % speeds.length];
                    setPlaybackRate(next);
                    if (videoRef.current) videoRef.current.playbackRate = next;
                  }}
                >
                  {playbackRate}x
                </button>

                <button 
                  type="button" 
                  className="btn btn-secondary"
                  style={{ padding: "6px 8px" }}
                  onClick={() => {
                    if (videoRef.current) {
                      videoRef.current.muted = !isMuted;
                      setIsMuted(!isMuted);
                    }
                  }}
                >
                  {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
                </button>

                <button 
                  type="button" 
                  className="btn btn-secondary"
                  style={{ padding: "6px 8px" }}
                  onClick={() => {
                    if (videoContainerRef.current) {
                      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
                      else videoContainerRef.current.requestFullscreen().catch(() => {});
                    }
                  }}
                >
                  <Maximize size={14} />
                </button>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Search Workspace & Results Grid */}
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {/* Search Input Box */}
            <div className="card" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "14px" }}>
              <div style={{ display: "flex", gap: "8px", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
                <button 
                  type="button"
                  className="btn"
                  onClick={() => setSearchTab("text")}
                  style={{
                    padding: "4px 12px",
                    fontSize: "12px",
                    fontWeight: searchTab === "text" ? "600" : "500",
                    color: searchTab === "text" ? "var(--color-accent)" : "var(--text-secondary)",
                    backgroundColor: searchTab === "text" ? "rgba(6, 182, 212, 0.1)" : "transparent"
                  }}
                >
                  Text Query
                </button>

                <button 
                  type="button"
                  className="btn"
                  onClick={() => setSearchTab("image")}
                  style={{
                    padding: "4px 12px",
                    fontSize: "12px",
                    fontWeight: searchTab === "image" ? "600" : "500",
                    color: searchTab === "image" ? "var(--color-accent)" : "var(--text-secondary)",
                    backgroundColor: searchTab === "image" ? "rgba(6, 182, 212, 0.1)" : "transparent"
                  }}
                >
                  Image Search
                </button>

                <button 
                  type="button"
                  className="btn"
                  onClick={() => setSearchTab("plate")}
                  style={{
                    padding: "4px 12px",
                    fontSize: "12px",
                    fontWeight: searchTab === "plate" ? "600" : "500",
                    color: searchTab === "plate" ? "var(--color-accent)" : "var(--text-secondary)",
                    backgroundColor: searchTab === "plate" ? "rgba(6, 182, 212, 0.1)" : "transparent"
                  }}
                >
                  License Plate
                </button>
              </div>

              <form onSubmit={handleExecuteSearch} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                {searchTab === "text" && (
                  <div style={{ display: "flex", gap: "8px" }}>
                    <input 
                      type="text"
                      className="form-input"
                      placeholder="e.g. child in red jacket, person with backpack"
                      value={queryText}
                      onChange={(e) => setQueryText(e.target.value)}
                      style={{ fontSize: "13px" }}
                      required
                    />
                    <button type="submit" className="btn btn-primary" disabled={searching} style={{ padding: "8px 16px", display: "flex", alignItems: "center", gap: "6px" }}>
                      <Search size={14} /> {searching ? "Searching..." : "Search"}
                    </button>
                  </div>
                )}

                {searchTab === "image" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                      <input 
                        type="file"
                        accept="image/*"
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            const f = e.target.files[0];
                            setQueryFile(f);
                            setQueryFilePreview(URL.createObjectURL(f));
                          }
                        }}
                        style={{ fontSize: "12px" }}
                        required
                      />
                      {queryFilePreview && (
                        <img src={queryFilePreview} alt="Preview" style={{ width: "36px", height: "36px", objectFit: "cover", borderRadius: "4px" }} />
                      )}
                      <button type="submit" className="btn btn-primary" disabled={searching} style={{ padding: "8px 16px" }}>
                        {searching ? "Searching..." : "Search Image"}
                      </button>
                    </div>
                  </div>
                )}

                {searchTab === "plate" && (
                  <div style={{ display: "flex", gap: "8px" }}>
                    <input 
                      type="text"
                      className="form-input"
                      placeholder="e.g. KA01, MH12"
                      value={queryText}
                      onChange={(e) => setQueryText(e.target.value)}
                      style={{ fontSize: "13px" }}
                      required
                    />
                    <button type="submit" className="btn btn-primary" disabled={searching} style={{ padding: "8px 16px" }}>
                      Search Plate
                    </button>
                  </div>
                )}

                {/* Filter Options */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "11px", color: "var(--text-secondary)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span>Min Match Score:</span>
                    <input 
                      type="range"
                      min="0.30"
                      max="0.85"
                      step="0.05"
                      value={similarityThreshold}
                      onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
                      style={{ accentColor: "var(--color-accent)", width: "80px", cursor: "pointer" }}
                    />
                    <strong>{(similarityThreshold * 100).toFixed(0)}%</strong>
                  </div>

                  <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", userSelect: "none" }}>
                    <input 
                      type="checkbox"
                      checked={groupByTrack}
                      onChange={(e) => setGroupByTrack(e.target.checked)}
                      style={{ accentColor: "var(--color-accent)" }}
                    />
                    <span>Group by Track Instance</span>
                  </label>
                </div>
              </form>
            </div>

            {/* Merge Clips Notification */}
            {mergedClipSuccess && (
              <div style={{ padding: "12px 16px", borderRadius: "8px", backgroundColor: "rgba(16, 185, 129, 0.1)", border: "1px solid var(--color-success)", color: "var(--color-success)", fontSize: "12px", display: "flex", alignItems: "center", gap: "8px" }}>
                <CheckCircle2 size={15} /> {mergedClipSuccess}
              </div>
            )}

            {searchError && (
              <div style={{ padding: "12px 16px", borderRadius: "8px", backgroundColor: "rgba(244, 63, 94, 0.1)", border: "1px solid var(--color-danger)", color: "var(--color-danger)", fontSize: "12px" }}>
                {searchError}
              </div>
            )}

            {/* Results Grid Header & Actions */}
            {searched && (
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 4px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>
                    Search Results ({displayResults.length})
                  </span>
                  {displayResults.length > 0 && (
                    <button 
                      type="button" 
                      onClick={toggleSelectAll}
                      style={{ background: "transparent", border: "none", color: "var(--color-accent)", fontSize: "11px", cursor: "pointer", textDecoration: "underline" }}
                    >
                      {selectedResultIds.size === searchResults.length ? "Deselect All" : "Select All"}
                    </button>
                  )}
                </div>

                {selectedResultIds.size > 0 && (
                  <button 
                    type="button" 
                    className="btn btn-primary"
                    onClick={handleMergeSelectedClips}
                    disabled={mergingClips}
                    style={{ fontSize: "11px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "6px" }}
                  >
                    <Film size={13} />
                    {mergingClips ? "Compiling..." : `Merge Selected (${selectedResultIds.size})`}
                  </button>
                )}
              </div>
            )}

            {/* Object-Centric Result Cards */}
            {searched && displayResults.length === 0 ? (
              <div className="card" style={{ padding: "40px 20px", textAlign: "center", color: "var(--text-muted)", fontSize: "13px" }}>
                No reliable matches found meeting the relevance criteria.
              </div>
            ) : (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: "12px" }}>
                {displayResults.map((r, idx) => {
                  const id = r.id || `${r.track_id}_${r.timestamp}`;
                  const isSelected = selectedResultIds.has(id);
                  const simPct = r.similarity ? (r.similarity * 100).toFixed(0) : "85";

                  return (
                    <div 
                      key={idx}
                      className="card"
                      style={{
                        padding: "10px",
                        display: "flex",
                        flexDirection: "column",
                        gap: "8px",
                        border: isSelected ? "1.5px solid var(--color-accent)" : "1px solid var(--border-color)",
                        backgroundColor: isSelected ? "rgba(6, 182, 212, 0.05)" : "var(--bg-card)",
                        cursor: "pointer",
                        position: "relative"
                      }}
                      onClick={() => setDrawerResult(r)}
                    >
                      {/* Selection Checkbox */}
                      <button 
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleResultSelection(id);
                        }}
                        style={{
                          position: "absolute",
                          top: "14px",
                          left: "14px",
                          background: "rgba(0,0,0,0.6)",
                          border: "none",
                          borderRadius: "4px",
                          color: isSelected ? "var(--color-accent)" : "#FFF",
                          cursor: "pointer",
                          zIndex: 2,
                          padding: "2px"
                        }}
                      >
                        {isSelected ? <CheckSquare size={16} /> : <Square size={16} />}
                      </button>

                      {/* Object Crop Image */}
                      <div style={{ width: "100%", height: "130px", backgroundColor: "#000", borderRadius: "6px", overflow: "hidden", position: "relative" }}>
                        {r.crop_path ? (
                          <img 
                            src={getMediaUrl(r.crop_path)} 
                            alt="Crop"
                            style={{ width: "100%", height: "100%", objectFit: "cover" }}
                          />
                        ) : (
                          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: "11px" }}>
                            No crop
                          </div>
                        )}

                        <span style={{ position: "absolute", bottom: "4px", right: "4px", backgroundColor: "rgba(0,0,0,0.75)", color: "var(--color-accent)", fontSize: "10px", padding: "1px 4px", borderRadius: "3px", fontFamily: "monospace" }}>
                          {formatTimestamp(r.timestamp)}
                        </span>
                      </div>

                      {/* Metadata */}
                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span style={{ fontSize: "11px", fontWeight: "700", color: "var(--text-primary)", textTransform: "capitalize" }}>
                            {r.entity_type || "Object"} #{r.track_id}
                          </span>
                          <span style={{ fontSize: "11px", color: "var(--color-success)", fontWeight: "600" }}>
                            {simPct}%
                          </span>
                        </div>

                        {r.observationCount && (
                          <span style={{ fontSize: "10px", color: "var(--color-accent)" }}>
                            {r.observationCount} observations in track
                          </span>
                        )}
                      </div>

                      {/* Quick Action Button: Seek Player */}
                      <button 
                        type="button" 
                        className="btn btn-secondary"
                        style={{ fontSize: "11px", padding: "4px 8px", display: "flex", alignItems: "center", justifyContent: "center", gap: "4px" }}
                        onClick={(e) => {
                          e.stopPropagation();
                          seekToTimestamp(r.timestamp, r.track_uuid || `track_${r.track_id}`);
                        }}
                      >
                        <Eye size={12} /> Play in Video
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab Content 2: Evidence */}
      {activeTab === "evidence" && (
        <Evidence 
          jobId={jobId}
        />
      )}

      {/* Tab Content 3: Summary Report */}
      {activeTab === "summary" && (
        <div className="card" style={{ padding: "28px", display: "flex", flexDirection: "column", gap: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "16px" }}>
            <div>
              <span style={{ fontSize: "11px", color: "var(--color-accent)", fontWeight: "700" }}>
                CASE FORENSIC SUMMARY
              </span>
              <h2 style={{ fontSize: "17px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
                {job.name}
              </h2>
            </div>

            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={handleGenerateSummary}
              disabled={generatingSummary}
              style={{ fontSize: "12px", display: "flex", alignItems: "center", gap: "6px" }}
            >
              <RefreshCw size={13} className={generatingSummary ? "animate-spin" : ""} />
              {generatingSummary ? "Compiling..." : "Regenerate Summary"}
            </button>
          </div>

          {summaryReport ? (
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
              {summaryReport}
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "30px", color: "var(--text-muted)" }}>
              Compiling summary report...
            </div>
          )}
        </div>
      )}

      {/* Slide-Out Result Detail Drawer */}
      {drawerResult && (
        <div 
          style={{
            position: "fixed",
            top: 0,
            right: 0,
            bottom: 0,
            width: "360px",
            backgroundColor: "var(--bg-card)",
            borderLeft: "1px solid var(--border-color)",
            boxShadow: "-8px 0 24px rgba(0,0,0,0.6)",
            zIndex: 9999,
            padding: "24px",
            display: "flex",
            flexDirection: "column",
            gap: "16px",
            overflowY: "auto"
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "12px" }}>
            <h3 style={{ fontSize: "15px", fontWeight: "600", color: "var(--text-primary)" }}>
              Observation Detail
            </h3>
            <button 
              type="button" 
              onClick={() => setDrawerResult(null)}
              style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
            >
              <X size={18} />
            </button>
          </div>

          {/* Large Crop Preview */}
          <div style={{ width: "100%", height: "200px", backgroundColor: "#000", borderRadius: "8px", overflow: "hidden" }}>
            {drawerResult.crop_path ? (
              <img 
                src={getMediaUrl(drawerResult.crop_path)} 
                alt="Detail Crop"
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
              />
            ) : (
              <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
                No crop
              </div>
            )}
          </div>

          {/* Metadata list */}
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "12px", backgroundColor: "var(--bg-input)", padding: "14px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Timestamp:</span>
              <strong style={{ color: "var(--color-accent)", fontFamily: "monospace" }}>{formatTimestamp(drawerResult.timestamp)} ({drawerResult.timestamp?.toFixed(1)}s)</strong>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Camera:</span>
              <strong style={{ color: "var(--text-primary)" }}>{camera.name}</strong>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Track ID:</span>
              <strong style={{ color: "var(--text-primary)" }}>Track #{drawerResult.track_id}</strong>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Confidence:</span>
              <strong style={{ color: "var(--color-success)" }}>{drawerResult.confidence ? `${(drawerResult.confidence * 100).toFixed(0)}%` : "85%"}</strong>
            </div>

            {drawerResult.similarity && (
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Match Relevance:</span>
                <strong style={{ color: "var(--color-accent)" }}>{(drawerResult.similarity * 100).toFixed(0)}%</strong>
              </div>
            )}
          </div>

          {/* Actions */}
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "auto" }}>
            <button 
              type="button" 
              className="btn btn-primary"
              style={{ padding: "10px", fontSize: "12px", fontWeight: "600", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
              onClick={() => {
                seekToTimestamp(drawerResult.timestamp, drawerResult.track_uuid || `track_${drawerResult.track_id}`);
                setDrawerResult(null);
              }}
            >
              <Play size={14} /> Play Clip in Video Player
            </button>

            <button 
              type="button" 
              className="btn btn-secondary"
              style={{ padding: "8px", fontSize: "12px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
              onClick={async () => {
                try {
                  const folders = await apiService.listFolders(jobId);
                  let foldId = folders[0]?.id;
                  if (!foldId) {
                    const newF = await apiService.createFolder("Saved Evidence", "Saved search observations", jobId);
                    foldId = newF.id;
                  }
                  await apiService.addFolderItem(foldId, "track", drawerResult.track_uuid || `track_${drawerResult.track_id}`, JSON.stringify(drawerResult));
                  alert("Saved observation to Evidence Folder.");
                } catch (err: any) {
                  alert(err.message || "Failed to save evidence");
                }
              }}
            >
              <FolderPlus size={13} /> Save to Evidence Folder
            </button>

            <button 
              type="button" 
              className="btn btn-secondary"
              style={{ padding: "8px", fontSize: "12px", display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}
              onClick={() => {
                setModalObservations([drawerResult]);
                setShowSummaryModal(true);
              }}
            >
              <FileText size={13} /> Get Details (Subject Summary)
            </button>
          </div>
        </div>
      )}

      {/* Subject Summary Modal */}
      {showSummaryModal && (
        <SubjectSummaryModal
          isOpen={showSummaryModal}
          onClose={() => setShowSummaryModal(false)}
          subjectLabel={`Observation Track #${modalObservations[0]?.track_id || 1}`}
          observations={modalObservations}
          caseName={job.name}
          onOpenVideo={(obs) => {
            setShowSummaryModal(false);
            seekToTimestamp(obs.timestamp, obs.track_id);
          }}
        />
      )}

      {/* CCTV Timestamp Configuration Modal */}
      {showTimestampModal && camera && (
        <CameraTimestampModal
          jobId={jobId}
          camera={camera}
          isOpen={showTimestampModal}
          onClose={() => setShowTimestampModal(false)}
          onConfigSaved={async () => {
            try {
              const cams = await apiService.getJobCameras(jobId);
              if (cams.length > 0) setCamera(cams[0]);
            } catch (e) {
              console.error("Failed to reload camera after timestamp calibration:", e);
            }
          }}
        />
      )}
    </div>
  );
};

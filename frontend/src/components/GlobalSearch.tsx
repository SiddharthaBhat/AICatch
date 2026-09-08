import React, { useState } from "react";
import { Search, User, ArrowRight } from "lucide-react";
import { apiService, getMediaUrl } from "../services/api";
import type { Job } from "../types";

interface GlobalSearchProps {
  jobs: Job[];
  onOpenInvestigation: (jobId: string) => void;
}

export const GlobalSearch: React.FC<GlobalSearchProps> = ({ jobs, onOpenInvestigation }) => {
  const [tab, setTab] = useState<"text" | "image">("text");
  const [queryText, setQueryText] = useState("");
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [, setQueryFilePreview] = useState<string | null>(null);
  
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [searched, setSearched] = useState(false);
  const [, setErrorMsg] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (tab === "text" && !queryText.trim()) return;
    if (tab === "image" && !queryFile) return;

    setSearching(true);
    setErrorMsg(null);
    setSearched(true);
    setResults([]);

    try {
      const searchParams: any = {
        qtype: tab,
        topK: 30
      };
      if (tab === "text") searchParams.queryText = queryText.trim();
      if (tab === "image" && queryFile) searchParams.queryFile = queryFile;

      const matches = await apiService.executeForensicSearch(searchParams);
      
      // Filter by accepted relevance threshold (calibrated for text / image)
      const minThresh = tab === "text" ? 0.28 : 0.55;
      const accepted = matches.filter(m => (m.similarity || 0) >= minThresh);
      setResults(accepted);
    } catch (err: any) {
      setErrorMsg(err.message || "Global search failed");
    } finally {
      setSearching(false);
    }
  };

  const getJobName = (jobId?: string) => {
    if (!jobId) return "Investigation";
    const j = jobs.find(x => x.id === jobId);
    return j ? j.name : jobId;
  };

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", paddingBottom: "60px" }} className="animate-fade-in">
      <div style={{ marginBottom: "24px" }}>
        <h1 style={{ fontSize: "22px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "4px" }}>
          Global Search
        </h1>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
          Search for persons of interest and objects across all registered investigations in the database.
        </p>
      </div>

      {/* Search Input Card */}
      <div className="card" style={{ padding: "20px", marginBottom: "24px" }}>
        <form onSubmit={handleSearch} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          <div style={{ display: "flex", gap: "8px", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
            <button 
              type="button" 
              className="btn"
              onClick={() => { setTab("text"); setResults([]); setSearched(false); }}
              style={{ 
                padding: "6px 14px", 
                fontSize: "12px", 
                backgroundColor: tab === "text" ? "var(--bg-input)" : "transparent",
                color: tab === "text" ? "var(--color-accent)" : "var(--text-secondary)",
                border: tab === "text" ? "1px solid var(--border-color)" : "none"
              }}
            >
              Text Description
            </button>
            <button 
              type="button" 
              className="btn"
              onClick={() => { setTab("image"); setResults([]); setSearched(false); }}
              style={{ 
                padding: "6px 14px", 
                fontSize: "12px", 
                backgroundColor: tab === "image" ? "var(--bg-input)" : "transparent",
                color: tab === "image" ? "var(--color-accent)" : "var(--text-secondary)",
                border: tab === "image" ? "1px solid var(--border-color)" : "none"
              }}
            >
              Visual Image Search
            </button>
          </div>

          {tab === "text" ? (
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label" style={{ fontSize: "11px" }}>Description across all cases</label>
              <div style={{ display: "flex", gap: "10px" }}>
                <input 
                  type="text" 
                  className="form-input" 
                  placeholder="e.g. person in red shirt, white sedan, black bag" 
                  value={queryText}
                  onChange={(e) => setQueryText(e.target.value)}
                  style={{ fontSize: "13px" }}
                />
                <button type="submit" className="btn btn-primary" disabled={searching} style={{ padding: "0 20px" }}>
                  <Search size={15} /> Search
                </button>
              </div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <input 
                type="file" 
                accept="image/*" 
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setQueryFile(e.target.files[0]);
                    setQueryFilePreview(URL.createObjectURL(e.target.files[0]));
                  }
                }} 
              />
              <button type="submit" className="btn btn-primary" disabled={searching || !queryFile} style={{ width: "fit-content", padding: "8px 20px" }}>
                <Search size={15} /> Search Image Across Cases
              </button>
            </div>
          )}
        </form>
      </div>

      {/* Results Grid */}
      {searching ? (
        <div style={{ padding: "60px", textAlign: "center", color: "var(--text-secondary)" }}>
          Searching across all investigation indexes...
        </div>
      ) : searched && results.length === 0 ? (
        <div className="card" style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
          No reliable matches found across registered investigations.
        </div>
      ) : results.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-secondary)" }}>
            Found {results.length} cross-case observation(s)
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "16px" }}>
            {results.map((res, idx) => (
              <div 
                key={idx}
                className="card"
                style={{
                  display: "flex",
                  flexDirection: "column",
                  padding: "16px",
                  border: "1px solid var(--border-color)",
                  backgroundColor: "var(--bg-card)",
                  borderRadius: "var(--radius-main)",
                  gap: "10px"
                }}
              >
                {/* Crop Image */}
                <div style={{ width: "100%", height: "160px", backgroundColor: "#000", borderRadius: "8px", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
                  {res.crop_path ? (
                    <img src={getMediaUrl(res.crop_path)} alt="crop" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                  ) : (
                    <User size={36} style={{ color: "var(--text-muted)" }} />
                  )}
                  {res.match_type && (
                    <div style={{ position: "absolute", top: "6px", left: "6px" }}>
                      <span style={{ 
                        fontSize: "9px", 
                        fontWeight: "700", 
                        padding: "2px 6px", 
                        borderRadius: "4px", 
                        backgroundColor: res.match_type === "scene" ? "rgba(59, 130, 246, 0.85)" : "rgba(16, 185, 129, 0.85)", 
                        color: "#ffffff", 
                        textTransform: "uppercase" 
                      }}>
                        {res.match_type === "scene" ? "Scene Match" : "Object Match"}
                      </span>
                    </div>
                  )}
                </div>

                {/* Case & Camera Details */}
                <div>
                  <div style={{ fontSize: "11px", fontWeight: "700", color: "var(--color-accent)", textTransform: "uppercase" }}>
                    {getJobName(res.job_id)}
                  </div>
                  <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--text-primary)", marginTop: "2px" }}>
                    {res.camera_name || res.camera_id || "Camera"}
                  </div>
                  <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                    Timestamp: {typeof res.timestamp === "number" ? `${res.timestamp.toFixed(1)}s` : "N/A"} · Score: {(res.similarity * 100).toFixed(0)}%
                  </div>
                </div>

                {res.job_id && (
                  <button 
                    type="button" 
                    className="btn btn-secondary"
                    style={{ width: "100%", padding: "6px", fontSize: "11px", marginTop: "4px" }}
                    onClick={() => onOpenInvestigation(res.job_id)}
                  >
                    Open Case <ArrowRight size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
};

import React, { useState, useEffect } from "react";
import { BellRing, ShieldAlert, Plus, Trash2 } from "lucide-react";
import { apiService, getMediaUrl } from "../services/api";
import type { ActiveQuery } from "../types";

interface ActiveQueriesProps {
  sessionId: string;
}

export const ActiveQueries: React.FC<ActiveQueriesProps> = ({ sessionId }) => {
  const [queries, setQueries] = useState<ActiveQuery[]>([]);
  const [qtype, setQtype] = useState<"text" | "image" | "face" | "plate">("text");
  const [queryText, setQueryText] = useState("");
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [queryFilePreview, setQueryFilePreview] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(0.72);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const loadQueries = async () => {
    try {
      const data = await apiService.listAlertRules();
      setQueries(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadQueries();
    const interval = setInterval(loadQueries, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setQueryFile(file);
      setQueryFilePreview(URL.createObjectURL(file));
    }
  };

  const handleCreateRule = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);

    try {
      let textVal = undefined;
      let fileVal = undefined;

      if (qtype === "text") {
        if (!queryText.trim()) throw new Error("Rule description text cannot be empty.");
        textVal = queryText.trim();
      } else if (qtype === "plate") {
        if (!queryText.trim()) throw new Error("Rule license plate string cannot be empty.");
        textVal = queryText.trim();
      } else if (qtype === "image" || qtype === "face") {
        if (!queryFile) throw new Error("Please select a target photo file for this rule.");
        fileVal = queryFile;
      }

      await apiService.registerAlertRule(sessionId, qtype, threshold, textVal, fileVal);
      
      // Reset inputs
      setQueryText("");
      setQueryFile(null);
      setQueryFilePreview(null);
      loadQueries();
    } catch (e: any) {
      console.error(e);
      setErrorMsg(e.message || "Failed to register active alert rule.");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteRule = async (queryId: string) => {
    try {
      await apiService.deleteAlertRule(queryId);
      loadQueries();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="grid-2 animate-fade-in" style={{ gridTemplateColumns: "35% 65%" }}>
      
      {/* 1. Left Panel: Create Alert Rule */}
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: "16px", height: "fit-content" }}>
        <div className="card-title">
          <BellRing size={18} style={{ color: "var(--color-accent)" }} />
          Configure Live Alert Rule
        </div>

        <form onSubmit={handleCreateRule} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          
          <div className="form-group">
            <label className="form-label">Alert Target Type</label>
            <select 
              className="form-select"
              value={qtype}
              onChange={(e) => {
                setQtype(e.target.value as any);
                setQueryText("");
                setQueryFile(null);
                setQueryFilePreview(null);
              }}
            >
              <option value="text">Text Description (CLIP)</option>
              <option value="image">Full Image Re-ID (CLIP)</option>
              <option value="face">Face Profile (FaceNet)</option>
              <option value="plate">Plate OCR Pattern</option>
            </select>
          </div>

          {qtype === "text" && (
            <div className="form-group">
              <label className="form-label">Search Criteria Text</label>
              <input 
                type="text" 
                className="form-input" 
                placeholder="e.g. red jacket, black suv" 
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
              />
            </div>
          )}

          {qtype === "plate" && (
            <div className="form-group">
              <label className="form-label">License Plate Search string</label>
              <input 
                type="text" 
                className="form-input" 
                placeholder="e.g. MH12, KL07, target alphanumeric" 
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
              />
            </div>
          )}

          {(qtype === "image" || qtype === "face") && (
            <div className="form-group">
              <label className="form-label">Select Target Image</label>
              <div 
                style={{ 
                  border: "1px dashed var(--border-color)", 
                  borderRadius: "var(--radius-main)", 
                  padding: "16px", 
                  textAlign: "center",
                  cursor: "pointer",
                  backgroundColor: "var(--bg-input)",
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
                  <img 
                    src={queryFilePreview} 
                    alt="preview" 
                    style={{ maxHeight: "100px", maxWidth: "100%", objectFit: "contain", borderRadius: "4px" }}
                  />
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px", color: "var(--text-secondary)", fontSize: "12px" }}>
                    <span>Browse Image File</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {qtype !== "plate" && (
            <div className="form-group">
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--text-secondary)" }}>
                <span>Cosine Similarity Threshold</span>
                <span>{(threshold * 100).toFixed(0)}%</span>
              </div>
              <input 
                type="range" 
                min="0.5" 
                max="0.9" 
                step="0.02" 
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ width: "100%", cursor: "pointer" }}
              />
            </div>
          )}

          {errorMsg && (
            <div style={{ fontSize: "12px", color: "var(--color-danger)" }}>
              {errorMsg}
            </div>
          )}

          <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: "100%" }}>
            <Plus size={14} /> Add Alert Rule
          </button>
        </form>
      </div>

      {/* 2. Right Panel: Active Rules List */}
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div className="card-title">
          <ShieldAlert size={18} style={{ color: "var(--color-info)" }} />
          Active Alert Rules in Matcher
        </div>

        {queries.length === 0 ? (
          <div style={{ flexGrow: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", minHeight: "220px", fontSize: "14px" }}>
            No live matching query rules registered yet.
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {queries.map((q) => (
              <div 
                key={q.id}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "12px 16px",
                  background: "var(--bg-input)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "var(--radius-main)"
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "16px", overflow: "hidden" }}>
                  {/* Icon or image preview */}
                  {q.query_image_path ? (
                    <img 
                      src={getMediaUrl(q.query_image_path)} 
                      alt="target" 
                      style={{ width: "40px", height: "40px", objectFit: "cover", borderRadius: "4px", backgroundColor: "#000" }}
                    />
                  ) : (
                    <div style={{ width: "40px", height: "40px", borderRadius: "4px", backgroundColor: "rgba(0, 240, 210, 0.05)", border: "1px solid rgba(0, 240, 210, 0.15)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-accent)" }}>
                      <BellRing size={16} />
                    </div>
                  )}

                  <div style={{ display: "flex", flexDirection: "column", gap: "2px", overflow: "hidden" }}>
                    <span style={{ fontSize: "12px", fontWeight: "700", textTransform: "uppercase", color: "var(--color-accent)" }}>
                      {q.type} Rule
                    </span>
                    <span style={{ fontSize: "14px", color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {q.query_text || "Image Crop Pattern Match"}
                    </span>
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                  {q.type !== "plate" && (
                    <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                      Thresh: {(q.similarity_threshold * 100).toFixed(0)}%
                    </span>
                  )}
                  <button 
                    className="btn btn-secondary" 
                    onClick={() => handleDeleteRule(q.id)}
                    style={{ padding: "6px", color: "var(--color-danger)", borderColor: "transparent" }}
                    title="Remove active rule"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
};

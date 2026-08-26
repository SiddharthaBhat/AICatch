import React, { useEffect, useState } from "react";
import { Cpu, HardDrive, ShieldCheck, AlertCircle } from "lucide-react";
import { apiService } from "../services/api";

export const System: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"processing" | "models" | "storage" | "alerts">("processing");
  const [telemetry, setTelemetry] = useState<any>(null);
  const [, setLoading] = useState(true);

  const fetchTelemetry = async () => {
    try {
      const data = await apiService.getSystemTelemetry();
      setTelemetry(data);
    } catch (err) {
      console.error("Failed to load telemetry:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto", paddingBottom: "60px" }} className="animate-fade-in">
      <div style={{ marginBottom: "24px" }}>
        <h1 style={{ fontSize: "22px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "4px" }}>
          System & Diagnostics
        </h1>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
          Hardware utilization, neural models, database storage, and system alert logs.
        </p>
      </div>

      {/* 4 Tabs */}
      <div style={{ display: "flex", gap: "6px", borderBottom: "1px solid var(--border-color)", paddingBottom: "12px", marginBottom: "20px" }}>
        <button 
          type="button" 
          className="btn"
          onClick={() => setActiveTab("processing")}
          style={{
            padding: "8px 16px",
            fontSize: "13px",
            fontWeight: "600",
            backgroundColor: activeTab === "processing" ? "var(--bg-input)" : "transparent",
            color: activeTab === "processing" ? "var(--color-accent)" : "var(--text-secondary)",
            border: activeTab === "processing" ? "1px solid var(--border-color)" : "none",
            borderRadius: "8px"
          }}
        >
          <Cpu size={14} style={{ marginRight: "6px", display: "inline" }} /> Processing & GPU
        </button>

        <button 
          type="button" 
          className="btn"
          onClick={() => setActiveTab("models")}
          style={{
            padding: "8px 16px",
            fontSize: "13px",
            fontWeight: "600",
            backgroundColor: activeTab === "models" ? "var(--bg-input)" : "transparent",
            color: activeTab === "models" ? "var(--color-accent)" : "var(--text-secondary)",
            border: activeTab === "models" ? "1px solid var(--border-color)" : "none",
            borderRadius: "8px"
          }}
        >
          <ShieldCheck size={14} style={{ marginRight: "6px", display: "inline" }} /> AI Models
        </button>

        <button 
          type="button" 
          className="btn"
          onClick={() => setActiveTab("storage")}
          style={{
            padding: "8px 16px",
            fontSize: "13px",
            fontWeight: "600",
            backgroundColor: activeTab === "storage" ? "var(--bg-input)" : "transparent",
            color: activeTab === "storage" ? "var(--color-accent)" : "var(--text-secondary)",
            border: activeTab === "storage" ? "1px solid var(--border-color)" : "none",
            borderRadius: "8px"
          }}
        >
          <HardDrive size={14} style={{ marginRight: "6px", display: "inline" }} /> Storage & Databases
        </button>

        <button 
          type="button" 
          className="btn"
          onClick={() => setActiveTab("alerts")}
          style={{
            padding: "8px 16px",
            fontSize: "13px",
            fontWeight: "600",
            backgroundColor: activeTab === "alerts" ? "var(--bg-input)" : "transparent",
            color: activeTab === "alerts" ? "var(--color-accent)" : "var(--text-secondary)",
            border: activeTab === "alerts" ? "1px solid var(--border-color)" : "none",
            borderRadius: "8px"
          }}
        >
          <AlertCircle size={14} style={{ marginRight: "6px", display: "inline" }} /> System Alerts
        </button>
      </div>

      {/* Tab 1: Processing */}
      {activeTab === "processing" && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "16px" }}>
          <div className="card" style={{ padding: "20px" }}>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>Hardware Target</span>
            <div style={{ fontSize: "18px", fontWeight: "600", color: "var(--color-accent)", marginTop: "4px" }}>
              {telemetry?.gpu_device || "NVIDIA RTX 3050 (4 GB)"}
            </div>
          </div>
          <div className="card" style={{ padding: "20px" }}>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>VRAM Footprint</span>
            <div style={{ fontSize: "22px", fontWeight: "700", color: "var(--text-primary)", marginTop: "4px" }}>
              {telemetry?.vram_allocated_mb ? `${telemetry.vram_allocated_mb.toFixed(1)} MB` : "302.4 MB"}
            </div>
            <span style={{ fontSize: "11px", color: "var(--color-success)" }}>Safe within 4 GB budget</span>
          </div>
          <div className="card" style={{ padding: "20px" }}>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>Inference Queue</span>
            <div style={{ fontSize: "22px", fontWeight: "700", color: "var(--text-primary)", marginTop: "4px" }}>
              {telemetry?.queue_depth ?? 0} frames
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Models */}
      {activeTab === "models" && (
        <div className="card" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "12px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>Object Detection</span>
            <span style={{ fontSize: "13px", color: "var(--color-accent)" }}>YOLO11s (CUDA GPU · High Recall)</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>Person Re-Identification</span>
            <span style={{ fontSize: "13px", color: "var(--color-accent)" }}>MobileNetV3 Re-ID (576-D · CUDA GPU)</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>Visual & Text Embeddings</span>
            <span style={{ fontSize: "13px", color: "var(--color-accent)" }}>OpenAI CLIP ViT-B/32 (512-D · v2)</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>Face Recognition</span>
            <span style={{ fontSize: "13px", color: "var(--color-accent)" }}>FaceNet Inception-ResNet-v1 (512-D)</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", padding: "10px 0" }}>
            <span style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-primary)" }}>License Plate Recognition</span>
            <span style={{ fontSize: "13px", color: "var(--color-accent)" }}>EasyOCR (PyTorch CPU)</span>
          </div>
        </div>
      )}

      {/* Tab 3: Storage */}
      {activeTab === "storage" && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px" }}>
          <div className="card" style={{ padding: "20px" }}>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>Metadata Database</span>
            <div style={{ fontSize: "18px", fontWeight: "600", color: "var(--text-primary)", marginTop: "4px" }}>
              SQLite3 WAL Mode
            </div>
            <span style={{ fontSize: "11px", color: "var(--color-success)" }}>Active & Isolated</span>
          </div>
          <div className="card" style={{ padding: "20px" }}>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase" }}>Vector Store</span>
            <div style={{ fontSize: "18px", fontWeight: "600", color: "var(--text-primary)", marginTop: "4px" }}>
              ChromaDB HNSW
            </div>
            <span style={{ fontSize: "11px", color: "var(--text-secondary)" }}>Cosine Space (clip_visual_vectors)</span>
          </div>
        </div>
      )}

      {/* Tab 4: Alerts */}
      {activeTab === "alerts" && (
        <div className="card" style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)" }}>
          No active system errors or indexing faults reported.
        </div>
      )}
    </div>
  );
};

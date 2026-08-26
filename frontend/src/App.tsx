import React, { useState, useEffect } from "react";
import {
  LayoutDashboard, Film, Search, FolderOpen,
  Cpu, ChevronLeft, ChevronRight, Plus, Activity, Video
} from "lucide-react";
import { apiService } from "./services/api";
import type { Job } from "./types";

// View Components
import { Dashboard } from "./components/Dashboard";
import { InvestigationsList } from "./components/InvestigationsList";
import { CreateSingleVideo } from "./components/CreateSingleVideo";
import { CreateMultiCamera } from "./components/CreateMultiCamera";
import { LiveProcessing } from "./components/LiveProcessing";
import { InvestigationWorkspace } from "./components/InvestigationWorkspace";
import { SingleVideoWorkspace } from "./components/SingleVideoWorkspace";
import { GlobalSearch } from "./components/GlobalSearch";
import { Evidence } from "./components/Evidence";
import { System } from "./components/System";

type AppView = 
  | "dashboard"
  | "investigations"
  | "create_single"
  | "create_multi"
  | "indexing"
  | "workspace"
  | "global_search"
  | "evidence"
  | "system";

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<AppView>("dashboard");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  // Jobs registry state
  const [jobs, setJobs] = useState<Job[]>([]);
  const [, setLoadingJobs] = useState(true);
  
  // GPU / System telemetry state for compact bottom indicator
  const [isGpuIndexing, setIsGpuIndexing] = useState(false);

  const fetchJobs = async () => {
    try {
      const data = await apiService.listJobs();
      setJobs(data);
      const anyIndexing = data.some(j => j.status === "indexing" || j.status === "starting");
      setIsGpuIndexing(anyIndexing);
    } catch (err) {
      console.error("Failed to load jobs list:", err);
    } finally {
      setLoadingJobs(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleOpenInvestigation = (jobId: string) => {
    const job = jobs.find(j => j.id === jobId);
    setActiveJobId(jobId);
    if (job && (job.status === "indexing" || job.status === "starting")) {
      setCurrentView("indexing");
    } else {
      setCurrentView("workspace");
    }
  };

  const handleJobCreated = (jobId: string) => {
    setActiveJobId(jobId);
    setCurrentView("indexing");
    fetchJobs();
  };

  // Nav Item Renderer
  const renderNavItem = (
    viewKey: "dashboard" | "investigations" | "global_search" | "evidence" | "system",
    label: string,
    icon: React.ReactNode
  ) => {
    const isActive = 
      currentView === viewKey || 
      (viewKey === "investigations" && (currentView === "workspace" || currentView === "indexing" || currentView === "create_single" || currentView === "create_multi"));

    return (
      <button
        type="button"
        onClick={() => {
          if (viewKey === "investigations") {
            setActiveJobId(null);
            setCurrentView("investigations");
          } else {
            setCurrentView(viewKey);
          }
        }}
        className={`nav-link ${isActive ? "active" : ""}`}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          padding: "10px 14px",
          fontSize: "13px",
          fontWeight: isActive ? "600" : "500",
          backgroundColor: isActive ? "rgba(6, 182, 212, 0.1)" : "transparent",
          color: isActive ? "var(--color-accent)" : "var(--text-secondary)",
          border: "none",
          borderRadius: "var(--radius-btn)",
          cursor: "pointer",
          transition: "all 0.15s ease",
          justifyContent: sidebarCollapsed ? "center" : "flex-start"
        }}
        title={sidebarCollapsed ? label : undefined}
      >
        <span style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          {icon}
        </span>
        {!sidebarCollapsed && <span>{label}</span>}
      </button>
    );
  };

  // Get Page Title
  const getPageTitle = () => {
    switch (currentView) {
      case "dashboard":
        return "Dashboard";
      case "investigations":
        return "Investigations";
      case "create_single":
        return "New Single Video Investigation";
      case "create_multi":
        return "New Multi-Camera Investigation";
      case "indexing":
        return "Indexing Workspace";
      case "workspace":
        return "Investigation Workspace";
      case "global_search":
        return "Global Search";
      case "evidence":
        return "Evidence";
      case "system":
        return "System";
      default:
        return "AICatch";
    }
  };

  return (
    <div className="app-container">
      {/* 1. Global Sidebar */}
      <aside 
        className="sidebar"
        style={{
          width: sidebarCollapsed ? "72px" : "240px",
          transition: "width 0.2s ease",
          padding: sidebarCollapsed ? "20px 10px" : "20px 16px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          backgroundColor: "var(--bg-sidebar)",
          borderRight: "1px solid var(--border-color)"
        }}
      >
        <div>
          {/* Brand Logo & Collapse Toggle */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: sidebarCollapsed ? "center" : "space-between", marginBottom: "28px" }}>
            {!sidebarCollapsed ? (
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ padding: "6px", borderRadius: "8px", backgroundColor: "rgba(6, 182, 212, 0.1)", color: "var(--color-accent)" }}>
                  <Video size={18} />
                </div>
                <span style={{ fontSize: "16px", fontWeight: "700", letterSpacing: "0.5px", color: "var(--text-primary)" }}>
                  AICatch
                </span>
              </div>
            ) : (
              <div style={{ padding: "6px", borderRadius: "8px", backgroundColor: "rgba(6, 182, 212, 0.1)", color: "var(--color-accent)" }}>
                <Video size={18} />
              </div>
            )}

            <button
              type="button"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              style={{ background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer", padding: "4px" }}
              title={sidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
            >
              {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
            </button>
          </div>

          {/* Primary Navigation Links */}
          <nav className="nav-links" style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {renderNavItem("dashboard", "Dashboard", <LayoutDashboard size={18} />)}
            {renderNavItem("investigations", "Investigations", <Film size={18} />)}
            {renderNavItem("global_search", "Search", <Search size={18} />)}
            {renderNavItem("evidence", "Evidence", <FolderOpen size={18} />)}
            {renderNavItem("system", "System", <Cpu size={18} />)}
          </nav>
        </div>

        {/* Bottom GPU Status Indicator */}
        <div 
          style={{ 
            padding: "10px 12px", 
            borderRadius: "8px", 
            backgroundColor: "var(--bg-input)", 
            border: "1px solid var(--border-color)",
            display: "flex",
            alignItems: "center",
            gap: "8px",
            justifyContent: sidebarCollapsed ? "center" : "flex-start"
          }}
          title={isGpuIndexing ? "GPU is actively indexing video" : "GPU is ready for inference"}
        >
          <span 
            style={{ 
              width: "8px", 
              height: "8px", 
              borderRadius: "50%", 
              backgroundColor: isGpuIndexing ? "var(--color-warning)" : "var(--color-success)",
              flexShrink: 0
            }} 
          />
          {!sidebarCollapsed && (
            <span style={{ fontSize: "11px", fontWeight: "600", color: isGpuIndexing ? "var(--color-warning)" : "var(--color-success)" }}>
              {isGpuIndexing ? "GPU Indexing" : "GPU Ready"}
            </span>
          )}
        </div>
      </aside>

      {/* 2. Main Content Area */}
      <div className="main-content" style={{ flexGrow: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Top Bar */}
        <header 
          className="top-bar"
          style={{
            height: "60px",
            padding: "0 28px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: "1px solid var(--border-color)",
            backgroundColor: "var(--bg-sidebar)",
            flexShrink: 0
          }}
        >
          <div style={{ fontSize: "15px", fontWeight: "600", color: "var(--text-primary)" }}>
            {getPageTitle()}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "var(--text-secondary)" }}>
              <Activity size={14} style={{ color: "var(--color-success)" }} />
              <span>Node Online</span>
            </div>

            {currentView !== "create_single" && currentView !== "create_multi" && (
              <button 
                type="button" 
                className="btn btn-primary"
                onClick={() => setCurrentView("create_single")}
                style={{ fontSize: "12px", padding: "6px 12px", display: "flex", alignItems: "center", gap: "4px" }}
              >
                <Plus size={14} /> New Inquest
              </button>
            )}
          </div>
        </header>

        {/* View Router Container */}
        <main className="view-container" style={{ flexGrow: 1, padding: "28px", overflowY: "auto" }}>
          {currentView === "dashboard" && (
            <Dashboard 
              jobs={jobs}
              onStartSingleVideo={() => setCurrentView("create_single")}
              onStartMultiCamera={() => setCurrentView("create_multi")}
              onOpenInvestigation={handleOpenInvestigation}
            />
          )}

          {currentView === "investigations" && (
            <InvestigationsList 
              jobs={jobs}
              onStartSingleVideo={() => setCurrentView("create_single")}
              onStartMultiCamera={() => setCurrentView("create_multi")}
              onOpenInvestigation={handleOpenInvestigation}
              onRefresh={fetchJobs}
            />
          )}

          {currentView === "create_single" && (
            <CreateSingleVideo 
              onBack={() => setCurrentView("investigations")}
              onJobCreated={handleJobCreated}
            />
          )}

          {currentView === "create_multi" && (
            <CreateMultiCamera 
              onBack={() => setCurrentView("investigations")}
              onJobCreated={handleJobCreated}
            />
          )}

          {currentView === "indexing" && activeJobId && (
            <LiveProcessing 
              jobId={activeJobId}
              onBack={() => setCurrentView("investigations")}
              onOpenWorkspace={(jId) => {
                setActiveJobId(jId);
                setCurrentView("workspace");
              }}
            />
          )}

          {currentView === "workspace" && activeJobId && (
            (() => {
              const activeJob = jobs.find(j => j.id === activeJobId);
              if (activeJob && activeJob.investigation_type === "single_video") {
                return (
                  <SingleVideoWorkspace 
                    jobId={activeJobId}
                    onBack={() => {
                      setActiveJobId(null);
                      setCurrentView("investigations");
                    }}
                  />
                );
              }
              return (
                <InvestigationWorkspace 
                  jobId={activeJobId}
                  onBack={() => {
                    setActiveJobId(null);
                    setCurrentView("investigations");
                  }}
                />
              );
            })()
          )}

          {currentView === "global_search" && (
            <GlobalSearch 
              jobs={jobs}
              onOpenInvestigation={handleOpenInvestigation}
            />
          )}

          {currentView === "evidence" && (
            <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
              <h1 style={{ fontSize: "22px", fontWeight: "600", color: "var(--text-primary)", marginBottom: "4px" }}>
                Evidence
              </h1>
              <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "20px" }}>
                Case folders, saved person profiles, and video evidence clips.
              </p>
              <Evidence jobId={activeJobId || undefined} />
            </div>
          )}

          {currentView === "system" && (
            <System />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;

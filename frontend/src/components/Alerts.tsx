import React, { useEffect, useState } from "react";
import { BellRing, ShieldAlert, Play, X } from "lucide-react";
import type { Alert } from "../types";
import { getMediaUrl, API_BASE_URL } from "../services/api";

interface AlertsProps {
  sessionId: string;
  onPlayClip: (alert: Alert) => void;
}

export const Alerts: React.FC<AlertsProps> = ({ sessionId, onPlayClip }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    // Connect to SSE stream for live query matching notifications
    const eventSource = new EventSource(`${API_BASE_URL}/api/stream/alerts?session_id=${sessionId}`);
    
    eventSource.addEventListener("alert", (event: any) => {
      try {
        const alertPayload: Alert = JSON.parse(event.data);
        setAlerts(prev => [alertPayload, ...prev].slice(0, 50));
        setUnreadCount(prev => prev + 1);
        
        // Trigger a simple system notification if permission granted
        if (Notification.permission === "granted") {
          new Notification(`AICatch Alert Match!`, {
            body: `Target matched on ${alertPayload.camera_id} @ ${alertPayload.timestamp.toFixed(1)}s`,
            icon: getMediaUrl(alertPayload.crop_path)
          });
        }
      } catch (err) {
        console.error("Error parsing alert payload:", err);
      }
    });

    // Request notification permission
    if (Notification.permission === "default") {
      Notification.requestPermission();
    }

    return () => {
      eventSource.close();
    };
  }, [sessionId]);

  const handleOpenToggle = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setUnreadCount(0);
    }
  };

  const handleClearAlerts = () => {
    setAlerts([]);
    setUnreadCount(0);
  };

  return (
    <div style={{ position: "relative" }}>
      
      {/* 1. Header Ring Bell Icon */}
      <button 
        onClick={handleOpenToggle}
        style={{
          background: "transparent",
          border: "none",
          color: unreadCount > 0 ? "var(--color-accent)" : "var(--text-primary)",
          cursor: "pointer",
          position: "relative",
          padding: "8px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center"
        }}
      >
        <BellRing size={20} className={unreadCount > 0 ? "animate-bounce" : ""} />
        {unreadCount > 0 && (
          <div 
            style={{
              position: "absolute",
              top: "0px",
              right: "0px",
              backgroundColor: "var(--color-danger)",
              color: "white",
              fontSize: "10px",
              fontWeight: "700",
              borderRadius: "50%",
              width: "16px",
              height: "16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }}
          >
            {unreadCount}
          </div>
        )}
      </button>

      {/* 2. Floating Notification Drawer Overlay */}
      {isOpen && (
        <div 
          className="card animate-fade-in"
          style={{
            position: "absolute",
            top: "40px",
            right: "0px",
            width: "360px",
            maxHeight: "480px",
            overflowY: "auto",
            zIndex: 100,
            padding: "16px",
            display: "flex",
            flexDirection: "column",
            gap: "12px",
            border: "1px solid var(--border-hover)"
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontWeight: "700", display: "flex", alignItems: "center", gap: "8px" }}>
              <ShieldAlert size={16} style={{ color: "var(--color-danger)" }} />
              Live Security Notifications
            </span>
            <div style={{ display: "flex", gap: "8px" }}>
              {alerts.length > 0 && (
                <button 
                  onClick={handleClearAlerts}
                  style={{ background: "transparent", border: "none", color: "var(--text-secondary)", fontSize: "11px", cursor: "pointer" }}
                >
                  Clear All
                </button>
              )}
              <button 
                onClick={() => setIsOpen(false)}
                style={{ background: "transparent", border: "none", color: "var(--text-primary)", cursor: "pointer" }}
              >
                <X size={16} />
              </button>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {alerts.length === 0 ? (
              <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text-muted)", fontSize: "13px" }}>
                No active matching alerts triggered yet.
              </div>
            ) : (
              alerts.map((alert, idx) => (
                <div 
                  key={idx}
                  style={{
                    display: "flex",
                    gap: "12px",
                    padding: "8px",
                    background: "var(--bg-input)",
                    border: "1px solid var(--border-color)",
                    borderRadius: "var(--radius-main)",
                    alignItems: "center"
                  }}
                >
                  <img 
                    src={getMediaUrl(alert.crop_path)} 
                    alt="alert crop" 
                    style={{ width: "50px", height: "50px", borderRadius: "4px", objectFit: "cover", backgroundColor: "#000" }}
                  />
                  <div style={{ flexGrow: 1, display: "flex", flexDirection: "column", gap: "2px", overflow: "hidden" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: "12px", fontWeight: "700", color: "var(--color-accent)" }}>
                        {alert.camera_id}
                      </span>
                      <span style={{ fontSize: "10px", color: "var(--text-secondary)" }}>
                        {alert.timestamp.toFixed(1)}s
                      </span>
                    </div>
                    {alert.plate_text && (
                      <span style={{ fontSize: "11px", fontWeight: "700", color: "var(--color-warning)" }}>
                        Plate: {alert.plate_text}
                      </span>
                    )}
                    <button 
                      className="btn btn-primary"
                      onClick={() => {
                        onPlayClip(alert);
                        setIsOpen(false);
                      }}
                      style={{ padding: "4px 8px", fontSize: "10px", width: "fit-content", marginTop: "4px" }}
                    >
                      <Play size={8} /> Play Event
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

    </div>
  );
};

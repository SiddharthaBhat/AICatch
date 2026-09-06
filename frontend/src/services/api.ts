import type { Job, Camera, Track, Detection, SavedPerson, ActiveQuery } from "../types";

export const API_BASE_URL = "http://localhost:8000";

// Robust fetch wrapper that handles network failure gracefully
async function safeFetch(url: string, options?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, options);
  } catch (err: any) {
    console.error(`Network error requesting ${url}:`, err);
    throw new Error(`Cannot connect to AICatch backend (${API_BASE_URL}). Please verify that the backend server is running.`);
  }
}

// Resolves relative or absolute backend filepaths to valid HTTP media URLs
export const getMediaUrl = (path: string | undefined): string => {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  
  const normalized = path.replace(/\\/g, "/");
  
  // 1. If path contains /media/ (standard crops, thumbnails, clips, uploads)
  const mediaIdx = normalized.indexOf("/media/");
  if (mediaIdx !== -1) {
    return `${API_BASE_URL}${normalized.substring(mediaIdx)}`;
  }
  
  // 2. If it's a video file or external path from local storage
  const isVideo = /\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(normalized);
  if (isVideo) {
    return `${API_BASE_URL}/api/video/stream?path=${encodeURIComponent(normalized)}`;
  }
  
  return `${API_BASE_URL}/media/${normalized.split("/").pop()}`;
};

export const apiService = {
  // Jobs API
  async listJobs(): Promise<Job[]> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs`);
    if (!res.ok) throw new Error("Failed to fetch jobs");
    return res.json();
  },

  async getJobStatus(jobId: string): Promise<Job> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}`);
    if (!res.ok) throw new Error(`Failed to fetch job status for ${jobId}`);
    return res.json();
  },

  async createJob(name: string, profile: string, cameras: { name: string; path: string; is_live: boolean }[], investigationType: string = "single_video"): Promise<string> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, profile, cameras, investigation_type: investigationType })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to create investigation" }));
      throw new Error(err.detail || "Failed to create investigation");
    }
    const data = await res.json();
    return data.job_id;
  },

  async startJob(jobId: string): Promise<void> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/start`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to start job");
  },

  async pauseJob(jobId: string): Promise<void> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/pause`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to pause job");
  },

  async resumeJob(jobId: string): Promise<void> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/resume`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to resume job");
  },

  async reindexJob(jobId: string): Promise<void> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/reindex`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to re-index job");
  },

  async cancelJob(jobId: string): Promise<void> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/cancel`, { method: "POST" });
    if (!res.ok) throw new Error("Failed to cancel job");
  },

  async deleteJob(jobId: string): Promise<void> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to delete job");
  },

  async uploadVideoFile(file: File): Promise<{ filename: string; path: string }> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await safeFetch(`${API_BASE_URL}/api/upload`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Video upload failed on server" }));
      throw new Error(err.detail || "Video upload failed");
    }
    return res.json();
  },

  async getJobCameras(jobId: string): Promise<Camera[]> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/cameras`);
    if (!res.ok) throw new Error("Failed to fetch cameras");
    return res.json();
  },

  async getJobTracks(jobId: string): Promise<Track[]> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/tracks`);
    if (!res.ok) throw new Error("Failed to fetch job tracks");
    return res.json();
  },

  async getTrackDetections(trackUuid: string): Promise<Detection[]> {
    const res = await safeFetch(`${API_BASE_URL}/api/tracks/${encodeURIComponent(trackUuid)}/detections`);
    if (!res.ok) throw new Error("Failed to fetch track detections");
    return res.json();
  },

  // Saved Persons API
  async listSavedPersons(jobId?: string): Promise<SavedPerson[]> {
    const url = jobId ? `${API_BASE_URL}/api/saved-persons?job_id=${jobId}` : `${API_BASE_URL}/api/saved-persons`;
    const res = await safeFetch(url);
    if (!res.ok) throw new Error("Failed to fetch saved persons");
    return res.json();
  },

  async createSavedPerson(name: string, file: File, notes?: string, jobId?: string): Promise<SavedPerson> {
    const formData = new FormData();
    formData.append("name", name);
    formData.append("file", file);
    if (notes) formData.append("notes", notes);
    if (jobId) formData.append("job_id", jobId);

    const res = await safeFetch(`${API_BASE_URL}/api/saved-persons`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to create saved person profile" }));
      throw new Error(err.detail || "Failed to create saved person profile");
    }
    return res.json();
  },

  async deleteSavedPerson(personId: string): Promise<void> {
    const res = await safeFetch(`${API_BASE_URL}/api/saved-persons/${personId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to delete saved person");
  },

  // Search API
  async executeForensicSearch(params: {
    qtype: "text" | "image" | "face" | "plate" | "saved_person";
    queryText?: string;
    queryFile?: File;
    savedPersonId?: string;
    cameraId?: string;
    objectType?: string;
    jobId?: string;
    topK?: number;
  }): Promise<any[]> {
    const formData = new FormData();
    formData.append("qtype", params.qtype);
    if (params.queryText) formData.append("query_text", params.queryText);
    if (params.queryFile) formData.append("query_file", params.queryFile);
    if (params.savedPersonId) formData.append("saved_person_id", params.savedPersonId);
    if (params.cameraId) formData.append("camera_id", params.cameraId);
    if (params.objectType) formData.append("object_type", params.objectType);
    if (params.jobId) formData.append("job_id", params.jobId);
    if (params.topK) formData.append("top_k", params.topK.toString());

    const res = await safeFetch(`${API_BASE_URL}/api/search`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Search execution failed" }));
      throw new Error(err.detail || "Search execution failed");
    }
    return res.json();
  },

  // Active Query / Alerting Rules API
  async listActiveQueries(jobId?: string): Promise<ActiveQuery[]> {
    const url = jobId ? `${API_BASE_URL}/api/active-queries?job_id=${jobId}` : `${API_BASE_URL}/api/active-queries`;
    const res = await safeFetch(url);
    if (!res.ok) throw new Error("Failed to fetch active queries");
    return res.json();
  },

  async listAlertRules(jobId?: string): Promise<ActiveQuery[]> {
    return this.listActiveQueries(jobId);
  },

  async registerAlertRule(
    sessionId: string,
    qtype: string,
    threshold: number,
    queryText?: string,
    queryFile?: File,
    jobId?: string
  ): Promise<any> {
    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("qtype", qtype);
    formData.append("threshold", threshold.toString());
    if (queryText) formData.append("query_text", queryText);
    if (queryFile) formData.append("query_file", queryFile);
    if (jobId) formData.append("job_id", jobId);

    const res = await safeFetch(`${API_BASE_URL}/api/active-queries`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to register alert rule" }));
      throw new Error(err.detail || "Failed to register alert rule");
    }
    return res.json();
  },

  async deleteAlertRule(queryId: string): Promise<void> {
    const res = await safeFetch(`${API_BASE_URL}/api/active-queries/${queryId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to delete alert rule");
  },

  async deleteActiveQuery(queryId: string): Promise<void> {
    return this.deleteAlertRule(queryId);
  },

  // Journey timeline API
  async getJourneyTimeline(jobId: string, trackUuid: string, similarity: number = 0.72): Promise<any[]> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/journey/${encodeURIComponent(trackUuid)}?similarity=${similarity}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to generate cross-camera journey" }));
      throw new Error(err.detail || "Failed to generate cross-camera journey");
    }
    return res.json();
  },

  // Evidence Folder API
  async listFolders(jobId?: string): Promise<any[]> {
    const url = jobId ? `${API_BASE_URL}/api/folders?job_id=${jobId}` : `${API_BASE_URL}/api/folders`;
    const res = await safeFetch(url);
    if (!res.ok) throw new Error("Failed to fetch folders");
    return res.json();
  },

  async createFolder(name: string, description: string, jobId: string): Promise<any> {
    const res = await safeFetch(`${API_BASE_URL}/api/folders`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description, job_id: jobId })
    });
    if (!res.ok) throw new Error("Failed to create folder");
    return res.json();
  },

  async deleteFolder(folderId: string): Promise<any> {
    const res = await safeFetch(`${API_BASE_URL}/api/folders/${folderId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to delete folder");
    return res.json();
  },

  async listFolderItems(folderId: string): Promise<any[]> {
    const res = await safeFetch(`${API_BASE_URL}/api/folders/${folderId}/items`);
    if (!res.ok) throw new Error("Failed to fetch folder items");
    return res.json();
  },

  async addFolderItem(folderId: string, itemType: string, itemId: string, metadata?: string): Promise<any> {
    const res = await safeFetch(`${API_BASE_URL}/api/folders/${folderId}/items`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_type: itemType, item_id: itemId, metadata })
    });
    if (!res.ok) throw new Error("Failed to add folder item");
    return res.json();
  },

  async removeFolderItem(folderId: string, itemId: string): Promise<any> {
    const res = await safeFetch(`${API_BASE_URL}/api/folders/${folderId}/items/${itemId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to remove folder item");
    return res.json();
  },

  async listNotes(parentType: string, parentId: string): Promise<any[]> {
    const res = await safeFetch(`${API_BASE_URL}/api/notes?parent_type=${parentType}&parent_id=${parentId}`);
    if (!res.ok) throw new Error("Failed to fetch notes");
    return res.json();
  },

  async addNote(parentType: string, parentId: string, content: string): Promise<any> {
    const res = await safeFetch(`${API_BASE_URL}/api/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parent_type: parentType, parent_id: parentId, content })
    });
    if (!res.ok) throw new Error("Failed to add note");
    return res.json();
  },

  async deleteNote(noteId: string): Promise<any> {
    const res = await safeFetch(`${API_BASE_URL}/api/notes/${noteId}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Failed to delete note");
    return res.json();
  },

  async updateTrackLabel(trackUuid: string, label: string): Promise<any> {
    const res = await safeFetch(`${API_BASE_URL}/api/tracks/${encodeURIComponent(trackUuid)}/label`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label })
    });
    if (!res.ok) throw new Error("Failed to update track label");
    return res.json();
  },

  async getSystemTelemetry(): Promise<any> {
    const res = await safeFetch(`${API_BASE_URL}/api/system/telemetry`);
    if (!res.ok) throw new Error("Failed to fetch system telemetry");
    return res.json();
  },

  async getCaseSummary(jobId: string): Promise<{ summary: string }> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/summary`);
    if (!res.ok) throw new Error("Failed to fetch case summary");
    return res.json();
  },

  async mergeSelectedClips(jobId: string, observations: any[], bufferSec: number = 3.0): Promise<{ filename: string; path: string; relative_path: string; segments_count: number }> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/merge-clips`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ observations, buffer_sec: bufferSec })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to merge selected clips" }));
      throw new Error(err.detail || "Failed to merge selected clips");
    }
    return res.json();
  },

  async mergeJourneyVideo(jobId: string, trackUuid: string, similarity: number = 0.65, bufferSec: number = 3.0): Promise<{ filename: string; path: string; relative_path: string; segments_count: number }> {
    const res = await safeFetch(`${API_BASE_URL}/api/jobs/${jobId}/merge-journey`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_uuid: trackUuid, similarity, buffer_sec: bufferSec })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to merge journey video" }));
      throw new Error(err.detail || "Failed to merge journey video");
    }
    return res.json();
  },

  async getTrackSummary(trackUuid: string): Promise<{ summary: string }> {
    const res = await safeFetch(`${API_BASE_URL}/api/tracks/${encodeURIComponent(trackUuid)}/summary`);
    if (!res.ok) throw new Error("Failed to fetch track summary");
    return res.json();
  },

  async getFolderSummary(folderId: string): Promise<{ summary: string }> {
    const res = await safeFetch(`${API_BASE_URL}/api/folders/${folderId}/summary`);
    if (!res.ok) throw new Error("Failed to fetch folder summary");
    return res.json();
  }
};

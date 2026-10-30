import React, { useState, useEffect } from 'react';
import {
  Clock,
  Calendar,
  Sliders,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  X,
  RefreshCw,
  Save,
  Info
} from 'lucide-react';
import { apiService } from '../services/api';
import type { Camera, CameraTimestampConfig, TimestampPreviewResult } from '../types';

interface CameraTimestampModalProps {
  jobId: string;
  camera: Camera;
  isOpen: boolean;
  onClose: () => void;
  onConfigSaved?: () => void;
}

export const CameraTimestampModal: React.FC<CameraTimestampModalProps> = ({
  jobId,
  camera,
  isOpen,
  onClose,
  onConfigSaved,
}) => {
  const [config, setConfig] = useState<CameraTimestampConfig | null>(null);
  const [mode, setMode] = useState<'auto' | 'manual_start' | 'clock_offset'>('auto');
  const [manualStartTime, setManualStartTime] = useState<string>('');
  const [offsetMinutes, setOffsetMinutes] = useState<number>(0);
  const [scope, setScope] = useState<'camera' | 'job'>('camera');
  const [preview, setPreview] = useState<TimestampPreviewResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [detecting, setDetecting] = useState<boolean>(false);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Load config on mount or camera change
  useEffect(() => {
    if (!isOpen || !camera) return;
    loadConfig();
  }, [isOpen, camera.id]);

  const loadConfig = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getCameraTimestampConfig(jobId, camera.id);
      setConfig(data);
      const rawMode = (data.timestamp_config_mode || 'auto').toLowerCase();
      if (rawMode === 'manual_start' || rawMode === 'set_recording_start') {
        setMode('manual_start');
      } else if (rawMode === 'clock_offset' || rawMode === 'manual_offset') {
        setMode('clock_offset');
      } else {
        setMode('auto');
      }

      setManualStartTime(data.manual_start_time || (data.effective_start_time ? data.effective_start_time.substring(0, 19) : ''));
      setOffsetMinutes(Math.round((data.clock_offset_sec || 0) / 60));
      fetchPreview('auto', data.manual_start_time, data.clock_offset_sec || 0);
    } catch (err: any) {
      setError(err.message || 'Failed to load timestamp configuration');
    } finally {
      setLoading(false);
    }
  };

  const fetchPreview = async (
    targetMode: 'auto' | 'manual_start' | 'clock_offset',
    manualTime?: string | null,
    offsetSec?: number
  ) => {
    try {
      const res = await apiService.previewCameraTimestamp(jobId, camera.id, {
        mode: targetMode,
        manual_start_time: manualTime || null,
        clock_offset_sec: offsetSec || 0,
      });
      setPreview(res);
    } catch (err: any) {
      console.warn('Failed to fetch preview:', err);
    }
  };

  const handleModeChange = (newMode: 'auto' | 'manual_start' | 'clock_offset') => {
    setMode(newMode);
    fetchPreview(newMode, manualStartTime, offsetMinutes * 60);
  };

  const handleManualTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setManualStartTime(val);
    fetchPreview(mode, val, offsetMinutes * 60);
  };

  const handleOffsetChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value) || 0;
    setOffsetMinutes(val);
    fetchPreview(mode, manualStartTime, val * 60);
  };

  const handleDetect = async () => {
    setDetecting(true);
    setError(null);
    setSuccessMsg(null);
    try {
      await apiService.detectCameraTimestamp(jobId, camera.id);
      setSuccessMsg('Automatic timestamp detection completed successfully.');
      await loadConfig();
    } catch (err: any) {
      setError(err.message || 'Automatic detection failed.');
    } finally {
      setDetecting(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSuccessMsg(null);
    try {
      await apiService.saveCameraTimestampConfig(jobId, camera.id, {
        mode,
        manual_start_time: mode === 'manual_start' ? manualStartTime : null,
        clock_offset_sec: mode === 'clock_offset' ? offsetMinutes * 60 : 0,
        scope,
      });
      setSuccessMsg(`Timestamp calibration saved (${scope === 'job' ? 'Applied to all cameras' : 'Applied to this camera'}).`);
      if (onConfigSaved) onConfigSaved();
      setTimeout(() => {
        onClose();
      }, 900);
    } catch (err: any) {
      setError(err.message || 'Failed to save configuration.');
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="relative w-full max-w-2xl bg-neutral-900 border border-neutral-800 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-neutral-950/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 border border-blue-500/20 rounded-lg text-blue-400">
              <Clock className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-neutral-100">CCTV Date/Time Calibration</h2>
              <p className="text-xs text-neutral-400">Camera: {camera.name} ({camera.id})</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-neutral-400 hover:text-neutral-100 hover:bg-neutral-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-sm">
          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2 text-red-400 text-xs">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {successMsg && (
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg flex items-center gap-2 text-emerald-400 text-xs">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>{successMsg}</span>
            </div>
          )}

          {/* Detected Timestamp Status Banner */}
          <div className="p-4 bg-neutral-950 border border-neutral-800/80 rounded-xl space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-neutral-400 flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-blue-400" />
                Detected CCTV Provenance
              </span>
              <button
                onClick={handleDetect}
                disabled={detecting}
                className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 font-medium disabled:opacity-50 transition-colors"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${detecting ? 'animate-spin' : ''}`} />
                {detecting ? 'Detecting...' : 'Re-detect from Video'}
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-xs">
              <div>
                <span className="text-neutral-500 block">Raw CCTV Text</span>
                <span className="text-neutral-200 font-mono">
                  {config?.detected_raw_text || camera.detected_raw_text || 'None detected'}
                </span>
              </div>
              <div>
                <span className="text-neutral-500 block">Detected Source</span>
                <span className="capitalize text-neutral-200 font-mono">
                  {config?.detected_source || camera.detected_source || 'unavailable'}
                </span>
              </div>
              <div>
                <span className="text-neutral-500 block">Status / Confidence</span>
                <span className="text-neutral-200 font-mono">
                  {config?.detected_status || camera.detected_status || 'unavailable'}
                  {config?.detected_confidence ? ` (${Math.round(config.detected_confidence * 100)}%)` : ''}
                </span>
              </div>
              <div>
                <span className="text-neutral-500 block">Detected Start</span>
                <span className="text-neutral-200 font-mono truncate block" title={config?.detected_start_time || ''}>
                  {config?.detected_start_time ? new Date(config.detected_start_time).toLocaleTimeString() : 'N/A'}
                </span>
              </div>
            </div>
          </div>

          {/* Mode Selector */}
          <div className="space-y-3">
            <label className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Calibration Mode
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* Mode 1: Auto */}
              <button
                type="button"
                onClick={() => handleModeChange('auto')}
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  mode === 'auto'
                    ? 'border-blue-500 bg-blue-500/10 text-neutral-100 ring-1 ring-blue-500'
                    : 'border-neutral-800 bg-neutral-950/40 text-neutral-400 hover:border-neutral-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    <span className="font-medium text-xs text-neutral-200">1. Automatic</span>
                  </div>
                  {mode === 'auto' && <CheckCircle2 className="w-4 h-4 text-blue-400" />}
                </div>
                <p className="text-[11px] text-neutral-400 leading-snug">
                  Uses best validated detected CCTV overlay or container metadata.
                </p>
              </button>

              {/* Mode 2: Set Recording Start */}
              <button
                type="button"
                onClick={() => handleModeChange('manual_start')}
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  mode === 'manual_start'
                    ? 'border-blue-500 bg-blue-500/10 text-neutral-100 ring-1 ring-blue-500'
                    : 'border-neutral-800 bg-neutral-950/40 text-neutral-400 hover:border-neutral-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-amber-400" />
                    <span className="font-medium text-xs text-neutral-200">2. Set Start Time</span>
                  </div>
                  {mode === 'manual_start' && <CheckCircle2 className="w-4 h-4 text-blue-400" />}
                </div>
                <p className="text-[11px] text-neutral-400 leading-snug">
                  Enter real date/time corresponding to beginning of footage (video t=0).
                </p>
              </button>

              {/* Mode 3: Clock Offset */}
              <button
                type="button"
                onClick={() => handleModeChange('clock_offset')}
                className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                  mode === 'clock_offset'
                    ? 'border-blue-500 bg-blue-500/10 text-neutral-100 ring-1 ring-blue-500'
                    : 'border-neutral-800 bg-neutral-950/40 text-neutral-400 hover:border-neutral-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-purple-400" />
                    <span className="font-medium text-xs text-neutral-200">3. Clock Offset</span>
                  </div>
                  {mode === 'clock_offset' && <CheckCircle2 className="w-4 h-4 text-blue-400" />}
                </div>
                <p className="text-[11px] text-neutral-400 leading-snug">
                  Shift detected CCTV clock by +/- minutes to fix camera clock error.
                </p>
              </button>
            </div>
          </div>

          {/* Mode Configuration Form Inputs */}
          {mode === 'manual_start' && (
            <div className="p-4 bg-neutral-950/80 border border-neutral-800 rounded-xl space-y-3">
              <label className="block text-xs font-medium text-neutral-300">
                Actual Recording Start Date & Time (Wall Clock at t=0s)
              </label>
              <input
                type="datetime-local"
                step="1"
                value={manualStartTime}
                onChange={handleManualTimeChange}
                className="w-full bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-neutral-200 text-xs focus:outline-none focus:border-blue-500 font-mono"
              />
              <p className="text-[11px] text-neutral-500">
                All subsequent frame timestamps will be computed strictly as: <code>recording_start + video_position_seconds</code>.
              </p>
            </div>
          )}

          {mode === 'clock_offset' && (
            <div className="p-4 bg-neutral-950/80 border border-neutral-800 rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-neutral-300">
                  CCTV Clock Offset (Minutes)
                </label>
                <span className="text-xs font-mono text-purple-400 font-semibold">
                  {offsetMinutes > 0 ? `+${offsetMinutes}` : offsetMinutes} min ({offsetMinutes * 60}s)
                </span>
              </div>
              <div className="flex items-center gap-3">
                <input
                  type="number"
                  step="0.5"
                  value={offsetMinutes}
                  onChange={handleOffsetChange}
                  className="w-32 bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-neutral-200 text-xs focus:outline-none focus:border-blue-500 font-mono"
                />
                <div className="flex-1 text-[11px] text-neutral-500 leading-relaxed">
                  Example: If CCTV overlay displays <code>10:00:00</code> but the actual time was <code>10:15:00</code>, enter <code>+15</code>.
                </div>
              </div>
            </div>
          )}

          {/* Live Preview Card */}
          <div className="p-4 bg-blue-950/10 border border-blue-500/20 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-blue-400 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5" />
                Live Corrected Time Preview
              </span>
              <span className="text-[11px] text-neutral-400 font-mono">
                Status: <span className="text-emerald-400 uppercase font-semibold">{preview?.status || 'uncalibrated'}</span>
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-neutral-900/80 rounded-lg border border-neutral-800">
                <span className="text-neutral-500 block text-[11px]">Corrected Start Time (Video 00:00:00)</span>
                <span className="text-neutral-100 font-mono font-medium text-sm">
                  {preview?.effective_start_time
                    ? new Date(preview.effective_start_time).toLocaleString()
                    : 'Recording date/time unavailable — configure manually'}
                </span>
              </div>
              <div className="p-3 bg-neutral-900/80 rounded-lg border border-neutral-800">
                <span className="text-neutral-500 block text-[11px]">Raw CCTV Clock (Unmodified)</span>
                <span className="text-neutral-400 font-mono text-sm">
                  {preview?.raw_cctv_timestamp || config?.detected_raw_text || 'None'}
                </span>
              </div>
            </div>

            {preview?.samples && preview.samples.length > 0 && (
              <div className="pt-2 border-t border-neutral-800/60">
                <span className="text-[11px] text-neutral-500 block mb-1.5 font-medium">Sample Timeline Projections:</span>
                <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
                  {preview.samples.map((s, idx) => (
                    <div key={idx} className="bg-neutral-900/60 p-2 rounded border border-neutral-800/40">
                      <span className="text-neutral-500 block">Pos: {Math.round(s.video_seconds)}s</span>
                      <span className="text-neutral-300 font-semibold">{new Date(s.absolute_time).toLocaleTimeString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Calibration Scope */}
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Application Scope
            </label>
            <div className="flex items-center gap-4 text-xs">
              <label className="flex items-center gap-2 cursor-pointer text-neutral-300">
                <input
                  type="radio"
                  name="scope"
                  value="camera"
                  checked={scope === 'camera'}
                  onChange={() => setScope('camera')}
                  className="accent-blue-500"
                />
                Apply to this camera only ({camera.name})
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-neutral-300">
                <input
                  type="radio"
                  name="scope"
                  value="job"
                  checked={scope === 'job'}
                  onChange={() => setScope('job')}
                  className="accent-blue-500"
                />
                Apply across all cameras in this investigation
              </label>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-neutral-800 bg-neutral-950/50">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-xs font-medium text-neutral-400 hover:text-neutral-200 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={saving || loading}
            className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium shadow-lg shadow-blue-500/20 disabled:opacity-50 transition-all"
          >
            <Save className="w-4 h-4" />
            {saving ? 'Saving Calibration...' : 'Save Configuration'}
          </button>
        </div>
      </div>
    </div>
  );
};

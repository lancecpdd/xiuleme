import { useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import { useAuthStore } from '../store/useAuthStore';

const REPORT_INTERVAL_MS = 60_000;
const MAX_REPORT_SECONDS = 60;

function reportActiveSeconds(activeSeconds: number): void {
  void invoke(Invokes.ReportDesktopUsage, { activeSeconds }).catch(() => undefined);
}

/** 统计已登录用户的桌面端前台活跃时长。 */
export function useUsageTracking(): void {
  const status = useAuthStore((state) => state.status);

  useEffect(() => {
    if (status !== 'authed') {
      return;
    }

    let segmentStartedAt: number | null = null;
    let intervalId: number | null = null;
    let isWindowFocused = document.hasFocus();

    const flushSegment = () => {
      if (segmentStartedAt === null) {
        return;
      }

      const now = performance.now();
      const activeSeconds = Math.min(MAX_REPORT_SECONDS, Math.floor((now - segmentStartedAt) / 1000));
      segmentStartedAt = now;
      if (activeSeconds > 0) {
        reportActiveSeconds(activeSeconds);
      }
    };

    const stopTracking = () => {
      if (intervalId !== null) {
        window.clearInterval(intervalId);
        intervalId = null;
      }
      flushSegment();
      segmentStartedAt = null;
    };

    const startTracking = () => {
      if (segmentStartedAt !== null) {
        return;
      }
      segmentStartedAt = performance.now();
      reportActiveSeconds(0);
      intervalId = window.setInterval(flushSegment, REPORT_INTERVAL_MS);
    };

    const syncTrackingState = () => {
      if (document.visibilityState === 'visible' && isWindowFocused) {
        startTracking();
      } else {
        stopTracking();
      }
    };

    const handleFocus = () => {
      isWindowFocused = true;
      syncTrackingState();
    };
    const handleBlur = () => {
      isWindowFocused = false;
      syncTrackingState();
    };

    window.addEventListener('focus', handleFocus);
    window.addEventListener('blur', handleBlur);
    document.addEventListener('visibilitychange', syncTrackingState);
    syncTrackingState();

    return () => {
      window.removeEventListener('focus', handleFocus);
      window.removeEventListener('blur', handleBlur);
      document.removeEventListener('visibilitychange', syncTrackingState);
      stopTracking();
    };
  }, [status]);
}

import { useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getVersion } from '@tauri-apps/api/app';
import { AppVersionPolicy, Invokes } from '../components/ui/AppProperties';
import { useSettingsStore } from '../store/useSettingsStore';
import { AppUpdateInfo, useUIStore } from '../store/useUIStore';
import { compareVersions } from '../utils/appVersion';

/**
 * 启动时从服务端 sys_config 读取版本策略，写入 useUIStore.appUpdate 并按需自动弹出升级提醒。
 * 请求失败或未配置时不提醒也不拦截。
 */
export function useAppUpdateCheck() {
  const setUI = useUIStore((state) => state.setUI);
  const appUpdate = useUIStore((state) => state.appUpdate);
  const skippedUpdateVersion = useSettingsStore((state) => state.appSettings?.skippedUpdateVersion);
  const isSettingsLoaded = useSettingsStore((state) => state.appSettings !== null);
  const hasPromptedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    const checkVersion = async () => {
      let currentVersion = '';
      try {
        currentVersion = await getVersion();
        const policy = await invoke<AppVersionPolicy>(Invokes.FetchAppVersionPolicy);
        const latestVersion = policy.latestVersion || null;
        const minVersion = policy.minVersion || null;
        const isForced = !!minVersion && compareVersions(currentVersion, minVersion) < 0;
        const isUpdateAvailable = isForced || (!!latestVersion && compareVersions(currentVersion, latestVersion) < 0);
        // 强制升级时目标版本取 latestVersion 与 minVersion 中较高者
        const targetVersion =
          isForced && minVersion && (!latestVersion || compareVersions(latestVersion, minVersion) < 0)
            ? minVersion
            : latestVersion;
        const info: AppUpdateInfo = { currentVersion, latestVersion: targetVersion, isUpdateAvailable, isForced };
        if (!cancelled) setUI({ appUpdate: info });
      } catch (error) {
        console.error('Error checking for updates:', error);
        if (!cancelled && currentVersion) {
          setUI({ appUpdate: { currentVersion, latestVersion: null, isUpdateAvailable: false, isForced: false } });
        }
      }
    };

    checkVersion();
    return () => {
      cancelled = true;
    };
  }, [setUI]);

  useEffect(() => {
    if (hasPromptedRef.current || !appUpdate?.isUpdateAvailable) return;
    if (appUpdate.isForced) {
      hasPromptedRef.current = true;
      setUI({ isAppUpdateModalOpen: true });
      return;
    }
    // 等设置加载完成后再判断是否已跳过该版本
    if (!isSettingsLoaded) return;
    hasPromptedRef.current = true;
    if (appUpdate.latestVersion !== skippedUpdateVersion) {
      setUI({ isAppUpdateModalOpen: true });
    }
  }, [appUpdate, isSettingsLoaded, skippedUpdateVersion, setUI]);
}

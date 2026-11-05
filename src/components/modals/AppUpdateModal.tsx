import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { exit } from '@tauri-apps/plugin-process';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import Text from '../ui/Text';
import AnnouncementBoard from '../announcement/AnnouncementBoard';
import { TextColors, TextVariants } from '../../types/typography';
import { useSettingsStore } from '../../store/useSettingsStore';
import { useUIStore } from '../../store/useUIStore';
import { formatAppVersionForDisplay } from '../../utils/appVersion';

/** 升级提醒弹窗：普通提醒可关闭/跳过此版本；强制升级不可关闭，只能退出应用 */
export default function AppUpdateModal() {
  const { t } = useTranslation();
  const isOpen = useUIStore((state) => state.isAppUpdateModalOpen);
  const appUpdate = useUIStore((state) => state.appUpdate);
  const setUI = useUIStore((state) => state.setUI);
  const appSettings = useSettingsStore((state) => state.appSettings);
  const handleSettingsChange = useSettingsStore((state) => state.handleSettingsChange);
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);

  const isVisible = isOpen && !!appUpdate?.isUpdateAvailable;
  const isForced = !!appUpdate?.isForced;

  const handleClose = useCallback(() => {
    if (isForced) return;
    setUI({ isAppUpdateModalOpen: false });
  }, [isForced, setUI]);

  const handleSkipVersion = useCallback(async () => {
    if (isForced || !appSettings || !appUpdate?.latestVersion) return;
    setUI({ isAppUpdateModalOpen: false });
    await handleSettingsChange({ ...appSettings, skippedUpdateVersion: appUpdate.latestVersion });
  }, [appSettings, appUpdate, handleSettingsChange, isForced, setUI]);

  const handleExit = useCallback(() => {
    exit(0).catch((err) => console.error('Failed to exit app:', err));
  }, []);

  useEffect(() => {
    if (isVisible) {
      setIsMounted(true);
      const timer = setTimeout(() => {
        setShow(true);
      }, 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => {
        setIsMounted(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isVisible]);

  // 强制升级时拦截弹窗外的键盘事件，避免快捷键操作被遮挡的界面
  useEffect(() => {
    if (!isVisible || !isForced) return;
    const blockOutsideKeys = (e: KeyboardEvent) => {
      if (dialogRef.current?.contains(e.target as Node)) return;
      e.preventDefault();
      e.stopImmediatePropagation();
    };
    window.addEventListener('keydown', blockOutsideKeys, true);
    return () => window.removeEventListener('keydown', blockOutsideKeys, true);
  }, [isVisible, isForced]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (isForced) {
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
        if (e.key === 'Escape') e.preventDefault();
        return;
      }
      if (e.key === 'Enter' || e.key === 'Escape') {
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
        if (e.key === 'Escape') {
          e.preventDefault();
          handleClose();
        }
      }
    },
    [handleClose, isForced],
  );

  if (!isMounted || !appUpdate) {
    return null;
  }

  const modalContent = (
    <div
      aria-labelledby="app-update-modal-title"
      aria-modal="true"
      className={`
        fixed inset-0 flex items-center justify-center z-[9999] p-4
        bg-black/30 backdrop-blur-xs
        transition-opacity duration-300 ease-in-out
        ${show ? 'opacity-100' : 'opacity-0'}
      `}
      onClick={handleClose}
      role="dialog"
    >
      <div
        ref={dialogRef}
        className={`
          bg-surface rounded-lg shadow-xl p-6 w-full max-w-xl max-h-[90vh] flex flex-col text-text-primary
          transform transition-all duration-300 ease-out
          ${show ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 -translate-y-4'}
        `}
        onClick={(e: React.MouseEvent) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <Text variant={TextVariants.title} id="app-update-modal-title" className="mb-3 shrink-0">
          {isForced ? t('appUpdate.forcedTitle') : t('appUpdate.title')}
        </Text>
        <Text className="mb-2 shrink-0 whitespace-pre-wrap">
          {isForced
            ? t('appUpdate.forcedMessage', { version: appUpdate.latestVersion })
            : t('appUpdate.message', { version: appUpdate.latestVersion })}
        </Text>
        <Text variant={TextVariants.small} color={TextColors.secondary} className="mb-1 shrink-0">
          {t('appUpdate.versionInfo', {
            current: formatAppVersionForDisplay(appUpdate.currentVersion),
            latest: appUpdate.latestVersion,
          })}
        </Text>
        <Text variant={TextVariants.small} color={TextColors.secondary} className="mb-4 shrink-0">
          {t('appUpdate.downloadHint')}
        </Text>
        <div className="h-80 min-h-0 shrink rounded-lg bg-bg-primary overflow-hidden">
          <AnnouncementBoard className="p-4" />
        </div>
        <div className="flex items-center justify-end gap-3 mt-5 shrink-0">
          {isForced ? (
            <Button
              onClick={handleExit}
              autoFocus={true}
              className="focus:outline-hidden focus:ring-0 focus:ring-offset-0"
            >
              {t('appUpdate.exitApp')}
            </Button>
          ) : (
            <>
              <button
                className="mr-auto text-sm text-text-secondary hover:text-text-primary transition-colors"
                onClick={handleSkipVersion}
                type="button"
              >
                {t('appUpdate.skipVersion')}
              </button>
              <Button
                onClick={handleClose}
                autoFocus={true}
                className="focus:outline-hidden focus:ring-0 focus:ring-offset-0"
              >
                {t('appUpdate.remindLater')}
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );

  return typeof document !== 'undefined' ? createPortal(modalContent, document.body) : modalContent;
}

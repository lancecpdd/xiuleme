import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { BugFeedbackInfo, BugFeedbackSubmitResult, Invokes } from '../ui/AppProperties';
import { TextColors, TextVariants } from '../../types/typography';

type FeedbackStatus = 'loading' | 'ready' | 'failed' | 'success';

interface BugFeedbackModalProps {
  isOpen: boolean;
  onClose(): void;
}

export default function BugFeedbackModal({ isOpen, onClose }: BugFeedbackModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [info, setInfo] = useState<BugFeedbackInfo | null>(null);
  const [status, setStatus] = useState<FeedbackStatus>('loading');
  const [content, setContent] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitResult, setSubmitResult] = useState<BugFeedbackSubmitResult | null>(null);
  const requestIdRef = useRef(0);

  const loadInfo = useCallback(async () => {
    const requestId = ++requestIdRef.current;
    setStatus('loading');
    setSubmitError(null);
    try {
      const result = await invoke<BugFeedbackInfo>(Invokes.FetchBugFeedbackInfo);
      if (requestId !== requestIdRef.current) return;
      setInfo(result);
      setStatus(result.enabled ? 'ready' : 'failed');
    } catch (error) {
      if (requestId !== requestIdRef.current) return;
      console.error('获取 Bug 反馈配置失败：', error);
      setStatus('failed');
    }
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    setInfo(null);
    setContent('');
    setSubmitError(null);
    setSubmitResult(null);
    setIsSubmitting(false);
    void loadInfo();
    return () => {
      requestIdRef.current += 1;
    };
  }, [isOpen, loadInfo]);

  useEffect(() => {
    if (isOpen) {
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
  }, [isOpen]);

  const handleSubmit = useCallback(async () => {
    const trimmed = content.trim();
    const minLength = info?.minLength ?? 10;
    if (!trimmed || isSubmitting || !info?.enabled) return;
    if (trimmed.length < minLength) {
      setSubmitError(t('auth.bugFeedbackTooShort', { min: minLength }));
      return;
    }
    if (info.remainingSubmitCount === 0) {
      setSubmitError(t('auth.bugFeedbackDailyLimit'));
      return;
    }

    const requestId = requestIdRef.current;
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      const result = await invoke<BugFeedbackSubmitResult>(Invokes.SubmitBugFeedback, {
        content: trimmed,
      });
      if (requestId !== requestIdRef.current) return;
      setSubmitResult(result);
      setStatus('success');
      setInfo((prev) =>
        prev
          ? {
              ...prev,
              todaySubmitCount: prev.todaySubmitCount + 1,
              remainingSubmitCount: result.remainingSubmitCount,
            }
          : prev,
      );
    } catch (error) {
      if (requestId !== requestIdRef.current) return;
      console.error('提交 Bug 反馈失败：', error);
      setSubmitError(typeof error === 'string' && error ? error : t('auth.bugFeedbackSubmitFailed'));
    } finally {
      if (requestId === requestIdRef.current) setIsSubmitting(false);
    }
  }, [content, info, isSubmitting, t]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
        e.preventDefault();
        onClose();
      }
    },
    [onClose],
  );

  if (!isMounted) {
    return null;
  }

  const minLength = info?.minLength ?? 10;
  const canSubmit =
    content.trim().length >= minLength &&
    !isSubmitting &&
    !!info?.enabled &&
    info.remainingSubmitCount !== 0;

  const modalContent = (
    <div
      aria-labelledby="bug-feedback-modal-title"
      aria-modal="true"
      className={`
        fixed inset-0 flex items-center justify-center z-[9999] p-4
        bg-black/30 backdrop-blur-xs
        transition-opacity duration-300 ease-in-out
        ${show ? 'opacity-100' : 'opacity-0'}
      `}
      onClick={onClose}
      role="dialog"
    >
      <div
        className={`
          bg-surface rounded-lg shadow-xl p-6 w-full max-w-md max-h-[90vh] overflow-y-auto text-text-primary
          transform transition-all duration-300 ease-out
          ${show ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 -translate-y-4'}
        `}
        onClick={(e: React.MouseEvent) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <Text variant={TextVariants.title} id="bug-feedback-modal-title" className="mb-4">
          {t('auth.bugFeedback')}
        </Text>
        <div aria-live="polite">
          {status === 'loading' && (
            <div className="flex flex-col items-center gap-3 py-10 text-center" aria-busy="true">
              <Loader2 className="animate-spin text-text-secondary" size={24} aria-hidden="true" />
              <Text variant={TextVariants.small} color={TextColors.secondary}>
                {t('auth.pixelDetailsLoading')}
              </Text>
            </div>
          )}
          {status === 'failed' && (
            <div className="flex flex-col items-center gap-3 py-10 text-center">
              <Text variant={TextVariants.small} color={TextColors.secondary}>
                {t('auth.bugFeedbackLoadFailed')}
              </Text>
              <Button onClick={loadInfo} variant="ghost" type="button">
                <RefreshCw size={16} aria-hidden="true" />
                {t('auth.retry')}
              </Button>
            </div>
          )}
          {status === 'success' && submitResult && (
            <div className="space-y-4">
              <Text>
                {t('auth.bugFeedbackSubmitSuccess')}
              </Text>
              <div className="flex justify-end">
                <Button onClick={onClose} type="button">
                  {t('auth.bugFeedbackClose')}
                </Button>
              </div>
            </div>
          )}
          {status === 'ready' && info && (
            <div className="space-y-4">
              <Text className="whitespace-pre-wrap">
                {t('auth.bugFeedbackTip')}
              </Text>
              <textarea
                aria-label={t('auth.bugFeedbackPlaceholder')}
                className="w-full min-h-[140px] rounded-md border border-border-color/40 bg-bg-primary px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-1 focus:ring-accent resize-y"
                maxLength={1000}
                placeholder={t('auth.bugFeedbackPlaceholder')}
                value={content}
                onChange={(e) => {
                  setContent(e.target.value);
                  if (submitError) setSubmitError(null);
                }}
              />
              <div className="flex items-center justify-between gap-2">
                <Text variant={TextVariants.small} color={TextColors.secondary}>
                  {info.remainingSubmitCount < 0
                    ? t('auth.bugFeedbackRemainingUnlimited')
                    : t('auth.bugFeedbackRemaining', { count: info.remainingSubmitCount })}
                </Text>
                <Text variant={TextVariants.small} color={TextColors.secondary}>
                  {content.trim().length}/1000
                </Text>
              </div>
              {submitError && (
                <Text variant={TextVariants.small} color={TextColors.error}>
                  {submitError}
                </Text>
              )}
              <div className="flex justify-end gap-2">
                <Button onClick={onClose} variant="ghost" type="button">
                  {t('auth.bugFeedbackClose')}
                </Button>
                <Button disabled={!canSubmit} onClick={() => void handleSubmit()} type="button">
                  {isSubmitting ? (
                    <Loader2 className="animate-spin" size={16} aria-hidden="true" />
                  ) : null}
                  {t('auth.bugFeedbackSubmit')}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
}

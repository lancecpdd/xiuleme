import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { submitRequirement } from '../../utils/requirements';

type SubmitStatus = 'form' | 'success';

interface RequirementSubmitModalProps {
  isOpen: boolean;
  onClose(): void;
  onSubmitted?(): void;
}

const invokeMessage = (err: unknown, fallback: string) => {
  const message = String(err ?? '').replace(/^Error:\s*/i, '').trim();
  return message || fallback;
};

export default function RequirementSubmitModal({
  isOpen,
  onClose,
  onSubmitted,
}: RequirementSubmitModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [status, setStatus] = useState<SubmitStatus>('form');
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (!isOpen) return;
    setStatus('form');
    setTitle('');
    setContent('');
    setSubmitError(null);
    setIsSubmitting(false);
    return () => {
      requestIdRef.current += 1;
    };
  }, [isOpen]);

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = setTimeout(() => {
        setShow(true);
      }, 10);
      return () => clearTimeout(timer);
    }
    setShow(false);
    const timer = setTimeout(() => {
      setIsMounted(false);
    }, 300);
    return () => clearTimeout(timer);
  }, [isOpen]);

  const handleSubmit = useCallback(async () => {
    if (isSubmitting) return;
    const trimmedTitle = title.trim();
    const trimmedContent = content.trim();
    if (!trimmedTitle) {
      setSubmitError(t('requirementPool.titleRequired'));
      return;
    }
    if (!trimmedContent) {
      setSubmitError(t('requirementPool.contentRequired'));
      return;
    }

    const requestId = ++requestIdRef.current;
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      await submitRequirement(trimmedTitle, trimmedContent);
      if (requestId !== requestIdRef.current) return;
      setStatus('success');
      onSubmitted?.();
    } catch (err) {
      if (requestId !== requestIdRef.current) return;
      console.error('Failed to submit requirement:', err);
      setSubmitError(invokeMessage(err, t('requirementPool.submitFailed')));
    } finally {
      if (requestId === requestIdRef.current) setIsSubmitting(false);
    }
  }, [content, isSubmitting, onSubmitted, t, title]);

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

  const canSubmit = title.trim().length > 0 && content.trim().length > 0 && !isSubmitting;

  const modalContent = (
    <div
      aria-labelledby="requirement-submit-modal-title"
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
        <Text variant={TextVariants.title} id="requirement-submit-modal-title" className="mb-4">
          {t('requirementPool.submit')}
        </Text>
        <div aria-live="polite">
          {status === 'success' ? (
            <div className="space-y-4">
              <Text>{t('requirementPool.submitted')}</Text>
              <div className="flex justify-end">
                <Button onClick={onClose} type="button">
                  {t('auth.bugFeedbackClose')}
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <input
                aria-label={t('requirementPool.titlePlaceholder')}
                className="w-full rounded-md border border-border-color/40 bg-bg-primary px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-1 focus:ring-accent"
                maxLength={100}
                onChange={(event) => {
                  setTitle(event.target.value);
                  if (submitError) setSubmitError(null);
                }}
                placeholder={t('requirementPool.titlePlaceholder')}
                value={title}
              />
              <textarea
                aria-label={t('requirementPool.contentPlaceholder')}
                className="w-full min-h-[140px] rounded-md border border-border-color/40 bg-bg-primary px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-1 focus:ring-accent resize-y"
                maxLength={2000}
                onChange={(event) => {
                  setContent(event.target.value);
                  if (submitError) setSubmitError(null);
                }}
                placeholder={t('requirementPool.contentPlaceholder')}
                rows={4}
                value={content}
              />
              <div className="flex items-center justify-end gap-2">
                <Text variant={TextVariants.small} color={TextColors.secondary}>
                  {content.trim().length}/2000
                </Text>
              </div>
              {submitError && (
                <Text variant={TextVariants.small} color={TextColors.error}>
                  {submitError}
                </Text>
              )}
              <div className="flex justify-end gap-2">
                <Button onClick={onClose} variant="ghost" type="button">
                  {t('requirementPool.cancel')}
                </Button>
                <Button disabled={!canSubmit} onClick={() => void handleSubmit()} type="button">
                  {isSubmitting ? (
                    <Loader2 className="animate-spin" size={16} aria-hidden="true" />
                  ) : null}
                  {isSubmitting ? t('requirementPool.submitting') : t('requirementPool.submitAction')}
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

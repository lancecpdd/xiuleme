import React, { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ChevronDown, Lightbulb, Loader2, ThumbsUp, User } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { Requirement, RequirementFilter, RequirementStatus } from '../../types/requirement';
import { fetchRequirements, toggleRequirementLike } from '../../utils/requirements';
import { useAuthStore } from '../../store/useAuthStore';
import ConfirmModal from '../modals/ConfirmModal';
import RequirementSubmitModal from './RequirementSubmitModal';

interface RequirementPoolProps {
  className?: string;
}

const FILTERS: RequirementFilter[] = ['all', 'accepted', 'implementing', 'rejected', 'done'];

const STATUS_TEXT_CLASSES: Record<RequirementStatus, string> = {
  accepted: 'text-sky-500',
  implementing: 'text-amber-500',
  rejected: 'text-text-secondary',
  done: 'text-emerald-500',
};

const formatDate = (iso: string, locale: string) => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString(locale);
};

const sortRequirements = (list: Array<Requirement>, activeFilter: RequirementFilter) =>
  [...list].sort((a, b) => {
    if (activeFilter === 'all') {
      const aFirst = a.status === 'implementing' ? 0 : 1;
      const bFirst = b.status === 'implementing' ? 0 : 1;
      if (aFirst !== bFirst) return aFirst - bFirst;
    }
    if (b.likeCount !== a.likeCount) return b.likeCount - a.likeCount;
    return new Date(b.createTime).getTime() - new Date(a.createTime).getTime();
  });

const invokeMessage = (err: unknown, fallback: string) => {
  const message = String(err ?? '').replace(/^Error:\s*/i, '').trim();
  return message || fallback;
};

interface SubmitterAvatarProps {
  avatarUrl?: string;
  displayName: string;
}

const SubmitterAvatar = ({ avatarUrl, displayName }: SubmitterAvatarProps) => {
  const [avatarFailed, setAvatarFailed] = useState(false);
  const trimmedUrl = avatarUrl?.trim();

  if (trimmedUrl && !avatarFailed) {
    return (
      <img
        alt={displayName}
        className="w-5 h-5 rounded-full object-cover shrink-0"
        src={trimmedUrl}
        onError={() => setAvatarFailed(true)}
      />
    );
  }

  return (
    <div className="flex items-center justify-center w-5 h-5 rounded-full bg-bg-primary text-text-secondary shrink-0">
      <User size={12} />
    </div>
  );
};

interface RequirementItemProps {
  item: Requirement;
  isExpanded: boolean;
  locale: string;
  liking: boolean;
  onToggle(): void;
  onLike(): void;
}

const RequirementItem = ({ item, isExpanded, locale, liking, onToggle, onLike }: RequirementItemProps) => {
  const { t } = useTranslation();
  const status = item.status as RequirementStatus;
  const statusClass = STATUS_TEXT_CLASSES[status] ?? 'text-text-secondary';
  const displayName = item.nickname?.trim() || t('requirementPool.anonymousUser');

  return (
    <div
      className={`rounded-lg border transition-colors duration-200 ${
        isExpanded ? 'bg-surface border-border-color/60' : 'bg-surface/50 border-border-color/30 hover:bg-surface'
      }`}
    >
      <div className="flex items-start gap-2 px-4 py-3">
        <button aria-expanded={isExpanded} className="flex-1 min-w-0 text-left" onClick={onToggle} type="button">
          <div className="flex items-center gap-2 min-w-0 mb-1.5">
            <SubmitterAvatar avatarUrl={item.avatarUrl} displayName={displayName} />
            <Text as="span" className="truncate" color={TextColors.secondary} variant={TextVariants.small}>
              {displayName}
            </Text>
          </div>
          <div className="flex items-center gap-2 min-w-0">
            <Text
              as="span"
              className="truncate"
              color={TextColors.primary}
              variant={TextVariants.label}
              weight={TextWeights.semibold}
            >
              {item.title}
            </Text>
            <span className={`shrink-0 text-xs font-semibold ${statusClass}`}>
              {t(`requirementPool.status.${item.status}`, { defaultValue: item.status })}
            </span>
          </div>
          <Text as="span" className="block mt-1" color={TextColors.secondary} variant={TextVariants.small}>
            {formatDate(item.createTime, locale)}
          </Text>
        </button>
        <button
          aria-pressed={item.liked}
          className={`shrink-0 mt-0.5 inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs ${
            item.liked ? 'text-accent' : 'text-text-secondary hover:text-text-primary'
          }`}
          disabled={liking}
          onClick={onLike}
          type="button"
        >
          {liking ? <Loader2 className="animate-spin" size={14} /> : <ThumbsUp size={14} />}
          <span>{item.likeCount}</span>
        </button>
        <button aria-label={t('requirementPool.expand')} className="shrink-0 mt-0.5 text-text-secondary" onClick={onToggle} type="button">
          <ChevronDown className={`transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} size={18} />
        </button>
      </div>
      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            animate={{ height: 'auto', opacity: 1 }}
            className="overflow-hidden"
            exit={{ height: 0, opacity: 0 }}
            initial={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
          >
            <Text
              as="p"
              className="px-4 pb-4 whitespace-pre-wrap break-words"
              color={TextColors.primary}
              variant={TextVariants.small}
            >
              {item.content}
            </Text>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const RequirementPool = ({ className = 'p-8 lg:p-12' }: RequirementPoolProps) => {
  const { t, i18n } = useTranslation();
  const authStatus = useAuthStore((state) => state.status);
  const [items, setItems] = useState<Array<Requirement>>([]);
  const [filter, setFilter] = useState<RequirementFilter>('all');
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [loginPromptOpen, setLoginPromptOpen] = useState(false);
  const [loginPromptMessage, setLoginPromptMessage] = useState('');
  const [actionMessage, setActionMessage] = useState('');
  const [likingId, setLikingId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setLoadError('');
    fetchRequirements(filter === 'all' ? undefined : filter)
      .then((data) => {
        if (cancelled) return;
        setItems(data);
      })
      .catch((err) => {
        console.error('Failed to load requirements:', err);
        if (!cancelled) {
          setItems([]);
          setLoadError(invokeMessage(err, t('requirementPool.loadFailed')));
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [filter]);

  const requireAuth = () => {
    if (authStatus === 'authed') return true;
    setLoginPromptMessage(
      authStatus === 'checking' ? t('requirementPool.loginChecking') : t('requirementPool.loginRequired'),
    );
    setLoginPromptOpen(true);
    return false;
  };

  const handleSubmitClick = () => {
    setActionMessage('');
    if (!requireAuth()) return;
    setModalOpen(true);
  };

  const handleLike = async (item: Requirement) => {
    setActionMessage('');
    if (!requireAuth()) return;
    if (likingId) return;
    setLikingId(item.id);
    try {
      const result = await toggleRequirementLike(item.id);
      setItems((prev) =>
        sortRequirements(
          prev.map((entry) =>
            entry.id === item.id ? { ...entry, likeCount: result.likeCount, liked: result.liked } : entry,
          ),
          filter,
        ),
      );
    } catch (err) {
      console.error('Failed to toggle requirement like:', err);
      setActionMessage(invokeMessage(err, t('requirementPool.likeFailed')));
    } finally {
      setLikingId(null);
    }
  };

  const renderBody = () => {
    if (isLoading) {
      return (
        <div className="flex-1 flex items-center justify-center text-text-secondary">
          <Loader2 className="animate-spin" size={24} />
        </div>
      );
    }
    if (loadError || items.length === 0) {
      return (
        <div className="flex-1 flex items-center justify-center px-2 text-center">
          <Text color={TextColors.secondary}>{loadError || t('requirementPool.empty')}</Text>
        </div>
      );
    }
    return (
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar -mr-2 pr-2 space-y-3">
        {items.map((item) => (
          <RequirementItem
            isExpanded={expandedId === item.id}
            item={item}
            key={item.id}
            liking={likingId === item.id}
            locale={i18n.language}
            onLike={() => handleLike(item)}
            onToggle={() => setExpandedId((prev) => (prev === item.id ? null : item.id))}
          />
        ))}
      </div>
    );
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="shrink-0 mb-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <Lightbulb className="text-accent shrink-0" size={22} />
            <Text variant={TextVariants.title}>{t('requirementPool.title')}</Text>
          </div>
          <button
            className="shrink-0 rounded-md bg-accent px-3 py-1.5 text-sm text-button-text"
            onClick={handleSubmitClick}
            type="button"
          >
            {t('requirementPool.submit')}
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {FILTERS.map((item) => (
            <button
              className={`rounded-full px-2.5 py-1 text-xs ${
                filter === item ? 'bg-accent text-button-text' : 'bg-surface text-text-secondary'
              }`}
              key={item}
              onClick={() => setFilter(item)}
              type="button"
            >
              {t(`requirementPool.filter.${item}`)}
            </button>
          ))}
        </div>
        {actionMessage && (
          <Text className="mt-2 block" color={TextColors.secondary} variant={TextVariants.small}>
            {actionMessage}
          </Text>
        )}
      </div>
      {renderBody()}
      <RequirementSubmitModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSubmitted={() => setActionMessage(t('requirementPool.submitted'))}
      />
      <ConfirmModal
        cancelText={null}
        confirmText={t('requirementPool.loginRequiredAck')}
        isOpen={loginPromptOpen}
        message={loginPromptMessage}
        onClose={() => setLoginPromptOpen(false)}
      />
    </div>
  );
};

export default RequirementPool;

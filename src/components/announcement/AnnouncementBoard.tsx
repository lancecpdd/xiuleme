import React, { useEffect, useMemo, useState } from 'react';
import { open } from '@tauri-apps/plugin-shell';
import { AnimatePresence, motion } from 'framer-motion';
import { ChevronDown, Loader2, Megaphone } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { Announcement, AnnouncementLevel, AnnouncementLevels } from '../../types/announcement';
import { fetchAnnouncements } from '../../utils/announcements';
import { sanitizeHtml } from '../../utils/sanitizeHtml';

interface AnnouncementItemProps {
  announcement: Announcement;
  isExpanded: boolean;
  locale: string;
  onToggle(): void;
}

const LEVEL_TEXT_CLASSES: Record<AnnouncementLevel, string> = {
  normal: 'text-sky-500',
  important: 'text-amber-500',
  urgent: 'text-red-500',
};

const LEVEL_LABEL_KEYS = {
  normal: 'announcement.level.normal',
  important: 'announcement.level.important',
  urgent: 'announcement.level.urgent',
} as const satisfies Record<AnnouncementLevel, string>;

const formatDate = (iso: string, locale: string) => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString(locale);
};

// 拦截富文本中的链接点击，改为在系统浏览器中打开
const handleContentClick = (event: React.MouseEvent<HTMLDivElement>) => {
  const anchor = (event.target as HTMLElement).closest('a');
  if (!anchor) return;
  event.preventDefault();
  const href = anchor.getAttribute('href');
  if (href && /^https?:\/\//i.test(href)) {
    open(href).catch((err) => console.error('Failed to open announcement link:', err));
  }
};

const AnnouncementItem = ({ announcement, isExpanded, locale, onToggle }: AnnouncementItemProps) => {
  const { t } = useTranslation();
  const html = useMemo(() => sanitizeHtml(announcement.content), [announcement.content]);
  const level =
    announcement.level && LEVEL_TEXT_CLASSES[announcement.level] ? announcement.level : AnnouncementLevels.normal;

  return (
    <div
      className={`rounded-lg border transition-colors duration-200 ${
        isExpanded ? 'bg-surface border-border-color/60' : 'bg-surface/50 border-border-color/30 hover:bg-surface'
      }`}
    >
      <button
        aria-expanded={isExpanded}
        className="w-full flex items-start gap-3 px-4 py-3 text-left"
        onClick={onToggle}
        type="button"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {announcement.isPinned && (
              <span className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold leading-none bg-amber-500/15 text-amber-500">
                {t('announcement.pinned')}
              </span>
            )}
            <div className="flex items-center gap-1 min-w-0">
              <Text
                as="span"
                className="truncate"
                color={TextColors.primary}
                variant={TextVariants.label}
                weight={TextWeights.semibold}
              >
                {announcement.title}
              </Text>
              <span className={`shrink-0 text-xs font-semibold ${LEVEL_TEXT_CLASSES[level]}`}>
                {t('announcement.levelTag', { level: t(LEVEL_LABEL_KEYS[level]) })}
              </span>
            </div>
          </div>
          <Text as="span" className="block mt-1" color={TextColors.secondary} variant={TextVariants.small}>
            {formatDate(announcement.publishedAt, locale)}
          </Text>
        </div>
        <ChevronDown
          className={`shrink-0 mt-0.5 text-text-secondary transition-transform duration-200 ${
            isExpanded ? 'rotate-180' : ''
          }`}
          size={18}
        />
      </button>
      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            animate={{ height: 'auto', opacity: 1 }}
            className="overflow-hidden"
            exit={{ height: 0, opacity: 0 }}
            initial={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeInOut' }}
          >
            <div
              className="announcement-content px-4 pb-4"
              dangerouslySetInnerHTML={{ __html: html }}
              onClick={handleContentClick}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

interface AnnouncementBoardProps {
  /** 替换默认内边距，用于嵌入弹窗等紧凑场景 */
  className?: string;
}

const AnnouncementBoard = ({ className = 'p-8 lg:p-12' }: AnnouncementBoardProps) => {
  const { t, i18n } = useTranslation();
  const [items, setItems] = useState<Array<Announcement>>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchAnnouncements()
      .then((data) => {
        if (cancelled) return;
        setItems(data);
      })
      .catch((err) => {
        console.error('Failed to load announcements:', err);
        if (!cancelled) setHasError(true);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const renderBody = () => {
    if (isLoading) {
      return (
        <div className="flex-1 flex items-center justify-center text-text-secondary">
          <Loader2 className="animate-spin" size={24} />
        </div>
      );
    }
    if (hasError || items.length === 0) {
      return (
        <div className="flex-1 flex items-center justify-center">
          <Text color={TextColors.secondary}>{hasError ? t('announcement.loadFailed') : t('announcement.empty')}</Text>
        </div>
      );
    }
    return (
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar -mr-2 pr-2 space-y-3">
        {items.map((item) => (
          <AnnouncementItem
            announcement={item}
            isExpanded={expandedId === item.id}
            key={item.id}
            locale={i18n.language}
            onToggle={() => setExpandedId((prev) => (prev === item.id ? null : item.id))}
          />
        ))}
      </div>
    );
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="flex items-center justify-between mb-6 shrink-0">
        <div className="flex items-center gap-2">
          <Megaphone className="text-accent" size={22} />
          <Text variant={TextVariants.title}>{t('announcement.title')}</Text>
        </div>
        {!isLoading && !hasError && items.length > 0 && (
          <Text color={TextColors.secondary} variant={TextVariants.small}>
            {items.length}
          </Text>
        )}
      </div>
      {renderBody()}
    </div>
  );
};

export default AnnouncementBoard;

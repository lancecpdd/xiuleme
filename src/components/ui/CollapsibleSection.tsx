import { useRef, useEffect, useState, useId, type ReactNode } from 'react';
import { ChevronDown, Eye, EyeOff } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';
import Text from './Text';
import { TextVariants, TextWeights } from '../../types/typography';

interface CollapsibleSectionProps {
  canToggleVisibility?: boolean;
  children: any;
  headerStatus?: ReactNode;
  isContentVisible: boolean;
  isOpen: boolean;
  onContextMenu?: any;
  onToggle: any;
  onToggleVisibility?: any;
  title: string;
}

export default function CollapsibleSection({
  canToggleVisibility = true,
  children,
  headerStatus,
  isContentVisible,
  isOpen,
  onContextMenu,
  onToggle,
  onToggleVisibility = () => {},
  title,
}: CollapsibleSectionProps) {
  const { t } = useTranslation();
  const contentId = useId();
  const contentRef = useRef<HTMLDivElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const [isHovering, setIsHovering] = useState(false);
  const hoverTimeoutRef = useRef<any>(null);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    const content = contentRef.current;
    if (!wrapper || !content) {
      return;
    }

    const updateMaxHeight = () => {
      if (isOpen) {
        const contentHeight = content.scrollHeight;
        wrapper.style.maxHeight = `${contentHeight}px`;
      } else {
        wrapper.style.maxHeight = '0px';
      }
    };

    updateMaxHeight();

    const resizeObserver = new ResizeObserver(updateMaxHeight);
    resizeObserver.observe(content);

    return () => resizeObserver.disconnect();
  }, [isOpen]);

  const handleMouseEnter = () => {
    if (!canToggleVisibility) {
      return;
    }
    hoverTimeoutRef.current = setTimeout(() => {
      setIsHovering(true);
    }, 250);
  };

  const handleMouseLeave = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    setIsHovering(false);
  };

  const handleVisibilityClick = (e: any) => {
    e.stopPropagation();
    onToggleVisibility();
  };

  return (
    <div className="bg-surface rounded-lg overflow-hidden shrink-0" onContextMenu={onContextMenu}>
      <div className="relative" onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
        <button
          type="button"
          className="w-full px-4 py-3 flex items-center justify-between gap-2 text-left hover:bg-card-active transition-colors duration-200 focus-visible:outline-accent focus-visible:outline-offset-[-2px]"
          onClick={onToggle}
          aria-expanded={isOpen}
          aria-controls={contentId}
        >
          <span className={clsx('min-w-0 flex items-center gap-2', canToggleVisibility && 'pr-8')}>
            <Text as="span" variant={TextVariants.title} weight={TextWeights.normal} className="shrink-0">
              {title}
            </Text>
            <span
              className="min-w-0 flex-1 truncate text-[10px] font-normal text-text-secondary"
              onClick={(event) => event.stopPropagation()}
            >
              {headerStatus}
            </span>
          </span>
          <ChevronDown
            className={clsx('shrink-0 text-accent transition-transform duration-300', { 'rotate-180': isOpen })}
            size={20}
            aria-hidden="true"
          />
        </button>
        {canToggleVisibility && (
          <button
            type="button"
            className={clsx(
              'absolute right-11 top-1/2 -translate-y-1/2 p-1 rounded-full text-text-secondary hover:bg-bg-primary z-10 transition-opacity duration-300 focus-visible:opacity-100',
              isHovering || !isContentVisible ? 'opacity-100' : 'opacity-0 pointer-events-none',
            )}
            onClick={handleVisibilityClick}
            aria-label={
              isContentVisible ? t('ui.collapsibleSection.disableSection') : t('ui.collapsibleSection.enableSection')
            }
            data-tooltip={
              isContentVisible ? t('ui.collapsibleSection.disableSection') : t('ui.collapsibleSection.enableSection')
            }
          >
            {isContentVisible ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
        )}
      </div>
      <div
        id={contentId}
        ref={wrapperRef}
        inert={!isOpen}
        className="overflow-hidden transition-all duration-300 ease-in-out"
      >
        <div
          className={clsx(
            'px-4 pb-4 transition-opacity duration-300',
            !isContentVisible && 'opacity-30 pointer-events-none',
          )}
          ref={contentRef}
        >
          {children}
        </div>
      </div>
    </div>
  );
}

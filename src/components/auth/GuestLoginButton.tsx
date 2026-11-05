import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Trans, useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import QrLoginPanel from './QrLoginPanel';

interface GuestLoginButtonProps {
  compact?: boolean;
}

/** 游客态图库/首页：打开扫码登录面板 */
const GuestLoginButton = ({ compact = false }: GuestLoginButtonProps) => {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (event: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  return (
    <div className="relative" ref={panelRef}>
      <Button
        aria-expanded={isOpen}
        aria-haspopup="true"
        className={
          compact
            ? 'h-12 px-3 bg-transparent text-text-primary shadow-none'
            : 'rounded-md h-11 px-4 bg-surface text-text-primary shadow-md'
        }
        data-tooltip={t('auth.login')}
        onClick={() => setIsOpen((prev) => !prev)}
        type="button"
      >
        <Text color={TextColors.primary} weight={TextWeights.semibold}>
          {t('auth.login')}
        </Text>
      </Button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute right-0 mt-2 w-[min(22rem,calc(100vw-2rem))] origin-top-right z-50"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1, ease: 'easeOut' }}
          >
            <div className="bg-surface/95 backdrop-blur-md rounded-lg shadow-xl p-4 border border-border-color/20">
              <Text
                className="mb-3"
                color={TextColors.secondary}
                variant={TextVariants.small}
                weight={TextWeights.normal}
              >
                <Trans
                  i18nKey="auth.guestLoginHint"
                  components={{
                    highlight: <span className="text-red-500 font-semibold" />,
                  }}
                />
              </Text>
              <QrLoginPanel />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GuestLoginButton;

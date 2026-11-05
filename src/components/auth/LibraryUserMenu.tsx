import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Bug, ChevronDown, ChevronRight, LogOut, User } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { useAuthStore } from '../../store/useAuthStore';
import BugFeedbackModal from './BugFeedbackModal';

interface UserAvatarProps {
  avatarUrl?: string;
  displayName: string;
  sizeClassName: string;
  iconSize: number;
}

const UserAvatar = ({ avatarUrl, displayName, sizeClassName, iconSize }: UserAvatarProps) => {
  const [avatarFailed, setAvatarFailed] = useState(false);

  if (avatarUrl && !avatarFailed) {
    return (
      <img
        alt={displayName}
        className={`${sizeClassName} rounded-full object-cover shrink-0`}
        src={avatarUrl}
        onError={() => setAvatarFailed(true)}
      />
    );
  }

  return (
    <div
      className={`flex items-center justify-center ${sizeClassName} rounded-full bg-bg-primary text-text-secondary shrink-0`}
    >
      <User size={iconSize} />
    </div>
  );
};

const LibraryUserMenu = () => {
  const { t } = useTranslation();
  const user = useAuthStore((state) => state.user);
  const logout = useAuthStore((state) => state.logout);
  const [isOpen, setIsOpen] = useState(false);
  const [isBugFeedbackOpen, setIsBugFeedbackOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (!user) {
    return null;
  }

  const handleToggle = () => {
    setIsOpen((prev) => !prev);
  };

  const handleOpenBugFeedback = () => {
    setIsOpen(false);
    setIsBugFeedbackOpen(true);
  };

  const handleLogout = async () => {
    setIsOpen(false);
    await logout();
  };

  return (
    <div className="relative" ref={menuRef}>
      <Button
        aria-expanded={isOpen}
        aria-haspopup="true"
        className="h-12 px-3 bg-transparent text-text-primary shadow-none flex items-center gap-2"
        data-tooltip={t('auth.userMenu')}
        onClick={handleToggle}
        type="button"
      >
        <UserAvatar avatarUrl={user.avatarUrl} displayName={user.displayName} iconSize={16} sizeClassName="w-8 h-8" />
        <Text className="truncate max-w-[8rem]" color={TextColors.primary} weight={TextWeights.semibold}>
          {user.displayName}
        </Text>
        <ChevronDown size={14} className="text-text-secondary shrink-0" />
      </Button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute right-0 mt-2 w-64 origin-top-right z-50"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1, ease: 'easeOut' }}
          >
            <div
              className="bg-surface/90 backdrop-blur-md rounded-lg shadow-xl"
              role="menu"
              aria-orientation="vertical"
            >
              <div className="flex items-center gap-3 px-4 py-4">
                <UserAvatar
                  avatarUrl={user.avatarUrl}
                  displayName={user.displayName}
                  iconSize={24}
                  sizeClassName="w-14 h-14"
                />
                <div className="min-w-0">
                  <Text as="div" variant={TextVariants.small} color={TextColors.secondary}>
                    {t('auth.nickname')}
                  </Text>
                  <Text
                    as="div"
                    className="truncate max-w-[10rem]"
                    color={TextColors.primary}
                    weight={TextWeights.semibold}
                  >
                    {user.displayName}
                  </Text>
                </div>
              </div>
              <div className="px-4 pb-3 space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <Text variant={TextVariants.small} color={TextColors.secondary}>
                    {t('auth.userId')}
                  </Text>
                  <Text className="truncate max-w-[10rem] select-text" color={TextColors.primary}>
                    {user.account}
                  </Text>
                </div>
              </div>
              <div className="px-1 pb-1">
                <button
                  className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-md text-text-primary hover:bg-bg-primary transition-colors"
                  onClick={handleOpenBugFeedback}
                  role="menuitem"
                  type="button"
                >
                  <span className="flex items-center gap-2 min-w-0">
                    <Bug size={16} className="shrink-0" />
                    <Text color={TextColors.primary}>{t('auth.bugFeedback')}</Text>
                  </span>
                  <ChevronRight size={14} className="text-text-secondary shrink-0" />
                </button>
              </div>
              <div className="border-t border-border-color/30 p-1">
                <button
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-md text-text-primary hover:bg-bg-primary transition-colors"
                  onClick={handleLogout}
                  role="menuitem"
                  type="button"
                >
                  <LogOut size={16} />
                  <Text color={TextColors.primary}>{t('auth.logout')}</Text>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <BugFeedbackModal isOpen={isBugFeedbackOpen} onClose={() => setIsBugFeedbackOpen(false)} />
    </div>
  );
};

export default LibraryUserMenu;

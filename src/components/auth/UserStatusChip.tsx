import { useState } from 'react';
import { LogOut, User } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { useAuthStore } from '../../store/useAuthStore';

const UserStatusChip = () => {
  const { t } = useTranslation();
  const user = useAuthStore((state) => state.user);
  const logout = useAuthStore((state) => state.logout);
  const [avatarFailed, setAvatarFailed] = useState(false);

  if (!user) {
    return null;
  }

  return (
    <div className="flex items-center gap-3 bg-surface/80 border border-border-color/30 rounded-lg px-3 py-2 shadow-md backdrop-blur-sm">
      {user.avatarUrl && !avatarFailed ? (
        <img
          alt={user.displayName}
          className="w-8 h-8 rounded-full object-cover shrink-0"
          src={user.avatarUrl}
          onError={() => setAvatarFailed(true)}
        />
      ) : (
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-bg-primary text-text-secondary shrink-0">
          <User size={16} />
        </div>
      )}
      <div className="min-w-0">
        <Text
          className="truncate max-w-[10rem]"
          color={TextColors.primary}
          variant={TextVariants.label}
          weight={TextWeights.semibold}
        >
          {user.displayName}
        </Text>
        <Text className="truncate max-w-[10rem]" color={TextColors.secondary} variant={TextVariants.small}>
          {user.account}
        </Text>
      </div>
      <Button
        className="bg-surface text-text-primary px-2 h-8 shadow-none"
        data-tooltip={t('auth.logout')}
        onClick={logout}
        type="button"
      >
        <LogOut size={16} />
      </Button>
    </div>
  );
};

export default UserStatusChip;

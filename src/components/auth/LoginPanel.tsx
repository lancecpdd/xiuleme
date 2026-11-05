import { useTranslation } from 'react-i18next';
import Text from '../ui/Text';
import Button from '../ui/Button';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { useAuthStore } from '../../store/useAuthStore';
import QrLoginPanel from './QrLoginPanel';

const LoginPanel = () => {
  const { t } = useTranslation();
  const enterAsGuest = useAuthStore((state) => state.enterAsGuest);

  return (
    <div className="my-auto text-left relative z-10 w-full">
      <Text variant={TextVariants.displayLarge}>{t('library.splash.brand')}</Text>
      <Text
        className="mb-8 max-w-md drop-shadow-sm"
        color={TextColors.secondary}
        variant={TextVariants.heading}
        weight={TextWeights.normal}
      >
        {t('auth.title')}
      </Text>
      <div className="flex flex-col items-start gap-4 max-w-xs">
        <QrLoginPanel />
        <Button
          className="rounded-md h-11 w-full bg-surface text-text-primary shadow-md transition-transform duration-200 hover:scale-[1.01] active:scale-[.98]"
          onClick={enterAsGuest}
          size="lg"
          type="button"
        >
          {t('auth.skipLogin')}
        </Button>
      </div>
    </div>
  );
};

export default LoginPanel;

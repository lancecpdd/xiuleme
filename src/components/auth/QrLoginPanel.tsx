import { useCallback, useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { QRCodeSVG } from 'qrcode.react';
import { Loader2, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { Invokes, QrLoginPollResult, QrLoginSession } from '../../components/ui/AppProperties';
import { useAuthStore } from '../../store/useAuthStore';

const QR_POLL_INTERVAL_MS = 2000;
const QR_LOGIN_PREFIX = 'xiuleme-login:';

type QrPanelStatus = 'loading' | 'ready' | 'expired' | 'canceled' | 'failed';

const QrLoginPanel = () => {
  const { t } = useTranslation();
  const applyQrLogin = useAuthStore((state) => state.applyQrLogin);
  const [ticket, setTicket] = useState<string | null>(null);
  const [status, setStatus] = useState<QrPanelStatus>('loading');

  const startSession = useCallback(async () => {
    setStatus('loading');
    setTicket(null);
    try {
      const session = await invoke<QrLoginSession>(Invokes.QrLoginCreateSession);
      setTicket(`${QR_LOGIN_PREFIX}${session.qrToken}`);
      setStatus('ready');
    } catch (err) {
      console.error('Failed to create QR login session:', err);
      setStatus('failed');
    }
  }, []);

  useEffect(() => {
    startSession();
  }, [startSession]);

  useEffect(() => {
    if (status !== 'ready' || !ticket) {
      return;
    }

    let cancelled = false;

    const timer = window.setInterval(async () => {
      const qrToken = ticket.slice(QR_LOGIN_PREFIX.length);
      try {
        const result = await invoke<QrLoginPollResult>(Invokes.QrLoginPoll, { qrToken });
        if (cancelled) {
          return;
        }

        if (result.status === 'CONFIRMED' && result.user) {
          window.clearInterval(timer);
          applyQrLogin(result.user);
        } else if (result.status === 'CANCELED') {
          window.clearInterval(timer);
          setStatus('canceled');
        } else if (result.status === 'EXPIRED') {
          window.clearInterval(timer);
          setStatus('expired');
        }
      } catch (err) {
        // 网络抖动时跳过本轮，下一轮继续轮询
        console.error('Failed to poll QR login status:', err);
      }
    }, QR_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [status, ticket, applyQrLogin]);

  const statusText =
    status === 'expired'
      ? t('auth.qrExpired')
      : status === 'canceled'
        ? t('auth.qrCanceled')
        : status === 'failed'
          ? t('auth.qrCreateFailed')
          : status === 'loading'
            ? t('auth.qrLoading')
            : t('auth.waitingScan');

  return (
    <div className="flex flex-col items-start w-full max-w-xs gap-4">
      <div className="bg-white p-3 rounded-lg shadow-md">
        <div className="relative">
          {ticket ? (
            <QRCodeSVG
              bgColor="#ffffff"
              fgColor="#111111"
              level="M"
              marginSize={1}
              size={176}
              title={t('auth.tabQr')}
              value={ticket}
              className={status === 'expired' || status === 'canceled' ? 'opacity-20' : undefined}
            />
          ) : (
            <div className="flex items-center justify-center w-[176px] h-[176px]">
              {status === 'failed' ? (
                <Button className="px-3 py-1.5 text-sm" onClick={startSession} type="button">
                  <RefreshCw size={16} />
                  {t('auth.retry')}
                </Button>
              ) : (
                <Loader2 className="animate-spin text-neutral-400" size={24} />
              )}
            </div>
          )}
          {status === 'expired' || status === 'canceled' ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <Button className="px-3 py-1.5 text-sm" onClick={startSession} type="button">
                <RefreshCw size={16} />
                {t('auth.refreshQr')}
              </Button>
            </div>
          ) : null}
        </div>
      </div>
      <Text color={TextColors.secondary} variant={TextVariants.small}>
        {statusText}
      </Text>
      <Text color={TextColors.secondary} variant={TextVariants.small}>
        {t('auth.qrHint')}
      </Text>
    </div>
  );
};

export default QrLoginPanel;

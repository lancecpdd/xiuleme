import i18n from 'i18next';

const PIXEL_ERROR_KEYS = {
  PIXEL_NOT_LOGGED_IN: 'export.pixel.notLoggedIn',
  PIXEL_SESSION_EXPIRED: 'export.pixel.sessionExpired',
} as const;

export const getExportErrorMessage = (error: unknown, fallback: string): string => {
  if (typeof error !== 'string') return fallback;

  const [code] = error.split(':');
  if (!Object.hasOwn(PIXEL_ERROR_KEYS, code)) return error;

  const key = PIXEL_ERROR_KEYS[code as keyof typeof PIXEL_ERROR_KEYS];
  return i18n.t(key);
};

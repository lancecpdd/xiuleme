import { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { RotateCcw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import {
  Adjustments,
  DEFAULT_LEVELS_CHANNEL,
  LevelsChannel,
  LevelsChannelSettings,
  LevelsSettings,
  getDefaultLevels,
  normalizeLevels,
  normalizeLevelsChannel,
} from '../../utils/adjustments';
import { Theme } from '../ui/AppProperties';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';

type DragHandle = 'inputBlack' | 'gamma' | 'inputWhite' | 'outputBlack' | 'outputWhite';

interface LevelsHistogram {
  red?: number[] | { data?: number[] };
  green?: number[] | { data?: number[] };
  blue?: number[] | { data?: number[] };
  luma?: number[] | { data?: number[] };
}

interface LevelsPanelProps {
  adjustments: Pick<Adjustments, 'levels'> & Record<string, unknown>;
  /** Pre-levels (identity levels) input histogram — Capture One style. */
  histogram: LevelsHistogram | null;
  setAdjustments(updater: (prev: Adjustments) => Adjustments): void;
  theme: string;
  onDragStateChange?: (isDragging: boolean) => void;
  isForMask?: boolean;
}

const asHistArray = (value: number[] | { data?: number[] } | undefined): number[] | undefined => {
  if (Array.isArray(value)) return value;
  if (value && Array.isArray(value.data)) return value.data;
  return undefined;
};

const CHANNELS: LevelsChannel[] = ['rgb', 'red', 'green', 'blue'];

const CHANNEL_COLORS: Record<LevelsChannel, string> = {
  rgb: 'var(--color-accent)',
  red: '#FF6B6B',
  green: '#6BCB77',
  blue: '#4D96FF',
};

function getHistogramPath(data: Array<number> | undefined) {
  if (!data || data.length === 0) return '';
  const maxVal = Math.max(...data);
  if (maxVal === 0) return '';
  const pathData = data
    .map((value: number, index: number) => `${(index / 255) * 255},${255 - (value / maxVal) * 255}`)
    .join(' L');
  return `M0,255 L${pathData} L255,255 Z`;
}

function getHistogramStrokePath(data: Array<number> | undefined) {
  if (!data || data.length === 0) return '';
  const maxVal = Math.max(...data);
  if (maxVal === 0) return '';
  return (
    'M' +
    data
      .map((value: number, index: number) => `${(index / 255) * 255},${255 - (value / maxVal) * 255}`)
      .join(' L')
  );
}

function getZeroHistogramPath(data: Array<number> | undefined) {
  if (!data || data.length === 0) return '';
  const pathData = data.map((_, index: number) => `${(index / 255) * 255},255`).join(' L');
  return `M0,255 L${pathData} L255,255 Z`;
}

const HIST_FILL_OPACITY = 0.4;
const HANDLE_BLACK = '#111111';
const HANDLE_WHITE = '#ffffff';
const HANDLE_STROKE = 'rgba(255,255,255,0.55)';

/** Home-plate marker pointing toward the histogram (top → down, bottom → up). */
function LevelsHandleMark({
  tone,
  pointing,
}: {
  tone: 'black' | 'white' | 'mid';
  pointing: 'down' | 'up';
}) {
  const fill = tone === 'black' ? HANDLE_BLACK : tone === 'white' ? HANDLE_WHITE : '#c8c8c8';
  const stroke = tone === 'white' ? 'rgba(0,0,0,0.35)' : HANDLE_STROKE;
  // Flat edge away from histogram, tip toward histogram.
  const points =
    pointing === 'down' ? '2,1.5 14,1.5 14,10 8,16.5 2,10' : '2,8 8,1.5 14,8 14,16.5 2,16.5';
  return (
    <svg
      width="16"
      height="18"
      viewBox="0 0 16 18"
      className="block drop-shadow-sm group-hover:scale-110 transition-transform"
      aria-hidden="true"
    >
      <polygon points={points} fill={fill} stroke={stroke} strokeWidth="1.1" strokeLinejoin="round" />
    </svg>
  );
}

/** Classic levels midtone: gamma=1 → midpoint 0.5 (center). */
function gammaToPosition(gamma: number, black: number, white: number): number {
  const range = Math.max(1, white - black);
  const g = Math.min(9.99, Math.max(0.1, gamma));
  const t = Math.min(0.98, Math.max(0.02, Math.pow(0.5, 1 / g)));
  return black + t * range;
}

function positionToGamma(x: number, black: number, white: number): number {
  const range = Math.max(1, white - black);
  const t = Math.min(0.98, Math.max(0.02, (x - black) / range));
  const gamma = Math.log(0.5) / Math.log(t);
  if (!Number.isFinite(gamma)) return 1;
  return Math.min(9.99, Math.max(0.1, Math.round(gamma * 100) / 100));
}

function pct(x: number) {
  return `${(x / 255) * 100}%`;
}

export default function LevelsPanel({
  adjustments,
  histogram,
  setAdjustments,
  theme,
  onDragStateChange,
}: LevelsPanelProps) {
  const { t } = useTranslation();
  const [activeChannel, setActiveChannel] = useState<LevelsChannel>('rgb');
  const [dragging, setDragging] = useState<DragHandle | null>(null);
  const [localSettings, setLocalSettings] = useState<LevelsChannelSettings | null>(null);

  const trackRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef<DragHandle | null>(null);
  const localSettingsRef = useRef<LevelsChannelSettings | null>(null);
  const activeChannelRef = useRef(activeChannel);
  const lastHistogramRef = useRef<LevelsHistogram | null>(null);
  const histogramEverShownRef = useRef(false);

  if (histogram) lastHistogramRef.current = histogram;
  const displayHistogram = histogram ?? lastHistogramRef.current;

  const levels: LevelsSettings = useMemo(
    () => normalizeLevels(adjustments?.levels),
    [adjustments?.levels],
  );
  const activeSettings = localSettings ?? levels[activeChannel];
  const accent = CHANNEL_COLORS[activeChannel];
  const isLightTheme = theme === Theme.Light || theme === Theme.Arctic;
  const linkStroke = isLightTheme ? 'rgba(0,0,0,0.35)' : 'rgba(255,255,255,0.55)';

  useEffect(() => {
    activeChannelRef.current = activeChannel;
    setLocalSettings(null);
    localSettingsRef.current = null;
    setDragging(null);
    draggingRef.current = null;
  }, [activeChannel]);

  useEffect(() => {
    onDragStateChange?.(dragging !== null);
    draggingRef.current = dragging;
  }, [dragging, onDragStateChange]);

  const commitSettings = (channel: LevelsChannel, settings: LevelsChannelSettings) => {
    const next = normalizeLevelsChannel(settings);
    localSettingsRef.current = next;
    setLocalSettings(next);
    setAdjustments((prev) => {
      const prevLevels = normalizeLevels(prev.levels);
      return {
        ...prev,
        levels: {
          ...prevLevels,
          [channel]: next,
        },
      };
    });
  };

  useEffect(() => {
    const handleMove = (e: MouseEvent | TouchEvent) => {
      const handle = draggingRef.current;
      const current = localSettingsRef.current;
      if (!handle || !current) return;
      if ('touches' in e && e.touches.length === 0) return;

      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      const track = trackRef.current;
      if (!track) return;

      const rect = track.getBoundingClientRect();
      const x = Math.max(0, Math.min(255, ((clientX - rect.left) / Math.max(rect.width, 1)) * 255));
      const next = { ...current };

      if (handle === 'inputBlack') {
        next.inputBlack = Math.min(Math.round(x), next.inputWhite - 1);
      } else if (handle === 'inputWhite') {
        next.inputWhite = Math.max(Math.round(x), next.inputBlack + 1);
      } else if (handle === 'gamma') {
        // Bottom mid handle only — map within output black/white range.
        next.gamma = positionToGamma(x, next.outputBlack, next.outputWhite);
      } else if (handle === 'outputBlack') {
        next.outputBlack = Math.min(Math.round(x), next.outputWhite - 1);
      } else if (handle === 'outputWhite') {
        next.outputWhite = Math.max(Math.round(x), next.outputBlack + 1);
      }

      commitSettings(activeChannelRef.current, next);
      if (e.cancelable) e.preventDefault();
    };

    const handleUp = () => {
      setDragging(null);
      draggingRef.current = null;
      localSettingsRef.current = null;
      setLocalSettings(null);
      onDragStateChange?.(false);
    };

    if (dragging !== null) {
      window.addEventListener('mousemove', handleMove, { passive: false });
      window.addEventListener('mouseup', handleUp);
      window.addEventListener('touchmove', handleMove, { passive: false });
      window.addEventListener('touchend', handleUp);
      window.addEventListener('touchcancel', handleUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
      window.removeEventListener('touchmove', handleMove);
      window.removeEventListener('touchend', handleUp);
      window.removeEventListener('touchcancel', handleUp);
    };
  }, [dragging, onDragStateChange, setAdjustments]);

  const startDrag = (handle: DragHandle, e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    e.stopPropagation();
    localSettingsRef.current = { ...activeSettings };
    setLocalSettings({ ...activeSettings });
    setDragging(handle);
    draggingRef.current = handle;
    onDragStateChange?.(true);
  };

  const updateField = (key: keyof LevelsChannelSettings, raw: string) => {
    const parsed = parseFloat(raw);
    if (!Number.isFinite(parsed)) return;
    commitSettings(activeChannel, { ...activeSettings, [key]: parsed });
  };

  const resetField = (key: keyof LevelsChannelSettings) => {
    commitSettings(activeChannel, { ...activeSettings, [key]: DEFAULT_LEVELS_CHANNEL[key] });
  };

  const resetChannel = () => {
    commitSettings(activeChannel, { ...DEFAULT_LEVELS_CHANNEL });
  };

  const resetAll = () => {
    setAdjustments((prev) => ({
      ...prev,
      levels: getDefaultLevels(),
    }));
  };

  // Top mid handle is a fixed anchor at the input black/white midpoint (not tied to gamma).
  const gammaTopFixedX = (activeSettings.inputBlack + activeSettings.inputWhite) / 2;
  const gammaBottomX = gammaToPosition(
    activeSettings.gamma,
    activeSettings.outputBlack,
    activeSettings.outputWhite,
  );

  const histRed = asHistArray(displayHistogram?.red);
  const histGreen = asHistArray(displayHistogram?.green);
  const histBlue = asHistArray(displayHistogram?.blue);
  const histLuma = asHistArray(displayHistogram?.luma);

  const activeHist =
    activeChannel === 'rgb'
      ? histLuma
      : activeChannel === 'red'
        ? histRed
        : activeChannel === 'green'
          ? histGreen
          : histBlue;

  const isFirstHistogramPaint = !!activeHist && !histogramEverShownRef.current;
  useEffect(() => {
    if (activeHist) histogramEverShownRef.current = true;
  }, [activeHist]);

  const numberInputClass =
    'w-12 min-w-12 shrink-0 px-1 py-0.5 rounded bg-surface text-text-primary text-xs text-center border border-border-color focus:border-accent outline-none tabular-nums [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none';

  return (
    // Cancel CollapsibleSection content `px-4` so the plot matches global histogram width.
    <div className="select-none touch-none -mx-4">
      <div className="flex items-center justify-between gap-2 mb-2 mt-2 px-4">
        <button
          type="button"
          className="w-8 h-8 rounded-md flex items-center justify-center text-text-secondary hover:text-text-primary hover:bg-surface-secondary transition-colors"
          onClick={resetChannel}
          onDoubleClick={resetAll}
          data-tooltip={t('adjustments.levels.resetChannel')}
          aria-label={t('adjustments.levels.resetChannel')}
        >
          <RotateCcw size={16} />
        </button>
        <div className="flex items-center gap-1 shrink-0">
          {CHANNELS.map((channel) => {
            const selected = activeChannel === channel;
            const label = t(`adjustments.levels.channels.${channel}`);
            return (
              <button
                key={channel}
                type="button"
                className={`w-7 h-7 rounded-full flex items-center justify-center transition-all ${
                  selected ? 'ring-2 ring-offset-2 ring-offset-surface ring-accent' : 'bg-surface-secondary'
                }`}
                onClick={() => setActiveChannel(channel)}
                style={{
                  backgroundColor: channel !== 'rgb' && !selected ? CHANNEL_COLORS[channel] + '40' : undefined,
                }}
                title={t('adjustments.levels.channelTitle', { channel: label })}
                aria-label={t('adjustments.levels.channelTitle', { channel: label })}
              >
                <Text variant={TextVariants.small} color={TextColors.primary} weight={TextWeights.bold}>
                  {channel === 'rgb' ? 'RGB' : label.charAt(0).toUpperCase()}
                </Text>
              </button>
            );
          })}
        </div>
      </div>

      {/* Corner inputs: top row */}
      <div className="flex items-center justify-between mb-1 px-0.5">
        <input
          type="number"
          className={numberInputClass}
          min={0}
          max={254}
          step={1}
          value={activeSettings.inputBlack}
          aria-label={t('adjustments.levels.inputBlack')}
          onChange={(e) => updateField('inputBlack', e.target.value)}
          onDoubleClick={() => resetField('inputBlack')}
        />
        <input
          type="number"
          className={numberInputClass}
          min={1}
          max={255}
          step={1}
          value={activeSettings.inputWhite}
          aria-label={t('adjustments.levels.inputWhite')}
          onChange={(e) => updateField('inputWhite', e.target.value)}
          onDoubleClick={() => resetField('inputWhite')}
        />
      </div>

      {/* Top handles sit above the histogram (outside the plot) */}
      <div className="relative h-5 mb-0.5 touch-none">
        {(
          [
            {
              key: 'inputBlack' as const,
              x: activeSettings.inputBlack,
              interactive: true,
              tone: 'black' as const,
            },
            { key: 'gamma' as const, x: gammaTopFixedX, interactive: false, tone: 'mid' as const },
            {
              key: 'inputWhite' as const,
              x: activeSettings.inputWhite,
              interactive: true,
              tone: 'white' as const,
            },
          ] as const
        ).map(({ key, x, interactive, tone }) =>
          interactive ? (
            <button
              key={`top-${key}`}
              type="button"
              aria-label={t(`adjustments.levels.${key}`)}
              className="absolute top-1/2 w-7 h-7 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize z-20 group flex items-center justify-center"
              style={{ left: pct(x) }}
              onMouseDown={(e) => startDrag(key, e)}
              onTouchStart={(e) => startDrag(key, e)}
              onDoubleClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                resetField(key);
              }}
            >
              <LevelsHandleMark tone={tone} pointing="down" />
            </button>
          ) : (
            <div
              key={`top-${key}`}
              aria-hidden="true"
              className="absolute top-1/2 w-7 h-7 -translate-x-1/2 -translate-y-1/2 z-10 pointer-events-none flex items-center justify-center"
              style={{ left: pct(x) }}
            >
              <span
                className="block w-3.5 h-3.5 rounded-full border border-black/30 shadow-sm"
                style={{ backgroundColor: HANDLE_WHITE }}
              />
            </div>
          ),
        )}
      </div>

      <div
        ref={trackRef}
        className="relative w-full aspect-[5/3] bg-surface-secondary rounded-md touch-none overflow-hidden"
      >
        <svg viewBox="0 0 255 255" className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
          <path
            d="M 0,63.75 H 255 M 0,127.5 H 255 M 0,191.25 H 255"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="0.5"
            fill="none"
          />
          {activeChannel === 'rgb' ? (
            (
              [
                { key: 'red', color: '#FF6B6B', data: histRed },
                { key: 'green', color: '#6BCB77', data: histGreen },
                { key: 'blue', color: '#4D96FF', data: histBlue },
              ] as const
            ).map(
              (ch) =>
                ch.data && (
                  <g key={ch.key} style={{ mixBlendMode: 'lighten' }}>
                    <path d={getHistogramPath(ch.data)} fill={ch.color} fillOpacity={HIST_FILL_OPACITY} />
                    <path
                      d={getHistogramStrokePath(ch.data)}
                      fill="none"
                      stroke={ch.color}
                      strokeWidth={1.5}
                      strokeOpacity={1}
                      vectorEffect="non-scaling-stroke"
                      strokeLinejoin="round"
                    />
                  </g>
                ),
            )
          ) : (
            activeHist && (
              <motion.g
                key={activeChannel}
                initial={isFirstHistogramPaint ? { opacity: 0 } : false}
                animate={{ opacity: 1 }}
                transition={{ opacity: { duration: 0.2 } }}
              >
                <motion.path
                  fill={accent}
                  fillOpacity={HIST_FILL_OPACITY}
                  initial={isFirstHistogramPaint ? { d: getZeroHistogramPath(activeHist) } : false}
                  animate={{ d: getHistogramPath(activeHist) }}
                  transition={{ d: { duration: 0.35, ease: [0.22, 1, 0.36, 1] } }}
                />
                <path
                  d={getHistogramStrokePath(activeHist)}
                  fill="none"
                  stroke={accent}
                  strokeWidth={1.5}
                  strokeOpacity={1}
                  vectorEffect="non-scaling-stroke"
                  strokeLinejoin="round"
                />
              </motion.g>
            )
          )}

          {(
            [
              [activeSettings.inputBlack, activeSettings.outputBlack],
              [gammaTopFixedX, gammaBottomX],
              [activeSettings.inputWhite, activeSettings.outputWhite],
            ] as const
          ).map(([top, bottom], i) => (
            <line
              key={i}
              x1={top}
              y1={0}
              x2={bottom}
              y2={255}
              stroke={linkStroke}
              strokeWidth="1.25"
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      </div>

      {/* Bottom handles sit below the histogram (outside the plot) */}
      <div className="relative h-5 mt-0.5 touch-none">
        {(
          [
            { key: 'outputBlack' as const, x: activeSettings.outputBlack, tone: 'black' as const },
            { key: 'gamma' as const, x: gammaBottomX, tone: 'mid' as const },
            { key: 'outputWhite' as const, x: activeSettings.outputWhite, tone: 'white' as const },
          ] as const
        ).map(({ key, x, tone }) => (
          <button
            key={`bottom-${key}`}
            type="button"
            aria-label={t(`adjustments.levels.${key}`)}
            className="absolute top-1/2 w-7 h-7 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize z-20 group flex items-center justify-center"
            style={{ left: pct(x) }}
            onMouseDown={(e) => startDrag(key, e)}
            onTouchStart={(e) => startDrag(key, e)}
            onDoubleClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              resetField(key);
            }}
          >
            <LevelsHandleMark tone={tone} pointing="up" />
          </button>
        ))}
      </div>

      {/* Corner inputs: bottom row + gamma center */}
      <div className="flex items-start justify-between mt-1 px-0.5 gap-2">
        {(
          [
            {
              key: 'outputBlack' as const,
              labelKey: 'blackPoint' as const,
              value: activeSettings.outputBlack,
              min: 0,
              max: 254,
              step: 1,
              className: numberInputClass,
            },
            {
              key: 'gamma' as const,
              labelKey: 'grayPoint' as const,
              value: Number(activeSettings.gamma.toFixed(2)),
              min: 0.1,
              max: 9.99,
              step: 0.01,
              className: `${numberInputClass} w-14`,
            },
            {
              key: 'outputWhite' as const,
              labelKey: 'whitePoint' as const,
              value: activeSettings.outputWhite,
              min: 1,
              max: 255,
              step: 1,
              className: numberInputClass,
            },
          ] as const
        ).map((field) => (
          <div key={field.key} className="flex flex-col items-center gap-0.5">
            <input
              type="number"
              className={field.className}
              min={field.min}
              max={field.max}
              step={field.step}
              value={field.value}
              aria-label={t(`adjustments.levels.${field.key}`)}
              onChange={(e) => updateField(field.key, e.target.value)}
              onDoubleClick={() => resetField(field.key)}
            />
            <Text variant={TextVariants.small} color={TextColors.secondary}>
              {t(`adjustments.levels.${field.labelKey}`)}
            </Text>
          </div>
        ))}
      </div>
    </div>
  );
}

import { useMemo, useState } from 'react';
import { RotateCcw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useShallow } from 'zustand/react/shallow';
import CollapsibleSection from '../../ui/CollapsibleSection';
import Switch from '../../ui/Switch';
import Text from '../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { useEditorStore } from '../../../store/useEditorStore';
import { useEditorActions, debouncedSave } from '../../../hooks/useEditorActions';
import {
  defaultFrame,
  detectBrand,
  exifFromXiuLeMe,
  formatExifDate,
  type FrameExif,
  type FrameState,
} from '../../../utils/frame';
import { INITIAL_ADJUSTMENTS } from '../../../utils/adjustments';

type FrameToggleKey = Exclude<keyof FrameState, 'enabled'>;

const TOGGLE_KEYS: FrameToggleKey[] = ['showParams', 'showDate', 'showLogo', 'showCamera', 'showLens'];

function paramsPreview(exif: FrameExif): string | null {
  const parts: string[] = [];
  if (exif.iso != null) parts.push(`ISO ${exif.iso}`);
  if (exif.aperture != null) parts.push(`f/${exif.aperture}`);
  if (exif.shutterSpeed) parts.push(`${exif.shutterSpeed}s`);
  return parts.length ? parts.join(' ') : null;
}

function valueOf(key: FrameToggleKey, exif: FrameExif): string | null {
  switch (key) {
    case 'showParams':
      return paramsPreview(exif);
    case 'showDate':
      return formatExifDate(exif.dateTime);
    case 'showLogo': {
      const brand = detectBrand(exif.cameraMake, exif.cameraModel);
      if (brand) return 'matched';
      if (exif.cameraMake) return 'fallback';
      return null;
    }
    case 'showCamera':
      return exif.cameraModel ?? exif.cameraMake ?? null;
    case 'showLens':
      return exif.lensModel;
    default:
      return null;
  }
}

export default function FramePanel() {
  const { t } = useTranslation();
  const { setAdjustments } = useEditorActions();
  const [isOpen, setIsOpen] = useState(true);
  const { selectedImage, frame } = useEditorStore(
    useShallow((s) => ({
      selectedImage: s.selectedImage,
      frame: s.adjustments.frame ?? INITIAL_ADJUSTMENTS.frame,
    })),
  );

  const exif = useMemo(() => exifFromXiuLeMe(selectedImage?.exif), [selectedImage?.exif]);

  const defaults = defaultFrame();
  const isPristine = (Object.keys(defaults) as Array<keyof FrameState>).every((k) => frame[k] === defaults[k]);

  const patchFrame = (patch: Partial<FrameState>) => {
    const path = selectedImage?.path;
    const prev = useEditorStore.getState().adjustments;
    const next = { ...prev, frame: { ...(prev.frame ?? defaultFrame()), ...patch } };
    setAdjustments(next);
    if (path) debouncedSave(path, next);
  };

  const resetFrame = () => {
    const path = selectedImage?.path;
    const prev = useEditorStore.getState().adjustments;
    const next = { ...prev, frame: defaultFrame() };
    setAdjustments(next);
    if (path) debouncedSave(path, next);
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto p-2 gap-2">
      <CollapsibleSection
        title={t('editor.frame.title')}
        isOpen={isOpen}
        isContentVisible
        canToggleVisibility={false}
        onToggle={() => setIsOpen((v) => !v)}
        headerStatus={
          <button
            type="button"
            className="p-1 rounded-md text-text-secondary hover:text-text-primary hover:bg-surface disabled:opacity-40 disabled:pointer-events-none"
            title={t('editor.frame.reset')}
            disabled={isPristine}
            onClick={(e) => {
              e.stopPropagation();
              resetFrame();
            }}
          >
            <RotateCcw size={14} />
          </button>
        }
      >
        <div className="flex flex-col gap-3 pt-1">
          <Text variant={TextVariants.small} color={TextColors.secondary}>
            {t('editor.frame.hint')}
          </Text>

          <Switch
            label={t('editor.frame.enabled')}
            checked={frame.enabled}
            onChange={(checked) => patchFrame({ enabled: checked })}
            tooltip={t('editor.frame.enabledTooltip')}
          />

          <Text variant={TextVariants.label} weight={TextWeights.medium} className="mt-1">
            {t('editor.frame.showContent')}
          </Text>

          {TOGGLE_KEYS.map((key) => {
            const raw = valueOf(key, exif);
            const display =
              key === 'showLogo'
                ? raw === 'matched'
                  ? t('editor.frame.logoMatched')
                  : raw === 'fallback'
                    ? t('editor.frame.logoFallback')
                    : '--'
                : (raw ?? '--');
            return (
              <div key={key} className="flex items-center gap-2 min-w-0">
                <div className="flex-1 min-w-0">
                  <Switch
                    label={t(`editor.frame.${key}`)}
                    checked={frame[key]}
                    disabled={!frame.enabled}
                    onChange={(checked) => patchFrame({ [key]: checked })}
                    tooltip={t(`editor.frame.${key}Tooltip`)}
                  />
                </div>
                <span
                  className="shrink-0 max-w-[40%] truncate text-[11px] text-text-tertiary"
                  title={display === '--' ? undefined : display}
                >
                  {display}
                </span>
              </div>
            );
          })}
        </div>
      </CollapsibleSection>
    </div>
  );
}

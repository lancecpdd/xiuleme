import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, RotateCcw, ScanFace } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';
import { useShallow } from 'zustand/react/shallow';
import CollapsibleSection from '../../ui/CollapsibleSection';
import Slider from '../../ui/Slider';
import Text from '../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';
import {
  isPortraitPreviewBusy,
  useEditorStore,
  type PortraitReshapeStatus,
} from '../../../store/useEditorStore';
import { useUIStore } from '../../../store/useUIStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { debouncedSetHistory, useEditorActions } from '../../../hooks/useEditorActions';
import {
  EYES_KEYS,
  INITIAL_ADJUSTMENTS,
  INITIAL_PORTRAIT_ADJUSTMENTS,
  UV_RESHAPE_KEYS,
  normalizePortraitAdjustments,
  hasPortraitEffect,
  PORTRAIT_KEYS,
  PORTRAIT_LABELS,
  PORTRAIT_SECTIONS,
  portraitSectionKeys,
  readPortraitReshape,
  type EyesKey,
  type PortraitAdjustments,
  type PortraitSection,
  type PortraitVisibilityKey,
  type UvReshapeKey,
} from '../../../utils/adjustments';
import {
  PORTRAIT_RESHAPE_REASON_LABELS,
  activateLegacyReshape,
  applyReshapeSliderChange,
  clearFaceOverrides,
  decideReshapePanel,
  mirrorRuntimeStrengths,
} from '../../../utils/portraitReshapePanel';

type FaceState = 'detecting' | 'detected' | 'noFace';

const APPLIED_FLASH_MS = 1200;

const VISIBILITY_SECTION: Record<PortraitVisibilityKey, PortraitSection> = {
  portraitSkin: 'skin',
  portraitBlemishes: 'blemishes',
  portraitMakeup: 'makeup',
  portraitReshape: 'reshape',
  portraitEyes: 'eyes',
  portraitBody: 'body',
};

const sectionForPortraitKeys = (keys: ReadonlyArray<string>): PortraitSection | null => {
  const matched = PORTRAIT_SECTIONS.filter((section) =>
    keys.some((key) => section.keys.includes(key as keyof PortraitAdjustments)),
  );
  return matched.length === 1 ? matched[0].id : null;
};

const FACE_STATE_STYLES: Record<FaceState, { icon: string; text: string }> = {
  detecting: { icon: 'text-text-secondary animate-pulse', text: 'text-text-secondary' },
  detected: { icon: 'text-green-500', text: 'text-green-500' },
  noFace: { icon: 'text-red-500', text: 'text-red-500' },
};

export default function PortraitPanel() {
  const { t } = useTranslation();
  const { setAdjustments } = useEditorActions();
  const {
    selectedImage,
    portrait,
    portraitReshape,
    portraitStatus,
    portraitReshapeStatus,
    skinVisible,
    blemishesVisible,
    makeupVisible,
    reshapeVisible,
    eyesVisible,
    bodyVisible,
    effectEnabled,
    previewBusy,
    previewError,
    isSliderDragging,
    setEditor,
    selectPortraitReshapeFace,
  } = useEditorStore(
    useShallow((state) => ({
      selectedImage: state.selectedImage,
      portrait: state.adjustments.portrait,
      portraitReshape: state.adjustments.portraitReshape,
      portraitStatus: state.portraitStatus,
      portraitReshapeStatus: state.portraitReshapeStatus,
      skinVisible: state.adjustments.sectionVisibility?.portraitSkin ?? true,
      blemishesVisible: state.adjustments.sectionVisibility?.portraitBlemishes ?? true,
      makeupVisible: state.adjustments.sectionVisibility?.portraitMakeup !== false,
      reshapeVisible: state.adjustments.sectionVisibility?.portraitReshape !== false,
      eyesVisible: state.adjustments.sectionVisibility?.portraitEyes !== false,
      bodyVisible: state.adjustments.sectionVisibility?.portraitBody !== false,
      effectEnabled: hasPortraitEffect(state.adjustments),
      previewBusy: isPortraitPreviewBusy(state),
      previewError: state.previewError,
      isSliderDragging: state.isSliderDragging,
      setEditor: state.setEditor,
      selectPortraitReshapeFace: state.selectPortraitReshapeFace,
    })),
  );
  const { portraitSectionsState, setUI } = useUIStore(
    useShallow((state) => ({ portraitSectionsState: state.portraitSectionsState, setUI: state.setUI })),
  );
  const enableFocusMode = useSettingsStore((state) => state.appSettings?.enableFocusMode);
  const values = normalizePortraitAdjustments(portrait);
  const isReady = !!selectedImage?.isReady;
  const hasChanges = (keys: ReadonlyArray<keyof PortraitAdjustments>) =>
    keys.some((key) => values[key] !== INITIAL_PORTRAIT_ADJUSTMENTS[key]);

  const faceStatus = useMemo<{ state: FaceState; label: string } | null>(() => {
    if (!isReady) return null;
    const isCurrent = !!selectedImage && portraitStatus.path === selectedImage.path;
    const phase = isCurrent ? portraitStatus.phase : 'idle';
    if (phase === 'idle' || phase === 'loading' || phase === 'analyzing') {
      return { state: 'detecting', label: t('editor.portrait.faceDetecting') };
    }
    if (phase === 'error') {
      const count = portraitStatus.summary?.faceCount ?? 0;
      return count > 0
        ? { state: 'detected', label: t('editor.portrait.faceDetected', { count }) }
        : { state: 'noFace', label: t('editor.portrait.noFaceDetected') };
    }
    const count = portraitStatus.summary?.faceCount ?? 0;
    return count > 0
      ? { state: 'detected', label: t('editor.portrait.faceDetected', { count }) }
      : { state: 'noFace', label: t('editor.portrait.noFaceDetected') };
  }, [isReady, selectedImage, portraitStatus, t]);

  const analysisFailed =
    !!selectedImage &&
    portraitStatus.path === selectedImage.path &&
    portraitStatus.phase === 'error';
  const faceBlocked = faceStatus?.state !== 'detected' || analysisFailed;

  const [applySection, setApplySection] = useState<PortraitSection | null>(null);
  const [showApplied, setShowApplied] = useState(false);
  const wasPreviewBusyRef = useRef(false);
  const markApplySection = useCallback((section: PortraitSection | null) => {
    setApplySection(section);
    setShowApplied(false);
  }, []);

  const toggleVisibility = (key: PortraitVisibilityKey) => {
    markApplySection(VISIBILITY_SECTION[key]);
    setAdjustments((prev) => ({
      ...prev,
      sectionVisibility: {
        ...(prev.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility),
        [key]: !(prev.sectionVisibility?.[key] ?? true),
      },
    }));
  };

  const sectionContentVisible = (sectionId: PortraitSection): boolean => {
    if (sectionId === 'skin') return skinVisible;
    if (sectionId === 'blemishes') return blemishesVisible;
    if (sectionId === 'makeup') return makeupVisible;
    if (sectionId === 'reshape') return reshapeVisible;
    if (sectionId === 'body') return bodyVisible;
    return eyesVisible;
  };

  const sectionVisibilityKey = (sectionId: PortraitSection): PortraitVisibilityKey => {
    if (sectionId === 'skin') return 'portraitSkin';
    if (sectionId === 'blemishes') return 'portraitBlemishes';
    if (sectionId === 'makeup') return 'portraitMakeup';
    if (sectionId === 'reshape') return 'portraitReshape';
    if (sectionId === 'body') return 'portraitBody';
    return 'portraitEyes';
  };

  const onDragStateChange = useCallback(
    (isDragging: boolean) => setEditor({ isSliderDragging: isDragging }),
    [setEditor],
  );

  useEffect(() => () => setEditor({ isSliderDragging: false }), [setEditor]);

  const updateValues = (changes: Partial<PortraitAdjustments>) => {
    // 滑块已通过 disabled={faceBlocked} 拦截交互；重置按钮仍可归零，因此这里只判断图像就绪
    if (!isReady) return;
    const current = normalizePortraitAdjustments(portrait);
    const next = normalizePortraitAdjustments({ ...current, ...changes });
    if (PORTRAIT_KEYS.every((key) => current[key] === next[key])) return;
    markApplySection(sectionForPortraitKeys(Object.keys(changes)));
    setAdjustments((prev) => ({ ...prev, portrait: next }));
  };

  // —— 09：重塑分组专用交互 ————————————————————————————————————————
  const reshapeRead = useMemo(() => readPortraitReshape(portraitReshape), [portraitReshape]);
  const reshapeStatus = portraitReshapeStatus as Pick<
    PortraitReshapeStatus,
    'path' | 'phase' | 'faces' | 'selectedFaceId'
  >;
  const decision = decideReshapePanel({
    reshapeRead,
    portrait,
    status: reshapeStatus,
    selectedPath: selectedImage?.path ?? null,
  });
  const analysisFaces = reshapeStatus.path === selectedImage?.path ? reshapeStatus.faces : [];
  const reshapePhase = reshapeStatus.path === selectedImage?.path ? reshapeStatus.phase : 'idle';
  const reshapeAnalyzing = reshapePhase === 'loading' || reshapePhase === 'analyzing';
  const reshapeSlidersDisabled = !isReady || !decision.editable;
  const showProtectionLimited =
    decision.editable &&
    (analysisFaces.some((face) => face.reason === 'protection-limited' && face.processable) ||
      portraitReshapeStatus.renderAttenuation === 'limited' ||
      portraitReshapeStatus.renderAttenuation === 'identity');
  const bodyAmount =
    values.bodySlimArms + values.bodySlimWaist + values.bodySlimLegs;
  const showBodyProtectionLimited =
    bodyAmount > 0 &&
    selectedImage?.path != null &&
    (portraitReshapeStatus.bodyAttenuation === 'limited' ||
      portraitReshapeStatus.bodyAttenuation === 'identity');

  const analysisStatusActive =
    portraitStatus.path === selectedImage?.path &&
    portraitStatus.phase !== 'idle' &&
    portraitStatus.phase !== 'ready' &&
    (effectEnabled || portraitStatus.phase === 'error');

  useEffect(() => {
    // 预览失败：立刻撤下分区 applying／applied 角标，避免与 Editor 的「正在应用面部重塑」叠死。
    if (previewError) {
      wasPreviewBusyRef.current = false;
      setShowApplied(false);
      setApplySection(null);
      return;
    }
    if (previewBusy) {
      wasPreviewBusyRef.current = true;
      setShowApplied(false);
      return;
    }
    if (!wasPreviewBusyRef.current) return;
    if (isSliderDragging) return;
    wasPreviewBusyRef.current = false;
    if (!applySection) {
      setShowApplied(false);
      return;
    }
    setShowApplied(true);
    const timer = setTimeout(() => {
      setShowApplied(false);
      setApplySection(null);
    }, APPLIED_FLASH_MS);
    return () => clearTimeout(timer);
  }, [previewBusy, previewError, isSliderDragging, applySection]);

  const headerBusy = analysisStatusActive || reshapeAnalyzing;
  const showStatus = analysisStatusActive || reshapeAnalyzing;

  useEffect(() => {
    const path = selectedImage?.path ?? null;
    let next: {
      path: string;
      kind: 'reshape' | 'effects';
      phase: 'applying' | 'applied';
    } | null = null;
    if (path && applySection && !isSliderDragging) {
      const kind = applySection === 'reshape' ? ('reshape' as const) : ('effects' as const);
      if (previewBusy) next = { path, kind, phase: 'applying' };
      else if (showApplied) next = { path, kind, phase: 'applied' };
    }
    const prev = useEditorStore.getState().portraitApplyHint;
    const same =
      (prev === null && next === null) ||
      (!!prev &&
        !!next &&
        prev.path === next.path &&
        prev.kind === next.kind &&
        prev.phase === next.phase);
    if (!same) setEditor({ portraitApplyHint: next });
  }, [
    selectedImage?.path,
    applySection,
    isSliderDragging,
    previewBusy,
    showApplied,
    setEditor,
  ]);

  useEffect(() => {
    return () => {
      setEditor({ portraitApplyHint: null });
    };
  }, [setEditor]);

  const commitReshape = (
    updater: (
      prev: ReturnType<typeof useEditorStore.getState>['adjustments'],
    ) => ReturnType<typeof useEditorStore.getState>['adjustments'] | null,
  ) => {
    setAdjustments((prev) => {
      const next = updater(prev);
      return next ?? prev;
    });
  };

  const updateReshapeValues = (changes: Partial<Record<UvReshapeKey | EyesKey, number>>) => {
    if (reshapeSlidersDisabled) return;
    markApplySection(sectionForPortraitKeys(Object.keys(changes)));
    commitReshape((prev) =>
      applyReshapeSliderChange(prev, changes, {
        selectedFaceId: decision.selectedFace?.faceId ?? null,
        faces: analysisFaces,
      }),
    );
  };

  const resetReshapeValues = () => {
    if (reshapeSlidersDisabled || !decision.selectedFace) return;
    const next = clearFaceOverrides(portraitReshape, decision.selectedFace.faceId);
    if (next) {
      markApplySection('reshape');
      commitReshape((prev) => ({
        ...prev,
        portrait: mirrorRuntimeStrengths(prev.portrait, next.faces),
        portraitReshape: next,
      }));
    }
    debouncedSetHistory.flush();
  };

  const resetEyesValues = () => {
    if (!reshapeSlidersDisabled && decision.selectedFace && decision.uvValues.bigEye !== 0) {
      updateReshapeValues({ bigEye: 0 });
    }
    debouncedSetHistory.flush();
  };

  const activateLegacy = () => {
    if (!isReady) return;
    markApplySection('reshape');
    commitReshape((prev) => activateLegacyReshape(prev));
    debouncedSetHistory.flush();
  };

  const reshapeSectionMessage = () => {
    if (decision.activation === 'legacy-inactive')
      return (
        <div className="text-xs text-text-secondary leading-relaxed" role="status">
          <p>{t('editor.portrait.reshape.activateLegacyHint')}</p>
          <button
            type="button"
            className="mt-1 px-2 py-1 rounded-md hover:bg-surface focus-visible:outline-accent"
            onClick={activateLegacy}
          >
            {t('editor.portrait.reshape.activateLegacy')}
          </button>
        </div>
      );
    if (decision.activation === 'unsupported')
      return (
        <p className="text-xs text-text-secondary leading-relaxed" role="status">
          {t('editor.portrait.reshape.unsupportedData')}
        </p>
      );
    if (decision.activation === 'invalid')
      return (
        <p className="text-xs text-red-500 leading-relaxed" role="status">
          {t('editor.portrait.reshape.invalidData')}
        </p>
      );
    if (reshapeAnalyzing)
      return (
        <div className="flex items-start gap-1.5 text-xs text-text-secondary leading-relaxed" role="status">
          <Loader2 size={14} className="animate-spin shrink-0 mt-0.5" aria-hidden="true" />
          <p>{t('editor.portrait.reshape.analyzing')}</p>
        </div>
      );
    if (reshapePhase === 'error')
      return (
        <div className="text-xs text-text-secondary leading-relaxed" role="status">
          <p>{t('editor.portrait.reshape.analysisFailed')}</p>
          {reshapeStatus.path === selectedImage?.path && 'error' in reshapeStatus && reshapeStatus.error ? (
            <p className="break-words">
              {t('editor.portrait.reshape.analysisFailedHint', { error: reshapeStatus.error })}
            </p>
          ) : null}
          <button
            type="button"
            className="mt-1 px-2 py-1 rounded-md hover:bg-surface focus-visible:outline-accent"
            onClick={() => setEditor((state) => ({ portraitReshapeRetry: state.portraitReshapeRetry + 1 }))}
          >
            {t('editor.portrait.status.retry')}
          </button>
        </div>
      );
    if (analysisFaces.length > 0 && decision.processableCount === 0)
      return (
        <div className="text-xs text-text-secondary leading-relaxed" role="status">
          <p>{t('editor.portrait.reshape.noProcessableFace')}</p>
          {analysisFaces.map((face) => (
            <p key={face.faceId} className="mt-0.5">
              {face.reason ? t(PORTRAIT_RESHAPE_REASON_LABELS[face.reason]) : null}
            </p>
          ))}
        </div>
      );
    // idle（尚未请求分析）不渲染提示：拖动滑块即进入新语义（02 第 7 条），
    // 且不能污染面板级 role="status" 的既有断言（磨皮/祛疵状态区）。
    return null;
  };

  const resetValues = (keys: ReadonlyArray<keyof PortraitAdjustments>) => {
    const defaults = Object.fromEntries(
      keys.map((key) => [key, INITIAL_PORTRAIT_ADJUSTMENTS[key]]),
    ) as unknown as Partial<PortraitAdjustments>;
    updateValues(defaults);
    debouncedSetHistory.flush();
  };

  const toggleSection = (section: PortraitSection) => {
    setUI((state) => {
      const isOpening = !state.portraitSectionsState[section];
      return {
        portraitSectionsState: {
          ...(enableFocusMode && isOpening
            ? { skin: false, blemishes: false, makeup: false, reshape: false, eyes: false, body: false }
            : state.portraitSectionsState),
          [section]: isOpening,
        },
      };
    });
  };

  const renderFaceList = (ariaLabel: string) =>
    analysisFaces.length > 0 ? (
      <div className="flex flex-col gap-1" role="listbox" aria-label={ariaLabel}>
        {analysisFaces.map((face, index) => {
          const reasonLabel = face.reason ? t(PORTRAIT_RESHAPE_REASON_LABELS[face.reason]) : null;
          const selected = decision.selectedFace?.faceId === face.faceId;
          return (
            <button
              key={face.faceId}
              type="button"
              role="option"
              aria-selected={selected}
              aria-disabled={!face.processable || undefined}
              disabled={!face.processable}
              title={reasonLabel ?? undefined}
              className={clsx(
                'flex items-center justify-between gap-2 px-2 py-1 rounded-md text-xs transition-colors focus-visible:outline-accent disabled:opacity-50 disabled:cursor-not-allowed',
                selected ? 'bg-card-active text-text-primary' : 'text-text-secondary hover:bg-surface',
              )}
              onClick={() => selectPortraitReshapeFace(face.faceId)}
              aria-label={t('editor.portrait.reshape.faceSelect', { index: index + 1 })}
            >
              <span>{t('editor.portrait.reshape.faceItem', { index: index + 1 })}</span>
              {reasonLabel && <span className="text-[10px] truncate text-right">{reasonLabel}</span>}
            </button>
          );
        })}
      </div>
    ) : null;

  return (
    <div className="flex flex-col h-full min-h-0" aria-label={t('editor.portrait.title')}>
      <div className="p-3 shrink-0 border-b border-surface">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <Text variant={TextVariants.title}>{t('editor.portrait.title')}</Text>
            {isReady && faceStatus && (
              <span className="flex items-center gap-1.5 min-w-0">
                {headerBusy ? (
                  <Loader2
                    size={18}
                    aria-hidden="true"
                    className="shrink-0 animate-spin text-text-secondary"
                  />
                ) : (
                  <ScanFace
                    size={20}
                    aria-hidden="true"
                    className={clsx('shrink-0', FACE_STATE_STYLES[faceStatus.state].icon)}
                  />
                )}
                <span className={clsx('text-xs truncate', FACE_STATE_STYLES[faceStatus.state].text)}>
                  {faceStatus.label}
                </span>
              </span>
            )}
          </div>
          <button
            type="button"
            className="p-2 rounded-full hover:bg-surface disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus-visible:outline-accent"
            disabled={!isReady || !hasChanges(PORTRAIT_KEYS)}
            onClick={() => resetValues(PORTRAIT_KEYS)}
            aria-label={t('editor.portrait.resetAll')}
            data-tooltip={t('editor.portrait.resetAll')}
          >
            <RotateCcw size={18} />
          </button>
        </div>
        {showStatus && (
          <div className="mt-2 text-xs text-text-secondary leading-relaxed" role="status" aria-live="polite">
            {analysisStatusActive && (
              <>
                <p>
                  {t(`editor.portrait.status.${portraitStatus.phase === 'idle' ? 'analyzing' : portraitStatus.phase}` as any, {
                    count: portraitStatus.summary?.faceCount ?? 0,
                  })}
                </p>
                {portraitStatus.error && <p className="break-words">{portraitStatus.error}</p>}
                {portraitStatus.phase === 'error' && (
                  <button
                    type="button"
                    className="mt-1 px-2 py-1 rounded-md hover:bg-surface focus-visible:outline-accent"
                    onClick={() => setEditor((state) => ({ portraitRetry: state.portraitRetry + 1 }))}
                  >
                    {t('editor.portrait.status.retry')}
                  </button>
                )}
              </>
            )}
            {!analysisStatusActive && reshapeAnalyzing && (
              <p className="flex items-center gap-1.5">
                <Loader2 size={12} className="animate-spin shrink-0" aria-hidden="true" />
                {t('editor.portrait.reshape.analyzing')}
              </p>
            )}
          </div>
        )}
        {analysisFaces.length > 0 && (
          <div className="mt-2">{renderFaceList(t('editor.portrait.title'))}</div>
        )}
      </div>
      <div className="grow min-h-0 overflow-y-auto p-3 flex flex-col gap-2">
        {selectedImage ? (
          PORTRAIT_SECTIONS.map((section) => (
            <CollapsibleSection
              key={section.id}
              title={t(section.title)}
              isOpen={portraitSectionsState[section.id]}
              isContentVisible={sectionContentVisible(section.id)}
              canToggleVisibility
              onToggleVisibility={() => toggleVisibility(sectionVisibilityKey(section.id))}
              onToggle={() => toggleSection(section.id)}
            >
              {section.id === 'reshape' ? (
                <div className="space-y-2">
                  {reshapeSectionMessage()}
                  <div className="flex flex-wrap justify-end gap-2">
                    <button
                      type="button"
                      className="px-2 py-1 rounded-md text-xs text-text-secondary hover:text-text-primary hover:bg-card-active disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus-visible:outline-accent"
                      disabled={
                        reshapeSlidersDisabled ||
                        !decision.selectedFace ||
                        !decision.selectedFaceHasOverride
                      }
                      onClick={resetReshapeValues}
                      aria-label={t('editor.portrait.resetSection', { section: t(section.title) })}
                    >
                      <RotateCcw size={12} aria-hidden="true" className="inline mr-1" />
                      {t('editor.portrait.resetGroup')}
                    </button>
                  </div>
                  {decision.selectedFaceHasOverride && (
                    <p className="text-[10px] text-text-secondary" data-testid="reshape-override-indicator">
                      {t('editor.portrait.reshape.overrideIndicator')}
                    </p>
                  )}
                  {showProtectionLimited && (
                    <p className="text-[10px] text-amber-500" data-testid="reshape-protection-limited">
                      {t('editor.portrait.reshape.protectionLimited')}
                      <span className="ml-1 text-text-secondary">
                        {t('editor.portrait.reshape.protectionLimitedHint')}
                      </span>
                    </p>
                  )}
                  <div className="space-y-3">
                    {UV_RESHAPE_KEYS.map((key) => (
                      <Slider
                        key={key}
                        label={t(PORTRAIT_LABELS[key])}
                        tooltip={t(`editor.portrait.tips.${key}`)}
                        min={0}
                        max={100}
                        step={1}
                        defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS[key]}
                        value={decision.uvValues[key]}
                        disabled={reshapeSlidersDisabled}
                        fillOrigin="min"
                        onChange={(event) =>
                          updateReshapeValues({ [key]: Number(event.target.value) })
                        }
                        onDragStateChange={onDragStateChange}
                        onPointerUp={() => debouncedSetHistory.flush()}
                      />
                    ))}
                  </div>
                </div>
              ) : section.id === 'eyes' ? (
                <div className="space-y-2">
                  {reshapeSectionMessage()}
                  <div className="flex flex-wrap justify-end gap-2">
                    <button
                      type="button"
                      className="px-2 py-1 rounded-md text-xs text-text-secondary hover:text-text-primary hover:bg-card-active disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus-visible:outline-accent"
                      disabled={
                        !isReady ||
                        !(
                          !reshapeSlidersDisabled &&
                          decision.selectedFace &&
                          decision.uvValues.bigEye !== 0
                        )
                      }
                      onClick={resetEyesValues}
                      aria-label={t('editor.portrait.resetSection', { section: t(section.title) })}
                    >
                      <RotateCcw size={12} aria-hidden="true" className="inline mr-1" />
                      {t('editor.portrait.resetGroup')}
                    </button>
                  </div>
                  <div className="space-y-3">
                    {EYES_KEYS.map((key) => (
                      <Slider
                        key={key}
                        label={t(PORTRAIT_LABELS[key])}
                        tooltip={t(`editor.portrait.tips.${key}`)}
                        min={0}
                        max={100}
                        step={1}
                        defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS[key]}
                        value={decision.uvValues[key]}
                        disabled={reshapeSlidersDisabled}
                        fillOrigin="min"
                        onChange={(event) =>
                          updateReshapeValues({ [key]: Number(event.target.value) })
                        }
                        onDragStateChange={onDragStateChange}
                        onPointerUp={() => debouncedSetHistory.flush()}
                      />
                    ))}
                  </div>
                </div>
              ) : section.id === 'body' ? (
                <div className="space-y-2">
                  <div className="flex justify-end">
                    <button
                      type="button"
                      className="px-2 py-1 rounded-md text-xs text-text-secondary hover:text-text-primary hover:bg-card-active disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus-visible:outline-accent"
                      disabled={!isReady || !hasChanges(section.keys)}
                      onClick={() => resetValues(section.keys)}
                      aria-label={t('editor.portrait.resetSection', { section: t(section.title) })}
                    >
                      <RotateCcw size={12} aria-hidden="true" className="inline mr-1" />
                      {t('editor.portrait.resetGroup')}
                    </button>
                  </div>
                  {bodyAmount > 0 && previewBusy ? (
                    <p className="flex items-center gap-1.5 text-xs text-text-secondary" role="status">
                      <Loader2 size={12} className="animate-spin shrink-0" aria-hidden="true" />
                      {t('editor.portrait.reshape.analyzing')}
                    </p>
                  ) : null}
                  {showBodyProtectionLimited ? (
                    <p className="text-[10px] text-amber-500" data-testid="body-protection-limited">
                      {t('editor.portrait.body.protectionLimited')}
                      <span className="ml-1 text-text-secondary">
                        {t('editor.portrait.body.protectionLimitedHint')}
                      </span>
                    </p>
                  ) : null}
                  <div className="space-y-3">
                    {section.keys.map((key) => (
                      <Slider
                        key={key}
                        label={t(PORTRAIT_LABELS[key])}
                        min={0}
                        max={100}
                        step={1}
                        defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS[key]}
                        value={values[key]}
                        disabled={!isReady}
                        fillOrigin="min"
                        onChange={(event) => updateValues({ [key]: Number(event.target.value) })}
                        onDragStateChange={onDragStateChange}
                        onPointerUp={() => debouncedSetHistory.flush()}
                      />
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex justify-end mb-2">
                    <button
                      type="button"
                      className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-text-secondary hover:text-text-primary hover:bg-card-active disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus-visible:outline-accent"
                      disabled={!isReady || !hasChanges(portraitSectionKeys(section.id))}
                      onClick={() => resetValues(portraitSectionKeys(section.id))}
                      aria-label={t('editor.portrait.resetSection', { section: t(section.title) })}
                    >
                      <RotateCcw size={12} aria-hidden="true" />
                      {t('editor.portrait.resetGroup')}
                    </button>
                  </div>
                  <div className="space-y-3">
                    {section.id === 'skin' ? (
                      <>
                        <Slider
                          label={t(PORTRAIT_LABELS.neutralGrayEvenness)}
                          tooltip={t('editor.portrait.tips.neutralGrayEvenness')}
                          min={0}
                          max={100}
                          step={1}
                          defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS.neutralGrayEvenness}
                          value={values.neutralGrayEvenness}
                          disabled={faceBlocked}
                          fillOrigin="min"
                          onChange={(event) =>
                            updateValues({ neutralGrayEvenness: Number(event.target.value) })
                          }
                          onDragStateChange={onDragStateChange}
                          onPointerUp={() => debouncedSetHistory.flush()}
                        />
                        <div className="space-y-3">
                          <Slider
                            label={t(PORTRAIT_LABELS.neutralGrayContour)}
                            tooltip={t('editor.portrait.tips.neutralGrayContour')}
                            min={0}
                            max={100}
                            step={1}
                            defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS.neutralGrayContour}
                            value={values.neutralGrayContour}
                            disabled={faceBlocked}
                            fillOrigin="min"
                            onChange={(event) =>
                              updateValues({ neutralGrayContour: Number(event.target.value) })
                            }
                            onDragStateChange={onDragStateChange}
                            onPointerUp={() => debouncedSetHistory.flush()}
                          />
                          {values.neutralGrayContour > 0 ? (
                            <div className="pl-4 border-l-2 border-border-color ml-1 space-y-3">
                              {(
                                [
                                  'neutralGrayContourFeatures',
                                  'neutralGrayContourOutline',
                                ] as const
                              ).map((key) => (
                                <Slider
                                  key={key}
                                  label={t(PORTRAIT_LABELS[key])}
                                  tooltip={t(`editor.portrait.tips.${key}`)}
                                  min={0}
                                  max={100}
                                  step={1}
                                  defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS[key]}
                                  value={values[key]}
                                  disabled={faceBlocked}
                                  fillOrigin="min"
                                  onChange={(event) =>
                                    updateValues({ [key]: Number(event.target.value) })
                                  }
                                  onDragStateChange={onDragStateChange}
                                  onPointerUp={() => debouncedSetHistory.flush()}
                                />
                              ))}
                            </div>
                          ) : null}
                        </div>
                        {(
                          [
                            {
                              master: 'frequencySeparationFace' as const,
                              bands: [
                                'frequencySeparationFaceLow',
                                'frequencySeparationFaceHigh',
                              ] as const,
                            },
                            {
                              master: 'frequencySeparationBody' as const,
                              bands: [
                                'frequencySeparationBodyLow',
                                'frequencySeparationBodyHigh',
                              ] as const,
                            },
                          ] as const
                        ).map(({ master, bands }) => (
                          <div key={master} className="space-y-3">
                            <Slider
                              label={t(PORTRAIT_LABELS[master])}
                              tooltip={t(`editor.portrait.tips.${master}`)}
                              min={0}
                              max={100}
                              step={1}
                              defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS[master]}
                              value={values[master]}
                              disabled={faceBlocked}
                              fillOrigin="min"
                              onChange={(event) =>
                                updateValues({ [master]: Number(event.target.value) })
                              }
                              onDragStateChange={onDragStateChange}
                              onPointerUp={() => debouncedSetHistory.flush()}
                            />
                            {values[master] > 0 ? (
                              <div className="pl-4 border-l-2 border-border-color ml-1 space-y-3">
                                {bands.map((key) => {
                                  const isHigh = key.endsWith('High');
                                  return (
                                    <Slider
                                      key={key}
                                      label={t(PORTRAIT_LABELS[key])}
                                      tooltip={t(`editor.portrait.tips.${key}`)}
                                      min={isHigh ? -100 : 0}
                                      max={100}
                                      step={1}
                                      defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS[key]}
                                      value={values[key]}
                                      disabled={faceBlocked}
                                      {...(isHigh ? {} : { fillOrigin: 'min' as const })}
                                      onChange={(event) =>
                                        updateValues({ [key]: Number(event.target.value) })
                                      }
                                      onDragStateChange={onDragStateChange}
                                      onPointerUp={() => debouncedSetHistory.flush()}
                                    />
                                  );
                                })}
                              </div>
                            ) : null}
                          </div>
                        ))}
                        {(
                          [
                            'hydratedSmoothingFace',
                            'hydratedSmoothingBody',
                            'evenSkinFace',
                            'evenSkinBody',
                            'whiteningFace',
                            'whiteningBody',
                          ] as const
                        ).map((key) => (
                          <Slider
                            key={key}
                            label={t(PORTRAIT_LABELS[key])}
                            tooltip={t(`editor.portrait.tips.${key}`)}
                            min={0}
                            max={100}
                            step={1}
                            defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS[key]}
                            value={values[key]}
                            disabled={faceBlocked}
                            fillOrigin="min"
                            onChange={(event) => updateValues({ [key]: Number(event.target.value) })}
                            onDragStateChange={onDragStateChange}
                            onPointerUp={() => debouncedSetHistory.flush()}
                          />
                        ))}
                      </>
                    ) : (
                      section.keys.map((key) => (
                        <Slider
                          key={key}
                          label={t(PORTRAIT_LABELS[key])}
                          tooltip={t(`editor.portrait.tips.${key}`)}
                          min={0}
                          max={100}
                          step={1}
                          defaultValue={INITIAL_PORTRAIT_ADJUSTMENTS[key]}
                          value={values[key]}
                          disabled={faceBlocked}
                          fillOrigin="min"
                          onChange={(event) => updateValues({ [key]: Number(event.target.value) })}
                          onDragStateChange={onDragStateChange}
                          onPointerUp={() => debouncedSetHistory.flush()}
                        />
                      ))
                    )}
                  </div>
                </>
              )}
            </CollapsibleSection>
          ))
        ) : (
          <div className="flex items-center justify-center h-full">
            <Text
              variant={TextVariants.heading}
              color={TextColors.secondary}
              weight={TextWeights.normal}
              className="text-center"
            >
              {t('editor.ai.noImageSelected')}
            </Text>
          </div>
        )}
      </div>
    </div>
  );
}

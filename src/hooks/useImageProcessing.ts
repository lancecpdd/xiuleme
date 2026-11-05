import React, { useCallback, useEffect, useRef, useMemo } from 'react';
import { Channel, invoke } from '@tauri-apps/api/core';
import debounce from 'lodash.debounce';
import throttle from 'lodash.throttle';
import {
  useEditorStore,
  acceptPreviewFrame,
  adjustmentsAffectLevelsInputHistogram,
  matchesPreviewRequest,
  type PortraitInitialApplyKind,
  type PortraitStatus,
  type PortraitSummary,
} from '../store/useEditorStore';
import { useUIStore } from '../store/useUIStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { useLibraryStore } from '../store/useLibraryStore';
import {
  Adjustments,
  COPYABLE_ADJUSTMENT_KEYS,
  deferredPortraitRetouch,
  deferredPortraitReshape,
  detectPortraitVisibilityFlip,
  flipPortraitSectionVisibility,
  readPortraitReshape,
  hasPortraitEffect,
  hasPortraitReshapePreviewEffect,
  isPortraitSectionVisible,
  PORTRAIT_VISIBILITY_KEYS,
  portraitSourceSignature,
  portraitVisibilityPreviewSignature,
  portraitVisibilitySectionHasEffect,
  type PortraitVisibilityKey,
} from '../utils/adjustments';
import { Invokes, Panel, decodePreviewResponse } from '../components/ui/AppProperties';
import { debouncedSave } from './useEditorActions';
import { globalImageCache } from '../utils/ImageLRUCache';
import { usePortraitReshapeAnalysis } from './usePortraitReshapeAnalysis';

/** 人像分区小眼睛 A/B：同 key+签名下各保留一帧 JPEG，仅该键可见性翻转时可即时换帧。 */
type PortraitVisibilityPreviewCache = {
  path: string;
  key: PortraitVisibilityKey;
  signature: string;
  onUrl: string | null;
  offUrl: string | null;
};

const pickPortraitVisibilityCacheKey = (adjustments: Adjustments): PortraitVisibilityKey | null => {
  for (const key of PORTRAIT_VISIBILITY_KEYS) {
    if (portraitVisibilitySectionHasEffect(adjustments, key)) return key;
  }
  return null;
};

const resolvePortraitInitialApplyKind = (
  adjustments: Adjustments,
  reshapeDeferred: boolean,
  retouchDeferred: boolean,
): PortraitInitialApplyKind => {
  if (reshapeDeferred || hasPortraitReshapePreviewEffect(adjustments)) return 'reshape';
  if (retouchDeferred || hasPortraitEffect(adjustments)) return 'retouch';
  return 'portrait';
};

export interface PreviewApplyJob {
  path: string;
  generation: number;
  readiness: string;
  adjustments: Adjustments;
  targetRes?: number;
  dragging: boolean;
  queuedAt: number;
}

/** 共享子树直接跳过，克隆参数按内容比较，避免序列化整份补丁像素。 */
const samePreviewValue = (a: unknown, b: unknown): boolean => {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const left = a as Record<string, unknown>;
  const right = b as Record<string, unknown>;
  const keys = Object.keys(left);
  return (
    keys.length === Object.keys(right).length &&
    keys.every((key) => Object.prototype.hasOwnProperty.call(right, key) && samePreviewValue(left[key], right[key]))
  );
};

const samePreviewApply = (a: PreviewApplyJob | null, b: PreviewApplyJob): boolean =>
  !!a &&
  a.path === b.path &&
  a.generation === b.generation &&
  a.readiness === b.readiness &&
  a.targetRes === b.targetRes &&
  a.dragging === b.dragging &&
  samePreviewValue(a.adjustments, b.adjustments);

/** 单飞队列：只保存最新意图，跨路径/代次不复用；失败允许重试。 */
export function createPreviewApplyQueue() {
  let inFlight: PreviewApplyJob | null = null;
  let pending: PreviewApplyJob | null = null;
  let completed: PreviewApplyJob | null = null;
  let revision = 0;
  let flightRevision = 0;
  return {
    offer(job: PreviewApplyJob) {
      pending = !inFlight && samePreviewApply(completed, job) ? null : job;
    },
    take(path: string | undefined, generation: number): PreviewApplyJob | null {
      if (inFlight || !pending) return null;
      const job = pending;
      pending = null;
      if (job.path !== path || job.generation !== generation) return null;
      inFlight = job;
      flightRevision = revision;
      return job;
    },
    finish(job: PreviewApplyJob, committed: boolean) {
      if (inFlight !== job) return;
      inFlight = null;
      completed = committed && flightRevision === revision ? job : null;
      if (pending && samePreviewApply(completed, pending)) pending = null;
    },
    hasPending: () => pending !== null,
    clearPending() {
      pending = null;
      completed = null;
      revision += 1;
    },
    dropPendingHighQuality() {
      if (pending && !pending.dragging) pending = null;
    },
  };
}

// 自研美肤交互帧已 bake 进预览基底，调色 settle 与无人像一致；仅重塑/GPUPixel Prepare 需要更长合并窗。
export const previewSettleDelay = (adjustments: Adjustments): number =>
  hasPortraitReshapePreviewEffect(adjustments) || hasPortraitEffect(adjustments) ? 150 : 50;

export function useImageProcessing(
  transformWrapperRef: any,
  prevAdjustmentsRef: React.RefObject<any>,
  renderRefs: {
    previewJobIdRef: React.RefObject<number>;
    latestRenderedJobIdRef: React.RefObject<number>;
    currentResRef: React.RefObject<number>;
  },
) {
  const { previewJobIdRef, latestRenderedJobIdRef, currentResRef } = renderRefs;

  const selectedImage = useEditorStore((state) => state.selectedImage);
  const adjustments = useEditorStore((state) => state.adjustments);
  const previewOverride = useEditorStore((state) => state.previewOverride);
  const isWaveformVisible = useEditorStore((state) => state.isWaveformVisible);
  const activeWaveformChannel = useEditorStore((state) => state.activeWaveformChannel);
  const displaySize = useEditorStore((state) => state.displaySize);
  const baseRenderSize = useEditorStore((state) => state.baseRenderSize);
  const originalSize = useEditorStore((state) => state.originalSize);
  const isSliderDragging = useEditorStore((state) => state.isSliderDragging);
  const setEditor = useEditorStore((state) => state.setEditor);
  const portraitStatus = useEditorStore((state) => state.portraitStatus);
  const portraitRetry = useEditorStore((state) => state.portraitRetry);
  const portraitReshapeKey = useEditorStore((state) => state.portraitReshapeStatus.key);
  const portraitReshapeRetry = useEditorStore((state) => state.portraitReshapeRetry);

  const activeView = useUIStore((state) => state.activeView);
  const activePanel = useUIStore((state) => state.activePanel);
  const activePanels = useUIStore((state) => state.activePanels);
  const appSettings = useSettingsStore((state) => state.appSettings);
  const multiSelectedPaths = useLibraryStore((state) => state.multiSelectedPaths);

  const uncroppedJobIdRef = useRef(0);
  const latestUncroppedJobIdRef = useRef(0);
  const portraitVisCacheRef = useRef<PortraitVisibilityPreviewCache | null>(null);
  const portraitVisPrefetchGenRef = useRef(0);
  /** Last adjustments used when levels-input histogram was requested (path-scoped via nulling on path change). */
  const lastLevelsInputAdjRef = useRef<Adjustments | null>(null);
  const lastLevelsInputPathRef = useRef<string | null>(null);
  const selectedImagePathRef = useRef<string | null>(null);
  selectedImagePathRef.current = selectedImage?.path ?? null;

  const clearPortraitVisCache = useCallback((keepUrl?: string | null) => {
    const cache = portraitVisCacheRef.current;
    if (!cache) return;
    for (const url of [cache.onUrl, cache.offUrl]) {
      if (url && url !== keepUrl && url.startsWith('blob:')) URL.revokeObjectURL(url);
    }
    portraitVisCacheRef.current = null;
  }, []);

  const rememberPortraitVisFrame = useCallback(
    (path: string, adjustments: Adjustments, url: string, preferredKey?: PortraitVisibilityKey) => {
      let cache = portraitVisCacheRef.current;
      let key = preferredKey;
      if (cache && cache.path === path) {
        const cacheSig = portraitVisibilityPreviewSignature(adjustments, cache.key);
        if (cacheSig === cache.signature) key = cache.key;
      }
      if (!key) key = pickPortraitVisibilityCacheKey(adjustments) ?? undefined;
      if (!key) return;
      const signature = portraitVisibilityPreviewSignature(adjustments, key);
      if (!cache || cache.path !== path || cache.key !== key || cache.signature !== signature) {
        clearPortraitVisCache(url);
        cache = { path, key, signature, onUrl: null, offUrl: null };
        portraitVisCacheRef.current = cache;
      }
      const slot = isPortraitSectionVisible(adjustments, key) ? 'onUrl' : 'offUrl';
      const previous = cache[slot];
      if (previous && previous !== url && previous.startsWith('blob:')) {
        setTimeout(() => {
          const latest = portraitVisCacheRef.current;
          const stillCached = latest && (latest.onUrl === previous || latest.offUrl === previous);
          if (!stillCached && !globalImageCache.isProtected(previous)) URL.revokeObjectURL(previous);
        }, 250);
      }
      cache[slot] = url;
    },
    [clearPortraitVisCache],
  );

  const trySwapPortraitVisCache = useCallback(
    (path: string, next: Adjustments, prev: Adjustments | null | undefined): boolean => {
      if (!prev) return false;
      const flipKey = detectPortraitVisibilityFlip(prev, next);
      if (!flipKey) return false;
      const cache = portraitVisCacheRef.current;
      const signature = portraitVisibilityPreviewSignature(next, flipKey);
      if (!cache || cache.path !== path || cache.key !== flipKey || cache.signature !== signature) {
        return false;
      }
      const url = isPortraitSectionVisible(next, flipKey) ? cache.onUrl : cache.offUrl;
      if (!url) return false;
      setEditor((state) => {
        if (state.selectedImage?.path !== path) return {};
        if (state.finalPreviewUrl === url) return { interactivePatch: null };
        const patch = state.interactivePatch?.url;
        if (patch) setTimeout(() => URL.revokeObjectURL(patch), 500);
        return { finalPreviewUrl: url, interactivePatch: null };
      });
      return true;
    },
    [setEditor],
  );

  /** 后台预热对侧小眼睛帧：不改 UI、不占主 previewRequest。 */
  const prefetchPortraitVisOpposite = useCallback(
    async (path: string, adjustments: Adjustments, key: PortraitVisibilityKey, targetRes?: number) => {
      if (!portraitVisibilitySectionHasEffect(adjustments, key)) return;
      const cache = portraitVisCacheRef.current;
      const signature = portraitVisibilityPreviewSignature(adjustments, key);
      if (!cache || cache.path !== path || cache.key !== key || cache.signature !== signature) return;
      const needOn = !cache.onUrl;
      const needOff = !cache.offUrl;
      if (!needOn && !needOff) return;
      const visible = isPortraitSectionVisible(adjustments, key);
      // 预热缺失的对侧：当前为开则求关，当前为关则求开。
      if ((visible && !needOff) || (!visible && !needOn)) return;

      const gen = ++portraitVisPrefetchGenRef.current;
      const flipped = flipPortraitSectionVisibility(adjustments, key);
      const payload = structuredClone(flipped);
      delete (payload as any).frame;
      try {
        const buffer: ArrayBuffer = await invoke(Invokes.ApplyAdjustments, {
          jsAdjustments: payload,
          path,
          requestId: -1,
          isInteractive: false,
          targetResolution: targetRes || null,
          roi: null,
          requestAnalytics: false,
          computeWaveform: false,
          activeWaveformChannel: null,
        });
        if (gen !== portraitVisPrefetchGenRef.current) return;
        if (path !== selectedImagePathRef.current) return;
        const latest = portraitVisCacheRef.current;
        if (!latest || latest.path !== path || latest.key !== key || latest.signature !== signature) {
          return;
        }
        const { metadata, image: imageBuffer } = decodePreviewResponse(buffer);
        if (metadata.path !== path || metadata.mode === 'native') return;
        const url = URL.createObjectURL(new Blob([imageBuffer], { type: 'image/jpeg' }));
        const still = portraitVisCacheRef.current;
        if (
          !still ||
          still.path !== path ||
          still.key !== key ||
          still.signature !== signature ||
          gen !== portraitVisPrefetchGenRef.current
        ) {
          URL.revokeObjectURL(url);
          return;
        }
        rememberPortraitVisFrame(path, flipped, url, key);
      } catch {
        // 预热失败静默忽略，下次切换仍可走全量 Apply。
      }
    },
    [rememberPortraitVisFrame],
  );

  const applyQueueRef = useRef(createPreviewApplyQueue());
  const lastAnalyticsTimeRef = useRef<number>(0);
  // 重塑预览很慢：最多 1 个在飞，新请求只覆盖 pending，避免代次互抢导致「重塑分析请求已过期」。
  const dragIdleTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // 同一路径首帧可 Bypass 重塑；提交后再允许完整 Prepare，避免直方图等人脸模型。
  const reshapePreviewAllowedRef = useRef<string | null>(null);
  const committedPortraitPathRef = useRef<string | null>(null);
  const activeWaveformChannelRef = useRef(activeWaveformChannel);
  activeWaveformChannelRef.current = activeWaveformChannel;
  const makeApplyJob = useCallback(
    (adj: Adjustments, dragging: boolean, targetRes?: number): PreviewApplyJob | null => {
      const state = useEditorStore.getState();
      const path = state.selectedImage?.path;
      if (!path || !state.selectedImage?.isReady) return null;
      const settings = useSettingsStore.getState().appSettings;
      return {
        path,
        generation: state.portraitStatus.generation,
        readiness: JSON.stringify([
          state.portraitStatus.phase,
          state.portraitReshapeStatus.key,
          state.portraitRetry,
          state.portraitReshapeRetry,
          reshapePreviewAllowedRef.current,
          state.isWaveformVisible,
          state.activeWaveformChannel,
          transformWrapperRef.current?.instance?.transformState,
          settings?.livePreviewQuality,
          settings?.useWgpuRenderer,
          settings?.rawHighlightCompression,
          settings?.linearRawMode,
          settings?.rawPreprocessingColorNr,
          settings?.rawPreprocessingSharpening,
          settings?.applyPreprocessingToNonRaws,
        ]),
        adjustments: adj,
        targetRes,
        dragging,
        queuedAt: performance.now(),
      };
    },
    [transformWrapperRef],
  );

  // 人像分析始终面向真实编辑状态；previewOverride（显示原始/LUT 预览）不得改变分析签名，
  // 否则临时预览会触发重新识别人脸并把分析结果缓存到错误的键下。
  const portraitEnabled = hasPortraitEffect(adjustments);
  const portraitSource = portraitSourceSignature(adjustments);
  const portraitSettings = JSON.stringify([
    appSettings?.rawHighlightCompression,
    appSettings?.linearRawMode,
    appSettings?.rawPreprocessingColorNr,
    appSettings?.rawPreprocessingSharpening,
    appSettings?.applyPreprocessingToNonRaws,
  ]);
  usePortraitReshapeAnalysis(activeView === 'editor', portraitSettings);
  const portraitPanelActive =
    activeView === 'editor' && (activePanel === Panel.Portrait || Object.values(activePanels).includes(Panel.Portrait));

  useEffect(() => {
    applyQueueRef.current.clearPending();
    const path = selectedImage?.path ?? null;
    const status = useEditorStore.getState().portraitStatus;
    const analysisSignature = `${path}|${portraitSource}|${portraitSettings}`;
    const upToDate =
      status.analysisSignature === analysisSignature &&
      status.path === path &&
      ['ready', 'empty', 'partial'].includes(status.phase);
    const shouldAnalyze = (portraitEnabled || portraitPanelActive) && !upToDate;
    if (!shouldAnalyze) {
      if (path && status.path !== path) {
        setEditor({
          portraitStatus: {
            path,
            generation: status.generation + 1,
            source: portraitSource,
            analysisSignature,
            phase: 'idle',
          },
        });
      }
      return;
    }
    const generation = status.generation + 1;
    let active = true;
    const update = (change: Partial<PortraitStatus>) => {
      if (!active) return;
      setEditor((state) =>
        state.selectedImage?.path === path && state.portraitStatus.generation === generation
          ? { portraitStatus: { ...state.portraitStatus, ...change } }
          : {},
      );
    };
    setEditor({
      portraitStatus: { path, generation, source: portraitSource, analysisSignature, phase: 'idle' },
    });
    if (!path || !selectedImage?.isReady)
      return () => {
        active = false;
      };
    update({ phase: 'analyzing' });
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;
    const run = () => {
      if (!active) return;
      const progress = new Channel<'loading' | 'analyzing'>();
      progress.onmessage = (phase) => update({ phase });
      const current = useEditorStore.getState();
      const analysisAdj = structuredClone(current.adjustments) as any;
      delete analysisAdj.frame;
      invoke<PortraitSummary>(Invokes.PreparePortraitAnalysis, {
        path,
        generation,
        jsAdjustments: analysisAdj,
        onProgress: progress,
      })
        .then((summary) => {
          update({ summary, phase: !summary.hasSkin ? 'empty' : summary.skippedFaces > 0 ? 'partial' : 'ready' });
        })
        .catch((error: unknown) => {
          // 前端缓存命中切回时后端 load_image 仍在进行（original_image 仍是上一张），
          // 后端返回 Superseded；这不是分析失败，保持「检测中」并稍后重试。
          if (String(error) === 'Superseded or worker failed' && attempts < 40) {
            attempts += 1;
            retryTimer = setTimeout(run, 250);
            return;
          }
          update({ phase: 'error', error: String(error) });
        });
    };
    run();
    return () => {
      active = false;
      if (retryTimer) clearTimeout(retryTimer);
      void invoke(Invokes.CancelPortraitAnalysis, { generation }).catch(() => {
        // 新请求可能已经接管同一代次；取消采用后端代次比较，迟到取消不会影响新任务。
      });
    };
  }, [
    selectedImage?.path,
    selectedImage?.isReady,
    portraitEnabled,
    portraitSource,
    portraitSettings,
    portraitRetry,
    portraitPanelActive,
    setEditor,
  ]);

  const calculateROI = useCallback(() => {
    if (!transformWrapperRef.current) return null;
    const state = transformWrapperRef.current.instance.transformState;
    if (!state) return null;

    if (!baseRenderSize) return null;

    const { scale, positionX, positionY } = state;
    const { width: baseW, height: baseH, offsetX, offsetY, containerWidth, containerHeight } = baseRenderSize;

    if (!baseW || !baseH || !containerWidth || !containerHeight) return null;
    if (scale <= 1.01) return null;

    const paddingPixels = 2.0;
    const paddingX = paddingPixels / baseW;
    const paddingY = paddingPixels / baseH;

    const visibleLeft = -positionX / scale;
    const visibleTop = -positionY / scale;
    const visibleRight = visibleLeft + containerWidth / scale;
    const visibleBottom = visibleTop + containerHeight / scale;

    const imgLeft = offsetX;
    const imgTop = offsetY;
    const imgRight = offsetX + baseW;
    const imgBottom = offsetY + baseH;

    const intersectLeft = Math.max(visibleLeft, imgLeft);
    const intersectTop = Math.max(visibleTop, imgTop);
    const intersectRight = Math.min(visibleRight, imgRight);
    const intersectBottom = Math.min(visibleBottom, imgBottom);

    if (intersectLeft >= intersectRight || intersectTop >= intersectBottom) {
      return null;
    }

    const roiX = (intersectLeft - imgLeft) / baseW;
    const roiY = (intersectTop - imgTop) / baseH;
    const roiW = (intersectRight - intersectLeft) / baseW;
    const roiH = (intersectBottom - intersectTop) / baseH;

    const newRoiX = roiX - paddingX;
    const newRoiY = roiY - paddingY;
    const newRoiW = roiW + paddingX * 2;
    const newRoiH = roiH + paddingY * 2;

    const clampedX = Math.max(0, newRoiX);
    const clampedY = Math.max(0, newRoiY);
    const clampedW = Math.min(1 - clampedX, newRoiW);
    const clampedH = Math.min(1 - clampedY, newRoiH);

    if (clampedW > 0.999 && clampedH > 0.999) return null;

    return [clampedX, clampedY, clampedW, clampedH] as [number, number, number, number];
  }, [baseRenderSize, transformWrapperRef]);

  const executeApplyAdjustments = useCallback(
    async (currentAdjustments: Adjustments, dragging: boolean = false, targetRes?: number, queuedAt?: number) => {
      const currentPath = selectedImage?.path;
      if (!currentPath || currentPath !== useEditorStore.getState().selectedImage?.path) return;
      const requestedAt = performance.now();
      const logCommittedFrame = (jobId: number) => {
        if (!portraitDeferred) {
          committedPortraitPathRef.current =
            hasPortraitEffect(currentAdjustments) || reshapeActive ? currentPath : null;
        }
        if (import.meta.env.DEV)
          console.debug('[portrait-preview] committed', {
            requestId: jobId,
            dragging,
            requestToCommitMs: performance.now() - requestedAt,
            intentToCommitMs: performance.now() - (queuedAt ?? requestedAt),
          });
      };

      let shouldRequestAnalytics = false;
      if (dragging) {
        const now = performance.now();
        // 拖动中降频直方图，减轻 Curves 重绘与 analytics 线程争用。
        if (now - lastAnalyticsTimeRef.current > 120) {
          shouldRequestAnalytics = true;
          lastAnalyticsTimeRef.current = now;
        }
      } else {
        shouldRequestAnalytics = true;
        lastAnalyticsTimeRef.current = 0;
      }

      // Capture One–style levels input hist: refresh only when upstream-of-levels changes (or missing).
      if (lastLevelsInputPathRef.current !== currentPath) {
        lastLevelsInputPathRef.current = currentPath;
        lastLevelsInputAdjRef.current = null;
      }
      const existingLevelsInput = useEditorStore.getState().levelsInputHistogram;
      const shouldRequestLevelsInputAnalytics =
        shouldRequestAnalytics &&
        (!existingLevelsInput ||
          adjustmentsAffectLevelsInputHistogram(lastLevelsInputAdjRef.current, currentAdjustments));

      const payload = structuredClone(currentAdjustments);
      // 信息边框仅前端 Canvas / 导出拼接，不参与 GPU 预览，避免污染哈希与渲染缓存
      delete (payload as any).frame;
      // 人像分析未就绪或本路径尚未放过色调首帧时，中性化修肤并 Bypass 重塑，
      // 预览与直方图即刻产出；分析完成／首帧提交后会再全量 Apply。
      const portraitReady =
        !hasPortraitEffect(currentAdjustments) ||
        (() => {
          const status = useEditorStore.getState().portraitStatus;
          return (
            status.path === currentPath &&
            status.source === portraitSourceSignature(currentAdjustments) &&
            ['ready', 'empty', 'partial'].includes(status.phase)
          );
        })();
      if (reshapePreviewAllowedRef.current !== currentPath && reshapePreviewAllowedRef.current !== null) {
        reshapePreviewAllowedRef.current = null;
      }
      const reshapeActive = hasPortraitReshapePreviewEffect(currentAdjustments);
      // 仅路径首帧未放行时 Bypass；已放行后拖调色保持重塑脸型，避免闪回「关重塑」。
      const hadCompletePortrait = committedPortraitPathRef.current === currentPath;
      const reshapeDeferred = reshapeActive && reshapePreviewAllowedRef.current !== currentPath && !hadCompletePortrait;
      const retouchDeferred = !portraitReady;
      const portraitDeferred = retouchDeferred || reshapeDeferred;
      // 已经展示过完整人像时，分析重试期间保留旧画面，不能发布临时关闭人像的调色帧。
      if (retouchDeferred && hadCompletePortrait) return;
      if (retouchDeferred) {
        payload.portrait = deferredPortraitRetouch();
      }
      if (reshapeDeferred) {
        payload.portraitReshape = deferredPortraitReshape(currentAdjustments.portraitReshape);
      }
      // 重塑的内容键必须看到实际像素，不能用 id 命中掩盖同长度补丁／蒙版替换。
      const reshapeEnabled =
        readPortraitReshape(payload.portraitReshape).settings?.enabled === true &&
        (payload.sectionVisibility?.portraitReshape !== false || payload.sectionVisibility?.portraitEyes !== false);
      const { patchesSentToBackend } = useEditorStore.getState();
      const newlySentPatches = new Set<string>();

      const processSubMasks = (subMasks: any[]) => {
        if (!Array.isArray(subMasks)) return;
        subMasks.forEach((sm: any) => {
          if (sm.id && sm.parameters) {
            const keys = ['mask_data_base64', 'maskDataBase64'];
            let foundMaskData = false;

            for (const key of keys) {
              if (sm.parameters[key] !== undefined && sm.parameters[key] !== null) {
                foundMaskData = true;
                if (!reshapeEnabled && patchesSentToBackend.has(sm.id)) {
                  sm.parameters[key] = null;
                }
              }
            }
            if (foundMaskData && !patchesSentToBackend.has(sm.id)) {
              newlySentPatches.add(sm.id);
            }
          }
        });
      };

      if (payload.aiPatches && Array.isArray(payload.aiPatches)) {
        payload.aiPatches.forEach((p: any) => {
          if (p.id && p.patchData && !p.isLoading) {
            if (!reshapeEnabled && patchesSentToBackend.has(p.id)) {
              p.patchData = null;
            } else {
              newlySentPatches.add(p.id);
            }
          }
          if (p.subMasks) processSubMasks(p.subMasks);
        });
      }

      if (payload.masks && Array.isArray(payload.masks)) {
        payload.masks.forEach((container: any) => {
          if (container.subMasks) processSubMasks(container.subMasks);
        });
      }

      const jobId = ++previewJobIdRef.current;
      const existingInitial = useEditorStore.getState().portraitInitialApply;
      const stillPendingInitial = !!existingInitial && existingInitial.path === currentPath && existingInitial.pending;
      // 仅刷新首开已挂起的角标 kind；不在日常拖滑块时新开 pending。
      const nextInitialApply = stillPendingInitial
        ? {
            path: currentPath,
            pending: true as const,
            kind: resolvePortraitInitialApplyKind(currentAdjustments, reshapeDeferred, retouchDeferred),
          }
        : undefined;
      // 同路径 HiFi／二次 Apply 保留旧直方图，直到新 analytics stamp 对齐。
      setEditor({
        previewRequest: { path: currentPath, requestId: jobId },
        previewError: null,
        pendingPreviewAnalytics: null,
        ...(nextInitialApply !== undefined ? { portraitInitialApply: nextInitialApply } : {}),
      });
      const requestGeneration = useEditorStore.getState().portraitStatus.generation;
      const roi = calculateROI();
      const armReshapeFollowUp = () => {
        // 拖动帧不排队二次全量重塑；仅松手／非交互提交后补完整 Prepare。
        if (!reshapeDeferred || dragging) return;
        reshapePreviewAllowedRef.current = currentPath;
        if (!applyQueueRef.current.hasPending()) {
          const followUp = makeApplyJob(currentAdjustments, false, targetRes);
          if (followUp) applyQueueRef.current.offer(followUp);
        }
      };

      try {
        const buffer: ArrayBuffer = await invoke(Invokes.ApplyAdjustments, {
          jsAdjustments: payload,
          path: currentPath,
          requestId: jobId,
          isInteractive: dragging,
          targetResolution: targetRes || null,
          roi: roi || null,
          requestAnalytics: shouldRequestAnalytics,
          requestLevelsInputAnalytics: shouldRequestLevelsInputAnalytics,
          computeWaveform: !!isWaveformVisible,
          activeWaveformChannel: activeWaveformChannelRef.current || null,
        });
        if (shouldRequestLevelsInputAnalytics) {
          lastLevelsInputAdjRef.current = currentAdjustments;
        }

        if (
          currentPath !== selectedImagePathRef.current ||
          requestGeneration !== useEditorStore.getState().portraitStatus.generation ||
          !matchesPreviewRequest(useEditorStore.getState(), currentPath, jobId)
        )
          return;
        // 延迟预览可以无人像效果先上屏；分析完成后由 phase／follow-up 再全量渲染。
        if (newlySentPatches.size > 0) {
          newlySentPatches.forEach((id) => patchesSentToBackend.add(id));
        }

        if (currentPath !== selectedImagePathRef.current) return;

        if (jobId < latestRenderedJobIdRef.current) return;
        const { metadata, image: imageBuffer } = decodePreviewResponse(buffer);
        if (metadata.path !== currentPath || metadata.stamp.requestId !== jobId) throw new Error('预览响应身份不匹配');
        if (metadata.mode === 'native') {
          let committedNative = false;
          setEditor((state) => {
            const frame = acceptPreviewFrame(state, metadata);
            if (!frame.previewStamp) return {};
            if (state.interactivePatch?.url) URL.revokeObjectURL(state.interactivePatch.url);
            latestRenderedJobIdRef.current = jobId;
            committedNative = true;
            return {
              ...frame,
              interactivePatch: null,
              ...(!portraitDeferred ? { portraitInitialApply: null } : {}),
            };
          });
          if (committedNative) {
            logCommittedFrame(jobId);
            armReshapeFollowUp();
          }
          return committedNative && !portraitDeferred;
        }

        const url = URL.createObjectURL(new Blob([imageBuffer], { type: 'image/jpeg' }));
        let committed = false;
        try {
          // 解码成功才提交画面和统计；失败保留旧图，解码期间仍可能切图或接受新请求。
          const image = new Image();
          image.src = url;
          await image.decode();
          if (image.naturalWidth !== metadata.roi.width || image.naturalHeight !== metadata.roi.height)
            throw new Error('预览图像尺寸与版本窗口不一致');
          if (requestGeneration !== useEditorStore.getState().portraitStatus.generation) return;
          setEditor((state) => {
            const frame = acceptPreviewFrame(state, metadata);
            if (!frame.previewStamp || jobId < latestRenderedJobIdRef.current) return {};
            const previousPatch = state.interactivePatch?.url;
            if (previousPatch) setTimeout(() => URL.revokeObjectURL(previousPatch), 500);
            latestRenderedJobIdRef.current = jobId;
            committed = true;
            if (dragging) {
              const { x, y, width, height } = metadata.roi;
              const [fullW, fullH] = metadata.dimensions;
              return {
                ...frame,
                interactivePatch: {
                  url,
                  normX: x / fullW,
                  normY: y / fullH,
                  normW: width / fullW,
                  normH: height / fullH,
                },
              };
            }
            const previousUrl = state.finalPreviewUrl;
            if (previousUrl?.startsWith('blob:')) {
              setTimeout(() => {
                const cached = portraitVisCacheRef.current;
                const inVisCache = cached && (cached.onUrl === previousUrl || cached.offUrl === previousUrl);
                if (!inVisCache && !globalImageCache.isProtected(previousUrl)) {
                  URL.revokeObjectURL(previousUrl);
                }
              }, 250);
            }
            if (!portraitDeferred) {
              rememberPortraitVisFrame(currentPath, currentAdjustments, url);
            }
            return {
              ...frame,
              finalPreviewUrl: url,
              interactivePatch: null,
              ...(!portraitDeferred ? { portraitInitialApply: null } : {}),
            };
          });
          if (committed && !portraitDeferred && !dragging) {
            const cache = portraitVisCacheRef.current;
            if (cache && cache.path === currentPath) {
              void prefetchPortraitVisOpposite(currentPath, currentAdjustments, cache.key, targetRes);
            }
          }
          if (committed) {
            logCommittedFrame(jobId);
            armReshapeFollowUp();
          }
        } finally {
          if (!committed) URL.revokeObjectURL(url);
        }
        return committed && !portraitDeferred;
      } catch (err) {
        if (
          currentPath !== selectedImagePathRef.current ||
          requestGeneration !== useEditorStore.getState().portraitStatus.generation ||
          !matchesPreviewRequest(useEditorStore.getState(), currentPath, jobId)
        )
          return;
        // 代次被更新的预览取代时后端会返回这些文案；不是美肤／重塑分析失败。
        const superseded = err === 'Superseded or worker failed' || err === '重塑分析请求已过期';
        if (!superseded) {
          console.error('Failed to apply adjustments:', err);
        }
        // 被取代：清掉本请求身份但不写 previewError，以免卡住后续成功帧的 matchesPreviewRequest。
        // 真失败才标 previewError；不得把美肤／重塑分析 phase 打成 error，否则面板误报「未检测到人脸」。
        if (superseded) {
          setEditor((state) =>
            matchesPreviewRequest(state, currentPath, jobId) ? { pendingPreviewAnalytics: null } : {},
          );
          return;
        }
        setEditor((state) => ({
          previewError: String(err),
          pendingPreviewAnalytics: null,
          // 预览失败必须撤下「正在应用…」角标；否则 portraitInitialApply.pending
          // 会一直为 true（例如冻结资产预算错误），UI 永久显示「正在应用面部重塑」。
          portraitInitialApply: null,
          // 预览／场校验失败不能把分析 phase 打成 error，否则面板误报「人脸分析失败」并禁用滑块。
        }));
      }
    },
    [
      selectedImage?.path,
      calculateROI,
      isWaveformVisible,
      setEditor,
      previewJobIdRef,
      latestRenderedJobIdRef,
      rememberPortraitVisFrame,
      prefetchPortraitVisOpposite,
      makeApplyJob,
    ],
  );

  const executeApplyRef = useRef(executeApplyAdjustments);
  executeApplyRef.current = executeApplyAdjustments;
  const flushPipelineRef = useRef<() => void>(() => {});
  const flushPipeline = useCallback(() => {
    const state = useEditorStore.getState();
    const queue = applyQueueRef.current;
    const job = queue.take(state.selectedImage?.path, state.portraitStatus.generation);
    if (!job) return;
    if (import.meta.env.DEV)
      console.debug('[portrait-preview] dispatch', {
        dragging: job.dragging,
        queueWaitMs: performance.now() - job.queuedAt,
      });
    void executeApplyRef
      .current(job.adjustments, job.dragging, job.targetRes, job.queuedAt)
      .then(
        (committed) => queue.finish(job, !!committed),
        () => queue.finish(job, false),
      )
      .finally(() => {
        if (queue.hasPending()) queueMicrotask(() => flushPipelineRef.current());
      });
  }, []);
  flushPipelineRef.current = flushPipeline;

  const applyAdjustments = useCallback(
    (currentAdjustments: Adjustments, dragging: boolean = false, targetRes?: number) => {
      if (!selectedImage?.isReady) return;
      // 拖动与松手一律进单飞队列：松手不再直冲 invoke，否则会取消正在算的重塑预览。
      const job = makeApplyJob(currentAdjustments, dragging, targetRes);
      if (!job) return;
      applyQueueRef.current.offer(job);
      flushPipeline();
    },
    [selectedImage?.isReady, flushPipeline, makeApplyJob],
  );

  const throttledDragApply = useMemo(
    () =>
      throttle(
        (adj: Adjustments, targetRes?: number) => {
          applyAdjustments(adj, true, targetRes);
        },
        30,
        { leading: true, trailing: true },
      ),
    [applyAdjustments],
  );

  const throttledUncroppedPreview = useMemo(
    () =>
      throttle(
        (adj: Adjustments, path: string, generation: number) => {
          const current = useEditorStore.getState();
          if (
            !current.selectedImage?.isReady ||
            current.selectedImage.path !== path ||
            current.portraitStatus.generation !== generation
          )
            return;
          // 人像分析未就绪时先中性化人像参数，让未裁剪预览即时生成；
          // 分析完成后 portraitStatus.phase 变化会重新生成含人像效果的版本。
          let renderAdjustments = adj;
          const retouchPending =
            hasPortraitEffect(adj) &&
            (current.portraitStatus.path !== path ||
              current.portraitStatus.source !== portraitSourceSignature(adj) ||
              !['ready', 'empty', 'partial'].includes(current.portraitStatus.phase));
          const reshapePending = hasPortraitReshapePreviewEffect(adj) && reshapePreviewAllowedRef.current !== path;
          if (retouchPending || reshapePending) {
            renderAdjustments = structuredClone(adj);
            if (retouchPending) {
              renderAdjustments.portrait = deferredPortraitRetouch();
            }
            if (reshapePending) {
              renderAdjustments.portraitReshape = deferredPortraitReshape(adj.portraitReshape);
            }
          } else {
            renderAdjustments = structuredClone(adj);
          }
          delete (renderAdjustments as any).frame;
          const jobId = ++uncroppedJobIdRef.current;
          invoke<string>(Invokes.GenerateUncroppedPreview, { jsAdjustments: renderAdjustments, path })
            .then((dataUrl) => {
              const latest = useEditorStore.getState();
              if (latest.selectedImage?.path !== path || latest.portraitStatus.generation !== generation) return;
              if (jobId >= latestUncroppedJobIdRef.current) {
                latestUncroppedJobIdRef.current = jobId;
                useEditorStore.getState().setEditor({ uncroppedAdjustedPreviewUrl: dataUrl });
              }
            })
            .catch(console.error);
        },
        30,
        { leading: true, trailing: true },
      ),
    [],
  );

  const generateUncroppedPreview = useCallback(
    (currentAdjustments: Adjustments) => {
      const current = useEditorStore.getState();
      if (current.selectedImage?.isReady) {
        throttledUncroppedPreview(currentAdjustments, current.selectedImage.path, current.portraitStatus.generation);
      }
    },
    [throttledUncroppedPreview],
  );

  useEffect(() => {
    if (activeView === 'editor' && activePanel === Panel.Crop && selectedImage?.isReady) {
      generateUncroppedPreview(adjustments);
    }
  }, [
    activeView,
    adjustments,
    activePanel,
    selectedImage?.path,
    selectedImage?.isReady,
    portraitStatus.phase,
    portraitReshapeKey,
    generateUncroppedPreview,
  ]);

  useEffect(
    () => () => {
      throttledUncroppedPreview.cancel();
      throttledDragApply.cancel();
      applyQueueRef.current.clearPending();
    },
    [throttledUncroppedPreview, throttledDragApply, selectedImage?.path],
  );

  useEffect(() => {
    portraitVisPrefetchGenRef.current += 1;
    clearPortraitVisCache();
    reshapePreviewAllowedRef.current = null;
    const path = selectedImage?.path ?? null;
    if (committedPortraitPathRef.current !== path || !selectedImage?.isReady) committedPortraitPathRef.current = null;
    const adj = useEditorStore.getState().adjustments;
    const state = useEditorStore.getState();
    const reshapeActive = !!(path && selectedImage?.isReady && hasPortraitReshapePreviewEffect(adj));
    const retouchActive = !!(path && selectedImage?.isReady && hasPortraitEffect(adj));
    const hasExactCachedFrame =
      !!state.finalPreviewUrl &&
      (!retouchActive ||
        (state.portraitStatus.path === path &&
          state.portraitStatus.analysisSignature === `${path}|${portraitSource}|${portraitSettings}` &&
          ['ready', 'empty', 'partial'].includes(state.portraitStatus.phase))) &&
      (!reshapeActive ||
        (state.portraitReshapeStatus.path === path &&
          ['ready', 'partial'].includes(state.portraitReshapeStatus.phase)));
    // 切图／就绪即挂角标，避免首帧 Apply 的短暂空窗；完整效果落地后由 commit 清除。
    setEditor({
      portraitInitialApply:
        (reshapeActive || retouchActive) && !hasExactCachedFrame
          ? {
              path,
              pending: true,
              kind: resolvePortraitInitialApplyKind(adj, reshapeActive, retouchActive),
            }
          : null,
    });
  }, [
    selectedImage?.path,
    selectedImage?.isReady,
    portraitReshapeRetry,
    portraitSource,
    portraitSettings,
    clearPortraitVisCache,
    setEditor,
  ]);

  const calculateTargetRes = useCallback(() => {
    const baseTargetRes = appSettings?.editorPreviewResolution || 1920;
    if (!(appSettings?.enableZoomHifi ?? true) || displaySize.width === 0) {
      return baseTargetRes;
    }

    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const sharpnessFactor = 1.25;
    const zoomMultiplier = appSettings?.highResZoomMultiplier || 1.0;
    const effectiveDpr = appSettings?.useFullDpiRendering ? dpr : 1;

    let targetRes = Math.max(displaySize.width, displaySize.height) * effectiveDpr * sharpnessFactor * zoomMultiplier;
    targetRes = Math.max(targetRes, 512);

    if (originalSize && originalSize.width > 0 && originalSize.height > 0) {
      const origMax = Math.max(originalSize.width, originalSize.height);
      targetRes = Math.min(targetRes, origMax);
      if (targetRes >= origMax * 0.8) {
        targetRes = origMax;
      }
    }

    if (originalSize && targetRes !== Math.max(originalSize.width, originalSize.height)) {
      targetRes = Math.ceil(targetRes / 256) * 256;
    }

    return Math.round(targetRes);
  }, [
    appSettings?.enableZoomHifi,
    appSettings?.editorPreviewResolution,
    appSettings?.highResZoomMultiplier,
    appSettings?.useFullDpiRendering,
    displaySize.width,
    displaySize.height,
    originalSize,
  ]);

  const settleDelay = previewSettleDelay(previewOverride ?? adjustments);
  const requestHiFiZoom = useMemo(
    () =>
      debounce((targetRes: number) => {
        if (useEditorStore.getState().isSliderDragging) return;
        if (targetRes > currentResRef.current) {
          currentResRef.current = targetRes;
          const { adjustments, previewOverride } = useEditorStore.getState();
          const renderAdjustments = previewOverride ?? adjustments;
          applyAdjustments(renderAdjustments, false, targetRes);
        }
      }, settleDelay),
    [applyAdjustments, currentResRef, settleDelay],
  );

  useEffect(() => {
    if (activeView === 'editor' && selectedImage?.isReady && displaySize.width > 0 && !isSliderDragging) {
      let baseRes = calculateTargetRes();
      if (originalSize.width > 0 && originalSize.height > 0) {
        const maxRes = Math.max(originalSize.width, originalSize.height);
        if (baseRes > maxRes) baseRes = maxRes;
      }
      const finalRes = Math.round(baseRes);

      if (finalRes > currentResRef.current) {
        requestHiFiZoom(finalRes);
      }
    }
    return () => {
      requestHiFiZoom.cancel();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    activeView,
    displaySize.width,
    displaySize.height,
    calculateTargetRes,
    selectedImage?.isReady,
    isSliderDragging,
    requestHiFiZoom,
    originalSize,
  ]);

  useEffect(() => {
    if (!selectedImage?.isReady) return;

    if (dragIdleTimer.current) clearTimeout(dragIdleTimer.current);

    const targetRes = calculateTargetRes();
    const renderAdjustments = previewOverride ?? adjustments;

    if (activeView !== 'editor') {
      if (isSliderDragging) return;
    }

    if (isSliderDragging) {
      requestHiFiZoom.cancel();
      applyQueueRef.current.dropPendingHighQuality();
      if (appSettings?.enableLivePreviews !== false) {
        // 人像交互只发送代理分辨率；松手分支仍提交完整 targetRes 做一次高质量精化。
        // 后端 GPUPixel 同样会在 bake 前缩到该尺寸，避免拖动每一档都处理整张原图。
        const portraitInteractive =
          hasPortraitEffect(renderAdjustments) || hasPortraitReshapePreviewEffect(renderAdjustments);
        throttledDragApply(renderAdjustments, portraitInteractive ? Math.min(targetRes, 960) : targetRes);
      }
    } else {
      throttledDragApply.cancel();
      dragIdleTimer.current = setTimeout(() => {
        currentResRef.current = targetRes;

        if (previewOverride) {
          applyAdjustments(renderAdjustments, false, targetRes);
          return;
        }

        const prev = prevAdjustmentsRef.current;
        const prevAdj = prev && prev.path === selectedImage.path ? (prev.adjustments as Adjustments) : null;

        // 仅一个人像分区小眼睛翻转且已有对应帧：即时换 blob，跳过整图 Apply。
        if (trySwapPortraitVisCache(selectedImage.path, adjustments, prevAdj)) {
          if (!prevAdj || prevAdj !== adjustments) {
            debouncedSave(selectedImage.path, adjustments);
            prevAdjustmentsRef.current = { path: selectedImage.path, adjustments };
          }
          return;
        }

        applyAdjustments(renderAdjustments, false, targetRes);

        if (!prev || prev.path !== selectedImage.path) {
          prevAdjustmentsRef.current = { path: selectedImage.path, adjustments };
          return;
        }

        const hasAdjustmentsChanged = prev.adjustments !== adjustments;

        if (hasAdjustmentsChanged) {
          debouncedSave(selectedImage.path, adjustments);

          const otherPaths = multiSelectedPaths.filter((p) => p !== selectedImage.path);
          if (appSettings?.copyPasteSettings?.autoSync && otherPaths.length > 0) {
            const delta: Partial<Adjustments> = {};
            const includedKeys = appSettings?.copyPasteSettings?.includedAdjustments || COPYABLE_ADJUSTMENT_KEYS;
            for (const key of Object.keys(adjustments) as Array<keyof Adjustments>) {
              if (includedKeys.includes(key as string)) {
                if (JSON.stringify(adjustments[key]) !== JSON.stringify(prev.adjustments[key])) {
                  (delta as any)[key] = adjustments[key];
                }
              }
            }
            if (Object.keys(delta).length > 0) {
              otherPaths.forEach((p) => globalImageCache.delete(p));
              invoke(Invokes.ApplyAdjustmentsToPaths, { paths: otherPaths, adjustments: delta }).catch((err) => {
                console.error('Failed to apply adjustments to multi-selection:', err);
              });
            }
          }

          prevAdjustmentsRef.current = { path: selectedImage.path, adjustments };
        }
      }, previewSettleDelay(renderAdjustments));
    }

    return () => {
      if (dragIdleTimer.current) clearTimeout(dragIdleTimer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    activeView,
    adjustments,
    previewOverride,
    selectedImage?.path,
    selectedImage?.isReady,
    isSliderDragging,
    multiSelectedPaths,
    appSettings?.enableLivePreviews,
    appSettings?.copyPasteSettings?.includedAdjustments,
    appSettings?.copyPasteSettings?.autoSync,
    isWaveformVisible,
    trySwapPortraitVisCache,
    throttledDragApply,
  ]);

  // 人像分析／重塑就绪后单独补全效果，避免与每次调色变化耦合成双 Apply。
  useEffect(() => {
    if (!selectedImage?.isReady || isSliderDragging) return;
    if (activeView !== 'editor' && !previewOverride) return;
    const renderAdjustments = previewOverride ?? adjustments;
    const retouchReady =
      hasPortraitEffect(renderAdjustments) && ['ready', 'empty', 'partial'].includes(portraitStatus.phase);
    const reshapeReady = hasPortraitReshapePreviewEffect(renderAdjustments) && !!portraitReshapeKey;
    if (!retouchReady && !reshapeReady) return;
    const targetRes = calculateTargetRes();
    applyAdjustments(renderAdjustments, false, targetRes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portraitStatus.phase, portraitReshapeKey]);

  return {
    applyAdjustments,
    executeApplyAdjustments,
  };
}

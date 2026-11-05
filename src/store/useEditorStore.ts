import { create } from 'zustand';
import {
  Adjustments,
  INITIAL_ADJUSTMENTS,
  MaskContainer,
  AiPatch,
  PortraitReshapeReason,
  ReshapeSourceAnchor,
  clearPortraitReshapeSnapshot,
} from '../utils/adjustments';
import {
  SelectedImage,
  WaveformData,
  BrushSettings,
  RenderStamp,
  PreviewMetadata,
  ReshapeAttenuation,
  sameRenderStamp,
  isRenderStamp,
} from '../components/ui/AppProperties';
import { ChannelConfig } from '../components/adjustments/Curves';
import { ImageDimensions } from '../hooks/useImageRenderSize';
import { ToolType } from '../components/panel/right/Masks';
import { OverlayMode } from '../components/panel/right/CropPanel';

interface InteractivePatch {
  url: string;
  normX: number;
  normY: number;
  normW: number;
  normH: number;
}

interface BaseRenderSize extends ImageDimensions {
  containerHeight: number;
  containerWidth: number;
  offsetX: number;
  offsetY: number;
}

export interface PortraitSummary {
  key: string;
  faceCount: number;
  skippedFaces: number;
  hasSkin: boolean;
}

export interface PortraitStatus {
  path: string | null;
  generation: number;
  source: string;
  /** 路径、人像基底和预处理设置的完整签名，用于切回图片时安全复用摘要。 */
  analysisSignature?: string;
  phase: 'idle' | 'loading' | 'analyzing' | 'ready' | 'empty' | 'partial' | 'error';
  summary?: PortraitSummary;
  error?: string;
}

/** 刚打开图像时，人像/重塑效果尚未完整落地前的预览角标状态。 */
export type PortraitInitialApplyKind = 'reshape' | 'retouch' | 'portrait';

export interface PortraitInitialApply {
  path: string;
  pending: boolean;
  kind: PortraitInitialApplyKind;
}

/** 人像分区滑块/可见性调节后的预览角标（统一显示在预览左下角）。 */
export type PortraitApplyHintKind = 'reshape' | 'effects';
export type PortraitApplyHintPhase = 'applying' | 'applied';

export interface PortraitApplyHint {
  path: string;
  kind: PortraitApplyHintKind;
  phase: PortraitApplyHintPhase;
}

export interface PortraitReshapeFaceSummary {
  faceId: string;
  sourceAnchor: ReshapeSourceAnchor;
  processable: boolean;
  reason?: PortraitReshapeReason;
}

export interface PortraitReshapeRequest {
  path: string;
  generation: number;
  source: string;
}

export interface PortraitReshapeCacheStats {
  bytes: number;
  entries: number;
  inFlight: number;
  hits: number;
  computations: number;
}

export interface PortraitReshapeAnalysisSummary {
  request: PortraitReshapeRequest;
  key: string;
  faces: PortraitReshapeFaceSummary[];
  rejectedFaces: number;
  reason: PortraitReshapeReason | null;
  cache: PortraitReshapeCacheStats;
  // 11：true 表示本次摘要由有效快照重建（未做全量检测）；false 时清除失配的旧快照。
  snapshotReused: boolean;
}

export interface PortraitReshapeStatus {
  path: string | null;
  generation: number;
  source: string;
  /** 不含滑块强度的分析作用域；切回同图时据此复用 faces/key。 */
  analysisScopeKey?: string;
  phase: 'idle' | 'loading' | 'analyzing' | 'ready' | 'partial' | 'error';
  faces: PortraitReshapeFaceSummary[];
  selectedFaceId: string | null;
  reason?: PortraitReshapeReason;
  key?: string;
  rejectedFaces?: number;
  cache?: PortraitReshapeCacheStats;
  error?: string;
  /** 最近一次预览中非零请求被安全回退限幅或压到恒等。 */
  renderAttenuation?: ReshapeAttenuation | null;
  /** 最近一次预览中身体塑形引擎旁路；与人脸重塑限幅隔离。 */
  bodyAttenuation?: ReshapeAttenuation | null;
}

export const emptyPortraitReshapeStatus = (path: string | null, generation: number): PortraitReshapeStatus => ({
  path,
  generation,
  source: '',
  phase: 'idle',
  faces: [],
  selectedFaceId: null,
});

export const matchesPortraitReshapeRequest = (
  state: Pick<EditorState, 'selectedImage' | 'portraitReshapeStatus'>,
  request: PortraitReshapeRequest,
): boolean =>
  state.selectedImage?.path === request.path &&
  state.selectedImage.isReady &&
  state.portraitReshapeStatus.path === request.path &&
  state.portraitReshapeStatus.generation === request.generation &&
  state.portraitReshapeStatus.source === request.source;

export const applyPortraitReshapeSummary = (
  state: Pick<EditorState, 'selectedImage' | 'portraitReshapeStatus' | 'adjustments'>,
  summary: PortraitReshapeAnalysisSummary,
): Partial<EditorState> => {
  if (!matchesPortraitReshapeRequest(state, summary.request)) return {};
  // 最多 32 个检测目标与 32 个未匹配的已保存目标；不接受完整关键点或概率图进入 store。
  if (summary.faces.length > 64 || new Set(summary.faces.map((face) => face.faceId)).size !== summary.faces.length)
    return { portraitReshapeStatus: { ...state.portraitReshapeStatus, phase: 'error', reason: 'model-error' } };
  const faces = summary.faces.map(({ faceId, sourceAnchor, processable, reason }) => ({
    faceId,
    sourceAnchor,
    processable,
    reason,
  }));
  const selectedFaceId = faces.some(
    (face) => face.faceId === state.portraitReshapeStatus.selectedFaceId && face.processable,
  )
    ? state.portraitReshapeStatus.selectedFaceId
    : (faces.find((face) => face.processable)?.faceId ?? null);
  // 全局强度清零／idle 暂存迁移由调用方 migrateStagedReshapeStrengths 完成，避免循环依赖。
  const portraitReshape = summary.snapshotReused
    ? state.adjustments.portraitReshape
    : clearPortraitReshapeSnapshot(state.adjustments.portraitReshape);
  return {
    ...(portraitReshape !== state.adjustments.portraitReshape
      ? { adjustments: { ...state.adjustments, portraitReshape } }
      : {}),
    portraitReshapeStatus: {
      ...state.portraitReshapeStatus,
      phase: summary.rejectedFaces > 0 || faces.some((face) => !face.processable) ? 'partial' : 'ready',
      key: summary.key,
      faces,
      selectedFaceId,
      rejectedFaces: summary.rejectedFaces,
      reason: summary.reason ?? undefined,
      cache: summary.cache,
      error: undefined,
    },
  };
};

// 选择仅存在于运行态；只有当前照片、当前代次的可处理脸可被选中，不写入 adjustments 或历史。
// 单人脸模式：不允许清空选择（null）；无有效 id 时落到第一张可处理脸。
export const selectPortraitReshapeFace = (
  state: Pick<EditorState, 'selectedImage' | 'portraitReshapeStatus'>,
  faceId: string | null,
): Partial<EditorState> => {
  const status = state.portraitReshapeStatus;
  if (status.path !== state.selectedImage?.path || !state.selectedImage.isReady) return {};
  const resolved =
    faceId !== null && status.faces.some((entry) => entry.faceId === faceId && entry.processable)
      ? faceId
      : (status.faces.find((entry) => entry.processable)?.faceId ?? null);
  if (resolved === status.selectedFaceId) return {};
  return { portraitReshapeStatus: { ...status, selectedFaceId: resolved } };
};

export interface PreviewAnalytics {
  path: string;
  stamp: RenderStamp;
  histogram: ChannelConfig | null;
  waveform: WaveformData | null;
  /** Pre-levels (identity levels) histogram for the Levels panel; omit to keep prior. */
  levelsInputHistogram?: ChannelConfig | null;
}

export const matchesPreviewRequest = (
  state: Pick<EditorState, 'selectedImage' | 'previewRequest' | 'previewError'>,
  path: string,
  requestId: number,
): boolean =>
  state.selectedImage?.isReady === true &&
  state.selectedImage.path === path &&
  state.previewRequest?.path === path &&
  state.previewRequest.requestId === requestId &&
  state.previewError === null;

/** 当前选中图的预览请求尚未被 stamp 对齐，且无错误 → 人像效果仍在应用中。 */
export const isPortraitPreviewBusy = (
  state: Pick<EditorState, 'selectedImage' | 'previewRequest' | 'previewStamp' | 'previewError'>,
): boolean => {
  const path = state.selectedImage?.path;
  const req = state.previewRequest;
  if (!path || !req || req.path !== path || state.previewError !== null) return false;
  return state.previewStamp?.requestId !== req.requestId;
};

export const acceptPreviewFrame = (state: EditorState, metadata: PreviewMetadata): Partial<EditorState> => {
  if (!matchesPreviewRequest(state, metadata.path, metadata.stamp.requestId)) return {};
  if (
    state.previewStamp?.requestId === metadata.stamp.requestId &&
    !sameRenderStamp(state.previewStamp, metadata.stamp)
  )
    return {};
  const analytics = state.pendingPreviewAnalytics;
  const matching = analytics && sameRenderStamp(analytics.stamp, metadata.stamp);
  const attenuation = metadata.reshapeAttenuation ?? null;
  const bodyAttenuation = metadata.bodyAttenuation ?? null;
  const samePhoto = state.selectedImage?.path === metadata.path;
  const sameReshapePath = state.portraitReshapeStatus.path === metadata.path;
  const reshapeUpdate = samePhoto
    ? {
        portraitReshapeStatus: {
          ...state.portraitReshapeStatus,
          ...(sameReshapePath ? { renderAttenuation: attenuation } : {}),
          bodyAttenuation,
        },
      }
    : {};
  return {
    previewStamp: metadata.stamp,
    pendingPreviewAnalytics: null,
    histogram: matching ? analytics.histogram : state.histogram,
    waveform: matching ? analytics.waveform : state.waveform,
    levelsInputHistogram:
      matching && analytics.levelsInputHistogram !== undefined
        ? analytics.levelsInputHistogram
        : state.levelsInputHistogram,
    hasRenderedFirstFrame: true,
    ...reshapeUpdate,
  };
};

export const acceptPreviewAnalytics = (state: EditorState, analytics: PreviewAnalytics): Partial<EditorState> => {
  if (!isRenderStamp(analytics.stamp) || !matchesPreviewRequest(state, analytics.path, analytics.stamp.requestId))
    return {};
  if (sameRenderStamp(state.previewStamp, analytics.stamp)) {
    return {
      histogram: analytics.histogram ?? null,
      waveform: analytics.waveform ?? null,
      ...(analytics.levelsInputHistogram !== undefined
        ? { levelsInputHistogram: analytics.levelsInputHistogram }
        : {}),
      pendingPreviewAnalytics: null,
    };
  }
  // 帧已经提交但 stamp 不同的统计是过期数据，不能暂存成下一帧的候选。
  if (state.previewStamp?.requestId === analytics.stamp.requestId) return {};
  return { pendingPreviewAnalytics: analytics };
};

const INVALIDATED_PREVIEW = {
  previewRequest: null,
  pendingPreviewAnalytics: null,
  previewError: null,
  histogram: null,
  waveform: null,
};

/** Keys that do not change the pre-levels input image (levels itself + post-levels curves). */
const LEVELS_INPUT_NEUTRAL_KEYS = new Set([
  'levels',
  'curves',
  'pointCurves',
  'parametricCurve',
  'curveMode',
]);

function stripLevelsInputNeutral(adjustments: Adjustments): unknown {
  const clone: Record<string, unknown> = { ...(adjustments as unknown as Record<string, unknown>) };
  for (const key of LEVELS_INPUT_NEUTRAL_KEYS) delete clone[key];
  if (Array.isArray(clone.masks)) {
    clone.masks = clone.masks.map((mask: MaskContainer) => {
      const adj = { ...(mask.adjustments as unknown as Record<string, unknown>) };
      for (const key of LEVELS_INPUT_NEUTRAL_KEYS) delete adj[key];
      return { ...mask, adjustments: adj };
    });
  }
  return clone;
}

/** True when upstream-of-levels adjustments changed (exposure etc.), so levels input hist should refresh. */
export function adjustmentsAffectLevelsInputHistogram(
  prev: Adjustments | null | undefined,
  next: Adjustments | null | undefined,
): boolean {
  if (!next) return false;
  if (!prev) return true;
  try {
    return JSON.stringify(stripLevelsInputNeutral(prev)) !== JSON.stringify(stripLevelsInputNeutral(next));
  } catch {
    return true;
  }
}

interface EditorState {
  // Core Image & Adjustments
  selectedImage: SelectedImage | null;
  adjustments: Adjustments;
  previewOverride: Adjustments | null;

  // History State
  history: Adjustments[];
  historyIndex: number;

  // Previews & Overlays
  finalPreviewUrl: string | null;
  uncroppedAdjustedPreviewUrl: string | null;
  interactivePatch: InteractivePatch | null;
  showOriginal: boolean;

  // 当前请求与最后完整发布帧分别记录；统计早到最多暂存一份。
  previewRequest: { path: string; requestId: number } | null;
  previewStamp: RenderStamp | null;
  previewError: string | null;
  pendingPreviewAnalytics: PreviewAnalytics | null;

  // Analytics
  histogram: ChannelConfig | null;
  /** Capture One–style pre-levels histogram for the Levels panel only. */
  levelsInputHistogram: ChannelConfig | null;
  waveform: WaveformData | null;
  isWaveformVisible: boolean;
  activeWaveformChannel: string;
  waveformHeight: number;

  // Interaction State
  isSliderDragging: boolean;
  zoom: number;
  displaySize: ImageDimensions;
  previewSize: ImageDimensions;
  baseRenderSize: BaseRenderSize;
  originalSize: ImageDimensions;

  // Tools State
  isRotationActive: boolean;
  overlayMode: OverlayMode;
  overlayRotation: number;
  isStraightenActive: boolean;
  isWbPickerActive: boolean;
  isGuidedPerspectiveActive: boolean;
  liveRotation: number | null;
  brushSettings: BrushSettings | null;

  // Masks & AI
  activeMaskContainerId: string | null;
  activeMaskId: string | null;
  activeAiPatchContainerId: string | null;
  activeAiSubMaskId: string | null;
  isMaskControlHovered: boolean;
  isGeneratingAiMask: boolean;
  isGeneratingAi: boolean;
  isAIConnectorConnected: boolean;
  hasRenderedFirstFrame: boolean;
  patchesSentToBackend: Set<string>;
  portraitStatus: PortraitStatus;
  portraitRetry: number;
  portraitReshapeStatus: PortraitReshapeStatus;
  portraitReshapeRetry: number;
  /** 首开渐进 Apply：效果 Bypass/分析中时 pending，完整帧落地后清空。 */
  portraitInitialApply: PortraitInitialApply | null;
  /** 分区调节时的 applying/applied 角标；由 PortraitPanel 写入，Editor 渲染。 */
  portraitApplyHint: PortraitApplyHint | null;

  // Clipboard
  copiedSectionAdjustments: any | null;
  copiedMask: MaskContainer | null;
  copiedAdjustments: Adjustments | null;

  // Actions
  setEditor: (updater: Partial<EditorState> | ((state: EditorState) => Partial<EditorState>)) => void;
  selectPortraitReshapeFace: (faceId: string | null) => void;
  pushHistory: (newAdjustments: Adjustments) => void;
  undo: () => void;
  redo: () => void;
  resetHistory: (initialState: Adjustments) => void;
  goToHistoryIndex: (index: number) => void;
}

export const useEditorStore = create<EditorState>((set) => ({
  selectedImage: null,
  adjustments: INITIAL_ADJUSTMENTS,
  previewOverride: null,
  history: [INITIAL_ADJUSTMENTS],
  historyIndex: 0,

  previewRequest: null,
  previewStamp: null,
  previewError: null,
  pendingPreviewAnalytics: null,
  finalPreviewUrl: null,
  uncroppedAdjustedPreviewUrl: null,
  showOriginal: false,
  histogram: null,
  levelsInputHistogram: null,
  waveform: null,
  isWaveformVisible: true,
  activeWaveformChannel: 'histogram',
  waveformHeight: 220,

  isSliderDragging: false,
  interactivePatch: null,
  activeMaskContainerId: null,
  activeMaskId: null,
  activeAiPatchContainerId: null,
  activeAiSubMaskId: null,

  zoom: 1,
  displaySize: { width: 0, height: 0 },
  previewSize: { width: 0, height: 0 },
  baseRenderSize: { width: 0, height: 0, offsetX: 0, offsetY: 0, containerWidth: 0, containerHeight: 0 },
  originalSize: { width: 0, height: 0 },

  isRotationActive: false,
  overlayMode: 'thirds',
  overlayRotation: 0,
  isStraightenActive: false,
  isWbPickerActive: false,
  isGuidedPerspectiveActive: false,
  liveRotation: null,

  copiedSectionAdjustments: null,
  copiedMask: null,
  brushSettings: { size: 50, feather: 50, tool: ToolType.Brush },
  copiedAdjustments: null,

  isGeneratingAiMask: false,
  isAIConnectorConnected: false,
  isGeneratingAi: false,
  isMaskControlHovered: false,
  hasRenderedFirstFrame: false,
  patchesSentToBackend: new Set<string>(),
  portraitStatus: { path: null, generation: 0, source: '', phase: 'idle' },
  portraitRetry: 0,
  portraitReshapeStatus: emptyPortraitReshapeStatus(null, 0),
  portraitReshapeRetry: 0,
  portraitInitialApply: null,
  portraitApplyHint: null,

  setEditor: (updater) =>
    set((state) => {
      const change = typeof updater === 'function' ? updater(state) : updater;
      if (
        Object.hasOwn(change, 'selectedImage') &&
        (change.selectedImage?.path !== state.selectedImage?.path ||
          (state.selectedImage?.isReady && !change.selectedImage?.isReady))
      ) {
        return {
          ...change,
          ...INVALIDATED_PREVIEW,
          levelsInputHistogram: null,
          previewStamp: null,
          portraitInitialApply: null,
          portraitApplyHint: null,
          portraitReshapeStatus: emptyPortraitReshapeStatus(
            change.selectedImage?.path ?? null,
            state.portraitReshapeStatus.generation + 1,
          ),
        };
      }
      if (
        (Object.hasOwn(change, 'adjustments') && change.adjustments !== state.adjustments) ||
        (Object.hasOwn(change, 'previewOverride') && change.previewOverride !== state.previewOverride)
      ) {
        const prevEffective = state.previewOverride ?? state.adjustments;
        const nextEffective =
          (Object.hasOwn(change, 'previewOverride')
            ? change.previewOverride
            : state.previewOverride) ??
          (Object.hasOwn(change, 'adjustments') ? change.adjustments : state.adjustments) ??
          state.adjustments;
        const clearLevelsInput = adjustmentsAffectLevelsInputHistogram(prevEffective, nextEffective);
        return {
          ...change,
          ...INVALIDATED_PREVIEW,
          ...(clearLevelsInput ? { levelsInputHistogram: null } : {}),
        };
      }
      return change;
    }),

  selectPortraitReshapeFace: (faceId) => set((state) => selectPortraitReshapeFace(state, faceId)),

  pushHistory: (newAdj) =>
    set((state) => {
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push(newAdj);
      if (newHistory.length > 50) newHistory.shift();
      return { history: newHistory, historyIndex: newHistory.length - 1 };
    }),

  undo: () =>
    set((state) => {
      if (state.historyIndex > 0) {
        const newIndex = state.historyIndex - 1;
        const next = state.history[newIndex];
        return {
          ...INVALIDATED_PREVIEW,
          ...(adjustmentsAffectLevelsInputHistogram(state.adjustments, next)
            ? { levelsInputHistogram: null }
            : {}),
          historyIndex: newIndex,
          adjustments: next,
        };
      }
      return state;
    }),

  redo: () =>
    set((state) => {
      if (state.historyIndex < state.history.length - 1) {
        const newIndex = state.historyIndex + 1;
        const next = state.history[newIndex];
        return {
          ...INVALIDATED_PREVIEW,
          ...(adjustmentsAffectLevelsInputHistogram(state.adjustments, next)
            ? { levelsInputHistogram: null }
            : {}),
          historyIndex: newIndex,
          adjustments: next,
        };
      }
      return state;
    }),

  resetHistory: (initialState) =>
    set({
      ...INVALIDATED_PREVIEW,
      levelsInputHistogram: null,
      history: [initialState],
      historyIndex: 0,
      adjustments: initialState,
    }),

  goToHistoryIndex: (index) =>
    set((state) => {
      if (index >= 0 && index < state.history.length) {
        const next = state.history[index];
        return {
          ...INVALIDATED_PREVIEW,
          ...(adjustmentsAffectLevelsInputHistogram(state.adjustments, next)
            ? { levelsInputHistogram: null }
            : {}),
          historyIndex: index,
          adjustments: next,
        };
      }
      return state;
    }),
}));

import { useEffect, useMemo } from 'react';
import { Channel, invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import {
  useEditorStore,
  applyPortraitReshapeSummary,
  emptyPortraitReshapeStatus,
  matchesPortraitReshapeRequest,
  type PortraitReshapeAnalysisSummary,
  type PortraitReshapeRequest,
  type PortraitReshapeStatus,
} from '../store/useEditorStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { portraitReshapeSourceSignature, readPortraitReshape } from '../utils/adjustments';
import { migrateStagedReshapeStrengths } from '../utils/portraitReshapePanel';

export const reshapePreprocessingSignature = (): string => {
  const settings = useSettingsStore.getState().appSettings;
  return JSON.stringify([
    settings?.rawHighlightCompression,
    settings?.linearRawMode,
    settings?.rawPreprocessingColorNr,
    settings?.rawPreprocessingSharpening,
    settings?.applyPreprocessingToNonRaws,
  ]);
};

// 04 仅开放开发构建中的内部分析，09 才提供正式逐脸入口。
export function usePortraitReshapeAnalysis(isEditor: boolean, preprocessingSignature: string) {
  const selectedImage = useEditorStore((state) => state.selectedImage);
  const adjustments = useEditorStore((state) => state.adjustments);
  const retry = useEditorStore((state) => state.portraitReshapeRetry);
  const setEditor = useEditorStore((state) => state.setEditor);
  const parameters = useMemo(
    () => readPortraitReshape(adjustments.portraitReshape).settings,
    [adjustments.portraitReshape],
  );
  // 分区显隐只影响渲染（sectionVisibility → RenderIntent::Bypass），不参与分析生命周期。
  // 若把 portraitReshape 可见性绑进 enabled，关小眼睛会 empty 状态并重跑 PreparePortraitReshapeAnalysis，
  // 再开时预览极慢；与 skin／blemishes（portraitSourceSignature 不含分区显隐）对齐。
  // 正式入口已开放：不再限制 DEV，否则 release 构建永远无分析脸、滑块无法写入。
  const enabled = isEditor && parameters?.enabled === true;
  const baseSignature = enabled ? portraitReshapeSourceSignature(adjustments) : '';
  // 身份改动只重新请求匹配摘要；强度和保护笔刷不触发基础分析。
  const identitySignature = JSON.stringify(
    enabled ? parameters?.faces.map(({ faceId, sourceAnchor }) => ({ faceId, sourceAnchor })) : [],
  );
  // 路径／基底／预处理／重试变化必须重分析；仅 identitySignature 因滑块写入 override 而变时可复用。
  // 不含 sectionVisibility：关／开小眼睛必须复用已有 faces／key，避免 bump generation 取消在途预览。
  const analysisScopeKey = JSON.stringify([
    selectedImage?.path ?? null,
    selectedImage?.isReady === true,
    enabled,
    baseSignature,
    preprocessingSignature,
    retry,
  ]);
  useEffect(() => {
    const state = useEditorStore.getState();
    const path = selectedImage?.path ?? null;
    const previous = state.portraitReshapeStatus.path === path ? state.portraitReshapeStatus.faces : [];
    const status = state.portraitReshapeStatus;
    const scopeUnchanged = status.analysisScopeKey === analysisScopeKey;
    // 滑块首次为已分析人脸写入 override 会让 faces 从 [] 变为有条目，identitySignature 变化。
    // 若作用域未变且锚点已与当前 ready/partial 摘要一致，禁止 bump generation／清空 faces，
    // 否则会取消进行中的预览（「重塑分析请求已过期」）并让后续滑块因无 processable 脸而写不进去。
    if (
      scopeUnchanged &&
      enabled &&
      path &&
      selectedImage?.isReady &&
      status.path === path &&
      (status.phase === 'ready' || status.phase === 'partial') &&
      (parameters?.faces ?? []).every((face) =>
        status.faces.some(
          (known) =>
            known.faceId === face.faceId &&
            known.sourceAnchor.sourceBaseHash === face.sourceAnchor.sourceBaseHash &&
            known.sourceAnchor.bounds.length === 4 &&
            face.sourceAnchor.bounds.length === 4 &&
            known.sourceAnchor.bounds.every((v, i) => v === face.sourceAnchor.bounds[i]),
        ),
      )
    ) {
      return;
    }
    const generation = status.generation + 1;
    setEditor({
      portraitReshapeStatus: {
        ...emptyPortraitReshapeStatus(path, generation),
        analysisScopeKey,
      },
    });
    if (!enabled || !path || !selectedImage?.isReady) return;

    let active = true;
    let request: PortraitReshapeRequest | null = null;
    const current = () => {
      const latest = useEditorStore.getState();
      return (
        active &&
        latest.selectedImage?.path === path &&
        latest.selectedImage.isReady &&
        latest.portraitReshapeStatus.generation === generation &&
        portraitReshapeSourceSignature(latest.adjustments) === baseSignature &&
        reshapePreprocessingSignature() === preprocessingSignature
      );
    };
    const update = (
      change: Pick<PortraitReshapeStatus, 'phase'> & Partial<Pick<PortraitReshapeStatus, 'reason' | 'error'>>,
    ) => {
      if (!current() || !request) return;
      const id = request;
      setEditor((latest) =>
        matchesPortraitReshapeRequest(latest, id)
          ? { portraitReshapeStatus: { ...latest.portraitReshapeStatus, ...change } }
          : {},
      );
    };

    const start = async () => {
      // IPC 只回传短哈希，不把有效像素补丁的序列化内容夹进每条进度消息。
      const bytes = new TextEncoder().encode(JSON.stringify([baseSignature, preprocessingSignature]));
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      if (!current()) return;
      request = {
        path,
        generation,
        source: Array.from(new Uint8Array(digest), (v) => v.toString(16).padStart(2, '0')).join(''),
      };
      const id = request;
      setEditor((latest) => ({
        portraitReshapeStatus: { ...latest.portraitReshapeStatus, source: id.source, phase: 'loading' },
      }));
      const progress = new Channel<{ request: PortraitReshapeRequest; phase: 'loading' | 'analyzing' }>();
      progress.onmessage = (message) => {
        if (
          message.request.path === id.path &&
          message.request.source === id.source &&
          message.request.generation === id.generation &&
          (message.phase === 'loading' || message.phase === 'analyzing')
        )
          update({ phase: message.phase });
      };
      const payload = structuredClone(useEditorStore.getState().adjustments);
      const parsed = readPortraitReshape(payload.portraitReshape).settings;
      if (!parsed) return;
      const ids = new Set(parsed.faces.map((face) => face.faceId));
      for (const face of previous) {
        if (parsed.faces.length >= 32) break;
        if (!ids.has(face.faceId) && face.reason !== 'unmatched-face') {
          parsed.faces.push({
            faceId: face.faceId,
            sourceAnchor: face.sourceAnchor,
            overrides: {},
            protection: { hair: true, background: true, strokes: [] },
          });
          ids.add(face.faceId);
        }
      }
      // 只在本次请求副本中携带身份锚点；不隐式保存、改历史或修改效果参数。
      payload.portraitReshape = parsed;
      let summary: PortraitReshapeAnalysisSummary | null = null;
      // 前端缓存命中切回时后端 load_image 可能尚未完成（original_image 仍是上一张），
      // 后端返回「已过期」；不是模型失败，等待后端就绪后重试。
      for (let attempt = 0; summary === null; attempt += 1) {
        try {
          summary = await invoke<PortraitReshapeAnalysisSummary>(Invokes.PreparePortraitReshapeAnalysis, {
            request: id,
            jsAdjustments: payload,
            onProgress: progress,
          });
        } catch (error) {
          if (String(error) !== '重塑分析请求已过期' || attempt >= 40) throw error;
          await new Promise((resolve) => setTimeout(resolve, 250));
          if (!current()) return;
        }
      }
      if (!summary) return;
      const ready = summary;
      if (current())
        setEditor((latest) => {
          const partial = applyPortraitReshapeSummary(latest, ready);
          const faces = partial.portraitReshapeStatus?.faces;
          if (!faces) return partial;
          const baseAdjustments = {
            ...latest.adjustments,
            ...(partial.adjustments ?? {}),
          };
          const migrated = migrateStagedReshapeStrengths(baseAdjustments, faces);
          return { ...partial, adjustments: migrated };
        });
    };
    void start().catch((error: unknown) => {
      if (!current()) return;
      if (request) update({ phase: 'error', reason: 'model-error', error: String(error) });
      else
        setEditor((latest) => ({
          portraitReshapeStatus: { ...latest.portraitReshapeStatus, phase: 'error', reason: 'model-error' },
        }));
    });
    return () => {
      active = false;
      if (request)
        void invoke(Invokes.CancelPortraitReshapeAnalysis, { request }).catch(() => {
          // 前端守卫已停用；取消命令失败也不能接受迟到的结果。
        });
    };
  }, [
    selectedImage?.path,
    selectedImage?.isReady,
    enabled,
    baseSignature,
    identitySignature,
    preprocessingSignature,
    retry,
    setEditor,
  ]);
}

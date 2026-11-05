import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { build } from 'esbuild';

// 在内存中编译真实模块；不生成构建文件，也不调用 Tauri 或读写照片。
const compiled = await build({
  stdin: {
    contents: `
      export * from './src/utils/adjustments.ts';
      export { Panel, Invokes, decodePreviewResponse, isPreviewMetadata, sameRenderStamp } from './src/components/ui/AppProperties.tsx';
      export * from './src/store/useEditorStore.ts';
      export { useUIStore, reconcileWorkspace } from './src/store/useUIStore.ts';
      export { useSettingsStore } from './src/store/useSettingsStore.ts';
      export { useProcessStore } from './src/store/useProcessStore.ts';
      export { createEstimateRequestGuard } from './src/hooks/useExportSettings.ts';
      export { createPreviewApplyQueue, previewSettleDelay } from './src/hooks/useImageProcessing.ts';
      export { Status, exportBatchSummary, exportCompletedOutputs } from './src/components/ui/ExportImportProperties.tsx';
      export { default as PortraitPanel } from './src/components/panel/right/PortraitPanel.tsx';
      export { default as CollapsibleSection } from './src/components/ui/CollapsibleSection.tsx';
      export { default as PortraitReshapeFaceOverlay } from './src/components/panel/editor/PortraitReshapeFaceOverlay.tsx';
      export * from './src/utils/portraitReshapePanel.ts';
      export { MobilePanelSwitcher } from './src/components/panel/PanelSwitcher.tsx';
      export { createElement } from 'react';
      export { renderToStaticMarkup } from 'react-dom/server';
      export { default as i18n } from 'i18next';
      import './src/i18n/index.ts';
    `,
    resolveDir: fileURLToPath(new URL('../', import.meta.url)),
    sourcefile: 'portrait-test-entry.ts',
    loader: 'ts',
  },
  bundle: true,
  write: false,
  platform: 'node',
  format: 'cjs',
  packages: 'external',
  define: { 'import.meta.env.DEV': 'false' },
});
const module = { exports: {} };
// frame.ts 的 import.meta.glob 在 CJS 下为空对象；替换为无 logo 的 stub。
const bundled = compiled.outputFiles[0].text.replaceAll(
  'import_meta.glob',
  '((..._args) => ({}))',
);
new Function('require', 'module', 'exports', bundled)(
  createRequire(import.meta.url),
  module,
  module.exports,
);
const {
  INITIAL_ADJUSTMENTS,
  INITIAL_PORTRAIT_ADJUSTMENTS,
  PORTRAIT_KEYS,
  PORTRAIT_LABELS,
  PORTRAIT_SECTIONS,
  portraitSectionKeys,
  normalizePortraitAdjustments,
  normalizeLoadedAdjustments,
  hasPortraitEffect,
  portraitSourceSignature,
  COPYABLE_ADJUSTMENT_KEYS,
  ADJUSTMENT_SECTIONS,
  Panel,
  useEditorStore,
  useUIStore,
  useSettingsStore,
  reconcileWorkspace,
  PortraitPanel,
  CollapsibleSection,
  PortraitReshapeFaceOverlay,
  MobilePanelSwitcher,
  createElement,
  renderToStaticMarkup,
  i18n,
} = module.exports;

const previewJob = (overrides = {}) => ({
  path: 'portrait.jpg',
  generation: 1,
  readiness: 'ready',
  adjustments: INITIAL_ADJUSTMENTS,
  targetRes: 1920,
  dragging: false,
  queuedAt: 0,
  ...overrides,
});

test('人像调色预览：单飞期间只保留最新参数，高清不并发抢占', () => {
  const queue = module.exports.createPreviewApplyQueue();
  const first = previewJob();
  queue.offer(first);
  assert.equal(queue.take(first.path, 1), first);
  for (let exposure = 1; exposure <= 30; exposure++) {
    queue.offer(previewJob({ dragging: true, adjustments: { ...INITIAL_ADJUSTMENTS, exposure } }));
    assert.equal(queue.take(first.path, 1), null);
  }
  queue.finish(first, true);
  const last = queue.take(first.path, 1);
  assert.equal(last.adjustments.exposure, 30);
  assert.equal(last.dragging, true);
  queue.finish(last, true);
  assert.equal(queue.hasPending(), false);
});

test('人像调色预览：同内容克隆高清去重，分辨率和分析就绪改变允许补帧', () => {
  const queue = module.exports.createPreviewApplyQueue();
  const first = previewJob();
  queue.offer(first);
  queue.take(first.path, 1);
  queue.offer(previewJob({ adjustments: structuredClone(first.adjustments), queuedAt: 10 }));
  queue.finish(first, true);
  assert.equal(queue.hasPending(), false);
  queue.offer(previewJob({ adjustments: structuredClone(first.adjustments) }));
  assert.equal(queue.hasPending(), false);
  for (const change of [{ targetRes: 2560 }, { readiness: 'analysis-complete' }, { dragging: true }]) {
    const job = previewJob(change);
    queue.offer(job);
    assert.equal(queue.take(job.path, 1), job);
    queue.finish(job, true);
  }
});

test('人像调色预览：失败保留同参数重试，迟到完成不得释放当前任务', () => {
  const queue = module.exports.createPreviewApplyQueue();
  const first = previewJob();
  const retry = previewJob({ queuedAt: 20 });
  queue.offer(first);
  queue.take(first.path, 1);
  queue.offer(retry);
  queue.finish(first, false);
  assert.equal(queue.take(first.path, 1), retry);
  const latest = previewJob({ targetRes: 2560 });
  queue.offer(latest);
  queue.finish(first, true);
  assert.equal(queue.take(first.path, 1), null);
  queue.finish(retry, false);
  assert.equal(queue.take(first.path, 1), latest);
});

test('人像调色预览：清理保留在飞锁，切图和源代次拒绝旧待执行', () => {
  const queue = module.exports.createPreviewApplyQueue();
  const first = previewJob();
  queue.offer(first);
  queue.take(first.path, 1);
  queue.clearPending();
  const other = previewJob({ path: 'other.raw', generation: 2 });
  queue.offer(other);
  assert.equal(queue.take(other.path, 2), null);
  queue.finish(first, true);
  assert.equal(queue.take(other.path, 2), other);
  queue.finish(other, true);
  queue.offer(previewJob());
  assert.equal(queue.take(other.path, 2), null);
  queue.offer(previewJob({ path: other.path }));
  assert.equal(queue.take(other.path, 2), null);
  assert.equal(queue.hasPending(), false);
});

test('人像调色预览：同源清理后的在飞完成不能恢复旧去重状态', () => {
  const queue = module.exports.createPreviewApplyQueue();
  const first = previewJob();
  queue.offer(first);
  queue.take(first.path, 1);
  queue.clearPending();
  queue.finish(first, true);
  const next = previewJob();
  queue.offer(next);
  assert.equal(queue.take(first.path, 1), next);
});

test('人像调色预览：重新拖动移除待发高清并保留最新交互', () => {
  const queue = module.exports.createPreviewApplyQueue();
  const first = previewJob({ dragging: true });
  queue.offer(first);
  queue.take(first.path, 1);
  queue.offer(previewJob());
  queue.dropPendingHighQuality();
  assert.equal(queue.hasPending(), false);
  const next = previewJob({ dragging: true, adjustments: { ...INITIAL_ADJUSTMENTS, exposure: 2 } });
  queue.offer(next);
  queue.dropPendingHighQuality();
  queue.finish(first, true);
  assert.equal(queue.take(first.path, 1), next);
});

test('人像调色预览：撤销回到在飞参数只在成功后去重，补丁按内容区分', () => {
  const queue = module.exports.createPreviewApplyQueue();
  const first = previewJob({ adjustments: { ...INITIAL_ADJUSTMENTS, aiPatches: [{ id: 'a', patchData: 'AAAA' }] } });
  queue.offer(first);
  queue.take(first.path, 1);
  queue.offer(previewJob({ adjustments: { ...first.adjustments, exposure: 2 } }));
  queue.offer(previewJob({ adjustments: structuredClone(first.adjustments) }));
  queue.finish(first, true);
  assert.equal(queue.hasPending(), false);
  const changed = previewJob({ adjustments: { ...first.adjustments, aiPatches: [{ id: 'a', patchData: 'BBBB' }] } });
  queue.offer(changed);
  assert.equal(queue.take(first.path, 1), changed);
});

test('人像调色预览：人像松手等待150ms，非人像维持50ms', () => {
  const { previewSettleDelay } = module.exports;
  assert.equal(previewSettleDelay(INITIAL_ADJUSTMENTS), 50);
  for (const key of ['lipstick', 'bigEye', 'hydratedSmoothingFace', 'bodySlimArms']) {
    const adj = { ...INITIAL_ADJUSTMENTS, portrait: { ...INITIAL_PORTRAIT_ADJUSTMENTS, [key]: 50 } };
    assert.equal(previewSettleDelay(adj), 150, key);
  }
});

test('08：估算防抖排队、乱序完成和清理立即作废旧请求', async () => {
  const guard = module.exports.createEstimateRequestGuard();
  const first = guard.begin();
  assert.equal(first(), true);
  const queued = guard.begin();
  assert.equal(first(), false);
  assert.equal(queued(), true);
  let displayed = 'pending';
  const late = Promise.resolve().then(() => {
    if (queued()) displayed = 'stale';
  });
  guard.invalidate();
  await late;
  assert.equal(displayed, 'pending');
  assert.equal(queued(), false);
  const latest = guard.begin();
  assert.equal(latest(), true);
  guard.invalidate();
  assert.equal(latest(), false);
});

test('08：取消或失败保留已提交文件，下一任务清空且取消恢复不清空', () => {
  const { useProcessStore, Status, exportCompletedOutputs } = module.exports;
  const store = useProcessStore.getState();
  const initial = useProcessStore.getInitialState();
  try {
    store.setExportState({ status: Status.Exporting });
    store.setExportState({ completedOutputs: exportCompletedOutputs({ outputPath: 'a.tiff' }) });
    store.setExportState({ status: Status.Cancelling });
    store.setExportState({ status: Status.Exporting });
    assert.deepEqual(useProcessStore.getState().exportState.completedOutputs, ['a.tiff']);
    store.setExportState({
      status: Status.Cancelled,
      completedOutputs: exportCompletedOutputs({ completedOutputs: ['a.tiff', 'b.tiff'] }),
    });
    store.setExportState({ status: Status.Error, errorMessage: 'third failed' });
    assert.deepEqual(useProcessStore.getState().exportState.completedOutputs, ['a.tiff', 'b.tiff']);
    store.setExportState({ status: Status.Idle });
    assert.deepEqual(useProcessStore.getState().exportState.completedOutputs, ['a.tiff', 'b.tiff']);
    store.setExportState({ status: Status.Exporting });
    assert.deepEqual(useProcessStore.getState().exportState.completedOutputs, []);
    assert.deepEqual(exportCompletedOutputs(null), []);
    assert.deepEqual(exportCompletedOutputs('old error'), []);
    assert.deepEqual(exportCompletedOutputs({ completedOutputs: ['', 12, 'b', 'b'], outputPath: 'a' }), ['b', 'a']);
  } finally {
    store.setExportState({ status: Status.Idle });
    useProcessStore.setState(initial, true);
  }
});

test('12：批量导出摘要拒绝损坏 payload，并按结果而非后端计数归类', () => {
  const { exportBatchSummary, Status, useProcessStore } = module.exports;
  const payload = {
    batchSummary: {
      failed: 999,
      skipped: 999,
      succeeded: 999,
      items: [
        { path: 'a.raw', outcome: 'success', elapsedMs: 12, outputPath: 'a.jpg', reshapePreparationMs: 4 },
        { path: 'b.raw', outcome: 'skipped', elapsedMs: 7, reason: 'noFace', reshapePreparationMs: 3 },
        { path: 'c.raw', outcome: 'failed', elapsedMs: 20, reason: 'modelError', reshapePreparationMs: 8 },
      ],
      performance: { p50Ms: 12, p95Ms: 20, totalMs: 42 },
    },
  };
  const summary = exportBatchSummary(payload);
  assert.equal(summary?.succeeded, 1);
  assert.equal(summary?.skipped, 1);
  assert.equal(summary?.failed, 1);
  assert.deepEqual(summary?.performance, payload.batchSummary.performance);
  assert.deepEqual(
    summary?.items.map(({ outcome, reason }) => ({ outcome, reason })),
    [
      { outcome: 'success', reason: null },
      { outcome: 'skipped', reason: 'noFace' },
      { outcome: 'failed', reason: 'modelError' },
    ],
  );
  assert.equal(
    exportBatchSummary({ batchSummary: { ...payload.batchSummary, items: [{ path: 'a', outcome: 'bad' }] } }),
    undefined,
  );
  const initial = useProcessStore.getState().exportState;
  try {
    useProcessStore.getState().setExportState({ status: Status.Exporting });
    useProcessStore.getState().setExportState({ batchSummary: summary });
    assert.equal(useProcessStore.getState().exportState.batchSummary?.skipped, 1);
    useProcessStore.getState().setExportState({ status: Status.Idle });
    useProcessStore.getState().setExportState({ status: Status.Exporting });
    assert.equal(useProcessStore.getState().exportState.batchSummary, undefined);
  } finally {
    useProcessStore.setState({ exportState: initial });
  }
});

// SSR 只验证结构和文案；将测试快照设为当前状态，不模拟浏览器事件或后端持久化。
const editorSnapshot = useEditorStore.getInitialState();
const uiSnapshot = useUIStore.getInitialState();
const settingsSnapshot = useSettingsStore.getInitialState();

const reshapeCases = JSON.parse(readFileSync(new URL('./portrait-reshape-cases.json', import.meta.url), 'utf8'));
const {
  createPortraitReshape,
  readPortraitReshape,
  normalizeReshapeStrengths,
  hasPortraitReshapeRequest,
  copyPortraitReshapeDefaults,
} = module.exports;

// —— 09 逐脸面板：决策、写入、复位、坐标映射与选择守卫 ——
const {
  PORTRAIT_RESHAPE_REASON_LABELS,
  decideReshapePanel,
  applyReshapeSliderChange,
  activateLegacyReshape,
  clearFaceOverrides,
  clearAllFaceOverrides,
  reshapeFaceFrame,
  selectPortraitReshapeFace,
} = module.exports;

for (const fixture of reshapeCases) {
  test(`重塑契约：${fixture.name}`, () => {
    const parsed = readPortraitReshape(fixture.input.portraitReshape);
    assert.equal(parsed.status, fixture.status);
    assert.deepEqual(parsed.value, fixture.normalized);
    assert.deepEqual(normalizeReshapeStrengths(fixture.input.portrait), fixture.strengths);
    assert.equal(hasPortraitReshapeRequest(fixture.input, fixture.targets), fixture.enabled);
    const loaded = normalizeLoadedAdjustments(fixture.input);
    assert.deepEqual(loaded.portraitReshape, fixture.normalized);
    const reopened = normalizeLoadedAdjustments(JSON.parse(JSON.stringify(loaded)));
    assert.deepEqual(reopened.portraitReshape, fixture.normalized);
    assert.equal(hasPortraitEffect(loaded), false);
  });
}

const reshapeFace = () => ({
  faceId: 'face-1',
  sourceAnchor: { sourceBaseHash: 'a'.repeat(64), bounds: [0.1, 0.1, 0.3, 0.3] },
  overrides: { slimFace: 0 },
  protection: { hair: true, background: true, strokes: [] },
});

test('重塑契约：非有限值、无穷、布尔值和非法 override 归零', () => {
  assert.deepEqual(
    normalizeReshapeStrengths({ slimFace: NaN, smallFace: Infinity, cheekbones: -Infinity, jawline: true }),
    { slimFace: 0, smallFace: 0, cheekbones: 0, jawline: 0 },
  );
  const config = {
    ...createPortraitReshape(),
    engineVersion: 'gpupixel-runtime-v1',
    enabled: true,
    faces: [{ ...reshapeFace(), overrides: { slimFace: NaN } }],
  };
  // 网格 slimFace override 读入时剥离；仅 portrait.slimFace 暂存可作 UV 默认。
  assert.deepEqual(readPortraitReshape(config).settings.faces[0].overrides, {});
  assert.equal(hasPortraitReshapeRequest({ portrait: { slimFace: 60 }, portraitReshape: config }, ['face-1']), true);
});

test('重塑契约：损坏保护、重复身份和越界锚点保留原文但禁止执行', () => {
  const face = reshapeFace();
  const invalidFaces = [
    [face, face],
    [{ ...face, sourceAnchor: { ...face.sourceAnchor, bounds: [0.9, 0.1, 0.3, 0.3] } }],
    [{ ...face, protection: { strokes: [{ mode: 'protect', radius: 0, points: [[0.5, 0.5]] }] } }],
    Array.from({ length: 33 }, (_, index) => ({ ...face, faceId: `face-${index}` })),
  ];
  for (const faces of invalidFaces) {
    const config = { ...createPortraitReshape(), enabled: true, faces };
    const parsed = readPortraitReshape(config);
    assert.equal(parsed.status, 'invalid');
    assert.deepEqual(parsed.value, config);
    assert.equal(hasPortraitReshapeRequest({ portrait: { slimFace: 60 }, portraitReshape: config }, ['face-1']), false);
  }
});

// —— 11 场景级快照：与 Rust portrait_snapshot::valid_snapshot 对齐的静态格式闸 ——
const reshapeSnapshot = (overrides = {}) => ({
  version: 'reshape-snapshot-v1',
  sourceBaseHash: 'a'.repeat(64),
  analysisKey: 'b'.repeat(64),
  parameterHash: 'c'.repeat(64),
  modelHashes: { landmarker: 'd'.repeat(64) },
  meshVersion: 'mediapipe-478-v1',
  baseDimensions: [1024, 1024],
  protectionMap: null,
  groups: [],
  faces: [],
  ...overrides,
});
const reshapeSnapshotFace = (overrides = {}) => ({
  faceId: 'face-1',
  bounds: [0.1, 0.1, 0.3, 0.3],
  roiMatrix: [1, 0, 0, 0, 1, 0],
  detectionConfidence: 0.9,
  landmarkConfidence: 0.8,
  refined: false,
  processable: true,
  reason: null,
  landmarks: Array.from({ length: 478 }, () => [0, 0, 0]),
  localProtectionMap: null,
  plan: { fixed: '0'.repeat(217), weights: [[0, 2.5]], regions: [[0, 0.5, true, 0]] },
  mesh: { target: Array.from({ length: 217 }, () => [0, 0]), requested: [10, 20, 30, 40], effective: [0, 0, 0, 1] },
  safetyScale: 0.8,
  backoffs: 1,
  detectedLines: 0,
  reasons: 0,
  ...overrides,
});
const snapshotConfig = (snapshot) => ({ ...createPortraitReshape(), enabled: true, snapshot });

test('11 快照状态机：网格快照读入剥离为 UV，缺失视为无快照', () => {
  const snapshot = reshapeSnapshot();
  const parsed = readPortraitReshape(snapshotConfig(snapshot));
  assert.equal(parsed.status, 'supported');
  assert.equal(parsed.settings.engineVersion, 'gpupixel-runtime-v1');
  assert.equal(Object.hasOwn(parsed.settings, 'snapshot'), false);
  assert.equal(snapshot.version, 'reshape-snapshot-v1');
  assert.equal(readPortraitReshape({ ...createPortraitReshape(), enabled: true }).status, 'supported');
  assert.equal(readPortraitReshape(snapshotConfig(null)).status, 'supported');
  // 任意快照负载均剥离，不再按版本/结构拒绝（网格引擎已下线）。
  for (const candidate of [{ faces: [] }, reshapeSnapshot({ version: 'reshape-snapshot-v9' }), { version: null }]) {
    const result = readPortraitReshape(snapshotConfig(candidate));
    assert.equal(result.status, 'supported');
    assert.equal(Object.hasOwn(result.settings, 'snapshot'), false);
  }
});

test('11 快照静态闸：损坏快照不再阻断 UV 契约', () => {
  for (const broken of [
    reshapeSnapshot({ analysisKey: 'B'.repeat(64) }),
    reshapeSnapshot({ meshVersion: 'mediapipe-468-v1' }),
    reshapeSnapshot({ modelHashes: {} }),
    reshapeSnapshot({ groups: [[0]] }),
    reshapeSnapshot({ faces: [reshapeSnapshotFace()] }),
  ]) {
    const result = readPortraitReshape(snapshotConfig(broken));
    assert.equal(result.status, 'supported');
    assert.equal(result.settings.engineVersion, 'gpupixel-runtime-v1');
    assert.equal(Object.hasOwn(result.settings, 'snapshot'), false);
  }
  const one = reshapeSnapshot({ faces: [reshapeSnapshotFace()], groups: [[0]] });
  const parsed = readPortraitReshape(snapshotConfig(one));
  assert.equal(parsed.status, 'supported');
  assert.equal(Object.hasOwn(parsed.settings, 'snapshot'), false);
  assert.equal(one.faces[0].mesh.target[0][0], 0);
});

test('重塑契约：旧数据不升级、历史保留未知版本，选择与安全复制隔离', () => {
  const initial = structuredClone(INITIAL_ADJUSTMENTS);
  assert.deepEqual(initial.portraitReshape, createPortraitReshape());
  const legacy = { ...initial, portrait: { ...initial.portrait, slimFace: 80 } };
  delete legacy.portraitReshape;
  assert.equal(normalizeLoadedAdjustments(legacy).portraitReshape, null);
  const future = reshapeCases.find((entry) => entry.status === 'unsupported').input.portraitReshape;
  const changed = { ...initial, portraitReshape: future };
  useEditorStore.getState().resetHistory(initial);
  useEditorStore.getState().pushHistory(changed);
  useEditorStore.getState().setEditor({ adjustments: changed });
  useEditorStore.getState().undo();
  assert.deepEqual(useEditorStore.getState().adjustments.portraitReshape, initial.portraitReshape);
  useEditorStore.getState().redo();
  assert.deepEqual(useEditorStore.getState().adjustments.portraitReshape, future);
  const signature = portraitSourceSignature(changed);
  useEditorStore.getState().setEditor({
    portraitReshapeStatus: {
      path: 'test.jpg',
      generation: 1,
      source: 'base',
      phase: 'idle',
      faces: [],
      selectedFaceId: 'face-1',
    },
  });
  assert.equal(portraitSourceSignature(useEditorStore.getState().adjustments), signature);
  assert.equal(Object.hasOwn(changed, 'portraitReshapeStatus'), false);
  assert.equal(COPYABLE_ADJUSTMENT_KEYS.includes('portraitReshape'), false);
  assert.deepEqual(copyPortraitReshapeDefaults({ ...reshapeFace(), slimFace: 23, exposure: 2 }), {
    slimFace: 23,
    smallFace: 0,
    cheekbones: 0,
    jawline: 0,
  });
  assert.equal(portraitSourceSignature(changed), portraitSourceSignature(initial));
});

const {
  portraitReshapeSourceSignature,
  reshapeVisibilityPreviewSignature,
  onlyPortraitReshapeVisibilityChanged,
  isPortraitReshapeSectionVisible,
  isPortraitSectionVisible,
  onlyPortraitVisibilityChanged,
  portraitVisibilityPreviewSignature,
  detectPortraitVisibilityFlip,
  portraitVisibilitySectionHasEffect,
  PORTRAIT_VISIBILITY_KEYS,
  hasPortraitReshapePreviewEffect,
  emptyPortraitReshapeStatus,
  matchesPortraitReshapeRequest,
  applyPortraitReshapeSummary,
  clearPortraitReshapeSnapshot,
  Invokes,
} = module.exports;

const analysisSummary = (request, faces = [], snapshotReused = true) => ({
  request,
  key: 'analysis-key',
  faces,
  rejectedFaces: 0,
  reason: faces.length ? null : 'no-face',
  cache: { bytes: 1024, entries: 1, inFlight: 0, hits: 1, computations: 1 },
  snapshotReused,
});
const analysisFace = (faceId) => ({
  faceId,
  processable: true,
  sourceAnchor: { sourceBaseHash: 'a'.repeat(64), bounds: [0.1, 0.2, 0.2, 0.3] },
});

test('04 分析签名：100 次滑块、调色、裁剪和保护笔刷不改变基础签名', () => {
  const initial = structuredClone(INITIAL_ADJUSTMENTS);
  const signature = portraitReshapeSourceSignature(initial);
  for (let amount = 0; amount < 100; amount++) {
    const changed = structuredClone(initial);
    for (const key of ['slimFace', 'smallFace', 'cheekbones', 'jawline']) changed.portrait[key] = amount;
    changed.exposure = amount;
    changed.crop = { x: amount, y: 0, width: 30, height: 40 };
    changed.portraitReshape.faces = [{ faceId: 'selected', protection: { strokes: [{ radius: amount }] } }];
    assert.equal(portraitReshapeSourceSignature(changed), signature);
  }
});

test('04 小眼睛可见性：不改变分析签名与 A/B 预览签名，且不进入分析 enabled 门控', () => {
  const initial = structuredClone(INITIAL_ADJUSTMENTS);
  initial.sectionVisibility.portraitReshape = true;
  const hidden = structuredClone(initial);
  hidden.sectionVisibility.portraitReshape = false;
  assert.equal(isPortraitReshapeSectionVisible(initial), true);
  assert.equal(isPortraitReshapeSectionVisible(hidden), false);
  assert.equal(portraitReshapeSourceSignature(hidden), portraitReshapeSourceSignature(initial));
  assert.equal(reshapeVisibilityPreviewSignature(hidden), reshapeVisibilityPreviewSignature(initial));
  assert.equal(onlyPortraitReshapeVisibilityChanged(initial, hidden), true);
  assert.equal(onlyPortraitReshapeVisibilityChanged(hidden, initial), true);
  const strength = structuredClone(initial);
  strength.portrait.slimFace = 40;
  assert.equal(onlyPortraitReshapeVisibilityChanged(initial, strength), false);
  const both = structuredClone(hidden);
  both.portrait.slimFace = 40;
  assert.equal(onlyPortraitReshapeVisibilityChanged(initial, both), false);
  // 契约：分析 hook 的 enabled 不得依赖 sectionVisibility.portraitReshape，否则关组会 empty 并重分析。
  const hook = readFileSync(new URL('../src/hooks/usePortraitReshapeAnalysis.ts', import.meta.url), 'utf8');
  assert.ok(hook.includes('parameters?.enabled === true'));
  assert.equal(
    /enabled\s*=\s*\n?\s*import\.meta\.env\.DEV[\s\S]*?sectionVisibility\?\.portraitReshape/.test(hook),
    false,
  );
  assert.ok(hook.includes('分区显隐只影响渲染') || hook.includes('sectionVisibility → RenderIntent::Bypass'));

  // 六段人像小眼睛：签名稳定、单键 flip 可检测；改强度或双键翻转不可 swap。
  assert.deepEqual(
    [...PORTRAIT_VISIBILITY_KEYS],
    ['portraitSkin', 'portraitBlemishes', 'portraitMakeup', 'portraitReshape', 'portraitEyes', 'portraitBody'],
  );
  for (const key of PORTRAIT_VISIBILITY_KEYS) {
    const on = structuredClone(INITIAL_ADJUSTMENTS);
    on.sectionVisibility[key] = true;
    const off = structuredClone(on);
    off.sectionVisibility[key] = false;
    assert.equal(isPortraitSectionVisible(on, key), true);
    assert.equal(isPortraitSectionVisible(off, key), false);
    assert.equal(portraitVisibilityPreviewSignature(on, key), portraitVisibilityPreviewSignature(off, key));
    assert.equal(onlyPortraitVisibilityChanged(on, off, key), true);
    assert.equal(detectPortraitVisibilityFlip(on, off), key);
    assert.equal(detectPortraitVisibilityFlip(off, on), key);
    const stronger = structuredClone(on);
    if (key === 'portraitSkin') stronger.portrait.neutralGrayEvenness = 40;
    else if (key === 'portraitBlemishes') stronger.portrait.aiBlemishRemoval = 40;
    else if (key === 'portraitMakeup') stronger.portrait.lipstick = 40;
    else if (key === 'portraitEyes') stronger.portrait.bigEye = 40;
    else if (key === 'portraitBody') stronger.portrait.bodySlimArms = 40;
    else stronger.portrait.slimFace = 40;
    assert.equal(onlyPortraitVisibilityChanged(on, stronger, key), false);
    assert.equal(detectPortraitVisibilityFlip(on, stronger), null);
    assert.equal(portraitVisibilitySectionHasEffect(stronger, key), true);
    assert.equal(portraitVisibilitySectionHasEffect(on, key), false);
  }
  const multi = structuredClone(INITIAL_ADJUSTMENTS);
  multi.sectionVisibility.portraitSkin = false;
  multi.sectionVisibility.portraitEyes = false;
  assert.equal(detectPortraitVisibilityFlip(INITIAL_ADJUSTMENTS, multi), null);
});

test('04 分析签名：有效补丁内容、镜头、旋转和前置液化改变签名', () => {
  const initial = structuredClone(INITIAL_ADJUSTMENTS);
  const signature = portraitReshapeSourceSignature(initial);
  for (const change of [
    { transformDistortion: 5 },
    { lensDistortionParams: { k1: 0.03 } },
    { rotation: 20 },
    { flipHorizontal: true },
    { liquify: [{ x: 0.4 }] },
  ]) {
    assert.notEqual(portraitReshapeSourceSignature({ ...initial, ...change }), signature);
  }
  const patch = { id: 'same-id', visible: true, patchData: { color: 'AAAA', mask: 'same' } };
  const a = portraitReshapeSourceSignature({ ...initial, aiPatches: [patch] });
  const b = portraitReshapeSourceSignature({
    ...initial,
    aiPatches: [{ ...patch, patchData: { color: 'BBBB', mask: 'same' } }],
  });
  assert.notEqual(a, b);
  const empty = portraitReshapeSourceSignature({ ...initial, aiPatches: [] });
  assert.equal(portraitReshapeSourceSignature({ ...initial, aiPatches: [{ ...patch, visible: false }] }), empty);
  assert.equal(portraitReshapeSourceSignature({ ...initial, aiPatches: [{ ...patch, isLoading: true }] }), empty);
});

test('04 结果守卫：路径、源签名、代次和 ready 必须同时一致', () => {
  const request = { path: 'A.jpg', source: 'a'.repeat(64), generation: 7 };
  const state = {
    selectedImage: { path: 'A.jpg', isReady: true },
    portraitReshapeStatus: { ...emptyPortraitReshapeStatus('A.jpg', 7), source: request.source },
    adjustments: structuredClone(INITIAL_ADJUSTMENTS),
  };
  assert.equal(matchesPortraitReshapeRequest(state, request), true);
  for (const stale of [
    { ...request, path: 'B.jpg' },
    { ...request, source: 'b'.repeat(64) },
    { ...request, generation: 6 },
  ]) {
    assert.equal(matchesPortraitReshapeRequest(state, stale), false);
    assert.deepEqual(applyPortraitReshapeSummary(state, analysisSummary(stale)), {});
  }
  assert.equal(
    matchesPortraitReshapeRequest({ ...state, selectedImage: { path: 'A.jpg', isReady: false } }, request),
    false,
  );
});

test('04 切图立即清空状态：A 的延迟结果不能写回 B，历史与参数不改变', () => {
  const snapshot = useEditorStore.getState();
  try {
    const request = { path: 'A.jpg', source: 'a'.repeat(64), generation: 50 };
    useEditorStore.setState({
      selectedImage: { path: 'A.jpg', isReady: true },
      portraitReshapeStatus: {
        ...emptyPortraitReshapeStatus('A.jpg', 50),
        source: request.source,
        faces: [analysisFace('a')],
      },
    });
    const before = useEditorStore.getState();
    before.setEditor({ selectedImage: { path: 'B.jpg', isReady: true } });
    const switched = useEditorStore.getState();
    assert.equal(switched.portraitReshapeStatus.generation, 51);
    assert.deepEqual(switched.portraitReshapeStatus.faces, []);
    switched.setEditor((state) => applyPortraitReshapeSummary(state, analysisSummary(request, [analysisFace('a')])));
    assert.equal(useEditorStore.getState().portraitReshapeStatus.path, 'B.jpg');
    assert.equal(useEditorStore.getState().adjustments, before.adjustments);
    assert.equal(useEditorStore.getState().history, before.history);
    assert.equal(useEditorStore.getState().portraitStatus, before.portraitStatus);
  } finally {
    useEditorStore.setState(snapshot, true);
  }
});

test('04 摘要只保留选脸数据：排序变化保留选择，失配清空选择，超限拒绝', () => {
  const request = { path: 'A.jpg', source: 'a'.repeat(64), generation: 1 };
  const state = {
    selectedImage: { path: 'A.jpg', isReady: true },
    portraitReshapeStatus: { ...emptyPortraitReshapeStatus('A.jpg', 1), source: request.source, selectedFaceId: 'a' },
    adjustments: structuredClone(INITIAL_ADJUSTMENTS),
  };
  const a = { ...analysisFace('a'), landmarks: Array(478).fill([0, 0, 0]), probabilities: [1] };
  const result = applyPortraitReshapeSummary(state, analysisSummary(request, [analysisFace('b'), a]));
  assert.equal(result.portraitReshapeStatus.selectedFaceId, 'a');
  assert.equal(Object.hasOwn(result.portraitReshapeStatus.faces[1], 'landmarks'), false);
  assert.equal(Object.hasOwn(result.portraitReshapeStatus.faces[1], 'probabilities'), false);
  const unmatched = applyPortraitReshapeSummary(
    state,
    analysisSummary(request, [{ ...a, processable: false, reason: 'unmatched-face' }]),
  );
  assert.equal(unmatched.portraitReshapeStatus.selectedFaceId, null);
  assert.equal(unmatched.portraitReshapeStatus.phase, 'partial');
  assert.equal(
    applyPortraitReshapeSummary(state, analysisSummary(request, Array(65).fill(a))).portraitReshapeStatus.phase,
    'error',
  );
  const empty = applyPortraitReshapeSummary(state, analysisSummary(request));
  assert.equal(empty.portraitReshapeStatus.reason, 'no-face');
});

test('11 修复通道：摘要未复用快照时剥离旧快照，复用与无快照时不写回', () => {
  const request = { path: 'A.jpg', source: 'a'.repeat(64), generation: 1 };
  const snapshot = reshapeSnapshot();
  const withSnapshot = { ...createPortraitReshape(), enabled: true, snapshot };
  const state = {
    selectedImage: { path: 'A.jpg', isReady: true },
    portraitReshapeStatus: { ...emptyPortraitReshapeStatus('A.jpg', 1), source: request.source },
    adjustments: {
      ...structuredClone(INITIAL_ADJUSTMENTS),
      portraitReshape: withSnapshot,
    },
  };
  // 有效复用：保留原对象引用，adjustments 不写回。
  const reused = applyPortraitReshapeSummary(state, analysisSummary(request, [], true));
  assert.equal(Object.hasOwn(reused, 'adjustments'), false);
  assert.equal(reused.portraitReshapeStatus.phase, 'ready');
  // 未复用：写回无 snapshot 的 UV 配置，输入原对象不变。
  const repaired = applyPortraitReshapeSummary(state, analysisSummary(request, [], false));
  assert.equal(Object.hasOwn(repaired.adjustments.portraitReshape, 'snapshot'), false);
  assert.equal(repaired.adjustments.portraitReshape.enabled, true);
  assert.equal(repaired.adjustments.portraitReshape.engineVersion, 'gpupixel-runtime-v1');
  assert.equal(repaired.adjustments.exposure, state.adjustments.exposure);
  assert.equal(state.adjustments.portraitReshape.snapshot, snapshot);
  const plain = createPortraitReshape();
  assert.equal(clearPortraitReshapeSnapshot(plain), plain);
  assert.equal(clearPortraitReshapeSnapshot(null), null);
  const unsupported = { schemaVersion: 9 };
  assert.equal(clearPortraitReshapeSnapshot(unsupported), unsupported);
});

test('08：缩略图、预设、社区与按路径预览接入共享原生渲染入口', () => {
  const lib = readFileSync(new URL('../src-tauri/src/lib.rs', import.meta.url), 'utf8');
  const fileManagement = readFileSync(new URL('../src-tauri/src/file_management.rs', import.meta.url), 'utf8');
  const render = readFileSync(new URL('../src-tauri/src/portrait_render.rs', import.meta.url), 'utf8');

  // 共享低清入口：旁路直返、发布在 lease 提交临界区内执行。
  assert.ok(render.includes('pub fn render_reduced_photo<T>'));
  assert.ok(render.includes('if render_intent(adjustments)? == RenderIntent::Bypass'));
  assert.ok(render.includes('purpose: RenderPurpose::Thumbnail'));
  assert.ok(render.includes('|| publish(image)'));
  // 社区中心方形裁剪并入照片最终窗口，裁剪不参与分析键。
  assert.ok(render.includes('square_target'));

  // 缩略图：会话隔离缓存键＋场／渲染版本；小／中图各自最终缩放。
  assert.ok(fileManagement.includes('fn reshape_thumbnail_session_key'));
  assert.ok(fileManagement.includes('crate::portrait_tiles::FIELD_VERSION.as_bytes()'));
  assert.ok(fileManagement.includes('crate::portrait_render::RENDER_VERSION.as_bytes()'));
  assert.ok(fileManagement.includes('encode_thumbnail_exact'));
  assert.ok(fileManagement.includes('render_thumbnail(target_width_medium, &medium_path)'));
  assert.ok(fileManagement.includes('render_thumbnail(target_width_small, &small_path)'));
  // 活动失败清理半写缓存，不缓存无效果照片。
  assert.ok(fileManagement.includes('fs::remove_file(&small_path)'));

  // 预设／社区／按路径三个预览入口接入共享渲染；旧美肤先缩小路径仅服务旁路。
  assert.equal(lib.split('render_reduced_photo(').length - 1, 3);
  assert.ok(fileManagement.split('render_reduced_photo(').length - 1 >= 2);
  assert.ok(lib.includes('crate::portrait_tiles::MAX_SIDE'));
  assert.ok(lib.includes('crate::portrait_render::RenderQuality::HighQuality'));
  assert.ok(lib.includes('portrait_processing::enabled(js_adjustments)'));
});

test('04 IPC 注册与独立入口一致，重塑使用严格 RAW 解码', () => {
  const lib = readFileSync(new URL('../src-tauri/src/lib.rs', import.meta.url), 'utf8');
  for (const command of [Invokes.PreparePortraitReshapeAnalysis, Invokes.CancelPortraitReshapeAnalysis]) {
    assert.ok(lib.includes(`portrait_analysis_cache::${command}`));
  }
  const source = readFileSync(new URL('../src-tauri/src/portrait_analysis_cache.rs', import.meta.url), 'utf8');
  const render = readFileSync(new URL('../src-tauri/src/portrait_render.rs', import.meta.url), 'utf8');
  // 08 起严格源加载集中在 portrait_render：分析入口必须经 editor_source，不允许绕过快照直接解码。
  assert.ok(source.includes('portrait_render.editor_source('));
  assert.ok(source.includes('prepare_source('));
  assert.ok(render.includes('SourceSnapshot::load('));
  assert.ok(render.includes('load_base_image_strict(&bytes'));
  assert.ok(render.includes('read_snapshot_bytes(&source, current)'));
  assert.ok(source.includes('ReshapeModels::approved_model_hashes'));
  assert.ok(!source.includes('PortraitModels::load'));
  assert.ok(!render.includes('load_base_image_fast'));
});

const defaults = () => structuredClone(INITIAL_ADJUSTMENTS);
const implementedPortraitKeys = [
  'neutralGrayEvenness',
  'neutralGrayContour',
  'frequencySeparationFace',
  'frequencySeparationBody',
  'hydratedSmoothingFace',
  'hydratedSmoothingBody',
  'evenSkinFace',
  'evenSkinBody',
  'whiteningFace',
  'whiteningBody',
  'aiBlemishRemoval',
  'faceShineRemoval',
  'eyeWhiteBrighten',
  'eyeRedness',
  'lipstick',
  'blusher',
  'bodySlimArms',
  'bodySlimWaist',
  'bodySlimLegs',
];
const selectedImage = { path: 'portrait-test.jpg', isReady: true, width: 800, height: 1200 };
const panelMarkup = () => {
  Object.assign(editorSnapshot, useEditorStore.getState());
  Object.assign(uiSnapshot, useUIStore.getState());
  Object.assign(settingsSnapshot, useSettingsStore.getState());
  return renderToStaticMarkup(createElement(PortraitPanel));
};

test('39 个数值参数；低频默认 100、高频默认 15，其余归零', () => {
  const lowKeys = ['frequencySeparationFaceLow', 'frequencySeparationBodyLow'];
  const highKeys = ['frequencySeparationFaceHigh', 'frequencySeparationBodyHigh'];
  assert.equal(PORTRAIT_KEYS.length, 41);
  assert.deepEqual(
    PORTRAIT_SECTIONS.map((section) => section.keys.length),
    [14, 2, 2, 1, 3, 3],
  );
  // 网格扩展键仍留在 INITIAL 供迁移归零，但不进入分区 UI；frequencyHighBipolar 亦不入分区。
  assert.deepEqual(
    PORTRAIT_SECTIONS.flatMap((section) => section.keys),
    [
      ...portraitSectionKeys('skin'),
      ...portraitSectionKeys('blemishes'),
      ...portraitSectionKeys('makeup'),
      'slimFace',
      'bigEye',
      'eyeWhiteBrighten',
      'eyeRedness',
      ...portraitSectionKeys('body'),
    ],
  );
  assert.ok(!portraitSectionKeys('skin').includes('frequencyHighBipolar'));
  assert.equal(new Set(PORTRAIT_KEYS).size, 41);
  assert.equal(INITIAL_PORTRAIT_ADJUSTMENTS.frequencyHighBipolar, 2);
  for (const key of lowKeys) assert.equal(INITIAL_PORTRAIT_ADJUSTMENTS[key], 100);
  for (const key of highKeys) assert.equal(INITIAL_PORTRAIT_ADJUSTMENTS[key], 15);
  assert.ok(
    PORTRAIT_KEYS.every((key) => {
      if (key === 'frequencyHighBipolar') return INITIAL_PORTRAIT_ADJUSTMENTS[key] === 2;
      if (lowKeys.includes(key)) return INITIAL_PORTRAIT_ADJUSTMENTS[key] === 100;
      if (highKeys.includes(key)) return INITIAL_PORTRAIT_ADJUSTMENTS[key] === 15;
      return INITIAL_PORTRAIT_ADJUSTMENTS[key] === 0;
    }),
  );
  assert.deepEqual(portraitSectionKeys('skin'), [
    'neutralGrayEvenness',
    'neutralGrayContour',
    'frequencySeparationFace',
    'frequencySeparationFaceLow',
    'frequencySeparationFaceHigh',
    'frequencySeparationBody',
    'frequencySeparationBodyLow',
    'frequencySeparationBodyHigh',
    'hydratedSmoothingFace',
    'hydratedSmoothingBody',
    'evenSkinFace',
    'evenSkinBody',
    'whiteningFace',
    'whiteningBody',
  ]);
  assert.deepEqual(portraitSectionKeys('blemishes'), [
    'aiBlemishRemoval',
    'faceShineRemoval',
  ]);
  assert.deepEqual(portraitSectionKeys('makeup'), ['lipstick', 'blusher']);
  assert.deepEqual(portraitSectionKeys('reshape'), ['slimFace']);
  assert.ok(!portraitSectionKeys('reshape').includes('bigEye'));
  assert.ok(!portraitSectionKeys('reshape').includes('noseWidth'));
  assert.ok(!portraitSectionKeys('reshape').includes('chinProjection'));
  assert.deepEqual(portraitSectionKeys('eyes'), ['bigEye', 'eyeWhiteBrighten', 'eyeRedness']);
  assert.deepEqual(portraitSectionKeys('body'), ['bodySlimArms', 'bodySlimWaist', 'bodySlimLegs']);
});

test('旧照片及缺失、非法参数回退，不接受字符串或越界值', () => {
  for (const value of [undefined, null, [], 'invalid', 4]) {
    assert.deepEqual(normalizePortraitAdjustments(value), INITIAL_PORTRAIT_ADJUSTMENTS);
  }
  const result = normalizePortraitAdjustments({
    slimFace: 150,
    smallFace: -20,
    cheekbones: 12.6,
    jawline: NaN,
    textureRetention: Infinity,
    blemishRemoval: '50',
    neutralGrayEvenness: null,
    frequencySeparationFaceLow: 40.4,
    frequencySeparationFaceHigh: -5,
    frequencySeparationLow: 70,
    frequencySeparationHigh: 20,
  });
  assert.equal(result.slimFace, 100);
  assert.equal(result.smallFace, 0);
  assert.equal(result.cheekbones, 13);
  assert.equal(result.jawline, 0);
  assert.equal(result.textureRetention, 0);
  assert.equal(result.frequencySeparationFaceLow, 40);
  // 无版本标记：旧保纹理量 High=-5 → 105 → clamp 100；legacy High=20 → 80。
  assert.equal(result.frequencySeparationFaceHigh, 100);
  assert.equal(result.frequencySeparationBodyLow, 70);
  assert.equal(result.frequencySeparationBodyHigh, 80);
  assert.equal(result.frequencyHighBipolar, 2);
  assert.equal(normalizePortraitAdjustments({}).frequencySeparationFaceLow, 100);
  assert.equal(normalizePortraitAdjustments({}).frequencySeparationBodyHigh, 15);
  assert.equal(normalizePortraitAdjustments({}).frequencySeparationFaceHigh, 15);
  const legacyShared = normalizePortraitAdjustments({
    frequencySeparationLow: 55,
    frequencySeparationHigh: 10,
  });
  assert.equal(legacyShared.frequencySeparationFaceLow, 55);
  assert.equal(legacyShared.frequencySeparationFaceHigh, 90);
  assert.equal(legacyShared.frequencySeparationBodyLow, 55);
  assert.equal(legacyShared.frequencySeparationBodyHigh, 90);
  assert.equal(legacyShared.frequencyHighBipolar, 2);
  // 无标记：旧 High=100→0、High=0→+100；v1 轴反号；v2 直接使用。
  assert.equal(
    normalizePortraitAdjustments({ frequencySeparationFaceHigh: 100 }).frequencySeparationFaceHigh,
    0,
  );
  assert.equal(
    normalizePortraitAdjustments({ frequencySeparationFaceHigh: 0 }).frequencySeparationFaceHigh,
    100,
  );
  assert.equal(
    normalizePortraitAdjustments({
      frequencyHighBipolar: 1,
      frequencySeparationFaceHigh: 100,
    }).frequencySeparationFaceHigh,
    -100,
  );
  assert.equal(
    normalizePortraitAdjustments({
      frequencyHighBipolar: 1,
      frequencySeparationFaceHigh: -40,
    }).frequencySeparationFaceHigh,
    40,
  );
  assert.equal(
    normalizePortraitAdjustments({
      frequencyHighBipolar: 2,
      frequencySeparationFaceHigh: 100,
    }).frequencySeparationFaceHigh,
    100,
  );
  assert.equal(
    normalizePortraitAdjustments({
      frequencyHighBipolar: 2,
      frequencySeparationFaceHigh: -40,
    }).frequencySeparationFaceHigh,
    -40,
  );
  assert.equal(result.blemishRemoval, 0);
  // 已下线滑条：旧 sidecar 数值一律强制归零。
  assert.equal(normalizePortraitAdjustments({ blemishRemoval: 60 }).blemishRemoval, 0);
  assert.equal(normalizePortraitAdjustments({ textureRetention: 90 }).textureRetention, 0);
  // 旧照片中的布尔开关迁移为强度值，非法值与越界按数值规则处理。
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: true }).aiBlemishRemoval, 100);
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: false }).aiBlemishRemoval, 0);
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: 150 }).aiBlemishRemoval, 100);
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: -10 }).aiBlemishRemoval, 0);
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: '50' }).aiBlemishRemoval, 0);
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: 42.6 }).aiBlemishRemoval, 43);
  // 旧 sidecar 的 slimArms 忽略；身体塑形用 bodySlim* 新键。
  assert.equal(normalizePortraitAdjustments({ slimArms: 120 }).slimArms, undefined);
  assert.equal(normalizePortraitAdjustments({ bodySlimArms: 120 }).bodySlimArms, 100);
  assert.equal(normalizePortraitAdjustments({ bodySlimWaist: -5 }).bodySlimWaist, 0);
  assert.equal(normalizePortraitAdjustments({ bodySlimLegs: 42.6 }).bodySlimLegs, 43);
  const legacy = defaults();
  delete legacy.portrait;
  assert.deepEqual(normalizeLoadedAdjustments(legacy).portrait, INITIAL_PORTRAIT_ADJUSTMENTS);
  assert.notEqual(normalizeLoadedAdjustments(legacy).portrait, INITIAL_PORTRAIT_ADJUSTMENTS);
});

test('编辑参数 JSON 往返与历史撤销、重做、分支互不污染', () => {
  const initial = defaults();
  const changed = { ...initial, portrait: { ...initial.portrait, slimFace: 42 } };
  const restored = normalizeLoadedAdjustments(JSON.parse(JSON.stringify(changed)));
  assert.deepEqual(restored.portrait, changed.portrait);
  useEditorStore.getState().resetHistory(initial);
  useEditorStore.getState().setEditor({ adjustments: changed });
  useEditorStore.getState().pushHistory(changed);
  useEditorStore.getState().undo();
  assert.equal(useEditorStore.getState().adjustments.portrait.slimFace, 0);
  useEditorStore.getState().redo();
  assert.equal(useEditorStore.getState().adjustments.portrait.slimFace, 42);
  useEditorStore.getState().undo();
  const branch = { ...initial, portrait: { ...initial.portrait, cheekbones: 25 } };
  useEditorStore.getState().setEditor({ adjustments: branch });
  useEditorStore.getState().pushHistory(branch);
  useEditorStore.getState().redo();
  assert.equal(useEditorStore.getState().history.length, 2);
  assert.equal(useEditorStore.getState().adjustments.portrait.slimFace, 0);
  assert.equal(useEditorStore.getState().adjustments.portrait.cheekbones, 25);
  assert.deepEqual(initial.portrait, INITIAL_PORTRAIT_ADJUSTMENTS);
});

test('已实现的主效果独立启用、隐藏、归零及历史恢复，其余项目不触发分析', () => {
  const initial = defaults();
  assert.equal(hasPortraitEffect(initial), false);
  for (const key of implementedPortraitKeys) {
    const changed = { ...initial, portrait: { ...initial.portrait, [key]: 50 } };
    assert.equal(hasPortraitEffect(changed), true);
    const visibility = portraitSectionKeys('blemishes').includes(key)
      ? 'portraitBlemishes'
      : portraitSectionKeys('makeup').includes(key)
        ? 'portraitMakeup'
        : portraitSectionKeys('eyes').includes(key)
          ? 'portraitEyes'
          : portraitSectionKeys('body').includes(key)
            ? 'portraitBody'
            : 'portraitSkin';
    assert.equal(hasPortraitEffect({ ...changed, sectionVisibility: { [visibility]: false } }), false);
    assert.equal(hasPortraitEffect({ ...changed, portrait: { [key]: '100' } }), false);
    useEditorStore.getState().resetHistory(initial);
    useEditorStore.getState().setEditor({ adjustments: changed });
    useEditorStore.getState().pushHistory(changed);
    useEditorStore.getState().undo();
    assert.equal(hasPortraitEffect(useEditorStore.getState().adjustments), false);
    useEditorStore.getState().redo();
    const reopened = normalizeLoadedAdjustments(JSON.parse(JSON.stringify(useEditorStore.getState().adjustments)));
    assert.equal(reopened.portrait[key], 50);
    reopened.portrait[key] = 0;
    assert.equal(hasPortraitEffect(reopened), false);
  }
  const other = defaults();
  for (const key of PORTRAIT_KEYS.filter((key) => !implementedPortraitKeys.includes(key))) other.portrait[key] = 100;
  assert.equal(hasPortraitEffect(other), false);
  assert.equal(hasPortraitEffect({ ...initial, portrait: { neutralGrayEvenness: '100' } }), false);
});

test('祛斑分区显隐独立、旧数据默认开启；已下线滑条归零不生效', () => {
  const initial = defaults();
  const legacy = defaults();
  delete legacy.sectionVisibility.portraitBlemishes;
  assert.equal(normalizeLoadedAdjustments(legacy).sectionVisibility.portraitBlemishes, true);
  const retired = {
    ...initial,
    portrait: {
      ...initial.portrait,
      textureRetention: 90,
      blemishRemoval: 60,
      beauty: 80,
      white: 70,
    },
    sectionVisibility: { ...initial.sectionVisibility, portraitSkin: false },
  };
  assert.equal(hasPortraitEffect(retired), false);
  const loaded = normalizeLoadedAdjustments(JSON.parse(JSON.stringify(retired)));
  assert.equal(loaded.portrait.textureRetention, 0);
  assert.equal(loaded.portrait.blemishRemoval, 0);
  assert.equal(loaded.portrait.beauty, 0);
  assert.equal(loaded.portrait.white, 0);
  const changed = {
    ...initial,
    portrait: { ...initial.portrait, faceShineRemoval: 60 },
    sectionVisibility: { ...initial.sectionVisibility, portraitSkin: false },
  };
  assert.equal(hasPortraitEffect(changed), true);
  useEditorStore.getState().resetHistory(initial);
  useEditorStore.getState().setEditor({ adjustments: changed });
  useEditorStore.getState().pushHistory(changed);
  useEditorStore.getState().undo();
  assert.equal(useEditorStore.getState().adjustments.portrait.faceShineRemoval, 0);
  useEditorStore.getState().redo();
  const reopened = normalizeLoadedAdjustments(JSON.parse(JSON.stringify(useEditorStore.getState().adjustments)));
  assert.equal(reopened.portrait.faceShineRemoval, 60);
  reopened.sectionVisibility.portraitBlemishes = false;
  assert.equal(hasPortraitEffect(reopened), false);
  reopened.sectionVisibility.portraitSkin = true;
  reopened.portrait.frequencySeparationFace = 50;
  assert.equal(hasPortraitEffect(reopened), true);
});

test('分析签名排除强度、裁剪及调色，包含几何和实际补丁内容', () => {
  const initial = defaults();
  const changed = defaults();
  changed.portrait.neutralGrayEvenness = 100;
  changed.portrait.neutralGrayContour = 100;
  changed.portrait.frequencySeparationFace = 100;
  changed.portrait.frequencySeparationBody = 100;
  changed.portrait.blemishRemoval = 80;
  changed.portrait.textureRetention = 70;
  changed.portrait.faceShineRemoval = 90;
  changed.sectionVisibility.portraitSkin = false;
  changed.sectionVisibility.portraitBlemishes = false;
  changed.crop = { x: 20, y: 10, width: 100, height: 100 };
  changed.exposure = 2;
  assert.equal(portraitSourceSignature(initial), portraitSourceSignature(changed));
  changed.rotation = 7;
  assert.notEqual(portraitSourceSignature(initial), portraitSourceSignature(changed));
  // AI 美肤开关与强度都不进入分析签名：滑块调整与分区显隐不重新识别人脸。
  const aiOn = defaults();
  aiOn.portrait.aiBlemishRemoval = 10;
  const aiStrong = defaults();
  aiStrong.portrait.aiBlemishRemoval = 50;
  assert.equal(portraitSourceSignature(initial), portraitSourceSignature(aiOn));
  assert.equal(portraitSourceSignature(aiOn), portraitSourceSignature(aiStrong));
  const aiHidden = structuredClone(aiOn);
  aiHidden.sectionVisibility.portraitBlemishes = false;
  aiHidden.sectionVisibility.portraitSkin = false;
  assert.equal(portraitSourceSignature(aiOn), portraitSourceSignature(aiHidden));
  const patched = { ...initial, aiPatches: [{ id: 'same-id', patchData: { color: 'aaaa', mask: 'mask' } }] };
  const modified = structuredClone(patched);
  modified.aiPatches[0].patchData.color = 'bbbb';
  assert.notEqual(portraitSourceSignature(patched), portraitSourceSignature(modified));
});

test('祛油光独立数值契约及分组重置，不依赖普通祛斑或 AI', () => {
  for (const [value, expected] of [
    [50.5, 51],
    [200, 100],
    [-1, 0],
    ['50', 0],
    [NaN, 0],
    [Infinity, 0],
    [null, 0],
  ]) {
    assert.equal(normalizePortraitAdjustments({ faceShineRemoval: value }).faceShineRemoval, expected);
  }
  const initial = defaults();
  const shine = defaults();
  shine.portrait.faceShineRemoval = 70;
  shine.sectionVisibility.portraitSkin = false;
  assert.equal(hasPortraitEffect(shine), true);
  assert.equal(portraitSourceSignature(shine), portraitSourceSignature(initial));
  shine.portrait.aiBlemishRemoval = 50;
  assert.equal(portraitSourceSignature(shine), portraitSourceSignature(initial));
  shine.portrait.faceShineRemoval = 100;
  assert.equal(portraitSourceSignature(shine), portraitSourceSignature(initial));
  assert.equal(hasPortraitEffect(shine), true);
  for (const key of portraitSectionKeys('blemishes')) shine.portrait[key] = INITIAL_PORTRAIT_ADJUSTMENTS[key];
  assert.equal(hasPortraitEffect(shine), false);
});

test('SSR：祛油光单独显示状态，AI 开启不禁用祛油光，无脸时禁用', async (t) => {
  const snapshot = useEditorStore.getState();
  t.after(() => useEditorStore.setState(snapshot));
  await i18n.changeLanguage('zh-CN');
  const adjustments = defaults();
  adjustments.portrait.faceShineRemoval = 65;
  adjustments.sectionVisibility.portraitSkin = false;
  const status = { path: selectedImage.path, phase: 'ready', summary: { faceCount: 1 } };
  const label = i18n.t(PORTRAIT_LABELS.faceShineRemoval);
  const shineInput = (html) =>
    (html.match(/<input[^>]*>/g) || []).find(
      (tag) => tag.includes('type="range"') && tag.includes(`aria-label="${label}"`),
    );
  for (const ai of [0, 60]) {
    adjustments.portrait.aiBlemishRemoval = ai;
    useEditorStore
      .getState()
      .setEditor({ selectedImage, adjustments: structuredClone(adjustments), portraitStatus: status });
    const html = panelMarkup();
    // ready 阶段不展示分析 status；仅断言祛油光滑块可用且不被 AI 禁用。
    assert.ok(shineInput(html));
    assert.ok(!shineInput(html).includes('disabled=""'));
    assert.ok(shineInput(html).includes('value="65"'));
  }
  useEditorStore.getState().setEditor({ portraitStatus: { ...status, phase: 'empty', summary: { faceCount: 0 } } });
  assert.ok(shineInput(panelMarkup()).includes('disabled=""'));
  adjustments.portrait.aiBlemishRemoval = 0;
  adjustments.sectionVisibility.portraitBlemishes = false;
  useEditorStore.getState().setEditor({ adjustments: structuredClone(adjustments), portraitStatus: status });
  assert.ok(!panelMarkup().includes('role="status"'));
});

test('人像不进入通用调整、蒙版或复制预设字段', () => {
  assert.ok(!COPYABLE_ADJUSTMENT_KEYS.includes('portrait'));
  assert.ok(!Object.keys(ADJUSTMENT_SECTIONS).includes('portrait'));
  assert.ok(!Object.values(ADJUSTMENT_SECTIONS).flat().includes('portrait'));
});

test('默认布局和旧布局：调整、人像、裁剪相邻，迁移幂等', () => {
  const expected = [Panel.Adjustments, Panel.Portrait, Panel.Crop, Panel.Masks, Panel.Ai, Panel.Presets];
  const workspace = reconcileWorkspace(undefined, false);
  assert.deepEqual(workspace.panelLayout.rightTop, expected);
  workspace.panelLayout.rightTop = expected.filter((panel) => panel !== Panel.Portrait);
  workspace.activePanels.rightTop = Panel.Crop;
  workspace.rightPanelWidth = 420;
  const migrated = reconcileWorkspace(workspace, false);
  assert.deepEqual(migrated.panelLayout.rightTop, expected);
  assert.equal(migrated.activePanels.rightTop, Panel.Crop);
  assert.equal(migrated.rightPanelWidth, 420);
  assert.deepEqual(reconcileWorkspace(migrated, false), migrated);
});

test('旧布局缺少右侧锚点时按规则插入，已有自定义位置保留', () => {
  const workspace = reconcileWorkspace(undefined, false);
  workspace.panelLayout.rightTop = [Panel.Crop, Panel.Masks, Panel.Ai, Panel.Presets];
  workspace.panelLayout.leftBottom = [Panel.Adjustments];
  assert.deepEqual(reconcileWorkspace(workspace, false).panelLayout.rightTop.slice(0, 2), [Panel.Portrait, Panel.Crop]);
  workspace.panelLayout.rightTop = [Panel.Masks, Panel.Ai, Panel.Presets];
  workspace.panelLayout.leftBottom.push(Panel.Crop);
  assert.equal(reconcileWorkspace(workspace, false).panelLayout.rightTop[0], Panel.Portrait);
  workspace.panelLayout.leftBottom.push(Panel.Portrait);
  const migrated = reconcileWorkspace(workspace, false);
  assert.deepEqual(migrated.panelLayout.leftBottom, [Panel.Adjustments, Panel.Crop, Panel.Portrait]);
  assert.equal(
    Object.values(migrated.panelLayout)
      .flat()
      .filter((panel) => panel === Panel.Portrait).length,
    1,
  );
});

test('SSR：中文空状态、加载禁用、皮肤主滑块标签及默认展开状态', async () => {
  await i18n.changeLanguage('zh-CN');
  useEditorStore.getState().setEditor({ selectedImage: null, adjustments: defaults() });
  assert.ok(panelMarkup().includes(i18n.t('editor.ai.noImageSelected')));
  assert.ok(!panelMarkup().includes('type="range"'));
  useEditorStore.getState().setEditor({ selectedImage: { ...selectedImage, isReady: false } });
  let html = panelMarkup();
  // 主滑块为 0 时子带隐藏：皮肤 12 + 祛疵 2 + 妆容 2 + 瘦脸 1 + 大眼 1 + 身体 3。
  assert.equal((html.match(/type="range"/g) || []).length, 21);
  assert.equal((html.match(/<input(?=[^>]*type="range")[^>]*disabled=""/g) || []).length, 21);
  useEditorStore.getState().setEditor({
    selectedImage,
    portraitStatus: { path: selectedImage.path, phase: 'idle', generation: 0, source: '' },
  });
  html = panelMarkup();
  assert.equal((html.match(/aria-expanded="true"/g) || []).length, 2);
  assert.equal((html.match(/aria-expanded="false"/g) || []).length, 4);
  // 美肤检测中阻断皮肤/祛疵/妆容；重塑与大眼由自身守卫。
  assert.equal((html.match(/<input(?=[^>]*type="range")[^>]*disabled=""/g) || []).length, 14);
  assert.ok(html.includes(i18n.t('editor.portrait.faceDetecting')));
  // 子带未挂载；只断言默认可见的主滑块标签。
  for (const key of [
    'neutralGrayEvenness',
    'neutralGrayContour',
    'frequencySeparationFace',
    'frequencySeparationBody',
    'hydratedSmoothingFace',
    'hydratedSmoothingBody',
    'evenSkinFace',
    'evenSkinBody',
    'whiteningFace',
    'whiteningBody',
    ...portraitSectionKeys('blemishes'),
    'slimFace',
    'bigEye',
  ]) {
    const label = i18n.t(PORTRAIT_LABELS[key]);
    assert.ok(html.includes(`aria-label="${label}"`), `${key} => ${label}`);
  }
});

test('SSR：处理阶段、错误重试、切图和归零提示隔离，状态不写入调整', async () => {
  await i18n.changeLanguage('zh-CN');
  const adjustments = defaults();
  adjustments.portrait.neutralGrayEvenness = 50;
  const source = portraitSourceSignature(adjustments);
  for (const phase of ['loading', 'analyzing', 'ready', 'empty', 'partial', 'error']) {
    const portraitStatus = {
      path: selectedImage.path,
      generation: 10,
      source,
      phase,
      summary: { key: 'test-analysis', faceCount: 2, skippedFaces: 1, hasSkin: true },
      error: phase === 'error' ? '缺少 retinaface_r34.onnx' : undefined,
    };
    useEditorStore.getState().setEditor({ selectedImage, adjustments, portraitStatus });
    const html = panelMarkup();
    // ready 不展示分析 status；其余非 idle 阶段在有美肤效果时展示。
    assert.equal(html.includes('role="status"'), phase !== 'ready');
    if (phase !== 'ready') {
      assert.ok(html.includes(i18n.t(`editor.portrait.status.${phase}`, { count: 2 })));
    }
    const blocked = phase === 'loading' || phase === 'analyzing' || phase === 'error';
    // 美肤阶段阻断皮肤+祛疵（子带隐藏）；ready 时这些滑条可用。
    const disabledCount = (html.match(/<input(?=[^>]*type="range")[^>]*disabled=""/g) || []).length;
    if (blocked) assert.ok(disabledCount >= 10);
    assert.ok(
      html.includes(
        blocked
          ? phase === 'error'
            ? i18n.t('editor.portrait.faceDetected', { count: 2 })
            : i18n.t('editor.portrait.faceDetecting')
          : i18n.t('editor.portrait.faceDetected', { count: 2 }),
      ),
    );
    assert.equal(html.includes(i18n.t('editor.portrait.status.retry')), phase === 'error');
    if (phase === 'error') assert.ok(html.includes('缺少 retinaface_r34.onnx'));
    assert.ok(!JSON.stringify(useEditorStore.getState().adjustments).includes('test-analysis'));
    useEditorStore.getState().setEditor({ selectedImage: { ...selectedImage, path: 'another.jpg' } });
    assert.ok(!panelMarkup().includes('role="status"'));
    useEditorStore.getState().setEditor({ selectedImage, adjustments: defaults() });
    assert.equal(panelMarkup().includes('role="status"'), phase === 'error');
  }
});

test('SSR：仅磨皮时显示分析错误和重试，无脸时身体磨皮仍禁用', async () => {
  await i18n.changeLanguage('zh-CN');
  for (const key of ['frequencySeparationFace', 'frequencySeparationBody']) {
    const adjustments = defaults();
    adjustments.portrait[key] = 60;
    const status = {
      path: selectedImage.path,
      generation: 20,
      source: portraitSourceSignature(adjustments),
      phase: 'error',
      error: '磨皮分析失败',
    };
    useEditorStore.getState().setEditor({ selectedImage, adjustments, portraitStatus: status });
    let html = panelMarkup();
    assert.ok(html.includes('role="status"'));
    assert.ok(html.includes('磨皮分析失败'));
    assert.ok(html.includes(i18n.t('editor.portrait.status.retry')));
    // 单侧高低频主滑块>0：皮肤 12（含该侧子带）+ 祛疵 2 + 妆容 2 = 16；眼白色彩不跟 faceBlocked。
    assert.equal((html.match(/<input(?=[^>]*type="range")[^>]*disabled=""/g) || []).length, 16);
    for (const faceCount of [0, 1]) {
      useEditorStore.getState().setEditor({
        portraitStatus: {
          ...status,
          phase: 'ready',
          error: undefined,
          summary: { key: 'test', faceCount, skippedFaces: 0, hasSkin: true },
        },
      });
      html = panelMarkup();
      // 无脸时阻断皮肤/祛疵/妆容；有脸后这些滑条可用（重塑/大眼可能仍禁用）。
      const disabledCount = (html.match(/<input(?=[^>]*type="range")[^>]*disabled=""/g) || []).length;
      if (faceCount === 0) assert.equal(disabledCount, 16);
      else assert.ok(disabledCount <= 2);
    }
    useEditorStore.getState().setEditor({
      adjustments: { ...adjustments, sectionVisibility: { ...adjustments.sectionVisibility, portraitSkin: false } },
    });
    assert.ok(!panelMarkup().includes('role="status"'));
    useEditorStore.getState().setEditor({ adjustments: defaults() });
    assert.ok(!panelMarkup().includes('role="status"'));
  }
});

test('SSR：祛斑分区仅保留 AI/祛油光；零参数分析错误可重试', async () => {
  await i18n.changeLanguage('zh-CN');
  const adjustments = defaults();
  adjustments.sectionVisibility.portraitSkin = false;
  const status = {
    path: selectedImage.path,
    generation: 30,
    source: portraitSourceSignature(adjustments),
    phase: 'ready',
    summary: { key: 'blemish-test', faceCount: 1, skippedFaces: 0, hasSkin: true },
  };
  useEditorStore.getState().setEditor({ selectedImage, adjustments, portraitStatus: status });
  let html = panelMarkup();
  assert.ok(!html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.blemishRemoval)}"`));
  assert.ok(!html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.textureRetention)}"`));
  assert.ok(html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.aiBlemishRemoval)}"`));
  assert.ok(html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.faceShineRemoval)}"`));
  assert.ok(!html.includes('role="status"'));
  adjustments.sectionVisibility.portraitBlemishes = false;
  useEditorStore.getState().setEditor({ adjustments: structuredClone(adjustments) });
  assert.ok(!panelMarkup().includes('role="status"'));
  useEditorStore.getState().setEditor({
    adjustments: defaults(),
    portraitStatus: { ...status, phase: 'error', error: '瑕疵分析失败', summary: undefined },
  });
  html = panelMarkup();
  assert.ok(html.includes(i18n.t('editor.portrait.status.retry')));
  assert.ok(html.includes('瑕疵分析失败'));
  // 美肤错误阻断磨皮/祛疵滑块（子带隐藏）。
  assert.ok((html.match(/<input(?=[^>]*type="range")[^>]*disabled=""/g) || []).length >= 10);
});

test('AI 美肤：数值契约、独立启用、显隐签名与历史恢复', () => {
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: 1 }).aiBlemishRemoval, 1);
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: 100 }).aiBlemishRemoval, 100);
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: 'true' }).aiBlemishRemoval, 0);
  assert.equal(normalizePortraitAdjustments({ aiBlemishRemoval: null }).aiBlemishRemoval, 0);
  const initial = defaults();
  const ai = defaults();
  ai.portrait.aiBlemishRemoval = 100;
  assert.equal(hasPortraitEffect(ai), true);
  // AI 强度不进入分析签名（与「分析签名排除强度」一致）。
  assert.equal(portraitSourceSignature(initial), portraitSourceSignature(ai));
  const hidden = structuredClone(ai);
  hidden.sectionVisibility.portraitBlemishes = false;
  assert.equal(hasPortraitEffect(hidden), false);
  assert.equal(portraitSourceSignature(initial), portraitSourceSignature(hidden));
  useEditorStore.getState().resetHistory(initial);
  useEditorStore.getState().setEditor({ adjustments: ai });
  useEditorStore.getState().pushHistory(ai);
  useEditorStore.getState().undo();
  assert.equal(useEditorStore.getState().adjustments.portrait.aiBlemishRemoval, 0);
  useEditorStore.getState().redo();
  assert.equal(
    normalizeLoadedAdjustments(JSON.parse(JSON.stringify(useEditorStore.getState().adjustments))).portrait
      .aiBlemishRemoval,
    100,
  );
  for (const key of portraitSectionKeys('blemishes')) ai.portrait[key] = INITIAL_PORTRAIT_ADJUSTMENTS[key];
  assert.equal(hasPortraitEffect(ai), false);
});

test('SSR：祛斑分区不再展示普通祛斑与质感保留', async () => {
  await i18n.changeLanguage('zh-CN');
  const adjustments = defaults();
  adjustments.portrait.aiBlemishRemoval = 100;
  useEditorStore.getState().setEditor({
    selectedImage,
    adjustments,
    portraitStatus: { path: selectedImage.path, phase: 'ready', summary: { faceCount: 1 } },
  });
  const html = panelMarkup();
  assert.ok(!html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.blemishRemoval)}"`));
  assert.ok(!html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.textureRetention)}"`));
  assert.ok(html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.aiBlemishRemoval)}"`));
  assert.ok(html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.faceShineRemoval)}"`));
});

test('SSR：应用中文案显示在分组标题旁而不是面板顶栏', async () => {
  await i18n.changeLanguage('zh-CN');
  const title = i18n.t('editor.portrait.sections.blemishes');
  const applying = i18n.t('editor.portrait.status.applying');
  const html = renderToStaticMarkup(
    createElement(
      CollapsibleSection,
      {
        title,
        isOpen: true,
        isContentVisible: true,
        onToggle() {},
        headerStatus: applying,
      },
      'slider-body',
    ),
  );
  const titleAt = html.indexOf(title);
  const applyingAt = html.indexOf(applying);
  const bodyAt = html.indexOf('slider-body');
  assert.ok(titleAt >= 0 && applyingAt > titleAt && applyingAt < bodyAt);
});

const previewMetadata = (requestId = 11) => ({
  path: 'photo-A.raw',
  stamp: {
    requestId,
    sourceRevision: 7,
    epoch: 19,
    compositeKey: 'a'.repeat(64),
    reshapeKey: 'b'.repeat(64),
    windowKey: 'c'.repeat(64),
  },
  mode: 'jpeg',
  dimensions: [101, 73],
  roi: { x: 3, y: 5, width: 17, height: 19 },
});
const previewEnvelope = (metadata, body = new Uint8Array([255, 216, 255, 217])) => {
  const json = new TextEncoder().encode(JSON.stringify(metadata));
  const buffer = new ArrayBuffer(8 + json.length + body.length);
  const view = new DataView(buffer);
  view.setUint32(0, 0x52525632, false);
  view.setUint32(4, json.length, true);
  new Uint8Array(buffer, 8, json.length).set(json);
  new Uint8Array(buffer, 8 + json.length).set(body);
  return buffer;
};

test('08 RRV2：JPEG／原生帧共用完整版本，图像体不含协议头', () => {
  const { decodePreviewResponse } = module.exports;
  const metadata = previewMetadata();
  const decoded = decodePreviewResponse(previewEnvelope(metadata));
  assert.deepEqual(decoded.metadata, metadata);
  assert.deepEqual([...new Uint8Array(decoded.image)], [255, 216, 255, 217]);
  const native = { ...metadata, mode: 'native', roi: { x: 0, y: 0, width: 101, height: 73 } };
  assert.equal(decodePreviewResponse(previewEnvelope(native, new Uint8Array())).image.byteLength, 0);
});

test('08 RRV2：拒绝截断、非法 UTF-8、旧协议、超限尺寸与混淆窗口', () => {
  const { decodePreviewResponse } = module.exports;
  const valid = previewEnvelope(previewMetadata());
  for (const length of [0, 1, 7, 8, 10, valid.byteLength - 5])
    assert.throws(() => decodePreviewResponse(valid.slice(0, length)));
  for (const length of [0, 65537, 0xffffffff]) {
    const bad = valid.slice(0);
    new DataView(bad).setUint32(4, length, true);
    assert.throws(() => decodePreviewResponse(bad));
  }
  const badUtf8 = valid.slice(0);
  new Uint8Array(badUtf8)[8] = 0xff;
  assert.throws(() => decodePreviewResponse(badUtf8));
  assert.throws(() => decodePreviewResponse(new TextEncoder().encode('WGPU_RENDER').buffer));
  for (const change of [
    { mode: 'raw' },
    { dimensions: [0, 73] },
    { dimensions: [32769, 1] },
    { dimensions: [10000, 10000] },
    { roi: { x: -1, y: 0, width: 17, height: 19 } },
    { roi: { x: 90, y: 0, width: 17, height: 19 } },
    { roi: { x: 0, y: 0, width: 0, height: 19 } },
    { mode: 'native' },
    { stamp: { ...previewMetadata().stamp, requestId: Number.MAX_SAFE_INTEGER + 1 } },
    { stamp: { ...previewMetadata().stamp, windowKey: 'unversioned' } },
  ])
    assert.throws(() => decodePreviewResponse(previewEnvelope({ ...previewMetadata(), ...change })));
  assert.throws(() => decodePreviewResponse(previewEnvelope(previewMetadata(), new Uint8Array())));
});

test('08 发布：统计早到最多一份，只有全部 stamp 字段相等才随帧提交', () => {
  const { acceptPreviewFrame, acceptPreviewAnalytics } = module.exports;
  const metadata = previewMetadata();
  let state = {
    ...editorSnapshot,
    selectedImage: { path: metadata.path, isReady: true },
    previewRequest: { path: metadata.path, requestId: 11 },
  };
  const analytics = { path: metadata.path, stamp: metadata.stamp, histogram: { red: [3] }, waveform: null };
  state = { ...state, ...acceptPreviewAnalytics(state, analytics) };
  assert.equal(state.histogram, null);
  assert.equal(state.pendingPreviewAnalytics, analytics);
  const replacement = { ...analytics, histogram: { red: [4] } };
  state = { ...state, ...acceptPreviewAnalytics(state, replacement) };
  assert.equal(state.pendingPreviewAnalytics, replacement);
  state = { ...state, ...acceptPreviewFrame(state, metadata) };
  assert.deepEqual(state.histogram, replacement.histogram);
  assert.equal(state.pendingPreviewAnalytics, null);
  for (const key of Object.keys(metadata.stamp)) {
    const changed = {
      ...metadata.stamp,
      [key]: typeof metadata.stamp[key] === 'number' ? metadata.stamp[key] + 1 : 'd'.repeat(64),
    };
    assert.deepEqual(acceptPreviewAnalytics(state, { ...analytics, stamp: changed }), {});
  }
  const next = previewMetadata(12);
  state = { ...state, previewRequest: { path: next.path, requestId: 12 } };
  assert.deepEqual(acceptPreviewAnalytics(state, analytics), {});
  state = { ...state, ...acceptPreviewFrame(state, next) };
  assert.deepEqual(state.histogram, replacement.histogram);
  assert.equal(state.waveform, null);
});

test('08 发布：失败、A→B→A、同路径重载和历史切换均拒绝迟到数据', () => {
  const { acceptPreviewFrame, acceptPreviewAnalytics } = module.exports;
  const metadata = previewMetadata();
  useEditorStore.setState({
    ...editorSnapshot,
    selectedImage: { path: metadata.path, isReady: true },
    previewRequest: { path: metadata.path, requestId: 11 },
    finalPreviewUrl: 'previous-success',
  });
  const analytics = { path: metadata.path, stamp: metadata.stamp, histogram: { red: [3] }, waveform: null };
  useEditorStore.getState().setEditor({ previewError: 'GPU allocation failed' });
  assert.deepEqual(acceptPreviewFrame(useEditorStore.getState(), metadata), {});
  assert.deepEqual(acceptPreviewAnalytics(useEditorStore.getState(), analytics), {});
  assert.equal(useEditorStore.getState().finalPreviewUrl, 'previous-success');
  for (const image of [
    { path: 'photo-B.raw', isReady: true },
    { path: metadata.path, isReady: true },
    { path: metadata.path, isReady: false },
  ]) {
    useEditorStore.getState().setEditor({ selectedImage: image });
    assert.equal(useEditorStore.getState().previewRequest, null);
    assert.deepEqual(acceptPreviewFrame(useEditorStore.getState(), metadata), {});
  }
  useEditorStore.getState().setEditor({ selectedImage: { path: metadata.path, isReady: true } });
  useEditorStore.getState().setEditor({ previewRequest: { path: metadata.path, requestId: 12 } });
  assert.deepEqual(acceptPreviewFrame(useEditorStore.getState(), metadata), {});
  useEditorStore.getState().setEditor({ adjustments: defaults() });
  assert.equal(useEditorStore.getState().previewRequest, null);
  for (const action of [
    () => useEditorStore.getState().resetHistory(defaults()),
    () => useEditorStore.getState().goToHistoryIndex(0),
  ]) {
    useEditorStore.getState().setEditor({ previewRequest: { path: metadata.path, requestId: 13 } });
    action();
    assert.equal(useEditorStore.getState().previewRequest, null);
  }
  useEditorStore.setState(editorSnapshot);
});

test('08 修肤延迟：中性化 GPUPixel／修肤', () => {
  const { deferredPortraitRetouch } = module.exports;
  const source = {
    ...INITIAL_PORTRAIT_ADJUSTMENTS,
    hydratedSmoothingFace: 81,
    aiBlemishRemoval: 70,
    slimFace: 31,
    smallFace: 47,
    cheekbones: 63,
    jawline: 91,
    bigEye: 55,
  };
  const copy = structuredClone(source);
  const result = deferredPortraitRetouch(source);
  assert.deepEqual(source, copy);
  // runtime：推迟修肤不透传瘦脸／大眼（由 deferredPortraitReshape 旁路）。
  assert.equal(result.slimFace, 0);
  assert.equal(result.bigEye, 0);
  assert.equal(result.smallFace, 0);
  assert.equal(result.hydratedSmoothingFace, 0);
});

test('08 预览推迟：重塑 Bypass 且修肤中性化后不再触发人像预览效果', () => {
  const { deferredPortraitRetouch, deferredPortraitReshape, hasPortraitReshapePreviewEffect, INITIAL_ADJUSTMENTS } =
    module.exports;
  const original = {
    ...INITIAL_ADJUSTMENTS,
    portrait: {
      ...INITIAL_PORTRAIT_ADJUSTMENTS,
      aiBlemishRemoval: 50,
      frequencySeparationFace: 40,
      slimFace: 31,
      bigEye: 55,
    },
    portraitReshape: {
      ...createPortraitReshape(),
      enabled: true,
      faces: [{ ...reshapeFace(), overrides: { slimFace: 40 } }],
    },
  };
  assert.equal(hasPortraitEffect(original), true);
  assert.equal(hasPortraitReshapePreviewEffect(original), true);
  const deferred = {
    ...original,
    portrait: deferredPortraitRetouch(original.portrait),
    portraitReshape: deferredPortraitReshape(original.portraitReshape),
  };
  assert.equal(deferred.portraitReshape.enabled, false);
  assert.equal(deferred.portraitReshape.faces.length, 1);
  assert.equal(hasPortraitEffect(deferred), false);
  assert.equal(hasPortraitReshapePreviewEffect(deferred), false);
  assert.equal(original.portraitReshape.enabled, true);
  // 拖调色时前端用同一 Bypass 载荷；松手后再恢复 enabled 的完整重塑。
  const release = {
    ...deferred,
    portraitReshape: original.portraitReshape,
    portrait: original.portrait,
  };
  assert.equal(hasPortraitReshapePreviewEffect(release), true);
  assert.equal(hasPortraitEffect(release), true);
});

test('SSR：移动端横向及纵向入口顺序，中英文与回退文案', async () => {
  for (const language of ['zh-CN', 'en', 'de']) {
    await i18n.changeLanguage(language);
    for (const key of PORTRAIT_KEYS) assert.notEqual(i18n.t(PORTRAIT_LABELS[key]), PORTRAIT_LABELS[key]);
    for (const placement of ['bottom', 'right']) {
      const html = renderToStaticMarkup(
        createElement(MobilePanelSwitcher, {
          activePanel: Panel.Portrait,
          onPanelSelect: () => {},
          placement,
        }),
      );
      const adjust = html.indexOf(`aria-label="${i18n.t('editor.switcher.tooltips.adjust')}"`);
      const portrait = html.indexOf(`aria-label="${i18n.t('editor.switcher.tooltips.portrait')}"`);
      const crop = html.indexOf(`aria-label="${i18n.t('editor.switcher.tooltips.crop')}"`);
      assert.ok(adjust >= 0 && adjust < portrait && portrait < crop);
      assert.equal((html.match(/aria-pressed="true"/g) || []).length, 1);
    }
  }
});

// —— 09 逐脸面板：决策矩阵、写入转换、复位/启用、坐标映射、选择守卫与 SSR ——

const reshapePanelStatus = (faces, selectedFaceId = null, phase = 'ready') => ({
  path: selectedImage.path,
  generation: 40,
  source: 'a'.repeat(64),
  phase,
  faces,
  selectedFaceId,
});
const panelDecision = (overrides = {}) =>
  decideReshapePanel({
    reshapeRead: readPortraitReshape(
      overrides.portraitReshape === undefined ? createPortraitReshape() : overrides.portraitReshape,
    ),
    portrait: overrides.portrait ?? {},
    status: overrides.status ?? { path: selectedImage.path, phase: 'idle', faces: [], selectedFaceId: null },
    selectedPath: overrides.selectedPath ?? selectedImage.path,
  });

test('09 决策：激活矩阵、编辑门槛与单脸继承值合并', () => {
  // 无分析（idle）即 active 可写：拖动滑块进入新语义（02 第 7 条）。
  let decision = panelDecision();
  assert.equal(decision.activation, 'active');
  assert.equal(decision.editable, true);
  assert.deepEqual(decision.values, { slimFace: 0, smallFace: 0, cheekbones: 0, jawline: 0 });
  assert.deepEqual(decision.uvValues, { slimFace: 0, bigEye: 0 });
  assert.equal(decision.engineVersion, 'gpupixel-runtime-v1');

  // 旧记录非零未激活：明确启用动作之前一律只读；零值旧记录不构成 legacy-inactive。
  decision = panelDecision({ portrait: { slimFace: 80 }, portraitReshape: null });
  assert.equal(decision.activation, 'legacy-inactive');
  assert.equal(decision.editable, false);
  assert.equal(panelDecision({ portrait: {}, portraitReshape: null }).activation, 'active');

  const future = reshapeCases.find((entry) => entry.status === 'unsupported').input.portraitReshape;
  assert.equal(panelDecision({ portraitReshape: future }).activation, 'unsupported');
  assert.equal(panelDecision({ portraitReshape: [] }).activation, 'invalid');

  // 重塑分析自身失败或有结果但无可处理脸才阻断；美肤状态不参与决策。
  decision = panelDecision({ status: reshapePanelStatus([], null, 'error') });
  assert.equal(decision.analysisBlocking, true);
  assert.equal(decision.editable, false);
  decision = panelDecision({
    status: reshapePanelStatus([{ ...analysisFace('a'), processable: false, reason: 'unsupported-pose' }]),
  });
  assert.equal(decision.processableCount, 0);
  assert.equal(decision.analysisBlocking, true);
  decision = panelDecision({
    status: reshapePanelStatus([analysisFace('a'), { ...analysisFace('b'), processable: false }]),
  });
  assert.equal(decision.processableCount, 1);
  assert.equal(decision.editable, true);

  // 选中脸只读该脸 overrides（不继承全局默认）；未选中或路径不符回退全局展示。
  const config = {
    ...createPortraitReshape(),
    enabled: true,
    faces: [
      { ...reshapeFace(), faceId: 'a', overrides: { uvSlimFace: 70 } },
      { ...reshapeFace(), faceId: 'b', overrides: { uvSlimFace: 20 } },
    ],
  };
  decision = panelDecision({
    portrait: { slimFace: 30 },
    portraitReshape: config,
    status: reshapePanelStatus([analysisFace('a'), analysisFace('b')], 'a'),
  });
  assert.equal(decision.selectedFace.faceId, 'a');
  assert.deepEqual(decision.values, { slimFace: 0, smallFace: 0, cheekbones: 0, jawline: 0 });
  assert.deepEqual(decision.uvValues, { slimFace: 70, bigEye: 0 });
  assert.equal(decision.selectedFaceHasOverride, true);
  assert.equal(decision.anyOverride, true);
  decision = panelDecision({
    portrait: { slimFace: 30 },
    portraitReshape: config,
    status: reshapePanelStatus([analysisFace('a'), analysisFace('b')], null),
  });
  assert.deepEqual(decision.uvValues, { slimFace: 30, bigEye: 0 });
  assert.equal(decision.selectedFaceHasOverride, false);
  decision = panelDecision({
    portrait: { smallFace: 30 },
    portraitReshape: config,
    status: reshapePanelStatus([analysisFace('a')], 'a'),
    selectedPath: 'other.jpg',
  });
  assert.equal(decision.selectedFace, null);
  assert.deepEqual(decision.values, { slimFace: 0, smallFace: 0, cheekbones: 0, jawline: 0 });
  assert.deepEqual(decision.uvValues, { slimFace: 0, bigEye: 0 });
});

test('09 写入：单脸 override、拒写无选中脸与旧记录', () => {
  const global = defaults();
  // 本用例覆盖网格引擎多滑块写入；默认引擎为 GPUPixel UV。
  global.portraitReshape = { ...createPortraitReshape(), engineVersion: 'gpupixel-runtime-v1' };
  // idle：摘要为空时暂存到 portrait 全局并启用，供分析就绪后迁移。
  const staged = applyReshapeSliderChange(global, { slimFace: 40 }, { selectedFaceId: null, faces: [] });
  assert.equal(staged.portrait.slimFace, 40);
  assert.equal(staged.portraitReshape.enabled, true);
  assert.equal(applyReshapeSliderChange(global, {}, { selectedFaceId: null, faces: [] }), null);

  // 单脸 override：锚点来自摘要，全局默认清零，越界值归一化。
  const faceWrite = applyReshapeSliderChange(
    global,
    { slimFace: 150, jawline: -5, cheekbones: NaN },
    {
      selectedFaceId: 'a',
      faces: [analysisFace('a')],
    },
  );
  assert.equal(faceWrite.portrait.slimFace, 0);
  assert.equal(faceWrite.portraitReshape.enabled, true);
  assert.deepEqual(faceWrite.portraitReshape.faces[0].overrides, { uvSlimFace: 100 });
  assert.deepEqual(faceWrite.portraitReshape.faces[0].sourceAnchor, analysisFace('a').sourceAnchor);
  // 已有脸条目：保留保护配置，只改 override。
  const withFace = {
    ...global,
    portraitReshape: {
      ...createPortraitReshape(),
      engineVersion: 'gpupixel-runtime-v1',
      enabled: true,
      faces: [structuredClone(reshapeFace())],
    },
  };
  const edited = applyReshapeSliderChange(
    withFace,
    { slimFace: 30 },
    {
      selectedFaceId: 'face-1',
      faces: [analysisFace('face-1')],
    },
  );
  assert.equal(edited.portraitReshape.faces[0].protection.hair, true);
  assert.deepEqual(edited.portraitReshape.faces[0].overrides, { uvSlimFace: 30 });

  // 摘要缺失或不可处理的脸不写入。
  assert.equal(applyReshapeSliderChange(global, { slimFace: 1 }, { selectedFaceId: 'a', faces: [] }), null);
  assert.equal(
    applyReshapeSliderChange(
      global,
      { slimFace: 1 },
      {
        selectedFaceId: 'b',
        faces: [{ ...analysisFace('b'), processable: false }],
      },
    ),
    null,
  );

  // 旧记录非零未激活 / 未知版本 / 损坏：一律拒绝。
  const future = reshapeCases.find((entry) => entry.status === 'unsupported').input.portraitReshape;
  const legacyActive = { ...global, portrait: { ...global.portrait, slimFace: 80 }, portraitReshape: null };
  assert.equal(applyReshapeSliderChange(legacyActive, { slimFace: 20 }, { selectedFaceId: null, faces: [] }), null);
  assert.equal(
    applyReshapeSliderChange(
      { ...global, portraitReshape: future },
      { slimFace: 1 },
      { selectedFaceId: 'a', faces: [analysisFace('a')] },
    ),
    null,
  );
  assert.equal(
    applyReshapeSliderChange(
      { ...global, portraitReshape: 'corrupt' },
      { slimFace: 1 },
      {
        selectedFaceId: 'a',
        faces: [analysisFace('a')],
      },
    ),
    null,
  );
});

test('扩展重塑：网格键读入剥离，仅 UV／大眼保留', () => {
  const { hasPortraitReshapeRequest, UV_SLIM_FACE_STORAGE_KEY } = module.exports;

  const config = {
    ...createPortraitReshape(),
    engineVersion: 'gpupixel-runtime-v1',
    enabled: true,
    faces: [
      {
        ...reshapeFace(),
        overrides: { noseWidth: 60, eyeDistance: 150, [UV_SLIM_FACE_STORAGE_KEY]: 40 },
      },
    ],
  };
  const parsed = readPortraitReshape(config);
  assert.equal(parsed.status, 'supported');
  assert.deepEqual(parsed.settings.faces[0].overrides, { uvSlimFace: 40 });
  assert.equal(
    hasPortraitReshapeRequest({ portrait: INITIAL_PORTRAIT_ADJUSTMENTS, portraitReshape: parsed.settings }, ['face-1']),
    true,
  );

  const uv = defaults();
  uv.portraitReshape = { ...createPortraitReshape(), engineVersion: 'gpupixel-runtime-v1' };
  const uvWrite = applyReshapeSliderChange(uv, { slimFace: 20 }, { selectedFaceId: 'a', faces: [analysisFace('a')] });
  assert.equal(uvWrite.portraitReshape.faces[0].overrides.uvSlimFace, 20);
  assert.equal(uvWrite.portraitReshape.faces[0].overrides.noseWidth, undefined);
});

test('09 复位与启用：清 override 恢复继承，旧记录需显式启用', () => {
  const config = () => ({
    ...createPortraitReshape(),
    enabled: true,
    faces: [
      { ...reshapeFace(), faceId: 'a', overrides: { uvSlimFace: 40 } },
      { ...reshapeFace(), faceId: 'b', overrides: { uvSlimFace: 25 } },
    ],
  });
  const single = clearFaceOverrides(config(), 'a');
  assert.deepEqual(single.faces[0].overrides, {});
  assert.deepEqual(single.faces[1].overrides, { uvSlimFace: 25 });
  assert.equal(clearFaceOverrides(config(), 'missing'), null);
  assert.equal(clearFaceOverrides(null, 'a'), null);

  const original = config();
  const unified = clearAllFaceOverrides(original);
  assert.deepEqual(
    unified.faces.map((face) => face.overrides),
    [{}, {}],
  );
  // readPortraitReshape 内部深拷贝：锚点与保护配置按结构保留而非同引用。
  assert.deepEqual(unified.faces[0].sourceAnchor, original.faces[0].sourceAnchor);
  assert.deepEqual(unified.faces[1].protection, original.faces[1].protection);
  assert.equal(
    clearAllFaceOverrides({ ...createPortraitReshape(), faces: [{ ...reshapeFace(), overrides: {} }] }),
    null,
  );

  // 显式启用：旧非零四项带入新结构作全局默认，旧对象非契约字段不迁移，原输入不被改写。
  const legacy = defaults();
  legacy.portrait.slimFace = 80;
  delete legacy.portraitReshape;
  const activated = activateLegacyReshape(legacy);
  assert.equal(activated.portrait.slimFace, 80);
  assert.equal(activated.portraitReshape.enabled, true);
  assert.deepEqual(activated.portraitReshape.faces, []);
  assert.equal(legacy.portraitReshape, undefined);
  assert.equal(activateLegacyReshape(defaults()), null);
  const legacyObject = { ...createPortraitReshape(), faces: [{ ...reshapeFace(), overrides: { slimFace: 55 } }] };
  delete legacyObject.schemaVersion;
  const migrated = activateLegacyReshape({ ...defaults(), portraitReshape: legacyObject });
  assert.equal(migrated.portraitReshape.enabled, true);
  assert.deepEqual(migrated.portraitReshape.faces, []);
});

test('09 画面框：线性几何映射与无效输入拒绝', () => {
  const geometry = (overrides = {}) => ({
    sourceWidth: 100,
    sourceHeight: 200,
    orientationSteps: 0,
    flipHorizontal: false,
    flipVertical: false,
    rotation: 0,
    crop: null,
    ...overrides,
  });
  assert.deepEqual(reshapeFaceFrame([0.25, 0.25, 0.5, 0.5], geometry()), {
    x: 0.25,
    y: 0.25,
    width: 0.5,
    height: 0.5,
  });
  assert.deepEqual(reshapeFaceFrame([0.25, 0.25, 0.25, 0.25], geometry({ flipHorizontal: true })), {
    x: 0.5,
    y: 0.25,
    width: 0.25,
    height: 0.25,
  });
  // EXIF 旋转：每步 (x, y) → (h - y, x) 并交换宽高。
  assert.deepEqual(reshapeFaceFrame([0.25, 0.25, 0.5, 0.5], geometry({ orientationSteps: 1 })), {
    x: 0.25,
    y: 0.25,
    width: 0.5,
    height: 0.5,
  });
  // 直角旋转：绕半像素中心 (w/2+0.5, h/2+0.5)，允许浮点误差。
  const rotated = reshapeFaceFrame([0.25, 0.25, 0.5, 0.5], geometry({ rotation: 90 }));
  assert.ok(Math.abs(rotated.x - 0.01) < 1e-9);
  assert.ok(Math.abs(rotated.y - 0.375) < 1e-9);
  assert.ok(Math.abs(rotated.width - 1) < 1e-9);
  assert.ok(Math.abs(rotated.height - 0.25) < 1e-9);
  // 裁剪：百分比与像素两种单位。
  assert.deepEqual(
    reshapeFaceFrame([0.25, 0.25, 0.5, 0.5], geometry({ crop: { unit: '%', x: 10, y: 20, width: 50, height: 40 } })),
    { x: 0.3, y: 0.125, width: 1, height: 1.25 },
  );
  assert.deepEqual(
    reshapeFaceFrame([0.25, 0.25, 0.5, 0.5], geometry({ crop: { x: 10, y: 20, width: 50, height: 40 } })),
    { x: 0.3, y: 0.75, width: 1, height: 2.5 },
  );
  assert.equal(reshapeFaceFrame([0.25, 0.25, 0.5, 0.5], geometry({ sourceWidth: 0 })), null);
});

test('09 选择守卫：路径、就绪与可处理脸同时满足才写入运行态', () => {
  const state = {
    selectedImage: { path: 'A.jpg', isReady: true },
    portraitReshapeStatus: {
      path: 'A.jpg',
      phase: 'ready',
      faces: [analysisFace('a'), { ...analysisFace('b'), processable: false, reason: 'low-confidence' }],
      selectedFaceId: null,
    },
  };
  assert.equal(selectPortraitReshapeFace(state, 'a').portraitReshapeStatus.selectedFaceId, 'a');
  // 单人脸模式：null / 非法 id 回落到第一张可处理脸，不清空选择。
  assert.equal(selectPortraitReshapeFace(state, null).portraitReshapeStatus.selectedFaceId, 'a');
  assert.equal(selectPortraitReshapeFace(state, 'b').portraitReshapeStatus.selectedFaceId, 'a');
  assert.equal(selectPortraitReshapeFace(state, 'missing').portraitReshapeStatus.selectedFaceId, 'a');
  assert.deepEqual(selectPortraitReshapeFace({ ...state, selectedImage: { path: 'B.jpg', isReady: true } }, 'a'), {});
  assert.deepEqual(selectPortraitReshapeFace({ ...state, selectedImage: { path: 'A.jpg', isReady: false } }, 'a'), {});
});

test('09 原因码与面板文案：中英文完整，六种原因码都有映射', async () => {
  assert.equal(Object.keys(PORTRAIT_RESHAPE_REASON_LABELS).length, 6);
  for (const language of ['zh-CN', 'en']) {
    await i18n.changeLanguage(language);
    for (const key of Object.values(PORTRAIT_RESHAPE_REASON_LABELS)) assert.notEqual(i18n.t(key), key);
    for (const key of [
      'faceItem',
      'faceSelect',
      'resetOverrides',
      'activateLegacy',
      'activateLegacyHint',
      'unsupportedData',
      'invalidData',
      'analysisFailed',
      'analysisFailedHint',
      'noProcessableFace',
      'protectionLimited',
      'protectionLimitedHint',
      'overrideIndicator',
      'extendedGroup',
      'extendedPreviewPending',
    ]) {
      assert.notEqual(i18n.t(`editor.portrait.reshape.${key}`), `editor.portrait.reshape.${key}`);
    }
    for (const key of module.exports.EXTENDED_RESHAPE_KEYS) {
      assert.notEqual(i18n.t(PORTRAIT_LABELS[key]), PORTRAIT_LABELS[key]);
    }
  }
});

test('SSR：09 逐脸列表、禁用原因、override 与保护限幅提示', async (t) => {
  const snapshot = useEditorStore.getState();
  const uiSnapshot = useUIStore.getState();
  t.after(() => {
    useEditorStore.setState(snapshot);
    useUIStore.setState(uiSnapshot);
  });
  await i18n.changeLanguage('zh-CN');
  const faces = [analysisFace('face-1'), { ...analysisFace('face-2'), processable: false, reason: 'unsupported-pose' }];
  useEditorStore.getState().setEditor({ selectedImage, adjustments: defaults() });
  useEditorStore.getState().setEditor({
    portraitReshapeStatus: reshapePanelStatus(faces, 'face-1'),
  });
  // 人脸 listbox 在人像标题下方统一展示，不依赖子模块是否展开。
  useUIStore.setState({
    activePanel: Panel.Portrait,
    portraitSectionsState: { skin: false, blemishes: false, reshape: false, eyes: false },
  });
  let html = panelMarkup();
  assert.equal(html.split('role="listbox"').length, 2);
  assert.ok(!html.includes(i18n.t('editor.portrait.reshape.allFaces')));
  assert.ok(html.includes(i18n.t('editor.portrait.reshape.faceItem', { index: 2 })));
  assert.ok(html.includes(i18n.t(PORTRAIT_RESHAPE_REASON_LABELS['unsupported-pose'])));
  assert.ok(html.includes('aria-disabled="true"'));

  useUIStore.setState({
    portraitSectionsState: { skin: false, blemishes: false, reshape: true, eyes: false },
  });
  html = panelMarkup();
  // 展开重塑后仍只有一处 listbox（不在子模块内重复）。
  assert.equal(html.split('role="listbox"').length, 2);
  // 选中 face-1 无 override：滑块为 0，不显示单脸参数提示。
  assert.ok(!html.includes('data-testid="reshape-override-indicator"'));
  assert.ok(!html.includes('data-testid="reshape-protection-limited"'));
  const slim = (markup) =>
    (markup.match(/<input[^>]*>/g) || []).find((tag) =>
      tag.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.slimFace)}"`),
    );
  assert.ok(slim(html).includes('value="0"'));
  assert.ok(!slim(html).includes('disabled=""'));

  const withOverride = defaults();
  withOverride.portraitReshape = {
    ...createPortraitReshape(),
    engineVersion: 'gpupixel-runtime-v1',
    enabled: true,
    faces: [{ ...reshapeFace(), faceId: 'face-1', overrides: { uvSlimFace: 30 } }],
  };
  useEditorStore.getState().setEditor({ adjustments: withOverride });
  html = panelMarkup();
  assert.ok(html.includes('data-testid="reshape-override-indicator"'));
  assert.ok(html.includes(i18n.t('editor.portrait.reshape.overrideIndicator')));
  assert.ok(slim(html).includes('value="30"'));
  assert.ok(!html.includes('data-testid="reshape-extended-group"'));
  assert.ok(!html.includes(`aria-label="${i18n.t(PORTRAIT_LABELS.noseWidth)}"`));
  useEditorStore.getState().setEditor({ adjustments: withOverride });

  useEditorStore.getState().setEditor({
    portraitReshapeStatus: reshapePanelStatus([{ ...analysisFace('face-1'), reason: 'protection-limited' }], 'face-1'),
  });
  html = panelMarkup();
  assert.ok(html.includes('data-testid="reshape-protection-limited"'));
  assert.ok(html.includes(i18n.t('editor.portrait.reshape.protectionLimited')));

  // 旧记录非零未激活：提示 + 明确启用按钮，滑块只读。
  const legacy = defaults();
  legacy.portrait.slimFace = 80;
  delete legacy.portraitReshape;
  useEditorStore
    .getState()
    .setEditor({ adjustments: legacy, portraitReshapeStatus: reshapePanelStatus(faces, 'face-1') });
  html = panelMarkup();
  assert.ok(html.includes(i18n.t('editor.portrait.reshape.activateLegacy')));
  assert.ok(html.includes(i18n.t('editor.portrait.reshape.activateLegacyHint')));
  assert.ok(slim(html).includes('disabled=""'));

  // 重塑分析失败：显示错误与重试，不写入调整参数。
  useEditorStore.getState().setEditor({
    adjustments: defaults(),
    portraitReshapeStatus: { ...reshapePanelStatus([], null, 'error'), error: '模型加载失败' },
  });
  html = panelMarkup();
  assert.ok(html.includes(i18n.t('editor.portrait.reshape.analysisFailed')));
  assert.ok(html.includes('模型加载失败'));
  assert.ok(html.includes(i18n.t('editor.portrait.status.retry')));
  assert.ok(!JSON.stringify(useEditorStore.getState().adjustments).includes('模型加载失败'));

  // 美肤错误阻断皮肤/祛疵/妆容；重塑滑块保持可拖动。
  const skinOnly = defaults();
  useEditorStore.getState().setEditor({
    adjustments: skinOnly,
    portraitReshapeStatus: reshapePanelStatus(faces, 'face-1'),
    portraitStatus: {
      path: selectedImage.path,
      generation: 9,
      source: portraitSourceSignature(skinOnly),
      phase: 'error',
      error: '磨皮分析失败',
      summary: undefined,
    },
  });
  html = panelMarkup();
  // 皮肤无子带展开：10 + 祛疵 2 + 妆容 2 = 14；眼白色彩不跟 faceBlocked。
  assert.equal((html.match(/<input(?=[^>]*type="range")[^>]*disabled=""/g) || []).length, 14);
  assert.ok(!slim(html).includes('disabled=""'));
  assert.ok(html.includes('磨皮分析失败'));
});

test('SSR：09 画面框随面板聚焦显示，对比原图与拖动时隐藏', async (t) => {
  const editorState = useEditorStore.getState();
  const uiState = useUIStore.getState();
  t.after(() => {
    useEditorStore.setState(editorState);
    useUIStore.setState(uiState);
  });
  await i18n.changeLanguage('zh-CN');
  const faces = [
    { ...analysisFace('face-1'), sourceAnchor: { sourceBaseHash: 'a'.repeat(64), bounds: [0.25, 0.25, 0.5, 0.5] } },
    {
      ...analysisFace('face-2'),
      sourceAnchor: { sourceBaseHash: 'a'.repeat(64), bounds: [0.5, 0.5, 0.25, 0.25] },
      processable: false,
      reason: 'unsupported-pose',
    },
  ];
  useEditorStore.setState({
    ...editorState,
    selectedImage: { path: selectedImage.path, isReady: true, width: 800, height: 600 },
    adjustments: defaults(),
    showOriginal: false,
    isSliderDragging: false,
    portraitReshapeStatus: reshapePanelStatus(faces, 'face-1'),
  });
  useUIStore.setState({ activePanel: Panel.Portrait, activePanels: { rightTop: Panel.Adjustments } });
  const renderSize = {
    width: 400,
    height: 300,
    offsetX: 50,
    offsetY: 25,
    scale: 0.5,
    containerWidth: 500,
    containerHeight: 350,
  };
  const overlayMarkup = () => {
    // 与 panelMarkup 同一 SSR 模式：React SSR 经 getServerSnapshot 读初始态对象，
    // 先把当前态合并进快照对象才能让组件渲染当前状态。
    Object.assign(editorSnapshot, useEditorStore.getState());
    Object.assign(uiSnapshot, useUIStore.getState());
    return renderToStaticMarkup(createElement(PortraitReshapeFaceOverlay, { imageRenderSize: renderSize }));
  };
  let html = overlayMarkup();
  assert.ok(html.includes('data-testid="reshape-face-overlay"'));
  assert.ok(html.includes('left:50px;top:25px;width:400px;height:300px'));
  assert.ok(html.includes('left:25%;top:25%;width:50%;height:50%'));
  assert.ok(html.includes('left:50%;top:50%;width:25%;height:25%'));

  useEditorStore.getState().setEditor({ showOriginal: true });
  assert.equal(overlayMarkup(), '');
  useEditorStore.getState().setEditor({ showOriginal: false, isSliderDragging: true });
  assert.equal(overlayMarkup(), '');
  useEditorStore.getState().setEditor({ isSliderDragging: false });
  useUIStore.setState({ activePanel: Panel.Adjustments, activePanels: { rightTop: Panel.Adjustments } });
  assert.equal(overlayMarkup(), '');
});

test('15 runtime 引擎：读写 bigEye、引擎切换与启用判定', () => {
  const {
    applyReshapeEngineChange,
    isUvReshapeEngine,
    isRuntimeReshapeEngine,
    normalizeUvReshapeStrengths,
    UV_SLIM_FACE_STORAGE_KEY,
  } = module.exports;

  assert.deepEqual(normalizeUvReshapeStrengths({ slimFace: 40, bigEye: 150 }), {
    slimFace: 40,
    bigEye: 100,
  });

  const uvConfig = {
    ...createPortraitReshape(),
    engineVersion: 'gpupixel-runtime-v1',
    enabled: true,
    faces: [{ ...reshapeFace(), overrides: { [UV_SLIM_FACE_STORAGE_KEY]: 50, bigEye: 40 } }],
  };
  const parsed = readPortraitReshape(uvConfig);
  assert.equal(parsed.status, 'supported');
  assert.equal(parsed.settings.engineVersion, 'gpupixel-runtime-v1');
  assert.equal(parsed.settings.faces[0].overrides.bigEye, 40);
  assert.equal(parsed.settings.faces[0].overrides.uvSlimFace, 50);
  assert.equal(isRuntimeReshapeEngine(parsed.settings.engineVersion), true);
  assert.equal(isUvReshapeEngine(parsed.settings.engineVersion), false);
  assert.equal(
    hasPortraitReshapeRequest({ portrait: INITIAL_PORTRAIT_ADJUSTMENTS, portraitReshape: uvConfig }, ['face-1']),
    true,
  );

  const meshOnly = {
    ...createPortraitReshape(),
    engineVersion: 'gpupixel-runtime-v1',
    enabled: true,
    faces: [{ ...reshapeFace(), overrides: { smallFace: 30 } }],
  };
  assert.equal(
    hasPortraitReshapeRequest({ portrait: INITIAL_PORTRAIT_ADJUSTMENTS, portraitReshape: meshOnly }, ['face-1']),
    false,
  );
  const uvZeroMesh = {
    ...createPortraitReshape(),
    engineVersion: 'gpupixel-runtime-v1',
    enabled: true,
    faces: [{ ...reshapeFace(), overrides: { smallFace: 30 } }],
  };
  assert.equal(
    hasPortraitReshapeRequest({ portrait: INITIAL_PORTRAIT_ADJUSTMENTS, portraitReshape: uvZeroMesh }, ['face-1']),
    false,
  );

  // 网格 slimFace 不计入 GPUPixel UV 请求。
  const meshSlimOnly = {
    ...createPortraitReshape(),
    engineVersion: 'gpupixel-runtime-v1',
    enabled: true,
    faces: [{ ...reshapeFace(), overrides: { slimFace: 80 } }],
  };
  assert.equal(
    hasPortraitReshapeRequest({ portrait: INITIAL_PORTRAIT_ADJUSTMENTS, portraitReshape: meshSlimOnly }, ['face-1']),
    false,
  );

  const switched = applyReshapeEngineChange(
    {
      ...INITIAL_ADJUSTMENTS,
      portraitReshape: { ...createPortraitReshape(), engineVersion: 'gpupixel-runtime-v1' },
    },
    'gpupixel-runtime-v1',
  );
  assert.equal(switched.portraitReshape.engineVersion, 'gpupixel-runtime-v1');
  assert.equal(switched.portraitReshape.enabled, true);
  assert.equal(Object.hasOwn(switched.portraitReshape, 'snapshot'), false);

  const decision = panelDecision({
    portraitReshape: uvConfig,
    status: reshapePanelStatus([analysisFace('face-1')], 'face-1'),
  });
  assert.equal(decision.engineVersion, 'gpupixel-runtime-v1');
  assert.deepEqual(decision.uvValues, { slimFace: 50, bigEye: 40 });

  const written = applyReshapeSliderChange(
    { ...INITIAL_ADJUSTMENTS, portraitReshape: uvConfig },
    { bigEye: 66 },
    { selectedFaceId: 'face-1', faces: [analysisFace('face-1')] },
  );
  assert.equal(written.portraitReshape.faces[0].overrides.bigEye, 66);
  assert.equal(written.portrait.bigEye, 0);

  // UV-only：slimFace 滑块写入 uvSlimFace。
  const uvAdj = {
    ...INITIAL_ADJUSTMENTS,
    portraitReshape: {
      ...createPortraitReshape(),
      enabled: true,
      engineVersion: 'gpupixel-runtime-v1',
      faces: [{ ...reshapeFace(), overrides: { [UV_SLIM_FACE_STORAGE_KEY]: 70 } }],
    },
  };
  const uvWrite = applyReshapeSliderChange(
    uvAdj,
    { slimFace: 88 },
    { selectedFaceId: 'face-1', faces: [analysisFace('face-1')] },
  );
  assert.equal(uvWrite.portraitReshape.faces[0].overrides.uvSlimFace, 88);

  const uvDecision = panelDecision({
    portraitReshape: uvWrite.portraitReshape,
    status: reshapePanelStatus([analysisFace('face-1')], 'face-1'),
  });
  assert.deepEqual(uvDecision.uvValues, { slimFace: 88, bigEye: 0 });

  // UV 下可写 bigEye，并保留已有 uvSlimFace。
  const meshBigEye = applyReshapeSliderChange(
    {
      ...INITIAL_ADJUSTMENTS,
      portraitReshape: {
        ...createPortraitReshape(),
        enabled: true,
        engineVersion: 'gpupixel-runtime-v1',
        faces: [{ ...reshapeFace(), overrides: { uvSlimFace: 20 } }],
      },
    },
    { bigEye: 45 },
    { selectedFaceId: 'face-1', faces: [analysisFace('face-1')] },
  );
  assert.equal(meshBigEye.portraitReshape.faces[0].overrides.bigEye, 45);
  assert.equal(meshBigEye.portraitReshape.faces[0].overrides.uvSlimFace, 20);
  assert.equal(
    hasPortraitReshapeRequest({ portrait: INITIAL_PORTRAIT_ADJUSTMENTS, portraitReshape: meshBigEye.portraitReshape }, [
      'face-1',
    ]),
    true,
  );
  assert.equal(
    hasPortraitReshapeRequest(
      {
        portrait: INITIAL_PORTRAIT_ADJUSTMENTS,
        portraitReshape: {
          ...createPortraitReshape(),
          enabled: true,
          engineVersion: 'gpupixel-runtime-v1',
          faces: [{ ...reshapeFace(), overrides: { bigEye: 40 } }],
        },
        sectionVisibility: { ...INITIAL_ADJUSTMENTS.sectionVisibility, portraitReshape: false },
      },
      ['face-1'],
    ),
    true,
  );
  assert.equal(
    hasPortraitReshapeRequest(
      {
        portrait: INITIAL_PORTRAIT_ADJUSTMENTS,
        portraitReshape: {
          ...createPortraitReshape(),
          enabled: true,
          engineVersion: 'gpupixel-runtime-v1',
          faces: [{ ...reshapeFace(), overrides: { bigEye: 40 } }],
        },
        sectionVisibility: { ...INITIAL_ADJUSTMENTS.sectionVisibility, portraitEyes: false },
      },
      ['face-1'],
    ),
    false,
  );
  assert.ok(meshBigEye);
  assert.equal(meshBigEye.portraitReshape.faces[0].overrides.bigEye, 45);
  // 已是 UV 且已启用时引擎切换为 no-op。
  assert.equal(applyReshapeEngineChange(meshBigEye, 'gpupixel-runtime-v1'), null);
});

import type { Crop } from 'react-image-crop';
import type { PortraitReshapeFaceSummary, PortraitReshapeStatus } from '../store/useEditorStore';
import {
  EYES_KEYS,
  UV_RESHAPE_KEYS,
  type Adjustments,
  type EyesKey,
  type PortraitAdjustments,
  type PortraitReshapeData,
  type PortraitReshapeFace,
  type PortraitReshapeReadResult,
  type PortraitReshapeReason,
  type PortraitReshapeV1,
  type ReshapeEngineVersion,
  type ReshapeSourceAnchor,
  type ReshapeStrengths,
  type UvReshapeKey,
  type UvReshapeStrengths,
  createPortraitReshape,
  normalizePortraitAdjustments,
  normalizeReshapeStrengths,
  normalizeUvReshapeStrengths,
  readPortraitReshape,
  UV_SLIM_FACE_STORAGE_KEY,
} from './adjustments';

// 09 ?????????????????adjustments??????????????11???
// ????????????????????????????????faceId??

export const PORTRAIT_RESHAPE_REASON_LABELS = {
  'no-face': 'editor.portrait.reshape.reasons.no-face',
  'model-error': 'editor.portrait.reshape.reasons.model-error',
  'low-confidence': 'editor.portrait.reshape.reasons.low-confidence',
  'unsupported-pose': 'editor.portrait.reshape.reasons.unsupported-pose',
  'protection-limited': 'editor.portrait.reshape.reasons.protection-limited',
  'unmatched-face': 'editor.portrait.reshape.reasons.unmatched-face',
} as const satisfies Record<PortraitReshapeReason, string>;

export type ReshapeActivation = 'active' | 'legacy-inactive' | 'unsupported' | 'invalid';

export interface ReshapePanelContext {
  reshapeRead: PortraitReshapeReadResult;
  portrait: PortraitAdjustments;
  status: Pick<PortraitReshapeStatus, 'path' | 'phase' | 'faces' | 'selectedFaceId'>;
  selectedPath: string | null;
}

export interface ReshapePanelDecision {
  activation: ReshapeActivation;
  editable: boolean;
  processableCount: number;
  analysisBlocking: boolean;
  selectedFace: PortraitReshapeFaceSummary | null;
  selectedFaceEntry: PortraitReshapeFace | null;
  values: ReshapeStrengths;
  uvValues: UvReshapeStrengths;
  engineVersion: ReshapeEngineVersion;
  anyOverride: boolean;
  selectedFaceHasOverride: boolean;
}

const analysisHasResult = (phase: PortraitReshapeStatus['phase']) => phase === 'ready' || phase === 'partial';

const hasLegacyUvStaging = (portrait: PortraitAdjustments) =>
  normalizeUvReshapeStrengths(portrait).slimFace !== 0 ||
  normalizeReshapeStrengths(portrait).slimFace !== 0;

const ZERO_MESH: ReshapeStrengths = {
  slimFace: 0,
  smallFace: 0,
  cheekbones: 0,
  jawline: 0,
};

const clearMeshPortrait = (portrait: PortraitAdjustments): PortraitAdjustments => ({
  ...portrait,
  slimFace: 0,
  smallFace: 0,
  cheekbones: 0,
  jawline: 0,
  noseWidth: 0,
  noseLength: 0,
  mouthSize: 0,
  lipThickness: 0,
  eyeAperture: 0,
  eyeDistance: 0,
  forehead: 0,
  chinProjection: 0,
  bigEye: 0,
});

export const decideReshapePanel = (context: ReshapePanelContext): ReshapePanelDecision => {
  const { reshapeRead, portrait, status, selectedPath } = context;
  const settings = reshapeRead.settings;
  const currentStatus = status.path === selectedPath ? status : null;
  const faces = currentStatus?.faces ?? [];
  const selectedFaceId = currentStatus?.selectedFaceId ?? null;
  const selectedFace = faces.find((face) => face.faceId === selectedFaceId) ?? null;
  const selectedFaceEntry = settings?.faces.find((face) => face.faceId === selectedFaceId) ?? null;
  const processableCount = faces.filter((face) => face.processable).length;

  let activation: ReshapeActivation;
  if (reshapeRead.status === 'unsupported') activation = 'unsupported';
  else if (reshapeRead.status === 'invalid') activation = 'invalid';
  else if (reshapeRead.status === 'legacy' && hasLegacyUvStaging(portrait))
    activation = 'legacy-inactive';
  else activation = 'active';

  const phase = currentStatus?.phase ?? 'idle';
  const analysisBlocking = phase === 'error' || (analysisHasResult(phase) && processableCount === 0);
  const editable = activation === 'active' && !analysisBlocking;

  const stagedUv = normalizeUvReshapeStrengths(portrait);
  const uvValues = selectedFaceEntry
    ? {
        slimFace: selectedFaceEntry.overrides.uvSlimFace ?? stagedUv.slimFace,
        bigEye: selectedFaceEntry.overrides.bigEye ?? stagedUv.bigEye,
      }
    : { slimFace: stagedUv.slimFace, bigEye: stagedUv.bigEye };

  const anyOverride = (settings?.faces ?? []).some((face) => Object.keys(face.overrides).length > 0);
  const selectedFaceHasOverride = selectedFaceEntry ? Object.keys(selectedFaceEntry.overrides).length > 0 : false;

  return {
    activation,
    editable,
    processableCount,
    analysisBlocking,
    selectedFace,
    selectedFaceEntry,
    values: ZERO_MESH,
    uvValues,
    engineVersion: 'gpupixel-runtime-v1',
    anyOverride,
    selectedFaceHasOverride,
  };
};

const reshapeFaceEntry = (
  settings: PortraitReshapeV1,
  faceId: string,
  anchor: ReshapeSourceAnchor,
): PortraitReshapeFace => {
  const existing = settings.faces.find((face) => face.faceId === faceId);
  if (existing) return existing;
  return { faceId, sourceAnchor: anchor, overrides: {}, protection: { hair: true, background: true, strokes: [] } };
};

const enabledSettings = (settings: PortraitReshapeV1): PortraitReshapeV1 =>
  settings.enabled ? settings : { ...settings, enabled: true };

export interface ReshapeWriteContext {
  selectedFaceId: string | null;
  faces: ReadonlyArray<PortraitReshapeFaceSummary>;
}

const amount = (value: unknown): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 0;
  return Math.round(Math.max(0, Math.min(100, value)));
};

export const applyReshapeSliderChange = (
  adjustments: Adjustments,
  changes: Partial<Record<UvReshapeKey | EyesKey, number>>,
  context: ReshapeWriteContext,
): Adjustments | null => {
  const read = readPortraitReshape(adjustments.portraitReshape);
  if (read.status === 'unsupported' || read.status === 'invalid') return null;
  if (read.status === 'legacy' && hasLegacyUvStaging(normalizePortraitAdjustments(adjustments.portrait)))
    return null;

  const settingsBase = read.settings ?? createPortraitReshape();
  const selectedFaceId =
    context.selectedFaceId ??
    context.faces.find((face) => face.processable)?.faceId ??
    null;

  if (selectedFaceId === null) {
    if (context.faces.length > 0) return null;
    const settings = enabledSettings(settingsBase);
    const portrait = { ...normalizePortraitAdjustments(adjustments.portrait) };
    let changed = settings.enabled !== true || read.settings == null;
    for (const key of UV_RESHAPE_KEYS) {
      if (!Object.hasOwn(changes, key)) continue;
      const value = amount(changes[key]);
      if (portrait[key] === value) continue;
      portrait[key] = value;
      changed = true;
    }
    for (const key of EYES_KEYS) {
      if (!Object.hasOwn(changes, key)) continue;
      const value = amount(changes[key]);
      if (portrait[key] === value) continue;
      portrait[key] = value;
      changed = true;
    }
    if (!changed) return null;
    return {
      ...adjustments,
      portrait,
      portraitReshape: { ...settings, engineVersion: 'gpupixel-runtime-v1', enabled: true },
    };
  }

  const summary = context.faces.find((face) => face.faceId === selectedFaceId);
  if (!summary?.processable) return null;

  const settings = enabledSettings(settingsBase);
  const entry = reshapeFaceEntry(settings, summary.faceId, summary.sourceAnchor);
  const overrides = { ...entry.overrides };
  let changed = false;
  for (const key of UV_RESHAPE_KEYS) {
    if (!Object.hasOwn(changes, key)) continue;
    const value = amount(changes[key]);
    const storageKey = UV_SLIM_FACE_STORAGE_KEY;
    if (overrides[storageKey] === value) continue;
    overrides[storageKey] = value;
    changed = true;
  }
  for (const key of EYES_KEYS) {
    if (!Object.hasOwn(changes, key)) continue;
    const value = amount(changes[key]);
    if (overrides[key] === value) continue;
    overrides[key] = value;
    changed = true;
  }
  if (!changed) return null;
  const nextEntry: PortraitReshapeFace = { ...entry, overrides };
  const faces = settings.faces.some((face) => face.faceId === nextEntry.faceId)
    ? settings.faces.map((face) => (face.faceId === nextEntry.faceId ? nextEntry : face))
    : [...settings.faces, nextEntry];
  // GPUPixel bake ? portrait.slimFace?bigEye??? override ???????????????
  const mirrored = clearMeshPortrait(normalizePortraitAdjustments(adjustments.portrait));
  if (overrides.uvSlimFace !== undefined) mirrored.slimFace = overrides.uvSlimFace;
  if (overrides.bigEye !== undefined) mirrored.bigEye = overrides.bigEye;
  return {
    ...adjustments,
    portrait: mirrored,
    portraitReshape: { ...settings, faces, engineVersion: 'gpupixel-runtime-v1', enabled: true },
  };
};

export const mirrorRuntimeStrengths = (
  portrait: PortraitAdjustments,
  faces: ReadonlyArray<PortraitReshapeFace>,
): PortraitAdjustments => {
  const next = clearMeshPortrait(portrait);
  let slim = 0;
  let bigEye = 0;
  for (const face of faces) {
    slim = Math.max(slim, amount(face.overrides.uvSlimFace));
    bigEye = Math.max(bigEye, amount(face.overrides.bigEye));
  }
  next.slimFace = slim;
  next.bigEye = bigEye;
  return next;
};

export const migrateStagedReshapeStrengths = (
  adjustments: Adjustments,
  processableFaces: ReadonlyArray<Pick<PortraitReshapeFaceSummary, 'faceId' | 'sourceAnchor' | 'processable'>>,
): Adjustments => {
  const target = processableFaces.find((face) => face.processable);
  if (!target) return adjustments;
  const read = readPortraitReshape(adjustments.portraitReshape);
  if (!read.settings) return adjustments;
  const settings = enabledSettings(read.settings);
  const stagedUv = normalizeUvReshapeStrengths(adjustments.portrait);
  const hasStaged = stagedUv.slimFace > 0 || stagedUv.bigEye > 0;
  const entry = reshapeFaceEntry(settings, target.faceId, target.sourceAnchor);
  const overrides = { ...entry.overrides };
  if (hasStaged) {
    if (stagedUv.slimFace > 0 && overrides[UV_SLIM_FACE_STORAGE_KEY] === undefined)
      overrides[UV_SLIM_FACE_STORAGE_KEY] = stagedUv.slimFace;
    if (stagedUv.bigEye > 0 && overrides.bigEye === undefined) overrides.bigEye = stagedUv.bigEye;
  }
  const nextEntry: PortraitReshapeFace = { ...entry, overrides };
  const faces = settings.faces.some((face) => face.faceId === nextEntry.faceId)
    ? settings.faces.map((face) => (face.faceId === nextEntry.faceId ? nextEntry : face))
    : [...settings.faces, nextEntry];
  return {
    ...adjustments,
    portrait: mirrorRuntimeStrengths(normalizePortraitAdjustments(adjustments.portrait), faces),
    portraitReshape: { ...settings, faces, engineVersion: 'gpupixel-runtime-v1', enabled: true },
  };
};

export const applyReshapeEngineChange = (
  adjustments: Adjustments,
  _engineVersion?: ReshapeEngineVersion,
): Adjustments | null => {
  const read = readPortraitReshape(adjustments.portraitReshape);
  if (read.status === 'unsupported' || read.status === 'invalid') return null;
  if (read.status === 'legacy' && hasLegacyUvStaging(normalizePortraitAdjustments(adjustments.portrait)))
    return null;
  const base = read.settings ?? createPortraitReshape();
  if (base.engineVersion === 'gpupixel-runtime-v1' && read.status === 'supported' && base.enabled) return null;
  const { snapshot: _snapshot, ...rest } = base as PortraitReshapeV1 & { snapshot?: unknown };
  return {
    ...adjustments,
    portraitReshape: { ...rest, engineVersion: 'gpupixel-runtime-v1', enabled: true },
  };
};

export const activateLegacyReshape = (adjustments: Adjustments): Adjustments | null => {
  const read = readPortraitReshape(adjustments.portraitReshape);
  if (read.status !== 'legacy') return null;
  const staged = normalizeUvReshapeStrengths(adjustments.portrait);
  const meshSlim = normalizeReshapeStrengths(adjustments.portrait).slimFace;
  return {
    ...adjustments,
    portrait: {
      ...normalizePortraitAdjustments(adjustments.portrait),
      slimFace: staged.slimFace || meshSlim,
      smallFace: 0,
      cheekbones: 0,
      jawline: 0,
    },
    portraitReshape: { ...createPortraitReshape(), enabled: true },
  };
};

export const clearFaceOverrides = (data: PortraitReshapeData, faceId: string): PortraitReshapeV1 | null => {
  const read = readPortraitReshape(data);
  const settings = read.settings;
  if (!settings || !settings.faces.some((face) => face.faceId === faceId)) return null;
  const faces = settings.faces.map((face) => (face.faceId === faceId ? { ...face, overrides: {} } : face));
  return { ...settings, faces };
};

export const clearAllFaceOverrides = (data: PortraitReshapeData): PortraitReshapeV1 | null => {
  const read = readPortraitReshape(data);
  const settings = read.settings;
  if (!settings || !settings.faces.some((face) => Object.keys(face.overrides).length > 0)) return null;
  const faces = settings.faces.map((face) => ({ ...face, overrides: {} }));
  return { ...settings, faces };
};

// ??Rust portrait_coordinates::CoordinateMap ??linear ?????
// ?????????????orientationSteps ??flip ??rotation?? (w/2+0.5,h/2+0.5)?? crop ????
// ??/??/?????????Rust ?????????????????????????????
// ???portrait ?????? phase ?? 'empty'?empty ????PortraitStatus ??????
// ??????idle/loading/analyzing/ready/partial/error ?????

export interface ReshapeFrameGeometry {
  sourceWidth: number;
  sourceHeight: number;
  orientationSteps: number;
  flipHorizontal: boolean;
  flipVertical: boolean;
  rotation: number;
  crop: Crop | null;
}

export interface NormalizedFrame {
  x: number;
  y: number;
  width: number;
  height: number;
}

const sourceToBasePixel = (
  x: number,
  y: number,
  geometry: ReshapeFrameGeometry,
): { x: number; y: number; w: number; h: number } | null => {
  let px = x;
  let py = y;
  let w = geometry.sourceWidth;
  let h = geometry.sourceHeight;
  if (!(w > 0 && h > 0) || !Number.isFinite(px) || !Number.isFinite(py)) return null;
  for (let step = 0; step < ((geometry.orientationSteps % 4) + 4) % 4; step++) {
    const nx = h - py;
    const ny = px;
    px = nx;
    py = ny;
    const swapped = w;
    w = h;
    h = swapped;
  }
  if (geometry.flipHorizontal) px = w - px;
  if (geometry.flipVertical) py = h - py;
  const angle = (geometry.rotation * Math.PI) / 180;
  if (Number.isFinite(angle)) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const cx = w / 2 + 0.5;
    const cy = h / 2 + 0.5;
    const rx = cos * px - sin * py + (cx - cos * cx + sin * cy);
    const ry = sin * px + cos * py + (cy - sin * cx - cos * cy);
    px = rx;
    py = ry;
  }
  if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
  return { x: px, y: py, w, h };
};

// ????? ????????B ??????????????????????
export const reshapeFaceFrame = (
  bounds: [number, number, number, number],
  geometry: ReshapeFrameGeometry,
): NormalizedFrame | null => {
  const [nx, ny, nw, nh] = bounds;
  const corners: Array<[number, number]> = [
    [nx, ny],
    [nx + nw, ny],
    [nx, ny + nh],
    [nx + nw, ny + nh],
  ];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let baseW = 0;
  let baseH = 0;
  for (const [u, v] of corners) {
    const mapped = sourceToBasePixel(u * geometry.sourceWidth, v * geometry.sourceHeight, geometry);
    if (!mapped) return null;
    minX = Math.min(minX, mapped.x);
    minY = Math.min(minY, mapped.y);
    maxX = Math.max(maxX, mapped.x);
    maxY = Math.max(maxY, mapped.y);
    baseW = mapped.w;
    baseH = mapped.h;
  }
  const crop = geometry.crop;
  let originX = 0;
  let originY = 0;
  let cropW = baseW;
  let cropH = baseH;
  if (crop && crop.width > 0 && crop.height > 0) {
    if (crop.unit === '%') {
      originX = ((crop.x ?? 0) / 100) * baseW;
      originY = ((crop.y ?? 0) / 100) * baseH;
      cropW = (crop.width / 100) * baseW;
      cropH = (crop.height / 100) * baseH;
    } else {
      originX = crop.x ?? 0;
      originY = crop.y ?? 0;
      cropW = crop.width;
      cropH = crop.height;
    }
  }
  if (!(cropW > 0 && cropH > 0)) return null;
  return {
    x: (minX - originX) / cropW,
    y: (minY - originY) / cropH,
    width: (maxX - minX) / cropW,
    height: (maxY - minY) / cropH,
  };
};

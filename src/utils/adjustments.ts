import { Crop } from 'react-image-crop';
import { v4 as uuidv4 } from 'uuid';
import { SubMask, SubMaskMode } from '../components/panel/right/Masks';
import type { AdjustmentLayout, AppSettings } from '../components/ui/AppProperties';
import type { WhiteBalance } from './whiteBalance';
import { defaultFrame, normalizeFrame, type FrameState } from './frame';

export enum ActiveChannel {
  Blue = 'blue',
  Green = 'green',
  Luma = 'luma',
  Red = 'red',
}

export enum DisplayMode {
  Luma = 'luma',
  Rgb = 'rgb',
  Parade = 'parade',
  Vectorscope = 'vectorscope',
  Histogram = 'histogram',
}

export enum PasteMode {
  Merge = 'merge',
  Replace = 'replace',
}

export interface CopyPasteSettings {
  mode: PasteMode;
  includedAdjustments: Array<string>;
  knownAdjustments: Array<string>;
  autoSync: boolean;
}

export enum BasicAdjustment {
  Blacks = 'blacks',
  Brightness = 'brightness',
  Contrast = 'contrast',
  Exposure = 'exposure',
  Highlights = 'highlights',
  Shadows = 'shadows',
  Whites = 'whites',
}

export enum ColorAdjustment {
  ColorGrading = 'colorGrading',
  Hsl = 'hsl',
  Hue = 'hue',
  Luminance = 'luminance',
  Saturation = 'saturation',
  Temperature = 'temperature',
  Tint = 'tint',
  Vibrance = 'vibrance',
  WhiteBalance = 'whiteBalance',
}

export enum ColorGrading {
  Balance = 'balance',
  Blending = 'blending',
  Global = 'global',
  Highlights = 'highlights',
  Midtones = 'midtones',
  Shadows = 'shadows',
}

export enum DetailsAdjustment {
  Clarity = 'clarity',
  Dehaze = 'dehaze',
  Structure = 'structure',
  Centré = 'centré',
  ColorNoiseReduction = 'colorNoiseReduction',
  LumaNoiseReduction = 'lumaNoiseReduction',
  Sharpness = 'sharpness',
  SharpnessThreshold = 'sharpnessThreshold',
  ChromaticAberrationRedCyan = 'chromaticAberrationRedCyan',
  ChromaticAberrationBlueYellow = 'chromaticAberrationBlueYellow',
}

export enum Effect {
  GrainAmount = 'grainAmount',
  GrainRoughness = 'grainRoughness',
  GrainSize = 'grainSize',
  LutData = 'lutData',
  LutIntensity = 'lutIntensity',
  LutName = 'lutName',
  LutPath = 'lutPath',
  LutSize = 'lutSize',
  LutIsSceneReferred = 'lutIsSceneReferred',
  VignetteAmount = 'vignetteAmount',
  VignetteFeather = 'vignetteFeather',
  VignetteMidpoint = 'vignetteMidpoint',
  VignetteRoundness = 'vignetteRoundness',
  LensBlurEnabled = 'lensBlurEnabled',
  LensBlurAmount = 'lensBlurAmount',
  lensBlurDiffusion = 'lensBlurDiffusion',
  LensBlurShape = 'lensBlurShape',
  LensBlurDepthMap = 'lensBlurDepthMap',
  LensBlurMinDepth = 'lensBlurMinDepth',
  LensBlurMaxDepth = 'lensBlurMaxDepth',
  LensBlurMinFade = 'lensBlurMinFade',
  LensBlurMaxFade = 'lensBlurMaxFade',
}

export enum CreativeAdjustment {
  GlowAmount = 'glowAmount',
  HalationAmount = 'halationAmount',
  FlareAmount = 'flareAmount',
}

enum TransformAdjustment {
  TransformDistortion = 'transformDistortion',
  TransformVertical = 'transformVertical',
  TransformHorizontal = 'transformHorizontal',
  TransformRotate = 'transformRotate',
  TransformAspect = 'transformAspect',
  TransformScale = 'transformScale',
  TransformXOffset = 'transformXOffset',
  TransformYOffset = 'transformYOffset',
}

export enum LensAdjustment {
  LensCorrectionMode = 'lensCorrectionMode',
  LensMaker = 'lensMaker',
  LensModel = 'lensModel',
  LensDistortionAmount = 'lensDistortionAmount',
  LensVignetteAmount = 'lensVignetteAmount',
  LensTcaAmount = 'lensTcaAmount',
  LensDistortionEnabled = 'lensDistortionEnabled',
  LensTcaEnabled = 'lensTcaEnabled',
  LensVignetteEnabled = 'lensVignetteEnabled',
}

export type GuideOrientation = 'vertical' | 'horizontal';

export interface GuideLine {
  id: string;
  type: GuideOrientation;
  p1: Coord;
  p2: Coord;
}

export interface GuidedPerspective {
  enabled: boolean;
  lines: GuideLine[];
  autoCrop: boolean;
}

export interface ColorCalibration {
  shadowsTint: number;
  redHue: number;
  redSaturation: number;
  greenHue: number;
  greenSaturation: number;
  blueHue: number;
  blueSaturation: number;
}

export interface ParametricCurveSettings {
  darks: number;
  shadows: number;
  highlights: number;
  lights: number;
  whiteLevel: number;
  blackLevel: number;
  split1: number;
  split2: number;
  split3: number;
}

export interface ParametricCurve {
  [index: string]: ParametricCurveSettings;
  blue: ParametricCurveSettings;
  green: ParametricCurveSettings;
  luma: ParametricCurveSettings;
  red: ParametricCurveSettings;
}

export type LevelsChannel = 'rgb' | 'red' | 'green' | 'blue';

export interface LevelsChannelSettings {
  inputBlack: number;
  gamma: number;
  inputWhite: number;
  outputBlack: number;
  outputWhite: number;
}

export interface LevelsSettings {
  [index: string]: LevelsChannelSettings;
  rgb: LevelsChannelSettings;
  red: LevelsChannelSettings;
  green: LevelsChannelSettings;
  blue: LevelsChannelSettings;
}

export const DEFAULT_LEVELS_CHANNEL: LevelsChannelSettings = {
  inputBlack: 0,
  gamma: 1,
  inputWhite: 255,
  outputBlack: 0,
  outputWhite: 255,
};

const clampLevelsNumber = (value: unknown, min: number, max: number, fallback: number): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
};

export const normalizeLevelsChannel = (value: unknown): LevelsChannelSettings => {
  const source = value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  let inputBlack = clampLevelsNumber(source.inputBlack, 0, 254, DEFAULT_LEVELS_CHANNEL.inputBlack);
  let inputWhite = clampLevelsNumber(source.inputWhite, 1, 255, DEFAULT_LEVELS_CHANNEL.inputWhite);
  if (inputBlack >= inputWhite) {
    inputBlack = Math.min(inputBlack, inputWhite - 1);
    inputWhite = Math.max(inputWhite, inputBlack + 1);
  }
  let outputBlack = clampLevelsNumber(source.outputBlack, 0, 254, DEFAULT_LEVELS_CHANNEL.outputBlack);
  let outputWhite = clampLevelsNumber(source.outputWhite, 1, 255, DEFAULT_LEVELS_CHANNEL.outputWhite);
  if (outputBlack >= outputWhite) {
    outputBlack = Math.min(outputBlack, outputWhite - 1);
    outputWhite = Math.max(outputWhite, outputBlack + 1);
  }
  return {
    inputBlack,
    gamma: clampLevelsNumber(source.gamma, 0.1, 9.99, DEFAULT_LEVELS_CHANNEL.gamma),
    inputWhite,
    outputBlack,
    outputWhite,
  };
};

export const getDefaultLevels = (): LevelsSettings => ({
  rgb: { ...DEFAULT_LEVELS_CHANNEL },
  red: { ...DEFAULT_LEVELS_CHANNEL },
  green: { ...DEFAULT_LEVELS_CHANNEL },
  blue: { ...DEFAULT_LEVELS_CHANNEL },
});

export const normalizeLevels = (value: unknown): LevelsSettings => {
  const source = value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
  return {
    rgb: normalizeLevelsChannel(source.rgb),
    red: normalizeLevelsChannel(source.red),
    green: normalizeLevelsChannel(source.green),
    blue: normalizeLevelsChannel(source.blue),
  };
};

export const isDefaultLevelsChannel = (settings: LevelsChannelSettings | undefined): boolean => {
  if (!settings) return true;
  return (
    settings.inputBlack === DEFAULT_LEVELS_CHANNEL.inputBlack &&
    settings.gamma === DEFAULT_LEVELS_CHANNEL.gamma &&
    settings.inputWhite === DEFAULT_LEVELS_CHANNEL.inputWhite &&
    settings.outputBlack === DEFAULT_LEVELS_CHANNEL.outputBlack &&
    settings.outputWhite === DEFAULT_LEVELS_CHANNEL.outputWhite
  );
};

export interface PortraitAdjustments {
  neutralGrayEvenness: number;
  neutralGrayContour: number;
  /** 中性灰立体 · 五官（结构高光）；仅当主滑块 > 0 时生效，默认 80。 */
  neutralGrayContourFeatures: number;
  /** 中性灰立体 · 轮廓（外轮廓压暗）；仅当主滑块 > 0 时生效，默认 80。 */
  neutralGrayContourOutline: number;
  frequencySeparationFace: number;
  frequencySeparationFaceLow: number;
  frequencySeparationFaceHigh: number;
  frequencySeparationBody: number;
  frequencySeparationBodyLow: number;
  frequencySeparationBodyHigh: number;
  /**
   * 高频轴版本：1=负柔化/正增强，2=正柔化/负增强。
   * 旧 sidecar 缺省标记时按 0..100 的保纹理量迁移；不入分区 UI。
   */
  frequencyHighBipolar: number;
  hydratedSmoothingFace: number;
  hydratedSmoothingBody: number;
  evenSkinFace: number;
  evenSkinBody: number;
  whiteningFace: number;
  whiteningBody: number;
  /** 已下线：GPUPixel 经典磨皮；旧 sidecar 经 normalize 强制归零。 */
  beauty: number;
  /** 已下线：GPUPixel 经典美白；旧 sidecar 经 normalize 强制归零。 */
  white: number;
  blemishRemoval: number;
  textureRetention: number;
  faceShineRemoval: number;
  aiBlemishRemoval: number;
  slimFace: number;
  smallFace: number;
  cheekbones: number;
  jawline: number;
  noseWidth: number;
  noseLength: number;
  mouthSize: number;
  lipThickness: number;
  eyeAperture: number;
  eyeDistance: number;
  forehead: number;
  chinProjection: number;
  bigEye: number;
  /** 提亮眼白（巩膜亮度）。 */
  eyeWhiteBrighten: number;
  /** 祛红血丝（巩膜去红）。 */
  eyeRedness: number;
  /** GPUPixel 口红。 */
  lipstick: number;
  /** GPUPixel 腮红。 */
  blusher: number;
  /** 原生引擎瘦手臂（`body.slim_arms`）；旧顶层 `slimArms` 忽略。 */
  bodySlimArms: number;
  /** 原生引擎瘦腰（`body.slim_waist`）。 */
  bodySlimWaist: number;
  /** 原生引擎瘦腿（`body.slim_legs`）。 */
  bodySlimLegs: number;
}

export type PortraitSection = 'skin' | 'blemishes' | 'makeup' | 'reshape' | 'eyes' | 'body';

export const RESHAPE_KEYS = ['slimFace', 'smallFace', 'cheekbones', 'jawline'] as const;
export type ReshapeKey = (typeof RESHAPE_KEYS)[number];
export type ReshapeStrengths = Record<ReshapeKey, number>;
/** 网格引擎遗留的扩展五官滑块；仅持久化，几何仍只消费 RESHAPE_KEYS。 */
export const EXTENDED_RESHAPE_KEYS = [
  'noseWidth',
  'noseLength',
  'mouthSize',
  'lipThickness',
  'eyeAperture',
  'eyeDistance',
  'forehead',
  'chinProjection',
] as const;
export type ExtendedReshapeKey = (typeof EXTENDED_RESHAPE_KEYS)[number];
export type ExtendedReshapeStrengths = Record<ExtendedReshapeKey, number>;
/** GPUPixel UV 引擎 UI 滑块：瘦脸（存储键为 uvSlimFace，与网格隔离）。 */
export const UV_RESHAPE_KEYS = ['slimFace'] as const;
export type UvReshapeKey = (typeof UV_RESHAPE_KEYS)[number];
/** 眼睛模块 UI 滑块；存储键为 overrides.bigEye，可叠加于 GPUPixel UV。 */
export const EYES_KEYS = ['bigEye'] as const;
export type EyesKey = (typeof EYES_KEYS)[number];
/** 眼睛色彩滑块（全局 portrait.*，非 reshape override）。 */
export const EYE_COLOR_KEYS = ['eyeWhiteBrighten', 'eyeRedness'] as const;
export type EyeColorKey = (typeof EYE_COLOR_KEYS)[number];
/** UV 强度结构仍含 bigEye，供读写与有效强度计算共用。 */
export type UvReshapeStrengths = Record<'slimFace' | 'bigEye', number>;
/** overrides 中 GPUPixel UV 瘦脸的存储键，避免与网格 slimFace 串值。 */
export const UV_SLIM_FACE_STORAGE_KEY = 'uvSlimFace' as const;
export type UvReshapeStorageKey = typeof UV_SLIM_FACE_STORAGE_KEY | 'bigEye';
/** 产品默认原版运行时；旧 UV／网格读入时迁到此值。 */
export type ReshapeEngineVersion = 'gpupixel-runtime-v1' | 'gpupixel-uv-v1';
export type PortraitReshapeReason =
  'no-face' | 'model-error' | 'low-confidence' | 'unsupported-pose' | 'protection-limited' | 'unmatched-face';

export interface ReshapeSourceAnchor {
  sourceBaseHash: string;
  // 未裁切照片空间，归一化的 x、y、width、height；不是显示画布坐标。
  bounds: [number, number, number, number];
}

export interface ReshapeProtectionStroke {
  mode: 'protect' | 'erase';
  radius: number;
  points: [number, number][];
}

export interface ReshapeProtection {
  hair: boolean;
  background: boolean;
  strokes: ReshapeProtectionStroke[];
}

export interface PortraitReshapeFace {
  faceId: string;
  sourceAnchor: ReshapeSourceAnchor;
  overrides: Partial<
    ReshapeStrengths & ExtendedReshapeStrengths & UvReshapeStrengths & { uvSlimFace: number }
  >;
  protection: ReshapeProtection;
}

export interface PortraitReshapeV1 {
  schemaVersion: 1;
  engineVersion: ReshapeEngineVersion;
  enabled: boolean;
  faces: PortraitReshapeFace[];
  // 11 场景级快照：由 Rust 保存流程注入，前端只做与 valid_snapshot 对齐的静态格式闸，
  // 不解码大数组。类型保持 unknown，消费者必须经 readPortraitReshape 校验后使用。
  snapshot?: unknown;
}

// 未知版本按原对象透传；仅 readPortraitReshape 返回的 settings 可供编辑和执行。
export type PortraitReshapeData = PortraitReshapeV1 | Readonly<Record<string, unknown>> | null;
export interface PortraitReshapeReadResult {
  status: 'legacy' | 'supported' | 'unsupported' | 'invalid';
  value: PortraitReshapeData;
  settings?: PortraitReshapeV1;
}

export const createPortraitReshape = (): PortraitReshapeV1 => ({
  schemaVersion: 1,
  engineVersion: 'gpupixel-runtime-v1',
  enabled: false,
  faces: [],
});

const reshapeRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const reshapeNumber = (value: unknown, min: number, max: number): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
const reshapeHash = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const reshapeId = (value: unknown): value is string =>
  typeof value === 'string' && /^[A-Za-z0-9_.:-]{1,128}$/.test(value);
const reshapeVector = (value: unknown, size: number, min = 0, max = 1): value is number[] =>
  Array.isArray(value) && value.length === size && value.every((n) => reshapeNumber(n, min, max));
const reshapeAmount = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.round(Math.max(0, Math.min(100, value))) : 0;

export const normalizeReshapeStrengths = (value: unknown): ReshapeStrengths => {
  const source = reshapeRecord(value) ? value : {};
  return {
    slimFace: reshapeAmount(source.slimFace),
    smallFace: reshapeAmount(source.smallFace),
    cheekbones: reshapeAmount(source.cheekbones),
    jawline: reshapeAmount(source.jawline),
  };
};

export const normalizeExtendedReshapeStrengths = (value: unknown): ExtendedReshapeStrengths => {
  const source = reshapeRecord(value) ? value : {};
  return {
    noseWidth: reshapeAmount(source.noseWidth),
    noseLength: reshapeAmount(source.noseLength),
    mouthSize: reshapeAmount(source.mouthSize),
    lipThickness: reshapeAmount(source.lipThickness),
    eyeAperture: reshapeAmount(source.eyeAperture),
    eyeDistance: reshapeAmount(source.eyeDistance),
    forehead: reshapeAmount(source.forehead),
    chinProjection: reshapeAmount(source.chinProjection),
  };
};

export const normalizeUvReshapeStrengths = (value: unknown): UvReshapeStrengths => {
  const source = reshapeRecord(value) ? value : {};
  return {
    slimFace: reshapeAmount(source.slimFace),
    bigEye: reshapeAmount(source.bigEye),
  };
};

export const isUvReshapeEngine = (engine: unknown): engine is 'gpupixel-uv-v1' =>
  engine === 'gpupixel-uv-v1';

export const isRuntimeReshapeEngine = (engine: unknown): engine is 'gpupixel-runtime-v1' =>
  engine === 'gpupixel-runtime-v1';

/** 旧 UV／网格引擎一并规范为 GPUPixel 原版运行时。 */
export const reshapeEngineVersion = (_value?: unknown): ReshapeEngineVersion =>
  'gpupixel-runtime-v1';

const stripMeshFaceOverrides = (
  overrides: PortraitReshapeFace['overrides'],
): PortraitReshapeFace['overrides'] => {
  const next: PortraitReshapeFace['overrides'] = {};
  if (overrides.uvSlimFace !== undefined) next.uvSlimFace = overrides.uvSlimFace;
  if (overrides.bigEye !== undefined) next.bigEye = overrides.bigEye;
  return next;
};

// —— 11 场景级快照静态格式闸：镜像 Rust portrait_snapshot::valid_snapshot 的有界结构检查 ——
// 只验证结构与上限，不解码保护图数据；损坏或超大快照在此拒绝执行（原值保留）。
const RESHAPE_SNAPSHOT_VERSION = 'reshape-snapshot-v1';
const RESHAPE_MESH_VERSION = 'mediapipe-478-v1';
const RESHAPE_MESH_VERTICES = 217;
const RESHAPE_MESH_TRIANGLES = 396;
const RESHAPE_MAX_MAP_SIDE = 256;
const RESHAPE_MAX_ENCODED_RUNS = 65536;
// base64(6 字节 × 65536 条 RLE) 的字符上限；解码前先按长度拒绝，避免无界分配。
const RESHAPE_MAX_MAP_BASE64 = Math.ceil((6 * RESHAPE_MAX_ENCODED_RUNS) / 3) * 4;
const RESHAPE_REASONS_MASK_MAX = (1 << 16) - 1;
const RESHAPE_REASONS: readonly string[] = [
  'no-face',
  'model-error',
  'low-confidence',
  'unsupported-pose',
  'protection-limited',
  'unmatched-face',
];

const reshapeInteger = (value: unknown, min: number, max: number): value is number =>
  reshapeNumber(value, min, max) && Number.isInteger(value);

const validReshapeReason = (value: unknown): boolean =>
  value === null || (typeof value === 'string' && RESHAPE_REASONS.includes(value));

const validReshapeMapBox = (value: unknown): boolean => {
  if (!reshapeRecord(value)) return false;
  const encoding = value.encoding;
  if (encoding !== 'rle-u8-v1' && encoding !== 'raw-u8-v1') return false;
  if (
    !reshapeInteger(value.width, 1, RESHAPE_MAX_MAP_SIDE) ||
    !reshapeInteger(value.height, 1, RESHAPE_MAX_MAP_SIDE) ||
    !reshapeHash(value.sha256) ||
    typeof value.data !== 'string'
  )
    return false;
  // RLE 展开量与 raw 像素和在解码阶段校验，这里只限制包装长度。
  const limit = encoding === 'raw-u8-v1' ? Math.ceil((4 * value.width * value.height) / 3) * 4 : RESHAPE_MAX_MAP_BASE64;
  return value.data.length <= limit;
};

const validReshapeSnapshotFace = (value: unknown): boolean => {
  if (!reshapeRecord(value) || !reshapeId(value.faceId) || !reshapeVector(value.bounds, 4)) return false;
  const [x, y, width, height] = value.bounds;
  if (width <= 0 || height <= 0 || x + width > 1 || y + height > 1) return false;
  // roi 直接存仿射矩阵的 6 个元素；不从角度／边长反推，重放位保持一致。
  if (!reshapeVector(value.roiMatrix, 6, -1e9, 1e9)) return false;
  if (
    !reshapeNumber(value.detectionConfidence, 0, 1) ||
    !reshapeNumber(value.landmarkConfidence, 0, 1) ||
    typeof value.refined !== 'boolean' ||
    typeof value.processable !== 'boolean' ||
    !validReshapeReason(value.reason)
  )
    return false;
  if (
    !Array.isArray(value.landmarks) ||
    value.landmarks.length !== 478 ||
    !value.landmarks.every((point) => reshapeVector(point, 3, -4, 4))
  )
    return false;
  if (value.localProtectionMap !== null && !validReshapeMapBox(value.localProtectionMap)) return false;
  const plan = value.plan;
  if (!reshapeRecord(plan)) return false;
  const fixed = plan.fixed;
  if (typeof fixed !== 'string' || fixed.length !== RESHAPE_MESH_VERTICES || !/^[01]+$/.test(fixed)) return false;
  const weights = plan.weights;
  if (!Array.isArray(weights) || weights.length > RESHAPE_MESH_VERTICES) return false;
  let previousWeight: number | null = null;
  for (const entry of weights) {
    if (!Array.isArray(entry) || entry.length !== 2) return false;
    const [index, weight] = entry;
    // 与 06 软约束契约一致：权重必须落在 (0, 10000]。
    if (
      !reshapeInteger(index, 0, RESHAPE_MESH_VERTICES - 1) ||
      typeof weight !== 'number' ||
      !Number.isFinite(weight) ||
      weight <= 0 ||
      weight > 10000
    )
      return false;
    if (previousWeight !== null && previousWeight >= index) return false;
    previousWeight = index;
  }
  const regions = plan.regions;
  if (!Array.isArray(regions) || regions.length > RESHAPE_MESH_TRIANGLES) return false;
  let previousTriangle: number | null = null;
  for (const region of regions) {
    if (!Array.isArray(region) || region.length !== 4) return false;
    const [triangle, strength, hard, reasons] = region;
    if (
      !reshapeInteger(triangle, 0, RESHAPE_MESH_TRIANGLES - 1) ||
      !reshapeNumber(strength, 0, 1) ||
      typeof hard !== 'boolean' ||
      !reshapeInteger(reasons, 0, RESHAPE_REASONS_MASK_MAX)
    )
      return false;
    if (previousTriangle !== null && previousTriangle >= triangle) return false;
    previousTriangle = triangle;
  }
  const mesh = value.mesh;
  if (!reshapeRecord(mesh)) return false;
  if (
    !Array.isArray(mesh.target) ||
    mesh.target.length !== RESHAPE_MESH_VERTICES ||
    !mesh.target.every((point) => reshapeVector(point, 2, -1e9, 1e9)) ||
    !reshapeVector(mesh.requested, 4, 0, 100) ||
    !reshapeVector(mesh.effective, 4, 0, 1)
  )
    return false;
  return (
    reshapeNumber(value.safetyScale, 0, 1) &&
    reshapeInteger(value.backoffs, 0, 16) &&
    reshapeInteger(value.detectedLines, 0, 32) &&
    reshapeInteger(value.reasons, 0, RESHAPE_REASONS_MASK_MAX)
  );
};

const validReshapeSceneSnapshot = (value: unknown): boolean => {
  if (!reshapeRecord(value)) return false;
  if (
    value.version !== RESHAPE_SNAPSHOT_VERSION ||
    !reshapeHash(value.sourceBaseHash) ||
    !reshapeHash(value.analysisKey) ||
    !reshapeHash(value.parameterHash) ||
    value.meshVersion !== RESHAPE_MESH_VERSION
  )
    return false;
  if (!reshapeRecord(value.modelHashes)) return false;
  const hashes = Object.entries(value.modelHashes);
  if (hashes.length === 0 || hashes.length > 8 || !hashes.every(([id, hash]) => reshapeId(id) && reshapeHash(hash)))
    return false;
  const dimensions = value.baseDimensions;
  if (!Array.isArray(dimensions) || dimensions.length !== 2) return false;
  if (!reshapeInteger(dimensions[0], 1, 32768) || !reshapeInteger(dimensions[1], 1, 32768)) return false;
  if (dimensions[0] * dimensions[1] > 64_000_000) return false;
  if (value.protectionMap !== null && !validReshapeMapBox(value.protectionMap)) return false;
  const faces = value.faces;
  if (!Array.isArray(faces) || faces.length > 32) return false;
  let previousId: string | null = null;
  for (const face of faces) {
    // 脸按 faceId 严格升序且份份有效；组索引随后校验为 faces 的一次划分。
    if (!reshapeRecord(face) || !reshapeId(face.faceId) || !validReshapeSnapshotFace(face)) return false;
    if (previousId !== null && previousId >= face.faceId) return false;
    previousId = face.faceId;
  }
  const groups = value.groups;
  if (!Array.isArray(groups) || groups.length > 32) return false;
  const seen = new Array<boolean>(faces.length).fill(false);
  for (const group of groups) {
    if (!Array.isArray(group) || group.length === 0 || group.length > faces.length) return false;
    let previous: number | null = null;
    for (const index of group) {
      if (!reshapeInteger(index, 0, faces.length - 1)) return false;
      if (seen[index] || (previous !== null && previous >= index)) return false;
      seen[index] = true;
      previous = index;
    }
  }
  return seen.every(Boolean);
};

const normalizeReshapeFace = (value: unknown): PortraitReshapeFace | null => {
  if (!reshapeRecord(value) || !reshapeId(value.faceId)) return null;
  const anchor = value.sourceAnchor;
  if (!reshapeRecord(anchor) || !reshapeHash(anchor.sourceBaseHash) || !reshapeVector(anchor.bounds, 4)) return null;
  const [x, y, width, height] = anchor.bounds;
  if (width <= 0 || height <= 0 || x + width > 1 || y + height > 1) return null;
  const protection = value.protection === undefined ? {} : value.protection;
  if (!reshapeRecord(protection)) return null;
  const strokes = protection.strokes === undefined ? [] : protection.strokes;
  if (!Array.isArray(strokes) || strokes.length > 32) return null;
  for (const stroke of strokes) {
    if (!reshapeRecord(stroke) || (stroke.mode !== 'protect' && stroke.mode !== 'erase')) return null;
    if (!reshapeNumber(stroke.radius, Number.MIN_VALUE, 1)) return null;
    if (
      !Array.isArray(stroke.points) ||
      stroke.points.length === 0 ||
      stroke.points.length > 256 ||
      !stroke.points.every((p) => reshapeVector(p, 2))
    )
      return null;
  }
  // 损坏的保护区让整份配置不可执行，不能丢掉保护后继续套用全局强度。
  if (value.overrides !== undefined && !reshapeRecord(value.overrides)) return null;
  const overrides: Partial<
    ReshapeStrengths & ExtendedReshapeStrengths & UvReshapeStrengths & { uvSlimFace: number }
  > = {};
  if (reshapeRecord(value.overrides)) {
    if (Object.hasOwn(value.overrides, 'bigEye')) {
      overrides.bigEye = reshapeAmount(value.overrides.bigEye);
    }
    if (Object.hasOwn(value.overrides, UV_SLIM_FACE_STORAGE_KEY)) {
      overrides.uvSlimFace = reshapeAmount(value.overrides[UV_SLIM_FACE_STORAGE_KEY]);
    }
  }
  return {
    ...value,
    faceId: value.faceId,
    sourceAnchor: { ...anchor, sourceBaseHash: anchor.sourceBaseHash, bounds: [x, y, width, height] },
    overrides,
    protection: {
      ...protection,
      hair: protection.hair !== false,
      background: protection.background !== false,
      strokes: structuredClone(strokes) as ReshapeProtectionStroke[],
    },
  };
};

export const readPortraitReshape = (input: unknown): PortraitReshapeReadResult => {
  if (input == null) return { status: 'legacy', value: null };
  if (!reshapeRecord(input)) return { status: 'invalid', value: null };
  const value = structuredClone(input);
  if (!Object.hasOwn(value, 'schemaVersion')) return { status: 'legacy', value };
  if (
    value.schemaVersion !== 1 ||
    // 3dmm-tddfa-v1／mesh-arap-v1 已下线，仅为读旧 sidecar 保留，读入后迁到 runtime。
    (value.engineVersion !== '3dmm-tddfa-v1' &&
      value.engineVersion !== 'mesh-arap-v1' &&
      value.engineVersion !== 'gpupixel-uv-v1' &&
      value.engineVersion !== 'gpupixel-runtime-v1')
  ) {
    return { status: 'unsupported', value };
  }
  // 网格快照已废弃：读入时剥离；引擎迁到 runtime bake。
  delete value.snapshot;
  const rawFaces = value.faces === undefined ? [] : value.faces;
  if (!Array.isArray(rawFaces) || rawFaces.length > 32) return { status: 'invalid', value };
  const faces: PortraitReshapeFace[] = [];
  const ids = new Set<string>();
  for (const rawFace of rawFaces) {
    const face = normalizeReshapeFace(rawFace);
    if (!face || ids.has(face.faceId)) return { status: 'invalid', value };
    ids.add(face.faceId);
    faces.push({ ...face, overrides: stripMeshFaceOverrides(face.overrides) });
  }
  const settings: PortraitReshapeV1 = {
    schemaVersion: 1,
    engineVersion: 'gpupixel-runtime-v1',
    enabled: value.enabled === true,
    faces,
  };
  return { status: 'supported', value: settings, settings };
};

export const normalizePortraitReshape = (value: unknown): PortraitReshapeData => readPortraitReshape(value).value;

// 11 修复通道：摘要未复用快照时移除 sidecar 内快照字段；
// 非 supported（缺失／未知版本／损坏）原样返回，保留语义由 readPortraitReshape 决定。
export const clearPortraitReshapeSnapshot = (data: PortraitReshapeData): PortraitReshapeData => {
  const read = readPortraitReshape(data);
  if (!read.settings) return data;
  // 读参已剥离 snapshot；若原对象仍挂着旧网格快照则写回无快照的 UV 配置。
  if (!reshapeRecord(data) || !Object.hasOwn(data, 'snapshot')) return data;
  return read.settings;
};

export const effectiveReshapeStrengths = (portrait: unknown, face?: PortraitReshapeFace): ReshapeStrengths => ({
  ...normalizeReshapeStrengths(portrait),
  ...Object.fromEntries(
    RESHAPE_KEYS.filter((key) => face?.overrides?.[key] !== undefined).map((key) => [
      key,
      face!.overrides[key]!,
    ]),
  ),
});

export const effectiveExtendedReshapeStrengths = (
  portrait: unknown,
  face?: PortraitReshapeFace,
): ExtendedReshapeStrengths => ({
  ...normalizeExtendedReshapeStrengths(portrait),
  ...Object.fromEntries(
    EXTENDED_RESHAPE_KEYS.filter((key) => face?.overrides?.[key] !== undefined).map((key) => [
      key,
      face!.overrides[key]!,
    ]),
  ),
});

export const effectiveUvReshapeStrengths = (
  portrait: unknown,
  face?: PortraitReshapeFace,
): UvReshapeStrengths => {
  const staged = normalizeUvReshapeStrengths(portrait);
  return {
    // idle 暂存写在 portrait.slimFace；有脸 override 后以 uvSlimFace 为准。
    slimFace: face?.overrides?.uvSlimFace ?? staged.slimFace,
    bigEye: face?.overrides?.bigEye ?? staged.bigEye,
  };
};

export const isPortraitEyesSectionVisible = (
  adjustments: Pick<Adjustments, 'sectionVisibility'>,
): boolean => adjustments.sectionVisibility?.portraitEyes !== false;

// processableFaceIds 必须来自当前照片、当前分析基底已匹配且通过质量门控的目标，不是 UI 选择。
// 此函数只表达请求有效，尚未接入渲染器，不能替代 hasPortraitEffect。
export const hasPortraitReshapeRequest = (
  adjustments: Pick<Adjustments, 'portrait' | 'portraitReshape' | 'sectionVisibility'>,
  processableFaceIds: readonly string[],
): boolean => {
  const { settings } = readPortraitReshape(adjustments.portraitReshape);
  if (!settings?.enabled) return false;
  const reshapeVisible = adjustments.sectionVisibility?.portraitReshape !== false;
  const eyesVisible = isPortraitEyesSectionVisible(adjustments);
  if (!reshapeVisible && !eyesVisible) return false;
  return processableFaceIds.some((id) => {
    const face = settings.faces.find((entry) => entry.faceId === id);
    const uvStrengths = effectiveUvReshapeStrengths(adjustments.portrait, face);
    const slimActive = reshapeVisible && uvStrengths.slimFace > 0;
    const bigEyeActive = eyesVisible && uvStrengths.bigEye > 0;
    return slimActive || bigEyeActive;
  });
};

// 跨照片传输仅白名单网格轮廓四键，不携带扩展五官、身份、位置、保护或快照。
export const copyPortraitReshapeDefaults = (portrait: unknown): ReshapeStrengths => normalizeReshapeStrengths(portrait);

/** 跨照片可选复制扩展五官默认值（与轮廓四键分离，调用方显式合并）。 */
export const copyPortraitExtendedReshapeDefaults = (portrait: unknown): ExtendedReshapeStrengths =>
  normalizeExtendedReshapeStrengths(portrait);

export const INITIAL_PORTRAIT_ADJUSTMENTS: Readonly<PortraitAdjustments> = {
  neutralGrayEvenness: 0,
  neutralGrayContour: 0,
  neutralGrayContourFeatures: 80,
  neutralGrayContourOutline: 80,
  frequencySeparationFace: 0,
  frequencySeparationFaceLow: 100,
  frequencySeparationFaceHigh: 15,
  frequencySeparationBody: 0,
  frequencySeparationBodyLow: 100,
  frequencySeparationBodyHigh: 15,
  frequencyHighBipolar: 2,
  hydratedSmoothingFace: 0,
  hydratedSmoothingBody: 0,
  evenSkinFace: 0,
  evenSkinBody: 0,
  whiteningFace: 0,
  whiteningBody: 0,
  beauty: 0,
  white: 0,
  aiBlemishRemoval: 0,
  blemishRemoval: 0,
  textureRetention: 0,
  faceShineRemoval: 0,
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
  eyeWhiteBrighten: 0,
  eyeRedness: 0,
  lipstick: 0,
  blusher: 0,
  bodySlimArms: 0,
  bodySlimWaist: 0,
  bodySlimLegs: 0,
};

export const PORTRAIT_LABELS = {
  neutralGrayEvenness: 'editor.portrait.controls.neutralGrayEvenness',
  neutralGrayContour: 'editor.portrait.controls.neutralGrayContour',
  neutralGrayContourFeatures: 'editor.portrait.controls.neutralGrayContourFeatures',
  neutralGrayContourOutline: 'editor.portrait.controls.neutralGrayContourOutline',
  frequencySeparationFace: 'editor.portrait.controls.frequencySeparationFace',
  frequencySeparationFaceLow: 'editor.portrait.controls.frequencySeparationFaceLow',
  frequencySeparationFaceHigh: 'editor.portrait.controls.frequencySeparationFaceHigh',
  frequencySeparationBody: 'editor.portrait.controls.frequencySeparationBody',
  frequencySeparationBodyLow: 'editor.portrait.controls.frequencySeparationBodyLow',
  frequencySeparationBodyHigh: 'editor.portrait.controls.frequencySeparationBodyHigh',
  frequencyHighBipolar: 'editor.portrait.controls.frequencySeparationFaceHigh',
  hydratedSmoothingFace: 'editor.portrait.controls.hydratedSmoothingFace',
  hydratedSmoothingBody: 'editor.portrait.controls.hydratedSmoothingBody',
  evenSkinFace: 'editor.portrait.controls.evenSkinFace',
  evenSkinBody: 'editor.portrait.controls.evenSkinBody',
  whiteningFace: 'editor.portrait.controls.whiteningFace',
  whiteningBody: 'editor.portrait.controls.whiteningBody',
  beauty: 'editor.portrait.controls.beauty',
  white: 'editor.portrait.controls.white',
  blemishRemoval: 'editor.portrait.controls.blemishRemoval',
  textureRetention: 'editor.portrait.controls.textureRetention',
  faceShineRemoval: 'editor.portrait.controls.faceShineRemoval',
  aiBlemishRemoval: 'editor.portrait.controls.aiBlemishRemoval',
  slimFace: 'editor.portrait.controls.slimFace',
  smallFace: 'editor.portrait.controls.smallFace',
  cheekbones: 'editor.portrait.controls.cheekbones',
  jawline: 'editor.portrait.controls.jawline',
  noseWidth: 'editor.portrait.controls.noseWidth',
  noseLength: 'editor.portrait.controls.noseLength',
  mouthSize: 'editor.portrait.controls.mouthSize',
  lipThickness: 'editor.portrait.controls.lipThickness',
  eyeAperture: 'editor.portrait.controls.eyeAperture',
  eyeDistance: 'editor.portrait.controls.eyeDistance',
  forehead: 'editor.portrait.controls.forehead',
  chinProjection: 'editor.portrait.controls.chinProjection',
  bigEye: 'editor.portrait.controls.bigEye',
  eyeWhiteBrighten: 'editor.portrait.controls.eyeWhiteBrighten',
  eyeRedness: 'editor.portrait.controls.eyeRedness',
  lipstick: 'editor.portrait.controls.lipstick',
  blusher: 'editor.portrait.controls.blusher',
  bodySlimArms: 'editor.portrait.controls.slimArms',
  bodySlimWaist: 'editor.portrait.controls.slimWaist',
  bodySlimLegs: 'editor.portrait.controls.slimLegs',
} as const satisfies Record<keyof PortraitAdjustments, string>;

export const PORTRAIT_SECTIONS: ReadonlyArray<{
  id: PortraitSection;
  title: `editor.portrait.sections.${PortraitSection}`;
  keys: ReadonlyArray<keyof PortraitAdjustments>;
}> = [
  {
    id: 'skin',
    title: 'editor.portrait.sections.skin',
    keys: [
      'neutralGrayEvenness',
      'neutralGrayContour',
      'neutralGrayContourFeatures',
      'neutralGrayContourOutline',
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
    ],
  },
  {
    id: 'blemishes',
    title: 'editor.portrait.sections.blemishes',
    keys: ['aiBlemishRemoval', 'faceShineRemoval'],
  },
  {
    id: 'makeup',
    title: 'editor.portrait.sections.makeup',
    keys: ['lipstick', 'blusher'],
  },
  {
    id: 'reshape',
    title: 'editor.portrait.sections.reshape',
    keys: ['slimFace'],
  },
  {
    id: 'eyes',
    title: 'editor.portrait.sections.eyes',
    keys: ['bigEye'],
  },
  {
    id: 'body',
    title: 'editor.portrait.sections.body',
    keys: ['bodySlimArms', 'bodySlimWaist', 'bodySlimLegs'],
  },
];

export const PORTRAIT_KEYS = Object.keys(INITIAL_PORTRAIT_ADJUSTMENTS) as Array<keyof PortraitAdjustments>;

export const portraitSectionKeys = (section: PortraitSection): ReadonlyArray<keyof PortraitAdjustments> =>
  PORTRAIT_SECTIONS.find((item) => item.id === section)?.keys ?? [];

// 拖调色时中性化修肤与 GPUPixel 妆容／瘦脸／大眼；重塑 enabled 由 deferredPortraitReshape 旁路。
export const deferredPortraitRetouch = (): PortraitAdjustments => ({ ...INITIAL_PORTRAIT_ADJUSTMENTS });

/** 预览推迟重塑：enabled=false → 后端 RenderIntent::Bypass，不改 sidecar。 */
export const deferredPortraitReshape = (data: PortraitReshapeData): PortraitReshapeV1 => {
  const settings = readPortraitReshape(data).settings;
  if (!settings) return createPortraitReshape();
  return { ...settings, enabled: false };
};

const FREQUENCY_LOW_KEYS = ['frequencySeparationFaceLow', 'frequencySeparationBodyLow'] as const;
const FREQUENCY_HIGH_KEYS = ['frequencySeparationFaceHigh', 'frequencySeparationBodyHigh'] as const;
const FREQUENCY_BAND_KEYS = [...FREQUENCY_LOW_KEYS, ...FREQUENCY_HIGH_KEYS] as const;

const clampPortraitAmount = (amount: number) => Math.max(0, Math.min(100, Math.round(amount)));
const clampFrequencyHigh = (amount: number) => Math.max(-100, Math.min(100, Math.round(amount)));
/** 各代高频轴 → v2：正值柔化、0 保纹理、负值增强。 */
const migrateFrequencyHigh = (amount: number, version: 0 | 1 | 2) => {
  if (version === 2) return clampFrequencyHigh(amount);
  if (version === 1) return clampFrequencyHigh(-Math.round(amount));
  return clampFrequencyHigh(100 - Math.round(amount));
};

export const normalizePortraitAdjustments = (value: unknown): PortraitAdjustments => {
  const source = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const record = source as Record<string, unknown>;
  const frequencyHighVersion: 0 | 1 | 2 =
    record.frequencyHighBipolar === 2 ? 2 : record.frequencyHighBipolar === 1 ? 1 : 0;
  const result = { ...INITIAL_PORTRAIT_ADJUSTMENTS };
  for (const key of PORTRAIT_KEYS) {
    if (key === 'frequencyHighBipolar') continue;
    const amount = record[key];
    if ((FREQUENCY_HIGH_KEYS as readonly string[]).includes(key)) {
      if (typeof amount === 'number' && Number.isFinite(amount)) {
        result[key as (typeof FREQUENCY_HIGH_KEYS)[number]] = migrateFrequencyHigh(
          amount,
          frequencyHighVersion,
        );
      } else {
        // 打开高低频时高频缺省轻度柔化。
        result[key as (typeof FREQUENCY_HIGH_KEYS)[number]] = 15;
      }
      continue;
    }
    if ((FREQUENCY_LOW_KEYS as readonly string[]).includes(key)) {
      if (typeof amount === 'number' && Number.isFinite(amount)) {
        result[key as (typeof FREQUENCY_LOW_KEYS)[number]] = clampPortraitAmount(amount);
      } else {
        // 旧 sidecar 缺省低频子带为满量。
        result[key as (typeof FREQUENCY_LOW_KEYS)[number]] = 100;
      }
      continue;
    }
    if (typeof amount === 'number' && Number.isFinite(amount)) {
      (result as unknown as Record<string, number>)[key] = clampPortraitAmount(amount);
    } else if (key === 'aiBlemishRemoval' && typeof amount === 'boolean') {
      // 旧照片中的布尔开关迁移为强度值。
      result.aiBlemishRemoval = amount ? 100 : 0;
    }
  }
  // 共享子带旧键 → 脸/身两对新区键（仅当对应新区键缺失时迁移）。
  const legacyLow = record.frequencySeparationLow;
  if (typeof legacyLow === 'number' && Number.isFinite(legacyLow)) {
    const amount = clampPortraitAmount(legacyLow);
    if (typeof record.frequencySeparationFaceLow !== 'number') result.frequencySeparationFaceLow = amount;
    if (typeof record.frequencySeparationBodyLow !== 'number') result.frequencySeparationBodyLow = amount;
  }
  const legacyHigh = record.frequencySeparationHigh;
  if (typeof legacyHigh === 'number' && Number.isFinite(legacyHigh)) {
    const amount = migrateFrequencyHigh(legacyHigh, frequencyHighVersion);
    if (typeof record.frequencySeparationFaceHigh !== 'number') result.frequencySeparationFaceHigh = amount;
    if (typeof record.frequencySeparationBodyHigh !== 'number') result.frequencySeparationBodyHigh = amount;
  }
  result.frequencyHighBipolar = 2;
  // 产品已下线普通祛斑／质感保留滑条；旧 sidecar 非零值强制归零，避免不可见效果。
  result.blemishRemoval = 0;
  result.textureRetention = 0;
  // 产品已下线 GPUPixel 经典磨皮／美白；旧 sidecar 非零值强制归零。
  result.beauty = 0;
  result.white = 0;
  // 提亮眼白／祛红血丝暂从 UI 下线；旧 sidecar 非零值强制归零，避免后台仍生效。
  result.eyeWhiteBrighten = 0;
  result.eyeRedness = 0;
  return result;
};

// 与后端人像基底字段保持一致；滑块强度、AI 美肤开关、分区显隐、裁剪和调色不触发重新识别。
export const PORTRAIT_BASE_KEYS = [
  'transformDistortion',
  'transformVertical',
  'transformHorizontal',
  'transformRotate',
  'transformAspect',
  'transformScale',
  'transformXOffset',
  'transformYOffset',
  'lensDistortionAmount',
  'lensVignetteAmount',
  'lensTcaAmount',
  'lensDistortionParams',
  'lensMaker',
  'lensModel',
  'lensDistortionEnabled',
  'lensTcaEnabled',
  'lensVignetteEnabled',
  'guidedPerspective',
  'orientationSteps',
  'rotation',
  'flipHorizontal',
  'flipVertical',
  'aiPatches',
  'lensBlurEnabled',
  'lensBlurAmount',
  'lensBlurDiffusion',
  'lensBlurShape',
  'lensBlurMinDepth',
  'lensBlurMaxDepth',
  'lensBlurMinFade',
  'lensBlurMaxFade',
  'lensBlurDepthMap',
] as const;

export const portraitSourceSignature = (adjustments: Adjustments): string =>
  JSON.stringify([
    ...PORTRAIT_BASE_KEYS.map((key) => adjustments[key]),
    adjustments.sectionVisibility?.effects ?? true,
  ]);

// 重塑只分析有效像素补丁；此签名用于调度，后端仍独立计算含源文件／模型的 SHA-256 缓存键。
export const portraitReshapeSourceSignature = (adjustments: Adjustments): string =>
  JSON.stringify([
    ...PORTRAIT_BASE_KEYS.map((key) =>
      key === 'aiPatches'
        ? adjustments.aiPatches?.filter((patch) => patch.visible !== false && patch.isLoading !== true)
        : adjustments[key],
    ),
    adjustments.sectionVisibility?.effects ?? true,
    adjustments.liquify ?? null,
  ]);

/** 人像分区小眼睛对应的 sectionVisibility 键。 */
export const PORTRAIT_VISIBILITY_KEYS = [
  'portraitSkin',
  'portraitBlemishes',
  'portraitMakeup',
  'portraitReshape',
  'portraitEyes',
  'portraitBody',
] as const;
export type PortraitVisibilityKey = (typeof PORTRAIT_VISIBILITY_KEYS)[number];

/** 分区小眼睛：false 为关组，缺省／true 为开组。 */
export const isPortraitSectionVisible = (
  adjustments: Pick<Adjustments, 'sectionVisibility'>,
  key: PortraitVisibilityKey,
): boolean => adjustments.sectionVisibility?.[key] !== false;

/** 重塑分组小眼睛：false 为关组（Bypass），缺省／true 为开组。 */
export const isPortraitReshapeSectionVisible = (
  adjustments: Pick<Adjustments, 'sectionVisibility'>,
): boolean => isPortraitSectionVisible(adjustments, 'portraitReshape');

/**
 * 预览 A/B 双帧签名：除指定 `sectionVisibility` 键外的完整 adjustments。
 * 仅该键小眼睛翻转时签名不变，可复用 on／off 已缓存 JPEG。
 */
export const portraitVisibilityPreviewSignature = (
  adjustments: Adjustments,
  key: PortraitVisibilityKey,
): string => {
  const sectionVisibility = {
    ...(adjustments.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility),
    [key]: true,
  };
  return JSON.stringify({ ...adjustments, sectionVisibility });
};

/** @deprecated 使用 portraitVisibilityPreviewSignature(adj, 'portraitReshape') */
export const reshapeVisibilityPreviewSignature = (adjustments: Adjustments): string =>
  portraitVisibilityPreviewSignature(adjustments, 'portraitReshape');

/** 是否仅指定人像分区可见性翻转（其它参数含其它 sectionVisibility 键均未变）。 */
export const onlyPortraitVisibilityChanged = (
  prev: Adjustments,
  next: Adjustments,
  key: PortraitVisibilityKey,
): boolean =>
  isPortraitSectionVisible(prev, key) !== isPortraitSectionVisible(next, key) &&
  portraitVisibilityPreviewSignature(prev, key) === portraitVisibilityPreviewSignature(next, key);

/** 若恰好一个人像可见性键翻转且签名匹配则返回该键，否则 null。 */
export const detectPortraitVisibilityFlip = (
  prev: Adjustments,
  next: Adjustments,
): PortraitVisibilityKey | null => {
  let flipped: PortraitVisibilityKey | null = null;
  for (const key of PORTRAIT_VISIBILITY_KEYS) {
    if (isPortraitSectionVisible(prev, key) === isPortraitSectionVisible(next, key)) continue;
    if (flipped !== null) return null;
    flipped = key;
  }
  if (flipped === null) return null;
  return onlyPortraitVisibilityChanged(prev, next, flipped) ? flipped : null;
};

/** 是否仅重塑分组可见性翻转（其它参数含其它 sectionVisibility 键均未变）。 */
export const onlyPortraitReshapeVisibilityChanged = (prev: Adjustments, next: Adjustments): boolean =>
  onlyPortraitVisibilityChanged(prev, next, 'portraitReshape');

/** 翻转指定人像分区小眼睛后的 adjustments（其它字段浅拷贝）。 */
export const flipPortraitSectionVisibility = (
  adjustments: Adjustments,
  key: PortraitVisibilityKey,
): Adjustments => ({
  ...adjustments,
  sectionVisibility: {
    ...(adjustments.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility),
    [key]: !isPortraitSectionVisible(adjustments, key),
  },
});

/**
 * 该分区是否有非零效果（忽略当前显隐）。用于决定是否值得预热对侧帧。
 * reshape/eyes 仅看配置层强度，不要求 processableFaceIds。
 */
export const portraitVisibilitySectionHasEffect = (
  adjustments: Adjustments,
  key: PortraitVisibilityKey,
): boolean => {
  const values = normalizePortraitAdjustments(adjustments.portrait);
  if (key === 'portraitSkin') {
    return (
      values.neutralGrayEvenness > 0 ||
      values.neutralGrayContour > 0 ||
      values.frequencySeparationFace > 0 ||
      values.frequencySeparationBody > 0 ||
      values.hydratedSmoothingFace > 0 ||
      values.hydratedSmoothingBody > 0 ||
      values.evenSkinFace > 0 ||
      values.evenSkinBody > 0 ||
      values.whiteningFace > 0 ||
      values.whiteningBody > 0
    );
  }
  if (key === 'portraitBlemishes') {
    return values.aiBlemishRemoval > 0 || values.faceShineRemoval > 0;
  }
  if (key === 'portraitMakeup') {
    return values.lipstick > 0 || values.blusher > 0;
  }
  if (key === 'portraitBody') {
    return values.bodySlimArms > 0 || values.bodySlimWaist > 0 || values.bodySlimLegs > 0;
  }
  const eyeColorActive = values.eyeWhiteBrighten > 0 || values.eyeRedness > 0;
  const { settings } = readPortraitReshape(adjustments.portraitReshape);
  if (!settings?.enabled) {
    if (key === 'portraitEyes') return values.bigEye > 0 || eyeColorActive;
    return values.slimFace > 0;
  }
  if (key === 'portraitEyes') {
    if (values.bigEye > 0 || eyeColorActive) return true;
    return settings.faces.some((face) => (face.overrides.bigEye ?? 0) > 0);
  }
  if (values.slimFace > 0) return true;
  return settings.faces.some((face) => (face.overrides.uvSlimFace ?? 0) > 0);
};

/** GPUPixel 口红／腮红（不含瘦脸／大眼，避免与重塑门控混淆；经典磨皮／美白已下线）。 */
export const hasPortraitGpuPixelBeautyEffect = (adjustments: Adjustments): boolean => {
  const values = normalizePortraitAdjustments(adjustments.portrait);
  const makeup = adjustments.sectionVisibility?.portraitMakeup !== false;
  return makeup && (values.lipstick > 0 || values.blusher > 0);
};

/** GPUPixel 瘦脸／大眼／口红／腮红任一非零且分区可见（与后端 `portrait_gpupixel::requested` 对齐）。 */
export const hasPortraitGpuPixelPreviewEffect = (adjustments: Adjustments): boolean => {
  const values = normalizePortraitAdjustments(adjustments.portrait);
  const reshape = adjustments.sectionVisibility?.portraitReshape !== false;
  const eyes = adjustments.sectionVisibility?.portraitEyes !== false;
  return (
    hasPortraitGpuPixelBeautyEffect(adjustments) ||
    (reshape && values.slimFace > 0) ||
    (eyes && values.bigEye > 0)
  );
};

/** 与后端 render_intent Prepare 对齐：GPUPixel／旧 UV。 */
export const hasPortraitReshapePreviewEffect = (adjustments: Adjustments): boolean => {
  if (hasPortraitGpuPixelPreviewEffect(adjustments)) return true;
  const { settings } = readPortraitReshape(adjustments.portraitReshape);
  if (!settings?.enabled) return false;
  const reshapeVisible = adjustments.sectionVisibility?.portraitReshape !== false;
  const eyesVisible = isPortraitEyesSectionVisible(adjustments);
  if (!reshapeVisible && !eyesVisible) return false;
  return (
    (reshapeVisible && portraitVisibilitySectionHasEffect(adjustments, 'portraitReshape')) ||
    (eyesVisible && portraitVisibilitySectionHasEffect(adjustments, 'portraitEyes'))
  );
};

export const hasPortraitEffect = (adjustments: Adjustments): boolean => {
  const values = normalizePortraitAdjustments(adjustments.portrait);
  return (
    hasPortraitGpuPixelBeautyEffect(adjustments) ||
    (adjustments.sectionVisibility?.portraitSkin !== false &&
      (values.neutralGrayEvenness > 0 ||
        values.neutralGrayContour > 0 ||
        values.frequencySeparationFace > 0 ||
        values.frequencySeparationBody > 0 ||
        values.hydratedSmoothingFace > 0 ||
        values.hydratedSmoothingBody > 0 ||
        values.evenSkinFace > 0 ||
        values.evenSkinBody > 0 ||
        values.whiteningFace > 0 ||
        values.whiteningBody > 0)) ||
    (adjustments.sectionVisibility?.portraitBlemishes !== false &&
      (values.aiBlemishRemoval > 0 || values.faceShineRemoval > 0)) ||
    (adjustments.sectionVisibility?.portraitEyes !== false &&
      (values.eyeWhiteBrighten > 0 || values.eyeRedness > 0)) ||
    (adjustments.sectionVisibility?.portraitBody !== false &&
      (values.bodySlimArms > 0 || values.bodySlimWaist > 0 || values.bodySlimLegs > 0))
  );
};

export interface Adjustments {
  [index: string]: any;
  portrait: PortraitAdjustments;
  portraitReshape: PortraitReshapeData;
  aiPatches: Array<AiPatch>;
  aspectRatio: number | null;
  blacks: number;
  brightness: number;
  centré: number;
  clarity: number;
  chromaticAberrationBlueYellow: number;
  chromaticAberrationRedCyan: number;
  colorCalibration: ColorCalibration;
  colorGrading: ColorGradingProps;
  colorNoiseReduction: number;
  contrast: number;
  curves: Curves;
  pointCurves?: Curves;
  parametricCurve?: ParametricCurve;
  curveMode?: 'point' | 'parametric';
  levels: LevelsSettings;
  crop: Crop | null;
  dehaze: number;
  exposure: number;
  flipHorizontal: boolean;
  flipVertical: boolean;
  flareAmount: number;
  /** 拍摄信息边框（sidecar 持久化；GPU 管线忽略） */
  frame: FrameState;
  glowAmount: number;
  grainAmount: number;
  grainRoughness: number;
  grainSize: number;
  guidedPerspective: GuidedPerspective;
  halationAmount: number;
  highlights: number;
  hsl: Hsl;
  hue: number;
  lensBlurAmount: number;
  lensBlurDiffusion: number;
  lensBlurShape: 'circle' | 'hexagon' | 'octagon' | 'ring';
  lensBlurDepthMap: string | null;
  lensBlurEnabled: boolean;
  lensBlurMaxDepth: number;
  lensBlurMaxFade: number;
  lensBlurMinDepth: number;
  lensBlurMinFade: number;
  lensCorrectionMode: 'auto' | 'manual';
  lensDistortionAmount: number;
  lensVignetteAmount: number;
  lensTcaAmount: number;
  lensDistortionEnabled: boolean;
  lensTcaEnabled: boolean;
  lensVignetteEnabled: boolean;
  lensDistortionParams: {
    k1: number;
    k2: number;
    k3: number;
    model: number;
    tca_vr: number;
    tca_vb: number;
    vig_k1: number;
    vig_k2: number;
    vig_k3: number;
  } | null;
  lensMaker: string | null;
  lensModel: string | null;
  lumaNoiseReduction: number;
  lutData?: string | null;
  lutIntensity?: number;
  lutName?: string | null;
  lutPath?: string | null;
  lutSize?: number;
  lutIsSceneReferred?: boolean;
  masks: Array<MaskContainer>;
  orientationSteps: number;
  rotation: number;
  saturation: number;
  sectionVisibility: SectionVisibility;
  shadows: number;
  sharpness: number;
  sharpnessThreshold: number;
  showClipping: boolean;
  structure: number;
  temperature: number;
  tint: number;
  toneMapper: 'agx' | 'basic';
  transformDistortion: number;
  transformVertical: number;
  transformHorizontal: number;
  transformRotate: number;
  transformAspect: number;
  transformScale: number;
  transformXOffset: number;
  transformYOffset: number;
  vibrance: number;
  vignetteAmount: number;
  vignetteFeather: number;
  vignetteMidpoint: number;
  vignetteRoundness: number;
  whiteBalance: WhiteBalance | null;
  whites: number;
}

export interface AiPatch {
  id: string;
  isLoading: boolean;
  invert: boolean;
  name: string;
  patchData: any | null;
  prompt: string;
  subMasks: Array<SubMask>;
  visible: boolean;
}

export interface Color {
  color: string;
  name: string;
}

interface ColorGradingProps {
  [index: string]: number | HueSatLum;
  balance: number;
  blending: number;
  global: HueSatLum;
  highlights: HueSatLum;
  midtones: HueSatLum;
  shadows: HueSatLum;
}

export interface Coord {
  x: number;
  y: number;
}

export interface Curves {
  [index: string]: Array<Coord>;
  blue: Array<Coord>;
  green: Array<Coord>;
  luma: Array<Coord>;
  red: Array<Coord>;
}

export interface HueSatLum {
  hue: number;
  saturation: number;
  luminance: number;
}

interface Hsl {
  [index: string]: HueSatLum;
  aquas: HueSatLum;
  blues: HueSatLum;
  greens: HueSatLum;
  magentas: HueSatLum;
  oranges: HueSatLum;
  purples: HueSatLum;
  reds: HueSatLum;
  yellows: HueSatLum;
}

export interface MaskAdjustments {
  [index: string]: any;
  blacks: number;
  brightness: number;
  clarity: number;
  colorGrading: ColorGradingProps;
  colorNoiseReduction: number;
  contrast: number;
  curves: Curves;
  pointCurves?: Curves;
  parametricCurve?: ParametricCurve;
  curveMode?: 'point' | 'parametric';
  levels: LevelsSettings;
  dehaze: number;
  exposure: number;
  flareAmount: number;
  glowAmount: number;
  halationAmount: number;
  highlights: number;
  hsl: Hsl;
  hue: number;
  id?: string;
  lumaNoiseReduction: number;
  saturation: number;
  sectionVisibility: SectionVisibility;
  shadows: number;
  sharpness: number;
  sharpnessThreshold: number;
  structure: number;
  temperature: number;
  tint: number;
  vibrance: number;
  whites: number;
}

export interface MaskContainer {
  adjustments: MaskAdjustments;
  id?: any;
  invert: boolean;
  name: string;
  opacity: number;
  subMasks: Array<SubMask>;
  visible: boolean;
}

export interface Sections {
  [index: string]: Array<string>;
  basic: Array<string>;
  curves: Array<string>;
  levels: Array<string>;
  color: Array<string>;
  colorGrading: Array<string>;
  colorMixer: Array<string>;
  colorCalibration: Array<string>;
  details: Array<string>;
  effects: Array<string>;
}

export interface SectionVisibility {
  [index: string]: boolean;
  basic: boolean;
  curves: boolean;
  levels: boolean;
  color: boolean;
  colorGrading: boolean;
  colorMixer: boolean;
  colorCalibration: boolean;
  details: boolean;
  effects: boolean;
  portraitSkin: boolean;
  portraitBlemishes: boolean;
  portraitMakeup: boolean;
  portraitReshape: boolean;
  portraitEyes: boolean;
  portraitBody: boolean;
}

export const COLOR_LABELS: Array<Color> = [
  { name: 'red', color: '#ef4444' },
  { name: 'yellow', color: '#facc15' },
  { name: 'green', color: '#4ade80' },
  { name: 'blue', color: '#60a5fa' },
  { name: 'purple', color: '#a78bfa' },
];

const INITIAL_COLOR_GRADING: ColorGradingProps = {
  balance: 0,
  blending: 50,
  global: { hue: 0, saturation: 0, luminance: 0 },
  highlights: { hue: 0, saturation: 0, luminance: 0 },
  midtones: { hue: 0, saturation: 0, luminance: 0 },
  shadows: { hue: 0, saturation: 0, luminance: 0 },
};

const INITIAL_COLOR_CALIBRATION: ColorCalibration = {
  shadowsTint: 0,
  redHue: 0,
  redSaturation: 0,
  greenHue: 0,
  greenSaturation: 0,
  blueHue: 0,
  blueSaturation: 0,
};

export const DEFAULT_PARAMETRIC_CURVE_SETTINGS: ParametricCurveSettings = {
  darks: 0,
  shadows: 0,
  highlights: 0,
  lights: 0,
  whiteLevel: 0,
  blackLevel: 0,
  split1: 25,
  split2: 50,
  split3: 75,
};

const getDefaultParametricCurve = (): ParametricCurve => ({
  luma: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS },
  red: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS },
  green: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS },
  blue: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS },
});

export const getDefaultCurves = (): Curves => ({
  blue: [
    { x: 0, y: 0 },
    { x: 255, y: 255 },
  ],
  green: [
    { x: 0, y: 0 },
    { x: 255, y: 255 },
  ],
  luma: [
    { x: 0, y: 0 },
    { x: 255, y: 255 },
  ],
  red: [
    { x: 0, y: 0 },
    { x: 255, y: 255 },
  ],
});

export const DEFAULT_PARAMETRIC_CURVE = getDefaultParametricCurve();

export const INITIAL_MASK_ADJUSTMENTS: MaskAdjustments = {
  blacks: 0,
  brightness: 0,
  clarity: 0,
  colorGrading: { ...INITIAL_COLOR_GRADING },
  colorNoiseReduction: 0,
  contrast: 0,
  curves: getDefaultCurves(),
  pointCurves: getDefaultCurves(),
  parametricCurve: getDefaultParametricCurve(),
  curveMode: 'point',
  levels: getDefaultLevels(),
  dehaze: 0,
  exposure: 0,
  flareAmount: 0,
  glowAmount: 0,
  halationAmount: 0,
  highlights: 0,
  hsl: {
    aquas: { hue: 0, saturation: 0, luminance: 0 },
    blues: { hue: 0, saturation: 0, luminance: 0 },
    greens: { hue: 0, saturation: 0, luminance: 0 },
    magentas: { hue: 0, saturation: 0, luminance: 0 },
    oranges: { hue: 0, saturation: 0, luminance: 0 },
    purples: { hue: 0, saturation: 0, luminance: 0 },
    reds: { hue: 0, saturation: 0, luminance: 0 },
    yellows: { hue: 0, saturation: 0, luminance: 0 },
  },
  hue: 0,
  lumaNoiseReduction: 0,
  saturation: 0,
  sectionVisibility: {
    basic: true,
    levels: true,
    curves: true,
    color: true,
    colorGrading: true,
    colorMixer: true,
    colorCalibration: true,
    details: true,
    effects: true,
    portraitSkin: true,
    portraitBlemishes: true,
    portraitMakeup: true,
    portraitReshape: true,
    portraitEyes: true,
    portraitBody: true,
  },
  shadows: 0,
  sharpness: 0,
  sharpnessThreshold: 15,
  structure: 0,
  temperature: 0,
  tint: 0,
  vibrance: 0,
  whites: 0,
};

export const INITIAL_MASK_CONTAINER: MaskContainer = {
  adjustments: INITIAL_MASK_ADJUSTMENTS,
  invert: false,
  name: 'New Mask',
  opacity: 100,
  subMasks: [],
  visible: true,
};

export const INITIAL_ADJUSTMENTS: Adjustments = {
  portrait: { ...INITIAL_PORTRAIT_ADJUSTMENTS },
  portraitReshape: createPortraitReshape(),
  aiPatches: [],
  aspectRatio: null,
  blacks: 0,
  brightness: 0,
  centré: 0,
  clarity: 0,
  chromaticAberrationBlueYellow: 0,
  chromaticAberrationRedCyan: 0,
  colorCalibration: { ...INITIAL_COLOR_CALIBRATION },
  colorGrading: { ...INITIAL_COLOR_GRADING },
  colorNoiseReduction: 0,
  contrast: 0,
  crop: null,
  curves: getDefaultCurves(),
  pointCurves: getDefaultCurves(),
  parametricCurve: getDefaultParametricCurve(),
  curveMode: 'point',
  levels: getDefaultLevels(),
  dehaze: 0,
  exposure: 0,
  flipHorizontal: false,
  flipVertical: false,
  flareAmount: 0,
  frame: defaultFrame(),
  glowAmount: 0,
  grainAmount: 0,
  grainRoughness: 50,
  grainSize: 25,
  guidedPerspective: { enabled: false, lines: [], autoCrop: true },
  halationAmount: 0,
  highlights: 0,
  hsl: {
    aquas: { hue: 0, saturation: 0, luminance: 0 },
    blues: { hue: 0, saturation: 0, luminance: 0 },
    greens: { hue: 0, saturation: 0, luminance: 0 },
    magentas: { hue: 0, saturation: 0, luminance: 0 },
    oranges: { hue: 0, saturation: 0, luminance: 0 },
    purples: { hue: 0, saturation: 0, luminance: 0 },
    reds: { hue: 0, saturation: 0, luminance: 0 },
    yellows: { hue: 0, saturation: 0, luminance: 0 },
  },
  hue: 0,
  lensBlurAmount: 40,
  lensBlurDiffusion: 0,
  lensBlurShape: 'circle',
  lensBlurDepthMap: null,
  lensBlurEnabled: false,
  lensBlurMaxDepth: 100,
  lensBlurMaxFade: 20,
  lensBlurMinDepth: 20,
  lensBlurMinFade: 20,
  lensCorrectionMode: 'manual',
  lensDistortionAmount: 100,
  lensVignetteAmount: 100,
  lensTcaAmount: 100,
  lensDistortionEnabled: true,
  lensTcaEnabled: true,
  lensVignetteEnabled: true,
  lensDistortionParams: null,
  lensMaker: null,
  lensModel: null,
  lumaNoiseReduction: 0,
  lutData: null,
  lutIntensity: 100,
  lutName: null,
  lutPath: null,
  lutSize: 0,
  lutIsSceneReferred: false,
  masks: [],
  orientationSteps: 0,
  rotation: 0,
  saturation: 0,
  sectionVisibility: {
    basic: true,
    levels: true,
    curves: true,
    color: true,
    colorGrading: true,
    colorMixer: true,
    colorCalibration: true,
    details: true,
    effects: true,
    portraitSkin: true,
    portraitBlemishes: true,
    portraitMakeup: true,
    portraitReshape: true,
    portraitEyes: true,
    portraitBody: true,
  },
  shadows: 0,
  sharpness: 0,
  sharpnessThreshold: 15,
  showClipping: false,
  structure: 0,
  temperature: 0,
  tint: 0,
  toneMapper: 'basic',
  transformDistortion: 0,
  transformVertical: 0,
  transformHorizontal: 0,
  transformRotate: 0,
  transformAspect: 0,
  transformScale: 100,
  transformXOffset: 0,
  transformYOffset: 0,
  vibrance: 0,
  vignetteAmount: 0,
  vignetteFeather: 50,
  vignetteMidpoint: 50,
  vignetteRoundness: 0,
  whiteBalance: null,
  whites: 0,
};

const deepCloneCurves = (curves: any): Curves => ({
  blue: curves?.blue?.map((p: Coord) => ({ ...p })) || [
    { x: 0, y: 0 },
    { x: 255, y: 255 },
  ],
  green: curves?.green?.map((p: Coord) => ({ ...p })) || [
    { x: 0, y: 0 },
    { x: 255, y: 255 },
  ],
  luma: curves?.luma?.map((p: Coord) => ({ ...p })) || [
    { x: 0, y: 0 },
    { x: 255, y: 255 },
  ],
  red: curves?.red?.map((p: Coord) => ({ ...p })) || [
    { x: 0, y: 0 },
    { x: 255, y: 255 },
  ],
});

const deepCloneParametric = (pCurve: any): ParametricCurve => ({
  luma: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS, ...(pCurve?.luma || {}) },
  red: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS, ...(pCurve?.red || {}) },
  green: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS, ...(pCurve?.green || {}) },
  blue: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS, ...(pCurve?.blue || {}) },
});

export const normalizeLoadedAdjustments = (loadedAdjustments: Adjustments): any => {
  if (!loadedAdjustments) {
    return INITIAL_ADJUSTMENTS;
  }

  const normalizeSubMasks = (subMasks: any[]) => {
    return (subMasks || []).map((subMask: Partial<SubMask>) => ({
      visible: true,
      mode: SubMaskMode.Additive,
      invert: false,
      opacity: 100,
      ...subMask,
    }));
  };

  const normalizedMasks = (loadedAdjustments.masks || []).map((maskContainer: MaskContainer) => {
    const containerAdjustments = maskContainer.adjustments || {};
    const normalizedSubMasks = normalizeSubMasks(maskContainer.subMasks);

    return {
      ...INITIAL_MASK_CONTAINER,
      id: maskContainer.id || uuidv4(),
      ...maskContainer,
      adjustments: {
        ...INITIAL_MASK_ADJUSTMENTS,
        ...containerAdjustments,
        flareAmount: containerAdjustments.flareAmount ?? INITIAL_MASK_ADJUSTMENTS.flareAmount,
        glowAmount: containerAdjustments.glowAmount ?? INITIAL_MASK_ADJUSTMENTS.glowAmount,
        halationAmount: containerAdjustments.halationAmount ?? INITIAL_MASK_ADJUSTMENTS.halationAmount,
        hue: containerAdjustments.hue ?? INITIAL_MASK_ADJUSTMENTS.hue,
        colorGrading: { ...INITIAL_MASK_ADJUSTMENTS.colorGrading, ...(containerAdjustments.colorGrading || {}) },
        hsl: { ...INITIAL_MASK_ADJUSTMENTS.hsl, ...(containerAdjustments.hsl || {}) },
        curves: containerAdjustments.curves ? deepCloneCurves(containerAdjustments.curves) : getDefaultCurves(),
        pointCurves: containerAdjustments.pointCurves
          ? deepCloneCurves(containerAdjustments.pointCurves)
          : getDefaultCurves(),
        parametricCurve: containerAdjustments.parametricCurve
          ? deepCloneParametric(containerAdjustments.parametricCurve)
          : getDefaultParametricCurve(),
        curveMode: containerAdjustments.curveMode || INITIAL_MASK_ADJUSTMENTS.curveMode,
        levels: normalizeLevels(containerAdjustments.levels),
        sectionVisibility: {
          ...INITIAL_MASK_ADJUSTMENTS.sectionVisibility,
          ...(containerAdjustments.sectionVisibility || {}),
        },
        sharpnessThreshold: containerAdjustments.sharpnessThreshold ?? INITIAL_MASK_ADJUSTMENTS.sharpnessThreshold,
      },
      subMasks: normalizedSubMasks,
    };
  });

  const normalizedAiPatches = (loadedAdjustments.aiPatches || []).map((patch: any) => ({
    visible: true,
    ...patch,
    subMasks: normalizeSubMasks(patch.subMasks),
  }));

  return {
    ...INITIAL_ADJUSTMENTS,
    ...loadedAdjustments,
    portrait: normalizePortraitAdjustments(loadedAdjustments.portrait),
    portraitReshape: normalizePortraitReshape(loadedAdjustments.portraitReshape),
    guidedPerspective: {
      enabled: loadedAdjustments.guidedPerspective?.enabled ?? false,
      lines: (() => {
        const clamp01 = (n: number) => Math.min(1, Math.max(0, n));
        const raw = (loadedAdjustments.guidedPerspective?.lines || []).map((l: any) => ({
          id: l.id || uuidv4(),
          type: (l.type === 'horizontal' ? 'horizontal' : 'vertical') as 'vertical' | 'horizontal',
          p1: { x: clamp01(l.p1?.x ?? 0), y: clamp01(l.p1?.y ?? 0) },
          p2: { x: clamp01(l.p2?.x ?? 1), y: clamp01(l.p2?.y ?? 1) },
        }));
        const verts = raw.filter((l) => l.type === 'vertical').slice(0, 2);
        const hors = raw.filter((l) => l.type === 'horizontal').slice(0, 2);
        return [...verts, ...hors];
      })(),
      autoCrop: loadedAdjustments.guidedPerspective?.autoCrop ?? true,
    },
    lutIsSceneReferred: loadedAdjustments.lutIsSceneReferred ?? false,
    flareAmount: loadedAdjustments.flareAmount ?? INITIAL_ADJUSTMENTS.flareAmount,
    glowAmount: loadedAdjustments.glowAmount ?? INITIAL_ADJUSTMENTS.glowAmount,
    halationAmount: loadedAdjustments.halationAmount ?? INITIAL_ADJUSTMENTS.halationAmount,
    lensBlurAmount: loadedAdjustments.lensBlurAmount ?? INITIAL_ADJUSTMENTS.lensBlurAmount,
    lensBlurDiffusion: loadedAdjustments.lensBlurDiffusion ?? INITIAL_ADJUSTMENTS.lensBlurDiffusion,
    lensBlurShape: loadedAdjustments.lensBlurShape ?? INITIAL_ADJUSTMENTS.lensBlurShape,
    lensBlurDepthMap: loadedAdjustments.lensBlurDepthMap ?? INITIAL_ADJUSTMENTS.lensBlurDepthMap,
    lensBlurEnabled: loadedAdjustments.lensBlurEnabled ?? INITIAL_ADJUSTMENTS.lensBlurEnabled,
    lensBlurMaxDepth: loadedAdjustments.lensBlurMaxDepth ?? INITIAL_ADJUSTMENTS.lensBlurMaxDepth,
    lensBlurMaxFade: loadedAdjustments.lensBlurMaxFade ?? INITIAL_ADJUSTMENTS.lensBlurMaxFade,
    lensBlurMinDepth: loadedAdjustments.lensBlurMinDepth ?? INITIAL_ADJUSTMENTS.lensBlurMinDepth,
    lensBlurMinFade: loadedAdjustments.lensBlurMinFade ?? INITIAL_ADJUSTMENTS.lensBlurMinFade,
    lensCorrectionMode: loadedAdjustments.lensCorrectionMode || 'manual',
    lensMaker: loadedAdjustments.lensMaker ?? INITIAL_ADJUSTMENTS.lensMaker,
    lensModel: loadedAdjustments.lensModel ?? INITIAL_ADJUSTMENTS.lensModel,
    lensDistortionAmount: loadedAdjustments.lensDistortionAmount ?? INITIAL_ADJUSTMENTS.lensDistortionAmount,
    lensVignetteAmount: loadedAdjustments.lensVignetteAmount ?? INITIAL_ADJUSTMENTS.lensVignetteAmount,
    lensTcaAmount: loadedAdjustments.lensTcaAmount ?? INITIAL_ADJUSTMENTS.lensTcaAmount,
    lensDistortionEnabled: loadedAdjustments.lensDistortionEnabled ?? INITIAL_ADJUSTMENTS.lensDistortionEnabled,
    lensTcaEnabled: loadedAdjustments.lensTcaEnabled ?? INITIAL_ADJUSTMENTS.lensTcaEnabled,
    lensVignetteEnabled: loadedAdjustments.lensVignetteEnabled ?? INITIAL_ADJUSTMENTS.lensVignetteEnabled,
    lensDistortionParams: loadedAdjustments.lensDistortionParams ?? INITIAL_ADJUSTMENTS.lensDistortionParams,
    transformDistortion: loadedAdjustments.transformDistortion ?? INITIAL_ADJUSTMENTS.transformDistortion,
    transformVertical: loadedAdjustments.transformVertical ?? INITIAL_ADJUSTMENTS.transformVertical,
    transformHorizontal: loadedAdjustments.transformHorizontal ?? INITIAL_ADJUSTMENTS.transformHorizontal,
    transformRotate: loadedAdjustments.transformRotate ?? INITIAL_ADJUSTMENTS.transformRotate,
    transformAspect: loadedAdjustments.transformAspect ?? INITIAL_ADJUSTMENTS.transformAspect,
    transformScale: loadedAdjustments.transformScale ?? INITIAL_ADJUSTMENTS.transformScale,
    transformXOffset: loadedAdjustments.transformXOffset ?? INITIAL_ADJUSTMENTS.transformXOffset,
    transformYOffset: loadedAdjustments.transformYOffset ?? INITIAL_ADJUSTMENTS.transformYOffset,
    colorCalibration: { ...INITIAL_ADJUSTMENTS.colorCalibration, ...(loadedAdjustments.colorCalibration || {}) },
    colorGrading: { ...INITIAL_ADJUSTMENTS.colorGrading, ...(loadedAdjustments.colorGrading || {}) },
    hsl: { ...INITIAL_ADJUSTMENTS.hsl, ...(loadedAdjustments.hsl || {}) },
    curves: loadedAdjustments.curves ? deepCloneCurves(loadedAdjustments.curves) : getDefaultCurves(),
    pointCurves: loadedAdjustments.pointCurves ? deepCloneCurves(loadedAdjustments.pointCurves) : getDefaultCurves(),
    parametricCurve: loadedAdjustments.parametricCurve
      ? deepCloneParametric(loadedAdjustments.parametricCurve)
      : getDefaultParametricCurve(),
    curveMode: loadedAdjustments.curveMode || INITIAL_ADJUSTMENTS.curveMode,
    levels: normalizeLevels(loadedAdjustments.levels),
    masks: normalizedMasks,
    aiPatches: normalizedAiPatches,
    sectionVisibility: {
      ...INITIAL_ADJUSTMENTS.sectionVisibility,
      ...(loadedAdjustments.sectionVisibility || {}),
    },
    sharpnessThreshold: loadedAdjustments.sharpnessThreshold ?? INITIAL_ADJUSTMENTS.sharpnessThreshold,
    frame: normalizeFrame(loadedAdjustments.frame),
  };
};

export interface AdjustmentGroup {
  label: string;
  keys: string[];
}

export const ADJUSTMENT_GROUPS: Record<string, AdjustmentGroup[]> = {
  basic: [
    {
      label: 'modals.copyPaste.groups.exposureToneMapper',
      keys: [BasicAdjustment.Exposure, 'toneMapper'],
    },
    {
      label: 'modals.copyPaste.groups.tone',
      keys: [
        BasicAdjustment.Brightness,
        BasicAdjustment.Contrast,
        BasicAdjustment.Highlights,
        BasicAdjustment.Shadows,
        BasicAdjustment.Whites,
        BasicAdjustment.Blacks,
      ],
    },
    {
      label: 'modals.copyPaste.groups.curves',
      keys: ['curves', 'pointCurves', 'parametricCurve', 'curveMode'],
    },
  ],
  levels: [{ label: 'modals.copyPaste.groups.levels', keys: ['levels'] }],
  color: [
    {
      label: 'modals.copyPaste.groups.whiteBalance',
      keys: [ColorAdjustment.Temperature, ColorAdjustment.Tint, ColorAdjustment.WhiteBalance],
    },
    { label: 'modals.copyPaste.groups.presence', keys: [ColorAdjustment.Saturation, ColorAdjustment.Vibrance] },
    {
      label: 'modals.copyPaste.groups.hueShift',
      keys: [ColorAdjustment.Hue],
    },
  ],
  colorGrading: [{ label: 'modals.copyPaste.groups.colorGrading', keys: [ColorAdjustment.ColorGrading] }],
  colorMixer: [{ label: 'modals.copyPaste.groups.colorMixer', keys: [ColorAdjustment.Hsl] }],
  colorCalibration: [{ label: 'modals.copyPaste.groups.colorCalibration', keys: ['colorCalibration'] }],
  details: [
    {
      label: 'modals.copyPaste.groups.clarityDehaze',
      keys: [
        DetailsAdjustment.Clarity,
        DetailsAdjustment.Structure,
        DetailsAdjustment.Dehaze,
        DetailsAdjustment.Centré,
      ],
    },
    {
      label: 'modals.copyPaste.groups.sharpness',
      keys: [DetailsAdjustment.Sharpness, DetailsAdjustment.SharpnessThreshold],
    },
    {
      label: 'modals.copyPaste.groups.noiseReduction',
      keys: [DetailsAdjustment.LumaNoiseReduction, DetailsAdjustment.ColorNoiseReduction],
    },
    {
      label: 'modals.copyPaste.groups.chromaticAberration',
      keys: [DetailsAdjustment.ChromaticAberrationRedCyan, DetailsAdjustment.ChromaticAberrationBlueYellow],
    },
  ],
  effects: [
    {
      label: 'modals.copyPaste.groups.vignette',
      keys: [Effect.VignetteAmount, Effect.VignetteFeather, Effect.VignetteMidpoint, Effect.VignetteRoundness],
    },
    { label: 'modals.copyPaste.groups.grain', keys: [Effect.GrainAmount, Effect.GrainRoughness, Effect.GrainSize] },
    {
      label: 'modals.copyPaste.groups.halationGlow',
      keys: [CreativeAdjustment.GlowAmount, CreativeAdjustment.HalationAmount, CreativeAdjustment.FlareAmount],
    },
    {
      label: 'modals.copyPaste.groups.lut',
      keys: [
        Effect.LutIntensity,
        Effect.LutName,
        Effect.LutPath,
        Effect.LutSize,
        Effect.LutData,
        Effect.LutIsSceneReferred,
      ],
    },
  ],
  geometry: [
    { label: 'modals.copyPaste.groups.cropAspectRatio', keys: ['crop', 'aspectRatio'] },
    {
      label: 'modals.copyPaste.groups.transformRotation',
      keys: [
        'rotation',
        'flipHorizontal',
        'flipVertical',
        'orientationSteps',
        TransformAdjustment.TransformDistortion,
        TransformAdjustment.TransformVertical,
        TransformAdjustment.TransformHorizontal,
        TransformAdjustment.TransformRotate,
        TransformAdjustment.TransformAspect,
        TransformAdjustment.TransformScale,
        TransformAdjustment.TransformXOffset,
        TransformAdjustment.TransformYOffset,
      ],
    },
    {
      label: 'modals.copyPaste.groups.lensCorrection',
      keys: [
        LensAdjustment.LensCorrectionMode,
        LensAdjustment.LensMaker,
        LensAdjustment.LensModel,
        LensAdjustment.LensDistortionAmount,
        LensAdjustment.LensVignetteAmount,
        LensAdjustment.LensTcaAmount,
        LensAdjustment.LensDistortionEnabled,
        LensAdjustment.LensTcaEnabled,
        LensAdjustment.LensVignetteEnabled,
      ],
    },
    {
      label: 'modals.copyPaste.groups.guidedPerspective',
      keys: ['guidedPerspective'],
    },
  ],
  masks: [{ label: 'modals.copyPaste.groups.masks', keys: ['masks'] }],
};

export const COPYABLE_ADJUSTMENT_KEYS: string[] = Object.values(ADJUSTMENT_GROUPS)
  .flat()
  .flatMap((group) => group.keys);

export const ADJUSTMENT_SECTIONS: Sections = {
  basic: [
    BasicAdjustment.Brightness,
    BasicAdjustment.Contrast,
    BasicAdjustment.Highlights,
    BasicAdjustment.Shadows,
    BasicAdjustment.Whites,
    BasicAdjustment.Blacks,
    BasicAdjustment.Exposure,
    'toneMapper',
  ],
  curves: ['curves', 'pointCurves', 'parametricCurve', 'curveMode'],
  levels: ['levels'],
  color: [
    ColorAdjustment.Saturation,
    ColorAdjustment.Temperature,
    ColorAdjustment.Tint,
    ColorAdjustment.WhiteBalance,
    ColorAdjustment.Vibrance,
    ColorAdjustment.Hue,
  ],
  colorGrading: [ColorAdjustment.ColorGrading],
  colorMixer: [ColorAdjustment.Hsl],
  colorCalibration: ['colorCalibration'],
  details: [
    DetailsAdjustment.Clarity,
    DetailsAdjustment.Dehaze,
    DetailsAdjustment.Structure,
    DetailsAdjustment.Centré,
    DetailsAdjustment.Sharpness,
    DetailsAdjustment.SharpnessThreshold,
    DetailsAdjustment.LumaNoiseReduction,
    DetailsAdjustment.ColorNoiseReduction,
    DetailsAdjustment.ChromaticAberrationRedCyan,
    DetailsAdjustment.ChromaticAberrationBlueYellow,
  ],
  effects: [
    CreativeAdjustment.GlowAmount,
    CreativeAdjustment.HalationAmount,
    CreativeAdjustment.FlareAmount,
    Effect.GrainAmount,
    Effect.GrainRoughness,
    Effect.GrainSize,
    Effect.LutIntensity,
    Effect.LutName,
    Effect.LutPath,
    Effect.LutSize,
    Effect.LutIsSceneReferred,
    Effect.VignetteAmount,
    Effect.VignetteFeather,
    Effect.VignetteMidpoint,
    Effect.VignetteRoundness,
    Effect.LensBlurEnabled,
    Effect.LensBlurAmount,
    Effect.lensBlurDiffusion,
    Effect.LensBlurShape,
    Effect.LensBlurDepthMap,
    Effect.LensBlurMinDepth,
    Effect.LensBlurMaxDepth,
    Effect.LensBlurMinFade,
    Effect.LensBlurMaxFade,
  ],
};

const reconcileOrder = (defaultOrder: string[], order: string[] = []): string[] => {
  const savedOrder = order.filter((id, index) => defaultOrder.includes(id) && order.indexOf(id) === index);
  return [...savedOrder, ...defaultOrder.filter((id) => !savedOrder.includes(id))];
};

export const getAdjustmentSectionOrder = (order?: string[]): string[] =>
  reconcileOrder(Object.keys(ADJUSTMENT_SECTIONS), order);

export const getVisibleAdjustmentSections = (layout?: AdjustmentLayout): string[] =>
  getAdjustmentSectionOrder(layout?.sectionOrder).filter((section) => !layout?.hiddenSections?.includes(section));

export const DEFAULT_HIDDEN_ADJUSTMENT_TOOLS: string[] = [];

export const getHiddenAdjustmentTools = (layout?: AdjustmentLayout): string[] =>
  layout?.hiddenTools ?? DEFAULT_HIDDEN_ADJUSTMENT_TOOLS;

export const withAdjustmentLayout = (settings: AppSettings, changes: Partial<AdjustmentLayout>): AppSettings => ({
  ...settings,
  adjustmentLayout: { ...settings.adjustmentLayout, ...changes },
});

export interface AdjustmentSectionTool {
  id: string;
  label: string;
}

export const ADJUSTMENT_SECTION_TOOLS: Record<string, Array<AdjustmentSectionTool>> = {
  // colorGrading / colorMixer / colorCalibration are top-level sections in XiuLeMe
  color: [
    { id: 'whiteBalance', label: 'adjustments.color.whiteBalance' },
    { id: 'colorPresence', label: 'adjustments.color.presence' },
    { id: 'hue', label: 'adjustments.color.hue' },
  ],
  details: [
    { id: 'sharpening', label: 'adjustments.details.sharpening' },
    { id: 'presence', label: 'adjustments.details.presence' },
    { id: 'noiseReduction', label: 'adjustments.details.noiseReduction' },
    { id: 'chromaticAberration', label: 'adjustments.details.chromaticAberration' },
  ],
  effects: [
    { id: 'creative', label: 'adjustments.effects.creative' },
    { id: 'lensBlur', label: 'adjustments.effects.lensBlur' },
    { id: 'lut', label: 'adjustments.effects.lut' },
    { id: 'vignette', label: 'adjustments.effects.vignette' },
    { id: 'grain', label: 'adjustments.effects.grain' },
  ],
};

export const getAdjustmentSectionToolIds = (section: string): string[] =>
  (ADJUSTMENT_SECTION_TOOLS[section] ?? []).map((tool) => tool.id);

export const getAdjustmentToolOrder = (section: string, toolOrder?: Record<string, string[]>): string[] =>
  reconcileOrder(getAdjustmentSectionToolIds(section), toolOrder?.[section]);


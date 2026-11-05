import { Progress } from './AppProperties';

export enum FileFormats {
  Jpeg = 'jpeg',
  Png = 'png',
  Tiff = 'tiff',
  Webp = 'webp',
  Jxl = 'jxl',
  Avif = 'avif',
  Cube = 'cube',
}

export const FILE_FORMATS: Array<FileFormat> = [
  { id: FileFormats.Jpeg, name: 'JPEG', extensions: ['jpg', 'jpeg'] },
  { id: FileFormats.Png, name: 'PNG', extensions: ['png'] },
  { id: FileFormats.Tiff, name: 'TIFF', extensions: ['tiff'] },
  { id: FileFormats.Webp, name: 'WebP', extensions: ['webp'] },
  { id: FileFormats.Jxl, name: 'JPEG XL', extensions: ['jxl'] },
  { id: FileFormats.Avif, name: 'AVIF', extensions: ['avif'] },
  { id: FileFormats.Cube, name: 'CUBE LUT', extensions: ['cube'] },
];

export const FILENAME_VARIABLES: Array<string> = [
  '{original_filename}',
  '{sequence}',
  '{YYYY}',
  '{MM}',
  '{DD}',
  '{hh}',
  '{mm}',
];

export type TiffBitDepth = 8 | 16;

export interface ExportSettings {
  filenameTemplate: string | null;
  jpegQuality: number;
  tiffBitDepth: TiffBitDepth;
  keepMetadata: boolean;
  preserveTimestamps: boolean;
  resize: {
    mode: string;
    value: number;
    dontEnlarge: boolean;
  } | null;
  border: {
    basis: BorderBasis;
    horizontalPercent: number;
    verticalPercent: number;
    color: string;
  } | null;
  pad: {
    ratioWidth: number;
    ratioHeight: number;
    color: string;
  } | null;
  stripGps: boolean;
  watermark: WatermarkSettings | null;
  exportMasks?: boolean;
  preserveFolders?: boolean;
  destinationType?: string;
  subfolder?: string;
  /** path → 前端绘制的信息边框 PNG 字节 */
  framePngByPath?: Record<string, number[]> | null;
}

export enum BorderBasis {
  LongEdge = 'longEdge',
  ShortEdge = 'shortEdge',
  EachEdge = 'eachEdge',
}

export enum WatermarkAnchor {
  TopLeft = 'topLeft',
  TopCenter = 'topCenter',
  TopRight = 'topRight',
  CenterLeft = 'centerLeft',
  Center = 'center',
  CenterRight = 'centerRight',
  BottomLeft = 'bottomLeft',
  BottomCenter = 'bottomCenter',
  BottomRight = 'bottomRight',
}

interface WatermarkSettings {
  path: string;
  anchor: WatermarkAnchor;
  scale: number;
  spacing: number;
  opacity: number;
}

export type ExportBatchOutcome = 'success' | 'skipped' | 'failed';

export interface ExportBatchItem {
  elapsedMs: number;
  outcome: ExportBatchOutcome;
  outputPath?: string | null;
  path: string;
  reason?: string | null;
  reshapePreparationMs?: number;
}

export interface ExportBatchPerformance {
  p50Ms: number;
  p95Ms: number;
  totalMs: number;
}

export interface ExportBatchSummary {
  failed: number;
  items: ExportBatchItem[];
  performance: ExportBatchPerformance;
  skipped: number;
  succeeded: number;
}

export interface ExportState {
  batchSummary?: ExportBatchSummary;
  errorMessage: string;
  progress: Progress;
  status: Status;
  completedOutputs: string[];
}

const finiteNonNegative = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;

const readBatchItem = (value: unknown): ExportBatchItem | null => {
  if (!value || typeof value !== 'object') return null;
  const item = value as Record<string, unknown>;
  const elapsedMs = finiteNonNegative(item.elapsedMs);
  if (
    typeof item.path !== 'string' ||
    elapsedMs === null ||
    !['success', 'skipped', 'failed'].includes(item.outcome as string)
  )
    return null;
  const reshapePreparationMs = finiteNonNegative(item.reshapePreparationMs);
  return {
    elapsedMs,
    outcome: item.outcome as ExportBatchOutcome,
    outputPath: typeof item.outputPath === 'string' ? item.outputPath : null,
    path: item.path,
    reason: typeof item.reason === 'string' ? item.reason : null,
    ...(reshapePreparationMs === null ? {} : { reshapePreparationMs }),
  };
};

// 兼容旧事件 payload：缺少 batchSummary 时忽略，不将未知数据写入全局导出状态。
export function exportBatchSummary(payload: unknown): ExportBatchSummary | undefined {
  if (!payload || typeof payload !== 'object') return undefined;
  const summary = (payload as Record<string, unknown>).batchSummary;
  if (!summary || typeof summary !== 'object') return undefined;
  const value = summary as Record<string, unknown>;
  if (!Array.isArray(value.items)) return undefined;
  const items = value.items.map(readBatchItem).filter((item): item is ExportBatchItem => item !== null);
  if (items.length !== value.items.length) return undefined;
  const performance = value.performance as Record<string, unknown> | undefined;
  const totalMs = finiteNonNegative(performance?.totalMs);
  const p50Ms = finiteNonNegative(performance?.p50Ms);
  const p95Ms = finiteNonNegative(performance?.p95Ms);
  if (totalMs === null || p50Ms === null || p95Ms === null) return undefined;
  const succeeded = items.filter((item) => item.outcome === 'success').length;
  const skipped = items.filter((item) => item.outcome === 'skipped').length;
  const failed = items.filter((item) => item.outcome === 'failed').length;
  return { failed, items, performance: { p50Ms, p95Ms, totalMs }, skipped, succeeded };
}

// 兼容旧终态空 payload；只保留后端明确报告的已提交路径。
export function exportCompletedOutputs(payload: unknown): string[] {
  if (!payload || typeof payload !== 'object') return [];
  const data = payload as Record<string, unknown>;
  const paths: unknown[] = Array.isArray(data.completedOutputs) ? data.completedOutputs : [];
  return [
    ...new Set(
      [...paths, data.outputPath].filter((path): path is string => typeof path === 'string' && path.length > 0),
    ),
  ];
}

export interface FileFormat {
  extensions: Array<string>;
  id: string;
  name: string;
}

export interface ImportState {
  errorMessage: string;
  path?: string;
  progress?: Progress;
  status: Status;
}

export enum Status {
  Cancelled = 'cancelled',
  Cancelling = 'cancelling',
  Exporting = 'exporting',
  Error = 'error',
  Idle = 'idle',
  Importing = 'importing',
  Success = 'success',
}

export interface ExportPreset {
  id: string;
  name: string;
  fileFormat: string;
  jpegQuality: number;
  tiffBitDepth?: TiffBitDepth;
  enableResize: boolean;
  resizeMode: string;
  resizeValue: number;
  dontEnlarge: boolean;
  enablePad?: boolean;
  padRatioWidth?: number;
  padRatioHeight?: number;
  padColor?: string;
  enableBorder?: boolean;
  borderBasis?: string;
  borderHorizontalPercent?: number;
  borderVerticalPercent?: number;
  borderColor?: string;
  keepMetadata: boolean;
  preserveTimestamps: boolean;
  stripGps: boolean;
  exportMasks?: boolean;
  preserveFolders?: boolean;
  filenameTemplate: string;
  enableWatermark: boolean;
  watermarkPath: string | null;
  watermarkAnchor: string;
  watermarkScale: number;
  watermarkSpacing: number;
  watermarkOpacity: number;
  lastExportPath?: string;
  destinationType?: string;
  subfolder?: string;
}

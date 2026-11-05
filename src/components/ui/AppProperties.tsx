import { ExportPreset } from './ExportImportProperties';
import { Adjustments, CopyPasteSettings } from '../../utils/adjustments';
import { ToolType } from '../panel/right/Masks';
import type { WhiteBalance, WhiteBalanceMode } from '../../utils/whiteBalance';

export const GLOBAL_KEYS = [
  ' ',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'f',
  'b',
  'a',
  's',
  'd',
  'r',
  'm',
  'k',
  'p',
  'i',
  'e',
  '0',
  '1',
  '2',
  '3',
  '4',
  '5',
  'Enter',
];
export const OPTION_SEPARATOR = 'separator';

export enum Invokes {
  AddTagForPaths = 'add_tag_for_paths',
  ApplyAdjustments = 'apply_adjustments',
  PreparePortraitAnalysis = 'prepare_portrait_analysis',
  CancelPortraitAnalysis = 'cancel_portrait_analysis',
  PreparePortraitReshapeAnalysis = 'prepare_portrait_reshape_analysis',
  CancelPortraitReshapeAnalysis = 'cancel_portrait_reshape_analysis',
  ApplyAdjustmentsToPaths = 'apply_adjustments_to_paths',
  ApplyAutoAdjustmentsToPaths = 'apply_auto_adjustments_to_paths',
  ApplyDenoising = 'apply_denoising',
  CalculateAutoAdjustments = 'calculate_auto_adjustments',
  SampleWhiteBalance = 'sample_white_balance',
  CancelAiTask = 'cancel_ai_task',
  CancelExport = 'cancel_export',
  CheckAIConnectorStatus = 'check_ai_connector_status',
  ClearAllSidecars = 'clear_all_sidecars',
  ClearAiTags = 'clear_ai_tags',
  ClearAllTags = 'clear_all_tags',
  ClearThumbnailCache = 'clear_thumbnail_cache',
  CopyFiles = 'copy_files',
  CreateFolder = 'create_folder',
  CreateVirtualCopy = 'create_virtual_copy',
  CullImages = 'cull_images',
  DeleteFolder = 'delete_folder',
  DuplicateFile = 'duplicate_file',
  EstimateExportSizes = 'estimate_export_sizes',
  ExportImages = 'export_images',
  FrontendLog = 'frontend_log',
  GenerateAiForegroundMask = 'generate_ai_foreground_mask',
  GenerateAiSkyMask = 'generate_ai_sky_mask',
  GenerateAiSubjectMask = 'generate_ai_subject_mask',
  GeneratePreviewForPath = 'generate_preview_for_path',
  GenerateMaskOverlay = 'generate_mask_overlay',
  GeneratePresetPreview = 'generate_preset_preview',
  GenerateUncroppedPreview = 'generate_uncropped_preview',
  GetFolderTree = 'get_folder_tree',
  GetFolderChildren = 'get_folder_children',
  GetLogFilePath = 'get_log_file_path',
  GetOrCreateInternalLibraryRoot = 'get_or_create_internal_library_root',
  GetPinnedFolderTrees = 'get_pinned_folder_trees',
  GetSupportedFileTypes = 'get_supported_file_types',
  HandleExportPresetsToFile = 'handle_export_presets_to_file',
  HandleImportPresetsFromFile = 'handle_import_presets_from_file',
  HandleImportPresetsFromFiles = 'handle_import_presets_from_files',
  HandleImportLegacyPresetsFromFile = 'handle_import_legacy_presets_from_file',
  ImportFiles = 'import_files',
  InvokeGenerativeReplaseWithMaskDef = 'invoke_generative_replace_with_mask_def',
  IsTetheringSupported = 'is_tethering_supported',
  IsRaw9Available = 'is_raw9_available',
  ListImagesInDir = 'list_images_in_dir',
  ListImagesRecursive = 'list_images_recursive',
  LoadImage = 'load_image',
  LoadMetadata = 'load_metadata',
  LoadPresets = 'load_presets',
  LoadSettings = 'load_settings',
  MoveFiles = 'move_files',
  ReadExifForPaths = 'read_exif_for_paths',
  RemoveTagForPaths = 'remove_tag_for_paths',
  RenameFiles = 'rename_files',
  RenameFolder = 'rename_folder',
  ResetAdjustmentsForPaths = 'reset_adjustments_for_paths',
  SaveMetadataAndUpdateThumbnail = 'save_metadata_and_update_thumbnail',
  SaveCollage = 'save_collage',
  SaveDenoisedImage = 'save_denoised_image',
  SavePanorama = 'save_panorama',
  SaveHdr = 'save_hdr',
  SavePresets = 'save_presets',
  SaveSettings = 'save_settings',
  SetColorLabelForPaths = 'set_color_label_for_paths',
  SetRatingForPaths = 'set_rating_for_paths',
  SetFlagForPaths = 'set_flag_for_paths',
  ShowInFinder = 'show_in_finder',
  StartBackgroundIndexing = 'start_background_indexing',
  StitchPanorama = 'stitch_panorama',
  StitchFocusStack = 'stitch_focus_stack',
  SaveFocusStack = 'save_focus_stack',
  MergeHdr = 'merge_hdr',
  TestAIConnectorConnection = 'test_ai_connector_connection',
  UpdateWgpuTransform = 'update_wgpu_transform',
  UpdateExifFields = 'update_exif_fields',
  FetchCommunityPresets = 'fetch_community_presets',
  GenerateAllCommunityPreviews = 'generate_all_community_previews',
  SaveCommunityPreset = 'save_community_preset',
  SaveTempFile = 'save_temp_file',
  GetAlbums = 'get_albums',
  SaveAlbums = 'save_albums',
  AddToAlbum = 'add_to_album',
  GetAlbumImages = 'get_album_images',
  TetherListCameras = 'tether_list_cameras',
  TetherConnect = 'tether_connect',
  TetherGetSettings = 'tether_get_settings',
  TetherSetSetting = 'tether_set_setting',
  TetherCapture = 'tether_capture',
  TetherGetPreview = 'tether_get_preview',
  QrLoginCreateSession = 'qr_login_create_session',
  QrLoginPoll = 'qr_login_poll',
  QrLoginVerifySaved = 'qr_login_verify_saved',
  QrLoginLogout = 'qr_login_logout',
  ReportDesktopUsage = 'report_desktop_usage',
  ReportFrontendError = 'report_frontend_error',
  FetchBugFeedbackInfo = 'fetch_bug_feedback_info',
  SubmitBugFeedback = 'submit_bug_feedback',
  FetchAnnouncements = 'fetch_announcements',
  FetchRequirementInfo = 'fetch_requirement_info',
  FetchRequirements = 'fetch_requirements',
  SubmitRequirement = 'submit_requirement',
  ToggleRequirementLike = 'toggle_requirement_like',
  FetchAppVersionPolicy = 'fetch_app_version_policy',
}

/** 服务端 sys_config 下发的版本策略；未配置或格式错误时为 null */
export interface AppVersionPolicy {
  latestVersion?: string | null;
  minVersion?: string | null;
}

/** Bug 反馈配置（后端 /wx/xiuleme/bug/info）；功能关闭时 enabled=false */
export interface BugFeedbackInfo {
  enabled: boolean;
  reward: number;
  dailySubmitLimit: number;
  todaySubmitCount: number;
  /** 今日剩余可提交次数，-1 表示不限 */
  remainingSubmitCount: number;
  minLength: number;
}

/** 提交 Bug 反馈结果 */
export interface BugFeedbackSubmitResult {
  id: number;
  reward: number;
  remainingSubmitCount: number;
}

/** 扫码登录会话（后端 /wx/qr-login/session 返回） */
export interface QrLoginSession {
  qrToken: string;
  expiresIn?: number;
}

/** 登录用户信息（token 仅保存在 Rust 端，不下发给前端） */
export interface QrLoginUser {
  userId?: number | null;
  userNo?: string | null;
  nickname?: string | null;
  avatarUrl?: string | null;
}

/** 扫码登录轮询结果 */
export interface QrLoginPollResult {
  status: 'WAITING' | 'CONFIRMED' | 'CANCELED' | 'EXPIRED';
  user?: QrLoginUser | null;
}

export interface RenderStamp {
  requestId: number;
  sourceRevision: number;
  epoch: number;
  compositeKey: string;
  reshapeKey: string;
  windowKey: string;
}

export type ReshapeAttenuation = 'limited' | 'identity';

export interface PreviewMetadata {
  path: string;
  stamp: RenderStamp;
  mode: 'native' | 'jpeg';
  dimensions: [number, number];
  roi: { x: number; y: number; width: number; height: number };
  /** 非零重塑被安全回退限幅或压到恒等时由后端带上；缺省表示无渲染期限幅提示。 */
  reshapeAttenuation?: ReshapeAttenuation | null;
  /** 身体塑形引擎旁路时由后端带上；与人脸重塑限幅隔离。 */
  bodyAttenuation?: ReshapeAttenuation | null;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const isPositiveInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
const isDigest = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

export const isRenderStamp = (value: unknown): value is RenderStamp =>
  isRecord(value) &&
  isPositiveInteger(value.requestId) &&
  isPositiveInteger(value.sourceRevision) &&
  isPositiveInteger(value.epoch) &&
  isDigest(value.compositeKey) &&
  (value.reshapeKey === 'bypass' || isDigest(value.reshapeKey)) &&
  isDigest(value.windowKey);

export const sameRenderStamp = (a: RenderStamp | null, b: RenderStamp): boolean =>
  a !== null &&
  a.requestId === b.requestId &&
  a.sourceRevision === b.sourceRevision &&
  a.epoch === b.epoch &&
  a.compositeKey === b.compositeKey &&
  a.reshapeKey === b.reshapeKey &&
  a.windowKey === b.windowKey;

export const isPreviewMetadata = (value: unknown): value is PreviewMetadata => {
  if (
    !isRecord(value) ||
    typeof value.path !== 'string' ||
    !value.path ||
    !isRenderStamp(value.stamp) ||
    (value.mode !== 'native' && value.mode !== 'jpeg') ||
    !Array.isArray(value.dimensions) ||
    value.dimensions.length !== 2 ||
    !value.dimensions.every(isPositiveInteger) ||
    !isRecord(value.roi)
  )
    return false;
  if (
    value.reshapeAttenuation !== undefined &&
    value.reshapeAttenuation !== null &&
    value.reshapeAttenuation !== 'limited' &&
    value.reshapeAttenuation !== 'identity'
  )
    return false;
  if (
    value.bodyAttenuation !== undefined &&
    value.bodyAttenuation !== null &&
    value.bodyAttenuation !== 'limited' &&
    value.bodyAttenuation !== 'identity'
  )
    return false;
  const [w, h] = value.dimensions as [number, number];
  const { x, y, width, height } = value.roi;
  return (
    w <= 32768 &&
    h <= 32768 &&
    w * h <= 64000000 &&
    typeof x === 'number' &&
    Number.isSafeInteger(x) &&
    x >= 0 &&
    typeof y === 'number' &&
    Number.isSafeInteger(y) &&
    y >= 0 &&
    isPositiveInteger(width) &&
    isPositiveInteger(height) &&
    x + width <= w &&
    y + height <= h &&
    (value.mode !== 'native' || (x === 0 && y === 0 && width === w && height === h))
  );
};

// 先检查长度再解码；拒绝旧裸 JPEG、旧 ROI 头、截断和非法 UTF-8，避免错误地发布半帧。
export const decodePreviewResponse = (buffer: ArrayBuffer): { metadata: PreviewMetadata; image: ArrayBuffer } => {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 8) throw new Error('预览响应长度非法');
  const view = new DataView(buffer);
  if (view.getUint32(0, false) !== 0x52525632) throw new Error('预览协议版本不匹配');
  const length = view.getUint32(4, true);
  if (length === 0 || length > 65536 || length > buffer.byteLength - 8) throw new Error('预览元数据长度非法');
  const metadata: unknown = JSON.parse(
    new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(buffer, 8, length)),
  );
  if (!isPreviewMetadata(metadata)) throw new Error('预览元数据非法');
  const image = buffer.slice(8 + length);
  if (metadata.mode === 'native' ? image.byteLength !== 0 : image.byteLength === 0) throw new Error('预览图像体非法');
  return { metadata, image };
};

export enum ExifOverlay {
  Off = 'off',
  Hover = 'hover',
  Always = 'always',
}

export enum Panel {
  Adjustments = 'adjustments',
  Portrait = 'portrait',
  Ai = 'ai',
  Crop = 'crop',
  Export = 'export',
  Frame = 'frame',
  Masks = 'masks',
  Metadata = 'metadata',
  Presets = 'presets',
  FolderTree = 'folderTree',
  Tethering = 'tethering',
}

export type PanelRegion = 'leftTop' | 'leftBottom' | 'rightTop' | 'rightBottom';

export enum RawStatus {
  All = 'all',
  NonRawOnly = 'nonRawOnly',
  RawOnly = 'rawOnly',
}

export enum SortDirection {
  Ascending = 'asc',
  Descending = 'desc',
}

type FolderSortKey = 'name' | 'modified' | 'created' | 'imageCount';

export interface FolderTreeSort {
  key: FolderSortKey;
  order: SortDirection;
}

export enum Theme {
  Arctic = 'arctic',
  Blue = 'blue',
  Dark = 'dark',
  Grey = 'grey',
  Light = 'light',
  MutedGreen = 'muted-green',
  Sepia = 'sepia',
  Snow = 'snow',
}

export enum ThumbnailAspectRatio {
  Cover = 'cover',
  Contain = 'contain',
  Justified = 'justified',
}

export interface WorkspaceState {
  leftPanelWidth: number;
  rightPanelWidth: number;
  leftTopHeight: number;
  rightTopHeight: number;
  panelLayout: Record<PanelRegion, Panel[]>;
  activePanels: Record<PanelRegion, Panel | null>;
  panelSwitcherPlacement: Record<PanelRegion, 'left' | 'right' | 'top' | 'bottom'>;
}

export interface CustomAspectRatio {
  width: number;
  height: number;
}

export type GroupPreference = 'jpeg' | 'raw';
export type GroupingMode = 'off' | GroupPreference;

export interface AppSettings {
  aiConnectorAddress?: string;
  aiProvider?: string;
  decorations?: any;
  editorPreviewResolution?: number;
  smallThumbnailResolution?: number;
  mediumThumbnailResolution?: number;
  enableZoomHifi?: boolean;
  useFullDpiRendering?: boolean;
  highResZoomMultiplier?: number;
  enableLivePreviews?: boolean;
  livePreviewQuality?: string;
  enableAiTagging?: boolean;
  aiTagCount?: number;
  customAiTags?: string[];
  filterCriteria?: FilterCriteria;
  lastFolderState?: any;
  pinnedFolders?: any;
  lastRootPath: string | null;
  rootFolders?: string[];
  libraryViewMode?: LibraryViewMode;
  sortCriteria?: SortCriteria;
  theme: Theme;
  thumbnailSize?: ThumbnailSize;
  thumbnailAspectRatio?: ThumbnailAspectRatio;
  uiVisibility?: UiVisibility;
  adjustmentVisibility?: { [key: string]: boolean };
  rawHighlightCompression?: number;
  rawPreprocessingColorNr?: number;
  rawPreprocessingSharpening?: number;
  applyPreprocessingToNonRaws?: boolean;
  useAppleRaw9?: boolean;
  processingBackend?: string;
  linuxGpuOptimization?: boolean;
  exportPresets?: ExportPreset[];
  myLenses?: any;
  customAspectRatios?: CustomAspectRatio[];
  adjustmentLayout?: AdjustmentLayout;
  enableFolderImageCounts?: boolean;
  displayEditIcon?: boolean;
  linearRawMode?: string;
  enableXmpSync?: boolean;
  createXmpIfMissing?: boolean;
  isWaveformVisible?: boolean;
  waveformHeight?: number;
  activeWaveformChannel?: string;
  useWgpuRenderer?: boolean;
  editorNeutralGreyBg?: boolean;
  canvasInputMode?: 'mouse' | 'trackpad';
  whiteBalanceMode?: WhiteBalanceMode;
  zoomSpeedMultiplier?: number;
  zoomPhotoToPixelClick?: boolean;
  keybinds?: { [action: string]: string[] };
  tonemapperOverrideEnabled?: boolean;
  defaultRawTonemapper?: string;
  defaultNonRawTonemapper?: string;
  copyPasteSettings?: CopyPasteSettings;
  enableFocusMode?: boolean;
  enableToolFocusMode?: boolean;
  openTreeSections?: string[];
  folderIcons?: Record<string, string>;
  exifOverlay?: ExifOverlay;
  language?: string;
  fontFamily?: string;
  folderTreeSort?: FolderTreeSort;
  taggingShortcuts?: string[];
  libraryDisplayMode?: LibraryDisplayMode;
  grouping?: GroupingMode;
  requireMatchingExif?: boolean;
  groupEditedFiles?: boolean;
  groupPreferredType?: GroupPreference; // legacy
  alwaysDecodeRawThumbnails?: boolean;
  workspace?: WorkspaceState;
  skippedUpdateVersion?: string | null;
}

export interface BrushSettings {
  feather: number;
  size: number;
  tool: ToolType;
}

export enum LibraryViewMode {
  Flat = 'flat',
  Recursive = 'recursive',
}

export const EditedStatus = {
  All: 'all',
  EditedOnly: 'editedOnly',
  UneditedOnly: 'uneditedOnly',
} as const;

export type EditedStatus = (typeof EditedStatus)[keyof typeof EditedStatus];

export const ImageFlag = {
  Pick: 'pick',
  Reject: 'reject',
} as const;

export type ImageFlag = (typeof ImageFlag)[keyof typeof ImageFlag];

export const FlagStatus = {
  All: 'all',
  Picked: 'picked',
  ExcludeRejected: 'excludeRejected',
  Rejected: 'rejected',
} as const;

export type FlagStatus = (typeof FlagStatus)[keyof typeof FlagStatus];

export type RatingOperator = 'gte' | 'eq' | 'lte';

export const RATING_OPERATORS = {
  lte: {
    symbol: '≤',
    next: 'eq',
    labelKey: 'library.header.viewOptions.ratingOperator.lte',
    suffixKey: 'library.filters.rating.andDownSuffix',
  },
  eq: {
    symbol: '=',
    next: 'gte',
    labelKey: 'library.header.viewOptions.ratingOperator.eq',
    suffixKey: 'library.filters.rating.onlySuffix',
  },
  gte: {
    symbol: '≥',
    next: 'lte',
    labelKey: 'library.header.viewOptions.ratingOperator.gte',
    suffixKey: 'library.filters.rating.andUpSuffix',
  },
} as const;

export interface FilterCriteria {
  colors: Array<string>;
  rating: number;
  ratingOperator?: RatingOperator;
  rawStatus: RawStatus;
  editedStatus?: EditedStatus;
  flagStatus?: FlagStatus;
}

export interface Folder {
  children: any;
  id?: string | undefined;
  name?: string | undefined;
  imageCount?: number;
}

export interface ImageFile {
  is_edited: boolean;
  modified: number;
  path: string;
  rating: number;
  flag: ImageFlag | null;
  tags: Array<string> | null;
  exif: { [key: string]: string } | null;
  is_virtual_copy: boolean;
  is_cloud_placeholder: boolean;
  is_raw: boolean;
  group_id: string | null;
}

export interface Option {
  color?: string;
  disabled?: boolean;
  icon?: any;
  isDestructive?: boolean;
  label?: string;
  onClick?(): void;
  onRightClick?(): void;
  submenu?: any;
  type?: string;
}

export enum Orientation {
  Horizontal = 'horizontal',
  Vertical = 'vertical',
}

export interface Preset {
  adjustments: Partial<Adjustments>;
  folder?: Folder;
  id: string;
  name: string;
  includeMasks?: boolean;
  includeCropTransform?: boolean;
  presetType?: 'tool' | 'style';
}

export interface Progress {
  completed?: number;
  current?: number;
  stage?: string;
  total: number;
}

export interface SelectedImage {
  asShotWhiteBalance?: WhiteBalance;
  exif: any;
  group_id?: string | null;
  height: number;
  isRaw: boolean;
  isReady: boolean;
  metadata?: any;
  original_base64?: string;
  path: string;
  thumbnailUrl: string;
  width: number;
}

export interface SortCriteria {
  key: string;
  label?: string;
  order: string;
}

export interface SupportedTypes {
  nonRaw: Array<string>;
  raw: Array<string>;
}

export enum LibraryDisplayMode {
  Grid = 'grid',
  Cull = 'cull',
  List = 'list',
}

export enum ThumbnailSize {
  Large = 'large',
  Medium = 'medium',
  Small = 'small',
}

export interface TransformState {
  positionX: number;
  positionY: number;
  scale: number;
}

export interface UiVisibility {
  filmstrip: boolean;
  leftPanel: boolean;
  rightPanel: boolean;
  quickFilter?: boolean;
}

/** Includes local HSL section keys (colorGrading / colorMixer / colorCalibration). */
export interface CollapsibleSectionsState {
  basic: boolean;
  color: boolean;
  colorGrading: boolean;
  colorMixer: boolean;
  colorCalibration: boolean;
  curves: boolean;
  levels: boolean;
  details: boolean;
  effects: boolean;
}

export interface AdjustmentLayout {
  collapsedTools?: string[];
  hiddenSections?: string[];
  hiddenTools?: string[];
  openSections?: Partial<CollapsibleSectionsState>;
  sectionOrder?: string[];
  toolOrder?: Record<string, string[]>;
}

export interface WaveformData {
  blue: string;
  green: string;
  height: number;
  luma: string;
  red: string;
  rgb: string;
  parade: string;
  vectorscope: string;
  width: number;
}

export interface CullingSettings {
  similarityThreshold: number;
  blurThreshold: number;
  groupSimilar: boolean;
  filterBlurry: boolean;
}

interface ImageAnalysisResult {
  path: string;
  qualityScore: number;
  sharpnessMetric: number;
  centerFocusMetric: number;
  exposureMetric: number;
  width: number;
  height: number;
}

interface CullGroup {
  representative: ImageAnalysisResult;
  duplicates: ImageAnalysisResult[];
}

export interface CullingSuggestions {
  similarGroups: CullGroup[];
  blurryImages: ImageAnalysisResult[];
  failedPaths: string[];
}

interface KeybindHandler {
  shouldFire?: () => boolean;
  execute: (event: KeyboardEvent) => void;
}

export type AlbumItem = Album | AlbumGroup;

export interface Album {
  type: 'album';
  id: string;
  name: string;
  icon?: string;
  images: string[];
}

export interface AlbumGroup {
  type: 'group';
  id: string;
  name: string;
  icon?: string;
  children: AlbumItem[];
}

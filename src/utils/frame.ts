// 拍摄信息边框：状态、品牌 Logo 匹配、Canvas 绘制。
// 预览与导出共用同一套绘制逻辑，保证所见即所得。

export interface FrameState {
  enabled: boolean;
  showParams: boolean;
  showDate: boolean;
  showLogo: boolean;
  showCamera: boolean;
  showLens: boolean;
}

/** 边框绘制用的规范化 EXIF（由 XiuLeMe 原始 tag map 适配而来） */
export interface FrameExif {
  iso: number | null;
  shutterSpeed: string | null;
  aperture: number | null;
  cameraMake: string | null;
  cameraModel: string | null;
  lensModel: string | null;
  dateTime: string | null;
}

export function defaultFrame(): FrameState {
  return {
    enabled: false,
    showParams: true,
    showDate: true,
    showLogo: true,
    showCamera: true,
    showLens: true,
  };
}

export function isFrameIdentity(f: FrameState | null | undefined): boolean {
  return !f?.enabled;
}

export function normalizeFrame(frame: Partial<FrameState> | null | undefined): FrameState {
  return { ...defaultFrame(), ...(frame || {}) };
}

/** 将 XiuLeMe EXIF tag map 转为边框绘制字段 */
export function exifFromXiuLeMe(exif: Record<string, string> | null | undefined): FrameExif {
  if (!exif) {
    return {
      iso: null,
      shutterSpeed: null,
      aperture: null,
      cameraMake: null,
      cameraModel: null,
      lensModel: null,
      dateTime: null,
    };
  }

  const isoRaw = exif.PhotographicSensitivity || exif.ISOSpeedRatings || exif.ISOSpeed || exif.ISO;
  const isoParsed = isoRaw ? parseInt(String(isoRaw).replace(/[^\d]/g, ''), 10) : NaN;

  let aperture: number | null = null;
  const fNumber = exif.FNumber || exif.ApertureValue;
  if (fNumber) {
    const cleaned = String(fNumber).replace(/^f\/?/i, '').trim();
    const n = parseFloat(cleaned);
    if (Number.isFinite(n)) aperture = n;
  }

  let shutterSpeed: string | null = null;
  const exposure = exif.ExposureTime || exif.ShutterSpeedValue;
  if (exposure) {
    shutterSpeed = String(exposure)
      .replace(/\s*s$/i, '')
      .trim();
  }

  return {
    iso: Number.isFinite(isoParsed) ? isoParsed : null,
    shutterSpeed: shutterSpeed || null,
    aperture,
    cameraMake: exif.Make?.trim() || null,
    cameraModel: exif.Model?.trim() || null,
    lensModel: exif.LensModel?.trim() || null,
    dateTime: exif.DateTimeOriginal || exif.CreateDate || exif.DateTime || null,
  };
}

const logoModules = import.meta.glob('../assets/camera-logos/*.svg', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>;

const logoUrls: Record<string, string> = {};
for (const [path, url] of Object.entries(logoModules)) {
  const m = path.match(/([^/\\]+)\.svg$/);
  if (m) logoUrls[m[1].toLowerCase()] = url;
}

const BRAND_MATCHERS: Array<[string, string]> = [
  ['nikon', 'nikon'],
  ['canon', 'canon'],
  ['sony', 'sony'],
  ['fujifilm', 'fujifilm'],
  ['fuji', 'fujifilm'],
  ['panasonic', 'panasonic'],
  ['om digital', 'omsystem'],
  ['om system', 'omsystem'],
  ['olympus', 'olympus'],
  ['leica', 'leica'],
  ['pentax', 'pentax'],
  ['ricoh', 'ricoh'],
  ['hasselblad', 'hasselblad'],
  ['sigma', 'sigma'],
  ['dji', 'dji'],
  ['apple', 'apple'],
  ['samsung', 'samsung'],
  ['xiaomi', 'xiaomi'],
  ['huawei', 'huawei'],
  ['gopro', 'gopro'],
];

export function detectBrand(make: string | null, model: string | null): string | null {
  const hay = `${make ?? ''} ${model ?? ''}`.toLowerCase();
  if (!hay.trim()) return null;
  for (const [needle, key] of BRAND_MATCHERS) {
    if (hay.includes(needle) && logoUrls[key]) return key;
  }
  return null;
}

const logoImageCache = new Map<string, Promise<HTMLImageElement | null>>();

function loadLogoImage(brand: string): Promise<HTMLImageElement | null> {
  const cached = logoImageCache.get(brand);
  if (cached) return cached;
  const url = logoUrls[brand];
  const p = new Promise<HTMLImageElement | null>((resolve) => {
    if (!url) return resolve(null);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
  logoImageCache.set(brand, p);
  return p;
}

export function formatExifDate(dateTime: string | null): string | null {
  if (!dateTime) return null;
  const m = dateTime.match(/^(\d{4})[:-](\d{2})[:-](\d{2})[ T](\d{2}:\d{2}:\d{2})/);
  if (!m) return dateTime;
  return `${m[1]}年${parseInt(m[2], 10)}月${parseInt(m[3], 10)}日 ${m[4]}`;
}

export function frameBarHeight(imageWidth: number): number {
  return Math.max(80, Math.round(imageWidth * 0.11));
}

const FONT_STACK = '-apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif';

function cameraDisplayName(exif: FrameExif): string | null {
  const { cameraMake, cameraModel } = exif;
  if (cameraModel && cameraMake) {
    return cameraModel.toLowerCase().includes(cameraMake.toLowerCase())
      ? cameraModel
      : `${cameraMake} ${cameraModel}`;
  }
  return cameraModel || cameraMake;
}

function paramsLine(exif: FrameExif): string | null {
  const parts: string[] = [];
  if (exif.iso != null) parts.push(`ISO ${exif.iso}`);
  if (exif.aperture != null) parts.push(`f/${exif.aperture}`);
  if (exif.shutterSpeed) parts.push(`${exif.shutterSpeed}s`);
  return parts.length ? parts.join('  ') : null;
}

export async function renderFrameBar(
  width: number,
  exif: FrameExif,
  frame: FrameState,
): Promise<HTMLCanvasElement> {
  const h = frameBarHeight(width);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, h);

  const padX = Math.round(h * 0.5);
  const boldSize = Math.round(h * 0.19);
  const graySize = Math.round(h * 0.15);
  const lineGap = Math.round(h * 0.09);
  const boldFont = `600 ${boldSize}px ${FONT_STACK}`;
  const grayFont = `400 ${graySize}px ${FONT_STACK}`;
  const darkColor = '#1a1a1a';
  const grayColor = '#8e8e8e';

  const line1 = frame.showParams ? paramsLine(exif) : null;
  const line2 = frame.showDate ? formatExifDate(exif.dateTime) : null;
  {
    const rows: Array<{ text: string; font: string; size: number; color: string }> = [];
    if (line1) rows.push({ text: line1, font: boldFont, size: boldSize, color: darkColor });
    if (line2) rows.push({ text: line2, font: grayFont, size: graySize, color: grayColor });
    const totalH = rows.reduce((s, r) => s + r.size, 0) + (rows.length > 1 ? lineGap : 0);
    let y = (h - totalH) / 2;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    for (const r of rows) {
      ctx.font = r.font;
      ctx.fillStyle = r.color;
      ctx.fillText(r.text, padX, y);
      y += r.size + lineGap;
    }
  }

  const cameraText = frame.showCamera ? cameraDisplayName(exif) : null;
  const lensText = frame.showLens ? exif.lensModel : null;
  const brand = frame.showLogo ? detectBrand(exif.cameraMake, exif.cameraModel) : null;
  const logoImg = brand ? await loadLogoImage(brand) : null;

  const rows: Array<{ text: string; font: string; size: number; color: string }> = [];
  if (cameraText) rows.push({ text: cameraText, font: boldFont, size: boldSize, color: darkColor });
  if (lensText) rows.push({ text: lensText, font: grayFont, size: graySize, color: grayColor });

  let textBlockW = 0;
  for (const r of rows) {
    ctx.font = r.font;
    textBlockW = Math.max(textBlockW, ctx.measureText(r.text).width);
  }
  const maxTextW = canvas.width * 0.45;
  textBlockW = Math.min(textBlockW, maxTextW);

  let cursorX = canvas.width - padX;
  if (rows.length) {
    const textX = cursorX - textBlockW;
    const totalH = rows.reduce((s, r) => s + r.size, 0) + (rows.length > 1 ? lineGap : 0);
    let y = (h - totalH) / 2;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    for (const r of rows) {
      ctx.font = r.font;
      ctx.fillStyle = r.color;
      ctx.fillText(r.text, textX, y, textBlockW);
      y += r.size + lineGap;
    }
    cursorX = textX;
  }

  const fallbackMake =
    !logoImg && frame.showLogo && exif.cameraMake && !cameraText ? exif.cameraMake : null;
  if (rows.length && (logoImg || fallbackMake)) {
    const divGap = Math.round(h * 0.22);
    const divX = cursorX - divGap;
    ctx.strokeStyle = '#d9d9d9';
    ctx.lineWidth = Math.max(1, Math.round(h * 0.012));
    ctx.beginPath();
    ctx.moveTo(divX, h * 0.28);
    ctx.lineTo(divX, h * 0.72);
    ctx.stroke();
    cursorX = divX - divGap;
  }

  if (logoImg) {
    const logoH = h * 0.34;
    const ratio = logoImg.width && logoImg.height ? logoImg.width / logoImg.height : 3;
    const logoW = logoH * ratio;
    ctx.drawImage(logoImg, cursorX - logoW, (h - logoH) / 2, logoW, logoH);
  } else if (fallbackMake) {
    ctx.font = boldFont;
    ctx.fillStyle = darkColor;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(fallbackMake, cursorX, h / 2);
    ctx.textAlign = 'left';
  }

  return canvas;
}

export async function renderFrameBarPng(
  width: number,
  exif: FrameExif,
  frame: FrameState,
): Promise<Uint8Array> {
  const canvas = await renderFrameBar(width, exif, frame);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('边框条编码失败');
  return new Uint8Array(await blob.arrayBuffer());
}

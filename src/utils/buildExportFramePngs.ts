import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import type { Adjustments } from './adjustments';
import {
  defaultFrame,
  exifFromXiuLeMe,
  isFrameIdentity,
  normalizeFrame,
  renderFrameBarPng,
  type FrameState,
} from './frame';

function frameFromAdjustments(adjustments: any): FrameState {
  return normalizeFrame(adjustments?.frame);
}

/**
 * 为导出路径生成信息边框 PNG（仅 enabled 的路径）。
 * 当前编辑图用 currentAdjustments；其余从 sidecar 读取。
 */
export async function buildExportFramePngs(
  paths: string[],
  currentEditPath: string | null | undefined,
  currentAdjustments: Adjustments | null | undefined,
  exifByPath: Map<string, Record<string, string> | null | undefined>,
): Promise<Record<string, number[]> | null> {
  const result: Record<string, number[]> = {};

  for (const path of paths) {
    let frame: FrameState = defaultFrame();
    if (currentEditPath && path === currentEditPath && currentAdjustments) {
      frame = frameFromAdjustments(currentAdjustments);
    } else {
      try {
        const metadata: any = await invoke(Invokes.LoadMetadata, { path });
        frame = frameFromAdjustments(metadata?.adjustments);
      } catch {
        continue;
      }
    }
    if (isFrameIdentity(frame)) continue;

    const exif = exifFromXiuLeMe(exifByPath.get(path) ?? null);
    // 用较高宽度绘制，后端按实际输出宽缩放
    const width = 4000;
    try {
      const png = await renderFrameBarPng(width, exif, frame);
      result[path] = Array.from(png);
    } catch (e) {
      console.warn('Failed to render info frame for', path, e);
    }
  }

  return Object.keys(result).length > 0 ? result : null;
}

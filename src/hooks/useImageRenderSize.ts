import { useState, useLayoutEffect } from 'react';

export interface ImageDimensions {
  height: number;
  width: number;
}

export interface RenderSize {
  containerHeight: number;
  containerWidth: number;
  height: number;
  offsetX: number;
  offsetY: number;
  scale: number;
  width: number;
}

const DEFAULT_SIZE: RenderSize = {
  width: 0,
  height: 0,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  containerWidth: 0,
  containerHeight: 0,
};

/** 与 frameBarHeight(displayWidth) 一致：max(80, round(w*0.11)) */
export function displayFrameBarHeight(displayWidth: number): number {
  return Math.max(80, Math.round(displayWidth * 0.11));
}

/**
 * @param withFrameBar 为 true 时为底部信息条预留高度（闭式计算，无反馈抖动）
 */
export const useImageRenderSize = (
  containerRef: React.RefObject<HTMLElement | null>,
  imageDimensions: ImageDimensions | null,
  withFrameBar = false,
) => {
  const [renderSize, setRenderSize] = useState<RenderSize>(DEFAULT_SIZE);
  const imgWidth = imageDimensions?.width;
  const imgHeight = imageDimensions?.height;

  useLayoutEffect(() => {
    const container = containerRef.current;

    if (!container || !imgWidth || !imgHeight) {
      setRenderSize(DEFAULT_SIZE);
      return;
    }

    const updateSize = () => {
      const { clientWidth: containerWidth, clientHeight: containerHeight } = container;
      const imageAspectRatio = imgWidth / imgHeight;

      let width: number;
      let height: number;

      if (withFrameBar) {
        // 条高 = max(80, 0.11*w)。先按比例段试算，不足 80 时改用固定 80px 条高。
        const heightPerWidth = 1 / imageAspectRatio + 0.11;
        let w = Math.min(containerWidth, containerHeight / heightPerWidth);
        let barH = displayFrameBarHeight(w);
        if (barH > w * 0.11 + 0.5) {
          // 落到 min-80 段：图高 + 80 <= containerH
          const maxW = Math.max(1, (containerHeight - 80) * imageAspectRatio);
          w = Math.min(containerWidth, maxW);
          barH = displayFrameBarHeight(w);
        }
        width = w;
        height = width / imageAspectRatio;
        const totalH = height + barH;
        const offsetX = (containerWidth - width) / 2;
        const offsetY = Math.max(0, (containerHeight - totalH) / 2);
        setRenderSize({ width, height, scale: width / imgWidth, offsetX, offsetY, containerWidth, containerHeight });
        return;
      }

      if (imageAspectRatio > containerWidth / Math.max(containerHeight, 1)) {
        width = containerWidth;
        height = containerWidth / imageAspectRatio;
      } else {
        height = containerHeight;
        width = containerHeight * imageAspectRatio;
      }

      const offsetX = (containerWidth - width) / 2;
      const offsetY = (containerHeight - height) / 2;

      setRenderSize({ width, height, scale: width / imgWidth, offsetX, offsetY, containerWidth, containerHeight });
    };

    updateSize();

    const resizeObserver = new ResizeObserver(() => {
      updateSize();
    });

    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
  }, [containerRef, imgWidth, imgHeight, withFrameBar]);

  return renderSize;
};

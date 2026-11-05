import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import { renderFrameBar, type FrameExif, type FrameState } from '../../../../utils/frame';

interface FrameBarOverlayProps {
  width: number;
  exif: FrameExif;
  frame: FrameState;
  style?: CSSProperties;
}

/** 画布预览信息条：与导出共用 renderFrameBar */
export default function FrameBarOverlay({ width, exif, frame, style }: FrameBarOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!frame.enabled || width <= 0) return;
    let cancelled = false;
    renderFrameBar(Math.round(width), exif, frame).then((bar) => {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = bar.width;
      canvas.height = bar.height;
      canvas.getContext('2d')!.drawImage(bar, 0, 0);
    });
    return () => {
      cancelled = true;
    };
  }, [exif, frame, width]);

  return <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: 'auto', ...style }} />;
}

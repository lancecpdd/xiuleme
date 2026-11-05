import { useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';
import { useEditorStore } from '../../../store/useEditorStore';
import { useUIStore } from '../../../store/useUIStore';
import { PORTRAIT_RESHAPE_REASON_LABELS, reshapeFaceFrame } from '../../../utils/portraitReshapePanel';
import { Panel } from '../../ui/AppProperties';
import type { RenderSize } from '../../../hooks/useImageRenderSize';

interface FaceBox {
  faceId: string;
  x: number;
  y: number;
  width: number;
  height: number;
  selected: boolean;
  disabled: boolean;
  reasonLabel: string | null;
}

interface PortraitReshapeFaceOverlayProps {
  // 与 ImageCanvas 内 svg 叠加层同一坐标系：相对 contentRef（zoom transform 上下文）的像素基准
  imageRenderSize: RenderSize;
}

// 09 画面框：位置基于 08 冻结分析的源锚点经 09 的 linear 几何映射；
// 保护笔刷叠加（10）与形变后正向映射不在此步实现。选中状态不写 adjustments。
export default function PortraitReshapeFaceOverlay({ imageRenderSize }: PortraitReshapeFaceOverlayProps) {
  const { t } = useTranslation();
  const { selectedImage, adjustments, portraitReshapeStatus, showOriginal, isSliderDragging } = useEditorStore(
    useShallow((state) => ({
      selectedImage: state.selectedImage,
      adjustments: state.adjustments,
      portraitReshapeStatus: state.portraitReshapeStatus,
      showOriginal: state.showOriginal,
      isSliderDragging: state.isSliderDragging,
    })),
  );
  const activePanels = useUIStore((state) => state.activePanels);
  const activePanel = useUIStore((state) => state.activePanel);
  const portraitPanelActive =
    (activePanel === Panel.Portrait || Object.values(activePanels).includes(Panel.Portrait)) === true;

  const reasonLabels = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(PORTRAIT_RESHAPE_REASON_LABELS).map(([reason, key]) => [reason, t(key)]),
      ) as Record<string, string>,
    [t],
  );

  const boxes = useMemo<FaceBox[]>(() => {
    // 仅在人像面板聚焦时显示；对比原图与拖动中隐藏，避免遮挡调色对比。
    if (!portraitPanelActive || showOriginal || isSliderDragging) return [];
    if (!selectedImage?.isReady || !selectedImage.width || !selectedImage.height) return [];
    const status = portraitReshapeStatus;
    if (status.path !== selectedImage.path) return [];
    if (status.phase !== 'ready' && status.phase !== 'partial') return [];
    if (status.faces.length === 0) return [];
    const geometry = {
      sourceWidth: selectedImage.width,
      sourceHeight: selectedImage.height,
      orientationSteps: adjustments.orientationSteps || 0,
      flipHorizontal: adjustments.flipHorizontal === true,
      flipVertical: adjustments.flipVertical === true,
      rotation: adjustments.rotation || 0,
      crop: adjustments.crop ?? null,
    };
    return status.faces.flatMap((face) => {
      const frame = reshapeFaceFrame(face.sourceAnchor.bounds, geometry);
      if (!frame || frame.width <= 0 || frame.height <= 0) return [];
      return [
        {
          faceId: face.faceId,
          x: frame.x * 100,
          y: frame.y * 100,
          width: frame.width * 100,
          height: frame.height * 100,
          selected: status.selectedFaceId === face.faceId,
          disabled: !face.processable,
          reasonLabel: face.reason ? (reasonLabels[face.reason] ?? null) : null,
        },
      ];
    });
  }, [
    portraitPanelActive,
    showOriginal,
    isSliderDragging,
    selectedImage,
    adjustments,
    portraitReshapeStatus,
    reasonLabels,
  ]);

  if (boxes.length === 0 || imageRenderSize.width <= 0 || imageRenderSize.height <= 0) return null;

  return (
    <div
      className="absolute pointer-events-none"
      aria-hidden="true"
      data-testid="reshape-face-overlay"
      style={{
        left: `${imageRenderSize.offsetX}px`,
        top: `${imageRenderSize.offsetY}px`,
        width: `${imageRenderSize.width}px`,
        height: `${imageRenderSize.height}px`,
      }}
    >
      {boxes.map((box) => (
        <div
          key={box.faceId}
          className={clsx(
            'absolute rounded-sm border-2 transition-colors',
            box.selected
              ? 'border-accent bg-accent/10'
              : box.disabled
                ? 'border-red-500/70 border-dashed'
                : 'border-accent/40',
          )}
          style={{
            left: `${box.x}%`,
            top: `${box.y}%`,
            width: `${box.width}%`,
            height: `${box.height}%`,
          }}
        />
      ))}
    </div>
  );
}

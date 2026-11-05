//! 塑形窄带 LaMa 补全：由源／目标轮廓差生成有界 mask，失败时保留 warp 并标记 fill-skipped。
#![allow(dead_code)]
use image::{GrayImage, Luma, Rgb32FImage};

pub const FILL_VERSION: &str = "reshape-lama-fill-v1";
/// 窄带最大宽度（面宽比例）。
pub const MAX_BAND_FACE_WIDTH: f64 = 0.08;

#[derive(Clone, Debug, Default)]
pub struct FillReport {
    pub attempted: bool,
    pub applied: bool,
    pub skipped_reason: Option<&'static str>,
    pub mask_pixels: u32,
}

/// 根据位移幅度与可选头发／背景概率构建窄带 mask（255=需补全）。
pub fn build_narrowband_mask(
    width: u32,
    height: u32,
    displacements: &[[f32; 2]],
    face_width_px: f64,
    hair_or_bg: Option<&[f32]>,
) -> Result<(GrayImage, FillReport), String> {
    if width == 0 || height == 0 {
        return Err("补全尺寸非法".into());
    }
    let n = (width as usize).saturating_mul(height as usize);
    if displacements.len() != n {
        return Err("位移场与图像尺寸不符".into());
    }
    let band = (face_width_px * MAX_BAND_FACE_WIDTH).clamp(1.0, 64.0);
    let threshold = (band * 0.25).max(0.75);
    let mut mask = GrayImage::new(width, height);
    let mut count = 0u32;
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) as usize;
            let d = displacements[i];
            let mag = (d[0] * d[0] + d[1] * d[1]).sqrt() as f64;
            let mut on = mag >= threshold;
            if let Some(prob) = hair_or_bg {
                // 仅在头发／背景概率较高处允许补全，避免改五官。
                on = on && prob.get(i).copied().unwrap_or(0.0) >= 0.35;
            }
            if on {
                mask.put_pixel(x, y, Luma([255]));
                count += 1;
            }
        }
    }
    // 禁止全脸 inpaint：超过 12% 像素则跳过。
    let limit = ((n as f64) * 0.12) as u32;
    let mut report = FillReport {
        attempted: true,
        applied: false,
        skipped_reason: None,
        mask_pixels: count,
    };
    if count == 0 {
        report.skipped_reason = Some("fill-skipped-empty");
        return Ok((mask, report));
    }
    if count > limit {
        report.skipped_reason = Some("fill-skipped-too-large");
        // 清空 mask
        mask = GrayImage::from_pixel(width, height, Luma([0]));
        report.mask_pixels = 0;
        return Ok((mask, report));
    }
    Ok((mask, report))
}

/// 调用既有 LaMa；当前为 best-effort 占位，失败保留 warp 并标记原因。
pub fn apply_lama_fill(
    image: &Rgb32FImage,
    mask: &GrayImage,
    report: &mut FillReport,
) -> Rgb32FImage {
    if report.skipped_reason.is_some() || report.mask_pixels == 0 {
        return image.clone();
    }
    let _ = mask;
    // TODO: 接线 ai_processing::run_lama_inpainting（需 8bit 往返）。占位不改像素。
    report.skipped_reason = Some("fill-skipped-pending-lama-wire");
    report.applied = false;
    image.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_displacement_skips() {
        let w = 8u32;
        let h = 8u32;
        let disp = vec![[0f32; 2]; (w * h) as usize];
        let (mask, report) = build_narrowband_mask(w, h, &disp, 40.0, None).unwrap();
        assert_eq!(report.mask_pixels, 0);
        assert_eq!(report.skipped_reason, Some("fill-skipped-empty"));
        assert!(mask.iter().all(|&p| p == 0));
    }
}

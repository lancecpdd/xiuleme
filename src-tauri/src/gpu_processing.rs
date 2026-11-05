use std::sync::{Arc, Mutex};
use std::time::Instant;

use half::f16;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgba};
use sha2::{Digest, Sha256};
use std::num::NonZero;

#[cfg(not(any(target_os = "android", target_os = "linux")))]
use tauri::Manager;
use wgpu::util::{DeviceExt, TextureDataOrder};

use crate::image_processing::{AllAdjustments, GpuContext, MAX_MASKS, MaskAdjustments};
use crate::lut_processing::Lut;
use crate::portrait_gpu as pg;
use crate::portrait_tiles::{CompositionDependencies, DeviceBudget, GPU_BUDGET, PixelRect};
use crate::{AppState, GpuImageCache, GpuProcessorState};

/// 原生合成路径按 analysis.key 复用美肤九通道，避免拖调色时每帧 digest/upload。
/// 人像场合并为 binding 12 的 texture_2d_array（11 层，层序与 shader.wgsl 一致；
/// 层 10 eye_white：RGB=祛红单位增量，A=提亮巩膜权重），
/// 使采样纹理绑定不超过 GLES/WebGPU 下限 16。
const PORTRAIT_ATLAS_BINDING: u32 = 12;
const PORTRAIT_ATLAS_LAYERS: u32 = 11;
static NATIVE_PORTRAIT_FIELD_CACHE: Mutex<Option<(String, pg::TextureSlot)>> = Mutex::new(None);

fn wrap_owned_texture(texture: wgpu::Texture) -> pg::TextureSlot {
    let view = texture.create_view(&Default::default());
    pg::TextureSlot { texture, view }
}

fn clone_texture_slot(slot: &pg::TextureSlot) -> pg::TextureSlot {
    pg::TextureSlot {
        texture: slot.texture.clone(),
        view: slot.view.clone(),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderOutputPrecision {
    #[default]
    EightBit,
    SixteenBit,
}

pub struct RenderRequest<'a> {
    pub adjustments: AllAdjustments,
    pub mask_bitmaps: &'a [ImageBuffer<Luma<u8>, Vec<u8>>],
    pub lut: Option<Arc<Lut>>,
    pub roi: Option<Roi>,
    pub portrait_maps: Option<crate::portrait_processing::PortraitRenderMaps>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DisplayTransform {
    pub rect: [f32; 4],
    pub clip: [f32; 4],
    pub window: [f32; 2],
    pub image_size: [f32; 2],
    pub texture_size: [f32; 2],
    pub pixelated: f32,
    pub _pad: f32,
    pub bg_primary: [f32; 4],
    pub bg_secondary: [f32; 4],
}

pub struct WgpuDisplay {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    pub transform_buffer: wgpu::Buffer,
    pub latest_transform: DisplayTransform,
    pub current_bind_group: Option<wgpu::BindGroup>,
    pub(crate) current_texture_bytes: u64,
}

impl WgpuDisplay {
    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if let Err(error) = self.try_render(device, queue) {
            log::warn!("显示提交失败：{error}");
        }
    }

    pub(crate) fn try_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        if let Some(bind_group) = &self.current_bind_group {
            let output = match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(tex)
                | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(device, &self.config);
                    match self.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(tex)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
                        _ => return Err("重新配置后仍无法获取显示纹理".into()),
                    }
                }
                _ => return Err("显示表面暂不可用".into()),
            };
            let view = output
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: self.latest_transform.bg_primary[0] as f64,
                                g: self.latest_transform.bg_primary[1] as f64,
                                b: self.latest_transform.bg_primary[2] as f64,
                                a: self.latest_transform.bg_primary[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: NonZero::new(0),
                });
                let clip_x1 = self.latest_transform.clip[0].max(0.0);
                let clip_y1 = self.latest_transform.clip[1].max(0.0);
                let clip_x2 =
                    (self.latest_transform.clip[0] + self.latest_transform.clip[2]).max(0.0);
                let clip_y2 =
                    (self.latest_transform.clip[1] + self.latest_transform.clip[3]).max(0.0);

                let final_clip_x = clip_x1.floor() as u32;
                let final_clip_y = clip_y1.floor() as u32;
                let final_clip_w = (clip_x2.ceil() as u32).saturating_sub(final_clip_x);
                let final_clip_h = (clip_y2.ceil() as u32).saturating_sub(final_clip_y);

                let max_x = self.config.width;
                let max_y = self.config.height;

                if final_clip_x < max_x && final_clip_y < max_y {
                    let clamped_width = final_clip_w.min(max_x - final_clip_x);
                    let clamped_height = final_clip_h.min(max_y - final_clip_y);

                    if clamped_width > 0 && clamped_height > 0 {
                        rpass.set_scissor_rect(
                            final_clip_x,
                            final_clip_y,
                            clamped_width,
                            clamped_height,
                        );

                        rpass.set_pipeline(&self.pipeline);
                        rpass.set_bind_group(0, bind_group, &[]);
                        rpass.draw(0..4, 0..1);
                    }
                }
            }
            queue.submit(Some(encoder.finish()));
            output.present();
            Ok(())
        } else {
            Err("没有可提交的显示纹理".into())
        }
    }
}

/// 仅暂存最终预览尺寸；不修改上一成功显示帧。调用者持 GPU 共用锁直至提交。
pub(crate) struct PreparedPreviewDisplay {
    bind_group: wgpu::BindGroup,
    dimensions: [u32; 2],
    bytes: u64,
}

impl PreparedPreviewDisplay {
    pub(crate) fn prepare(
        context: &GpuContext,
        image: &image::RgbaImage,
        current: &dyn Fn() -> bool,
    ) -> Result<Self, String> {
        crate::portrait_analysis::check_current(current)?;
        finish_upload(context, current)?;
        let dimensions = [image.width(), image.height()];
        crate::portrait_tiles::check_dimensions(dimensions)?;
        let display = context.display.lock().unwrap_or_else(|e| e.into_inner());
        let display = display.as_ref().ok_or("原生显示尚未初始化")?;
        let (bytes, staging) =
            crate::portrait_tiles::texture_bytes([dimensions[0], dimensions[1], 1], 4)?;
        let total = bytes
            .checked_add(staging)
            .and_then(|n| n.checked_add(display.current_texture_bytes))
            .and_then(|n| {
                n.checked_add(LEGACY_READBACK_BYTES.load(std::sync::atomic::Ordering::SeqCst))
            })
            .ok_or("显示纹理预算溢出")?;
        if total > GPU_BUDGET
            || dimensions
                .iter()
                .any(|n| *n > context.device.limits().max_texture_dimension_2d)
            || staging > context.device.limits().max_buffer_size
        {
            return Err("最终预览显示纹理超过 GPU 预算或设备限制".into());
        }
        let texture = pg::TextureSlot::try_new(
            &context.device,
            dimensions,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )?;
        pg::checked_gpu(&context.device, || {
            context.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                image.as_raw(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(dimensions[0] * 4),
                    rows_per_image: Some(dimensions[1]),
                },
                pg::extent(dimensions),
            );
            finish_upload(context, current)
        })?;
        let bind_group = pg::checked_gpu(&context.device, || {
            Ok(context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("08 最终预览暂存"),
                    layout: &display.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: display.transform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&display.sampler),
                        },
                    ],
                }))
        })?;
        Ok(Self {
            bind_group,
            dimensions,
            bytes,
        })
    }

    pub(crate) fn publish(self, context: &GpuContext) -> Result<(), String> {
        let mut display = context.display.lock().unwrap_or_else(|e| e.into_inner());
        let display = display.as_mut().ok_or("原生显示已关闭")?;
        let previous = display.current_bind_group.replace(self.bind_group);
        let previous_bytes = display.current_texture_bytes;
        let previous_transform = display.latest_transform;
        display.current_texture_bytes = self.bytes;
        display.latest_transform.image_size = self.dimensions.map(|v| v as f32);
        display.latest_transform.texture_size = self.dimensions.map(|v| v as f32);
        let result = pg::checked_gpu(&context.device, || {
            context.queue.write_buffer(
                &display.transform_buffer,
                0,
                bytemuck::bytes_of(&display.latest_transform),
            );
            display.try_render(&context.device, &context.queue)
        });
        if result.is_err() {
            display.current_bind_group = previous;
            display.current_texture_bytes = previous_bytes;
            display.latest_transform = previous_transform;
            context.queue.write_buffer(
                &display.transform_buffer,
                0,
                bytemuck::bytes_of(&display.latest_transform),
            );
        }
        result
    }
}

pub fn get_or_init_gpu_context(
    state: &tauri::State<AppState>,
    _app_handle: &tauri::AppHandle,
) -> Result<GpuContext, String> {
    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    let app_handle = _app_handle;

    let mut context_lock = match state.gpu_context.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            log::warn!(
                "GPU context lock was poisoned (previous crash). Wiping state to self-heal."
            );
            let mut guard = poisoned.into_inner();
            *guard = None;
            guard
        }
    };
    if let Some(context) = &*context_lock {
        return Ok(context.clone());
    }

    #[allow(unused_mut)]
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();

    // Windows：始终强制 DX12/Vulkan（即使设置/环境写成 gl），避免与 GPUPixel OpenGL 争用。
    #[cfg(target_os = "windows")]
    {
        if std::env::var("WGPU_BACKEND")
            .ok()
            .is_some_and(|v| v.eq_ignore_ascii_case("gl"))
        {
            log::warn!("忽略 WGPU_BACKEND=gl，Windows 预览改用 DX12/Vulkan");
            // SAFETY: 仅在启动后、单线程设置路径清除冲突的后端环境变量。
            unsafe {
                std::env::remove_var("WGPU_BACKEND");
            }
        }
        instance_desc.backends = wgpu::Backends::DX12 | wgpu::Backends::VULKAN;
    }

    let flag_path = state.gpu_crash_flag_path.lock().unwrap().clone();
    if let Some(p) = &flag_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(p, "initializing_gpu");
    }

    let instance = wgpu::Instance::new(instance_desc);

    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    let surface_opt = {
        let settings = crate::app_settings::load_settings(app_handle.clone()).unwrap_or_default();
        let use_wgpu_renderer = settings.use_wgpu_renderer.unwrap_or(true);

        if use_wgpu_renderer {
            if let Some(window) = app_handle.get_webview_window("main") {
                match instance.create_surface(window) {
                    Ok(surface) => Some(surface),
                    Err(e) => {
                        log::warn!(
                            "Failed to create surface, falling back to compute-only: {}",
                            e
                        );
                        if let Some(p) = &flag_path {
                            let _ = std::fs::remove_file(p);
                        }
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    #[cfg(any(target_os = "android", target_os = "linux"))]
    let surface_opt: Option<wgpu::Surface> = None;

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: surface_opt.as_ref(),
        ..Default::default()
    }))
    .map_err(|e| {
        if let Some(p) = &flag_path {
            let _ = std::fs::remove_file(p);
        }
        format!("Failed to find a wgpu adapter: {}", e)
    })?;

    let mut required_features = wgpu::Features::empty();
    if adapter
        .features()
        .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
    {
        required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    }

    let limits = adapter.limits();
    let adapter_info = adapter.get_info();
    log::info!(
        "[gpu] adapter={:?} backend={:?} max_sampled_textures={}",
        adapter_info.name,
        adapter_info.backend,
        limits.max_sampled_textures_per_shader_stage
    );
    // Windows 上 wgpu GLES 与 GPUPixel 的 OpenGL 争用会死锁并毒化预览锁。
    #[cfg(target_os = "windows")]
    if adapter_info.backend == wgpu::Backend::Gl {
        if let Some(p) = &flag_path {
            let _ = std::fs::remove_file(p);
        }
        return Err(
            "Windows 预览后端为 OpenGL/GLES，与 GPUPixel 冲突。请勿设置 WGPU_BACKEND=gl，应使用 DX12/Vulkan"
                .into(),
        );
    }
    if limits.max_sampled_textures_per_shader_stage < 11 {
        return Err(format!(
            "GPU 采样纹理上限过低（{} < 11），无法创建调色管线",
            limits.max_sampled_textures_per_shader_stage
        ));
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Processing Device"),
        required_features,
        required_limits: limits.clone(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| {
        if let Some(p) = &flag_path {
            let _ = std::fs::remove_file(p);
        }
        e.to_string()
    })?;

    device.on_uncaptured_error(Arc::new(|err: wgpu::Error| {
        log::error!("[wgpu-error] {}", err);
    }));

    if let Some(p) = &flag_path {
        let _ = std::fs::remove_file(p);
    }

    #[cfg(not(any(target_os = "android", target_os = "linux")))]
    let display_opt = if let Some(surface) = surface_opt {
        let window = app_handle
            .get_webview_window("main")
            .ok_or("Failed to get main window")?;

        let swapchain_caps = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(swapchain_caps.formats[0]);

        let alpha_mode = if cfg!(target_os = "windows")
            && swapchain_caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::Opaque)
        {
            wgpu::CompositeAlphaMode::Opaque
        } else if swapchain_caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else if swapchain_caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
        {
            wgpu::CompositeAlphaMode::PostMultiplied
        } else {
            swapchain_caps.alpha_modes[0]
        };

        let size = window
            .inner_size()
            .unwrap_or(tauri::PhysicalSize::new(1280, 720));
        let config = wgpu::SurfaceConfiguration {
            width: size.width.max(1),
            height: size.height.max(1),
            format: swapchain_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Display Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Display BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    count: None,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Display Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Display Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: NonZero::new(0),
            cache: None,
        });

        let transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transform Buffer"),
            size: std::mem::size_of::<DisplayTransform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Display Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Some(WgpuDisplay {
            surface,
            config,
            pipeline,
            bind_group_layout,
            transform_buffer,
            latest_transform: DisplayTransform {
                rect: [0.0, 0.0, 100.0, 100.0],
                clip: [0.0, 0.0, 10000.0, 10000.0],
                window: [1280.0, 720.0],
                image_size: [100.0, 100.0],
                texture_size: [100.0, 100.0],
                pixelated: 0.0,
                _pad: 0.0,
                bg_primary: [24.0 / 255.0, 24.0 / 255.0, 24.0 / 255.0, 1.0],
                bg_secondary: [35.0 / 255.0, 35.0 / 255.0, 35.0 / 255.0, 1.0],
            },
            sampler,
            current_bind_group: None,
            current_texture_bytes: 0,
        })
    } else {
        None
    };

    #[cfg(any(target_os = "android", target_os = "linux"))]
    let display_opt = None;

    let new_context = GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        limits,
        display: Arc::new(std::sync::Mutex::new(display_opt)),
    };
    *context_lock = Some(new_context.clone());
    Ok(new_context)
}

pub(crate) fn read_texture_data_roi(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, String> {
    read_texture_data_roi_typed(device, queue, texture, origin, size, bytes_per_pixel)
        .map_err(Into::into)
}

fn read_texture_data_roi_typed(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
    bytes_per_pixel: u32,
) -> Result<Vec<u8>, pg::GpuFailure> {
    if size.width == 0
        || size.height == 0
        || size.depth_or_array_layers != 1
        || !matches!(bytes_per_pixel, 4 | 8 | 16)
    {
        return Err("GPU 回读尺寸或格式非法".into());
    }
    let unpadded_bytes_per_row = bytes_per_pixel
        .checked_mul(size.width)
        .ok_or("GPU 回读行宽溢出")?;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row
        .checked_add(align - 1)
        .ok_or("GPU 行对齐溢出")?
        & !(align - 1);
    let output_buffer_size = u64::from(padded_bytes_per_row) * u64::from(size.height);
    if output_buffer_size > device.limits().max_buffer_size {
        return Err("GPU 回读超过设备缓冲区限额".into());
    }
    let output_buffer = pg::checked_gpu_typed(device, || {
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }))
    })?;

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );

    queue.submit(Some(encoder.finish()));
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        })
        .map_err(|e| format!("Failed while polling mapped GPU buffer: {}", e))?;
    let map_result = rx
        .recv()
        .map_err(|e| format!("Failed receiving GPU map result: {}", e))?;
    map_result.map_err(|e| e.to_string())?;

    let length = usize::try_from(u64::from(unpadded_bytes_per_row) * u64::from(size.height))
        .map_err(|_| "GPU 回读 CPU 长度超限")?;
    let mut data = Vec::new();
    data.try_reserve_exact(length)
        .map_err(|_| "GPU 回读 CPU 内存不足")?;
    {
        let mapped = buffer_slice.get_mapped_range();
        for row in mapped.chunks_exact(padded_bytes_per_row as usize) {
            data.extend_from_slice(&row[..unpadded_bytes_per_row as usize]);
        }
    }
    output_buffer.unmap();
    if data.len() != length {
        return Err("GPU 回读映射长度不符".into());
    }
    Ok(data)
}

fn to_rgba_f16(img: &DynamicImage) -> Vec<f16> {
    let rgba_f32 = img.to_rgba32f();
    rgba_f32.into_raw().into_iter().map(f16::from_f32).collect()
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    radius: u32,
    tile_offset_x: u32,
    tile_offset_y: u32,
    input_width: u32,
    input_height: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

const GF_SHORT_EDGE: u32 = 1080;
const GF_CLARITY_RADIUS: f32 = 8.0;
const GF_STRUCTURE_RADIUS: f32 = 40.0;
const GF_CLARITY_EPS: f32 = 0.25;
const GF_STRUCTURE_EPS: f32 = 0.5;
const GF_COARSE_DIV: u32 = 4;
const GF_TONAL_RADIUS: f32 = 10.0;
const GF_TONAL_EPS: f32 = 0.25;

/// Guided-filter working resolution (short edge ≤ GF_SHORT_EDGE). Full-res upload
/// would exceed the 256MB soft budget on native Prepare paths (e.g. 6240×4160 RAW).
fn gf_guide_dims(width: u32, height: u32) -> (u32, u32) {
    let short = width.min(height).max(1);
    let k = (GF_SHORT_EDGE as f32 / short as f32).min(1.0);
    (
        ((width as f32 * k).round() as u32).max(1),
        ((height as f32 * k).round() as u32).max(1),
    )
}

/// Box-average RGB to match `gf_downsample` when the guide is already at working size.
fn gf_guide_sample(base: &image::Rgba32FImage, lw: u32, lh: u32, x: u32, y: u32) -> [f32; 4] {
    let [src_w, src_h] = [base.width(), base.height()];
    if src_w == lw && src_h == lh {
        return base.get_pixel(x, y).0;
    }
    let ratio = [src_w as f32 / lw as f32, src_h as f32 / lh as f32];
    let taps = [
        ratio[0].ceil().clamp(1.0, 8.0) as i32,
        ratio[1].ceil().clamp(1.0, 8.0) as i32,
    ];
    let origin = [x as f32 * ratio[0], y as f32 * ratio[1]];
    let max_idx = [
        src_w.saturating_sub(1) as i32,
        src_h.saturating_sub(1) as i32,
    ];
    let mut sum = [0.0f32; 4];
    let mut count = 0.0f32;
    for j in 0..taps[1] {
        for i in 0..taps[0] {
            let px = (origin[0] + (i as f32 + 0.5) / taps[0] as f32 * ratio[0])
                .floor()
                .clamp(0.0, max_idx[0] as f32) as u32;
            let py = (origin[1] + (j as f32 + 0.5) / taps[1] as f32 * ratio[1])
                .floor()
                .clamp(0.0, max_idx[1] as f32) as u32;
            let p = base.get_pixel(px, py).0;
            for k in 0..4 {
                sum[k] += p[k];
            }
            count += 1.0;
        }
    }
    sum.map(|v| v / count)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GfParams {
    radius: u32,
    is_raw: u32,
    eps: f32,
    _pad: f32,
}

struct GuidedFilterGpu {
    bgl: wgpu::BindGroupLayout,
    downsample: wgpu::ComputePipeline,
    box_down: wgpu::ComputePipeline,
    blur_h: wgpu::ComputePipeline,
    blur_v: wgpu::ComputePipeline,
    coeffs: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
}

impl GuidedFilterGpu {
    fn new(device: &wgpu::Device) -> Self {
        let gf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Guided Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/guided.wgsl").into()),
        });
        let gf_texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Guided Filter BGL"),
            entries: &[
                gf_texture_entry(0),
                gf_texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Guided Filter Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &gf_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            bgl,
            downsample: pipeline("gf_downsample"),
            box_down: pipeline("gf_box_down"),
            blur_h: pipeline("gf_blur_h"),
            blur_v: pipeline("gf_blur_v"),
            coeffs: pipeline("gf_coeffs"),
            pack: pipeline("gf_pack"),
        }
    }

    fn build_coeffs(
        &self,
        context: &GpuContext,
        input_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        is_raw: u32,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ) {
        let device = &context.device;
        let queue = &context.queue;
        let short = width.min(height).max(1);
        let k = (GF_SHORT_EDGE as f32 / short as f32).min(1.0);
        let lw = ((width as f32 * k).round() as u32).max(1);
        let lh = ((height as f32 * k).round() as u32).max(1);
        let low_short = lw.min(lh) as f32;
        let cw = lw.div_ceil(GF_COARSE_DIV).max(1);
        let ch = lh.div_ceil(GF_COARSE_DIV).max(1);

        let make_tex = |label: &'static str, w: u32, h: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        };
        let make_params = |base_radius: f32, eps: f32, div: f32| {
            let radius = (base_radius * low_short / 1080.0 / div).ceil().max(1.0) as u32;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Guided Filter Params"),
                contents: bytemuck::bytes_of(&GfParams {
                    radius,
                    is_raw,
                    eps,
                    _pad: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };

        let moments_tex = make_tex("GF Moments", lw, lh);
        let moments = moments_tex.create_view(&Default::default());
        let tmp_tex = make_tex("GF Ping Pong", lw, lh);
        let tmp = tmp_tex.create_view(&Default::default());
        let blurred_tex = make_tex("GF Blurred", lw, lh);
        let blurred = blurred_tex.create_view(&Default::default());
        let ab_tex = make_tex("GF Coeffs", lw, lh);
        let ab = ab_tex.create_view(&Default::default());
        let ab_clarity_tex = make_tex("GF Coeffs Clarity", lw, lh);
        let ab_clarity = ab_clarity_tex.create_view(&Default::default());
        let c_moments_tex = make_tex("GF Coarse Moments", cw, ch);
        let c_moments = c_moments_tex.create_view(&Default::default());
        let c_tmp_tex = make_tex("GF Coarse Ping Pong", cw, ch);
        let c_tmp = c_tmp_tex.create_view(&Default::default());
        let c_blurred_tex = make_tex("GF Coarse Blurred", cw, ch);
        let c_blurred = c_blurred_tex.create_view(&Default::default());
        let c_ab_tex = make_tex("GF Coarse Coeffs", cw, ch);
        let c_ab = c_ab_tex.create_view(&Default::default());

        let params_clarity = make_params(GF_CLARITY_RADIUS, GF_CLARITY_EPS, 1.0);
        let params_tonal = make_params(GF_TONAL_RADIUS, GF_TONAL_EPS, 1.0);
        let params_coarse =
            make_params(GF_STRUCTURE_RADIUS, GF_STRUCTURE_EPS, GF_COARSE_DIV as f32);

        let mut encoder = device.create_command_encoder(&Default::default());
        let mut pass = |pipeline: &wgpu::ComputePipeline,
                        src_a: &wgpu::TextureView,
                        src_b: &wgpu::TextureView,
                        dst: &wgpu::TextureView,
                        params: &wgpu::Buffer,
                        (w, h): (u32, u32)| {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Guided Filter BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_a),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src_b),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(dst),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        };
        let fine = (lw, lh);
        let coarse = (cw, ch);

        pass(
            &self.downsample,
            input_view,
            input_view,
            &moments,
            &params_clarity,
            fine,
        );
        pass(
            &self.box_down,
            &moments,
            &moments,
            &c_moments,
            &params_coarse,
            coarse,
        );
        pass(
            &self.blur_h,
            &c_moments,
            &c_moments,
            &c_tmp,
            &params_coarse,
            coarse,
        );
        pass(
            &self.blur_v,
            &c_tmp,
            &c_tmp,
            &c_blurred,
            &params_coarse,
            coarse,
        );
        pass(
            &self.coeffs,
            &c_blurred,
            &c_blurred,
            &c_ab,
            &params_coarse,
            coarse,
        );
        pass(&self.blur_h, &c_ab, &c_ab, &c_tmp, &params_coarse, coarse);
        // Reuse c_moments as coarse output storage (matches upstream).
        pass(
            &self.blur_v,
            &c_tmp,
            &c_tmp,
            &c_moments,
            &params_coarse,
            coarse,
        );

        for (params, out) in [(&params_clarity, &ab_clarity), (&params_tonal, &blurred)] {
            pass(&self.blur_h, &moments, &moments, &tmp, params, fine);
            pass(&self.blur_v, &tmp, &tmp, &blurred, params, fine);
            pass(&self.coeffs, &blurred, &blurred, &ab, params, fine);
            pass(&self.blur_h, &ab, &ab, &tmp, params, fine);
            pass(&self.blur_v, &tmp, &tmp, out, params, fine);
        }

        pass(
            &self.pack,
            &ab_clarity,
            &blurred,
            &ab,
            &params_clarity,
            fine,
        );
        queue.submit(Some(encoder.finish()));

        // Keep owning textures for coeffs (ab) and dehaze/structure (c_moments).
        let coeffs_view = ab_tex.create_view(&Default::default());
        let dehaze_view = c_moments_tex.create_view(&Default::default());
        (ab_tex, coeffs_view, c_moments_tex, dehaze_view)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FlareParams {
    amount: f32,
    is_raw: u32,
    exposure: f32,
    brightness: f32,
    contrast: f32,
    whites: f32,
    aspect_ratio: f32,
    _pad: f32,
}

pub struct GpuProcessor {
    context: GpuContext,
    blur_bgl: wgpu::BindGroupLayout,
    h_blur_pipeline: wgpu::ComputePipeline,
    v_blur_pipeline: wgpu::ComputePipeline,
    blur_params_buffer: wgpu::Buffer,

    flare_bgl_0: wgpu::BindGroupLayout,
    flare_bgl_1: wgpu::BindGroupLayout,
    flare_threshold_pipeline: wgpu::ComputePipeline,
    flare_ghosts_pipeline: wgpu::ComputePipeline,
    flare_params_buffer: wgpu::Buffer,
    flare_threshold_view: wgpu::TextureView,
    flare_ghosts_view: wgpu::TextureView,
    flare_final_view: wgpu::TextureView,
    flare_sampler: wgpu::Sampler,

    guided: GuidedFilterGpu,

    main_bgl: wgpu::BindGroupLayout,
    main_pipeline: wgpu::ComputePipeline,
    high_precision_bgl: wgpu::BindGroupLayout,
    high_precision_pipeline: wgpu::ComputePipeline,
    high_precision_tile: std::sync::OnceLock<HighPrecisionTile>,
    reshape_gpu: Option<crate::portrait_gpu::ReshapeGpu>,
    reshape_layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    tile_output_size: wgpu::Extent3d,
    adjustments_buffer: wgpu::Buffer,
    portrait_texture: std::sync::Mutex<Option<(String, wgpu::TextureView)>>,
    dummy_blur_view: wgpu::TextureView,
    dummy_portrait_view: wgpu::TextureView,
    dummy_lut_view: wgpu::TextureView,
    dummy_lut_sampler: wgpu::Sampler,
    ping_pong_view: wgpu::TextureView,
    sharpness_blur_view: wgpu::TextureView,
    tonal_blur_view: wgpu::TextureView,
    clarity_blur_view: wgpu::TextureView,
    structure_blur_view: wgpu::TextureView,

    pub tile_output_texture: wgpu::Texture,
    pub tile_output_texture_view: wgpu::TextureView,
    pub working_texture: wgpu::Texture,
    pub working_texture_view: wgpu::TextureView,
    pub output_texture: wgpu::Texture,
    pub output_texture_view: wgpu::TextureView,
}

struct HighPrecisionTile {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct FloatBaseTarget<'a> {
    stage: &'a crate::portrait_gpu::Stage,
    slot: &'a crate::portrait_gpu::TextureSlot,
}

/// 07 内部入口的源／代次绑定。调用方仍须使用 04 的源签名和代次守卫。
pub(crate) struct FixedReshapeRequest<'a> {
    pub scene: Option<&'a crate::portrait_safety::SafeScene>,
    pub expected_analysis_key: &'a str,
    pub generation: &'a str,
    pub profile: crate::portrait_gpu::Profile,
}

// 07 GPU 验证通路的输出；生产调度不读取，字段由 portrait_gpu_integration_tests 消费。
// 仅生产编译豁免 dead_code，测试编译保留检测，避免将来真实退化被掩盖。
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct FixedReshapeOutput {
    pub image: DynamicImage,
    pub field_precision: Option<crate::portrait_gpu_field::FieldPrecision>,
    pub quantization_error: f64,
    pub resource_budget: u64,
}

enum RenderedPixels {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

fn high_precision_shader_source() -> String {
    include_str!("shaders/shader.wgsl").replace("rgba8unorm, write>", "rgba16float, write>")
}

#[cfg(test)]
#[path = "portrait_gpu_integration_tests.rs"]
mod portrait_gpu_integration_tests;

/// 08：与 shader.wgsl／blur.wgsl 共用的 64 字节窗口布局。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CompositionWindow {
    pub full_dimensions: [u32; 2],
    pub input_origin: [i32; 2],
    pub output_origin: [i32; 2],
    pub source_min: [i32; 2],
    pub source_max: [i32; 2],
    pub mask_origin: [i32; 2],
    pub radius: u32,
    pub direction: u32,
    pub reserved: [u32; 2],
}
const _: () = assert!(std::mem::size_of::<CompositionWindow>() == 64);
impl CompositionWindow {
    fn new(
        full: [u32; 2],
        input: PixelRect,
        output: PixelRect,
        source: PixelRect,
    ) -> Result<Self, String> {
        crate::portrait_tiles::check_dimensions(full)?;
        if !source.contains(output) {
            return Err("运算窗口不在真实源域内".into());
        }
        Ok(Self {
            full_dimensions: full,
            input_origin: [input.x, input.y],
            output_origin: [output.x, output.y],
            source_min: [source.x, source.y],
            source_max: [
                i32::try_from(source.right()).map_err(|_| "源域溢出")?,
                i32::try_from(source.bottom()).map_err(|_| "源域溢出")?,
            ],
            mask_origin: [output.x, output.y],
            radius: 0,
            direction: 0,
            reserved: [0; 2],
        })
    }
}

/// 辅助资源由同一原生基底准备；蒙版是当前合成窗口切片，其他场为全局共享资源。
pub(crate) struct ColorTileResources<'a> {
    pub masks: &'a wgpu::Texture,
    pub lut: &'a wgpu::Texture,
    pub lut_sampler: &'a wgpu::Sampler,
    pub flare: &'a pg::TextureSlot,
    pub flare_sampler: &'a wgpu::Sampler,
    pub portrait: &'a pg::TextureSlot,
    pub gf_coeffs: &'a pg::TextureSlot,
    pub gf_dehaze: &'a pg::TextureSlot,
    /// 其他活跃资源（尤其旧显示纹理）仍须计入，不能隐式排除旧工作池。
    pub other_resident_bytes: u64,
}
impl ColorTileResources<'_> {
    fn bytes(&self) -> Result<u64, String> {
        let textures = [
            self.masks,
            self.lut,
            &self.flare.texture,
            &self.portrait.texture,
            &self.gf_coeffs.texture,
            &self.gf_dehaze.texture,
        ];
        textures
            .into_iter()
            .try_fold(self.other_resident_bytes, |sum, texture| {
                let bpp = match texture.format() {
                    wgpu::TextureFormat::R8Unorm => 1,
                    wgpu::TextureFormat::Rgba16Float => 8,
                    wgpu::TextureFormat::Rgba32Float => 16,
                    _ => return Err("有界调色资源格式未计账".into()),
                };
                let (bytes, _) = crate::portrait_tiles::texture_bytes(
                    [
                        texture.width(),
                        texture.height(),
                        texture.depth_or_array_layers(),
                    ],
                    bpp,
                )?;
                sum.checked_add(bytes)
                    .ok_or_else(|| "辅助资源预算溢出".into())
            })
    }
}

/// 不创建旧完整输入／工作池；只分配当前合成窗及串行 blur 必需的中间行。
pub(crate) struct WindowedColorGpu {
    base: pg::Stage,
    blur: pg::Stage,
    tail: pg::Stage,
    status: wgpu::Buffer,
    profile: pg::Profile,
}
impl WindowedColorGpu {
    pub(crate) fn new(context: &GpuContext, profile: pg::Profile) -> Result<Self, String> {
        pg::checked_gpu(&context.device, || {
            // 仅声明 shader 使用的绑定；人像九场合并为 binding 12 的 D2Array。
            // 布局顺序：0..=12 后接 18..=21，故 binding b≤12 的下标即 b，18..=21 为 13..=16。
            let mut entries: Vec<_> = (0..=12).chain(18..=21).map(pg::texture_entry).collect();
            entries[1] = pg::storage_entry(1, profile.format());
            entries[2] = pg::buffer_entry(
                2,
                wgpu::BufferBindingType::Storage { read_only: true },
                std::mem::size_of::<AllAdjustments>() as u64,
            );
            entries[3].ty = wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            };
            entries[4].ty = wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            };
            entries[5].ty = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering);
            entries[10].ty = wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            };
            entries[11].ty = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
            entries[12].ty = wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            };
            entries[13] = pg::buffer_entry(18, wgpu::BufferBindingType::Uniform, 64);
            entries[14] =
                pg::buffer_entry(19, wgpu::BufferBindingType::Storage { read_only: false }, 4);
            let source = include_str!("shaders/shader.wgsl").replace(
                "rgba8unorm, write>",
                &format!("{}, write>", profile.wgsl_format()),
            );
            let base = pg::Stage::new(
                &context.device,
                &entries,
                source.clone(),
                "window_main",
                &[("RESHAPE_BASE_OUTPUT", 1.0), ("HIGH_PRECISION_OUTPUT", 1.0)],
            );
            let tail = pg::Stage::new(
                &context.device,
                // entries 布局：0..=12 后接 binding 18/19/20/21（下标 13..=16）
                &[entries[0], entries[1], entries[2], entries[13], entries[14]],
                source,
                "finish_window",
                &[(
                    "HIGH_PRECISION_OUTPUT",
                    if profile == pg::Profile::Preview {
                        0.0
                    } else {
                        1.0
                    },
                )],
            );
            let blur = pg::Stage::new(
                &context.device,
                &[
                    pg::texture_entry(0),
                    pg::storage_entry(1, wgpu::TextureFormat::Rgba16Float),
                    pg::buffer_entry(3, wgpu::BufferBindingType::Uniform, 64),
                    pg::buffer_entry(4, wgpu::BufferBindingType::Storage { read_only: false }, 4),
                ],
                include_str!("shaders/blur.wgsl").into(),
                "window_blur",
                &[],
            );
            let status = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("08 调色安全状态"),
                size: 4,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
            Ok(Self {
                base,
                blur,
                tail,
                status,
                profile,
            })
        })
    }

    fn buffer(&self, context: &GpuContext, window: &CompositionWindow) -> wgpu::Buffer {
        context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("08 调色窗口"),
                contents: bytemuck::bytes_of(window),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }
    fn budget(
        &self,
        context: &GpuContext,
        input: PixelRect,
        output: PixelRect,
        resident: u64,
    ) -> Result<(), pg::GpuFailure> {
        let budget = DeviceBudget {
            bytes: GPU_BUDGET,
            texture_side: context.limits.max_texture_dimension_2d,
            buffer_bytes: context.limits.max_buffer_size,
            resident_bytes: resident,
        };
        if !budget.fits(input, output, self.profile == pg::Profile::HighQuality)? {
            return Err(pg::GpuFailure::OutOfMemory(
                "08 调色资源不足，必须缩小核心块".into(),
            ));
        }
        Ok(())
    }
    fn blur_pass(
        &self,
        context: &GpuContext,
        input: &pg::TextureSlot,
        window: CompositionWindow,
        output: PixelRect,
        current: &dyn Fn() -> bool,
    ) -> Result<pg::TextureSlot, pg::GpuFailure> {
        crate::portrait_analysis::check_current(current)?;
        let result = pg::TextureSlot::try_new(
            &context.device,
            [output.width, output.height],
            wgpu::TextureFormat::Rgba16Float,
            true,
        )?;
        let parameters = self.buffer(context, &window);
        context.queue.write_buffer(&self.status, 0, &[0; 4]);
        pg::dispatch(
            context,
            &self.blur,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&result.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.status.as_entire_binding(),
                },
            ],
            [output.width, output.height],
        );
        pg::check_status(context, &self.status, current)?;
        Ok(result)
    }

    // 07 合成入口：上下文、输入、输入／输出／源窗口、全图、调整与资源句柄各自独立。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compose(
        &self,
        context: &GpuContext,
        input: &pg::TextureSlot,
        input_window: PixelRect,
        output: PixelRect,
        source: PixelRect,
        full: [u32; 2],
        adjustments: &AllAdjustments,
        resources: &ColorTileResources<'_>,
        current: &dyn Fn() -> bool,
    ) -> Result<pg::TextureSlot, pg::GpuFailure> {
        crate::portrait_analysis::check_current(current)?;
        let window = CompositionWindow::new(full, input_window, output, source)?;
        if input.texture.width() != input_window.width
            || input.texture.height() != input_window.height
            || input.texture.format() != self.profile.format()
        {
            return Err("调色输入窗口或精度错误".into());
        }
        if adjustments.mask_count > MAX_MASKS as u32
            || (adjustments.mask_count > 0
                && (resources.masks.width() != output.width
                    || resources.masks.height() != output.height
                    || resources.masks.depth_or_array_layers() < adjustments.mask_count))
        {
            return Err("照片蒙版切片与调整索引不匹配".into());
        }
        let deps = CompositionDependencies::native(
            full,
            true,
            true,
            [
                adjustments.global.chromatic_aberration_red_cyan as f64,
                adjustments.global.chromatic_aberration_blue_yellow as f64,
            ],
        )?;
        let required = deps
            .input(output, full)?
            .intersect(source)
            .ok_or("调色足迹为空")?;
        if !input_window.contains(required) {
            return Err("调色局部输入缺少滤波或色差足迹".into());
        }
        self.budget(context, input_window, output, resources.bytes()?)?;
        pg::checked_gpu_typed(&context.device, || {
            let mut blurs = Vec::with_capacity(4);
            for radius in deps.blur_radii {
                // 横通输出保留纵通需要的行；不得在临时块边缘提前 clamp。
                let extended = output
                    .expand(radius)?
                    .intersect(source)
                    .ok_or("blur 足迹为空")?;
                let horizontal =
                    PixelRect::new(output.x, extended.y, output.width, extended.height)?;
                let mut h = CompositionWindow::new(full, input_window, horizontal, source)?;
                h.radius = radius;
                let ping = self.blur_pass(context, input, h, horizontal, current)?;
                let mut v = CompositionWindow::new(full, horizontal, output, source)?;
                v.radius = radius;
                v.direction = 1;
                blurs.push(self.blur_pass(context, &ping, v, output, current)?);
            }
            let result = pg::TextureSlot::try_new(
                &context.device,
                [output.width, output.height],
                self.profile.format(),
                true,
            )?;
            let parameters = self.buffer(context, &window);
            let a = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("08 原生调色参数"),
                    contents: bytemuck::bytes_of(adjustments),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let mask_view = resources.masks.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            let lut_view = resources.lut.create_view(&Default::default());
            let mut bindings = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&result.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(resources.lut_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&resources.flare.view),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::Sampler(resources.flare_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: self.status.as_entire_binding(),
                },
            ];
            for (i, blur) in blurs.iter().enumerate() {
                bindings.push(wgpu::BindGroupEntry {
                    binding: 6 + i as u32,
                    resource: wgpu::BindingResource::TextureView(&blur.view),
                });
            }
            bindings.push(wgpu::BindGroupEntry {
                binding: PORTRAIT_ATLAS_BINDING,
                resource: wgpu::BindingResource::TextureView(&resources.portrait.view),
            });
            bindings.push(wgpu::BindGroupEntry {
                binding: 20,
                resource: wgpu::BindingResource::TextureView(&resources.gf_coeffs.view),
            });
            bindings.push(wgpu::BindGroupEntry {
                binding: 21,
                resource: wgpu::BindingResource::TextureView(&resources.gf_dehaze.view),
            });
            context.queue.write_buffer(&self.status, 0, &[0; 4]);
            pg::dispatch(
                context,
                &self.base,
                &bindings,
                [output.width, output.height],
            );
            pg::check_status(context, &self.status, current)?;
            Ok(result)
        })
    }

    // 07 尾部拆分入口：上下文、变形纹理、窗口、全图、调整、常驻预算与取消谓词各自独立。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finish(
        &self,
        context: &GpuContext,
        warped: &pg::TextureSlot,
        window: PixelRect,
        full: [u32; 2],
        adjustments: &AllAdjustments,
        resident: u64,
        current: &dyn Fn() -> bool,
    ) -> Result<pg::TextureSlot, pg::GpuFailure> {
        crate::portrait_analysis::check_current(current)?;
        self.budget(context, window, window, resident)?;
        if warped.texture.width() != window.width
            || warped.texture.height() != window.height
            || warped.texture.format() != self.profile.format()
        {
            return Err("重塑尾部窗口错误".into());
        }
        pg::checked_gpu_typed(&context.device, || {
            let result = pg::TextureSlot::try_new(
                &context.device,
                [window.width, window.height],
                self.profile.format(),
                true,
            )?;
            let parameters = self.buffer(
                context,
                &CompositionWindow::new(full, window, window, window)?,
            );
            let a = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("08 原生输出尾部"),
                    contents: bytemuck::bytes_of(adjustments),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            context.queue.write_buffer(&self.status, 0, &[0; 4]);
            pg::dispatch(
                context,
                &self.tail,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&warped.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&result.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 18,
                        resource: parameters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 19,
                        resource: self.status.as_entire_binding(),
                    },
                ],
                [window.width, window.height],
            );
            pg::check_status(context, &self.status, current)?;
            Ok(result)
        })
    }
}

const FLARE_MAP_SIZE: u32 = 512;

/// 等待本轮上传完成后才释放 staging 准入额度。调用者须串行持有照片 GPU 锁。
fn finish_upload(context: &GpuContext, current: &dyn Fn() -> bool) -> Result<(), String> {
    let submission = context.queue.submit(std::iter::empty());
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .map_err(|e| format!("08 上传等待失败：{e}"))?;
    crate::portrait_analysis::check_current(current)
}

fn upload_budget(context: &GpuContext, resident: u64) -> DeviceBudget {
    DeviceBudget {
        bytes: GPU_BUDGET,
        texture_side: context.limits.max_texture_dimension_2d,
        buffer_bytes: context.limits.max_buffer_size,
        resident_bytes: resident,
    }
}

/// 同一上传器用于局部主图和全局辅助场；辅助场的第四通道允许有符号语义。
fn upload_native_rgba(
    context: &GpuContext,
    dims: [u32; 3],
    dimension: wgpu::TextureDimension,
    profile: pg::Profile,
    resident: u64,
    sample: impl Fn(u32, u32, u32) -> [f32; 4],
    current: &dyn Fn() -> bool,
) -> Result<pg::TextureSlot, pg::GpuFailure> {
    crate::portrait_analysis::check_current(current)?;
    let bpp = if profile == pg::Profile::Preview {
        8
    } else {
        16
    };
    if dimension == wgpu::TextureDimension::D3
        && dims
            .iter()
            .any(|v| *v > context.limits.max_texture_dimension_3d)
    {
        return Err("LUT 超过设备三维纹理上限".into());
    }
    if !upload_budget(context, resident).admits_upload(dims, bpp)? {
        return Err("08 纹理上传超过显存或设备限额".into());
    }
    let (bytes, _) = crate::portrait_tiles::texture_bytes(dims, bpp)?;
    let mut data = Vec::new();
    data.try_reserve_exact(usize::try_from(bytes).map_err(|_| "上传长度溢出")?)
        .map_err(|_| "上传临时内存不足")?;
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            crate::portrait_analysis::check_current(current)?;
            for x in 0..dims[0] {
                for v in sample(x, y, z) {
                    if !v.is_finite() || v.abs() > 65504.0 {
                        return Err("08 上传包含非有限或超出浮点范围的值".into());
                    }
                    if bpp == 8 {
                        data.extend_from_slice(&f16::from_f32(v).to_le_bytes());
                    } else {
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }
    pg::checked_gpu_typed(&context.device, || {
        let texture = pg::checked_gpu_typed(&context.device, || {
            Ok(context.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("08 已准入的局部或辅助纹理"),
                size: wgpu::Extent3d {
                    width: dims[0],
                    height: dims[1],
                    depth_or_array_layers: dims[2],
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension,
                format: profile.format(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }))
        })?;
        pg::write_texture(context, &texture, &data, bpp);
        finish_upload(context, current)?;
        let view = if dimension == wgpu::TextureDimension::D2 && dims[2] > 1 {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        } else {
            texture.create_view(&Default::default())
        };
        Ok(pg::TextureSlot { texture, view })
    })
}

/// 与旧整图 FP16 纹理的双线性读法一致：先量化四个源 texel，再做 f32 插值。
/// 512 输入保留 FP32，避免插值后额外量化；两种输出质量共用同一 flare。
fn flare_sample(base: &image::Rgba32FImage, x: u32, y: u32) -> [f32; 4] {
    let [w, h] = [base.width(), base.height()];
    let u = (x as f32 + 0.5) / FLARE_MAP_SIZE as f32 * w as f32 - 0.5;
    let v = (y as f32 + 0.5) / FLARE_MAP_SIZE as f32 * h as f32 - 0.5;
    let [ix, iy] = [u.floor() as i32, v.floor() as i32];
    let [tx, ty] = [u - u.floor(), v - v.floor()];
    let tap = |dx: i32, dy: i32| {
        base.get_pixel(
            (ix + dx).clamp(0, w as i32 - 1) as u32,
            (iy + dy).clamp(0, h as i32 - 1) as u32,
        )
        .0
        .map(|v| f16::from_f32(v).to_f32())
    };
    let [a, b, c, d] = [tap(0, 0), tap(1, 0), tap(0, 1), tap(1, 1)];
    std::array::from_fn(|k| {
        (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty
    })
}

fn prepare_native_flare(
    context: &GpuContext,
    base: &image::Rgba32FImage,
    a: &AllAdjustments,
    resident: u64,
    current: &dyn Fn() -> bool,
) -> Result<pg::TextureSlot, String> {
    if a.global.flare_amount == 0.0 {
        return upload_native_rgba(
            context,
            [1, 1, 1],
            wgpu::TextureDimension::D2,
            pg::Profile::Preview,
            resident,
            |_, _, _| [0.0; 4],
            current,
        )
        .map_err(Into::into);
    }
    if !a.global.flare_amount.is_finite() || a.global.flare_amount < 0.0 {
        return Err("flare 强度非法".into());
    }
    let dims = [FLARE_MAP_SIZE; 2];
    // FP32 输入、两张 FP16 输出及上传阶段；状态／参数保留 1 MiB 余量。
    let transient = u64::from(FLARE_MAP_SIZE).pow(2) * 32;
    if resident
        .checked_add(transient)
        .and_then(|v| v.checked_add(1024 * 1024))
        .is_none_or(|v| v > GPU_BUDGET)
    {
        return Err("全局 flare 准备超过显存预算".into());
    }
    let input = upload_native_rgba(
        context,
        [dims[0], dims[1], 1],
        wgpu::TextureDimension::D2,
        pg::Profile::HighQuality,
        resident,
        |x, y, _| flare_sample(base, x, y),
        current,
    )?;
    pg::checked_gpu(&context.device, || {
        // 仅此变体将 ghosts 的 group1 展平；保留原 shader 和旧管线的绑定契约。
        let source = include_str!("shaders/flare.wgsl")
            .replace("@group(1) @binding(0)", "@group(0) @binding(4)")
            .replace("@group(1) @binding(1)", "@group(0) @binding(5)");
        let params_entry = pg::buffer_entry(2, wgpu::BufferBindingType::Uniform, 32);
        let status_entry =
            pg::buffer_entry(6, wgpu::BufferBindingType::Storage { read_only: false }, 4);
        let threshold = pg::Stage::new(
            &context.device,
            &[
                pg::texture_entry(0),
                pg::storage_entry(1, wgpu::TextureFormat::Rgba16Float),
                params_entry,
                status_entry,
            ],
            source.clone(),
            "threshold_prepared",
            &[],
        );
        let ghosts = pg::Stage::new(
            &context.device,
            &[
                params_entry,
                pg::texture_entry(4),
                pg::storage_entry(5, wgpu::TextureFormat::Rgba16Float),
                status_entry,
            ],
            source,
            "ghosts_window",
            &[],
        );
        let parameters = FlareParams {
            amount: a.global.flare_amount,
            is_raw: a.global.is_raw_image,
            exposure: a.global.exposure,
            brightness: a.global.brightness,
            contrast: a.global.contrast,
            whites: a.global.whites,
            aspect_ratio: base.width() as f32 / base.height() as f32,
            _pad: 0.0,
        };
        let parameters = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("08 全局 flare 参数"),
                contents: bytemuck::bytes_of(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let status = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("08 flare 安全状态"),
                contents: &[0; 4],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let bright = pg::TextureSlot::new(
            &context.device,
            dims,
            wgpu::TextureFormat::Rgba16Float,
            true,
        );
        let result = pg::TextureSlot::new(
            &context.device,
            dims,
            wgpu::TextureFormat::Rgba16Float,
            true,
        );
        pg::dispatch(
            context,
            &threshold,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bright.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: status.as_entire_binding(),
                },
            ],
            dims,
        );
        pg::check_status(context, &status, current)?;
        pg::dispatch(
            context,
            &ghosts,
            &[
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&bright.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&result.view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: status.as_entire_binding(),
                },
            ],
            dims,
        );
        pg::check_status(context, &status, current)?;
        Ok(result)
    })
}

/// 交互资源分别按像素输入和 LUT 内容失效；设备切换时整体释放。
#[derive(Default)]
struct NativePreviewResources {
    device: std::sync::Weak<wgpu::Device>,
    raw: Option<(String, pg::TextureSlot)>,
    guide: Option<(String, [pg::TextureSlot; 2])>,
    lut: Option<(String, pg::TextureSlot)>,
    guided_pipeline: Option<GuidedFilterGpu>,
}

static NATIVE_PREVIEW_RESOURCES: std::sync::OnceLock<Mutex<NativePreviewResources>> =
    std::sync::OnceLock::new();

pub(crate) fn clear_native_preview_resources() {
    if let Some(cache) = NATIVE_PREVIEW_RESOURCES.get() {
        let mut cache = cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.raw = None;
        cache.guide = None;
        cache.lut = None;
    }
}

/// 原生照片的一次性全局资源；不做人脸检测，不接受裁剪后的美肤坐标。
pub(crate) struct NativeColorAssets {
    raw: Option<pg::TextureSlot>,
    lut: pg::TextureSlot,
    portrait: pg::TextureSlot,
    flare: pg::TextureSlot,
    gf_coeffs: pg::TextureSlot,
    gf_dehaze: pg::TextureSlot,
    lut_sampler: wgpu::Sampler,
    flare_sampler: wgpu::Sampler,
    // 缓存身份仅由 08 集成测试断言（original.key != changed.key）；生产路径不读取。
    #[cfg_attr(not(test), allow(dead_code))]
    pub key: String,
    pub resident_bytes: u64,
    other_resident_bytes: u64,
}
impl NativeColorAssets {
    // 08 原生辅助资源准备：上下文、基底、键、调整、LUT、美肤场、他人预算与取消谓词各自独立。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        context: &GpuContext,
        base: &image::Rgba32FImage,
        base_key: &str,
        adjustments: &mut AllAdjustments,
        lut: Option<&Lut>,
        maps: Option<&crate::portrait_processing::PortraitRenderMaps>,
        other_resident_bytes: u64,
        current: &dyn Fn() -> bool,
    ) -> Result<Self, String> {
        Self::prepare_cached(
            context,
            base,
            base_key,
            adjustments,
            lut,
            maps,
            other_resident_bytes,
            false,
            current,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_cached(
        context: &GpuContext,
        base: &image::Rgba32FImage,
        base_key: &str,
        adjustments: &mut AllAdjustments,
        lut: Option<&Lut>,
        maps: Option<&crate::portrait_processing::PortraitRenderMaps>,
        other_resident_bytes: u64,
        cache_preview: bool,
        current: &dyn Fn() -> bool,
    ) -> Result<Self, String> {
        crate::portrait_tiles::check_dimensions([base.width(), base.height()])?;
        crate::portrait_analysis::check_current(current)?;
        let mut digest = Sha256::new();
        digest.update(b"native-auxiliary-v1");
        digest.update(base_key.as_bytes());
        let cache_preview = cache_preview && other_resident_bytes < GPU_BUDGET / 2;
        let mut cache = NATIVE_PREVIEW_RESOURCES
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cache
            .device
            .upgrade()
            .is_none_or(|device| !Arc::ptr_eq(&device, &context.device))
        {
            *cache = NativePreviewResources {
                device: Arc::downgrade(&context.device),
                ..Default::default()
            };
        }
        let guide_key = format!(
            "{base_key}:{}x{}:{}",
            base.width(),
            base.height(),
            adjustments.global.is_raw_image
        );
        if !cache_preview
            || cache
                .guide
                .as_ref()
                .is_some_and(|(key, _)| key != &guide_key)
        {
            cache.guide = None;
        }
        if !cache_preview || cache.raw.as_ref().is_some_and(|(key, _)| key != &guide_key) {
            cache.raw = None;
        }
        let lut_key = if adjustments.global.has_lut != 0 {
            let lut = lut.ok_or("活动 LUT 未准备")?;
            let mut hash = Sha256::new();
            hash.update(lut.size.to_le_bytes());
            for values in lut.data.chunks(4096) {
                crate::portrait_analysis::check_current(current)?;
                hash.update(bytemuck::cast_slice(values));
            }
            hex::encode(hash.finalize())
        } else {
            "no-lut".to_owned()
        };
        if !cache_preview || cache.lut.as_ref().is_some_and(|(key, _)| key != &lut_key) {
            cache.lut = None;
        }
        // 复用句柄也占 GPU 内存，先计入准入预算；后续资源计账重复计入只会更保守。
        let retained = cache.guide.as_ref().map_or(0, |(_, fields)| {
            fields
                .iter()
                .map(|field| {
                    u64::from(field.texture.width()) * u64::from(field.texture.height()) * 16
                })
                .sum::<u64>()
        }) + cache.lut.as_ref().map_or(0, |(_, slot)| {
            u64::from(slot.texture.width())
                * u64::from(slot.texture.height())
                * u64::from(slot.texture.depth_or_array_layers())
                * 8
        }) + cache.raw.as_ref().map_or(0, |(_, slot)| {
            u64::from(slot.texture.width()) * u64::from(slot.texture.height()) * 8
        });
        let other_resident_bytes = other_resident_bytes
            .checked_add(retained)
            .ok_or("交互 GPU 缓存预算溢出")?;
        let mut resident = other_resident_bytes;
        // 局部 fn 显式传借用：闭包会持有 &mut resident 且不实现 Drop，需要 drop 才能归还。
        fn add_resident(resident: &mut u64, texture: &wgpu::Texture) -> Result<(), String> {
            let (bytes, _) = crate::portrait_tiles::texture_bytes(
                [
                    texture.width(),
                    texture.height(),
                    texture.depth_or_array_layers(),
                ],
                8,
            )?;
            *resident = resident.checked_add(bytes).ok_or("辅助场预算溢出")?;
            Ok(())
        }
        digest.update(lut_key.as_bytes());
        let lut = if let Some((_, slot)) = &cache.lut {
            clone_texture_slot(slot)
        } else if adjustments.global.has_lut != 0 {
            let lut = lut.ok_or("活动 LUT 未准备，禁止渲染遗漏效果的照片")?;
            let count = u64::from(lut.size)
                .checked_pow(3)
                .and_then(|v| v.checked_mul(3))
                .ok_or("LUT 长度溢出")?;
            if lut.size < 2 || count != lut.data.len() as u64 {
                return Err("LUT 数据长度非法".into());
            }
            upload_native_rgba(
                context,
                [lut.size; 3],
                wgpu::TextureDimension::D3,
                pg::Profile::Preview,
                other_resident_bytes,
                |x, y, z| {
                    let i = ((z * lut.size + y) * lut.size + x) as usize * 3;
                    [lut.data[i], lut.data[i + 1], lut.data[i + 2], 1.0]
                },
                current,
            )?
        } else {
            upload_native_rgba(
                context,
                [1; 3],
                wgpu::TextureDimension::D3,
                pg::Profile::Preview,
                other_resident_bytes,
                |_, _, _| [0.0; 4],
                current,
            )?
        };
        add_resident(&mut resident, &lut.texture)?;
        if cache_preview {
            crate::portrait_analysis::check_current(current)?;
            cache.lut = Some((lut_key, clone_texture_slot(&lut)));
        }
        let active = adjustments.portrait.strengths.iter().any(|v| *v > 0.0)
            || adjustments.portrait.hydrated[0] > 0.0
            || adjustments.portrait.hydrated[1] > 0.0
            || adjustments.portrait.even[0] > 0.0
            || adjustments.portrait.even[1] > 0.0
            || adjustments.portrait.whitening[0] > 0.0
            || adjustments.portrait.whitening[1] > 0.0
            || adjustments.portrait.eyes[0] > 0.0
            || adjustments.portrait.eyes[1] > 0.0
            || adjustments.portrait.blemishes[0] > 0.0
            || adjustments.portrait.blemishes[2] > 0.0;
        let maps = if active {
            Some(maps.ok_or("旧美肤分析尚未就绪")?)
        } else {
            None
        };
        let analysis_key = if let Some(maps) = maps {
            if maps.uv_rect != [0.0, 0.0, 1.0, 1.0]
                || maps.analysis.full_dimensions != base.dimensions()
            {
                return Err("美肤场不是同一未裁剪原生基底".into());
            }
            adjustments.portrait.uv_rect = maps.uv_rect;
            if maps.analysis.summary.face_count == 0 {
                adjustments.portrait.strengths[2] = 0.0;
                adjustments.portrait.strengths[3] = 0.0;
                adjustments.portrait.hydrated[0] = 0.0;
                adjustments.portrait.even[0] = 0.0;
                adjustments.portrait.whitening[0] = 0.0;
                adjustments.portrait.eyes = [0.0; 4];
                adjustments.portrait.blemishes = [0.0; 4];
            }
            digest.update(maps.analysis.summary.key.as_bytes());
            Some(maps.analysis.summary.key.as_str())
        } else {
            None
        };
        let cached_fields = analysis_key.and_then(|key| {
            let guard = NATIVE_PORTRAIT_FIELD_CACHE
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            guard
                .as_ref()
                .filter(|(cached_key, _)| cached_key == key)
                .map(|(_, slot)| clone_texture_slot(slot))
        });
        let fields = if let Some(fields) = cached_fields {
            add_resident(&mut resident, &fields.texture)?;
            fields
        } else {
            let (width, height, layers) = if let Some(maps) = maps {
                let m = &maps.analysis;
                (
                    m.width,
                    m.height,
                    [
                        m.pixels.as_ref(),
                        m.frequency_pixels.as_ref(),
                        m.hydrated_pixels.as_ref(),
                        m.blemish_pixels.as_ref(),
                        m.texture_pixels.as_ref(),
                        m.shine_pixels.as_ref(),
                        m.frequency_detail_pixels.as_ref(),
                        m.even_pixels.as_ref(),
                        m.whitening_pixels.as_ref(),
                        m.evenness_chroma_pixels.as_ref(),
                        m.eye_white_pixels.as_ref(),
                    ],
                )
            } else {
                (1, 1, [&[[0.0; 4]][..]; 11])
            };
            for layer in &layers {
                if u64::from(width) * u64::from(height) != layer.len() as u64 {
                    return Err("美肤场像素长度错误".into());
                }
            }
            crate::portrait_analysis::check_current(current)?;
            let dims = [width, height, PORTRAIT_ATLAS_LAYERS];
            let slot = upload_native_rgba(
                context,
                dims,
                wgpu::TextureDimension::D2,
                pg::Profile::Preview,
                resident,
                |x, y, z| layers[z as usize][(y * width + x) as usize],
                current,
            )?;
            resident = resident
                .checked_add(crate::portrait_tiles::texture_bytes(dims, 8)?.0)
                .ok_or("美肤场预算溢出")?;
            if let Some(key) = analysis_key {
                let mut guard = NATIVE_PORTRAIT_FIELD_CACHE
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                *guard = Some((key.to_owned(), clone_texture_slot(&slot)));
            }
            slot
        };
        let flare = prepare_native_flare(context, base, adjustments, resident, current)?;
        resident = resident
            .checked_add(
                crate::portrait_tiles::texture_bytes(
                    [flare.texture.width(), flare.texture.height(), 1],
                    8,
                )?
                .0,
            )
            .ok_or("flare 预算溢出")?;
        for v in [
            adjustments.global.flare_amount,
            adjustments.global.exposure,
            adjustments.global.brightness,
            adjustments.global.contrast,
            adjustments.global.whites,
        ] {
            digest.update(v.to_le_bytes());
        }
        digest.update(adjustments.global.is_raw_image.to_le_bytes());
        let lut_sampler = context
            .device
            .create_sampler(&wgpu::SamplerDescriptor::default());
        let flare_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Guided-filter guide at working resolution only — full native bases (Prepare /
        // reshape) are often 20MP+ and fail the 256MB admits_upload peak check.
        let [gf_coeffs, gf_dehaze] = if let Some((_, fields)) = &cache.guide {
            log::debug!("[portrait-preview] guided input cache hit");
            std::array::from_fn(|i| clone_texture_slot(&fields[i]))
        } else {
            let guided = cache
                .guided_pipeline
                .get_or_insert_with(|| GuidedFilterGpu::new(&context.device));
            let (gf_w, gf_h) = gf_guide_dims(base.width(), base.height());
            let base_upload = upload_native_rgba(
                context,
                [gf_w, gf_h, 1],
                wgpu::TextureDimension::D2,
                pg::Profile::Preview,
                resident,
                |x, y, _| gf_guide_sample(base, gf_w, gf_h, x, y),
                current,
            )?;
            let (gf_coeffs_tex, _, gf_dehaze_tex, _) = guided.build_coeffs(
                context,
                &base_upload.view,
                gf_w,
                gf_h,
                adjustments.global.is_raw_image,
            );
            drop(base_upload);
            let fields = [
                wrap_owned_texture(gf_coeffs_tex),
                wrap_owned_texture(gf_dehaze_tex),
            ];
            crate::portrait_analysis::check_current(current)?;
            if cache_preview {
                cache.guide = Some((
                    guide_key.clone(),
                    std::array::from_fn(|i| clone_texture_slot(&fields[i])),
                ));
            }
            fields
        };
        for slot in [&gf_coeffs, &gf_dehaze] {
            let (bytes, _) = crate::portrait_tiles::texture_bytes(
                [slot.texture.width(), slot.texture.height(), 1],
                16,
            )?;
            resident = resident
                .checked_add(bytes)
                .ok_or("guided filter 预算溢出")?;
        }

        let raw = if let Some((_, slot)) = &cache.raw {
            log::debug!("[portrait-preview] GPU input cache hit");
            Some(clone_texture_slot(slot))
        } else if cache_preview {
            match upload_native_rgba(
                context,
                [base.width(), base.height(), 1],
                wgpu::TextureDimension::D2,
                pg::Profile::Preview,
                resident,
                |x, y, _| base.get_pixel(x, y).0,
                current,
            ) {
                Ok(slot) => Some(slot),
                Err(error) if error.is_out_of_memory() => None,
                Err(error) => return Err(error.into()),
            }
        } else {
            None
        };
        let mut other_resident_bytes = other_resident_bytes;
        if let Some(slot) = &raw {
            crate::portrait_analysis::check_current(current)?;
            add_resident(&mut resident, &slot.texture)?;
            add_resident(&mut other_resident_bytes, &slot.texture)?;
            cache.raw = Some((guide_key, clone_texture_slot(slot)));
        }
        Ok(Self {
            raw,
            lut,
            portrait: fields,
            flare,
            gf_coeffs,
            gf_dehaze,
            lut_sampler,
            flare_sampler,
            key: hex::encode(digest.finalize()),
            resident_bytes: resident,
            other_resident_bytes,
        })
    }

    pub(crate) fn resources<'a>(&'a self, masks: &'a wgpu::Texture) -> ColorTileResources<'a> {
        ColorTileResources {
            masks,
            lut: &self.lut.texture,
            lut_sampler: &self.lut_sampler,
            flare: &self.flare,
            flare_sampler: &self.flare_sampler,
            portrait: &self.portrait,
            gf_coeffs: &self.gf_coeffs,
            gf_dehaze: &self.gf_dehaze,
            other_resident_bytes: self.other_resident_bytes,
        }
    }
}

/// 不进入旧完整 GPU 工作池的原生照片输入；所有借用均绑定同一源快照。
pub(crate) struct NativePhotoInput<'a> {
    pub base: &'a image::Rgba32FImage,
    pub base_key: &'a str,
    pub field: &'a crate::portrait_tiles::CanonicalField,
    pub adjustments: AllAdjustments,
    pub masks: &'a [crate::mask_generation::MaskDefinition],
    pub mask_cache: &'a crate::portrait_render::NativeMaskCache,
    /// 原尺寸 Alpha 的空间尺寸与独立基底键；调色参数不应使 Alpha 缓存失效。
    pub mask_source: Option<([u32; 2], &'a str)>,
    pub cache_preview_resources: bool,
    /// 颜色／亮度蒙版所用的已补丁、已镜头校正但尚未定向／旋转的源。
    pub warped: Option<&'a DynamicImage>,
    pub lut: Option<&'a Lut>,
    pub portrait: Option<&'a crate::portrait_processing::PortraitRenderMaps>,
    pub other_resident_bytes: u64,
}

fn upload_native_masks(
    context: &GpuContext,
    input: &NativePhotoInput<'_>,
    window: PixelRect,
    resident: u64,
    current: &dyn Fn() -> bool,
) -> Result<wgpu::Texture, pg::GpuFailure> {
    let layers = input.adjustments.mask_count.max(1);
    if layers > context.limits.max_texture_array_layers
        || !upload_budget(context, resident)
            .admits_upload([window.width, window.height, layers], 1)?
    {
        return Err("照片蒙版上传超过设备或显存限额".into());
    }
    let mut data = Vec::new();
    let bytes = crate::portrait_tiles::texture_bytes([window.width, window.height, layers], 1)?.0;
    data.try_reserve_exact(usize::try_from(bytes).map_err(|_| "蒙版上传长度溢出")?)
        .map_err(|_| "蒙版上传内存不足")?;
    for mask in input.masks.iter().filter(|m| m.visible) {
        let dims = [input.base.width(), input.base.height()];
        let (source_dims, mask_key) = input.mask_source.unwrap_or((dims, input.base_key));
        input.mask_cache.append_resampled_slice(
            mask_key,
            mask,
            source_dims,
            dims,
            window,
            input.warped,
            &mut data,
            current,
        )?;
    }
    if input.adjustments.mask_count == 0 {
        data.resize(bytes as usize, 0);
    }
    if data.len() as u64 != bytes {
        return Err("照片蒙版切片索引或长度错误".into());
    }
    pg::checked_gpu_typed(&context.device, || {
        let texture = pg::checked_gpu_typed(&context.device, || {
            Ok(context.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("08 局部照片蒙版 atlas"),
                size: wgpu::Extent3d {
                    width: window.width,
                    height: window.height,
                    depth_or_array_layers: layers,
                },
                dimension: wgpu::TextureDimension::D2,
                mip_level_count: 1,
                sample_count: 1,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }))
        })?;
        pg::write_texture(context, &texture, &data, 1);
        finish_upload(context, current)?;
        Ok(texture)
    })
}

static LEGACY_READBACK_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct TrackedAnalyticsReadback {
    buffer: wgpu::Buffer,
    bytes: u64,
}
impl Drop for TrackedAnalyticsReadback {
    fn drop(&mut self) {
        self.buffer.destroy();
        LEGACY_READBACK_BYTES.fetch_sub(self.bytes, std::sync::atomic::Ordering::SeqCst);
    }
}

fn legacy_gpu_resident_bytes(
    processor: &Option<GpuProcessorState>,
    cache: &Option<GpuImageCache>,
) -> Result<u64, String> {
    let mut bytes = 0u64;
    if let Some(state) = processor {
        // working / output / tile / ping-pong / 四路 blur ≈ 8 张 Rgba16Float。
        let (one, _) = crate::portrait_tiles::texture_bytes([state.width, state.height, 1], 8)?;
        bytes = bytes
            .checked_add(one.checked_mul(8).ok_or("旧 GPU 工作池预算溢出")?)
            .ok_or("旧 GPU 工作池预算溢出")?;
    }
    if let Some(image) = cache {
        let (one, _) = crate::portrait_tiles::texture_bytes([image.width, image.height, 1], 8)?;
        bytes = bytes.checked_add(one).ok_or("旧 GPU 输入缓存预算溢出")?;
    }
    Ok(bytes)
}

/// 与旧照片处理器共用锁；Preview 保留热缓存并计账，HighQuality 才释放整图工作池。
pub(crate) fn render_native_photo_shared(
    app: &tauri::AppHandle,
    state: &tauri::State<AppState>,
    mut input: NativePhotoInput<'_>,
    profile: pg::Profile,
    current: &dyn Fn() -> bool,
) -> Result<image::Rgba32FImage, String> {
    let context = get_or_init_gpu_context(state, app)?;
    let mut processor = state
        .gpu_processor
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut cached = state
        .gpu_image_cache
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    crate::portrait_analysis::check_current(current)?;
    let legacy_bytes = if profile == pg::Profile::HighQuality {
        drop(processor.take());
        drop(cached.take());
        0
    } else {
        legacy_gpu_resident_bytes(&processor, &cached)?
    };
    drop(processor);
    drop(cached);
    finish_upload(&context, current)?;
    let display_bytes = context
        .display
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |d| {
            if d.current_bind_group.is_some() {
                d.current_texture_bytes
            } else {
                0
            }
        });
    input.other_resident_bytes = input
        .other_resident_bytes
        .checked_add(display_bytes)
        .and_then(|v| {
            v.checked_add(LEGACY_READBACK_BYTES.load(std::sync::atomic::Ordering::SeqCst))
        })
        .and_then(|v| v.checked_add(legacy_bytes))
        .ok_or("已有 GPU 资源预算溢出")?;
    render_native_photo(&context, input, profile, current)
}

/// pipeline 含可写安全状态，使用期间串行持锁；按设备及精度隔离，不持有照片像素。
#[derive(Default)]
struct NativePipelineCache {
    device: std::sync::Weak<wgpu::Device>,
    profiles: [Option<(WindowedColorGpu, pg::WindowedReshapeGpu)>; 2],
}

static NATIVE_PIPELINES: std::sync::OnceLock<std::sync::Mutex<NativePipelineCache>> =
    std::sync::OnceLock::new();

/// 返回原生 B 空间的预乘线性参考浮点结果；裁剪、缩放及编码在此之后。
/// 只渲染传入的冻结场，绝不在这里检测或求解。调用者持有 GPU 锁并核验源／发布代次。
pub(crate) fn render_native_photo(
    context: &GpuContext,
    mut input: NativePhotoInput<'_>,
    profile: pg::Profile,
    current: &dyn Fn() -> bool,
) -> Result<image::Rgba32FImage, String> {
    let full = [input.base.width(), input.base.height()];
    let source = PixelRect::full(full)?;
    if input.field.dimensions() != full {
        return Err("冻结场与原生合成基底尺寸不符".into());
    }
    if input.adjustments.mask_count as usize != input.masks.iter().filter(|m| m.visible).count()
        || input.adjustments.mask_count > MAX_MASKS as u32
    {
        return Err("照片蒙版数量与调整索引不一致，禁止静默截断".into());
    }
    let prepare_started = Instant::now();
    let assets = NativeColorAssets::prepare_cached(
        context,
        input.base,
        input.base_key,
        &mut input.adjustments,
        input.lut,
        input.portrait,
        input.other_resident_bytes,
        input.cache_preview_resources
            && profile == pg::Profile::Preview
            && input.field.is_identity()
            && full[0].max(full[1]) <= 1280,
        current,
    )?;
    log::debug!(
        "[portrait-preview] GPU resources {:?} elapsed={:?}",
        full,
        prepare_started.elapsed()
    );
    let mut color_elapsed = std::time::Duration::ZERO;
    let mut readback_elapsed = std::time::Duration::ZERO;
    let dependencies = CompositionDependencies::native(
        full,
        true,
        true,
        [
            input.adjustments.global.chromatic_aberration_red_cyan as f64,
            input.adjustments.global.chromatic_aberration_blue_yellow as f64,
        ],
    )?;
    let composition_window = |tile| -> Result<PixelRect, String> {
        input
            .field
            .inverse_footprint(
                tile,
                if profile == pg::Profile::HighQuality {
                    2
                } else {
                    1
                },
                current,
            )?
            .intersect(source)
            .ok_or_else(|| "合成逆足迹为空".into())
    };
    let tile_resources = |tile| -> Result<(PixelRect, PixelRect, u64, u64), String> {
        crate::portrait_analysis::check_current(current)?;
        let composition = composition_window(tile)?;
        let raw = dependencies
            .input(composition, full)?
            .intersect(source)
            .ok_or("调色足迹为空")?;
        let (mask_bytes, upload_bytes) = crate::portrait_tiles::texture_bytes(
            [
                composition.width,
                composition.height,
                input.adjustments.mask_count.max(1),
            ],
            1,
        )?;
        Ok((
            raw,
            composition,
            mask_bytes.checked_add(upload_bytes).ok_or("蒙版预算溢出")?,
            upload_bytes,
        ))
    };
    // 足迹闭包作为参数传入而非捕获：按值传给 impl Fn 会移动闭包、使 plan 退化为 FnOnce，
    // 而 OOM 分块重试会对子窗口再次调用 plan；&dyn 转换保留 Fn 语义且借用是必需的。
    let plan = |window: PixelRect,
                footprint: &dyn Fn(
        PixelRect,
    ) -> Result<(PixelRect, PixelRect, u64, u64), String>| {
        crate::portrait_tiles::plan_tiles_with_resources(
            window,
            upload_budget(context, assets.resident_bytes),
            profile == pg::Profile::HighQuality,
            footprint,
        )
    };
    let mut pending = plan(source, &tile_resources)?;
    pending.reverse();
    let mut pipelines = NATIVE_PIPELINES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if pipelines
        .device
        .upgrade()
        .is_none_or(|device| !Arc::ptr_eq(&device, &context.device))
    {
        *pipelines = NativePipelineCache {
            device: Arc::downgrade(&context.device),
            profiles: Default::default(),
        };
    }
    let slot = if profile == pg::Profile::Preview {
        0
    } else {
        1
    };
    if pipelines.profiles[slot].is_none() {
        pipelines.profiles[slot] = Some((
            WindowedColorGpu::new(context, profile)?,
            pg::WindowedReshapeGpu::new(context, profile)?,
        ));
    }
    let (color, reshape) = pipelines.profiles[slot]
        .as_ref()
        .ok_or("原生调色 pipeline 未就绪")?;
    crate::portrait_analysis::check_current(current)?;
    let len = (full[0] as usize)
        .checked_mul(full[1] as usize)
        .and_then(|v| v.checked_mul(4))
        .ok_or("原生 CPU 输出长度溢出")?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(len)
        .map_err(|_| "原生 CPU 浮点输出内存不足")?;
    pixels.resize(len, 0.0f32);
    let bpp = if profile == pg::Profile::Preview {
        8
    } else {
        16
    };
    while let Some((core, raw_window)) = pending.pop() {
        crate::portrait_analysis::check_current(current)?;
        // 本次尝试拥有所有局部资源；错误返回时先 Drop，再等待释放并重试子块。
        let attempt = (|| -> Result<Vec<u8>, pg::GpuFailure> {
            let composition = composition_window(core)?;
            let masks =
                upload_native_masks(context, &input, composition, assets.resident_bytes, current)?;
            let mask_bytes = crate::portrait_tiles::texture_bytes(
                [masks.width(), masks.height(), masks.depth_or_array_layers()],
                1,
            )?
            .0;
            let resident = assets
                .resident_bytes
                .checked_add(mask_bytes)
                .ok_or("活动资源溢出")?;
            let raw = if raw_window == source
                && let Some(raw) = &assets.raw
            {
                clone_texture_slot(raw)
            } else {
                upload_native_rgba(
                    context,
                    [raw_window.width, raw_window.height, 1],
                    wgpu::TextureDimension::D2,
                    profile,
                    resident,
                    |x, y, _| {
                        input
                            .base
                            .get_pixel(raw_window.x as u32 + x, raw_window.y as u32 + y)
                            .0
                    },
                    current,
                )?
            };
            let color_started = Instant::now();
            let composed = color.compose(
                context,
                &raw,
                raw_window,
                composition,
                source,
                full,
                &input.adjustments,
                &assets.resources(&masks),
                current,
            )?;
            drop(raw);
            drop(masks);
            let warped = reshape.render_tile(
                context,
                &composed,
                composition,
                core,
                source,
                input.field,
                assets.resident_bytes,
                current,
            )?;
            drop(composed);
            let finished = color.finish(
                context,
                &warped,
                core,
                full,
                &input.adjustments,
                assets.resident_bytes,
                current,
            )?;
            drop(warped);
            color_elapsed += color_started.elapsed();
            let readback_started = Instant::now();
            let readback = pg::checked_gpu_typed(&context.device, || {
                read_texture_data_roi_typed(
                    &context.device,
                    &context.queue,
                    &finished.texture,
                    wgpu::Origin3d::ZERO,
                    pg::extent([core.width, core.height]),
                    bpp,
                )
            });
            readback_elapsed += readback_started.elapsed();
            readback
        })();
        crate::portrait_analysis::check_current(current)?;
        let bytes = match attempt {
            Ok(bytes) => bytes,
            Err(error) if error.is_out_of_memory() => {
                // 仅 OOM 进入这里。验证／安全／设备等待失败直接终止，不改变场或精度。
                pg::checked_gpu(&context.device, || finish_upload(context, current))?;
                let children = crate::portrait_tiles::split_core(core)
                    .map_err(|limit| format!("{error}；{limit}"))?;
                for child in children.into_iter().rev() {
                    let replacements = plan(child, &tile_resources)?;
                    pending
                        .try_reserve(replacements.len())
                        .map_err(|_| "分块重试索引内存不足")?;
                    pending.extend(replacements.into_iter().rev());
                }
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let expected = core.width as usize * core.height as usize * bpp as usize;
        if bytes.len() != expected {
            return Err("原生核心块回读长度错误".into());
        }
        for y in 0..core.height as usize {
            crate::portrait_analysis::check_current(current)?;
            for x in 0..core.width as usize {
                let offset = (y * core.width as usize + x) * bpp as usize;
                let rgba: [f32; 4] = std::array::from_fn(|c| {
                    let i = offset + c * (bpp as usize / 4);
                    if bpp == 8 {
                        f16::from_le_bytes([bytes[i], bytes[i + 1]]).to_f32()
                    } else {
                        f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap())
                    }
                });
                if rgba.iter().any(|v| !v.is_finite())
                    || !(0.0..=1.0).contains(&rgba[3])
                    || (rgba[3] == 0.0 && rgba[..3].iter().any(|v| *v != 0.0))
                {
                    return Err("原生核心块浮点或透明度非法".into());
                }
                let index = ((core.y as usize + y) * full[0] as usize + core.x as usize + x) * 4;
                pixels[index..index + 4].copy_from_slice(&rgba);
            }
        }
    }
    crate::portrait_analysis::check_current(current)?;
    log::debug!(
        "[portrait-preview] GPU color={:?} readback={:?}",
        color_elapsed,
        readback_elapsed
    );
    image::Rgba32FImage::from_raw(full[0], full[1], pixels)
        .ok_or_else(|| "原生 CPU 结果长度错误".into())
}

impl GpuProcessor {
    pub fn new(context: GpuContext, max_width: u32, max_height: u32) -> Result<Self, String> {
        let device = context.device.clone();
        pg::checked_gpu(&device, || {
            Self::new_unchecked(context, max_width, max_height)
        })
    }

    fn new_unchecked(context: GpuContext, max_width: u32, max_height: u32) -> Result<Self, String> {
        let device = &context.device;
        const MAX_MASK_BINDINGS: u32 = 1;

        let blur_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });

        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blur BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[Some(&blur_bgl)],
            immediate_size: 0,
        });

        let h_blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Horizontal Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_shader_module,
            entry_point: Some("horizontal_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        let v_blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vertical Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_shader_module,
            entry_point: Some("vertical_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blur_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur Params Buffer"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flare Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flare.wgsl").into()),
        });

        let flare_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let flare_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare BGL 1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let flare_threshold_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Flare Threshold Layout"),
                bind_group_layouts: &[Some(&flare_bgl_0)],
                immediate_size: 0,
            });

        let flare_ghosts_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flare Ghosts Layout"),
            bind_group_layouts: &[Some(&flare_bgl_0), Some(&flare_bgl_1)],
            immediate_size: 0,
        });

        let flare_threshold_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flare Threshold Pipeline"),
                layout: Some(&flare_threshold_layout),
                module: &flare_shader,
                entry_point: Some("threshold_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let flare_ghosts_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flare Ghosts Pipeline"),
                layout: Some(&flare_ghosts_layout),
                module: &flare_shader,
                entry_point: Some("ghosts_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let flare_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flare Params Buffer"),
            size: std::mem::size_of::<FlareParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flare_tex_desc = wgpu::TextureDescriptor {
            label: Some("Flare Tex"),
            size: wgpu::Extent3d {
                width: FLARE_MAP_SIZE,
                height: FLARE_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let flare_threshold_texture = device.create_texture(&flare_tex_desc);
        let flare_threshold_view = flare_threshold_texture.create_view(&Default::default());
        let flare_ghosts_texture = device.create_texture(&flare_tex_desc);
        let flare_ghosts_view = flare_ghosts_texture.create_view(&Default::default());
        let flare_final_texture = device.create_texture(&flare_tex_desc);
        let flare_final_view = flare_final_texture.create_view(&Default::default());

        let flare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Flare Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let guided = GuidedFilterGpu::new(device);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image Processing Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        let mut bind_group_layout_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 10 + MAX_MASK_BINDINGS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: PORTRAIT_ATLAS_BINDING,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        // Guided filter textures (shader bindings 20/21).
        for binding in [20, 21] {
            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }

        let main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main BGL"),
            entries: &bind_group_layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&main_bgl)],
            immediate_size: 0,
        });

        let main_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        bind_group_layout_entries[1].ty = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let high_precision_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("High Precision Main BGL"),
                entries: &bind_group_layout_entries,
            });
        let high_precision_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("High Precision Pipeline Layout"),
                bind_group_layouts: &[Some(&high_precision_bgl)],
                immediate_size: 0,
            });
        let high_precision_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("High Precision Image Processing Shader"),
            source: wgpu::ShaderSource::Wgsl(high_precision_shader_source().into()),
        });
        let high_precision_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("High Precision Compute Pipeline"),
                layout: Some(&high_precision_pipeline_layout),
                module: &high_precision_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("HIGH_PRECISION_OUTPUT", 1.0)],
                    ..Default::default()
                },
                cache: None,
            });

        let adjustments_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjustments Buffer"),
            size: std::mem::size_of::<AllAdjustments>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_texture_desc = wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let dummy_blur_texture = device.create_texture(&dummy_texture_desc);
        let dummy_blur_view = dummy_blur_texture.create_view(&Default::default());

        let dummy_portrait_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Portrait Atlas"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: PORTRAIT_ATLAS_LAYERS,
            },
            ..dummy_texture_desc
        });
        let dummy_portrait_view =
            dummy_portrait_texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        let dummy_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            dimension: wgpu::TextureDimension::D3,
            ..dummy_texture_desc
        });
        let dummy_lut_view = dummy_lut_texture.create_view(&Default::default());
        let dummy_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        const TILE_SIZE: u32 = 2048;
        const TILE_OVERLAP: u32 = 128;

        let clamped_tile_width = max_width.min(TILE_SIZE + TILE_OVERLAP * 2);
        let clamped_tile_height = max_height.min(TILE_SIZE + TILE_OVERLAP * 2);

        let clamped_tile_size = wgpu::Extent3d {
            width: clamped_tile_width,
            height: clamped_tile_height,
            depth_or_array_layers: 1,
        };

        let full_image_size = wgpu::Extent3d {
            width: max_width,
            height: max_height,
            depth_or_array_layers: 1,
        };

        let reusable_texture_desc = wgpu::TextureDescriptor {
            label: None,
            size: clamped_tile_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let ping_pong_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Ping Pong Texture"),
            ..reusable_texture_desc
        });
        let ping_pong_view = ping_pong_texture.create_view(&Default::default());

        let sharpness_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Sharpness Blur Texture"),
            ..reusable_texture_desc
        });
        let sharpness_blur_view = sharpness_blur_texture.create_view(&Default::default());

        let tonal_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tonal Blur Texture"),
            ..reusable_texture_desc
        });
        let tonal_blur_view = tonal_blur_texture.create_view(&Default::default());

        let clarity_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Clarity Blur Texture"),
            ..reusable_texture_desc
        });
        let clarity_blur_view = clarity_blur_texture.create_view(&Default::default());

        let structure_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Structure Blur Texture"),
            ..reusable_texture_desc
        });
        let structure_blur_view = structure_blur_texture.create_view(&Default::default());

        let tile_output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Tile Output Texture"),
            size: clamped_tile_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let tile_output_texture_view = tile_output_texture.create_view(&Default::default());

        let working_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Working Output Texture"),
            size: full_image_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let working_texture_view = working_texture.create_view(&Default::default());

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Full Output Texture"),
            size: full_image_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_texture_view = output_texture.create_view(&Default::default());

        Ok(Self {
            context,
            blur_bgl,
            h_blur_pipeline,
            v_blur_pipeline,
            blur_params_buffer,
            flare_bgl_0,
            flare_bgl_1,
            flare_threshold_pipeline,
            flare_ghosts_pipeline,
            flare_params_buffer,
            flare_threshold_view,
            flare_ghosts_view,
            flare_final_view,
            flare_sampler,
            guided,
            main_bgl,
            main_pipeline,
            high_precision_bgl,
            high_precision_pipeline,
            high_precision_tile: std::sync::OnceLock::new(),
            reshape_gpu: None,
            reshape_layout_entries: bind_group_layout_entries,
            tile_output_size: clamped_tile_size,
            adjustments_buffer,
            portrait_texture: std::sync::Mutex::new(None),
            dummy_blur_view,
            dummy_portrait_view,
            dummy_lut_view,
            dummy_lut_sampler,
            ping_pong_view,
            sharpness_blur_view,
            tonal_blur_view,
            clarity_blur_view,
            structure_blur_view,
            tile_output_texture,
            tile_output_texture_view,
            working_texture,
            working_texture_view,
            output_texture,
            output_texture_view,
        })
    }

    fn high_precision_tile(&self) -> &HighPrecisionTile {
        self.high_precision_tile.get_or_init(|| {
            let texture = self
                .context
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("High Precision Tile Output Texture"),
                    size: self.tile_output_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
            let view = texture.create_view(&Default::default());
            HighPrecisionTile { texture, view }
        })
    }

    /// 固定完整基底验证，不是 08 的生产预览／导出调度。&mut self 保证复用资源串行。
    #[allow(dead_code)]
    pub(crate) fn render_reshape_fixed(
        &mut self,
        base_image: &DynamicImage,
        request: RenderRequest,
        reshape: FixedReshapeRequest<'_>,
        current: &dyn Fn() -> bool,
    ) -> Result<FixedReshapeOutput, String> {
        use crate::portrait_gpu::{
            Profile, ReshapeGpu, TextureSlot, checked_gpu, estimated_bytes, write_texture,
        };
        use crate::portrait_gpu_field::ValidatedField;
        let dims = [base_image.width(), base_image.height()];
        estimated_bytes(dims)?;
        if !current() {
            return Err("重塑请求已取消".into());
        }
        if request.roi.is_some()
            || dims[0] > self.tile_output_size.width
            || dims[1] > self.tile_output_size.height
            || reshape.generation.is_empty()
            || reshape.generation.len() > 4096
            || request.mask_bitmaps.len() > MAX_MASKS
            || request
                .mask_bitmaps
                .iter()
                .any(|m| [m.width(), m.height()] != dims)
            || request.lut.as_ref().is_some_and(|l| {
                l.size < 2
                    || l.size > 65
                    || l.data.len() != l.size as usize * l.size as usize * l.size as usize * 3
            })
        {
            return Err("07 要求完整基底、有效代次和有界蒙版／LUT 资源".into());
        }
        let field = if let Some(scene) = reshape.scene {
            if scene.base_dimensions() != dims
                || scene.analysis_key() != reshape.expected_analysis_key
            {
                return Err("重塑分析尺寸或代次已过期".into());
            }
            ValidatedField::from_scene(scene, reshape.profile == Profile::Preview, current)?
        } else {
            None
        };
        let Some(field) = field else {
            // 关闭／全零／无脸：释放上一张的场，只调用原 run，不做颜色往返或额外末尾 pass。
            self.reshape_gpu = None;
            let texture = TextureSlot::new(
                &self.context.device,
                dims,
                wgpu::TextureFormat::Rgba16Float,
                false,
            );
            write_texture(
                &self.context,
                &texture.texture,
                bytemuck::cast_slice(&to_rgba_f16(base_image)),
                8,
            );
            let precision = if reshape.profile == Profile::Preview {
                RenderOutputPrecision::EightBit
            } else {
                RenderOutputPrecision::SixteenBit
            };
            let (_gf_tex_a, gf_coeffs_view, _gf_tex_b, gf_dehaze_view) = self.guided.build_coeffs(
                &self.context,
                &texture.view,
                dims[0],
                dims[1],
                request.adjustments.global.is_raw_image,
            );
            let (pixels, w, h, ..) = self.run(
                &texture.view,
                &gf_coeffs_view,
                &gf_dehaze_view,
                dims[0],
                dims[1],
                request,
                false,
                precision,
            )?;
            let image = match pixels {
                RenderedPixels::U8(v) => DynamicImage::ImageRgba8(
                    ImageBuffer::from_raw(w, h, v).ok_or("原路径输出长度错误")?,
                ),
                RenderedPixels::U16(v) => DynamicImage::ImageRgba16(
                    ImageBuffer::from_raw(w, h, v).ok_or("原路径输出长度错误")?,
                ),
            };
            if !current() {
                return Err("重塑请求已取消".into());
            }
            return Ok(FixedReshapeOutput {
                image,
                field_precision: None,
                quantization_error: 0.0,
                resource_budget: 0,
            });
        };
        if self
            .reshape_gpu
            .as_ref()
            .is_some_and(|n| !n.compatible(dims, reshape.profile, field.precision()))
        {
            self.reshape_gpu = None;
        }
        let mut node = match self.reshape_gpu.take() {
            Some(node) => node,
            None => ReshapeGpu::new(
                &self.context,
                dims,
                reshape.profile,
                field.precision(),
                &self.reshape_layout_entries,
            )?,
        };
        let adjustments = request.adjustments;
        let result = checked_gpu(&self.context.device, || {
            node.upload(
                &self.context,
                base_image.to_rgba32f().as_raw(),
                &field,
                reshape.generation,
            )?;
            let (_gf_tex_a, gf_coeffs_view, _gf_tex_b, gf_dehaze_view) = self.guided.build_coeffs(
                &self.context,
                &node.input.view,
                dims[0],
                dims[1],
                request.adjustments.global.is_raw_image,
            );
            self.run_internal(
                &node.input.view,
                &gf_coeffs_view,
                &gf_dehaze_view,
                dims[0],
                dims[1],
                request,
                false,
                RenderOutputPrecision::EightBit,
                Some(FloatBaseTarget {
                    stage: &node.base_stage,
                    slot: &node.base,
                }),
            )?;
            if !current() {
                return Err("重塑请求已取消".into());
            }
            node.warp(&self.context)?;
            let image = node.finish(&self.context, &adjustments)?;
            if !current() {
                return Err("重塑请求已取消".into());
            }
            Ok(FixedReshapeOutput {
                image,
                field_precision: Some(field.precision()),
                quantization_error: field.quantization_error(),
                resource_budget: node.allocated_budget,
            })
        });
        // 失败资源不复用，避免已排队的无效上传留下缓存命中标记。
        if result.is_ok() {
            self.reshape_gpu = Some(node);
        }
        result
    }

    #[cfg(test)]
    pub(crate) fn reshape_layout_entries(&self) -> &[wgpu::BindGroupLayoutEntry] {
        &self.reshape_layout_entries
    }

    fn run_auto_guided(
        &self,
        input_texture_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        request: RenderRequest,
        output_to_display: bool,
        output_precision: RenderOutputPrecision,
    ) -> Result<(RenderedPixels, u32, u32, u32, u32), String> {
        let (_a, coeffs, _b, dehaze) = self.guided.build_coeffs(
            &self.context,
            input_texture_view,
            width,
            height,
            request.adjustments.global.is_raw_image,
        );
        self.run(
            input_texture_view,
            &coeffs,
            &dehaze,
            width,
            height,
            request,
            output_to_display,
            output_precision,
        )
    }

    fn run(
        &self,
        input_texture_view: &wgpu::TextureView,
        gf_coeffs_view: &wgpu::TextureView,
        gf_dehaze_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        request: RenderRequest,
        output_to_display: bool,
        output_precision: RenderOutputPrecision,
    ) -> Result<(RenderedPixels, u32, u32, u32, u32), String> {
        self.run_internal(
            input_texture_view,
            gf_coeffs_view,
            gf_dehaze_view,
            width,
            height,
            request,
            output_to_display,
            output_precision,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_internal(
        &self,
        input_texture_view: &wgpu::TextureView,
        gf_coeffs_view: &wgpu::TextureView,
        gf_dehaze_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        request: RenderRequest,
        output_to_display: bool,
        output_precision: RenderOutputPrecision,
        float_target: Option<FloatBaseTarget<'_>>,
    ) -> Result<(RenderedPixels, u32, u32, u32, u32), String> {
        if let Some(target) = &float_target {
            crate::portrait_gpu_field::check_dimensions([width, height])?;
            if output_to_display
                || request.roi.is_some()
                || target.slot.texture.width() != width
                || target.slot.texture.height() != height
            {
                return Err("浮点基底目标必须与固定完整基底一致".into());
            }
        }
        let device = &self.context.device;
        let queue = &self.context.queue;
        let scale = (width.min(height) as f32) / 1080.0;
        const MAX_MASK_BINDINGS: u32 = 1;

        if output_precision == RenderOutputPrecision::SixteenBit && output_to_display {
            return Err(
                "High-precision GPU output is only supported for CPU-readback renders.".to_string(),
            );
        }

        let high_precision_tile =
            if float_target.is_none() && output_precision == RenderOutputPrecision::SixteenBit {
                Some(self.high_precision_tile())
            } else {
                None
            };
        let (output_pipeline, output_bgl, output_tile_texture, output_tile_view, bytes_per_pixel) =
            if let Some(target) = &float_target {
                (
                    &target.stage.pipeline,
                    &target.stage.layout,
                    &target.slot.texture,
                    &target.slot.view,
                    0,
                )
            } else if let Some(high_precision_tile) = high_precision_tile {
                (
                    &self.high_precision_pipeline,
                    &self.high_precision_bgl,
                    &high_precision_tile.texture,
                    &high_precision_tile.view,
                    8,
                )
            } else {
                (
                    &self.main_pipeline,
                    &self.main_bgl,
                    &self.tile_output_texture,
                    &self.tile_output_texture_view,
                    4,
                )
            };

        let bounds = request.roi.unwrap_or(Roi {
            x: 0,
            y: 0,
            width,
            height,
        });
        let out_width = bounds.width;
        let out_height = bounds.height;
        let mask_layer_count = request.mask_bitmaps.len().clamp(2, MAX_MASKS) as u32;
        let full_texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: mask_layer_count,
        };
        let buffer_size = (width as usize) * (height as usize) * (mask_layer_count as usize);
        let mut mask_texture_data = Vec::with_capacity(buffer_size);
        if request.mask_bitmaps.is_empty() {
            mask_texture_data.resize(buffer_size, 0);
        } else {
            for mask_bitmap in request.mask_bitmaps.iter().take(MAX_MASKS) {
                mask_texture_data.extend_from_slice(mask_bitmap.as_raw());
            }
            if mask_texture_data.len() < buffer_size {
                mask_texture_data.resize(buffer_size, 0);
            }
        }
        let mask_texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Full Mask Texture Array"),
                size: full_texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            &mask_texture_data,
        );
        let mask_texture_view = mask_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let (lut_texture_view, lut_sampler) = if let Some(lut_arc) = &request.lut {
            let lut_data = &lut_arc.data;
            let size = lut_arc.size;
            let mut rgba_lut_data_f16 = Vec::with_capacity(lut_data.len() / 3 * 4);
            for chunk in lut_data.as_chunks::<3>().0 {
                rgba_lut_data_f16.push(f16::from_f32(chunk[0]));
                rgba_lut_data_f16.push(f16::from_f32(chunk[1]));
                rgba_lut_data_f16.push(f16::from_f32(chunk[2]));
                rgba_lut_data_f16.push(f16::ONE);
            }
            let lut_texture = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("LUT 3D Texture"),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                bytemuck::cast_slice(&rgba_lut_data_f16),
            );
            let view = lut_texture.create_view(&Default::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (view, sampler)
        } else {
            (self.dummy_lut_view.clone(), self.dummy_lut_sampler.clone())
        };

        let mut adjustments = request.adjustments;
        let portrait_view = if let Some(maps) = &request.portrait_maps {
            adjustments.portrait.uv_rect = maps.uv_rect;
            let analysis = &maps.analysis;
            let pixel_count = analysis.width as usize * analysis.height as usize;
            if pixel_count == 0
                || analysis.pixels.len() != pixel_count
                || analysis.frequency_pixels.len() != pixel_count
                || analysis.frequency_detail_pixels.len() != pixel_count
                || analysis.hydrated_pixels.len() != pixel_count
                || analysis.even_pixels.len() != pixel_count
                || analysis.whitening_pixels.len() != pixel_count
                || analysis.evenness_chroma_pixels.len() != pixel_count
                || analysis.blemish_pixels.len() != pixel_count
                || analysis.texture_pixels.len() != pixel_count
                || analysis.shine_pixels.len() != pixel_count
                || analysis.eye_white_pixels.len() != pixel_count
            {
                return Err("人像修正场尺寸无效，请重新分析".into());
            }
            if analysis.summary.face_count == 0 {
                adjustments.portrait.strengths[2] = 0.0;
                adjustments.portrait.strengths[3] = 0.0;
                adjustments.portrait.hydrated[0] = 0.0;
                adjustments.portrait.even[0] = 0.0;
                adjustments.portrait.whitening[0] = 0.0;
                adjustments.portrait.eyes = [0.0; 4];
                adjustments.portrait.blemishes = [0.0; 4];
            }
            let mut cached = self
                .portrait_texture
                .lock()
                .map_err(|_| "人像 GPU 缓存锁异常")?;
            if cached
                .as_ref()
                .is_none_or(|(key, _)| key != &analysis.summary.key)
            {
                let layers: [&[[f32; 4]]; 11] = [
                    analysis.pixels.as_ref(),
                    analysis.frequency_pixels.as_ref(),
                    analysis.hydrated_pixels.as_ref(),
                    analysis.blemish_pixels.as_ref(),
                    analysis.texture_pixels.as_ref(),
                    analysis.shine_pixels.as_ref(),
                    analysis.frequency_detail_pixels.as_ref(),
                    analysis.even_pixels.as_ref(),
                    analysis.whitening_pixels.as_ref(),
                    analysis.evenness_chroma_pixels.as_ref(),
                    analysis.eye_white_pixels.as_ref(),
                ];
                let mut data = Vec::with_capacity(pixel_count * 4 * PORTRAIT_ATLAS_LAYERS as usize);
                for pixels in layers {
                    for px in pixels {
                        for &c in px {
                            data.push(f16::from_f32(c));
                        }
                    }
                }
                let texture = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("Portrait Field Atlas"),
                        size: wgpu::Extent3d {
                            width: analysis.width,
                            height: analysis.height,
                            depth_or_array_layers: PORTRAIT_ATLAS_LAYERS,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    TextureDataOrder::MipMajor,
                    bytemuck::cast_slice(&data),
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                });
                *cached = Some((analysis.summary.key.clone(), view));
            }
            cached.as_ref().ok_or("人像 GPU 纹理缺失")?.1.clone()
        } else {
            if adjustments.portrait.strengths.iter().any(|&v| v > 0.0)
                || adjustments.portrait.hydrated[0] > 0.0
                || adjustments.portrait.hydrated[1] > 0.0
                || adjustments.portrait.even[0] > 0.0
                || adjustments.portrait.even[1] > 0.0
                || adjustments.portrait.whitening[0] > 0.0
                || adjustments.portrait.whitening[1] > 0.0
                || adjustments.portrait.eyes[0] > 0.0
                || adjustments.portrait.eyes[1] > 0.0
                || adjustments.portrait.blemishes[0] > 0.0
                || adjustments.portrait.blemishes[2] > 0.0
            {
                return Err("人像分析尚未就绪，不能渲染遗漏效果的照片".into());
            }
            self.dummy_portrait_view.clone()
        };
        if adjustments.global.flare_amount > 0.0 {
            let mut encoder = device.create_command_encoder(&Default::default());

            let aspect_ratio = if height > 0 {
                width as f32 / height as f32
            } else {
                1.0
            };
            let f_params = FlareParams {
                amount: adjustments.global.flare_amount,
                is_raw: adjustments.global.is_raw_image,
                exposure: adjustments.global.exposure,
                brightness: adjustments.global.brightness,
                contrast: adjustments.global.contrast,
                whites: adjustments.global.whites,
                aspect_ratio,
                _pad: 0.0,
            };
            queue.write_buffer(&self.flare_params_buffer, 0, bytemuck::bytes_of(&f_params));

            let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG0"),
                layout: &self.flare_bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.flare_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.flare_threshold_pipeline);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 16, FLARE_MAP_SIZE / 16, 1);
            }

            let bg0_ghosts = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG0 Ghosts"),
                layout: &self.flare_bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_final_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.flare_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                    },
                ],
            });

            let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG1"),
                layout: &self.flare_bgl_1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_ghosts_view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.flare_ghosts_pipeline);
                cpass.set_bind_group(0, &bg0_ghosts, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 16, FLARE_MAP_SIZE / 16, 1);
            }

            queue.submit(Some(encoder.finish()));
        }

        const TILE_SIZE: u32 = 2048;
        const TILE_OVERLAP: u32 = 128;

        let output_len = if output_to_display || float_target.is_some() {
            0
        } else {
            (out_width * out_height * 4) as usize
        };
        let mut final_pixels = match output_precision {
            RenderOutputPrecision::EightBit => RenderedPixels::U8(vec![0u8; output_len]),
            RenderOutputPrecision::SixteenBit => RenderedPixels::U16(vec![0u16; output_len]),
        };

        let start_tile_x = bounds.x / TILE_SIZE;
        let start_tile_y = bounds.y / TILE_SIZE;
        let end_tile_x = (bounds.x + bounds.width).div_ceil(TILE_SIZE);
        let end_tile_y = (bounds.y + bounds.height).div_ceil(TILE_SIZE);

        for tile_y in start_tile_y..end_tile_y {
            for tile_x in start_tile_x..end_tile_x {
                let x_start_unclamped = tile_x * TILE_SIZE;
                let y_start_unclamped = tile_y * TILE_SIZE;

                let x_start = x_start_unclamped.max(bounds.x);
                let y_start = y_start_unclamped.max(bounds.y);
                let x_end = (x_start_unclamped + TILE_SIZE)
                    .min(bounds.x + bounds.width)
                    .min(width);
                let y_end = (y_start_unclamped + TILE_SIZE)
                    .min(bounds.y + bounds.height)
                    .min(height);

                let tile_width = x_end - x_start;
                let tile_height = y_end - y_start;

                let input_x_start = (x_start as i32 - TILE_OVERLAP as i32).max(0) as u32;
                let input_y_start = (y_start as i32 - TILE_OVERLAP as i32).max(0) as u32;
                let input_x_end = (x_end + TILE_OVERLAP).min(width);
                let input_y_end = (y_end + TILE_OVERLAP).min(height);
                let input_width = input_x_end - input_x_start;
                let input_height = input_y_end - input_y_start;

                let input_texture_size = wgpu::Extent3d {
                    width: input_width,
                    height: input_height,
                    depth_or_array_layers: 1,
                };

                let run_blur = |base_radius: f32, output_view: &wgpu::TextureView| -> bool {
                    let radius = (base_radius * scale).ceil().max(1.0) as u32;
                    if radius == 0 {
                        return false;
                    }

                    let params = BlurParams {
                        radius,
                        tile_offset_x: input_x_start,
                        tile_offset_y: input_y_start,
                        input_width,
                        input_height,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    };
                    queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&params));

                    let mut blur_encoder = device.create_command_encoder(&Default::default());

                    let h_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("H-Blur BG"),
                        layout: &self.blur_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(input_texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.ping_pong_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.blur_params_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    {
                        let mut cpass = blur_encoder.begin_compute_pass(&Default::default());
                        cpass.set_pipeline(&self.h_blur_pipeline);
                        cpass.set_bind_group(0, &h_blur_bg, &[]);
                        cpass.dispatch_workgroups(input_width.div_ceil(256), input_height, 1);
                    }

                    let v_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("V-Blur BG"),
                        layout: &self.blur_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.ping_pong_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.blur_params_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    {
                        let mut cpass = blur_encoder.begin_compute_pass(&Default::default());
                        cpass.set_pipeline(&self.v_blur_pipeline);
                        cpass.set_bind_group(0, &v_blur_bg, &[]);
                        cpass.dispatch_workgroups(input_width, input_height.div_ceil(256), 1);
                    }

                    queue.submit(Some(blur_encoder.finish()));
                    true
                };

                let active_masks = adjustments
                    .mask_adjustments
                    .iter()
                    .take(adjustments.mask_count as usize);
                let any_term =
                    |global: f32, pick: &dyn Fn(&MaskAdjustments) -> f32, positive: bool| {
                        let hit = |v: f32| if positive { v > 0.0 } else { v != 0.0 };
                        hit(global) || active_masks.clone().any(|m| hit(pick(m)))
                    };
                let any_neg_dehaze =
                    adjustments.global.dehaze < 0.0 || active_masks.clone().any(|m| m.dehaze < 0.0);
                let need_sharpen = any_term(adjustments.global.sharpness, &|m| m.sharpness, false);
                let need_halation = any_term(
                    adjustments.global.halation_amount,
                    &|m| m.halation_amount,
                    true,
                );
                let need_structure =
                    any_term(adjustments.global.glow_amount, &|m| m.glow_amount, true)
                        || any_neg_dehaze;

                let did_create_sharpness_blur =
                    need_sharpen && run_blur(1.0, &self.sharpness_blur_view);
                let did_create_tonal_blur = need_sharpen && run_blur(3.5, &self.tonal_blur_view);
                let did_create_clarity_blur =
                    need_halation && run_blur(8.0, &self.clarity_blur_view);
                let did_create_structure_blur =
                    need_structure && run_blur(40.0, &self.structure_blur_view);

                let mut main_encoder = device.create_command_encoder(&Default::default());

                let mut tile_adjustments = adjustments;
                tile_adjustments.tile_offset_x = input_x_start;
                tile_adjustments.tile_offset_y = input_y_start;
                queue.write_buffer(
                    &self.adjustments_buffer,
                    0,
                    bytemuck::bytes_of(&tile_adjustments),
                );

                let mut bind_group_entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(output_tile_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.adjustments_buffer.as_entire_binding(),
                    },
                ];
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&mask_texture_view),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 3 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(&lut_texture_view),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 4 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                });

                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 5 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if did_create_sharpness_blur {
                        &self.sharpness_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 6 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if did_create_tonal_blur {
                        &self.tonal_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 7 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if did_create_clarity_blur {
                        &self.clarity_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 8 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if did_create_structure_blur {
                        &self.structure_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });

                let use_flare = adjustments.global.flare_amount > 0.0;
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 9 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::TextureView(if use_flare {
                        &self.flare_ghosts_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 10 + MAX_MASK_BINDINGS,
                    resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                });

                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: PORTRAIT_ATLAS_BINDING,
                    resource: wgpu::BindingResource::TextureView(&portrait_view),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 20,
                    resource: wgpu::BindingResource::TextureView(gf_coeffs_view),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 21,
                    resource: wgpu::BindingResource::TextureView(gf_dehaze_view),
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Tile Bind Group"),
                    layout: output_bgl,
                    entries: &bind_group_entries,
                });

                {
                    let mut compute_pass = main_encoder.begin_compute_pass(&Default::default());
                    compute_pass.set_pipeline(output_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(
                        input_width.div_ceil(8),
                        input_height.div_ceil(8),
                        1,
                    );
                }

                let crop_x_start = x_start - input_x_start;
                let crop_y_start = y_start - input_y_start;

                if output_to_display {
                    main_encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.tile_output_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: crop_x_start,
                                y: crop_y_start,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.working_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: x_start,
                                y: y_start,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: tile_width,
                            height: tile_height,
                            depth_or_array_layers: 1,
                        },
                    );
                }

                queue.submit(Some(main_encoder.finish()));

                if !output_to_display && float_target.is_none() {
                    let processed_tile_data = read_texture_data_roi(
                        device,
                        queue,
                        output_tile_texture,
                        wgpu::Origin3d::ZERO,
                        input_texture_size,
                        bytes_per_pixel,
                    )?;

                    match &mut final_pixels {
                        RenderedPixels::U8(final_pixels) => {
                            for row in 0..tile_height {
                                let final_y = y_start + row - bounds.y;
                                let final_x = x_start - bounds.x;
                                let final_row_offset = (final_y * out_width + final_x) as usize * 4;
                                let source_y = crop_y_start + row;
                                let source_row_offset =
                                    (source_y * input_width + crop_x_start) as usize * 4;
                                let copy_bytes = (tile_width * 4) as usize;

                                final_pixels[final_row_offset..final_row_offset + copy_bytes]
                                    .copy_from_slice(
                                        &processed_tile_data
                                            [source_row_offset..source_row_offset + copy_bytes],
                                    );
                            }
                        }
                        RenderedPixels::U16(final_pixels) => {
                            for row in 0..tile_height {
                                let final_y = y_start + row - bounds.y;
                                let final_x = x_start - bounds.x;
                                let final_row_offset = (final_y * out_width + final_x) as usize * 4;
                                let source_y = crop_y_start + row;
                                let source_pixel_offset =
                                    (source_y * input_width + crop_x_start) as usize;

                                for sample in 0..(tile_width as usize * 4) {
                                    let source_byte_offset = (source_pixel_offset * 4 + sample) * 2;
                                    let bits = u16::from_ne_bytes([
                                        processed_tile_data[source_byte_offset],
                                        processed_tile_data[source_byte_offset + 1],
                                    ]);
                                    let value = f16::from_bits(bits).to_f32();
                                    final_pixels[final_row_offset + sample] = if value.is_finite() {
                                        (value.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16
                                    } else {
                                        0
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok((final_pixels, out_width, out_height, bounds.x, bounds.y))
    }
}

pub fn process_and_get_dynamic_image(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
) -> Result<DynamicImage, String> {
    process_and_get_dynamic_image_inner(
        context,
        state,
        base_image,
        transform_hash,
        request,
        caller_id,
        RenderOutputPrecision::EightBit,
        false,
        None,
    )
}

pub fn process_and_get_dynamic_image_with_precision(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    output_precision: RenderOutputPrecision,
) -> Result<DynamicImage, String> {
    process_and_get_dynamic_image_inner(
        context,
        state,
        base_image,
        transform_hash,
        request,
        caller_id,
        output_precision,
        false,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn process_and_get_dynamic_image_with_analytics(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    output_to_display: bool,
    analytics_config: Option<crate::AnalyticsConfig>,
) -> Result<DynamicImage, String> {
    process_and_get_dynamic_image_inner(
        context,
        state,
        base_image,
        transform_hash,
        request,
        caller_id,
        RenderOutputPrecision::EightBit,
        output_to_display,
        analytics_config,
    )
}

#[allow(clippy::too_many_arguments)]
fn process_and_get_dynamic_image_inner(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    request: RenderRequest,
    caller_id: &str,
    output_precision: RenderOutputPrecision,
    output_to_display: bool,
    analytics_config: Option<crate::AnalyticsConfig>,
) -> Result<DynamicImage, String> {
    let start_time = Instant::now();
    let (width, height) = base_image.dimensions();
    let device = &context.device;
    let queue = &context.queue;

    let max_dim = context.limits.max_texture_dimension_2d;
    if width > max_dim || height > max_dim {
        if request.portrait_maps.is_some() {
            return Err(format!(
                "图像尺寸 {width}×{height} 超过 GPU 上限 {max_dim}，无法完整应用人像效果"
            ));
        }
        log::warn!(
            "Image dimensions ({}x{}) exceed GPU limits ({}). Bypassing GPU processing and returning unprocessed image to prevent a crash. Try upgrading your GPU :)",
            width,
            height,
            max_dim
        );
        return Ok(base_image.clone());
    }

    let mut processor_lock = match state.gpu_processor.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            log::warn!("GPU processor lock was poisoned. Resetting to self-heal.");
            let mut guard = poisoned.into_inner();
            *guard = None;
            guard
        }
    };
    let mut needs_new_processor = false;
    let new_width = (width + 255) & !255;
    let new_height = (height + 255) & !255;

    if let Some(p) = processor_lock.as_ref() {
        if p.width < width || p.height < height {
            needs_new_processor = true;
        }
    } else {
        needs_new_processor = true;
    }

    if needs_new_processor {
        log::info!(
            "Creating new GPU Processor for dimensions up to {}x{}",
            new_width,
            new_height
        );

        let old_processor = processor_lock.take();

        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(500)),
        });

        match GpuProcessor::new(context.clone(), new_width, new_height) {
            Ok(new_processor) => {
                *processor_lock = Some(crate::GpuProcessorState {
                    processor: new_processor,
                    width: new_width,
                    height: new_height,
                });
            }
            Err(err) => {
                *processor_lock = old_processor;
                return Err(err);
            }
        }
    }

    let processor_state = processor_lock.as_ref().unwrap();
    let processor = &processor_state.processor;

    let mut cache_lock = match state.gpu_image_cache.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            log::warn!("GPU image cache lock was poisoned. Resetting to self-heal.");
            let mut guard = poisoned.into_inner();
            *guard = None;
            guard
        }
    };
    let mut needs_new_cache = false;

    let is_raw = request.adjustments.global.is_raw_image;
    if let Some(cache) = &*cache_lock {
        if cache.transform_hash != transform_hash
            || cache.width != width
            || cache.height != height
            || cache.is_raw != is_raw
        {
            needs_new_cache = true;
        }
    } else {
        needs_new_cache = true;
    }

    if needs_new_cache {
        let old_cache = cache_lock.take();
        drop(old_cache);

        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(500)),
        });

        let img_rgba_f16 = to_rgba_f16(base_image);
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Input Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(&img_rgba_f16),
        );
        let texture_view = texture.create_view(&Default::default());
        let (gf_coeffs_texture, gf_coeffs_view, gf_dehaze_texture, gf_dehaze_view) = processor
            .guided
            .build_coeffs(context, &texture_view, width, height, is_raw);

        *cache_lock = Some(GpuImageCache {
            texture,
            texture_view,
            gf_coeffs_texture,
            gf_coeffs_view,
            gf_dehaze_texture,
            gf_dehaze_view,
            is_raw,
            width,
            height,
            transform_hash,
        });
    }

    let cache = cache_lock.as_ref().unwrap();

    let skip_readback = output_to_display;

    let (processed_pixels, out_w, out_h, out_x, out_y) = processor.run(
        &cache.texture_view,
        &cache.gf_coeffs_view,
        &cache.gf_dehaze_view,
        cache.width,
        cache.height,
        request,
        output_to_display,
        output_precision,
    )?;

    let mut final_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Final Passes Encoder"),
    });
    let mut submit_final_encoder = false;

    let mut async_readback_buffer: Option<TrackedAnalyticsReadback> = None;
    let mut async_padded_bpr: u32 = 0;
    let mut async_unpadded_bpr: u32 = 0;

    if analytics_config.is_some() && skip_readback {
        let unpadded_bytes_per_row = 4 * out_w;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);
        let output_buffer_size = (padded_bytes_per_row * out_h) as u64;

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Async Analytics Readback Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        final_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &processor.working_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: out_x,
                    y: out_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(out_h),
                },
            },
            wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
        );

        LEGACY_READBACK_BYTES.fetch_add(output_buffer_size, std::sync::atomic::Ordering::SeqCst);
        async_readback_buffer = Some(TrackedAnalyticsReadback {
            buffer: output_buffer,
            bytes: output_buffer_size,
        });
        async_padded_bpr = padded_bytes_per_row;
        async_unpadded_bpr = unpadded_bytes_per_row;
        submit_final_encoder = true;
    }

    if output_to_display {
        final_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &processor.working_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: out_x,
                    y: out_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &processor.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: out_x,
                    y: out_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
        );
        submit_final_encoder = true;
    }

    if submit_final_encoder {
        queue.submit(Some(final_encoder.finish()));
    }

    if let Some(analytics) = analytics_config {
        if let Some(buffer) = async_readback_buffer {
            let output_buffer = buffer;
            let padded_bytes_per_row: u32 = async_padded_bpr;
            let unpadded_bytes_per_row: u32 = async_unpadded_bpr;
            let device_clone = context.device.clone();

            std::thread::spawn(move || {
                let buffer_slice = output_buffer.buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();

                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });

                if let Err(e) = device_clone.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(60)),
                }) {
                    log::error!("Async analytics readback poll failed: {}", e);
                    return;
                }

                if let Ok(Ok(())) = rx.recv() {
                    let padded_data = buffer_slice.get_mapped_range().to_vec();
                    output_buffer.buffer.unmap();

                    let mut unpadded_data =
                        Vec::with_capacity((unpadded_bytes_per_row * out_h) as usize);
                    if padded_bytes_per_row == unpadded_bytes_per_row {
                        unpadded_data = padded_data;
                    } else {
                        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
                            unpadded_data
                                .extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
                        }
                    }

                    if let Some(img_buf) =
                        ImageBuffer::<Rgba<u8>, _>::from_raw(out_w, out_h, unpadded_data)
                    {
                        let dynamic_img = DynamicImage::ImageRgba8(img_buf);
                        let _ = analytics.sender.send(crate::AnalyticsJob {
                            path: analytics.path,
                            stamp: analytics.stamp,
                            editor_epoch: analytics.editor_epoch,
                            image: std::sync::Arc::new(dynamic_img),
                            compute_waveform: analytics.compute_waveform,
                            active_waveform_channel: analytics.active_waveform_channel,
                            levels_input_histogram: analytics.levels_input_histogram,
                            reuse_final_as_levels_input: analytics.reuse_final_as_levels_input,
                        });
                    }
                }
            });
        } else {
            if let RenderedPixels::U8(pixels) = &processed_pixels {
                let pixels_clone = pixels.clone();
                std::thread::spawn(move || {
                    if let Some(img_buf) =
                        ImageBuffer::<Rgba<u8>, _>::from_raw(out_w, out_h, pixels_clone)
                    {
                        let dynamic_img = DynamicImage::ImageRgba8(img_buf);
                        let _ = analytics.sender.send(crate::AnalyticsJob {
                            path: analytics.path,
                            stamp: analytics.stamp,
                            editor_epoch: analytics.editor_epoch,
                            image: std::sync::Arc::new(dynamic_img),
                            compute_waveform: analytics.compute_waveform,
                            active_waveform_channel: analytics.active_waveform_channel,
                            levels_input_histogram: analytics.levels_input_histogram,
                            reuse_final_as_levels_input: analytics.reuse_final_as_levels_input,
                        });
                    }
                });
            } else {
                log::warn!("Skipping analytics for a high-precision CPU readback");
            }
        }
    }

    if output_to_display
        && let Some(display) = context
            .display
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
    {
        display.latest_transform.image_size = [width as f32, height as f32];
        display.latest_transform.texture_size =
            [processor_state.width as f32, processor_state.height as f32];

        queue.write_buffer(
            &display.transform_buffer,
            0,
            bytemuck::bytes_of(&display.latest_transform),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &display.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: display.transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&processor.output_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&display.sampler),
                },
            ],
            label: None,
        });
        display.current_bind_group = Some(bind_group);
        display.current_texture_bytes = u64::from(processor.output_texture.width())
            * u64::from(processor.output_texture.height())
            * 4;
        display.render(device, queue);
    }

    if skip_readback {
        let duration = start_time.elapsed();
        let fps = 1.0 / duration.as_secs_f64();
        log::info!(
            "[{}] {}x{} native WGPU display updated in {:?} ({:.2} FPS)",
            caller_id,
            width,
            height,
            duration,
            fps
        );
        return Ok(DynamicImage::new_rgba8(0, 0));
    }

    let duration = start_time.elapsed();
    let fps = 1.0 / duration.as_secs_f64();
    log::info!(
        "[{}] {}x{} processed (ROI: {}x{}) on GPU in {:?} ({:.2} FPS)",
        caller_id,
        width,
        height,
        out_w,
        out_h,
        duration,
        fps
    );

    match processed_pixels {
        RenderedPixels::U8(pixels) => {
            let img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(out_w, out_h, pixels)
                .ok_or("Failed to create 8-bit image buffer from GPU data")?;
            Ok(DynamicImage::ImageRgba8(img_buf))
        }
        RenderedPixels::U16(pixels) => {
            let img_buf = ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(out_w, out_h, pixels)
                .ok_or("Failed to create 16-bit image buffer from GPU data")?;
            Ok(DynamicImage::ImageRgba16(img_buf))
        }
    }
}

#[cfg(test)]
mod portrait_gpu_tests {
    use super::*;
    use crate::portrait_processing::{PortraitAnalysis, PortraitSummary};

    #[test]
    #[ignore = "需要本地 GPU 及 beauty_reference_images 输出；导出真实样片并比较 CPU"]
    fn portrait_beauty_reference_export() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let directory = std::env::var_os("RETOUCH_REFERENCE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| root.join(".research/retouch/full-validation"));
        let name = std::env::var("RETOUCH_CASE").unwrap_or_else(|_| "demo_examples_3".into());
        let input = image::open(directory.join(format!("onnx_{name}_00_input.png")))
            .unwrap()
            .to_rgb32f();
        let expected = image::open(directory.join(format!("rust_{name}_08_full.png")))
            .unwrap()
            .to_rgb8();
        let repaired = DynamicImage::ImageRgb8(expected.clone()).to_rgb32f();
        let (w, h) = input.dimensions();
        let linear = image::Rgb32FImage::from_fn(w, h, |x, y| {
            image::Rgb(input.get_pixel(x, y).0.map(|v| {
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            }))
        });
        let delta = crate::portrait_processing::beauty_delta_fields(&repaired, &input, &linear);
        assert!(delta.iter().any(|v| v[3] > 0.0));
        let analysis = Arc::new(PortraitAnalysis {
            summary: PortraitSummary {
                key: "beauty-export".into(),
                face_count: 1,
                skipped_faces: 0,
                has_skin: true,
            },
            width: w,
            height: h,
            full_dimensions: (w, h),
            pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            frequency_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            frequency_detail_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            hydrated_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            even_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            whitening_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            evenness_chroma_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            texture_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            blemish_pixels: Arc::from(delta),
            shine_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
            eye_white_pixels: Arc::from(vec![[0.0; 4]; (w * h) as usize]),
        });
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        eprintln!("AI 样片 GPU：{:?}", adapter.get_info());
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .unwrap();
        let context = GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
            display: Arc::new(std::sync::Mutex::new(None)),
        };
        let processor = GpuProcessor::new(context.clone(), w, h).unwrap();
        let texture = context.device.create_texture_with_data(
            &context.queue,
            &wgpu::TextureDescriptor {
                label: Some("AI 完整美肤样片"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(&to_rgba_f16(&DynamicImage::ImageRgb32F(input))),
        );
        let view = texture.create_view(&Default::default());
        let render = |json: &serde_json::Value, roi, precision| {
            processor
                .run_auto_guided(
                    &view,
                    w,
                    h,
                    RenderRequest {
                        adjustments: crate::image_processing::get_all_adjustments_from_json(
                            json,
                            false,
                            crate::white_balance::WhiteBalance::reference(),
                            None,
                        ),
                        mask_bitmaps: &[],
                        lut: None,
                        roi,
                        portrait_maps: Some(analysis.for_render(&serde_json::json!({}))),
                    },
                    false,
                    precision,
                )
                .unwrap()
                .0
        };
        let ai = serde_json::json!({"portrait":{"aiBlemishRemoval":100,"blemishRemoval":23,"textureRetention":99}});
        for precision in [
            RenderOutputPrecision::EightBit,
            RenderOutputPrecision::SixteenBit,
        ] {
            let started = std::time::Instant::now();
            let full = render(&ai, None, precision);
            eprintln!("AI GPU 全图渲染 {:?}", started.elapsed());
            let (data, scale, suffix): (Vec<u16>, f64, &str) = match full {
                RenderedPixels::U8(v) => {
                    image::RgbaImage::from_raw(w, h, v.clone())
                        .unwrap()
                        .save(directory.join(format!("gpu_{name}_export8.png")))
                        .unwrap();
                    (v.into_iter().map(u16::from).collect(), 1., "8")
                }
                RenderedPixels::U16(v) => {
                    ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(w, h, v.clone())
                        .unwrap()
                        .save(directory.join(format!("gpu_{name}_export16.png")))
                        .unwrap();
                    (v, 257., "16")
                }
            };
            let mut errors: Vec<f64> = data
                .chunks_exact(4)
                .zip(expected.pixels())
                .flat_map(|(a, b)| (0..3).map(move |c| (a[c] as f64 / scale - b[c] as f64).abs()))
                .collect();
            errors.sort_by(f64::total_cmp);
            let mean = errors.iter().sum::<f64>() / errors.len() as f64;
            let p99 = errors[errors.len() * 99 / 100];
            eprintln!(
                "GPU {suffix}bit 对 CPU：mean={mean}, p99={p99}, max={}",
                errors.last().unwrap()
            );
            assert!(mean <= 1. && p99 <= 2.);
            let roi = Roi {
                x: w / 4,
                y: h / 4,
                width: w / 2,
                height: h / 2,
            };
            let region: Vec<u16> = match render(&ai, Some(roi), precision) {
                RenderedPixels::U8(v) => v.into_iter().map(u16::from).collect(),
                RenderedPixels::U16(v) => v,
            };
            for y in 0..h / 2 {
                let start = ((y + h / 4) * w + w / 4) as usize * 4;
                assert_eq!(
                    &region[y as usize * (w / 2) as usize * 4
                        ..(y + 1) as usize * (w / 2) as usize * 4],
                    &data[start..start + (w / 2) as usize * 4]
                );
            }
            let zero = serde_json::json!({});
            let hidden = serde_json::json!({"portrait":{"aiBlemishRemoval":100},"sectionVisibility":{"portraitBlemishes":false}});
            let bytes = |pixels| match pixels {
                RenderedPixels::U8(v) => v.into_iter().map(u16::from).collect::<Vec<_>>(),
                RenderedPixels::U16(v) => v,
            };
            assert_eq!(
                bytes(render(&zero, None, precision)),
                bytes(render(&hidden, None, precision))
            );
            let started = std::time::Instant::now();
            assert_eq!(
                data,
                bytes(render(&ai, None, precision)),
                "隐藏再打开应复用正确的完整美肤纹理"
            );
            eprintln!("AI GPU 缓存命中渲染 {:?}", started.elapsed());
        }
    }

    #[test]
    #[ignore = "需要本地 GPU；验证祛油光跨 2048 像素分块与缩放"]
    fn portrait_shine_tiles() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .unwrap();
        let context = GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
            display: Arc::new(std::sync::Mutex::new(None)),
        };
        let (w, h) = (2112, 16);
        let processor = GpuProcessor::new(context.clone(), w, h).unwrap();
        let input =
            DynamicImage::ImageRgba32F(ImageBuffer::from_pixel(w, h, Rgba([0.2, 0.2, 0.2, 0.75])));
        let texture = context.device.create_texture_with_data(
            &context.queue,
            &wgpu::TextureDescriptor {
                label: Some("祛油光分块回归"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(&to_rgba_f16(&input)),
        );
        let view = texture.create_view(&Default::default());
        let analysis = Arc::new(PortraitAnalysis {
            summary: PortraitSummary {
                key: "shine-tiles".into(),
                face_count: 1,
                skipped_faces: 0,
                has_skin: true,
            },
            width: 64,
            height: 16,
            full_dimensions: (w, h),
            pixels: Arc::from(vec![[0.0, 0.0, 1.0, 0.2]; 64 * 16]),
            frequency_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
            frequency_detail_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
            hydrated_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
            even_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
            whitening_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
            evenness_chroma_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
            blemish_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
            texture_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
            // 合成保色增量（按肤色比例），验证滑块缩放与安全限，非分析烘焙产物。
            shine_pixels: Arc::from(vec![[-0.048, -0.033, -0.024, 1.0]; 64 * 16]),
            eye_white_pixels: Arc::from(vec![[0.0; 4]; 64 * 16]),
        });
        let json = serde_json::json!({"portrait": {"faceShineRemoval": 100}});
        for precision in [
            RenderOutputPrecision::EightBit,
            RenderOutputPrecision::SixteenBit,
        ] {
            let render = |amount, roi| {
                let mut json = json.clone();
                json["portrait"]["faceShineRemoval"] = serde_json::json!(amount);
                let (pixels, ..) = processor
                    .run_auto_guided(
                        &view,
                        w,
                        h,
                        RenderRequest {
                            adjustments: crate::image_processing::get_all_adjustments_from_json(
                                &json,
                                true,
                                crate::white_balance::WhiteBalance::reference(),
                                None,
                            ),
                            mask_bitmaps: &[],
                            lut: None,
                            roi,
                            portrait_maps: Some(analysis.for_render(&json)),
                        },
                        false,
                        precision,
                    )
                    .unwrap();
                match pixels {
                    RenderedPixels::U8(v) => v.into_iter().map(u16::from).collect::<Vec<_>>(),
                    RenderedPixels::U16(v) => v,
                }
            };
            let zero = render(0, None);
            let full = render(100, None);
            assert!(full[0] < zero[0]);
            assert!(
                full.chunks_exact(4).all(|p| p == &full[..4]),
                "缩放和分块不能产生接缝"
            );
            let roi = render(
                100,
                Some(Roi {
                    x: 2032,
                    y: 4,
                    width: 64,
                    height: 8,
                }),
            );
            for y in 0..8 {
                assert_eq!(
                    &roi[y * 64 * 4..(y + 1) * 64 * 4],
                    &full[((y + 4) * w as usize + 2032) * 4..((y + 4) * w as usize + 2096) * 4]
                );
            }
        }
    }

    #[test]
    #[ignore = "需要本地 GPU；验证真实 WGSL、8/16 位和 ROI 读回"]
    fn portrait_pipeline_pixels() {
        assert_eq!(
            std::mem::size_of::<crate::image_processing::PortraitGpuParams>(),
            144
        );
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .unwrap();
        let context = GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
            display: Arc::new(std::sync::Mutex::new(None)),
        };
        let processor = GpuProcessor::new(context.clone(), 64, 48).unwrap();
        let input = DynamicImage::ImageRgba32F(ImageBuffer::from_pixel(
            64,
            48,
            Rgba([0.2, 0.2, 0.2, 0.75]),
        ));
        let texture = context.device.create_texture_with_data(
            &context.queue,
            &wgpu::TextureDescriptor {
                label: Some("人像回归输入"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 48,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(&to_rgba_f16(&input)),
        );
        let view = texture.create_view(&Default::default());
        let analysis = Arc::new(PortraitAnalysis {
            summary: PortraitSummary {
                key: "gpu-test".into(),
                face_count: 1,
                skipped_faces: 0,
                has_skin: true,
            },
            width: 64,
            height: 48,
            full_dimensions: (64, 48),
            shine_pixels: Arc::from(
                (0..64 * 48)
                    .map(|i| {
                        if i / 64 >= 8 && i % 64 < 32 {
                            [-0.048, -0.033, -0.024, 1.0]
                        } else {
                            [0.0; 4]
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            blemish_pixels: Arc::from(
                (0..64 * 48)
                    .map(|i| {
                        if i / 64 >= 8 && i % 64 < 32 {
                            [0.06, 0.03, 0.02, 1.0]
                        } else {
                            [0.0; 4]
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            texture_pixels: Arc::from(
                (0..64 * 48)
                    .map(|i| {
                        if i / 64 >= 8 && i % 64 < 32 {
                            [0.008, -0.005, 0.004, 1.0]
                        } else {
                            [0.0; 4]
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            frequency_pixels: Arc::from(
                (0..64 * 48)
                    .map(|i| {
                        if i / 64 < 8 {
                            [0.0; 4]
                        } else if i % 64 < 32 {
                            [-0.025, 0.012, 0.008, 1.0]
                        } else {
                            [0.025, -0.01, 0.015, -1.0]
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            frequency_detail_pixels: Arc::from(
                (0..64 * 48)
                    .map(|i| {
                        if i / 64 < 8 {
                            [0.0; 4]
                        } else if i % 64 < 32 {
                            [0.02, -0.01, 0.008, 1.0]
                        } else {
                            [-0.015, 0.012, -0.006, -1.0]
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            hydrated_pixels: Arc::from(
                (0..64 * 48)
                    .map(|i| {
                        if i / 64 < 8 {
                            [0.0; 4]
                        } else if i % 64 < 32 {
                            [0.01, 0.008, 0.006, 1.0]
                        } else {
                            [0.006, 0.005, 0.004, -1.0]
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            even_pixels: Arc::from(vec![[0.0; 4]; 64 * 48]),
            whitening_pixels: Arc::from(vec![[0.0; 4]; 64 * 48]),
            evenness_chroma_pixels: Arc::from(vec![[0.0; 4]; 64 * 48]),
            eye_white_pixels: Arc::from(vec![[0.0; 4]; 64 * 48]),
            pixels: Arc::from(
                (0..64 * 48)
                    .map(|i| {
                        if i / 64 < 8 {
                            [0.0, 0.0, 0.0, 0.2]
                        } else {
                            [if i % 64 < 32 { -0.25 } else { 0.25 }, 0.1, 1.0, 0.2]
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
        });
        let render = |json: serde_json::Value,
                      maps: Option<crate::portrait_processing::PortraitRenderMaps>,
                      roi: Option<Roi>,
                      precision|
         -> Result<Vec<u16>, String> {
            let adjustments = crate::image_processing::get_all_adjustments_from_json(
                &json,
                true,
                crate::white_balance::WhiteBalance::reference(),
                None,
            );
            let (pixels, ..) = processor.run_auto_guided(
                &view,
                64,
                48,
                RenderRequest {
                    adjustments,
                    mask_bitmaps: &[],
                    lut: None,
                    roi,
                    portrait_maps: maps,
                },
                false,
                precision,
            )?;
            Ok(match pixels {
                RenderedPixels::U8(v) => v.into_iter().map(u16::from).collect::<Vec<_>>(),
                RenderedPixels::U16(v) => v,
            })
        };
        let json = |amounts: [u32; 4]| {
            serde_json::json!({"portrait": {
                "neutralGrayEvenness": amounts[0], "neutralGrayContour": amounts[1],
                "frequencySeparationFace": amounts[2], "frequencySeparationBody": amounts[3],
            }})
        };
        let run = |amounts, maps: bool, roi, precision| {
            render(
                json(amounts),
                maps.then(|| analysis.for_render(&serde_json::json!({}))),
                roi,
                precision,
            )
            .unwrap()
        };
        for precision in [
            RenderOutputPrecision::EightBit,
            RenderOutputPrecision::SixteenBit,
        ] {
            let zero = run([0; 4], false, None, precision);
            assert_eq!(
                zero,
                run([0; 4], true, None, precision),
                "归零必须逐像素相等"
            );
            let half = run([50, 0, 0, 0], true, None, precision);
            let full = run([100, 0, 0, 0], true, None, precision);
            let dark = (24 * 64 + 16) * 4;
            let light = (24 * 64 + 48) * 4;
            assert!(full[dark] < half[dark] && half[dark] < zero[dark]);
            assert!(full[light] > half[light] && half[light] > zero[light]);
            assert_eq!(&zero[..64 * 7 * 4], &full[..64 * 7 * 4], "肤外保持不变");
            for i in (3..full.len()).step_by(4) {
                assert_eq!(zero[i], full[i], "保留 alpha");
            }
            let roi = run(
                [100, 0, 0, 0],
                true,
                Some(Roi {
                    x: 16,
                    y: 12,
                    width: 30,
                    height: 16,
                }),
                precision,
            );
            for y in 0..16 {
                assert_eq!(
                    &roi[y * 30 * 4..(y + 1) * 30 * 4],
                    &full[((y + 12) * 64 + 16) * 4..((y + 12) * 64 + 46) * 4],
                    "ROI 绝对坐标一致"
                );
            }
            for slot in [2, 3] {
                let mut amounts = [0; 4];
                amounts[slot] = 100;
                assert!(
                    render(json(amounts), None, None, precision)
                        .unwrap_err()
                        .contains("人像分析尚未就绪")
                );
                let full = run(amounts, true, None, precision);
                amounts[slot] = 50;
                let half = run(amounts, true, None, precision);
                let (changed, unchanged) = if slot == 2 {
                    (dark, light)
                } else {
                    (light, dark)
                };
                assert!(
                    full[changed].abs_diff(zero[changed]) > half[changed].abs_diff(zero[changed])
                );
                assert!(half[changed] != zero[changed]);
                assert_eq!(
                    &full[unchanged..unchanged + 4],
                    &zero[unchanged..unchanged + 4],
                    "另一皮肤区域不受影响"
                );
                assert_eq!(
                    &full[..64 * 8 * 4],
                    &zero[..64 * 8 * 4],
                    "保护区不受磨皮影响"
                );
            }
            let face = run([0, 0, 100, 0], true, None, precision);
            let body = run([0, 0, 0, 100], true, None, precision);
            let both = run([0, 0, 100, 100], true, None, precision);
            for i in 0..64 * 48 {
                let separate = if i % 64 < 32 { &face } else { &body };
                assert_eq!(&both[i * 4..i * 4 + 4], &separate[i * 4..i * 4 + 4]);
                assert_eq!(both[i * 4 + 3], zero[i * 4 + 3], "磨皮保留 alpha");
            }
            let combined = run([40, 30, 70, 80], true, None, precision);
            let region = run(
                [40, 30, 70, 80],
                true,
                Some(Roi {
                    x: 16,
                    y: 12,
                    width: 30,
                    height: 16,
                }),
                precision,
            );
            for y in 0..16 {
                assert_eq!(
                    &region[y * 30 * 4..(y + 1) * 30 * 4],
                    &combined[((y + 12) * 64 + 16) * 4..((y + 12) * 64 + 46) * 4]
                );
            }
            let blemishes = |amount, retention| {
                serde_json::json!({"portrait": {
                    "blemishRemoval": amount, "textureRetention": retention,
                }})
            };
            let maps = || Some(analysis.for_render(&serde_json::json!({})));
            let shine = |amount| serde_json::json!({"portrait": {"faceShineRemoval": amount}});
            assert!(
                render(shine(50), None, None, precision)
                    .unwrap_err()
                    .contains("人像分析尚未就绪")
            );
            let shine25 = render(shine(25), maps(), None, precision).unwrap();
            let shine50 = render(shine(50), maps(), None, precision).unwrap();
            let shine100 = render(shine(100), maps(), None, precision).unwrap();
            assert!(
                shine100[dark] < shine50[dark]
                    && shine50[dark] < shine25[dark]
                    && shine25[dark] < zero[dark]
            );
            assert_eq!(&shine100[..64 * 8 * 4], &zero[..64 * 8 * 4]);
            for y in 0..48 {
                assert_eq!(
                    &shine100[(y * 64 + 32) * 4..(y + 1) * 64 * 4],
                    &zero[(y * 64 + 32) * 4..(y + 1) * 64 * 4]
                );
            }
            for i in (3..zero.len()).step_by(4) {
                assert_eq!(shine100[i], zero[i]);
            }
            let mut hidden_shine = shine(100);
            hidden_shine["sectionVisibility"] = serde_json::json!({"portraitBlemishes": false});
            assert_eq!(zero, render(hidden_shine, None, None, precision).unwrap());
            let mut no_skin = shine(100);
            no_skin["sectionVisibility"] = serde_json::json!({"portraitSkin": false});
            assert_eq!(shine100, render(no_skin, maps(), None, precision).unwrap());
            let shine_roi = render(
                shine(100),
                maps(),
                Some(Roi {
                    x: 16,
                    y: 12,
                    width: 30,
                    height: 16,
                }),
                precision,
            )
            .unwrap();
            for y in 0..16 {
                assert_eq!(
                    &shine_roi[y * 30 * 4..(y + 1) * 30 * 4],
                    &shine100[((y + 12) * 64 + 16) * 4..((y + 12) * 64 + 46) * 4]
                );
            }
            let crop = serde_json::json!({"crop": {"x": 16, "y": 12, "width": 32, "height": 24}});
            let cropped_shine = render(
                shine(100),
                Some(analysis.for_render(&crop)),
                None,
                precision,
            )
            .unwrap();
            assert_eq!(&cropped_shine[dark..dark + 4], &shine100[dark..dark + 4]);
            assert_eq!(&cropped_shine[light..light + 4], &zero[light..light + 4]);
            let mut ai = serde_json::json!({"portrait": {"aiBlemishRemoval": 70}});
            let ai_only = render(ai.clone(), maps(), None, precision).unwrap();
            ai["portrait"]["faceShineRemoval"] = serde_json::json!(100);
            let ai_shine = render(ai, maps(), None, precision).unwrap();
            assert!(ai_shine[dark] < ai_only[dark]);
            assert_eq!(&ai_shine[light..light + 4], &ai_only[light..light + 4]);
            assert_eq!(
                zero,
                render(blemishes(0, 100), None, None, precision).unwrap()
            );
            assert!(render(blemishes(50, 50), None, None, precision).is_err());
            let repaired = render(blemishes(100, 0), maps(), None, precision).unwrap();
            let half = render(blemishes(50, 0), maps(), None, precision).unwrap();
            let retained = render(blemishes(100, 100), maps(), None, precision).unwrap();
            assert!(repaired[dark] > half[dark] && half[dark] > zero[dark]);
            // 神经祛痘路径下 textureRetention 恒为 0，保留值不再改变输出。
            assert_eq!(repaired[dark], retained[dark]);
            assert_eq!(&repaired[light..light + 4], &zero[light..light + 4]);
            assert_eq!(&repaired[..64 * 8 * 4], &zero[..64 * 8 * 4]);
            for i in (3..zero.len()).step_by(4) {
                assert_eq!(zero[i], retained[i]);
            }
            let mut hidden_skin = blemishes(100, 0);
            hidden_skin["sectionVisibility"] = serde_json::json!({"portraitSkin": false});
            assert_eq!(
                repaired,
                render(hidden_skin, maps(), None, precision).unwrap()
            );
            let mut hidden_blemishes = blemishes(100, 100);
            hidden_blemishes["sectionVisibility"] = serde_json::json!({"portraitBlemishes": false});
            assert_eq!(
                zero,
                render(hidden_blemishes, None, None, precision).unwrap()
            );
            let mut mixed = json([40, 30, 70, 80]);
            mixed["portrait"]["blemishRemoval"] = serde_json::json!(70);
            mixed["portrait"]["textureRetention"] = serde_json::json!(60);
            mixed["portrait"]["faceShineRemoval"] = serde_json::json!(80);
            let mixed_full = render(mixed.clone(), maps(), None, precision).unwrap();
            let mixed_roi = render(
                mixed,
                maps(),
                Some(Roi {
                    x: 16,
                    y: 12,
                    width: 30,
                    height: 16,
                }),
                precision,
            )
            .unwrap();
            for y in 0..16 {
                assert_eq!(
                    &mixed_roi[y * 30 * 4..(y + 1) * 30 * 4],
                    &mixed_full[((y + 12) * 64 + 16) * 4..((y + 12) * 64 + 46) * 4]
                );
            }
            let mut hidden = json([100; 4]);
            hidden["sectionVisibility"] = serde_json::json!({"portraitSkin": false});
            assert_eq!(zero, render(hidden, None, None, precision).unwrap());
            let no_face = Arc::new(PortraitAnalysis {
                summary: PortraitSummary {
                    key: "no-face".into(),
                    face_count: 0,
                    skipped_faces: 1,
                    has_skin: true,
                },
                width: 64,
                height: 48,
                full_dimensions: (64, 48),
                pixels: analysis.pixels.clone(),
                frequency_pixels: analysis.frequency_pixels.clone(),
                frequency_detail_pixels: analysis.frequency_detail_pixels.clone(),
                hydrated_pixels: analysis.hydrated_pixels.clone(),
                even_pixels: analysis.even_pixels.clone(),
                whitening_pixels: analysis.whitening_pixels.clone(),
                evenness_chroma_pixels: analysis.evenness_chroma_pixels.clone(),
                blemish_pixels: analysis.blemish_pixels.clone(),
                texture_pixels: analysis.texture_pixels.clone(),
                shine_pixels: analysis.shine_pixels.clone(),
                eye_white_pixels: analysis.eye_white_pixels.clone(),
            });
            assert_eq!(
                zero,
                render(
                    shine(100),
                    Some(no_face.for_render(&serde_json::json!({}))),
                    None,
                    precision
                )
                .unwrap()
            );
            let malformed = Arc::new(PortraitAnalysis {
                summary: analysis.summary.clone(),
                width: 64,
                height: 48,
                full_dimensions: (64, 48),
                pixels: analysis.pixels.clone(),
                frequency_pixels: analysis.frequency_pixels.clone(),
                frequency_detail_pixels: analysis.frequency_detail_pixels.clone(),
                hydrated_pixels: analysis.hydrated_pixels.clone(),
                even_pixels: analysis.even_pixels.clone(),
                whitening_pixels: analysis.whitening_pixels.clone(),
                evenness_chroma_pixels: analysis.evenness_chroma_pixels.clone(),
                blemish_pixels: analysis.blemish_pixels.clone(),
                texture_pixels: analysis.texture_pixels.clone(),
                shine_pixels: Arc::from(vec![]),
                eye_white_pixels: analysis.eye_white_pixels.clone(),
            });
            assert!(
                render(
                    shine(100),
                    Some(malformed.for_render(&serde_json::json!({}))),
                    None,
                    precision
                )
                .unwrap_err()
                .contains("修正场尺寸无效")
            );
            assert_eq!(
                zero,
                render(
                    json([0, 0, 100, 100]),
                    Some(no_face.for_render(&serde_json::json!({}))),
                    None,
                    precision
                )
                .unwrap()
            );
            assert_eq!(
                zero,
                render(
                    blemishes(100, 100),
                    Some(no_face.for_render(&serde_json::json!({}))),
                    None,
                    precision
                )
                .unwrap()
            );
            // 整幅渲染裁剪区域的放大图，采样必须落在同一未裁剪坐标系。
            let crop = serde_json::json!({"crop": {"x": 16, "y": 12, "width": 32, "height": 24}});
            let cropped = render(
                json([0, 0, 100, 100]),
                Some(analysis.for_render(&crop)),
                None,
                precision,
            )
            .unwrap();
            assert_eq!(&cropped[dark..dark + 4], &both[dark..dark + 4]);
            assert_eq!(&cropped[light..light + 4], &both[light..light + 4]);
            let cached = processor
                .portrait_texture
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .1
                .clone();
            run([20, 10, 30, 40], true, None, precision);
            render(blemishes(30, 80), maps(), None, precision).unwrap();
            render(shine(75), maps(), None, precision).unwrap();
            assert_eq!(
                processor
                    .portrait_texture
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .1,
                cached,
                "仅强度变化复用 portrait atlas"
            );
        }
        assert_eq!(
            processor
                .portrait_texture
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .0,
            "gpu-test"
        );
    }
}

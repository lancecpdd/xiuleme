//! 07：复用 GpuContext 的有界浮点重塑资源。仅供固定尺寸内部验证，不接 IPC。
#![allow(dead_code)]
use crate::image_processing::{AllAdjustments, GpuContext};
use crate::portrait_gpu_field::{FieldPrecision, ValidatedField, check_dimensions};
use half::f16;
use image::{DynamicImage, ImageBuffer, Rgba};
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Preview,
    HighQuality,
}
impl Profile {
    pub fn format(self) -> wgpu::TextureFormat {
        match self {
            Self::Preview => wgpu::TextureFormat::Rgba16Float,
            Self::HighQuality => wgpu::TextureFormat::Rgba32Float,
        }
    }
    pub(crate) fn wgsl_format(self) -> &'static str {
        match self {
            Self::Preview => "rgba16float",
            Self::HighQuality => "rgba32float",
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub cubic: u32,
    pub reserved: u32,
}
const _: () = assert!(std::mem::size_of::<Params>() == 16);
pub const RESOURCE_BUDGET: u64 = 256 * 1024 * 1024;

pub fn estimated_bytes(dims: [u32; 2]) -> Result<u64, String> {
    check_dimensions(dims)?;
    // 包含本节点纹理、上传临时数组、字段数据、对齐读回及 CPU 输出；不包含既有调色池。
    let [w, h] = dims.map(u64::from);
    let bytes = w * h * 176 + (w + 1) * (h + 1) * 48 + h * 256 + 1024 * 1024;
    if bytes > RESOURCE_BUDGET {
        return Err("重塑附加资源超过 256 MiB 预算".into());
    }
    Ok(bytes)
}

pub struct TextureSlot {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}
impl TextureSlot {
    pub fn new(
        device: &wgpu::Device,
        dims: [u32; 2],
        format: wgpu::TextureFormat,
        storage: bool,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("07 浮点资源"),
            size: extent(dims),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | if storage {
                    wgpu::TextureUsages::STORAGE_BINDING
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }

    /// 分配完成即检查错误，避免把 OOM 无效纹理继续用于 bind／dispatch。
    pub(crate) fn try_new(
        device: &wgpu::Device,
        dims: [u32; 2],
        format: wgpu::TextureFormat,
        storage: bool,
    ) -> Result<Self, GpuFailure> {
        let texture = checked_gpu_typed(device, || {
            Ok(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("08 局部浮点资源"),
                size: extent(dims),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | if storage {
                        wgpu::TextureUsages::STORAGE_BINDING
                    } else {
                        wgpu::TextureUsages::empty()
                    },
                view_formats: &[],
            }))
        })?;
        checked_gpu_typed(device, || {
            let view = texture.create_view(&Default::default());
            Ok(Self { texture, view })
        })
    }
}
pub fn extent([width, height]: [u32; 2]) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}
pub fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
pub fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}
pub(crate) fn buffer_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size),
        },
        count: None,
    }
}
pub struct Stage {
    pub layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
}
impl Stage {
    pub(crate) fn new(
        device: &wgpu::Device,
        entries: &[wgpu::BindGroupLayoutEntry],
        source: String,
        entry: &str,
        constants: &[(&str, f64)],
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("07 布局"),
            entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("07 管线布局"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("07 着色器"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("07 浮点管线"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        });
        Self { layout, pipeline }
    }
}
/// 只有 wgpu 明确报告且没有伴随安全／验证失败的 OOM 才能缩块重试。
#[derive(Debug)]
pub(crate) enum GpuFailure {
    OutOfMemory(String),
    Failed(String),
}
impl std::fmt::Display for GpuFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory(message) | Self::Failed(message) => f.write_str(message),
        }
    }
}
impl From<String> for GpuFailure {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}
impl From<&str> for GpuFailure {
    fn from(message: &str) -> Self {
        Self::Failed(message.to_owned())
    }
}
impl From<GpuFailure> for String {
    fn from(error: GpuFailure) -> Self {
        error.to_string()
    }
}
impl GpuFailure {
    pub(crate) fn is_out_of_memory(&self) -> bool {
        matches!(self, Self::OutOfMemory(_))
    }
}

pub(crate) fn resolve_gpu_result<T>(
    result: Result<T, GpuFailure>,
    errors: impl IntoIterator<Item = wgpu::Error>,
) -> Result<T, GpuFailure> {
    let mut allocation = None;
    let mut fatal = None;
    for error in errors {
        let message = format!("GPU 能力或执行错误：{error}");
        match error {
            wgpu::Error::OutOfMemory { .. } => allocation = Some(message),
            _ => fatal = Some(message),
        }
    }
    if let Some(message) = fatal {
        return Err(GpuFailure::Failed(message));
    }
    // 安全、缺足迹、取消和等待失败不能被同时发生的 OOM 掩盖。
    let value = result?;
    if let Some(message) = allocation {
        return Err(GpuFailure::OutOfMemory(message));
    }
    Ok(value)
}

pub(crate) fn checked_gpu_typed<T>(
    device: &wgpu::Device,
    run: impl FnOnce() -> Result<T, GpuFailure>,
) -> Result<T, GpuFailure> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = run();
    let errors = [
        pollster::block_on(validation.pop()),
        pollster::block_on(internal.pop()),
        pollster::block_on(oom.pop()),
    ];
    resolve_gpu_result(result, errors.into_iter().flatten())
}

/// wgpu 29 使用作用域 guard；格式／WGSL／内存失败转为明确错误，不自动回退到 8 位。
pub fn checked_gpu<T>(
    device: &wgpu::Device,
    run: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    checked_gpu_typed(device, || run().map_err(Into::into)).map_err(Into::into)
}

/// 08 规范窗口参数，顺序与 WGSL TileParams 严格一致。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileParams {
    pub full_dimensions: [u32; 2],
    pub output_dimensions: [u32; 2],
    pub input_origin: [i32; 2],
    pub field_origin: [i32; 2],
    pub output_origin: [i32; 2],
    pub source_min: [i32; 2],
    pub source_max: [i32; 2],
    pub cubic: u32,
    pub reserved: u32,
}
const _: () = assert!(std::mem::size_of::<TileParams>() == 64);

/// 只持有管线与 4 字节状态；图像、规范场上传和读回均以局部窗口为单位。
pub struct WindowedReshapeGpu {
    stage: Stage,
    status: wgpu::Buffer,
    profile: Profile,
}
impl WindowedReshapeGpu {
    pub fn new(context: &GpuContext, profile: Profile) -> Result<Self, String> {
        checked_gpu(&context.device, || {
            let source = include_str!("shaders/portrait_reshape.wgsl").replace(
                "rgba16float, write>",
                &format!("{}, write>", profile.wgsl_format()),
            );
            let stage = Stage::new(
                &context.device,
                &[
                    texture_entry(0),
                    texture_entry(1),
                    storage_entry(2, profile.format()),
                    buffer_entry(4, wgpu::BufferBindingType::Storage { read_only: false }, 4),
                    buffer_entry(5, wgpu::BufferBindingType::Uniform, 64),
                ],
                source,
                "tile_main",
                &[],
            );
            let status = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("08 分块安全状态"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            Ok(Self {
                stage,
                status,
                profile,
            })
        })
    }

    /// 输入已经是预乘、线性化显示参考浮点合成图；禁止传上一轮重塑结果。
    // 07 分块渲染入口：上下文、基底、输入／输出／源窗口、场、预算与取消谓词各自独立。
    #[allow(clippy::too_many_arguments)]
    pub fn render_tile(
        &self,
        context: &GpuContext,
        base: &TextureSlot,
        input: crate::portrait_tiles::PixelRect,
        output: crate::portrait_tiles::PixelRect,
        source_bounds: crate::portrait_tiles::PixelRect,
        field: &crate::portrait_tiles::CanonicalField,
        resident_bytes: u64,
        current: &dyn Fn() -> bool,
    ) -> Result<TextureSlot, GpuFailure> {
        use crate::portrait_tiles::{DeviceBudget, GPU_BUDGET};
        crate::portrait_analysis::check_current(current)?;
        if base.texture.width() != input.width
            || base.texture.height() != input.height
            || base.texture.format() != self.profile.format()
        {
            return Err("浮点输入纹理与窗口或精度不匹配".into());
        }
        let required = field
            .inverse_footprint(
                output,
                if self.profile == Profile::HighQuality {
                    2
                } else {
                    1
                },
                current,
            )?
            .intersect(source_bounds)
            .ok_or("逆足迹未与真实源域相交")?;
        if !input.contains(required) {
            return Err("局部输入缺少逆映射滤波足迹".into());
        }
        let budget = DeviceBudget {
            bytes: GPU_BUDGET,
            texture_side: context.limits.max_texture_dimension_2d,
            buffer_bytes: context.limits.max_buffer_size,
            resident_bytes,
        };
        if !budget.fits(input, output, self.profile == Profile::HighQuality)? {
            return Err(GpuFailure::OutOfMemory(
                "08 分块 GPU 资源不足，必须缩小核心块".into(),
            ));
        }
        let nodes = field.window_samples(output, current)?;
        checked_gpu_typed(&context.device, || {
            let dims = [output.width, output.height];
            let displacement = TextureSlot::try_new(
                &context.device,
                [dims[0] + 1, dims[1] + 1],
                wgpu::TextureFormat::Rgba32Float,
                false,
            )?;
            write_texture(
                context,
                &displacement.texture,
                bytemuck::cast_slice(&nodes),
                16,
            );
            let result = TextureSlot::try_new(&context.device, dims, self.profile.format(), true)?;
            let parameters = TileParams {
                full_dimensions: field.dimensions(),
                output_dimensions: dims,
                input_origin: [input.x, input.y],
                field_origin: [output.x, output.y],
                output_origin: [output.x, output.y],
                source_min: [source_bounds.x, source_bounds.y],
                source_max: [
                    i32::try_from(source_bounds.right()).map_err(|_| "源域溢出")?,
                    i32::try_from(source_bounds.bottom()).map_err(|_| "源域溢出")?,
                ],
                cubic: u32::from(self.profile == Profile::HighQuality),
                reserved: 0,
            };
            let buffer = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("08 窗口参数"),
                    contents: bytemuck::bytes_of(&parameters),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            context.queue.write_buffer(&self.status, 0, &[0u8; 4]);
            dispatch(
                context,
                &self.stage,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&base.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&displacement.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&result.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.status.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buffer.as_entire_binding(),
                    },
                ],
                dims,
            );
            crate::portrait_analysis::check_current(current)?;
            let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("08 安全回读"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = context.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.status, 0, &readback, 0, 4);
            context.queue.submit(Some(encoder.finish()));
            let bytes = read_buffer(&context.device, &readback)?;
            let status = u32::from_le_bytes(
                bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| "安全状态长度错误")?,
            );
            if status != 0 {
                return Err(format!("08 GPU 分块安全检查失败（{status}），禁止发布").into());
            }
            crate::portrait_analysis::check_current(current)?;
            Ok(result)
        })
    }
}

/// 每次分派后回读 4 字节状态；调用方必须在共享 GPU 锁内使用。
pub(crate) fn check_status(
    context: &GpuContext,
    status: &wgpu::Buffer,
    current: &dyn Fn() -> bool,
) -> Result<(), String> {
    crate::portrait_analysis::check_current(current)?;
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("08 安全状态"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(status, 0, &readback, 0, 4);
    context.queue.submit(Some(encoder.finish()));
    let bytes = read_buffer(&context.device, &readback)?;
    let value = u32::from_le_bytes(
        bytes
            .as_slice()
            .try_into()
            .map_err(|_| "安全状态长度错误")?,
    );
    if value != 0 {
        return Err(format!("08 GPU 安全检查失败（{value}），禁止发布"));
    }
    crate::portrait_analysis::check_current(current)
}

pub struct ReshapeGpu {
    pub profile: Profile,
    pub dimensions: [u32; 2],
    field_precision: FieldPrecision,
    field_key: Option<(String, String)>,
    pub input: TextureSlot,
    pub base: TextureSlot,
    pub warped: TextureSlot,
    output: TextureSlot,
    field: TextureSlot,
    pub base_stage: Stage,
    warp_stage: Stage,
    finish_stage: Stage,
    params: wgpu::Buffer,
    status: wgpu::Buffer,
    pub allocated_budget: u64,
}
impl ReshapeGpu {
    pub fn compatible(&self, dims: [u32; 2], profile: Profile, precision: FieldPrecision) -> bool {
        self.dimensions == dims && self.profile == profile && self.field_precision == precision
    }
    pub fn new(
        context: &GpuContext,
        dims: [u32; 2],
        profile: Profile,
        precision: FieldPrecision,
        main_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<Self, String> {
        let allocated_budget = estimated_bytes(dims)?;
        if dims
            .iter()
            .any(|&v| v + 1 > context.limits.max_texture_dimension_2d)
            || context.limits.max_buffer_size < (dims[0] as u64 * 16 + 255) * dims[1] as u64
        {
            return Err("设备纹理或读回缓冲区上限不足".into());
        }
        let device = &context.device;
        checked_gpu(device, || {
            let format = profile.format();
            let mut entries = main_entries.to_vec();
            entries[1] = storage_entry(1, format);
            let source = include_str!("shaders/shader.wgsl").replace(
                "rgba8unorm, write>",
                &format!("{}, write>", profile.wgsl_format()),
            );
            let base_stage = Stage::new(
                device,
                &entries,
                source,
                "main",
                &[("RESHAPE_BASE_OUTPUT", 1.0), ("HIGH_PRECISION_OUTPUT", 1.0)],
            );
            let finish_format = if profile == Profile::Preview {
                wgpu::TextureFormat::Rgba8Unorm
            } else {
                wgpu::TextureFormat::Rgba32Float
            };
            let finish_source = include_str!("shaders/shader.wgsl").replace(
                "rgba8unorm, write>",
                if profile == Profile::Preview {
                    "rgba8unorm, write>"
                } else {
                    "rgba32float, write>"
                },
            );
            let finish_stage = Stage::new(
                device,
                &[
                    texture_entry(0),
                    storage_entry(1, finish_format),
                    buffer_entry(
                        2,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        std::mem::size_of::<AllAdjustments>() as u64,
                    ),
                ],
                finish_source,
                "finish_reshape",
                &[(
                    "HIGH_PRECISION_OUTPUT",
                    if profile == Profile::Preview {
                        0.0
                    } else {
                        1.0
                    },
                )],
            );
            let warp_source = include_str!("shaders/portrait_reshape.wgsl").replace(
                "rgba16float, write>",
                &format!("{}, write>", profile.wgsl_format()),
            );
            let warp_stage = Stage::new(
                device,
                &[
                    texture_entry(0),
                    texture_entry(1),
                    storage_entry(2, format),
                    buffer_entry(3, wgpu::BufferBindingType::Uniform, 16),
                    buffer_entry(4, wgpu::BufferBindingType::Storage { read_only: false }, 4),
                ],
                warp_source,
                "main",
                &[],
            );
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("07 参数"),
                contents: bytemuck::bytes_of(&Params {
                    width: dims[0],
                    height: dims[1],
                    cubic: u32::from(profile == Profile::HighQuality),
                    reserved: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let status = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("07 安全状态"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            Ok(Self {
                profile,
                dimensions: dims,
                field_precision: precision,
                field_key: None,
                input: TextureSlot::new(device, dims, format, false),
                base: TextureSlot::new(device, dims, format, true),
                warped: TextureSlot::new(device, dims, format, true),
                output: TextureSlot::new(device, dims, finish_format, true),
                field: TextureSlot::new(
                    device,
                    [dims[0] + 1, dims[1] + 1],
                    if precision == FieldPrecision::F16 {
                        wgpu::TextureFormat::Rgba16Float
                    } else {
                        wgpu::TextureFormat::Rgba32Float
                    },
                    false,
                ),
                base_stage,
                warp_stage,
                finish_stage,
                params,
                status,
                allocated_budget,
            })
        })
    }
    /// 图像每次完整覆盖。只缓存按代次＋实际量化场哈希绑定的位移，不缓存调色结果。
    pub fn upload(
        &mut self,
        context: &GpuContext,
        pixels: &[f32],
        field: &ValidatedField,
        generation: &str,
    ) -> Result<(), String> {
        if generation.is_empty()
            || generation.len() > 4096
            || field.dimensions() != self.dimensions
            || field.precision() != self.field_precision
        {
            return Err("重塑资源代次、尺寸或精度不匹配".into());
        }
        if pixels.len() != (self.dimensions[0] * self.dimensions[1] * 4) as usize
            || pixels.as_chunks::<4>().0.iter().any(|p| {
                p.iter().any(|v| !v.is_finite() || v.abs() > 65504.0)
                    || !(0.0..=1.0).contains(&p[3])
            })
        {
            return Err("重塑输入不是有效浮点 RGBA".into());
        }
        let input_bytes = if self.profile == Profile::Preview {
            pixels
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect::<Vec<_>>()
        } else {
            pixels.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        write_texture(
            context,
            &self.input.texture,
            &input_bytes,
            if self.profile == Profile::Preview {
                8
            } else {
                16
            },
        );
        let key = (generation.to_owned(), field.key().to_owned());
        if self.field_key.as_ref() != Some(&key) {
            write_texture(
                context,
                &self.field.texture,
                &field.upload_bytes(),
                if self.field_precision == FieldPrecision::F16 {
                    8
                } else {
                    16
                },
            );
            self.field_key = Some(key);
        }
        Ok(())
    }
    pub fn warp(&self, context: &GpuContext) -> Result<(), String> {
        checked_gpu(&context.device, || {
            context.queue.write_buffer(&self.status, 0, &[0u8; 4]);
            dispatch(
                context,
                &self.warp_stage,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.base.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.field.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.warped.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.status.as_entire_binding(),
                    },
                ],
                self.dimensions,
            );
            let device = &context.device;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("07 安全回读"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.status, 0, &readback, 0, 4);
            context.queue.submit(Some(encoder.finish()));
            let bytes = read_buffer(device, &readback)?;
            let status = u32::from_le_bytes(bytes[..4].try_into().map_err(|_| "安全状态长度错误")?);
            if status != 0 {
                return Err(format!(
                    "GPU 上传／插值后验证失败（位标志 {status}），丢弃结果"
                ));
            }
            Ok(())
        })
    }
    pub fn finish(
        &self,
        context: &GpuContext,
        adjustments: &AllAdjustments,
    ) -> Result<DynamicImage, String> {
        checked_gpu(&context.device, || {
            let buffer = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("07 末尾参数"),
                    contents: bytemuck::bytes_of(adjustments),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            dispatch(
                context,
                &self.finish_stage,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.warped.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.output.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
                self.dimensions,
            );
            let [w, h] = self.dimensions;
            let bytes = crate::gpu_processing::read_texture_data_roi(
                &context.device,
                &context.queue,
                &self.output.texture,
                wgpu::Origin3d::ZERO,
                extent(self.dimensions),
                if self.profile == Profile::Preview {
                    4
                } else {
                    16
                },
            )?;
            if self.profile == Profile::Preview {
                Ok(DynamicImage::ImageRgba8(
                    ImageBuffer::from_raw(w, h, bytes).ok_or("8 位输出长度错误")?,
                ))
            } else {
                let floats = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                if floats.iter().any(|v| !v.is_finite()) {
                    return Err("FP32 输出包含非有限值".into());
                }
                let pixels = floats
                    .into_iter()
                    .map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
                    .collect::<Vec<_>>();
                Ok(DynamicImage::ImageRgba16(
                    ImageBuffer::<Rgba<u16>, _>::from_raw(w, h, pixels)
                        .ok_or("16 位输出长度错误")?,
                ))
            }
        })
    }
}
pub fn write_texture(context: &GpuContext, texture: &wgpu::Texture, bytes: &[u8], bpp: u32) {
    context.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(texture.width() * bpp),
            rows_per_image: Some(texture.height()),
        },
        texture.size(),
    );
}
pub(crate) fn dispatch(
    context: &GpuContext,
    stage: &Stage,
    entries: &[wgpu::BindGroupEntry<'_>],
    dims: [u32; 2],
) {
    let bg = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("07 资源绑定"),
            layout: &stage.layout,
            entries,
        });
    let mut encoder = context.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&stage.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dims[0].div_ceil(8), dims[1].div_ceil(8), 1);
    }
    context.queue.submit(Some(encoder.finish()));
}
pub(crate) fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .map_err(|e| format!("重塑 GPU 等待失败：{e}"))?;
    rx.recv_timeout(std::time::Duration::from_secs(1))
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let data = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    Ok(data)
}

#[cfg(test)]
#[path = "portrait_gpu_tests.rs"]
mod tests;

#[cfg(test)]
pub(crate) use tests::context as test_context;

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

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: BlurParams;

// 08 与 Rust CompositionWindow 同布局；独立入口不改变旧 blur 参数。
struct CompositionWindow {
    full_dimensions: vec2<u32>,
    input_origin: vec2<i32>,
    output_origin: vec2<i32>,
    source_min: vec2<i32>,
    source_max: vec2<i32>,
    mask_origin: vec2<i32>,
    radius: u32,
    direction: u32,
    reserved: vec2<u32>,
}
@group(0) @binding(3) var<uniform> window: CompositionWindow;
@group(0) @binding(4) var<storage, read_write> failure: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn window_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(output_texture))) { return; }
    let at = vec2<i32>(id.xy) + window.output_origin;
    let radius = i32(window.radius);
    let sigma = f32(radius) / 2.0;
    if (radius < 1 || window.direction > 1u) { atomicOr(&failure, 1u); return; }
    var total = vec3<f32>(0.0);
    var weights = 0.0;
    for (var offset = -radius; offset <= radius; offset++) {
        let delta = select(vec2<i32>(offset, 0), vec2<i32>(0, offset), window.direction == 1u);
        let global = clamp(at + delta, window.source_min, window.source_max - vec2<i32>(1));
        let local = global - window.input_origin;
        if (any(local < vec2<i32>(0)) || any(local >= vec2<i32>(textureDimensions(input_texture)))) {
            atomicOr(&failure, 8u); return;
        }
        let value = textureLoad(input_texture, local, 0);
        if (!all(abs(value) <= vec4<f32>(65504.0))) { atomicOr(&failure, 4u); return; }
        let weight = gaussian(f32(offset), sigma);
        total += clamp(value.rgb, vec3<f32>(0.0), vec3<f32>(F16_MAX)) * weight;
        weights += weight;
    }
    let result = total / weights;
    if (!all(abs(result) <= vec3<f32>(65504.0))) { atomicOr(&failure, 4u); return; }
    textureStore(output_texture, id.xy, vec4<f32>(result, 1.0));
}

const F16_MAX = 65504.0;

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

@compute @workgroup_size(256, 1, 1)
fn horizontal_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(output_texture));
    if (id.x >= u32(dims.x)) {
        return;
    }

    let radius = i32(params.radius);
    let sigma = f32(radius) / 2.0;

    let absolute_coord = vec2<u32>(id.x + params.tile_offset_x, id.y + params.tile_offset_y);
    let full_dims = vec2<i32>(textureDimensions(input_texture));

    let center_color = clamp(textureLoad(input_texture, absolute_coord, 0).rgb, vec3(0.0), vec3(F16_MAX));

    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var offset = -radius; offset <= radius; offset = offset + 1) {
        let sample_x = clamp(i32(absolute_coord.x) + offset, 0, full_dims.x - 1);
        let sample_coord = vec2<i32>(sample_x, i32(absolute_coord.y));

        let sample_color = clamp(textureLoad(input_texture, vec2<u32>(sample_coord), 0).rgb, vec3(0.0), vec3(F16_MAX));
        let weight = gaussian(f32(offset), sigma);

        total_color += sample_color * weight;
        total_weight += weight;
    }

    let final_color = total_color / total_weight;
    textureStore(output_texture, id.xy, vec4<f32>(final_color, 1.0));
}

@compute @workgroup_size(1, 256, 1)
fn vertical_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.y >= params.input_height) {
        return;
    }

    let radius = i32(params.radius);
    let sigma = f32(radius) / 2.0;

    let local_coord = vec2<i32>(id.xy);
    let max_y = i32(params.input_height) - 1;

    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var offset = -radius; offset <= radius; offset = offset + 1) {
        let sample_y = clamp(local_coord.y + offset, 0, max_y);
        let sample_coord = vec2<i32>(local_coord.x, sample_y);

        let sample_color = clamp(textureLoad(input_texture, vec2<u32>(sample_coord), 0).rgb, vec3(0.0), vec3(F16_MAX));
        let weight = gaussian(f32(offset), sigma);

        total_color += sample_color * weight;
        total_weight += weight;
    }

    let final_color = total_color / total_weight;
    textureStore(output_texture, id.xy, vec4<f32>(final_color, 1.0));
}

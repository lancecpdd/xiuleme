// 07：整数基底边界格点上的相对逆位移，输出像素中心为 id+.5。
struct Params { width: u32, height: u32, cubic: u32, reserved: u32 }
@group(0) @binding(0) var base: texture_2d<f32>;
@group(0) @binding(1) var field: texture_2d<f32>;
@group(0) @binding(2) var output_image: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> failure: atomic<u32>;

fn valid_node(p: vec4<f32>, at: vec2<u32>) -> bool {
    let dims = vec2<u32>(params.width, params.height);
    let source = vec2<f32>(at) + p.xy;
    let locked = p.z == 1.0 || any(at == vec2<u32>(0u)) || any(at == dims);
    return all(abs(p) <= vec4<f32>(65504.0)) && (p.z == 0.0 || p.z == 1.0) && p.w == 0.0
        && (!locked || all(p.xy == vec2<f32>(0.0)))
        && all(source >= vec2<f32>(0.0)) && all(source <= vec2<f32>(dims));
}
fn norm_bound(dx: vec2<f32>, dy: vec2<f32>) -> bool {
    // 与 portrait_gpu_field::GRADIENT_LIMIT 对齐。
    let limit = 4.0;
    return dot(dx,dx) + dot(dy,dy) <= limit * limit;
}
fn load_base(p: vec2<i32>) -> vec4<f32> {
    // 仅重建核 ClampToEdge，不接受越界映射中心。
    let q = clamp(p, vec2<i32>(0), vec2<i32>(vec2<u32>(params.width,params.height))-vec2<i32>(1));
    let v = textureLoad(base,q,0);
    if (!all(abs(v) <= vec4<f32>(65504.0)) || v.a < 0.0 || v.a > 1.0) { atomicOr(&failure,4u); }
    // FP16 下溢也可能让 Alpha 归零，隐藏颜色必须一并清零。
    if (v.a == 0.0) { return vec4<f32>(0.0); }
    return v;
}
fn cubic_weight(x: f32) -> f32 {
    let a = abs(x);
    if (a <= 1.0) { return ((1.5*a-2.5)*a)*a+1.0; }
    if (a < 2.0) { return ((-0.5*a+2.5)*a-4.0)*a+2.0; }
    return 0.0;
}
fn reconstruct(source: vec2<f32>) -> vec4<f32> {
    let q = source-vec2<f32>(0.5);
    let p = vec2<i32>(floor(q));
    let t = fract(q);
    if (params.cubic == 0u) {
        return mix(mix(load_base(p),load_base(p+vec2<i32>(1,0)),t.x),
            mix(load_base(p+vec2<i32>(0,1)),load_base(p+vec2<i32>(1,1)),t.x),t.y);
    }
    var value = vec4<f32>(0.0);
    var low = vec4<f32>(65504.0);
    var high = vec4<f32>(-65504.0);
    var color_low = vec3<f32>(1e30);
    var color_high = vec3<f32>(-1e30);
    // 三次结果限制到中心 2x2 的局部范围，避免棋盘／透明边过冲。
    for (var y = -1; y <= 2; y++) {
        for (var x = -1; x <= 2; x++) {
            let v = load_base(p+vec2<i32>(x,y));
            value += v*cubic_weight(f32(x)-t.x)*cubic_weight(f32(y)-t.y);
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
                low=min(low,v); high=max(high,v);
                if (v.a > 0.0) { color_low=min(color_low,v.rgb/v.a); color_high=max(color_high,v.rgb/v.a); }
            }
        }
    }
    value=clamp(value,low,high);
    // Alpha 和 RGB 均在预乘空间；零 Alpha 的隐藏颜色不得渗入。
    if (value.a <= 0.0) { return vec4<f32>(0.0); }
    // 分量限幅后还要保持 RGB／Alpha 的范围关系，避免半透明彩边被放大。
    return vec4<f32>(clamp(value.rgb,color_low*value.a,color_high*value.a),value.a);
}
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) { return; }
    let p = vec2<i32>(id.xy);
    let a = textureLoad(field,p,0);
    let b = textureLoad(field,p+vec2<i32>(1,0),0);
    let c = textureLoad(field,p+vec2<i32>(0,1),0);
    let d = textureLoad(field,p+vec2<i32>(1,1),0);
    if (!valid_node(a,id.xy) || !valid_node(b,id.xy+vec2<u32>(1u,0u))
        || !valid_node(c,id.xy+vec2<u32>(0u,1u)) || !valid_node(d,id.xy+vec2<u32>(1u,1u))
        || !norm_bound(b.xy-a.xy,c.xy-a.xy) || !norm_bound(b.xy-a.xy,d.xy-b.xy)
        || !norm_bound(d.xy-c.xy,c.xy-a.xy) || !norm_bound(d.xy-c.xy,d.xy-b.xy)) {
        atomicOr(&failure,1u); return;
    }
    let displacement = mix(mix(a.xy,b.xy,0.5),mix(c.xy,d.xy,0.5),0.5);
    let source = vec2<f32>(id.xy)+vec2<f32>(0.5)+displacement;
    if (!all(source >= vec2<f32>(0.0)) || !all(source <= vec2<f32>(vec2<u32>(params.width,params.height)))) {
        atomicOr(&failure,2u); return;
    }
    textureStore(output_image,id.xy,reconstruct(source));
}

// 08 单独入口：07 完整基底入口不引用此绑定，保留其原始布局与回归语义。
struct TileParams {
    full_dimensions: vec2<u32>,
    output_dimensions: vec2<u32>,
    input_origin: vec2<i32>,
    field_origin: vec2<i32>,
    output_origin: vec2<i32>,
    source_min: vec2<i32>,
    source_max: vec2<i32>,
    cubic: u32,
    reserved: u32,
}
@group(0) @binding(5) var<uniform> tile: TileParams;

fn tile_valid_node(p: vec4<f32>, at: vec2<i32>) -> bool {
    let dims = vec2<i32>(tile.full_dimensions);
    let outside = any(at < vec2<i32>(0)) || any(at > dims);
    let locked = p.z == 1.0 || any(at == vec2<i32>(0)) || any(at == dims) || outside;
    let source = vec2<f32>(at) + p.xy;
    let in_bounds = all(source >= vec2<f32>(0.0)) && all(source <= vec2<f32>(dims));
    return all(abs(p) <= vec4<f32>(65504.0)) && (p.z == 0.0 || p.z == 1.0) && p.w == 0.0
        && (!locked || all(p.xy == vec2<f32>(0.0))) && (outside || in_bounds);
}
fn tile_field(at: vec2<i32>) -> vec4<f32> {
    let p = at - tile.field_origin;
    if (any(p < vec2<i32>(0)) || any(p >= vec2<i32>(textureDimensions(field)))) {
        atomicOr(&failure, 8u); return vec4<f32>(0.0);
    }
    return textureLoad(field, p, 0);
}
fn tile_load(at: vec2<i32>) -> vec4<f32> {
    // 仅真实源域允许 clamp，上传块边缘缺失是错误，不能用黑条或边缘复制填补。
    let global = clamp(at, tile.source_min, tile.source_max - vec2<i32>(1));
    let local = global - tile.input_origin;
    if (any(local < vec2<i32>(0)) || any(local >= vec2<i32>(textureDimensions(base)))) {
        atomicOr(&failure, 8u); return vec4<f32>(0.0);
    }
    let v = textureLoad(base, local, 0);
    if (!all(abs(v) <= vec4<f32>(65504.0)) || v.a < 0.0 || v.a > 1.0) {
        atomicOr(&failure, 4u); return vec4<f32>(0.0);
    }
    if (v.a == 0.0) { return vec4<f32>(0.0); }
    return v;
}
fn tile_reconstruct(source: vec2<f32>) -> vec4<f32> {
    let q = source - vec2<f32>(0.5);
    let p = vec2<i32>(floor(q));
    let t = fract(q);
    if (tile.cubic == 0u) {
        return mix(mix(tile_load(p),tile_load(p+vec2<i32>(1,0)),t.x),
            mix(tile_load(p+vec2<i32>(0,1)),tile_load(p+vec2<i32>(1,1)),t.x),t.y);
    }
    var value = vec4<f32>(0.0);
    var low = vec4<f32>(65504.0);
    var high = vec4<f32>(-65504.0);
    var color_low = vec3<f32>(1e30);
    var color_high = vec3<f32>(-1e30);
    for (var y = -1; y <= 2; y++) {
        for (var x = -1; x <= 2; x++) {
            let v = tile_load(p+vec2<i32>(x,y));
            value += v*cubic_weight(f32(x)-t.x)*cubic_weight(f32(y)-t.y);
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
                low=min(low,v); high=max(high,v);
                if (v.a > 0.0) { color_low=min(color_low,v.rgb/v.a); color_high=max(color_high,v.rgb/v.a); }
            }
        }
    }
    value=clamp(value,low,high);
    if (value.a <= 0.0) { return vec4<f32>(0.0); }
    return vec4<f32>(clamp(value.rgb,color_low*value.a,color_high*value.a),value.a);
}
@compute @workgroup_size(8,8,1)
fn tile_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= tile.output_dimensions)) { return; }
    let at = vec2<i32>(id.xy) + tile.output_origin;
    let a = tile_field(at);
    let b = tile_field(at+vec2<i32>(1,0));
    let c = tile_field(at+vec2<i32>(0,1));
    let d = tile_field(at+vec2<i32>(1,1));
    if (!tile_valid_node(a,at) || !tile_valid_node(b,at+vec2<i32>(1,0))
        || !tile_valid_node(c,at+vec2<i32>(0,1)) || !tile_valid_node(d,at+vec2<i32>(1,1))
        || !norm_bound(b.xy-a.xy,c.xy-a.xy) || !norm_bound(b.xy-a.xy,d.xy-b.xy)
        || !norm_bound(d.xy-c.xy,c.xy-a.xy) || !norm_bound(d.xy-c.xy,d.xy-b.xy)) {
        atomicOr(&failure,1u); return;
    }
    let displacement = mix(mix(a.xy,b.xy,0.5),mix(c.xy,d.xy,0.5),0.5);
    let source = vec2<f32>(at)+vec2<f32>(0.5)+displacement;
    if (!all(abs(source) <= vec2<f32>(1e8))) { atomicOr(&failure,2u); return; }
    textureStore(output_image,id.xy,tile_reconstruct(source));
}

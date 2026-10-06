struct FusionPlan { destination: vec4<u32>, parents: vec4<u32> }
@group(0) @binding(0) var<storage, read> plans: array<FusionPlan>;
@group(0) @binding(1) var<storage, read> events: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> types: array<u32>;
@group(0) @binding(4) var<storage, read_write> embryo9: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> embryo10: array<vec4<u32>>;
fn hash(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}
fn remap_float(value: u32, delta: i32) -> u32 {
    let index = bitcast<f32>(value);
    if (index < 0.0) { return value; }
    return bitcast<u32>(index + f32(delta));
}
fn remap_int(value: u32, delta: i32) -> u32 {
    let index = bitcast<i32>(value);
    if (index < 0) { return value; }
    return bitcast<u32>(index + delta);
}

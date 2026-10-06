// Additive local radiance; fixed point permits race-free overlapping emitters.
struct Params { origin: vec3<f32>, cell_size: f32, }
struct Emission { r: atomic<u32>, g: atomic<u32>, b: atomic<u32>, strength: atomic<u32>, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> glow: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> count: array<u32>;
@group(0) @binding(4) var<storage, read> solid: array<u32>;
@group(0) @binding(5) var<storage, read_write> emission: array<Emission>;
@group(0) @binding(6) var<storage, read> cell_occupancy: array<u32>;
const RES: i32 = 128;
// RGB <= 16 per cell: safe even if all 200k supported cells overlap.
const FP: f32 = 1024.0;
fn index(p: vec3<i32>) -> u32 { return u32(p.x + p.y * RES + p.z * RES * RES); }
fn inside(p: vec3<i32>) -> bool { return all(p >= vec3<i32>(0)) && all(p < vec3<i32>(RES)); }
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell = id.x;
    if cell >= count[0] { return; }
    let source = glow[cell];
    if source.w <= 0.001 { return; }
    let center = (positions[cell].xyz - params.origin) / params.cell_size;
    let base = vec3<i32>(floor(center));
    // Six voxels gives useful reach without population-wide voxel gathers.
    let radius = 6.0;
    for (var z = -6; z <= 6; z++) {
        for (var y = -6; y <= 6; y++) {
            for (var x = -6; x <= 6; x++) {
                let p = base + vec3<i32>(x,y,z);
                if !inside(p) { continue; }
                let delta = vec3<f32>(p) + 0.5 - center;
                let distance = length(delta);
                if distance >= radius { continue; }
                if !visible_between(center, vec3<f32>(p) + 0.5) { continue; }
                let edge = 1.0 - distance / radius;
                let power = min(source.w, 4.0) * edge * edge;
                let rgb = clamp(source.xyz, vec3<f32>(0.0), vec3<f32>(4.0)) * power;
                let i = index(p);
                atomicAdd(&emission[i].r, u32(round(rgb.r * FP)));
                atomicAdd(&emission[i].g, u32(round(rgb.g * FP)));
                atomicAdd(&emission[i].b, u32(round(rgb.b * FP)));
                // Strength drives heat and photocyte food. Round down so fixed-
                // point quantization can discard energy but can never create it.
                atomicAdd(&emission[i].strength, u32(power * FP));
            }
        }
    }
}

// OCCLUSION_IMPLEMENTATION

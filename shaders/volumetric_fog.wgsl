// Volumetric Fog Shader
// Full-screen ray march through the light field to produce:
//   1. Volumetric fog / haze in shadowed regions
//   2. God rays (light shafts) where light penetrates through openings
//   3. Atmospheric scattering for depth cues
//
// Rendered as a full-screen triangle pass that composites over the scene.
// Uses the depth buffer to stop ray marching at scene geometry.

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
}

struct FogParams {
    grid_resolution: u32,
    cell_size: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,
    world_radius: f32,
    // Light direction (normalized, pointing toward light)
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    // Fog density (0.0 = no fog, 1.0 = dense fog)
    fog_density: f32,
    // Number of ray march steps for fog
    fog_steps: u32,
    // Light color RGB
    light_color_r: f32,
    light_color_g: f32,
    light_color_b: f32,
    // Light intensity multiplier
    light_intensity: f32,
    // Fog color RGB (ambient/shadow color)
    fog_color_r: f32,
    fog_color_g: f32,
    fog_color_b: f32,
    // Scattering anisotropy (Henyey-Greenstein g parameter, 0 = isotropic, 0.7 = forward scatter)
    scattering_anisotropy: f32,
    // Absorption coefficient (how much light fog absorbs)
    absorption: f32,
    // Height fog: density falloff with height
    height_fog_density: f32,
    height_fog_falloff: f32,
    // Near/far for ray march
    ray_start: f32,
    ray_end: f32,
    _pad0: f32,
    _pad1: f32,
}

// Group 0: Camera
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// Group 1: Fog parameters and data (solid_mask removed — light_field encodes solid as 0.0)
@group(1) @binding(0)
var<uniform> fog_params: FogParams;

@group(1) @binding(1)
var<storage, read> light_field: array<f32>;

@group(1) @binding(2)
var depth_texture: texture_depth_2d;

@group(1) @binding(3)
var depth_sampler: sampler;

// Vertex output
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Full-screen triangle (3 vertices, no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate full-screen triangle from vertex index
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    
    return out;
}

// Henyey-Greenstein phase function (fast approximation)
// Models forward/backward scattering of light in participating media
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    // Fast approximation: pow(x, 1.5) ≈ x * sqrt(x)
    let denom_sqrt = sqrt(max(denom, 0.001));
    return (1.0 - g2) / (4.0 * 3.14159265 * denom * denom_sqrt);
}

// Sample light field at world position with nearest neighbor (fast, 1 read)
fn sample_light_field(world_pos: vec3<f32>) -> f32 {
    let res = fog_params.grid_resolution;
    let fres = f32(res);
    
    // Convert to grid-space
    let gx = (world_pos.x - fog_params.grid_origin_x) / fog_params.cell_size;
    let gy = (world_pos.y - fog_params.grid_origin_y) / fog_params.cell_size;
    let gz = (world_pos.z - fog_params.grid_origin_z) / fog_params.cell_size;
    
    // Round to nearest voxel
    let ix = i32(round(gx));
    let iy = i32(round(gy));
    let iz = i32(round(gz));
    let ires = i32(res);
    
    // Bounds check - outside grid = fully lit
    if (ix < 0 || ix >= ires || iy < 0 || iy >= ires || iz < 0 || iz >= ires) {
        return 1.0;
    }
    
    let idx = u32(ix) + u32(iy) * res + u32(iz) * res * res;
    return light_field[idx];
}

// Solid detection threshold: solid voxels have light_field = 0.0,
// non-solid always have >= ambient_floor (0.02). Using 0.01 as threshold.
const SOLID_LIGHT_THRESHOLD: f32 = 0.01;

// Ray-sphere intersection. Returns (t_near, t_far) or (-1, -1) if no hit.
// Sphere centered at origin with given radius.
fn intersect_sphere(ray_origin: vec3<f32>, ray_dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(ray_origin, ray_dir);
    let c = dot(ray_origin, ray_origin) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    let sqrt_disc = sqrt(discriminant);
    let t0 = (-b - sqrt_disc) / (2.0 * a);
    let t1 = (-b + sqrt_disc) / (2.0 * a);
    return vec2<f32>(t0, t1);
}

// Reconstruct world position from UV + depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV to NDC
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    // Flip Y for wgpu NDC convention
    let ndc_corrected = vec4<f32>(ndc.x, -ndc.y, ndc.z, 1.0);
    
    // Unproject
    let world_h = camera.inv_view_proj * ndc_corrected;
    return world_h.xyz / world_h.w;
}

// Compute fog density at a given world position (no solid check — handled by caller via light_field)
fn fog_density_at(world_pos: vec3<f32>) -> f32 {
    var density = fog_params.fog_density;
    
    // Smooth fade near sphere boundary to prevent hard edges
    let dist_from_center = length(world_pos);
    let fade_start = fog_params.world_radius * 0.85;
    if (dist_from_center > fade_start) {
        let fade = 1.0 - smoothstep(fade_start, fog_params.world_radius, dist_from_center);
        density *= fade;
    }
    
    // Height-based fog: denser at lower elevations (fast exp approximation)
    if (fog_params.height_fog_density > 0.0) {
        let height = world_pos.y - fog_params.grid_origin_y;
        let x = height * fog_params.height_fog_falloff;
        let height_factor = 1.0 / (1.0 + x + x * x * 0.5);
        density += fog_params.height_fog_density * height_factor;
    }
    
    return density;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    
    // Sample depth buffer
    let depth = textureSample(depth_texture, depth_sampler, uv);
    
    // Reconstruct world position of the scene geometry at this pixel
    let scene_world_pos = reconstruct_world_pos(uv, depth);
    
    // Ray direction from camera
    let ray_dir = normalize(scene_world_pos - camera.camera_pos);
    
    // Intersect ray with world sphere (centered at origin)
    let sphere_hit = intersect_sphere(camera.camera_pos, ray_dir, fog_params.world_radius);
    if (sphere_hit.y < 0.0) {
        // Ray misses sphere entirely — no fog
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Calculate ray march distance, clamped to sphere interior
    let scene_dist = length(scene_world_pos - camera.camera_pos);
    let sphere_near = max(sphere_hit.x, 0.0); // clamp entry to camera if inside
    let sphere_far = sphere_hit.y;
    
    let march_start = max(fog_params.ray_start, sphere_near);
    let march_end = min(min(scene_dist, fog_params.ray_end), sphere_far);
    let march_dist = march_end - march_start;
    
    if (march_dist <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Light direction and color
    let light_dir = normalize(vec3<f32>(fog_params.light_dir_x, fog_params.light_dir_y, fog_params.light_dir_z));
    let light_color = vec3<f32>(fog_params.light_color_r, fog_params.light_color_g, fog_params.light_color_b);
    let fog_color = vec3<f32>(fog_params.fog_color_r, fog_params.fog_color_g, fog_params.fog_color_b);
    
    // Phase function: blend between isotropic (even lighting) and directional (god rays)
    // Isotropic phase = uniform scattering in all directions
    let isotropic_phase = 1.0 / (4.0 * 3.14159265);
    // Directional phase = Henyey-Greenstein forward scattering (creates god rays)
    let cos_theta = dot(ray_dir, light_dir);
    let directed_phase = henyey_greenstein(cos_theta, fog_params.scattering_anisotropy);
    // scattering_anisotropy controls the blend: 0 = perfectly even, 1 = full god rays
    let phase = mix(isotropic_phase, directed_phase, fog_params.scattering_anisotropy);
    
    // Ray march through fog with adaptive step count
    // Reduce steps for longer rays to maintain consistent per-pixel cost
    let max_efficient_dist = fog_params.world_radius * 0.5;
    let step_reduction = min(march_dist / max_efficient_dist, 2.0);
    let step_count = max(fog_params.fog_steps / u32(1.0 + step_reduction * 0.5), 4u);
    let step_size = march_dist / f32(step_count);
    
    var accumulated_color = vec3<f32>(0.0);
    var transmittance = 1.0;
    
    // Dithered start to reduce banding
    let dither = fract(sin(dot(uv * 1000.0, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let start_offset = march_start + dither * step_size;
    
    for (var i = 0u; i < step_count; i++) {
        let t = start_offset + f32(i) * step_size;
        let sample_pos = camera.camera_pos + ray_dir * t;
        
        // Sample light field FIRST — also detects solid voxels (light = 0.0)
        // This single read replaces both the old solid_mask check and the light sample
        let light_intensity = sample_light_field(sample_pos);
        
        // Solid voxels have light = 0.0, non-solid have >= ambient_floor (0.02)
        if (light_intensity < SOLID_LIGHT_THRESHOLD) {
            continue;
        }
        
        // Get fog density at this position (no solid check needed)
        let density = fog_density_at(sample_pos);
        
        if (density <= 0.0) {
            continue;
        }
        
        // In-scattered light: light that scatters toward camera at this point
        let in_scattered = light_color * light_intensity * phase * fog_params.light_intensity;
        
        // Ambient fog color (visible even in shadow)
        let ambient = fog_color * 0.1;
        
        // Total light contribution at this sample
        let sample_color = in_scattered + ambient;
        
        // Beer-Lambert absorption (fast exp approximation)
        let sample_extinction = density * (fog_params.absorption + fog_params.fog_density) * step_size * 0.01;
        let sample_transmittance = 1.0 / (1.0 + sample_extinction + sample_extinction * sample_extinction * 0.5);
        
        // Accumulate (energy-conserving integration)
        let integrand = sample_color * density * step_size * 0.01;
        accumulated_color += transmittance * integrand;
        transmittance *= sample_transmittance;
        
        // Early exit if fully opaque
        if (transmittance < 0.01) {
            break;
        }
    }
    
    // Output: RGB = fog color contribution, A = opacity (1 - transmittance)
    let opacity = 1.0 - transmittance;
    return vec4<f32>(accumulated_color, opacity);
}

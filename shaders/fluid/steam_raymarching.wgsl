// Steam ray marching shader
// Renders steam (fluid type 3) as volumetric effect using ray marching

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct SteamParams {
    grid_resolution: u32,
    world_radius: f32,
    cell_size: f32,
    _pad0: u32,

    grid_origin: vec3<f32>,
    max_steps: u32,

    // Steam appearance
    steam_color: vec3<f32>,
    density_multiplier: f32,

    // Light direction
    light_dir: vec3<f32>,
    light_intensity: f32,

    // Scattering params
    absorption: f32,
    scattering: f32,
    phase_g: f32,  // Henyey-Greenstein phase function parameter
    _pad1: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> params: SteamParams;

@group(0) @binding(2)
var<storage, read> fluid_state: array<u32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle vertices
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Fullscreen triangle covering clip space
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;

    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, y * 0.5 + 0.5); // Remove Y flip for correct orientation

    return out;
}

// Convert world position to grid index (matching fluid simulator coordinate system)
fn world_to_grid(world_pos: vec3<f32>) -> vec3<i32> {
    let local_pos = (world_pos - params.grid_origin) / params.cell_size - 0.5;
    return vec3<i32>(floor(local_pos));
}

// Check if grid position is valid
fn is_valid_grid_pos(grid_pos: vec3<i32>) -> bool {
    let res = i32(params.grid_resolution);
    return grid_pos.x >= 0 && grid_pos.x < res &&
           grid_pos.y >= 0 && grid_pos.y < res &&
           grid_pos.z >= 0 && grid_pos.z < res;
}

// Get fluid type at grid position (0=empty, 1=water, 2=lava, 3=steam)
// The fluid state is packed: lower 16 bits = fluid type
fn get_fluid_type(grid_pos: vec3<i32>) -> u32 {
    if !is_valid_grid_pos(grid_pos) {
        return 0u;
    }
    let idx = u32(grid_pos.x) + u32(grid_pos.y) * params.grid_resolution +
              u32(grid_pos.z) * params.grid_resolution * params.grid_resolution;
    // Extract fluid type from lower 16 bits
    return fluid_state[idx] & 0xFFFFu;
}

// Sample steam density - fast single voxel lookup
fn sample_steam_density(world_pos: vec3<f32>) -> f32 {
    let grid_pos = world_to_grid(world_pos);
    let fluid_type = get_fluid_type(grid_pos);

    // Only count steam (type 3)
    if fluid_type == 3u {
        return 1.0;
    }

    return 0.0;
}

// Henyey-Greenstein phase function for anisotropic scattering
fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(denom, 1.5));
}

// Ray-box intersection
fn intersect_box(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray_dir;
    let t1 = (box_min - ray_origin) * inv_dir;
    let t2 = (box_max - ray_origin) * inv_dir;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    return vec2<f32>(max(t_near, 0.0), t_far);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct ray direction from UV coordinates
    let clip_pos = vec4<f32>(in.uv.x * 2.0 - 1.0, in.uv.y * 2.0 - 1.0, 1.0, 1.0);
    let world_pos = camera.inv_view_proj * clip_pos;
    let world_pos_3d = world_pos.xyz / world_pos.w;
    
    let ray_origin = camera.camera_pos;
    let ray_dir = normalize(world_pos_3d - ray_origin);

    // Calculate bounding box of fluid grid
    let world_size = f32(params.grid_resolution) * params.cell_size;
    let box_min = params.grid_origin;
    let box_max = params.grid_origin + vec3<f32>(world_size);

    // Intersect ray with bounding box
    let t_bounds = intersect_box(ray_origin, ray_dir, box_min, box_max);

    if t_bounds.x >= t_bounds.y {
        // Ray misses the volume
        discard;
    }

    // Ray marching parameters - use larger steps for performance
    let step_size = params.cell_size * 2.0;  // 2 voxels per step for speed
    let max_steps = min(params.max_steps, 64u);  // Cap at 64 steps

    // Accumulated values
    var accumulated_color = vec3<f32>(0.0);
    var transmittance = 1.0;

    // Light direction (normalized)
    let light_dir = normalize(params.light_dir);

    // March through volume with early termination
    var t = t_bounds.x;
    var hit_something = false;

    for (var i = 0u; i < max_steps; i++) {
        if t >= t_bounds.y {
            break;
        }

        // Early out if mostly opaque
        if transmittance < 0.05 {
            break;
        }

        let sample_pos = ray_origin + ray_dir * t;
        let density = sample_steam_density(sample_pos) * params.density_multiplier;

        if density > 0.01 {
            hit_something = true;

            // Calculate extinction
            let extinction = density * (params.absorption + params.scattering);
            let sample_transmittance = exp(-extinction * step_size);

            // In-scattering from light - simplified
            let cos_theta = dot(ray_dir, light_dir);
            let phase = phase_hg(cos_theta, params.phase_g);

            // Simple ambient + directional light
            let light_contribution = 0.4 + params.light_intensity * phase * 0.6;

            // Steam color with lighting
            let sample_color = params.steam_color * light_contribution * density * params.scattering;

            // Accumulate color with transmittance
            accumulated_color += sample_color * transmittance * step_size;
            transmittance *= sample_transmittance;
        }

        t += step_size;
    }

    // Discard if we didn't hit any steam
    if !hit_something {
        discard;
    }

    // Final alpha is 1 - transmittance (how much light was absorbed/scattered)
    let alpha = 1.0 - transmittance;

    if alpha < 0.01 {
        discard;
    }

    return vec4<f32>(accumulated_color, alpha);
}

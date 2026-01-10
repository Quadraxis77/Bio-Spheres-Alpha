// Compute shader for building cell instance buffers on the GPU
// Includes frustum culling and occlusion culling via Hi-Z depth buffer
// Eliminates CPU-side iteration and reduces CPU→GPU data transfer

// Input: Cell simulation data (SoA layout from CanonicalState)
struct CellData {
    position: vec3<f32>,
    _pad0: f32,
    rotation: vec4<f32>,  // Quaternion (x, y, z, w)
}

// Output: Packed instance data for rendering (96 bytes, 16-byte aligned)
// Matches Rust CellInstance struct in instance_builder.rs
struct CellInstance {
    position: vec3<f32>,        // 12 bytes - World position
    radius: f32,                // 4 bytes - Cell radius
    color: vec4<f32>,           // 16 bytes - RGBA color from mode
    visual_params: vec4<f32>,   // 16 bytes - x: specular, y: power, z: fresnel, w: emissive
    rotation: vec4<f32>,        // 16 bytes - Quaternion rotation
    type_data_0: vec4<f32>,     // 16 bytes - Type-specific data [0-3]
    type_data_1: vec4<f32>,     // 16 bytes - Type-specific data [4-7]
    // Total: 96 bytes
    //
    // Type data interpretation by cell type:
    // Test cell:
    //   type_data_0.x = membrane_noise_scale
    //   type_data_0.y = membrane_noise_strength
    //   type_data_0.z = membrane_noise_speed
    //   type_data_0.w = membrane_anim_offset
    //   type_data_1 = reserved (zeros)
    // Flagellocyte (future):
    //   type_data_0.x = flagella_angle
    //   type_data_0.y = flagella_speed
    //   type_data_0.zw = reserved
    //   type_data_1 = reserved
    // Neurocyte (future):
    //   type_data_0.xyz = sensor_direction
    //   type_data_0.w = reserved
    //   type_data_1 = reserved
}

// Mode visual data (from genome)
// Note: Using vec4 for color to ensure consistent 32-byte struct size
// that matches Rust layout (w component always 1.0 for opaque cells)
struct ModeVisuals {
    color: vec4<f32>,      // xyz = color, w = 1.0 (always opaque)
    emissive_pad: vec4<f32>, // x = emissive, yzw = padding
}

// Cell type visual settings
struct CellTypeVisuals {
    specular_strength: f32,
    specular_power: f32,
    fresnel_strength: f32,
    membrane_noise_scale: f32,
    membrane_noise_strength: f32,
    membrane_noise_speed: f32,
    _pad: vec2<f32>,
}

// Frustum plane (normal.xyz, distance)
struct FrustumPlane {
    normal_and_dist: vec4<f32>,
}

// Uniforms
struct BuildParams {
    cell_count: u32,
    mode_count: u32,
    cell_type_count: u32,
    culling_enabled: u32,  // 0 = disabled, 1 = frustum only, 2 = frustum + occlusion, 3 = occlusion only
    // View-projection matrix for occlusion culling
    view_proj: mat4x4<f32>,
    // Camera position for distance-based LOD
    camera_pos: vec3<f32>,
    near_plane: f32,
    far_plane: f32,
    // Screen dimensions for Hi-Z lookup
    screen_width: f32,
    screen_height: f32,
    hiz_mip_count: u32,
    // Occlusion culling parameters
    occlusion_bias: f32,
    occlusion_mip_override: i32,  // -1 = auto, 0+ = force specific mip level
    min_screen_size: f32,         // Minimum screen-space size (pixels) to cull
    min_distance: f32,            // Don't cull objects closer than this distance
    // Frustum planes (6 planes: left, right, bottom, top, near, far)
    frustum_planes: array<FrustumPlane, 6>,
}

@group(0) @binding(0) var<uniform> params: BuildParams;

// Input buffers (read-only)
// Note: positions.w contains mass, radius is calculated from mass in shader
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> radii: array<f32>;  // Fallback, prefer mass-based calculation
@group(0) @binding(4) var<storage, read> mode_indices: array<u32>;
@group(0) @binding(5) var<storage, read> cell_ids: array<u32>;
@group(0) @binding(6) var<storage, read> genome_ids: array<u32>;

// Constants for radius calculation
const PI: f32 = 3.14159265359;

// Calculate radius from mass (assuming unit density spheres)
// radius = (mass * 3 / (4 * PI))^(1/3)
fn calculate_radius_from_mass(mass: f32) -> f32 {
    let volume = mass / 1.0;  // density = 1.0
    return pow(volume * 3.0 / (4.0 * PI), 1.0 / 3.0);
}

// Lookup tables
@group(0) @binding(7) var<storage, read> mode_visuals: array<ModeVisuals>;
@group(0) @binding(8) var<storage, read> cell_type_visuals: array<CellTypeVisuals>;

// Output buffer (write-only)
@group(0) @binding(9) var<storage, read_write> instances: array<CellInstance>;

// Atomic counter for visible instance count
@group(0) @binding(10) var<storage, read_write> counters: array<atomic<u32>>;
// counters[0] = visible count, counters[1] = total processed, counters[2] = frustum culled, counters[3] = occluded

// Hi-Z depth texture for occlusion culling (optional, binding 11)
// Note: R32Float is not filterable, so we use textureLoad instead of textureSample
@group(0) @binding(11) var hiz_texture: texture_2d<f32>;

// GPU-side cell count buffer: [0] = total cells, [1] = live cells
@group(0) @binding(12) var<storage, read> cell_count_buffer: array<u32>;

// ============================================================================
// Frustum Culling
// ============================================================================

// Frustum culling for spheres using clip-space bounds.
// Projects the sphere center and checks if it's within NDC bounds,
// with a conservative margin for the sphere radius.
fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    // Project center to clip space
    let clip = params.view_proj * vec4<f32>(center, 1.0);
    
    // Behind camera check (w <= 0 means behind or at camera)
    // Use small epsilon to handle cells very close to near plane
    if (clip.w <= 0.001) {
        return false;
    }
    
    // For proper frustum culling, we need to check if the sphere
    // is completely outside any of the 6 frustum planes.
    // In clip space, a point is inside if: -w <= x,y,z <= w
    // For a sphere, we add radius margin in world space, which maps to clip space.
    
    // Calculate clip-space radius (conservative estimate)
    // The sphere extends 'radius' in world space, which in clip space
    // is approximately radius * (projection_scale / distance)
    // Since clip.w ≈ distance for perspective, we use:
    let clip_radius = radius * 2.0;  // Conservative multiplier for edge cases
    
    // Check against frustum planes in clip space
    // Left plane: x >= -w  =>  x + w >= 0
    if (clip.x < -clip.w - clip_radius) {
        return false;
    }
    // Right plane: x <= w  =>  w - x >= 0
    if (clip.x > clip.w + clip_radius) {
        return false;
    }
    // Bottom plane: y >= -w
    if (clip.y < -clip.w - clip_radius) {
        return false;
    }
    // Top plane: y <= w
    if (clip.y > clip.w + clip_radius) {
        return false;
    }
    // Near plane: z >= 0 (wgpu uses [0,1] depth range)
    if (clip.z < -clip_radius) {
        return false;
    }
    // Far plane: z <= w
    if (clip.z > clip.w + clip_radius) {
        return false;
    }
    
    return true;
}

// ============================================================================
// Occlusion Culling (Hi-Z)
// ============================================================================

// Project a world-space point to NDC
fn project_to_ndc(world_pos: vec3<f32>) -> vec4<f32> {
    let clip = params.view_proj * vec4<f32>(world_pos, 1.0);
    return clip;
}

// Convert NDC to screen UV [0, 1]
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * 0.5 + 0.5;
}

// Get the appropriate Hi-Z mip level for a given screen-space size
fn get_hiz_mip_level(screen_size: f32) -> u32 {
    // Use a mip level where one texel covers approximately the object's screen size
    let mip = log2(max(screen_size, 1.0));
    return clamp(u32(mip), 0u, params.hiz_mip_count - 1u);
}

// Test if a billboard sprite is occluded using Hi-Z
// Returns true if occluded (should be culled), false if potentially visible
// Note: Cells are rendered as camera-facing billboards with sphere depth
fn sphere_occluded_hiz(center: vec3<f32>, radius: f32) -> bool {
    // Distance check - don't cull objects closer than min_distance
    let dist_to_camera = length(params.camera_pos - center);
    if (dist_to_camera < params.min_distance) {
        return false;
    }
    
    // Project billboard center to clip space
    let center_clip = project_to_ndc(center);
    
    // Behind camera check
    if (center_clip.w <= 0.0) {
        return false;
    }
    
    // Convert to NDC
    let center_ndc = center_clip.xyz / center_clip.w;
    
    // Check if in screen bounds (with some margin)
    if (abs(center_ndc.x) > 1.2 || abs(center_ndc.y) > 1.2) {
        return false;
    }
    
    // Calculate screen-space size (diameter in pixels)
    let screen_size = (radius * 2.0 / dist_to_camera) * max(params.screen_width, params.screen_height) * 0.5;
    
    // Don't cull if screen size is below threshold
    if (screen_size < params.min_screen_size) {
        return false;
    }
    
    // Convert NDC to UV (flip Y for texture coordinates)
    let uv = vec2<f32>(center_ndc.x * 0.5 + 0.5, -center_ndc.y * 0.5 + 0.5);
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    
    // Select mip level: use override if >= 0, otherwise use mip 0
    var mip_level: u32 = 0u;
    if (params.occlusion_mip_override >= 0) {
        mip_level = clamp(u32(params.occlusion_mip_override), 0u, params.hiz_mip_count - 1u);
    }
    
    // Get texture dimensions at selected mip level
    let tex_dims = textureDimensions(hiz_texture, i32(mip_level));
    let texel_coord = vec2<i32>(clamped_uv * vec2<f32>(tex_dims));
    let clamped_coord = clamp(texel_coord, vec2<i32>(0), vec2<i32>(tex_dims) - vec2<i32>(1));
    
    // Load Hi-Z depth (max depth = farthest surface in this region)
    // With MAX depth, we only cull if the cell is behind EVERYTHING in the region
    let hiz_depth = textureLoad(hiz_texture, clamped_coord, i32(mip_level)).r;
    
    // Calculate the front surface depth of this cell
    // The front surface is at (center - radius) along the view direction
    // Project the front surface point to get its depth
    let view_dir = normalize(center - params.camera_pos);
    let front_surface_world = center - view_dir * radius;
    let front_clip = params.view_proj * vec4<f32>(front_surface_world, 1.0);
    let front_depth = front_clip.z / front_clip.w;
    
    // Occluded if the cell's front surface is behind the Hi-Z depth (farthest in region)
    // This means the cell is behind everything visible in that screen region
    // Bias helps prevent z-fighting artifacts
    return front_depth > (hiz_depth + params.occlusion_bias);
}

// ============================================================================
// Main Compute Shader
// ============================================================================

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Read cell count from GPU buffer (zero CPU involvement)
    let cell_count = cell_count_buffer[0];
    
    if (idx >= cell_count) {
        return;
    }
    
    // Read cell data
    let pos_and_mass = positions[idx];
    let position = pos_and_mass.xyz;
    let mass = pos_and_mass.w;
    
    // Skip dead cells (mass below death threshold)
    // This handles cells marked for removal before the lifecycle pipeline runs
    const DEATH_MASS_THRESHOLD: f32 = 0.1;
    if (mass < DEATH_MASS_THRESHOLD) {
        return;
    }
    
    let rotation = rotations[idx];
    // Calculate radius from mass (GPU-side) for proper growth visualization
    // This ensures cells visually grow as mass increases from mass_accum shader
    let radius = calculate_radius_from_mass(mass);
    let mode_index = mode_indices[idx];
    let cell_id = cell_ids[idx];
    let cell_type = genome_ids[idx];
    
    // Increment total processed counter
    atomicAdd(&counters[1], 1u);
    
    // ============== CULLING ==============
    if (params.culling_enabled >= 1u) {
        // Frustum culling (modes 1 and 2)
        if (params.culling_enabled == 1u || params.culling_enabled == 2u) {
            if (!sphere_in_frustum(position, radius)) {
                atomicAdd(&counters[2], 1u); // Frustum culled counter
                return;
            }
        }
        
        // Occlusion culling (modes 2 and 3)
        if ((params.culling_enabled == 2u || params.culling_enabled == 3u) && params.hiz_mip_count > 0u) {
            if (sphere_occluded_hiz(position, radius)) {
                atomicAdd(&counters[3], 1u); // Occluded counter
                return;
            }
        }
    }
    
    // ============== BUILD INSTANCE ==============
    
    // Look up mode visuals (with bounds check)
    var color = vec3<f32>(0.5, 0.5, 0.5);
    var emissive = 0.0;
    if (mode_index < params.mode_count) {
        let mode = mode_visuals[mode_index];
        color = mode.color.xyz;
        // Skip opacity - cells are always opaque
        emissive = mode.emissive_pad.x;
    }
    
    // Look up cell type visuals (with bounds check)
    var specular_strength = 0.5;
    var specular_power = 32.0;
    var fresnel_strength = 0.3;
    var noise_scale = 8.0;
    var noise_strength = 0.15;
    var noise_speed = 0.0;
    if (cell_type < params.cell_type_count) {
        let visuals = cell_type_visuals[cell_type];
        specular_strength = visuals.specular_strength;
        specular_power = visuals.specular_power;
        fresnel_strength = visuals.fresnel_strength;
        noise_scale = 0.0;        // Noise disabled
        noise_strength = 0.0;     // Noise disabled  
        noise_speed = 0.0;        // Noise disabled
    }
    
    // Calculate stable animation offset from cell ID
    let anim_offset = (f32(cell_id) * 0.1) % 100.0;
    
    // Build instance
    var instance: CellInstance;
    instance.position = position;
    instance.radius = radius;
    instance.color = vec4<f32>(color, 1.0);  // Always fully opaque
    instance.visual_params = vec4<f32>(specular_strength, specular_power, fresnel_strength, emissive);
    instance.rotation = rotation;
    // Type-specific data: membrane params in type_data_0 for Test cells
    instance.type_data_0 = vec4<f32>(noise_scale, noise_strength, noise_speed, anim_offset);
    instance.type_data_1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);  // Reserved for future use
    
    // Atomically get output index and write instance
    let output_idx = atomicAdd(&counters[0], 1u);
    instances[output_idx] = instance;
}

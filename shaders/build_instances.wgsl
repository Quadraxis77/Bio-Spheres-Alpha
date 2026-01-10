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
    // Test cell (cell_type = 0):
    //   type_data_0.x = membrane_noise_scale
    //   type_data_0.y = membrane_noise_strength
    //   type_data_0.z = membrane_noise_speed
    //   type_data_0.w = membrane_anim_offset
    //   type_data_1 = reserved (zeros)
    // Flagellocyte (cell_type = 1):
    //   type_data_0.x = tail_length (0.5 - 3.0)
    //   type_data_0.y = tail_thickness (0.01 - 0.3)
    //   type_data_0.z = tail_amplitude (0.0 - 0.5)
    //   type_data_0.w = tail_frequency (0.5 - 10.0)
    //   type_data_1.x = tail_speed (0.0 - 15.0, calculated from swim_force)
    //   type_data_1.y = tail_taper (0.0 - 1.0)
    //   type_data_1.z = tail_segments (4 - 64)
    //   type_data_1.w = reserved
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
    // Flagella parameters (used by Flagellocyte cell type)
    tail_length: f32,
    tail_thickness: f32,
    tail_amplitude: f32,
    tail_frequency: f32,
    // tail_speed removed - now calculated from swim_force in mode_properties
    tail_taper: f32,
    tail_segments: f32,
    _pad: vec4<f32>, // Padding to 64 bytes (16 floats)
}

// Mode properties (per-mode settings from genome)
// Layout: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass, nutrient_priority, swim_force, prioritize_when_low]
struct ModeProperties {
    nutrient_gain_rate: f32,
    max_cell_size: f32,
    membrane_stiffness: f32,
    split_interval: f32,
    split_mass: f32,
    nutrient_priority: f32,
    swim_force: f32,
    prioritize_when_low: f32,
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

// Calculate radius from mass (matching preview scene: radius = mass clamped to 0.5-2.0)
fn calculate_radius_from_mass(mass: f32) -> f32 {
    return clamp(mass, 0.5, 2.0);
}

// Lookup tables
@group(0) @binding(7) var<storage, read> mode_visuals: array<ModeVisuals>;
@group(0) @binding(8) var<storage, read> cell_type_visuals: array<CellTypeVisuals>;

// Output buffer (write-only)
@group(0) @binding(9) var<storage, read_write> instances: array<CellInstance>;

// Atomic counter for visible instance count
@group(0) @binding(10) var<storage, read_write> counters: array<atomic<u32>>;
// counters[0] = visible count (total), counters[1] = total processed, counters[2] = frustum culled, counters[3] = occluded
// counters[4] = Test cell count, counters[5] = Flagellocyte count
// Per-type counts are used for multi-pipeline rendering

// Hi-Z depth texture for occlusion culling (optional, binding 11)
// Note: R32Float is not filterable, so we use textureLoad instead of textureSample
@group(0) @binding(11) var hiz_texture: texture_2d<f32>;

// GPU-side cell count buffer: [0] = total cells, [1] = live cells
@group(0) @binding(12) var<storage, read> cell_count_buffer: array<u32>;

// Cell types per cell (0 = Test, 1 = Flagellocyte, etc.)
// NOTE: This buffer may be stale if mode cell_type settings changed after cell creation.
// The shader now derives cell_type from mode_cell_types[mode_index] for correctness.
@group(0) @binding(13) var<storage, read> cell_types: array<u32>;

// Mode properties (per-mode settings including swim_force for tail animation)
@group(0) @binding(14) var<storage, read> mode_properties: array<ModeProperties>;

// Mode cell types lookup table: mode_cell_types[mode_index] = cell_type
// This is always up-to-date with current genome settings, unlike cell_types buffer
@group(0) @binding(15) var<storage, read> mode_cell_types: array<u32>;

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
    const DEATH_MASS_THRESHOLD: f32 = 0.5;
    if (mass < DEATH_MASS_THRESHOLD) {
        return;
    }
    
    let rotation = rotations[idx];
    // Calculate radius from mass (GPU-side) for proper growth visualization
    // This ensures cells visually grow as mass increases from mass_accum shader
    let radius = calculate_radius_from_mass(mass);
    let mode_index = mode_indices[idx];
    let cell_id = cell_ids[idx];
    
    // Read cell_type from buffer - this is set during cell insertion and updated by triple buffer sync
    let cell_type = cell_types[idx];
    
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
    
    // Type-specific data based on cell type
    // Cell type 0 = Test: membrane noise params
    // Cell type 1 = Flagellocyte: flagella params
    // type_data_1.w always stores cell_type for the unified shader
    if (cell_type == 1u && cell_type < params.cell_type_count) {
        // Flagellocyte: populate type_data with flagella parameters from visuals
        let visuals = cell_type_visuals[cell_type];
        
        // Calculate tail_speed from swim_force (animation speed proportional to thrust)
        // swim_force is in range 0.0-1.0, scale to tail_speed 0.0-15.0
        var tail_speed = 0.0;
        if (mode_index < params.mode_count) {
            let props = mode_properties[mode_index];
            tail_speed = props.swim_force * 15.0;
        }
        
        // type_data layout for Flagellocyte:
        // [0]=tail_length, [1]=tail_thickness, [2]=tail_amplitude, [3]=tail_frequency
        // [4]=tail_speed, [5]=tail_taper, [6]=tail_segments, [7]=cell_type
        instance.type_data_0 = vec4<f32>(
            visuals.tail_length,
            visuals.tail_thickness,
            visuals.tail_amplitude,
            visuals.tail_frequency
        );
        instance.type_data_1 = vec4<f32>(
            tail_speed,
            visuals.tail_taper,
            visuals.tail_segments,
            f32(cell_type)  // Store cell_type for unified shader
        );
    } else {
        // Test cell or unknown: membrane params (currently disabled)
        // type_data_1.w = cell_type for unified shader
        instance.type_data_0 = vec4<f32>(noise_scale, noise_strength, noise_speed, anim_offset);
        instance.type_data_1 = vec4<f32>(0.0, 0.0, 0.0, f32(cell_type));
    }
    
    // Write instance at same index as input cell (deterministic 1:1 mapping)
    // Track counts by type for indirect draw buffers
    if (cell_type == 0u) {
        atomicAdd(&counters[4], 1u); // Test count
    } else {
        atomicAdd(&counters[5], 1u); // Flagellocyte count
    }
    
    // Increment total visible count
    atomicAdd(&counters[0], 1u);
    
    // Write to same index as input - deterministic 1:1 mapping
    instances[idx] = instance;
}

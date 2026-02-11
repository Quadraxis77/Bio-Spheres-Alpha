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
    // Goldberg ridge parameters (used by Photocyte membrane)
    goldberg_scale: f32,
    goldberg_ridge_width: f32,
    goldberg_meander: f32,
    goldberg_ridge_strength: f32,
    nucleus_scale: f32,
    _pad: f32,
}

// Mode properties (per-mode settings from genome)
// Layout: [nutrient_gain_rate, max_cell_size, membrane_stiffness, split_interval, split_mass, nutrient_priority, swim_force, prioritize_when_low, max_splits, split_ratio, buoyancy_force, padding]
// Total: 12 floats = 48 bytes per mode
struct ModeProperties {
    nutrient_gain_rate: f32,
    max_cell_size: f32,
    membrane_stiffness: f32,
    split_interval: f32,
    split_mass: f32,
    nutrient_priority: f32,
    swim_force: f32,
    prioritize_when_low: f32,
    max_splits: f32,
    split_ratio: f32,
    buoyancy_force: f32,
    _pad0: f32,
}

// Cell type behavior flags for parameterized shader logic
struct CellTypeBehaviorFlags {
    ignores_split_interval: u32,
    applies_swim_force: u32,
    uses_texture_atlas: u32,
    has_procedural_tail: u32,
    gains_mass_from_light: u32,
    is_storage_cell: u32,
    applies_buoyancy: u32,
    _padding: array<u32, 9>,
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
    // Camera focal length for LOD calculation
    focal_length: f32,
    // LOD parameters for configurable thresholds
    lod_scale_factor: f32,
    lod_threshold_low: f32,
    lod_threshold_medium: f32,
    lod_threshold_high: f32,
    // Debug colors flag for LOD visualization
    lod_debug_colors: u32,  // 0 = disabled, 1 = enabled
    // Cell buffer capacity for partition size calculation
    cell_capacity: u32,
    // Padding to maintain 16-byte alignment
    _padding0: f32,
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
// counters[4..4+MAX_TYPES] = per-type instance counts (dynamically allocated)
// Per-type counts are used for multi-pipeline rendering with dynamic partitioning

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

// Cell type behavior flags (one per type, up to MAX_TYPES=30)
@group(0) @binding(16) var<storage, read> type_behaviors: array<CellTypeBehaviorFlags>;

// ============================================================================
// Random Number Generation
// ============================================================================

// Hash function for generating pseudo-random values from cell ID
fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

// Generate a random float in [0, 1) from a seed
fn random_float(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967296.0;
}

// Generate a random unit quaternion from cell ID (for texture orientation)
fn random_quaternion(cell_id: u32) -> vec4<f32> {
    // Use different seeds for each component
    let u0 = random_float(cell_id);
    let u1 = random_float(cell_id + 12345u);
    let u2 = random_float(cell_id + 67890u);
    
    // Uniform random quaternion using Shoemake's method
    let sqrt_u0 = sqrt(u0);
    let sqrt_1_u0 = sqrt(1.0 - u0);
    let theta1 = 2.0 * PI * u1;
    let theta2 = 2.0 * PI * u2;
    
    return vec4<f32>(
        sqrt_1_u0 * sin(theta1),  // x
        sqrt_1_u0 * cos(theta1),  // y
        sqrt_u0 * sin(theta2),    // z
        sqrt_u0 * cos(theta2)     // w
    );
}

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
    
    // Derive cell_type from cell's stored type (preserves original type)
    // This prevents existing cells from changing when preview mode changes
    var cell_type = 0u;
    if (idx < arrayLength(&cell_types)) {
        cell_type = cell_types[idx];
    }
    
    // Fallback: if cell_types buffer is empty or invalid, use mode_cell_types
    // This can happen for newly inserted cells before cell_types is updated
    if (cell_type == 0u && mode_index < params.mode_count) {
        cell_type = mode_cell_types[mode_index];
    }
    
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
    
    // Rotation: Cells with texture atlas get randomized orientation, others use physics rotation
    let behavior = type_behaviors[cell_type];
    if (behavior.uses_texture_atlas != 0u) {
        // Texture atlas cells: use random quaternion based on cell_id for varied appearance
        instance.rotation = random_quaternion(cell_id);
    } else {
        // Other cell types: use physics-driven rotation
        instance.rotation = rotation;
    }
    
    // Type-specific data based on behavior flags
    // Cells with procedural tails: flagella params for 3D geometry
    // Cells with texture atlas: UV coordinates for texture-based rendering
    // type_data_1.w always stores cell_type for the shader
    if (behavior.has_procedural_tail != 0u && cell_type < params.cell_type_count) {
        // Procedural tail cells (e.g., Flagellocyte): populate type_data with tail parameters
        let visuals = cell_type_visuals[cell_type];
        
        // Calculate tail_speed from swim_force (animation speed proportional to thrust)
        // swim_force is in range 0.0-1.0, scale to tail_speed 0.0-15.0
        var tail_speed = 0.0;
        if (mode_index < params.mode_count) {
            let props = mode_properties[mode_index];
            tail_speed = props.swim_force * 15.0;
        }
        
        // type_data layout for Flagellocyte (pixelated 3D geometry):
        // [0]=tail_length, [1]=tail_thickness, [2]=tail_amplitude, [3]=tail_frequency
        // [4]=tail_speed, [5]=tail_taper, [6]=debug_colors_enabled, [7]=cell_type
        // Note: tail_segments is not stored - the tail shader uses a fixed LOD mesh
        instance.type_data_0 = vec4<f32>(
            visuals.tail_length,
            visuals.tail_thickness,
            visuals.tail_amplitude,
            visuals.tail_frequency
        );
        instance.type_data_1 = vec4<f32>(
            tail_speed,
            visuals.tail_taper,
            f32(params.lod_debug_colors),  // debug_colors in .z for consistency with cell shader
            f32(cell_type)  // Store cell_type for hybrid shader
        );
    } else if (behavior.uses_texture_atlas != 0u) {
        // Texture atlas cells (e.g., Test, Phagocyte, Photocyte, Lipocyte):
        // Calculate screen radius for LOD selection with configurable scale factor
        let camera_distance = max(length(position - params.camera_pos), 1.0);
        // Use configurable scale factor for more spread out transitions
        let screen_radius = (radius / camera_distance) * params.lod_scale_factor;
        let clamped_screen_radius = clamp(screen_radius, 0.1, 200.0);
        
        // Select LOD based on configurable thresholds
        var lod_level = 1u; // Default to Medium (64x64)
        if (clamped_screen_radius < params.lod_threshold_low) {
            lod_level = 0u; // Low (32x32) - Far away
        } else if (clamped_screen_radius < params.lod_threshold_medium) {
            lod_level = 1u; // Medium (64x64) - Medium distance
        } else if (clamped_screen_radius < params.lod_threshold_high) {
            lod_level = 2u; // High (128x128) - Close
        } else {
            lod_level = 3u; // Ultra (256x256) - Very close
        }
        
        // Calculate atlas UV coordinates for Test cell type
        // Atlas layout: 4 LODs horizontally, 2 cell types vertically
        // Test cells are in row 0, Flagellocyte cells in row 1
        let atlas_width = 1024.0;  // 4 * 256
        let atlas_height = 512.0;  // 2 * 256
        let slot_size = 256.0;     // Each slot is 256x256
        
        let x_offset = f32(lod_level) * slot_size;
        let y_offset = 0.0; // Test cells in row 0
        
        // Calculate actual texture size for this LOD
        let actual_size = slot_size / pow(2.0, f32(3u - lod_level)); // 32, 64, 128, 256
        
        // Center the texture within the slot
        let x_padding = (slot_size - actual_size) * 0.5;
        let y_padding = (slot_size - actual_size) * 0.5;
        
        let uv_min_x = (x_offset + x_padding) / atlas_width;
        let uv_min_y = (y_offset + y_padding) / atlas_height;
        let uv_max_x = (x_offset + x_padding + actual_size) / atlas_width;
        let uv_max_y = (y_offset + y_padding + actual_size) / atlas_height;
        
        // type_data layout for texture atlas cells:
        // [0]=uv_min.x, [1]=uv_min.y, [2]=uv_max.x, [3]=uv_max.y
        // [4]=lod_level, [5]=screen_radius, [6]=debug_colors_enabled, [7]=cell_type
        instance.type_data_0 = vec4<f32>(uv_min_x, uv_min_y, uv_max_x, uv_max_y);
        instance.type_data_1 = vec4<f32>(f32(lod_level), clamped_screen_radius, f32(params.lod_debug_colors), f32(cell_type)); // cell_type in .w
    } else {
        // Default: pack Goldberg ridge params for photocytes (and zeros for others)
        // type_data_0: [goldberg_scale, goldberg_ridge_width, goldberg_meander, goldberg_ridge_strength]
        // type_data_1: [0, 0, debug_colors, cell_type]
        if (cell_type < params.cell_type_count) {
            let visuals = cell_type_visuals[cell_type];
            instance.type_data_0 = vec4<f32>(
                visuals.goldberg_scale,
                visuals.goldberg_ridge_width,
                visuals.goldberg_meander,
                visuals.goldberg_ridge_strength
            );
        } else {
            instance.type_data_0 = vec4<f32>(3.0, 0.12, 0.08, 0.15);
        }
        var nuc_scale = 0.6;
        if (cell_type < params.cell_type_count) {
            nuc_scale = cell_type_visuals[cell_type].nucleus_scale;
        }
        instance.type_data_1 = vec4<f32>(nuc_scale, 0.0, f32(params.lod_debug_colors), f32(cell_type));
    }

    // Dynamic instance allocation - single buffer for all cell types
    // No partitioning - cells are allocated sequentially regardless of type
    let output_index = atomicAdd(&counters[0], 1u);

    // Bounds check
    if (output_index >= params.cell_capacity) {
        return;
    }

    // Also track per-type counts for stats (counters[4..4+MAX_TYPES])
    let counter_index = 4u + cell_type;
    atomicAdd(&counters[counter_index], 1u);

    // Write instance
    instances[output_index] = instance;
}

// Compute shader for building cell instance buffers on the GPU
// Includes frustum culling and occlusion culling via Hi-Z depth buffer
// Eliminates CPU-side iteration and reduces CPU->GPU data transfer

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
    // Ciliocyte (cell_type = 8):
    //   type_data_0.x = effective_speed (cilia_speed from mode_properties_v5)
    //   type_data_0.y = cilia_ring_frequency (from CellTypeVisuals)
    //   type_data_0.z = cilia_ring_depth (from CellTypeVisuals)
    //   type_data_0.w = cilia_ring_speed (from CellTypeVisuals)
    //   type_data_1.x = reserved (0)
    //   type_data_1.y = reserved (0)
    //   type_data_1.z = debug_colors
    //   type_data_1.w = cell_type
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
    // Cilia ring parameters (used by Ciliocyte cell type)
    cilia_ring_frequency: f32,
    cilia_ring_depth: f32,
    cilia_ring_speed: f32,
    _pad2: f32,
    // Generic type params packed into type_data_0 for default-branch types
    param_a: f32,
    param_b: f32,
    param_c: f32,
    param_d: f32,
}

// mode_properties_v1 per mode: [split_mass, nutrient_priority, swim_force, prioritize_when_low]
// swim_force is at .z

// Cell type behavior flags for parameterized shader logic
struct CellTypeBehaviorFlags {
    ignores_split_interval: u32,
    applies_swim_force: u32,
    uses_texture_atlas: u32,
    has_procedural_tail: u32,
    gains_mass_from_light: u32,
    is_storage_cell: u32,
    applies_buoyancy: u32,
    applies_cilia_force: u32,
    _padding: array<u32, 8>,
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
    current_time: f32,
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
@group(0) @binding(7) var<storage, read> mode_colors: array<vec4<f32>>;   // RGB color per mode (xyz=RGB, w=1.0)
@group(0) @binding(8) var<storage, read> cell_type_visuals: array<CellTypeVisuals>;
@group(0) @binding(18) var<storage, read> mode_emissive: array<vec4<f32>>; // emissive per mode (x=emissive, yzw=padding)

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

// mode_properties sub-buffer v1 (swim_force at .z): copied from GpuTripleBufferSystem.mode_properties_v1
@group(0) @binding(14) var<storage, read> mode_properties: array<vec4<f32>>;

// Mode cell types lookup table: mode_cell_types[mode_index] = cell_type
// This is always up-to-date with current genome settings, unlike cell_types buffer
@group(0) @binding(15) var<storage, read> mode_cell_types: array<u32>;

// Cell type behavior flags (one per type, up to MAX_TYPES=30)
@group(0) @binding(16) var<storage, read> type_behaviors: array<CellTypeBehaviorFlags>;

// Death flags - cells marked for removal by lifecycle system
@group(0) @binding(17) var<storage, read> death_flags: array<u32>;

// mode_properties_v5 per mode: [cilia_speed, cilia_push_bonded, cilia_use_signal, cilia_signal_channel]
@group(0) @binding(19) var<storage, read> mode_properties_v5: array<vec4<f32>>;

// Signal flags per cell: cell_idx * SIGNAL_CHANNELS + channel → packed u32 (bits 0-10 = signal value)
@group(0) @binding(20) var<storage, read> signal_flags: array<u32>;

// mode_properties_v7 per mode: [luminocyte_invert, unused, signal_channel, threshold]
@group(0) @binding(21) var<storage, read> mode_properties_v7: array<vec4<f32>>;

// Cave solid-mask data (bindings 22-23). This stays bound for compatibility
// with the instance-builder layout, but cells are deliberately not culled by
// this mask: contact with cave walls or the world boundary can place a cell
// center inside a solid voxel for a frame, and render-culling that state makes
// the cell blink out until physics pushes it back.
struct CaveCullParams {
    enabled: u32,           // 0 = disabled (no cave), 1 = enabled
    grid_resolution: u32,   // voxel grid side length (128)
    cell_size: f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    _pad0: f32,
    _pad1: f32,
}
@group(0) @binding(22) var<uniform> cave_cull: CaveCullParams;
@group(0) @binding(23) var<storage, read> cave_solid_mask: array<u32>;
@group(0) @binding(24) var<storage, read> cell_thermal_state: array<u32>;

// mode_properties_v14 per mode: [siphon_intake_rate, siphon_expel_rate, siphon_impulse, packed signal settings]
// packed = threshold * 128 + mode * 32 + invert * 16 + channel
@group(0) @binding(25) var<storage, read> mode_properties_v14: array<vec4<f32>>;
@group(0) @binding(26) var<storage, read> cell_water: array<f32>;
@group(0) @binding(27) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(28) var<storage, read_write> instance_velocities: array<vec4<f32>>;
@group(0) @binding(29) var<storage, read_write> siphon_instances: array<CellInstance>;
@group(0) @binding(30) var<storage, read_write> siphon_instance_velocities: array<vec4<f32>>;

const SIGNAL_CHANNELS: u32 = 16u;
const SIGNAL_VALUE_MASK: u32 = 2047u;
const THERMAL_STATE_FROZEN: u32 = 1u;

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

// Frustum culling for spheres using pre-extracted frustum planes.
// Each plane has an inward-pointing normal; a sphere is outside the frustum
// if its signed distance to ANY plane is less than -radius.
fn sphere_in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = params.frustum_planes[i].normal_and_dist;
        let normal = vec3<f32>(plane.x, plane.y, plane.z);
        let dist = dot(normal, center) + plane.w;
        if (dist < -radius) {
            return false;
        }
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
    
    // Skip dead cells (check death_flags set by lifecycle system)
    // Death flags are set when nutrients < 1.0 (DEATH_NUTRIENT_THRESHOLD)
    if (death_flags[idx] == 1u) {
        return;
    }

    let rotation = rotations[idx];
    // Calculate radius from mass (GPU-side) for proper growth visualization
    // This ensures cells visually grow as mass increases from mass_accum shader
    let radius = calculate_radius_from_mass(mass);
    let mode_index = mode_indices[idx];
    let cell_id = cell_ids[idx];
    
    // Derive visual cell_type from the current mode. The per-cell cell_types
    // buffer is a physics cache and can lag behind mode edits or mode switches.
    var cell_type = 0u;
    if (mode_index < arrayLength(&mode_cell_types)) {
        cell_type = mode_cell_types[mode_index];
    } else if (idx < arrayLength(&cell_types)) {
        cell_type = cell_types[idx];
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
    if (mode_index < arrayLength(&mode_colors)) {
        color = mode_colors[mode_index].xyz;
        emissive = mode_emissive[mode_index].x;
    }

    // For luminocytes (type 16), gate visual emissive on signal state to match
    // the light-field gating in photocyte_light.wgsl. In preview (no signals),
    // signal value is 0 → dim glow unless threshold is also 0.
    if (cell_type == 16u && mode_index < arrayLength(&mode_properties_v7)) {
        if (emissive < 0.001) { emissive = 0.5; } // default brightness when unset
        let lum = mode_properties_v7[mode_index];
        let invert = lum.x >= 0.5;
        let channel = min(u32(clamp(lum.z, 0.0, 15.0)), SIGNAL_CHANNELS - 1u);
        let threshold = lum.w;
        var sig_val = 0.0;
        let sig_idx = idx * SIGNAL_CHANNELS + channel;
        if (sig_idx < arrayLength(&signal_flags)) {
            sig_val = f32(signal_flags[sig_idx] & SIGNAL_VALUE_MASK);
        }
        let above = sig_val >= threshold;
        emissive = select(emissive * 0.15, emissive, above != invert);
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
        if (mode_index < arrayLength(&mode_colors)) {
            let props = mode_properties[mode_index]; // v1: [split_mass, nutrient_priority, swim_force, prioritize_when_low]
            tail_speed = props.z * 15.0; // swim_force
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
    } else if (behavior.applies_cilia_force != 0u && cell_type < params.cell_type_count) {
        // Ciliocyte (cell_type 8): pack cilia ring visual parameters
        // type_data_0: [effective_speed, cilia_ring_frequency, cilia_ring_depth, cilia_ring_speed]
        // type_data_1: [0, 0, debug_colors, cell_type]
        let visuals = cell_type_visuals[cell_type];

        // effective_speed: use fixed cilia_speed from mode_properties_v5
        // Signal evaluation happens on GPU in the cilia_force shader, so we just use the fixed speed here
        var cilia_speed = 0.0;
        if (mode_index < arrayLength(&mode_properties_v5)) {
            cilia_speed = mode_properties_v5[mode_index].x; // v5.x = cilia_speed
        }

        instance.type_data_0 = vec4<f32>(
            cilia_speed,
            visuals.cilia_ring_frequency,
            visuals.cilia_ring_depth,
            visuals.cilia_ring_speed
        );
        instance.type_data_1 = vec4<f32>(0.0, 0.0, f32(params.lod_debug_colors), f32(cell_type));
    } else {
        // Default: pack type-specific params for all remaining cell types.
        // type_data_0 layout per cell type:
        //   Test (0):        param_a/b/c/d -> [ring_frequency, ring_sharpness, ring_brightness, unused]
        //   Phagocyte (2):   param_a/b/c/d -> [nucleus_radius, nucleus_darkness, nucleus_sharpness, unused]
        //   Lipocyte (4):    param_a/b/c/d -> [droplet_scale, droplet_threshold, boundary_sharpness, brightness]
        //   Buoyocyte (5):   param_a/b/c/d -> [bubble_scale, rotation_speed, wall_brightness, gas_brightness]
        //   Devorocyte (11): param_a/b/c/d -> [spike_height, spike_sharpness, spike_embed, tip_fade]
        //   Luminocyte (16): param_a/b/c/d -> [band_frequency, band_width, core_glow, color_shift]
        //   Siphonocyte (17):param_a/b/c/d -> [aperture_radius, aperture_darkness, rim_brightness, nozzle_height]
        //                   goldberg_ridge_strength -> visual_params.w nozzle_embed_depth
        //   Plumocyte (18):  param_a/b/c/d -> [feather_length, feather_width, feather_brightness, stroke_speed]
        //   Stemocyte (19):  param_a/b/c/d -> [core_radius, bud_radius, branch_brightness, pulse_speed]
        //   Photocyte (3):   goldberg -> [subdivision, ridge_width, meander, ridge_strength]
        //   Glueocyte (6):   goldberg -> [voro_scale, border_width, meander, border_dark]
        //   Oculocyte (7):   goldberg -> [pupil_size, iris_freq, iris_texture, pupil_dark]
        //   Myocyte (9):     goldberg -> [line_freq, bulge_strength, warp_amt, unused]
        //   Embryocyte (10): goldberg -> [yolk_radius, -yolk_drop (negated), yolk_brightness, unused]
        //   Vasculocyte (12):goldberg -> [cell_scale, border_width, meander, border_depth]
        // type_data_1: [nucleus_scale/frozen_flag, anim_speed, debug_colors, cell_type]
        if (cell_type < params.cell_type_count) {
            let visuals = cell_type_visuals[cell_type];
            // For types with dedicated params (Test=0, Phagocyte=2, Lipocyte=4,
            // Buoyocyte=5, Devorocyte=11), use param_a/b/c/d.
            // For all others (Photocyte=3, Glueocyte=6, Oculocyte=7, Myocyte=9,
            // Embryocyte=10, Vasculocyte=12) use goldberg fields.
            let use_params = (cell_type == 0u || cell_type == 2u || cell_type == 4u
                           || cell_type == 5u || cell_type == 11u || cell_type == 16u
                           || cell_type == 17u || cell_type == 18u || cell_type == 19u);
            if (cell_type == 17u) {
                instance.type_data_0 = vec4<f32>(
                    visuals.param_a,
                    visuals.param_b,
                    visuals.param_c,
                    visuals.param_d
                );
            } else if (use_params) {
                instance.type_data_0 = vec4<f32>(
                    visuals.param_a,
                    visuals.param_b,
                    visuals.param_c,
                    visuals.param_d
                );
            } else if (cell_type == 10u) {
                // Embryocyte: yolk_offset_y must be negative in shader (clamped to -0.35..0.0).
                // goldberg_ridge_width stores the drop as a positive UI value - negate it here.
                instance.type_data_0 = vec4<f32>(
                    visuals.goldberg_scale,
                    -visuals.goldberg_ridge_width,
                    visuals.goldberg_meander,
                    visuals.goldberg_ridge_strength
                );
            } else {
                instance.type_data_0 = vec4<f32>(
                    visuals.goldberg_scale,
                    visuals.goldberg_ridge_width,
                    visuals.goldberg_meander,
                    visuals.goldberg_ridge_strength
                );
            }
        } else {
            instance.type_data_0 = vec4<f32>(3.0, 0.12, 0.08, 0.15);
        }
        var nuc_scale = 0.6;
        if (cell_type < params.cell_type_count) {
            nuc_scale = cell_type_visuals[cell_type].nucleus_scale;
        }
        var anim_speed_val = 0.0;
        if (cell_type < params.cell_type_count) {
            anim_speed_val = cell_type_visuals[cell_type].membrane_noise_speed;
        }
        // type_data_1: x=nucleus_scale (photocyte hex sphere radius), y=anim_speed, z=debug_colors, w=cell_type
        // NOTE: glueocyte reads .x as cell_seed via type_data_1.x - use nuc_scale here since
        // glueocyte reads it as fract(type_data_1.x * small_constant) which is stable for any value.
        var extra_x = nuc_scale;
        var extra_y = anim_speed_val;
        if (cell_type == 17u && mode_index < arrayLength(&mode_properties_v14)) {
            let siphon = mode_properties_v14[mode_index];
            let packed = u32(clamp(siphon.w, 0.0, 262143.0));
            let siphon_channel = packed & 15u;
            let siphon_invert = (packed & 16u) != 0u;
            let siphon_mode = min((packed / 32u) & 3u, 3u);
            let siphon_threshold = f32(packed / 128u);
            let stroke_phase = fract(params.current_time * mix(0.85, 2.4, clamp(siphon.z / 3.0, 0.0, 1.0)) + f32(cell_id & 1023u) * 0.013);
            let expel_stroke = smoothstep(0.54, 0.64, stroke_phase) * (1.0 - smoothstep(0.82, 0.96, stroke_phase));
            if (siphon_mode == 0u || siphon_mode == 1u) {
                extra_y = clamp((siphon.y + siphon.z) / 7.0, 0.2, 1.0);
            } else if (siphon_mode == 2u) {
                extra_y = clamp(siphon.x / 4.0, 0.15, 1.0);
            } else {
                extra_y = clamp(siphon.y / 4.0, 0.15, 1.0);
            }
            let raw_signal = signal_flags[idx * 16u + siphon_channel];
            let signal_active = select(f32(raw_signal & 2047u) >= siphon_threshold, f32(raw_signal & 2047u) < siphon_threshold, siphon_invert);
            var output_active = siphon_mode == 0u || ((siphon_mode == 1u || siphon_mode == 3u) && signal_active);
            let has_water = idx < arrayLength(&cell_water) && cell_water[idx] > 0.001;
            if (output_active && !has_water) {
                extra_y = select(extra_y, 0.18, siphon_mode != 0u);
            }
            if (output_active && has_water && expel_stroke > 0.05 && (siphon.y > 0.001 || siphon.z > 0.001)) {
                extra_x = select(1.0, 2.0, idx < arrayLength(&cell_thermal_state) && cell_thermal_state[idx] >= 6u);
            } else {
                extra_x = 0.0;
            }
        }
        if (cell_type == 18u && idx < arrayLength(&cell_thermal_state)) {
            extra_x = select(0.0, 1.0, cell_thermal_state[idx] <= THERMAL_STATE_FROZEN);
        }
        instance.type_data_1 = vec4<f32>(extra_x, extra_y, f32(params.lod_debug_colors), f32(cell_type));
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
    let type_output_index = atomicAdd(&counters[counter_index], 1u);

    // Write instance
    instances[output_index] = instance;
    instance_velocities[output_index] = velocities[idx];

    if (cell_type == 17u && type_output_index < params.cell_capacity) {
        var siphon_instance = instance;
        if (cell_type < params.cell_type_count) {
            siphon_instance.visual_params.w = clamp(cell_type_visuals[cell_type].goldberg_ridge_strength, 0.0, 0.28);
        }
        siphon_instances[type_output_index] = siphon_instance;
        siphon_instance_velocities[type_output_index] = velocities[idx];
    }
}

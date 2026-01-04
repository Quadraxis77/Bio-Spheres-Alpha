// Instance extraction compute shader for GPU scene rendering integration
// Extracts rendering data from GPU physics buffers to instance buffers
// Handles cell type-specific visual properties and mode-based appearance

// Input: GPU physics simulation data (SoA layout from triple buffer system)
// Output: Packed instance data for CellRenderer

// Instance data structure matching CellRenderer expectations
struct CellInstance {
    position: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
    visual_params: vec4<f32>,   // x: specular, y: power, z: fresnel, w: emissive
    membrane_params: vec4<f32>, // x: noise_scale, y: noise_strength, z: noise_speed, w: anim_offset
    rotation: vec4<f32>,        // quaternion (x, y, z, w)
}

// Mode visual data from genome system
struct ModeVisuals {
    color: vec4<f32>,      // xyz = color, w = opacity
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

// Extraction parameters
struct ExtractionParams {
    cell_count: u32,
    mode_count: u32,
    cell_type_count: u32,
    current_time: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: ExtractionParams;

// Input buffers from GPU physics simulation (read-only)
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;        // position_and_mass
@group(0) @binding(2) var<storage, read> orientations: array<vec4<f32>>;     // cell orientations
@group(0) @binding(3) var<storage, read> radii: array<f32>;                  // cell radii (from mass)
@group(0) @binding(4) var<storage, read> mode_indices: array<u32>;           // absolute mode indices
@group(0) @binding(5) var<storage, read> cell_ids: array<u32>;               // stable cell IDs
@group(0) @binding(6) var<storage, read> genome_ids: array<u32>;             // genome IDs (used as cell type)

// Lookup tables for visual properties
@group(0) @binding(7) var<storage, read> mode_visuals: array<ModeVisuals>;
@group(0) @binding(8) var<storage, read> cell_type_visuals: array<CellTypeVisuals>;

// Output buffer for rendering instances (write-only)
@group(0) @binding(9) var<storage, read_write> instances: array<CellInstance>;

// Extract visual properties from mode data
fn extract_mode_properties(mode_index: u32) -> vec4<f32> {
    // Default values if mode index is out of bounds
    var color = vec3<f32>(0.5, 0.5, 0.5);
    var opacity = 1.0;
    var emissive = 0.0;
    
    if (mode_index < params.mode_count) {
        let mode = mode_visuals[mode_index];
        color = mode.color.xyz;
        opacity = mode.color.w;
        emissive = mode.emissive_pad.x;
    }
    
    return vec4<f32>(color.x, color.y, color.z, opacity);
}

// Extract emissive value from mode data
fn extract_mode_emissive(mode_index: u32) -> f32 {
    if (mode_index < params.mode_count) {
        return mode_visuals[mode_index].emissive_pad.x;
    }
    return 0.0;
}

// Extract visual parameters from cell type data
fn extract_cell_type_properties(cell_type: u32) -> vec4<f32> {
    // Default visual parameters
    var specular_strength = 0.5;
    var specular_power = 32.0;
    var fresnel_strength = 0.3;
    
    if (cell_type < params.cell_type_count) {
        let visuals = cell_type_visuals[cell_type];
        specular_strength = visuals.specular_strength;
        specular_power = visuals.specular_power;
        fresnel_strength = visuals.fresnel_strength;
    }
    
    return vec4<f32>(specular_strength, specular_power, fresnel_strength, 0.0);
}

// Extract membrane parameters from cell type data
fn extract_membrane_properties(cell_type: u32, cell_id: u32) -> vec4<f32> {
    // Default membrane parameters
    var noise_scale = 8.0;
    var noise_strength = 0.15;
    var noise_speed = 0.0;
    
    if (cell_type < params.cell_type_count) {
        let visuals = cell_type_visuals[cell_type];
        noise_scale = visuals.membrane_noise_scale;
        noise_strength = visuals.membrane_noise_strength;
        noise_speed = visuals.membrane_noise_speed;
    }
    
    // Create stable animation offset from cell ID (doesn't change on split)
    let anim_offset = (f32(cell_id) * 0.1) % 100.0;
    
    return vec4<f32>(noise_scale, noise_strength, noise_speed, anim_offset);
}

// Main compute shader for instance extraction
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.cell_count) {
        return;
    }
    
    // Read cell data from GPU physics buffers
    let position = positions[idx].xyz;
    let radius = radii[idx];
    let orientation = orientations[idx];
    let mode_index = mode_indices[idx];
    let cell_id = cell_ids[idx];
    let cell_type = genome_ids[idx]; // Use genome_id as cell type
    
    // Extract visual properties from mode and cell type data
    let color = extract_mode_properties(mode_index);
    let emissive = extract_mode_emissive(mode_index);
    let visual_params = extract_cell_type_properties(cell_type);
    let membrane_params = extract_membrane_properties(cell_type, cell_id);
    
    // Combine emissive with visual parameters (w component)
    let final_visual_params = vec4<f32>(
        visual_params.x, // specular_strength
        visual_params.y, // specular_power
        visual_params.z, // fresnel_strength
        emissive         // emissive
    );
    
    // Build instance data for rendering
    var instance: CellInstance;
    instance.position = position;
    instance.radius = radius;
    instance.color = color;
    instance.visual_params = final_visual_params;
    instance.membrane_params = membrane_params;
    instance.rotation = orientation;
    
    // Write to output buffer
    instances[idx] = instance;
}
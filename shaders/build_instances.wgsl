// Compute shader for building cell instance buffers on the GPU
// Eliminates CPU-side iteration and reduces CPU→GPU data transfer

// Input: Cell simulation data (SoA layout from CanonicalState)
struct CellData {
    position: vec3<f32>,
    _pad0: f32,
    rotation: vec4<f32>,  // Quaternion (x, y, z, w)
}

// Output: Packed instance data for rendering
struct CellInstance {
    position: vec3<f32>,
    radius: f32,
    color: vec4<f32>,
    visual_params: vec4<f32>,   // x: specular, y: power, z: fresnel, w: emissive
    membrane_params: vec4<f32>, // x: noise_scale, y: noise_strength, z: noise_speed, w: anim_offset
    rotation: vec4<f32>,
}

// Mode visual data (from genome)
struct ModeVisuals {
    color: vec3<f32>,
    opacity: f32,
    emissive: f32,
    _pad: vec3<f32>,
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

// Uniforms
struct BuildParams {
    cell_count: u32,
    mode_count: u32,
    cell_type_count: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: BuildParams;

// Input buffers (read-only)
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> radii: array<f32>;
@group(0) @binding(4) var<storage, read> mode_indices: array<u32>;
@group(0) @binding(5) var<storage, read> cell_ids: array<u32>;
@group(0) @binding(6) var<storage, read> genome_ids: array<u32>;

// Lookup tables
@group(0) @binding(7) var<storage, read> mode_visuals: array<ModeVisuals>;
@group(0) @binding(8) var<storage, read> cell_type_visuals: array<CellTypeVisuals>;

// Output buffer (write-only)
@group(0) @binding(9) var<storage, read_write> instances: array<CellInstance>;

// Atomic counter for output index (for separating opaque/transparent)
@group(0) @binding(10) var<storage, read_write> counters: array<atomic<u32>>;
// counters[0] = opaque count, counters[1] = transparent start index

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.cell_count) {
        return;
    }
    
    // Read cell data
    let position = positions[idx].xyz;
    let rotation = rotations[idx];
    let radius = radii[idx];
    let mode_index = mode_indices[idx];
    let cell_id = cell_ids[idx];
    let cell_type = genome_ids[idx];
    
    // Look up mode visuals (with bounds check)
    var color = vec3<f32>(0.5, 0.5, 0.5);
    var opacity = 1.0;
    var emissive = 0.0;
    if (mode_index < params.mode_count) {
        let mode = mode_visuals[mode_index];
        color = mode.color;
        opacity = mode.opacity;
        emissive = mode.emissive;
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
        noise_scale = visuals.membrane_noise_scale;
        noise_strength = visuals.membrane_noise_strength;
        noise_speed = visuals.membrane_noise_speed;
    }
    
    // Calculate stable animation offset from cell ID
    let anim_offset = (f32(cell_id) * 0.1) % 100.0;
    
    // Build instance
    var instance: CellInstance;
    instance.position = position;
    instance.radius = radius;
    instance.color = vec4<f32>(color, opacity);
    instance.visual_params = vec4<f32>(specular_strength, specular_power, fresnel_strength, emissive);
    instance.membrane_params = vec4<f32>(noise_scale, noise_strength, noise_speed, anim_offset);
    instance.rotation = rotation;
    
    // Write to output buffer
    // For now, write sequentially (opaque/transparent sorting can be added later)
    instances[idx] = instance;
}

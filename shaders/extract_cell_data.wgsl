// GPU Cell Data Extraction Compute Shader
// Extracts selected cell data from GPU buffers for cell inspection
// Uses bounds validation and calculates derived values like age
//
// Algorithm:
// 1. Validate cell index bounds against cell count
// 2. Extract all cell properties from GPU buffers
// 3. Calculate derived values (age = current_time - birth_time)
// 4. Set is_valid flag based on bounds checking
// 5. Write extracted data to output buffer
//
// Workgroup size: (1,1,1) for single cell extraction

struct PhysicsParams {
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    cell_capacity: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct CellExtractionParams {
    cell_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct InspectedCellData {
    // Cell position (12 bytes + 4 padding = 16 bytes)
    position: vec3<f32>,
    _pad0: f32,
    
    // Cell velocity (12 bytes + 4 padding = 16 bytes)
    velocity: vec3<f32>,
    _pad1: f32,
    
    // Cell physics properties (16 bytes)
    mass: f32,
    radius: f32,
    birth_time: f32,
    age: f32,
    
    // Cell division properties (16 bytes)
    nutrient_threshold: f32,
    split_interval: f32,
    split_count: u32,
    max_splits: u32,
    
    // Cell genome and mode info (16 bytes)
    genome_id: u32,
    mode_index: u32,
    cell_id: u32,
    cell_slot_index: u32,
    
    // Cell state properties (16 bytes)
    nutrient_gain_rate: f32,
    max_cell_size: f32,
    stiffness: f32,
    is_valid: u32,
    
    // Additional properties (16 bytes)
    nutrients: f32,
    cell_type: u32,
    adhesion_count: u32,
    is_dead: u32,

    // Organism identity (16 bytes)
    organism_id: u32,  // min cell index in connected component; 0xFFFFFFFF = dead/isolated
    reserve: u32,      // embryocyte reserve (0-65535); also used by non-embryocytes as head-start buffer
    _pad3: u32,
    _pad4: u32,
}

// Physics bind group (group 0) - standard 6-binding layout
@group(0) @binding(0)
var<uniform> params: PhysicsParams;

@group(0) @binding(1)
var<storage, read> positions_in: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> velocities_in: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> positions_out: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> velocities_out: array<vec4<f32>>;

// GPU-side cell count: [0] = total cells, [1] = live cells
@group(0) @binding(5)
var<storage, read_write> cell_count_buffer: array<u32>;

// Cell extraction parameters uniform buffer
@group(1) @binding(0)
var<uniform> extraction_params: CellExtractionParams;

// Cell state buffers (read-only for extraction)
@group(2) @binding(0)
var<storage, read> birth_times: array<f32>;

@group(2) @binding(1)
var<storage, read> split_intervals: array<f32>;

@group(2) @binding(2)
var<storage, read> split_masses: array<f32>;

@group(2) @binding(3)
var<storage, read> split_counts: array<u32>;

@group(2) @binding(4)
var<storage, read> split_ready_frame: array<i32>;

@group(2) @binding(5)
var<storage, read> max_splits: array<u32>;

@group(2) @binding(6)
var<storage, read> genome_ids: array<u32>;

@group(2) @binding(7)
var<storage, read> mode_indices: array<u32>;

@group(2) @binding(8)
var<storage, read> cell_ids: array<u32>;

@group(2) @binding(9)
var<storage, read> nutrient_gain_rates: array<f32>;

@group(2) @binding(10)
var<storage, read> max_cell_sizes: array<f32>;

@group(2) @binding(11)
var<storage, read> stiffnesses: array<f32>;

@group(2) @binding(12)
var<storage, read> nutrients_buffer: array<i32>;

@group(2) @binding(13)
var<storage, read> cell_types: array<u32>;

@group(2) @binding(14)
var<storage, read> death_flags: array<u32>;

const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const FIXED_POINT_SCALE: f32 = 1000.0;

@group(2) @binding(15)
var<storage, read> cell_adhesion_indices: array<i32>;

// Organism label buffer — label[i] = min cell index in connected component
// 0xFFFFFFFF means the cell is dead or isolated
@group(2) @binding(16)
var<storage, read> label_buffer: array<u32>;

// Embryocyte reserve buffer (one u32 per cell, 0-65535)
@group(2) @binding(17)
var<storage, read> embryocyte_reserves: array<u32>;

// Output buffer for extracted cell data
@group(3) @binding(0)
var<storage, read_write> extracted_data: InspectedCellData;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Only thread 0 should execute (single workgroup dispatch)
    if (global_id.x != 0u || global_id.y != 0u || global_id.z != 0u) {
        return;
    }
    
    let cell_index = extraction_params.cell_index;
    let current_cell_count = cell_count_buffer[0];
    
    // Bounds validation
    if (cell_index >= current_cell_count || cell_index >= params.cell_capacity) {
        // Invalid cell index - set is_valid to false and return default values
        extracted_data.position = vec3<f32>(0.0, 0.0, 0.0);
        extracted_data.velocity = vec3<f32>(0.0, 0.0, 0.0);
        extracted_data.mass = 0.0;
        extracted_data.radius = 0.0;
        extracted_data.birth_time = 0.0;
        extracted_data.age = 0.0;
        extracted_data.nutrient_threshold = 0.0;
        extracted_data.split_interval = 0.0;
        extracted_data.split_count = 0u;
        extracted_data.max_splits = 0u;
        extracted_data.genome_id = 0u;
        extracted_data.mode_index = 0u;
        extracted_data.cell_id = 0u;
        extracted_data.cell_slot_index = cell_index;
        extracted_data.is_valid = 0u;
        extracted_data.nutrient_gain_rate = 0.0;
        extracted_data.max_cell_size = 0.0;
        extracted_data.stiffness = 0.0;
        extracted_data.nutrients = 0.0;
        extracted_data.cell_type = 0u;
        extracted_data.adhesion_count = 0u;
        extracted_data.is_dead = 0u;
        extracted_data.organism_id = 0xFFFFFFFFu;
        extracted_data.reserve = 0u;
        return;
    }
    
    // Extract position and mass from position buffer
    let position_mass = positions_in[cell_index];
    extracted_data.position = position_mass.xyz;
    extracted_data.mass = position_mass.w;
    
    // Extract velocity
    let velocity_data = velocities_in[cell_index];
    extracted_data.velocity = velocity_data.xyz;
    
    // Extract cell state properties
    extracted_data.birth_time = birth_times[cell_index];
    extracted_data.split_interval = split_intervals[cell_index];
    extracted_data.nutrient_threshold = split_masses[cell_index];
    extracted_data.split_count = split_counts[cell_index];
    extracted_data.max_splits = max_splits[cell_index];
    extracted_data.genome_id = genome_ids[cell_index];
    extracted_data.mode_index = mode_indices[cell_index];
    extracted_data.cell_id = cell_ids[cell_index];
    extracted_data.cell_slot_index = cell_index;
    extracted_data.nutrient_gain_rate = nutrient_gain_rates[cell_index];
    extracted_data.max_cell_size = max_cell_sizes[cell_index];
    extracted_data.stiffness = stiffnesses[cell_index];
    
    // Nutrients: convert from fixed-point i32 (scale 1000) to f32
    let nutrients_fixed = nutrients_buffer[cell_index];
    extracted_data.nutrients = f32(nutrients_fixed) / FIXED_POINT_SCALE;
    
    // Cell type
    extracted_data.cell_type = cell_types[cell_index];
    
    // Death flag
    extracted_data.is_dead = death_flags[cell_index];
    
    // Count active adhesion connections
    let adhesion_base = cell_index * MAX_ADHESIONS_PER_CELL;
    var adh_count = 0u;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        if (cell_adhesion_indices[adhesion_base + i] >= 0) {
            adh_count++;
        }
    }
    extracted_data.adhesion_count = adh_count;
    
    // Calculate derived values
    extracted_data.age = params.current_time - extracted_data.birth_time;
    extracted_data.radius = clamp(extracted_data.mass, 0.5, 2.0);
    
    // Organism ID from label buffer
    extracted_data.organism_id = label_buffer[cell_index];
    
    // Reserve
    extracted_data.reserve = embryocyte_reserves[cell_index];
    
    // Debug: log organism ID values
    if (extracted_data.organism_id == 0u) {
        // This shouldn't happen with DEAD_LABEL initialization
        // Add some visual indicator or store in a debug buffer if needed
    }
    
    // Set valid flag
    extracted_data.is_valid = 1u;
}
//! # Cell Division Execution Compute Shader
//!
//! This shader executes cell division by creating Child A and Child B from parent cells
//! that have been assigned division slots. It handles:
//! - Mass splitting (50/50 between children)
//! - Position offset and velocity inheritance
//! - Adhesion inheritance using zone classification (A, B, C)
//! - Deterministic adhesion placement in assigned slots
//!
//! ## Algorithm Overview
//! 1. **Child Creation**: Create Child A (replaces parent) and Child B (new cell)
//! 2. **Mass Distribution**: Split parent mass equally between children
//! 3. **Position Offset**: Offset children along division direction
//! 4. **Velocity Inheritance**: Both children inherit parent velocity
//! 5. **Adhesion Inheritance**: Classify and redistribute parent adhesions
//!
//! ## Zone Classification for Adhesion Inheritance
//! - **Zone A**: Adhesions closer to Child A position (inherited by Child A)
//! - **Zone B**: Adhesions closer to Child B position (inherited by Child B)  
//! - **Zone C**: Adhesions equidistant (duplicated to both children)
//!
//! ## Deterministic Execution
//! All operations use deterministic algorithms to ensure same input produces
//! same output across all execution modes (Preview, CPU, GPU).

// Physics parameters uniform buffer
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
    dragged_cell_index: i32,
    _padding1: vec3<f32>,
    _padding: array<vec4<f32>, 12>,
}

// Cell count buffer for tracking live cells
struct CellCountBuffer {
    total_cell_count: u32,
    live_cell_count: u32,
    total_adhesion_count: u32,
    free_adhesion_top: u32,
}

// Adhesion connection structure (96 bytes, matching reference exactly)
struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    anchor_direction_a: vec3<f32>,
    _padding_a: f32,
    anchor_direction_b: vec3<f32>,
    _padding_b: f32,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    _padding: vec2<u32>,
}

// GPU mode structure for division parameters
struct GpuMode {
    color: vec4<f32>,
    orientation_a: vec4<f32>,
    orientation_b: vec4<f32>,
    split_direction: vec4<f32>,
    child_modes: vec2<i32>,
    split_interval: f32,
    genome_offset: i32,
    // Additional mode fields would be here...
}

// Bind group 0: Physics parameters and cell count
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> cell_count_buffer: CellCountBuffer;

// Bind group 1: Cell data buffers (triple buffered)
@group(1) @binding(0) var<storage, read_write> position_and_mass: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> velocity: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> acceleration: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> cell_age: array<f32>;
@group(1) @binding(4) var<storage, read_write> nutrients: array<f32>;
@group(1) @binding(5) var<storage, read_write> signaling_substances: array<f32>;
@group(1) @binding(6) var<storage, read_write> toxins: array<f32>;

// Bind group 2: Division management buffers
@group(2) @binding(0) var<storage, read> division_candidates: array<u32>;
@group(2) @binding(1) var<storage, read> division_assignments: array<u32>;
@group(2) @binding(2) var<storage, read> division_count: u32;
@group(2) @binding(3) var<storage, read_write> free_slots: array<u32>;

// Bind group 3: Adhesion system buffers
@group(3) @binding(0) var<storage, read_write> adhesion_connections: array<AdhesionConnection>;
@group(3) @binding(1) var<storage, read_write> adhesion_indices: array<array<u32, 20>>;
@group(3) @binding(2) var<storage, read_write> adhesion_counts: array<u32>;

// Bind group 4: Genome mode data
@group(4) @binding(0) var<storage, read> modes: array<GpuMode>;
@group(4) @binding(1) var<storage, read> genome_ids: array<u32>;

/// Calculate division direction from mode data.
///
/// Uses the split_direction from the cell's genome mode to determine
/// the direction for positioning Child A and Child B.
///
/// # Arguments
/// * `mode` - Genome mode containing split direction
/// * `cell_index` - Index of dividing cell for deterministic randomization
///
/// # Returns
/// Normalized division direction vector
fn calculate_division_direction(mode: GpuMode, cell_index: u32) -> vec3<f32> {
    var direction = mode.split_direction.xyz;
    
    // If no specific direction, use deterministic pseudo-random direction
    if (length(direction) < 0.001) {
        // Simple deterministic pseudo-random based on cell index and frame
        let seed = cell_index * 1103515245u + u32(physics_params.current_frame) * 12345u;
        let x = f32((seed >> 16u) & 0xFFFFu) / 65535.0 * 2.0 - 1.0;
        let y = f32((seed >> 8u) & 0xFFFFu) / 65535.0 * 2.0 - 1.0;
        let z = f32(seed & 0xFFFFu) / 65535.0 * 2.0 - 1.0;
        direction = vec3<f32>(x, y, z);
    }
    
    return normalize(direction);
}

/// Classify adhesion into zones for inheritance.
///
/// Determines whether an adhesion should be inherited by Child A, Child B,
/// or both children based on geometric proximity to child positions.
///
/// # Arguments
/// * `adhesion` - Adhesion connection to classify
/// * `parent_pos` - Position of parent cell
/// * `child_a_pos` - Position of Child A
/// * `child_b_pos` - Position of Child B
/// * `other_cell_pos` - Position of the other cell in the adhesion
///
/// # Returns
/// Zone classification: 0 = Zone A, 1 = Zone B, 2 = Zone C (both)
fn classify_adhesion_zone(
    adhesion: AdhesionConnection,
    parent_pos: vec3<f32>,
    child_a_pos: vec3<f32>,
    child_b_pos: vec3<f32>,
    other_cell_pos: vec3<f32>
) -> u32 {
    let dist_to_a = distance(other_cell_pos, child_a_pos);
    let dist_to_b = distance(other_cell_pos, child_b_pos);
    let threshold = 0.1; // Small threshold for Zone C classification
    
    if (abs(dist_to_a - dist_to_b) < threshold) {
        return 2u; // Zone C - equidistant, duplicate to both
    } else if (dist_to_a < dist_to_b) {
        return 0u; // Zone A - closer to Child A
    } else {
        return 1u; // Zone B - closer to Child B
    }
}

/// Create new adhesion connection for child cell.
///
/// Creates a new adhesion connection by copying parent adhesion data
/// and updating cell indices and geometric properties.
///
/// # Arguments
/// * `parent_adhesion` - Original adhesion from parent
/// * `child_index` - Index of child cell
/// * `child_pos` - Position of child cell
/// * `other_cell_pos` - Position of other cell in adhesion
///
/// # Returns
/// New adhesion connection for child
fn create_child_adhesion(
    parent_adhesion: AdhesionConnection,
    child_index: u32,
    child_pos: vec3<f32>,
    other_cell_pos: vec3<f32>
) -> AdhesionConnection {
    var child_adhesion = parent_adhesion;
    
    // Update cell indices (child replaces parent in connection)
    if (parent_adhesion.cell_a_index == child_index) {
        child_adhesion.cell_a_index = child_index;
    } else {
        child_adhesion.cell_b_index = child_index;
    }
    
    // Recalculate anchor directions based on new positions
    let connection_vector = other_cell_pos - child_pos;
    let connection_length = length(connection_vector);
    
    if (connection_length > 0.001) {
        let normalized_direction = connection_vector / connection_length;
        
        if (parent_adhesion.cell_a_index == child_index) {
            child_adhesion.anchor_direction_a = normalized_direction;
        } else {
            child_adhesion.anchor_direction_b = -normalized_direction;
        }
    }
    
    return child_adhesion;
}

/// Execute cell division for a single parent cell.
///
/// Creates Child A (replaces parent) and Child B (new cell) with proper
/// mass distribution, positioning, and adhesion inheritance.
///
/// # Arguments
/// * `parent_index` - Index of parent cell to divide
/// * `child_b_index` - Assigned index for Child B (from slot assignment)
fn execute_cell_division(parent_index: u32, child_b_index: u32) {
    // Get parent cell data
    let parent_pos_mass = position_and_mass[parent_index];
    let parent_pos = parent_pos_mass.xyz;
    let parent_mass = parent_pos_mass.w;
    let parent_velocity = velocity[parent_index].xyz;
    
    // Get genome mode for division parameters
    let genome_id = genome_ids[parent_index];
    let mode = modes[genome_id];
    
    // Calculate division direction and child positions
    let division_direction = calculate_division_direction(mode, parent_index);
    let offset_distance = 0.6; // Small offset to separate children
    
    let child_a_pos = parent_pos - division_direction * offset_distance * 0.5;
    let child_b_pos = parent_pos + division_direction * offset_distance * 0.5;
    
    // Split mass equally between children
    let child_mass = parent_mass * 0.5;
    
    // Create Child A (replaces parent at parent_index)
    position_and_mass[parent_index] = vec4<f32>(child_a_pos, child_mass);
    velocity[parent_index] = vec4<f32>(parent_velocity, 0.0);
    acceleration[parent_index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    // Reset child properties
    cell_age[parent_index] = 0.0;
    nutrients[parent_index] = nutrients[parent_index] * 0.5;
    signaling_substances[parent_index] = signaling_substances[parent_index] * 0.5;
    toxins[parent_index] = toxins[parent_index] * 0.5;
    
    // Create Child B (new cell at child_b_index)
    position_and_mass[child_b_index] = vec4<f32>(child_b_pos, child_mass);
    velocity[child_b_index] = vec4<f32>(parent_velocity, 0.0);
    acceleration[child_b_index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    // Initialize Child B properties
    cell_age[child_b_index] = 0.0;
    nutrients[child_b_index] = nutrients[parent_index] * 0.5;
    signaling_substances[child_b_index] = signaling_substances[parent_index] * 0.5;
    toxins[child_b_index] = toxins[parent_index] * 0.5;
    
    // Set Child B genome (from mode child_modes)
    genome_ids[child_b_index] = u32(mode.child_modes.y);
    
    // Handle adhesion inheritance
    let parent_adhesion_count = adhesion_counts[parent_index];
    var child_a_adhesion_count = 0u;
    var child_b_adhesion_count = 0u;
    
    // Process each parent adhesion
    for (var i = 0u; i < parent_adhesion_count && i < 20u; i++) {
        let adhesion_index = adhesion_indices[parent_index][i];
        let parent_adhesion = adhesion_connections[adhesion_index];
        
        if (parent_adhesion.is_active == 0u) {
            continue;
        }
        
        // Get other cell position for zone classification
        let other_cell_index = select(parent_adhesion.cell_b_index, parent_adhesion.cell_a_index, 
                                     parent_adhesion.cell_a_index == parent_index);
        let other_cell_pos = position_and_mass[other_cell_index].xyz;
        
        // Classify adhesion zone
        let zone = classify_adhesion_zone(parent_adhesion, parent_pos, child_a_pos, child_b_pos, other_cell_pos);
        
        if (zone == 0u || zone == 2u) {
            // Zone A or Zone C: inherit to Child A
            if (child_a_adhesion_count < 20u) {
                let child_a_adhesion = create_child_adhesion(parent_adhesion, parent_index, child_a_pos, other_cell_pos);
                adhesion_connections[adhesion_index] = child_a_adhesion;
                adhesion_indices[parent_index][child_a_adhesion_count] = adhesion_index;
                child_a_adhesion_count++;
            }
        }
        
        if (zone == 1u || zone == 2u) {
            // Zone B or Zone C: inherit to Child B
            if (child_b_adhesion_count < 20u) {
                // Need to allocate new adhesion slot for Child B
                let new_adhesion_index = atomicAdd(&cell_count_buffer.total_adhesion_count, 1u);
                if (new_adhesion_index < arrayLength(&adhesion_connections)) {
                    let child_b_adhesion = create_child_adhesion(parent_adhesion, child_b_index, child_b_pos, other_cell_pos);
                    adhesion_connections[new_adhesion_index] = child_b_adhesion;
                    adhesion_indices[child_b_index][child_b_adhesion_count] = new_adhesion_index;
                    child_b_adhesion_count++;
                }
            }
        }
    }
    
    // Update adhesion counts
    adhesion_counts[parent_index] = child_a_adhesion_count;
    adhesion_counts[child_b_index] = child_b_adhesion_count;
    
    // Clear unused adhesion slots
    for (var i = child_a_adhesion_count; i < 20u; i++) {
        adhesion_indices[parent_index][i] = 0u;
    }
    for (var i = child_b_adhesion_count; i < 20u; i++) {
        adhesion_indices[child_b_index][i] = 0u;
    }
}

/// Main compute shader entry point.
///
/// Processes division candidates and executes cell division using assigned slots.
/// Each thread handles one division candidate, creating Child A and Child B
/// with proper mass distribution and adhesion inheritance.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_index = global_id.x;
    
    // Check if this thread has a division to process
    if (thread_index >= division_count) {
        return;
    }
    
    // Get parent cell index and assigned Child B slot
    let parent_index = division_candidates[thread_index];
    let child_b_index = division_assignments[thread_index];
    
    // Validate indices
    if (parent_index >= physics_params.cell_count || child_b_index >= physics_params.cell_count) {
        return;
    }
    
    // Execute the division
    execute_cell_division(parent_index, child_b_index);
    
    // Update live cell count (Child B is a new live cell)
    if (thread_index == 0u) {
        atomicAdd(&cell_count_buffer.live_cell_count, division_count);
    }
}
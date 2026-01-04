//! # Lifecycle Death Compaction Compute Shader
//!
//! This shader implements the second phase of prefix-sum lifecycle management by
//! compacting the cell arrays to remove dead cells. It uses prefix-sum results
//! to deterministically remove cells and update all related data structures.
//!
//! ## Functionality
//! - **Cell Removal**: Remove dead cells using prefix-sum compaction
//! - **Data Compaction**: Compact all cell property arrays
//! - **Index Remapping**: Update all indices to account for removed cells
//! - **Adhesion Cleanup**: Clean up adhesion connections for dead cells
//! - **Deterministic Processing**: Ensure consistent removal across frames
//!
//! ## Requirements Addressed
//! - 6.6: Cell aging and death conditions using GPU compute shaders
//! - 6.7: Cell removal and state compaction using GPU compute shaders

// Physics parameters and configuration
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
    _padding: array<f32, 48>,
}

// Cell count buffer for tracking live cells
struct CellCountBuffer {
    total_cell_count: u32,
    live_cell_count: u32,
    total_adhesion_count: u32,
    free_adhesion_top: u32,
}

// Adhesion connection structure (96 bytes)
struct GpuAdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    _padding_zones: vec2<u32>,
    anchor_direction_a: vec3<f32>,
    _padding_a: f32,
    anchor_direction_b: vec3<f32>,
    _padding_b: f32,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
}

// Bind group 0: Physics parameters and cell data
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> cell_count_buffer: CellCountBuffer;
@group(0) @binding(2) var<storage, read> death_flags: array<u32>;
@group(0) @binding(3) var<storage, read> death_prefix_sum: array<u32>;

// Bind group 1: Cell property arrays (all read_write for compaction)
@group(1) @binding(0) var<storage, read_write> position_and_mass: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> velocity: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> acceleration: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> prev_acceleration: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read_write> orientation: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> genome_orientation: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> angular_velocity: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read_write> angular_acceleration: array<vec4<f32>>;
@group(1) @binding(8) var<storage, read_write> prev_angular_acceleration: array<vec4<f32>>;
@group(1) @binding(9) var<storage, read_write> signalling_substances: array<vec4<f32>>;
@group(1) @binding(10) var<storage, read_write> mode_indices: array<i32>;
@group(1) @binding(11) var<storage, read_write> ages: array<f32>;
@group(1) @binding(12) var<storage, read_write> toxins: array<f32>;
@group(1) @binding(13) var<storage, read_write> nitrates: array<f32>;
@group(1) @binding(14) var<storage, read_write> cell_ids: array<u32>;
@group(1) @binding(15) var<storage, read_write> genome_ids: array<u32>;
@group(1) @binding(16) var<storage, read_write> birth_times: array<f32>;
@group(1) @binding(17) var<storage, read_write> split_intervals: array<f32>;
@group(1) @binding(18) var<storage, read_write> split_masses: array<f32>;
@group(1) @binding(19) var<storage, read_write> split_counts: array<i32>;
@group(1) @binding(20) var<storage, read_write> split_ready_frame: array<i32>;

// Bind group 2: Adhesion system (for cleanup)
@group(2) @binding(0) var<storage, read_write> adhesion_connections: array<GpuAdhesionConnection>;
@group(2) @binding(1) var<storage, read_write> adhesion_indices: array<i32>;
@group(2) @binding(2) var<storage, read_write> adhesion_counts: array<u32>;

/// Compact a single cell's data from source to destination index
fn compact_cell_data(source_index: u32, dest_index: u32) {
    // Copy all cell properties from source to destination
    position_and_mass[dest_index] = position_and_mass[source_index];
    velocity[dest_index] = velocity[source_index];
    acceleration[dest_index] = acceleration[source_index];
    prev_acceleration[dest_index] = prev_acceleration[source_index];
    orientation[dest_index] = orientation[source_index];
    genome_orientation[dest_index] = genome_orientation[source_index];
    angular_velocity[dest_index] = angular_velocity[source_index];
    angular_acceleration[dest_index] = angular_acceleration[source_index];
    prev_angular_acceleration[dest_index] = prev_angular_acceleration[source_index];
    signalling_substances[dest_index] = signalling_substances[source_index];
    mode_indices[dest_index] = mode_indices[source_index];
    ages[dest_index] = ages[source_index];
    toxins[dest_index] = toxins[source_index];
    nitrates[dest_index] = nitrates[source_index];
    cell_ids[dest_index] = cell_ids[source_index];
    genome_ids[dest_index] = genome_ids[source_index];
    birth_times[dest_index] = birth_times[source_index];
    split_intervals[dest_index] = split_intervals[source_index];
    split_masses[dest_index] = split_masses[source_index];
    split_counts[dest_index] = split_counts[source_index];
    split_ready_frame[dest_index] = split_ready_frame[source_index];
    
    // Copy adhesion data
    let source_adhesion_start = source_index * 10u;
    let dest_adhesion_start = dest_index * 10u;
    for (var i = 0u; i < 10u; i++) {
        adhesion_indices[dest_adhesion_start + i] = adhesion_indices[source_adhesion_start + i];
    }
    adhesion_counts[dest_index] = adhesion_counts[source_index];
}

/// Clean up adhesion connections that reference a dead cell
fn cleanup_adhesion_connections_for_cell(dead_cell_index: u32) {
    // Mark all adhesion connections involving this cell as inactive
    let adhesion_start_index = dead_cell_index * 10u;
    for (var i = 0u; i < 10u; i++) {
        let adhesion_index = adhesion_indices[adhesion_start_index + i];
        if (adhesion_index >= 0) {
            // Mark the connection as inactive
            adhesion_connections[adhesion_index].is_active = 0u;
        }
    }
    
    // Clear the cell's adhesion indices
    for (var i = 0u; i < 10u; i++) {
        adhesion_indices[adhesion_start_index + i] = -1;
    }
    adhesion_counts[dead_cell_index] = 0u;
}

/// Update adhesion connection indices after cell compaction
fn update_adhesion_indices_after_compaction(old_index: u32, new_index: u32) {
    // Update all adhesion connections that reference the moved cell
    for (var conn_idx = 0u; conn_idx < cell_count_buffer.total_adhesion_count; conn_idx++) {
        var connection = adhesion_connections[conn_idx];
        
        if (connection.is_active != 0u) {
            var updated = false;
            
            // Update cell_a_index if it matches
            if (connection.cell_a_index == old_index) {
                connection.cell_a_index = new_index;
                updated = true;
            }
            
            // Update cell_b_index if it matches
            if (connection.cell_b_index == old_index) {
                connection.cell_b_index = new_index;
                updated = true;
            }
            
            // Write back if updated
            if (updated) {
                adhesion_connections[conn_idx] = connection;
            }
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Check if this cell is dead
    let is_dead = death_flags[cell_index];
    
    if (is_dead != 0u) {
        // This cell is dead - clean up its adhesion connections
        cleanup_adhesion_connections_for_cell(cell_index);
    } else {
        // This cell is alive - calculate its new position after compaction
        let new_index = cell_index - death_prefix_sum[cell_index];
        
        // If the new index is different, we need to move the cell data
        if (new_index != cell_index) {
            compact_cell_data(cell_index, new_index);
            
            // Update adhesion connection indices that reference this cell
            update_adhesion_indices_after_compaction(cell_index, new_index);
        }
    }
    
    // Update cell count (only the first thread does this)
    if (cell_index == 0u) {
        let total_dead_cells = death_prefix_sum[physics_params.cell_count - 1u] + death_flags[physics_params.cell_count - 1u];
        let new_live_count = physics_params.cell_count - total_dead_cells;
        cell_count_buffer.live_cell_count = new_live_count;
    }
}
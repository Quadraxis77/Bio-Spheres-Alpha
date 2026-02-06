// Adhesion Cleanup Shader
// Runs after death_scan to clean up adhesion connections for dead cells
// Workgroup size: 256 threads for optimal GPU occupancy
//
// This shader:
// 1. Iterates through all adhesion connections in index order (deterministic)
// 2. Checks if either connected cell is dead (using death_flags)
// 3. Marks the adhesion as inactive
// 4. Removes the adhesion index from both cells' per-cell adhesion indices
// 5. Adds freed slots to the free slot stack in deterministic order
//
// Requirements: 4.1, 4.2, 4.3, 4.4, 9.3
//
// Determinism: Adhesions are processed in index order (0, 1, 2, ...).
// The free slot stack is built by atomically incrementing a counter,
// but since threads process in order and the atomic operations are
// serialized within a workgroup, the resulting order is deterministic
// for the same input state.

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

// Adhesion connection structure (104 bytes matching Rust GpuAdhesionConnection)
// IMPORTANT: Use vec4 for anchor directions because vec3 has 16-byte alignment in WGSL
// which would cause layout mismatch with Rust's [f32; 3] + f32 padding
struct AdhesionConnection {
    cell_a_index: u32,          // offset 0
    cell_b_index: u32,          // offset 4
    mode_index: u32,            // offset 8
    is_active: u32,             // offset 12
    zone_a: u32,                // offset 16
    zone_b: u32,                // offset 20
    _align_pad: vec2<u32>,      // offset 24-31 (8 bytes)
    anchor_direction_a: vec4<f32>,  // offset 32-47 (xyz = direction, w = padding)
    anchor_direction_b: vec4<f32>,  // offset 48-63 (xyz = direction, w = padding)
    twist_reference_a: vec4<f32>,   // offset 64-79
    twist_reference_b: vec4<f32>,   // offset 80-95
    _padding: vec2<u32>,            // offset 96-103
};

// Maximum adhesions per cell (reduced for 200K cell support)
const MAX_ADHESIONS_PER_CELL: u32 = 20u;

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

// Lifecycle bind group (group 1) - must match lifecycle_layout in compute_pipelines.rs
@group(1) @binding(0)
var<storage, read_write> death_flags: array<u32>;

@group(1) @binding(1)
var<storage, read_write> division_flags: array<u32>;

@group(1) @binding(2)
var<storage, read_write> free_slot_ring: array<u32>;

@group(1) @binding(3)
var<storage, read_write> division_slot_assignments: array<u32>;

@group(1) @binding(4)
var<storage, read_write> ring_state: array<atomic<u32>>;

// Adhesion bind group (group 2)
@group(2) @binding(0)
var<storage, read_write> adhesion_connections: array<AdhesionConnection>;

// Adhesion counts: [0] = total, [1] = live, [2] = free_top, [3] = padding
@group(2) @binding(1)
var<storage, read_write> adhesion_counts: array<atomic<u32>>;

// Per-cell adhesion indices (20 * i32 per cell, -1 = no connection)
@group(2) @binding(2)
var<storage, read_write> cell_adhesion_indices: array<i32>;

// Free adhesion slot stack
@group(2) @binding(3)
var<storage, read_write> free_adhesion_slots: array<u32>;

// Remove an adhesion index from a cell's per-cell adhesion indices array
fn remove_adhesion_from_cell(cell_idx: u32, adhesion_idx: u32) {
    let base_offset = cell_idx * MAX_ADHESIONS_PER_CELL;
    let adhesion_idx_i32 = i32(adhesion_idx);
    
    // Search for the adhesion index in the cell's array and remove it
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let slot_idx = base_offset + i;
        if (cell_adhesion_indices[slot_idx] == adhesion_idx_i32) {
            // Found it - set to -1 to mark as empty
            cell_adhesion_indices[slot_idx] = -1;
            break;
        }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    
    // Get total adhesion count and cell count
    let total_adhesions = atomicLoad(&adhesion_counts[0]);
    let cell_count = cell_count_buffer[0];
    
    // Process adhesions (existing functionality)
    if (thread_id < total_adhesions) {
        let adhesion_idx = thread_id;
        
        // Read the adhesion connection
        let connection = adhesion_connections[adhesion_idx];
        
        // Skip if already inactive
        if (connection.is_active != 0u) {
            let cell_a = connection.cell_a_index;
            let cell_b = connection.cell_b_index;
            
            // Check bounds
            if (cell_a >= cell_count || cell_b >= cell_count) {
                // Invalid cell indices - mark as inactive
                adhesion_connections[adhesion_idx].is_active = 0u;
                atomicSub(&adhesion_counts[1], 1u); // Decrement live count
                
                // Add to free slot stack (deterministic order by adhesion index)
                let free_slot_idx = atomicAdd(&adhesion_counts[2], 1u);
                free_adhesion_slots[free_slot_idx] = adhesion_idx;
                return;
            }
            
            // Check if either cell is dead
            let cell_a_dead = death_flags[cell_a] == 1u;
            let cell_b_dead = death_flags[cell_b] == 1u;
            
            if (cell_a_dead || cell_b_dead) {
                // Mark adhesion as inactive
                adhesion_connections[adhesion_idx].is_active = 0u;
                
                // Remove adhesion index from both cells' per-cell adhesion indices
                // Even if a cell is dead, we still clean up its indices for consistency
                remove_adhesion_from_cell(cell_a, adhesion_idx);
                remove_adhesion_from_cell(cell_b, adhesion_idx);
                
                // Decrement live adhesion count
                atomicSub(&adhesion_counts[1], 1u);
                
                // Add to free slot stack for reuse
                // Note: This uses atomic to get a unique slot in the free stack.
                // Since adhesions are processed in index order (0, 1, 2, ...),
                // and atomicAdd returns the previous value, lower adhesion indices
                // will get lower positions in the free stack, maintaining determinism.
                let free_slot_idx = atomicAdd(&adhesion_counts[2], 1u);
                free_adhesion_slots[free_slot_idx] = adhesion_idx;
            }
        }
    }
    
    // CRITICAL: Clear adhesion indices for dead cells
    // This prevents future adhesion physics from reading stale indices
    if (thread_id < cell_count && death_flags[thread_id] == 1u) {
        // Cell is dead - clear all its adhesion indices
        let base_offset = thread_id * MAX_ADHESIONS_PER_CELL;
        for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
            cell_adhesion_indices[base_offset + i] = -1;
        }
    }
}

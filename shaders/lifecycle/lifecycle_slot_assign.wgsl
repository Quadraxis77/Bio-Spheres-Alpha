//! # Lifecycle Slot Assignment Compute Shader
//!
//! This shader implements deterministic slot allocation using prefix-sum for
//! cell division. It uses the assignment formula: assignments[i] = freeSlots[reservations[i]]
//! to ensure deterministic and efficient slot allocation.
//!
//! ## Functionality
//! - **Prefix-Sum Allocation**: Use prefix-sum results for deterministic slot assignment
//! - **Free Slot Management**: Allocate slots from compacted free slot arrays
//! - **Availability Checking**: Ensure N ≥ M (available ≥ needed) before assignment
//! - **Assignment Formula**: Apply assignments[i] = freeSlots[reservations[i]]
//!
//! ## Requirements Addressed
//! - 6.1: Division mechanics using GPU compute shaders exclusively
//! - 6.2: Split_ready_frame tracking using GPU buffers and compute shaders
//! - 6.4: Division attempts using GPU compute shaders without CPU synchronization

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
    _padding: array<vec4<f32>, 12>,
}

// Cell count buffer for tracking available slots
struct CellCountBuffer {
    total_cell_count: u32,
    live_cell_count: u32,
    total_adhesion_count: u32,
    free_adhesion_top: u32,
}

// Bind group 0: Physics parameters and slot management
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read_write> cell_count_buffer: CellCountBuffer;
@group(0) @binding(2) var<storage, read> division_candidates: array<u32>;
@group(0) @binding(3) var<storage, read> division_reservations: array<u32>;
@group(0) @binding(4) var<storage, read> reservation_prefix_sum: array<u32>;
@group(0) @binding(5) var<storage, read> free_slots: array<u32>;
@group(0) @binding(6) var<storage, read> free_slot_count: u32;
@group(0) @binding(7) var<storage, read_write> division_assignments: array<u32>;

// Assignment constants
const INVALID_ASSIGNMENT: u32 = 0xFFFFFFFFu; // Invalid assignment marker

/// Check if there are enough free slots for all division requests
fn check_slot_availability(total_reservations: u32, available_slots: u32) -> bool {
    return available_slots >= total_reservations;
}

/// Apply the assignment formula: assignments[i] = freeSlots[reservations[i]]
/// This ensures deterministic slot allocation using prefix-sum results
fn calculate_slot_assignment(
    cell_index: u32,
    reservation_count: u32,
    prefix_sum_offset: u32,
    free_slots_array: ptr<storage, array<u32>, read>
) -> u32 {
    // Check if we have enough slots
    if (prefix_sum_offset + reservation_count > free_slot_count) {
        return INVALID_ASSIGNMENT;
    }
    
    // Assignment formula: assignments[i] = freeSlots[reservations[i]]
    // The prefix sum gives us the starting offset in the free slots array
    let assigned_slot = free_slots_array[prefix_sum_offset];
    
    return assigned_slot;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Check if this cell is a division candidate
    let is_candidate = division_candidates[cell_index];
    
    if (is_candidate != 0u) {
        // This cell wants to divide
        let reservation_count = division_reservations[cell_index];
        let prefix_sum_offset = reservation_prefix_sum[cell_index];
        
        // Calculate slot assignment using the assignment formula
        let assigned_slot = calculate_slot_assignment(
            cell_index,
            reservation_count,
            prefix_sum_offset,
            &free_slots
        );
        
        // Store the assignment
        division_assignments[cell_index] = assigned_slot;
    } else {
        // Not a division candidate
        division_assignments[cell_index] = INVALID_ASSIGNMENT;
    }
    
    // Global availability check (only first thread does this)
    if (cell_index == 0u) {
        // Calculate total reservations needed
        let last_index = physics_params.cell_count - 1u;
        let total_reservations = reservation_prefix_sum[last_index] + division_reservations[last_index];
        
        // Check if we have enough slots
        let slots_available = check_slot_availability(total_reservations, free_slot_count);
        
        // If not enough slots, we could mark some divisions as failed
        // For now, we proceed with available slots (first-come-first-served)
        if (!slots_available) {
            // Could implement priority-based allocation here
            // For now, divisions that don't get slots will be marked invalid
        }
    }
}
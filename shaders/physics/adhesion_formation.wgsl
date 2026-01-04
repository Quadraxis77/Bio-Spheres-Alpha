//! # Adhesion Bond Formation Compute Shader
//!
//! This compute shader implements adhesion bond formation logic including
//! proximity detection, genome-based formation criteria, zone classification,
//! and maximum adhesion limits per cell. It creates new bonds between cells
//! that meet the formation requirements.
//!
//! ## Bond Formation Process
//! - **Proximity Detection**: Find cells within adhesion range
//! - **Genome Compatibility**: Check if cell types can form bonds
//! - **Zone Classification**: Determine adhesion zones (A, B, C)
//! - **Capacity Limits**: Respect maximum adhesions per cell
//! - **Bond Creation**: Initialize new adhesion connections
//!
//! ## Formation Criteria
//! - Distance between cells must be within adhesion range
//! - Both cells must have available adhesion slots
//! - Genome modes must allow adhesion formation
//! - Zone classification determines inheritance behavior
//!
//! ## Performance Characteristics
//! - **Workgroup Size**: 64 threads for balanced compute and memory access
//! - **Memory Access**: Random access to cell arrays and adhesion data
//! - **Compute Intensity**: Moderate - distance calculations and zone classification
//! - **Atomic Operations**: Used for thread-safe adhesion slot allocation

// Physics parameters uniform buffer
struct PhysicsParams {
    // Time and frame info
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    
    // World and physics settings
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    
    // Grid settings
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    
    // UI interaction
    dragged_cell_index: i32,
    _padding1: vec3<f32>,
    
    // Padding to 256 bytes
    _padding: array<f32, 48>,
}

// GPU Mode adhesion settings structure (48 bytes)
struct GpuModeAdhesionSettings {
    can_break: i32,                         // 0 = false, 1 = true
    break_force: f32,                       // Force threshold for breaking
    rest_length: f32,                       // Natural bond length
    linear_spring_stiffness: f32,           // Spring constant for distance
    linear_spring_damping: f32,             // Damping coefficient
    orientation_spring_stiffness: f32,      // Rotational constraint stiffness
    orientation_spring_damping: f32,        // Rotational damping
    max_angular_deviation: f32,             // Maximum allowed angle deviation
    twist_constraint_stiffness: f32,        // Twist resistance stiffness
    twist_constraint_damping: f32,          // Twist damping
    enable_twist_constraint: i32,           // 0 = false, 1 = true
    _padding: i32,                          // Pad to 48 bytes
}

// GPU Mode structure for genome-based behavior
struct GpuMode {
    // Visual properties (64 bytes)
    color: vec4<f32>,
    orientation_a: vec4<f32>,
    orientation_b: vec4<f32>,
    split_direction: vec4<f32>,
    
    // Child modes (8 bytes)
    child_modes: vec2<i32>,
    
    // Division properties (8 bytes)
    split_interval: f32,
    genome_offset: i32,
    
    // Adhesion settings (48 bytes)
    adhesion_settings: GpuModeAdhesionSettings,
    
    // Adhesion behavior (16 bytes)
    parent_make_adhesion: i32,
    child_a_keep_adhesion: i32,
    child_b_keep_adhesion: i32,
    max_adhesions: i32,
}

// Adhesion connection structure (96 bytes total)
struct GpuAdhesionConnection {
    cell_a_index: u32,                      // Index of first cell
    cell_b_index: u32,                      // Index of second cell
    mode_index: u32,                        // Mode index for settings lookup
    is_active: u32,                         // 1 = active, 0 = inactive
    
    zone_a: u32,                            // Zone classification for cell A
    zone_b: u32,                            // Zone classification for cell B
    _padding_zones: vec2<u32>,              // Padding for alignment
    
    anchor_direction_a: vec3<f32>,          // Anchor direction for cell A (local space)
    _padding_a: f32,                        // Padding for 16-byte alignment
    
    anchor_direction_b: vec3<f32>,          // Anchor direction for cell B (local space)
    _padding_b: f32,                        // Padding for 16-byte alignment
    
    twist_reference_a: vec4<f32>,           // Reference quaternion for cell A twist
    twist_reference_b: vec4<f32>,           // Reference quaternion for cell B twist
}

// Bind group 0: Physics parameters and cell properties
@group(0) @binding(0) var<uniform> physics_params: PhysicsParams;
@group(0) @binding(1) var<storage, read> position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> orientation: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> genome_orientation: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mode_indices: array<i32>;
@group(0) @binding(5) var<storage, read> genome_modes: array<GpuMode>;

// Bind group 1: Adhesion system
@group(1) @binding(0) var<storage, read_write> adhesion_connections: array<GpuAdhesionConnection>;
@group(1) @binding(1) var<storage, read_write> adhesion_indices: array<i32>; // 10 per cell
@group(1) @binding(2) var<storage, read_write> adhesion_counts: array<atomic<u32>>;

// Bind group 2: Spatial grid data (for proximity detection)
@group(2) @binding(0) var<storage, read> grid_counts: array<u32>;
@group(2) @binding(1) var<storage, read> grid_offsets: array<u32>;
@group(2) @binding(2) var<storage, read> grid_indices: array<u32>;
@group(2) @binding(3) var<storage, read> grid_assignments: array<u32>;

/// Adhesion zone classification constants
const ZONE_A: u32 = 0u;  // Opposite to split direction
const ZONE_B: u32 = 1u;  // Same as split direction
const ZONE_C: u32 = 2u;  // Equatorial band

/// Maximum adhesions per cell
const MAX_ADHESIONS_PER_CELL: u32 = 10u;

/// Empty adhesion slot marker
const EMPTY_ADHESION_SLOT: i32 = -1;

/// Equatorial threshold in radians (4 degrees)
const EQUATORIAL_THRESHOLD: f32 = 0.06981317; // 4 degrees in radians

/// Rotate a vector by a quaternion.
///
/// Applies quaternion rotation to transform a vector from one coordinate
/// system to another. Used for converting world directions to local space.
///
/// # Arguments
/// * `v` - Vector to rotate
/// * `q` - Quaternion rotation (w, x, y, z)
///
/// # Returns
/// Rotated vector in new coordinate system
fn rotate_vector_by_quaternion(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    // Quaternion format: (w, x, y, z) where w is scalar part
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    
    // Quaternion rotation formula: v' = q * v * q^(-1)
    // Optimized version using cross products
    let qvec = vec3<f32>(x, y, z);
    let uv = cross(qvec, v);
    let uuv = cross(qvec, uv);
    
    return v + ((uv * w) + uuv) * 2.0;
}

/// Classify adhesion bond direction relative to split direction.
///
/// This determines which zone the adhesion belongs to for division inheritance:
/// - Zone A: Opposite to split direction → inherit to child B
/// - Zone B: Same as split direction → inherit to child A  
/// - Zone C: Equatorial band (90° ± 4°) → inherit to both children
///
/// # Arguments
/// * `bond_direction` - Direction of the adhesion bond (normalized)
/// * `split_direction` - Direction of cell division (normalized)
///
/// # Returns
/// Zone classification (ZONE_A, ZONE_B, or ZONE_C)
fn classify_bond_direction(bond_direction: vec3<f32>, split_direction: vec3<f32>) -> u32 {
    let dot_product = dot(bond_direction, split_direction);
    let angle = acos(clamp(abs(dot_product), 0.0, 1.0));
    let equatorial_angle = 1.5707963; // π/2 radians (90 degrees)
    
    // Check if within equatorial threshold (90° ± 4°)
    if (abs(angle - equatorial_angle) <= EQUATORIAL_THRESHOLD) {
        return ZONE_C; // Equatorial band
    }
    // Classify based on which side relative to split direction
    else if (dot_product > 0.0) {
        return ZONE_B; // Positive dot product (same direction as split)
    } else {
        return ZONE_A; // Negative dot product (opposite to split)
    }
}

/// Convert world position to grid coordinates.
///
/// Maps a world space position to discrete grid cell coordinates
/// for spatial partitioning and proximity detection.
///
/// # Arguments
/// * `world_pos` - Position in world space
/// * `world_size` - Size of simulation world
/// * `grid_resolution` - Number of grid cells per dimension
///
/// # Returns
/// Grid coordinates (x, y, z)
fn world_to_grid(world_pos: vec3<f32>, world_size: f32, grid_resolution: i32) -> vec3<i32> {
    let half_world = world_size * 0.5;
    let normalized_pos = (world_pos + vec3<f32>(half_world)) / world_size;
    let grid_pos = normalized_pos * f32(grid_resolution);
    return vec3<i32>(
        clamp(i32(grid_pos.x), 0, grid_resolution - 1),
        clamp(i32(grid_pos.y), 0, grid_resolution - 1),
        clamp(i32(grid_pos.z), 0, grid_resolution - 1)
    );
}

/// Convert grid coordinates to linear index.
///
/// Maps 3D grid coordinates to a linear array index for
/// accessing grid data structures.
///
/// # Arguments
/// * `grid_coords` - Grid coordinates (x, y, z)
/// * `grid_resolution` - Number of grid cells per dimension
///
/// # Returns
/// Linear grid index
fn grid_coords_to_index(grid_coords: vec3<i32>, grid_resolution: i32) -> u32 {
    return u32(grid_coords.x + grid_coords.y * grid_resolution + grid_coords.z * grid_resolution * grid_resolution);
}

/// Find a free adhesion slot in a cell's adhesion indices array.
///
/// Searches through the cell's 10 adhesion slots to find an empty one
/// (marked with EMPTY_ADHESION_SLOT). Uses atomic operations for thread safety.
///
/// # Arguments
/// * `cell_index` - Index of the cell to search
///
/// # Returns
/// Slot index if found, or MAX_ADHESIONS_PER_CELL if no free slots
fn find_free_adhesion_slot(cell_index: u32) -> u32 {
    let base_index = cell_index * MAX_ADHESIONS_PER_CELL;
    
    for (var slot = 0u; slot < MAX_ADHESIONS_PER_CELL; slot++) {
        let slot_index = base_index + slot;
        if (adhesion_indices[slot_index] == EMPTY_ADHESION_SLOT) {
            return slot;
        }
    }
    
    return MAX_ADHESIONS_PER_CELL; // No free slots
}

/// Find a free adhesion connection slot in the global connections array.
///
/// Searches through the adhesion connections array to find an inactive slot
/// that can be used for a new connection.
///
/// # Returns
/// Connection index if found, or arrayLength if no free connections
fn find_free_connection_slot() -> u32 {
    let connection_count = arrayLength(&adhesion_connections);
    
    for (var i = 0u; i < connection_count; i++) {
        if (adhesion_connections[i].is_active == 0u) {
            return i;
        }
    }
    
    return connection_count; // No free connections
}

/// Check if two cells can form an adhesion bond.
///
/// Evaluates formation criteria including distance, genome compatibility,
/// and available adhesion slots.
///
/// # Arguments
/// * `cell_a_idx` - Index of first cell
/// * `cell_b_idx` - Index of second cell
/// * `pos_a` - Position of first cell
/// * `pos_b` - Position of second cell
/// * `mode_a` - Genome mode of first cell
/// * `mode_b` - Genome mode of second cell
///
/// # Returns
/// True if cells can form a bond, false otherwise
fn can_form_bond(
    cell_a_idx: u32,
    cell_b_idx: u32,
    pos_a: vec3<f32>,
    pos_b: vec3<f32>,
    mode_a: GpuMode,
    mode_b: GpuMode
) -> bool {
    // Don't bond to self
    if (cell_a_idx == cell_b_idx) {
        return false;
    }
    
    // Check distance (use rest length as formation distance)
    let distance = length(pos_b - pos_a);
    let formation_distance = mode_a.adhesion_settings.rest_length * 1.2; // 20% tolerance
    if (distance > formation_distance) {
        return false;
    }
    
    // Check if both cells have available adhesion slots
    let count_a = atomicLoad(&adhesion_counts[cell_a_idx]);
    let count_b = atomicLoad(&adhesion_counts[cell_b_idx]);
    
    if (count_a >= u32(mode_a.max_adhesions) || count_b >= u32(mode_b.max_adhesions)) {
        return false;
    }
    
    // Check if cells can make adhesions (genome setting)
    if (mode_a.parent_make_adhesion == 0 || mode_b.parent_make_adhesion == 0) {
        return false;
    }
    
    return true;
}

/// Create a new adhesion bond between two cells.
///
/// Initializes a new adhesion connection with proper zone classification,
/// anchor directions, and twist references. Updates adhesion indices and counts.
///
/// # Arguments
/// * `cell_a_idx` - Index of first cell
/// * `cell_b_idx` - Index of second cell
/// * `connection_idx` - Index of connection slot to use
/// * `pos_a` - Position of first cell
/// * `pos_b` - Position of second cell
/// * `orient_a` - Orientation of first cell
/// * `orient_b` - Orientation of second cell
/// * `genome_orient_a` - Genome orientation of first cell
/// * `genome_orient_b` - Genome orientation of second cell
/// * `mode_a` - Genome mode of first cell
/// * `mode_b` - Genome mode of second cell
///
/// # Returns
/// True if bond was successfully created, false otherwise
fn create_adhesion_bond(
    cell_a_idx: u32,
    cell_b_idx: u32,
    connection_idx: u32,
    pos_a: vec3<f32>,
    pos_b: vec3<f32>,
    orient_a: vec4<f32>,
    orient_b: vec4<f32>,
    genome_orient_a: vec4<f32>,
    genome_orient_b: vec4<f32>,
    mode_a: GpuMode,
    mode_b: GpuMode
) -> bool {
    // Find free slots in both cells
    let slot_a = find_free_adhesion_slot(cell_a_idx);
    let slot_b = find_free_adhesion_slot(cell_b_idx);
    
    if (slot_a >= MAX_ADHESIONS_PER_CELL || slot_b >= MAX_ADHESIONS_PER_CELL) {
        return false; // No free slots
    }
    
    // Calculate bond direction and anchor directions
    let bond_vector = pos_b - pos_a;
    let bond_length = length(bond_vector);
    
    if (bond_length < 0.001) {
        return false; // Cells too close
    }
    
    let bond_direction = bond_vector / bond_length;
    
    // Calculate anchor directions in local space
    // For cell A: anchor points toward cell B
    let anchor_direction_a = rotate_vector_by_quaternion(-bond_direction, 
        vec4<f32>(orient_a.x, -orient_a.y, -orient_a.z, -orient_a.w)); // Conjugate for inverse rotation
    
    // For cell B: anchor points toward cell A
    let anchor_direction_b = rotate_vector_by_quaternion(bond_direction,
        vec4<f32>(orient_b.x, -orient_b.y, -orient_b.z, -orient_b.w)); // Conjugate for inverse rotation
    
    // Classify zones using genome orientations and split directions
    let split_direction_a = mode_a.split_direction.xyz;
    let split_direction_b = mode_b.split_direction.xyz;
    
    // Transform split directions to world space using genome orientations
    let world_split_a = rotate_vector_by_quaternion(split_direction_a, genome_orient_a);
    let world_split_b = rotate_vector_by_quaternion(split_direction_b, genome_orient_b);
    
    let zone_a = classify_bond_direction(bond_direction, world_split_a);
    let zone_b = classify_bond_direction(-bond_direction, world_split_b);
    
    // Initialize the connection
    adhesion_connections[connection_idx].cell_a_index = cell_a_idx;
    adhesion_connections[connection_idx].cell_b_index = cell_b_idx;
    adhesion_connections[connection_idx].mode_index = u32(mode_indices[cell_a_idx]); // Use cell A's mode
    adhesion_connections[connection_idx].is_active = 1u;
    adhesion_connections[connection_idx].zone_a = zone_a;
    adhesion_connections[connection_idx].zone_b = zone_b;
    adhesion_connections[connection_idx].anchor_direction_a = anchor_direction_a;
    adhesion_connections[connection_idx].anchor_direction_b = anchor_direction_b;
    adhesion_connections[connection_idx].twist_reference_a = genome_orient_a;
    adhesion_connections[connection_idx].twist_reference_b = genome_orient_b;
    
    // Update adhesion indices
    let base_a = cell_a_idx * MAX_ADHESIONS_PER_CELL;
    let base_b = cell_b_idx * MAX_ADHESIONS_PER_CELL;
    adhesion_indices[base_a + slot_a] = i32(connection_idx);
    adhesion_indices[base_b + slot_b] = i32(connection_idx);
    
    // Update adhesion counts
    atomicAdd(&adhesion_counts[cell_a_idx], 1u);
    atomicAdd(&adhesion_counts[cell_b_idx], 1u);
    
    return true;
}

/// Main compute shader entry point for adhesion bond formation.
///
/// Each thread processes one cell and attempts to form new adhesion bonds
/// with nearby cells found through spatial grid traversal. The formation
/// process respects genome settings, distance criteria, and capacity limits.
///
/// ## Thread Mapping
/// - `global_invocation_id.x` maps directly to cell index
/// - Each thread processes one cell's bond formation attempts
/// - Total threads needed: cell_count (variable, up to capacity)
/// - Workgroups needed: ceil(cell_count / 64)
///
/// ## Bond Formation Pipeline
/// 1. Read cell properties and genome mode
/// 2. Find nearby cells using spatial grid traversal
/// 3. Check formation criteria for each nearby cell
/// 4. Classify adhesion zones based on split directions
/// 5. Create new bonds if criteria are met and slots are available
/// 6. Update adhesion indices and counts atomically
///
/// ## Thread Safety
/// - Atomic operations for adhesion counts and slot allocation
/// - Careful ordering to prevent race conditions
/// - Each thread only modifies its own cell's data primarily
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;
    
    // Bounds check: ensure we don't process beyond the active cell count
    if (cell_index >= physics_params.cell_count) {
        return;
    }
    
    // Read cell properties
    let pos_mass = position_and_mass[cell_index];
    let pos = pos_mass.xyz;
    let mass = pos_mass.w;
    let orientation = orientation[cell_index];
    let genome_orientation = genome_orientation[cell_index];
    let mode_index = mode_indices[cell_index];
    let mode = genome_modes[mode_index];
    
    // Skip if cell can't make adhesions
    if (mode.parent_make_adhesion == 0) {
        return;
    }
    
    // Check if cell has reached maximum adhesions
    let current_adhesion_count = atomicLoad(&adhesion_counts[cell_index]);
    if (current_adhesion_count >= u32(mode.max_adhesions)) {
        return;
    }
    
    // Get grid assignment for this cell
    let grid_assignment = grid_assignments[cell_index];
    let grid_resolution = physics_params.grid_resolution;
    
    // Convert grid index back to coordinates for neighbor traversal
    let grid_z = i32(grid_assignment) / (grid_resolution * grid_resolution);
    let grid_y = (i32(grid_assignment) - grid_z * grid_resolution * grid_resolution) / grid_resolution;
    let grid_x = i32(grid_assignment) - grid_z * grid_resolution * grid_resolution - grid_y * grid_resolution;
    
    // Search neighboring grid cells for potential adhesion partners
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_x = grid_x + dx;
                let neighbor_y = grid_y + dy;
                let neighbor_z = grid_z + dz;
                
                // Check bounds
                if (neighbor_x < 0 || neighbor_x >= grid_resolution ||
                    neighbor_y < 0 || neighbor_y >= grid_resolution ||
                    neighbor_z < 0 || neighbor_z >= grid_resolution) {
                    continue;
                }
                
                // Calculate neighbor grid index
                let neighbor_grid_index = u32(neighbor_x + neighbor_y * grid_resolution + neighbor_z * grid_resolution * grid_resolution);
                
                // Get cells in this grid cell
                let grid_count = grid_counts[neighbor_grid_index];
                let grid_offset = grid_offsets[neighbor_grid_index];
                
                // Check each cell in the grid cell
                for (var i = 0u; i < grid_count && i < u32(physics_params.max_cells_per_grid); i++) {
                    let neighbor_index = grid_indices[grid_offset + i];
                    
                    // Skip if same cell or invalid index
                    if (neighbor_index == cell_index || neighbor_index >= physics_params.cell_count) {
                        continue;
                    }
                    
                    // Read neighbor properties
                    let neighbor_pos_mass = position_and_mass[neighbor_index];
                    let neighbor_pos = neighbor_pos_mass.xyz;
                    let neighbor_orientation = orientation[neighbor_index];
                    let neighbor_genome_orientation = genome_orientation[neighbor_index];
                    let neighbor_mode_index = mode_indices[neighbor_index];
                    let neighbor_mode = genome_modes[neighbor_mode_index];
                    
                    // Check if bond formation is possible
                    if (!can_form_bond(cell_index, neighbor_index, pos, neighbor_pos, mode, neighbor_mode)) {
                        continue;
                    }
                    
                    // Find a free connection slot
                    let connection_index = find_free_connection_slot();
                    if (connection_index >= arrayLength(&adhesion_connections)) {
                        continue; // No free connections available
                    }
                    
                    // Attempt to create the bond
                    let bond_created = create_adhesion_bond(
                        cell_index, neighbor_index, connection_index,
                        pos, neighbor_pos,
                        orientation, neighbor_orientation,
                        genome_orientation, neighbor_genome_orientation,
                        mode, neighbor_mode
                    );
                    
                    // If bond was created and we've reached max adhesions, stop searching
                    if (bond_created) {
                        let new_count = atomicLoad(&adhesion_counts[cell_index]);
                        if (new_count >= u32(mode.max_adhesions)) {
                            return;
                        }
                    }
                }
            }
        }
    }
}
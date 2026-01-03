//! # CPU Physics Engine - Deterministic Spatial Grid and Force Calculations
//! 
//! This module implements the CPU-based physics simulation for Bio-Spheres, designed to match
//! the reference BioSpheres-Q implementation. It uses a deterministic spatial grid for efficient
//! collision detection and supports both single-threaded and parallel execution.
//! 
//! ## Physics Pipeline
//! 
//! The physics simulation follows this sequence each frame:
//! 
//! 1. **Position Integration**: Update positions using Verlet integration
//! 2. **Rotation Update**: Integrate angular velocities to update orientations
//! 3. **Spatial Partitioning**: Rebuild spatial grid for collision detection
//! 4. **Collision Detection**: Find overlapping cells using spatial grid
//! 5. **Force Calculation**: Compute collision, adhesion, and boundary forces
//! 6. **Velocity Integration**: Update velocities from accumulated forces
//! 7. **Angular Integration**: Update angular velocities from torques
//! 8. **Division Processing**: Handle cell division events
//! 
//! ## Deterministic Spatial Grid
//! 
//! The spatial grid divides 3D space into uniform cells to accelerate collision detection.
//! Instead of checking all O(n²) cell pairs, we only check cells in the same or adjacent
//! grid cells, reducing complexity to approximately O(n).
//! 
//! ### Key Features:
//! - **Zero Allocations**: Uses pre-allocated arrays and prefix-sum algorithm
//! - **Deterministic**: Same input always produces same output (important for reproducibility)
//! - **Cache Friendly**: Contiguous memory access patterns
//! - **Parallel Safe**: Can be rebuilt and queried from multiple threads
//! 
//! ## Integration Schemes
//! 
//! ### Verlet Integration
//! Used for position/velocity updates. Provides better stability than Euler integration
//! for collision-heavy scenarios:
//! 
//! ```
//! x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
//! v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
//! ```
//! 
//! ### Benefits:
//! - **Stability**: Better energy conservation in collisions
//! - **Accuracy**: Second-order accurate for smooth forces
//! - **Reversibility**: Time-reversible integration (important for physics)
//! 
//! ## Performance Characteristics
//! 
//! - **Single-threaded**: Good for <1000 cells, simple debugging
//! - **Multi-threaded**: Uses `rayon` for parallel processing of grid cells
//! - **Memory**: Pre-allocated buffers avoid runtime allocations
//! - **Cache**: SoA layout provides excellent cache locality
//! 
//! ## Force Types
//! 
//! 1. **Collision Forces**: Repulsive forces between overlapping cells
//! 2. **Adhesion Forces**: Spring-damper connections between bonded cells
//! 3. **Boundary Forces**: Soft spherical boundary to contain simulation
//! 4. **Gravity**: Optional downward force for realistic settling

// CPU-based physics simulation
// Matches the BioSpheres-Q reference implementation

use glam::{Vec3, Quat, IVec3, UVec3};
use std::collections::HashMap;
use crate::simulation::canonical_state::{CanonicalState, DivisionEvent};
use crate::simulation::physics_config::PhysicsConfig;
use crate::genome::Genome;
use crate::cell::division;

// ============================================================================
// Deterministic Spatial Grid for Collision Detection
// ============================================================================

/// Deterministic spatial grid using fixed-size arrays and prefix-sum algorithm
/// 
/// This spatial partitioning system divides 3D space into a uniform grid to accelerate
/// collision detection. Instead of checking all O(n²) possible cell pairs, we only
/// check cells within the same or adjacent grid cells.
/// 
/// ## Algorithm Overview
/// 
/// 1. **Grid Division**: 3D space is divided into `grid_dimensions³` uniform cells
/// 2. **Cell Assignment**: Each simulation cell is assigned to grid cells based on position
/// 3. **Prefix Sum**: Grid cell contents are stored using prefix-sum for cache efficiency
/// 4. **Neighbor Search**: Collision detection only checks adjacent grid cells
/// 
/// ## Memory Layout
/// 
/// The grid uses a flat array with prefix sums instead of nested vectors:
/// ```
/// Grid Cell 0: [cell_0, cell_3, cell_7]     -> offset=0, count=3
/// Grid Cell 1: [cell_1, cell_5]             -> offset=3, count=2  
/// Grid Cell 2: [cell_2, cell_4, cell_6]     -> offset=5, count=3
/// ```
/// 
/// This provides:
/// - **Cache Efficiency**: Contiguous memory access
/// - **Zero Allocations**: No runtime vector resizing
/// - **Parallel Safety**: Multiple threads can read simultaneously
/// 
/// ## Performance Characteristics
/// 
/// - **Build Time**: O(n) where n is number of cells
/// - **Query Time**: O(k) where k is average cells per grid cell
/// - **Memory**: O(grid_dimensions³ + n) 
/// - **Cache Misses**: Minimal due to contiguous layout
#[derive(Clone)]
pub struct DeterministicSpatialGrid {
    /// Number of grid cells in each dimension (creates dimensions³ total cells)
    pub grid_dimensions: UVec3,
    
    /// Total world size (grid extends from -world_size/2 to +world_size/2)
    pub world_size: f32,
    
    /// Size of each individual grid cell
    pub cell_size: f32,
    
    /// Radius of spherical boundary (for active cell precomputation)
    pub sphere_radius: f32,

    /// Pre-computed list of active grid cell coordinates
    /// 
    /// Only grid cells within the spherical boundary are considered "active".
    /// This avoids processing empty regions of space.
    pub active_cells: Vec<IVec3>,

    /// HashMap for O(1) lookup of active cell index from coordinates
    /// 
    /// Maps grid coordinates (x,y,z) to index in active_cells array.
    /// Used during grid rebuild to quickly find where to place cells.
    pub active_cell_map: HashMap<IVec3, usize>,

    /// Flat array containing all cell indices, organized by grid cell
    /// 
    /// Layout: [grid_0_cells..., grid_1_cells..., grid_2_cells...]
    /// Use cell_offsets to find where each grid cell's data starts.
    pub cell_contents: Vec<usize>,

    /// Starting offset in cell_contents for each active grid cell
    /// 
    /// cell_contents[cell_offsets[i]..cell_offsets[i]+cell_counts[i]]
    /// contains all simulation cells in active grid cell i.
    pub cell_offsets: Vec<usize>,

    /// Number of simulation cells in each active grid cell
    /// 
    /// Used together with cell_offsets to slice cell_contents array.
    /// Reset to zero at start of each rebuild.
    pub cell_counts: Vec<usize>,

    /// Track which grid cells were used in the last rebuild
    /// 
    /// Optimization: only clear counts for grid cells that were actually used,
    /// avoiding O(grid_dimensions³) clear operation each frame.
    pub used_grid_cells: Vec<usize>,
}

impl DeterministicSpatialGrid {
    /// Create a new deterministic spatial grid with default capacity
    /// 
    /// Uses a default capacity of 10,000 cells, which is suitable for most
    /// preview-mode simulations. For larger simulations, use `with_capacity()`.
    /// 
    /// # Arguments
    /// * `grid_dim` - Number of grid cells per dimension (creates grid_dim³ total cells)
    /// * `world_size` - Total world size (grid extends from -world_size/2 to +world_size/2)
    /// * `sphere_radius` - Radius of spherical boundary for active cell computation
    /// 
    /// # Grid Density Guidelines
    /// - 32: Small simulations (<1000 cells)
    /// - 64: Medium simulations (1000-10000 cells)
    /// - 128: Large simulations (>10000 cells)
    pub fn new(grid_dim: u32, world_size: f32, sphere_radius: f32) -> Self {
        Self::with_capacity(grid_dim, world_size, sphere_radius, 10_000)
    }
    
    /// Create a new deterministic spatial grid with specified cell capacity
    /// 
    /// The capacity determines the size of pre-allocated arrays. Choose based on
    /// the maximum expected number of simulation cells, not the initial count.
    /// 
    /// # Arguments
    /// * `grid_dim` - Number of grid cells per dimension
    /// * `world_size` - Total world size  
    /// * `sphere_radius` - Radius of spherical boundary
    /// * `max_cells` - Maximum number of simulation cells to support
    /// 
    /// # Memory Usage
    /// - Grid metadata: ~grid_dim³ * 8 bytes
    /// - Cell storage: max_cells * 4 bytes
    /// - Active cell map: ~grid_dim³ * 24 bytes (HashMap overhead)
    /// 
    /// Total for 64³ grid with 10k cells: ~8MB
    pub fn with_capacity(grid_dim: u32, world_size: f32, sphere_radius: f32, max_cells: usize) -> Self {
        let grid_dimensions = UVec3::splat(grid_dim);
        let cell_size = world_size / grid_dim as f32;

        // Precompute which grid cells are within the spherical boundary
        // This avoids processing empty regions of space during collision detection
        let active_cells = Self::precompute_active_cells(grid_dimensions);
        let active_count = active_cells.len();

        // Build HashMap for O(1) coordinate -> index lookups during rebuild
        let mut active_cell_map = HashMap::new();
        for (idx, &coord) in active_cells.iter().enumerate() {
            active_cell_map.insert(coord, idx);
        }

        Self {
            grid_dimensions,
            world_size,
            cell_size,
            sphere_radius,
            active_cells,
            active_cell_map,
            
            // Pre-allocate arrays to avoid runtime allocations
            cell_contents: vec![0; max_cells],           // Flat array of cell indices
            cell_offsets: vec![0; active_count],         // Starting offset for each grid cell
            cell_counts: vec![0; active_count],          // Count of cells in each grid cell
            used_grid_cells: Vec::with_capacity(max_cells), // Track which cells were used
        }
    }

    /// Precompute which grid cells are active (within simulation bounds)
    /// 
    /// Currently includes all grid cells, but could be optimized to only include
    /// cells within a spherical or other shaped boundary to reduce memory usage
    /// and processing time for sparse simulations.
    /// 
    /// # Returns
    /// Vector of 3D grid coordinates for all active cells
    fn precompute_active_cells(grid_dimensions: UVec3) -> Vec<IVec3> {
        let mut active_cells = Vec::new();

        // Include all grid cells (could be optimized for spherical boundaries)
        for x in 0..grid_dimensions.x as i32 {
            for y in 0..grid_dimensions.y as i32 {
                for z in 0..grid_dimensions.z as i32 {
                    active_cells.push(IVec3::new(x, y, z));
                }
            }
        }

        active_cells
    }

    /// Convert world position to grid coordinates
    /// 
    /// Maps a 3D world position to discrete grid cell coordinates.
    /// Handles boundary clamping to ensure coordinates are always valid.
    /// 
    /// # Arguments
    /// * `position` - World space position
    /// 
    /// # Returns
    /// Grid cell coordinates (clamped to valid range)
    /// 
    /// # Coordinate System
    /// - World space: [-world_size/2, +world_size/2] in each dimension
    /// - Grid space: [0, grid_dimensions-1] in each dimension
    fn world_to_grid(&self, position: Vec3) -> IVec3 {
        // Shift position to [0, world_size] range
        let offset_position = position + Vec3::splat(self.world_size / 2.0);
        
        // Scale to grid coordinates
        let grid_pos = offset_position / self.cell_size;
        
        // Clamp to valid grid range to handle boundary cases
        let max_coord = (self.grid_dimensions.x - 1) as i32;
        IVec3::new(
            (grid_pos.x as i32).clamp(0, max_coord),
            (grid_pos.y as i32).clamp(0, max_coord),
            (grid_pos.z as i32).clamp(0, max_coord),
        )
    }

    /// Find the index of a grid coordinate in the active_cells array
    fn active_cell_index(&self, grid_coord: IVec3) -> Option<usize> {
        self.active_cell_map.get(&grid_coord).copied()
    }

    /// Rebuild the spatial grid using prefix sum algorithm (parallel for large cell counts)
    pub fn rebuild(&mut self, positions: &[Vec3], cell_count: usize) {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        
        // Clear only previously used counts
        for &idx in &self.used_grid_cells {
            self.cell_counts[idx] = 0;
        }
        self.used_grid_cells.clear();

        // Use parallel counting for large cell counts (>500 cells)
        if cell_count > 500 {
            // Parallel counting with atomic operations
            let atomic_counts: Vec<AtomicUsize> = (0..self.active_cells.len())
                .map(|_| AtomicUsize::new(0))
                .collect();
            
            (0..cell_count).into_par_iter().for_each(|i| {
                let grid_coord = self.world_to_grid(positions[i]);
                if let Some(idx) = self.active_cell_index(grid_coord) {
                    atomic_counts[idx].fetch_add(1, Ordering::Relaxed);
                }
            });
            
            // Convert atomic counts and track used cells
            for (idx, atomic_count) in atomic_counts.iter().enumerate() {
                let count = atomic_count.load(Ordering::Relaxed);
                if count > 0 {
                    self.cell_counts[idx] = count;
                    self.used_grid_cells.push(idx);
                }
            }
        } else {
            // Sequential counting for small cell counts
            for i in 0..cell_count {
                let grid_coord = self.world_to_grid(positions[i]);
                if let Some(idx) = self.active_cell_index(grid_coord) {
                    if self.cell_counts[idx] == 0 {
                        self.used_grid_cells.push(idx);
                    }
                    self.cell_counts[idx] += 1;
                }
            }
        }

        // Compute offsets using prefix sum
        let mut offset = 0;
        for &idx in &self.used_grid_cells {
            self.cell_offsets[idx] = offset;
            offset += self.cell_counts[idx];
        }

        // Reset counts for insertion phase
        for &idx in &self.used_grid_cells {
            self.cell_counts[idx] = 0;
        }

        // Parallel insertion for large cell counts
        if cell_count > 500 {
            let atomic_offsets: Vec<AtomicUsize> = self.cell_offsets
                .iter()
                .map(|&offset| AtomicUsize::new(offset))
                .collect();
            
            (0..cell_count).into_par_iter().for_each(|i| {
                let grid_coord = self.world_to_grid(positions[i]);
                if let Some(idx) = self.active_cell_index(grid_coord) {
                    let insert_pos = atomic_offsets[idx].fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        // Safe because each thread writes to a unique position
                        let ptr = self.cell_contents.as_ptr() as *mut usize;
                        *ptr.add(insert_pos) = i;
                    }
                }
            });
            
            // Update counts from atomic offsets
            for (idx, atomic_offset) in atomic_offsets.iter().enumerate() {
                let final_offset = atomic_offset.load(Ordering::Relaxed);
                self.cell_counts[idx] = final_offset - self.cell_offsets[idx];
            }
        } else {
            // Sequential insertion for small cell counts
            for i in 0..cell_count {
                let grid_coord = self.world_to_grid(positions[i]);
                if let Some(idx) = self.active_cell_index(grid_coord) {
                    let insert_pos = self.cell_offsets[idx] + self.cell_counts[idx];
                    self.cell_contents[insert_pos] = i;
                    self.cell_counts[idx] += 1;
                }
            }
        }
    }

    /// Get a slice of cell indices in a specific grid cell
    pub fn get_cell_contents(&self, grid_idx: usize) -> &[usize] {
        let start = self.cell_offsets[grid_idx];
        let count = self.cell_counts[grid_idx];
        &self.cell_contents[start..start + count]
    }
}

// ============================================================================
// Collision Detection
// ============================================================================

/// Collision pair between two cells
#[derive(Clone, Copy, Debug)]
pub struct CollisionPair {
    pub index_a: usize,
    pub index_b: usize,
    pub overlap: f32,
    pub normal: Vec3,
}

/// Forward neighbors for half-space optimization (13 neighbors instead of 27)
const FORWARD_NEIGHBORS: [IVec3; 13] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 1, 0), IVec3::new(0, 1, 0), IVec3::new(1, 1, 0),
    IVec3::new(-1, -1, 1), IVec3::new(0, -1, 1), IVec3::new(1, -1, 1),
    IVec3::new(-1, 0, 1), IVec3::new(0, 0, 1), IVec3::new(1, 0, 1),
    IVec3::new(-1, 1, 1), IVec3::new(0, 1, 1), IVec3::new(1, 1, 1),
];

/// Detect collisions using the deterministic spatial grid - Single-threaded version
pub fn detect_collisions_st(
    state: &CanonicalState,
) -> Vec<CollisionPair> {
    let mut collision_pairs = Vec::new();

    // Iterate through only the grid cells that contain simulation cells
    for &grid_idx in &state.spatial_grid.used_grid_cells {
        let grid_coord = state.spatial_grid.active_cells[grid_idx];
        let cells_in_grid = state.spatial_grid.get_cell_contents(grid_idx);

        // Check collisions within the same grid cell
        for i in 0..cells_in_grid.len() {
            let idx_a = cells_in_grid[i];
            for j in (i + 1)..cells_in_grid.len() {
                let idx_b = cells_in_grid[j];

                let delta = state.positions[idx_b] - state.positions[idx_a];
                let distance = delta.length();
                let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                if distance < combined_radius {
                    // Skip collision if cells are in the same organism
                    // (adhesion forces handle their interaction instead)
                    if are_cells_in_same_organism(state, idx_a, idx_b) {
                        continue;
                    }
                    
                    let overlap = combined_radius - distance;
                    let normal = if distance > 0.0001 {
                        delta / distance
                    } else {
                        Vec3::X
                    };

                    collision_pairs.push(CollisionPair {
                        index_a: idx_a,
                        index_b: idx_b,
                        overlap,
                        normal,
                    });
                }
            }
        }

        // Check forward neighbors to avoid duplicate checks
        for &offset in &FORWARD_NEIGHBORS {
            let neighbor_coord = grid_coord + offset;

            let Some(neighbor_idx) = state.spatial_grid.active_cell_index(neighbor_coord) else {
                continue;
            };

            let neighbor_cells = state.spatial_grid.get_cell_contents(neighbor_idx);

            for &idx_a in cells_in_grid {
                for &idx_b in neighbor_cells {
                    let delta = state.positions[idx_b] - state.positions[idx_a];
                    let distance = delta.length();
                    let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                    if distance < combined_radius {
                        // Skip collision if cells are in the same organism
                        // (adhesion forces handle their interaction instead)
                        if are_cells_in_same_organism(state, idx_a, idx_b) {
                            continue;
                        }
                        
                        let overlap = combined_radius - distance;
                        let normal = if distance > 0.0001 {
                            delta / distance
                        } else {
                            Vec3::X
                        };

                        collision_pairs.push(CollisionPair {
                            index_a: idx_a,
                            index_b: idx_b,
                            overlap,
                            normal,
                        });
                    }
                }
            }
        }
    }

    // Sort collision pairs by indices to maintain deterministic ordering
    collision_pairs.sort_unstable_by_key(|pair| (pair.index_a, pair.index_b));

    collision_pairs
}

/// Detect collisions using the deterministic spatial grid - Multithreaded version
/// Uses parallel iteration over grid cells for improved performance with large cell counts.
pub fn detect_collisions_parallel(
    state: &CanonicalState,
) -> Vec<CollisionPair> {
    use rayon::prelude::*;
    
    // Process each grid cell in parallel
    let mut collision_pairs: Vec<CollisionPair> = state.spatial_grid.used_grid_cells
        .par_iter()
        .flat_map(|&grid_idx| {
            let mut local_pairs = Vec::new();
            let grid_coord = state.spatial_grid.active_cells[grid_idx];
            let cells_in_grid = state.spatial_grid.get_cell_contents(grid_idx);

            // Check collisions within the same grid cell
            for i in 0..cells_in_grid.len() {
                let idx_a = cells_in_grid[i];
                for j in (i + 1)..cells_in_grid.len() {
                    let idx_b = cells_in_grid[j];

                    let delta = state.positions[idx_b] - state.positions[idx_a];
                    let distance = delta.length();
                    let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                    if distance < combined_radius {
                        // Skip collision if cells are in the same organism
                        if are_cells_in_same_organism(state, idx_a, idx_b) {
                            continue;
                        }
                        
                        let overlap = combined_radius - distance;
                        let normal = if distance > 0.0001 {
                            delta / distance
                        } else {
                            Vec3::X
                        };

                        local_pairs.push(CollisionPair {
                            index_a: idx_a,
                            index_b: idx_b,
                            overlap,
                            normal,
                        });
                    }
                }
            }

            // Check forward neighbors
            for &offset in &FORWARD_NEIGHBORS {
                let neighbor_coord = grid_coord + offset;

                let Some(neighbor_idx) = state.spatial_grid.active_cell_index(neighbor_coord) else {
                    continue;
                };

                let neighbor_cells = state.spatial_grid.get_cell_contents(neighbor_idx);

                for &idx_a in cells_in_grid {
                    for &idx_b in neighbor_cells {
                        let delta = state.positions[idx_b] - state.positions[idx_a];
                        let distance = delta.length();
                        let combined_radius = state.radii[idx_a] + state.radii[idx_b];

                        if distance < combined_radius {
                            if are_cells_in_same_organism(state, idx_a, idx_b) {
                                continue;
                            }
                            
                            let overlap = combined_radius - distance;
                            let normal = if distance > 0.0001 {
                                delta / distance
                            } else {
                                Vec3::X
                            };

                            local_pairs.push(CollisionPair {
                                index_a: idx_a,
                                index_b: idx_b,
                                overlap,
                                normal,
                            });
                        }
                    }
                }
            }
            
            local_pairs
        })
        .collect();

    // Sort for deterministic ordering
    collision_pairs.sort_unstable_by_key(|pair| (pair.index_a, pair.index_b));

    collision_pairs
}

/// Check if two cells are in the same organism (connected via adhesions)
/// Cells in the same organism rely on adhesion forces for their interactions
#[inline]
fn are_cells_in_same_organism(state: &CanonicalState, cell_a: usize, cell_b: usize) -> bool {
    state.adhesion_manager.are_cells_connected(&state.adhesion_connections, cell_a, cell_b)
}

// ============================================================================
// Force Computation
// ============================================================================

/// Compute collision forces from detected collision pairs
pub fn compute_collision_forces(
    state: &mut CanonicalState,
    collision_pairs: &[CollisionPair],
    config: &PhysicsConfig,
) {
    // Clear all forces and torques
    for i in 0..state.cell_count {
        state.forces[i] = Vec3::ZERO;
        state.torques[i] = Vec3::ZERO;
    }

    // Process each collision pair
    for pair in collision_pairs {
        let idx_a = pair.index_a;
        let idx_b = pair.index_b;

        let stiffness_a = state.stiffnesses[idx_a];
        let stiffness_b = state.stiffnesses[idx_b];

        // Compute combined stiffness using harmonic mean
        let combined_stiffness = if stiffness_a > 0.0 && stiffness_b > 0.0 {
            (stiffness_a * stiffness_b) / (stiffness_a + stiffness_b)
        } else if stiffness_a > 0.0 {
            stiffness_a
        } else if stiffness_b > 0.0 {
            stiffness_b
        } else {
            // Both cells have zero stiffness - use zero (no repulsion)
            0.0
        };

        // Calculate spring force
        let spring_force_magnitude = combined_stiffness * pair.overlap;

        // Calculate relative velocity along collision normal
        let relative_velocity = state.velocities[idx_b] - state.velocities[idx_a];
        let relative_velocity_normal = relative_velocity.dot(pair.normal);

        // Calculate damping force
        let damping_force_magnitude = -config.damping * relative_velocity_normal;

        // Total force magnitude
        let total_force_magnitude = spring_force_magnitude + damping_force_magnitude;

        // Clamp force magnitude
        let max_force = 10000.0;
        let clamped_force_magnitude = total_force_magnitude.clamp(-max_force, max_force);
        let force = clamped_force_magnitude * pair.normal;

        // Apply equal and opposite forces
        state.forces[idx_b] += force;
        state.forces[idx_a] -= force;

        // Rolling friction torque
        if config.friction_coefficient > 0.0 && pair.overlap > 0.0 {
            let contact_offset_a = pair.normal * state.radii[idx_a];
            let contact_offset_b = -pair.normal * state.radii[idx_b];

            let vel_at_contact_a = state.velocities[idx_a] +
                state.angular_velocities[idx_a].cross(contact_offset_a);
            let vel_at_contact_b = state.velocities[idx_b] +
                state.angular_velocities[idx_b].cross(contact_offset_b);

            let relative_vel_at_contact = vel_at_contact_b - vel_at_contact_a;
            let tangential_velocity = relative_vel_at_contact - pair.normal * relative_vel_at_contact.dot(pair.normal);
            let tangential_speed = tangential_velocity.length();

            if tangential_speed > 0.0001 {
                let tangent_direction = tangential_velocity / tangential_speed;
                let max_friction_torque = config.friction_coefficient * clamped_force_magnitude.abs();

                let torque_axis_a = contact_offset_a.cross(tangent_direction);
                let torque_axis_b = contact_offset_b.cross(tangent_direction);

                let torque_magnitude = (tangential_speed * state.radii[idx_a]).min(max_friction_torque);

                let resistance_torque_a = -torque_axis_a.normalize_or_zero() * torque_magnitude;
                let resistance_torque_b = -torque_axis_b.normalize_or_zero() * torque_magnitude;

                state.torques[idx_a] += resistance_torque_a;
                state.torques[idx_b] += resistance_torque_b;
            }
        }
    }
}

// ============================================================================
// Boundary Forces
// ============================================================================

/// Apply boundary forces that push cells toward center near the spherical boundary
/// 
/// Creates a smooth inward force that increases as cells approach the boundary.
/// The force activates in a "soft zone" near the boundary and becomes stronger closer to the edge.
/// Also applies torque to rotate cells to face inward.
pub fn apply_boundary_forces(
    state: &mut CanonicalState,
    config: &PhysicsConfig,
) {
    // Boundary force parameters
    let boundary_radius = config.sphere_radius;
    let soft_zone_thickness = 5.0; // Start applying force 5 units before boundary
    let soft_zone_start = boundary_radius - soft_zone_thickness;
    let max_boundary_force = 500.0; // Maximum inward force at the boundary

    for i in 0..state.cell_count {
        let distance_from_origin = state.positions[i].length();

        // Skip if at origin
        if distance_from_origin < 0.0001 {
            continue;
        }

        let r_hat = state.positions[i] / distance_from_origin;

        // Apply smooth inward force in the soft zone
        if distance_from_origin > soft_zone_start {
            // Calculate how far into the soft zone (0.0 at start, 1.0 at boundary)
            let penetration = (distance_from_origin - soft_zone_start) / soft_zone_thickness;
            let penetration_clamped = penetration.clamp(0.0, 1.0);

            // Quadratic force curve: gentle at first, strong near boundary
            let force_magnitude = max_boundary_force * penetration_clamped * penetration_clamped;

            // Apply inward force (negative radial direction)
            let inward_force = -r_hat * force_magnitude;
            // CRITICAL: Use hardcoded 0.016 to match reference implementation exactly
            state.velocities[i] += inward_force * 0.016;

            // Apply torque to rotate cell to face inward
            // Get the cell's forward direction (local +Z axis)
            let forward = state.rotations[i] * Vec3::Z;

            // Calculate desired direction (toward center)
            let desired_direction = -r_hat;

            // Calculate rotation axis (cross product of current forward and desired direction)
            let rotation_axis = forward.cross(desired_direction);
            let rotation_axis_length = rotation_axis.length();

            // Only apply torque if there's a meaningful rotation needed
            if rotation_axis_length > 0.001 {
                let normalized_axis = rotation_axis / rotation_axis_length;

                // Calculate angle between current and desired direction
                let dot_product = forward.dot(desired_direction).clamp(-1.0, 1.0);
                let angle = dot_product.acos();

                // Torque magnitude increases with penetration and angle
                let torque_strength = 50.0 * penetration_clamped * angle;

                // Apply torque to rotate toward center
                state.torques[i] += normalized_axis * torque_strength;
            }
        }

        // Hard clamp: if somehow past boundary, push back and reverse velocity
        if distance_from_origin > boundary_radius {
            state.positions[i] = r_hat * boundary_radius;

            // Reverse any outward velocity component
            let radial_velocity = state.velocities[i].dot(r_hat);
            if radial_velocity > 0.0 {
                state.velocities[i] -= r_hat * radial_velocity * 2.0; // Remove and reverse
            }
        }
    }
}

// ============================================================================
// Integration Functions
// ============================================================================

/// Verlet integration position update
pub fn verlet_integrate_positions(
    positions: &mut [Vec3],
    velocities: &[Vec3],
    accelerations: &[Vec3],
    dt: f32,
) {
    let dt_sq = dt * dt;
    for i in 0..positions.len() {
        if velocities[i].is_finite() && accelerations[i].is_finite() {
            positions[i] += velocities[i] * dt + 0.5 * accelerations[i] * dt_sq;
        }
    }
}

/// Verlet integration velocity update
pub fn verlet_integrate_velocities(
    velocities: &mut [Vec3],
    accelerations: &mut [Vec3],
    prev_accelerations: &mut [Vec3],
    forces: &[Vec3],
    masses: &[f32],
    dt: f32,
    velocity_damping: f32,
) {
    let velocity_damping_factor = velocity_damping.powf(dt * 100.0);

    for i in 0..velocities.len() {
        if masses[i] <= 0.0 || !masses[i].is_finite() {
            continue;
        }
        
        let old_acceleration = accelerations[i];
        let new_acceleration = forces[i] / masses[i];

        if new_acceleration.is_finite() {
            let velocity_change = 0.5 * (old_acceleration + new_acceleration) * dt;
            velocities[i] = (velocities[i] + velocity_change) * velocity_damping_factor;
            accelerations[i] = new_acceleration;
            prev_accelerations[i] = old_acceleration;
        }
    }
}

/// Update rotations from angular velocities
pub fn integrate_rotations(
    rotations: &mut [Quat],
    angular_velocities: &[Vec3],
    dt: f32,
) {
    for i in 0..rotations.len() {
        let ang_vel = angular_velocities[i];
        if ang_vel.length_squared() > 0.0001 {
            let angle = ang_vel.length() * dt;
            let axis = ang_vel.normalize();
            let delta_rotation = Quat::from_axis_angle(axis, angle);
            rotations[i] = (delta_rotation * rotations[i]).normalize();
        }
    }
}

/// Update angular velocities from torques (SoA version) - Single-threaded
pub fn integrate_angular_velocities(
    angular_velocities: &mut [Vec3],
    torques: &[Vec3],
    radii: &[f32],
    masses: &[f32],
    dt: f32,
    angular_damping: f32,
) {
    let angular_damping_factor = angular_damping.powf(dt * 100.0);

    for i in 0..angular_velocities.len() {
        if masses[i] <= 0.0 || !masses[i].is_finite() || radii[i] <= 0.0 {
            continue;
        }

        // Moment of inertia for a sphere: I = (2/5) * m * r²
        let moment_of_inertia = 0.4 * masses[i] * radii[i] * radii[i];

        if moment_of_inertia > 0.0 {
            let angular_acceleration = torques[i] / moment_of_inertia;
            angular_velocities[i] = (angular_velocities[i] + angular_acceleration * dt) * angular_damping_factor;
        }
    }
}

// ============================================================================
// Nutrient Growth
// ============================================================================

/// Update nutrient growth for cells
pub fn update_nutrient_growth(
    state: &mut CanonicalState,
    genome: &Genome,
    dt: f32,
) {
    for i in 0..state.cell_count {
        let mode_index = state.mode_indices[i];
        if let Some(mode) = genome.modes.get(mode_index) {
            // Only grow if we haven't reached the maximum size yet
            let current_mass = state.masses[i];
            let max_mass = mode.max_cell_size;
            
            if current_mass < max_mass {
                // Cells gain mass over time based on their mode's nutrient_gain_rate
                let mass_gain = mode.nutrient_gain_rate * dt;
                if mass_gain > 0.0 {
                    let new_mass = (current_mass + mass_gain).min(max_mass);
                    
                    // Only update if mass actually changed
                    if new_mass != current_mass {
                        state.masses[i] = new_mass;
                        
                        // Update radius based on mass (clamped to reasonable bounds)
                        let new_radius = new_mass.clamp(0.5, 2.0);
                        if new_radius != state.radii[i] {
                            state.radii[i] = new_radius;
                            state.masses_changed = true; // Mark that masses/radii have changed
                        }
                    }
                }
            }
        }
    }
}

/// Update nutrient growth for cells with multiple genomes
/// Each cell uses its genome_id to look up the correct genome for growth parameters
pub fn update_nutrient_growth_multi(
    state: &mut CanonicalState,
    genomes: &[Genome],
    dt: f32,
) {
    for i in 0..state.cell_count {
        let genome_id = state.genome_ids[i];
        let mode_index = state.mode_indices[i];
        
        if let Some(genome) = genomes.get(genome_id) {
            if let Some(mode) = genome.modes.get(mode_index) {
                // Only grow if we haven't reached the maximum size yet
                let current_mass = state.masses[i];
                let max_mass = mode.max_cell_size;
                
                if current_mass < max_mass {
                    // Cells gain mass over time based on their mode's nutrient_gain_rate
                    let mass_gain = mode.nutrient_gain_rate * dt;
                    if mass_gain > 0.0 {
                        let new_mass = (current_mass + mass_gain).min(max_mass);
                        
                        // Only update if mass actually changed
                        if new_mass != current_mass {
                            state.masses[i] = new_mass;
                            
                            // Update radius based on mass (clamped to reasonable bounds)
                            let new_radius = new_mass.clamp(0.5, 2.0);
                            if new_radius != state.radii[i] {
                                state.radii[i] = new_radius;
                                state.masses_changed = true; // Mark that masses/radii have changed
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Main Physics Step
// ============================================================================

/// Single-threaded physics step function
/// 
/// Performs one complete physics simulation step using CPU-based calculations.
/// This version is simpler and easier to debug than the parallel version, making
/// it suitable for small simulations and development.
/// 
/// ## Physics Pipeline
/// 
/// 1. **Verlet Position Integration**: Update positions from velocities and accelerations
/// 2. **Rotation Integration**: Update orientations from angular velocities  
/// 3. **Spatial Grid Rebuild**: Update collision detection data structure
/// 4. **Collision Detection**: Find overlapping cell pairs using spatial grid
/// 5. **Force Calculation**: Compute repulsive forces between colliding cells
/// 6. **Boundary Forces**: Apply soft spherical boundary constraints
/// 7. **Verlet Velocity Integration**: Update velocities from accumulated forces
/// 8. **Angular Integration**: Update angular velocities from torques
/// 
/// ## Performance Characteristics
/// 
/// - **Suitable for**: <1000 cells, debugging, deterministic testing
/// - **Time Complexity**: O(n) for most operations, O(n*k) for collisions where k is average neighbors
/// - **Memory**: No additional allocations during simulation
/// 
/// ## Integration Scheme
/// 
/// Uses Verlet integration for better stability in collision-heavy scenarios:
/// - Position: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
/// - Velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
/// 
/// # Arguments
/// * `state` - Mutable reference to simulation state (SoA layout)
/// * `config` - Physics parameters (timestep, damping, forces)
pub fn physics_step_st(
    state: &mut CanonicalState,
    config: &PhysicsConfig,
) {
    let dt = config.fixed_timestep;

    // 1. Verlet integration (position update)
    // Update positions based on current velocities and accelerations
    // This must happen first to maintain Verlet integration stability
    verlet_integrate_positions(
        &mut state.positions[..state.cell_count],
        &state.velocities[..state.cell_count],
        &state.accelerations[..state.cell_count],
        dt,
    );

    // 2. Update rotations from angular velocities
    // Integrate rotational motion independently of linear motion
    integrate_rotations(
        &mut state.rotations[..state.cell_count],
        &state.angular_velocities[..state.cell_count],
        dt,
    );

    // 3. Update spatial partitioning
    // Rebuild the spatial grid with new positions for efficient collision detection
    // This is O(n) and must be done after position updates
    state.spatial_grid.rebuild(&state.positions, state.cell_count);

    // 4. Detect collisions
    // Find all overlapping cell pairs using the spatial grid
    // Single-threaded version for simplicity and determinism
    let collisions = detect_collisions_st(state);

    // 5. Compute forces and torques
    // Calculate repulsive forces between colliding cells
    // Forces are accumulated in state.forces array
    compute_collision_forces(state, &collisions, config);

    // 6. Apply boundary conditions
    // Soft spherical boundary prevents cells from escaping simulation volume
    // Also applies torques to orient cells toward center
    apply_boundary_forces(state, config);

    // 7. Verlet integration (velocity update)
    // Update velocities based on accumulated forces
    // This completes the Verlet integration cycle
    verlet_integrate_velocities(
        &mut state.velocities[..state.cell_count],
        &mut state.accelerations[..state.cell_count],
        &mut state.prev_accelerations[..state.cell_count],
        &state.forces[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.velocity_damping,
    );

    // 8. Update angular velocities from torques
    // Integrate rotational dynamics with damping
    integrate_angular_velocities(
        &mut state.angular_velocities[..state.cell_count],
        &state.torques[..state.cell_count],
        &state.radii[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.angular_damping,
    );
}

/// Physics step with genome-based features (nutrient growth, division)
/// 
/// Extended physics simulation that includes biological behaviors defined by genomes.
/// This version adds nutrient accumulation, cell division, and adhesion forces to
/// the basic physics simulation.
/// 
/// ## Additional Features Beyond Basic Physics
/// 
/// - **Nutrient Growth**: Cells accumulate mass over time based on genome parameters
/// - **Cell Division**: Cells divide when reaching mass and time thresholds
/// - **Adhesion Forces**: Spring-damper forces between connected cells
/// - **Genome Integration**: Cell behavior driven by genome mode settings
/// 
/// ## Division Processing
/// 
/// Cell division is handled in multiple passes to avoid conflicts:
/// 1. Identify cells ready to divide (mass + time thresholds)
/// 2. Filter out conflicting divisions (prevent cascade effects)
/// 3. Create new cells and inherit adhesion connections
/// 4. Return division events for external processing
/// 
/// ## Performance Notes
/// 
/// Uses parallel versions of collision detection and adhesion force calculation
/// for better performance with larger cell populations.
/// 
/// # Arguments
/// * `state` - Mutable reference to simulation state
/// * `genome` - Genome defining cell behaviors
/// * `config` - Physics parameters
/// * `current_time` - Current simulation time (for division timing)
/// 
/// # Returns
/// Vector of division events that occurred this step
pub fn physics_step_with_genome(
    state: &mut CanonicalState,
    genome: &Genome,
    config: &PhysicsConfig,
    current_time: f32,
) -> Vec<DivisionEvent> {
    let dt = config.fixed_timestep;

    // 1. Verlet integration (position update)
    verlet_integrate_positions(
        &mut state.positions[..state.cell_count],
        &state.velocities[..state.cell_count],
        &state.accelerations[..state.cell_count],
        dt,
    );

    // 2. Update rotations from angular velocities
    integrate_rotations(
        &mut state.rotations[..state.cell_count],
        &state.angular_velocities[..state.cell_count],
        dt,
    );

    // 3. Update spatial partitioning
    state.spatial_grid.rebuild(&state.positions, state.cell_count);

    // 4. Detect collisions (use parallel version for better performance)
    let collisions = detect_collisions_parallel(state);

    // 5. Compute collision forces and torques
    compute_collision_forces(state, &collisions, config);

    // 5.5. Compute adhesion forces (biological cell-cell connections)
    // Update adhesion settings cache if genome changed
    state.update_adhesion_settings_cache(genome);
    
    // Update membrane stiffness from genome mode settings
    state.update_membrane_stiffness_from_genome(genome);
    
    // Apply adhesion forces using parallel version for better performance
    if state.adhesion_connections.active_count > 0 && !state.cached_adhesion_settings.is_empty() {
        crate::cell::compute_adhesion_forces_parallel(
            &state.adhesion_connections,
            &state.positions[..state.cell_count],
            &state.velocities[..state.cell_count],
            &state.rotations[..state.cell_count],
            &state.angular_velocities[..state.cell_count],
            &state.masses[..state.cell_count],
            &state.cached_adhesion_settings,
            &mut state.forces[..state.cell_count],
            &mut state.torques[..state.cell_count],
        );
    }

    // 6. Apply boundary conditions
    apply_boundary_forces(state, config);

    // 7. Verlet integration (velocity update)
    verlet_integrate_velocities(
        &mut state.velocities[..state.cell_count],
        &mut state.accelerations[..state.cell_count],
        &mut state.prev_accelerations[..state.cell_count],
        &state.forces[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.velocity_damping,
    );

    // 8. Update angular velocities from torques
    integrate_angular_velocities(
        &mut state.angular_velocities[..state.cell_count],
        &state.torques[..state.cell_count],
        &state.radii[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.angular_damping,
    );

    // 9. Update nutrient growth
    update_nutrient_growth(state, genome, dt);

    // 10. Run division step
    let max_cells = state.capacity;
    let rng_seed = 12345;
    division::division_step(state, genome, current_time, max_cells, rng_seed)
}


/// Physics step with multiple genomes support
/// Each cell uses its genome_id to look up the correct genome for division/adhesion
pub fn physics_step_with_genomes(
    state: &mut CanonicalState,
    genomes: &[Genome],
    config: &PhysicsConfig,
    current_time: f32,
) -> Vec<DivisionEvent> {
    if genomes.is_empty() {
        return Vec::new();
    }
    
    let dt = config.fixed_timestep;

    // 1. Verlet integration (position update)
    verlet_integrate_positions(
        &mut state.positions[..state.cell_count],
        &state.velocities[..state.cell_count],
        &state.accelerations[..state.cell_count],
        dt,
    );

    // 2. Update rotations from angular velocities
    integrate_rotations(
        &mut state.rotations[..state.cell_count],
        &state.angular_velocities[..state.cell_count],
        dt,
    );

    // 3. Update spatial partitioning
    state.spatial_grid.rebuild(&state.positions, state.cell_count);

    // 4. Detect collisions (use parallel version for better performance)
    let collisions = detect_collisions_parallel(state);

    // 5. Compute forces and torques
    compute_collision_forces(state, &collisions, config);

    // 5.5. Compute adhesion forces
    // Update adhesion settings cache from all genomes (combined mode indices)
    state.update_adhesion_settings_cache_multi(genomes);
    
    // Update membrane stiffness from all genomes
    state.update_membrane_stiffness_from_genomes(genomes);
    
    // Apply adhesion forces using parallel version for better performance
    if state.adhesion_connections.active_count > 0 && !state.cached_adhesion_settings.is_empty() {
        crate::cell::compute_adhesion_forces_parallel(
            &state.adhesion_connections,
            &state.positions[..state.cell_count],
            &state.velocities[..state.cell_count],
            &state.rotations[..state.cell_count],
            &state.angular_velocities[..state.cell_count],
            &state.masses[..state.cell_count],
            &state.cached_adhesion_settings,
            &mut state.forces[..state.cell_count],
            &mut state.torques[..state.cell_count],
        );
    }

    // 6. Apply boundary conditions
    apply_boundary_forces(state, config);

    // 7. Verlet integration (velocity update)
    verlet_integrate_velocities(
        &mut state.velocities[..state.cell_count],
        &mut state.accelerations[..state.cell_count],
        &mut state.prev_accelerations[..state.cell_count],
        &state.forces[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.velocity_damping,
    );

    // 8. Update angular velocities from torques
    integrate_angular_velocities(
        &mut state.angular_velocities[..state.cell_count],
        &state.torques[..state.cell_count],
        &state.radii[..state.cell_count],
        &state.masses[..state.cell_count],
        dt,
        config.angular_damping,
    );

    // 9. Update nutrient growth with per-cell genome support
    update_nutrient_growth_multi(state, genomes, dt);

    // 10. Run division step with multiple genomes
    let max_cells = state.capacity;
    let rng_seed = 12345;
    division::division_step_multi(state, genomes, current_time, max_cells, rng_seed)
}

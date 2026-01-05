//! Deterministic Spatial Grid for Collision Detection
//! 
//! This spatial partitioning system divides 3D space into a uniform grid to accelerate
//! collision detection. Used by preview scene for CPU-based physics.

use glam::{Vec3, IVec3, UVec3};
use std::collections::HashMap;

/// Deterministic spatial grid using fixed-size arrays and prefix-sum algorithm
#[derive(Clone)]
pub struct DeterministicSpatialGrid {
    /// Number of grid cells in each dimension
    pub grid_dimensions: UVec3,
    /// Total world size
    pub world_size: f32,
    /// Size of each grid cell
    pub cell_size: f32,
    /// Radius of spherical boundary
    pub sphere_radius: f32,
    /// Pre-computed list of active grid cell coordinates
    pub active_cells: Vec<IVec3>,
    /// HashMap for O(1) lookup of active cell index
    pub active_cell_map: HashMap<IVec3, usize>,
    /// Flat array containing all cell indices
    pub cell_contents: Vec<usize>,
    /// Starting offset for each active grid cell
    pub cell_offsets: Vec<usize>,
    /// Number of cells in each active grid cell
    pub cell_counts: Vec<usize>,
    /// Track which grid cells were used
    pub used_grid_cells: Vec<usize>,
}

impl DeterministicSpatialGrid {
    pub fn new(grid_dim: u32, world_size: f32, sphere_radius: f32) -> Self {
        Self::with_capacity(grid_dim, world_size, sphere_radius, 10_000)
    }
    
    pub fn with_capacity(grid_dim: u32, world_size: f32, sphere_radius: f32, max_cells: usize) -> Self {
        let grid_dimensions = UVec3::splat(grid_dim);
        let cell_size = world_size / grid_dim as f32;

        let active_cells = Self::precompute_active_cells(grid_dimensions);
        let active_count = active_cells.len();

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
            cell_contents: vec![0; max_cells],
            cell_offsets: vec![0; active_count],
            cell_counts: vec![0; active_count],
            used_grid_cells: Vec::with_capacity(max_cells),
        }
    }

    fn precompute_active_cells(grid_dimensions: UVec3) -> Vec<IVec3> {
        let mut active_cells = Vec::new();
        for x in 0..grid_dimensions.x as i32 {
            for y in 0..grid_dimensions.y as i32 {
                for z in 0..grid_dimensions.z as i32 {
                    active_cells.push(IVec3::new(x, y, z));
                }
            }
        }
        active_cells
    }

    fn world_to_grid(&self, position: Vec3) -> IVec3 {
        let offset_position = position + Vec3::splat(self.world_size / 2.0);
        let grid_pos = offset_position / self.cell_size;
        let max_coord = (self.grid_dimensions.x - 1) as i32;
        IVec3::new(
            (grid_pos.x as i32).clamp(0, max_coord),
            (grid_pos.y as i32).clamp(0, max_coord),
            (grid_pos.z as i32).clamp(0, max_coord),
        )
    }

    pub fn active_cell_index(&self, grid_coord: IVec3) -> Option<usize> {
        self.active_cell_map.get(&grid_coord).copied()
    }

    pub fn rebuild(&mut self, positions: &[Vec3], cell_count: usize) {
        // Clear only previously used counts
        for &idx in &self.used_grid_cells {
            self.cell_counts[idx] = 0;
        }
        self.used_grid_cells.clear();

        // Count cells per grid cell
        for i in 0..cell_count {
            let grid_coord = self.world_to_grid(positions[i]);
            if let Some(idx) = self.active_cell_index(grid_coord) {
                if self.cell_counts[idx] == 0 {
                    self.used_grid_cells.push(idx);
                }
                self.cell_counts[idx] += 1;
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

        // Insert cell indices
        for i in 0..cell_count {
            let grid_coord = self.world_to_grid(positions[i]);
            if let Some(idx) = self.active_cell_index(grid_coord) {
                let insert_pos = self.cell_offsets[idx] + self.cell_counts[idx];
                self.cell_contents[insert_pos] = i;
                self.cell_counts[idx] += 1;
            }
        }
    }

    pub fn get_cell_contents(&self, grid_idx: usize) -> &[usize] {
        let start = self.cell_offsets[grid_idx];
        let count = self.cell_counts[grid_idx];
        &self.cell_contents[start..start + count]
    }
}
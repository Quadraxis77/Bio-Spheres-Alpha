//! Adhesion Integration for GPU Scene
//!
//! This module provides the integration layer between the adhesion system
//! and the GPU physics pipeline. It handles:
//!
//! 1. **Sibling adhesion creation** during cell division
//! 2. **Adhesion inheritance** when parent cells divide
//! 3. **Adhesion physics** execution in the compute pipeline
//! 4. **Adhesion removal** when cells die
//!
//! ## Deterministic Execution
//!
//! The adhesion system is designed to be fully deterministic:
//! - Same input always produces same output
//! - No race conditions or non-deterministic atomics
//! - Uses prefix-sum compaction for slot allocation
//!
//! ## Integration Points
//!
//! ### Division Pipeline
//! After `lifecycle_division_execute.wgsl` creates child cells:
//! 1. CPU reads division results (which cells divided)
//! 2. CPU creates sibling adhesions deterministically
//! 3. CPU syncs adhesion buffers to GPU
//!
//! ### Physics Pipeline
//! Between collision detection and position integration:
//! 1. `adhesion_physics.wgsl` computes spring-damper forces
//! 2. Forces are accumulated to cell force buffers
//! 3. Position/velocity integration includes adhesion forces
//!
//! ### Death Pipeline
//! When cells die:
//! 1. CPU identifies dead cells
//! 2. CPU removes adhesions connected to dead cells
//! 3. CPU syncs adhesion buffers to GPU

use super::adhesion::AdhesionZone;
use super::adhesion_buffers::AdhesionBuffers;

/// Adhesion creation request for deterministic processing
#[derive(Debug, Clone)]
pub struct AdhesionCreationRequest {
    /// Index of cell A (parent slot for sibling bonds)
    pub cell_a_index: u32,
    /// Index of cell B (child slot for sibling bonds)
    pub cell_b_index: u32,
    /// Mode index for adhesion settings lookup
    pub mode_index: u32,
    /// Anchor direction for cell A in local space
    pub anchor_a: glam::Vec3,
    /// Anchor direction for cell B in local space
    pub anchor_b: glam::Vec3,
    /// Twist reference quaternion for cell A
    pub twist_ref_a: glam::Quat,
    /// Twist reference quaternion for cell B
    pub twist_ref_b: glam::Quat,
    /// Zone classification for cell A
    pub zone_a: AdhesionZone,
    /// Zone classification for cell B
    pub zone_b: AdhesionZone,
}

impl AdhesionCreationRequest {
    /// Create a sibling adhesion request (between newly divided cells)
    pub fn new_sibling(
        child_a_index: u32,
        child_b_index: u32,
        mode_index: u32,
        parent_rotation: glam::Quat,
        child_a_orientation: glam::Quat,
        child_b_orientation: glam::Quat,
        _split_direction: glam::Vec3,
    ) -> Self {
        // Anchor directions in child local space.
        // The child's world orientation = parent * split_rotation * child.orientation
        // The split direction in world = parent * split_rotation * Vec3::Z
        // So in the child's local frame, the split axis is just:
        //   child.orientation.inverse() * Vec3::Z
        // Child A is at +split, points toward B (at -split): use -Z
        // Child B is at -split, points toward A (at +split): use +Z
        let anchor_a = (child_a_orientation.inverse() * -glam::Vec3::Z).normalize();
        let anchor_b = (child_b_orientation.inverse() * glam::Vec3::Z).normalize();
        
        // Calculate child rotations for twist reference
        let child_a_rotation = parent_rotation * child_a_orientation;
        let child_b_rotation = parent_rotation * child_b_orientation;
        
        // Zone classification for sibling bonds
        // Child A is at +offset (Zone B), Child B is at -offset (Zone A)
        let zone_a = AdhesionZone::ZoneB;
        let zone_b = AdhesionZone::ZoneA;
        
        Self {
            cell_a_index: child_a_index,
            cell_b_index: child_b_index,
            mode_index,
            anchor_a,
            anchor_b,
            twist_ref_a: child_a_rotation,
            twist_ref_b: child_b_rotation,
            zone_a,
            zone_b,
        }
    }
    
    /// Create deterministic hash for sorting
    pub fn deterministic_hash(&self) -> u64 {
        // Use cell indices and mode for deterministic ordering
        // Lower cell index first for consistent ordering
        let (a, b) = if self.cell_a_index <= self.cell_b_index {
            (self.cell_a_index, self.cell_b_index)
        } else {
            (self.cell_b_index, self.cell_a_index)
        };
        
        ((a as u64) << 32) | ((b as u64) << 16) | (self.mode_index as u64)
    }
}

/// Adhesion integration manager
/// 
/// Coordinates adhesion operations between CPU and GPU.
/// Ensures deterministic execution order.
pub struct AdhesionIntegration {
    /// Pending adhesion creation requests
    pending_creations: Vec<AdhesionCreationRequest>,
    
    /// Pending adhesion removal requests (adhesion indices)
    pending_removals: Vec<u32>,
    
    /// Whether parent_make_adhesion is enabled per mode
    /// Cached from genome for quick lookup
    parent_make_adhesion_flags: Vec<bool>,
}

impl AdhesionIntegration {
    /// Create new adhesion integration manager
    pub fn new() -> Self {
        Self {
            pending_creations: Vec::new(),
            pending_removals: Vec::new(),
            parent_make_adhesion_flags: Vec::new(),
        }
    }
    
    /// Update parent_make_adhesion flags from genomes
    pub fn update_from_genomes(&mut self, genomes: &[crate::genome::Genome]) {
        self.parent_make_adhesion_flags.clear();
        for genome in genomes {
            for mode in &genome.modes {
                self.parent_make_adhesion_flags.push(mode.parent_make_adhesion);
            }
        }
    }
    
    /// Queue a sibling adhesion creation (called after division)
    pub fn queue_sibling_adhesion(&mut self, request: AdhesionCreationRequest) {
        self.pending_creations.push(request);
    }
    
    /// Queue an adhesion removal (called when cell dies)
    pub fn queue_adhesion_removal(&mut self, adhesion_index: u32) {
        self.pending_removals.push(adhesion_index);
    }
    
    /// Process all pending adhesion operations in deterministic order
    /// 
    /// Returns the number of adhesions created and removed.
    pub fn process_pending(
        &mut self,
        adhesion_buffers: &mut AdhesionBuffers,
        queue: &wgpu::Queue,
    ) -> (usize, usize) {
        // Process removals first (in sorted order for determinism)
        self.pending_removals.sort_unstable();
        let removals = self.pending_removals.len();
        for &adhesion_idx in &self.pending_removals {
            adhesion_buffers.remove_adhesion(adhesion_idx);
        }
        self.pending_removals.clear();
        
        // Process creations in deterministic order
        self.pending_creations.sort_by_key(|r| r.deterministic_hash());
        let mut creations = 0;
        for request in &self.pending_creations {
            // Check if parent_make_adhesion is enabled for this mode
            let mode_idx = request.mode_index as usize;
            let should_create = mode_idx < self.parent_make_adhesion_flags.len()
                && self.parent_make_adhesion_flags[mode_idx];
            
            if should_create {
                if adhesion_buffers.create_sibling_adhesion(
                    request.cell_a_index,
                    request.cell_b_index,
                    request.mode_index,
                    request.anchor_a,
                    request.anchor_b,
                    request.twist_ref_a,
                    request.twist_ref_b,
                ).is_some() {
                    creations += 1;
                }
            }
        }
        self.pending_creations.clear();
        
        // Sync to GPU
        adhesion_buffers.sync_to_gpu(queue);
        
        (creations, removals)
    }
    
    /// Handle cell division - create sibling adhesions
    /// 
    /// Called after division execute shader completes.
    /// 
    /// # Arguments
    /// * `divisions` - List of (parent_idx, child_b_idx, mode_idx, parent_rotation, child_a_orient, child_b_orient, split_dir)
    pub fn handle_divisions(
        &mut self,
        divisions: &[(u32, u32, u32, glam::Quat, glam::Quat, glam::Quat, glam::Vec3)],
    ) {
        for &(parent_idx, child_b_idx, mode_idx, parent_rot, child_a_orient, child_b_orient, split_dir) in divisions {
            let request = AdhesionCreationRequest::new_sibling(
                parent_idx,    // Child A is at parent slot
                child_b_idx,   // Child B is at new slot
                mode_idx,
                parent_rot,
                child_a_orient,
                child_b_orient,
                split_dir,
            );
            self.queue_sibling_adhesion(request);
        }
    }
    
    /// Handle cell deaths - remove connected adhesions
    /// 
    /// Called when cells die.
    /// 
    /// # Arguments
    /// * `dead_cells` - List of dead cell indices
    /// * `adhesion_buffers` - Adhesion buffer system
    pub fn handle_deaths(
        &mut self,
        dead_cells: &[u32],
        adhesion_buffers: &AdhesionBuffers,
    ) {
        for &cell_idx in dead_cells {
            // Get all adhesions connected to this cell
            let adhesions = adhesion_buffers.get_cell_adhesions(cell_idx);
            for adhesion_idx in adhesions {
                self.queue_adhesion_removal(adhesion_idx);
            }
        }
    }
    
    /// Get pending creation count
    pub fn pending_creation_count(&self) -> usize {
        self.pending_creations.len()
    }
    
    /// Get pending removal count
    pub fn pending_removal_count(&self) -> usize {
        self.pending_removals.len()
    }
    
    /// Clear all pending operations
    pub fn clear_pending(&mut self) {
        self.pending_creations.clear();
        self.pending_removals.clear();
    }
    
    /// Reset the integration manager
    pub fn reset(&mut self) {
        self.pending_creations.clear();
        self.pending_removals.clear();
        self.parent_make_adhesion_flags.clear();
    }
}

impl Default for AdhesionIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Quat};
    
    #[test]
    fn test_sibling_adhesion_request() {
        let request = AdhesionCreationRequest::new_sibling(
            0,  // child_a_index
            1,  // child_b_index
            0,  // mode_index
            Quat::IDENTITY,  // parent_rotation
            Quat::IDENTITY,  // child_a_orientation
            Quat::IDENTITY,  // child_b_orientation
            Vec3::Z,         // split_direction
        );
        
        // Anchor A should point toward B (negative Z in local space)
        assert!(request.anchor_a.z < 0.0);
        
        // Anchor B should point toward A (positive Z in local space)
        assert!(request.anchor_b.z > 0.0);
        
        // Zone classifications
        assert_eq!(request.zone_a, AdhesionZone::ZoneB);
        assert_eq!(request.zone_b, AdhesionZone::ZoneA);
    }
    
    #[test]
    fn test_deterministic_hash_ordering() {
        let request1 = AdhesionCreationRequest {
            cell_a_index: 0,
            cell_b_index: 1,
            mode_index: 0,
            anchor_a: Vec3::Z,
            anchor_b: -Vec3::Z,
            twist_ref_a: Quat::IDENTITY,
            twist_ref_b: Quat::IDENTITY,
            zone_a: AdhesionZone::ZoneB,
            zone_b: AdhesionZone::ZoneA,
        };
        
        let request2 = AdhesionCreationRequest {
            cell_a_index: 1,
            cell_b_index: 0,  // Swapped order
            mode_index: 0,
            anchor_a: Vec3::Z,
            anchor_b: -Vec3::Z,
            twist_ref_a: Quat::IDENTITY,
            twist_ref_b: Quat::IDENTITY,
            zone_a: AdhesionZone::ZoneB,
            zone_b: AdhesionZone::ZoneA,
        };
        
        // Same cells should have same hash regardless of order
        assert_eq!(request1.deterministic_hash(), request2.deterministic_hash());
    }
}

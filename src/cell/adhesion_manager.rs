use glam::{Vec3, Quat};
use super::adhesion::{AdhesionConnections, AdhesionIndices, MAX_ADHESIONS_PER_CELL, init_adhesion_indices};
use super::adhesion_zones::classify_bond_direction;

/// Adhesion connection manager
/// Handles proper adhesion index slot management (20 slots per cell, -1 for empty)
#[derive(Clone)]
pub struct AdhesionConnectionManager {
    /// Adhesion indices for each cell (20 slots per cell)
    pub cell_adhesion_indices: Vec<AdhesionIndices>,
}

impl AdhesionConnectionManager {
    pub fn new(max_cells: usize) -> Self {
        Self {
            cell_adhesion_indices: vec![init_adhesion_indices(); max_cells],
        }
    }
    
    /// Initialize adhesion indices for a cell (all slots to -1)
    pub fn init_cell_adhesion_indices(&mut self, cell_index: usize) {
        if cell_index < self.cell_adhesion_indices.len() {
            self.cell_adhesion_indices[cell_index] = init_adhesion_indices();
        }
    }
    
    /// Reset all adhesion indices to empty (-1)
    pub fn reset(&mut self) {
        for indices in &mut self.cell_adhesion_indices {
            *indices = init_adhesion_indices();
        }
    }
    
    /// Find a free adhesion slot in a cell
    pub fn find_free_adhesion_slot(&self, cell_index: usize) -> Option<usize> {
        if cell_index >= self.cell_adhesion_indices.len() {
            return None;
        }
        
        for (slot_idx, &connection_idx) in self.cell_adhesion_indices[cell_index].iter().enumerate() {
            if connection_idx < 0 {
                return Some(slot_idx);
            }
        }
        
        None
    }
    
    /// Set adhesion index in a cell slot
    pub fn set_adhesion_index(&mut self, cell_index: usize, slot_index: usize, connection_index: i32) -> bool {
        if cell_index >= self.cell_adhesion_indices.len() {
            return false;
        }
        
        if slot_index >= MAX_ADHESIONS_PER_CELL {
            return false;
        }
        
        self.cell_adhesion_indices[cell_index][slot_index] = connection_index;
        true
    }
    
    /// Remove adhesion index from a cell
    pub fn remove_adhesion_index(&mut self, cell_index: usize, connection_index: i32) -> bool {
        if cell_index >= self.cell_adhesion_indices.len() {
            return false;
        }
        
        for slot in &mut self.cell_adhesion_indices[cell_index] {
            if *slot == connection_index {
                *slot = -1;
                return true;
            }
        }
        
        false
    }

    /// Add adhesion connection with proper slot management and zone classification
    /// 
    /// # Arguments
    /// * `connections` - Adhesion connections data
    /// * `cell_a` - Index of cell A
    /// * `cell_b` - Index of cell B
    /// * `mode_index` - Mode index for adhesion settings
    /// * `anchor_direction_a` - Anchor direction for cell A (local space)
    /// * `anchor_direction_b` - Anchor direction for cell B (local space)
    /// * `split_direction_a` - Split direction for cell A (for zone classification)
    /// * `split_direction_b` - Split direction for cell B (for zone classification)
    /// * `genome_orientation_a` - Genome orientation for cell A (for twist reference)
    /// * `genome_orientation_b` - Genome orientation for cell B (for twist reference)
    #[allow(clippy::too_many_arguments)]
    pub fn add_adhesion_with_directions(
        &mut self,
        connections: &mut AdhesionConnections,
        cell_a: usize,
        cell_b: usize,
        mode_index: usize,
        anchor_direction_a: Vec3,
        anchor_direction_b: Vec3,
        split_direction_a: Vec3,
        split_direction_b: Vec3,
        genome_orientation_a: Quat,
        genome_orientation_b: Quat,
    ) -> Option<usize> {
        // Check if cells are the same
        if cell_a == cell_b {
            return None;
        }
        
        // Check connection capacity
        if connections.active_count >= connections.cell_a_index.len() {
            return None;
        }
        
        // Find free slots in both cells
        let slot_a = match self.find_free_adhesion_slot(cell_a) {
            Some(slot) => slot,
            None => return None,
        };
        
        let slot_b = match self.find_free_adhesion_slot(cell_b) {
            Some(slot) => slot,
            None => return None,
        };
        
        // Find free connection slot
        let connection_index = match self.find_free_connection_slot(connections) {
            Some(idx) => idx,
            None => return None,
        };
        
        // Create the connection
        connections.cell_a_index[connection_index] = cell_a;
        connections.cell_b_index[connection_index] = cell_b;
        connections.mode_index[connection_index] = mode_index;
        connections.is_active[connection_index] = 1;
        
        // Classify zones using anchor directions and each cell's split direction
        let zone_a = classify_bond_direction(anchor_direction_a, split_direction_a);
        let zone_b = classify_bond_direction(anchor_direction_b, split_direction_b);
        
        connections.zone_a[connection_index] = zone_a as u8;
        connections.zone_b[connection_index] = zone_b as u8;
        
        // Set anchor directions (normalized)
        let normalized_anchor_a = if anchor_direction_a.length() > 0.001 {
            anchor_direction_a.normalize()
        } else {
            Vec3::X
        };
        
        let normalized_anchor_b = if anchor_direction_b.length() > 0.001 {
            anchor_direction_b.normalize()
        } else {
            -Vec3::X
        };
        
        connections.anchor_direction_a[connection_index] = normalized_anchor_a;
        connections.anchor_direction_b[connection_index] = normalized_anchor_b;
        
        // Set twist references from genome orientations (matches C++ implementation)
        // This is critical for proper twist constraint behavior
        connections.twist_reference_a[connection_index] = genome_orientation_a;
        connections.twist_reference_b[connection_index] = genome_orientation_b;
        
        // Update adhesion indices in both cells
        if !self.set_adhesion_index(cell_a, slot_a, connection_index as i32) ||
           !self.set_adhesion_index(cell_b, slot_b, connection_index as i32) {
            // Rollback
            connections.is_active[connection_index] = 0;
            return None;
        }
        
        // Update active count if needed
        if connection_index >= connections.active_count {
            connections.active_count = connection_index + 1;
        }
        
        Some(connection_index)
    }
    
    /// Remove adhesion connection
    pub fn remove_adhesion(
        &mut self,
        connections: &mut AdhesionConnections,
        connection_index: usize,
    ) -> bool {
        if connection_index >= connections.active_count {
            return false;
        }
        
        if connections.is_active[connection_index] == 0 {
            return false;
        }
        
        let cell_a = connections.cell_a_index[connection_index];
        let cell_b = connections.cell_b_index[connection_index];
        
        // Remove adhesion indices from both cells
        self.remove_adhesion_index(cell_a, connection_index as i32);
        self.remove_adhesion_index(cell_b, connection_index as i32);
        
        // Mark connection as inactive
        connections.is_active[connection_index] = 0;
        
        true
    }
    
    /// Remove all connections for a cell
    pub fn remove_all_connections_for_cell(
        &mut self,
        connections: &mut AdhesionConnections,
        cell_index: usize,
    ) {
        if cell_index >= self.cell_adhesion_indices.len() {
            return;
        }
        
        // Collect connection indices to remove
        let mut to_remove = Vec::new();
        for &connection_idx in &self.cell_adhesion_indices[cell_index] {
            if connection_idx >= 0 {
                to_remove.push(connection_idx as usize);
            }
        }
        
        // Remove each connection
        for connection_idx in to_remove {
            self.remove_adhesion(connections, connection_idx);
        }
    }
    
    /// Find free connection slot
    fn find_free_connection_slot(&self, connections: &AdhesionConnections) -> Option<usize> {
        for i in 0..connections.cell_a_index.len() {
            if i >= connections.active_count || connections.is_active[i] == 0 {
                return Some(i);
            }
        }
        None
    }
    
    /// Get connections for a cell
    pub fn get_connections_for_cell(&self, connections: &AdhesionConnections, cell_index: usize) -> Vec<usize> {
        let mut result = Vec::new();
        
        if cell_index >= self.cell_adhesion_indices.len() {
            return result;
        }
        
        for &connection_idx in &self.cell_adhesion_indices[cell_index] {
            if connection_idx >= 0 {
                let idx = connection_idx as usize;
                if idx < connections.active_count && connections.is_active[idx] == 1 {
                    result.push(idx);
                }
            }
        }
        
        result
    }
    
    /// Get active connection count
    pub fn get_active_connection_count(&self, connections: &AdhesionConnections) -> usize {
        let mut count = 0;
        for i in 0..connections.active_count {
            if connections.is_active[i] == 1 {
                count += 1;
            }
        }
        count
    }

    /// Get all active connections efficiently (returns iterator over connection indices)
    /// This is more cache-friendly than repeated lookups
    pub fn iter_active_connections<'a>(
        &'a self,
        connections: &'a AdhesionConnections,
    ) -> impl Iterator<Item = usize> + 'a {
        (0..connections.active_count)
            .filter(move |&i| connections.is_active[i] == 1)
    }
    
    /// Count active adhesions for a cell (optimized)
    pub fn count_active_adhesions(&self, cell_index: usize) -> usize {
        if cell_index >= self.cell_adhesion_indices.len() {
            return 0;
        }
        
        self.cell_adhesion_indices[cell_index]
            .iter()
            .filter(|&&idx| idx >= 0)
            .count()
    }
    
    /// Check if two cells are directly connected by adhesion (optimized with early exit)
    pub fn are_cells_connected(&self, connections: &AdhesionConnections, cell_a: usize, cell_b: usize) -> bool {
        if cell_a >= self.cell_adhesion_indices.len() {
            return false;
        }
        
        // Check cell A's adhesion list (typically much smaller than all connections)
        for &conn_idx in &self.cell_adhesion_indices[cell_a] {
            if conn_idx < 0 {
                continue;
            }
            
            let conn_idx = conn_idx as usize;
            if conn_idx >= connections.active_count || connections.is_active[conn_idx] == 0 {
                continue;
            }
            
            let other = if connections.cell_a_index[conn_idx] == cell_a {
                connections.cell_b_index[conn_idx]
            } else {
                connections.cell_a_index[conn_idx]
            };
            
            if other == cell_b {
                return true;
            }
        }
        
        false
    }
    
    /// Check if two cells are in the same organism (connected component via adhesions)
    /// Uses BFS to traverse the adhesion graph from cell_a to find cell_b
    /// This is more expensive than are_cells_connected but correctly identifies
    /// cells that are part of the same contiguous organism
    pub fn are_cells_in_same_organism(&self, connections: &AdhesionConnections, cell_a: usize, cell_b: usize) -> bool {
        // Same cell is trivially in same organism
        if cell_a == cell_b {
            return true;
        }
        
        // If either cell has no adhesions, they can't be in the same organism
        if cell_a >= self.cell_adhesion_indices.len() || cell_b >= self.cell_adhesion_indices.len() {
            return false;
        }
        
        // Quick check: if cell_a has no adhesions, it's a single-cell organism
        let has_adhesions_a = self.cell_adhesion_indices[cell_a].iter().any(|&idx| idx >= 0);
        if !has_adhesions_a {
            return false;
        }
        
        // BFS to find if cell_b is reachable from cell_a
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        visited.insert(cell_a);
        queue.push_back(cell_a);
        
        while let Some(current) = queue.pop_front() {
            // Check all adhesion connections of current cell
            for &conn_idx in &self.cell_adhesion_indices[current] {
                if conn_idx < 0 {
                    continue;
                }
                
                let conn_idx = conn_idx as usize;
                if conn_idx >= connections.active_count || connections.is_active[conn_idx] == 0 {
                    continue;
                }
                
                // Find the neighbor cell
                let neighbor = if connections.cell_a_index[conn_idx] == current {
                    connections.cell_b_index[conn_idx]
                } else {
                    connections.cell_a_index[conn_idx]
                };
                
                // Found target!
                if neighbor == cell_b {
                    return true;
                }
                
                // Add to queue if not visited
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        false
    }
}

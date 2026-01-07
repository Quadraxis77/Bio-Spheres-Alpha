//! Deterministic Adhesion System for GPU Scene
//!
//! This module implements a deterministic adhesion system compatible with the GPU physics pipeline.
//! Adhesions are permanent connections between sibling cells created during division.
//!
//! ## Key Features
//! - **Deterministic**: Same input produces same output across runs
//! - **GPU-compatible**: Uses prefix-sum compaction for slot allocation
//! - **Zone-based inheritance**: Adhesions inherited based on geometric zones
//!
//! ## Adhesion Connection Structure (96 bytes, matching reference)
//! - cell_a_index: u32 - Index of first cell
//! - cell_b_index: u32 - Index of second cell  
//! - mode_index: u32 - Mode index for adhesion settings lookup
//! - is_active: u32 - 1 = active, 0 = inactive
//! - zone_a: u32 - Zone classification for cell A (0=A, 1=B, 2=C)
//! - zone_b: u32 - Zone classification for cell B
//! - anchor_direction_a: vec3<f32> + padding - Local anchor direction for cell A
//! - anchor_direction_b: vec3<f32> + padding - Local anchor direction for cell B
//! - twist_reference_a: vec4<f32> - Reference quaternion for twist constraint
//! - twist_reference_b: vec4<f32> - Reference quaternion for twist constraint
//! - _padding: [u32; 2] - Padding to 96 bytes

use bytemuck::{Pod, Zeroable};

/// Maximum adhesions per cell (matching reference)
pub const MAX_ADHESIONS_PER_CELL: usize = 20;

/// Maximum total adhesion connections
pub const MAX_ADHESION_CONNECTIONS: u32 = 100_000;

/// Adhesion connection structure (exactly 96 bytes)
/// Matches the reference implementation for GPU compatibility
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuAdhesionConnection {
    /// Index of first cell in the connection
    pub cell_a_index: u32,
    /// Index of second cell in the connection
    pub cell_b_index: u32,
    /// Mode index for adhesion settings lookup
    pub mode_index: u32,
    /// Whether this connection is active (1 = active, 0 = inactive)
    pub is_active: u32,
    /// Zone classification for cell A (0=ZoneA, 1=ZoneB, 2=ZoneC)
    pub zone_a: u32,
    /// Zone classification for cell B (0=ZoneA, 1=ZoneB, 2=ZoneC)
    pub zone_b: u32,
    /// Anchor direction for cell A in local cell space (normalized)
    pub anchor_direction_a: [f32; 3],
    /// Padding for 16-byte alignment
    pub padding_a: f32,
    /// Anchor direction for cell B in local cell space (normalized)
    pub anchor_direction_b: [f32; 3],
    /// Padding for 16-byte alignment
    pub padding_b: f32,
    /// Reference quaternion for twist constraint for cell A (x, y, z, w)
    pub twist_reference_a: [f32; 4],
    /// Reference quaternion for twist constraint for cell B (x, y, z, w)
    pub twist_reference_b: [f32; 4],
    /// Padding to ensure 96 bytes total
    pub _padding: [u32; 2],
}

// Verify size at compile time
const _: () = assert!(std::mem::size_of::<GpuAdhesionConnection>() == 96);

impl GpuAdhesionConnection {
    /// Create a new inactive adhesion connection
    pub fn inactive() -> Self {
        Self {
            cell_a_index: 0,
            cell_b_index: 0,
            mode_index: 0,
            is_active: 0,
            zone_a: 0,
            zone_b: 0,
            anchor_direction_a: [0.0, 0.0, 1.0],
            padding_a: 0.0,
            anchor_direction_b: [0.0, 0.0, -1.0],
            padding_b: 0.0,
            twist_reference_a: [0.0, 0.0, 0.0, 1.0], // Identity quaternion
            twist_reference_b: [0.0, 0.0, 0.0, 1.0],
            _padding: [0, 0],
        }
    }

    /// Create a new sibling adhesion between two cells
    pub fn new_sibling(
        cell_a_index: u32,
        cell_b_index: u32,
        mode_index: u32,
        anchor_a: glam::Vec3,
        anchor_b: glam::Vec3,
        twist_ref_a: glam::Quat,
        twist_ref_b: glam::Quat,
    ) -> Self {
        Self {
            cell_a_index,
            cell_b_index,
            mode_index,
            is_active: 1,
            zone_a: 1, // Zone B (positive split direction)
            zone_b: 0, // Zone A (negative split direction)
            anchor_direction_a: anchor_a.to_array(),
            padding_a: 0.0,
            anchor_direction_b: anchor_b.to_array(),
            padding_b: 0.0,
            twist_reference_a: [twist_ref_a.x, twist_ref_a.y, twist_ref_a.z, twist_ref_a.w],
            twist_reference_b: [twist_ref_b.x, twist_ref_b.y, twist_ref_b.z, twist_ref_b.w],
            _padding: [0, 0],
        }
    }
}

/// GPU-side adhesion settings (48 bytes, matching reference GPUModeAdhesionSettings)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuAdhesionSettings {
    /// Whether adhesion can break (bool as i32)
    pub can_break: i32,
    /// Force threshold for breaking
    pub break_force: f32,
    /// Natural rest length of the bond
    pub rest_length: f32,
    /// Linear spring stiffness
    pub linear_spring_stiffness: f32,
    /// Linear spring damping
    pub linear_spring_damping: f32,
    /// Orientation spring stiffness
    pub orientation_spring_stiffness: f32,
    /// Orientation spring damping
    pub orientation_spring_damping: f32,
    /// Maximum angular deviation (degrees)
    pub max_angular_deviation: f32,
    /// Twist constraint stiffness
    pub twist_constraint_stiffness: f32,
    /// Twist constraint damping
    pub twist_constraint_damping: f32,
    /// Whether twist constraint is enabled (bool as i32)
    pub enable_twist_constraint: i32,
    /// Padding to 48 bytes
    pub _padding: i32,
}

// Verify size at compile time
const _: () = assert!(std::mem::size_of::<GpuAdhesionSettings>() == 48);

impl Default for GpuAdhesionSettings {
    fn default() -> Self {
        Self {
            can_break: 1,
            break_force: 10.0,
            rest_length: 1.0,
            linear_spring_stiffness: 150.0,
            linear_spring_damping: 5.0,
            orientation_spring_stiffness: 10.0,
            orientation_spring_damping: 2.0,
            max_angular_deviation: 0.0,
            twist_constraint_stiffness: 2.0,
            twist_constraint_damping: 0.5,
            enable_twist_constraint: 1,
            _padding: 0,
        }
    }
}

/// Adhesion counts buffer structure
/// Used for tracking adhesion allocation state
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct AdhesionCounts {
    /// Total allocated adhesion connections
    pub total_adhesion_count: u32,
    /// Number of active adhesion connections
    pub live_adhesion_count: u32,
    /// Top of free adhesion slot stack
    pub free_adhesion_top: u32,
    /// Padding
    pub _padding: u32,
}

/// Zone classification for adhesion inheritance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdhesionZone {
    /// Zone A: Negative split direction (kept by Child B)
    ZoneA = 0,
    /// Zone B: Positive split direction (kept by Child A)
    ZoneB = 1,
    /// Zone C: Equatorial (shared by both children)
    ZoneC = 2,
}

impl AdhesionZone {
    /// Classify a bond direction into zones based on angle from split direction
    /// 
    /// # Arguments
    /// * `bond_dir_local` - Bond direction in local cell space (normalized)
    /// * `split_dir_local` - Split direction in local cell space (normalized)
    /// * `inheritance_angle_deg` - Half-width of equatorial zone in degrees
    pub fn classify(bond_dir_local: glam::Vec3, split_dir_local: glam::Vec3, inheritance_angle_deg: f32) -> Self {
        let dot = bond_dir_local.dot(split_dir_local);
        let angle = dot.clamp(-1.0, 1.0).acos().to_degrees();
        let half_width = inheritance_angle_deg;
        let equatorial_angle = 90.0;

        if (angle - equatorial_angle).abs() <= half_width {
            AdhesionZone::ZoneC // Equatorial
        } else if dot > 0.0 {
            AdhesionZone::ZoneB // Positive dot product
        } else {
            AdhesionZone::ZoneA // Negative dot product
        }
    }
}

/// Deterministic adhesion slot allocator
/// Uses sorted free list for deterministic allocation order
#[derive(Debug, Clone)]
pub struct AdhesionSlotAllocator {
    /// Free slot indices (sorted for deterministic allocation)
    free_slots: Vec<u32>,
    /// Total capacity
    capacity: u32,
    /// Current allocation count
    allocated_count: u32,
}

impl AdhesionSlotAllocator {
    /// Create new allocator with given capacity
    pub fn new(capacity: u32) -> Self {
        Self {
            free_slots: Vec::new(),
            capacity,
            allocated_count: 0,
        }
    }

    /// Allocate next available slot deterministically
    /// Returns slot index or None if no slots available
    pub fn allocate_slot(&mut self) -> Option<u32> {
        // First try to reuse a free slot
        if !self.free_slots.is_empty() {
            // Always take the first (lowest) slot for deterministic allocation
            return Some(self.free_slots.remove(0));
        }

        // Otherwise allocate from unused capacity
        if self.allocated_count < self.capacity {
            let slot = self.allocated_count;
            self.allocated_count += 1;
            return Some(slot);
        }

        None // No slots available
    }

    /// Free a slot (mark for reuse)
    pub fn free_slot(&mut self, slot: u32) {
        if slot < self.capacity && !self.free_slots.contains(&slot) {
            self.free_slots.push(slot);
            // Keep sorted for deterministic allocation
            self.free_slots.sort_unstable();
        }
    }

    /// Get current allocated count
    pub fn allocated_count(&self) -> u32 {
        self.allocated_count
    }

    /// Get available slot count
    pub fn available_slots(&self) -> usize {
        self.free_slots.len() + (self.capacity - self.allocated_count) as usize
    }

    /// Reset allocator to initial state
    pub fn reset(&mut self) {
        self.free_slots.clear();
        self.allocated_count = 0;
    }
}

/// Per-cell adhesion indices array
/// Each cell can have up to MAX_ADHESIONS_PER_CELL connections
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CellAdhesionIndices {
    /// Indices into adhesion_connections array (-1 = no connection)
    pub indices: [i32; MAX_ADHESIONS_PER_CELL],
}

impl Default for CellAdhesionIndices {
    fn default() -> Self {
        Self {
            indices: [-1; MAX_ADHESIONS_PER_CELL],
        }
    }
}

impl CellAdhesionIndices {
    /// Find first available slot
    pub fn find_free_slot(&self) -> Option<usize> {
        self.indices.iter().position(|&idx| idx < 0)
    }

    /// Add adhesion index to first available slot
    pub fn add_adhesion(&mut self, adhesion_idx: u32) -> bool {
        if let Some(slot) = self.find_free_slot() {
            self.indices[slot] = adhesion_idx as i32;
            true
        } else {
            false
        }
    }

    /// Remove adhesion index
    pub fn remove_adhesion(&mut self, adhesion_idx: u32) {
        for idx in &mut self.indices {
            if *idx == adhesion_idx as i32 {
                *idx = -1;
                break;
            }
        }
    }

    /// Count active adhesions
    pub fn count(&self) -> usize {
        self.indices.iter().filter(|&&idx| idx >= 0).count()
    }
}

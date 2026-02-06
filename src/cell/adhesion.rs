use glam::{Vec3, Quat};

/// Maximum adhesions per cell (reduced from 20 for 200K cell support)
pub const MAX_ADHESIONS_PER_CELL: usize = 20;

/// Maximum total adhesion connections (20 × max cells)
pub const MAX_ADHESION_CONNECTIONS: usize = 5120;

/// Adhesion connection between two cells (Structure-of-Arrays layout)
#[derive(Clone)]
pub struct AdhesionConnections {
    /// Cell A indices
    pub cell_a_index: Vec<usize>,
    /// Cell B indices
    pub cell_b_index: Vec<usize>,
    /// Mode index for adhesion settings
    pub mode_index: Vec<usize>,
    /// Active flag (1 = active, 0 = inactive)
    pub is_active: Vec<u8>,
    /// Zone classification for cell A
    pub zone_a: Vec<u8>,
    /// Zone classification for cell B
    pub zone_b: Vec<u8>,
    
    /// Anchor direction for cell A (local space, normalized)
    pub anchor_direction_a: Vec<Vec3>,
    /// Anchor direction for cell B (local space, normalized)
    pub anchor_direction_b: Vec<Vec3>,
    
    /// Twist reference quaternion for cell A
    pub twist_reference_a: Vec<Quat>,
    /// Twist reference quaternion for cell B
    pub twist_reference_b: Vec<Quat>,
    
    /// Number of active connections
    pub active_count: usize,
}

impl AdhesionConnections {
    pub fn new(capacity: usize) -> Self {
        Self {
            cell_a_index: vec![0; capacity],
            cell_b_index: vec![0; capacity],
            mode_index: vec![0; capacity],
            is_active: vec![0; capacity],
            zone_a: vec![0; capacity],
            zone_b: vec![0; capacity],
            anchor_direction_a: vec![Vec3::X; capacity],
            anchor_direction_b: vec![-Vec3::X; capacity],
            twist_reference_a: vec![Quat::IDENTITY; capacity],
            twist_reference_b: vec![Quat::IDENTITY; capacity],
            active_count: 0,
        }
    }
}

/// Adhesion indices for each cell (20 slots, -1 for empty)
pub type AdhesionIndices = [i32; MAX_ADHESIONS_PER_CELL];

/// Initialize adhesion indices for a cell (all slots to -1)
pub fn init_adhesion_indices() -> AdhesionIndices {
    [-1; MAX_ADHESIONS_PER_CELL]
}

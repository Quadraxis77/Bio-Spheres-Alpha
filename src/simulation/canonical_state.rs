use glam::{Vec3, Quat};

/// Canonical simulation state using Structure-of-Arrays (SoA) layout
/// Pure data structure with no ECS dependencies
#[derive(Clone)]
pub struct CanonicalState {
    /// Number of active cells
    pub cell_count: usize,
    
    /// Maximum capacity
    pub capacity: usize,
    
    /// Cell unique IDs
    pub cell_ids: Vec<u32>,
    
    // Position and Motion (SoA)
    pub positions: Vec<Vec3>,
    pub prev_positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    
    // Cell Properties (SoA)
    pub masses: Vec<f32>,
    pub radii: Vec<f32>,
    pub genome_ids: Vec<usize>,
    pub mode_indices: Vec<usize>,
    
    // Orientation (SoA)
    pub rotations: Vec<Quat>,
    pub angular_velocities: Vec<Vec3>,
    
    // Physics State (SoA)
    pub forces: Vec<Vec3>,
    pub torques: Vec<Vec3>,
    pub accelerations: Vec<Vec3>,
    pub prev_accelerations: Vec<Vec3>,
    pub stiffnesses: Vec<f32>,
    
    // Division Timers (SoA)
    pub birth_times: Vec<f32>,
    pub split_intervals: Vec<f32>,
    pub split_masses: Vec<f32>,
    pub split_counts: Vec<i32>,
    
    /// Next cell ID to assign
    pub next_cell_id: u32,
}

impl CanonicalState {
    pub fn new(capacity: usize) -> Self {
        Self {
            cell_count: 0,
            capacity,
            cell_ids: vec![0; capacity],
            positions: vec![Vec3::ZERO; capacity],
            prev_positions: vec![Vec3::ZERO; capacity],
            velocities: vec![Vec3::ZERO; capacity],
            masses: vec![1.0; capacity],
            radii: vec![1.0; capacity],
            genome_ids: vec![0; capacity],
            mode_indices: vec![0; capacity],
            rotations: vec![Quat::IDENTITY; capacity],
            angular_velocities: vec![Vec3::ZERO; capacity],
            forces: vec![Vec3::ZERO; capacity],
            torques: vec![Vec3::ZERO; capacity],
            accelerations: vec![Vec3::ZERO; capacity],
            prev_accelerations: vec![Vec3::ZERO; capacity],
            stiffnesses: vec![10.0; capacity],
            birth_times: vec![0.0; capacity],
            split_intervals: vec![10.0; capacity],
            split_masses: vec![1.5; capacity],
            split_counts: vec![0; capacity],
            next_cell_id: 0,
        }
    }
    
    /// Add a new cell to the canonical state
    pub fn add_cell(
        &mut self,
        position: Vec3,
        velocity: Vec3,
        rotation: Quat,
        angular_velocity: Vec3,
        mass: f32,
        radius: f32,
        genome_id: usize,
        mode_index: usize,
        birth_time: f32,
        split_interval: f32,
        split_mass: f32,
        stiffness: f32,
    ) -> Option<usize> {
        if self.cell_count >= self.capacity {
            return None;
        }
        
        let index = self.cell_count;
        self.cell_ids[index] = self.next_cell_id;
        self.positions[index] = position;
        self.prev_positions[index] = position;
        self.velocities[index] = velocity;
        self.masses[index] = mass;
        self.radii[index] = radius;
        self.genome_ids[index] = genome_id;
        self.mode_indices[index] = mode_index;
        self.rotations[index] = rotation;
        self.angular_velocities[index] = angular_velocity;
        self.forces[index] = Vec3::ZERO;
        self.torques[index] = Vec3::ZERO;
        self.accelerations[index] = Vec3::ZERO;
        self.prev_accelerations[index] = Vec3::ZERO;
        self.stiffnesses[index] = stiffness;
        self.birth_times[index] = birth_time;
        self.split_intervals[index] = split_interval;
        self.split_masses[index] = split_mass;
        self.split_counts[index] = 0;
        
        self.cell_count += 1;
        self.next_cell_id += 1;
        
        Some(index)
    }
}

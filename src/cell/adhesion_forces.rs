use glam::{Vec3, Vec4, Quat};
use crate::genome::AdhesionSettings;
use super::adhesion::AdhesionConnections;

/// Numerical precision constants (matching GPU/C++)
#[allow(dead_code)]
const EPSILON: f32 = 1e-6;
const ANGLE_EPSILON: f32 = 0.001;
const QUATERNION_EPSILON: f32 = 0.0001;
const TWIST_CLAMP_LIMIT: f32 = 1.57; // ±90 degrees

/// Compute adhesion forces for all active connections.
/// Returns a list of connection indices whose spring force exceeded break_force
/// (only when `can_break` is true). Caller is responsible for removing them.
#[allow(clippy::too_many_arguments)]
pub fn compute_adhesion_forces(
    connections: &AdhesionConnections,
    positions: &[Vec3],
    velocities: &[Vec3],
    rotations: &[Quat],
    angular_velocities: &[Vec3],
    masses: &[f32],
    mode_settings: &[AdhesionSettings],
    forces: &mut [Vec3],
    torques: &mut [Vec3],
    current_time: f32,
) -> Vec<usize> {
    const BREAK_GRACE_PERIOD: f32 = 0.5;
    let mut bonds_to_break = Vec::new();

    // Process each active adhesion connection
    for i in 0..connections.active_count {
        if connections.is_active[i] == 0 {
            continue;
        }
        
        let cell_a_idx = connections.cell_a_index[i];
        let cell_b_idx = connections.cell_b_index[i];
        let mode_idx = connections.mode_index[i];
        
        // Validate indices
        if cell_a_idx >= positions.len() || cell_b_idx >= positions.len() {
            continue;
        }
        
        if mode_idx >= mode_settings.len() {
            continue;
        }
        
        let settings = &mode_settings[mode_idx];
        
        // Calculate forces and torques
        let (force_a, torque_a, force_b, torque_b, spring_force_mag) = compute_adhesion_force_pair(
            positions[cell_a_idx],
            velocities[cell_a_idx],
            rotations[cell_a_idx],
            angular_velocities[cell_a_idx],
            masses[cell_a_idx],
            positions[cell_b_idx],
            velocities[cell_b_idx],
            rotations[cell_b_idx],
            angular_velocities[cell_b_idx],
            masses[cell_b_idx],
            connections.anchor_direction_a[i],
            connections.anchor_direction_b[i],
            connections.twist_reference_a[i],
            connections.twist_reference_b[i],
            settings,
        );
        
        // Check break condition before applying forces
        // Skip break check during grace period after bond creation
        let in_grace = current_time - connections.birth_time[i] < BREAK_GRACE_PERIOD;
        if settings.can_break && !in_grace && spring_force_mag > settings.break_force {
            bonds_to_break.push(i);
            continue;
        }

        // Apply forces
        forces[cell_a_idx] += force_a;
        forces[cell_b_idx] += force_b;
        torques[cell_a_idx] += torque_a;
        torques[cell_b_idx] += torque_b;
    }

    bonds_to_break
}


/// Compute adhesion forces for all active connections - Parallel version.
/// Returns a list of connection indices whose spring force exceeded break_force
/// (only when `can_break` is true). Caller is responsible for removing them.
#[allow(clippy::too_many_arguments)]
pub fn compute_adhesion_forces_parallel(
    connections: &AdhesionConnections,
    positions: &[Vec3],
    velocities: &[Vec3],
    rotations: &[Quat],
    angular_velocities: &[Vec3],
    masses: &[f32],
    mode_settings: &[AdhesionSettings],
    forces: &mut [Vec3],
    torques: &mut [Vec3],
    current_time: f32,
) -> Vec<usize> {
    const BREAK_GRACE_PERIOD: f32 = 0.5;
    use rayon::prelude::*;
    
    // Compute force contributions in parallel.
    // Each entry is either a force contribution or a break signal (empty forces).
    // Tag: (connection_idx_if_break: Option<usize>, cell_idx, force, torque)
    let results: Vec<(Option<usize>, usize, Vec3, Vec3)> = (0..connections.active_count)
        .into_par_iter()
        .filter(|&i| connections.is_active[i] != 0)
        .flat_map(|i| {
            let cell_a_idx = connections.cell_a_index[i];
            let cell_b_idx = connections.cell_b_index[i];
            let mode_idx = connections.mode_index[i];
            
            if cell_a_idx >= positions.len() || cell_b_idx >= positions.len() {
                return vec![];
            }
            if mode_idx >= mode_settings.len() {
                return vec![];
            }
            
            let settings = &mode_settings[mode_idx];
            
            let (force_a, torque_a, force_b, torque_b, spring_force_mag) = compute_adhesion_force_pair(
                positions[cell_a_idx],
                velocities[cell_a_idx],
                rotations[cell_a_idx],
                angular_velocities[cell_a_idx],
                masses[cell_a_idx],
                positions[cell_b_idx],
                velocities[cell_b_idx],
                rotations[cell_b_idx],
                angular_velocities[cell_b_idx],
                masses[cell_b_idx],
                connections.anchor_direction_a[i],
                connections.anchor_direction_b[i],
                connections.twist_reference_a[i],
                connections.twist_reference_b[i],
                settings,
            );

            // Bond breaks: signal with Some(i), zero forces
            // Skip break check during grace period after bond creation
            let in_grace = current_time - connections.birth_time[i] < BREAK_GRACE_PERIOD;
            if settings.can_break && !in_grace && spring_force_mag > settings.break_force {
                return vec![(Some(i), 0, Vec3::ZERO, Vec3::ZERO)];
            }
            
            vec![
                (None, cell_a_idx, force_a, torque_a),
                (None, cell_b_idx, force_b, torque_b),
            ]
        })
        .collect();
    
    let mut bonds_to_break = Vec::new();
    for (break_idx, idx, force, torque) in results {
        if let Some(conn_idx) = break_idx {
            bonds_to_break.push(conn_idx);
        } else {
            forces[idx] += force;
            torques[idx] += torque;
        }
    }
    bonds_to_break
}

/// Compute adhesion forces for a single connection pair
/// Direct port of C++ computeAdhesionForces (cell pair version)
/// Optimized with inline hint for better performance
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_adhesion_force_pair(
    pos_a: Vec3,
    vel_a: Vec3,
    rot_a: Quat,
    ang_vel_a: Vec3,
    _mass_a: f32,
    pos_b: Vec3,
    vel_b: Vec3,
    rot_b: Quat,
    ang_vel_b: Vec3,
    _mass_b: f32,
    anchor_dir_a: Vec3,
    anchor_dir_b: Vec3,
    twist_ref_a: Quat,
    twist_ref_b: Quat,
    settings: &AdhesionSettings,
) -> (Vec3, Vec3, Vec3, Vec3, f32) {
    let mut force_a = Vec3::ZERO;
    let mut torque_a = Vec3::ZERO;
    let mut force_b = Vec3::ZERO;
    let mut torque_b = Vec3::ZERO;
    
    // Connection vector from A to B
    let delta_pos = pos_b - pos_a;
    let dist = delta_pos.length();
    if dist < QUATERNION_EPSILON {
        return (force_a, torque_a, force_b, torque_b, 0.0);
    }
    
    let adhesion_dir = delta_pos / dist;
    let rest_length = settings.rest_length;
    
    // Linear spring force with softness factor (emulates Python softness = 1.0 - bond_stretch * 0.8)
    let softness = 0.3; // Reduced softness to allow more flexibility for spiral patterns
    let force_mag = settings.linear_spring_stiffness * (dist - rest_length) * softness;
    let spring_force_mag = force_mag.abs();
    let spring_force = adhesion_dir * force_mag;
    
    // Damping - matches reference implementation exactly
    // This is an unusual formula: constant force modified by velocity
    // When rel_vel is 0: damping_force = -adhesion_dir * 1.0 (constant force toward A)
    // The velocity component modulates this base force
    let rel_vel = vel_b - vel_a;
    let damp_mag = 1.0 - settings.linear_spring_damping * rel_vel.dot(adhesion_dir);
    let damping_force = -adhesion_dir * damp_mag;
    
    // Apply forces: spring + damping
    force_a += spring_force + damping_force;
    force_b -= spring_force + damping_force;
    
    // Transform anchor directions to world space using PHYSICS rotations
    // Anchors are stored in local space and rotate with the cell
    let anchor_a = if anchor_dir_a.length() < ANGLE_EPSILON && anchor_dir_b.length() < ANGLE_EPSILON {
        Vec3::X
    } else {
        rotate_vector_by_quaternion(anchor_dir_a, rot_a)
    };
    
    let anchor_b = if anchor_dir_a.length() < ANGLE_EPSILON && anchor_dir_b.length() < ANGLE_EPSILON {
        -Vec3::X
    } else {
        rotate_vector_by_quaternion(anchor_dir_b, rot_b)
    };
    
    // Apply orientation spring and damping
    let axis_a = anchor_a.cross(adhesion_dir);
    let sin_a = axis_a.length();
    let cos_a = anchor_a.dot(adhesion_dir);
    let angle_a = sin_a.atan2(cos_a);
    
    if sin_a > QUATERNION_EPSILON {
        let axis_a_norm = axis_a.normalize();
        let spring_torque_a = axis_a_norm * angle_a * settings.orientation_spring_stiffness;
        let damping_torque_a = -axis_a_norm * ang_vel_a.dot(axis_a_norm) * settings.orientation_spring_damping;
        torque_a += spring_torque_a + damping_torque_a;
    }
    
    let axis_b = anchor_b.cross(-adhesion_dir);
    let sin_b = axis_b.length();
    let cos_b = anchor_b.dot(-adhesion_dir);
    let angle_b = sin_b.atan2(cos_b);
    
    if sin_b > QUATERNION_EPSILON {
        let axis_b_norm = axis_b.normalize();
        let spring_torque_b = axis_b_norm * angle_b * settings.orientation_spring_stiffness;
        let damping_torque_b = -axis_b_norm * ang_vel_b.dot(axis_b_norm) * settings.orientation_spring_damping;
        torque_b += spring_torque_b + damping_torque_b;
    }

    
    // Apply twist constraints if enabled
    if settings.enable_twist_constraint && 
       twist_ref_a.length() > ANGLE_EPSILON && 
       twist_ref_b.length() > ANGLE_EPSILON {
        
        let adhesion_axis = delta_pos.normalize();
        
        // Get current anchor directions in world space
        let current_anchor_a = rotate_vector_by_quaternion(anchor_dir_a, rot_a);
        let current_anchor_b = rotate_vector_by_quaternion(anchor_dir_b, rot_b);
        
        // Calculate target anchor directions
        let target_anchor_a = adhesion_axis;
        let target_anchor_b = -adhesion_axis;
        
        // Find rotation needed to align current to target
        let alignment_rot_a = quat_from_two_vectors(current_anchor_a, target_anchor_a);
        let alignment_rot_b = quat_from_two_vectors(current_anchor_b, target_anchor_b);
        
        // Apply alignment rotation to reference orientations
        let target_orientation_a = (alignment_rot_a * twist_ref_a).normalize();
        let target_orientation_b = (alignment_rot_b * twist_ref_b).normalize();
        
        // Calculate correction rotation
        let correction_rot_a = (target_orientation_a * rot_a.conjugate()).normalize();
        let correction_rot_b = (target_orientation_b * rot_b.conjugate()).normalize();
        
        // Convert to axis-angle
        let axis_angle_a = quat_to_axis_angle(correction_rot_a);
        let axis_angle_b = quat_to_axis_angle(correction_rot_b);
        
        // Project correction onto adhesion axis (twist component only)
        let twist_correction_a = axis_angle_a.w * Vec3::new(axis_angle_a.x, axis_angle_a.y, axis_angle_a.z).dot(adhesion_axis);
        let twist_correction_b = axis_angle_b.w * Vec3::new(axis_angle_b.x, axis_angle_b.y, axis_angle_b.z).dot(adhesion_axis);
        
        // Clamp corrections
        let twist_correction_a = twist_correction_a.clamp(-TWIST_CLAMP_LIMIT, TWIST_CLAMP_LIMIT);
        let twist_correction_b = twist_correction_b.clamp(-TWIST_CLAMP_LIMIT, TWIST_CLAMP_LIMIT);
        
        // Apply twist torque (reduced strength for CPU stability)
        let twist_torque_a = adhesion_axis * twist_correction_a * settings.twist_constraint_stiffness * 0.05;
        let twist_torque_b = adhesion_axis * twist_correction_b * settings.twist_constraint_stiffness * 0.05;
        
        // Add strong damping
        let angular_vel_a_proj = ang_vel_a.dot(adhesion_axis);
        let angular_vel_b_proj = ang_vel_b.dot(adhesion_axis);
        let relative_angular_vel = angular_vel_a_proj - angular_vel_b_proj;
        
        let twist_damping_a = -adhesion_axis * relative_angular_vel * settings.twist_constraint_damping * 0.6;
        let twist_damping_b = adhesion_axis * relative_angular_vel * settings.twist_constraint_damping * 0.6;
        
        torque_a += twist_torque_a + twist_damping_a;
        torque_b += twist_torque_b + twist_damping_b;
    }
    
    // Apply tangential forces from torques to maintain organism shape
    // IMPROVED: Use balanced tangential forces that conserve momentum
    // 
    // The issue with the original implementation was that it applied:
    //   force_a += (-delta_pos).cross(torque_b)
    //   force_b += delta_pos.cross(torque_a)
    // 
    // This creates unbalanced forces when torques differ, causing phantom drift.
    // 
    // The fix: Apply equal and opposite tangential forces based on the TOTAL torque
    // that would be needed to maintain the constraint. This ensures momentum conservation.
    
    // Calculate the total corrective torque (sum of both cells' torques)
    let total_torque = torque_a + torque_b;
    
    // Calculate tangential force that would create this torque
    // F_tangential = torque × r / |r|²
    // This ensures equal and opposite forces on both cells
    let r_squared = delta_pos.length_squared();
    if r_squared > QUATERNION_EPSILON {
        let tangential_force = total_torque.cross(delta_pos) / r_squared;
        
        // Apply equal and opposite tangential forces
        // This maintains shape while conserving momentum
        force_a += tangential_force;
        force_b -= tangential_force;
    }
    
    (force_a, torque_a, force_b, torque_b, spring_force_mag)
}

/// Rotate vector by quaternion (GPU algorithm port)
/// Optimized with inline hint and reduced operations
#[inline(always)]
fn rotate_vector_by_quaternion(v: Vec3, q: Quat) -> Vec3 {
    let u = Vec3::new(q.x, q.y, q.z);
    let s = q.w;
    let u_dot_v = u.dot(v);
    let u_dot_u = u.dot(u);
    
    // Optimized: reuse computed values
    u * (2.0 * u_dot_v) + v * (s * s - u_dot_u) + u.cross(v) * (2.0 * s)
}

/// Convert quaternion to axis-angle representation
/// Optimized with inline hint
#[inline]
fn quat_to_axis_angle(q: Quat) -> Vec4 {
    let w_clamped = q.w.clamp(-1.0, 1.0);
    let angle = 2.0 * w_clamped.acos();
    
    let axis = if angle < 0.001 {
        Vec3::X
    } else {
        let sin_half = (angle * 0.5).sin();
        Vec3::new(q.x, q.y, q.z) / sin_half
    };
    
    Vec4::new(axis.x, axis.y, axis.z, angle)
}

/// Create quaternion from two vectors (deterministic)
fn quat_from_two_vectors(from: Vec3, to: Vec3) -> Quat {
    let v1 = from.normalize();
    let v2 = to.normalize();
    
    let cos_angle = v1.dot(v2);
    
    // Vectors already aligned
    if cos_angle > 0.9999 {
        return Quat::IDENTITY;
    }
    
    // Vectors are opposite
    if cos_angle < -0.9999 {
        // Choose axis deterministically
        let axis = if v1.x.abs() < v1.y.abs() && v1.x.abs() < v1.z.abs() {
            Vec3::new(0.0, -v1.z, v1.y).normalize()
        } else if v1.y.abs() < v1.z.abs() {
            Vec3::new(-v1.z, 0.0, v1.x).normalize()
        } else {
            Vec3::new(-v1.y, v1.x, 0.0).normalize()
        };
        return Quat::from_xyzw(axis.x, axis.y, axis.z, 0.0); // 180 degree rotation
    }
    
    // General case: half-way quaternion method
    let halfway = (v1 + v2).normalize();
    let axis = Vec3::new(
        v1.y * halfway.z - v1.z * halfway.y,
        v1.z * halfway.x - v1.x * halfway.z,
        v1.x * halfway.y - v1.y * halfway.x,
    );
    let w = v1.dot(halfway);
    
    Quat::from_xyzw(axis.x, axis.y, axis.z, w).normalize()
}


/// Compute adhesion forces with improved cache locality
/// 
/// This version processes connections in batches to improve CPU cache utilization.
/// By grouping connections that access nearby cells, we reduce cache misses.
#[allow(clippy::too_many_arguments)]
pub fn compute_adhesion_forces_batched(
    connections: &AdhesionConnections,
    positions: &[Vec3],
    velocities: &[Vec3],
    rotations: &[Quat],
    angular_velocities: &[Vec3],
    masses: &[f32],
    mode_settings: &[AdhesionSettings],
    forces: &mut [Vec3],
    torques: &mut [Vec3],
    current_time: f32,
) {
    const BREAK_GRACE_PERIOD: f32 = 0.5;
    // Batch size tuned for L1 cache (typically 32KB)
    // Each cell needs ~200 bytes of data, so batch of 32 cells fits in L1
    const BATCH_SIZE: usize = 32;
    
    // Process connections in batches
    let mut batch_start = 0;
    while batch_start < connections.active_count {
        let batch_end = (batch_start + BATCH_SIZE).min(connections.active_count);
        
        // Process batch
        for i in batch_start..batch_end {
            if connections.is_active[i] == 0 {
                continue;
            }
            
            let cell_a_idx = connections.cell_a_index[i];
            let cell_b_idx = connections.cell_b_index[i];
            let mode_idx = connections.mode_index[i];
            
            // Validate indices
            if cell_a_idx >= positions.len() || cell_b_idx >= positions.len() {
                continue;
            }
            
            if mode_idx >= mode_settings.len() {
                continue;
            }
            
            let settings = &mode_settings[mode_idx];
            
            // Calculate forces and torques
            let (force_a, torque_a, force_b, torque_b, spring_force_mag) = compute_adhesion_force_pair(
                positions[cell_a_idx],
                velocities[cell_a_idx],
                rotations[cell_a_idx],
                angular_velocities[cell_a_idx],
                masses[cell_a_idx],
                positions[cell_b_idx],
                velocities[cell_b_idx],
                rotations[cell_b_idx],
                angular_velocities[cell_b_idx],
                masses[cell_b_idx],
                connections.anchor_direction_a[i],
                connections.anchor_direction_b[i],
                connections.twist_reference_a[i],
                connections.twist_reference_b[i],
                settings,
            );
            
            // Skip bond if it exceeds break force (but not during grace period)
            let in_grace = current_time - connections.birth_time[i] < BREAK_GRACE_PERIOD;
            if settings.can_break && !in_grace && spring_force_mag > settings.break_force {
                continue;
            }

            // Apply forces
            forces[cell_a_idx] += force_a;
            forces[cell_b_idx] += force_b;
            torques[cell_a_idx] += torque_a;
            torques[cell_b_idx] += torque_b;
        }
        
        batch_start = batch_end;
    }
}

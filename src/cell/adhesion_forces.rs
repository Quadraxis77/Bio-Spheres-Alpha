use super::adhesion::{AdhesionConnections, BOND_FLAG_BARRIER_BALL};
use crate::genome::AdhesionSettings;
use glam::{Quat, Vec3};

/// Numerical precision constants (matching GPU/C++)
#[allow(dead_code)]
const EPSILON: f32 = 1e-6;
const ANGLE_EPSILON: f32 = 0.001;
const QUATERNION_EPSILON: f32 = 0.0001;
const TWIST_CLAMP_LIMIT: f32 = 1.57; // 90 degrees

/// Per-bond force cap. Without this, large blobs accumulate phantom net forces
/// because spring errors across many bonds don't cancel perfectly - the residual
/// is enough to produce locomotion. Matches the GPU adhesion_physics.wgsl cap.
const MAX_BOND_FORCE: f32 = 500.0;

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
    genome_orientations: &[Quat],
    mode_settings: &[AdhesionSettings],
    forces: &mut [Vec3],
    torques: &mut [Vec3],
    current_time: f32,
    dt: f32,
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

        let genome_rot_a = if cell_a_idx < genome_orientations.len() {
            genome_orientations[cell_a_idx]
        } else {
            rotations[cell_a_idx]
        };
        let genome_rot_b = if cell_b_idx < genome_orientations.len() {
            genome_orientations[cell_b_idx]
        } else {
            rotations[cell_b_idx]
        };

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
            genome_rot_a,
            genome_rot_b,
            connections.anchor_direction_a[i],
            connections.anchor_direction_b[i],
            connections.twist_reference_a[i],
            connections.twist_reference_b[i],
            settings,
            connections.rest_length_overrides[i],
            (connections.bond_flags[i] & BOND_FLAG_BARRIER_BALL) != 0,
            0.0,
            0.0,
            dt,
            (current_time - connections.birth_time[i]).max(0.0),
        );

        // Check break condition before applying forces
        // Skip break check during grace period after bond creation
        let in_grace = current_time - connections.birth_time[i] < BREAK_GRACE_PERIOD;
        if settings.can_break && !in_grace && spring_force_mag > settings.break_force {
            bonds_to_break.push(i);
            continue;
        }

        // Cap per-bond force before accumulating to prevent phantom locomotion
        // in large blobs where spring errors across many bonds don't cancel.
        let fa_mag = force_a.length();
        let fb_mag = force_b.length();
        let capped_a = if fa_mag > MAX_BOND_FORCE {
            force_a * (MAX_BOND_FORCE / fa_mag)
        } else {
            force_a
        };
        let capped_b = if fb_mag > MAX_BOND_FORCE {
            force_b * (MAX_BOND_FORCE / fb_mag)
        } else {
            force_b
        };

        forces[cell_a_idx] += capped_a;
        forces[cell_b_idx] += capped_b;
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
    genome_orientations: &[Quat],
    mode_settings: &[AdhesionSettings],
    forces: &mut [Vec3],
    torques: &mut [Vec3],
    current_time: f32,
    muscle_contractions: &[f32],
    dt: f32,
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

            let genome_rot_a = if cell_a_idx < genome_orientations.len() {
                genome_orientations[cell_a_idx]
            } else {
                rotations[cell_a_idx]
            };
            let genome_rot_b = if cell_b_idx < genome_orientations.len() {
                genome_orientations[cell_b_idx]
            } else {
                rotations[cell_b_idx]
            };

            let settings = &mode_settings[mode_idx];

            // Get per-cell muscle contraction values
            let contraction_a = if cell_a_idx < muscle_contractions.len() {
                muscle_contractions[cell_a_idx]
            } else {
                0.0
            };
            let contraction_b = if cell_b_idx < muscle_contractions.len() {
                muscle_contractions[cell_b_idx]
            } else {
                0.0
            };

            let (force_a, torque_a, force_b, torque_b, spring_force_mag) =
                compute_adhesion_force_pair(
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
                    genome_rot_a,
                    genome_rot_b,
                    connections.anchor_direction_a[i],
                    connections.anchor_direction_b[i],
                    connections.twist_reference_a[i],
                    connections.twist_reference_b[i],
                    settings,
                    connections.rest_length_overrides[i],
                    (connections.bond_flags[i] & BOND_FLAG_BARRIER_BALL) != 0,
                    contraction_a,
                    contraction_b,
                    dt,
                    (current_time - connections.birth_time[i]).max(0.0),
                );

            // Bond breaks: signal with Some(i), zero forces
            // Skip break check during grace period after bond creation
            let in_grace = current_time - connections.birth_time[i] < BREAK_GRACE_PERIOD;
            if settings.can_break && !in_grace && spring_force_mag > settings.break_force {
                return vec![(Some(i), 0, Vec3::ZERO, Vec3::ZERO)];
            }

            // Cap per-bond force before accumulating
            let fa_mag = force_a.length();
            let fb_mag = force_b.length();
            let capped_a = if fa_mag > MAX_BOND_FORCE {
                force_a * (MAX_BOND_FORCE / fa_mag)
            } else {
                force_a
            };
            let capped_b = if fb_mag > MAX_BOND_FORCE {
                force_b * (MAX_BOND_FORCE / fb_mag)
            } else {
                force_b
            };

            vec![
                (None, cell_a_idx, capped_a, torque_a),
                (None, cell_b_idx, capped_b, torque_b),
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

/// Compute adhesion forces for a single connection pair.
/// Matches GPU adhesion_physics.wgsl exactly:
/// - Physics rotations for all springs (geometric, orientation, twist)
/// - Anchor-based geometric spring (not simple distance spring)
/// - Full-strength twist constraints (no reduction factors)
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_adhesion_force_pair(
    pos_a: Vec3,
    vel_a: Vec3,
    rot_a: Quat,
    ang_vel_a: Vec3,
    mass_a: f32,
    pos_b: Vec3,
    vel_b: Vec3,
    rot_b: Quat,
    ang_vel_b: Vec3,
    mass_b: f32,
    _genome_rot_a: Quat,
    _genome_rot_b: Quat,
    anchor_dir_a: Vec3,
    anchor_dir_b: Vec3,
    twist_ref_a: Quat,
    twist_ref_b: Quat,
    settings: &AdhesionSettings,
    rest_length_override: f32,
    is_ball_joint: bool,
    contraction_a: f32,
    contraction_b: f32,
    dt: f32,
    bond_age: f32,
) -> (Vec3, Vec3, Vec3, Vec3, f32) {
    let mut force_a = Vec3::ZERO;
    let mut torque_a = Vec3::ZERO;
    let mut force_b = Vec3::ZERO;
    let mut torque_b = Vec3::ZERO;

    let delta_pos = pos_b - pos_a;
    let dist = delta_pos.length();
    if dist < QUATERNION_EPSILON {
        return (force_a, torque_a, force_b, torque_b, 0.0);
    }

    let adhesion_dir = delta_pos / dist;
    let rest_length = if rest_length_override > 0.0 {
        rest_length_override
    } else {
        settings.rest_length
    };

    // Per-cell contraction: each cell's contraction reduces the total rest length by half.
    // One myocyte at full contraction (1.0) shortens the bond by 50%.
    // Two myocytes at full contraction shorten it to zero.
    let effective_rest_length =
        rest_length * (1.0 - contraction_a * 0.5 - contraction_b * 0.5).max(0.0);
    if is_ball_joint {
        const SETTLE_DURATION: f32 = 0.3;
        let settle_factor = (bond_age / SETTLE_DURATION).clamp(0.0, 1.0);
        let spring =
            (dist - effective_rest_length) * settings.linear_spring_stiffness * settle_factor;
        let rel_vel = vel_b - vel_a;
        let damping = settings.linear_spring_damping * rel_vel.dot(adhesion_dir);
        let force = adhesion_dir * (spring + damping);
        return (force, Vec3::ZERO, -force, Vec3::ZERO, force.length());
    }

    // Transform anchor directions to world space using physics rotations.
    // Both the geometric spring and orientation spring use physics rotations so the
    // structure moves freely with the creature.
    let anchor_a = if anchor_dir_a.length() < ANGLE_EPSILON && anchor_dir_b.length() < ANGLE_EPSILON
    {
        Vec3::X
    } else {
        rotate_vector_by_quaternion(anchor_dir_a, rot_a)
    };
    let anchor_b = if anchor_dir_a.length() < ANGLE_EPSILON && anchor_dir_b.length() < ANGLE_EPSILON
    {
        -Vec3::X
    } else {
        rotate_vector_by_quaternion(anchor_dir_b, rot_b)
    };
    // Each cell uses its own contracted reach for its anchor.
    // Symmetric anchor spring: each cell is pulled toward the position defined by the
    // OTHER cell's anchor. This naturally enforces both bond length and inter-cell angle,
    // and is exactly zero at the true equilibrium (dist == rest_length and anchors aligned).
    //
    // Settle ramp: newly created bonds start soft and ramp to full stiffness over
    // SETTLE_DURATION seconds. This lets cells find their natural equilibrium positions
    // before the geometric spring locks them in place, preventing bonds from fighting
    // each other during the initial placement phase.
    const SETTLE_DURATION: f32 = 0.3;
    let settle_factor = (bond_age / SETTLE_DURATION).clamp(0.0, 1.0);
    let effective_stiffness = settings.linear_spring_stiffness * settle_factor;

    let target_b = pos_a + anchor_a * effective_rest_length;
    let target_a = pos_b + anchor_b * effective_rest_length;
    let error_a = target_a - pos_a;
    let error_b = target_b - pos_b;
    let geo_force_on_a = (error_a - error_b) * 0.5 * effective_stiffness;
    let spring_force_mag = geo_force_on_a.length();

    // Linear damping: pure velocity-damping along the bond axis.
    // Only the component of relative velocity along the bond is damped,
    // so there is no constant-bias force when cells are stationary.
    let rel_vel = vel_b - vel_a;
    let rel_vel_along_bond = rel_vel.dot(adhesion_dir);
    let damping_force = adhesion_dir * (settings.linear_spring_damping * rel_vel_along_bond);

    force_a += geo_force_on_a + damping_force;
    force_b -= geo_force_on_a + damping_force;

    // Orientation spring: aligns each cell's anchor toward the live adhesion direction.
    // This is what enforces the angular shape - it pulls the cell's anchor to point at
    // its bonded neighbor, creating the rigid inter-cell angle defined by the genome.
    let axis_a = anchor_a.cross(adhesion_dir);
    let sin_a = axis_a.length();
    let cos_a = anchor_a.dot(adhesion_dir);
    let angle_a = sin_a.atan2(cos_a);
    if sin_a > QUATERNION_EPSILON {
        let axis_a_norm = axis_a.normalize();
        torque_a += axis_a_norm * angle_a * settings.orientation_spring_stiffness;
        torque_a += -axis_a_norm * ang_vel_a.dot(axis_a_norm) * settings.orientation_spring_damping;
    }
    // Damp spin around the bond axis itself - the orientation spring's axis is always
    // perpendicular to adhesion_dir so it cannot see bond-axis spin. This plugs that gap.
    torque_a -=
        adhesion_dir * ang_vel_a.dot(adhesion_dir) * settings.orientation_spring_damping * 0.5;

    // Orientation spring for cell B: aligns B's anchor toward -adhesion_dir (toward A)
    let axis_b = anchor_b.cross(-adhesion_dir);
    let sin_b = axis_b.length();
    let cos_b = anchor_b.dot(-adhesion_dir);
    let angle_b = sin_b.atan2(cos_b);
    if sin_b > QUATERNION_EPSILON {
        let axis_b_norm = axis_b.normalize();
        torque_b += axis_b_norm * angle_b * settings.orientation_spring_stiffness;
        torque_b += -axis_b_norm * ang_vel_b.dot(axis_b_norm) * settings.orientation_spring_damping;
    }
    // Same bond-axis damping for cell B.
    torque_b -=
        adhesion_dir * ang_vel_b.dot(adhesion_dir) * settings.orientation_spring_damping * 0.5;

    // Relative twist constraint: constrains A's rotation relative to B about the bond axis.
    // This allows the whole structure to spin freely while preventing cells from
    // twisting against each other beyond their birth relative orientation.
    if settings.enable_twist_constraint
        && twist_ref_a.length() > ANGLE_EPSILON
        && twist_ref_b.length() > ANGLE_EPSILON
    {
        let adhesion_axis = delta_pos.normalize();

        let birth_relative = (twist_ref_b.conjugate() * twist_ref_a).normalize();
        let current_relative = (rot_b.conjugate() * rot_a).normalize();
        let twist_error_quat = (current_relative * birth_relative.conjugate()).normalize();

        // Extract twist component about the adhesion axis directly from the quaternion's
        // imaginary part. This avoids the axis-angle double-cover ambiguity that causes
        // the spring to fire in the wrong direction when the error crosses the hemisphere.
        // twist_scalar ~= sin(half_angle) * sign, which is monotone in [-1, 1] for 180 deg.
        let twist_imag = Vec3::new(twist_error_quat.x, twist_error_quat.y, twist_error_quat.z);
        let twist_error_scalar = twist_imag
            .dot(adhesion_axis)
            .clamp(-TWIST_CLAMP_LIMIT, TWIST_CLAMP_LIMIT);

        let angular_vel_a_proj = ang_vel_a.dot(adhesion_axis);
        let angular_vel_b_proj = ang_vel_b.dot(adhesion_axis);
        let relative_angular_vel = angular_vel_a_proj - angular_vel_b_proj;

        // Clamp the damping coefficient to the explicit-integration stability limit.
        // A relative angular damper applied as comega_rel is only stable when
        // cdt(1/I_a + 1/I_b) <= 1, otherwise it overshoots and reverses omega_rel,
        // pumping energy into the structure until it spins out and collapses into a blob.
        // I = 0.4mr^2 (solid sphere), matching the angular integrator.
        let damp = stable_twist_damping(settings.twist_constraint_damping, mass_a, mass_b, dt);

        let twist_spring = adhesion_axis * twist_error_scalar * settings.twist_constraint_stiffness;
        let twist_damp = adhesion_axis * relative_angular_vel * damp;
        torque_a += -twist_spring - twist_damp;
        torque_b += twist_spring + twist_damp;
    }

    (force_a, torque_a, force_b, torque_b, spring_force_mag)
}

/// Clamp a relative angular damping coefficient to the explicit-integration stability
/// limit. Returns the largest coefficient <= the requested value that keeps the per-step
/// relative-angular-velocity update non-divergent: cdt(1/I_a + 1/I_b) <= 1.
/// Moment of inertia uses I = 0.4mr^2 with r = clamp(mass, 0.5, 2.0), matching the
/// angular velocity integrator.
#[inline]
fn stable_twist_damping(requested: f32, mass_a: f32, mass_b: f32, dt: f32) -> f32 {
    if requested <= 0.0 || dt <= 0.0 {
        return requested.max(0.0);
    }
    let inertia = |m: f32| {
        let r = m.clamp(0.5, 2.0);
        0.4 * m * r * r
    };
    let inv_inertia_sum = 1.0 / inertia(mass_a).max(1e-4) + 1.0 / inertia(mass_b).max(1e-4);
    // Use 0.9 of the theoretical limit for a safety margin against the spring term.
    let max_stable = 0.9 / (dt * inv_inertia_sum);
    requested.min(max_stable)
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

/// Compute one adhesion constraint substep iteration (mirrors GPU adhesion_substep.wgsl).
///
/// Unlike the main adhesion force pass which accumulates into force/torque buffers,
/// this directly integrates positions, velocities, rotations, and angular velocities.
/// Uses genome orientations for geometric spring anchors (genome-pure) and physics
/// rotations for orientation/twist constraints (detects actual misalignment).
///
/// Run N iterations after the main physics pass for stiffer joints.
#[allow(clippy::too_many_arguments)]
pub fn compute_adhesion_substep(
    connections: &AdhesionConnections,
    positions: &mut [Vec3],
    velocities: &mut [Vec3],
    rotations: &mut [Quat],
    angular_velocities: &mut [Vec3],
    masses: &[f32],
    _genome_orientations: &[Quat],
    mode_settings: &[AdhesionSettings],
    cell_count: usize,
    dt: f32,
    muscle_contractions: &[f32],
    angular_damping: f32,
) {
    const MAX_FORCE: f32 = 5000.0;
    const MAX_TORQUE: f32 = 500.0;
    const MAX_SPEED: f32 = 500.0;

    // Accumulate per-cell forces and torques (Jacobi-style: read current, write corrections)
    let mut cell_forces: Vec<Vec3> = vec![Vec3::ZERO; cell_count];
    let mut cell_torques: Vec<Vec3> = vec![Vec3::ZERO; cell_count];

    // Compute force contributions from all active connections
    for i in 0..connections.active_count {
        if connections.is_active[i] == 0 {
            continue;
        }

        let cell_a_idx = connections.cell_a_index[i];
        let cell_b_idx = connections.cell_b_index[i];
        let mode_idx = connections.mode_index[i];

        if cell_a_idx >= cell_count || cell_b_idx >= cell_count {
            continue;
        }
        if mode_idx >= mode_settings.len() {
            continue;
        }

        let settings = &mode_settings[mode_idx];

        // Compute substep forces for this pair
        let contraction_a = if cell_a_idx < muscle_contractions.len() {
            muscle_contractions[cell_a_idx]
        } else {
            0.0
        };
        let contraction_b = if cell_b_idx < muscle_contractions.len() {
            muscle_contractions[cell_b_idx]
        } else {
            0.0
        };

        let (force_a, torque_a, force_b, torque_b) = compute_substep_force_pair(
            positions[cell_a_idx],
            velocities[cell_a_idx],
            rotations[cell_a_idx],
            angular_velocities[cell_a_idx],
            positions[cell_b_idx],
            velocities[cell_b_idx],
            rotations[cell_b_idx],
            angular_velocities[cell_b_idx],
            rotations[cell_a_idx],
            rotations[cell_b_idx],
            connections.anchor_direction_a[i],
            connections.anchor_direction_b[i],
            connections.twist_reference_a[i],
            connections.twist_reference_b[i],
            settings,
            connections.rest_length_overrides[i],
            (connections.bond_flags[i] & BOND_FLAG_BARRIER_BALL) != 0,
            contraction_a,
            contraction_b,
            masses[cell_a_idx],
            masses[cell_b_idx],
            dt,
        );

        cell_forces[cell_a_idx] += force_a;
        cell_forces[cell_b_idx] += force_b;
        cell_torques[cell_a_idx] += torque_a;
        cell_torques[cell_b_idx] += torque_b;
    }

    // Integrate directly (matching GPU adhesion_substep.wgsl lines 462-503)
    for idx in 0..cell_count {
        let total_force = cell_forces[idx];
        let total_torque = cell_torques[idx];

        if total_force.length_squared() < 1e-12 && total_torque.length_squared() < 1e-12 {
            continue;
        }

        // Clamp forces and torques
        let force_mag = total_force.length();
        let torque_mag = total_torque.length();
        let clamped_force = if force_mag > MAX_FORCE {
            total_force * (MAX_FORCE / force_mag)
        } else {
            total_force
        };
        let clamped_torque = if torque_mag > MAX_TORQUE {
            total_torque * (MAX_TORQUE / torque_mag)
        } else {
            total_torque
        };

        let safe_mass = masses[idx].max(0.001);
        let acceleration = clamped_force / safe_mass;

        let old_vel = velocities[idx];
        let mut new_vel = old_vel + acceleration * dt;

        // Clamp velocity
        let speed = new_vel.length();
        if speed > MAX_SPEED {
            new_vel = new_vel * (MAX_SPEED / speed);
        }

        // Position correction: only apply the velocity *change* (matches GPU: new_pos = my_pos + (new_vel - my_vel) * dt)
        let new_pos = positions[idx] + (new_vel - old_vel) * dt;

        // Angular integration with damping
        let radius = masses[idx].clamp(0.5, 2.0); // calculate_radius_from_mass
        let moment_of_inertia = 0.4 * safe_mass * radius * radius;
        let mut new_ang_vel = angular_velocities[idx];
        if moment_of_inertia > 0.0001 {
            let angular_acceleration = clamped_torque / moment_of_inertia;
            // Match integrate_angular_velocities: powf(dt * 100.0) so damping is
            // consistent regardless of substep dt. Without the * 100.0 the substep
            // damps ~100x less than the main pass.
            let angular_damping_factor = angular_damping.powf(dt * 100.0);
            new_ang_vel =
                (angular_velocities[idx] + angular_acceleration * dt) * angular_damping_factor;
        }

        // Integrate rotation
        let ang_vel_mag = new_ang_vel.length();
        let mut new_rot = rotations[idx];
        if ang_vel_mag > 0.0001 {
            let angle = ang_vel_mag * dt;
            let axis = new_ang_vel / ang_vel_mag;
            let delta_rotation = Quat::from_axis_angle(axis, angle);
            new_rot = (delta_rotation * rotations[idx]).normalize();
        }

        // Write corrected state
        positions[idx] = new_pos;
        velocities[idx] = new_vel;
        rotations[idx] = new_rot;
        angular_velocities[idx] = new_ang_vel;
    }
}

/// Compute substep forces for a single connection pair.
/// Mirrors GPU adhesion_substep.wgsl compute_substep_forces().
/// Uses genome orientations for geometric spring, physics rotations for orientation/twist.
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_substep_force_pair(
    pos_a: Vec3,
    vel_a: Vec3,
    rot_a: Quat,
    ang_vel_a: Vec3,
    pos_b: Vec3,
    vel_b: Vec3,
    rot_b: Quat,
    ang_vel_b: Vec3,
    genome_rot_a: Quat,
    genome_rot_b: Quat,
    anchor_dir_a: Vec3,
    anchor_dir_b: Vec3,
    twist_ref_a: Quat,
    twist_ref_b: Quat,
    settings: &AdhesionSettings,
    rest_length_override: f32,
    is_ball_joint: bool,
    contraction_a: f32,
    contraction_b: f32,
    mass_a: f32,
    mass_b: f32,
    dt: f32,
) -> (Vec3, Vec3, Vec3, Vec3) {
    let mut force_a = Vec3::ZERO;
    let mut torque_a = Vec3::ZERO;
    let mut force_b = Vec3::ZERO;
    let mut torque_b = Vec3::ZERO;

    let delta_pos = pos_b - pos_a;
    let dist = delta_pos.length();
    if dist < QUATERNION_EPSILON {
        return (force_a, torque_a, force_b, torque_b);
    }

    let adhesion_dir = delta_pos / dist;
    let rest_length = if rest_length_override > 0.0 {
        rest_length_override
    } else {
        settings.rest_length
    };
    let effective_rest_length =
        rest_length * (1.0 - contraction_a * 0.5 - contraction_b * 0.5).max(0.0);
    if is_ball_joint {
        let spring = (dist - effective_rest_length) * settings.linear_spring_stiffness;
        let rel_vel = vel_b - vel_a;
        let damping = settings.linear_spring_damping * rel_vel.dot(adhesion_dir);
        let force = adhesion_dir * (spring + damping);
        return (force, Vec3::ZERO, -force, Vec3::ZERO);
    }

    // Geometric spring using GENOME orientations (matches GPU)
    let genome_anchor_a =
        if anchor_dir_a.length() < ANGLE_EPSILON && anchor_dir_b.length() < ANGLE_EPSILON {
            Vec3::X
        } else {
            rotate_vector_by_quaternion(anchor_dir_a, genome_rot_a)
        };
    let genome_anchor_b =
        if anchor_dir_a.length() < ANGLE_EPSILON && anchor_dir_b.length() < ANGLE_EPSILON {
            -Vec3::X
        } else {
            rotate_vector_by_quaternion(anchor_dir_b, genome_rot_b)
        };

    // Anchor-based geometric spring force (matches GPU, NOT the simple distance spring)
    let target_b = pos_a + genome_anchor_a * effective_rest_length;
    let target_a = pos_b + genome_anchor_b * effective_rest_length;
    let error_a = target_a - pos_a;
    let error_b = target_b - pos_b;
    let geo_force = (error_a - error_b) * 0.5 * settings.linear_spring_stiffness;

    // Linear damping: pure velocity-damping along the bond axis.
    let rel_vel = vel_b - vel_a;
    let rel_vel_along_bond = rel_vel.dot(adhesion_dir);
    let damping_force = adhesion_dir * (settings.linear_spring_damping * rel_vel_along_bond);

    force_a += geo_force + damping_force;
    force_b -= geo_force + damping_force;

    // Physics-based anchors for orientation/twist springs (detects actual rotation)
    let phys_anchor_a =
        if anchor_dir_a.length() < ANGLE_EPSILON && anchor_dir_b.length() < ANGLE_EPSILON {
            Vec3::X
        } else {
            rotate_vector_by_quaternion(anchor_dir_a, rot_a)
        };
    let phys_anchor_b =
        if anchor_dir_a.length() < ANGLE_EPSILON && anchor_dir_b.length() < ANGLE_EPSILON {
            -Vec3::X
        } else {
            rotate_vector_by_quaternion(anchor_dir_b, rot_b)
        };

    // Orientation spring for cell A: aligns A's anchor toward the live adhesion direction.
    // Matches GPU adhesion_physics.wgsl orientation spring logic.
    let axis_a = phys_anchor_a.cross(adhesion_dir);
    let sin_a = axis_a.length();
    let cos_a = phys_anchor_a.dot(adhesion_dir);
    let angle_a = sin_a.atan2(cos_a);
    if sin_a > QUATERNION_EPSILON {
        let axis_a_norm = axis_a.normalize();
        torque_a += axis_a_norm * angle_a * settings.orientation_spring_stiffness;
        torque_a += -axis_a_norm * ang_vel_a.dot(axis_a_norm) * settings.orientation_spring_damping;
    }
    // Damp spin around the bond axis itself - the orientation spring's axis is always
    // perpendicular to adhesion_dir so it cannot see bond-axis spin. This plugs that gap.
    torque_a -=
        adhesion_dir * ang_vel_a.dot(adhesion_dir) * settings.orientation_spring_damping * 0.5;

    // Orientation spring for cell B: aligns B's anchor toward -adhesion_dir (toward A)
    let axis_b = phys_anchor_b.cross(-adhesion_dir);
    let sin_b = axis_b.length();
    let cos_b = phys_anchor_b.dot(-adhesion_dir);
    let angle_b = sin_b.atan2(cos_b);
    if sin_b > QUATERNION_EPSILON {
        let axis_b_norm = axis_b.normalize();
        torque_b += axis_b_norm * angle_b * settings.orientation_spring_stiffness;
        torque_b += -axis_b_norm * ang_vel_b.dot(axis_b_norm) * settings.orientation_spring_damping;
    }
    // Same bond-axis damping for cell B.
    torque_b -=
        adhesion_dir * ang_vel_b.dot(adhesion_dir) * settings.orientation_spring_damping * 0.5;

    // Relative twist constraint: constrains A's rotation relative to B about the bond axis.
    if settings.enable_twist_constraint
        && twist_ref_a.length() > ANGLE_EPSILON
        && twist_ref_b.length() > ANGLE_EPSILON
    {
        let adhesion_axis = delta_pos.normalize();

        let birth_relative = (twist_ref_b.conjugate() * twist_ref_a).normalize();
        let current_relative = (rot_b.conjugate() * rot_a).normalize();
        let twist_error_quat = (current_relative * birth_relative.conjugate()).normalize();

        // Extract twist component about the adhesion axis directly from the quaternion's
        // imaginary part, avoiding the axis-angle double-cover ambiguity.
        let twist_imag = Vec3::new(twist_error_quat.x, twist_error_quat.y, twist_error_quat.z);
        let twist_error_scalar = twist_imag
            .dot(adhesion_axis)
            .clamp(-TWIST_CLAMP_LIMIT, TWIST_CLAMP_LIMIT);

        let angular_vel_a_proj = ang_vel_a.dot(adhesion_axis);
        let angular_vel_b_proj = ang_vel_b.dot(adhesion_axis);
        let relative_angular_vel = angular_vel_a_proj - angular_vel_b_proj;

        let twist_spring = adhesion_axis * twist_error_scalar * settings.twist_constraint_stiffness;
        let twist_damp = adhesion_axis
            * relative_angular_vel
            * stable_twist_damping(settings.twist_constraint_damping, mass_a, mass_b, dt);
        torque_a += -twist_spring - twist_damp;
        torque_b += twist_spring + twist_damp;
    }

    (force_a, torque_a, force_b, torque_b)
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
    genome_orientations: &[Quat],
    mode_settings: &[AdhesionSettings],
    forces: &mut [Vec3],
    torques: &mut [Vec3],
    current_time: f32,
    dt: f32,
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

            let genome_rot_a = if cell_a_idx < genome_orientations.len() {
                genome_orientations[cell_a_idx]
            } else {
                rotations[cell_a_idx]
            };
            let genome_rot_b = if cell_b_idx < genome_orientations.len() {
                genome_orientations[cell_b_idx]
            } else {
                rotations[cell_b_idx]
            };

            // Calculate forces and torques
            let (force_a, torque_a, force_b, torque_b, spring_force_mag) =
                compute_adhesion_force_pair(
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
                    genome_rot_a,
                    genome_rot_b,
                    connections.anchor_direction_a[i],
                    connections.anchor_direction_b[i],
                    connections.twist_reference_a[i],
                    connections.twist_reference_b[i],
                    settings,
                    connections.rest_length_overrides[i],
                    (connections.bond_flags[i] & BOND_FLAG_BARRIER_BALL) != 0,
                    0.0,
                    0.0,
                    dt,
                    (current_time - connections.birth_time[i]).max(0.0),
                );

            // Skip bond if it exceeds break force (but not during grace period)
            let in_grace = current_time - connections.birth_time[i] < BREAK_GRACE_PERIOD;
            if settings.can_break && !in_grace && spring_force_mag > settings.break_force {
                continue;
            }

            // Cap per-bond force before accumulating
            let fa_mag = force_a.length();
            let fb_mag = force_b.length();
            let capped_a = if fa_mag > MAX_BOND_FORCE {
                force_a * (MAX_BOND_FORCE / fa_mag)
            } else {
                force_a
            };
            let capped_b = if fb_mag > MAX_BOND_FORCE {
                force_b * (MAX_BOND_FORCE / fb_mag)
            } else {
                force_b
            };

            forces[cell_a_idx] += capped_a;
            forces[cell_b_idx] += capped_b;
            torques[cell_a_idx] += torque_a;
            torques[cell_b_idx] += torque_b;
        }

        batch_start = batch_end;
    }
}

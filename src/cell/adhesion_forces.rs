use glam::{Vec3, Quat};
use crate::genome::AdhesionSettings;
use super::adhesion::AdhesionConnections;

/// PBD adhesion solver constants
const ADHESION_ITERATIONS: usize = 8;
const MAX_PBD_CORRECTION: f32 = 8.0;
const MAX_HINGE_SPRING: f32 = 8.0;
const HINGE_CORRECTION_RATE: f32 = 0.8;
const TWIST_CORRECTION_RATE: f32 = 0.2;
const MAX_TWIST_CORRECTION: f32 = 0.5;

/// Solve all adhesion constraints using Position-Based Dynamics.
///
/// This directly modifies positions and rotations (no forces/torques).
/// Returns a list of connection indices whose bonds should be broken.
///
/// Three constraint passes:
/// 1. **Distance constraint** — iterative PBD keeping bonded cells at target distance
/// 2. **Hinge spring** — corrects cell orientation based on bond angle deviation,
///    with perpendicular translational forces for center-to-center lever action
/// 3. **Twist constraint** — hardcoded PBD twist correction using anchor reference frames
///
/// Bond breaking is evaluated after all constraint passes.
#[allow(clippy::too_many_arguments)]
pub fn solve_adhesion_pbd(
    connections: &AdhesionConnections,
    positions: &mut [Vec3],
    rotations: &mut [Quat],
    radii: &[f32],
    masses: &[f32],
    mode_settings: &[AdhesionSettings],
) -> Vec<usize> {
    let cell_count = positions.len();

    // =========================================================================
    // PASS 1: PBD DISTANCE CONSTRAINT (iterative position correction)
    // =========================================================================
    for _iter in 0..ADHESION_ITERATIONS {
        for i in 0..connections.active_count {
            if connections.is_active[i] == 0 {
                continue;
            }

            let idx_a = connections.cell_a_index[i];
            let idx_b = connections.cell_b_index[i];
            let mode_idx = connections.mode_index[i];

            if idx_a >= cell_count || idx_b >= cell_count || mode_idx >= mode_settings.len() {
                continue;
            }

            let settings = &mode_settings[mode_idx];

            let rest_offset = settings.adhesin_length * 50.0;
            let softness = 1.0 - settings.adhesin_stretch * 0.8;

            let delta = positions[idx_b] - positions[idx_a];
            let dist = delta.length();
            let sum_radii = radii[idx_a] + radii[idx_b];
            let target_dist = sum_radii + rest_offset;
            let error = dist - target_dist;

            let normal = if dist < 0.001 {
                if error.abs() < 0.001 {
                    continue;
                }
                // Fallback: use anchor_a direction in world space
                let anchor_world = rotations[idx_a] * connections.anchor_direction_a[i];
                if anchor_world.length() > 0.001 {
                    anchor_world.normalize()
                } else {
                    Vec3::X
                }
            } else {
                delta / dist
            };

            let correction = (error * softness).clamp(-MAX_PBD_CORRECTION, MAX_PBD_CORRECTION);

            let inv_m1 = 1.0 / masses[idx_a].max(0.001);
            let inv_m2 = 1.0 / masses[idx_b].max(0.001);
            let w_total = inv_m1 + inv_m2;

            if w_total > 1e-10 {
                let s = correction / w_total;
                positions[idx_a] += normal * s * inv_m1;
                positions[idx_b] -= normal * s * inv_m2;
            }
        }
    }

    // =========================================================================
    // PASS 2: HINGE RESTORATIVE SPRING (orientation + perpendicular lever forces)
    // =========================================================================
    for i in 0..connections.active_count {
        if connections.is_active[i] == 0 {
            continue;
        }

        let idx_a = connections.cell_a_index[i];
        let idx_b = connections.cell_b_index[i];
        let mode_idx = connections.mode_index[i];

        if idx_a >= cell_count || idx_b >= cell_count || mode_idx >= mode_settings.len() {
            continue;
        }

        let settings = &mode_settings[mode_idx];
        if settings.stiffness <= 0.0 {
            continue;
        }
        let correction_strength = HINGE_CORRECTION_RATE * settings.stiffness;

        let delta = positions[idx_b] - positions[idx_a];
        let dist = delta.length();
        if dist < 0.001 {
            continue;
        }
        let bond_dir = delta / dist;

        // Perpendicular direction for translational lever forces
        // In 3D we compute it per-cell from the deviation plane
        let inv_m1 = 1.0 / masses[idx_a].max(0.001);
        let inv_m2 = 1.0 / masses[idx_b].max(0.001);
        let total_inv_m = inv_m1 + inv_m2;

        // --- Cell A hinge ---
        {
            // Current bond direction in cell A's local frame
            let local_bond_dir_a = rotations[idx_a].conjugate() * bond_dir;
            // Rest anchor direction for cell A (stored in local space)
            let rest_anchor_a = connections.anchor_direction_a[i];

            if rest_anchor_a.length() > 0.001 {
                // Deviation: rotation from rest anchor to current local bond direction
                let cross_a = rest_anchor_a.cross(local_bond_dir_a);
                let sin_a = cross_a.length();
                let cos_a = rest_anchor_a.dot(local_bond_dir_a);
                let dev_angle_a = sin_a.atan2(cos_a);

                if sin_a > 0.0001 {
                    let axis_local = cross_a / sin_a; // normalized rotation axis in local space
                    let correction_angle = dev_angle_a * correction_strength;

                    // Apply orientation correction to cell A
                    let half_angle = correction_angle * 0.5;
                    let delta_rot = Quat::from_xyzw(
                        axis_local.x * half_angle.sin(),
                        axis_local.y * half_angle.sin(),
                        axis_local.z * half_angle.sin(),
                        half_angle.cos(),
                    ).normalize();
                    rotations[idx_a] = (rotations[idx_a] * delta_rot).normalize();

                    // Perpendicular translational correction (lever action)
                    if total_inv_m > 1e-10 {
                        // Axis in world space
                        let axis_world = rotations[idx_a] * axis_local;
                        // Perpendicular to bond in the deviation plane
                        let perp = axis_world.cross(bond_dir);
                        if perp.length() > 0.001 {
                            let perp_n = perp.normalize();
                            let trans = (dev_angle_a * correction_strength * dist)
                                .clamp(-MAX_HINGE_SPRING, MAX_HINGE_SPRING);
                            positions[idx_a] += perp_n * trans * (inv_m1 / total_inv_m);
                            positions[idx_b] -= perp_n * trans * (inv_m2 / total_inv_m);
                        }
                    }
                }
            }
        }

        // --- Cell B hinge ---
        {
            // Cell B's anchor should point toward A, so negate bond direction
            let local_bond_dir_b = rotations[idx_b].conjugate() * (-bond_dir);
            let rest_anchor_b = connections.anchor_direction_b[i];

            if rest_anchor_b.length() > 0.001 {
                let cross_b = rest_anchor_b.cross(local_bond_dir_b);
                let sin_b = cross_b.length();
                let cos_b = rest_anchor_b.dot(local_bond_dir_b);
                let dev_angle_b = sin_b.atan2(cos_b);

                if sin_b > 0.0001 {
                    let axis_local = cross_b / sin_b;
                    let correction_angle = dev_angle_b * correction_strength;

                    let half_angle = correction_angle * 0.5;
                    let delta_rot = Quat::from_xyzw(
                        axis_local.x * half_angle.sin(),
                        axis_local.y * half_angle.sin(),
                        axis_local.z * half_angle.sin(),
                        half_angle.cos(),
                    ).normalize();
                    rotations[idx_b] = (rotations[idx_b] * delta_rot).normalize();

                    if total_inv_m > 1e-10 {
                        let axis_world = rotations[idx_b] * axis_local;
                        let perp = axis_world.cross(-bond_dir);
                        if perp.length() > 0.001 {
                            let perp_n = perp.normalize();
                            let trans = (dev_angle_b * correction_strength * dist)
                                .clamp(-MAX_HINGE_SPRING, MAX_HINGE_SPRING);
                            positions[idx_b] += perp_n * trans * (inv_m2 / total_inv_m);
                            positions[idx_a] -= perp_n * trans * (inv_m1 / total_inv_m);
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // PASS 3: HARDCODED PBD TWIST CONSTRAINT
    // =========================================================================
    // Prevents cells from spinning around the bond axis.
    // Uses twist reference quaternions stored at bond creation time.
    // The twist component is extracted by projecting the full rotation
    // correction onto the bond axis, then applying a fractional correction.
    for i in 0..connections.active_count {
        if connections.is_active[i] == 0 {
            continue;
        }

        let idx_a = connections.cell_a_index[i];
        let idx_b = connections.cell_b_index[i];

        if idx_a >= cell_count || idx_b >= cell_count {
            continue;
        }

        let twist_ref_a = connections.twist_reference_a[i];
        let twist_ref_b = connections.twist_reference_b[i];

        // Skip if no twist references stored
        if twist_ref_a.length() < 0.001 || twist_ref_b.length() < 0.001 {
            continue;
        }

        let delta = positions[idx_b] - positions[idx_a];
        let dist = delta.length();
        if dist < 0.001 {
            continue;
        }
        let bond_axis = delta / dist;

        // --- Twist correction for cell A ---
        {
            let anchor_world_a = rotations[idx_a] * connections.anchor_direction_a[i];
            let target_dir_a = bond_axis;
            let alignment_rot = quat_from_two_vectors(anchor_world_a, target_dir_a);
            let target_orientation = (alignment_rot * twist_ref_a).normalize();
            let correction_rot = (target_orientation * rotations[idx_a].conjugate()).normalize();

            // Extract twist component around bond axis
            let (axis, angle) = quat_to_axis_angle_pair(correction_rot);
            let twist_amount = (angle * axis.dot(bond_axis))
                .clamp(-MAX_TWIST_CORRECTION, MAX_TWIST_CORRECTION);

            if twist_amount.abs() > 0.0001 {
                let half = twist_amount * TWIST_CORRECTION_RATE * 0.5;
                let twist_delta = Quat::from_xyzw(
                    bond_axis.x * half.sin(),
                    bond_axis.y * half.sin(),
                    bond_axis.z * half.sin(),
                    half.cos(),
                ).normalize();
                rotations[idx_a] = (twist_delta * rotations[idx_a]).normalize();
            }
        }

        // --- Twist correction for cell B ---
        {
            let anchor_world_b = rotations[idx_b] * connections.anchor_direction_b[i];
            let target_dir_b = -bond_axis;
            let alignment_rot = quat_from_two_vectors(anchor_world_b, target_dir_b);
            let target_orientation = (alignment_rot * twist_ref_b).normalize();
            let correction_rot = (target_orientation * rotations[idx_b].conjugate()).normalize();

            let (axis, angle) = quat_to_axis_angle_pair(correction_rot);
            let twist_amount = (angle * axis.dot(bond_axis))
                .clamp(-MAX_TWIST_CORRECTION, MAX_TWIST_CORRECTION);

            if twist_amount.abs() > 0.0001 {
                let half = twist_amount * TWIST_CORRECTION_RATE * 0.5;
                let twist_delta = Quat::from_xyzw(
                    bond_axis.x * half.sin(),
                    bond_axis.y * half.sin(),
                    bond_axis.z * half.sin(),
                    half.cos(),
                ).normalize();
                rotations[idx_b] = (twist_delta * rotations[idx_b]).normalize();
            }
        }
    }

    // =========================================================================
    // BOND BREAKING (stretch-distance based)
    // =========================================================================
    let mut bonds_to_remove = Vec::new();
    for i in 0..connections.active_count {
        if connections.is_active[i] == 0 {
            continue;
        }

        let idx_a = connections.cell_a_index[i];
        let idx_b = connections.cell_b_index[i];
        let mode_idx = connections.mode_index[i];

        if idx_a >= cell_count || idx_b >= cell_count || mode_idx >= mode_settings.len() {
            continue;
        }

        let settings = &mode_settings[mode_idx];
        if !settings.can_break {
            continue;
        }

        let delta = positions[idx_b] - positions[idx_a];
        let dist = delta.length();
        let sum_radii = radii[idx_a] + radii[idx_b];
        let bond_length_offset = settings.adhesin_length * 50.0;
        let max_stretch_dist =
            sum_radii * (1.3 + settings.adhesin_stretch * 3.0) + bond_length_offset;

        if dist > max_stretch_dist {
            bonds_to_remove.push(i);
        }
    }

    bonds_to_remove
}

/// Compute adhesion PBD for all active connections - Parallel version (stub)
///
/// For now this delegates to the sequential solver since PBD position corrections
/// are inherently order-dependent. A future Jacobi-style parallel solver could
/// be implemented here.
#[allow(clippy::too_many_arguments)]
pub fn solve_adhesion_pbd_parallel(
    connections: &AdhesionConnections,
    positions: &mut [Vec3],
    rotations: &mut [Quat],
    radii: &[f32],
    masses: &[f32],
    mode_settings: &[AdhesionSettings],
) -> Vec<usize> {
    solve_adhesion_pbd(connections, positions, rotations, radii, masses, mode_settings)
}

// =============================================================================
// Helper functions
// =============================================================================

/// Convert quaternion to (axis, angle) pair
#[inline]
fn quat_to_axis_angle_pair(q: Quat) -> (Vec3, f32) {
    let w_clamped = q.w.clamp(-1.0, 1.0);
    let angle = 2.0 * w_clamped.acos();

    let axis = if angle < 0.001 {
        Vec3::X
    } else {
        let sin_half = (angle * 0.5).sin();
        Vec3::new(q.x, q.y, q.z) / sin_half
    };

    (axis, angle)
}

/// Create quaternion from two vectors (deterministic)
fn quat_from_two_vectors(from: Vec3, to: Vec3) -> Quat {
    let v1_len = from.length();
    let v2_len = to.length();
    if v1_len < 0.0001 || v2_len < 0.0001 {
        return Quat::IDENTITY;
    }
    let v1 = from / v1_len;
    let v2 = to / v2_len;

    let cos_angle = v1.dot(v2);

    if cos_angle > 0.9999 {
        return Quat::IDENTITY;
    }

    if cos_angle < -0.9999 {
        let axis = if v1.x.abs() < v1.y.abs() && v1.x.abs() < v1.z.abs() {
            Vec3::new(0.0, -v1.z, v1.y).normalize()
        } else if v1.y.abs() < v1.z.abs() {
            Vec3::new(-v1.z, 0.0, v1.x).normalize()
        } else {
            Vec3::new(-v1.y, v1.x, 0.0).normalize()
        };
        return Quat::from_xyzw(axis.x, axis.y, axis.z, 0.0);
    }

    let halfway = (v1 + v2).normalize();
    let axis = Vec3::new(
        v1.y * halfway.z - v1.z * halfway.y,
        v1.z * halfway.x - v1.x * halfway.z,
        v1.x * halfway.y - v1.y * halfway.x,
    );
    let w = v1.dot(halfway);

    Quat::from_xyzw(axis.x, axis.y, axis.z, w).normalize()
}

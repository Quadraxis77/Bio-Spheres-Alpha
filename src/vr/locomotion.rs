//! Gravity-relative locomotion. All positions are in simulation world units.
use glam::{Mat3, Quat, Vec2, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Gravity {
    mode: u32,
    sign: f32,
}
impl Default for Gravity {
    fn default() -> Self {
        Self { mode: 1, sign: 1.0 }
    }
}
impl Gravity {
    pub fn new(strength: f32, mode: u32) -> Self {
        Self {
            mode,
            sign: if strength < 0.0 { -1.0 } else { 1.0 },
        }
    }
    /// Up is opposite the simulation's force: positive radial gravity pulls inward.
    /// Zero gravity retains the selected axis; the center has no radial normal.
    pub fn up(self, head: Vec3, fallback: Vec3) -> Vec3 {
        match self.mode {
            0 => Vec3::X * self.sign,
            2 => Vec3::Z * self.sign,
            3 if head.length_squared() > 1.0e-10 => head.normalize() * self.sign,
            3 => fallback.normalize_or_zero(),
            _ => Vec3::Y * self.sign,
        }
    }
    /// Level the tracking rig, preserving its heading and the actual head position.
    /// Headset pitch/roll remain natural tracking offsets, not navigation axes.
    pub fn align(self, rig: Vec3, rotation: Quat, head: Vec3, scale: f32) -> (Vec3, Quat) {
        let world_head = rig + rotation * head * scale;
        let up = self.up(world_head, rotation * Vec3::Y);
        if (rotation * Vec3::Y).dot(up) > 0.999999 {
            return (rig, rotation);
        }
        let forward = tangent_forward(rotation, up, rotation);
        let right = forward.cross(up).normalize();
        let leveled = Quat::from_mat3(&Mat3::from_cols(right, up, -forward)).normalize();
        (world_head - leveled * head * scale, leveled)
    }
    /// A tangent arc transports both position and orientation, preserving altitude.
    /// Grip + stick Y changes altitude; X still strafes along the ground plane.
    pub fn travel(
        self,
        rig: Vec3,
        rotation: Quat,
        head: Vec3,
        head_rotation: Quat,
        motion: Vec2,
        step: f32,
        scale: f32,
        climbing: bool,
    ) -> (Vec3, Quat) {
        let mut world_head = rig + rotation * head * scale;
        let up = self.up(world_head, rotation * Vec3::Y);
        let forward = tangent_forward(rotation * head_rotation, up, rotation);
        let right = forward.cross(up).normalize();
        let tangent = (right * motion.x
            + if climbing {
                Vec3::ZERO
            } else {
                forward * motion.y
            })
            * step;
        let lift = if climbing { motion.y * step } else { 0.0 };
        if self.mode != 3 {
            world_head += tangent + up * lift;
            return (world_head - rotation * head * scale, rotation);
        }
        // Use double precision for the arc. Repeated f32 rotations introduce
        // a systematic radius bias that otherwise grows with the refresh rate.
        let radial = world_head.as_dvec3();
        let radius = radial.length();
        let minimum_radius = f64::from((scale * 0.005).max(0.001));
        if radius < minimum_radius {
            // There is no spherical ground at the exact center. Use the last
            // stable plane until the player has moved far enough to define one.
            world_head += tangent + up * lift;
            return self.align(world_head - rotation * head * scale, rotation, head, scale);
        }
        let normal = radial / radius;
        let radius = (radius + f64::from(self.sign * lift)).max(minimum_radius);
        let tangent = tangent.as_dvec3();
        let distance = tangent.length();
        let transport = if distance > 1.0e-6 {
            glam::DQuat::from_axis_angle(
                normal.cross(tangent / distance).normalize(),
                distance / radius,
            )
        } else {
            glam::DQuat::IDENTITY
        };
        world_head = ((transport * normal).normalize() * radius).as_vec3();
        let rotation = (transport * rotation.as_dquat()).normalize().as_quat();
        (world_head - rotation * head * scale, rotation)
    }
}
fn tangent_forward(view: Quat, up: Vec3, fallback: Quat) -> Vec3 {
    let projected = |direction: Vec3| direction - up * direction.dot(up);
    let mut forward = projected(view * Vec3::NEG_Z);
    // Looking straight up/down must not destroy or invert the walking heading.
    if forward.length_squared() < 0.01 {
        forward = projected(fallback * Vec3::NEG_Z);
    }
    if forward.length_squared() < 0.01 {
        forward = up.cross(fallback * Vec3::X);
    }
    if forward.length_squared() < 0.01 {
        forward = up.cross(if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X });
    }
    forward.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn near(a: Vec3, b: Vec3) {
        assert!((a - b).length() < 0.002, "{a:?} != {b:?}");
    }
    #[test]
    fn all_axial_planes_and_gravity_signs_preserve_height_despite_head_pitch() {
        for mode in 0..3 {
            for sign in [1.0, -1.0] {
                let g = Gravity::new(sign, mode);
                let head = Vec3::new(0.2, 1.1, -0.15);
                let (rig, q) = g.align(
                    Vec3::new(40.0, 30.0, 20.0),
                    Quat::from_rotation_z(0.7),
                    head,
                    20.0,
                );
                let before = rig + q * head * 20.0;
                let up = g.up(before, Vec3::Y);
                near(q * Vec3::Y, up);
                for pitch in [-1.56, -0.8, 0.8, 1.56] {
                    let (p, nq) = g.travel(
                        rig,
                        q,
                        head,
                        Quat::from_rotation_x(pitch),
                        Vec2::new(0.6, 0.8),
                        10.0,
                        20.0,
                        false,
                    );
                    let delta = p + nq * head * 20.0 - before;
                    assert!(delta.dot(up).abs() < 0.002);
                    assert!((delta.length() - 10.0).abs() < 0.002);
                    near(nq * Vec3::Y, up);
                }
                let (p, nq) = g.travel(
                    rig,
                    q,
                    head,
                    Quat::from_rotation_x(0.8),
                    Vec2::Y,
                    10.0,
                    20.0,
                    true,
                );
                near(p + nq * head * 20.0 - before, up * 10.0);
            }
        }
    }
    #[test]
    fn radial_walk_follows_a_great_circle_and_transports_the_seated_frame() {
        for sign in [1.0, -1.0] {
            for motion in [Vec2::Y, -Vec2::Y, Vec2::X, -Vec2::X, Vec2::new(0.6, 0.8)] {
                let g = Gravity::new(sign, 3);
                let head = Vec3::new(0.2, 1.0, -0.1);
                let radius = 500.0;
                let initial = Vec3::Y * radius;
                let (rig, q) = g.align(initial - head * 20.0, Quat::IDENTITY, head, 20.0);
                let (p, nq) = g.travel(
                    rig,
                    q,
                    head,
                    Quat::IDENTITY,
                    motion,
                    radius * std::f32::consts::FRAC_PI_2,
                    20.0,
                    false,
                );
                let result = p + nq * head * 20.0;
                assert!((result.length() - radius).abs() < 0.002);
                assert!(
                    result.dot(initial).abs() < 0.2,
                    "Quarter circumference reaches the equator"
                );
                near(nq * Vec3::Y, result.normalize() * sign);
                let (back, bq) = g.travel(
                    p,
                    nq,
                    head,
                    Quat::IDENTITY,
                    -motion,
                    radius * std::f32::consts::FRAC_PI_2,
                    20.0,
                    false,
                );
                near(back + bq * head * 20.0, initial);
                assert!(bq.angle_between(q) < 0.001);
            }
        }
    }
    #[test]
    fn radial_walk_crosses_both_poles_without_drift_or_coordinate_singularities() {
        for sign in [1.0, -1.0] {
            let g = Gravity::new(sign, 3);
            let head = Vec3::new(0.2, 1.0, -0.1);
            let start = Vec3::Y * 100.0;
            let (mut p, mut q) = g.align(start - head * 20.0, Quat::IDENTITY, head, 20.0);
            let initial_rotation = q;
            for _ in 0..720 {
                (p, q) = g.travel(
                    p,
                    q,
                    head,
                    Quat::IDENTITY,
                    Vec2::Y,
                    100.0 * std::f32::consts::TAU / 720.0,
                    20.0,
                    false,
                );
                let world = p + q * head * 20.0;
                assert!((world.length() - 100.0).abs() < 0.02);
                near(q * Vec3::Y, world.normalize() * sign);
                assert!(p.is_finite() && q.is_finite());
            }
            near(p + q * head * 20.0, start);
            assert!(q.angle_between(initial_rotation) < 0.002);
        }
    }
    #[test]
    fn radial_lift_respects_inward_and_outward_gravity_and_cannot_cross_the_center() {
        for sign in [1.0, -1.0] {
            let g = Gravity::new(sign, 3);
            let (p, q) = g.align(Vec3::Y * 100.0, Quat::IDENTITY, Vec3::ZERO, 20.0);
            let (raised, _) = g.travel(p, q, Vec3::ZERO, Quat::IDENTITY, Vec2::Y, 10.0, 20.0, true);
            assert!((raised.length() - (100.0 + sign * 10.0)).abs() < 0.001);
            let (lowered, nq) = g.travel(
                p,
                q,
                Vec3::ZERO,
                Quat::IDENTITY,
                -Vec2::Y * sign,
                200.0,
                20.0,
                true,
            );
            assert!(lowered.length() >= 0.099 && lowered.is_finite() && nq.is_finite());
            near(lowered.normalize(), Vec3::Y);
        }
        let g = Gravity::new(30.0, 3);
        let (p, q) = g.travel(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec2::Y,
            1.0,
            20.0,
            true,
        );
        near(p, Vec3::Y);
        assert!(q.is_finite());
    }
    #[test]
    fn radial_distance_is_world_distance_and_independent_of_refresh_rate() {
        let run = |hz: u32| {
            let g = Gravity::new(30.0, 3);
            let mut p = Vec3::Y * 500.0;
            let mut q = Quat::IDENTITY;
            for _ in 0..hz {
                (p, q) = g.travel(
                    p,
                    q,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec2::Y,
                    30.0 / hz as f32,
                    20.0,
                    false,
                );
            }
            (p, q)
        };
        let (a, qa) = run(60);
        let (b, qb) = run(120);
        near(a, b);
        assert!(qa.angle_between(qb) < 0.001);
        assert!((a.normalize().angle_between(Vec3::Y) - 30.0 / 500.0).abs() < 0.001);
    }
}

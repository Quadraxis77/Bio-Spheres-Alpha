//! Visual inspection spotlight for the desktop cursor and tracked VR controller.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Three vec4s shared with surface and volumetric shaders. A zero range disables it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CursorSpotlight {
    pub position_range: [f32; 4],
    pub direction_outer: [f32; 4],
    pub color_inner: [f32; 4],
}

impl CursorSpotlight {
    pub fn from_cursor(
        cursor: Option<(f32, f32)>,
        width: u32,
        height: u32,
        view_proj: Mat4,
        camera_pos: Vec3,
        world_radius: f32,
    ) -> Self {
        let Some((x, y)) = cursor else {
            return Self::default();
        };
        if width == 0
            || height == 0
            || x < 0.0
            || y < 0.0
            || x >= width as f32
            || y >= height as f32
        {
            return Self::default();
        }
        let ndc = Vec3::new(
            2.0 * x / width as f32 - 1.0,
            1.0 - 2.0 * y / height as f32,
            0.5,
        );
        let direction = (view_proj.inverse().project_point3(ndc) - camera_pos).normalize_or_zero();
        if !direction.is_finite() || direction.length_squared() < 0.5 {
            return Self::default();
        }
        // Emit from the optical center of the camera, along the world-space ray
        // through the cursor. The cursor only steers the beam, never its origin.
        let mut light = Self::from_ray(camera_pos, direction, world_radius);
        light.direction_outer[3] = 24.0_f32.to_radians().cos();
        light.color_inner = [0.6, 0.57, 0.51, 7.0_f32.to_radians().cos()];
        light
    }

    pub fn from_ray(origin: Vec3, direction: Vec3, world_radius: f32) -> Self {
        let direction = direction.normalize_or_zero();
        if !origin.is_finite() || !direction.is_finite() || direction.length_squared() < 0.5 {
            return Self::default();
        }
        Self {
            position_range: origin.extend((world_radius * 4.0).max(100.0)).to_array(),
            direction_outer: direction.extend(13.0_f32.to_radians().cos()).to_array(),
            color_inner: [2.4, 2.28, 2.04, 7.0_f32.to_radians().cos()],
        }
    }

    pub fn enabled(self) -> bool {
        self.position_range[3] > 0.0
    }
}

/// Detect deliberate back-and-forth strokes in tracking-space meters.
/// Two reversals and a cooldown avoid toggling during ordinary pointing or jitter.
#[derive(Default)]
pub struct ShakeToggle {
    previous: Option<Vec3>,
    anchor: Option<Vec3>,
    stroke: Option<Vec3>,
    reversals: u8,
    window: f32,
    cooldown: f32,
}

impl ShakeToggle {
    pub fn update(&mut self, position: Option<Vec3>, dt: f32) -> bool {
        self.cooldown = (self.cooldown - dt.max(0.0)).max(0.0);
        let Some(position) = position.filter(|p| p.is_finite()) else {
            self.reset_motion();
            return false;
        };
        if dt <= 0.0 || dt > 0.15 || self.cooldown > 0.0 {
            self.reset_motion();
            self.previous = Some(position);
            self.anchor = Some(position);
            return false;
        }
        let previous = self.previous.replace(position);
        let anchor = *self.anchor.get_or_insert(position);
        self.window += dt;
        if self.window > 0.7 {
            self.stroke = None;
            self.reversals = 0;
            self.window = 0.0;
        }
        let travel = position - anchor;
        let speed = previous.map_or(0.0, |p| (position - p).length() / dt);
        if travel.length() < 0.055 || speed < 0.65 {
            return false;
        }
        self.anchor = Some(position);
        let direction = travel.normalize();
        if let Some(stroke) = self.stroke {
            if stroke.dot(direction) < -0.5 {
                self.reversals += 1;
            }
        } else {
            self.window = 0.0;
        }
        self.stroke = Some(direction);
        if self.reversals >= 2 {
            self.cooldown = 1.0;
            self.reset_motion();
            return true;
        }
        false
    }

    fn reset_motion(&mut self) {
        self.previous = None;
        self.anchor = None;
        self.stroke = None;
        self.reversals = 0;
        self.window = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shake_requires_reversals_and_debounces() {
        for hz in [72.0, 90.0, 120.0] {
            let mut detector = ShakeToggle::default();
            let dt = 1.0 / hz;
            let mut toggles = 0;
            for i in 0..(hz as usize) {
                let t = i as f32 * dt;
                let hand = Vec3::X * (t * 4.0 * std::f32::consts::TAU).sin() * 0.12;
                toggles += detector.update(Some(hand), dt) as usize;
            }
            assert_eq!(toggles, 1, "one toggle per shake burst at {hz} Hz");
            // Tracking loss discards any partially recognized gesture.
            detector.update(None, dt);
            for i in 0..(hz as usize * 2) {
                assert!(!detector.update(Some(Vec3::X * i as f32 * dt * 0.1), dt));
            }
            for i in 0..100 {
                assert!(
                    !detector.update(Some(Vec3::X * if i % 2 == 0 { 0.002 } else { -0.002 }), dt)
                );
            }
        }
    }

    #[test]
    fn flashlight_shaders_validate_and_share_uniform_layout() {
        for source in [
            include_str!("../../shaders/cells/cell_unified.wgsl"),
            include_str!("../../shaders/cells/tail_gpu.wgsl"),
            include_str!("../../shaders/cave_system.wgsl"),
            include_str!("../../shaders/fluid/density_mesh.wgsl"),
            include_str!("../../shaders/fluid/ice_mesh.wgsl"),
            include_str!("../../shaders/volumetric_fog.wgsl"),
            include_str!("../../shaders/organism_skin_render.wgsl"),
            include_str!("../../shaders/boulder.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            if let Some((_, ty)) = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some("ShadowFieldParams"))
            {
                let naga::TypeInner::Struct { members, span } = &ty.inner else {
                    panic!()
                };
                assert_eq!(
                    *span as usize,
                    std::mem::size_of::<
                        crate::simulation::gpu_physics::light_field::ShadowFieldParams,
                    >()
                );
                assert_eq!(
                    members
                        .iter()
                        .find(|m| m.name.as_deref() == Some("spotlight_position_range"))
                        .unwrap()
                        .offset as usize,
                    std::mem::offset_of!(
                        crate::simulation::gpu_physics::light_field::ShadowFieldParams,
                        cursor_spotlight
                    )
                );
            };
        }
    }

    #[test]
    fn spotlight_tracks_cursor_through_rotated_camera() {
        let position = Vec3::new(12.0, -8.0, 40.0);
        let rotation = glam::Quat::from_rotation_y(0.6);
        let projection = Mat4::perspective_rh(60.0_f32.to_radians(), 2.0, 0.1, 5000.0);
        let view_proj = projection * Mat4::from_rotation_translation(rotation, position).inverse();
        for cursor in [(400.0, 200.0), (120.0, 80.0), (720.0, 350.0)] {
            let light =
                CursorSpotlight::from_cursor(Some(cursor), 800, 400, view_proj, position, 200.0);
            let direction = Vec3::from_array(light.direction_outer[..3].try_into().unwrap());
            assert_eq!(&light.position_range[..3], &position.to_array());
            let projected = view_proj.project_point3(position + direction * 100.0);
            let screen = ((projected.x + 1.0) * 400.0, (1.0 - projected.y) * 200.0);
            assert!((screen.0 - cursor.0).abs() < 0.05);
            assert!((screen.1 - cursor.1).abs() < 0.05);
            assert!(light.enabled());
        }
        for cursor in [None, Some((-1.0, 20.0)), Some((800.0, 20.0))] {
            assert!(
                !CursorSpotlight::from_cursor(cursor, 800, 400, view_proj, position, 200.0)
                    .enabled()
            );
        }
    }
}

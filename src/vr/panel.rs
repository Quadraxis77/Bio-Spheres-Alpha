//! A menu anchored in tracking space, shared by presentation and controller picking.

use glam::{Quat, Vec2, Vec3};
use openxr as xr;

pub(super) const WIDTH_METERS: f32 = 1.8;

#[derive(Clone, Copy, Debug)]
pub struct PanelSettings {
    /// Horizontal arc in degrees; zero is a flat screen.
    pub curvature: f32,
    pub distance: f32,
    pub aspect: f32,
}
impl Default for PanelSettings {
    fn default() -> Self {
        Self {
            curvature: 0.0,
            distance: 1.6,
            aspect: 16.0 / 9.0,
        }
    }
}
impl PanelSettings {
    pub fn sanitized(self) -> Self {
        let finite = |value: f32, fallback: f32, min: f32, max: f32| {
            if value.is_finite() {
                value.clamp(min, max)
            } else {
                fallback
            }
        };
        Self {
            curvature: finite(self.curvature, 0.0, 0.0, 110.0),
            distance: finite(self.distance, 1.6, 0.6, 4.0),
            aspect: finite(self.aspect, 16.0 / 9.0, 1.0, 3.0),
        }
    }
    pub(super) fn point(self, uv: Vec2) -> Vec3 {
        let settings = self.sanitized();
        let x = (uv.x - 0.5) * WIDTH_METERS;
        let y = (0.5 - uv.y) * WIDTH_METERS / settings.aspect;
        let angle = settings.curvature.to_radians();
        if angle < 0.001 {
            return Vec3::new(x, y, 0.0);
        }
        let radius = f64::from(WIDTH_METERS) / f64::from(angle);
        let theta = f64::from(uv.x - 0.5) * f64::from(angle);
        Vec3::new(
            (radius * theta.sin()) as f32,
            y,
            (radius * (1.0 - theta.cos())) as f32,
        )
    }
}

#[derive(Default)]
pub(super) struct PanelAnchor {
    origin: Option<xr::Posef>,
    pub settings: PanelSettings,
}
impl PanelAnchor {
    pub fn pose(&self) -> Option<xr::Posef> {
        let mut pose = self.origin?;
        let center =
            position(pose) + rotation(pose) * Vec3::NEG_Z * self.settings.sanitized().distance;
        pose.position = xr::Vector3f {
            x: center.x,
            y: center.y,
            z: center.z,
        };
        Some(pose)
    }
    pub fn reset(&mut self) {
        self.origin = None;
    }
    pub fn capture(&mut self, head: xr::Posef) {
        if self.origin.is_some() {
            return;
        }
        // Capture the seated origin once. Adjusting screen distance must not
        // recenter it on a moving HMD or change the player's navigation rig.
        let forward = rotation(head) * Vec3::NEG_Z;
        let forward = Vec3::new(forward.x, 0.0, forward.z)
            .try_normalize()
            .unwrap_or(Vec3::NEG_Z);
        let orientation = Quat::from_rotation_y((-forward.x).atan2(-forward.z));
        self.origin = Some(xr::Posef {
            orientation: xr::Quaternionf {
                x: orientation.x,
                y: orientation.y,
                z: orientation.z,
                w: orientation.w,
            },
            position: head.position,
        });
    }
}

fn position(pose: xr::Posef) -> Vec3 {
    Vec3::new(pose.position.x, pose.position.y, pose.position.z)
}

fn rotation(pose: xr::Posef) -> Quat {
    Quat::from_xyzw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    .normalize()
}

/// Off-axis window projection. The aperture stays fixed; eye translation gives
/// stereo disparity and motion parallax without making the panel follow the head.
pub(super) fn render_view_with_settings(
    panel: xr::Posef,
    eye: xr::Posef,
    camera: &crate::ui::camera::CameraController,
    rect: egui::Rect,
    panel_width: u32,
    panel_height: u32,
    width: u32,
    height: u32,
    settings: PanelSettings,
) -> Option<crate::rendering::RenderView> {
    if panel_width == 0 || panel_height == 0 || rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }
    let settings = settings.sanitized();
    let meters_per_pixel = WIDTH_METERS / panel_width as f32;
    let meters_per_pixel_y = WIDTH_METERS / settings.aspect / panel_height as f32;
    let center = Vec3::new(
        (rect.center().x - panel_width as f32 * 0.5) * meters_per_pixel,
        (panel_height as f32 * 0.5 - rect.center().y) * meters_per_pixel_y,
        0.0,
    );
    let eye = rotation(panel).conjugate() * (position(eye) - position(panel)) - center;
    if eye.z <= 0.05 || !eye.is_finite() {
        return None;
    }
    let half_width = rect.width() * meters_per_pixel * 0.5;
    let half_height = rect.height() * meters_per_pixel_y * 0.5;
    // Preserve the preview's framing and put its orbit target at the panel plane.
    let distance = camera.distance.max(0.1);
    let scale = distance * (camera.horizontal_fov_degrees.to_radians() * 0.5).tan() / half_width;
    let orientation = camera.view_rotation();
    let focus = camera.position() + orientation * Vec3::NEG_Z * distance;
    Some(crate::rendering::RenderView {
        position: focus + orientation * eye * scale,
        rotation: orientation,
        projection: crate::rendering::CameraProjection::from_fov(
            ((-half_width - eye.x) / eye.z).atan(),
            ((half_width - eye.x) / eye.z).atan(),
            ((-half_height - eye.y) / eye.z).atan(),
            ((half_height - eye.y) / eye.z).atan(),
            0.1,
            5000.0,
        ),
        width,
        height,
    })
}

pub(super) fn ray_hit_with_settings(
    pose: xr::Posef,
    settings: PanelSettings,
    origin: Vec3,
    direction: Vec3,
    width: u32,
    height: u32,
) -> Option<(Vec2, f32)> {
    if width == 0 || height == 0 || !origin.is_finite() || !direction.is_finite() {
        return None;
    }
    let inverse = rotation(pose).conjugate();
    let origin = inverse * (origin - position(pose));
    let direction = inverse * direction;
    let settings = settings.sanitized();
    let angle = settings.curvature.to_radians();
    let uv_for_hit = |t: f32| {
        if t < 0.0 || !t.is_finite() {
            return None;
        }
        let point = origin + direction * t;
        let u = if angle < 0.001 {
            if direction.z >= -1e-5 {
                return None;
            }
            point.x / WIDTH_METERS + 0.5
        } else {
            let radius = WIDTH_METERS / angle;
            let theta = point.x.atan2(radius - point.z);
            let normal = Vec3::new(-theta.sin(), 0.0, theta.cos());
            if normal.dot(direction) >= -1e-5 {
                return None;
            }
            theta / angle + 0.5
        };
        let uv = Vec2::new(u, 0.5 - point.y * settings.aspect / WIDTH_METERS);
        if !uv.is_finite() || uv.min_element() < -1e-5 || uv.max_element() > 1.00001 {
            return None;
        }
        Some((
            uv.clamp(Vec2::ZERO, Vec2::ONE) * Vec2::new(width as f32, height as f32),
            t,
        ))
    };
    if angle < 0.001 {
        if direction.z.abs() < 1e-5 {
            return None;
        }
        return uv_for_hit(-origin.z / direction.z);
    }
    // Near-flat curves have a very large radius. Use f64 for the cylinder
    // intersection to avoid subtracting almost-equal f32 squared distances.
    let radius = f64::from(WIDTH_METERS) / f64::from(angle);
    let center = glam::DVec2::new(f64::from(origin.x), f64::from(origin.z) - radius);
    let ray = glam::DVec2::new(f64::from(direction.x), f64::from(direction.z));
    let a = ray.length_squared();
    if a < 1e-8 {
        return None;
    }
    let b = center.dot(ray);
    let discriminant = b * b - a * (center.length_squared() - radius * radius);
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    uv_for_hit(((-b - root) / a) as f32).or_else(|| uv_for_hit(((-b + root) / a) as f32))
}
pub(super) fn ray_hit(
    pose: xr::Posef,
    origin: Vec3,
    direction: Vec3,
    width: u32,
    height: u32,
) -> Option<Vec2> {
    ray_hit_with_settings(
        pose,
        PanelSettings::default(),
        origin,
        direction,
        width,
        height,
    )
    .map(|hit| hit.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curved_screen_picking_tracks_distance_aspect_and_transformed_anchor() {
        let q = Quat::from_rotation_y(0.7);
        let mut head = xr::Posef::IDENTITY;
        head.position = xr::Vector3f {
            x: 2.0,
            y: 1.5,
            z: 3.0,
        };
        head.orientation = xr::Quaternionf {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        };
        let mut anchor = PanelAnchor::default();
        anchor.capture(head);
        for curvature in [0.0, 0.01, 0.1, 1.0, 45.0, 110.0] {
            for distance in [0.6, 1.6, 4.0] {
                for aspect in [1.0, 16.0 / 9.0, 3.0] {
                    anchor.settings = PanelSettings {
                        curvature,
                        distance,
                        aspect,
                    };
                    let pose = anchor.pose().unwrap();
                    for uv in [Vec2::splat(0.5), Vec2::new(0.02, 0.1), Vec2::new(0.98, 0.9)] {
                        let point = position(pose) + rotation(pose) * anchor.settings.point(uv);
                        let origin = position(head) + q * Vec3::new(0.02, 0.01, 0.0);
                        let direction = (point - origin).normalize();
                        let (pixel, ray_distance) = ray_hit_with_settings(
                            pose,
                            anchor.settings,
                            origin,
                            direction,
                            1920,
                            1080,
                        )
                        .unwrap();
                        assert!(
                            (pixel - uv * Vec2::new(1920.0, 1080.0)).length() < 0.3,
                            "{curvature}°, {distance}m, {aspect}:1 -> {pixel:?}"
                        );
                        assert!((origin + direction * ray_distance - point).length() < 0.001);
                    }
                }
            }
        }
        let captured = anchor.pose().unwrap();
        head.position.x += 1.0;
        anchor.capture(head);
        assert_eq!(
            position(anchor.pose().unwrap()),
            position(captured),
            "screen adjustments cannot turn it into a head-locked panel"
        );
        anchor.settings.distance = 2.0;
        assert!((position(anchor.pose().unwrap()) - position(captured)).length() > 1.9);
        let settings = PanelSettings {
            curvature: f32::NAN,
            distance: f32::INFINITY,
            aspect: 0.0,
        }
        .sanitized();
        assert!(settings.point(Vec2::splat(0.5)).is_finite());
    }
    #[test]
    fn stereo_window_has_depth_and_keeps_its_aperture_fixed_when_the_head_moves() {
        let mut anchor = PanelAnchor::default();
        anchor.capture(xr::Posef::IDENTITY);
        let camera = crate::ui::camera::CameraController::new_for_preview_scene();
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1920.0, 1080.0));
        let mut projections = Vec::new();
        for x in [-0.032, 0.032, 0.35] {
            let mut eye = xr::Posef::IDENTITY;
            eye.position.x = x;
            let view = render_view_with_settings(
                anchor.pose().unwrap(),
                eye,
                &camera,
                rect,
                1920,
                1080,
                1920,
                1080,
                PanelSettings::default(),
            )
            .unwrap();
            let projection = view.projection.matrix(16.0 / 9.0, 0.1, 5000.0);
            let view_proj = projection * view.view_matrix();
            let distance = camera.distance.max(0.1);
            let focus = camera.position() + camera.view_rotation() * Vec3::NEG_Z * distance;
            // Objects at the panel plane align; objects behind it have eye disparity.
            assert!(view_proj.project_point3(focus).truncate().length() < 1e-4);
            let behind = focus + camera.view_rotation() * Vec3::NEG_Z * 10.0;
            projections.push(view_proj.project_point3(behind).x);
            // The left and right physical aperture edges project exactly to NDC edges.
            let scale = distance * (camera.horizontal_fov_degrees.to_radians() * 0.5).tan()
                / (WIDTH_METERS * 0.5);
            for edge in [-1.0, 1.0] {
                let world_edge =
                    focus + camera.view_rotation() * Vec3::X * edge * WIDTH_METERS * 0.5 * scale;
                assert!((view_proj.project_point3(world_edge).x - edge).abs() < 1e-4);
            }
        }
        assert!(projections[1] > projections[0]);
        assert!(projections[2] > projections[1]);
        let mut behind_panel = xr::Posef::IDENTITY;
        behind_panel.position.z = -2.0;
        assert!(render_view_with_settings(
            anchor.pose().unwrap(),
            behind_panel,
            &camera,
            rect,
            1920,
            1080,
            1920,
            1080,
            PanelSettings::default(),
        )
        .is_none());
    }

    #[test]
    fn moving_and_turning_the_head_does_not_move_the_panel() {
        let mut anchor = PanelAnchor::default();
        anchor.capture(xr::Posef::IDENTITY);
        let original = anchor.pose().unwrap();
        let mut moved = xr::Posef::IDENTITY;
        moved.position.x = 0.5;
        moved.position.y = 0.2;
        moved.orientation.y = std::f32::consts::FRAC_1_SQRT_2;
        moved.orientation.w = std::f32::consts::FRAC_1_SQRT_2;
        anchor.capture(moved);
        assert_eq!(position(anchor.pose().unwrap()), position(original));
        assert_eq!(rotation(anchor.pose().unwrap()), rotation(original));
        let origin = position(moved);
        let direction = (position(original) - origin).normalize();
        let pointer = ray_hit(anchor.pose().unwrap(), origin, direction, 1920, 1080).unwrap();
        assert!((pointer - Vec2::new(960.0, 540.0)).length() < 0.01);
        anchor.reset();
        anchor.capture(moved);
        assert!((position(anchor.pose().unwrap()) - position(original)).length() > 1.0);
    }

    #[test]
    fn rotated_and_translated_panel_uses_the_same_controller_coordinates() {
        let mut head = xr::Posef::IDENTITY;
        head.position = xr::Vector3f {
            x: 2.0,
            y: 1.5,
            z: 3.0,
        };
        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        head.orientation = xr::Quaternionf {
            x: rotation.x,
            y: rotation.y,
            z: rotation.z,
            w: rotation.w,
        };
        let mut anchor = PanelAnchor::default();
        anchor.capture(head);
        let origin = position(head) + rotation * Vec3::new(0.45, 0.253125, 0.0);
        let pointer = ray_hit(
            anchor.pose().unwrap(),
            origin,
            rotation * Vec3::NEG_Z,
            1920,
            1080,
        )
        .unwrap();
        assert!((pointer - Vec2::new(1440.0, 270.0)).length() < 0.01);
        assert!(ray_hit(
            anchor.pose().unwrap(),
            origin,
            rotation * Vec3::Z,
            1920,
            1080
        )
        .is_none());
    }

    #[test]
    fn panel_stays_upright_when_the_initial_head_pose_is_tilted() {
        let tilted =
            Quat::from_rotation_y(0.8) * Quat::from_rotation_x(-0.4) * Quat::from_rotation_z(0.3);
        let mut head = xr::Posef::IDENTITY;
        head.position.y = 1.7;
        head.orientation = xr::Quaternionf {
            x: tilted.x,
            y: tilted.y,
            z: tilted.z,
            w: tilted.w,
        };
        let mut anchor = PanelAnchor::default();
        anchor.capture(head);
        let pose = anchor.pose().unwrap();
        assert!((rotation(pose) * Vec3::Y - Vec3::Y).length() < 1e-5);
        assert_eq!(pose.position.y, head.position.y);
    }
}

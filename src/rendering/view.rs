//! View data supplied by either a desktop camera or a tracked headset eye.

use glam::{Mat4, Quat, Vec3};

#[derive(Clone, Copy, Debug)]
pub enum CameraProjection {
    HorizontalFov(f32),
    Explicit(Mat4),
}

impl From<f32> for CameraProjection {
    fn from(fov: f32) -> Self {
        Self::HorizontalFov(fov)
    }
}

impl CameraProjection {
    pub fn matrix(self, aspect: f32, near: f32, far: f32) -> Mat4 {
        match self {
            Self::HorizontalFov(fov) => Mat4::perspective_rh(
                crate::ui::camera::CameraController::vertical_fov_radians_for_horizontal(
                    fov, aspect,
                ),
                aspect,
                near,
                far,
            ),
            Self::Explicit(matrix) => matrix,
        }
    }

    /// OpenXR supplies separate signed angles for all four sides of the view.
    /// Right-handed, forward -Z, WebGPU depth range 0..1.
    pub fn from_fov(left: f32, right: f32, down: f32, up: f32, near: f32, far: f32) -> Self {
        let (left, right, down, up) = (left.tan(), right.tan(), down.tan(), up.tan());
        Self::Explicit(Mat4::from_cols_array(&[
            2.0 / (right - left),
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / (up - down),
            0.0,
            0.0,
            (right + left) / (right - left),
            (up + down) / (up - down),
            far / (near - far),
            -1.0,
            0.0,
            0.0,
            near * far / (near - far),
            0.0,
        ]))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderView {
    pub position: Vec3,
    pub rotation: Quat,
    pub projection: CameraProjection,
    pub width: u32,
    pub height: u32,
}

impl RenderView {
    pub fn view_matrix(self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position).inverse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_projection_maps_eye_frustum_edges_and_depth() {
        let (left, right, down, up) = (-0.7_f32, 0.9_f32, -0.6_f32, 0.8_f32);
        let projection =
            CameraProjection::from_fov(left, right, down, up, 0.1, 100.0).matrix(1.0, 0.1, 100.0);
        for (x, y, expected_x, expected_y) in [
            (left.tan(), down.tan(), -1.0, -1.0),
            (right.tan(), up.tan(), 1.0, 1.0),
        ] {
            let ndc = projection.project_point3(Vec3::new(x, y, -1.0));
            assert!((ndc.x - expected_x).abs() < 1e-5);
            assert!((ndc.y - expected_y).abs() < 1e-5);
        }
        assert!(projection.project_point3(Vec3::new(0.0, 0.0, -0.1)).z.abs() < 1e-5);
        assert!((projection.project_point3(Vec3::new(0.0, 0.0, -100.0)).z - 1.0).abs() < 1e-5);
    }
}

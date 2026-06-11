use glam::{Quat, Vec3};
use std::time::Instant;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Horizontal field of view used by the main render camera.
///
/// The default preserves the old 45 degree vertical FOV at a 16:9 viewport.
pub const DEFAULT_HORIZONTAL_FOV_DEGREES: f32 = 72.73435;
pub const MIN_HORIZONTAL_FOV_DEGREES: f32 = 30.0;
pub const MAX_HORIZONTAL_FOV_DEGREES: f32 = 140.0;
/// Fraction of the vertical FOV the world sphere should occupy when orbit
/// distance is auto-fit, leaving a small margin at the top and bottom.
const ORBIT_FIT_FOV_FRACTION: f32 = 0.9;
/// Maximum gap (in seconds) between middle-mouse clicks to count as a double-click.
const DOUBLE_CLICK_SECONDS: f32 = 0.35;

/// Camera mode - matches BioSpheres-Q
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Orbit,
    FreeFly,
}

/// Scene type for camera behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneType {
    GpuScene,
    PreviewScene,
}

/// Camera controller - Space Engineers style 6DOF camera (matches BioSpheres-Q)
pub struct CameraController {
    // Core camera state
    pub center: Vec3,
    pub distance: f32,
    pub target_distance: f32,
    pub rotation: Quat,
    pub target_rotation: Quat,
    pub mode: CameraMode,
    pub scene_type: SceneType, // Add scene type for different behaviors

    // Stable yaw/up axis for orbit mode. This follows the selected gravity axis,
    // not the gravity sign, so orbit controls never flip upside down.
    pub up_direction: Vec3,
    // Current gravity mode (0=X, 1=Y, 2=Z, 3=radial)
    pub gravity_mode: u32,

    // Mouse state
    is_dragging: bool,
    last_mouse_pos: Option<PhysicalPosition<f64>>,
    accumulated_mouse_delta: Vec3,
    accumulated_scroll: f32,

    // Middle-mouse "free look" state (Orbit mode only). Lets the camera angle
    // away from the orbit center without changing orbit position/distance/logic.
    pub look_offset: Quat,
    is_look_dragging: bool,
    last_look_mouse_pos: Option<PhysicalPosition<f64>>,
    accumulated_look_delta: Vec3,
    last_middle_click: Option<Instant>,

    // Radius of the world boundary sphere, used to auto-fit the orbit distance
    // when switching to Orbit mode (see `toggle_mode` and `reset_orbit_view`).
    pub world_radius: f32,

    // Configuration (matches BioSpheres-Q defaults)
    pub move_speed: f32,
    pub sprint_multiplier: f32,
    pub mouse_sensitivity: f32,
    pub roll_speed: f32,
    pub zoom_speed: f32,
    pub horizontal_fov_degrees: f32,
    pub enable_spring: bool,
    pub spring_stiffness: f32,
    pub spring_damping: f32,

    // Keyboard state
    keys_pressed: KeyState,
}

#[derive(Default)]
struct KeyState {
    w: bool,
    s: bool,
    a: bool,
    d: bool,
    space: bool,
    c: bool,
    q: bool,
    e: bool,
    shift: bool,
    tab: bool,
}

impl Default for CameraController {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraController {
    pub fn new() -> Self {
        // Initial rotation: looking down at the scene from a 45-degree angle
        let initial_rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4);

        Self {
            center: Vec3::ZERO,
            distance: 600.0,
            target_distance: 600.0,
            rotation: initial_rotation,
            target_rotation: initial_rotation,
            mode: CameraMode::FreeFly,
            scene_type: SceneType::GpuScene, // Default to GPU scene
            up_direction: Vec3::Y,           // Default up is +Y (gravity pulls down in -Y)
            gravity_mode: 1,
            is_dragging: false,
            last_mouse_pos: None,
            accumulated_mouse_delta: Vec3::ZERO,
            accumulated_scroll: 0.0,
            look_offset: Quat::IDENTITY,
            is_look_dragging: false,
            last_look_mouse_pos: None,
            accumulated_look_delta: Vec3::ZERO,
            last_middle_click: None,
            world_radius: 200.0,
            move_speed: 15.0,
            sprint_multiplier: 6.0, // Increased from 3.0 for faster shift movement
            mouse_sensitivity: 0.003,
            roll_speed: 1.5,
            zoom_speed: 0.2,
            horizontal_fov_degrees: DEFAULT_HORIZONTAL_FOV_DEGREES,
            enable_spring: true,
            spring_stiffness: 50.0,
            spring_damping: 0.9,
            keys_pressed: KeyState::default(),
        }
    }

    /// Create camera for GPU scene (Orbit at 500 units)
    pub fn new_for_gpu_scene() -> Self {
        // Initial rotation: looking down at the scene from a 45-degree angle
        let initial_rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4);

        Self {
            center: Vec3::ZERO,
            distance: 500.0, // Orbit at 500 units
            target_distance: 500.0,
            rotation: initial_rotation,
            target_rotation: initial_rotation,
            mode: CameraMode::Orbit, // GPU scene now defaults to Orbit mode
            scene_type: SceneType::GpuScene,
            up_direction: Vec3::Y,
            gravity_mode: 1,
            is_dragging: false,
            last_mouse_pos: None,
            accumulated_mouse_delta: Vec3::ZERO,
            accumulated_scroll: 0.0,
            look_offset: Quat::IDENTITY,
            is_look_dragging: false,
            last_look_mouse_pos: None,
            accumulated_look_delta: Vec3::ZERO,
            last_middle_click: None,
            world_radius: 200.0,
            move_speed: 15.0,
            sprint_multiplier: 6.0,
            mouse_sensitivity: 0.003,
            roll_speed: 1.5,
            zoom_speed: 0.2,
            horizontal_fov_degrees: DEFAULT_HORIZONTAL_FOV_DEGREES,
            enable_spring: true,
            spring_stiffness: 50.0,
            spring_damping: 0.9,
            keys_pressed: KeyState::default(),
        }
    }

    /// Create camera for preview scene (Orbit at 50 units)
    pub fn new_for_preview_scene() -> Self {
        // Initial rotation: looking down at the scene from a 45-degree angle
        let initial_rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4);

        Self {
            center: Vec3::ZERO,
            distance: 50.0, // Orbit at 50 units
            target_distance: 50.0,
            rotation: initial_rotation,
            target_rotation: initial_rotation,
            mode: CameraMode::Orbit, // Preview starts in Orbit mode
            scene_type: SceneType::PreviewScene,
            up_direction: Vec3::Y,
            gravity_mode: 1,
            is_dragging: false,
            last_mouse_pos: None,
            accumulated_mouse_delta: Vec3::ZERO,
            accumulated_scroll: 0.0,
            look_offset: Quat::IDENTITY,
            is_look_dragging: false,
            last_look_mouse_pos: None,
            accumulated_look_delta: Vec3::ZERO,
            last_middle_click: None,
            world_radius: 200.0,
            move_speed: 15.0,
            sprint_multiplier: 6.0,
            mouse_sensitivity: 0.003,
            roll_speed: 1.5,
            zoom_speed: 0.2,
            horizontal_fov_degrees: DEFAULT_HORIZONTAL_FOV_DEGREES,
            enable_spring: true,
            spring_stiffness: 50.0,
            spring_damping: 0.9,
            keys_pressed: KeyState::default(),
        }
    }

    /// Set yaw/up axis from gravity mode.
    /// gravity_mode: 0=X axis, 1=Y axis, 2=Z axis, 3=radial
    /// For axial modes, yaw is aligned to the selected gravity axis without
    /// flipping when gravity changes sign. For radial mode, keep the current
    /// stable axis because there is no single global yaw axis.
    pub fn set_gravity_direction(&mut self, _gravity: f32, gravity_mode: u32) {
        // Store gravity mode so orbit rotation can use it
        self.gravity_mode = gravity_mode;

        // In radial mode, don't realign - there is no fixed yaw axis.
        // Recomputing up from camera position every frame causes a feedback spin loop.
        if gravity_mode == 3 {
            return;
        }

        let new_up = match gravity_mode {
            0 => Vec3::X,
            1 => Vec3::Y,
            2 => Vec3::Z,
            _ => Vec3::Y,
        };

        if self.up_direction != new_up {
            self.up_direction = new_up;
            self.realign_camera_to_up(new_up);
        }
    }

    /// Helper method to realign camera to a new up direction
    fn realign_camera_to_up(&mut self, new_up: Vec3) {
        let forward = self.rotation * Vec3::NEG_Z;
        let new_rotation = Self::rotation_from_forward_and_up(forward, new_up);

        self.rotation = new_rotation;
        self.target_rotation = new_rotation;
    }

    /// Build a camera rotation with no roll relative to the supplied up axis.
    fn rotation_from_forward_and_up(forward: Vec3, up: Vec3) -> Quat {
        let up = up.normalize_or_zero();
        let forward = forward.normalize_or_zero();
        let stable_forward =
            if forward.length_squared() < 0.001 || forward.cross(up).length_squared() < 0.001 {
                if up.y.abs() < 0.9 {
                    Vec3::Y.cross(up).normalize()
                } else {
                    Vec3::X.cross(up).normalize()
                }
            } else {
                forward
            };

        let right = stable_forward.cross(up).normalize();
        let corrected_up = right.cross(stable_forward).normalize();
        let rot_matrix = glam::Mat3::from_cols(right, corrected_up, -stable_forward);
        Quat::from_mat3(&rot_matrix).normalize()
    }

    /// Default camera rotation: looking down at the scene from a 45-degree angle.
    fn default_rotation() -> Quat {
        Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4)
    }

    /// Orbit distance at which a sphere of the given radius fills
    /// `ORBIT_FIT_FOV_FRACTION` of the vertical field of view, leaving a small
    /// margin at the top and bottom regardless of world size.
    fn fit_distance_for_radius(radius: f32) -> f32 {
        let vertical_fov =
            Self::vertical_fov_radians_for_horizontal(DEFAULT_HORIZONTAL_FOV_DEGREES, 16.0 / 9.0);
        let half_fov = (vertical_fov * 0.5) * ORBIT_FIT_FOV_FRACTION;
        radius / half_fov.sin()
    }

    pub fn vertical_fov_radians_for_horizontal(horizontal_fov_degrees: f32, aspect: f32) -> f32 {
        let aspect = aspect.max(0.001);
        let horizontal_fov_degrees =
            horizontal_fov_degrees.clamp(MIN_HORIZONTAL_FOV_DEGREES, MAX_HORIZONTAL_FOV_DEGREES);
        let tan_half_horizontal = (horizontal_fov_degrees.to_radians() * 0.5).tan();
        2.0 * (tan_half_horizontal / aspect).atan()
    }

    pub fn vertical_fov_radians(&self, aspect: f32) -> f32 {
        Self::vertical_fov_radians_for_horizontal(self.horizontal_fov_degrees, aspect)
    }

    pub fn projection_matrix(&self, aspect: f32, near: f32, far: f32) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.vertical_fov_radians(aspect), aspect, near, far)
    }

    pub fn view_ray_direction(&self, ndc_x: f32, ndc_y: f32, aspect: f32) -> glam::Vec3 {
        let aspect = aspect.max(0.001);
        let horizontal_fov_degrees = self
            .horizontal_fov_degrees
            .clamp(MIN_HORIZONTAL_FOV_DEGREES, MAX_HORIZONTAL_FOV_DEGREES);
        let tan_half_horizontal = (horizontal_fov_degrees.to_radians() * 0.5).tan();
        let tan_half_vertical = tan_half_horizontal / aspect;

        glam::Vec3::new(ndc_x * tan_half_horizontal, ndc_y * tan_half_vertical, -1.0).normalize()
    }

    /// Update the world boundary sphere radius used to auto-fit the orbit distance.
    pub fn set_world_radius(&mut self, radius: f32) {
        self.world_radius = radius;
    }

    /// Reset the orbit camera to its default view: centered on the world origin,
    /// looking down at 45 degrees, with the world sphere fit vertically in view
    /// (small margin top and bottom), and any free-look offset cleared.
    pub fn reset_orbit_view(&mut self) {
        self.center = Vec3::ZERO;
        self.look_offset = Quat::IDENTITY;

        let default_rotation = Self::default_rotation();
        self.rotation = default_rotation;
        self.target_rotation = default_rotation;

        let orbit_distance = match self.scene_type {
            SceneType::GpuScene => Self::fit_distance_for_radius(self.world_radius),
            SceneType::PreviewScene => 50.0,
        };
        self.distance = orbit_distance;
        self.target_distance = orbit_distance;
    }

    /// Reset only the render/view direction while preserving the orbit center,
    /// rotation, and distance. In orbit mode this returns middle-mouse free-look
    /// to the current orbit camera direction without moving the camera position.
    pub fn reset_view_direction(&mut self) {
        self.look_offset = Quat::IDENTITY;
    }

    /// Get the current camera position in world space
    pub fn position(&self) -> Vec3 {
        let offset = self.rotation * Vec3::new(0.0, 0.0, self.distance);
        self.center + offset
    }

    /// Handle mouse button input
    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let control_button = match self.mode {
            CameraMode::Orbit => MouseButton::Right,
            CameraMode::FreeFly => MouseButton::Right,
        };

        if button == control_button {
            self.is_dragging = state == ElementState::Pressed;
            if !self.is_dragging {
                self.last_mouse_pos = None;
            }
        }

        // Middle mouse button: free-look in Orbit mode (angle the camera away
        // from the orbit center without changing orbit position/distance).
        // Double-click resets only the view direction.
        if button == MouseButton::Middle {
            self.is_look_dragging = state == ElementState::Pressed;
            if !self.is_look_dragging {
                self.last_look_mouse_pos = None;
            }

            if state == ElementState::Pressed {
                let now = Instant::now();
                let is_double_click = self
                    .last_middle_click
                    .is_some_and(|t| now.duration_since(t).as_secs_f32() <= DOUBLE_CLICK_SECONDS);

                if is_double_click {
                    if self.mode == CameraMode::Orbit {
                        self.reset_view_direction();
                    }
                    self.last_middle_click = None;
                } else {
                    self.last_middle_click = Some(now);
                }
            }
        }
    }

    /// Returns true if the camera is currently rotating (right-mouse held).
    pub fn is_dragging(&self) -> bool {
        self.is_dragging
    }

    /// Returns true if the camera is currently free-looking (middle-mouse held).
    pub fn is_look_dragging(&self) -> bool {
        self.is_look_dragging
    }

    /// Handle mouse movement
    pub fn handle_mouse_move(&mut self, position: PhysicalPosition<f64>) {
        if self.is_dragging {
            if let Some(last_pos) = self.last_mouse_pos {
                let delta_x = (position.x - last_pos.x) as f32;
                let delta_y = (position.y - last_pos.y) as f32;

                self.accumulated_mouse_delta.x += delta_x;
                self.accumulated_mouse_delta.y += delta_y;
            }
            self.last_mouse_pos = Some(position);
        }

        if self.is_look_dragging {
            if let Some(last_pos) = self.last_look_mouse_pos {
                let delta_x = (position.x - last_pos.x) as f32;
                let delta_y = (position.y - last_pos.y) as f32;

                self.accumulated_look_delta.x += delta_x;
                self.accumulated_look_delta.y += delta_y;
            }
            self.last_look_mouse_pos = Some(position);
        }
    }

    /// Returns the rotation to use for rendering/view direction, including any
    /// middle-mouse free-look offset (Orbit mode only). Orbit position/distance
    /// (see `position()`) is unaffected by this offset.
    pub fn view_rotation(&self) -> Quat {
        if self.mode == CameraMode::Orbit {
            (self.rotation * self.look_offset).normalize()
        } else {
            self.rotation
        }
    }

    /// Handle mouse scroll for zooming in Orbit mode.
    pub fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_x, y) => y,
            MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
        };

        if self.mode == CameraMode::Orbit {
            self.accumulated_scroll += scroll_amount;
        }
    }

    /// Handle keyboard input
    pub fn handle_keyboard(&mut self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;

        if let PhysicalKey::Code(keycode) = event.physical_key {
            match keycode {
                KeyCode::KeyW => self.keys_pressed.w = pressed,
                KeyCode::KeyS => self.keys_pressed.s = pressed,
                KeyCode::KeyA => self.keys_pressed.a = pressed,
                KeyCode::KeyD => self.keys_pressed.d = pressed,
                KeyCode::Space => self.keys_pressed.space = pressed,
                KeyCode::KeyC => self.keys_pressed.c = pressed,
                KeyCode::KeyQ => self.keys_pressed.q = pressed,
                KeyCode::KeyE => self.keys_pressed.e = pressed,
                KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys_pressed.shift = pressed,
                KeyCode::Tab => {
                    if pressed && !self.keys_pressed.tab {
                        self.toggle_mode();
                        log::info!("Camera mode switched to: {:?}", self.mode);
                    }
                    self.keys_pressed.tab = pressed;
                }
                KeyCode::BracketLeft if pressed && !event.repeat => {
                    if self.mode == CameraMode::Orbit {
                        self.zoom_speed = (self.zoom_speed / 1.25).clamp(0.01, 2.0);
                        log::info!("Orbit zoom sensitivity: {:.3}", self.zoom_speed);
                    }
                }
                KeyCode::BracketRight if pressed && !event.repeat => {
                    if self.mode == CameraMode::Orbit {
                        self.zoom_speed = (self.zoom_speed * 1.25).clamp(0.01, 2.0);
                        log::info!("Orbit zoom sensitivity: {:.3}", self.zoom_speed);
                    }
                }
                KeyCode::Backslash if pressed && !event.repeat => {
                    if self.mode == CameraMode::Orbit {
                        self.zoom_speed = 0.2;
                        log::info!("Orbit zoom sensitivity reset: {:.3}", self.zoom_speed);
                    }
                }
                _ => {}
            }
        }
    }

    /// Toggle between Orbit and FreeFly modes
    fn toggle_mode(&mut self) {
        match self.mode {
            CameraMode::Orbit => {
                // Switch to FreeFly: move orbit center to current camera position
                self.center = self.position();
                self.distance = 0.0;
                self.target_distance = 0.0;
                self.target_rotation = self.rotation;
                self.mode = CameraMode::FreeFly;
            }
            CameraMode::FreeFly => {
                // Switch to Orbit: set orbit center to world origin, use a distance that
                // fits the world sphere vertically in view (small margin top/bottom).
                self.center = Vec3::ZERO;
                let orbit_distance = match self.scene_type {
                    SceneType::GpuScene => Self::fit_distance_for_radius(self.world_radius),
                    SceneType::PreviewScene => 50.0,
                };
                self.distance = orbit_distance;
                self.target_distance = orbit_distance;

                let forward = (self.rotation * Vec3::NEG_Z).normalize();
                self.target_rotation =
                    Self::rotation_from_forward_and_up(forward, self.up_direction);
                self.rotation = self.target_rotation;

                self.mode = CameraMode::Orbit;
            }
        }
    }

    /// Update camera state (call once per frame)
    pub fn update(&mut self, dt: f32) {
        // 1. ZOOM (scroll) - Only in Orbit mode
        if self.mode == CameraMode::Orbit && self.accumulated_scroll.abs() > 0.001 {
            // Additive zoom - constant speed regardless of distance.
            self.target_distance -= self.accumulated_scroll * self.zoom_speed * 30.0;
            self.target_distance = self.target_distance.max(0.1);
        }
        self.accumulated_scroll = 0.0;

        // 2. FREE LOOK (middle mouse drag) - Only in Orbit mode
        if self.mode == CameraMode::Orbit && self.accumulated_look_delta.length_squared() > 0.0 {
            let delta = self.accumulated_look_delta.truncate() * self.mouse_sensitivity;
            let yaw = Quat::from_axis_angle(Vec3::Y, -delta.x);
            let pitch = Quat::from_axis_angle(Vec3::X, -delta.y);
            self.look_offset = (self.look_offset * yaw * pitch).normalize();
        }
        self.accumulated_look_delta = Vec3::ZERO;

        // Apply spring interpolation to distance and rotation in orbit mode
        if self.mode == CameraMode::Orbit {
            if self.enable_spring {
                // Spring for distance
                let distance_error = self.target_distance - self.distance;
                let velocity = distance_error * self.spring_stiffness * dt;
                self.distance += velocity * (1.0 - self.spring_damping);

                // Spring for rotation
                self.rotation = self.rotation.slerp(
                    self.target_rotation,
                    self.spring_stiffness * dt * (1.0 - self.spring_damping),
                );
            } else {
                self.distance = self.target_distance;
                self.rotation = self.target_rotation;
            }
        }

        // 2. ROTATION (mouse)
        if self.accumulated_mouse_delta.length_squared() > 0.0 {
            let delta = self.accumulated_mouse_delta.truncate() * self.mouse_sensitivity;

            if self.mode == CameraMode::Orbit {
                if self.gravity_mode == 3 {
                    // Radial mode: fully quaternion-based orbit - no fixed axis.
                    // Yaw around camera's local Y, pitch around camera's local X.
                    let local_y = self.target_rotation * Vec3::Y;
                    let local_x = self.target_rotation * Vec3::X;
                    let yaw = Quat::from_axis_angle(local_y, -delta.x);
                    let pitch = Quat::from_axis_angle(local_x, -delta.y);
                    self.target_rotation = (yaw * pitch * self.target_rotation).normalize();
                } else {
                    // Axial mode: longitude/latitude rotation around gravity-defined up axis
                    let yaw_rotation = Quat::from_axis_angle(self.up_direction, -delta.x);
                    self.target_rotation = yaw_rotation * self.target_rotation;

                    let camera_right = self.target_rotation * Vec3::X;
                    let right_axis = (camera_right
                        - self.up_direction * camera_right.dot(self.up_direction))
                    .normalize_or_zero();
                    if right_axis.length_squared() > 0.001 {
                        let pitch_rotation = Quat::from_axis_angle(right_axis, -delta.y);
                        let proposed_rotation = (pitch_rotation * self.target_rotation).normalize();
                        let proposed_forward = proposed_rotation * Vec3::NEG_Z;
                        const MAX_UP_DOT: f32 = 0.98;
                        if proposed_forward.dot(self.up_direction).abs() < MAX_UP_DOT {
                            self.target_rotation = Self::rotation_from_forward_and_up(
                                proposed_forward,
                                self.up_direction,
                            );
                        } else {
                            let current_forward = self.target_rotation * Vec3::NEG_Z;
                            self.target_rotation = Self::rotation_from_forward_and_up(
                                current_forward,
                                self.up_direction,
                            );
                        }
                    } else {
                        let current_forward = self.target_rotation * Vec3::NEG_Z;
                        self.target_rotation =
                            Self::rotation_from_forward_and_up(current_forward, self.up_direction);
                    }
                }
            } else {
                // FreeFly mode: yaw around world up to prevent roll accumulation,
                // pitch around camera's local X axis.
                let pitch = Quat::from_axis_angle(self.target_rotation * Vec3::X, -delta.y);
                let free_yaw = Quat::from_axis_angle(self.up_direction, -delta.x);

                self.target_rotation = (free_yaw * pitch) * self.target_rotation;
                self.target_rotation = self.target_rotation.normalize();
            }
        }
        self.accumulated_mouse_delta = Vec3::ZERO;

        // Apply spring interpolation to rotation in free fly mode
        if self.mode == CameraMode::FreeFly {
            // Direct rotation - no spring damping for free fly
            self.rotation = self.target_rotation;
        }

        // 3. ROLL (Q/E) - Only in radial FreeFly mode
        if self.mode == CameraMode::FreeFly && self.gravity_mode == 3 {
            let mut roll_amount = 0.0;
            if self.keys_pressed.q {
                roll_amount += 1.0;
            }
            if self.keys_pressed.e {
                roll_amount -= 1.0;
            }

            if roll_amount != 0.0 {
                let roll_axis = self.target_rotation * Vec3::Z;
                let roll = Quat::from_axis_angle(roll_axis, roll_amount * self.roll_speed * dt);
                self.target_rotation = (roll * self.target_rotation).normalize();
            }
        }

        // 4. MOVEMENT (WASD + Space + C) - Only in FreeFly mode
        if self.mode == CameraMode::FreeFly {
            let mut speed = self.move_speed * dt;
            if self.keys_pressed.shift {
                speed *= self.sprint_multiplier;
            }

            let mut move_vec = Vec3::ZERO;

            if self.keys_pressed.w {
                move_vec += self.rotation * Vec3::Z * -1.0; // forward
            }
            if self.keys_pressed.s {
                move_vec += self.rotation * Vec3::Z; // backward
            }
            if self.keys_pressed.a {
                move_vec += self.rotation * Vec3::X * -1.0; // left
            }
            if self.keys_pressed.d {
                move_vec += self.rotation * Vec3::X; // right
            }
            if self.keys_pressed.space {
                move_vec += self.rotation * Vec3::Y; // up
            }
            if self.keys_pressed.c {
                move_vec += self.rotation * Vec3::Y * -1.0; // down
            }

            if move_vec.length_squared() > 0.0 {
                self.center += move_vec.normalize() * speed;
                log::debug!(
                    "FreeFly movement: center={:?}, speed={}",
                    self.center,
                    speed
                );
            }
        }
    }
}

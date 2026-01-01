use glam::{Quat, Vec3};
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta};
use winit::dpi::PhysicalPosition;
use winit::keyboard::{KeyCode, PhysicalKey};

/// Camera mode - matches BioSpheres-Q
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    Orbit,
    FreeFly,
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
    
    // Mouse state
    is_dragging: bool,
    last_mouse_pos: Option<PhysicalPosition<f64>>,
    accumulated_mouse_delta: Vec3,
    accumulated_scroll: f32,
    
    // Configuration (matches BioSpheres-Q defaults)
    pub move_speed: f32,
    pub sprint_multiplier: f32,
    pub mouse_sensitivity: f32,
    pub roll_speed: f32,
    pub zoom_speed: f32,
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
            distance: 50.0,
            target_distance: 50.0,
            rotation: initial_rotation,
            target_rotation: initial_rotation,
            mode: CameraMode::Orbit,
            is_dragging: false,
            last_mouse_pos: None,
            accumulated_mouse_delta: Vec3::ZERO,
            accumulated_scroll: 0.0,
            move_speed: 15.0,
            sprint_multiplier: 3.0,
            mouse_sensitivity: 0.003,
            roll_speed: 1.5,
            zoom_speed: 0.2,
            enable_spring: true,
            spring_stiffness: 50.0,
            spring_damping: 0.9,
            keys_pressed: KeyState::default(),
        }
    }
    
    /// Get the current camera position in world space
    pub fn position(&self) -> Vec3 {
        let offset = self.rotation * Vec3::new(0.0, 0.0, self.distance);
        self.center + offset
    }
    
    /// Handle mouse button input
    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let control_button = match self.mode {
            CameraMode::Orbit => MouseButton::Middle,
            CameraMode::FreeFly => MouseButton::Right,
        };
        
        if button == control_button {
            self.is_dragging = state == ElementState::Pressed;
            if !self.is_dragging {
                self.last_mouse_pos = None;
            }
        }
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
    }
    
    /// Handle mouse scroll for zooming (Orbit mode) or focal plane (FreeFly mode)
    pub fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_x, y) => y,
            MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
        };
        
        self.accumulated_scroll += scroll_amount;
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
                // Switch to Orbit: set orbit center to world origin, calculate distance
                let current_pos = self.position();
                self.center = Vec3::ZERO;
                let new_distance = current_pos.distance(Vec3::ZERO);
                self.distance = new_distance;
                self.target_distance = new_distance;
                self.target_rotation = self.rotation;
                self.mode = CameraMode::Orbit;
            }
        }
    }
    
    /// Update camera state (call once per frame)
    pub fn update(&mut self, dt: f32) {
        // 1. ZOOM (scroll) - Only in Orbit mode
        if self.mode == CameraMode::Orbit && self.accumulated_scroll.abs() > 0.001 {
            // Additive zoom - constant speed regardless of distance
            self.target_distance -= self.accumulated_scroll * self.zoom_speed * 30.0;
            self.target_distance = self.target_distance.max(0.1);
        }
        self.accumulated_scroll = 0.0;
        
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
                    self.spring_stiffness * dt * (1.0 - self.spring_damping)
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
                // Orbit mode: rotate around world origin
                let yaw = Quat::from_axis_angle(Vec3::Y, -delta.x);
                let right = self.target_rotation * Vec3::X;
                let pitch = Quat::from_axis_angle(right, -delta.y);
                
                self.target_rotation = yaw * pitch * self.target_rotation;
                self.target_rotation = self.target_rotation.normalize();
            } else {
                // FreeFly mode: free rotation
                let pitch = Quat::from_axis_angle(self.target_rotation * Vec3::X, -delta.y);
                let local_up = self.target_rotation * Vec3::Y;
                let free_yaw = Quat::from_axis_angle(local_up, -delta.x);
                
                self.target_rotation = (free_yaw * pitch) * self.target_rotation;
                self.target_rotation = self.target_rotation.normalize();
            }
        }
        self.accumulated_mouse_delta = Vec3::ZERO;
        
        // Apply spring interpolation to rotation in free fly mode
        if self.mode == CameraMode::FreeFly {
            if self.enable_spring {
                self.rotation = self.rotation.slerp(
                    self.target_rotation,
                    self.spring_stiffness * dt * (1.0 - self.spring_damping)
                );
            } else {
                self.rotation = self.target_rotation;
            }
        }
        
        // 3. ROLL (Q/E) - Only in FreeFly mode
        if self.mode == CameraMode::FreeFly {
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
                log::debug!("FreeFly movement: center={:?}, speed={}", self.center, speed);
            }
        }
    }
}

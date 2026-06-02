//! Main menu scene.
//!
//! Holds two live PreviewScenes rendered into off-screen textures that are then
//! displayed as egui images on either side of the centred button column.

use crate::cell::types::CellTypeVisuals;
use crate::genome::Genome;
use crate::scene::traits::Scene as _;
use egui::TextureId;
use glam::{Quat, Vec3};
use std::sync::atomic::{AtomicU64, Ordering};

/// Counter seeded from the current time on first use, then incremented by 2 per
/// menu creation. This ensures a different genome pair is shown on each app launch
/// rather than always starting at index 0 of the sorted genomes directory.
static MENU_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Returns a time-seeded starting value for MENU_COUNTER, initialised once.
fn time_seeded_counter() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xdeadbeef_cafebabe)
}

// -- orbit constants -----------------------------------------------------------

/// Radians-per-second for the auto-orbit.
const ORBIT_SPEED: f32 = 0.28;

/// Camera distance from the organism centre.
const ORBIT_DISTANCE: f32 = 40.0;

/// Downward camera tilt in radians.
const ORBIT_PITCH: f32 = 0.35;

// -- MainMenuScene -------------------------------------------------------------

pub struct MainMenuScene {
    pub left_preview: crate::scene::PreviewScene,
    pub right_preview: crate::scene::PreviewScene,

    /// Names shown under each panel (genome names).
    pub left_genome_name: String,
    pub right_genome_name: String,

    left_color_tex: wgpu::Texture,
    right_color_tex: wgpu::Texture,

    /// egui TextureIds for the off-screen render targets.
    pub left_tex_id: TextureId,
    pub right_tex_id: TextureId,

    left_orbit_angle: f32,
    right_orbit_angle: f32,

    pub panel_width: u32,
    pub panel_height: u32,
    format: wgpu::TextureFormat,
}

impl MainMenuScene {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        egui_renderer: &mut egui_wgpu::Renderer,
    ) -> Self {
        let panel_width = (surface_config.width / 3).max(1);
        let panel_height = surface_config.height.max(1);

        // Build a fake surface config at panel dimensions so the preview scene
        // pipelines and depth texture are created at the right size.
        let panel_config = wgpu::SurfaceConfiguration {
            usage: surface_config.usage,
            format: surface_config.format,
            width: panel_width,
            height: panel_height,
            present_mode: surface_config.present_mode,
            alpha_mode: surface_config.alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: surface_config.desired_maximum_frame_latency,
        };

        // Seed the counter from the current time on first use (compare-exchange from 0),
        // then advance by 2 for this menu creation. This ensures a different genome pair
        // is shown on each app launch instead of always starting at index 0.
        let seed = time_seeded_counter();
        let _ = MENU_COUNTER.compare_exchange(0, seed, Ordering::Relaxed, Ordering::Relaxed);
        let counter = MENU_COUNTER.fetch_add(2, Ordering::Relaxed);

        // Left preview - load a saved genome, fall back to procedural generation
        let left_genome = Genome::load_from_genomes_dir_at(counter)
            .unwrap_or_else(|| Genome::generate_procedural(counter));
        let left_name = left_genome.name.clone();
        let mut left_preview = crate::scene::PreviewScene::new(device, queue, &panel_config);
        left_preview.show_adhesion_lines = true;
        left_preview.show_skybox = false;
        left_preview.clear_color = [7.0 / 255.0, 10.0 / 255.0, 22.0 / 255.0];
        left_preview.camera.distance = ORBIT_DISTANCE;
        left_preview.camera.target_distance = ORBIT_DISTANCE;
        left_preview.update_genome(&left_genome);
        // Disable gizmos and split rings - not appropriate for the menu backdrop
        left_preview.gizmo_renderer.update_config(&crate::rendering::orientation_gizmo::GizmoConfig {
            visible: false,
            ..Default::default()
        });
        left_preview.split_ring_renderer.update_config(&crate::rendering::split_rings::SplitRingConfig {
            visible: false,
            ..Default::default()
        });

        // Right preview - load a different saved genome (counter+1 ensures a different pick)
        let right_genome = Genome::load_from_genomes_dir_at(counter + 1)
            .unwrap_or_else(|| Genome::generate_procedural(counter + 1));
        let right_name = right_genome.name.clone();
        let mut right_preview = crate::scene::PreviewScene::new(device, queue, &panel_config);
        right_preview.show_adhesion_lines = true;
        right_preview.show_skybox = false;
        right_preview.clear_color = [7.0 / 255.0, 10.0 / 255.0, 22.0 / 255.0];
        right_preview.camera.distance = ORBIT_DISTANCE;
        right_preview.camera.target_distance = ORBIT_DISTANCE;
        right_preview.update_genome(&right_genome);
        // Disable gizmos and split rings - not appropriate for the menu backdrop
        right_preview.gizmo_renderer.update_config(&crate::rendering::orientation_gizmo::GizmoConfig {
            visible: false,
            ..Default::default()
        });
        right_preview.split_ring_renderer.update_config(&crate::rendering::split_rings::SplitRingConfig {
            visible: false,
            ..Default::default()
        });

        // Off-screen render targets
        let (left_color_tex, right_color_tex) =
            Self::create_textures(device, surface_config.format, panel_width, panel_height);

        // Register with egui renderer
        let left_view = left_color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let right_view = right_color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let left_tex_id =
            egui_renderer.register_native_texture(device, &left_view, wgpu::FilterMode::Linear);
        let right_tex_id =
            egui_renderer.register_native_texture(device, &right_view, wgpu::FilterMode::Linear);

        Self {
            left_preview,
            right_preview,
            left_genome_name: left_name,
            right_genome_name: right_name,
            left_color_tex,
            right_color_tex,
            left_tex_id,
            right_tex_id,
            left_orbit_angle: 0.0,
            right_orbit_angle: std::f32::consts::PI, // opposite start
            panel_width,
            panel_height,
            format: surface_config.format,
        }
    }

    // -- per-frame update ------------------------------------------------------

    pub fn update(&mut self, dt: f32) {
        self.left_orbit_angle += ORBIT_SPEED * dt;
        self.right_orbit_angle -= ORBIT_SPEED * dt; // counter-rotate for visual contrast

        let pitch = Quat::from_axis_angle(Vec3::X, -ORBIT_PITCH);

        let left_rot = Quat::from_axis_angle(Vec3::Y, self.left_orbit_angle) * pitch;
        self.left_preview.camera.rotation = left_rot;
        self.left_preview.camera.target_rotation = left_rot;

        let right_rot = Quat::from_axis_angle(Vec3::Y, self.right_orbit_angle) * pitch;
        self.right_preview.camera.rotation = right_rot;
        self.right_preview.camera.target_rotation = right_rot;

        // Advance simulations forward by dt - no seeking, no checkpoints.
        let left_genome = self.left_preview.genome.clone();
        let left_config = self.left_preview.config.clone();
        self.left_preview.state.step_forward(dt, &left_genome, &left_config);

        let right_genome = self.right_preview.genome.clone();
        let right_config = self.right_preview.config.clone();
        self.right_preview.state.step_forward(dt, &right_genome, &right_config);

        // Run one incremental resimulation chunk each frame.
        self.left_preview.update(dt);
        self.right_preview.update(dt);
    }

    // -- render to off-screen textures -----------------------------------------

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cell_type_visuals: Option<&[CellTypeVisuals]>,
    ) {
        let left_view = self.left_color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let right_view =
            self.right_color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Render each preview into its off-screen texture.
        // lod_scale_factor must match the editor default (500.0) so that cells at the
        // menu's orbit distance (~40 units) compute a screen_radius above lod_threshold_low
        // and render at LOD >= 1 (textured). With scale 1.0 every cell falls to LOD 0
        // (plain sphere) because screen_radius = radius/distance * 1.0 ~= 0.025 < 10.0.
        self.left_preview.render(
            device,
            queue,
            &left_view,
            cell_type_visuals,
            400.0,
            500.0,
            10.0,
            25.0,
            50.0,
            false,
            0.12, // thick black outline for menu backdrop
        );
        self.right_preview.render(
            device,
            queue,
            &right_view,
            cell_type_visuals,
            400.0,
            500.0,
            10.0,
            25.0,
            50.0,
            false,
            0.12, // thick black outline for menu backdrop
        );

        // NOTE: Do NOT call update_egui_texture_from_wgpu_texture here.
        // The TextureIds already point to the correct off-screen textures from
        // register_native_texture at construction time. Calling update every frame
        // creates a new wgpu::Sampler + BindGroup each call (120/sec at 60fps)
        // that leak into wgpu's internal ref-count. Rebinding only happens in
        // resize() where the underlying texture actually changes.
    }

    // -- resize ----------------------------------------------------------------

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        egui_renderer: &mut egui_wgpu::Renderer,
        width: u32,
        height: u32,
    ) {
        let panel_width = (width / 3).max(1);
        let panel_height = height.max(1);
        self.panel_width = panel_width;
        self.panel_height = panel_height;

        self.left_preview.resize(device, panel_width, panel_height);
        self.right_preview.resize(device, panel_width, panel_height);

        let (left_tex, right_tex) =
            Self::create_textures(device, self.format, panel_width, panel_height);
        self.left_color_tex = left_tex;
        self.right_color_tex = right_tex;

        // Rebind new texture views
        let left_view = self.left_color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let right_view =
            self.right_color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        egui_renderer.update_egui_texture_from_wgpu_texture(
            device,
            &left_view,
            wgpu::FilterMode::Linear,
            self.left_tex_id,
        );
        egui_renderer.update_egui_texture_from_wgpu_texture(
            device,
            &right_view,
            wgpu::FilterMode::Linear,
            self.right_tex_id,
        );
    }

    // -- helpers ---------------------------------------------------------------

    fn create_textures(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::Texture) {
        let usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        let desc = wgpu::TextureDescriptor {
            label: Some("menu_left_tex"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        };
        let left = device.create_texture(&desc);
        let right = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("menu_right_tex"),
            ..desc
        });
        (left, right)
    }
}

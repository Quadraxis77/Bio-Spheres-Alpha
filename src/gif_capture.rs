//! GIF thumbnail capture for genomes — incremental, non-blocking.
//!
//! Instead of blocking for 1–2 seconds, the capture runs one frame per app
//! frame. The caller drives it by calling `GifCaptureState::step()` each frame
//! and checking `is_done()`. Progress is exposed so the UI can show a bar.
//!
//! Orbit constants match the main menu for visual consistency.
//!
//! ## Dynamic capture window
//!
//! At `begin()` time the simulation is pre-scanned (CPU-only, no rendering) to
//! find the moment when the organism either reaches 256 cells or dies (cell
//! count drops to 0 after having been non-zero). The GIF frames are then spread
//! evenly over that window so the thumbnail always shows the full life-cycle
//! regardless of how fast or slow the genome grows.

use std::path::PathBuf;
use glam::{Quat, Vec3};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Resolution of the GIF thumbnail (square).
pub const GIF_SIZE: u32 = 256;

/// Number of frames in the GIF.
pub const GIF_FRAMES: u32 = 120;

/// Delay between frames in centiseconds (5cs = 50ms → 20fps playback).
pub const FRAME_DELAY_CS: u16 = 5;

/// Orbit speed in radians per second — matches the main menu exactly.
const ORBIT_SPEED: f32 = 0.28;

/// Pre-scan step size (seconds). Smaller = more accurate window, but slower begin().
const SCAN_STEP: f32 = 0.25;

/// Maximum sim time to scan before giving up (seconds).
const SCAN_MAX_TIME: f32 = 300.0;

/// Target cell count that ends the scan.
const SCAN_TARGET_CELLS: usize = 256;

/// Fallback capture window when the scan finds nothing interesting (seconds).
const FALLBACK_WINDOW: f32 = 60.0;

/// How many consecutive scan steps with no cell-count change counts as stagnation.
/// At SCAN_STEP=0.25s this is 20 steps = 5 seconds of no change.
const STAGNATION_STEPS: usize = 20;

// ── GifCaptureState ───────────────────────────────────────────────────────────

/// Incremental GIF capture state. Drive by calling `step()` once per app frame.
pub struct GifCaptureState {
    /// Off-screen preview scene used for rendering.
    scene: crate::scene::PreviewScene,
    /// Off-screen render target.
    color_tex: wgpu::Texture,
    /// Staging buffer for pixel readback.
    staging: wgpu::Buffer,
    /// Padded bytes per row (wgpu alignment requirement).
    padded_bpr: u32,
    /// Unpadded bytes per row (actual pixel data width).
    unpadded_bpr: u32,
    /// Whether the surface format is BGRA (needs R/B swap).
    is_bgra: bool,
    /// Accumulated RGBA frames.
    frames: Vec<Vec<u8>>,
    /// Current orbit angle (radians) — starts at the saved camera yaw.
    orbit_angle: f32,
    /// Initial camera rotation from the preview (used to extract pitch/yaw).
    initial_rotation: Quat,
    /// Camera distance to use for the orbit.
    orbit_distance: f32,
    /// Path to save the GIF to.
    pub output_path: PathBuf,
    /// Frame index to display as the static thumbnail (from the time slider position).
    pub static_frame_idx: u32,
    /// Result of the completed capture (set when done).
    pub result: Option<Result<PathBuf, String>>,
    /// Sim time step per captured frame (total_window / GIF_FRAMES).
    frame_interval: f32,
}

// Manual trait impls — GPU resources are not Clone or Debug.
impl std::fmt::Debug for GifCaptureState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GifCaptureState")
            .field("frames_done", &self.frames.len())
            .field("output_path", &self.output_path)
            .field("frame_interval", &self.frame_interval)
            .finish()
    }
}
impl Clone for GifCaptureState {
    fn clone(&self) -> Self {
        panic!("GifCaptureState should never be cloned")
    }
}

impl GifCaptureState {
    /// Begin a new capture.
    ///
    /// `cam_rotation` and `cam_distance` come from the preview camera at save time.
    /// The GIF orbits at that distance, starting from the saved camera angle, so
    /// the static thumbnail frame matches exactly what the user was looking at.
    pub fn begin(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        genome: &crate::genome::Genome,
        sim_time: f32,
        cam_rotation: Quat,
        cam_distance: f32,
        cam_center: Vec3,
        cell_type_visuals: Option<&[crate::cell::types::CellTypeVisuals]>,
    ) -> Result<Self, String> {
        let panel_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: GIF_SIZE,
            height: GIF_SIZE,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // ── Pre-scan: find the capture window ────────────────────────────────
        // We run a lightweight CPU-only simulation (no rendering) to find when
        // the organism reaches SCAN_TARGET_CELLS or dies after being alive.
        let capture_window = {
            let mut scan_scene = crate::scene::PreviewScene::new(device, queue, &panel_config);
            scan_scene.update_genome(genome);

            let g = scan_scene.genome.clone();
            let c = scan_scene.config.clone();

            let mut elapsed = 0.0f32;
            let mut ever_alive = false;
            let mut last_count = 0usize;
            let mut stagnant_steps = 0usize;

            loop {
                scan_scene.state.step_forward(SCAN_STEP, &g, &c);
                elapsed += SCAN_STEP;

                let count = scan_scene.state.work_state.cell_count;

                if count > 0 { ever_alive = true; }

                // Track stagnation: count consecutive steps with no change.
                if count == last_count {
                    stagnant_steps += 1;
                } else {
                    stagnant_steps = 0;
                }
                last_count = count;

                // Stop when we hit the target cell count.
                if count >= SCAN_TARGET_CELLS {
                    break;
                }

                // Stop when the organism dies (was alive, now 0 cells).
                if ever_alive && count == 0 {
                    break;
                }

                // Stop when the cell count has been completely flat for long enough.
                // Only trigger stagnation once the organism has actually been alive —
                // we don't want to stop immediately if the genome starts with 0 cells
                // and takes a moment to spawn.
                if ever_alive && stagnant_steps >= STAGNATION_STEPS {
                    // Wind back to just before stagnation started so the GIF
                    // doesn't waste frames on a frozen organism.
                    elapsed -= SCAN_STEP * stagnant_steps as f32;
                    elapsed = elapsed.max(SCAN_STEP);
                    break;
                }

                // Hard cap.
                if elapsed >= SCAN_MAX_TIME {
                    break;
                }
            }

            // If nothing interesting happened, use the fallback window.
            if elapsed < 1.0 { FALLBACK_WINDOW } else { elapsed }
        };

        let frame_interval = capture_window / GIF_FRAMES as f32;

        // ── Render scene (starts fresh from t=0) ─────────────────────────────
        let mut scene = crate::scene::PreviewScene::new(device, queue, &panel_config);
        scene.show_adhesion_lines = true;
        scene.show_skybox = false;
        scene.camera.distance = cam_distance;
        scene.camera.target_distance = cam_distance;
        scene.camera.center = cam_center;
        scene.gizmo_renderer.update_config(&crate::rendering::orientation_gizmo::GizmoConfig {
            visible: false, ..Default::default()
        });
        scene.split_ring_renderer.update_config(&crate::rendering::split_rings::SplitRingConfig {
            visible: false, ..Default::default()
        });
        scene.update_genome(genome);

        // Extract the yaw angle from the saved camera rotation so the orbit
        // starts at the same horizontal angle the user was looking from.
        // We decompose the rotation into a yaw around Y and ignore pitch/roll —
        // the orbit will re-apply pitch each frame from the saved rotation.
        let forward = cam_rotation * Vec3::NEG_Z;
        let initial_yaw = forward.x.atan2(-forward.z); // atan2(x, -z) = yaw around Y

        // Map the slider position into the capture window to pick the static frame.
        let static_frame_idx = if capture_window > 0.0 {
            ((sim_time / capture_window) * GIF_FRAMES as f32).round() as u32
        } else {
            0
        }.min(GIF_FRAMES.saturating_sub(1));

        // ── Off-screen render target ──────────────────────────────────────────
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("GIF Capture Color Texture"),
            size: wgpu::Extent3d { width: GIF_SIZE, height: GIF_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let bytes_per_pixel = 4u32;
        let unpadded_bpr = GIF_SIZE * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = (unpadded_bpr + align - 1) / align * align;
        let staging_size = (padded_bpr * GIF_SIZE) as u64;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GIF Capture Staging Buffer"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let is_bgra = matches!(
            surface_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        let output_path = crate::app_dirs::genomes_dir()
            .join(format!("{}.gif", crate::app_dirs::sanitize_filename(&genome.name)));

        let _ = cell_type_visuals;

        Ok(Self {
            scene,
            color_tex,
            staging,
            padded_bpr,
            unpadded_bpr,
            is_bgra,
            frames: Vec::with_capacity(GIF_FRAMES as usize),
            orbit_angle: initial_yaw,
            initial_rotation: cam_rotation,
            orbit_distance: cam_distance,
            output_path,
            static_frame_idx,
            result: None,
            frame_interval,
        })
    }

    /// How many frames have been captured so far.
    pub fn frames_done(&self) -> u32 { self.frames.len() as u32 }

    /// Total frames to capture.
    pub fn frames_total(&self) -> u32 { GIF_FRAMES }

    /// Progress 0.0–1.0.
    pub fn progress(&self) -> f32 { self.frames_done() as f32 / GIF_FRAMES as f32 }

    /// Whether capture + encoding is complete.
    pub fn is_done(&self) -> bool { self.result.is_some() }

    /// Capture one frame. Call once per app frame until `is_done()`.
    pub fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cell_type_visuals: Option<&[crate::cell::types::CellTypeVisuals]>,
    ) {
        if self.is_done() { return; }

        // Advance simulation by the dynamic per-frame interval.
        {
            let g = self.scene.genome.clone();
            let c = self.scene.config.clone();
            self.scene.state.step_forward(self.frame_interval, &g, &c);
        }

        // Set camera for this frame — orbit around the saved center point at
        // the saved distance, starting from the saved yaw, with the saved pitch.
        let saved_forward = self.initial_rotation * Vec3::NEG_Z;
        let pitch_angle = saved_forward.y.asin();
        let pitch = Quat::from_axis_angle(Vec3::X, pitch_angle);
        let yaw   = Quat::from_axis_angle(Vec3::Y, self.orbit_angle);
        let rot   = yaw * pitch;
        self.orbit_angle += ORBIT_SPEED * self.frame_interval;
        self.scene.camera.distance = self.orbit_distance;
        self.scene.camera.target_distance = self.orbit_distance;
        self.scene.camera.rotation = rot;
        self.scene.camera.target_rotation = rot;

        // Render.
        let view = self.color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        use crate::scene::traits::Scene as _;
        self.scene.render(device, queue, &view, cell_type_visuals, 0.0, 500.0, 10.0, 25.0, 50.0, false, 0.08);

        // Copy to staging.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GIF Frame Copy"),
        });
        encoder.copy_texture_to_buffer(
            self.color_tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bpr),
                    rows_per_image: Some(GIF_SIZE),
                },
            },
            wgpu::Extent3d { width: GIF_SIZE, height: GIF_SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Readback (blocking for this one frame — ~1ms).
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        if device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).is_err() {
            self.result = Some(Err("GIF capture: device.poll failed".into()));
            return;
        }
        if rx.recv().map_err(|_| ()).and_then(|r| r.map_err(|_| ())).is_err() {
            self.result = Some(Err("GIF capture: staging map failed".into()));
            return;
        }

        let mapped = self.staging.slice(..).get_mapped_range();
        let mut rgba: Vec<u8> = Vec::with_capacity((GIF_SIZE * GIF_SIZE * 4) as usize);
        for row in 0..GIF_SIZE {
            let row_start = (row * self.padded_bpr) as usize;
            let row_bytes = &mapped[row_start..row_start + self.unpadded_bpr as usize];
            if self.is_bgra {
                for chunk in row_bytes.chunks_exact(4) {
                    rgba.push(chunk[2]); rgba.push(chunk[1]); rgba.push(chunk[0]); rgba.push(chunk[3]);
                }
            } else {
                rgba.extend_from_slice(row_bytes);
            }
        }
        drop(mapped);
        self.staging.unmap();
        self.frames.push(rgba);

        // All frames captured — encode and save.
        if self.frames.len() as u32 == GIF_FRAMES {
            self.result = Some(self.encode_and_save());
        }
    }

    fn encode_and_save(&mut self) -> Result<PathBuf, String> {
        let file = std::fs::File::create(&self.output_path)
            .map_err(|e| format!("GIF: could not create {:?}: {}", self.output_path, e))?;

        let mut encoder = gif::Encoder::new(
            std::io::BufWriter::new(file),
            GIF_SIZE as u16,
            GIF_SIZE as u16,
            &[],
        ).map_err(|e| format!("GIF: encoder init: {}", e))?;

        encoder.set_repeat(gif::Repeat::Infinite)
            .map_err(|e| format!("GIF: set_repeat: {}", e))?;

        for (i, rgba) in self.frames.iter().enumerate() {
            let mut frame = gif::Frame::from_rgba_speed(
                GIF_SIZE as u16, GIF_SIZE as u16,
                &mut rgba.clone(),
                10,
            );
            // Hold the last frame for 1 second (100cs) before looping.
            frame.delay = if i + 1 == self.frames.len() { 100 } else { FRAME_DELAY_CS };
            encoder.write_frame(&frame)
                .map_err(|e| format!("GIF: write_frame: {}", e))?;
        }

        drop(encoder);

        // Write sidecar .meta file storing the static thumbnail frame index.
        let meta_path = self.output_path.with_extension("gif.meta");
        let meta = format!("{{\"static_frame\":{}}}", self.static_frame_idx);
        let _ = std::fs::write(&meta_path, meta);

        Ok(self.output_path.clone())
    }
}

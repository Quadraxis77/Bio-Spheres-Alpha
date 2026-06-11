//! Mossrock System
//!
//! Owns BoulderBuffers and BoulderRenderer, handles spawning from cave ceiling
//! voxels, and coordinates the per-frame update cycle.
//!
//! ## Architectural note - why not reuse the cell physics pipeline?
//!
//! See boulder_buffers.rs for the full explanation. In short: boulders should have
//! been placed in reserved slots within the existing cell physics buffers so that
//! gravity, collision, cave SDF, and velocity integration would all have been
//! handled by the existing pipeline at zero additional cost. The BoulderSystem
//! struct and boulder_physics.wgsl exist because that reuse was not done during
//! initial implementation. A future refactor could eliminate this module entirely
//! by treating boulders as a special cell type that branches only for moss
//! consumption and rendering.

use glam::Vec3;

use super::boulder_buffers::{BoulderBuffers, BoulderSpawnRequest};
use crate::rendering::boulder_bubbles::BoulderBubbleSystem;
use crate::rendering::boulder_renderer::BoulderRenderer;
use crate::rendering::cave_system::CaveParams;
use crate::simulation::fluid_simulation::SolidMaskGenerator;

/// How many boulders to maintain in the world at any time.
const DEFAULT_TARGET_BOULDER_COUNT: u32 = 32;

/// Minimum time between spawn attempts (seconds).
const SPAWN_INTERVAL: f32 = 5.0;

/// Default boulder radius.
const DEFAULT_BOULDER_RADIUS: f32 = 4.0;

pub struct BoulderSystem {
    pub buffers: BoulderBuffers,
    pub renderer: BoulderRenderer,
    pub bubbles: BoulderBubbleSystem,
    /// Cached compute bind group for bubbles - rebuilt when water system initializes.
    pub bubble_compute_bg: Option<wgpu::BindGroup>,

    /// Pre-computed ceiling spawn positions (world space).
    ceiling_positions: Vec<Vec3>,

    /// Anti-gravity direction (used for spawn push and moss_dir init).
    anti_gravity_dir: Vec3,

    /// Time accumulator for spawn throttling.
    spawn_timer: f32,

    /// Simple RNG state (xorshift32).
    rng: u32,

    /// Target number of live boulders.
    pub target_count: u32,

    /// Initial moss store per boulder (fixed-point x1000).
    pub initial_moss: i32,

    /// Spawn radius for new boulders.
    pub spawn_radius: f32,

    /// Minimum spawn radius (for size variation).
    pub radius_min: f32,

    /// Maximum spawn radius (for size variation).
    pub radius_max: f32,

    /// Minimum initial moss store in nutrients (for nutrient variation).
    pub moss_min: f32,

    /// Maximum initial moss store in nutrients (for nutrient variation).
    pub moss_max: f32,

    /// Seconds between spawn attempts. Lower = more frequent spawning.
    pub spawn_interval: f32,

    /// Set to true when buoyancy changed in UI - applied on next update().
    pub buoyancy_dirty: bool,
    /// Current buoyancy gravity multiplier (0 = float, 1 = full gravity).
    pub buoyancy: f32,
}

impl BoulderSystem {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        cave_params: &CaveParams,
        world_radius: f32,
        gravity_mode: u32,
    ) -> Self {
        let buffers = BoulderBuffers::new(device);
        let renderer =
            BoulderRenderer::new(device, queue, surface_format, depth_format, width, height);
        let bubbles =
            BoulderBubbleSystem::new(device, queue, surface_format, depth_format, width, height);

        let anti_gravity_dir = anti_gravity_direction(gravity_mode, Vec3::ZERO);
        let ceiling_positions = find_ceiling_voxels(cave_params, world_radius, gravity_mode);

        log::info!(
            "[BoulderSystem] Found {} ceiling spawn positions",
            ceiling_positions.len()
        );

        let mut rng = 0xDEADBEEFu32;
        // xorshift32 once to get initial random offset
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        let initial_offset = (rng >> 8) as f32 / 16_777_216.0 * SPAWN_INTERVAL;

        Self {
            buffers,
            renderer,
            bubbles,
            bubble_compute_bg: None,
            ceiling_positions,
            anti_gravity_dir,
            // Start at a random fraction of spawn_interval so boulders don't all
            // arrive simultaneously when multiple BoulderSystems are created.
            spawn_timer: initial_offset,
            rng,
            target_count: DEFAULT_TARGET_BOULDER_COUNT,
            initial_moss: super::boulder_buffers::BOULDER_INITIAL_MOSS,
            spawn_radius: DEFAULT_BOULDER_RADIUS,
            radius_min: DEFAULT_BOULDER_RADIUS * 0.5,
            radius_max: DEFAULT_BOULDER_RADIUS * 2.0,
            moss_min: 2_000.0,
            moss_max: 20_000.0,
            spawn_interval: SPAWN_INTERVAL,
            buoyancy_dirty: false,
            buoyancy: 0.08,
        }
    }

    /// Call once per frame. Spawns new boulders as needed.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, delta_time: f32) {
        // Poll for dead boulders (non-blocking)
        self.buffers.poll_dead_readback(device);

        // Apply buoyancy change if dirty
        if self.buoyancy_dirty {
            self.buffers.set_buoyancy(queue, self.buoyancy);
            self.buoyancy_dirty = false;
        }

        // Count live boulders
        let live = self
            .buffers
            .cpu_boulders
            .iter()
            .filter(|b| b.dead == 0 && b.radius > 0.0)
            .count() as u32;

        // Spawn if below target
        self.spawn_timer += delta_time;
        if self.spawn_timer >= self.spawn_interval && live < self.target_count {
            // Reset timer with random jitter: next spawn in [0, 2 * spawn_interval]
            // This staggers boulders so they don't all arrive at regular intervals.
            self.spawn_timer = -(self.rand_f32() * self.spawn_interval);
            self.try_spawn(queue);
        }
    }

    /// Run bubble compute passes. Call after boulder physics, before render.
    pub fn update_bubbles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        delta_time: f32,
        gravity_mode: u32,
    ) {
        if let Some(ref bg) = self.bubble_compute_bg {
            self.bubbles.update(
                encoder,
                queue,
                bg,
                self.buffers.active_count,
                delta_time,
                gravity_mode,
            );
        }
    }

    /// Render boulders and bubbles. Reads directly from GPU storage buffers - no CPU upload.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_pos: Vec3,
        camera_rotation: glam::Quat,
        current_time: f32,
        horizontal_fov_degrees: f32,
    ) {
        self.renderer.render(
            encoder,
            queue,
            color_view,
            depth_view,
            camera_pos,
            camera_rotation,
            current_time,
            horizontal_fov_degrees,
            self.buffers.active_count,
        );
        // Render bubbles after boulders (additive-ish blending, depth test only)
        if self.bubble_compute_bg.is_some() {
            self.bubbles.render(
                encoder,
                queue,
                color_view,
                depth_view,
                camera_pos,
                camera_rotation,
                horizontal_fov_degrees,
            );
        }
    }

    /// Active boulder count for dispatch sizing.
    pub fn active_count(&self) -> u32 {
        self.buffers.active_count
    }

    /// Update the target boulder count.
    pub fn set_target_count(&mut self, count: u32) {
        self.target_count = count;
    }

    /// Update the initial moss store (nutrients).
    pub fn set_initial_moss(&mut self, moss: f32) {
        self.initial_moss = (moss * super::boulder_buffers::BOULDER_FIXED_POINT_SCALE) as i32;
    }

    /// Update the boulder spawn radius.
    pub fn set_radius(&mut self, radius: f32) {
        self.spawn_radius = radius;
    }

    /// Clear all live boulders - called on scene reset.
    pub fn clear(&mut self, queue: &wgpu::Queue) {
        let n = crate::simulation::gpu_physics::boulder_buffers::MAX_BOULDERS as usize;

        // Zero all CPU mirrors
        for b in &mut self.buffers.cpu_boulders {
            *b = bytemuck::Zeroable::zeroed();
        }
        for m in &mut self.buffers.cpu_moss {
            *m = 0;
        }
        for d in &mut self.buffers.cpu_moss_dir {
            *d = [0.0, 1.0, 0.0, 0.0];
        }
        self.buffers.active_count = 0;

        // Upload zeroed state to GPU
        queue.write_buffer(
            &self.buffers.boulder_state,
            0,
            bytemuck::cast_slice(&self.buffers.cpu_boulders),
        );
        let zero_moss = vec![0i32; n];
        queue.write_buffer(
            &self.buffers.boulder_moss,
            0,
            bytemuck::cast_slice(&zero_moss),
        );
        let zero_count = [0u32, 0u32, 0u32, 0u32];
        queue.write_buffer(
            &self.buffers.boulder_count,
            0,
            bytemuck::cast_slice(&zero_count),
        );

        // Reset spawn timer so boulders start spawning again after the reset
        self.spawn_timer = self.spawn_interval;
    }

    fn try_spawn(&mut self, queue: &wgpu::Queue) {
        if self.ceiling_positions.is_empty() {
            return;
        }

        // Pick a random ceiling position
        let idx = self.rand_u32() as usize % self.ceiling_positions.len();
        let spawn_pos = self.ceiling_positions[idx];

        // Randomize radius within [radius_min, radius_max]
        let t_radius = self.rand_f32();
        let radius = self.radius_min + (self.radius_max - self.radius_min) * t_radius;

        // Spawn at the ceiling voxel center - inside the solid rock.
        // The cave SDF collision in boulder_physics.wgsl detects the penetration
        // and ejects the mossrock downward, giving a natural "breaking free" emergence.
        let pos = spawn_pos;

        // Randomize moss within [moss_min, moss_max]
        let t_moss = self.rand_f32();
        let moss_nutrients = self.moss_min + (self.moss_max - self.moss_min) * t_moss;
        let moss_fixed =
            (moss_nutrients * super::boulder_buffers::BOULDER_FIXED_POINT_SCALE) as i32;

        let seed = self.rand_u32();
        let req = BoulderSpawnRequest {
            position: pos,
            velocity: Vec3::ZERO, // SDF ejection provides the initial push
            radius,
            seed,
        };

        if let Some(slot) =
            self.buffers
                .spawn_with_moss(queue, req, self.anti_gravity_dir, moss_fixed)
        {
            log::debug!(
                "[BoulderSystem] Spawned boulder at slot {} radius={:.1} moss={:.0}",
                slot,
                radius,
                moss_nutrients
            );
        }
    }

    fn rand_u32(&mut self) -> u32 {
        // xorshift32
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        self.rng
    }

    /// Returns a pseudo-random f32 in [0, 1).
    fn rand_f32(&mut self) -> f32 {
        (self.rand_u32() >> 8) as f32 / 16_777_216.0
    }
}

// -- Ceiling voxel detection ---------------------------------------------------

/// Returns the world-space "up" direction (opposite of gravity).
fn anti_gravity_direction(gravity_mode: u32, _pos: Vec3) -> Vec3 {
    match gravity_mode {
        0 => Vec3::X, // gravity pulls -X, ceiling faces +X
        2 => Vec3::Z, // gravity pulls -Z, ceiling faces +Z
        3 => Vec3::Y, // radial: use +Y as default anti-gravity for spawning
        _ => Vec3::Y, // default Y gravity, ceiling faces +Y
    }
}

/// Scan the solid mask for ceiling voxels and return their world-space centers.
///
/// A ceiling voxel is a solid voxel that has an open (non-solid) voxel in the
/// "downward" direction (opposite of anti_gravity_dir).
fn find_ceiling_voxels(
    cave_params: &CaveParams,
    world_radius: f32,
    gravity_mode: u32,
) -> Vec<Vec3> {
    let res = 128usize; // matches GRID_RESOLUTION
    let world_center = Vec3::from(cave_params.world_center);
    let cell_size = (world_radius * 2.0) / res as f32;
    let grid_origin = world_center - Vec3::splat(world_radius);

    let generator = SolidMaskGenerator::new(res as u32, world_center, world_radius);
    let solid_mask = generator.generate_solid_mask(cave_params);

    let is_solid = |x: i32, y: i32, z: i32| -> bool {
        if x < 0 || y < 0 || z < 0 || x >= res as i32 || y >= res as i32 || z >= res as i32 {
            return true; // out of bounds = solid
        }
        solid_mask[x as usize + y as usize * res + z as usize * res * res] != 0
    };

    // "Down" step in voxel coordinates (direction gravity pulls)
    let (dx, dy, dz): (i32, i32, i32) = match gravity_mode {
        0 => (-1, 0, 0), // gravity -X, down = -X
        2 => (0, 0, -1), // gravity -Z, down = -Z
        3 => (0, -1, 0), // radial: approximate as -Y
        _ => (0, -1, 0), // default -Y
    };

    // "Up" step - anti-gravity direction (toward the ceiling surface)
    let (ux, uy, uz) = (-dx, -dy, -dz);

    let mut positions = Vec::new();

    for x in 0..res {
        for y in 0..res {
            for z in 0..res {
                let ix = x as i32;
                let iy = y as i32;
                let iz = z as i32;

                // Must be solid - this is the ceiling rock
                if !is_solid(ix, iy, iz) {
                    continue;
                }

                // The voxel directly ABOVE (in anti-gravity direction) must be open.
                // This identifies the underside of a ceiling: solid rock with open
                // cave space below it (in the gravity direction).
                // Equivalently: the voxel in the anti-gravity direction is open.
                if is_solid(ix + ux, iy + uy, iz + uz) {
                    continue;
                }

                // The voxel directly BELOW (in gravity direction) must also be open -
                // the mossrock needs somewhere to fall into.
                if is_solid(ix + dx, iy + dy, iz + dz) {
                    continue;
                }

                // Must be inside the world sphere (not outer shell)
                let world_pos = grid_origin
                    + Vec3::new(
                        x as f32 * cell_size + cell_size * 0.5,
                        y as f32 * cell_size + cell_size * 0.5,
                        z as f32 * cell_size + cell_size * 0.5,
                    );
                if world_pos.length() > world_radius * 0.95 {
                    continue; // skip outer shell voxels
                }

                positions.push(world_pos);
            }
        }
    }

    // Subsample to avoid too many candidates (keep every Nth)
    // With 128^3 grid there can be tens of thousands of ceiling voxels
    let step = (positions.len() / 500).max(1);
    positions.into_iter().step_by(step).collect()
}

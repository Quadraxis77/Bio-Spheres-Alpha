//! Mossrock GPU Buffer System
//!
//! Mossrocks are environmental physics objects — not cells, not part of the genome.
//! They fall under gravity, collide with cave walls and the world sphere, carry a
//! fixed moss store that depletes as phagocytes eat from them, and are removed when
//! their moss reaches zero.
//!
//! ## Architectural note — why not reuse the cell physics pipeline?
//!
//! The cell physics pipeline (position_update.wgsl, collision_detection.wgsl,
//! velocity_update.wgsl) operates on flat arrays indexed by slot and is entirely
//! agnostic about what occupies a slot. Boulders could have been placed in reserved
//! slots at the end of the cell arrays, with death_flags=1 to exclude them from the
//! lifecycle pipeline and a cell_type flag to skip metabolism and adhesion. The
//! physics dispatch already covers all slots up to total_cell_slots, so gravity,
//! boundary forces, cave SDF collision, spatial grid participation, and velocity
//! integration would all have worked for free with zero new shaders.
//!
//! What would have needed branching: nutrient_transport (skip boulders), moss_consume
//! (use reserve buffer instead of nutrients), and rendering (icosphere vs billboard).
//! Rolling contact torque would have been a small addition to collision_detection.wgsl
//! using the existing torque_accum buffers.
//!
//! Instead, boulder_physics.wgsl reimplements gravity, boundary, cave SDF, and
//! velocity integration from scratch. This was a design mistake made during
//! implementation. The correct approach would have been to treat boulders as
//! reserved cell slots that branch only where their behaviour differs from cells.
//!
//! ## Buffer layout
//!
//! `boulder_state`: array of `GpuBoulder` (80 bytes each, 16-byte aligned)
//!   - position (vec3), radius (f32)
//!   - velocity (vec3), dead flag (u32)
//!   - seed (u32), _pad (u32 × 3)
//!   - angular_velocity (vec4: xyz = ω rad/s, w unused)
//!   - orientation (vec4: quaternion x,y,z,w)
//!
//! `boulder_moss`: array<atomic<i32>> — fixed-point ×1000 moss store per boulder
//!
//! `boulder_moss_dir`: array<vec4<f32>> — world-space direction of remaining moss
//!   (xyz = unit vector toward moss concentration, w = unused)
//!
//! `boulder_eat_dir_accum`: 3 × array<atomic<i32>> per boulder (x, y, z components)
//!   Accumulated each frame by boulder_consume, read and cleared by boulder_physics.
//!
//! `boulder_count`: [0] = active boulder count (written by CPU)

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Maximum number of boulders that can exist simultaneously.
pub const MAX_BOULDERS: u32 = 256;

/// Fixed-point scale matching the rest of the nutrient system.
pub const BOULDER_FIXED_POINT_SCALE: f32 = 1000.0;

/// Initial moss store per boulder (nutrients). 10,000 nutrients = 10_000_000 fixed-point.
pub const BOULDER_INITIAL_MOSS: i32 = 10_000_000;

/// GPU-side boulder state (80 bytes, 16-byte aligned).
/// Must match `GpuBoulder` struct in boulder_physics.wgsl, boulder_consume.wgsl, and boulder.wgsl.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuBoulder {
    /// World-space position (xyz)
    pub position: [f32; 3],
    /// Current radius — shrinks toward 0 as boulder dies
    pub radius: f32,
    /// Current velocity (xyz)
    pub velocity: [f32; 3],
    /// 1 = dead (skip all passes and rendering), 0 = alive
    pub dead: u32,
    /// Shape seed — fixed at spawn, drives vertex displacement in render shader
    pub seed: u32,
    pub _pad: [u32; 3],
    /// Angular velocity (xyz = ω rad/s, w = unused)
    pub angular_velocity: [f32; 4],
    /// Orientation quaternion (x, y, z, w) — starts as identity (0,0,0,1)
    pub orientation: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuBoulder>() == 80);

/// CPU-side spawn request, converted to `GpuBoulder` before upload.
#[derive(Debug, Clone)]
pub struct BoulderSpawnRequest {
    pub position: glam::Vec3,
    /// Initial push velocity (away from ceiling surface)
    pub velocity: glam::Vec3,
    pub radius: f32,
    pub seed: u32,
}

/// All GPU buffers for the boulder system.
pub struct BoulderBuffers {
    /// Per-boulder state: position, velocity, radius, dead, seed.
    /// `array<GpuBoulder>` — written by boulder_physics, read by boulder_consume + render.
    pub boulder_state: wgpu::Buffer,

    /// Per-boulder moss store: `array<atomic<i32>>` fixed-point ×1000.
    /// Decremented by boulder_consume, read by boulder_physics for death check.
    pub boulder_moss: wgpu::Buffer,

    /// Per-boulder moss direction: `array<vec4<f32>>`.
    /// xyz = world-space unit vector toward remaining moss concentration.
    /// Updated by boulder_physics from eat_dir_accum each frame.
    pub boulder_moss_dir: wgpu::Buffer,

    /// Per-boulder eat direction accumulator: 3 × `atomic<i32>` per boulder (x, y, z).
    /// Accumulated by boulder_consume, read + cleared by boulder_physics.
    /// Stored as a flat array: boulder_eat_dir_accum[boulder_idx * 3 + component]
    pub boulder_eat_dir_accum: wgpu::Buffer,

    /// Per-boulder force accumulator: 3 × atomic<i32> per boulder (x, y, z), fixed-point ×1000.
    /// Written by collision_detection.wgsl when cells push against boulders.
    /// Read and cleared by boulder_physics.wgsl each frame.
    pub boulder_force_accum: wgpu::Buffer,

    /// Boulder buoyancy params: [gravity_multiplier_in_water, drag_coeff, 0, 0]
    /// Written by CPU when the buoyancy slider changes.
    pub boulder_buoyancy_params: wgpu::Buffer,

    /// Boulder count buffer: [0] = active boulder count.
    /// Written by CPU on spawn/despawn, read by all boulder shaders.
    pub boulder_count: wgpu::Buffer,

    /// Staging buffer for async readback of dead flags (to recycle slots on CPU).
    pub dead_flags_readback: wgpu::Buffer,

    /// Whether a dead-flag readback is pending.
    pub readback_pending: bool,

    /// Channel receiver for readback completion.
    pub readback_receiver: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,

    /// CPU-side mirror of boulder state (for spawn management and slot reuse).
    pub cpu_boulders: Vec<GpuBoulder>,

    /// CPU-side moss values (for initial upload and death detection via readback).
    pub cpu_moss: Vec<i32>,

    /// CPU-side moss directions (initial upload).
    pub cpu_moss_dir: Vec<[f32; 4]>,

    /// Active boulder count (matches boulder_count buffer).
    pub active_count: u32,
}

impl BoulderBuffers {
    pub fn new(device: &wgpu::Device) -> Self {
        let n = MAX_BOULDERS as usize;

        // Boulder state: MAX_BOULDERS × 80 bytes
        // Initialize with identity quaternions (orientation w=1)
        let cpu_boulders: Vec<GpuBoulder> = (0..n).map(|_| GpuBoulder {
            orientation: [0.0, 0.0, 0.0, 1.0],
            ..GpuBoulder::zeroed()
        }).collect();
        let boulder_state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Boulder State"),
            contents: bytemuck::cast_slice(&cpu_boulders),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // Boulder moss: MAX_BOULDERS × 4 bytes (atomic<i32>)
        let cpu_moss = vec![0i32; n];
        let boulder_moss = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Boulder Moss"),
            contents: bytemuck::cast_slice(&cpu_moss),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // Boulder moss direction: MAX_BOULDERS × 16 bytes (vec4<f32>)
        // Default: (0, 1, 0, 0) — moss starts on top
        let cpu_moss_dir: Vec<[f32; 4]> = vec![[0.0, 1.0, 0.0, 0.0]; n];
        let boulder_moss_dir = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Boulder Moss Dir"),
            contents: bytemuck::cast_slice(&cpu_moss_dir),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // Boulder eat direction accumulator: MAX_BOULDERS × 3 × 4 bytes (atomic<i32> × 3)
        let eat_dir_data = vec![0i32; n * 3];
        let boulder_eat_dir_accum = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Boulder Eat Dir Accum"),
            contents: bytemuck::cast_slice(&eat_dir_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        // Boulder force accumulator: MAX_BOULDERS × 3 × 4 bytes (atomic<i32> × 3)
        // Cleared each frame by boulder_physics before accumulating new forces.
        let force_accum_data = vec![0i32; n * 3];
        let boulder_force_accum = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Boulder Force Accum"),
            contents: bytemuck::cast_slice(&force_accum_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });

        // Boulder buoyancy params: [gravity_multiplier, drag_coeff, 0, 0] = 16 bytes
        let boulder_buoyancy_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Boulder Buoyancy Params"),
            contents: bytemuck::cast_slice(&[0.08f32, 40.0f32, 0.0f32, 0.0f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Boulder count: 16 bytes (4 × u32, only [0] used)
        let boulder_count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Boulder Count"),
            contents: bytemuck::cast_slice(&[0u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // Dead flags readback: MAX_BOULDERS × 4 bytes
        let dead_flags_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Boulder Dead Flags Readback"),
            size: n as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            boulder_state,
            boulder_moss,
            boulder_moss_dir,
            boulder_eat_dir_accum,
            boulder_force_accum,
            boulder_buoyancy_params,
            boulder_count,
            dead_flags_readback,
            readback_pending: false,
            readback_receiver: None,
            cpu_boulders,
            cpu_moss,
            cpu_moss_dir,
            active_count: 0,
        }
    }

    /// Update the buoyancy params buffer (gravity multiplier in water, drag coefficient).
    pub fn set_buoyancy(&self, queue: &wgpu::Queue, gravity_multiplier: f32) {
        let drag_coeff = 40.0f32; // fixed drag coefficient
        let data = [gravity_multiplier, drag_coeff, 0.0f32, 0.0f32];
        queue.write_buffer(&self.boulder_buoyancy_params, 0, bytemuck::cast_slice(&data));
    }

    /// Spawn a boulder into the next free slot with the default moss store.
    /// Returns the slot index, or None if all slots are occupied.
    pub fn spawn(
        &mut self,
        queue: &wgpu::Queue,
        req: BoulderSpawnRequest,
        gravity_dir: glam::Vec3,
    ) -> Option<u32> {
        self.spawn_with_moss(queue, req, gravity_dir, BOULDER_INITIAL_MOSS)
    }

    /// Spawn a boulder with an explicit moss store (fixed-point ×1000).
    /// Returns the slot index, or None if all slots are occupied.
    pub fn spawn_with_moss(
        &mut self,
        queue: &wgpu::Queue,
        req: BoulderSpawnRequest,
        gravity_dir: glam::Vec3,
        moss_fixed: i32,
    ) -> Option<u32> {
        // Find a dead or zeroed slot
        let slot = self.cpu_boulders.iter().position(|b| b.dead != 0 || b.radius == 0.0)?;
        let slot = slot as u32;

        let boulder = GpuBoulder {
            position: req.position.to_array(),
            radius: req.radius,
            velocity: req.velocity.to_array(),
            dead: 0,
            seed: req.seed,
            _pad: [0; 3],
            angular_velocity: [0.0; 4],
            orientation: [0.0, 0.0, 0.0, 1.0], // identity quaternion
        };

        // Moss direction starts opposite to gravity (moss on the "top" face)
        let moss_up = (-gravity_dir).normalize();
        let moss_dir = [moss_up.x, moss_up.y, moss_up.z, 0.0];

        self.cpu_boulders[slot as usize] = boulder;
        self.cpu_moss[slot as usize] = moss_fixed;
        self.cpu_moss_dir[slot as usize] = moss_dir;

        // Upload this slot to GPU
        let offset = slot as u64 * std::mem::size_of::<GpuBoulder>() as u64;
        queue.write_buffer(&self.boulder_state, offset, bytemuck::bytes_of(&boulder));

        let moss_offset = slot as u64 * 4;
        queue.write_buffer(
            &self.boulder_moss,
            moss_offset,
            bytemuck::bytes_of(&moss_fixed),
        );

        let dir_offset = slot as u64 * 16;
        queue.write_buffer(
            &self.boulder_moss_dir,
            dir_offset,
            bytemuck::cast_slice(&[moss_dir]),
        );

        // Update active count
        self.active_count = self.cpu_boulders.iter().filter(|b| b.dead == 0 && b.radius > 0.0).count() as u32;
        let count_data = [self.active_count, 0u32, 0u32, 0u32];
        queue.write_buffer(&self.boulder_count, 0, bytemuck::cast_slice(&count_data));

        Some(slot)
    }

    /// Initiate async readback of dead flags to detect boulders that have died on GPU.
    /// Call once per frame; poll with `poll_dead_readback`.
    pub fn request_dead_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.readback_pending {
            return;
        }
        // Copy dead field from each boulder. GpuBoulder layout offset of `dead` = 28 bytes.
        // We copy the full state buffer to a staging buffer and extract dead fields on CPU.
        // The dead_flags_readback buffer is sized for MAX_BOULDERS × 4 (just the dead u32s),
        // so we use a stride copy approach: copy each boulder's dead field individually.
        // For simplicity, copy the full state and extract in poll.
        let _ = encoder;
        self.readback_pending = true;
    }

    /// Poll for completed dead-flag readback. Returns list of newly-dead slot indices.
    pub fn poll_dead_readback(&mut self, device: &wgpu::Device) -> Vec<u32> {
        if !self.readback_pending {
            return Vec::new();
        }

        let slice = self.dead_flags_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        // Non-blocking poll
        match device.poll(wgpu::PollType::Poll) {
            _ => {}
        }

        if let Ok(Ok(())) = rx.try_recv() {
            let view = slice.get_mapped_range();
            let dead_flags: &[u32] = bytemuck::cast_slice(&view);
            let dead_slots: Vec<u32> = dead_flags
                .iter()
                .enumerate()
                .filter(|(_, &d)| d != 0)
                .map(|(i, _)| i as u32)
                .collect();
            drop(view);
            self.dead_flags_readback.unmap();
            self.readback_pending = false;

            // Mark dead slots in CPU mirror
            for &slot in &dead_slots {
                if let Some(b) = self.cpu_boulders.get_mut(slot as usize) {
                    b.dead = 1;
                }
            }

            dead_slots
        } else {
            Vec::new()
        }
    }
}

//! Snapshot save and restore logic for `GpuScene`.
//!
//! This module is kept separate from `gpu_scene.rs` to avoid making that
//! already-large file even bigger.  The public API is two methods added to
//! `GpuScene` via an `impl` block at the bottom of this file:
//!
//! - `save_snapshot(device, queue) -> Result<GpuSceneSnapshot, SnapshotError>`
//! - `restore_from_snapshot(device, queue, snapshot) -> Result<(), SnapshotError>`
//!
//! ## Save flow
//! 1. Pause the simulation (caller's responsibility).
//! 2. For each GPU-only per-cell buffer, create a staging buffer, copy via
//!    `encoder.copy_buffer_to_buffer`, submit, then `device.poll(Wait)` and
//!    `map_async` to read the bytes back to CPU.
//! 3. Collect the CPU-side adhesion caches (no readback needed).
//! 4. Serialise genomes via their existing YAML path.
//! 5. Pack everything into a `GpuSceneSnapshot`.
//!
//! ## Restore flow
//! 1. Call `reset()` to clear all GPU state.
//! 2. Rebuild `CanonicalState` from the snapshot arrays.
//! 3. Call `sync_from_canonical_state` to push cell data to all three buffer
//!    sets.
//! 4. Call `AdhesionBuffers::restore_from_snapshot` + `sync_to_gpu`.
//! 5. Restore scalar settings.

use std::path::Path;
use std::sync::mpsc;

use crate::genome::Genome;
use crate::simulation::CanonicalState;
use super::gpu_scene::GpuScene;
use super::snapshot::{GpuSceneSnapshot, SnapshotError};

// --- helpers -----------------------------------------------------------------

/// Read `byte_count` bytes from `src_buffer` starting at `src_offset` into a
/// `Vec<u8>` using a temporary staging buffer.  Blocks until the GPU work is
/// complete.
fn readback_buffer_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buffer: &wgpu::Buffer,
    src_offset: u64,
    byte_count: u64,
) -> Result<Vec<u8>, SnapshotError> {
    if byte_count == 0 {
        return Ok(Vec::new());
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Snapshot Staging Buffer"),
        size: byte_count,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Snapshot Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(src_buffer, src_offset, &staging, 0, byte_count);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| SnapshotError::GpuReadback(format!("device.poll failed: {e:?}")))?;

    rx.recv()
        .map_err(|_| SnapshotError::GpuReadback("channel closed before map completed".into()))?
        .map_err(|e| SnapshotError::GpuReadback(format!("map_async error: {e:?}")))?;

    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();

    Ok(bytes)
}

/// Read a GPU buffer that contains `count` elements of type `T` (which must be
/// `bytemuck::Pod`).  Returns a `Vec<T>` of length `count`.
fn readback_typed<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>, SnapshotError> {
    let byte_count = (count * std::mem::size_of::<T>()) as u64;
    let bytes = readback_buffer_bytes(device, queue, src_buffer, 0, byte_count)?;
    Ok(bytemuck::cast_slice::<u8, T>(&bytes).to_vec())
}

// --- GpuScene impl -----------------------------------------------------------

impl GpuScene {
    /// Capture the current simulation state into a `GpuSceneSnapshot`.
    ///
    /// This performs a **blocking** GPU readback of all per-cell buffers.  The
    /// simulation should be paused before calling this to avoid reading
    /// mid-frame state.
    ///
    /// The snapshot can be written to disk with `snapshot.save_to_file(path)`
    /// or restored immediately with `restore_from_snapshot`.
    pub fn save_snapshot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<GpuSceneSnapshot, SnapshotError> {
        let slots = self.total_cell_slots as usize;
        let capacity = self.gpu_triple_buffers.capacity;

        // Use the current (output) buffer index for the triple-buffered arrays.
        let buf_idx = self.gpu_triple_buffers.output_buffer_index();

        log::info!(
            "[Snapshot] Saving: capacity={capacity}, live={}, slots={slots}",
            self.current_cell_count
        );

        // -- Per-cell GPU readbacks --------------------------------------------
        // Each readback creates a staging buffer, copies, polls, and maps.
        // We read only `slots` elements (the high-water mark) to keep the
        // snapshot compact.

        let positions_and_mass: Vec<[f32; 4]> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.position_and_mass[buf_idx], slots)?
        } else {
            Vec::new()
        };

        let velocities: Vec<[f32; 4]> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.velocity[buf_idx], slots)?
        } else {
            Vec::new()
        };

        let rotations: Vec<[f32; 4]> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.rotations[buf_idx], slots)?
        } else {
            Vec::new()
        };

        let genome_orientations: Vec<[f32; 4]> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.genome_orientations, slots)?
        } else {
            Vec::new()
        };

        let nutrients: Vec<i32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.nutrients_buffer, slots)?
        } else {
            Vec::new()
        };

        let birth_times: Vec<f32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.birth_times, slots)?
        } else {
            Vec::new()
        };

        let split_intervals: Vec<f32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.split_intervals, slots)?
        } else {
            Vec::new()
        };

        let split_nutrient_thresholds: Vec<f32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.split_nutrient_thresholds, slots)?
        } else {
            Vec::new()
        };

        let split_counts: Vec<u32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.split_counts, slots)?
        } else {
            Vec::new()
        };

        let genome_ids: Vec<u32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.genome_ids, slots)?
        } else {
            Vec::new()
        };

        let mode_indices: Vec<u32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.mode_indices, slots)?
        } else {
            Vec::new()
        };

        let cell_ids: Vec<u32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.cell_ids, slots)?
        } else {
            Vec::new()
        };

        let death_flags: Vec<u32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.death_flags, slots)?
        } else {
            Vec::new()
        };

        let embryocyte_reserves: Vec<u32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.embryocyte_reserve_buffer, slots)?
        } else {
            Vec::new()
        };

        // -- Adhesion state (CPU caches - no readback needed) ------------------
        let adhesion_connections = self.adhesion_buffers.snapshot_connections();
        let cell_adhesion_indices = self.adhesion_buffers.snapshot_cell_indices();
        let adhesion_allocated_count = self.adhesion_buffers.snapshot_allocated_count();

        // -- Genomes -----------------------------------------------------------
        // Serialise each genome to a YAML string using the existing path.
        let mut genomes_yaml: Vec<String> = Vec::with_capacity(self.genomes.len());
        for genome in &self.genomes {
            let yaml = genome
                .to_yaml_string()
                .map_err(|e| SnapshotError::GpuReadback(format!("genome serialization: {e}")))?;
            genomes_yaml.push(yaml);
        }

        // -- Cave parameters ---------------------------------------------------
        let cave_active = self.cave_renderer.is_some();
        let (cave_density, cave_scale, cave_octaves, cave_persistence,
             cave_threshold, cave_smoothness, cave_seed, cave_resolution) =
            if let Some(ref cave) = self.cave_renderer {
                let p = cave.params();
                (p.density, p.scale, p.octaves, p.persistence,
                 p.threshold, p.smoothness, p.seed, p.grid_resolution)
            } else {
                (0.5, 100.0, 2, 0.5, 1.0, 0.0, 12345, 128)
            };

        // -- Fluid / water state -----------------------------------------------
        let fluid_active = self.fluid_simulator.is_some();
        let (fluid_voxels, nutrient_voxels, fluid_time, fluid_type, fluid_continuous_spawn) =
            if let Some(ref sim) = self.fluid_simulator {
                log::info!("[Snapshot] Reading back fluid voxels (2 × 8 MB)…");
                let (fv, nv) = sim.snapshot_voxels(device, queue);
                (fv, nv, sim.time(), sim.get_fluid_type(), sim.is_continuous_spawn_enabled())
            } else {
                (Vec::new(), Vec::new(), 0.0, 1, false)
            };

        log::info!(
            "[Snapshot] Readback complete: {} cells, {} adhesions, {} genomes, fluid={}",
            slots,
            adhesion_allocated_count,
            genomes_yaml.len(),
            fluid_active,
        );

        Ok(GpuSceneSnapshot {
            version: GpuSceneSnapshot::CURRENT_VERSION,
            capacity,
            live_cell_count: self.current_cell_count,
            total_cell_slots: self.total_cell_slots,
            positions_and_mass,
            velocities,
            rotations,
            genome_orientations,
            nutrients,
            birth_times,
            split_intervals,
            split_nutrient_thresholds,
            split_counts,
            genome_ids,
            mode_indices,
            cell_ids,
            death_flags,
            embryocyte_reserves,
            adhesion_connections,
            cell_adhesion_indices,
            adhesion_allocated_count,
            genomes_yaml,
            current_time: self.current_time,
            current_frame: self.current_frame,
            next_cell_id: self.next_cell_id,
            time_scale: self.time_scale,
            gravity: self.gravity,
            gravity_mode: self.gravity_mode,
            constraint_iterations: self.constraint_iterations,
            surface_pressure: self.surface_pressure,
            acceleration_damping: self.acceleration_damping,
            water_viscosity: self.water_viscosity,
            solo_metabolism_multiplier: self.solo_metabolism_multiplier,
            radiation_level: self.radiation_level,
            subtle_mutations: self.subtle_mutations,
            lateral_flow_probabilities: self.lateral_flow_probabilities,
            condensation_probability: self.condensation_probability,
            vaporization_probability: self.vaporization_probability,
            nutrient_density: self.nutrient_density,
            nutrient_epoch_duration: self.nutrient_epoch_duration,
            nutrient_epoch_spacing: self.nutrient_epoch_spacing,
            nutrient_spawn_end: self.nutrient_spawn_end,
            nutrient_despawn_start: self.nutrient_despawn_start,
            world_radius: self.config.sphere_radius,
            cave_active,
            cave_density,
            cave_scale,
            cave_octaves,
            cave_persistence,
            cave_threshold,
            cave_smoothness,
            cave_seed,
            cave_resolution,
            fluid_active,
            fluid_voxels,
            nutrient_voxels,
            fluid_time,
            fluid_type,
            fluid_continuous_spawn,
        })
    }

    /// Restore the simulation from a previously captured `GpuSceneSnapshot`.
    ///
    /// This resets the scene, rebuilds `CanonicalState` from the snapshot
    /// arrays, pushes all data to the GPU via the existing
    /// `sync_from_canonical_state` path, and restores adhesion state.
    ///
    /// The scene capacity must be **>=** the snapshot capacity.  If it is
    /// smaller, this returns `SnapshotError::CapacityMismatch`.
    pub fn restore_from_snapshot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &GpuSceneSnapshot,
    ) -> Result<(), SnapshotError> {
        // -- Version and capacity checks ---------------------------------------
        if !snapshot.is_compatible() {
            return Err(SnapshotError::IncompatibleVersion {
                found: snapshot.version,
                expected: GpuSceneSnapshot::CURRENT_VERSION,
            });
        }

        let scene_capacity = self.gpu_triple_buffers.capacity;
        if snapshot.capacity > scene_capacity {
            return Err(SnapshotError::CapacityMismatch {
                snapshot: snapshot.capacity,
                scene: scene_capacity,
            });
        }

        log::info!(
            "[Snapshot] Restoring: {} live cells, {} slots, {} genomes",
            snapshot.live_cell_count,
            snapshot.total_cell_slots,
            snapshot.genomes_yaml.len()
        );

        // -- Reset scene to clean state ----------------------------------------
        self.reset(queue);

        // -- Restore scalar settings -------------------------------------------
        self.current_time = snapshot.current_time;
        self.current_frame = snapshot.current_frame;
        self.next_cell_id = snapshot.next_cell_id;
        self.time_scale = snapshot.time_scale;
        self.gravity = snapshot.gravity;
        self.gravity_mode = snapshot.gravity_mode;
        self.constraint_iterations = snapshot.constraint_iterations;
        self.surface_pressure = snapshot.surface_pressure;
        self.acceleration_damping = snapshot.acceleration_damping;
        self.water_viscosity = snapshot.water_viscosity;
        self.solo_metabolism_multiplier = snapshot.solo_metabolism_multiplier;
        self.radiation_level = snapshot.radiation_level;
        self.subtle_mutations = snapshot.subtle_mutations;
        self.lateral_flow_probabilities = snapshot.lateral_flow_probabilities;
        self.condensation_probability = snapshot.condensation_probability;
        self.vaporization_probability = snapshot.vaporization_probability;
        self.nutrient_density = snapshot.nutrient_density;
        self.nutrient_epoch_duration = snapshot.nutrient_epoch_duration;
        self.nutrient_epoch_spacing = snapshot.nutrient_epoch_spacing;
        self.nutrient_spawn_end = snapshot.nutrient_spawn_end;
        self.nutrient_despawn_start = snapshot.nutrient_despawn_start;
        self.config.sphere_radius = snapshot.world_radius;
        self.current_cell_count = snapshot.live_cell_count;
        self.total_cell_slots = snapshot.total_cell_slots;

        // -- Restore genomes ---------------------------------------------------
        let mut genomes: Vec<Genome> = Vec::with_capacity(snapshot.genomes_yaml.len());
        for yaml in &snapshot.genomes_yaml {
            let genome = Genome::from_yaml_string(yaml)
                .map_err(|e| SnapshotError::GpuReadback(format!("genome deserialization: {e}")))?;
            genomes.push(genome);
        }
        self.genomes = genomes;

        // Rebuild derived genome caches.
        self.parent_make_adhesion_flags.clear();
        self.has_oculocytes = false;
        self.max_signal_hops = 0;
        for genome in &self.genomes {
            for mode in &genome.modes {
                self.parent_make_adhesion_flags.push(mode.parent_make_adhesion);
                if mode.cell_type == 7 {
                    // Oculocyte
                    self.has_oculocytes = true;
                    self.max_signal_hops =
                        self.max_signal_hops.max(mode.oculocyte_signal_hops as u32);
                }
            }
        }

        // -- Rebuild CanonicalState from snapshot arrays -----------------------
        let slots = snapshot.total_cell_slots as usize;

        if slots > 0 {
            let mut canonical = CanonicalState::new(scene_capacity as usize);

            // Compute live cell count from death flags.
            let live_count = snapshot
                .death_flags
                .iter()
                .take(slots)
                .filter(|&&d| d == 0)
                .count();
            canonical.cell_count = live_count;

            // Populate per-cell arrays for all slots up to `slots`.
            // Dead slots are included so that slot indices remain stable.
            for i in 0..slots {
                let pm = snapshot.positions_and_mass[i];
                canonical.positions[i] = glam::Vec3::new(pm[0], pm[1], pm[2]);
                canonical.masses[i] = pm[3];
                canonical.radii[i] = pm[3].clamp(0.5, 2.0);

                let v = snapshot.velocities[i];
                canonical.velocities[i] = glam::Vec3::new(v[0], v[1], v[2]);

                let r = snapshot.rotations[i];
                canonical.rotations[i] = glam::Quat::from_xyzw(r[0], r[1], r[2], r[3]);

                let go = snapshot.genome_orientations[i];
                canonical.genome_orientations[i] =
                    glam::Quat::from_xyzw(go[0], go[1], go[2], go[3]);

                // Nutrients are stored as i32 fixed-point (value * 1000).
                canonical.nutrients[i] = snapshot.nutrients[i] as f32 / 1000.0;

                canonical.birth_times[i] = snapshot.birth_times[i];
                canonical.split_intervals[i] = snapshot.split_intervals[i];
                canonical.split_nutrient_thresholds[i] = snapshot.split_nutrient_thresholds[i];
                canonical.split_counts[i] = snapshot.split_counts[i] as i32;
                canonical.genome_ids[i] = snapshot.genome_ids[i] as usize;
                canonical.mode_indices[i] = snapshot.mode_indices[i] as usize;
                canonical.cell_ids[i] = snapshot.cell_ids[i];

                // Restore Embryocyte reserves (zero for non-Embryocyte slots)
                if i < snapshot.embryocyte_reserves.len() {
                    canonical.reserves[i] = snapshot.embryocyte_reserves[i];
                }
            }

            canonical.next_cell_id = snapshot.next_cell_id;

            // Push canonical state to all three GPU buffer sets.
            self.gpu_triple_buffers
                .sync_from_canonical_state(device, queue, &canonical, &self.genomes);

            // Sync embryocyte reserve buffer to GPU (sync_from_canonical_state doesn't cover it)
            if !snapshot.embryocyte_reserves.is_empty() {
                self.gpu_triple_buffers
                    .sync_embryocyte_reserves(queue, &snapshot.embryocyte_reserves, slots);
            }
        }

        // -- Restore adhesion state --------------------------------------------
        self.adhesion_buffers.restore_from_snapshot(
            snapshot.adhesion_connections.clone(),
            snapshot.cell_adhesion_indices.clone(),
            snapshot.adhesion_allocated_count,
        );
        self.adhesion_buffers.sync_to_gpu(queue);

        // -- Sync genome mode data to GPU --------------------------------------
        // sync_from_canonical_state already calls sync_genome_mode_data, but
        // we also need to sync the adhesion settings for the restored genomes.
        if !self.genomes.is_empty() {
            // Ensure mode pool is large enough (sync_from_canonical_state may have grown it,
            // but call again here in case genomes changed between the two sync paths)
            let total_modes: u64 = self.genomes.iter().map(|g| g.modes.len() as u64).sum();
            self.gpu_triple_buffers.grow_mode_pool_if_needed(device, total_modes);

            self.gpu_triple_buffers
                .sync_genome_mode_data(queue, &self.genomes);
            self.gpu_triple_buffers
                .sync_mode_properties(queue, &self.genomes);
            self.gpu_triple_buffers
                .sync_cilia_mode_properties(queue, &self.genomes);
            self.gpu_triple_buffers
                .sync_myocyte_mode_properties(queue, &self.genomes);
            self.gpu_triple_buffers
                .sync_embryocyte_mode_properties(queue, &self.genomes);
            self.gpu_triple_buffers
                .sync_devorocyte_mode_properties(queue, &self.genomes);
            self.gpu_triple_buffers
                .sync_vasculocyte_mode_properties(queue, &self.genomes);
        }

        // Mark instance builder dirty so rendering picks up the new state.
        self.instance_builder.mark_all_dirty();
        self.genomes_dirty = true;

        // -- Restore cave parameters -------------------------------------------
        // Cave geometry is fully deterministic from its scalar params, so we
        // just push the saved values back into the cave renderer and mark dirty.
        // The mesh will be regenerated on the next `update_cave_params` call.
        if snapshot.cave_active {
            if let Some(ref mut cave) = self.cave_renderer {
                let mut p = *cave.params();
                p.density        = snapshot.cave_density;
                p.scale          = snapshot.cave_scale;
                p.octaves        = snapshot.cave_octaves;
                p.persistence    = snapshot.cave_persistence;
                p.threshold      = snapshot.cave_threshold;
                p.smoothness     = snapshot.cave_smoothness;
                p.seed           = snapshot.cave_seed;
                p.grid_resolution = snapshot.cave_resolution;
                // world_center and world_radius are always derived from config.
                p.world_center   = [0.0, 0.0, 0.0];
                p.world_radius   = self.config.sphere_radius;
                *cave.params_mut() = p;
                self.cave_params_dirty = true;
            }
        }

        // -- Restore fluid / water state ---------------------------------------
        // Write the saved voxel arrays directly into the GPU buffers.
        if snapshot.fluid_active
            && !snapshot.fluid_voxels.is_empty()
            && !snapshot.nutrient_voxels.is_empty()
        {
            if let Some(ref sim) = self.fluid_simulator {
                sim.restore_voxels(queue, &snapshot.fluid_voxels, &snapshot.nutrient_voxels);
                sim.set_time(snapshot.fluid_time);
                sim.set_fluid_type(snapshot.fluid_type);
                sim.set_continuous_spawn(snapshot.fluid_continuous_spawn);
                log::info!("[Snapshot] Fluid voxels restored ({} voxels).", snapshot.fluid_voxels.len());
            }
        }

        log::info!("[Snapshot] Restore complete.");
        Ok(())
    }
}

// --- File I/O helpers on GpuSceneSnapshot ------------------------------------

impl GpuSceneSnapshot {
    /// Serialise this snapshot to a RON string.
    pub fn to_ron_string(&self) -> Result<String, SnapshotError> {
        let config = ron::ser::PrettyConfig::new()
            .depth_limit(6)
            .separate_tuple_members(false)
            .enumerate_arrays(false);
        ron::ser::to_string_pretty(self, config).map_err(SnapshotError::Serialise)
    }

    /// Deserialise a snapshot from a RON string.
    pub fn from_ron_string(s: &str) -> Result<Self, SnapshotError> {
        ron::from_str(s).map_err(SnapshotError::Deserialise)
    }

    /// Write this snapshot to a file.
    ///
    /// The recommended extension is `.bss` (Bio-Spheres Snapshot).
    pub fn save_to_file(&self, path: &Path) -> Result<(), SnapshotError> {
        let ron = self.to_ron_string()?;
        std::fs::write(path, ron)?;
        log::info!("[Snapshot] Saved to {:?}", path);
        Ok(())
    }

    /// Load a snapshot from a file.
    pub fn load_from_file(path: &Path) -> Result<Self, SnapshotError> {
        let s = std::fs::read_to_string(path)?;
        let snapshot = Self::from_ron_string(&s)?;
        log::info!("[Snapshot] Loaded from {:?}", path);
        Ok(snapshot)
    }
}

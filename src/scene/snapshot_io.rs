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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::mpsc;

use image::ImageEncoder;

use super::gpu_scene::GpuScene;
use super::lineage::{LineageAdultCellSnapshot, LineageAdultSnapshot, LineagePopulationSample};
use super::snapshot::{GpuSceneSnapshot, SnapshotError};
use crate::genome::Genome;
use crate::simulation::gpu_physics::GenomeMeta;
use crate::simulation::CanonicalState;

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

fn capture_gpu_adult_snapshots(
    genomes: &[Genome],
    connections: &[crate::simulation::gpu_physics::adhesion::GpuAdhesionConnection],
    cell_indices: &[crate::simulation::gpu_physics::adhesion::CellAdhesionIndices],
    positions_and_mass: &[[f32; 4]],
    mode_indices: &[u32],
    genome_ids: &[u32],
    death_flags: &[u32],
    organism_labels: Option<&[u32]>,
    labels_reliable: bool,
    nutrients: &[i32],
    birth_times: &[f32],
    split_intervals: &[f32],
    split_thresholds: &[f32],
    current_time: f32,
    slots_used: u32,
) -> HashMap<u32, LineageAdultSnapshot> {
    const MAX_CAPTURE_CELLS: usize = 96;

    let mut groups: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, &genome_id) in genome_ids.iter().enumerate() {
        if death_flags.get(i).copied().unwrap_or(1) != 0 || genome_id == u32::MAX {
            continue;
        }
        let group_id = if labels_reliable {
            organism_labels
                .and_then(|labels| labels.get(i).copied())
                .filter(|&label| label < slots_used)
                .unwrap_or(i as u32)
        } else {
            genome_id
        };
        groups.entry((genome_id, group_id)).or_default().push(i);
    }

    let mut best_by_genome: HashMap<u32, (f32, f32, Vec<usize>)> = HashMap::new();
    for ((genome_id, _), cells) in groups {
        let readiness = cells
            .iter()
            .map(|&cell| {
                free_division_readiness(
                    cell,
                    cell_indices,
                    connections,
                    nutrients,
                    birth_times,
                    split_intervals,
                    split_thresholds,
                    current_time,
                )
            })
            .fold(0.0f32, f32::max);
        let score = readiness * 10.0 + cells.len().min(MAX_CAPTURE_CELLS) as f32 * 0.001;

        let replace = best_by_genome
            .get(&genome_id)
            .map(|(best_score, _, _)| score > *best_score)
            .unwrap_or(true);
        if replace {
            best_by_genome.insert(genome_id, (score, readiness, cells));
        }
    }

    best_by_genome
        .into_iter()
        .filter_map(|(genome_id, (_, readiness, cells))| {
            let snapshot = snapshot_from_gpu_group(
                genomes.get(genome_id as usize),
                genome_id,
                &cells,
                connections,
                positions_and_mass,
                mode_indices,
                current_time,
                readiness >= 1.0,
                MAX_CAPTURE_CELLS,
            )?;
            Some((genome_id, snapshot))
        })
        .collect()
}

fn free_division_readiness(
    cell: usize,
    cell_indices: &[crate::simulation::gpu_physics::adhesion::CellAdhesionIndices],
    connections: &[crate::simulation::gpu_physics::adhesion::GpuAdhesionConnection],
    nutrients: &[i32],
    birth_times: &[f32],
    split_intervals: &[f32],
    split_thresholds: &[f32],
    current_time: f32,
) -> f32 {
    if active_adhesion_count(cell, cell_indices, connections) > 0 {
        return 0.0;
    }

    let age = current_time - birth_times.get(cell).copied().unwrap_or(current_time);
    let interval = split_intervals.get(cell).copied().unwrap_or(1.0).max(0.001);
    let nutrient = nutrients.get(cell).copied().unwrap_or(0) as f32 / 1000.0;
    let threshold = split_thresholds
        .get(cell)
        .copied()
        .unwrap_or(100.0)
        .max(0.001);
    let time_ready = (age / interval).clamp(0.0, 1.5);
    let nutrient_ready = (nutrient / threshold).clamp(0.0, 1.5);
    time_ready.min(nutrient_ready)
}

fn active_adhesion_count(
    cell: usize,
    cell_indices: &[crate::simulation::gpu_physics::adhesion::CellAdhesionIndices],
    connections: &[crate::simulation::gpu_physics::adhesion::GpuAdhesionConnection],
) -> usize {
    cell_indices
        .get(cell)
        .map(|indices| {
            indices
                .indices
                .iter()
                .filter(|&&idx| {
                    idx >= 0
                        && connections
                            .get(idx as usize)
                            .map(|conn| conn.is_active == 1)
                            .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

fn snapshot_from_gpu_group(
    genome: Option<&Genome>,
    genome_id: u32,
    source_cells: &[usize],
    connections: &[crate::simulation::gpu_physics::adhesion::GpuAdhesionConnection],
    positions_and_mass: &[[f32; 4]],
    mode_indices: &[u32],
    current_time: f32,
    captured_before_division: bool,
    max_cells: usize,
) -> Option<LineageAdultSnapshot> {
    let mut cells: Vec<_> = source_cells.to_vec();
    cells.truncate(max_cells.min(u16::MAX as usize));
    if cells.is_empty() {
        return None;
    }

    let mut center = glam::Vec3::ZERO;
    for &cell in &cells {
        let position = positions_and_mass.get(cell).copied().unwrap_or([0.0; 4]);
        center += glam::vec3(position[0], position[1], position[2]);
    }
    center /= cells.len() as f32;

    let mut remap = HashMap::new();
    for (local, &cell) in cells.iter().enumerate() {
        remap.insert(cell, local as u16);
    }

    let snapshot_cells = cells
        .iter()
        .map(|&cell| {
            let pm = positions_and_mass.get(cell).copied().unwrap_or([0.0; 4]);
            let world_pos = glam::vec3(pm[0], pm[1], pm[2]);
            let mode_index = mode_indices.get(cell).copied().unwrap_or(0) as usize;
            let mode = genome.and_then(|genome| genome.modes.get(mode_index));
            let color = mode
                .map(|mode| [mode.color.x, mode.color.y, mode.color.z])
                .unwrap_or([0.35, 0.75, 0.85]);
            let cell_type = mode
                .map(|mode| mode.cell_type.clamp(0, u8::MAX as i32) as u8)
                .unwrap_or(0);
            let pos = world_pos - center;
            LineageAdultCellSnapshot {
                position: [pos.x, pos.y, pos.z],
                radius: pm[3].max(0.1).cbrt().max(0.25),
                mode_index: mode_index.min(u16::MAX as usize) as u16,
                cell_type,
                color,
                emissive: mode.map(|mode| mode.emissive).unwrap_or(0.0),
            }
        })
        .collect();

    let mut bonds = Vec::new();
    for connection in connections {
        if connection.is_active == 0 {
            continue;
        }
        let a = connection.cell_a_index as usize;
        let b = connection.cell_b_index as usize;
        if let (Some(&local_a), Some(&local_b)) = (remap.get(&a), remap.get(&b)) {
            bonds.push([local_a, local_b]);
        }
    }

    let mut world_radius = 0.0f32;
    for &cell in &cells {
        let position = positions_and_mass.get(cell).copied().unwrap_or([0.0; 4]);
        let world_pos = glam::vec3(position[0], position[1], position[2]);
        world_radius =
            world_radius.max((world_pos - center).length() + position[3].max(0.1).cbrt());
    }

    let mut snapshot = LineageAdultSnapshot {
        genome_hash: genome_id as u64,
        captured_time: current_time,
        captured_frame: 0, // filled in by push_adult_snapshot_for_genome
        captured_before_division,
        world_center: [center.x, center.y, center.z],
        world_radius: world_radius.max(1.0),
        cells: snapshot_cells,
        bonds,
        scene_thumbnail_png: None,
    };
    snapshot.scene_thumbnail_png = encode_lineage_scene_thumbnail_png(
        &snapshot,
        crate::scene::lineage::LINEAGE_THUMBNAIL_WIDTH,
        crate::scene::lineage::LINEAGE_THUMBNAIL_HEIGHT,
    );
    Some(snapshot)
}

fn encode_lineage_scene_thumbnail_png(
    snapshot: &LineageAdultSnapshot,
    width: u32,
    height: u32,
) -> Option<Vec<u8>> {
    if snapshot.cells.is_empty() || width == 0 || height == 0 {
        return None;
    }

    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for px in pixels.chunks_exact_mut(4) {
        px.copy_from_slice(&[5, 8, 12, 255]);
    }

    let mut min = glam::Vec2::splat(f32::INFINITY);
    let mut max = glam::Vec2::splat(f32::NEG_INFINITY);
    for cell in &snapshot.cells {
        let p = glam::vec2(cell.position[0], cell.position[1]);
        min = min.min(p);
        max = max.max(p);
    }
    let span = (max - min).max(glam::Vec2::splat(1.0));
    let scale = ((width as f32 - 14.0) / span.x)
        .min((height as f32 - 14.0) / span.y)
        .max(1.0)
        * 0.82;
    let center = glam::vec2(width as f32 * 0.5, height as f32 * 0.48);
    let world_center = (min + max) * 0.5;
    let project = |position: [f32; 3]| {
        let p = glam::vec2(position[0], position[1]);
        center + (p - world_center) * scale
    };

    for bond in &snapshot.bonds {
        let a = bond[0] as usize;
        let b = bond[1] as usize;
        if let (Some(cell_a), Some(cell_b)) = (snapshot.cells.get(a), snapshot.cells.get(b)) {
            draw_rgba_line(
                &mut pixels,
                width,
                height,
                project(cell_a.position),
                project(cell_b.position),
                [10, 18, 24, 160],
                4.0,
            );
            draw_rgba_line(
                &mut pixels,
                width,
                height,
                project(cell_a.position),
                project(cell_b.position),
                [132, 212, 255, 230],
                2.0,
            );
        }
    }

    let mut order: Vec<_> = (0..snapshot.cells.len()).collect();
    order.sort_by(|&a, &b| snapshot.cells[a].position[2].total_cmp(&snapshot.cells[b].position[2]));
    for i in order {
        let cell = &snapshot.cells[i];
        let p = project(cell.position);
        let radius = (cell.radius * scale * 0.38).clamp(2.5, 9.0);
        let color = [
            (cell.color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (cell.color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (cell.color[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ];
        draw_rgba_circle(
            &mut pixels,
            width,
            height,
            p + glam::vec2(2.0, 2.0),
            radius * 1.18,
            [0, 0, 0, 115],
        );
        draw_rgba_circle(
            &mut pixels,
            width,
            height,
            p,
            radius * 1.14,
            [210, 238, 255, 80],
        );
        draw_rgba_circle(&mut pixels, width, height, p, radius, color);
    }

    let mut png = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png);
    encoder
        .write_image(&pixels, width, height, image::ColorType::Rgba8.into())
        .ok()?;
    Some(png)
}

fn draw_rgba_circle(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    center: glam::Vec2,
    radius: f32,
    color: [u8; 4],
) {
    let radius_sq = radius * radius;
    let min_x = (center.x - radius).floor().max(0.0) as u32;
    let max_x = (center.x + radius).ceil().min(width as f32 - 1.0) as u32;
    let min_y = (center.y - radius).floor().max(0.0) as u32;
    let max_y = (center.y + radius).ceil().min(height as f32 - 1.0) as u32;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as f32 + 0.5 - center.x;
            let dy = y as f32 + 0.5 - center.y;
            if dx * dx + dy * dy <= radius_sq {
                blend_pixel(pixels, width, x, y, color);
            }
        }
    }
}

fn draw_rgba_line(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    a: glam::Vec2,
    b: glam::Vec2,
    color: [u8; 4],
    thickness: f32,
) {
    let delta = b - a;
    let steps = delta.length().ceil().max(1.0) as u32;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let p = a.lerp(b, t);
        draw_rgba_circle(pixels, width, height, p, thickness * 0.5, color);
    }
}

fn blend_pixel(pixels: &mut [u8], width: u32, x: u32, y: u32, color: [u8; 4]) {
    let idx = ((y * width + x) * 4) as usize;
    if idx + 3 >= pixels.len() {
        return;
    }
    let alpha = color[3] as f32 / 255.0;
    let inv = 1.0 - alpha;
    pixels[idx] = (color[0] as f32 * alpha + pixels[idx] as f32 * inv) as u8;
    pixels[idx + 1] = (color[1] as f32 * alpha + pixels[idx + 1] as f32 * inv) as u8;
    pixels[idx + 2] = (color[2] as f32 * alpha + pixels[idx + 2] as f32 * inv) as u8;
    pixels[idx + 3] = 255;
}

fn read_sampled_mutation_parents(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mutation_system: Option<&crate::simulation::gpu_physics::MutationSystem>,
    genome_ids: impl Iterator<Item = u32>,
    cpu_genome_count: usize,
) -> HashMap<u32, u32> {
    let Some(mutation_system) = mutation_system else {
        return HashMap::new();
    };

    let mut parents = HashMap::new();
    for genome_id in genome_ids {
        if (genome_id as usize) < cpu_genome_count
            || genome_id >= crate::simulation::gpu_physics::mutation::GENOME_RING_CAPACITY
        {
            continue;
        }

        let byte_offset = genome_id as u64 * std::mem::size_of::<GenomeMeta>() as u64;
        let Ok(bytes) = readback_buffer_bytes(
            device,
            queue,
            mutation_system.genome_meta_buffer(),
            byte_offset,
            std::mem::size_of::<GenomeMeta>() as u64,
        ) else {
            continue;
        };
        let meta = bytemuck::from_bytes::<GenomeMeta>(&bytes);
        if meta.mode_count > 0 && meta.parent_genome_id != u32::MAX {
            parents.insert(genome_id, meta.parent_genome_id);
        }
    }

    parents
}

/// Score a sampled GPU mutation genome and decide whether it deserves a full
/// lineage node.  Returns true only when the score clears the promotion
/// threshold.
///
/// Scoring factors (all log-scaled to avoid a handful of huge populations
/// crowding out everything else):
///
/// 1. **Multi-cellularity** — the primary signal of structural innovation.
///    Single-celled mutations (avg ≤ 1.5 cells/organism) are immediately
///    disqualified unless they have achieved true ecological dominance.
/// 2. **Ecological spread** — independent organisms indicate the genome
///    survives division and competes successfully.
/// 3. **Persistence** — lineages seen across multiple scan intervals are much
///    more likely to be stable innovations rather than transient drift.
/// 4. **Vitality** — a lineage that is crashing toward zero is discounted;
///    one that is holding or growing receives a bonus.
fn sampled_mutation_is_meaningful(
    cells: u32,
    organisms: u32,
    prior: Option<&crate::scene::lineage::LineageGenomeStatSample>,
) -> bool {
    // --- Hard gates -----------------------------------------------------------

    // Genome must have at least one organism with ≥ 2 cells on average.
    // Single-cell → single-cell mutations are structurally trivial.
    // Exception: extreme ecological dominance (≥ 400 cells) may still be
    // worth tracking even for single-celled genomes.
    let avg_body = if organisms > 0 {
        cells as f32 / organisms as f32
    } else {
        cells as f32
    };
    const SINGLE_CELL_DOMINANCE: u32 = 400;
    if avg_body < 2.0 && cells < SINGLE_CELL_DOMINANCE {
        return false;
    }

    // Must be seen at least once before with a credible population, OR must
    // already show substantial multi-cellular presence right now.  This
    // filters one-frame transient blips.
    const INSTANT_PROMOTION_ORGANISMS: u32 = 8;
    const INSTANT_PROMOTION_AVG_BODY: f32 = 6.0;
    let instant =
        organisms >= INSTANT_PROMOTION_ORGANISMS && avg_body >= INSTANT_PROMOTION_AVG_BODY;
    if prior.is_none() && !instant {
        return false;
    }

    // --- Scoring --------------------------------------------------------------

    // Multi-cellularity: log2 of avg body size above 1.  Zero for single-cell;
    // 1.0 for 2 cells/organism; 2.0 for 4; 3.0 for 8; etc.
    let complexity = (avg_body - 1.0).max(0.0).log2().max(0.0) * 5.0;

    // Ecological spread: independent organisms competing in the world.
    let spread = (organisms as f32 + 1.0).log2() * 2.5;

    // Persistence and vitality from prior observations.
    let (persistence, vitality_factor) = match prior {
        Some(p) => {
            // Each additional scan interval the lineage survives is strong
            // evidence it is a stable innovation.
            let obs = p.observations.saturating_add(1) as f32;
            let persistence_score = obs.log2() * 4.0;

            // Vitality: ratio of current cells to the peak ever seen.
            // A declining lineage is discounted; a stable or growing one gets
            // full weight.
            let peak = p.cells.max(cells).max(1) as f32;
            let vitality = (cells as f32 / peak).clamp(0.05, 1.0);
            // Map vitality to [0.2, 1.0] so even declining lineages can pass
            // if they are structurally complex and persistent enough.
            let vf = 0.2 + vitality * 0.8;

            (persistence_score, vf)
        }
        None => (0.0, 0.6), // instant promotion path: partial credit
    };

    let score = (complexity + spread + persistence) * vitality_factor;

    // Threshold chosen so that:
    // - A 4-cell avg / 5-organism / 2-scan genome scores ≈ 8 → promotes.
    // - A 2-cell avg / 2-organism / 2-scan genome scores ≈ 3.8 → borderline.
    // - Any single-scan observation of a modestly multi-cellular genome is
    //   blocked by the prior==None gate above unless it's very large.
    const THRESHOLD: f32 = 5.0;
    score >= THRESHOLD
}

// --- GpuScene impl -----------------------------------------------------------

impl GpuScene {
    pub fn maybe_capture_lineage_interval(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<bool, SnapshotError> {
        let interval = self.lineage_capture_interval_seconds;
        if self.current_time < interval {
            return Ok(false);
        }

        let due = self
            .lineage_archive
            .last_scan_frame
            .map(|_| self.current_time - self.lineage_archive.last_scan_time >= interval)
            .unwrap_or(true);

        if due {
            self.scan_lineage_for_viewer(device, queue)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Capture a frozen lineage/bestiary population scan for the UI.
    ///
    /// This is deliberately an explicit, blocking, user-initiated scan. It does
    /// not run during normal gameplay and it reads only compact ID buffers:
    /// genome IDs, death flags, and stable organism IDs when available.
    pub fn scan_lineage_for_viewer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), SnapshotError> {
        let cell_counters: Vec<u32> =
            readback_typed(device, queue, &self.gpu_triple_buffers.cell_count_buffer, 2)?;
        let slots_used = cell_counters
            .first()
            .copied()
            .unwrap_or(0)
            .min(self.gpu_triple_buffers.capacity);
        let live_cells = cell_counters.get(1).copied().unwrap_or(0);

        self.total_cell_slots = slots_used;
        self.current_cell_count = live_cells;

        // Snapshot pre-scan state for stable-period detection (done at end of fn).
        let pre_node_count = self.lineage_archive.nodes.len();
        let pre_event_count = self.lineage_archive.events.len();
        let pre_live_cells = self.lineage_archive.last_scan_live_cells;
        let pre_scan_time = self.lineage_archive.last_scan_time;
        let pre_scan_frame = self.lineage_archive.last_scan_frame;

        let slots = slots_used as usize;
        self.lineage_archive
            .ensure_scene_genomes(&self.genomes, self.current_frame);

        if slots == 0 {
            self.lineage_archive.record_scan_population(
                self.current_frame,
                self.current_time,
                live_cells,
                0,
                live_cells,
                true,
            );
            return Ok(());
        }

        if let Some(labels) = &mut self.organism_label_system {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Lineage Viewer Label Refresh"),
            });
            labels.encode_frame(&mut encoder, slots_used, true);
            queue.submit(Some(encoder.finish()));
        }

        let genome_ids: Vec<u32> =
            readback_typed(device, queue, &self.gpu_triple_buffers.genome_ids, slots)?;
        let death_flags: Vec<u32> =
            readback_typed(device, queue, &self.gpu_triple_buffers.death_flags, slots)?;
        let buf_idx = self.gpu_triple_buffers.output_buffer_index();
        let positions_and_mass: Vec<[f32; 4]> = readback_typed(
            device,
            queue,
            &self.gpu_triple_buffers.position_and_mass[buf_idx],
            slots,
        )?;
        let mode_indices: Vec<u32> =
            readback_typed(device, queue, &self.gpu_triple_buffers.mode_indices, slots)?;
        let nutrients: Vec<i32> = readback_typed(
            device,
            queue,
            &self.gpu_triple_buffers.nutrients_buffer,
            slots,
        )?;
        let birth_times: Vec<f32> =
            readback_typed(device, queue, &self.gpu_triple_buffers.birth_times, slots)?;
        let split_intervals: Vec<f32> = readback_typed(
            device,
            queue,
            &self.gpu_triple_buffers.split_intervals,
            slots,
        )?;
        let split_nutrient_thresholds: Vec<f32> = readback_typed(
            device,
            queue,
            &self.gpu_triple_buffers.split_nutrient_thresholds,
            slots,
        )?;

        let organism_labels: Option<Vec<u32>> = self
            .organism_label_system
            .as_ref()
            .map(|labels| readback_typed(device, queue, &labels.label_buffer, slots))
            .transpose()?;

        let mut cell_counts: BTreeMap<u32, u32> = BTreeMap::new();
        let mut organism_sets: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut invalid_live_cells = 0u32;
        let mut labeled_live_cells = 0u32;

        for i in 0..slots {
            if death_flags.get(i).copied().unwrap_or(1) != 0 {
                continue;
            }

            let genome_id = genome_ids.get(i).copied().unwrap_or(u32::MAX);
            if genome_id == u32::MAX {
                invalid_live_cells = invalid_live_cells.saturating_add(1);
                continue;
            }

            let count = cell_counts.entry(genome_id).or_default();
            *count = count.saturating_add(1);

            if let Some(labels) = &organism_labels {
                let label = labels.get(i).copied().unwrap_or(u32::MAX);
                if label < slots_used {
                    labeled_live_cells = labeled_live_cells.saturating_add(1);
                    organism_sets.entry(genome_id).or_default().insert(label);
                }
            }
        }

        let organism_counts_reliable =
            organism_labels.is_some() && (live_cells == 0 || labeled_live_cells > 0);
        let adult_snapshots = capture_gpu_adult_snapshots(
            &self.genomes,
            &self.adhesion_buffers.snapshot_connections(),
            &self.adhesion_buffers.snapshot_cell_indices(),
            &positions_and_mass,
            &mode_indices,
            &genome_ids,
            &death_flags,
            organism_labels.as_deref(),
            organism_counts_reliable,
            &nutrients,
            &birth_times,
            &split_intervals,
            &split_nutrient_thresholds,
            self.current_time,
            slots_used,
        );
        let mutation_parents = read_sampled_mutation_parents(
            device,
            queue,
            self.mutation_system.as_ref(),
            cell_counts.keys().copied(),
            self.genomes.len(),
        );
        let mut samples = Vec::with_capacity(cell_counts.len());
        let mut tracked_cells = 0u32;
        for (&genome_id, &cells) in &cell_counts {
            let organisms = if organism_counts_reliable {
                organism_sets
                    .get(&genome_id)
                    .map(|set| set.len() as u32)
                    .unwrap_or(0)
            } else {
                0
            };

            let lineage_id =
                if let Some(lineage_id) = self.lineage_archive.lineage_for_genome_id(genome_id) {
                    lineage_id
                } else if (genome_id as usize) < self.genomes.len() {
                    self.lineage_archive.ensure_user_lineage(
                        genome_id,
                        &self.genomes[genome_id as usize],
                        self.current_frame,
                    )
                } else {
                    let parent_genome_id = mutation_parents.get(&genome_id).copied();
                    let prior_sample = self
                        .lineage_archive
                        .unpromoted_genome_sample(genome_id)
                        .cloned();
                    if !sampled_mutation_is_meaningful(cells, organisms, prior_sample.as_ref()) {
                        self.lineage_archive.record_unpromoted_genome_sample(
                            genome_id,
                            parent_genome_id,
                            cells,
                            organisms,
                            self.current_frame,
                        );
                        continue;
                    }

                    let parent_lineage = parent_genome_id.and_then(|parent_genome_id| {
                        if parent_genome_id == genome_id {
                            None
                        } else if let Some(lineage_id) =
                            self.lineage_archive.lineage_for_genome_id(parent_genome_id)
                        {
                            Some(lineage_id)
                        } else if (parent_genome_id as usize) < self.genomes.len() {
                            Some(self.lineage_archive.ensure_user_lineage(
                                parent_genome_id,
                                &self.genomes[parent_genome_id as usize],
                                self.current_frame,
                            ))
                        } else {
                            None
                        }
                    });
                    self.lineage_archive.ensure_sampled_mutation_lineage(
                        genome_id,
                        parent_lineage,
                        self.current_frame,
                    )
                };

            if (genome_id as usize) >= self.genomes.len() {
                let display_name = self
                    .lineage_archive
                    .nodes
                    .iter()
                    .find(|node| node.id == lineage_id)
                    .map(|node| node.display_name.clone())
                    .unwrap_or_else(|| format!("GPU Genome #{genome_id}"));

                if let Some(genome) = self.read_back_genome(device, queue, genome_id) {
                    match genome.to_yaml_string() {
                        Ok(genome_yaml) => {
                            self.lineage_archive.upsert_loadable_genome(
                                lineage_id,
                                genome_id,
                                display_name,
                                "Rolling lineage interval payload",
                                self.current_frame,
                                genome_yaml,
                            );
                        }
                        Err(error) => {
                            log::warn!(
                                "[Lineage] Failed to serialize loadable genome {}: {}",
                                genome_id,
                                error
                            );
                        }
                    }
                }
            }

            if let Some(snapshot) = adult_snapshots.get(&genome_id).cloned() {
                self.lineage_archive.push_adult_snapshot_for_genome(
                    genome_id,
                    self.current_frame,
                    snapshot,
                );
            }
            tracked_cells = tracked_cells.saturating_add(cells);
            samples.push(LineagePopulationSample {
                lineage_id,
                cells,
                organisms,
                frame: self.current_frame,
            });
        }

        self.lineage_archive
            .apply_population_samples(&samples, self.current_frame);
        let untracked_cells = live_cells
            .saturating_sub(tracked_cells)
            .max(invalid_live_cells);
        self.lineage_archive.record_scan_population(
            self.current_frame,
            self.current_time,
            live_cells,
            tracked_cells,
            untracked_cells,
            organism_counts_reliable,
        );

        // Detect stable periods: if nothing meaningful changed since the last scan,
        // record a compressed time gap so the UI can collapse it.
        if let Some(start_frame) = pre_scan_frame {
            let new_noteworthy = self
                .lineage_archive
                .events
                .get(pre_event_count..)
                .map(|events| events.iter().any(|e| e.noteworthy))
                .unwrap_or(false);
            let new_extinctions = self
                .lineage_archive
                .nodes
                .iter()
                .any(|n| n.extinct_frame.map(|f| f > start_frame).unwrap_or(false));
            // Count nodes added beyond what ensure_scene_genomes would guarantee.
            let guaranteed_node_count = pre_node_count.max(self.genomes.len());
            let new_promoted = self.lineage_archive.nodes.len() > guaranteed_node_count;
            let cell_change = if pre_live_cells > 0 {
                (live_cells as f32 - pre_live_cells as f32).abs() / pre_live_cells as f32
            } else if live_cells > 0 {
                1.0
            } else {
                0.0
            };

            let is_stable = !new_noteworthy
                && !new_extinctions
                && !new_promoted
                && cell_change < crate::scene::lineage::STABLE_LIVE_CELL_CHANGE_THRESHOLD;

            if is_stable {
                self.lineage_archive.record_stable_period(
                    start_frame,
                    self.current_frame,
                    pre_scan_time,
                    self.current_time,
                    pre_live_cells,
                );
            }
        }

        log::info!(
            "[Lineage] Captured viewer scan: {} slots, {} live cells, {} tracked cells, {} samples",
            slots,
            live_cells,
            tracked_cells,
            samples.len()
        );

        Ok(())
    }

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
            readback_typed(
                device,
                queue,
                &self.gpu_triple_buffers.position_and_mass[buf_idx],
                slots,
            )?
        } else {
            Vec::new()
        };

        let velocities: Vec<[f32; 4]> = if slots > 0 {
            readback_typed(
                device,
                queue,
                &self.gpu_triple_buffers.velocity[buf_idx],
                slots,
            )?
        } else {
            Vec::new()
        };

        let rotations: Vec<[f32; 4]> = if slots > 0 {
            readback_typed(
                device,
                queue,
                &self.gpu_triple_buffers.rotations[buf_idx],
                slots,
            )?
        } else {
            Vec::new()
        };

        let genome_orientations: Vec<[f32; 4]> = if slots > 0 {
            readback_typed(
                device,
                queue,
                &self.gpu_triple_buffers.genome_orientations,
                slots,
            )?
        } else {
            Vec::new()
        };

        let nutrients: Vec<i32> = if slots > 0 {
            readback_typed(
                device,
                queue,
                &self.gpu_triple_buffers.nutrients_buffer,
                slots,
            )?
        } else {
            Vec::new()
        };

        let birth_times: Vec<f32> = if slots > 0 {
            readback_typed(device, queue, &self.gpu_triple_buffers.birth_times, slots)?
        } else {
            Vec::new()
        };

        let split_intervals: Vec<f32> = if slots > 0 {
            readback_typed(
                device,
                queue,
                &self.gpu_triple_buffers.split_intervals,
                slots,
            )?
        } else {
            Vec::new()
        };

        let split_nutrient_thresholds: Vec<f32> = if slots > 0 {
            readback_typed(
                device,
                queue,
                &self.gpu_triple_buffers.split_nutrient_thresholds,
                slots,
            )?
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
            readback_typed(
                device,
                queue,
                &self.gpu_triple_buffers.embryocyte_reserve_buffer,
                slots,
            )?
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
        let (
            cave_density,
            cave_scale,
            cave_octaves,
            cave_persistence,
            cave_threshold,
            cave_smoothness,
            cave_seed,
            cave_resolution,
        ) = if let Some(ref cave) = self.cave_renderer {
            let p = cave.params();
            (
                p.density,
                p.scale,
                p.octaves,
                p.persistence,
                p.threshold,
                p.smoothness,
                p.seed,
                p.grid_resolution,
            )
        } else {
            (0.5, 100.0, 2, 0.5, 1.0, 0.0, 12345, 128)
        };

        // -- Fluid / water state -----------------------------------------------
        let fluid_active = self.fluid_simulator.is_some();
        let (fluid_voxels, nutrient_voxels, fluid_time, fluid_type, fluid_continuous_spawn) =
            if let Some(ref sim) = self.fluid_simulator {
                log::info!("[Snapshot] Reading back fluid voxels (2 × 8 MB)…");
                let (fv, nv) = sim.snapshot_voxels(device, queue);
                (
                    fv,
                    nv,
                    sim.time(),
                    sim.get_fluid_type(),
                    sim.is_continuous_spawn_enabled(),
                )
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

        let mut lineage_archive = self.lineage_archive.clone();
        lineage_archive.prepare_for_save();

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
            lineage_archive,
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
        self.lineage_archive = snapshot.lineage_archive.clone();
        self.lineage_archive.migrate_legacy_snapshots();
        self.lineage_archive
            .ensure_scene_genomes(&self.genomes, self.current_frame);

        // Rebuild derived genome caches.
        self.parent_make_adhesion_flags.clear();
        self.has_oculocytes = false;
        self.max_signal_hops = 0;
        for genome in &self.genomes {
            for mode in &genome.modes {
                self.parent_make_adhesion_flags
                    .push(mode.parent_make_adhesion);
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
            self.gpu_triple_buffers.sync_from_canonical_state(
                device,
                queue,
                &canonical,
                &self.genomes,
            );

            // Sync embryocyte reserve buffer to GPU (sync_from_canonical_state doesn't cover it)
            if !snapshot.embryocyte_reserves.is_empty() {
                self.gpu_triple_buffers.sync_embryocyte_reserves(
                    queue,
                    &snapshot.embryocyte_reserves,
                    slots,
                );
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
            self.gpu_triple_buffers
                .grow_mode_pool_if_needed(device, total_modes);

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
                p.density = snapshot.cave_density;
                p.scale = snapshot.cave_scale;
                p.octaves = snapshot.cave_octaves;
                p.persistence = snapshot.cave_persistence;
                p.threshold = snapshot.cave_threshold;
                p.smoothness = snapshot.cave_smoothness;
                p.seed = snapshot.cave_seed;
                p.grid_resolution = snapshot.cave_resolution;
                // world_center and world_radius are always derived from config.
                p.world_center = [0.0, 0.0, 0.0];
                p.world_radius = self.config.sphere_radius;
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
                log::info!(
                    "[Snapshot] Fluid voxels restored ({} voxels).",
                    snapshot.fluid_voxels.len()
                );
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

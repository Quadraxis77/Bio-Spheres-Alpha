// Glueocyte Cell-to-Cell Adhesion Shader
//
// Two entry points dispatched once per physics step:
//
//   bond_create  - Each glueocyte thread scans its spatial-grid neighbourhood.
//                  When it finds an overlapping cell it is not yet bonded to, it
//                  allocates an adhesion slot and writes the connection.
//                  Skipped when the signal gate is active and the signal is below
//                  the threshold (glueocyte is "inactive").
//
//   bond_release - Each glueocyte thread walks its own per-cell adhesion list and
//                  marks every bond that has BOND_FLAG_GLUEOCYTE set as inactive,
//                  then removes it from both cells' index arrays and frees the slot.
//                  Only runs when the signal gate is active AND the signal is below
//                  the threshold (glueocyte just became "inactive").
//
// Signal gate semantics (per mode):
//   glueocyte_cell_adhesion_flags[mode_idx * 4 + 0] = enabled (0/1)
//   glueocyte_cell_adhesion_flags[mode_idx * 4 + 1] = signal_channel
//                                                      (0xFFFFFFFF = always active)
//   glueocyte_cell_adhesion_flags[mode_idx * 4 + 2] = signal_threshold (f32 bits)
//   glueocyte_cell_adhesion_flags[mode_idx * 4 + 3] = flags:
//       bit 0 = self_adhesion (0 = skip own organism, 1 = bond to own organism)
//       bit 1 = invert gate   (0 = active when sig >= threshold, 1 = active when sig < threshold)
//
// Bond origin flag stored in AdhesionConnection._align_pad.x (offset 24):
//   BOND_FLAG_GLUEOCYTE = 1u  - created by this shader, released on deactivation
//
// Group 0: Standard physics bind group
// Group 1: Adhesion buffers (connections, indices, next_id, free_slots, counts)
// Group 2: Spatial grid (counts, offsets, cells, cell_grid_indices)
// Group 3: Mode / signal data (mode_indices, mode_cell_types, cell_adhesion_flags,
//                              signal_flags, rotations, genome_orientations)

// ---- Structs ----

struct PhysicsParams {
    delta_time: f32,
    current_time: f32,
    current_frame: i32,
    cell_count: u32,
    world_size: f32,
    boundary_stiffness: f32,
    gravity: f32,
    acceleration_damping: f32,
    grid_resolution: i32,
    grid_cell_size: f32,
    max_cells_per_grid: i32,
    enable_thrust_force: i32,
    cell_capacity: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct AdhesionConnection {
    cell_a_index: u32,
    cell_b_index: u32,
    mode_index: u32,
    is_active: u32,
    zone_a: u32,
    zone_b: u32,
    // _align_pad.x repurposed as bond_flags:
    //   bit 0 = BOND_FLAG_GLUEOCYTE (created by this shader)
    bond_flags: u32,
    _align_pad1: u32,
    anchor_direction_a: vec4<f32>,
    anchor_direction_b: vec4<f32>,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    birth_time: f32,
    _pad: u32,
}

// ---- Constants ----

const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const SIGNAL_CHANNELS: u32 = 16u;
const GLUEOCYTE_CELL_TYPE: u32 = 6u;
const BOND_FLAG_GLUEOCYTE: u32 = 1u;

// ---- Group 0: Physics ----

@group(0) @binding(0) var<uniform> params: PhysicsParams;
@group(0) @binding(1) var<storage, read> positions_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> cell_count_buffer: array<u32>;

// ---- Group 1: Adhesion buffers ----

@group(1) @binding(0) var<storage, read_write> adhesion_connections: array<AdhesionConnection>;
@group(1) @binding(1) var<storage, read_write> cell_adhesion_indices: array<atomic<i32>>;
@group(1) @binding(2) var<storage, read_write> next_adhesion_id: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> free_adhesion_slots: array<u32>;
@group(1) @binding(4) var<storage, read_write> adhesion_counts: array<atomic<u32>>;

// ---- Group 2: Spatial grid ----

@group(2) @binding(0) var<storage, read> spatial_grid_counts: array<u32>;
@group(2) @binding(1) var<storage, read> spatial_grid_offsets: array<u32>;
@group(2) @binding(3) var<storage, read> spatial_grid_cells: array<u32>;

// ---- Group 3: Mode / signal data ----

@group(3) @binding(0) var<storage, read> mode_indices: array<u32>;
@group(3) @binding(1) var<storage, read> mode_cell_types: array<u32>;
// 4 u32 per mode: [enabled, signal_channel, signal_threshold_bits, flags(bit0=self_adhesion, bit1=invert)]
@group(3) @binding(2) var<storage, read> glueocyte_cell_adhesion_flags: array<u32>;
// Per-cell signal flags (packed u32: bits 0-10 = value, bits 11-15 = hops, bit 16 = source)
@group(3) @binding(3) var<storage, read> signal_flags: array<u32>;
// Per-cell genome orientations (pure genome chain, no physics perturbation)
@group(3) @binding(4) var<storage, read> genome_orientations: array<vec4<f32>>;
// Per-cell death flags
@group(3) @binding(5) var<storage, read> death_flags: array<u32>;
// Per-cell organism labels (for self-adhesion filtering)
@group(3) @binding(6) var<storage, read> organism_labels: array<u32>;
// ---- Helpers ----

fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn rotate_by_quat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn rotate_by_quat_inv(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return rotate_by_quat(v, quat_conjugate(q));
}

// Allocate an adhesion slot, preferring freed slots over monotonic growth.
fn allocate_adhesion_slot() -> u32 {
    // Try to pop from free stack
    loop {
        let free_top = atomicLoad(&adhesion_counts[2]);
        if (free_top == 0u) { break; }
        let result = atomicCompareExchangeWeak(&adhesion_counts[2], free_top, free_top - 1u);
        if (result.exchanged) {
            atomicAdd(&adhesion_counts[1], 1u);
            return free_adhesion_slots[free_top - 1u];
        }
    }
    // Monotonic fallback
    let slot = atomicAdd(&next_adhesion_id[0], 1u);
    if (slot < arrayLength(&adhesion_connections)) {
        atomicAdd(&adhesion_counts[1], 1u);
        return slot;
    }
    return 0xFFFFFFFFu; // at capacity
}

// Try to write adhesion_idx into the first free slot of cell's index array.
// Returns true on success.
fn try_add_adhesion_to_cell(cell_idx: u32, adhesion_idx: u32) -> bool {
    let base = cell_idx * MAX_ADHESIONS_PER_CELL;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let result = atomicCompareExchangeWeak(
            &cell_adhesion_indices[base + i],
            -1i,
            i32(adhesion_idx),
        );
        if (result.exchanged) { return true; }
    }
    return false;
}

// Remove adhesion_idx from cell's index array (set slot to -1).
fn remove_adhesion_from_cell(cell_idx: u32, adhesion_idx: u32) {
    let base = cell_idx * MAX_ADHESIONS_PER_CELL;
    let search_val = i32(adhesion_idx);
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        if (atomicLoad(&cell_adhesion_indices[base + i]) == search_val) {
            atomicStore(&cell_adhesion_indices[base + i], -1i);
            break;
        }
    }
}

// Count active adhesions for a cell (used for max_adhesions gate).
fn count_active_adhesions(cell_idx: u32) -> u32 {
    let base = cell_idx * MAX_ADHESIONS_PER_CELL;
    var count = 0u;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        if (atomicLoad(&cell_adhesion_indices[base + i]) >= 0i) {
            count++;
        }
    }
    return count;
}

// Check whether two cells are already connected.
fn already_connected(cell_a: u32, cell_b: u32) -> bool {
    let base = cell_a * MAX_ADHESIONS_PER_CELL;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let adh_idx = atomicLoad(&cell_adhesion_indices[base + i]);
        if (adh_idx < 0i) { continue; }
        let conn = adhesion_connections[u32(adh_idx)];
        if (conn.is_active != 0u) {
            if ((conn.cell_a_index == cell_a && conn.cell_b_index == cell_b) ||
                (conn.cell_a_index == cell_b && conn.cell_b_index == cell_a)) {
                return true;
            }
        }
    }
    return false;
}

// Read the signal value for a cell on a given channel (lower 11 bits).
fn read_signal(cell_idx: u32, channel: u32) -> f32 {
    let raw = signal_flags[cell_idx * SIGNAL_CHANNELS + channel];
    return f32(raw & 0x7FFu);
}

// Determine whether a glueocyte is currently "active" (should form/keep bonds).
// Returns true when:
//   - cell adhesion is enabled for this mode, AND
//   - either no signal gate is configured, OR the signal condition is met.
// The condition is sig >= threshold normally, or sig < threshold when invert is set.
fn is_glueocyte_active(mode_idx: u32, cell_idx: u32) -> bool {
    let base = mode_idx * 4u;
    if (base + 3u >= arrayLength(&glueocyte_cell_adhesion_flags)) { return false; }

    let enabled = glueocyte_cell_adhesion_flags[base + 0u];
    if (enabled == 0u) { return false; }

    let channel = glueocyte_cell_adhesion_flags[base + 1u];
    if (channel == 0xFFFFFFFFu) {
        // No signal gate - always active
        return true;
    }

    let threshold_bits = glueocyte_cell_adhesion_flags[base + 2u];
    let threshold = bitcast<f32>(threshold_bits);
    let sig = read_signal(cell_idx, clamp(channel, 0u, 7u));
    let flags = glueocyte_cell_adhesion_flags[base + 3u];
    let invert = (flags & 2u) != 0u;
    return select(sig >= threshold, sig < threshold, invert);
}

// ---- Entry point 1: bond_create ----
// One thread per cell. Glueocytes that are active scan their spatial-grid
// neighbourhood and form bonds with overlapping cells they are not yet bonded to.

@compute @workgroup_size(256)
fn bond_create(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) { return; }
    if (death_flags[cell_idx] == 1u) { return; }

    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_cell_types)) { return; }

    // Only glueocytes run this shader
    if (mode_cell_types[mode_idx] != GLUEOCYTE_CELL_TYPE) { return; }

    // Only form bonds when active
    if (!is_glueocyte_active(mode_idx, cell_idx)) { return; }

    // Read max_adhesions for this mode from adhesion_counts[0] (total capacity).
    // We use the per-cell count gate instead - read from the index array directly.
    // The max_adhesions limit is encoded in the mode; we approximate it as
    // MAX_ADHESIONS_PER_CELL here (the shader doesn't have mode_properties access).
    // Callers can tighten this via the genome's max_adhesions setting which the
    // adhesion_physics shader already enforces for spring forces.
    let my_adhesion_count = count_active_adhesions(cell_idx);
    if (my_adhesion_count >= MAX_ADHESIONS_PER_CELL) { return; }

    let my_pos = positions_in[cell_idx].xyz;
    let my_mass = positions_in[cell_idx].w;
    let my_radius = clamp(my_mass, 0.5, 2.0);
    let my_rot = genome_orientations[cell_idx];

    // Spatial grid lookup
    let res = params.grid_resolution;
    let cs = params.grid_cell_size;
    let half_world = params.world_size * 0.5;
    let grid_pos = (my_pos + vec3<f32>(half_world)) / cs;
    let gx = i32(grid_pos.x);
    let gy = i32(grid_pos.y);
    let gz = i32(grid_pos.z);

    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                let nx = gx + dx;
                let ny = gy + dy;
                let nz = gz + dz;
                if (nx < 0 || nx >= res || ny < 0 || ny >= res || nz < 0 || nz >= res) { continue; }

                let grid_idx = u32(nx + ny * res + nz * res * res);
                let count = spatial_grid_counts[grid_idx];
                let offset = spatial_grid_offsets[grid_idx];

                for (var k = 0u; k < count; k++) {
                    let other_idx = spatial_grid_cells[offset + k];
                    if (other_idx == cell_idx) { continue; }
                    if (other_idx >= cell_count) { continue; }
                    if (death_flags[other_idx] == 1u) { continue; }

                    let other_pos = positions_in[other_idx].xyz;
                    let other_mass = positions_in[other_idx].w;
                    let other_radius = clamp(other_mass, 0.5, 2.0);

                    // Overlap test
                    let delta = other_pos - my_pos;
                    let dist_sq = dot(delta, delta);
                    let contact_dist = my_radius + other_radius;
                    if (dist_sq >= contact_dist * contact_dist) { continue; }
                    if (dist_sq < 0.0001) { continue; }

                    // Already bonded?
                    if (already_connected(cell_idx, other_idx)) { continue; }

                    // Self-adhesion gate: skip same-organism cells unless self_adhesion is enabled.
                    let flags_base = mode_idx * 4u;
                    let self_adhesion_flag = glueocyte_cell_adhesion_flags[flags_base + 3u];
                    if (self_adhesion_flag == 0u) {
                        let my_org    = organism_labels[cell_idx];
                        let other_org = organism_labels[other_idx];
                        if (my_org != 0xFFFFFFFFu && other_org != 0xFFFFFFFFu && my_org == other_org) {
                            continue;
                        }
                    }

                    // Other cell adhesion count gate
                    let other_count = count_active_adhesions(other_idx);
                    if (other_count >= MAX_ADHESIONS_PER_CELL) { continue; }

                    // Allocate slot
                    let slot = allocate_adhesion_slot();
                    if (slot == 0xFFFFFFFFu) { return; } // at capacity

                    // Compute local-space anchor directions
                    let dist = sqrt(dist_sq);
                    let dir_a_to_b = delta / dist;
                    let dir_b_to_a = -dir_a_to_b;

                    let anchor_a = normalize(rotate_by_quat_inv(dir_a_to_b, my_rot));
                    let other_rot = genome_orientations[other_idx];
                    let anchor_b = normalize(rotate_by_quat_inv(dir_b_to_a, other_rot));

                    // Build connection - use parent mode_index for adhesion settings lookup
                    var conn: AdhesionConnection;
                    conn.cell_a_index = cell_idx;
                    conn.cell_b_index = other_idx;
                    conn.mode_index   = mode_idx;
                    conn.is_active    = 1u;
                    conn.zone_a       = 2u; // ZoneC (equatorial - no zone preference)
                    conn.zone_b       = 2u;
                    conn.bond_flags   = BOND_FLAG_GLUEOCYTE;
                    conn._align_pad1  = 0u;
                    conn.anchor_direction_a = vec4<f32>(anchor_a, 0.0);
                    conn.anchor_direction_b = vec4<f32>(anchor_b, 0.0);
                    // Identity quaternions for twist references (no twist constraint)
                    conn.twist_reference_a = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                    conn.twist_reference_b = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                    conn.birth_time   = params.current_time;
                    conn._pad         = 0u;

                    adhesion_connections[slot] = conn;

                    // Register in both cells' index arrays
                    let ok_a = try_add_adhesion_to_cell(cell_idx, slot);
                    let ok_b = try_add_adhesion_to_cell(other_idx, slot);

                    if (!ok_a || !ok_b) {
                        // Couldn't register - roll back: mark inactive and free slot
                        adhesion_connections[slot].is_active = 0u;
                        if (ok_a) { remove_adhesion_from_cell(cell_idx, slot); }
                        if (ok_b) { remove_adhesion_from_cell(other_idx, slot); }
                        atomicSub(&adhesion_counts[1], 1u);
                        let free_top = atomicAdd(&adhesion_counts[2], 1u);
                        free_adhesion_slots[free_top] = slot;
                    }

                    // Only form one new bond per frame per glueocyte to avoid
                    // exhausting the adhesion budget in a single step.
                    return;
                }
            }
        }
    }
}

// ---- Entry point 2: bond_release ----
// One thread per cell. Glueocytes that are INACTIVE walk their adhesion list and
// release every bond they created (BOND_FLAG_GLUEOCYTE set).

@compute @workgroup_size(256)
fn bond_release(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    let cell_count = cell_count_buffer[0];
    if (cell_idx >= cell_count) { return; }
    if (death_flags[cell_idx] == 1u) { return; }

    let mode_idx = mode_indices[cell_idx];
    if (mode_idx >= arrayLength(&mode_cell_types)) { return; }

    // Only glueocytes run this shader
    if (mode_cell_types[mode_idx] != GLUEOCYTE_CELL_TYPE) { return; }

    // Only release bonds when INACTIVE (signal gate enabled and signal below threshold)
    let base = mode_idx * 4u;
    if (base + 3u >= arrayLength(&glueocyte_cell_adhesion_flags)) { return; }

    let enabled = glueocyte_cell_adhesion_flags[base + 0u];
    if (enabled == 0u) { return; } // cell adhesion disabled entirely - nothing to release

    let channel = glueocyte_cell_adhesion_flags[base + 1u];
    if (channel == 0xFFFFFFFFu) { return; } // no signal gate - never releases

    let threshold_bits = glueocyte_cell_adhesion_flags[base + 2u];
    let threshold = bitcast<f32>(threshold_bits);
    let sig = read_signal(cell_idx, clamp(channel, 0u, 7u));
    let flags = glueocyte_cell_adhesion_flags[base + 3u];
    let invert = (flags & 2u) != 0u;
    let still_active = select(sig >= threshold, sig < threshold, invert);
    if (still_active) { return; } // still active - don't release

    // Glueocyte is inactive: release all bonds it created
    let adh_base = cell_idx * MAX_ADHESIONS_PER_CELL;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let adh_idx_i32 = atomicLoad(&cell_adhesion_indices[adh_base + i]);
        if (adh_idx_i32 < 0i) { continue; }
        let adh_idx = u32(adh_idx_i32);

        let conn = adhesion_connections[adh_idx];
        if (conn.is_active == 0u) { continue; }

        // Only release bonds this glueocyte created
        if ((conn.bond_flags & BOND_FLAG_GLUEOCYTE) == 0u) { continue; }

        // This glueocyte must be cell_a (it created the bond)
        if (conn.cell_a_index != cell_idx) { continue; }

        // Mark inactive
        adhesion_connections[adh_idx].is_active = 0u;

        // Remove from both cells' index arrays
        atomicStore(&cell_adhesion_indices[adh_base + i], -1i);
        remove_adhesion_from_cell(conn.cell_b_index, adh_idx);

        // Decrement live count and push to free stack
        atomicSub(&adhesion_counts[1], 1u);
        let free_top = atomicAdd(&adhesion_counts[2], 1u);
        free_adhesion_slots[free_top] = adh_idx;
    }
}

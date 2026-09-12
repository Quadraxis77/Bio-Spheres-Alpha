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
    bond_flags: u32,
    _align_pad1: u32,
    anchor_direction_a: vec4<f32>,
    anchor_direction_b: vec4<f32>,
    twist_reference_a: vec4<f32>,
    twist_reference_b: vec4<f32>,
    birth_time: f32,
    _pad: u32,
}

struct ScaffoldRule {
    id: u32,
    genome_id: u32,
    endpoint_a_kind: u32,
    endpoint_b_kind: u32,
    endpoint_a_mode: u32,
    endpoint_b_mode: u32,
    endpoint_a_hash_lo: u32,
    endpoint_a_hash_hi: u32,
    endpoint_b_hash_lo: u32,
    endpoint_b_hash_hi: u32,
    rest_length_bits: u32,
    max_range_bits: u32,
    endpoint_a_branch_slot: u32,
    endpoint_b_branch_slot: u32,
    preferred_generation_delta: i32,
    _pad1: u32,
}

struct ScaffoldParams {
    rule_count: u32,
    cell_slots: u32,
    pass_index: u32,
    source_phase: u32,
}

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cell_count_buffer: array<u32>;
@group(0) @binding(3) var<storage, read> spatial_grid_counts: array<u32>;
@group(0) @binding(4) var<storage, read> spatial_grid_cells: array<u32>;
@group(0) @binding(5) var<storage, read> cell_grid_indices: array<u32>;

@group(1) @binding(0) var<storage, read> mode_indices: array<u32>;
@group(1) @binding(1) var<storage, read> genome_ids: array<u32>;
@group(1) @binding(2) var<storage, read> development_addresses: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read> death_flags: array<u32>;
@group(1) @binding(4) var<storage, read> parent_lineage_hashes: array<vec2<u32>>;
@group(1) @binding(5) var<storage, read> organism_cell_ids: array<u32>;
// Current connected component through normal developmental adhesions only.
// Barrier-ball scaffold bonds are deliberately excluded by organism_label.wgsl.
@group(1) @binding(9) var<storage, read> organism_labels: array<u32>;

@group(2) @binding(0) var<storage, read_write> adhesion_connections: array<AdhesionConnection>;
@group(2) @binding(1) var<storage, read_write> cell_adhesion_indices: array<atomic<i32>>;
@group(2) @binding(2) var<storage, read_write> next_adhesion_id: array<atomic<u32>>;
@group(2) @binding(3) var<storage, read_write> free_adhesion_slots: array<u32>;
@group(2) @binding(4) var<storage, read_write> adhesion_counts: array<atomic<u32>>;

@group(3) @binding(0) var<storage, read> scaffold_rules: array<ScaffoldRule>;
@group(3) @binding(1) var<uniform> scaffold_params: ScaffoldParams;

const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const BOND_FLAG_BARRIER_BALL: u32 = 2u;
const SELECTOR_ANY: u32 = 0u;
const SELECTOR_MODE: u32 = 1u;
const SELECTOR_LINEAGE: u32 = 2u;
const SELECTOR_LINEAGE_OR_MODE: u32 = 3u;
const SELECTOR_ORGANISM_CELL_ID: u32 = 4u;
const INVALID_ORGANISM_LABEL: u32 = 0xFFFFFFFFu;
const INVALID_CELL: u32 = 0xFFFFFFFFu;
const MAX_CELLS_PER_GRID: u32 = 16u;
const SCAFFOLD_SOURCE_PHASE_COUNT: u32 = 8u;

fn is_in_scaffold_scope(cell_idx: u32, organism_id: u32, component_label: u32) -> bool {
    return development_addresses[cell_idx].x == organism_id
        && organism_labels[cell_idx] == component_label;
}

fn rule_formation_range(rule: ScaffoldRule) -> f32 {
    return max(bitcast<f32>(rule.max_range_bits), 0.0);
}

fn neighborhood_radius(max_range: f32) -> u32 {
    let cell_size = max(physics.grid_cell_size, 0.001);
    return min(u32(ceil(max_range / cell_size)), u32(max(physics.grid_resolution - 1, 0)));
}

fn neighborhood_slot_capacity(max_range: f32) -> u32 {
    let radius = neighborhood_radius(max_range);
    let side = radius * 2u + 1u;
    return side * side * side * MAX_CELLS_PER_GRID;
}

// Map a compact ordinal to one of the fixed spatial-grid occupants in the
// source's formation-range neighborhood. This replaces the old all-live-cell
// scan. Overflow occupants are intentionally deferred until the grid's
// overcrowding cull makes room in the fixed bucket on a later frame.
fn neighborhood_candidate(source: u32, ordinal: u32, max_range: f32) -> u32 {
    let resolution = u32(max(physics.grid_resolution, 1));
    let source_grid = cell_grid_indices[source];
    let source_x = source_grid % resolution;
    let source_y = (source_grid / resolution) % resolution;
    let source_z = source_grid / (resolution * resolution);
    let radius = neighborhood_radius(max_range);
    let side = radius * 2u + 1u;
    let bucket_ordinal = ordinal / MAX_CELLS_PER_GRID;
    let occupant_slot = ordinal % MAX_CELLS_PER_GRID;
    let local_x = bucket_ordinal % side;
    let local_y = (bucket_ordinal / side) % side;
    let local_z = bucket_ordinal / (side * side);
    let grid_x = i32(source_x) + i32(local_x) - i32(radius);
    let grid_y = i32(source_y) + i32(local_y) - i32(radius);
    let grid_z = i32(source_z) + i32(local_z) - i32(radius);
    let resolution_i = i32(resolution);
    if (grid_x < 0 || grid_y < 0 || grid_z < 0
        || grid_x >= resolution_i || grid_y >= resolution_i || grid_z >= resolution_i) {
        return INVALID_CELL;
    }

    let grid_index = u32(grid_x) + u32(grid_y) * resolution
        + u32(grid_z) * resolution * resolution;
    let occupant_count = min(spatial_grid_counts[grid_index], MAX_CELLS_PER_GRID);
    if (occupant_slot >= occupant_count) {
        return INVALID_CELL;
    }
    return spatial_grid_cells[grid_index * MAX_CELLS_PER_GRID + occupant_slot];
}

fn is_within_formation_range(source: u32, candidate: u32, max_range: f32) -> bool {
    let delta = positions[candidate].xyz - positions[source].xyz;
    return dot(delta, delta) <= max_range * max_range;
}

fn selector_matches(cell_idx: u32, kind: u32, mode_idx: u32, hash_lo: u32, hash_hi: u32, branch_slot: u32) -> bool {
    if (kind == SELECTOR_ANY) {
        return true;
    }
    if (kind == SELECTOR_MODE) {
        return mode_indices[cell_idx] == mode_idx;
    }
    if (kind == SELECTOR_ORGANISM_CELL_ID) {
        return organism_cell_ids[cell_idx] == hash_lo;
    }

    let dev = development_addresses[cell_idx];
    let lineage_matches = dev.y == hash_lo && dev.z == hash_hi;
    let cell_branch_slot = dev.w & 0xFFFFu;
    let branch_matches = branch_slot == 0u || cell_branch_slot == branch_slot;
    if (kind == SELECTOR_LINEAGE) {
        return lineage_matches;
    }
    if (kind == SELECTOR_LINEAGE_OR_MODE) {
        return lineage_matches || (mode_indices[cell_idx] == mode_idx && branch_matches);
    }
    return false;
}

fn structural_match_rank(cell_idx: u32, kind: u32, mode_idx: u32, hash_lo: u32, hash_hi: u32, branch_slot: u32) -> u32 {
    if (kind != SELECTOR_LINEAGE_OR_MODE) {
        return select(0u, 3u, selector_matches(cell_idx, kind, mode_idx, hash_lo, hash_hi, branch_slot));
    }

    let dev = development_addresses[cell_idx];
    if (dev.y == hash_lo && dev.z == hash_hi) {
        return 3u;
    }

    if (mode_indices[cell_idx] != mode_idx) {
        return 0u;
    }

    let cell_branch_slot = dev.w & 0xFFFFu;
    if (branch_slot == 0u || cell_branch_slot == branch_slot) {
        return 2u;
    }

    return 1u;
}

fn lineage_depth(cell_idx: u32) -> i32 {
    return i32(development_addresses[cell_idx].w >> 16u);
}

fn abs_i32(value: i32) -> i32 {
    return select(value, -value, value < 0);
}

// Follow the preferred-branch lineage chain from `root_hash_lo/hi` and return the
// index of the current living tip cell, or 0xFFFFFFFF if not found.
//
// Algorithm mirrors the CPU `find_structural_match_in_org` BFS:
//   1. If a cell with the exact lineage hash is alive → it IS the tip (rank 3).
//   2. Otherwise find the living child whose parent_lineage_hash == current_hash
//      and whose branch_slot == preferred_branch_slot.  Follow it forward one level.
//   3. If no preferred child, try any child (fallback).
//   Repeat up to MAX_GENERATIONS times.
// Mirrors CPU find_structural_match_in_org:
//   1. Exact lineage hash match within organism.
//   2. BFS through parent_lineage_hashes following preferred_branch_slot.
//   3. Mode-only fallback (fallback_mode = 0xFFFFFFFF disables it).
fn find_preferred_chain_tip(
    live_slots: u32,
    genome_id: u32,
    organism_id: u32,
    component_label: u32,
    source: u32,
    max_range: f32,
    root_hash_lo: u32,
    root_hash_hi: u32,
    preferred_branch_slot: u32,
    fallback_mode: u32,
) -> u32 {
    var cur_lo = root_hash_lo;
    var cur_hi = root_hash_hi;
    let neighbor_slots = neighborhood_slot_capacity(max_range);

    for (var gen = 0u; gen < 24u; gen++) {
        // Step 1: living cell with exact lineage hash.
        for (var ordinal = 0u; ordinal < neighbor_slots; ordinal++) {
            let candidate = neighborhood_candidate(source, ordinal, max_range);
            if (candidate == INVALID_CELL || candidate >= live_slots) { continue; }
            if (!is_within_formation_range(source, candidate, max_range)) { continue; }
            if (death_flags[candidate] != 0u) { continue; }
            if (genome_ids[candidate] != genome_id) { continue; }
            if (!is_in_scaffold_scope(candidate, organism_id, component_label)) { continue; }
            let dev = development_addresses[candidate];
            if (dev.y == cur_lo && dev.z == cur_hi) {
                return candidate;
            }
        }

        // Step 2: cell with this hash has divided — follow preferred child.
        var preferred_child = 0xFFFFFFFFu;
        var any_child = 0xFFFFFFFFu;
        for (var ordinal = 0u; ordinal < neighbor_slots; ordinal++) {
            let candidate = neighborhood_candidate(source, ordinal, max_range);
            if (candidate == INVALID_CELL || candidate >= live_slots) { continue; }
            if (!is_within_formation_range(source, candidate, max_range)) { continue; }
            if (death_flags[candidate] != 0u) { continue; }
            if (genome_ids[candidate] != genome_id) { continue; }
            if (!is_in_scaffold_scope(candidate, organism_id, component_label)) { continue; }
            let ph = parent_lineage_hashes[candidate];
            if (ph.x != cur_lo || ph.y != cur_hi) { continue; }
            let cell_branch = development_addresses[candidate].w & 0xFFFFu;
            if (cell_branch == preferred_branch_slot) {
                preferred_child = candidate;
                break;
            }
            if (any_child == 0xFFFFFFFFu) {
                any_child = candidate;
            }
        }

        let next_child = select(any_child, preferred_child, preferred_child != 0xFFFFFFFFu);
        if (next_child == 0xFFFFFFFFu) {
            break;
        }
        let next_dev = development_addresses[next_child];
        cur_lo = next_dev.y;
        cur_hi = next_dev.z;
    }

    // Step 3: mode-only fallback — mirrors CPU step 3.
    if (fallback_mode != 0xFFFFFFFFu) {
        for (var ordinal = 0u; ordinal < neighbor_slots; ordinal++) {
            let candidate = neighborhood_candidate(source, ordinal, max_range);
            if (candidate == INVALID_CELL || candidate >= live_slots) { continue; }
            if (!is_within_formation_range(source, candidate, max_range)) { continue; }
            if (death_flags[candidate] != 0u) { continue; }
            if (genome_ids[candidate] != genome_id) { continue; }
            if (!is_in_scaffold_scope(candidate, organism_id, component_label)) { continue; }
            if (mode_indices[candidate] == fallback_mode) { return candidate; }
        }
    }

    return 0xFFFFFFFFu;
}

// Returns the slot of an existing SCAFFOLD (barrier-ball) bond between a and b,
// or 0xFFFFFFFF if none exists. Deliberately ignores normal (non-barrier-ball)
// bonds so that scaffold bonds are always created alongside them rather than
// being suppressed by the early-return.
fn existing_scaffold_connection(a: u32, b: u32) -> u32 {
    let base = a * MAX_ADHESIONS_PER_CELL;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let signed_idx = atomicLoad(&cell_adhesion_indices[base + i]);
        if (signed_idx < 0) { continue; }
        let idx = u32(signed_idx);
        if (idx >= arrayLength(&adhesion_connections)) { continue; }
        let conn = adhesion_connections[idx];
        if (conn.is_active == 0u) { continue; }
        if ((conn.bond_flags & BOND_FLAG_BARRIER_BALL) == 0u) { continue; }
        if ((conn.cell_a_index == a && conn.cell_b_index == b)
            || (conn.cell_a_index == b && conn.cell_b_index == a)) {
            return idx;
        }
    }
    return 0xFFFFFFFFu;
}

fn existing_connection(a: u32, b: u32) -> u32 {
    let base = a * MAX_ADHESIONS_PER_CELL;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let signed_idx = atomicLoad(&cell_adhesion_indices[base + i]);
        if (signed_idx < 0) { continue; }
        let idx = u32(signed_idx);
        if (idx >= arrayLength(&adhesion_connections)) { continue; }
        let conn = adhesion_connections[idx];
        if (conn.is_active == 0u) { continue; }
        if ((conn.cell_a_index == a && conn.cell_b_index == b)
            || (conn.cell_a_index == b && conn.cell_b_index == a)) {
            return idx;
        }
    }
    return 0xFFFFFFFFu;
}

fn allocate_adhesion_slot() -> u32 {
    loop {
        let free_top = atomicLoad(&adhesion_counts[2]);
        if (free_top == 0u) {
            break;
        }
        let result = atomicCompareExchangeWeak(&adhesion_counts[2], free_top, free_top - 1u);
        if (result.exchanged) {
            let slot = free_adhesion_slots[free_top - 1u];
            atomicAdd(&adhesion_counts[1], 1u);
            return slot;
        }
    }

    let slot = atomicAdd(&next_adhesion_id[0], 1u);
    if (slot < arrayLength(&adhesion_connections)) {
        atomicMax(&adhesion_counts[0], slot + 1u);
        atomicAdd(&adhesion_counts[1], 1u);
        return slot;
    }
    return 0xFFFFFFFFu;
}

fn attach_index(cell_idx: u32, adhesion_id: u32) -> bool {
    let base = cell_idx * MAX_ADHESIONS_PER_CELL;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        let result = atomicCompareExchangeWeak(&cell_adhesion_indices[base + i], -1, i32(adhesion_id));
        if (result.exchanged) {
            return true;
        }
    }
    return false;
}

fn detach_index(cell_idx: u32, adhesion_id: u32) {
    let base = cell_idx * MAX_ADHESIONS_PER_CELL;
    for (var i = 0u; i < MAX_ADHESIONS_PER_CELL; i++) {
        atomicCompareExchangeWeak(&cell_adhesion_indices[base + i], i32(adhesion_id), -1);
    }
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let source = gid.x;
    let rule_idx = gid.y;
    let live_slots = min(cell_count_buffer[0], scaffold_params.cell_slots);
    if (source >= live_slots || rule_idx >= scaffold_params.rule_count) {
        return;
    }
    // Persistent scaffold bonds do not need every eligible source to search on
    // every rendered frame. Deterministically shard sources so both endpoint
    // passes cover the whole population once every eight frames without spikes.
    if ((source % SCAFFOLD_SOURCE_PHASE_COUNT) != scaffold_params.source_phase) {
        return;
    }

    let rule = scaffold_rules[rule_idx];
    if (death_flags[source] != 0u) {
        return;
    }
    if (genome_ids[source] != rule.genome_id) {
        return;
    }

    let source_org = development_addresses[source].x;
    let source_component = organism_labels[source];
    if (source_org == 0u || source_component == INVALID_ORGANISM_LABEL) {
        return;
    }
    let max_range = rule_formation_range(rule);
    if (max_range <= 0.0) {
        return;
    }
    let neighbor_slots = neighborhood_slot_capacity(max_range);

    if (rule.endpoint_a_kind == SELECTOR_ORGANISM_CELL_ID && rule.endpoint_b_kind == SELECTOR_ORGANISM_CELL_ID) {
        if (scaffold_params.pass_index == 1u) {
            return;
        }
        // Exact organism-cell IDs have one authoritative source per organism;
        // no global "first source" scan is needed.
        if (organism_cell_ids[source] != rule.endpoint_a_hash_lo) {
            return;
        }

        let endpoint_a = source;
        var endpoint_b = 0xFFFFFFFFu;
        for (var ordinal = 0u; ordinal < neighbor_slots; ordinal++) {
            let candidate = neighborhood_candidate(source, ordinal, max_range);
            if (candidate == INVALID_CELL || candidate >= live_slots) { continue; }
            if (!is_within_formation_range(source, candidate, max_range)) { continue; }
            if (death_flags[candidate] != 0u) { continue; }
            if (genome_ids[candidate] != rule.genome_id) { continue; }
            if (!is_in_scaffold_scope(candidate, source_org, source_component)) { continue; }
            let cell_id = organism_cell_ids[candidate];
            if (cell_id == rule.endpoint_b_hash_lo) {
                endpoint_b = candidate;
                break;
            }
        }
        if (endpoint_b != 0xFFFFFFFFu && endpoint_a != endpoint_b) {
            create_or_update_scaffold_connection(endpoint_a, endpoint_b, rule);
        }
        return;
    }

    let source_kind = select(rule.endpoint_a_kind, rule.endpoint_b_kind, scaffold_params.pass_index == 1u);
    let source_mode = select(rule.endpoint_a_mode, rule.endpoint_b_mode, scaffold_params.pass_index == 1u);
    let source_hash_lo = select(rule.endpoint_a_hash_lo, rule.endpoint_b_hash_lo, scaffold_params.pass_index == 1u);
    let source_hash_hi = select(rule.endpoint_a_hash_hi, rule.endpoint_b_hash_hi, scaffold_params.pass_index == 1u);
    let source_branch_slot = select(rule.endpoint_a_branch_slot, rule.endpoint_b_branch_slot, scaffold_params.pass_index == 1u);
    let target_kind = select(rule.endpoint_b_kind, rule.endpoint_a_kind, scaffold_params.pass_index == 1u);
    let target_mode = select(rule.endpoint_b_mode, rule.endpoint_a_mode, scaffold_params.pass_index == 1u);
    let target_hash_lo = select(rule.endpoint_b_hash_lo, rule.endpoint_a_hash_lo, scaffold_params.pass_index == 1u);
    let target_hash_hi = select(rule.endpoint_b_hash_hi, rule.endpoint_a_hash_hi, scaffold_params.pass_index == 1u);
    let target_branch_slot = select(rule.endpoint_b_branch_slot, rule.endpoint_a_branch_slot, scaffold_params.pass_index == 1u);

    if (!selector_matches(source, source_kind, source_mode, source_hash_lo, source_hash_hi, source_branch_slot)) {
        return;
    }

    // Structural = ByLineageHashOrMode on either endpoint (matches CPU is_structural check).
    // ByLineageHash falls through to the pattern path exactly like the CPU does.
    let is_structural_rule = rule.endpoint_a_kind == SELECTOR_LINEAGE_OR_MODE
        || rule.endpoint_b_kind == SELECTOR_LINEAGE_OR_MODE
        || rule.endpoint_a_kind == SELECTOR_ORGANISM_CELL_ID
        || rule.endpoint_b_kind == SELECTOR_ORGANISM_CELL_ID;
    if (is_structural_rule && scaffold_params.pass_index == 1u) {
        return;
    }

    if (is_structural_rule) {
        if (source_kind != SELECTOR_LINEAGE_OR_MODE || target_kind != SELECTOR_LINEAGE_OR_MODE) {
            return;
        }
        // source_mode / target_mode carry the fallback mode index for step 3.
        let tip_a = find_preferred_chain_tip(
            live_slots, rule.genome_id, source_org, source_component,
            source, max_range,
            source_hash_lo, source_hash_hi, source_branch_slot,
            source_mode,
        );
        if (tip_a != source) {
            return;
        }
        let tip_b = find_preferred_chain_tip(
            live_slots, rule.genome_id, source_org, source_component,
            source, max_range,
            target_hash_lo, target_hash_hi, target_branch_slot,
            target_mode,
        );
        if (tip_b == 0xFFFFFFFFu || tip_b == source) {
            return;
        }
        create_or_update_scaffold_connection(source, tip_b, rule);
        return;
    }

    let same_selector_rule = source_kind == target_kind
        && source_mode == target_mode
        && source_hash_lo == target_hash_lo
        && source_hash_hi == target_hash_hi
        && source_branch_slot == target_branch_slot;
    if (same_selector_rule) {
        let source_cell_id = organism_cell_ids[source];
        var next_cell = 0xFFFFFFFFu;
        var next_cell_id = 0xFFFFFFFFu;
        var first_cell = 0xFFFFFFFFu;
        var first_cell_id = 0xFFFFFFFFu;

        for (var ordinal = 0u; ordinal < neighbor_slots; ordinal++) {
            let candidate = neighborhood_candidate(source, ordinal, max_range);
            if (candidate == INVALID_CELL || candidate >= live_slots) { continue; }
            if (candidate == source) { continue; }
            if (!is_within_formation_range(source, candidate, max_range)) { continue; }
            if (death_flags[candidate] != 0u) { continue; }
            if (genome_ids[candidate] != rule.genome_id) { continue; }
            if (!is_in_scaffold_scope(candidate, source_org, source_component)) { continue; }
            if (!selector_matches(candidate, target_kind, target_mode, target_hash_lo, target_hash_hi, target_branch_slot)) {
                continue;
            }

            let candidate_cell_id = organism_cell_ids[candidate];
            if (candidate_cell_id < first_cell_id || (candidate_cell_id == first_cell_id && candidate < first_cell)) {
                first_cell_id = candidate_cell_id;
                first_cell = candidate;
            }
            if (candidate_cell_id > source_cell_id
                && (candidate_cell_id < next_cell_id || (candidate_cell_id == next_cell_id && candidate < next_cell))) {
                next_cell_id = candidate_cell_id;
                next_cell = candidate;
            }
        }

        let cycle_target = select(first_cell, next_cell, next_cell != 0xFFFFFFFFu);
        if (cycle_target != 0xFFFFFFFFu) {
            create_or_update_scaffold_connection(source, cycle_target, rule);
        }
        return;
    }

    // Pattern bond: match the rule's undirected generation separation first,
    // then use deterministic organism-cell identity as the tie-breaker.
    var best_target = 0xFFFFFFFFu;
    var best_delta_error = 0x7FFFFFFF;
    var best_organism_cell_id = 0xFFFFFFFFu;
    let source_depth = lineage_depth(source);
    let preferred_delta = abs_i32(rule.preferred_generation_delta);

    for (var ordinal = 0u; ordinal < neighbor_slots; ordinal++) {
        let candidate = neighborhood_candidate(source, ordinal, max_range);
        if (candidate == INVALID_CELL || candidate >= live_slots) { continue; }
        if (candidate == source) { continue; }
        if (!is_within_formation_range(source, candidate, max_range)) { continue; }
        if (death_flags[candidate] != 0u) { continue; }
        if (genome_ids[candidate] != rule.genome_id) { continue; }
        if (!is_in_scaffold_scope(candidate, source_org, source_component)) { continue; }
        if (!selector_matches(candidate, target_kind, target_mode, target_hash_lo, target_hash_hi, target_branch_slot)) {
            continue;
        }
        let generation_delta = abs_i32(lineage_depth(candidate) - source_depth);
        let delta_error = abs_i32(generation_delta - preferred_delta);
        let candidate_organism_cell_id = organism_cell_ids[candidate];
        if (delta_error < best_delta_error
            || (delta_error == best_delta_error && candidate_organism_cell_id < best_organism_cell_id)
            || (delta_error == best_delta_error && candidate_organism_cell_id == best_organism_cell_id && candidate < best_target)) {
            best_delta_error = delta_error;
            best_organism_cell_id = candidate_organism_cell_id;
            best_target = candidate;
        }
    }

    if (best_target == 0xFFFFFFFFu) {
        return;
    }

    create_or_update_scaffold_connection(source, best_target, rule);
}

fn create_or_update_scaffold_connection(source: u32, best_target: u32, rule: ScaffoldRule) {
    let existing = existing_scaffold_connection(source, best_target);
    if (existing != 0xFFFFFFFFu) {
        adhesion_connections[existing]._pad = rule.rest_length_bits;
        return;
    }

    if (!is_within_formation_range(source, best_target, rule_formation_range(rule))) {
        return;
    }

    if (existing_connection(source, best_target) != 0xFFFFFFFFu) {
        return;
    }

    let source_mode = mode_indices[source];
    let adhesion_id = allocate_adhesion_slot();
    if (adhesion_id == 0xFFFFFFFFu) {
        return;
    }

    var conn: AdhesionConnection;
    conn.cell_a_index = source;
    conn.cell_b_index = best_target;
    conn.mode_index = source_mode;
    conn.is_active = 1u;
    conn.zone_a = 2u;
    conn.zone_b = 2u;
    conn.bond_flags = BOND_FLAG_BARRIER_BALL;
    conn._align_pad1 = 0u;
    conn.anchor_direction_a = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    conn.anchor_direction_b = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    conn.twist_reference_a = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    conn.twist_reference_b = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    conn.birth_time = physics.current_time;
    conn._pad = rule.rest_length_bits;
    adhesion_connections[adhesion_id] = conn;

    let attached_a = attach_index(source, adhesion_id);
    let attached_b = attach_index(best_target, adhesion_id);
    if (!attached_a || !attached_b) {
        adhesion_connections[adhesion_id].is_active = 0u;
        if (attached_a) { detach_index(source, adhesion_id); }
        if (attached_b) { detach_index(best_target, adhesion_id); }
        atomicSub(&adhesion_counts[1], 1u);
        let free_top = atomicAdd(&adhesion_counts[2], 1u);
        if (free_top < arrayLength(&free_adhesion_slots)) {
            free_adhesion_slots[free_top] = adhesion_id;
        }
    }
}

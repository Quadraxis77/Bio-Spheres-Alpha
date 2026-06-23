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
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> physics: PhysicsParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cell_count_buffer: array<u32>;

@group(1) @binding(0) var<storage, read> mode_indices: array<u32>;
@group(1) @binding(1) var<storage, read> genome_ids: array<u32>;
@group(1) @binding(2) var<storage, read> development_addresses: array<vec4<u32>>;
@group(1) @binding(3) var<storage, read> death_flags: array<u32>;
@group(1) @binding(4) var<storage, read> parent_lineage_hashes: array<vec2<u32>>;
@group(1) @binding(5) var<storage, read> organism_cell_ids: array<u32>;

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

fn is_first_structural_match(cell_idx: u32, live_slots: u32, rule: ScaffoldRule, kind: u32, mode_idx: u32, hash_lo: u32, hash_hi: u32, branch_slot: u32, organism_id: u32) -> bool {
    let rank = structural_match_rank(cell_idx, kind, mode_idx, hash_lo, hash_hi, branch_slot);
    if (rank == 0u) {
        return false;
    }

    for (var candidate = 0u; candidate < live_slots; candidate++) {
        if (candidate == cell_idx) {
            return true;
        }
        if (death_flags[candidate] != 0u) { continue; }
        if (genome_ids[candidate] != rule.genome_id) { continue; }
        if (development_addresses[candidate].x != organism_id) { continue; }

        let candidate_rank = structural_match_rank(candidate, kind, mode_idx, hash_lo, hash_hi, branch_slot);
        if (candidate_rank >= rank) {
            return false;
        }
    }
    return true;
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
    root_hash_lo: u32,
    root_hash_hi: u32,
    preferred_branch_slot: u32,
    fallback_mode: u32,
) -> u32 {
    var cur_lo = root_hash_lo;
    var cur_hi = root_hash_hi;

    for (var gen = 0u; gen < 24u; gen++) {
        // Step 1: living cell with exact lineage hash.
        for (var i = 0u; i < live_slots; i++) {
            if (death_flags[i] != 0u) { continue; }
            if (genome_ids[i] != genome_id) { continue; }
            if (development_addresses[i].x != organism_id) { continue; }
            let dev = development_addresses[i];
            if (dev.y == cur_lo && dev.z == cur_hi) {
                return i;
            }
        }

        // Step 2: cell with this hash has divided — follow preferred child.
        var preferred_child = 0xFFFFFFFFu;
        var any_child = 0xFFFFFFFFu;
        for (var i = 0u; i < live_slots; i++) {
            if (death_flags[i] != 0u) { continue; }
            if (genome_ids[i] != genome_id) { continue; }
            if (development_addresses[i].x != organism_id) { continue; }
            let ph = parent_lineage_hashes[i];
            if (ph.x != cur_lo || ph.y != cur_hi) { continue; }
            let cell_branch = development_addresses[i].w & 0xFFFFu;
            if (cell_branch == preferred_branch_slot) {
                preferred_child = i;
                break;
            }
            if (any_child == 0xFFFFFFFFu) {
                any_child = i;
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
        for (var i = 0u; i < live_slots; i++) {
            if (death_flags[i] != 0u) { continue; }
            if (genome_ids[i] != genome_id) { continue; }
            if (development_addresses[i].x != organism_id) { continue; }
            if (mode_indices[i] == fallback_mode) { return i; }
        }
    }

    return 0xFFFFFFFFu;
}

fn find_best_structural_match(live_slots: u32, rule: ScaffoldRule, kind: u32, mode_idx: u32, hash_lo: u32, hash_hi: u32, branch_slot: u32, organism_id: u32, exclude: u32) -> u32 {
    var best = 0xFFFFFFFFu;
    var best_rank = 0u;
    var best_delta_error = 0x7FFFFFFF;
    let source_depth = lineage_depth(exclude);

    for (var candidate = 0u; candidate < live_slots; candidate++) {
        if (candidate == exclude) { continue; }
        if (death_flags[candidate] != 0u) { continue; }
        if (genome_ids[candidate] != rule.genome_id) { continue; }
        if (development_addresses[candidate].x != organism_id) { continue; }

        let rank = structural_match_rank(candidate, kind, mode_idx, hash_lo, hash_hi, branch_slot);
        let generation_delta = abs_i32(lineage_depth(candidate) - source_depth);
        let delta_error = abs_i32(generation_delta - abs_i32(rule.preferred_generation_delta));
        if (rank > best_rank || (rank == best_rank && delta_error < best_delta_error)) {
            best_rank = rank;
            best_delta_error = delta_error;
            best = candidate;
        }
    }
    return best;
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

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let source = gid.x;
    let rule_idx = gid.y;
    let live_slots = min(cell_count_buffer[0], scaffold_params.cell_slots);
    if (source >= live_slots || rule_idx >= scaffold_params.rule_count) {
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
    if (source_org == 0u) {
        return;
    }

    if (rule.endpoint_a_kind == SELECTOR_ORGANISM_CELL_ID && rule.endpoint_b_kind == SELECTOR_ORGANISM_CELL_ID) {
        if (scaffold_params.pass_index == 1u) {
            return;
        }

        for (var candidate = 0u; candidate < source; candidate++) {
            if (death_flags[candidate] != 0u) { continue; }
            if (genome_ids[candidate] != rule.genome_id) { continue; }
            if (development_addresses[candidate].x == source_org) {
                return;
            }
        }

        var endpoint_a = 0xFFFFFFFFu;
        var endpoint_b = 0xFFFFFFFFu;
        for (var candidate = 0u; candidate < live_slots; candidate++) {
            if (death_flags[candidate] != 0u) { continue; }
            if (genome_ids[candidate] != rule.genome_id) { continue; }
            if (development_addresses[candidate].x != source_org) { continue; }
            let cell_id = organism_cell_ids[candidate];
            if (cell_id == rule.endpoint_a_hash_lo) {
                endpoint_a = candidate;
            }
            if (cell_id == rule.endpoint_b_hash_lo) {
                endpoint_b = candidate;
            }
        }
        if (endpoint_a != 0xFFFFFFFFu && endpoint_b != 0xFFFFFFFFu && endpoint_a != endpoint_b) {
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
            live_slots, rule.genome_id, source_org,
            source_hash_lo, source_hash_hi, source_branch_slot,
            source_mode,
        );
        if (tip_a != source) {
            return;
        }
        let tip_b = find_preferred_chain_tip(
            live_slots, rule.genome_id, source_org,
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

        for (var candidate = 0u; candidate < live_slots; candidate++) {
            if (candidate == source) { continue; }
            if (death_flags[candidate] != 0u) { continue; }
            if (genome_ids[candidate] != rule.genome_id) { continue; }
            if (development_addresses[candidate].x != source_org) { continue; }
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

    for (var candidate = 0u; candidate < live_slots; candidate++) {
        if (candidate == source) { continue; }
        if (death_flags[candidate] != 0u) { continue; }
        if (genome_ids[candidate] != rule.genome_id) { continue; }
        if (development_addresses[candidate].x != source_org) { continue; }
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

    if (existing_connection(source, best_target) != 0xFFFFFFFFu) {
        return;
    }

    let adhesion_id = allocate_adhesion_slot();
    if (adhesion_id == 0xFFFFFFFFu) {
        return;
    }

    var conn: AdhesionConnection;
    conn.cell_a_index = source;
    conn.cell_b_index = best_target;
    conn.mode_index = mode_indices[source];
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
    }
}

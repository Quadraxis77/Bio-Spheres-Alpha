// Phase 4 deterministic topology lifecycle oracle/kernel.
// One invocation consumes a bounded number of already sorted topology jobs.
// Physical invalidation is written by lifecycle shaders before this pass.

struct Params {
    node_count: u32,
    bond_count: u32,
    job_count: u32,
    job_budget: u32,
}

struct Bond {
    stable_lo: u32,
    stable_hi: u32,
    a: u32,
    b: u32,
    resistance: u32,
    flags: u32,
    generation: u32,
    _padding: u32,
}

struct Job {
    kind: u32,
    bond_index: u32,
    cut_a: u32,
    cut_b: u32,
}

struct Control {
    cursor: atomic<u32>,
    topology_generation: atomic<u32>,
    processed: atomic<u32>,
    invalid_jobs: atomic<u32>,
    phase: atomic<u32>,
    head: atomic<u32>,
    tail: atomic<u32>,
    scan: atomic<u32>,
    selected: atomic<u32>,
    best_lo: atomic<u32>,
    best_hi: atomic<u32>,
    stamp: atomic<u32>,
}

struct NodeWork {
    stamp_a: u32,
    stamp_b: u32,
    parent_node: u32,
    parent_bond: u32,
    distance_a_lo: u32,
    distance_a_hi: u32,
    distance_b_lo: u32,
    distance_b_hi: u32,
}

struct U64 {
    lo: u32,
    hi: u32,
}

const INVALID: u32 = 0xffffffffu;
const MAX_ADHESIONS_PER_CELL: u32 = 20u;
const BOND_VALID: u32 = 1u;
const BOND_BACKBONE: u32 = 2u;
const BOND_ACTIVE: u32 = 4u;
const BOND_PENDING: u32 = 8u;
const JOB_ADD: u32 = 0u;
const JOB_REPAIR: u32 = 1u;
const PHASE_IDLE: u32 = 0u;
const PHASE_REPAIR_A: u32 = 1u;
const PHASE_REPAIR_B: u32 = 2u;
const PHASE_REPAIR_SCAN: u32 = 3u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> bonds: array<Bond>;
@group(0) @binding(2) var<storage, read> adjacency: array<u32>;
@group(0) @binding(3) var<storage, read> jobs: array<Job>;
@group(0) @binding(4) var<storage, read_write> control: Control;
@group(0) @binding(5) var<storage, read_write> work: array<NodeWork>;
@group(0) @binding(6) var<storage, read_write> queue: array<u32>;

fn stable_less(left: Bond, right: Bond) -> bool {
    return left.stable_hi < right.stable_hi
        || (left.stable_hi == right.stable_hi && left.stable_lo < right.stable_lo);
}

fn u64_add_u32(value: U64, increment: u32) -> U64 {
    let lo = value.lo + increment;
    return U64(lo, value.hi + select(0u, 1u, lo < value.lo));
}

fn u64_add(left: U64, right: U64) -> U64 {
    let lo = left.lo + right.lo;
    return U64(lo, left.hi + right.hi + select(0u, 1u, lo < left.lo));
}

fn u64_less(left: U64, right: U64) -> bool {
    return left.hi < right.hi || (left.hi == right.hi && left.lo < right.lo);
}

fn u64_equal(left: U64, right: U64) -> bool {
    return left.lo == right.lo && left.hi == right.hi;
}

fn u64_sub(left: U64, right: U64) -> U64 {
    return U64(left.lo - right.lo, left.hi - right.hi - select(0u, 1u, left.lo < right.lo));
}

fn u64_twice(value: U64) -> U64 {
    return U64(value.lo << 1u, (value.hi << 1u) | (value.lo >> 31u));
}

fn usable_active(index: u32) -> bool {
    if (index >= params.bond_count) { return false; }
    let flags = bonds[index].flags;
    return (flags & (BOND_VALID | BOND_BACKBONE | BOND_ACTIVE))
        == (BOND_VALID | BOND_BACKBONE | BOND_ACTIVE);
}

fn other_endpoint(bond: Bond, node: u32) -> u32 {
    return select(bond.a, bond.b, bond.a == node);
}

fn has_active_neighbor(node: u32) -> bool {
    let base = node * MAX_ADHESIONS_PER_CELL;
    for (var slot = 0u; slot < MAX_ADHESIONS_PER_CELL; slot++) {
        if (usable_active(adjacency[base + slot])) { return true; }
    }
    return false;
}

fn bfs(root: u32, goal: u32, stamp: u32, second: bool) -> bool {
    if (root >= params.node_count) { return false; }
    var head = 0u;
    var tail = 1u;
    queue[0] = root;
    if (second) {
        work[root].stamp_b = stamp;
        work[root].distance_b_lo = 0u;
        work[root].distance_b_hi = 0u;
    } else {
        work[root].stamp_a = stamp;
        work[root].distance_a_lo = 0u;
        work[root].distance_a_hi = 0u;
        work[root].parent_node = root;
        work[root].parent_bond = INVALID;
    }
    loop {
        if (head >= tail) { break; }
        let node = queue[head];
        head++;
        if (node == goal) { return true; }
        let base = node * MAX_ADHESIONS_PER_CELL;
        for (var slot = 0u; slot < MAX_ADHESIONS_PER_CELL; slot++) {
            let bond_index = adjacency[base + slot];
            if (!usable_active(bond_index)) { continue; }
            let bond = bonds[bond_index];
            let neighbor = other_endpoint(bond, node);
            if (neighbor >= params.node_count) { continue; }
            if (second) {
                if (work[neighbor].stamp_b == stamp) { continue; }
                work[neighbor].stamp_b = stamp;
                let distance = u64_add_u32(
                    U64(work[node].distance_b_lo, work[node].distance_b_hi),
                    bond.resistance
                );
                work[neighbor].distance_b_lo = distance.lo;
                work[neighbor].distance_b_hi = distance.hi;
            } else {
                if (work[neighbor].stamp_a == stamp) { continue; }
                work[neighbor].stamp_a = stamp;
                let distance = u64_add_u32(
                    U64(work[node].distance_a_lo, work[node].distance_a_hi),
                    bond.resistance
                );
                work[neighbor].distance_a_lo = distance.lo;
                work[neighbor].distance_a_hi = distance.hi;
                work[neighbor].parent_node = node;
                work[neighbor].parent_bond = bond_index;
            }
            if (tail < params.node_count) {
                queue[tail] = neighbor;
                tail++;
            }
        }
    }
    return goal != INVALID && goal < params.node_count
        && select(work[goal].stamp_a, work[goal].stamp_b, second) == stamp;
}

fn commit_addition(bond_index: u32, stamp: u32) -> bool {
    if (bond_index >= params.bond_count) { return false; }
    let candidate = bonds[bond_index];
    if ((candidate.flags & (BOND_VALID | BOND_BACKBONE | BOND_PENDING))
        != (BOND_VALID | BOND_BACKBONE | BOND_PENDING)
        || candidate.a >= params.node_count || candidate.b >= params.node_count) {
        return false;
    }
    bonds[bond_index].flags &= ~BOND_PENDING;
    // Developmental leaf attachment is the dominant lifecycle event and does
    // not need a component search when either endpoint is isolated.
    if (!has_active_neighbor(candidate.a) || !has_active_neighbor(candidate.b)) {
        bonds[bond_index].flags |= BOND_ACTIVE;
        return true;
    }
    if (!bfs(candidate.a, candidate.b, stamp, false)) {
        bonds[bond_index].flags |= BOND_ACTIVE;
        return true;
    }

    var node = candidate.b;
    var path_resistance = U64(0u, 0u);
    var maximum = 0u;
    loop {
        if (node == candidate.a) { break; }
        let edge = work[node].parent_bond;
        if (edge == INVALID) { return false; }
        path_resistance = u64_add_u32(path_resistance, bonds[edge].resistance);
        maximum = max(maximum, bonds[edge].resistance);
        node = work[node].parent_node;
    }
    if (!u64_less(U64(candidate.resistance, 0u), path_resistance)) { return false; }

    var demoted_edge = INVALID;
    var best_midpoint_delta = U64(0xffffffffu, 0xffffffffu);
    var suffix = U64(0u, 0u);
    node = candidate.b;
    loop {
        if (node == candidate.a) { break; }
        let edge = work[node].parent_bond;
        let edge_resistance = bonds[edge].resistance;
        if (edge_resistance == maximum) {
            let center_twice_from_b = u64_add_u32(u64_twice(suffix), edge_resistance);
            var delta = u64_sub(path_resistance, center_twice_from_b);
            if (u64_less(path_resistance, center_twice_from_b)) {
                delta = u64_sub(center_twice_from_b, path_resistance);
            }
            if (demoted_edge == INVALID || u64_less(delta, best_midpoint_delta)
                || (u64_equal(delta, best_midpoint_delta) && stable_less(bonds[edge], bonds[demoted_edge]))) {
                demoted_edge = edge;
                best_midpoint_delta = delta;
            }
        }
        suffix = u64_add_u32(suffix, edge_resistance);
        node = work[node].parent_node;
    }
    if (demoted_edge == INVALID) { return false; }
    bonds[demoted_edge].flags &= ~BOND_ACTIVE;
    bonds[bond_index].flags |= BOND_ACTIVE;
    return true;
}

fn commit_repair(job: Job, stamp: u32) -> bool {
    if (job.cut_a >= params.node_count || job.cut_b >= params.node_count) {
        return false;
    }
    _ = bfs(job.cut_a, INVALID, stamp, false);
    _ = bfs(job.cut_b, INVALID, stamp, true);

    var selected = INVALID;
    var best_resistance = U64(0xffffffffu, 0xffffffffu);
    for (var index = 0u; index < params.bond_count; index++) {
        let bond = bonds[index];
        if ((bond.flags & (BOND_VALID | BOND_BACKBONE)) != (BOND_VALID | BOND_BACKBONE)
            || (bond.flags & (BOND_ACTIVE | BOND_PENDING)) != 0u) {
            continue;
        }
        var left = bond.a;
        var right = bond.b;
        if (left >= params.node_count || right >= params.node_count) { continue; }
        if (!(work[left].stamp_a == stamp && work[right].stamp_b == stamp)) {
            if (work[right].stamp_a == stamp && work[left].stamp_b == stamp) {
                let swap = left;
                left = right;
                right = swap;
            } else {
                continue;
            }
        }
        let replacement = u64_add(
            u64_add_u32(U64(work[left].distance_a_lo, work[left].distance_a_hi), bond.resistance),
            U64(work[right].distance_b_lo, work[right].distance_b_hi)
        );
        if (selected == INVALID || u64_less(replacement, best_resistance)
            || (u64_equal(replacement, best_resistance) && stable_less(bond, bonds[selected]))) {
            selected = index;
            best_resistance = replacement;
        }
    }
    if (selected == INVALID) { return false; }
    bonds[selected].flags |= BOND_ACTIVE;
    return true;
}

fn begin_repair(job: Job, stamp: u32) {
    queue[0] = job.cut_a;
    work[job.cut_a].stamp_a = stamp;
    work[job.cut_a].distance_a_lo = 0u;
    work[job.cut_a].distance_a_hi = 0u;
    atomicStore(&control.head, 0u);
    atomicStore(&control.tail, 1u);
    atomicStore(&control.scan, 0u);
    atomicStore(&control.selected, INVALID);
    atomicStore(&control.best_lo, 0xffffffffu);
    atomicStore(&control.best_hi, 0xffffffffu);
    atomicStore(&control.stamp, stamp);
    atomicStore(&control.phase, PHASE_REPAIR_A);
}

fn repair_bfs_step(second: bool, stamp: u32) -> bool {
    let head = atomicLoad(&control.head);
    let tail = atomicLoad(&control.tail);
    if (head >= tail) { return true; }
    let node = queue[head];
    atomicStore(&control.head, head + 1u);
    let base = node * MAX_ADHESIONS_PER_CELL;
    var next_tail = tail;
    for (var slot = 0u; slot < MAX_ADHESIONS_PER_CELL; slot++) {
        let bond_index = adjacency[base + slot];
        if (!usable_active(bond_index)) { continue; }
        let bond = bonds[bond_index];
        let neighbor = other_endpoint(bond, node);
        if (neighbor >= params.node_count) { continue; }
        if (second) {
            if (work[neighbor].stamp_b == stamp) { continue; }
            work[neighbor].stamp_b = stamp;
            let distance = u64_add_u32(U64(work[node].distance_b_lo, work[node].distance_b_hi), bond.resistance);
            work[neighbor].distance_b_lo = distance.lo;
            work[neighbor].distance_b_hi = distance.hi;
        } else {
            if (work[neighbor].stamp_a == stamp) { continue; }
            work[neighbor].stamp_a = stamp;
            let distance = u64_add_u32(U64(work[node].distance_a_lo, work[node].distance_a_hi), bond.resistance);
            work[neighbor].distance_a_lo = distance.lo;
            work[neighbor].distance_a_hi = distance.hi;
        }
        if (next_tail < params.node_count) {
            queue[next_tail] = neighbor;
            next_tail++;
        }
    }
    atomicStore(&control.tail, next_tail);
    return atomicLoad(&control.head) >= next_tail;
}

fn repair_scan_step(stamp: u32) -> bool {
    let index = atomicLoad(&control.scan);
    if (index >= params.bond_count) { return true; }
    atomicStore(&control.scan, index + 1u);
    let bond = bonds[index];
    if ((bond.flags & (BOND_VALID | BOND_BACKBONE)) == (BOND_VALID | BOND_BACKBONE)
        && (bond.flags & (BOND_ACTIVE | BOND_PENDING)) == 0u
        && bond.a < params.node_count && bond.b < params.node_count) {
        var left = bond.a;
        var right = bond.b;
        if (!(work[left].stamp_a == stamp && work[right].stamp_b == stamp)) {
            if (work[right].stamp_a == stamp && work[left].stamp_b == stamp) {
                let swap = left; left = right; right = swap;
            } else {
                return atomicLoad(&control.scan) >= params.bond_count;
            }
        }
        let replacement = u64_add(
            u64_add_u32(U64(work[left].distance_a_lo, work[left].distance_a_hi), bond.resistance),
            U64(work[right].distance_b_lo, work[right].distance_b_hi)
        );
        let best = U64(atomicLoad(&control.best_lo), atomicLoad(&control.best_hi));
        let selected = atomicLoad(&control.selected);
        if (selected == INVALID || u64_less(replacement, best)
            || (u64_equal(replacement, best) && stable_less(bond, bonds[selected]))) {
            atomicStore(&control.selected, index);
            atomicStore(&control.best_lo, replacement.lo);
            atomicStore(&control.best_hi, replacement.hi);
        }
    }
    return atomicLoad(&control.scan) >= params.bond_count;
}

@compute @workgroup_size(1)
fn process_jobs(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) { return; }
    var cursor = atomicLoad(&control.cursor);
    var operations = 0u;
    loop {
        if (operations >= params.job_budget || cursor >= params.job_count) { break; }
        let job = jobs[cursor];
        let stamp = cursor + 1u;
        var changed = false;
        if (job.kind == JOB_ADD) {
            changed = commit_addition(job.bond_index, stamp);
            cursor++;
            atomicStore(&control.cursor, cursor);
            atomicAdd(&control.processed, 1u);
        } else if (job.kind == JOB_REPAIR) {
            var phase = atomicLoad(&control.phase);
            if (phase == PHASE_IDLE) {
                begin_repair(job, stamp);
                phase = PHASE_REPAIR_A;
            }
            if (phase == PHASE_REPAIR_A) {
                if (repair_bfs_step(false, stamp)) {
                    queue[0] = job.cut_b;
                    work[job.cut_b].stamp_b = stamp;
                    work[job.cut_b].distance_b_lo = 0u;
                    work[job.cut_b].distance_b_hi = 0u;
                    atomicStore(&control.head, 0u);
                    atomicStore(&control.tail, 1u);
                    atomicStore(&control.phase, PHASE_REPAIR_B);
                }
            } else if (phase == PHASE_REPAIR_B) {
                if (repair_bfs_step(true, stamp)) {
                    atomicStore(&control.scan, 0u);
                    atomicStore(&control.phase, PHASE_REPAIR_SCAN);
                }
            } else if (phase == PHASE_REPAIR_SCAN) {
                if (repair_scan_step(stamp)) {
                    let selected = atomicLoad(&control.selected);
                    if (selected != INVALID) {
                        bonds[selected].flags |= BOND_ACTIVE;
                        changed = true;
                    }
                    cursor++;
                    atomicStore(&control.cursor, cursor);
                    atomicAdd(&control.processed, 1u);
                    atomicStore(&control.phase, PHASE_IDLE);
                }
            }
        } else {
            atomicAdd(&control.invalid_jobs, 1u);
            cursor++;
            atomicStore(&control.cursor, cursor);
            atomicAdd(&control.processed, 1u);
        }
        if (changed) {
            let old = atomicAdd(&control.topology_generation, 1u);
            if (old == 0xffffffffu) {
                atomicStore(&control.topology_generation, 1u);
            }
        }
        operations++;
    }
}

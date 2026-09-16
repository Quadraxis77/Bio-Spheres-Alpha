// Reserve every resource before committing a fusion. Detection only locks parents;
// failed allocations release both locks. This pass is serial over at most 64 pairs,
// avoiding speculative ring-pop races and requiring no CPU acknowledgement.
struct FusionLimits { mode_capacity: u32, genome_capacity: u32, _pad0: u32, _pad1: u32 }
struct FusionPlan { destination: vec4<u32>, parents: vec4<u32> }
@group(0) @binding(0) var<storage, read> events: array<u32>;
@group(0) @binding(1) var<storage, read_write> plans: array<FusionPlan>;
@group(0) @binding(2) var<storage, read_write> metadata: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> refs: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> ring: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> free_genomes: array<u32>;
@group(0) @binding(6) var<storage, read_write> deaths: array<u32>;
@group(0) @binding(7) var<uniform> limits: FusionLimits;
@group(0) @binding(8) var<storage, read_write> parentage: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read_write> initial_orientations: array<vec4<f32>>;
const INVALID: u32 = 0xffffffffu;

fn allocate_genome() -> u32 {
    let head = atomicLoad(&ring[0]);
    if (head < atomicLoad(&ring[1])) {
        atomicStore(&ring[0], head + 1u);
        return free_genomes[head % limits.genome_capacity];
    }
    for (var attempt = 0u; attempt < 256u; attempt++) {
        let id = atomicLoad(&ring[2]);
        if (id >= limits.genome_capacity) { return INVALID; }
        atomicStore(&ring[2], id + 1u);
        if (atomicLoad(&refs[id]) == 0u) { return id; }
    }
    return INVALID;
}

@compute @workgroup_size(1)
fn main() {
    for (var event = 0u; event < min(events[0], 64u); event++) {
        plans[event].destination = vec4<u32>(INVALID, 0u, 0u, 0u);
        let e = 1u + event * 8u;
        let a = metadata[events[e + 2u]];
        let b = metadata[events[e + 3u]];
        let count = max(a.x, b.x);
        if (count == 0u || count > 128u) {
            deaths[events[e]] = 0u;
            deaths[events[e + 1u]] = 0u;
            continue;
        }
        let id = allocate_genome();
        if (id == INVALID) {
            deaths[events[e]] = 0u;
            deaths[events[e + 1u]] = 0u;
            continue;
        }
        let recycled = metadata[id];
        let end = atomicLoad(&ring[3]);
        var base = end;
        // GC invalidates ranges reclaimed from the tail. An intact free block
        // below that tail can be reused safely without advancing the allocator.
        let reuse = recycled.x == 0u && recycled.z >= count
            && recycled.y <= limits.mode_capacity
            && count <= limits.mode_capacity - recycled.y
            && recycled.y + recycled.z <= end;
        if (reuse) {
            base = recycled.y;
        } else {
            if (base > limits.mode_capacity || count > limits.mode_capacity - base) {
                // Return the uncommitted ID. No parent or mode data was modified.
                let tail = atomicLoad(&ring[1]);
                free_genomes[tail % limits.genome_capacity] = id;
                atomicStore(&ring[1], tail + 1u);
                deaths[events[e]] = 0u;
                deaths[events[e + 1u]] = 0u;
                continue;
            }
            atomicStore(&ring[3], base + count);
        }
        atomicMax(&ring[4], base + count);
        let initial = min(a.z, count - 1u);
        metadata[id] = vec4<u32>(count, base, initial, events[e + 2u]);
        atomicStore(&refs[id], 1u);
        initial_orientations[id] = initial_orientations[events[e + 2u]];
        parentage[id] = vec2<u32>(events[e + 2u], events[e + 3u]);
        plans[event].destination = vec4<u32>(id, base, count, initial);
        plans[event].parents = vec4<u32>(a.y, a.x, b.y, b.x);
    }
}

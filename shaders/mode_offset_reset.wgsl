// Reclaim only the unused tail, preserving the authored prefix. Invalidate free
// blocks in that tail before future allocation so they cannot alias new genomes.
// These partitions match mutation::{AUTHORED_GENOME_RESERVE, AUTHORED_MODE_RESERVE}.
@group(0) @binding(0) var<storage, read_write> genome_ring_state: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> metadata: array<vec4<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let max_offset = max(65536u, atomicLoad(&genome_ring_state[4]));
    if (gid.x == 0u) {
        atomicStore(&genome_ring_state[3], max_offset);
        atomicStore(&genome_ring_state[2], 4096u);
    }
    if (gid.x < arrayLength(&metadata)) {
        let m = metadata[gid.x];
        if (m.x == 0u && m.y + m.z > max_offset) {
            metadata[gid.x] = vec4<u32>(0u);
        }
    }
}

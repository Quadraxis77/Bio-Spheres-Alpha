// Finalize a bounded instance count after particle extraction/spawning.
struct Limits { vertices: u32, capacity: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> counter: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@group(0) @binding(2) var<uniform> limits: Limits;
@compute @workgroup_size(1)
fn main() {
    args[0] = limits.vertices;
    args[1] = min(counter[0], limits.capacity);
    args[2] = 0u;
    args[3] = 0u;
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch: array<u32>;
@compute @workgroup_size(1)
fn prepare_dispatch() {
    let groups=(min(counts[0],params.count)+127u)/128u;
    dispatch[0]=groups;dispatch[1]=1u;dispatch[2]=1u;
    dispatch[3]=groups;dispatch[4]=1u;dispatch[5]=1u;
}

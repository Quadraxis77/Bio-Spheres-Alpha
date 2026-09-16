@compute @workgroup_size(64)
fn copy_types(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.y >= min(events[0], 64u)) { return; }
    let p = plans[gid.y];
    let local = gid.x;
    if (p.destination.x == 0xffffffffu || local >= p.destination.z) { return; }
    let e = 1u + gid.y * 8u;
    let choose_b = local >= p.parents.y || (local < p.parents.w &&
        (hash(ids[events[e]] ^ hash(ids[events[e + 1u]]) ^ hash(local)) & 1u) != 0u);
    let src = select(p.parents.x, p.parents.z, choose_b) + local;
    types[p.destination.y + local] = select(types[src], 10u, local == p.destination.w);
}

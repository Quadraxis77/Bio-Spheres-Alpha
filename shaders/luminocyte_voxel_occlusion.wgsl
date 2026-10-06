fn visible_between(center: vec3<f32>, receiver: vec3<f32>) -> bool {
    let delta = receiver - center;
    let base = vec3<i32>(floor(center));
    let p = vec3<i32>(floor(receiver));
    let steps = max(i32(ceil(length(delta) * 2.0)), 1);
    for (var s = 1; s < steps; s++) {
        let q = vec3<i32>(floor(center + delta * (f32(s) / f32(steps))));
        if all(q == base) || all(q == p) { continue; }
        if inside(q) {
            let i = index(q);
            if solid[i] != 0u || cell_occupancy[i] != 0u { return false; }
        }
    }
    return true;
}

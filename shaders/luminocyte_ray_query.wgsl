@group(0) @binding(7) var cave_acceleration: acceleration_structure;

// Distance to the exit of the containing voxel in the supplied direction.
fn voxel_exit(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let cell_min = floor(origin);
    let face = select(cell_min, cell_min + 1.0, direction > vec3<f32>(0.0));
    let safe_dir = select(vec3<f32>(1.0), direction, abs(direction) > vec3<f32>(0.000001));
    let distances = select(vec3<f32>(1e20), (face - origin) / safe_dir, abs(direction) > vec3<f32>(0.000001));
    return min(distances.x, min(distances.y, distances.z));
}
fn visible_between(center: vec3<f32>, receiver: vec3<f32>) -> bool {
    let delta = receiver - center;
    let distance = length(delta);
    if distance < 0.0001 { return true; }
    let direction = delta / distance;
    let base = vec3<i32>(floor(center));
    let receiver_voxel = vec3<i32>(floor(receiver));
    // The cave is queried through the acceleration structure. Dynamic cells use
    // the occupancy volume built earlier in this frame, so both backends cast
    // the same cell shadows without rebuilding the TLAS as cells move.
    let occupancy_steps = max(i32(ceil(distance * 2.0)), 1);
    for (var s = 1; s < occupancy_steps; s++) {
        let q = vec3<i32>(floor(center + delta * (f32(s) / f32(occupancy_steps))));
        if all(q == base) || all(q == receiver_voxel) { continue; }
        if inside(q) && cell_occupancy[index(q)] != 0u { return false; }
    }
    // Exclude source and receiver voxels, so a wall can receive light itself.
    let start = voxel_exit(center, direction) + 0.0001;
    let end = distance - voxel_exit(receiver, -direction) - 0.0001;
    if end <= start { return true; }
    // A ray starting inside a solid run need not cross its triangles.
    let first = vec3<i32>(floor(center + direction * start));
    if inside(first) && solid[index(first)] != 0u { return false; }
    var query: ray_query;
    // Opaque, terminate on first hit. No back-face culling for cave walls.
    rayQueryInitialize(&query, cave_acceleration, RayDesc(5u, 255u, start, end, center, direction));
    while rayQueryProceed(&query) {}
    return rayQueryGetCommittedIntersection(&query).kind == 0u;
}

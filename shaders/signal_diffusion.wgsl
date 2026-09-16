// Each invocation gathers balanced transfers from one immutable field and
// writes its own next value. Shared edge records guarantee symmetric weights.
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> current: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> next: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> production: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> edges: array<u32>;
@group(0) @binding(5) var<storage, read> adjacency: array<i32>;
@group(0) @binding(6) var<storage, read> cells: array<CellState>;

@group(0) @binding(7) var<storage, read> activity: u32;

@compute @workgroup_size(128)
fn transport(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell = id.x;
    if (cell >= params.count) { return; }
    if (activity == 0u || cells[cell].live == 0u) {
        for(var group=0u;group<4u;group++){next[cell*4u+group]=vec4<f32>(0.0);} return;
    }
    var neighbors: array<vec4<f32>,4>;
    var degree = 0u;
    for (var slot = 0u; slot < params.degree; slot++) {
        let edge = adjacency[cell * params.degree + slot];
        if (edge < 0) { continue; }
        let base = u32(edge) * 26u;
        if (base + 25u >= arrayLength(&edges)) { continue; }
        let a = edges[base]; let b = edges[base + 1u];
        if (edges[base + 3u] == 0u || (edges[base + 6u] & 2u) != 0u || a == b) { continue; }
        if (a >= params.count || b >= params.count || (a != cell && b != cell)) { continue; }
        let other = select(a, b, a == cell);
        if (cells[other].live == 0u) { continue; }
        for(var group=0u;group<4u;group++){neighbors[group] += current[other * 4u + group];}
        degree++;
    }
    let weight = params.dt * params.conductance;
    // Convex form of C + dt*g*sum(C_neighbor-C); no negative clamp.
    for(var group=0u;group<4u;group++){
        let offset=cell*4u+group;
        let transported = current[offset] * (1.0 - weight * f32(degree)) + neighbors[group] * weight;
        let degraded = transported * params.retention;
        next[offset] = degraded + params.dt * params.production_scale * production[offset];
    }
}

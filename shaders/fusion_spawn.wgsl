// Appended to the shared insertion shader. The offspring replaces parent A;
// parent B remains flagged as fused and is recycled by the normal death scan.
struct FusionPlan { destination: vec4<u32>, parents: vec4<u32> }
@group(3) @binding(0) var<storage, read> fusion_plans: array<FusionPlan>;
@group(3) @binding(1) var<storage, read> fusion_events: array<u32>;
@group(3) @binding(2) var<storage, read> fusion_v0: array<vec4<f32>>;
@group(3) @binding(3) var<storage, read> fusion_v1: array<vec4<f32>>;
@group(3) @binding(4) var<storage, read> fusion_v2: array<vec4<f32>>;

@group(3) @binding(5) var<storage, read> initial_orientations: array<vec4<f32>>;

@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let event = gid.x;
    if (event >= min(fusion_events[0], 64u)) { return; }
    let plan = fusion_plans[event].destination;
    if (plan.x == 0xffffffffu) { return; }
    let e = 1u + event * 8u;
    let slot = fusion_events[e];
    let mode = plan.y + plan.w;
    let v0 = fusion_v0[mode];
    let v1 = fusion_v1[mode];
    let v2 = fusion_v2[mode];
    var newborn: CellInsertionParams;
    newborn.position = vec3<f32>(bitcast<f32>(fusion_events[e + 4u]),
        bitcast<f32>(fusion_events[e + 5u]), bitcast<f32>(fusion_events[e + 6u]));
    newborn.mass = 1.0;
    newborn.rotation = initial_orientations[plan.x];
    newborn.genome_id = plan.x;
    newborn.mode_index = mode;
    newborn.birth_time = params.current_time;
    newborn.split_interval = v0.w;
    newborn.split_mass = max(0.0, (v1.x - 1.0) * 100.0);
    newborn.stiffness = v0.z;
    newborn.nutrient_gain_rate = v0.x;
    newborn.max_cell_size = v0.y;
    newborn.max_splits = select(u32(max(v2.x, 0.0)), 0xffffffffu, v2.x < 0.0);
    newborn.cell_type = 10u;
    newborn.initial_reserve = fusion_events[e + 7u];
    // initialize_cell assigns a fresh GPU cell/organism ID, clears angular
    // velocity and developmental state, and initializes all three physics slots.
    initialize_cell(slot, newborn);
}

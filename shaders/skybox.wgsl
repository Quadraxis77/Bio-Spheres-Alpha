// Procedural Starfield Skybox
//
// Simple black background with procedural stars.
// Stars are placed using 3D Voronoi cell noise directly on the unit sphere -
// no longitude/latitude mapping, no seams, no poles.

struct CameraUniforms {
    inv_view_rot_proj: mat4x4<f32>,
    time:  f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct SkyboxParams {
    star_density:    f32,
    star_brightness: f32,
    twinkle_speed:   f32,
    twinkle_amount:  f32,
    nebula_intensity:    f32,
    milky_way_intensity: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> params: SkyboxParams;

// -- Vertex shader -------------------------------------------------------------

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32((vi >> 1u) & 1u) * 4 - 1);
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 1.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// -- Hash helpers --------------------------------------------------------------

fn hash3(p: vec3<f32>) -> vec3<f32> {
    var q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453123);
}

fn hash1(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453123);
}

// -- Stars via 3D Voronoi on the unit sphere -----------------------------------
//
// Divide space into a regular grid of cubes. Each cube may contain one star
// at a random position inside it. We check the 3x3x3 neighborhood of the
// cube the ray passes through.
//
// Because we normalize the star position to the unit sphere before measuring
// angular distance, there are no seams or poles - the grid is in Cartesian
// space, not angular space.

fn stars(dir: vec3<f32>, scale: f32, density: f32, time: f32) -> vec3<f32> {
    let p    = dir * scale;
    let cell = floor(p);

    var color = vec3<f32>(0.0);

    for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
        let nc  = cell + vec3<f32>(f32(dx), f32(dy), f32(dz));
        let rnd = hash3(nc);

        // Skip cells that don't contain a star
        if (rnd.x > density) { continue; }

        // Star sits at a random point inside the cell, projected onto the sphere
        let star_pos = normalize(nc + rnd * 0.999);

        // Cosine of the angle between view ray and star direction
        let cos_a = dot(dir, star_pos);
        if (cos_a < 0.9995) { continue; } // outside a ~2 deg cone - early out

        // Convert to angular distance in radians
        let angle = acos(clamp(cos_a, -1.0, 1.0));

        // Star angular radius - sharp pinpoint
        let radius = 0.0006 + rnd.y * 0.0008;
        if (angle > radius) { continue; }

        // Smooth disc falloff
        let t = 1.0 - angle / radius;
        var brightness = t * t * params.star_brightness;

        // Magnitude: most stars dim, a few bright
        let mag = pow(rnd.z, 2.0);
        brightness *= 0.15 + mag * 0.85;

        // Twinkle
        let phase = rnd.y * 6.2832;
        brightness *= 1.0 + params.twinkle_amount *
            sin(time * params.twinkle_speed + phase);

        // Color: mostly white, occasional tint
        var star_color: vec3<f32>;
        let c = hash1(nc + vec3<f32>(5.1, 2.7, 8.3));
        if      (c < 0.10) { star_color = vec3<f32>(0.75, 0.88, 1.00); } // blue-white
        else if (c < 0.18) { star_color = vec3<f32>(1.00, 0.95, 0.80); } // warm white
        else               { star_color = vec3<f32>(1.00, 1.00, 1.00); } // white

        color += star_color * brightness;
    }}}

    return color;
}

// -- Fragment shader -----------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct view direction from rotation-only inverse view-proj.
    // No camera_pos subtraction - no precision loss, no shaking.
    let ndc    = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    let wh     = camera.inv_view_rot_proj * ndc;
    let dir    = normalize(wh.xyz / wh.w);

    // Black background
    var color = vec3<f32>(0.0);

    // Single star layer - the 3x3x3 neighbor search is expensive per pixel,
    // so one layer at moderate density gives a good starfield without tanking FPS.
    color += stars(dir, 8.0, params.star_density * 0.3, camera.time);

    return vec4<f32>(color, 1.0);
}

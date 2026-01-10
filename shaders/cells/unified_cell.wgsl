// Unified Cell Appearance Shader
//
// Renders all cell types with a common base sphere and type-specific modifications.
// Cell type determines which visual features are applied:
// - Type 0 (Test): Basic sphere with membrane effects
// - Type 1 (Flagellocyte): Sphere with animated helical tail
// - Future types will add more visual features
//
// Instance Data Layout (96 bytes):
// - position: vec3<f32> (12 bytes)
// - radius: f32 (4 bytes)
// - color: vec4<f32> (16 bytes)
// - visual_params: vec4<f32> (16 bytes) - specular, power, fresnel, emissive
// - rotation: vec4<f32> (16 bytes)
// - type_data_0: vec4<f32> (16 bytes) - type-specific params
// - type_data_1: vec4<f32> (16 bytes) - type-specific params
//
// Type Data Layout by Cell Type:
// Test (type 0):
//   type_data_0: (noise_scale, noise_strength, noise_speed, anim_offset)
//   type_data_1: (cell_type, 0, 0, 0)
// Flagellocyte (type 1):
//   type_data_0: (tail_length, tail_thickness, tail_amplitude, tail_frequency)
//   type_data_1: (tail_speed, tail_taper, tail_segments, cell_type)

// Camera uniform
struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
}

// Lighting uniform
struct Lighting {
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<uniform> lighting: Lighting;

// Instance data from vertex buffer
struct InstanceInput {
    @location(0) position: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
    @location(3) visual_params: vec4<f32>,
    @location(4) rotation: vec4<f32>,
    @location(5) type_data_0: vec4<f32>,
    @location(6) type_data_1: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
    @location(4) visual_params: vec4<f32>,
    @location(5) uv: vec2<f32>,
    @location(6) type_data_0: vec4<f32>,
    @location(7) type_data_1: vec4<f32>,
    @location(8) rotation: vec4<f32>,
    @location(9) cam_right: vec3<f32>,
    @location(10) cam_up: vec3<f32>,
}

// Billboard quad vertices
const QUAD_VERTICES: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Cell type constants
const CELL_TYPE_TEST: u32 = 0u;
const CELL_TYPE_FLAGELLOCYTE: u32 = 1u;

// ============================================================================
// Helper Functions
// ============================================================================

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + ((uv * q.w) + uuv) * 2.0;
}

fn projectToScreen(point3D: vec3<f32>, camRight: vec3<f32>, camUp: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(dot(point3D, camRight), dot(point3D, camUp));
}

fn sdTaperedCapsule2(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, ra: f32, rb: f32) -> vec2<f32> {
    let pa = p - a;
    let ba = b - a;
    let ba_len_sq = dot(ba, ba);
    if (ba_len_sq < 0.0001) {
        return vec2<f32>(length(pa) - ra, 0.0);
    }
    let h = clamp(dot(pa, ba) / ba_len_sq, 0.0, 1.0);
    let r = mix(ra, rb, h);
    return vec2<f32>(length(pa - ba * h) - r, h);
}

fn flagellaNormal2D(
    p: vec2<f32>,
    closestPoint: vec2<f32>,
    camRight: vec3<f32>,
    camUp: vec3<f32>,
    camFwd: vec3<f32>,
    radius: f32
) -> vec3<f32> {
    let toSurface = p - closestPoint;
    let dist2D = length(toSurface);
    if (dist2D < 0.001) {
        return -camFwd;
    }
    let dir2D = toSurface / dist2D;
    let r = clamp(dist2D / radius, 0.0, 1.0);
    let z = sqrt(max(1.0 - r * r, 0.0));
    return normalize(camRight * dir2D.x * r + camUp * dir2D.y * r - camFwd * z);
}


// ============================================================================
// Flagella SDF (for Flagellocyte type)
// ============================================================================

fn sdFlagella2D(
    p: vec2<f32>,
    flagellaDir3D: vec3<f32>,
    camRight: vec3<f32>,
    camUp: vec3<f32>,
    camFwd: vec3<f32>,
    ro: vec3<f32>,
    cellCenter: vec3<f32>,
    time: f32,
    radius: f32,
    tail_length: f32,
    tail_thickness: f32,
    tail_amplitude: f32,
    tail_frequency: f32,
    tail_speed: f32,
    tail_taper: f32,
    numSegments: i32
) -> vec4<f32> {
    let attach3D = flagellaDir3D * radius * 0.95;
    let attachScreen = projectToScreen(attach3D, camRight, camUp);
    
    var helixUp = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(flagellaDir3D, helixUp)) > 0.99) {
        helixUp = vec3<f32>(1.0, 0.0, 0.0);
    }
    let helixRight = normalize(cross(flagellaDir3D, helixUp));
    helixUp = normalize(cross(helixRight, flagellaDir3D));
    
    var minDist: f32 = 1000.0;
    var bestT: f32 = 0.0;
    var minDepth: f32 = 1000.0;
    var bestClosestPoint: vec2<f32> = attachScreen;
    
    var prevCenter = attachScreen;
    var prevRadius = tail_thickness * radius;
    var prevT: f32 = 0.0;
    var prevPos3D = attach3D;
    
    for (var i = 1; i < numSegments; i = i + 1) {
        let linearT = f32(i) / f32(numSegments - 1);
        let t = pow(linearT, 0.7);
        let phase = t * tail_frequency * TWO_PI - time * tail_speed;
        
        let pos3D = attach3D 
                  + flagellaDir3D * tail_length * radius * t
                  + helixUp * sin(phase) * tail_amplitude * radius * t
                  + helixRight * cos(phase) * tail_amplitude * radius * t;
        
        let center = attachScreen + projectToScreen(pos3D - attach3D, camRight, camUp);
        let segRadius = tail_thickness * radius * (1.0 - t * tail_taper);
        
        let result = sdTaperedCapsule2(p, prevCenter, center, prevRadius, segRadius);
        let d = result.x;
        let h = result.y;
        
        if (d < 0.0) {
            let interpPos3D = mix(prevPos3D, pos3D, h);
            let interpRadius = mix(prevRadius, segRadius, h);
            let frontPosWorld = cellCenter + interpPos3D - camFwd * interpRadius;
            let depth = dot(frontPosWorld - ro, camFwd);
            minDepth = min(minDepth, depth);
        }
        
        if (d < minDist) {
            minDist = d;
            bestT = mix(prevT, t, h);
            let ba = center - prevCenter;
            bestClosestPoint = prevCenter + ba * h;
        }
        
        prevCenter = center;
        prevRadius = segRadius;
        prevT = t;
        prevPos3D = pos3D;
    }
    
    return vec4<f32>(minDist, bestT, bestClosestPoint.x, minDepth);
}

// ============================================================================
// Vertex Shader
// ============================================================================

@vertex
fn vs_main(
    instance: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    
    let quad_pos = QUAD_VERTICES[vertex_index];
    
    // Camera-facing billboard
    let to_camera = normalize(camera.camera_pos - instance.position);
    
    // Calculate billboard basis vectors
    // Handle the case when looking straight up/down (to_camera nearly parallel to Y)
    var right: vec3<f32>;
    var up: vec3<f32>;
    let cross_result = cross(vec3<f32>(0.0, 1.0, 0.0), to_camera);
    let cross_len = length(cross_result);
    if (cross_len > 0.001) {
        right = cross_result / cross_len;
        up = cross(to_camera, right);
    } else {
        // Looking straight up or down - use X as right, Z as up
        right = vec3<f32>(1.0, 0.0, 0.0);
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    
    // Determine cell type from type_data_1.w (round to handle floating point precision)
    let cell_type = u32(round(instance.type_data_1.w));
    
    // Calculate billboard scale based on cell type
    var scale = instance.radius * 1.5; // Default: sphere with margin
    
    if (cell_type == CELL_TYPE_FLAGELLOCYTE) {
        // Flagellocyte needs larger billboard for tail
        let tail_length = instance.type_data_0.x;
        let tail_amplitude = instance.type_data_0.z;
        let tail_extent = tail_length * instance.radius;
        let helix_width = tail_amplitude * instance.radius;
        let total_extent = instance.radius + tail_extent + helix_width;
        scale = total_extent * 1.5;
    }
    
    let world_offset = right * quad_pos.x * scale + up * quad_pos.y * scale;
    let world_pos = instance.position + world_offset;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.center = instance.position;
    out.radius = instance.radius;
    out.color = instance.color;
    out.visual_params = instance.visual_params;
    out.uv = quad_pos * 0.5 + 0.5;
    out.type_data_0 = instance.type_data_0;
    out.type_data_1 = instance.type_data_1;
    out.rotation = instance.rotation;
    out.cam_right = right;
    out.cam_up = up;
    
    return out;
}


// ============================================================================
// Fragment Shader
// ============================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Determine cell type (round to handle floating point precision)
    let cell_type = u32(round(in.type_data_1.w));
    
    // Camera basis vectors
    let camRight = in.cam_right;
    let camUp = in.cam_up;
    let camFwd = normalize(in.center - camera.camera_pos);
    
    // Ray setup for sphere intersection
    let ray_origin = camera.camera_pos;
    let ray_dir = normalize(in.world_pos - ray_origin);
    
    // Ray-sphere intersection for body
    let oc = ray_origin - in.center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - in.radius * in.radius;
    let discriminant = b * b - 4.0 * a * c;
    
    var hitSphere = false;
    var sphereT: f32 = 1000.0;
    if (discriminant >= 0.0) {
        let t = (-b - sqrt(discriminant)) / (2.0 * a);
        if (t > 0.0) {
            sphereT = t;
            hitSphere = true;
        }
    }
    
    // Type-specific features
    var hitFlagella = false;
    var flagellaDist: f32 = 1000.0;
    var flagellaT: f32 = 0.0;
    var flagellaDepth: f32 = 1000.0;
    var flagellaDir3D: vec3<f32> = vec3<f32>(0.0, 0.0, -1.0);
    
    if (cell_type == CELL_TYPE_FLAGELLOCYTE) {
        // Extract flagella parameters
        let tail_length = in.type_data_0.x;
        let tail_thickness = in.type_data_0.y;
        let tail_amplitude = in.type_data_0.z;
        let tail_frequency = in.type_data_0.w;
        let tail_speed = in.type_data_1.x;
        let tail_taper = in.type_data_1.y;
        let tail_segments = i32(in.type_data_1.z);
        
        // Flagella direction from rotation
        flagellaDir3D = quat_rotate(in.rotation, vec3<f32>(0.0, 0.0, -1.0));
        
        // Billboard position relative to cell center
        let toPixel = in.world_pos - in.center;
        let billboardPos = projectToScreen(toPixel, camRight, camUp);
        
        // Compute flagella SDF
        let flagellaResult = sdFlagella2D(
            billboardPos,
            flagellaDir3D,
            camRight,
            camUp,
            camFwd,
            ray_origin,
            in.center,
            camera.time,
            in.radius,
            tail_length,
            tail_thickness,
            tail_amplitude,
            tail_frequency,
            tail_speed,
            tail_taper,
            tail_segments
        );
        
        flagellaDist = flagellaResult.x;
        flagellaT = flagellaResult.y;
        flagellaDepth = flagellaResult.w;
        hitFlagella = flagellaDist < 0.0;
    }
    
    // Determine what to render based on depth
    var renderSphere = false;
    var renderFlagella = false;
    
    if (hitSphere && hitFlagella) {
        let sphereHitPoint = ray_origin + ray_dir * sphereT;
        let sphereDepth = dot(sphereHitPoint - ray_origin, camFwd);
        if (flagellaDepth < sphereDepth) {
            renderFlagella = true;
        } else {
            renderSphere = true;
        }
    } else if (hitSphere) {
        renderSphere = true;
    } else if (hitFlagella) {
        renderFlagella = true;
    }
    
    if (!renderSphere && !renderFlagella) {
        discard;
    }
    
    // Extract visual parameters
    let specular_strength = in.visual_params.x;
    let specular_power = in.visual_params.y;
    let fresnel_strength = in.visual_params.z;
    let emissive = in.visual_params.w;
    
    var normal: vec3<f32>;
    var baseColor: vec3<f32>;
    
    if (renderSphere) {
        // Sphere rendering (common to all cell types)
        let hit_point = ray_origin + ray_dir * sphereT;
        normal = normalize(hit_point - in.center);
        baseColor = in.color.rgb;
        
        let n_dot_l = max(dot(normal, -lighting.light_dir), 0.0);
        let diffuse = n_dot_l * lighting.light_color;
        
        let view_dir = normalize(camera.camera_pos - hit_point);
        let half_dir = normalize(-lighting.light_dir + view_dir);
        let spec = pow(max(dot(normal, half_dir), 0.0), specular_power) * specular_strength;
        let specular = spec * lighting.light_color;
        
        let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0) * fresnel_strength;
        
        let ambient_color = baseColor * lighting.ambient;
        let lit_color = baseColor * diffuse + specular + fresnel * baseColor;
        let final_color = ambient_color + lit_color + baseColor * emissive;
        
        return vec4<f32>(final_color, in.color.a);
    } else {
        // Flagella rendering (Flagellocyte only)
        let tail_thickness = in.type_data_0.y;
        let tail_frequency = in.type_data_0.w;
        let tail_speed = in.type_data_1.x;
        let tail_taper = in.type_data_1.y;
        
        let phase = flagellaT * tail_frequency * TWO_PI - camera.time * tail_speed;
        
        var helixUp = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(dot(flagellaDir3D, helixUp)) > 0.99) {
            helixUp = vec3<f32>(1.0, 0.0, 0.0);
        }
        let helixRight = normalize(cross(flagellaDir3D, helixUp));
        helixUp = normalize(cross(helixRight, flagellaDir3D));
        
        let tail_length = in.type_data_0.x;
        let tail_amplitude = in.type_data_0.z;
        let attach3D = flagellaDir3D * in.radius * 0.95;
        let attachScreen = projectToScreen(attach3D, camRight, camUp);
        
        let pos3D = attach3D 
                  + flagellaDir3D * tail_length * in.radius * flagellaT
                  + helixUp * sin(phase) * tail_amplitude * in.radius * flagellaT
                  + helixRight * cos(phase) * tail_amplitude * in.radius * flagellaT;
        let closestPoint = attachScreen + projectToScreen(pos3D - attach3D, camRight, camUp);
        
        let segRadius = tail_thickness * in.radius * (1.0 - flagellaT * tail_taper);
        let toPixel = in.world_pos - in.center;
        let billboardPos = projectToScreen(toPixel, camRight, camUp);
        
        normal = flagellaNormal2D(billboardPos, closestPoint, camRight, camUp, camFwd, segRadius);
        baseColor = in.color.rgb * 0.9 + vec3<f32>(0.1, 0.15, 0.2);
        
        let outlineWidth = 0.008 * in.radius;
        let distFromEdge = -flagellaDist;
        
        if (distFromEdge < outlineWidth) {
            let outlineColor = baseColor * 0.15;
            return vec4<f32>(outlineColor, in.color.a);
        }
        
        let view_dir = -ray_dir;
        let n_dot_l = max(dot(normal, -lighting.light_dir), 0.0);
        let n_dot_v = max(dot(normal, view_dir), 0.0);
        
        var color = baseColor * (n_dot_l * lighting.light_color * 0.7 + lighting.ambient);
        color += baseColor * pow(1.0 - n_dot_v, 2.5) * fresnel_strength * 0.4;
        
        let half_dir = normalize(-lighting.light_dir + view_dir);
        let spec = pow(max(dot(normal, half_dir), 0.0), 32.0) * 0.3;
        color += lighting.light_color * spec;
        
        return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), in.color.a);
    }
}

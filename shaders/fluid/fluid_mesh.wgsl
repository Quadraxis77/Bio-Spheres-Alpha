// Fluid mesh shader for surface nets rendering
// Renders smooth fluid surfaces with Fresnel effects and lighting
// Supports per-vertex fluid type for multi-fluid rendering

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
}

struct FluidParams {
    water_color: vec4<f32>,
    lava_color: vec4<f32>,
    steam_color: vec4<f32>,
    // Lighting vec4
    ambient: f32,
    diffuse: f32,
    specular_intensity: f32,
    shininess: f32,
    // Fresnel vec4
    fresnel_strength: f32,
    fresnel_power: f32,
    reflection: f32,
    alpha: f32,
    // Emissive vec4
    emissive: f32,
    rim_strength: f32,
    fluid_type: u32,
    ssr_enabled: f32,
    // SSR params vec4
    ssr_max_steps: f32,
    ssr_step_size: f32,
    ssr_max_distance: f32,
    ssr_thickness: f32,
    // SSR quality vec4
    ssr_intensity: f32,
    ssr_fade_start: f32,
    ssr_fade_end: f32,
    ssr_roughness: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> params: FluidParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) fluid_type: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_dir: vec3<f32>,
    @location(3) @interpolate(flat) fluid_type: u32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    out.world_position = in.position;
    out.world_normal = normalize(in.normal);
    out.view_dir = normalize(camera.camera_pos - in.position);
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.fluid_type = in.fluid_type;
    
    return out;
}

// Screen-space reflection approximation using ray marching in view space
fn compute_ssr(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    base_color: vec3<f32>,
) -> vec4<f32> {
    if params.ssr_enabled < 0.5 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Calculate reflection direction
    let reflect_dir = reflect(-view_dir, normal);
    
    // SSR ray marching approximation
    var ray_pos = world_pos;
    var accumulated_color = vec3<f32>(0.0);
    var hit_count = 0.0;
    
    let max_steps = u32(params.ssr_max_steps);
    let step_size = params.ssr_step_size;
    
    // March along reflection ray
    for (var i = 0u; i < max_steps; i++) {
        ray_pos += reflect_dir * step_size * (1.0 + f32(i) * 0.1);
        
        let dist = length(ray_pos - world_pos);
        if dist > params.ssr_max_distance {
            break;
        }
        
        // Simulate hitting geometry by sampling environment
        // In a full implementation, this would sample depth buffer
        let height_factor = ray_pos.y / 100.0;
        let dist_factor = dist / params.ssr_max_distance;
        
        // Sky gradient based on ray height
        let sky_top = vec3<f32>(0.4, 0.6, 0.9);
        let sky_horizon = vec3<f32>(0.7, 0.8, 0.95);
        let ground = vec3<f32>(0.3, 0.25, 0.2);
        
        var env_color: vec3<f32>;
        if reflect_dir.y > 0.0 {
            // Looking up - sky colors
            let sky_blend = pow(reflect_dir.y, 0.5);
            env_color = mix(sky_horizon, sky_top, sky_blend);
        } else {
            // Looking down - darker/ground colors
            let ground_blend = pow(-reflect_dir.y, 0.5);
            env_color = mix(sky_horizon * 0.5, ground, ground_blend);
        }
        
        // Add some variation based on position (fake detail)
        let noise = sin(ray_pos.x * 0.5) * cos(ray_pos.z * 0.5) * 0.1;
        env_color += vec3<f32>(noise);
        
        // Accumulate with distance falloff
        let falloff = 1.0 - smoothstep(params.ssr_fade_start, params.ssr_fade_end, dist_factor);
        accumulated_color += env_color * falloff;
        hit_count += falloff;
    }
    
    if hit_count > 0.0 {
        accumulated_color /= hit_count;
    }
    
    // Apply roughness blur simulation
    let roughness_blend = params.ssr_roughness;
    accumulated_color = mix(accumulated_color, base_color * 0.5 + accumulated_color * 0.5, roughness_blend);
    
    // Calculate reflection strength based on fresnel
    let fresnel_ssr = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
    let reflection_strength = fresnel_ssr * params.ssr_intensity;
    
    return vec4<f32>(accumulated_color, reflection_strength);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(in.view_dir);
    
    // Select base color based on per-vertex fluid type
    var base_color: vec4<f32>;
    if in.fluid_type == 1u {
        base_color = params.water_color;
    } else if in.fluid_type == 2u {
        base_color = params.lava_color;
    } else if in.fluid_type == 3u {
        base_color = params.steam_color;
    } else {
        base_color = params.water_color;
    }
    
    // Light direction (sun-like from upper right)
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    
    // Ambient lighting
    let ambient_term = params.ambient;
    
    // Diffuse lighting (Lambert)
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let diffuse_term = n_dot_l * params.diffuse;
    
    // Specular lighting (Blinn-Phong)
    let half_dir = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    var specular_term = pow(n_dot_h, params.shininess) * params.specular_intensity;
    
    // Reduce specular for lava (more matte/molten look)
    if in.fluid_type == 2u {
        specular_term *= 0.3;
    }
    // Increase specular for water (wet/shiny)
    if in.fluid_type == 1u {
        specular_term *= 1.2;
    }
    
    // Fresnel effect - more reflective at glancing angles
    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), params.fresnel_power) * params.fresnel_strength;
    
    // Rim lighting (backlight effect)
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0) * params.rim_strength;
    
    // Combine base lighting
    let lighting = ambient_term + diffuse_term;
    var final_color = base_color.rgb * lighting;
    
    // Add specular highlight
    let specular_color = vec3<f32>(1.0, 1.0, 1.0);
    final_color += specular_color * specular_term;
    
    // Screen Space Reflections
    var ssr_result = vec4<f32>(0.0);
    if in.fluid_type == 1u || in.fluid_type == 3u {
        // SSR for water and steam
        ssr_result = compute_ssr(in.world_position, normal, view_dir, base_color.rgb);
    }
    
    // Environment reflection (fallback/blend with SSR)
    var reflection_color: vec3<f32>;
    if in.fluid_type == 1u {
        let sky_color = vec3<f32>(0.6, 0.8, 1.0);
        let horizon_color = vec3<f32>(0.9, 0.95, 1.0);
        let reflect_dir = reflect(-view_dir, normal);
        let sky_factor = max(reflect_dir.y, 0.0);
        reflection_color = mix(horizon_color, sky_color, sky_factor);
    } else if in.fluid_type == 2u {
        reflection_color = vec3<f32>(1.0, 0.5, 0.2);
    } else {
        reflection_color = vec3<f32>(0.95, 0.95, 1.0);
    }
    
    // Blend SSR with environment reflection
    let ssr_blend = ssr_result.a;
    reflection_color = mix(reflection_color, ssr_result.rgb, ssr_blend);
    
    // Apply reflection
    final_color = mix(final_color, reflection_color, fresnel * params.reflection);
    
    // Rim light (different color per type)
    var rim_color: vec3<f32>;
    if in.fluid_type == 1u {
        rim_color = vec3<f32>(0.7, 0.85, 1.0);
    } else if in.fluid_type == 2u {
        rim_color = vec3<f32>(1.0, 0.6, 0.2);
    } else {
        rim_color = vec3<f32>(1.0, 1.0, 1.0);
    }
    final_color += rim_color * rim;
    
    // Emissive glow (mainly for lava)
    if in.fluid_type == 2u {
        let emissive_color = base_color.rgb * params.emissive;
        final_color += emissive_color;
        let glow = base_color.rgb * params.emissive * 0.5;
        final_color += glow;
    }
    
    // Subsurface scattering simulation for water
    if in.fluid_type == 1u {
        let sss = pow(max(dot(view_dir, -light_dir), 0.0), 2.0) * 0.15;
        let sss_color = vec3<f32>(0.0, 0.4, 0.6);
        final_color += sss_color * sss;
    }
    
    // Alpha varies by fluid type
    var final_alpha = params.alpha;
    if params.alpha >= 0.99 {
        // User wants fully opaque - respect that
        final_alpha = 1.0;
    } else if in.fluid_type == 1u {
        // Water: blend between base alpha and opaque at edges
        final_alpha = mix(base_color.a * params.alpha, min(params.alpha + 0.2, 1.0), fresnel * 0.3);
    } else if in.fluid_type == 2u {
        // Lava: use alpha directly (can be opaque)
        final_alpha = params.alpha;
    } else if in.fluid_type == 3u {
        // Steam: always somewhat transparent
        final_alpha = base_color.a * params.alpha * 0.7;
    }
    
    return vec4<f32>(final_color, final_alpha);
}

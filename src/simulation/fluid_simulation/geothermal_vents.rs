use crate::rendering::cave_system::CaveParams;
use glam::{IVec3, Vec3};

#[derive(Clone, Copy, Debug)]
struct VentSpec {
    surface: IVec3,
    normal: IVec3,
    tangent: IVec3,
    bitangent: IVec3,
    length: i32,
    width: i32,
    depth: i32,
    heat_radius: i32,
    glow_radius: i32,
}

#[derive(Debug)]
pub struct GeothermalFields {
    pub heat: Vec<f32>,
    pub glow: Vec<[f32; 4]>,
}

const SEARCH_STRIDE: usize = 3;

fn param_i32(value: f32, min: i32, max: i32) -> i32 {
    (value.round() as i32).clamp(min, max)
}

#[inline]
fn idx(p: IVec3, res: usize) -> usize {
    p.x as usize + p.y as usize * res + p.z as usize * res * res
}

#[inline]
fn in_bounds(p: IVec3, res: i32) -> bool {
    p.x > 1 && p.y > 1 && p.z > 1 && p.x < res - 2 && p.y < res - 2 && p.z < res - 2
}

#[inline]
fn is_solid(solid: &[bool], p: IVec3, res: usize) -> bool {
    in_bounds(p, res as i32) && solid[idx(p, res)]
}

fn dominant_axis_dir(v: Vec3) -> IVec3 {
    let a = v.abs();
    if a.x >= a.y && a.x >= a.z {
        IVec3::new(v.x.signum() as i32, 0, 0)
    } else if a.y >= a.x && a.y >= a.z {
        IVec3::new(0, v.y.signum() as i32, 0)
    } else {
        IVec3::new(0, 0, v.z.signum() as i32)
    }
}

fn push_unique(out: &mut Vec<IVec3>, p: IVec3) {
    if !out.contains(&p) {
        out.push(p);
    }
}

fn round_vec3(v: Vec3) -> IVec3 {
    IVec3::new(v.x.round() as i32, v.y.round() as i32, v.z.round() as i32)
}

fn hash_u32(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= (x as u32).wrapping_mul(0x9E37_79B9);
    h = h.rotate_left(7) ^ (y as u32).wrapping_mul(0x85EB_CA6B);
    h = h.rotate_left(11) ^ (z as u32).wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^ (h >> 15)
}

fn choose_tangent(normal: IVec3, p: IVec3, seed: u32) -> IVec3 {
    let mut axes = Vec::new();
    for axis in [IVec3::X, IVec3::Y, IVec3::Z] {
        if axis.dot(normal).abs() == 0 {
            axes.push(axis);
        }
    }
    let choice = (hash_u32(p.x, p.y, p.z, seed) as usize) % axes.len();
    if hash_u32(p.z, p.x, p.y, seed ^ 0xA53A_9A5D) & 1 == 0 {
        axes[choice]
    } else {
        -axes[choice]
    }
}

fn generate_specs(solid: &[bool], res: usize, params: &CaveParams, seed: u32) -> Vec<VentSpec> {
    if params.geothermal_enabled == 0 || params.geothermal_count == 0 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    let center = Vec3::splat(res as f32 * 0.5);
    let boundary_radius = res as f32 * 0.5;

    for y in (3..res - 3).step_by(SEARCH_STRIDE) {
        for z in (3..res - 3).step_by(SEARCH_STRIDE) {
            for x in (3..res - 3).step_by(SEARCH_STRIDE) {
                let p = IVec3::new(x as i32, y as i32, z as i32);
                if !is_solid(solid, p, res) {
                    continue;
                }
                let pf = Vec3::new(x as f32, y as f32, z as f32);
                let from_center = pf - center;
                let dist = from_center.length();
                if dist < boundary_radius * 0.88 {
                    continue;
                }

                let normal = dominant_axis_dir(center - pf);
                if normal == IVec3::ZERO {
                    continue;
                }

                // The stack lives on the solid sphere shell and exhales inward
                // into the first open interior voxel. No solid voxels are opened.
                if !is_solid(solid, p + normal, res) {
                    let tangent = choose_tangent(normal, p, seed);
                    let score = hash_u32(p.x, p.y, p.z, seed ^ 0x6D2B_79F5);
                    candidates.push((score, p, normal, tangent));
                }
            }
        }
    }

    candidates.sort_by_key(|(score, _, _, _)| *score);
    candidates
        .into_iter()
        .take(params.geothermal_count as usize)
        .map(|(_, surface, normal, tangent)| VentSpec {
            surface,
            normal,
            tangent,
            bitangent: normal.cross(tangent),
            length: param_i32(params.geothermal_length, 1, 64),
            width: param_i32(params.geothermal_width, 1, 16),
            depth: param_i32(params.geothermal_depth, 1, 32),
            heat_radius: param_i32(params.geothermal_heat_radius, 1, 64),
            glow_radius: param_i32(params.geothermal_glow_radius, 1, 64),
        })
        .collect()
}

fn centerline_jitter(spec: VentSpec, s: i32, seed: u32) -> i32 {
    let h = hash_u32(
        spec.surface.x + s * spec.tangent.x,
        spec.surface.y + s * spec.tangent.y,
        spec.surface.z + s * spec.tangent.z,
        seed ^ 0xB529_7A4D,
    );
    (h % 3) as i32 - 1
}

fn footprint_centers(spec: VentSpec, seed: u32) -> Vec<IVec3> {
    let mut out = Vec::new();
    let half_len = spec.length / 2;
    for s in -half_len..=half_len {
        let jitter = centerline_jitter(spec, s, seed);
        out.push(spec.surface + spec.tangent * s + spec.bitangent * jitter);
    }
    out
}

fn stack_voxels(
    spec: VentSpec,
    seed: u32,
    res: usize,
) -> (Vec<IVec3>, Vec<IVec3>, Vec<IVec3>, Vec<IVec3>) {
    let mut walls = Vec::new();
    let mut core = Vec::new();
    let mut openings = Vec::new();
    let mut glow_sources = Vec::new();
    let top_outer = spec.width.max(2);
    let base_outer = (spec.width + 2).max(top_outer + 1);
    let inner = (spec.width / 2).max(1);
    let grid_center = Vec3::splat(res as f32 * 0.5);
    let sphere_radius = res as f32 * 0.5 - 1.0;
    let axis_hint = Vec3::new(
        spec.normal.x as f32,
        spec.normal.y as f32,
        spec.normal.z as f32,
    );
    let tangent = Vec3::new(
        spec.tangent.x as f32,
        spec.tangent.y as f32,
        spec.tangent.z as f32,
    );
    let bitangent = Vec3::new(
        spec.bitangent.x as f32,
        spec.bitangent.y as f32,
        spec.bitangent.z as f32,
    );

    for base in footprint_centers(spec, seed) {
        let base_f = Vec3::new(base.x as f32, base.y as f32, base.z as f32);
        let radial_in = (grid_center - base_f).try_normalize().unwrap_or(axis_hint);
        let radial_out = -radial_in;

        for h in 0..=spec.depth {
            let height_t = h as f32 / spec.depth.max(1) as f32;
            let outer = (base_outer as f32 * (1.0 - height_t) + top_outer as f32 * height_t)
                .round()
                .max(top_outer as f32) as i32;

            for a in -base_outer..=base_outer {
                for b in -base_outer..=base_outer {
                    let lateral = tangent * a as f32 + bitangent * b as f32;
                    let radial_len = lateral.length();
                    if radial_len > outer as f32 + 0.25 {
                        continue;
                    }

                    // Each footprint sample is projected onto the sphere before
                    // the stack rises inward, so the flared base follows the
                    // curved boundary instead of cutting a flat plane through it.
                    let shell_point = (base_f + lateral - grid_center)
                        .try_normalize()
                        .unwrap_or(radial_out)
                        * sphere_radius
                        + grid_center;
                    let p = round_vec3(shell_point + radial_in * h as f32);

                    if h == 0 {
                        push_unique(&mut walls, p);
                    } else if radial_len <= inner as f32 + 0.25 {
                        push_unique(&mut core, p);
                        if h == spec.depth {
                            push_unique(&mut openings, p);
                        } else if h <= 1 {
                            push_unique(&mut glow_sources, p);
                        }
                    } else {
                        push_unique(&mut walls, p);
                    }
                }
            }
        }
    }

    (walls, core, openings, glow_sources)
}

pub fn carve_density_grid(density: &mut [Vec<Vec<f32>>], params: &CaveParams, threshold: f32) {
    let res = density.len();
    if res < 8 {
        return;
    }
    let mut solid = vec![false; res * res * res];
    for x in 0..res {
        for y in 0..res {
            for z in 0..res {
                solid[x + y * res + z * res * res] = density[x][y][z] >= threshold;
            }
        }
    }

    for spec in generate_specs(&solid, res, params, params.seed ^ 0xCE11_5EED) {
        let (walls, core, _, _) = stack_voxels(spec, params.seed, res);
        for p in walls {
            if in_bounds(p, res as i32) {
                density[p.x as usize][p.y as usize][p.z as usize] = threshold + 0.35;
            }
        }
        for p in core {
            if in_bounds(p, res as i32) {
                density[p.x as usize][p.y as usize][p.z as usize] = threshold - 0.5;
            }
        }
    }
}

pub fn apply_to_solid_mask(
    solid_mask: &mut [u32],
    params: &CaveParams,
    world_center: Vec3,
    world_radius: f32,
    res: usize,
) -> GeothermalFields {
    let mut heat = vec![0.0f32; solid_mask.len()];
    let mut glow = vec![[0.0f32; 4]; solid_mask.len()];
    let mut solid: Vec<bool> = solid_mask.iter().map(|&v| v != 0).collect();
    let specs = generate_specs(&solid, res, params, params.seed ^ 0xCE11_5EED);

    for spec in specs {
        let (walls, core, openings, glow_sources) = stack_voxels(spec, params.seed, res);

        for p in walls {
            if in_bounds(p, res as i32) {
                let i = idx(p, res);
                solid[i] = true;
                solid_mask[i] = 1;
            }
        }
        for p in core {
            if in_bounds(p, res as i32) {
                let i = idx(p, res);
                solid[i] = false;
                solid_mask[i] = 0;
            }
        }

        let grid_center = Vec3::splat(res as f32 * 0.5);
        let tangent = Vec3::new(
            spec.tangent.x as f32,
            spec.tangent.y as f32,
            spec.tangent.z as f32,
        );
        let bitangent = Vec3::new(
            spec.bitangent.x as f32,
            spec.bitangent.y as f32,
            spec.bitangent.z as f32,
        );

        for &p in &openings {
            let p_f = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            let radial_in = (grid_center - p_f).try_normalize().unwrap_or(Vec3::Y);
            let source_depth = (spec.heat_radius / 8).clamp(1, 3);
            let source_radius = (spec.width / 2).max(1);

            for h in 0..=source_depth {
                let forward = p_f + radial_in * h as f32;
                for a in -source_radius..=source_radius {
                    for b in -source_radius..=source_radius {
                        let lateral = ((a * a + b * b) as f32).sqrt();
                        if lateral > source_radius as f32 + 0.25 {
                            continue;
                        }

                        let q = round_vec3(forward + tangent * a as f32 + bitangent * b as f32);
                        if !in_bounds(q, res as i32) || solid[idx(q, res)] {
                            continue;
                        }

                        let axial = 1.0 - h as f32 / (source_depth + 1) as f32;
                        let radial = 1.0 - (lateral / source_radius.max(1) as f32).min(1.0) * 0.35;
                        let heat_value =
                            params.geothermal_heat_output * axial.max(0.0) * radial.max(0.0);
                        let i = idx(q, res);
                        heat[i] = heat[i].max(heat_value);
                    }
                }
            }
        }

        let shaft_inner_radius = (spec.width / 2).max(1);
        for &source in &glow_sources {
            let source_f = Vec3::new(source.x as f32, source.y as f32, source.z as f32);
            let radial_in = (grid_center - source_f).try_normalize().unwrap_or(Vec3::Y);

            for h in 0..=spec.glow_radius {
                // Before the glow reaches the chimney lip, keep it inside the
                // hollow shaft. Past the opening it may spread as a small cone.
                let cone_width = if h <= spec.depth {
                    shaft_inner_radius
                } else {
                    shaft_inner_radius + (h - spec.depth) / 3
                };
                let forward = source_f + radial_in * h as f32;
                for a in -cone_width..=cone_width {
                    for b in -cone_width..=cone_width {
                        let lateral_len = Vec3::new(a as f32, b as f32, 0.0).length();
                        if lateral_len > cone_width as f32 + 0.25 {
                            continue;
                        }

                        let q = round_vec3(forward + tangent * a as f32 + bitangent * b as f32);
                        if !in_bounds(q, res as i32) || solid[idx(q, res)] {
                            continue;
                        }

                        let axial_falloff = 1.0 - h as f32 / spec.glow_radius.max(1) as f32;
                        let lateral_falloff =
                            1.0 - (lateral_len / cone_width.max(1) as f32).min(1.0) * 0.6;
                        let shaft_occlusion = if h <= spec.depth { 0.72 } else { 1.0 };
                        let falloff = (axial_falloff * lateral_falloff * shaft_occlusion)
                            .max(0.0)
                            .powf(1.35);
                        let i = idx(q, res);
                        let strength = params.geothermal_glow_strength * falloff;
                        glow[i][0] = glow[i][0].max(params.geothermal_glow_color[0] * strength);
                        glow[i][1] = glow[i][1].max(params.geothermal_glow_color[1] * strength);
                        glow[i][2] = glow[i][2].max(params.geothermal_glow_color[2] * strength);
                        glow[i][3] = glow[i][3].max(strength);
                    }
                }
            }
        }
    }

    let _ = (solid_mask, world_center, world_radius);
    GeothermalFields { heat, glow }
}

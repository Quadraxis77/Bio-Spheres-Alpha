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
const SIDE_MARGIN_CELLS: i32 = 1;
const SOLID_MASK_OPEN_DEPTH_CELLS: i32 = 1;

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

fn has_required_thickness(
    solid: &[bool],
    p: IVec3,
    normal: IVec3,
    tangent: IVec3,
    res: usize,
    params: &CaveParams,
) -> bool {
    let inward = -normal;
    let bitangent = normal.cross(tangent);
    let half_len = param_i32(params.geothermal_length, 1, 64) / 2;
    let probe_depth =
        param_i32(params.geothermal_depth, 1, 32) + param_i32(params.geothermal_back_margin, 0, 32);
    let probe_width = param_i32(params.geothermal_width, 1, 16) + SIDE_MARGIN_CELLS;
    let top_margin = param_i32(params.geothermal_top_margin, 0, 16);

    for s in -half_len..=half_len {
        for w in -probe_width..=probe_width {
            for d in 0..=probe_depth {
                let base = p + tangent * s + bitangent * w + inward * d;
                for top in 0..=top_margin {
                    let q = base + IVec3::Y * top;
                    if !is_solid(solid, q, res) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn generate_specs(solid: &[bool], res: usize, params: &CaveParams, seed: u32) -> Vec<VentSpec> {
    if params.geothermal_enabled == 0 || params.geothermal_count == 0 {
        return Vec::new();
    }

    let dirs = [
        IVec3::X,
        -IVec3::X,
        IVec3::Y,
        -IVec3::Y,
        IVec3::Z,
        -IVec3::Z,
    ];
    let mut candidates = Vec::new();

    for y in (3..res - 3).step_by(SEARCH_STRIDE) {
        for z in (3..res - 3).step_by(SEARCH_STRIDE) {
            for x in (3..res - 3).step_by(SEARCH_STRIDE) {
                let p = IVec3::new(x as i32, y as i32, z as i32);
                if !is_solid(solid, p, res) {
                    continue;
                }
                for normal in dirs {
                    if !is_solid(solid, p + normal, res) {
                        let tangent = choose_tangent(normal, p, seed);
                        if has_required_thickness(solid, p, normal, tangent, res, params) {
                            let score = hash_u32(p.x, p.y, p.z, seed ^ 0x6D2B_79F5);
                            candidates.push((score, p, normal, tangent));
                        }
                        break;
                    }
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

fn carved_voxels(spec: VentSpec, seed: u32) -> Vec<(IVec3, i32)> {
    let mut out = Vec::new();
    let inward = -spec.normal;
    let half_len = spec.length / 2;
    for s in -half_len..=half_len {
        let jitter = centerline_jitter(spec, s, seed);
        for d in 0..=spec.depth {
            let depth_frac = d as f32 / spec.depth.max(1) as f32;
            let width = (spec.width as f32 * (1.0 - depth_frac).powf(1.7)).ceil() as i32;
            for w in -width..=width {
                if w.abs() > width {
                    continue;
                }
                let p =
                    spec.surface + spec.tangent * s + spec.bitangent * (w + jitter) + inward * d;
                out.push((p, d));
            }
        }
    }
    out
}

fn mouth_voxels(spec: VentSpec, seed: u32) -> Vec<(IVec3, i32)> {
    carved_voxels(spec, seed)
        .into_iter()
        .filter(|(_, depth)| *depth <= SOLID_MASK_OPEN_DEPTH_CELLS.min(spec.depth))
        .collect()
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
        for (p, depth) in carved_voxels(spec, params.seed) {
            if in_bounds(p, res as i32) {
                let depth_frac = depth as f32 / spec.depth.max(1) as f32;
                let cut_strength = 0.5 * (1.0 - depth_frac * 0.7).max(0.18);
                density[p.x as usize][p.y as usize][p.z as usize] = threshold - cut_strength;
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
        let voxels = mouth_voxels(spec, params.seed);
        let mut lowest = spec.surface;

        for (p, depth) in &voxels {
            if !in_bounds(*p, res as i32) {
                continue;
            }
            solid[idx(*p, res)] = false;
            solid_mask[idx(*p, res)] = 0;
            if depth == &spec.depth && p.y < lowest.y {
                lowest = *p;
            }
        }

        for (p, depth) in voxels {
            if !in_bounds(p, res as i32) {
                continue;
            }
            let heat_weight = 0.55 + 0.45 * (depth as f32 / spec.depth.max(1) as f32);
            for h in 0..=spec.heat_radius {
                let forward = p + spec.normal * h;
                let cone_width = 1 + h / 2;
                for a in -cone_width..=cone_width {
                    for b in -cone_width..=cone_width {
                        let q = forward + spec.tangent * a + spec.bitangent * b;
                        if !in_bounds(q, res as i32) || solid[idx(q, res)] {
                            continue;
                        }
                        let lateral = ((a * a + b * b) as f32).sqrt() / cone_width.max(1) as f32;
                        if lateral > 1.0 {
                            continue;
                        }
                        let falloff = (1.0 - h as f32 / spec.heat_radius.max(1) as f32)
                            * (1.0 - lateral * 0.65)
                            * heat_weight;
                        let heat_value = params.geothermal_heat_output * falloff.max(0.0);
                        let i = idx(q, res);
                        heat[i] = heat[i].max(heat_value);
                    }
                }
            }
        }

        for dz in -spec.glow_radius..=spec.glow_radius {
            for dy in -spec.glow_radius..=spec.glow_radius {
                for dx in -spec.glow_radius..=spec.glow_radius {
                    let q = lowest + IVec3::new(dx, dy, dz);
                    if !in_bounds(q, res as i32) || solid[idx(q, res)] {
                        continue;
                    }
                    let dist = Vec3::new(dx as f32, dy as f32, dz as f32).length();
                    if dist > spec.glow_radius as f32 {
                        continue;
                    }
                    let falloff = (1.0 - dist / spec.glow_radius as f32).powf(1.8);
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

    let _ = (world_center, world_radius);
    GeothermalFields { heat, glow }
}

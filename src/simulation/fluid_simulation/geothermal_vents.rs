use crate::rendering::cave_system::CaveParams;
use glam::{IVec3, Vec3};

#[derive(Clone, Copy, Debug)]
pub(crate) struct VentSpec {
    surface: IVec3,
    /// Cardinal direction toward the open neighbor cell. Used to mold the
    /// base flush with the wall along a precise grid axis.
    normal: IVec3,
    tangent: IVec3,
    bitangent: IVec3,
    /// Direction the stack grows from base to opening, following the local
    /// average surface normal rather than snapping to a grid axis.
    axis: Vec3,
    placement_mode: u32,
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

/// Estimates the local outward surface direction at a solid voxel by
/// averaging (inverse-distance weighted) directions toward nearby non-solid
/// voxels. Returns `None` if no open voxels are found in the search radius.
fn local_outward_axis(solid: &[bool], p: IVec3, res: usize, radius: i32) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                let q = p + IVec3::new(dx, dy, dz);
                if !in_bounds(q, res as i32) || solid[idx(q, res)] {
                    continue;
                }
                let dir = Vec3::new(dx as f32, dy as f32, dz as f32);
                let dist_sq = dir.length_squared();
                sum += dir / dist_sq;
            }
        }
    }
    sum.try_normalize()
}

/// Walks outward from `origin` along `axis`, looking for the solid/open
/// boundary closest to `origin`. Returns the signed offset (in voxels) that
/// should be added along `axis` so the base sits flush on the surface
/// instead of floating above it (overhang) or burying into it.
fn surface_shift(solid: &[bool], origin: Vec3, axis: Vec3, res: usize, max_search: i32) -> f32 {
    for d in -max_search..=max_search {
        let cur = round_vec3(origin + axis * d as f32);
        let next = round_vec3(origin + axis * (d + 1) as f32);
        let cur_solid = in_bounds(cur, res as i32) && solid[idx(cur, res)];
        let next_solid = in_bounds(next, res as i32) && solid[idx(next, res)];
        if cur_solid && !next_solid {
            return d as f32;
        }
    }
    0.0
}

/// Checks that the voxels ahead of `origin` along `axis` are open for at
/// least `min_clear` steps, so the vent opening doesn't exhale straight into
/// more rock.
fn has_clear_opening(
    solid: &[bool],
    origin: IVec3,
    axis: Vec3,
    res: usize,
    min_clear: i32,
) -> bool {
    let origin_f = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
    for d in 1..=min_clear {
        let q = round_vec3(origin_f + axis * d as f32);
        if !in_bounds(q, res as i32) || solid[idx(q, res)] {
            return false;
        }
    }
    true
}

/// Checks that the wall stays solid and the surface direction stays
/// consistent across the footprint area, so the base isn't molded onto a
/// sharp edge or corner where the wall geometry breaks down.
fn is_locally_flat(
    solid: &[bool],
    p: IVec3,
    axis: Vec3,
    tangent: IVec3,
    bitangent: IVec3,
    radius: i32,
    res: usize,
) -> bool {
    const MIN_DOT: f32 = 0.6;
    for &(ta, bb) in &[(radius, 0), (-radius, 0), (0, radius), (0, -radius)] {
        let q = p + tangent * ta + bitangent * bb;
        if !is_solid(solid, q, res) {
            return false;
        }
        match local_outward_axis(solid, q, res, 2) {
            Some(local_axis) if local_axis.dot(axis) >= MIN_DOT => {}
            _ => return false,
        }
    }
    true
}

fn gravity_down_dir(params: &CaveParams) -> Option<Vec3> {
    if params.geothermal_gravity_mode == 3 {
        return None;
    }

    let sign = if params.geothermal_gravity < 0.0 {
        1.0
    } else {
        -1.0
    };

    Some(match params.geothermal_gravity_mode {
        0 => Vec3::X * sign,
        2 => Vec3::Z * sign,
        _ => Vec3::Y * sign,
    })
}

fn passes_lower_hemisphere(p: IVec3, res: usize, params: &CaveParams) -> bool {
    if params.geothermal_lower_hemisphere == 0 {
        return true;
    }

    let Some(down) = gravity_down_dir(params) else {
        return true;
    };

    let center = Vec3::splat(res as f32 * 0.5);
    let pf = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
    (pf - center).dot(down) >= 0.0
}

/// Generates vent placement specs against the given solid/empty grid. Both
/// the geothermal heat/glow fields and the cave mesh's stack geometry must
/// call this against the *same* `solid`/`res` grid (the canonical
/// resolution^3/world_radius grid from [`SolidMaskGenerator::build_solid_array`])
/// so they agree on placement -- otherwise glow can appear with no matching
/// stack geometry, or vice versa.
pub(crate) fn generate_specs(
    solid: &[bool],
    res: usize,
    params: &CaveParams,
    seed: u32,
) -> Vec<VentSpec> {
    if params.geothermal_enabled == 0 || params.geothermal_count == 0 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    let center = Vec3::splat(res as f32 * 0.5);
    let boundary_radius = res as f32 * 0.5;
    let placement_mode = params.geothermal_placement_mode.min(1);
    let dirs = [
        IVec3::X,
        -IVec3::X,
        IVec3::Y,
        -IVec3::Y,
        IVec3::Z,
        -IVec3::Z,
    ];

    let length = param_i32(params.geothermal_length, 1, 64);
    let width = param_i32(params.geothermal_width, 1, 16);
    let depth = param_i32(params.geothermal_depth, 1, 32);
    let heat_radius = param_i32(params.geothermal_heat_radius, 1, 64);
    let glow_radius = param_i32(params.geothermal_glow_radius, 1, 64);
    let min_clear = depth + 2;
    let top_outer = width.max(2);

    for y in (3..res - 3).step_by(SEARCH_STRIDE) {
        for z in (3..res - 3).step_by(SEARCH_STRIDE) {
            for x in (3..res - 3).step_by(SEARCH_STRIDE) {
                let p = IVec3::new(x as i32, y as i32, z as i32);
                if !is_solid(solid, p, res) {
                    continue;
                }
                if !passes_lower_hemisphere(p, res, params) {
                    continue;
                }
                let pf = Vec3::new(x as f32, y as f32, z as f32);
                let from_center = pf - center;
                let dist = from_center.length();

                if placement_mode == 0 {
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
                        let axis = (center - pf).try_normalize().unwrap_or(Vec3::new(
                            normal.x as f32,
                            normal.y as f32,
                            normal.z as f32,
                        ));
                        let tangent = choose_tangent(normal, p, seed);
                        let score = hash_u32(p.x, p.y, p.z, seed ^ 0x6D2B_79F5);
                        candidates.push((score, p, normal, tangent, axis));
                    }
                } else {
                    // Interior mode: choose ordinary cave wall voxels and face the
                    // stack out into the adjacent open cave cell.
                    if dist > boundary_radius * 0.86 {
                        continue;
                    }

                    // Face the stack toward the average of every open
                    // cardinal neighbor direction, rather than just the
                    // first one found. On an edge or corner this points
                    // along a blended diagonal instead of snapping to
                    // whichever single face happened to be open.
                    let mut face_sum = Vec3::ZERO;
                    for dir in dirs {
                        if !is_solid(solid, p + dir, res) {
                            face_sum += Vec3::new(dir.x as f32, dir.y as f32, dir.z as f32);
                        }
                    }
                    let Some(face_avg) = face_sum.try_normalize() else {
                        continue;
                    };

                    // `normal` stays a single cardinal axis: it's used to
                    // mold the base flush with the wall along a precise grid
                    // step and to pick a perpendicular cardinal
                    // tangent/bitangent pair. Use whichever cardinal axis is
                    // closest to the averaged open-face direction, and
                    // require that axis itself to be open.
                    let normal = dominant_axis_dir(face_avg);
                    if is_solid(solid, p + normal, res) {
                        continue;
                    }

                    // Angle the stack along the averaged surface normal of the
                    // surrounding wall instead of snapping to a cardinal axis.
                    // The estimate is blended with (and biased toward) the
                    // averaged open-face direction so the tilt stays modest --
                    // a wildly angled axis would tip the footprint disc off the
                    // flat wall section it was sampled from, leaving most
                    // columns to mold against nothing (floating in open air or
                    // buried with no nearby surface).
                    let axis = match local_outward_axis(solid, p, res, 2) {
                        Some(avg) => (face_avg * 1.5 + avg).try_normalize().unwrap_or(face_avg),
                        None => face_avg,
                    };

                    // Reject spots where the vent would exhale straight back into rock.
                    if !has_clear_opening(solid, p, axis, res, min_clear) {
                        continue;
                    }

                    let tangent = choose_tangent(normal, p, seed);
                    let bitangent = normal.cross(tangent);

                    // Reject sharp edges/corners where the wall geometry
                    // breaks down across the footprint area.
                    if !is_locally_flat(solid, p, axis, tangent, bitangent, top_outer, res) {
                        continue;
                    }

                    let score = hash_u32(
                        p.x + normal.x * 17,
                        p.y + normal.y * 17,
                        p.z + normal.z * 17,
                        seed ^ 0x6D2B_79F5,
                    );
                    candidates.push((score, p, normal, tangent, axis));
                }
            }
        }
    }

    candidates.sort_by_key(|(score, _, _, _, _)| *score);

    let mut specs = Vec::new();
    for (_, surface, normal, tangent, axis) in candidates {
        if specs.len() >= params.geothermal_count as usize {
            break;
        }

        let spec = VentSpec {
            surface,
            normal,
            tangent,
            bitangent: normal.cross(tangent),
            axis,
            placement_mode,
            length,
            width,
            depth,
            heat_radius,
            glow_radius,
        };

        if placement_mode == 1 {
            // Make sure the molded stack actually reaches open space and
            // doesn't end up sealed inside the rock. Only the central
            // (anchor) column is checked here -- fringe footprint columns
            // are allowed to mold against whatever local terrain they land
            // on without invalidating the whole vent. Molding walks along
            // the cardinal `normal` (precise, single-axis steps) while the
            // opening is offset along the angled `axis` to match how
            // `stack_voxels` builds the stack.
            let surface_f = Vec3::new(surface.x as f32, surface.y as f32, surface.z as f32);
            let normal_vec = Vec3::new(normal.x as f32, normal.y as f32, normal.z as f32);
            let shift = surface_shift(solid, surface_f, normal_vec, res, 3);
            let opening = round_vec3(surface_f + normal_vec * shift + axis * depth as f32);
            if !in_bounds(opening, res as i32) || solid[idx(opening, res)] {
                continue;
            }
        }

        specs.push(spec);
    }

    specs
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

pub(crate) fn stack_voxels(
    spec: VentSpec,
    seed: u32,
    solid: &[bool],
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
    let normal_vec = Vec3::new(
        spec.normal.x as f32,
        spec.normal.y as f32,
        spec.normal.z as f32,
    );

    for base in footprint_centers(spec, seed) {
        let base_f = Vec3::new(base.x as f32, base.y as f32, base.z as f32);
        let stack_axis = if spec.placement_mode == 0 {
            (grid_center - base_f).try_normalize().unwrap_or(spec.axis)
        } else {
            spec.axis
        };
        let radial_out = -stack_axis;

        // For angled (mode 1) stacks the cardinal tangent/bitangent pair is
        // generally not perpendicular to `stack_axis` any more. Re-project
        // them onto the plane perpendicular to the axis so each cross
        // section stays a clean disc instead of a sheared ellipse, which
        // otherwise leaves gaps in the extruded stack.
        let (cs_tangent, cs_bitangent) = if spec.placement_mode == 1 {
            let t = (tangent - stack_axis * tangent.dot(stack_axis))
                .try_normalize()
                .unwrap_or(tangent);
            let b = stack_axis.cross(t).try_normalize().unwrap_or(bitangent);
            (t, b)
        } else {
            (tangent, bitangent)
        };

        for a in -base_outer..=base_outer {
            for b in -base_outer..=base_outer {
                let lateral = cs_tangent * a as f32 + cs_bitangent * b as f32;
                let radial_len = lateral.length();
                if radial_len > base_outer as f32 + 0.25 {
                    continue;
                }

                let col_origin = if spec.placement_mode == 0 {
                    // Each footprint sample is projected onto the sphere
                    // before the stack rises inward, so the flared base
                    // follows the curved boundary instead of cutting a flat
                    // plane through it.
                    (base_f + lateral - grid_center)
                        .try_normalize()
                        .unwrap_or(radial_out)
                        * sphere_radius
                        + grid_center
                } else {
                    // Cave-wall stacks mold their base to the local wall: walk
                    // along the cardinal normal (a precise, single-axis step
                    // per voxel) to find where this column actually crosses
                    // the rock surface, so the base neither floats over a
                    // recess (overhang) nor buries into a bulge. The stack
                    // then grows away from this molded point along the
                    // angled `stack_axis`.
                    let flat = base_f + lateral;
                    let shift = surface_shift(solid, flat, normal_vec, res, 3);
                    flat + normal_vec * shift
                };

                for h in 0..=spec.depth {
                    let height_t = h as f32 / spec.depth.max(1) as f32;
                    let outer = (base_outer as f32 * (1.0 - height_t) + top_outer as f32 * height_t)
                        .round()
                        .max(top_outer as f32) as i32;
                    if radial_len > outer as f32 + 0.25 {
                        continue;
                    }

                    let p = round_vec3(col_origin + stack_axis * h as f32);

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
    let grid_size = density.len();
    if grid_size < 8 {
        return;
    }
    if params.geothermal_enabled == 0 || params.geothermal_count == 0 {
        return;
    }

    // Vent placement is decided on the canonical resolution^3/world_radius
    // grid (the same grid `SolidMaskGenerator::build_solid_array` produces
    // for the geothermal heat/glow fields), then the resulting voxels are
    // mapped into this mesh's own density grid by world position. Deciding
    // placement on this grid's own (mesh) resolution would let the two
    // disagree on local terrain, leaving glow with no matching stack
    // geometry (or vice versa).
    let world_center = Vec3::from(params.world_center);
    let world_radius = params.world_radius;
    let res_canonical = crate::simulation::fluid_simulation::GRID_RESOLUTION as usize;
    let solid_mask_gen = crate::simulation::fluid_simulation::SolidMaskGenerator::new(
        res_canonical as u32,
        world_center,
        world_radius,
    );
    let solid_canonical = solid_mask_gen.build_solid_array(params);

    let cell_size_canonical = world_radius * 2.0 / res_canonical as f32;
    let origin_canonical = world_center - Vec3::splat(world_radius);

    let resolution = grid_size - 1;
    let cave_generation_radius = world_radius + 3.0;
    let cell_size_mesh = cave_generation_radius * 2.0 / resolution as f32;
    let origin_mesh = world_center - Vec3::splat(cave_generation_radius);

    let to_mesh_grid = |p: IVec3| -> Option<IVec3> {
        let world_pos =
            origin_canonical + Vec3::new(p.x as f32, p.y as f32, p.z as f32) * cell_size_canonical;
        let g = round_vec3((world_pos - origin_mesh) / cell_size_mesh);
        if g.x >= 0
            && g.y >= 0
            && g.z >= 0
            && (g.x as usize) < grid_size
            && (g.y as usize) < grid_size
            && (g.z as usize) < grid_size
        {
            Some(g)
        } else {
            None
        }
    };

    for spec in generate_specs(
        &solid_canonical,
        res_canonical,
        params,
        params.seed ^ 0xCE11_5EED,
    ) {
        let (walls, core, _, _) = stack_voxels(spec, params.seed, &solid_canonical, res_canonical);
        for p in walls {
            if let Some(g) = to_mesh_grid(p) {
                density[g.x as usize][g.y as usize][g.z as usize] = threshold + 0.35;
            }
        }
        for p in core {
            if let Some(g) = to_mesh_grid(p) {
                density[g.x as usize][g.y as usize][g.z as usize] = threshold - 0.5;
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
        let (walls, core, openings, glow_sources) = stack_voxels(spec, params.seed, &solid, res);

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
            let radial_in = if spec.placement_mode == 0 {
                (grid_center - p_f).try_normalize().unwrap_or(Vec3::Y)
            } else {
                spec.axis
            };
            let source_radius = (spec.width / 2).max(1);

            for h in 0..=spec.heat_radius {
                let axial_t = h as f32 / spec.heat_radius.max(1) as f32;
                let cone_radius = source_radius + (h / 4).max(0);
                let forward = p_f + radial_in * h as f32;
                for a in -cone_radius..=cone_radius {
                    for b in -cone_radius..=cone_radius {
                        let lateral = ((a * a + b * b) as f32).sqrt();
                        if lateral > cone_radius as f32 + 0.25 {
                            continue;
                        }

                        let q = round_vec3(forward + tangent * a as f32 + bitangent * b as f32);
                        if !in_bounds(q, res as i32) || solid[idx(q, res)] {
                            continue;
                        }

                        let axial = (1.0 - axial_t).max(0.0).powf(1.25);
                        let radial = 1.0 - (lateral / cone_radius.max(1) as f32).min(1.0) * 0.55;
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
            let radial_in = if spec.placement_mode == 0 {
                (grid_center - source_f).try_normalize().unwrap_or(Vec3::Y)
            } else {
                spec.axis
            };

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

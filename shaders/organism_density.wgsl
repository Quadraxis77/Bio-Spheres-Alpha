// Organism Density Compute Shader
//
// Generates a 3D density field from cell positions for organism skin rendering.
// Each live cell contributes a smooth metaball blob to nearby voxels.
//
// Three passes:
//   1. clear_density  - zero the atomic i32 accumulation buffer
//   2. generate_density - per-cell dispatch; atomicAdd contributions into accumulation buffer
//   3. normalize_density - convert fixed-point i32 accumulators to f32 density output

struct OrganismDensityParams {
    // vec3 + f32 = 16 bytes (matches [f32;3] + f32 in Rust repr(C))
    grid_origin: vec3<f32>,
    cell_size: f32,
    // 4 × u32/f32 = 16 bytes
    grid_resolution: u32,
    skin_radius_scale: f32,
    max_cells: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform>            params: OrganismDensityParams;
@group(0) @binding(1) var<storage, read>      position_and_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>      death_flags: array<u32>;
@group(0) @binding(3) var<storage, read>      cell_count: array<u32>;
@group(0) @binding(4) var<storage, read_write> density_accum: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> organism_density: array<f32>;

// Fixed-point scale: contribution [0,1] stored as [0, SCALE]
const FIXED_SCALE: f32 = 32768.0;

// Maximum voxel search radius per cell.
// Limits inner-loop iterations for large cells relative to voxel size.
const MAX_VOXEL_RADIUS: i32 = 28;  // Doubled for 256 grid

fn voxel_index(x: u32, y: u32, z: u32) -> u32 {
    let r = params.grid_resolution;
    // Standard indexing for original resolution (no padding)
    return x + y * r + z * r * r;
}

// Smooth Hermite (smoothstep) metaball kernel.
// t = 1 at cell center, 0 at influence radius.
fn metaball_kernel(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: clear_density
// Each thread zeros one element of density_accum.
// Dispatch: ceil(grid_resolution^3 / 256) workgroups of (256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn clear_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Clear original resolution buffer like water surface nets
    let total_size = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    if gid.x >= total_size { return; }
    atomicStore(&density_accum[gid.x], 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: generate_density
// One thread per cell slot up to max_cells.
// Each thread iterates over its voxel bounding box and atomicAdds contributions.
// Dispatch: ceil(max_cells / 64) workgroups of (64, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(64, 1, 1)
fn generate_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;

    // Early-out: beyond allocated cells
    let total_cells = cell_count[0];
    if cell_idx >= total_cells { return; }

    // Skip dead cells
    if death_flags[cell_idx] != 0u { return; }

    let pos_mass = position_and_mass[cell_idx];
    let pos      = pos_mass.xyz;
    let mass     = pos_mass.w;
    if mass <= 0.0 { return; }

    let influence_radius = clamp(mass, 0.5, 2.0) * params.skin_radius_scale;
    let cell_size        = params.cell_size;
    let grid_origin      = params.grid_origin;
    let res              = i32(params.grid_resolution);

    // Map cell position to voxel coordinates (center voxel)
    let local = pos - grid_origin;
    let cx    = i32(local.x / cell_size);
    let cy    = i32(local.y / cell_size);
    let cz    = i32(local.z / cell_size);

    // Voxel radius to cover the influence sphere
    let vr = min(MAX_VOXEL_RADIUS, i32(influence_radius / cell_size) + 1);

    // Extended bounds with boundary padding to prevent edge holes during movement
    // Add 1 voxel margin to ensure smooth transitions at boundaries
    let vmin = max(vec3<i32>(0),       vec3<i32>(cx - vr, cy - vr, cz - vr));
    let vmax = min(vec3<i32>(res - 1), vec3<i32>(cx + vr, cy + vr, cz + vr));

    let inv_radius = 1.0 / influence_radius;

    for (var vz = vmin.z; vz <= vmax.z; vz++) {
        for (var vy = vmin.y; vy <= vmax.y; vy++) {
            for (var vx = vmin.x; vx <= vmax.x; vx++) {
                // World-space position matching surface nets coordinate system
                let voxel_ws = grid_origin
                    + vec3<f32>(f32(vx), f32(vy), f32(vz)) * cell_size;

                let dist = length(voxel_ws - pos);
                if dist >= influence_radius { continue; }

                let t    = 1.0 - dist * inv_radius;
                let contrib = metaball_kernel(t);
                let fixed   = i32(contrib * FIXED_SCALE);

                let idx = voxel_index(u32(vx), u32(vy), u32(vz));
                atomicAdd(&density_accum[idx], fixed);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3: normalize_density
// Converts the fixed-point i32 accumulator to a normalised f32 [0,N] density.
// Dispatch: ceil(grid_resolution / 4)^3 workgroups of (4, 4, 4)
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(4, 4, 4)
fn normalize_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.grid_resolution;
    // Handle padded resolution in normalize pass too
    if gid.x >= res || gid.y >= res || gid.z >= res { return; }

    let idx = voxel_index(gid.x, gid.y, gid.z);
    let raw = atomicLoad(&density_accum[idx]);
    organism_density[idx] = f32(raw) / FIXED_SCALE;
}

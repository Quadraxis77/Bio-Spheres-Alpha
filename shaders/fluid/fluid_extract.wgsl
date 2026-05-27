// Extract density from fluid state for rendering
// Both fluid simulation and surface nets use 128³ grid with 128-stride indexing

struct ExtractParams {
    grid_resolution: u32,  // Fluid simulation resolution (128)
    gravity_mode: u32,     // 0=X, 1=Y, 2=Z, 3=radial
    _pad1: u32,
    _pad2: u32,
    grid_origin: vec3<f32>,
    cell_size: f32,
    gravity_magnitude: f32,  // 0 = zero-g, blobs allowed
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

@group(0) @binding(0) var<uniform> extract_params: ExtractParams;
@group(0) @binding(1) var<storage, read> fluid_state: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> density_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> fluid_type_out: array<u32>;
@group(0) @binding(4) var<storage, read> solid_mask: array<u32>;

fn get_fluid_type(state: u32) -> u32 {
    return state & 0xFFFFu;
}

fn sample_fluid_state(x: u32, y: u32, z: u32) -> u32 {
    let res = extract_params.grid_resolution;
    if x >= res || y >= res || z >= res {
        return 0u; // Out of bounds = empty
    }
    let idx = x + y * res + z * res * res;
    return atomicLoad(&fluid_state[idx]);
}

fn is_solid(x: u32, y: u32, z: u32) -> bool {
    let res = extract_params.grid_resolution;
    if x >= res || y >= res || z >= res {
        return false;
    }
    return solid_mask[x + y * res + z * res * res] == 1u;
}

// Returns true if the water voxel at (x,y,z) is resting on something solid or liquid.
// Unsupported (falling) water is excluded from the density field so it doesn't
// form blobs in the surface nets mesh — only puddles and pools are rendered.
fn is_water_supported(x: u32, y: u32, z: u32) -> bool {
    let res = extract_params.grid_resolution;

    var dx: i32 = 0;
    var dy: i32 = 0;
    var dz: i32 = 0;

    if extract_params.gravity_mode == 3u {
        // Radial gravity: "down" is toward the origin from this voxel's world position.
        // Convert voxel centre to world space, then pick the neighbour whose grid offset
        // most closely aligns with the inward (toward-origin) direction.
        let world_pos = extract_params.grid_origin
            + vec3<f32>(f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5) * extract_params.cell_size;
        let r = length(world_pos);
        if r < 0.001 {
            dy = -1; // fallback at origin
        } else {
            // Inward unit vector
            let inward = -world_pos / r;
            // Pick the axis-aligned neighbour most aligned with inward direction
            let ax = abs(inward.x);
            let ay = abs(inward.y);
            let az = abs(inward.z);
            if ax >= ay && ax >= az {
                dx = select(-1, 1, inward.x < 0.0);
            } else if ay >= az {
                dy = select(-1, 1, inward.y < 0.0);
            } else {
                dz = select(-1, 1, inward.z < 0.0);
            }
        }
    } else if extract_params.gravity_mode == 0u {
        dx = -1;  // gravity pulls -X
    } else if extract_params.gravity_mode == 2u {
        dz = -1;  // gravity pulls -Z
    } else {
        dy = -1;  // gravity pulls -Y (mode 1, default)
    }

    let nx = i32(x) + dx;
    let ny = i32(y) + dy;
    let nz = i32(z) + dz;

    if nx < 0 || ny < 0 || nz < 0 {
        return false;
    }
    let ux = u32(nx);
    let uy = u32(ny);
    let uz = u32(nz);
    if ux >= res || uy >= res || uz >= res {
        return false;
    }

    if is_solid(ux, uy, uz) {
        return true;
    }

    let below_type = get_fluid_type(atomicLoad(&fluid_state[ux + uy * res + uz * res * res]));
    return below_type == 1u || below_type == 2u;
}

@compute @workgroup_size(4, 4, 4)
fn extract_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = extract_params.grid_resolution;  // 128

    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    // Output index in 128³ grid
    let out_idx = gid.x + gid.y * res + gid.z * res * res;
    
    let state = sample_fluid_state(gid.x, gid.y, gid.z);
    let fluid_type = get_fluid_type(state);
    let fill = f32((state >> 16u) & 0xFFFFu) / 65535.0;

    if (fluid_type == 1u || fluid_type == 2u) && fill > 0.0 {
        // In zero gravity, all water contributes to the mesh (floating blobs allowed).
        // With gravity, only supported water forms the mesh surface.
        if fluid_type == 1u
            && extract_params.gravity_magnitude > 0.01
            && !is_water_supported(gid.x, gid.y, gid.z) {
            density_out[out_idx] = 0.0;
        } else {
            density_out[out_idx] = fill;
        }
    } else {
        density_out[out_idx] = 0.0;
    }

    fluid_type_out[out_idx] = fluid_type;
}

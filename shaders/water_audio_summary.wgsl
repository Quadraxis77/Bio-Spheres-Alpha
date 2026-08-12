struct SummaryParams {
    grid_resolution: u32,
    bucket_resolution: u32,
    gravity_mode: u32,
    // Only every sample_stride-th voxel along each axis is examined (see
    // WATER_AUDIO_SAMPLE_STRIDE on the Rust side) - counts are scaled back up
    // by stride^3 below so bucket strength stays comparable to a full scan.
    sample_stride: u32,
    grid_origin: vec3<f32>,
    cell_size: f32,
}

struct AudioBucket {
    flow_count: atomic<u32>,
    rain_count: atomic<u32>,
}

@group(0) @binding(0) var<uniform> params: SummaryParams;
@group(0) @binding(1) var<storage, read> fluid_state: array<u32>;
@group(0) @binding(2) var<storage, read> water_velocity: array<u32>;
@group(0) @binding(3) var<storage, read_write> buckets: array<AudioBucket>;

fn grid_index(x: u32, y: u32, z: u32) -> u32 {
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

fn decode_velocity_component(v: u32) -> f32 {
    if v == 1u { return 1.0; }
    if v == 2u { return -1.0; }
    return 0.0;
}

fn decode_water_velocity(packed: u32) -> vec3<f32> {
    if packed == 0u { return vec3<f32>(0.0); }
    return vec3<f32>(
        decode_velocity_component(packed & 3u),
        decode_velocity_component((packed >> 2u) & 3u),
        decode_velocity_component((packed >> 4u) & 3u)
    );
}

fn world_center() -> vec3<f32> {
    let diameter = f32(params.grid_resolution) * params.cell_size;
    return params.grid_origin + vec3<f32>(diameter * 0.5);
}

fn gravity_dir(world_pos: vec3<f32>) -> vec3<f32> {
    switch (params.gravity_mode) {
        case 0u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 2u: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 3u: {
            let radial = world_center() - world_pos;
            let r = length(radial);
            if r > 0.001 {
                return radial / r;
            }
            return vec3<f32>(0.0, -1.0, 0.0);
        }
        default: { return vec3<f32>(0.0, -1.0, 0.0); }
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) sample_id: vec3<u32>) {
    let res = params.grid_resolution;
    let stride = max(params.sample_stride, 1u);
    let gid = sample_id * stride;
    if gid.x >= res || gid.y >= res || gid.z >= res {
        return;
    }

    let idx = grid_index(gid.x, gid.y, gid.z);
    if (fluid_state[idx] & 0x7u) != 1u {
        return;
    }

    let velocity = decode_water_velocity(water_velocity[idx]);
    if dot(velocity, velocity) < 0.5 {
        return;
    }

    let bucket_res = params.bucket_resolution;
    let bucket = min(gid * bucket_res / res, vec3<u32>(bucket_res - 1u));
    let bucket_idx = bucket.x + bucket.y * bucket_res + bucket.z * bucket_res * bucket_res;

    let world_pos = params.grid_origin + (vec3<f32>(gid) + vec3<f32>(0.5)) * params.cell_size;
    let falling = dot(normalize(velocity), gravity_dir(world_pos)) > 0.65;
    // Scale back up by the voxels each sample stands in for, so bucket
    // strength stays comparable to what a full (stride=1) scan would report.
    let weight = stride * stride * stride;
    if falling {
        atomicAdd(&buckets[bucket_idx].rain_count, weight);
    } else {
        atomicAdd(&buckets[bucket_idx].flow_count, weight);
    }
}

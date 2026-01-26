//! Solid mask generation for fluid system
//!
//! This module generates a solid mask that treats the entire cave volume as solid.
//! It uses the exact same noise function and cave settings to replicate the cave volume.

use glam::Vec3;
use crate::rendering::cave_system::CaveParams;

/// Solid mask generator for fluid system
pub struct SolidMaskGenerator {
    grid_resolution: u32,
    world_center: Vec3,
    world_radius: f32,
}

impl SolidMaskGenerator {
    /// Create a new solid mask generator
    pub fn new(grid_resolution: u32, world_center: Vec3, world_radius: f32) -> Self {
        Self {
            grid_resolution,
            world_center,
            world_radius,
        }
    }

    /// Generate solid mask data using cave generation logic
    /// Returns a vector of u32 values where 1 = solid, 0 = empty
    pub fn generate_solid_mask(&self, cave_params: &CaveParams) -> Vec<u32> {
        let resolution = self.grid_resolution as usize;
        let total_voxels = resolution * resolution * resolution;
        let mut solid_mask = vec![0u32; total_voxels];

        // Calculate cell size
        let world_diameter = self.world_radius * 2.0;
        let cell_size = world_diameter / self.grid_resolution as f32;
        
        // Grid origin (bottom-back-left corner)
        let grid_origin = self.world_center - Vec3::splat(self.world_radius);

        // Generate solid mask for each voxel
        for x in 0..resolution {
            for y in 0..resolution {
                for z in 0..resolution {
                    let voxel_index = x + y * resolution + z * resolution * resolution;
                    
                    // Calculate world position of voxel center
                    let world_pos = grid_origin + Vec3::new(
                        (x as f32 + 0.5) * cell_size,
                        (y as f32 + 0.5) * cell_size,
                        (z as f32 + 0.5) * cell_size,
                    );

                    // Use cave generation logic to determine if this is solid
                    let is_solid = self.is_solid_cave_volume(world_pos, cave_params);
                    
                    solid_mask[voxel_index] = if is_solid { 1 } else { 0 };
                }
            }
        }

        solid_mask
    }

    /// Check if a world position is inside the solid cave volume
    /// Uses the same logic as cave_system.rs::sample_density
    fn is_solid_cave_volume(&self, pos: Vec3, params: &CaveParams) -> bool {
        // Distance from world center (spherical constraint)
        let world_center = Vec3::from(params.world_center);
        let dist_from_center = (pos - world_center).length();
        
        // Extend cave generation 3 units beyond world sphere (same as cave system)
        let cave_generation_radius = params.world_radius + 3.0;
        let sphere_sdf = dist_from_center - cave_generation_radius;

        // Outside extended sphere = solid
        if sphere_sdf > 0.0 {
            return true;
        }

        // Apply domain warping for organic shapes (same as cave system)
        let warped_pos = self.warp_domain(pos, params);

        // Get base noise value using FBM (same as cave system)
        let noise = self.fbm(warped_pos, params);

        // Map noise to density based on cave density parameter
        // density parameter controls how much solid rock vs open space
        // Higher density = more solid rock, lower = more open tunnels
        let cave_threshold = params.density.clamp(0.0, 1.0);

        // Solid rock where noise is above threshold, open tunnels where below
        // This matches the cave generation logic exactly
        noise > cave_threshold
    }

    /// Hash function for single random value at integer coordinates (same as cave system)
    fn hash1(&self, x: i32, y: i32, z: i32, seed: u32) -> f32 {
        let mut h = seed;
        h = h.wrapping_mul(374761393).wrapping_add(x as u32);
        h = h.wrapping_mul(668265263).wrapping_add(y as u32);
        h = h.wrapping_mul(1274126177).wrapping_add(z as u32);
        h ^= h >> 13;
        h = h.wrapping_mul(1274126177);
        h ^= h >> 16;

        h as f32 / u32::MAX as f32
    }

    /// Smooth interpolation (same as cave system)
    fn smoothstep(&self, t: f32) -> f32 {
        t * t * (3.0 - 2.0 * t)
    }

    /// 3D value noise - interpolates between random values at lattice points (same as cave system)
    fn value_noise_3d(&self, pos: Vec3, seed: u32) -> f32 {
        // Integer and fractional parts
        let ix = pos.x.floor() as i32;
        let iy = pos.y.floor() as i32;
        let iz = pos.z.floor() as i32;

        let fx = pos.x - pos.x.floor();
        let fy = pos.y - pos.y.floor();
        let fz = pos.z - pos.z.floor();

        // Smooth interpolation weights
        let ux = self.smoothstep(fx);
        let uy = self.smoothstep(fy);
        let uz = self.smoothstep(fz);

        // Random values at 8 corners of the cube
        let c000 = self.hash1(ix, iy, iz, seed);
        let c100 = self.hash1(ix + 1, iy, iz, seed);
        let c010 = self.hash1(ix, iy + 1, iz, seed);
        let c110 = self.hash1(ix + 1, iy + 1, iz, seed);
        let c001 = self.hash1(ix, iy, iz + 1, seed);
        let c101 = self.hash1(ix + 1, iy, iz + 1, seed);
        let c011 = self.hash1(ix, iy + 1, iz + 1, seed);
        let c111 = self.hash1(ix + 1, iy + 1, iz + 1, seed);

        // Trilinear interpolation with smooth weights
        let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;

        let x00 = mix(c000, c100, ux);
        let x10 = mix(c010, c110, ux);
        let x01 = mix(c001, c101, ux);
        let x11 = mix(c011, c111, ux);

        let y0 = mix(x00, x10, uy);
        let y1 = mix(x01, x11, uy);

        mix(y0, y1, uz)
    }

    /// Fractal Brownian Motion - combines multiple octaves of value noise (same as cave system)
    fn fbm(&self, pos: Vec3, params: &CaveParams) -> f32 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_value = 0.0;

        for i in 0..params.octaves {
            let sample_pos = pos * frequency / params.scale;
            // Use different seed for each octave to avoid correlation
            let octave_seed = params.seed.wrapping_add(i * 1337);
            value += amplitude * self.value_noise_3d(sample_pos, octave_seed);
            max_value += amplitude;
            amplitude *= params.persistence;
            frequency *= 2.0;
        }

        // Normalize to 0-1 range
        value / max_value
    }

    /// Domain warping - distorts the input coordinates for more organic shapes (same as cave system)
    fn warp_domain(&self, pos: Vec3, params: &CaveParams) -> Vec3 {
        let warp_scale = params.scale * 0.5;
        let warp_strength = params.smoothness * params.scale;

        // Sample noise at offset positions to get warp vectors
        let warp_seed = params.seed.wrapping_add(9999);
        let wx = self.value_noise_3d(pos / warp_scale, warp_seed) - 0.5;
        let wy = self.value_noise_3d(pos / warp_scale + Vec3::new(31.7, 47.3, 13.1), warp_seed) - 0.5;
        let wz = self.value_noise_3d(pos / warp_scale + Vec3::new(73.9, 19.4, 67.2), warp_seed) - 0.5;

        Vec3::new(
            pos.x + wx * warp_strength,
            pos.y + wy * warp_strength,
            pos.z + wz * warp_strength,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solid_mask_generation() {
        let generator = SolidMaskGenerator::new(32, Vec3::ZERO, 200.0);
        let cave_params = CaveParams::default();
        
        let solid_mask = generator.generate_solid_mask(&cave_params);
        
        // Should have correct number of voxels
        assert_eq!(solid_mask.len(), 32 * 32 * 32);
        
        // Should have both solid and empty voxels
        let solid_count = solid_mask.iter().sum::<u32>();
        assert!(solid_count > 0, "Should have some solid voxels");
        assert!(solid_count < (32 * 32 * 32) as u32, "Should have some empty voxels");
    }

    #[test]
    fn test_solid_cave_volume() {
        let generator = SolidMaskGenerator::new(64, Vec3::ZERO, 200.0);
        let cave_params = CaveParams::default();
        
        // Test outside world sphere (should be solid)
        let outside_pos = Vec3::new(300.0, 0.0, 0.0);
        assert!(generator.is_solid_cave_volume(outside_pos, &cave_params));
        
        // Test inside world sphere (depends on noise)
        let inside_pos = Vec3::new(0.0, 0.0, 0.0);
        let result = generator.is_solid_cave_volume(inside_pos, &cave_params);
        // Result could be either solid or empty depending on noise
        // Just ensure the function runs without panic
        let _ = result;
    }
}

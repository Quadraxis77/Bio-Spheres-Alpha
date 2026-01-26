//! Test module for solid mask generation

use super::solid_mask::*;
use crate::rendering::cave_system::CaveParams;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solid_mask_basic_functionality() {
        let generator = SolidMaskGenerator::new(32, glam::Vec3::ZERO, 200.0);
        let cave_params = CaveParams::default();
        
        let solid_mask = generator.generate_solid_mask(&cave_params);
        
        // Should have correct number of voxels
        assert_eq!(solid_mask.len(), 32 * 32 * 32);
        
        // Should have both solid and empty voxels
        let solid_count = solid_mask.iter().sum::<u32>();
        assert!(solid_count > 0, "Should have some solid voxels");
        assert!(solid_count < (32 * 32 * 32) as u32, "Should have some empty voxels");
        
        println!("Generated solid mask with {} solid voxels out of {} total", 
            solid_count, 32 * 32 * 32);
    }

    #[test]
    fn test_solid_cave_volume_consistency() {
        let generator = SolidMaskGenerator::new(64, glam::Vec3::ZERO, 200.0);
        let cave_params = CaveParams::default();
        
        // Test outside world sphere (should be solid)
        let outside_pos = glam::Vec3::new(300.0, 0.0, 0.0);
        assert!(generator.is_solid_cave_volume(outside_pos, &cave_params));
        
        // Test inside world sphere (depends on noise)
        let inside_pos = glam::Vec3::new(0.0, 0.0, 0.0);
        let result = generator.is_solid_cave_volume(inside_pos, &cave_params);
        // Result could be either solid or empty depending on noise
        // Just ensure the function runs without panic
        let _ = result;
    }

    #[test]
    fn test_cave_params_changes() {
        let generator = SolidMaskGenerator::new(16, glam::Vec3::ZERO, 100.0);
        
        // Test with different cave parameters
        let mut params = CaveParams::default();
        
        // High density (should be mostly solid)
        params.density = 0.9;
        let high_density_mask = generator.generate_solid_mask(&params);
        let high_density_solid = high_density_mask.iter().sum::<u32>();
        
        // Low density (should be mostly empty)
        params.density = 0.1;
        let low_density_mask = generator.generate_solid_mask(&params);
        let low_density_solid = low_density_mask.iter().sum::<u32>();
        
        // High density should have more solid voxels than low density
        assert!(high_density_solid > low_density_solid, 
            "High density should have more solid voxels");
        
        println!("High density: {} solid voxels", high_density_solid);
        println!("Low density: {} solid voxels", low_density_solid);
    }

    #[test]
    fn test_seed_consistency() {
        let generator = SolidMaskGenerator::new(16, glam::Vec3::ZERO, 100.0);
        
        let mut params = CaveParams::default();
        params.seed = 12345;
        
        let mask1 = generator.generate_solid_mask(&params);
        let mask2 = generator.generate_solid_mask(&params);
        
        // Same seed should produce identical results
        assert_eq!(mask1, mask2, "Same seed should produce identical masks");
        
        // Different seed should produce different results
        params.seed = 54321;
        let mask3 = generator.generate_solid_mask(&params);
        assert_ne!(mask1, mask3, "Different seed should produce different masks");
    }
}

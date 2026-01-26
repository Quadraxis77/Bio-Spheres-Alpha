# Solid Mask Implementation for Fluid System

## Overview

This document describes the implementation of a solid mask for the fluid system that treats the entire cave volume as solid. The solid mask uses the exact same noise function and user-set cave settings to replicate the cave volume and updates automatically when cave settings change.

## Architecture

### Components

1. **SolidMaskGenerator** (`src/simulation/fluid_simulation/solid_mask.rs`)
   - Generates solid mask data using cave generation logic
   - Uses identical noise functions as cave system
   - Supports all cave parameters (density, scale, octaves, persistence, threshold, smoothness, seed)

2. **FluidBuffers Integration** (`src/simulation/fluid_simulation/buffers.rs`)
   - Added `update_solid_mask()` method to update GPU buffer
   - Solid mask buffer stores u32 values (1 = solid, 0 = empty)

3. **GPU Scene Integration** (`src/scene/gpu_scene.rs`)
   - Automatic solid mask generation when cave system initializes
   - Updates solid mask when cave parameters change
   - Integrates with existing fluid simulation pipeline

## Key Features

### 1. Exact Cave Replication
The solid mask uses the same noise generation logic as the cave system:

```rust
// Same hash function as cave_system.rs
fn hash1(x: i32, y: i32, z: i32, seed: u32) -> f32

// Same value noise with smoothstep interpolation  
fn value_noise_3d(pos: Vec3, seed: u32) -> f32

// Same FBM with multiple octaves
fn fbm(pos: Vec3, params: &CaveParams) -> f32

// Same domain warping for organic shapes
fn warp_domain(pos: Vec3, params: &CaveParams) -> Vec3
```

### 2. Parameter Synchronization
The solid mask automatically updates when:
- Cave system is first initialized
- Cave parameters are changed via UI
- World diameter changes (affecting cave generation radius)

### 3. Grid Resolution Matching
- Uses same 128³ grid as fluid simulation
- Cell size calculated from world diameter
- World positions mapped to voxel coordinates correctly

## Usage

### Initialization
```rust
// Solid mask generator created automatically when fluid system initializes
let solid_mask_generator = SolidMaskGenerator::new(
    GRID_RESOLUTION,  // 128
    world_center,     // Vec3::ZERO
    world_radius,     // From physics config
);
```

### Updating Mask
```rust
// Called automatically when cave params change
pub fn update_solid_mask(&mut self, queue: &wgpu::Queue) {
    if let (Some(ref fluid_buffers), Some(ref solid_mask_generator), Some(ref cave_renderer)) = 
        (&self.fluid_buffers, &self.solid_mask_generator, &self.cave_renderer) {
        
        let cave_params = cave_renderer.params();
        let solid_mask = solid_mask_generator.generate_solid_mask(cave_params);
        
        // Update GPU buffer
        fluid_buffers.update_solid_mask(queue, &solid_mask);
    }
}
```

## Cave Parameters

The solid mask responds to all cave parameters:

- **density**: Controls solid vs empty ratio (0.0 = all empty, 1.0 = all solid)
- **scale**: Base scale for cave features  
- **octaves**: Number of noise layers for detail
- **persistence**: Detail falloff between octaves
- **threshold**: Boundary threshold for solid/empty
- **smoothness**: Boundary smoothness and domain warping strength
- **seed**: Random seed for reproducible generation

## Performance Considerations

### Memory Usage
- Solid mask: 128³ × 4 bytes = ~2 MB
- Generated on CPU, uploaded to GPU
- Only regenerated when parameters change

### Generation Cost
- O(n³) where n = grid resolution (128)
- Uses efficient noise functions
- Acceptable since regeneration is infrequent

## Testing

The implementation includes comprehensive tests:

```bash
cargo test solid_mask
```

Tests verify:
- Correct mask size and format
- Solid/empty voxel distribution
- Cave parameter effects
- Seed consistency
- Boundary conditions

## Integration Points

### With Fluid Simulation
The solid mask buffer is available to fluid shaders as:
```wgsl
@group(0) @binding(2) var<storage, read> solid_mask: array<u32>;
```

### With Cave System
Shares cave parameters and generation logic:
```rust
let cave_params = cave_renderer.params();
let solid_mask = solid_mask_generator.generate_solid_mask(cave_params);
```

## Future Enhancements

1. **GPU-side Generation**: Move mask generation to compute shader for faster updates
2. **Compression**: Use RLE or bit-packing for large grids
3. **Level of Detail**: Different resolution masks for different simulation scales
4. **Dynamic Updates**: Partial mask updates for localized cave changes

## Conclusion

The solid mask implementation provides a robust, efficient way to treat the entire cave volume as solid in the fluid simulation. It maintains perfect synchronization with cave generation parameters and updates automatically when settings change, ensuring consistent behavior between the cave system and fluid simulation.

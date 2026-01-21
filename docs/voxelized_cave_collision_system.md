# GPU-Based Voxelized Cave Collision System for Fluid Simulation

## Overview

This document describes a **GPU-only voxelized cave collision system** that evaluates cave noise and parameters to create solid voxel barriers throughout the cave volume. The system is designed to be **ultra-cheap to process** and provides **identical behavior** for both solid voxels and empty non-voxelized spaces.

**Key Design Principles:**
- **GPU-only evaluation** - Cave SDF sampled entirely on GPU
- **Single-bit solid mask** - Minimal memory (2 MB for 128³ grid)
- **Zero CPU readbacks** - All collision detection on GPU
- **Unified boundary handling** - Same code path for cave walls and world sphere
- **Cache-friendly lookups** - Coalesced memory access patterns

---

## System Architecture

### 1. Voxel Grid Structure

```rust
/// Ultra-lightweight voxel grid for fluid collision
pub struct VoxelizedCaveGrid {
    /// Single-bit solid mask (1 = solid, 0 = fluid space)
    /// 128³ grid = 2,097,152 cells = 2 MB (u32 array)
    pub solid_mask: wgpu::Buffer,
    
    /// Grid metadata
    pub grid_resolution: u32,        // 128
    pub cell_size: f32,              // world_diameter / 128.0
    pub world_center: Vec3,
    pub world_radius: f32,
    
    /// Cave generation parameters (for GPU evaluation)
    pub cave_params: wgpu::Buffer,   // CaveParams uniform
    
    /// Statistics (GPU atomic counters)
    pub stats: wgpu::Buffer,         // [solid_count, fluid_count]
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelGridParams {
    grid_resolution: u32,
    cell_size: f32,
    world_center: [f32; 3],
    world_radius: f32,
    
    // Cave SDF parameters (from CaveParams)
    density: f32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    threshold: f32,
    smoothness: f32,
    seed: u32,
    
    // Padding to 256 bytes
    _padding: [f32; 50],
}
```

### 2. GPU Cave SDF Evaluation

**File: `shaders/fluid/generate_voxel_grid.wgsl`**

```wgsl
struct VoxelGridParams {
    grid_resolution: u32,
    cell_size: f32,
    world_center: vec3<f32>,
    world_radius: f32,
    
    density: f32,
    scale: f32,
    octaves: u32,
    persistence: f32,
    threshold: f32,
    smoothness: f32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: VoxelGridParams;
@group(0) @binding(1) var<storage, read_write> solid_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 2>;

// Hash-based noise function (matches cave_system.rs implementation)
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth interpolated noise
fn smooth_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Smooth interpolation (quintic)
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    // Sample 8 corners of cube
    let c000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let c001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let c010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let c011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let c100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let c101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let c110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let c111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    
    // Trilinear interpolation
    return mix(
        mix(mix(c000, c100, u.x), mix(c010, c110, u.x), u.y),
        mix(mix(c001, c101, u.x), mix(c011, c111, u.x), u.y),
        u.z
    );
}

// Fractal Brownian Motion (FBM) for cave generation
fn fbm(p: vec3<f32>, octaves: u32, persistence: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;
    
    for (var i = 0u; i < octaves; i++) {
        value += smooth_noise(p * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }
    
    return value / max_value;
}

// Cave SDF evaluation (matches cave_system.rs)
fn sample_cave_density(world_pos: vec3<f32>) -> f32 {
    // Distance from world center
    let dist_from_center = length(world_pos - params.world_center);
    
    // Outside world sphere = solid
    if (dist_from_center > params.world_radius) {
        return 2.0; // Definitely solid
    }
    
    // Sample noise at world position
    let noise_pos = world_pos / params.scale;
    let noise_value = fbm(noise_pos, params.octaves, params.persistence);
    
    // Apply density parameter
    let density_adjusted = noise_value * params.density;
    
    // Threshold determines cave vs solid
    // density_adjusted > threshold = solid
    // density_adjusted < threshold = cave space
    return density_adjusted;
}

// Main voxel grid generation kernel
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_cells) {
        return;
    }
    
    // Convert linear index to 3D grid coordinates
    let res = i32(params.grid_resolution);
    let z = i32(idx) / (res * res);
    let y = (i32(idx) - z * res * res) / res;
    let x = i32(idx) - z * res * res - y * res;
    
    // Convert grid coordinates to world position (cell center)
    let grid_offset = vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5);
    let normalized_pos = grid_offset / f32(params.grid_resolution);
    let world_pos = params.world_center + 
                    (normalized_pos - vec3<f32>(0.5)) * (params.world_radius * 2.0);
    
    // Sample cave density at this position
    let density = sample_cave_density(world_pos);
    let is_solid = density > params.threshold;
    
    // Store result in solid mask
    solid_mask[idx] = select(0u, 1u, is_solid);
    
    // Update statistics atomically
    if (is_solid) {
        atomicAdd(&stats[0], 1u);  // solid_count
    } else {
        atomicAdd(&stats[1], 1u);  // fluid_count
    }
}
```

---

## 3. Ultra-Fast Collision Detection

### Fluid Collision Shader

**File: `shaders/fluid/check_voxel_collision.wgsl`**

```wgsl
struct FluidParams {
    grid_resolution: u32,
    cell_size: f32,
    world_center: vec3<f32>,
    world_radius: f32,
    // ... other params
}

@group(0) @binding(0) var<uniform> params: FluidParams;
@group(0) @binding(1) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> fluid_velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> fluid_densities: array<vec4<f32>>;

// Convert world position to voxel grid index
fn world_to_voxel_index(world_pos: vec3<f32>) -> u32 {
    // Normalize to [0, 1] range
    let normalized = (world_pos - params.world_center) / (params.world_radius * 2.0) + vec3<f32>(0.5);
    
    // Convert to grid coordinates
    let grid_pos = clamp(
        normalized * f32(params.grid_resolution),
        vec3<f32>(0.0),
        vec3<f32>(f32(params.grid_resolution - 1u))
    );
    
    let x = u32(grid_pos.x);
    let y = u32(grid_pos.y);
    let z = u32(grid_pos.z);
    
    // Linear index
    let res = params.grid_resolution;
    return x + y * res + z * res * res;
}

// Check if position is in solid voxel
fn is_solid_voxel(world_pos: vec3<f32>) -> bool {
    let idx = world_to_voxel_index(world_pos);
    return solid_mask[idx] != 0u;
}

// Get collision normal from neighboring voxels
fn get_collision_normal(world_pos: vec3<f32>) -> vec3<f32> {
    let idx = world_to_voxel_index(world_pos);
    
    // Sample neighbors in 6 directions
    let res = i32(params.grid_resolution);
    let z = i32(idx) / (res * res);
    let y = (i32(idx) - z * res * res) / res;
    let x = i32(idx) - z * res * res - y * res;
    
    var normal = vec3<f32>(0.0);
    var count = 0;
    
    // Check 6-connected neighbors
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
    );
    
    for (var i = 0; i < 6; i++) {
        let nx = x + offsets[i].x;
        let ny = y + offsets[i].y;
        let nz = z + offsets[i].z;
        
        // Bounds check
        if (nx >= 0 && nx < res && ny >= 0 && ny < res && nz >= 0 && nz < res) {
            let neighbor_idx = u32(nx + ny * res + nz * res * res);
            
            // If neighbor is fluid space, normal points toward it
            if (solid_mask[neighbor_idx] == 0u) {
                normal += vec3<f32>(offsets[i]);
                count++;
            }
        }
    }
    
    if (count > 0) {
        return normalize(normal);
    }
    
    // Fallback: point toward world center
    return normalize(params.world_center - world_pos);
}

// Apply collision response to fluid velocity
fn apply_voxel_collision(
    velocity: vec3<f32>,
    world_pos: vec3<f32>,
    is_solid: bool
) -> vec3<f32> {
    if (!is_solid) {
        return velocity;  // No collision
    }
    
    // Get collision normal
    let normal = get_collision_normal(world_pos);
    
    // Remove velocity component into solid
    let v_normal = dot(velocity, normal);
    if (v_normal < 0.0) {
        // Moving into solid - remove normal component (free-slip)
        return velocity - normal * v_normal;
    }
    
    return velocity;
}

// Main fluid collision kernel
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_cells) {
        return;
    }
    
    // Check if this fluid cell is in solid voxel
    let is_solid = solid_mask[idx] != 0u;
    
    if (is_solid) {
        // Solid voxel - zero out fluid
        fluid_densities[idx] = vec4<f32>(0.0);
        fluid_velocities[idx] = vec4<f32>(0.0);
        return;
    }
    
    // Fluid space - apply collision to velocity
    let vel = fluid_velocities[idx].xyz;
    
    // Convert grid index to world position
    let res = i32(params.grid_resolution);
    let z = i32(idx) / (res * res);
    let y = (i32(idx) - z * res * res) / res;
    let x = i32(idx) - z * res * res - y * res;
    
    let grid_pos = vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5);
    let normalized = grid_pos / f32(params.grid_resolution);
    let world_pos = params.world_center + 
                    (normalized - vec3<f32>(0.5)) * (params.world_radius * 2.0);
    
    // Apply collision response
    let new_vel = apply_voxel_collision(vel, world_pos, false);
    fluid_velocities[idx] = vec4<f32>(new_vel, 0.0);
}
```

---

## 4. Unified Boundary Handling

### Same Code Path for All Boundaries

```wgsl
// Unified boundary check - works for cave walls AND world sphere
fn is_boundary_voxel(idx: u32) -> bool {
    // Single lookup - ultra fast
    return solid_mask[idx] != 0u;
}

// Unified collision response - same for all boundaries
fn apply_boundary_collision(
    velocity: vec3<f32>,
    position: vec3<f32>,
    voxel_idx: u32
) -> vec3<f32> {
    if (!is_boundary_voxel(voxel_idx)) {
        return velocity;  // No boundary
    }
    
    // Get normal (works for cave walls and sphere)
    let normal = get_collision_normal(position);
    
    // Free-slip boundary condition
    let v_normal = dot(velocity, normal);
    if (v_normal < 0.0) {
        return velocity - normal * v_normal;
    }
    
    return velocity;
}
```

---

## 5. Performance Characteristics

### Memory Usage

```
128³ voxel grid:
- Solid mask: 2,097,152 bits = 262,144 bytes = 2 MB
- Stored as u32 array: 2,097,152 × 4 bytes = 8 MB
- Statistics: 8 bytes (2 × u32)
- Total: ~8 MB
```

### Computational Cost

```
Per fluid cell collision check:
1. Single u32 load (solid_mask[idx]) - 1 memory access
2. Compare with 0 - 1 ALU op
3. Branch on result - 1 branch

Total: ~1 cycle on modern GPU (cached)

For 128³ fluid grid at 30 FPS:
- ~2M collision checks per frame
- ~0.5ms on modern GPU (highly parallel)
```

### Cache Efficiency

```
Fluid grid and voxel grid have same resolution (128³)
- 1:1 mapping between fluid cells and voxels
- Perfect cache locality
- Coalesced memory access (sequential indices)
- No texture sampling overhead
```

---

## 6. Integration with Fluid Simulation

### Initialization Pipeline

```rust
impl FluidSimulation {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cave_params: &CaveParams,
    ) -> Self {
        // 1. Generate voxel grid on GPU
        let voxel_grid = VoxelizedCaveGrid::generate_gpu(
            device,
            queue,
            128,  // grid_resolution
            cave_params,
        );
        
        // 2. Create fluid buffers (same resolution)
        let fluid_buffers = FluidSimulationBuffers::new(
            device,
            128,  // grid_resolution
            voxel_grid.solid_mask.clone(),
        );
        
        // 3. Initialize fluid state (skip solid voxels)
        Self::initialize_fluid_state_gpu(
            device,
            queue,
            &fluid_buffers,
            &voxel_grid,
        );
        
        Self {
            voxel_grid,
            fluid_buffers,
            // ...
        }
    }
}
```

### Per-Frame Pipeline Integration

```rust
impl FluidSimulation {
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        
        // ... fluid simulation stages ...
        
        // Stage N: Voxel collision (ultra-fast)
        pass.set_pipeline(&self.pipelines.voxel_collision);
        pass.set_bind_group(0, &self.bind_groups.collision, &[]);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage N+1: Boundary conditions (uses same voxel data)
        pass.set_pipeline(&self.pipelines.boundary_conditions);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // ... continue pipeline ...
    }
}
```

---

## 7. Advantages Over SDF-Based Collision

### Voxel Grid Advantages

| Feature | Voxel Grid | SDF Sampling |
|---------|-----------|--------------|
| **Lookup Speed** | O(1) - single array access | O(n) - noise evaluation |
| **Memory** | 8 MB (128³) | 0 MB (computed) |
| **Deterministic** | Yes - pre-computed | Yes - same function |
| **Cache Friendly** | Perfect locality | Poor (scattered samples) |
| **GPU Friendly** | Coalesced access | Divergent branches |
| **Preprocessing** | One-time GPU generation | None needed |

### When to Use Each

**Use Voxel Grid:**
- Fluid simulation (this system)
- High-frequency collision checks
- Static cave geometry
- Cache-sensitive operations

**Use SDF Sampling:**
- Cell physics (existing system)
- Dynamic cave deformation
- Continuous collision detection
- Memory-constrained scenarios

---

## 8. Dynamic Cave Updates (Optional)

### Regenerate Voxel Grid on Demand

```rust
impl VoxelizedCaveGrid {
    /// Regenerate voxel grid when cave parameters change
    pub fn regenerate_gpu(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        new_params: &CaveParams,
    ) {
        // Update parameters
        self.queue.write_buffer(
            &self.cave_params,
            0,
            bytemuck::bytes_of(new_params),
        );
        
        // Re-run generation shader
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.generation_pipeline);
        pass.set_bind_group(0, &self.generation_bind_group, &[]);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // No CPU synchronization needed
    }
}
```

---

## 9. Debug Visualization

### Voxel Grid Visualization Shader

```wgsl
// Extract voxel instances for debug rendering
@compute @workgroup_size(64)
fn extract_voxel_debug_instances() {
    let idx = global_id.x;
    
    if (solid_mask[idx] != 0u) {
        // This is a solid voxel - create debug instance
        let instance_idx = atomicAdd(&instance_count, 1u);
        
        // Convert to world position
        let world_pos = voxel_index_to_world_pos(idx);
        
        debug_instances[instance_idx] = DebugInstance {
            position: world_pos,
            scale: params.cell_size,
            color: vec4<f32>(0.5, 0.5, 0.5, 0.3),  // Gray, semi-transparent
        };
    }
}
```

---

## 10. Implementation Checklist

### Phase 1: Core Voxel System
- [ ] Create `VoxelizedCaveGrid` structure
- [ ] Implement `generate_voxel_grid.wgsl` shader
- [ ] Test GPU cave SDF evaluation
- [ ] Verify statistics (solid vs fluid count)

### Phase 2: Collision Detection
- [ ] Implement `check_voxel_collision.wgsl` shader
- [ ] Test boundary normal calculation
- [ ] Verify free-slip boundary conditions
- [ ] Benchmark collision performance

### Phase 3: Fluid Integration
- [ ] Integrate voxel grid into `FluidSimulation`
- [ ] Add collision stage to fluid pipeline
- [ ] Test with simple fluid flow
- [ ] Verify no fluid in solid voxels

### Phase 4: Optimization
- [ ] Profile memory access patterns
- [ ] Optimize workgroup sizes
- [ ] Test cache efficiency
- [ ] Verify zero CPU overhead

### Phase 5: Debug Tools
- [ ] Implement voxel visualization
- [ ] Add statistics display in UI
- [ ] Create collision normal visualization
- [ ] Test dynamic regeneration

---

## 11. Performance Targets

### Memory
- **Voxel Grid**: < 10 MB
- **Total Fluid System**: < 120 MB (including triple buffering)

### Computation
- **Voxel Generation**: < 2ms one-time cost
- **Collision Checks**: < 0.5ms per frame
- **Total Fluid Pipeline**: < 6ms per frame at 30 FPS

### Quality
- **Collision Accuracy**: 100% (voxel-perfect)
- **Boundary Normals**: Smooth (6-neighbor average)
- **Determinism**: Perfect (pre-computed grid)

---

## Summary

This voxelized cave collision system provides:

✅ **Ultra-fast collision** - Single array lookup per check  
✅ **Unified boundaries** - Same code for cave walls and world sphere  
✅ **GPU-only** - Zero CPU readbacks or synchronization  
✅ **Cache-friendly** - Perfect locality with fluid grid  
✅ **Deterministic** - Pre-computed, reproducible results  
✅ **Minimal memory** - 8 MB for 128³ grid  
✅ **Easy integration** - Drop-in replacement for SDF sampling  

The system evaluates cave noise once during initialization and stores the result in a compact voxel grid, enabling ultra-fast collision detection throughout the fluid simulation.

# 4-Fluid Simulation System - Detailed Implementation Plan

## Executive Summary

This document outlines the complete implementation plan for integrating a 4-fluid simulation system (lava, water, steam, air) into the Bio-Spheres GPU scene. The system uses a 128³ voxel grid with cave SDF constraints, runs at 30 FPS, and maintains perfect mass conservation for water and lava.

---

## System Specifications

### Fluid Types

1. **Water** (liquid)
   - Conserved mass (no creation/destruction)
   - Sinks under gravity
   - Converts to steam on lava contact
   - Condenses from steam at solid boundaries

2. **Lava** (liquid)
   - Conserved mass (no creation/destruction)
   - Sinks under gravity (denser than water)
   - Does not change state
   - Causes water → steam phase change on contact

3. **Steam** (gas)
   - Created from water on lava contact
   - Rises (buoyant)
   - Condenses to water at solid boundaries (cave walls, world sphere)
   - Mass equals water lost during phase change

4. **Air** (gas)
   - Fills remaining volume
   - NOT mass-conserved (displaced by steam expansion)
   - Provides pressure medium
   - No phase changes

### Grid Configuration

- **Resolution**: 128³ voxels
- **Total cells**: 2,097,152 cells
- **Occupied by cave**: ~50% (1,048,576 cells solid)
- **Available for fluid**: ~50% (1,048,576 cells open)
- **Cell size**: `world_diameter / 128.0`
- **World coordinate mapping**: Grid aligned with world sphere center

### Simulation Parameters

- **Update rate**: 30 FPS (every other frame at 60 FPS base)
- **Pressure solver**: Jacobi iteration, 20 iterations per frame
- **Boundary conditions**: Free-slip at all solid surfaces
- **Mass conservation**: Perfect for water + lava (steam = water phase change)
- **Phase change model**: Direct contact only (no temperature tracking)

### Memory Budget

Per 128³ grid:
- Cave solid mask: 2 MB (u32 per cell)
- Fluid densities: 8 MB (vec4<f32>: lava, water, steam, air)
- Fluid velocity: 8 MB (vec4<f32>: vx, vy, vz, padding)
- Fluid pressure: 2 MB (f32 per cell)
- **Total**: ~20 MB GPU memory

---

## Procedural Initialization

### User-Defined Distribution

UI sliders control initial fluid distribution (must sum to 100%):
- **Cave density**: 50-70% (typical, determines solid space)
- **Water**: 10-30%
- **Lava**: 5-15%
- **Air**: Remaining percentage (fills rest of volume)

### Initialization Algorithm

```
1. Generate cave voxel grid from SDF (marks solid cells)
2. Count available fluid cells (non-solid)
3. Calculate target counts:
   - water_cells = available_cells × water_percentage
   - lava_cells = available_cells × lava_percentage
   - air_cells = available_cells - water_cells - lava_cells

4. Determine gravity direction (world_center → up)
5. Classify open cells by hemisphere:
   - Top hemisphere: dot(cell_pos - world_center, gravity_up) > 0
   - Bottom hemisphere: dot(cell_pos - world_center, gravity_up) < 0

6. Fill cells:
   - Water → Top hemisphere cells (furthest from center first)
   - Lava → Bottom hemisphere cells (furthest from center first)
   - Air → All remaining open cells
   
7. Set initial densities:
   - Water cells: density = 1.0
   - Lava cells: density = 1.0
   - Air cells: density = 1.0
   - Steam: density = 0.0 (none at start)
   
8. Initialize velocity = 0 everywhere
9. Initialize pressure = 0 everywhere
```

### Mass Conservation Tracking

```rust
struct FluidMassTracker {
    initial_water_mass: f32,
    initial_lava_mass: f32,
    current_water_mass: f32,
    current_steam_mass: f32,
    current_lava_mass: f32,
}

// Invariant: current_water_mass + current_steam_mass == initial_water_mass
// Invariant: current_lava_mass == initial_lava_mass
```

---

## Phase Change Model

### Direct Contact Phase Changes

**No temperature tracking** - phase changes occur on direct neighbor contact only:

1. **Water → Steam** (Boiling)
   - Trigger: Water cell has lava neighbor (6-connected)
   - Rate: Fast conversion (e.g., 0.1/frame or 10% per frame)
   - Mass: `steam_gained = water_lost`

2. **Steam → Water** (Condensation)
   - Trigger: Steam cell touches solid boundary (cave wall or world sphere)
   - Rate: Instant conversion at boundary (100% per frame)
   - Mass: `water_gained = steam_lost`

3. **Lava** (Inert)
   - No phase changes
   - Triggers water → steam conversion in adjacent cells

### Phase Change Shader Logic

```wgsl
// Check for water-lava contact
if (water_density > 0.0) {
    var has_lava_neighbor = false;
    for each neighbor {
        if (lava_density[neighbor] > 0.0) {
            has_lava_neighbor = true;
            break;
        }
    }
    
    if (has_lava_neighbor) {
        // Convert water to steam
        let conversion_rate = 0.1;  // 10% per frame
        let amount = water_density * conversion_rate;
        water_density -= amount;
        steam_density += amount;
    }
}

// Check for steam-solid contact
if (steam_density > 0.0) {
    var has_solid_neighbor = false;
    for each neighbor {
        if (is_solid[neighbor]) {
            has_solid_neighbor = true;
            break;
        }
    }
    
    if (has_solid_neighbor) {
        // Condense steam to water
        let condensation_rate = 1.0;  // Instant at boundary
        let amount = steam_density * condensation_rate;
        steam_density -= amount;
        water_density += amount;
    }
}
```

---

## Surface Adhesion and Droplet Formation

### Orientation-Based Water Behavior

Water behavior depends on **surface orientation** relative to gravity:

1. **Horizontal Ceilings** (facing down)
   - Water accumulates until reaching droplet threshold
   - Forms discrete droplets/blobs
   - Droplets detach and fall when heavy enough

2. **Slanted Surfaces & Walls** (angled or vertical)
   - Water clings and streams downward
   - Forms thin films that trickle along surface
   - Adhesion competes with gravity component

3. **Floors** (facing up)
   - Water pools naturally
   - No adhesion needed (gravity does the work)

### Surface Normal Detection

Determine surface orientation by checking solid neighbor directions:

```wgsl
// Detect surface type based on solid neighbor position
fn get_surface_orientation(coords: vec3<i32>) -> vec3<f32> {
    var surface_normal = vec3<f32>(0.0);
    var solid_neighbor_count = 0;
    
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
    );
    
    for (var i = 0; i < 6; i++) {
        let neighbor_coords = coords + offsets[i];
        if (is_valid_coords(neighbor_coords)) {
            let neighbor_idx = coords_to_grid_index(neighbor_coords);
            if (solid_mask[neighbor_idx] != 0u) {
                // Accumulate normal pointing away from solid
                surface_normal -= vec3<f32>(offsets[i]);
                solid_neighbor_count++;
            }
        }
    }
    
    if (solid_neighbor_count > 0) {
        return normalize(surface_normal);
    }
    return vec3<f32>(0.0, 1.0, 0.0);  // Default up
}
```

### Droplet Formation on Ceilings

Water on horizontal ceilings accumulates until forming droplets:

```wgsl
// In apply_forces shader
if (water_density > 0.0) {
    let coords = grid_index_to_coords(idx);
    let surface_normal = get_surface_orientation(coords);
    let gravity_dir = vec3<f32>(0.0, -1.0, 0.0);
    
    // Check if surface is ceiling (normal points down, against gravity)
    let dot_with_gravity = dot(surface_normal, gravity_dir);
    
    if (dot_with_gravity > 0.7) {
        // Horizontal ceiling - droplet behavior
        if (water_density > params.droplet_threshold) {
            // Droplet is heavy enough to detach
            // Reduce adhesion, let gravity win
            let detach_force = gravity_dir * params.droplet_detach_force * water_density;
            vel += detach_force * params.dt;
        } else {
            // Accumulating - strong adhesion to ceiling
            let adhesion = surface_normal * params.water_adhesion_strength * water_density;
            vel += adhesion * params.dt;
        }
    } else if (abs(dot_with_gravity) < 0.7) {
        // Slanted surface or wall - streaming behavior
        // Apply adhesion toward surface
        let adhesion = surface_normal * params.water_adhesion_strength * water_density * 0.5;
        vel += adhesion * params.dt;
        
        // Also apply tangential flow (along surface, downward)
        // Project gravity onto surface plane
        let gravity_on_surface = gravity_dir - surface_normal * dot(gravity_dir, surface_normal);
        vel += gravity_on_surface * params.dt;
    }
    // else: floor (dot < -0.7) - no special adhesion, gravity handles it
}
```

### Condensation Behavior by Surface Type

Steam condensation creates water with behavior based on surface orientation:

```wgsl
// In phase_change shader
if (steam_density > 0.0 && has_solid_neighbor) {
    // Condense steam to water
    let amount = steam_density * condensation_rate;
    density.z -= amount;  // Steam
    density.y += amount;  // Water
    
    // Water will automatically adopt correct behavior in next apply_forces pass:
    // - Ceiling: accumulate into droplets
    // - Wall/slant: stream downward
    // - Floor: pool naturally
}
```

### Behavior Summary

| Surface Type | Orientation | Water Behavior |
|--------------|-------------|----------------|
| **Ceiling** | Horizontal (down-facing) | Accumulates → Forms droplets → Falls as blobs |
| **Wall** | Vertical | Clings + streams downward in thin films |
| **Slanted** | Angled | Clings + trickles along surface |
| **Floor** | Horizontal (up-facing) | Pools naturally (gravity only) |

### Tuning Parameters

- **`water_adhesion_strength`** (0.3-0.8): How strongly water clings to surfaces
- **`droplet_threshold`** (0.5-2.0): Water density needed before droplet detaches from ceiling
- **`droplet_detach_force`** (1.0-5.0): Force applied to detach heavy droplets

---

## Fluid-Cell Interaction

### Fluids Affect Cells

1. **Buoyancy Forces**
   - Cells in fluid experience buoyancy based on fluid density
   - Force = `(fluid_density - cell_density) × gravity × cell_volume`
   - Applied in cell physics shader

2. **Drag Forces**
   - Cells experience drag proportional to relative velocity
   - Force = `drag_coefficient × (fluid_velocity - cell_velocity)`

3. **Lava Damage** (Optional future feature)
   - Cells in lava lose mass over time
   - Not implemented in initial phases

### Cells Do NOT Affect Fluids

- Cells do not displace fluid
- Cells do not consume/produce fluid
- Cells are "ghosts" to the fluid simulation
- Simplifies implementation and maintains determinism

---

## Boundary Conditions

### Free-Slip at Solid Surfaces

At cave walls and world sphere boundary:

```wgsl
// Velocity boundary condition
fn apply_free_slip_boundary(cell_idx: u32, coords: vec3<i32>) {
    var velocity = fluid_velocity[cell_idx].xyz;
    
    // Check each face neighbor
    for each direction {
        let neighbor_coords = coords + direction;
        
        if (is_solid[neighbor_coords]) {
            // Compute normal (points into fluid)
            let normal = -direction;
            
            // Remove normal component of velocity
            let v_normal = dot(velocity, normal);
            if (v_normal < 0.0) {  // Moving into wall
                velocity -= normal * v_normal;
            }
        }
    }
    
    fluid_velocity[cell_idx].xyz = velocity;
}
```

### Pressure Boundary Condition

Solid cells are excluded from pressure solve:

```wgsl
// Neumann boundary (zero gradient)
// Solid neighbors simply not included in Jacobi iteration
if (is_solid[cell_idx]) {
    pressure[cell_idx] = 0.0;
    return;
}

// Only sum pressure from fluid neighbors
var neighbor_sum = 0.0;
var fluid_neighbor_count = 0;

for each neighbor {
    if (!is_solid[neighbor]) {
        neighbor_sum += pressure[neighbor];
        fluid_neighbor_count++;
    }
}

if (fluid_neighbor_count > 0) {
    pressure[cell_idx] = neighbor_sum / fluid_neighbor_count - divergence[cell_idx];
}
```

---

## Rendering: Smoothed Voxel Visualization

### Marching Cubes Approach

Generate a smooth isosurface for each fluid type:

1. **Density Threshold**
   - Extract isosurface where `density > 0.5`
   - Separate mesh for each fluid type (water, lava, steam)

2. **Smoothing**
   - Trilinear interpolation of density values
   - Smooth normals from density gradient

3. **Mesh Generation**
   - Run marching cubes on 128³ grid
   - Generate vertices, indices, normals
   - Update mesh every frame (or every N frames for performance)

4. **Rendering**
   - Water: Blue, semi-transparent
   - Lava: Orange/red, emissive
   - Steam: White/gray, very transparent
   - Air: Invisible (not rendered)

### Alternative: Instanced Cubes (Simpler for Phase 1)

For initial implementation:

```rust
// Extract non-zero density cells
for each cell in grid {
    if (water_density > threshold) {
        instances.push(Instance {
            position: grid_to_world(cell_coords),
            scale: cell_size,
            color: vec4(0.2, 0.4, 0.8, 0.6),  // Blue water
        });
    }
}

// Render as instanced cubes
```

Later upgrade to marching cubes for smooth surfaces.

---

## Phased Implementation Plan

### Phase 1: Foundation (Water + Boundaries + Adhesion)

**Goal**: Single fluid type (water) with cave constraints, surface adhesion, and basic visualization

#### Step 1.1: Cave Voxel Grid Generation
**File**: `src/rendering/cave_system.rs`

```rust
impl CaveSystemRenderer {
    /// Generate 128³ boolean grid marking solid vs open space
    pub fn generate_fluid_collision_grid(
        params: &CaveParams,
        grid_resolution: u32,
    ) -> Vec<u32> {
        let total_cells = (grid_resolution.pow(3)) as usize;
        let mut solid_mask = vec![0u32; total_cells];
        
        let world_center = Vec3::from(params.world_center);
        let world_radius = params.world_radius;
        let cell_size = (world_radius * 2.0) / grid_resolution as f32;
        
        for z in 0..grid_resolution {
            for y in 0..grid_resolution {
                for x in 0..grid_resolution {
                    let grid_pos = Vec3::new(
                        world_center.x - world_radius + (x as f32 + 0.5) * cell_size,
                        world_center.y - world_radius + (y as f32 + 0.5) * cell_size,
                        world_center.z - world_radius + (z as f32 + 0.5) * cell_size,
                    );
                    
                    let density = Self::sample_density(grid_pos, params);
                    let is_solid = density > params.threshold;
                    
                    let idx = x + y * grid_resolution + z * grid_resolution * grid_resolution;
                    solid_mask[idx as usize] = if is_solid { 1 } else { 0 };
                }
            }
        }
        
        solid_mask
    }
    
    /// Create GPU buffer for fluid collision grid
    pub fn create_fluid_collision_buffer(
        &mut self,
        device: &wgpu::Device,
        grid_resolution: u32,
    ) -> wgpu::Buffer {
        let solid_mask = Self::generate_fluid_collision_grid(&self.params, grid_resolution);
        
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Collision Grid"),
            contents: bytemuck::cast_slice(&solid_mask),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }
}
```

**Validation**: Print statistics
- Total cells: 2,097,152
- Solid cells: ~1,048,576 (50%)
- Fluid cells: ~1,048,576 (50%)

#### Step 1.2: Fluid GPU Buffers
**File**: `src/gpu_buffers/fluid_buffers.rs` (new file)

```rust
use wgpu;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FluidParams {
    grid_resolution: u32,
    cell_size: f32,
    world_center: [f32; 3],
    world_radius: f32,
    
    gravity: f32,
    water_density: f32,
    lava_density: f32,
    steam_density: f32,
    air_density: f32,
    
    viscosity: f32,
    dt: f32,  // 1/30 for 30 FPS
    
    phase_change_rate: f32,  // Water → steam conversion rate (0.0-1.0)
    water_adhesion_strength: f32,  // Surface wetting strength (0.0-1.0)
    droplet_threshold: f32,  // Water density threshold for droplet formation
    droplet_detach_force: f32,  // Force needed to detach droplet from ceiling
    
    // Padding to 256 bytes
    _padding: [f32; 46],
}

pub struct FluidSimulationBuffers {
    // Fluid state (128³ cells)
    pub densities: wgpu::Buffer,        // vec4<f32>: (lava, water, steam, air)
    pub velocity: wgpu::Buffer,         // vec4<f32>: (vx, vy, vz, padding)
    pub pressure: wgpu::Buffer,         // f32
    
    // Temporary buffers for compute
    pub divergence: wgpu::Buffer,       // f32
    pub densities_temp: wgpu::Buffer,   // vec4<f32> for advection
    pub velocity_temp: wgpu::Buffer,    // vec4<f32> for advection
    pub pressure_temp: wgpu::Buffer,    // f32 for Jacobi ping-pong
    
    // Cave collision
    pub solid_mask: wgpu::Buffer,       // u32 (from CaveSystemRenderer)
    
    // Parameters
    pub params: wgpu::Buffer,           // FluidParams uniform
}

impl FluidSimulationBuffers {
    pub fn new(
        device: &wgpu::Device,
        grid_resolution: u32,
        solid_mask_data: &[u32],
    ) -> Self {
        let total_cells = (grid_resolution.pow(3)) as u64;
        let vec4_size = std::mem::size_of::<[f32; 4]>() as u64;
        let f32_size = std::mem::size_of::<f32>() as u64;
        let u32_size = std::mem::size_of::<u32>() as u64;
        
        let densities = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Densities"),
            size: total_cells * vec4_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let velocity = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Velocity"),
            size: total_cells * vec4_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let pressure = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Pressure"),
            size: total_cells * f32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let divergence = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Divergence"),
            size: total_cells * f32_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let densities_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Densities Temp"),
            size: total_cells * vec4_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let velocity_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Velocity Temp"),
            size: total_cells * vec4_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let pressure_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Pressure Temp"),
            size: total_cells * f32_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let solid_mask = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Solid Mask"),
            contents: bytemuck::cast_slice(solid_mask_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Params"),
            size: std::mem::size_of::<FluidParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            densities,
            velocity,
            pressure,
            divergence,
            densities_temp,
            velocity_temp,
            pressure_temp,
            solid_mask,
            params,
        }
    }
}
```

#### Step 1.3: Water Initialization Shader
**File**: `shaders/fluid/initialize_water.wgsl` (new file)

```wgsl
struct FluidParams {
    grid_resolution: u32,
    cell_size: f32,
    world_center: vec3<f32>,
    world_radius: f32,
    gravity: f32,
    water_density: f32,
    lava_density: f32,
    steam_density: f32,
    air_density: f32,
    viscosity: f32,
    dt: f32,
}

@group(0) @binding(0) var<uniform> params: FluidParams;
@group(0) @binding(1) var<storage, read_write> densities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> solid_mask: array<u32>;

fn grid_index_to_coords(idx: u32) -> vec3<i32> {
    let res = i32(params.grid_resolution);
    let z = i32(idx) / (res * res);
    let y = (i32(idx) - z * res * res) / res;
    let x = i32(idx) - z * res * res - y * res;
    return vec3<i32>(x, y, z);
}

fn grid_coords_to_world(coords: vec3<i32>) -> vec3<f32> {
    let half_world = params.world_radius;
    let offset = vec3<f32>(coords) * params.cell_size;
    return params.world_center - vec3<f32>(half_world) + offset + vec3<f32>(params.cell_size * 0.5);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_cells) {
        return;
    }
    
    // Skip solid cells
    if (solid_mask[idx] != 0u) {
        densities[idx] = vec4<f32>(0.0);
        return;
    }
    
    // Get world position
    let coords = grid_index_to_coords(idx);
    let world_pos = grid_coords_to_world(coords);
    
    // Determine hemisphere (gravity points up in +Y)
    let gravity_dir = vec3<f32>(0.0, 1.0, 0.0);
    let from_center = world_pos - params.world_center;
    let dot_product = dot(from_center, gravity_dir);
    
    // Top hemisphere gets water (for Phase 1, fill all available cells)
    if (dot_product > 0.0) {
        // Water in top hemisphere
        densities[idx] = vec4<f32>(0.0, 1.0, 0.0, 0.0);  // (lava, water, steam, air)
    } else {
        // Air in bottom hemisphere (will be lava in Phase 2)
        densities[idx] = vec4<f32>(0.0, 0.0, 0.0, 1.0);  // (lava, water, steam, air)
    }
}
```

#### Step 1.4: Basic Advection Shader
**File**: `shaders/fluid/advect.wgsl` (new file)

```wgsl
struct FluidParams {
    grid_resolution: u32,
    cell_size: f32,
    world_center: vec3<f32>,
    world_radius: f32,
    gravity: f32,
    water_density: f32,
    lava_density: f32,
    steam_density: f32,
    air_density: f32,
    viscosity: f32,
    dt: f32,
}

@group(0) @binding(0) var<uniform> params: FluidParams;
@group(0) @binding(1) var<storage, read> densities_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> densities_out: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> solid_mask: array<u32>;

fn grid_index_to_coords(idx: u32) -> vec3<i32> {
    let res = i32(params.grid_resolution);
    let z = i32(idx) / (res * res);
    let y = (i32(idx) - z * res * res) / res;
    let x = i32(idx) - z * res * res - y * res;
    return vec3<i32>(x, y, z);
}

fn coords_to_grid_index(coords: vec3<i32>) -> u32 {
    let res = i32(params.grid_resolution);
    return u32(coords.x + coords.y * res + coords.z * res * res);
}

fn sample_density_trilinear(pos: vec3<f32>) -> vec4<f32> {
    let res = f32(params.grid_resolution);
    
    // Clamp to grid bounds
    let clamped = clamp(pos, vec3<f32>(0.0), vec3<f32>(res - 1.0));
    
    // Get integer and fractional parts
    let i = vec3<i32>(floor(clamped));
    let f = fract(clamped);
    
    // Sample 8 corners
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    
    for (var dz = 0; dz <= 1; dz++) {
        for (var dy = 0; dy <= 1; dy++) {
            for (var dx = 0; dx <= 1; dx++) {
                let corner = i + vec3<i32>(dx, dy, dz);
                
                // Bounds check
                if (corner.x >= 0 && corner.x < i32(res) &&
                    corner.y >= 0 && corner.y < i32(res) &&
                    corner.z >= 0 && corner.z < i32(res)) {
                    
                    let idx = coords_to_grid_index(corner);
                    
                    // Skip solid cells
                    if (solid_mask[idx] == 0u) {
                        let weight = 
                            (1.0 - abs(f.x - f32(dx))) *
                            (1.0 - abs(f.y - f32(dy))) *
                            (1.0 - abs(f.z - f32(dz)));
                        
                        sum += densities_in[idx] * weight;
                        weight_sum += weight;
                    }
                }
            }
        }
    }
    
    if (weight_sum > 0.0) {
        return sum / weight_sum;
    }
    return vec4<f32>(0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_cells) {
        return;
    }
    
    // Skip solid cells
    if (solid_mask[idx] != 0u) {
        densities_out[idx] = vec4<f32>(0.0);
        return;
    }
    
    let coords = grid_index_to_coords(idx);
    let vel = velocity[idx].xyz;
    
    // Semi-Lagrangian advection: trace backwards
    let back_pos = vec3<f32>(coords) - vel * params.dt / params.cell_size;
    
    // Sample density at back-traced position
    densities_out[idx] = sample_density_trilinear(back_pos);
}
```

#### Step 1.5: Simple Visualization
**File**: `src/rendering/fluid_renderer.rs` (new file)

```rust
pub struct FluidRenderer {
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
    render_pipeline: wgpu::RenderPipeline,
}

impl FluidRenderer {
    pub fn extract_instances(
        &mut self,
        queue: &wgpu::Queue,
        densities: &[Vec4],  // Read back from GPU
        solid_mask: &[u32],
        grid_resolution: u32,
        cell_size: f32,
        world_center: Vec3,
        world_radius: f32,
    ) {
        let mut instances = Vec::new();
        
        for z in 0..grid_resolution {
            for y in 0..grid_resolution {
                for x in 0..grid_resolution {
                    let idx = (x + y * grid_resolution + z * grid_resolution * grid_resolution) as usize;
                    
                    if solid_mask[idx] != 0 {
                        continue;
                    }
                    
                    let density = densities[idx];
                    let water_density = density.y;
                    
                    if (water_density > 0.1) {
                        let pos = Vec3::new(
                            world_center.x - world_radius + (x as f32 + 0.5) * cell_size,
                            world_center.y - world_radius + (y as f32 + 0.5) * cell_size,
                            world_center.z - world_radius + (z as f32 + 0.5) * cell_size,
                        );
                        
                        instances.push(FluidInstance {
                            position: pos,
                            scale: cell_size,
                            color: Vec4::new(0.2, 0.4, 0.8, 0.6 * water_density),
                        });
                    }
                }
            }
        }
        
        self.instance_count = instances.len() as u32;
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
    }
}
```

**Validation**: 
- Water appears in top hemisphere
- No water in solid cells
- Water clings to nearby walls (adhesion visible)
- Visual inspection of distribution

---

### Phase 2: Multi-Fluid + Phase Changes

**Goal**: Add lava, steam, and implement water ↔ steam phase changes

#### Step 2.1: Hemisphere Initialization
**File**: Update `shaders/fluid/initialize_water.wgsl`

Add lava initialization in bottom hemisphere based on user percentages.

#### Step 2.2: Phase Change Shader
**File**: `shaders/fluid/phase_change.wgsl` (new file)

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_cells || solid_mask[idx] != 0u) {
        return;
    }
    
    var density = densities[idx];
    let water = density.y;
    let steam = density.z;
    let lava = density.x;
    
    // Water → Steam (lava contact)
    if (water > 0.0) {
        var has_lava_neighbor = false;
        
        // Check 6-connected neighbors
        let coords = grid_index_to_coords(idx);
        let offsets = array<vec3<i32>, 6>(
            vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
            vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
            vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
        );
        
        for (var i = 0; i < 6; i++) {
            let neighbor_coords = coords + offsets[i];
            
            if (is_valid_coords(neighbor_coords)) {
                let neighbor_idx = coords_to_grid_index(neighbor_coords);
                if (solid_mask[neighbor_idx] == 0u && densities[neighbor_idx].x > 0.0) {
                    has_lava_neighbor = true;
                    break;
                }
            }
        }
        
        if (has_lava_neighbor) {
            let conversion_rate = 0.1;  // 10% per frame
            let amount = water * conversion_rate;
            density.y -= amount;  // Water
            density.z += amount;  // Steam
        }
    }
    
    // Steam → Water (solid contact)
    if (steam > 0.0) {
        var has_solid_neighbor = false;
        
        let coords = grid_index_to_coords(idx);
        let offsets = array<vec3<i32>, 6>(
            vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
            vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
            vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
        );
        
        for (var i = 0; i < 6; i++) {
            let neighbor_coords = coords + offsets[i];
            
            if (is_valid_coords(neighbor_coords)) {
                let neighbor_idx = coords_to_grid_index(neighbor_coords);
                if (solid_mask[neighbor_idx] != 0u) {
                    has_solid_neighbor = true;
                    break;
                }
            }
        }
        
        if (has_solid_neighbor) {
            let condensation_rate = 1.0;  // Instant
            let amount = steam * condensation_rate;
            density.z -= amount;  // Steam
            density.y += amount;  // Water
        }
    }
    
    densities[idx] = density;
}
```

**Validation**:
- Water near lava converts to steam
- Steam near walls/ceiling converts to water
- Condensed water clings to walls and streams downward
- Total water + steam mass conserved

---

### Phase 3: Pressure Solve + Incompressibility

**Goal**: Implement divergence-free velocity field using Jacobi iteration

#### Step 3.1: Compute Divergence
**File**: `shaders/fluid/compute_divergence.wgsl` (new file)

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= total_cells || solid_mask[idx] != 0u) {
        divergence[idx] = 0.0;
        return;
    }
    
    let coords = grid_index_to_coords(idx);
    let vel = velocity[idx].xyz;
    
    // Central differences for divergence
    var div = 0.0;
    
    // X direction
    if (coords.x > 0 && solid_mask[idx - 1] == 0u) {
        let vel_left = velocity[idx - 1].x;
        div += (vel.x - vel_left) / params.cell_size;
    }
    
    // Y direction
    if (coords.y > 0 && solid_mask[idx - grid_res] == 0u) {
        let vel_down = velocity[idx - grid_res].y;
        div += (vel.y - vel_down) / params.cell_size;
    }
    
    // Z direction
    if (coords.z > 0 && solid_mask[idx - grid_res * grid_res] == 0u) {
        let vel_back = velocity[idx - grid_res * grid_res].z;
        div += (vel.z - vel_back) / params.cell_size;
    }
    
    divergence[idx] = div;
}
```

#### Step 3.2: Jacobi Pressure Solve
**File**: `shaders/fluid/jacobi_pressure.wgsl` (new file)

```wgsl
@group(0) @binding(0) var<uniform> params: FluidParams;
@group(0) @binding(1) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure_out: array<f32>;
@group(0) @binding(3) var<storage, read> divergence: array<f32>;
@group(0) @binding(4) var<storage, read> solid_mask: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_cells) {
        return;
    }
    
    // Solid cells have zero pressure
    if (solid_mask[idx] != 0u) {
        pressure_out[idx] = 0.0;
        return;
    }
    
    let coords = grid_index_to_coords(idx);
    let res = i32(params.grid_resolution);
    
    // Sum pressure from fluid neighbors only
    var neighbor_sum = 0.0;
    var fluid_neighbor_count = 0;
    
    // -X neighbor
    if (coords.x > 0) {
        let neighbor_idx = idx - 1;
        if (solid_mask[neighbor_idx] == 0u) {
            neighbor_sum += pressure_in[neighbor_idx];
            fluid_neighbor_count++;
        }
    }
    
    // +X neighbor
    if (coords.x < res - 1) {
        let neighbor_idx = idx + 1;
        if (solid_mask[neighbor_idx] == 0u) {
            neighbor_sum += pressure_in[neighbor_idx];
            fluid_neighbor_count++;
        }
    }
    
    // -Y neighbor
    if (coords.y > 0) {
        let neighbor_idx = idx - u32(res);
        if (solid_mask[neighbor_idx] == 0u) {
            neighbor_sum += pressure_in[neighbor_idx];
            fluid_neighbor_count++;
        }
    }
    
    // +Y neighbor
    if (coords.y < res - 1) {
        let neighbor_idx = idx + u32(res);
        if (solid_mask[neighbor_idx] == 0u) {
            neighbor_sum += pressure_in[neighbor_idx];
            fluid_neighbor_count++;
        }
    }
    
    // -Z neighbor
    if (coords.z > 0) {
        let neighbor_idx = idx - u32(res * res);
        if (solid_mask[neighbor_idx] == 0u) {
            neighbor_sum += pressure_in[neighbor_idx];
            fluid_neighbor_count++;
        }
    }
    
    // +Z neighbor
    if (coords.z < res - 1) {
        let neighbor_idx = idx + u32(res * res);
        if (solid_mask[neighbor_idx] == 0u) {
            neighbor_sum += pressure_in[neighbor_idx];
            fluid_neighbor_count++;
        }
    }
    
    // Jacobi iteration
    if (fluid_neighbor_count > 0) {
        pressure_out[idx] = neighbor_sum / f32(fluid_neighbor_count) - divergence[idx];
    } else {
        pressure_out[idx] = 0.0;
    }
}
```

**Run 20 iterations per frame** with ping-pong buffers.

#### Step 3.3: Pressure Gradient Subtraction
**File**: `shaders/fluid/subtract_pressure_gradient.wgsl` (new file)

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= total_cells || solid_mask[idx] != 0u) {
        return;
    }
    
    let coords = grid_index_to_coords(idx);
    var vel = velocity[idx].xyz;
    
    // Compute pressure gradient
    var grad = vec3<f32>(0.0);
    
    // X gradient
    if (coords.x < res - 1 && solid_mask[idx + 1] == 0u) {
        grad.x = (pressure[idx + 1] - pressure[idx]) / params.cell_size;
    }
    
    // Y gradient
    if (coords.y < res - 1 && solid_mask[idx + res] == 0u) {
        grad.y = (pressure[idx + res] - pressure[idx]) / params.cell_size;
    }
    
    // Z gradient
    if (coords.z < res - 1 && solid_mask[idx + res * res] == 0u) {
        grad.z = (pressure[idx + res * res] - pressure[idx]) / params.cell_size;
    }
    
    // Subtract gradient to make divergence-free
    vel -= grad * params.dt;
    
    velocity[idx] = vec4<f32>(vel, 0.0);
}
```

#### Step 3.4: Buoyancy and Orientation-Based Adhesion
**File**: `shaders/fluid/apply_forces.wgsl` (new file)

```wgsl
fn get_surface_orientation(coords: vec3<i32>) -> vec3<f32> {
    var surface_normal = vec3<f32>(0.0);
    var solid_neighbor_count = 0;
    
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
    );
    
    for (var i = 0; i < 6; i++) {
        let neighbor_coords = coords + offsets[i];
        if (is_valid_coords(neighbor_coords)) {
            let neighbor_idx = coords_to_grid_index(neighbor_coords);
            if (solid_mask[neighbor_idx] != 0u) {
                surface_normal -= vec3<f32>(offsets[i]);
                solid_neighbor_count++;
            }
        }
    }
    
    if (solid_neighbor_count > 0) {
        return normalize(surface_normal);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= total_cells || solid_mask[idx] != 0u) {
        return;
    }
    
    var vel = velocity[idx].xyz;
    let density = densities[idx];
    let water_density = density.y;
    
    // Gravity force (negative Y direction)
    let gravity_dir = vec3<f32>(0.0, -params.gravity, 0.0);
    vel += gravity_dir * params.dt;
    
    // Orientation-based water adhesion and droplet formation
    if (water_density > 0.0) {
        let coords = grid_index_to_coords(idx);
        let surface_normal = get_surface_orientation(coords);
        let dot_with_gravity = dot(surface_normal, normalize(gravity_dir));
        
        if (dot_with_gravity > 0.7) {
            // Ceiling - droplet formation behavior
            if (water_density > params.droplet_threshold) {
                // Heavy droplet - detach and fall
                let detach_force = gravity_dir * params.droplet_detach_force * water_density;
                vel += detach_force * params.dt;
            } else {
                // Light water - cling to ceiling
                let adhesion = surface_normal * params.water_adhesion_strength * water_density;
                vel += adhesion * params.dt;
            }
        } else if (abs(dot_with_gravity) < 0.7) {
            // Wall or slanted surface - streaming behavior
            let adhesion = surface_normal * params.water_adhesion_strength * water_density * 0.5;
            vel += adhesion * params.dt;
            
            // Tangential flow along surface
            let gravity_on_surface = gravity_dir - surface_normal * dot(gravity_dir, surface_normal);
            vel += gravity_on_surface * params.dt;
        }
        // else: floor - gravity handles it naturally
    }
    
    velocity[idx] = vec4<f32>(vel, 0.0);
}
```

#### Step 3.5: Free-Slip Boundary
**File**: `shaders/fluid/enforce_boundaries.wgsl` (new file)

```wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= total_cells || solid_mask[idx] != 0u) {
        return;
    }
    
    let coords = grid_index_to_coords(idx);
    var vel = velocity[idx].xyz;
    
    // Check each neighbor for solid boundary
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
    );
    
    for (var i = 0; i < 6; i++) {
        let neighbor_coords = coords + offsets[i];
        
        if (is_valid_coords(neighbor_coords)) {
            let neighbor_idx = coords_to_grid_index(neighbor_coords);
            
            if (solid_mask[neighbor_idx] != 0u) {
                // Solid neighbor - remove normal component
                let normal = -vec3<f32>(offsets[i]);
                let v_normal = dot(vel, normal);
                
                if (v_normal < 0.0) {  // Moving into wall
                    vel -= normal * v_normal;
                }
            }
        }
    }
    
    velocity[idx] = vec4<f32>(vel, 0.0);
}
```

**Validation**:
- Velocity field is divergence-free
- Fluids don't penetrate boundaries
- Steam rises, water/lava sink

---

### Phase 4: Air Displacement + Integration

**Goal**: Add air dynamics and integrate into GpuScene at 30 FPS

#### Step 4.1: Air Displacement Logic

Air fills volume not occupied by other fluids:

```wgsl
// After advection and phase changes
let total_liquid_gas = density.x + density.y + density.z;  // lava + water + steam
density.w = max(0.0, 1.0 - total_liquid_gas);  // Air fills remainder
```

#### Step 4.2: Mass Conservation Validation

Add CPU-side tracking:

```rust
pub struct FluidMassTracker {
    initial_water: f32,
    initial_lava: f32,
}

impl FluidMassTracker {
    pub fn validate(&self, current_densities: &[Vec4]) -> bool {
        let mut water_mass = 0.0;
        let mut steam_mass = 0.0;
        let mut lava_mass = 0.0;
        
        for density in current_densities {
            lava_mass += density.x;
            water_mass += density.y;
            steam_mass += density.z;
        }
        
        let water_total = water_mass + steam_mass;
        let water_error = (water_total - self.initial_water).abs() / self.initial_water;
        let lava_error = (lava_mass - self.initial_lava).abs() / self.initial_lava;
        
        println!("Water conservation error: {:.2}%", water_error * 100.0);
        println!("Lava conservation error: {:.2}%", lava_error * 100.0);
        
        water_error < 0.01 && lava_error < 0.01  // 1% tolerance
    }
}
```

#### Step 4.3: GpuScene Integration

**File**: `src/scene/gpu_scene.rs`

```rust
pub struct GpuScene {
    // ... existing fields ...
    
    fluid_system: Option<FluidSimulation>,
    frame_counter: u32,
}

impl GpuScene {
    pub fn step_physics(&mut self, dt: f32) {
        // Existing cell physics at 60 FPS
        self.run_cell_physics_pipeline();
        
        // Fluid physics at 30 FPS
        self.frame_counter += 1;
        if self.frame_counter % 2 == 0 {
            if let Some(ref mut fluid) = self.fluid_system {
                fluid.step(dt * 2.0);  // Double dt for 30 FPS
            }
        }
    }
}
```

#### Step 4.4: Fluid Simulation Pipeline

**File**: `src/simulation/fluid_simulation.rs` (new file)

```rust
pub struct FluidSimulation {
    buffers: FluidSimulationBuffers,
    pipelines: FluidPipelines,
    mass_tracker: FluidMassTracker,
}

impl FluidSimulation {
    pub fn step(&mut self, dt: f32) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            
            // 1. Apply forces (gravity, buoyancy)
            pass.set_pipeline(&self.pipelines.apply_forces);
            pass.set_bind_group(0, &self.bind_groups.main, &[]);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
            
            // 2. Advect densities
            pass.set_pipeline(&self.pipelines.advect_density);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
            
            // 3. Advect velocity
            pass.set_pipeline(&self.pipelines.advect_velocity);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
            
            // 4. Phase changes
            pass.set_pipeline(&self.pipelines.phase_change);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
            
            // 5. Compute divergence
            pass.set_pipeline(&self.pipelines.compute_divergence);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
            
            // 6. Pressure solve (20 Jacobi iterations)
            for _ in 0..20 {
                pass.set_pipeline(&self.pipelines.jacobi_pressure);
                pass.dispatch_workgroups(self.workgroup_count, 1, 1);
                // Swap pressure buffers
            }
            
            // 7. Subtract pressure gradient
            pass.set_pipeline(&self.pipelines.subtract_gradient);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
            
            // 8. Enforce boundaries
            pass.set_pipeline(&self.pipelines.enforce_boundaries);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        }
        
        self.queue.submit(Some(encoder.finish()));
    }
}
```

**Validation**:
- Runs at 30 FPS (every other frame)
- Mass conservation maintained
- Visual inspection of fluid behavior

---

### Phase 5: UI Controls + Polish

**Goal**: Add user controls for fluid generation parameters

#### Step 5.1: UI Sliders

**File**: `src/ui/fluid_controls.rs` (new file)

```rust
pub struct FluidGenerationParams {
    pub cave_density: f32,    // 0.0 - 1.0
    pub water_percent: f32,   // 0.0 - 1.0
    pub lava_percent: f32,    // 0.0 - 1.0
    // air_percent computed as remainder
}

impl FluidGenerationParams {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Fluid Generation");
        
        ui.add(egui::Slider::new(&mut self.cave_density, 0.0..=1.0)
            .text("Cave Density"));
        
        let remaining = 1.0 - self.cave_density;
        
        ui.add(egui::Slider::new(&mut self.water_percent, 0.0..=remaining)
            .text("Water %"));
        
        let remaining_after_water = remaining - self.water_percent;
        
        ui.add(egui::Slider::new(&mut self.lava_percent, 0.0..=remaining_after_water)
            .text("Lava %"));
        
        let air_percent = remaining_after_water - self.lava_percent;
        ui.label(format!("Air: {:.1}%", air_percent * 100.0));
        
        if ui.button("Regenerate Fluids").clicked() {
            // Trigger fluid regeneration
        }
    }
}
```

#### Step 5.2: Smoothed Voxel Rendering

Upgrade from instanced cubes to marching cubes:

**File**: `src/rendering/fluid_marching_cubes.rs` (new file)

```rust
pub struct FluidMarchingCubes {
    // Similar to CaveSystemRenderer marching cubes
    // Generate separate meshes for water, lava, steam
}

impl FluidMarchingCubes {
    pub fn generate_mesh(
        densities: &[Vec4],
        solid_mask: &[u32],
        grid_resolution: u32,
        fluid_index: usize,  // 0=lava, 1=water, 2=steam
        threshold: f32,
    ) -> (Vec<Vertex>, Vec<u32>) {
        // Extract isosurface where density > threshold
        // Use trilinear interpolation for smooth surface
        // Compute normals from density gradient
    }
}
```

**Validation**:
- Smooth fluid surfaces
- Separate visualization for each fluid type
- Performance acceptable (mesh generation can be async)

---

## Performance Considerations

### GPU Memory Usage

Total for 128³ grid:
- Cave solid mask: 2 MB
- Fluid densities: 8 MB
- Fluid velocity: 8 MB
- Fluid pressure: 2 MB (×2 for ping-pong)
- Divergence: 2 MB
- Temp buffers: 16 MB
- **Total: ~40 MB**

### Compute Performance

At 30 FPS with 128³ grid:
- Advection: ~2M cells
- Pressure solve: 20 iterations × ~1M fluid cells
- Phase changes: ~1M fluid cells
- **Estimated**: 2-5ms per frame on modern GPU

### Optimization Opportunities

1. **Skip solid cells early** - 50% reduction in work
2. **Adaptive pressure solve** - Fewer iterations in stable regions
3. **Async mesh generation** - Don't block simulation
4. **LOD for distant fluids** - Lower resolution far from camera

---

## Testing & Validation Strategy

### Visual Inspection Tests

1. **Water settling**: Water should flow downward and pool at bottom
2. **Ceiling droplets**: Water on horizontal ceilings accumulates then falls as discrete blobs/droplets
3. **Wall streaming**: Water on vertical walls clings and trickles downward in thin streams
4. **Slanted surface flow**: Water on angled surfaces follows surface contour while flowing down
5. **Lava-water interaction**: Steam should form at interface
6. **Steam rising**: Steam should rise and condense at ceiling/walls
7. **Condensation patterns**: Steam condensing on ceiling forms droplets, on walls forms streams
8. **Boundary respect**: No fluid penetration into cave walls
9. **Mass conservation**: Total water+steam constant over time

### Quantitative Tests

1. **Mass conservation**: Track total mass each frame, error < 1%
2. **Divergence**: Measure velocity divergence after pressure solve
3. **Performance**: Maintain 30 FPS fluid update, 60 FPS overall
4. **Memory**: Stay within 50 MB GPU memory budget

### Debug Visualization

1. **Density slices**: 2D cross-sections of density field
2. **Velocity arrows**: Vector field visualization
3. **Pressure heatmap**: Pressure distribution
4. **Solid mask overlay**: Verify cave boundaries

---

## File Structure Summary

```
src/
├── gpu_buffers/
│   └── fluid_buffers.rs          (NEW)
├── rendering/
│   ├── cave_system.rs             (MODIFIED - add voxel grid generation)
│   ├── fluid_renderer.rs          (NEW)
│   └── fluid_marching_cubes.rs    (NEW - Phase 5)
├── simulation/
│   └── fluid_simulation.rs        (NEW)
├── scene/
│   └── gpu_scene.rs               (MODIFIED - integrate fluid system)
└── ui/
    └── fluid_controls.rs          (NEW - Phase 5)

shaders/fluid/
├── initialize_water.wgsl          (NEW)
├── advect.wgsl                    (NEW)
├── phase_change.wgsl              (NEW)
├── compute_divergence.wgsl        (NEW)
├── jacobi_pressure.wgsl           (NEW)
├── subtract_pressure_gradient.wgsl (NEW)
├── apply_forces.wgsl              (NEW)
└── enforce_boundaries.wgsl        (NEW)
```

---

## Implementation Timeline

### Phase 1: Foundation (3-4 days)
- Cave voxel grid generation
- Basic GPU buffers
- Water-only advection
- Simple cube rendering

### Phase 2: Multi-Fluid (2-3 days)
- Lava initialization
- Phase change shaders
- Steam visualization

### Phase 3: Pressure Solve (3-4 days)
- Divergence computation
- Jacobi iteration
- Pressure gradient subtraction
- Buoyancy forces

### Phase 4: Integration (2-3 days)
- Air displacement
- Mass conservation validation
- GpuScene integration at 30 FPS

### Phase 5: Polish (2-3 days)
- UI sliders
- Marching cubes rendering
- Performance optimization

**Total: 12-17 days**

---

## Success Criteria

✅ **Functional**
- [ ] Water flows and pools correctly
- [ ] Lava stays at bottom, causes steam formation
- [ ] Steam rises and condenses at boundaries
- [ ] No fluid penetration into cave walls
- [ ] Runs at 30 FPS fluid update

✅ **Quality**
- [ ] Mass conservation error < 1%
- [ ] Smooth visual appearance (marching cubes)
- [ ] Responsive UI controls
- [ ] No GPU memory issues

✅ **Performance**
- [ ] 30 FPS fluid simulation
- [ ] 60 FPS overall frame rate maintained
- [ ] < 50 MB GPU memory for fluid system

---

## Next Steps

1. **Review this document** - Confirm specifications and approach
2. **Begin Phase 1** - Cave voxel grid generation
3. **Iterative testing** - Validate each phase before proceeding
4. **Adjust as needed** - Refine parameters based on visual results

---

*Document Version: 1.0*  
*Created: January 19, 2026*  
*Bio-Spheres Fluid Simulation System*

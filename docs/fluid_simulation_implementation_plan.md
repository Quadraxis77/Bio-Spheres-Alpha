# GPU-Only 4-Fluid Simulation System - Implementation Plan

## Executive Summary

This document outlines a **pure GPU implementation** of a 4-fluid simulation system (lava, water, steam, air) for Bio-Spheres. The system follows the GPU scene architecture with **zero CPU readbacks**, **zero synchronization**, and **triple-buffered execution** for maximum performance.

**Core Principles:**
- **Absolutely zero CPU physics** - All computation on GPU
- **Zero CPU readbacks** - No GPU-to-CPU data transfer during simulation
- **Triple buffering** - Lock-free parallel execution
- **GPU-only rendering** - Extract instances directly on GPU
- **Deterministic execution** - Reproducible results

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
- **Mass conservation**: GPU-tracked atomic counters (no CPU validation)
- **Phase change model**: Direct contact only (no temperature tracking)
- **Buffer management**: Triple-buffered for zero-sync performance

### Memory Budget (Triple Buffered)

Per 128³ grid with triple buffering:
- Cave solid mask: 2 MB (u32 per cell, static)
- Fluid densities: 24 MB (vec4<f32> × 3 buffer sets)
- Fluid velocity: 24 MB (vec4<f32> × 3 buffer sets)
- Fluid pressure: 6 MB (f32 × 3 buffer sets for ping-pong)
- Divergence temp: 2 MB (f32 per cell)
- Instance extraction: 16 MB (GPU instance buffer)
- **Total**: ~74 MB GPU memory

---

## GPU-Only Initialization

### User-Defined Distribution (CPU Parameters Only)

UI sliders set GPU uniform parameters (no CPU computation):
- **Cave density**: 50-70% (typical, determines solid space)
- **Water percentage**: 10-30%
- **Lava percentage**: 5-15%
- **Air**: Computed on GPU as remainder

### GPU Initialization Pipeline

**All initialization happens in compute shaders:**

```wgsl
// Stage 1: Cave voxel grid generation (GPU compute)
@compute @workgroup_size(64)
fn generate_cave_grid() {
    // Sample cave SDF on GPU
    // Mark solid vs fluid cells
    // No CPU involvement
}

// Stage 2: Hemisphere classification (GPU compute)
@compute @workgroup_size(64)
fn classify_hemispheres() {
    // Classify each cell as top/bottom hemisphere
    // Store in temporary buffer
}

// Stage 3: Fluid distribution (GPU compute)
@compute @workgroup_size(64)
fn initialize_fluids() {
    // Water → Top hemisphere
    // Lava → Bottom hemisphere
    // Air → Remainder
    // All densities initialized on GPU
}

// Stage 4: Initialize velocity and pressure (GPU compute)
@compute @workgroup_size(64)
fn initialize_dynamics() {
    // velocity = vec3(0.0)
    // pressure = 0.0
}
```

### GPU Mass Conservation Tracking

**No CPU readbacks - use GPU atomic counters:**

```wgsl
// GPU-side mass tracking buffer
struct MassCounters {
    initial_water: atomic<u32>,    // Fixed-point representation
    initial_lava: atomic<u32>,
    current_water: atomic<u32>,
    current_steam: atomic<u32>,
    current_lava: atomic<u32>,
}

// Validation shader (runs on GPU, no CPU readback)
@compute @workgroup_size(64)
fn validate_mass_conservation() {
    // Check: current_water + current_steam == initial_water
    // Check: current_lava == initial_lava
    // Store error flags in GPU buffer for UI display
}
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

## GPU-Only Fluid-Cell Interaction

### Fluids Affect Cells (GPU Compute)

**Cell physics shader reads fluid data directly from GPU buffers:**

```wgsl
// In cell_physics_spatial.wgsl
@compute @workgroup_size(64)
fn cell_physics_with_fluids() {
    let cell_pos = positions_in[cell_idx].xyz;
    
    // Sample fluid grid at cell position (GPU-to-GPU)
    let fluid_idx = world_pos_to_fluid_grid_index(cell_pos);
    let fluid_density = fluid_densities[fluid_idx];
    let fluid_velocity = fluid_velocities[fluid_idx].xyz;
    
    // 1. Buoyancy force
    let buoyancy = (fluid_density.y - cell_density) * gravity * cell_volume;
    force += buoyancy;
    
    // 2. Drag force
    let relative_vel = fluid_velocity - cell_velocity;
    let drag = drag_coefficient * relative_vel;
    force += drag;
    
    // 3. Lava damage (optional)
    if (fluid_density.x > 0.5) {  // In lava
        mass_loss += lava_damage_rate * dt;
    }
}
```

### Cells Do NOT Affect Fluids

- Cells do not displace fluid
- Cells do not consume/produce fluid
- Cells are "ghosts" to the fluid simulation
- **Zero coupling from cells to fluids** - maintains determinism
- One-way interaction: fluids → cells only

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

## GPU-Only Rendering: Zero CPU Readback

### GPU Instance Extraction (No CPU Involvement)

**Extract rendering instances directly on GPU:**

```wgsl
// Compute shader extracts instances from fluid grid
@compute @workgroup_size(64)
fn extract_fluid_instances() {
    let idx = global_id.x;
    
    if (solid_mask[idx] != 0u) {
        return;
    }
    
    let density = fluid_densities[idx];
    let coords = grid_index_to_coords(idx);
    let world_pos = grid_coords_to_world(coords);
    
    // Water instances
    if (density.y > 0.1) {
        let instance_idx = atomicAdd(&water_instance_count, 1u);
        water_instances[instance_idx] = FluidInstance {
            position: world_pos,
            scale: params.cell_size,
            color: vec4<f32>(0.2, 0.4, 0.8, 0.6 * density.y),
        };
    }
    
    // Lava instances
    if (density.x > 0.1) {
        let instance_idx = atomicAdd(&lava_instance_count, 1u);
        lava_instances[instance_idx] = FluidInstance {
            position: world_pos,
            scale: params.cell_size,
            color: vec4<f32>(1.0, 0.3, 0.0, density.x),
        };
    }
    
    // Steam instances
    if (density.z > 0.1) {
        let instance_idx = atomicAdd(&steam_instance_count, 1u);
        steam_instances[instance_idx] = FluidInstance {
            position: world_pos,
            scale: params.cell_size,
            color: vec4<f32>(0.9, 0.9, 0.9, 0.3 * density.z),
        };
    }
}
```

### Triple-Buffered Instance Rendering

**Rendering uses instances from 2 frames ago (no sync stalls):**

```rust
// Render pipeline uses triple-buffered instance data
pub fn render_fluids(&self, encoder: &mut CommandEncoder, view: &TextureView) {
    let render_buffer_set = self.triple_buffer.get_render_index();
    
    // Render water
    render_pass.set_vertex_buffer(0, self.water_instances[render_buffer_set].slice(..));
    render_pass.draw_indirect(&self.water_draw_args[render_buffer_set], 0);
    
    // Render lava
    render_pass.set_vertex_buffer(0, self.lava_instances[render_buffer_set].slice(..));
    render_pass.draw_indirect(&self.lava_draw_args[render_buffer_set], 0);
    
    // Render steam
    render_pass.set_vertex_buffer(0, self.steam_instances[render_buffer_set].slice(..));
    render_pass.draw_indirect(&self.steam_draw_args[render_buffer_set], 0);
}
```

### Future: GPU Marching Cubes

**Phase 5 upgrade - still GPU-only:**
- Marching cubes compute shader
- Generate mesh vertices/indices on GPU
- Use indirect draw for dynamic mesh
- Zero CPU mesh generation

---

## Phased Implementation Plan

### Phase 1: Foundation (Water + Boundaries + Adhesion)

**Goal**: Single fluid type (water) with cave constraints, surface adhesion, and basic visualization

#### Step 1.1: GPU Cave Voxel Grid Generation
**File**: `shaders/fluid/generate_cave_grid.wgsl` (new file)

**Generate cave grid entirely on GPU:**

```wgsl
struct CaveParams {
    world_center: vec3<f32>,
    world_radius: f32,
    grid_resolution: u32,
    threshold: f32,
    noise_scale: f32,
    noise_octaves: u32,
}

@group(0) @binding(0) var<uniform> cave_params: CaveParams;
@group(0) @binding(1) var<storage, read_write> solid_mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 2>;  // [solid_count, fluid_count]

// Cave SDF sampling (same as CPU version)
fn sample_cave_density(pos: vec3<f32>) -> f32 {
    // Implement cave SDF on GPU
    // Use noise functions for procedural generation
    // Return density value
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = cave_params.grid_resolution * cave_params.grid_resolution * cave_params.grid_resolution;
    
    if (idx >= total_cells) {
        return;
    }
    
    // Convert linear index to 3D coordinates
    let res = i32(cave_params.grid_resolution);
    let z = i32(idx) / (res * res);
    let y = (i32(idx) - z * res * res) / res;
    let x = i32(idx) - z * res * res - y * res;
    
    // Convert grid coords to world position
    let cell_size = (cave_params.world_radius * 2.0) / f32(cave_params.grid_resolution);
    let grid_pos = cave_params.world_center - vec3<f32>(cave_params.world_radius) + 
                   vec3<f32>(f32(x), f32(y), f32(z)) * cell_size + 
                   vec3<f32>(cell_size * 0.5);
    
    // Sample cave density
    let density = sample_cave_density(grid_pos);
    let is_solid = density > cave_params.threshold;
    
    // Store result
    solid_mask[idx] = select(0u, 1u, is_solid);
    
    // Update statistics atomically (GPU-only tracking)
    if (is_solid) {
        atomicAdd(&stats[0], 1u);  // solid_count
    } else {
        atomicAdd(&stats[1], 1u);  // fluid_count
    }
}
```

**Rust integration (no CPU computation):**

```rust
impl CaveSystemRenderer {
    /// Generate cave grid on GPU (no CPU computation)
    pub fn generate_fluid_collision_buffer_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid_resolution: u32,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        // Create output buffers
        let total_cells = grid_resolution.pow(3) as u64;
        let solid_mask = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Collision Grid"),
            size: total_cells * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let stats = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cave Grid Stats"),
            size: 8,  // 2 × u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Execute GPU generation
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.cave_grid_pipeline);
            pass.set_bind_group(0, &self.cave_grid_bind_group, &[]);
            pass.dispatch_workgroups((total_cells as u32 + 63) / 64, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        
        // No CPU readback - stats buffer can be read by UI shader if needed
        (solid_mask, stats)
    }
}
```

**Validation**: GPU stats buffer (no CPU readback)
- Stats displayed via UI shader reading GPU buffer
- Optional: Copy stats to staging buffer only for UI display

#### Step 1.2: Triple-Buffered Fluid GPU Buffers
**File**: `src/gpu_buffers/fluid_buffers.rs` (new file)

```rust
use wgpu;
use bytemuck::{Pod, Zeroable};
use std::sync::atomic::{AtomicUsize, Ordering};

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
    
    phase_change_rate: f32,
    water_adhesion_strength: f32,
    droplet_threshold: f32,
    droplet_detach_force: f32,
    
    // Padding to 256 bytes
    _padding: [f32; 46],
}

/// Triple-buffered fluid simulation buffers (zero CPU sync)
pub struct FluidSimulationBuffers {
    // Triple-buffered fluid state (128³ cells × 3)
    pub densities: [wgpu::Buffer; 3],        // vec4<f32>: (lava, water, steam, air)
    pub velocity: [wgpu::Buffer; 3],         // vec4<f32>: (vx, vy, vz, padding)
    pub pressure: [wgpu::Buffer; 3],         // f32 (for ping-pong in pressure solve)
    
    // Temporary buffers (single copy)
    pub divergence: wgpu::Buffer,            // f32
    pub pressure_temp: wgpu::Buffer,         // f32 for Jacobi ping-pong
    
    // Cave collision (static, single copy)
    pub solid_mask: wgpu::Buffer,            // u32
    
    // Triple-buffered instance extraction
    pub water_instances: [wgpu::Buffer; 3],  // FluidInstance array
    pub lava_instances: [wgpu::Buffer; 3],   // FluidInstance array
    pub steam_instances: [wgpu::Buffer; 3],  // FluidInstance array
    pub instance_counts: [wgpu::Buffer; 3],  // Atomic counters for each fluid type
    pub draw_args: [wgpu::Buffer; 3],        // Indirect draw arguments
    
    // Mass conservation tracking (GPU-only)
    pub mass_counters: wgpu::Buffer,         // Atomic counters for mass tracking
    
    // Parameters (uniform)
    pub params: wgpu::Buffer,                // FluidParams uniform
    
    // Triple buffer rotation (CPU atomic for index management)
    current_physics_index: AtomicUsize,
    current_render_index: AtomicUsize,
}

impl FluidSimulationBuffers {
    pub fn new(
        device: &wgpu::Device,
        grid_resolution: u32,
        solid_mask_buffer: wgpu::Buffer,  // From GPU cave generation
    ) -> Self {
        let total_cells = (grid_resolution.pow(3)) as u64;
        let vec4_size = 16u64;
        let f32_size = 4u64;
        let max_instances = total_cells;  // Worst case: all cells have fluid
        
        // Helper to create triple-buffered array
        let create_triple_buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            [
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{} 0", label)),
                    size,
                    usage,
                    mapped_at_creation: false,
                }),
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{} 1", label)),
                    size,
                    usage,
                    mapped_at_creation: false,
                }),
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{} 2", label)),
                    size,
                    usage,
                    mapped_at_creation: false,
                }),
            ]
        };
        
        let densities = create_triple_buffer(
            "Fluid Densities",
            total_cells * vec4_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        
        let velocity = create_triple_buffer(
            "Fluid Velocity",
            total_cells * vec4_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        
        let pressure = create_triple_buffer(
            "Fluid Pressure",
            total_cells * f32_size,
            wgpu::BufferUsages::STORAGE,
        );
        
        let water_instances = create_triple_buffer(
            "Water Instances",
            max_instances * 32,  // FluidInstance size
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        );
        
        let lava_instances = create_triple_buffer(
            "Lava Instances",
            max_instances * 32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        );
        
        let steam_instances = create_triple_buffer(
            "Steam Instances",
            max_instances * 32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        );
        
        let instance_counts = create_triple_buffer(
            "Instance Counts",
            12,  // 3 × u32 (water, lava, steam)
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        
        let draw_args = create_triple_buffer(
            "Draw Args",
            20,  // wgpu::util::DrawIndirectArgs size
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        
        let divergence = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Divergence"),
            size: total_cells * f32_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let pressure_temp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Pressure Temp"),
            size: total_cells * f32_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let mass_counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mass Conservation Counters"),
            size: 20,  // 5 × u32 atomic counters
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Params"),
            size: 256,  // FluidParams aligned to 256 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            densities,
            velocity,
            pressure,
            divergence,
            pressure_temp,
            solid_mask: solid_mask_buffer,
            water_instances,
            lava_instances,
            steam_instances,
            instance_counts,
            draw_args,
            mass_counters,
            params,
            current_physics_index: AtomicUsize::new(0),
            current_render_index: AtomicUsize::new(2),
        }
    }
    
    /// Rotate triple buffers atomically (zero sync)
    pub fn rotate_buffers(&self) -> (usize, usize) {
        let physics = self.current_physics_index.fetch_add(1, Ordering::Relaxed) % 3;
        let render = self.current_render_index.fetch_add(1, Ordering::Relaxed) % 3;
        (physics, render)
    }
    
    /// Get current physics buffer index
    pub fn physics_index(&self) -> usize {
        self.current_physics_index.load(Ordering::Relaxed) % 3
    }
    
    /// Get current render buffer index (2 frames behind)
    pub fn render_index(&self) -> usize {
        self.current_render_index.load(Ordering::Relaxed) % 3
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

### Phase 4: Complete GPU Pipeline Integration

**Goal**: Integrate fluid system into GpuScene with zero CPU dependencies

#### Step 4.1: Complete GPU Fluid Pipeline

**File**: `src/simulation/fluid_simulation.rs` (new file)

```rust
pub struct FluidSimulation {
    buffers: FluidSimulationBuffers,
    pipelines: FluidPipelines,
    bind_groups: FluidBindGroups,
    workgroup_count: u32,
}

impl FluidSimulation {
    /// Execute complete fluid pipeline on GPU (zero CPU sync)
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        // Rotate triple buffers atomically
        let (physics_idx, render_idx) = self.buffers.rotate_buffers();
        
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fluid Simulation Pipeline"),
            timestamp_writes: None,
        });
        
        // Stage 1: Clear instance counters
        pass.set_pipeline(&self.pipelines.clear_instance_counts);
        pass.set_bind_group(0, &self.bind_groups.instance_counts[physics_idx], &[]);
        pass.dispatch_workgroups(1, 1, 1);
        
        // Stage 2: Apply forces (gravity, buoyancy, adhesion)
        pass.set_pipeline(&self.pipelines.apply_forces);
        pass.set_bind_group(0, &self.bind_groups.main[physics_idx], &[]);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 3: Advect densities (semi-Lagrangian)
        pass.set_pipeline(&self.pipelines.advect_density);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 4: Advect velocity
        pass.set_pipeline(&self.pipelines.advect_velocity);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 5: Phase changes (water ↔ steam)
        pass.set_pipeline(&self.pipelines.phase_change);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 6: Air displacement
        pass.set_pipeline(&self.pipelines.air_displacement);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 7: Compute divergence
        pass.set_pipeline(&self.pipelines.compute_divergence);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 8-27: Pressure solve (20 Jacobi iterations with ping-pong)
        for i in 0..20 {
            pass.set_pipeline(&self.pipelines.jacobi_pressure);
            let src_idx = i % 2;
            pass.set_bind_group(0, &self.bind_groups.pressure[src_idx], &[]);
            pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        }
        
        // Stage 28: Subtract pressure gradient
        pass.set_pipeline(&self.pipelines.subtract_gradient);
        pass.set_bind_group(0, &self.bind_groups.main[physics_idx], &[]);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 29: Enforce boundaries (free-slip)
        pass.set_pipeline(&self.pipelines.enforce_boundaries);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 30: Update mass counters (GPU-only validation)
        pass.set_pipeline(&self.pipelines.update_mass_counters);
        pass.set_bind_group(0, &self.bind_groups.mass_tracking, &[]);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 31: Extract rendering instances (GPU-to-GPU)
        pass.set_pipeline(&self.pipelines.extract_instances);
        pass.set_bind_group(0, &self.bind_groups.instance_extraction[physics_idx], &[]);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // Stage 32: Build indirect draw arguments
        pass.set_pipeline(&self.pipelines.build_draw_args);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
```

#### Step 4.2: GpuScene Integration (Zero CPU Sync)

**File**: `src/scene/gpu_scene.rs` (modified)

```rust
pub struct GpuScene {
    // ... existing fields ...
    
    fluid_system: Option<FluidSimulation>,
    frame_counter: u32,
}

impl Scene for GpuScene {
    fn step_physics(&mut self, dt: f32) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        
        // Cell physics at 60 FPS (existing 15-stage pipeline)
        self.execute_cell_physics_pipeline(&mut encoder);
        
        // Fluid physics at 30 FPS (every other frame)
        self.frame_counter += 1;
        if self.frame_counter % 2 == 0 {
            if let Some(ref mut fluid) = self.fluid_system {
                fluid.step(&mut encoder);  // 32-stage fluid pipeline
            }
        }
        
        // Single submission for both systems
        self.queue.submit(Some(encoder.finish()));
        
        // ZERO CPU SYNCHRONIZATION - no readbacks, no waits
    }
    
    fn render(&self, encoder: &mut CommandEncoder, view: &TextureView) {
        // Render cells (existing)
        self.render_cells(encoder, view);
        
        // Render fluids using triple-buffered instances (2 frames behind)
        if let Some(ref fluid) = self.fluid_system {
            fluid.render(encoder, view);
        }
    }
}
```

#### Step 4.3: GPU Mass Conservation Validation

**File**: `shaders/fluid/update_mass_counters.wgsl` (new file)

```wgsl
struct MassCounters {
    initial_water: atomic<u32>,
    initial_lava: atomic<u32>,
    current_water: atomic<u32>,
    current_steam: atomic<u32>,
    current_lava: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> mass_counters: MassCounters;
@group(0) @binding(1) var<storage, read> densities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> solid_mask: array<u32>;
@group(0) @binding(3) var<uniform> params: FluidParams;

// Clear current counters
@compute @workgroup_size(1)
fn clear_current_mass() {
    atomicStore(&mass_counters.current_water, 0u);
    atomicStore(&mass_counters.current_steam, 0u);
    atomicStore(&mass_counters.current_lava, 0u);
}

// Accumulate mass from all cells
@compute @workgroup_size(64)
fn accumulate_mass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_cells = params.grid_resolution * params.grid_resolution * params.grid_resolution;
    
    if (idx >= total_cells || solid_mask[idx] != 0u) {
        return;
    }
    
    let density = densities[idx];
    
    // Convert to fixed-point (multiply by 1000 for precision)
    let lava_fixed = u32(density.x * 1000.0);
    let water_fixed = u32(density.y * 1000.0);
    let steam_fixed = u32(density.z * 1000.0);
    
    atomicAdd(&mass_counters.current_lava, lava_fixed);
    atomicAdd(&mass_counters.current_water, water_fixed);
    atomicAdd(&mass_counters.current_steam, steam_fixed);
}
```

#### Step 4.4: GPU-Only UI Display

**Optional: Display mass conservation stats in UI without CPU readback**

```rust
// UI can optionally copy mass counters to staging buffer for display
// This is ONLY for UI display, not for validation
impl FluidSimulation {
    pub fn copy_stats_for_ui(&self, encoder: &mut CommandEncoder) {
        // Only if UI panel is open
        if self.ui_stats_visible {
            encoder.copy_buffer_to_buffer(
                &self.buffers.mass_counters,
                0,
                &self.staging_buffer,
                0,
                20,
            );
        }
    }
}
```

**Validation**:
- Runs at 30 FPS (every other frame)
- Zero CPU synchronization during simulation
- Mass conservation tracked on GPU
- Visual inspection of fluid behavior

---

### Phase 5: UI Controls + Polish (GPU-Only)

**Goal**: Add user controls that trigger GPU reinitialization

#### Step 5.1: UI Sliders (CPU Parameters → GPU Uniforms)

**File**: `src/ui/fluid_controls.rs` (new file)

```rust
pub struct FluidGenerationParams {
    pub cave_density: f32,
    pub water_percent: f32,
    pub lava_percent: f32,
    pub water_adhesion: f32,
    pub droplet_threshold: f32,
}

impl FluidGenerationParams {
    pub fn ui(&mut self, ui: &mut egui::Ui, fluid_system: &mut FluidSimulation) -> bool {
        ui.heading("Fluid Generation");
        
        let mut regenerate = false;
        
        ui.add(egui::Slider::new(&mut self.cave_density, 0.5..=0.7)
            .text("Cave Density"));
        
        let remaining = 1.0 - self.cave_density;
        
        ui.add(egui::Slider::new(&mut self.water_percent, 0.1..=remaining)
            .text("Water %"));
        
        let remaining_after_water = remaining - self.water_percent;
        
        ui.add(egui::Slider::new(&mut self.lava_percent, 0.05..=remaining_after_water)
            .text("Lava %"));
        
        let air_percent = remaining_after_water - self.lava_percent;
        ui.label(format!("Air: {:.1}%", air_percent * 100.0));
        
        ui.separator();
        ui.heading("Fluid Behavior");
        
        if ui.add(egui::Slider::new(&mut self.water_adhesion, 0.0..=1.0)
            .text("Water Adhesion")).changed() {
            // Update GPU uniform immediately
            fluid_system.update_adhesion_param(self.water_adhesion);
        }
        
        if ui.add(egui::Slider::new(&mut self.droplet_threshold, 0.5..=2.0)
            .text("Droplet Threshold")).changed() {
            fluid_system.update_droplet_param(self.droplet_threshold);
        }
        
        if ui.button("Regenerate Fluids").clicked() {
            regenerate = true;
        }
        
        regenerate
    }
}

impl FluidSimulation {
    /// Regenerate fluids entirely on GPU
    pub fn regenerate_gpu(&mut self, encoder: &mut CommandEncoder, params: &FluidGenerationParams) {
        // Update GPU uniform parameters
        self.queue.write_buffer(&self.buffers.params, 0, bytemuck::bytes_of(&FluidParams {
            water_percent: params.water_percent,
            lava_percent: params.lava_percent,
            water_adhesion_strength: params.water_adhesion,
            droplet_threshold: params.droplet_threshold,
            // ... other params
        }));
        
        // Execute GPU initialization pipeline
        let mut pass = encoder.begin_compute_pass(&Default::default());
        
        pass.set_pipeline(&self.pipelines.initialize_fluids);
        pass.set_bind_group(0, &self.bind_groups.init, &[]);
        pass.dispatch_workgroups(self.workgroup_count, 1, 1);
        
        // No CPU involvement in regeneration
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

## Performance Considerations (GPU-Only)

### GPU Memory Usage (Triple Buffered)

Total for 128³ grid with triple buffering:
- Cave solid mask: 2 MB (static)
- Fluid densities: 24 MB (×3 buffer sets)
- Fluid velocity: 24 MB (×3 buffer sets)
- Fluid pressure: 6 MB (×3 for ping-pong)
- Instance buffers: 48 MB (×3 for water/lava/steam)
- Divergence: 2 MB
- Mass counters: 20 bytes
- **Total: ~106 MB**

### Compute Performance (Zero CPU Sync)

At 30 FPS with 128³ grid:
- 32-stage GPU pipeline per frame
- Advection: ~1M fluid cells (skip solids)
- Pressure solve: 20 Jacobi iterations
- Instance extraction: GPU-to-GPU atomic operations
- **Estimated**: 3-6ms per frame on modern GPU
- **Zero CPU overhead** - no readbacks, no synchronization

### Triple Buffering Benefits

1. **Physics computation** - Uses buffer set N
2. **Instance extraction** - Uses buffer set N-1
3. **Rendering** - Uses buffer set N-2
4. **Zero stalls** - All stages run in parallel
5. **Maximum throughput** - GPU never waits for CPU

### Optimization Opportunities

1. **Early solid cell rejection** - 50% work reduction
2. **Workgroup size tuning** - Optimize for GPU architecture
3. **Coalesced memory access** - SoA layout for cache efficiency
4. **Atomic operation minimization** - Batch updates where possible
5. **Indirect draw** - GPU-driven rendering without CPU

---

## Testing & Validation Strategy (GPU-Only)

### Visual Inspection Tests

1. **Water settling**: Water flows downward and pools at bottom
2. **Ceiling droplets**: Water accumulates on ceilings → forms droplets → falls
3. **Wall streaming**: Water clings to walls and trickles downward
4. **Slanted surface flow**: Water follows surface contours
5. **Lava-water interaction**: Steam forms at lava-water interface
6. **Steam rising**: Steam rises and condenses at boundaries
7. **Condensation patterns**: Ceiling droplets, wall streams
8. **Boundary respect**: No fluid penetration into cave walls
9. **Mass conservation**: Total water+steam visually constant

### GPU-Only Quantitative Tests

1. **Mass conservation**: GPU atomic counters track mass (no CPU readback)
2. **Performance**: Maintain 30 FPS fluid, 60 FPS overall
3. **Memory**: Stay within 110 MB GPU memory budget
4. **Triple buffer rotation**: Verify atomic index management
5. **Instance extraction**: Verify GPU-to-GPU instance generation

### Debug Visualization (GPU-Only)

**All debug visualization uses GPU compute shaders:**

```wgsl
// Debug shader: Extract 2D slice for visualization
@compute @workgroup_size(64)
fn extract_debug_slice() {
    // Extract density slice at Z=64
    // Write to debug texture (GPU-to-GPU)
    // No CPU readback required
}

// Debug shader: Velocity field arrows
@compute @workgroup_size(64)
fn generate_velocity_arrows() {
    // Generate arrow instances on GPU
    // Use indirect draw for rendering
}
```

**Debug features:**
1. **Density slices**: GPU-generated 2D cross-sections
2. **Velocity arrows**: GPU-generated vector field instances
3. **Pressure heatmap**: GPU-generated color-coded visualization
4. **Mass counters**: Display GPU atomic counter values in UI

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

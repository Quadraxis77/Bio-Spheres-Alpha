# Simplified Voxel-Based Fluid Simulation - Implementation Plan

## Executive Summary

This document outlines a **simplified, pure voxel-based fluid simulation** for Bio-Spheres. The system uses incompressible flow with discrete fluid types (water, lava, steam) and treats air as empty space. All computation happens on GPU with zero CPU involvement.

**Core Simplifications:**
- **Incompressible fluids only** - No density tracking, just fluid type + fill fraction
- **Air is empty space** - No computation for empty voxels
- **Discrete fluid types** - Water, lava, steam (no mixing)
- **Unified solid boundaries** - Cave walls and world sphere treated identically
- **Pure voxel rendering** - No particles, no marching cubes, GPU instance extraction

---

## System Architecture

### Fluid Model

**3 Fluid Types:**
1. **Water** (liquid)
   - Sinks under gravity
   - Converts to steam on lava contact
   - Condenses from steam at solid boundaries
   - Incompressible

2. **Lava** (liquid)
   - Sinks under gravity (denser than water)
   - Does not change state
   - Triggers water → steam conversion
   - Incompressible

3. **Steam** (gas)
   - Rises (buoyancy force)
   - Condenses to water at solid boundaries
   - Created from water-lava contact
   - Incompressible

**Air:**
- Not simulated
- Empty voxels represent air
- Acts as free surface boundary

### Grid Configuration

- **Resolution**: 128³ voxels (hardcoded)
- **Total cells**: 2,097,152 cells
- **Occupied by cave**: ~50% (1,048,576 cells solid)
- **Available for fluid**: ~50% (1,048,576 cells open)
- **Cell size**: `world_diameter / 128.0`
- **World coordinate mapping**: Grid aligned with world sphere center

### Memory Layout (Per Voxel)

```
fluid_type: u32        // 0=empty, 1=water, 2=lava, 3=steam, 4=solid
fill_fraction: f32     // 0.0-1.0 (how full the voxel is)
velocity: vec3<f32>    // Velocity field
pressure: f32          // Pressure field

Total: 24 bytes per voxel
```

### Memory Budget (128³ Grid, Triple Buffered)

- Fluid state (type + fill): 16 MB × 3 = 48 MB
- Velocity: 24 MB × 3 = 72 MB
- Pressure: 8 MB × 2 = 16 MB (ping-pong only)
- Divergence: 8 MB (single buffer)
- Solid mask: 8 MB (static, single buffer)
- Face buffers: 48 MB × 3 = 144 MB (1M faces typical)
- **Total: ~296 MB**

**Note:** Face count varies with fluid distribution. Worst case 6M faces (all surface), typical ~1M faces (some interior).

---

## Solid Boundary System

### Unified Solid Mask

**Two boundary types, one treatment:**

1. **Cave Volume Boundaries**
   - Interior cave walls from SDF
   - Voxelized during cave generation
   - Marked as solid in mask

2. **World Sphere Boundaries**
   - Outer edge of simulation domain
   - Voxels outside world sphere marked solid
   - Prevents fluid escape

**Generation Process:**
```
For each voxel (x, y, z):
  world_pos = grid_to_world(x, y, z)
  
  // Check world sphere boundary
  if distance(world_pos, sphere_center) > sphere_radius:
    solid_mask[idx] = SOLID
    continue
  
  // Check cave interior
  if sample_cave_sdf(world_pos) > threshold:
    solid_mask[idx] = SOLID
    continue
  
  // Otherwise fluid-capable
  solid_mask[idx] = EMPTY
```

### Boundary Conditions

**Velocity Boundary (Free-Slip):**
- Zero normal component at solid boundaries
- Tangential component preserved
- Applied after pressure projection

**Pressure Boundary (Neumann):**
- Solid voxels excluded from pressure solve
- Zero gradient at boundaries (natural from Jacobi)
- Fluid voxels adjacent to solid handle boundary implicitly

**Advection Boundary:**
- Cannot advect into solid voxels
- Back-trace clamped to fluid domain
- Prevents penetration

**Phase Change at Boundaries:**
- Steam touching solid → condenses to water
- Applies to both cave walls and sphere edge
- Instant conversion (100% per frame)

---

## Simulation Pipeline

### GPU Compute Pipeline (30 FPS)

**Stage 1: Apply Forces**
- Gravity to water/lava (downward, proportional to gravity strength)
- Buoyancy to steam (upward, proportional to gravity strength)
- Surface adhesion forces (water only, disabled if gravity = 0)

**Stage 2: Advect Velocity (RK2)**
- Semi-Lagrangian advection
- RK2 for accuracy
- Sample velocity at midpoint

**Stage 3: Advect Fluid State**
- Move fluid_type and fill_fraction
- Semi-Lagrangian with RK2
- Empty voxels don't advect

**Stage 4: Phase Changes**
- Water + lava neighbor → steam
- Steam + solid neighbor → water
- Discrete type changes

**Stage 5: Vorticity Confinement**
- Compute curl of velocity
- Apply confinement force
- Creates realistic swirling

**Stage 6: Compute Divergence**
- Only for fluid voxels
- ∇·v = (∂u/∂x + ∂v/∂y + ∂w/∂z)
- Exact on collocated grid

**Stage 7: Pressure Solve (Red-Black SOR)**
- 10 iterations (red-black pattern)
- Successive over-relaxation (ω=1.9)
- Only fluid voxels participate
- Solid voxels skipped

**Stage 8: Subtract Pressure Gradient**
- Make velocity divergence-free
- v_new = v_old - ∇p
- Only for fluid voxels

**Stage 9: Enforce Boundaries**
- Apply free-slip at solid boundaries
- Zero normal velocity component
- Clamp to domain

**Stage 10: Extract Faces (GPU)**
- Scan fluid voxels
- Check 6 neighbors per voxel
- Generate face if neighbor is different type
- Atomic counter for face count
- Indirect draw preparation

---

## Rendering System

### Minecraft-Style Voxel Surface

**Continuous surface with face culling** - render only visible faces, creating a solid fluid volume.

**Key Concept:**
- Fluid voxels form a continuous surface
- Only render faces adjacent to empty/different-type voxels
- Interior faces are culled (never rendered)
- Exactly like Minecraft block rendering

### Face Culling System

**GPU Face Extraction (Compute Shader):**
```
For each fluid voxel:
  if fluid_type == EMPTY or fluid_type == SOLID:
    continue
  
  // Check 6 neighbors
  for each direction (±X, ±Y, ±Z):
    neighbor_type = get_neighbor_fluid_type(direction)
    
    // Render face if neighbor is different
    if neighbor_type != current_type:
      face_idx = atomic_increment(face_counter)
      faces[face_idx] = {
        position: voxel_center,
        normal: direction,
        fluid_type: current_type,
        fill_fraction: current_fill
      }
```

**Face Rendering Conditions:**

| Current Voxel | Neighbor Voxel | Render Face? |
|---------------|----------------|--------------|
| Water | Water | No (interior) |
| Water | Empty | Yes (surface) |
| Water | Lava | Yes (boundary) |
| Water | Steam | Yes (boundary) |
| Water | Solid | No (boundary handled by solid) |
| Lava | Lava | No (interior) |
| Lava | Empty | Yes (surface) |
| Steam | Steam | No (interior) |
| Steam | Empty | Yes (surface) |

**Benefits:**
- Massive reduction in geometry (only surface faces)
- Continuous appearance (no gaps between voxels)
- Natural smooth surface from lighting
- Efficient GPU rendering

### Rendering Pipeline

**Stage 1: Extract Visible Faces (GPU Compute)**
```
Compute shader scans all fluid voxels
→ For each voxel, check 6 neighbors
→ Generate face quad if neighbor is different type
→ Write face data to buffer
→ Atomic counter tracks total face count
```

**Stage 2: Render Faces (Indirect Draw)**
```
Vertex shader generates quad from face data
→ Position + normal define the face
→ Fragment shader applies lighting
→ Smooth appearance from per-pixel lighting
```

**Memory:**
- Max faces: 6 × 1M fluid voxels = 6M faces (worst case)
- Typical: ~1M faces (most voxels have some interior faces)
- Face data: 32 bytes per face
- Triple buffered: 32 MB × 3 = 96 MB

### Smooth Surface Appearance

**Lighting-Based Smoothness:**
- Each face has exact normal (±X, ±Y, ±Z)
- Per-pixel lighting creates smooth appearance
- No need for gradient normals
- Phong or PBR shading

**Optional: Ambient Occlusion**
- Check corner neighbors for each face
- Darken corners where voxels meet
- Adds depth and definition
- Computed in vertex shader

**Optional: Smooth Normals**
- Average normals from adjacent faces
- Creates rounded appearance
- Computed from fill_fraction gradient
- Applied in fragment shader

**Color Variation:**
- Base color from fluid type
- Modulate by fill_fraction (fuller = more opaque)
- Add slight noise for variation
- Depth-based darkening

### Face Data Structure

```
struct FluidFace {
  position: vec3<f32>,      // Voxel center
  normal: vec3<f32>,        // Face direction (±1 in one axis)
  fluid_type: u32,          // Water, lava, or steam
  fill_fraction: f32,       // 0-1 (affects color/alpha)
  ao_factors: vec4<f32>,    // Ambient occlusion per corner (optional)
}

Size: 48 bytes per face
```

### Rendering Per Fluid Type

**Water (Opaque, Reflective):**
- Solid blue color
- Specular highlights
- Optional: Screen-space reflections
- Faces rendered front-to-back

**Lava (Opaque, Emissive):**
- Orange/red gradient
- Emissive glow
- Optional: Animated texture
- Faces rendered front-to-back

**Steam (Transparent, Soft):**
- White/gray with alpha
- Soft edges
- Additive or alpha blending
- Faces rendered back-to-front

### Optimization: Greedy Meshing

**Optional future enhancement:**
- Merge adjacent coplanar faces into larger quads
- Reduces face count by 10-100x
- More complex extraction shader
- Same visual result

**Example:**
```
Before: 10×10 water surface = 100 faces
After greedy meshing: 1 large quad = 1 face
```

**Implementation:**
- Scan each axis plane
- Find rectangular regions of same type
- Emit one large quad instead of many small ones
- Requires more complex GPU algorithm

### Visual Quality Techniques

**Per-Face:**
- Exact normals (no interpolation needed)
- Per-pixel lighting (smooth appearance)
- Ambient occlusion (depth at corners)
- Color variation (fill_fraction modulation)

**Per-Fluid:**
- Water: Blue, specular, reflective
- Lava: Orange, emissive, glowing
- Steam: White, transparent, soft

**Post-Processing:**
- Optional: Screen-space smoothing
- Optional: Depth-based fog
- Optional: Bloom for lava glow

---

## Phase Change System

### Direct Contact Model

**No temperature tracking** - phase changes occur on direct neighbor contact.

**Water → Steam (Boiling):**
- Trigger: Water voxel has lava neighbor (6-connected)
- Rate: Instant conversion (100% per frame)
- Result: Water voxel becomes steam voxel
- Mass: Conserved (fill_fraction preserved)

**Steam → Water (Condensation):**
- Trigger: Steam voxel touches solid boundary
- Rate: Instant conversion (100% per frame)
- Result: Steam voxel becomes water voxel
- Location: At boundary surface

**Lava (Inert):**
- No phase changes
- Triggers water → steam in adjacent voxels
- Remains lava indefinitely

### Implementation Details

**Phase Change Shader:**
```
For each fluid voxel:
  if fluid_type == WATER:
    // Check 6 neighbors for lava
    if has_lava_neighbor():
      fluid_type = STEAM
      // fill_fraction unchanged (mass conserved)
  
  if fluid_type == STEAM:
    // Check 6 neighbors for solid
    if has_solid_neighbor():
      fluid_type = WATER
      // fill_fraction unchanged (mass conserved)
```

**Mass Conservation:**
- Fill_fraction preserved during phase change
- Total fluid volume conserved
- No creation or destruction
- GPU atomic counters track totals (optional validation)

---

## Surface Adhesion (Water Only)

### Orientation-Based Behavior

Water behavior depends on surface orientation relative to gravity.

**Surface Types:**

1. **Ceilings** (horizontal, facing down)
   - Water accumulates until threshold
   - Forms droplets
   - Detaches and falls when heavy enough

2. **Walls** (vertical)
   - Water clings and streams downward
   - Thin films along surface
   - Adhesion competes with gravity

3. **Slanted Surfaces** (angled)
   - Water trickles along surface
   - Follows surface contours
   - Combination of adhesion + gravity

4. **Floors** (horizontal, facing up)
   - Water pools naturally
   - Gravity handles it
   - No special adhesion needed

### Surface Normal Detection

```
For each water voxel:
  // Check 6 neighbors for solid
  surface_normal = vec3(0)
  
  for each direction:
    if neighbor_is_solid(direction):
      surface_normal -= direction_vector
  
  if length(surface_normal) > 0:
    surface_normal = normalize(surface_normal)
    apply_adhesion_force(surface_normal)
```

### Adhesion Forces

**Parameters:**
- `water_adhesion_strength`: 0.3-0.8
- `droplet_threshold`: 0.5-2.0 (fill_fraction)
- `droplet_detach_force`: 1.0-5.0

**Force Application:**
```
dot_gravity = dot(surface_normal, gravity_dir)

if dot_gravity > 0.7:  // Ceiling
  if fill_fraction > droplet_threshold:
    // Heavy droplet - detach
    force = gravity_dir * droplet_detach_force
  else:
    // Light water - cling
    force = surface_normal * adhesion_strength

else if abs(dot_gravity) < 0.7:  // Wall/slant
  // Adhesion + tangential flow
  force = surface_normal * adhesion_strength * 0.5
  force += project_gravity_on_surface(surface_normal)

// else: floor - no special handling
```

---

## Numerical Methods

### Advection (RK2 Semi-Lagrangian)

**Why RK2:**
- 2x more accurate than Euler
- Minimal extra cost (one extra sample)
- Reduces numerical diffusion significantly

**Algorithm:**
```
// Current position
pos = voxel_center

// Sample velocity at current position
vel = sample_velocity(pos)

// RK2: Sample at midpoint
mid_pos = pos - vel * dt * 0.5
mid_vel = sample_velocity(mid_pos)

// Use midpoint velocity for full step
back_pos = pos - mid_vel * dt

// Sample quantity at back-traced position
result = sample_trilinear(back_pos)
```

**Trilinear Interpolation:**
- Sample 8 surrounding voxels
- Weight by distance
- Smooth continuous field from discrete grid

### Pressure Solve (Red-Black SOR)

**Why Red-Black:**
- GPU-friendly parallelization
- Faster convergence than Jacobi
- Checkerboard pattern avoids data races

**Why SOR:**
- Over-relaxation accelerates convergence
- ω=1.9 typically optimal
- Reduces iterations from 20 → 10

**Algorithm:**
```
// Red pass (even cells)
for each voxel where (x+y+z) % 2 == 0:
  if fluid_type != EMPTY and fluid_type != SOLID:
    neighbor_sum = sum_pressure_from_fluid_neighbors()
    neighbor_count = count_fluid_neighbors()
    
    new_pressure = neighbor_sum / neighbor_count - divergence
    
    // SOR update
    pressure = pressure + omega * (new_pressure - pressure)

// Black pass (odd cells)
for each voxel where (x+y+z) % 2 == 1:
  // Same as red pass
```

**Convergence:**
- 10 iterations sufficient for 128³ grid
- Each iteration = 2 passes (red + black)
- Total: 20 compute dispatches

### Vorticity Confinement

**Purpose:**
- Adds realistic swirling motion
- Prevents numerical dissipation from killing vortices
- Makes fluid look "alive"

**Algorithm:**
```
// Compute curl (vorticity)
curl = compute_curl(velocity_field)

// Compute gradient of curl magnitude
grad_curl_mag = compute_gradient(length(curl))

// Confinement force
N = normalize(grad_curl_mag)
force = epsilon * cross(N, curl)

// Apply to velocity
velocity += force * dt
```

**Parameter:**
- `vorticity_epsilon`: 0.01-0.1 (tunable)

---

## GPU Optimization Strategy

### Workgroup Configuration

**For 3D grid operations:**
- Workgroup size: 4×4×4 = 64 threads
- Good spatial locality
- Cache-friendly neighbor access

**For 1D scans:**
- Workgroup size: 64×1×1
- Good for instance extraction
- Coalesced memory access

### Memory Access Patterns

**Structure of Arrays (SoA):**
- Separate buffers for each field
- Coalesced access in compute shaders
- Cache-friendly

**Triple Buffering:**
- Physics reads from buffer N-1, writes to buffer N
- Rendering reads from buffer N-2
- Zero synchronization, maximum throughput

### Early Termination

**Solid cell rejection:**
```
if solid_mask[idx] != 0:
  return  // Skip immediately
```

**Empty cell rejection:**
```
if fluid_type[idx] == EMPTY:
  return  // Skip immediately
```

**Impact:**
- Cave is ~50% solid → 50% work reduction
- Fluid fills ~25% of remaining → 75% total reduction
- Only ~500K voxels processed out of 2M

### Compute Pass Organization

**Separate passes for clarity:**
- Each stage is one compute shader
- Easy to debug and profile
- Can merge later if needed

**Potential merges:**
- Apply forces + advect velocity
- Advect state + phase changes
- Profile first, optimize later

### Indirect Rendering

**GPU builds draw list:**
- Compute shader scans grid
- Atomic counter for instance count
- Write instance data to buffer
- Indirect draw uses GPU count

**Benefits:**
- Zero CPU involvement
- Only render visible voxels
- Automatic LOD (skip low fill_fraction)

---

## Integration with Bio-Spheres

### GpuScene Integration

**Fluid system as component:**
- Runs at 30 FPS (every other frame)
- Cell physics runs at 60 FPS
- Single command buffer submission
- Zero CPU synchronization

**Execution order:**
```
Frame N:
  - Cell physics (15 stages)
  - Fluid physics (10 stages) if frame % 2 == 0
  - Submit command buffer
  - Render cells
  - Render fluids (from buffer N-2)
```

### Fluid-Cell Interaction

**One-way coupling (fluids affect cells):**

**Cell physics reads fluid data:**
- Sample fluid grid at cell position
- Apply buoyancy force
- Apply drag force
- Optional: lava damage

**Cells do NOT affect fluids:**
- Cells are "ghosts" to fluid
- No displacement
- No consumption
- Maintains determinism

**Implementation:**
```
In cell physics shader:
  cell_pos = positions[cell_idx]
  
  // Sample fluid grid (GPU-to-GPU)
  fluid_idx = world_pos_to_grid_index(cell_pos)
  fluid_type = fluid_types[fluid_idx]
  fill = fill_fractions[fluid_idx]
  
  // Apply forces based on fluid
  if fluid_type == WATER:
    force += buoyancy_force(cell_density, water_density)
  
  if fluid_type == LAVA:
    force += buoyancy_force(cell_density, lava_density)
    mass_loss += lava_damage_rate * dt
  
  if fluid_type == STEAM:
    force += buoyancy_force(cell_density, steam_density)
```

### Cave System Integration

**Reuse existing cave generation:**
- Cave SDF already exists
- Voxelize for fluid collision
- Generate on GPU (no CPU computation)

**Shared solid mask:**
- Cave renderer uses same mask
- Fluid system uses same mask
- Single source of truth

---

## Phased Implementation Plan

### Phase 1: Foundation (4-5 days)

**Goal:** Single fluid (water) flowing and settling

**Tasks:**
1. Create GPU buffer structures (fluid state, velocity, pressure)
2. Generate unified solid mask (cave + sphere)
3. Implement basic advection (Euler first, then RK2)
4. Implement Jacobi pressure solve (20 iterations)
5. Implement boundary conditions (free-slip)
6. Simple cube rendering (GPU instance extraction)

**Validation:**
- Water flows downward
- Water settles in pools
- Water respects cave boundaries
- No penetration through walls

### Phase 2: Numerical Quality (2-3 days)

**Goal:** Smooth, realistic fluid motion

**Tasks:**
1. Upgrade to RK2 advection
2. Implement Red-Black SOR (reduce to 10 iterations)
3. Implement vorticity confinement
4. Add smooth normals from gradient
5. Tune rendering (lighting, scaling)

**Validation:**
- Fluid swirls realistically
- Less numerical diffusion
- Smooth appearance
- Good performance

### Phase 3: Multi-Fluid (2-3 days)

**Goal:** Water, lava, steam with phase changes

**Tasks:**
1. Add lava and steam fluid types
2. Implement phase change logic
3. Add surface adhesion (water only)
4. Separate rendering per fluid type
5. Tune buoyancy forces

**Validation:**
- Lava sinks below water
- Steam rises and condenses
- Water clings to ceilings/walls
- Phase changes work correctly

### Phase 4: Scale & Optimize (2-3 days)

**Goal:** 128³ resolution at 30 FPS

**Tasks:**
1. Scale from 64³ to 128³
2. Profile GPU performance
3. Optimize bottlenecks
4. Tune workgroup sizes
5. Merge compute passes if needed

**Validation:**
- Maintains 30 FPS fluid update
- 60 FPS overall frame rate
- Memory within budget
- Visual quality acceptable

### Phase 5: Integration & Polish (2-3 days)

**Goal:** Integrated with cell physics, production ready

**Tasks:**
1. Implement fluid-cell interaction
2. Add UI controls (regenerate, parameters)
3. Tune visual parameters
4. Add debug visualization
5. Performance optimization

**Validation:**
- Cells react to fluids
- User can control fluid generation
- Stable and performant
- Ready for gameplay

**Total Timeline: 12-16 days**

---

## Success Criteria

### Functional Requirements

- [ ] Water flows and pools correctly
- [ ] Lava stays at bottom, denser than water
- [ ] Steam rises and condenses at boundaries
- [ ] Phase changes work (water ↔ steam)
- [ ] No fluid penetration through boundaries
- [ ] Surface adhesion creates droplets/streams
- [ ] Cells affected by fluid (buoyancy, drag)

### Performance Requirements

- [ ] 30 FPS fluid simulation (128³ grid)
- [ ] 60 FPS overall frame rate maintained
- [ ] < 250 MB GPU memory for fluid system
- [ ] Zero CPU involvement during simulation
- [ ] Stable over long runs (no drift/explosion)

### Quality Requirements

- [ ] Smooth visual appearance
- [ ] Realistic swirling motion (vorticity)
- [ ] Mass conservation error < 1%
- [ ] No visible artifacts (flickering, popping)
- [ ] Responsive to parameter changes

---

## Technical Specifications

### Shader Files

```
shaders/fluid/
├── apply_forces.wgsl              # Gravity, buoyancy, adhesion
├── advect_velocity_rk2.wgsl       # RK2 velocity advection
├── advect_state_rk2.wgsl          # RK2 fluid state advection
├── phase_change.wgsl              # Water ↔ steam conversion
├── vorticity_confinement.wgsl     # Curl computation and force
├── compute_divergence.wgsl        # ∇·v calculation
├── pressure_solve_red.wgsl        # Red-Black SOR (red pass)
├── pressure_solve_black.wgsl      # Red-Black SOR (black pass)
├── subtract_gradient.wgsl         # Pressure projection
├── enforce_boundaries.wgsl        # Free-slip boundary conditions
└── extract_faces.wgsl             # GPU face extraction with culling
```

### Rust Modules

```
src/simulation/fluid_simulation/
├── mod.rs                         # Module exports
├── buffers.rs                     # GPU buffer management
├── pipelines.rs                   # Compute pipeline creation
├── simulation.rs                  # Main simulation loop
└── renderer.rs                    # Voxel rendering

src/rendering/
└── fluid_renderer.rs              # Rendering integration
```

### Parameters Structure

```
FluidParams (uniform buffer, 256-byte aligned):
  grid_resolution: u32
  cell_size: f32
  world_center: vec3<f32>
  world_radius: f32
  
  gravity: f32
  water_density: f32
  lava_density: f32
  steam_density: f32
  
  dt: f32
  vorticity_epsilon: f32
  sor_omega: f32
  pressure_iterations: u32
  
  water_adhesion_strength: f32
  droplet_threshold: f32
  droplet_detach_force: f32
  
  phase_change_rate: f32
  
  [padding to 256 bytes]
```

---

## Testing & Validation

### Visual Tests

1. **Water settling** - Water flows down and pools at bottom
2. **Boundary respect** - No penetration through cave walls or sphere
3. **Lava-water separation** - Lava sinks below water
4. **Steam rising** - Steam rises and condenses at ceiling
5. **Droplet formation** - Water accumulates on ceilings, falls as droplets
6. **Wall streaming** - Water clings to walls and trickles down
7. **Phase changes** - Steam forms at lava-water interface
8. **Vorticity** - Fluid swirls around obstacles

### Quantitative Tests

1. **Mass conservation** - Total fluid volume constant over time
2. **Performance** - 30 FPS fluid, 60 FPS overall
3. **Memory** - Within 250 MB budget
4. **Stability** - No explosion or drift over 1000 frames
5. **Divergence** - Velocity field divergence-free after projection

### Debug Visualization

**GPU-generated debug views:**
- Density slice (2D cross-section)
- Velocity field arrows
- Pressure heatmap
- Fluid type overlay
- Fill fraction visualization

---

## Future Enhancements

### Potential Upgrades (Post-MVP)

1. **MAC Grid** - Staggered velocity for better accuracy
2. **Higher Resolution** - 256³ or adaptive refinement
3. **Better Rendering** - Screen-space smoothing or marching cubes
4. **Two-Way Coupling** - Cells displace fluid
5. **Temperature Field** - More realistic phase changes
6. **Multiple Fluids** - Oil, acid, etc.
7. **Viscosity** - Different fluid viscosities
8. **Surface Tension** - Better droplet formation

### Not Planned

- Particle-based fluids (SPH, FLIP)
- Compressible flow
- Turbulence modeling
- Chemical reactions
- Erosion/deformation

---

## References

### Techniques Used

- **Semi-Lagrangian Advection** - Stable, unconditionally stable
- **Pressure Projection** - Enforces incompressibility
- **Red-Black SOR** - Fast pressure solve on GPU
- **Vorticity Confinement** - Preserves swirling motion
- **RK2 Integration** - Accurate time stepping

### Inspirations

- Sebastian Lague's fluid simulation video
- GPU Gems 3 - Chapter 30 (Real-Time Simulation and Rendering of 3D Fluids)
- Bridson's "Fluid Simulation for Computer Graphics"
- Stam's "Stable Fluids" paper

---

*Document Version: 1.0*  
*Created: January 21, 2026*  
*Bio-Spheres Simplified Voxel Fluid System*

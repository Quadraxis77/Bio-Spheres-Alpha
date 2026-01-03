# Bio-Spheres Developer Guide

Welcome to Bio-Spheres! This guide will help you understand the codebase architecture and get started with development.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Core Concepts](#core-concepts)
4. [Key Files to Understand](#key-files-to-understand)
5. [Development Workflow](#development-workflow)
6. [Adding New Features](#adding-new-features)
7. [Performance Considerations](#performance-considerations)
8. [Debugging and Tools](#debugging-and-tools)
9. [Common Patterns](#common-patterns)
10. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Rust 1.70+ with cargo
- GPU with Vulkan/DirectX 12/Metal support
- Git for version control

### Building and Running

```bash
# Clone the repository
git clone <repository-url>
cd bio-spheres

# Build and run in debug mode
cargo run

# Build optimized release version
cargo build --release
```

### First Steps

1. **Explore the UI**: The application starts in Preview mode with a default genome
2. **Add cells**: Use the "Add Cell" button to populate the simulation
3. **Edit genomes**: Open the Genome Editor panel to modify cell behavior
4. **Switch modes**: Try GPU mode for larger simulations (>1000 cells)
5. **Experiment**: Modify physics parameters and observe the effects

## Architecture Overview

Bio-Spheres is organized into four main subsystems:

```
┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   User Interface│
│     (app.rs)    │◄──►│    (ui/*.rs)    │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Simulation    │    │    Rendering    │
│(simulation/*.rs)│◄──►│ (rendering/*.rs)│
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Genome System │    │   Cell System   │
│ (genome/*.rs)   │◄──►│  (cell/*.rs)    │
└─────────────────┘    └─────────────────┘
```

### Data Flow

```
Input Events → Camera Update → Physics Step → State Update → Render → Present
     ▲                                                           │
     └───────────────── UI Interaction ◄────────────────────────┘
```

## Core Concepts

### Structure-of-Arrays (SoA) Layout

Bio-Spheres uses SoA instead of traditional object-oriented design for performance:

```rust
// Traditional AoS (Array-of-Structures) - AVOID
struct Cell {
    position: Vec3,
    velocity: Vec3,
    mass: f32,
    // ... 20+ fields
}
let cells: Vec<Cell> = vec![...];

// Bio-Spheres SoA (Structure-of-Arrays) - PREFERRED
struct CanonicalState {
    positions: Vec<Vec3>,    // All positions together
    velocities: Vec<Vec3>,   // All velocities together  
    masses: Vec<f32>,        // All masses together
    // ...
}
```

**Benefits:**
- **Cache Efficiency**: Only load data you need
- **SIMD Friendly**: Vectorized operations on contiguous arrays
- **GPU Compatible**: Matches compute shader memory patterns
- **Parallel Processing**: Different systems can work on different arrays

### Simulation Modes

#### Preview Mode (CPU Physics)
- **Use Case**: Genome editing, small simulations (<1000 cells)
- **Physics**: CPU-based with deterministic spatial grid
- **Features**: Time scrubbing, checkpoints, real-time genome editing
- **Performance**: Good for interactive development

#### GPU Mode (GPU Compute)
- **Use Case**: Large-scale simulations (10k+ cells)
- **Physics**: GPU compute shaders with parallel processing
- **Features**: Frustum culling, Hi-Z occlusion culling, interactive tools
- **Performance**: Scales to massive cell populations

### Genome System

Genomes define cell behavior through **modes** and **transitions**:

```rust
struct Genome {
    modes: Vec<ModeSettings>,  // Different behavior states
}

struct ModeSettings {
    division_settings: DivisionSettings,    // When/how to divide
    adhesion_settings: AdhesionSettings,    // Cell-cell connections
    visual_settings: VisualSettings,        // Appearance
    // ...
}
```

Cells transition between modes based on:
- Time since birth
- Mass accumulation
- Environmental conditions
- Random events

## Key Files to Understand

### Tier 1: Critical Foundation
Start with these files to understand the core architecture:

1. **`src/main.rs`** - Application entry point
2. **`src/lib.rs`** - Module organization and public API
3. **`src/app.rs`** - Main application coordinator with wgpu setup
4. **`src/simulation/canonical_state.rs`** - Central SoA data structure
5. **`src/scene/manager.rs`** - Scene lifecycle management

### Tier 2: Physics and Rendering
Once you understand the foundation:

6. **`src/simulation/cpu_physics.rs`** - CPU physics engine
7. **`src/rendering/cells.rs`** - GPU instanced cell rendering
8. **`src/rendering/mod.rs`** - Rendering pipeline organization
9. **`src/simulation/physics_config.rs`** - Physics parameters

### Tier 3: Genome and UI
For behavior and interaction systems:

10. **`src/genome/mod.rs`** - Genome representation
11. **`src/ui/camera.rs`** - 6DOF camera controller
12. **`src/cell/division.rs`** - Cell division logic
13. **`src/cell/adhesion.rs`** - Cell-cell connections

## Development Workflow

### Setting Up Your Environment

1. **IDE Setup**: Use VS Code with rust-analyzer extension
2. **Logging**: Set `RUST_LOG=debug` for detailed logging
3. **GPU Debugging**: Install RenderDoc for GPU profiling

### Code Organization

```
src/
├── app.rs              # Application entry, event loop, wgpu setup
├── lib.rs              # Module exports and documentation
├── main.rs             # Binary entry point
├── cell/               # Cell types and behaviors
├── genome/             # Genome representation and editing
├── rendering/          # wgpu rendering pipelines
├── simulation/         # Physics and state management
├── scene/              # Scene coordination (Preview/GPU modes)
└── ui/                 # User interface and camera
```

### Testing Strategy

```bash
# Run all tests
cargo test

# Run specific test module
cargo test simulation::tests

# Run with logging
RUST_LOG=debug cargo test -- --nocapture

# Property-based testing (uses proptest)
cargo test proptest
```

### Performance Profiling

```bash
# Profile CPU performance
cargo build --release
perf record --call-graph=dwarf ./target/release/bio-spheres
perf report

# Profile GPU with RenderDoc
# 1. Launch RenderDoc
# 2. Set executable to ./target/release/bio-spheres
# 3. Capture frame and analyze
```

## Adding New Features

### Adding a New Cell Property

1. **Add to CanonicalState** (`src/simulation/canonical_state.rs`):
```rust
pub struct CanonicalState {
    // ... existing fields
    pub my_new_property: Vec<f32>,  // Add your property
}
```

2. **Initialize in constructors**:
```rust
impl CanonicalState {
    pub fn new(capacity: usize) -> Self {
        Self {
            // ... existing initialization
            my_new_property: vec![default_value; capacity],
        }
    }
    
    pub fn add_cell(&mut self, /* params */, new_property: f32) -> Option<usize> {
        // ... existing code
        self.my_new_property[index] = new_property;
        // ...
    }
}
```

3. **Update physics** (`src/simulation/cpu_physics.rs`):
```rust
pub fn step(state: &mut CanonicalState, dt: f32) {
    // ... existing physics
    
    // Update your property
    for i in 0..state.cell_count {
        state.my_new_property[i] += some_calculation(dt);
    }
}
```

4. **Expose in UI** (appropriate panel in `src/ui/`):
```rust
ui.add(egui::Slider::new(&mut my_property, 0.0..=10.0).text("My Property"));
```

### Adding a New Render Effect

1. **Create shader** (`shaders/my_effect.wgsl`):
```wgsl
@vertex
fn vs_main(/* vertex input */) -> VertexOutput {
    // Vertex shader logic
}

@fragment  
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Fragment shader logic
}
```

2. **Create renderer** (`src/rendering/my_effect.rs`):
```rust
pub struct MyEffectRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    // ... other resources
}

impl MyEffectRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        // Create pipeline, bind groups, buffers
    }
    
    pub fn render(&self, pass: &mut wgpu::RenderPass, /* params */) {
        // Render commands
    }
}
```

3. **Integrate into render pass** (`src/scene/preview_scene.rs` or `src/scene/gpu_scene.rs`):
```rust
impl Scene {
    fn render(&mut self, /* params */) {
        // ... existing render calls
        self.my_effect_renderer.render(&mut render_pass, params);
    }
}
```

### Adding a New Physics Force

1. **Implement in CPU physics** (`src/simulation/cpu_physics.rs`):
```rust
fn calculate_my_force(state: &CanonicalState, cell_idx: usize) -> Vec3 {
    // Force calculation logic
}

pub fn step(state: &mut CanonicalState, dt: f32) {
    // ... existing force calculations
    
    // Add your force
    for i in 0..state.cell_count {
        let my_force = calculate_my_force(state, i);
        state.forces[i] += my_force;
    }
}
```

2. **Mirror in GPU compute** (`shaders/physics.wgsl`):
```wgsl
fn calculate_my_force(cell_idx: u32) -> vec3<f32> {
    // Same logic as CPU version
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // ... existing GPU physics
    let my_force = calculate_my_force(global_id.x);
    forces[global_id.x] += my_force;
}
```

3. **Add parameters** (`src/simulation/physics_config.rs`):
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    // ... existing parameters
    pub my_force_strength: f32,
}
```

## Performance Considerations

### Memory Layout

- **Use SoA**: Keep related data in separate arrays
- **Pre-allocate**: Avoid runtime allocations in hot paths
- **Align data**: Use `glam` types for SIMD alignment

### CPU Optimization

```rust
// GOOD: Cache-friendly iteration
for i in 0..state.cell_count {
    state.positions[i] += state.velocities[i] * dt;
}

// AVOID: Jumping between arrays
for i in 0..state.cell_count {
    let pos = state.positions[i];
    let vel = state.velocities[i];
    let mass = state.masses[i];  // Cache miss if arrays are large
    // ...
}

// BETTER: Process in chunks or use separate loops
```

### GPU Optimization

- **Minimize draw calls**: Use instancing for repeated geometry
- **Batch state changes**: Sort by pipeline, then bind group
- **Avoid CPU-GPU sync**: Use double buffering for compute

### Parallel Processing

```rust
use rayon::prelude::*;

// Parallel iteration over large datasets
state.forces[0..state.cell_count]
    .par_iter_mut()
    .enumerate()
    .for_each(|(i, force)| {
        *force = calculate_forces(i, &state);
    });
```

## Debugging and Tools

### Logging

```rust
// Use structured logging
log::info!("Cell count: {}", state.cell_count);
log::debug!("Position: {:?}", pos);  // Only in debug builds
log::warn!("Unusual condition: {}", condition);
log::error!("Failed to allocate: {}", error);
```

### Visual Debugging

- **Debug renderer**: Enable collision bounds visualization
- **Adhesion lines**: Show cell-cell connections
- **Camera info**: Display position and orientation in UI
- **Performance metrics**: Monitor FPS, cell count, culling stats

### GPU Debugging

1. **RenderDoc**: Capture and analyze GPU frames
2. **wgpu labels**: All resources have descriptive names
3. **Validation layers**: Enable in debug builds for error checking

```rust
// Always use descriptive labels
let buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Cell Instance Buffer"),  // Shows up in debuggers
    // ...
});
```

### Common Debug Scenarios

#### Physics Issues
- Check force accumulation in `cpu_physics.rs`
- Verify spatial grid is updating correctly
- Look for NaN values in position/velocity arrays

#### Rendering Issues  
- Verify instance buffer updates
- Check pipeline state changes
- Use RenderDoc to inspect draw calls

#### Performance Issues
- Profile with `perf` or `cargo flamegraph`
- Check for unnecessary allocations
- Monitor GPU utilization

## Common Patterns

### Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SimulationError {
    #[error("Cell capacity exceeded: {0}/{1}")]
    CapacityExceeded(usize, usize),
    
    #[error("Invalid genome ID: {0}")]
    InvalidGenome(usize),
    
    #[error("GPU error: {0}")]
    GpuError(#[from] wgpu::Error),
}

// Use Result types consistently
pub type Result<T> = std::result::Result<T, SimulationError>;
```

### Resource Management

```rust
// RAII pattern for GPU resources
pub struct MyRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
}

impl MyRenderer {
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Recreate size-dependent resources
        self.bind_group = create_bind_group(device, width, height);
    }
}
```

### Configuration

```rust
// Use serde for serializable config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyConfig {
    pub parameter: f32,
    pub enabled: bool,
}

impl Default for MyConfig {
    fn default() -> Self {
        Self {
            parameter: 1.0,
            enabled: true,
        }
    }
}
```

## Troubleshooting

### Build Issues

**Problem**: Compilation errors with wgpu
**Solution**: Ensure you have the latest GPU drivers and Vulkan/DirectX support

**Problem**: Missing dependencies
**Solution**: Run `cargo update` and check Cargo.toml versions

### Runtime Issues

**Problem**: Black screen or no rendering
**Solution**: Check GPU compatibility and wgpu adapter selection

**Problem**: Poor performance
**Solution**: Build in release mode (`cargo build --release`)

**Problem**: Simulation instability
**Solution**: Reduce physics timestep or check for NaN values

### GPU Issues

**Problem**: Compute shader not working
**Solution**: Verify GPU supports compute shaders and check device limits

**Problem**: Rendering artifacts
**Solution**: Use RenderDoc to inspect draw calls and pipeline state

### Memory Issues

**Problem**: High memory usage
**Solution**: Check capacity settings and pre-allocation sizes

**Problem**: Memory leaks
**Solution**: Verify proper resource cleanup and avoid circular references

## Getting Help

1. **Read the code**: Start with the files listed in "Key Files to Understand"
2. **Check logs**: Enable debug logging to see what's happening
3. **Use debugger**: Step through code to understand execution flow
4. **Profile performance**: Use tools to identify bottlenecks
5. **Ask questions**: Reach out to the development team

## Contributing

1. **Follow patterns**: Use existing code patterns and conventions
2. **Add tests**: Include unit tests for new functionality
3. **Document changes**: Update comments and documentation
4. **Performance test**: Verify changes don't regress performance
5. **Code review**: Have changes reviewed before merging
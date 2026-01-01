# Bio-Spheres

A biological cell simulation written in Rust using wgpu/wgsl for GPU-accelerated physics and rendering.

## Features

- GPU-accelerated cell physics simulation
- Cell division and adhesion mechanics
- Genome-based cell behavior
- Real-time 3D rendering with volumetric effects
- Interactive UI with genome editor

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

## Project Structure

- `src/cell/` - Cell types, adhesion, and division logic
- `src/genome/` - Genome representation and node graph
- `src/simulation/` - Physics simulation (CPU and GPU)
- `src/rendering/` - wgpu rendering pipeline
- `src/ui/` - egui-based user interface
- `src/input/` - Input handling
- `shaders/` - WGSL compute and render shaders
- `assets/` - Textures, models, and other resources

## License

See LICENSE file for details.

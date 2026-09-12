# Bio-Spheres

A biological cell simulation written in Rust using wgpu/wgsl for GPU-accelerated physics and rendering.

## Architecture

This project implements two simulation modes:

- **Preview Scene**: CPU-based simulation for genome editing and testing
- **GPU Scene**: Full GPU-accelerated physics and rendering for large-scale simulations

The project uses local forks of egui and egui_dock for the user interface, with all Bevy dependencies removed.

## Features

- GPU-accelerated cell physics simulation
- Cell division and adhesion mechanics
- Genome-based cell behavior
- Real-time 3D rendering with volumetric effects
- Interactive UI with genome editor

## Building

### Linux prerequisites

On Debian, Ubuntu, or Linux Mint, install the native libraries used by the windowing, audio,
and GPU backends before building:

```bash
sudo apt update
sudo apt install pkg-config libasound2-dev libx11-dev libxcursor-dev libxrandr-dev \
	libxi-dev libwayland-dev libxkbcommon-dev libvulkan-dev libudev-dev
```

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
- `egui_local/` - Local fork of egui (not tracked)
- `egui_dock_local/` - Local fork of egui_dock (not tracked)

## License

See LICENSE file for details.

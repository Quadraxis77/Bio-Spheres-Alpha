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

```bash
cargo build --release
```

## Running

```bash
cargo run --release
```

## Native VR

Native VR is included in the default build. Connect your PCVR headset and select
its OpenXR runtime (Virtual Desktop, SteamVR, Meta Link, or another Vulkan-capable
runtime), then launch:

```powershell
cargo run --release --bin bio-spheres -- --vr
```

The headset receives tracked stereo views at its recommended eye resolution.
Head position and orientation are applied directly at the runtime's predicted
display time; the navigation camera acts as the movement rig. The simulation
advances once for both eyes. The desktop window mirrors the left eye.

Point a controller at the floating menu panel and use its trigger to select or
activate the current tool. Hold B (Y on the left Touch controller) to open the
tool wheel, point at a tool, then release. The left stick moves at 1.5 meters per
second relative to your head direction. The right stick turns in 30-degree steps;
return it to the center between turns. Simple, Touch, Index, Vive, and Windows
Motion Controller interaction profiles have bindings. Tracking or focus loss
releases selection and dragging.

VR pacing comes from OpenXR and bypasses the desktop frame limiter. The game
requests 120 Hz when the runtime exposes that rate; otherwise it uses the runtime's
selected rate. Reaching that rate depends on scene complexity and GPU performance.
Desktop rendering defaults to 120 FPS and can be set between 30 and 120 in Settings.

```powershell
# Identify the active runtime and connected headset:
target\release\bio-spheres.exe --vr-info
# Check Vulkan device sharing and native swapchain creation without a game window:
target\release\bio-spheres.exe --vr-check
# Build a desktop executable without OpenXR or the bundled loader:
cargo build --release --no-default-features --bin bio-spheres
```

The OpenXR loader is bundled at build time; building VR requires CMake and the
platform C/C++ toolchain. Lens distortion and reprojection are handled by the
runtime. Desktop depth-of-field and temporal occlusion culling are disabled for
headset views. File dialogs still use the operating system's desktop interface.

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

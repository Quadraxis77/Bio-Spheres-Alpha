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

## Native VR

Native VR is included in the default build. Launch `bio-spheres.exe` normally;
the game detects connected PCVR headsets automatically, including connections
made after startup. For Quest Steam Link, connect Steam Link to the PC so
SteamVR is running, then Run main or launch the EXE normally. Biospheres tries
that running SteamVR connection first, then falls back to the Windows OpenXR
runtime (including Virtual Desktop/VDXR). It does not change the system runtime
setting or launch SteamVR just because it is installed. An explicit
`XR_RUNTIME_JSON` environment override takes priority.
Wearing the headset switches to VR; removing it returns to desktop rendering
without ending the simulation. The VR session stays available for putting it
back on. `XR_EXT_user_presence` supplies wear detection where supported; other
runtimes use session focus/visibility, so switching follows their headset state.

**Launch VR.cmd** explicitly connects to VR at startup. Beside a distributed
`bio-spheres.exe`, it launches that executable. In a source checkout, it first
builds the latest release executable. The launcher checks GPU synchronization
and VR session creation before opening the game. If that check fails, it
displays the error and stops.

For an explicit VR startup check, use the launcher or start with `--vr`.
Use `--no-vr` to keep a session in desktop mode and disable automatic detection:

```powershell
.\target\release\bio-spheres.exe --vr
```

To build and launch from source:

```powershell
cargo run --release --bin bio-spheres -- --vr
```

For Virtual Desktop, connect the headset to your PC through its headset app and
select **VDXR** as the OpenXR runtime in the PC Streamer settings. The
[VDXR setup guide](https://github.com/mbucchia/VirtualDesktop-OpenXR/wiki#setup)
describes this setting and how to verify the runtime in the performance overlay.
If `--vr-info` reports `Headset: Unavailable`, reconnect the headset before
launching native VR.

The main menu and genome editor appear in a fixed stereo panel 1.6 meters in
front of the player. The panel stays upright and does not follow head movement.
Its scene images use separate off-axis views for each eye, providing depth and
motion parallax through the panel. Headset recentering repositions the panel;
returning from the GPU simulation places it in front of the current head position.
VR menus and the wrist wheel are rendered at up to twice their display resolution
to sharpen text and interface edges; the scale is reduced automatically when
needed to respect the GPU's maximum texture size.
The ordinary VR panels also use heavier glyph outlines, stronger lines, and
higher contrast for subdued labels. This does not change the wrist-wheel style.

Menu and slider usage is saved locally in the app's configuration directory as
`control_usage.json`. **Help → Control usage** ranks controls for the last seven
days and all time, with separate desktop, VR screen, and wrist-wheel counts.
Rapid repeats within two seconds count as one use; a held adjustment stays one
use even when you pause during the drag. Switching controls starts a new use.
The history stores aggregate counts, not setting values or raw input events.

The main GPU simulation fills the headset view, with no desktop screen border.
The headset receives tracked stereo views at its recommended eye resolution.
Head position and orientation are applied directly at the runtime's predicted
display time; the navigation camera acts as the movement rig. The simulation
advances once for both eyes. The desktop window does not mirror the headset.
Desktop rendering resumes when OpenXR reports the headset inactive (for example,
when it is unworn), without ending the VR session or simulation.
When first connecting a headset to a desktop-started session, graphics resources
are rebuilt for the runtime's Vulkan device while retaining preview state and a
snapshot of the GPU cells, genomes, and fluid. Wearing/removing the headset after
that switches presentation on the existing device.

The immersive GPU scene starts with desktop panels hidden. Quest 3 controls:

- The **left-hand wheel opens on first VR entry**. Raise your left hand to see
  it. Tap **X** to open/close it again (Y or left-stick click also work), or
  point at the floating **Menu / X** button above the hand and use the right
  trigger. Point with the **right controller** and press its **trigger** to
  select a highlighted sector. Release the trigger before making another choice.
  The wrist menu renders over the left controller; its selection cursor stays
  visible on the buttons.

- **Play/Pause**, **Simulation speed**, and **Reset scene** stay on the home
  wheel, grouped together at the top. Reset opens a scope selector: **Cells only**
  keeps water; **Cells + water** clears both. A fresh trigger press confirms the
  chosen scope. The hub shows the live population; the curved chart records the
  last 60 simulation seconds, scales to the rolling peak, and clears on reset.
- **Scenes** opens Simulation, Genome editor, and Main menu. **World** expands
  into Water, Physics, Lighting, and Biology; select a group to see its controls.
  **Tools** is an outer ring on the main simulation home wheel: Navigate, Insert,
  Inspect, Drag, Boost, and Remove are directly selectable. Selecting a
  tool closes the wheel. **View / screen** contains travel modes, sensitivity,
  VR screen adjustments, reset view, and control help. **All settings** opens
  the normal UI. **Main menu** also has its own button on the home wheel.
  Scene switching retains the existing simulation/editor state; simulation
  controls are dimmed outside the simulation.
- The entire wrist console follows the selected UI theme, including custom
  palettes: translucent panel surfaces, text, borders, category accents,
  population graph, and circular sliders. Deep grooves separate groups;
  shallow seams separate buttons within each group. Buttons depress on a
  trigger press and show a lit indicator for hover or active selection.
- Selecting a numeric setting on the wheel, including simulation speed, opens a
  **dedicated circular slider view**. Other wheel buttons and population displays
  stay hidden until **Back / B** or **Close / X**. Significant values are marked,
  labeled, and gently snap into place; small stick movements still allow fine tuning. Release the selection trigger, then point at the
  ring and hold the **right trigger** while moving around it. The grab stays
  active when the cursor leaves the ring or visible panel; releasing the trigger
  ends it. Missing tracking pauses the grab without changing the value.
  **Right stick right/up increases** the selected slider; **left/down decreases**
  it, with smaller deflections for fine control. The bottom gap separates minimum
  and maximum; dragging past an endpoint clamps the value.
  Release to keep the value, then select **Back** or press **B**. **Close / X**
  closes the wheel.
  **Left-stick travel and left-grip rise/fall remain available while the wheel
  or settings panel is open.** The right stick adjusts the selected slider;
  camera turns and scene-grab gestures are suspended until the menu closes.
  Center the right stick after closing to re-arm turning.
- **B** goes back one wheel page, closes the home wheel, or returns from the
  full settings panel to the home wheel. The wheel center also offers Back/Close.
- **Left stick** moves forward/back and strafes along the gravity ground plane.
  Looking up/down does not change your height. Hold **left grip** to make
  up/down rise/fall along the local gravity axis; left/right still strafes.
  Release the grip to return immediately to ground-plane travel.
- **Right stick left/right** turns around the local gravity up axis in
  **45-degree** steps, fading to black before the turn and back in afterward.
  Center the stick between steps. When pointing at a menu, right-stick
  left/right edits a focused slider and up/down scrolls the menu; up/down does
  not pitch the camera. You look up/down naturally with the headset. Hands and
  the wheel stay visible.
- **Right thumbstick click** takes a screenshot while in-game VR, including with
  the wheel or settings panel open.
- For **radial gravity**, walking follows the sphere around the simulation's
  center, preserving your radius and carrying your orientation around the
  curvature. Left-grip rise/fall changes that radius. Up is outward for inward
  gravity and inward for outward gravity. X/Y/Z gravity uses the corresponding
  flat ground plane. This also works seated.
- **View / screen** offers **Ground travel**, **Grab scene**, **Tune speed**,
  **Reset view**, and **Slow / Normal / Fast** speeds. Tune speed opens a
  page with separate **Travel speed** and **Turn sensitivity** circular sliders.
  B returns from the slider to that page. Speed does not change automatically
  with distance.
- **Grab scene** enables an optional right-grip gesture to pull/push/slide the
  scene, or both grips to rotate and scale it. Sticks still use gravity-relative
  travel and turning; releasing the scene grab returns to a level gravity frame.


- In the **editor preview**, the **right stick** smoothly rotates the scene
  horizontally around the ground-plane up axis, preserving its pitch and
  keeping the horizon level. Right-stick up/down adjusts pitch separately,
  stopping before the poles so the view cannot flip. **Left stick** pans along the
  ground plane; hold **left grip** for up/down instead of forward/back.
  These work with the wheel open. A selected slider takes priority over right
  stick rotation, while left-stick pan remains available. The physical screen
  stays anchored; preview controls preserve the orbit distance and zoom.
- **View / screen → VR screen** adjusts the main menu, editor preview, and
  settings screen. Choose **Curvature** (flat through 110°), **Distance**
  (0.6–4 m), or **Aspect ratio** (1:1–3:1), then use the circular slider:
  hold the right trigger and drag, or move the right thumbstick. Settings apply
  live, carry across scenes, and persist between launches. Changes save after a
  short pause in adjustment and when leaving the slider or removing the headset.
  The screen stays anchored to the seated origin;
  changing its distance does not recenter it on a moving head. Both eye images
  remain separate, and controller pointing follows the curved surface.
- **All settings** opens the normal UI on a stationary panel in front of you.

  **F1** toggles this panel in the simulation. Controller pointing, desktop mouse
  clicks, and keyboard UI navigation work while VR is active. Mouse activity
  temporarily takes pointer priority; a trigger press resumes controller pointing.
  Focusing a numeric input replaces the left-controller wrist menu with a number
  pad showing its current value. Point at it with the right controller and use
  the number, sign, and decimal keys to edit; **Del** removes the last character,
  **Clear** resets it to zero, and **Done** returns to the wrist menu.
  With the wheel closed, use the right trigger to drag a regular slider, or
  select it and use the right thumbstick. Tap **X** to expand the selected slider
  into the circular view while keeping its settings panel visible. Logarithmic
  ranges, integer steps, and the setting's existing limits are preserved.
  Desktop mouse dragging and keyboard adjustments also work normally.
- To inspect a cell, choose **Tools → Inspect**, point at the cell, and press the
  right trigger. Open the wheel with **X**, then choose **Cell details**. Tabs
  organize **Identity**, **Biology**, **Physics**, and all sixteen **Signals**.
  **Load genome into Preview** loads the retained genome into the editor preview.
  A death notice appears when the selected cell dies; its genome and last readings
  stay selected and loadable. Reused GPU slots cannot replace that selection.

The simulation starts outside the sphere, looking inward with +Y up, nearly
level with the Y-gravity ground layer. The selected gravity mode then determines
your travel plane. Removing and wearing the headset does not reset your position.

Main-menu and genome-preview scenes retain their stationary stereoscopic panels.
Both hands use the published Quest 3 **Touch Plus** models and their original
textures/labels, embedded in the EXE. Source and license are in
[assets/vr/touch-plus](assets/vr/touch-plus/README.md). Buttons are rendered in
their resting position. Index, Vive, Windows Motion Controller, Simple, Touch,
Touch Plus, and Touch Pro input profiles are supported; analog navigation needs
sticks and grip inputs.

VR pacing comes from OpenXR and bypasses the desktop frame limiter. The game
requests 120 Hz when the runtime exposes that rate; otherwise it uses the runtime's
selected rate. Reaching that rate depends on scene complexity and GPU performance.
Desktop rendering defaults to 120 FPS and can be set between 30 and 120 in Settings.

```powershell
# Identify the active runtime and connected headset:
target\release\bio-spheres.exe --vr-info
# Check GPU synchronization, Vulkan device sharing, and native swapchain creation:
target\release\bio-spheres.exe --vr-check
# Build a desktop executable without OpenXR or the bundled loader:
cargo build --release --no-default-features --bin bio-spheres
```

The OpenXR loader is bundled at build time; building VR requires CMake and the
platform C/C++ toolchain. Native VR requires Vulkan 1.2 for timeline semaphores.
It requests the highest Vulkan version supported by both the OpenXR runtime and
the local loader, up to Vulkan 1.3. Lens distortion and reprojection are handled
by the runtime. Desktop depth-of-field and temporal occlusion culling are
disabled for headset views. File dialogs still use the operating system's
desktop interface.

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

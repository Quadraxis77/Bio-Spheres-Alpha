// Generated from the user-reviewed menu checklist; Hide maps to Developer (F11).
use super::Visibility::{Advanced as A, Developer as D, Show as S};

pub(super) const CHOICES: [super::Visibility; 402] = [
    S, // 0001 Top bar and general menus / Scene switch
    S, // 0002 Top bar and general menus / Tutorial
    S, // 0003 Top bar and general menus / Help
    S, // 0004 Top bar and general menus / Themes
    S, // 0005 Top bar and general menus / Panels
    S, // 0006 Top bar and general menus / Save Sphere…
    S, // 0007 Top bar and general menus / Load Sphere…
    S, // 0008 Top bar and general menus / Save genome
    S, // 0009 Top bar and general menus / Load genome
    S, // 0010 Top bar and general menus / New genome
    S, // 0011 Top bar and general menus / Genome name
    S, // 0012 Top bar and general menus / UI Scale
    S, // 0013 Top bar and general menus / Frame rate limit
    S, // 0014 Top bar and general menus / Horizontal FOV
    A, // 0015 Top bar and general menus / Speed 1
    A, // 0016 Top bar and general menus / Speed 2
    A, // 0017 Top bar and general menus / Scroll Sensitivity
    A, // 0018 Top bar and general menus / Panel visibility toggles
    A, // 0019 Top bar and general menus / Panel Lock/Unlock
    S, // 0020 Top bar and general menus / Reset to Default (layout)
    D, // 0021 Top bar and general menus / Layout Files: Active / Default
    S, // 0022 Left button rail / Generate Procedural Genome
    S, // 0023 Left button rail / Take Screenshot
    S, // 0024 Left button rail / Open Bio-Spheres Folder
    S, // 0025 Left button rail / Advanced Options
    S, // 0026 Left button rail / Lineage Viewer
    S, // 0027 Left button rail / Toggle Water Fill
    S, // 0028 Left button rail / Static Water World (fill world, disable dynamic water physics)
    S, // 0029 Left button rail / Flat Ground (fill lower third with sandy ocean floor)
    S, // 0030 Left button rail / Reset scene settings
    S, // 0031 Panels and windows / Viewport
    S, // 0032 Panels and windows / Scene Manager
    S, // 0033 Panels and windows / Cell Inspector
    S, // 0034 Panels and windows / Lineage Viewer
    S, // 0035 Panels and windows / Genome Editor
    A, // 0036 Panels and windows / Performance
    A, // 0037 Panels and windows / Rendering
    S, // 0038 Panels and windows / Time Scrubber
    A, // 0039 Panels and windows / Theme Editor
    S, // 0040 Panels and windows / Camera
    S, // 0041 Panels and windows / Lighting
    S, // 0042 Panels and windows / Gizmo Rendering
    S, // 0043 Panels and windows / Cell Visuals
    S, // 0044 Panels and windows / Cave System
    S, // 0045 Panels and windows / Fluid System
    S, // 0046 Panels and windows / World Settings
    S, // 0047 Panels and windows / Modes
    S, // 0048 Panels and windows / Adhesion Settings
    S, // 0049 Panels and windows / Parent Settings
    S, // 0050 Panels and windows / Circle Sliders
    S, // 0051 Panels and windows / Child Rotation
    S, // 0052 Panels and windows / Time Slider
    A, // 0053 Panels and windows / Mode Graph
    S, // 0054 Panels and windows / Help
    D, // 0055 Panels and windows / Left Panel
    D, // 0056 Panels and windows / Right Panel
    D, // 0057 Panels and windows / Bottom Panel
    S, // 0058 Cave System / 🏔️ Generate Cave System
    S, // 0059 Cave System / Density:
    A, // 0060 Cave System / Scale:
    D, // 0061 Cave System / Octaves:
    D, // 0062 Cave System / Smoothness:
    D, // 0063 Cave System / Resolution:
    D, // 0064 Cave System / Fragment Cull:
    D, // 0065 Cave System / Mesh Smoothing Iterations:
    D, // 0066 Cave System / Mesh Smoothing Factor:
    D, // 0067 Cave System / Smooth Normals
    S, // 0068 Cave System / Seed:
    S, // 0069 Cave System / Preset:
    D, // 0070 Cave System / Basalt: / Dark:
    D, // 0071 Cave System / Glass: / Cool:
    D, // 0072 Cave System / Iron: / Warm:
    D, // 0073 Cave System / Ash: / Pale:
    D, // 0074 Cave System / Flow Scale: / Layer Scale:
    D, // 0075 Cave System / Warp:
    D, // 0076 Cave System / Plate Variation: / Fine Bands:
    D, // 0077 Cave System / Blistering: / Cool Mottle:
    D, // 0078 Cave System / Glass Grain: / Grain:
    D, // 0079 Cave System / Ash Deposits: / Patch Contrast:
    D, // 0080 Cave System / Seam Darkness: / Seam Darkening:
    D, // 0081 Cave System / Shelf Scuffs: / Wall Lines:
    D, // 0082 Cave System / Texture Scale:
    D, // 0083 Cave System / Conform:
    D, // 0084 Cave System / Relief:
    D, // 0085 Cave System / Plate Size: / Major Lines:
    D, // 0086 Cave System / Surface Breakup: / Fine Lines:
    D, // 0087 Cave System / Crack Density: / Seam Lines:
    D, // 0088 Cave System / Flow Noise Scale: / Fine Warp Scale:
    D, // 0089 Cave System / Flow Warp: / Fine Warp:
    D, // 0090 Cave System / Crack Noise Scale: / Seam Warp Scale:
    D, // 0091 Cave System / Crack Warp: / Seam Warp:
    D, // 0092 Cave System / Plate Low: / Major Low:
    D, // 0093 Cave System / Plate High: / Major High:
    D, // 0094 Cave System / Blister Low: / Fine Low:
    D, // 0095 Cave System / Blister High: / Fine High:
    D, // 0096 Cave System / Crack Thin: / Seam Low:
    D, // 0097 Cave System / Crack Thick: / Seam High:
    D, // 0098 Cave System / Ambient:
    D, // 0099 Cave System / Diffuse:
    D, // 0100 Cave System / Specular:
    D, // 0101 Cave System / Spec Power:
    D, // 0102 Cave System / Min Tone:
    D, // 0103 Cave System / Max Tone:
    S, // 0104 Cave System / Enable Vents
    S, // 0105 Cave System / Placement:
    S, // 0106 Cave System / Lower Hemisphere Only
    S, // 0107 Cave System / Frequency:
    A, // 0108 Cave System / Length:
    A, // 0109 Cave System / Width:
    A, // 0110 Cave System / Depth:
    A, // 0111 Cave System / Back Margin:
    A, // 0112 Cave System / Top Margin:
    A, // 0113 Cave System / Heat Output:
    A, // 0114 Cave System / Heat Radius:
    A, // 0115 Cave System / Color:
    A, // 0116 Cave System / Glow Strength:
    A, // 0117 Cave System / Glow Radius:
    A, // 0118 Cave System / Enable Moss
    A, // 0119 Cave System / Growth Rate:
    A, // 0120 Cave System / Min Light Level:
    A, // 0121 Cave System / Water Radius (voxels):
    A, // 0122 Cave System / Decay Rate:
    A, // 0123 Cave System / Water Erosion Rate:
    A, // 0124 Cave System / Wetness Evaporation:
    A, // 0125 Cave System / Graze Cooldown:
    A, // 0126 Cave System / Nutrients per Moss:
    A, // 0127 Cave System / Texture Scale:
    A, // 0128 Cave System / Parallax Depth:
    A, // 0129 Cave System / Noise Type:
    A, // 0130 Cave System / Noise Frequency:
    A, // 0131 Cave System / Noise Lacunarity:
    A, // 0132 Cave System / Height Sharpness (low):
    A, // 0133 Cave System / Height Sharpness (high):
    A, // 0134 Cave System / Bump Strength:
    A, // 0135 Cave System / Dark (base):
    A, // 0136 Cave System / Bright (tips):
    S, // 0137 Cave System / Enable Mossrocks
    S, // 0138 Cave System / Target Count:
    A, // 0139 Cave System / Approx. Spawn Interval (seconds):
    A, // 0140 Cave System / Radius Range: Min
    A, // 0141 Cave System / Radius Range: Max
    A, // 0142 Cave System / Moss Store Range (nutrients): Min
    A, // 0143 Cave System / Moss Store Range (nutrients): Max
    A, // 0144 Cave System / Buoyancy in Water:
    A, // 0145 Cave System / Size Gate (half-saturation):
    S, // 0146 Fluid System / Density:
    A, // 0147 Fluid System / Epoch Duration:
    A, // 0148 Fluid System / Epoch Spacing:
    A, // 0149 Fluid System / Spawn Ramp:
    A, // 0150 Fluid System / Despawn Ramp:
    A, // 0151 Fluid System / Surface Pressure:
    A, // 0152 Fluid System / Humidity Diffusion Rate:
    A, // 0153 Fluid System / Freeze Rate:
    A, // 0154 Fluid System / Melt Rate:
    A, // 0155 Fluid System / Snow Melt Rate:
    A, // 0156 Fluid System / Snow Compaction Rate:
    A, // 0157 Fluid System / Lateral Flow Probability:
    D, // 0158 Fluid System / Facet Size:
    D, // 0159 Fluid System / Facet Depth:
    D, // 0160 Fluid System / Facet Shading:
    D, // 0161 Fluid System / Glint Intensity:
    D, // 0162 Fluid System / Opacity:
    D, // 0163 Fluid System / Surface Color
    D, // 0164 Fluid System / Deep Color
    D, // 0165 Fluid System / Glint Sharpness:
    D, // 0166 Fluid System / Reflection:
    D, // 0167 Fluid System / Reflection Brightness:
    D, // 0168 Fluid System / Ambient:
    D, // 0169 Fluid System / Diffuse:
    D, // 0170 Fluid System / Specular:
    D, // 0171 Fluid System / Shininess:
    D, // 0172 Fluid System / Rim Light:
    D, // 0173 Fluid System / Alpha (Transparency):
    D, // 0174 Fluid System / Fresnel Strength:
    D, // 0175 Fluid System / Fresnel Power:
    D, // 0176 Fluid System / Reflection Intensity:
    D, // 0177 Fluid System / Reflection Brightness:
    D, // 0178 Fluid System / Below-Waterline Opacity:
    D, // 0179 Fluid System / Wave Height:
    D, // 0180 Fluid System / Wave Speed:
    D, // 0181 Fluid System / Noise Scale (lower = larger waves):
    D, // 0182 Fluid System / Octaves:
    D, // 0183 Fluid System / Lacunarity (freq per octave):
    D, // 0184 Fluid System / Persistence (amp per octave):
    A, // 0185 Fluid System / Sunlight Attenuation:
    A, // 0186 Fluid System / Ambient Spread:
    A, // 0187 Fluid System / Shadow Depth:
    A, // 0188 Fluid System / Intensity:
    A, // 0189 Fluid System / Scale:
    A, // 0190 Fluid System / Speed:
    S, // 0191 Fluid System / Water Fill: ON / OFF
    A, // 0192 Fluid System / Static Water World: ON / OFF
    D, // 0193 Organism Skins / Show Organism Skins
    D, // 0194 Organism Skins / Skin Offset (gap from cell surface, world units):
    D, // 0195 Organism Skins / Shrink Speed (fraction of gap closed per iteration):
    D, // 0196 Organism Skins / Smooth Factor (Laplacian blend per iteration):
    D, // 0197 Organism Skins / Shrink Iterations:
    D, // 0198 Organism Skins / Smooth Iterations:
    D, // 0199 Organism Skins / Min Cells for Skin:
    D, // 0200 Organism Skins / Skin Colour:
    D, // 0201 Organism Skins / Opacity:
    D, // 0202 Organism Skins / Subsurface Scattering:
    D, // 0203 Organism Skins / Rim Light:
    D, // 0204 Organism Skins / Cell Membrane
    D, // 0205 Organism Skins / Bioluminescent
    D, // 0206 Organism Skins / Opaque Shell
    D, // 0207 Lighting / Sun Disc
    S, // 0208 Lighting / Orbit
    S, // 0209 Lighting / Show Ring
    S, // 0210 Lighting / Brightness (0-5)
    A, // 0211 Lighting / Tint
    S, // 0212 Lighting / Position
    S, // 0213 Lighting / Speed
    S, // 0214 Lighting / Path Tilt
    A, // 0215 Lighting / Path Yaw
    D, // 0216 Lighting / Light Direction
    D, // 0217 Lighting / Disc Size
    S, // 0218 Lighting / Night Ratio (0-1)
    S, // 0219 Lighting / Cycle Length
    A, // 0220 Lighting / Thermal Inertia (0-5)
    S, // 0221 Lighting / Seasonal Cycle
    S, // 0222 Lighting / Season Min
    S, // 0223 Lighting / Season Max
    A, // 0224 Lighting / Hardware Ray Tracing
    D, // 0225 Lighting / Surface Shadows
    D, // 0226 Lighting / Strength
    D, // 0227 Lighting / Quality
    D, // 0228 Lighting / Ray Steps
    D, // 0229 Lighting / Step Size
    D, // 0230 Lighting / Rock Absorption
    D, // 0231 Lighting / Cell Absorption
    D, // 0232 Lighting / Ambient Floor
    D, // 0233 Lighting / Enabled
    D, // 0234 Lighting / Density
    D, // 0235 Lighting / Glow
    D, // 0236 Lighting / Directionality
    D, // 0237 Lighting / Absorption
    D, // 0238 Lighting / Ambient Color
    D, // 0239 Lighting / Bottom Density
    D, // 0240 Lighting / Falloff
    D, // 0241 Lighting / Steps
    D, // 0242 Lighting / Smooth Voxels
    D, // 0243 Lighting / Blur
    D, // 0244 Lighting / Strength
    D, // 0245 Lighting / Scale
    A, // 0246 Lighting / Enabled
    A, // 0247 Lighting / Halo Radius
    A, // 0248 Lighting / Depth of Field
    D, // 0249 Lighting / Contrast
    D, // 0250 Lighting / Eye Adaptation
    D, // 0251 Lighting / Adapt Speed
    D, // 0252 Lighting / Min Exposure
    D, // 0253 Lighting / Max Exposure
    D, // 0254 Lighting / Focal Distance
    D, // 0255 Lighting / Focal Range
    D, // 0256 Lighting / Max Blur
    D, // 0257 Lighting / Blur Strength
    D, // 0258 Lighting / Max Steps / Frame:
    D, // 0259 Lighting / Light Field Interval: Every /2 /4 /8
    D, // 0260 Lighting / Physics Rate: 32 / 48 / 64 Hz
    S, // 0261 World Settings / World Radius:
    S, // 0262 World Settings / Cell Capacity:
    S, // 0263 World Settings / Gravity:
    D, // 0264 World Settings / Velocity Damping:
    D, // 0265 World Settings / Constraint Iterations:
    D, // 0266 World Settings / Water Viscosity:
    S, // 0267 World Settings / Gravity Axis: X / Y / Z / Radial
    D, // 0268 World Settings / Solo cell metabolism penalty
    D, // 0269 World Settings / Metabolism Multiplier:
    S, // 0270 World Settings / Radiation Level:
    D, // 0271 World Settings / Subtle mutations
    S, // 0272 World Settings / All
    S, // 0273 World Settings / Clear
    S, // 0274 World Settings / Gene Pool: cell type selection
    D, // 0275 Camera / Base Movement Speed:
    D, // 0276 Camera / Speed 1:
    D, // 0277 Camera / Speed 2:
    D, // 0278 Camera / Zoom Speed:
    D, // 0279 Camera / Mouse Sensitivity:
    D, // 0280 Camera / Enable Spring
    S, // 0281 Cell Visuals / Outline Width:
    S, // 0282 Cell Visuals / Cell Type:
    S, // 0283 Cell Visuals / Specular Strength:
    S, // 0284 Cell Visuals / Specular Sharpness:
    S, // 0285 Cell Visuals / Rim Lighting:
    S, // 0286 Cell Visuals / Matte
    S, // 0287 Cell Visuals / Glossy
    S, // 0288 Cell Visuals / Shiny
    S, // 0289 Cell Visuals / Tail Length:
    S, // 0290 Cell Visuals / Tail Thickness:
    S, // 0291 Cell Visuals / Wave Amplitude:
    S, // 0292 Cell Visuals / Wave Frequency:
    S, // 0293 Cell Visuals / Taper:
    S, // 0294 Cell Visuals / Segments:
    S, // 0295 Cell Visuals / Cell Scale:
    S, // 0296 Cell Visuals / Border Width:
    S, // 0297 Cell Visuals / Meander:
    S, // 0298 Cell Visuals / Border Darkness:
    S, // 0299 Cell Visuals / Anim Speed:
    S, // 0300 Cell Visuals / Pattern Scale:
    S, // 0301 Cell Visuals / Pattern Strength:
    S, // 0302 Cell Visuals / Nucleus Scale:
    S, // 0303 Cell Visuals / Subdivision:
    S, // 0304 Cell Visuals / Ridge Width:
    S, // 0305 Cell Visuals / Meander:
    S, // 0306 Cell Visuals / Ridge Depth:
    S, // 0307 Cell Visuals / Ring Frequency:
    S, // 0308 Cell Visuals / Ring Depth:
    S, // 0309 Cell Visuals / Scroll Speed:
    S, // 0310 Cell Visuals / Ring Frequency:
    S, // 0311 Cell Visuals / Ring Sharpness:
    S, // 0312 Cell Visuals / Ring Brightness:
    S, // 0313 Cell Visuals / Nucleus Radius:
    S, // 0314 Cell Visuals / Nucleus Darkness:
    S, // 0315 Cell Visuals / Edge Softness:
    S, // 0316 Cell Visuals / Droplet Scale:
    S, // 0317 Cell Visuals / Droplet Threshold:
    S, // 0318 Cell Visuals / Boundary Sharpness:
    S, // 0319 Cell Visuals / Brightness:
    S, // 0320 Cell Visuals / Bubble Size:
    S, // 0321 Cell Visuals / Rotation Speed:
    S, // 0322 Cell Visuals / Membrane Brightness:
    S, // 0323 Cell Visuals / Gas Brightness:
    S, // 0324 Cell Visuals / Spike Length:
    S, // 0325 Cell Visuals / Spike Sharpness:
    S, // 0326 Cell Visuals / Embed Depth:
    S, // 0327 Cell Visuals / Tip Fade:
    S, // 0328 Cell Visuals / Fiber Count:
    S, // 0329 Cell Visuals / Bulge Strength:
    S, // 0330 Cell Visuals / Fiber Warp:
    S, // 0331 Cell Visuals / Yolk Radius:
    S, // 0332 Cell Visuals / Yolk Drop:
    S, // 0333 Cell Visuals / Yolk Brightness:
    S, // 0334 Cell Visuals / Pupil Size:
    S, // 0335 Cell Visuals / Iris Detail:
    S, // 0336 Cell Visuals / Iris Texture:
    S, // 0337 Cell Visuals / Pupil Darkness:
    S, // 0338 Cell Visuals / Cell Scale:
    S, // 0339 Cell Visuals / Border Width:
    S, // 0340 Cell Visuals / Meander:
    S, // 0341 Cell Visuals / Border Depth:
    S, // 0342 Cell Visuals / Crater Radius:
    S, // 0343 Cell Visuals / Throat Darkness:
    S, // 0344 Cell Visuals / Rim Brightness:
    S, // 0345 Cell Visuals / Nozzle Height:
    S, // 0346 Cell Visuals / Embed Depth:
    S, // 0347 Cell Visuals / Feather Length:
    S, // 0348 Cell Visuals / Feather Width:
    S, // 0349 Cell Visuals / Feather Brightness:
    S, // 0350 Cell Visuals / Stroke Speed:
    S, // 0351 Cell Visuals / Core Radius:
    S, // 0352 Cell Visuals / Daughter Bud Size:
    S, // 0353 Cell Visuals / Branch Brightness:
    S, // 0354 Cell Visuals / Developmental Pulse:
    S, // 0355 Cell Visuals / Save
    S, // 0356 Cell Visuals / Reset Type
    S, // 0357 Cell Visuals / Reset All
    S, // 0358 Theme Editor / Active Theme
    S, // 0359 Theme Editor / Copy current theme to Custom
    S, // 0360 Theme Editor / Darkest Background
    S, // 0361 Theme Editor / Panel Background
    S, // 0362 Theme Editor / Widget Background
    S, // 0363 Theme Editor / Hover Background
    S, // 0364 Theme Editor / Active Background
    S, // 0365 Theme Editor / Selected Background
    S, // 0366 Theme Editor / Primary Accent
    S, // 0367 Theme Editor / Secondary Accent
    S, // 0368 Theme Editor / Primary Text
    S, // 0369 Theme Editor / Secondary Text
    S, // 0370 Theme Editor / Dim Text
    S, // 0371 Theme Editor / Subtle Border
    S, // 0372 Theme Editor / Normal Border
    S, // 0373 Theme Editor / Bright Border
    S, // 0374 Theme Editor / Top Bar Background
    S, // 0375 Theme Editor / Top Bar Border
    S, // 0376 Theme Editor / Status OK
    S, // 0377 Theme Editor / Status Warn
    S, // 0378 Theme Editor / Status Error
    S, // 0379 Theme Editor / Status Info
    S, // 0380 Theme Editor / Reset to Defaults
    S, // 0381 Gizmo Rendering / Show Orientation Lines
    S, // 0382 Gizmo Rendering / Show Split Rings
    A, // 0383 Performance / GPU Telemetry
    A, // 0384 Performance / GPU Frame Timing
    D, // 0385 Performance / Frustum
    D, // 0386 Performance / Occlusion
    D, // 0387 Performance / Scale
    D, // 0388 Performance / Full Detail
    D, // 0389 Performance / Debug Colors
    A, // 0390 Lineage Viewer — Headless Simulation / Simulation speed
    A, // 0391 Lineage Viewer — Headless Simulation / Adaptive
    A, // 0392 Lineage Viewer — Headless Simulation / Fast Forward
    A, // 0393 Lineage Viewer — Headless Simulation / Auto target FPS
    A, // 0394 Lineage Viewer — Headless Simulation / Auto bounds
    A, // 0395 Lineage Viewer — Headless Simulation / GPU telemetry
    S, // 0396 Genome Editor — authoring panels / Modes: cell type, color, mode name and mode selection
    S, // 0397 Genome Editor — authoring panels / Adhesion Settings: breaking, spring and connection properties
    S, // 0398 Genome Editor — authoring panels / Parent Settings: cell functions, signaling and division rules
    S, // 0399 Genome Editor — authoring panels / Circle Sliders: split angles
    S, // 0400 Genome Editor — authoring panels / Child Rotation: quaternion/orientation controls
    S, // 0401 Genome Editor — authoring panels / Time Slider: preview time and resimulation progress
    S, // 0402 Genome Editor — authoring panels / Mode Graph: nodes and mode transitions
];

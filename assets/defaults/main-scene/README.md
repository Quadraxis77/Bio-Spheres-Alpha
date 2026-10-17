This preset captures the developer's saved main-scene configuration on 2026-10-17: cave seed 6064 and sun brightness 3.0.

The cave, fluid, water-rendering, lighting, and cell-appearance files are embedded fallbacks, independent of each player's saved settings. The UI and GPU dock snapshots are mirrored in the root `default_ui_state.ron` and `default_dock_state_gpu.ron` files. Preview layout and onboarding progress are excluded from the personal snapshot; new players still receive the introductions.

Scene-specific reset restores these tuning values, including cave generation and seed, without restarting the simulation or clearing its population or fluid buffers. Changes to cave geometry apply in the existing world. Shared preferences remain separate, and world size/capacity controls take effect through the existing explicit simulation-reset flow.

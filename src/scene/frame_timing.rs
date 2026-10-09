//! Clocks shared by frame-driven effects and the fixed-rate fluid solver.

pub(super) const FLUID_TIMESTEP: f64 = 1.0 / 60.0;

#[derive(Default)]
pub(super) struct FluidMeshClock {
    tick: u64,
    initialized: bool,
}

#[derive(Default)]
pub(super) struct MeshUpdates {
    pub prepare_water: bool,
    pub finalize_water: bool,
    pub prepare_ice: bool,
    pub finalize_ice: bool,
}

impl FluidMeshClock {
    pub fn take(&mut self, fluid_steps: u32) -> MeshUpdates {
        let initial = !self.initialized;
        self.initialized = true;
        let start = self.tick;
        self.tick += u64::from(fluid_steps);
        // Coalesce catch-up ticks into one rebuild using the newest density.
        let due = |period: u64, phase: u64| {
            let first = start + 1;
            let offset = (phase + period - first % period) % period;
            first + offset <= self.tick
        };
        MeshUpdates {
            prepare_water: initial || due(2, 0),
            finalize_water: initial || due(2, 1),
            prepare_ice: initial || due(8, 0),
            finalize_ice: initial || due(8, 1),
        }
    }
}

#[derive(Default)]
pub(super) struct SceneFrameClock {
    wall_elapsed: f64,
    simulation_elapsed: f64,
    fluid_remainder: f64,
}

#[derive(Default)]
pub(super) struct FrameTimes {
    pub wall_seconds: f32,
    pub simulation_seconds: f32,
    pub fluid_steps: u32,
}

impl SceneFrameClock {
    pub fn record(&mut self, dt: f32, paused: bool, time_scale: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        self.wall_elapsed += f64::from(dt);
        if !paused && time_scale.is_finite() && time_scale > 0.0 {
            self.simulation_elapsed += f64::from(dt) * f64::from(time_scale);
        }
    }

    /// Consume each update once, even when the same scene is rendered again.
    /// Fluid retains its historical wall-time cadence, independent of cell time scale.
    pub fn take(&mut self, fluid_active: bool) -> FrameTimes {
        let wall_seconds = std::mem::take(&mut self.wall_elapsed);
        let simulation_seconds = std::mem::take(&mut self.simulation_elapsed);
        let fluid_steps = if fluid_active {
            self.fluid_remainder += wall_seconds;
            // dt enters as f32; tolerate sub-microsecond rounding at tick boundaries.
            let steps = ((self.fluid_remainder + 1e-7) / FLUID_TIMESTEP).floor() as u32;
            self.fluid_remainder =
                (self.fluid_remainder - f64::from(steps) * FLUID_TIMESTEP).max(0.0);
            steps
        } else {
            // A paused/absent solver must not catch up on time spent inactive.
            self.fluid_remainder = 0.0;
            0
        };
        FrameTimes {
            wall_seconds: wall_seconds as f32,
            simulation_seconds: simulation_seconds as f32,
            fluid_steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_and_ice_mesh_work_does_not_scale_with_render_frequency() {
        for fps in [30, 60, 90, 120, 144] {
            let mut frame_clock = SceneFrameClock::default();
            let mut mesh_clock = FluidMeshClock::default();
            let mut water = 0;
            let mut ice = 0;
            for _ in 0..fps * 8 {
                frame_clock.record(1.0 / fps as f32, false, 1.0);
                let mesh = mesh_clock.take(frame_clock.take(true).fluid_steps);
                water += u32::from(mesh.finalize_water);
                ice += u32::from(mesh.finalize_ice);
                // A second eye or an idle render must reuse the prepared mesh.
                let idle = mesh_clock.take(0);
                assert!(
                    !idle.prepare_water
                        && !idle.finalize_water
                        && !idle.prepare_ice
                        && !idle.finalize_ice
                );
            }
            assert!(
                (240..=241).contains(&water),
                "water rebuilds at {fps} FPS: {water}"
            );
            assert!((60..=61).contains(&ice), "ice rebuilds at {fps} FPS: {ice}");
        }
    }

    #[test]
    fn clocks_advance_equally_at_different_render_rates() {
        for fps in [30, 60, 72, 90, 120, 144] {
            let mut clock = SceneFrameClock::default();
            let mut ticks = 0;
            let mut wall = 0.0;
            let mut simulation = 0.0;
            for _ in 0..fps * 10 {
                clock.record(1.0 / fps as f32, false, 2.0);
                let frame = clock.take(true);
                ticks += frame.fluid_steps;
                wall += f64::from(frame.wall_seconds);
                simulation += f64::from(frame.simulation_seconds);
            }
            assert_eq!(ticks, 600, "render rate: {fps}");
            assert!((wall - 10.0).abs() < 1e-5);
            assert!((simulation - 20.0).abs() < 1e-5);
        }
    }

    #[test]
    fn rendering_again_does_not_advance_clocks() {
        let mut clock = SceneFrameClock::default();
        clock.record(1.0 / 60.0, false, 1.0);
        assert_eq!(clock.take(true).fluid_steps, 1);
        let second = clock.take(true);
        assert_eq!(second.wall_seconds, 0.0);
        assert_eq!(second.simulation_seconds, 0.0);
        assert_eq!(second.fluid_steps, 0);
    }

    #[test]
    fn paused_time_is_not_replayed_on_resume() {
        let mut clock = SceneFrameClock::default();
        clock.record(1.0 / 120.0, false, 1.0);
        assert_eq!(clock.take(true).fluid_steps, 0);
        clock.record(1.0, true, 1.0);
        let paused = clock.take(false);
        assert_eq!(paused.wall_seconds, 1.0);
        assert_eq!(paused.simulation_seconds, 0.0);
        assert_eq!(paused.fluid_steps, 0);
        clock.record(1.0 / 120.0, false, 1.0);
        assert_eq!(clock.take(true).fluid_steps, 0);
        clock.record(1.0 / 120.0, false, 1.0);
        assert_eq!(clock.take(true).fluid_steps, 1);
    }

    #[test]
    fn fluid_catches_up_after_a_slow_frame() {
        let mut clock = SceneFrameClock::default();
        clock.record(0.1, false, 1.0);
        assert_eq!(clock.take(true).fluid_steps, 6);
        assert_eq!(clock.take(true).fluid_steps, 0);
    }

    #[test]
    fn jitter_and_deferred_rendering_preserve_elapsed_time() {
        let mut clock = SceneFrameClock::default();
        let mut ticks = 0;
        for _ in 0..100 {
            clock.record(0.003, false, 1.0);
            clock.record(0.007, false, 1.0);
            clock.record(0.015, false, 1.0);
            ticks += clock.take(true).fluid_steps;
        }
        assert_eq!(ticks, 150);
    }
}

//! Confirm water crossings before triggering the full-screen lens effect.

const ENTER_FRACTION: f32 = 0.8;
const EXIT_FRACTION: f32 = 0.2;
const CONFIRM_SECONDS: f32 = 0.15;

#[derive(Default)]
pub(super) struct WaterTransition {
    underwater: Option<bool>,
    pending: Option<bool>,
    pending_seconds: f32,
}

impl WaterTransition {
    /// Returns true once per confirmed crossing. The ambiguous waterline band
    /// retains the last side and cancels short-lived waves/readback noise.
    pub fn update(&mut self, fraction: f32, dt: f32) -> bool {
        let next = if fraction >= ENTER_FRACTION {
            Some(true)
        } else if fraction <= EXIT_FRACTION {
            Some(false)
        } else {
            None
        };
        let Some(next) = next else {
            self.pending = None;
            self.pending_seconds = 0.0;
            return false;
        };
        // Establish the initial environment without flashing on scene entry.
        let Some(underwater) = self.underwater else {
            self.underwater = Some(next);
            return false;
        };
        if next == underwater {
            self.pending = None;
            self.pending_seconds = 0.0;
            return false;
        }
        if self.pending != Some(next) {
            self.pending = Some(next);
            self.pending_seconds = 0.0;
            return false;
        }
        // A long stall is not evidence of a sustained crossing; require more
        // than one new observation before committing the opposite side.
        self.pending_seconds += dt.clamp(0.0, 0.1);
        if self.pending_seconds < CONFIRM_SECONDS {
            return false;
        }
        self.underwater = Some(next);
        self.pending = None;
        self.pending_seconds = 0.0;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waterline_jitter_and_short_waves_do_not_retrigger_transitions() {
        for hz in [30.0f32, 60.0, 144.0] {
            for submerged in [false, true] {
                let mut transition = WaterTransition::default();
                let dt = 1.0 / hz;
                assert!(!transition.update(if submerged { 1.0 } else { 0.0 }, dt));
                for _ in 0..100 {
                    // Mid-surface averages, single-sample occupancy flips,
                    // and short fully wet/dry wave bursts all occur at a waterline.
                    for fraction in [0.45, 0.55, 0.0, 1.0, 0.5] {
                        assert!(!transition.update(fraction, dt));
                    }
                    for _ in 0..(0.08 / dt).ceil() as u32 {
                        assert!(!transition.update(if submerged { 0.0 } else { 1.0 }, dt));
                    }
                    assert!(!transition.update(0.5, dt));
                }
            }
        }
    }

    #[test]
    fn sustained_entry_and_exit_each_fire_once_at_different_frame_rates() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut transition = WaterTransition::default();
            let dt = 1.0 / hz;
            assert!(!transition.update(0.0, dt));
            for fraction in [1.0, 0.0] {
                let mut flashes = 0;
                let mut crossed_after = 0.0;
                for frame in 0..hz as u32 {
                    if transition.update(fraction, dt) {
                        flashes += 1;
                        crossed_after = frame as f32 * dt;
                    }
                }
                assert_eq!(flashes, 1);
                assert!(crossed_after >= CONFIRM_SECONDS);
                assert!(crossed_after <= CONFIRM_SECONDS + dt * 2.0);
            }
        }
    }

    #[test]
    fn initial_submersion_and_long_frame_stalls_do_not_flash() {
        let mut transition = WaterTransition::default();
        assert!(!transition.update(1.0, 1.0 / 60.0));
        assert!(!transition.update(0.0, 5.0));
        assert!(!transition.update(0.0, 5.0));
        assert!(!transition.update(1.0, 1.0 / 60.0));
    }
}

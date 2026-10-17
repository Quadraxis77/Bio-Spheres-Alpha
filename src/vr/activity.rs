//! Keep desktop presentation available when the runtime pauses headset frames.

use std::time::{Duration, Instant};

// Allow ordinary frame waits and brief stalls without switching views. Some
// runtimes keep reporting focus/presence after the headset goes to sleep.
const FRAME_IDLE_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Default)]
pub(super) struct FrameActivity {
    last_renderable: Option<Instant>,
}

impl FrameActivity {
    pub fn observe(&mut self, should_render: bool, now: Instant) {
        self.last_renderable = should_render.then_some(now);
    }

    pub fn active(&self, now: Instant) -> bool {
        self.last_renderable
            .is_some_and(|last| now.saturating_duration_since(last) < FRAME_IDLE_TIMEOUT)
    }

    pub fn reset(&mut self) {
        self.last_renderable = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_frame_switches_to_desktop_until_rendering_resumes() {
        let now = Instant::now();
        let mut activity = FrameActivity::default();
        assert!(!activity.active(now));
        activity.observe(true, now);
        assert!(activity.active(now));
        activity.observe(false, now + Duration::from_millis(10));
        assert!(!activity.active(now + Duration::from_millis(10)));
        activity.observe(true, now + Duration::from_millis(20));
        assert!(activity.active(now + Duration::from_millis(20)));
    }

    #[test]
    fn normal_waits_retain_vr_but_sleeping_waits_release_desktop() {
        let now = Instant::now();
        let mut activity = FrameActivity::default();
        activity.observe(true, now);
        assert!(activity.active(now + Duration::from_millis(30)));
        assert!(!activity.active(now + FRAME_IDLE_TIMEOUT));
        assert!(!activity.active(now + Duration::from_secs(1)));
        activity.observe(true, now + Duration::from_secs(1));
        assert!(activity.active(now + Duration::from_secs(1)));
    }

    #[test]
    fn session_restart_requires_a_new_renderable_frame() {
        let now = Instant::now();
        let mut activity = FrameActivity::default();
        activity.observe(true, now);
        activity.reset();
        assert!(!activity.active(now));
        activity.observe(true, now + Duration::from_millis(10));
        assert!(activity.active(now + Duration::from_millis(10)));
    }
}

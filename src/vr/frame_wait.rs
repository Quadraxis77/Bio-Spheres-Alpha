//! Keep an idle/sleeping headset's blocking frame wait off the desktop UI thread.

use super::VrResult;
use openxr as xr;
use std::sync::mpsc;

pub(super) struct FrameWait {
    request: Option<mpsc::Sender<u64>>,
    result: mpsc::Receiver<(u64, VrResult<xr::FrameState>)>,
    pending: bool,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl FrameWait {
    pub fn new(waiter: xr::FrameWaiter, queue: wgpu::Queue, device: wgpu::Device) -> Self {
        // Retain the graphics device until the waiter and its session are gone.
        let mut owner = (waiter, queue, device);
        fn wait_owned(
            owner: &mut (xr::FrameWaiter, wgpu::Queue, wgpu::Device),
        ) -> VrResult<xr::FrameState> {
            owner
                .0
                .wait()
                .map_err(|e| format!("OpenXR wait frame: {e}"))
        }
        Self::spawn(move || wait_owned(&mut owner))
    }

    fn spawn(mut wait: impl FnMut() -> VrResult<xr::FrameState> + Send + 'static) -> Self {
        let (request, requests) = mpsc::channel();
        let (results, result) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("OpenXR frame wait".into())
            .spawn(move || {
                while let Ok(epoch) = requests.recv() {
                    if results.send((epoch, wait())).is_err() {
                        break;
                    }
                }
            })
            .expect("OpenXR frame wait thread");
        Self {
            request: Some(request),
            result,
            pending: false,
            worker: Some(worker),
        }
    }

    pub fn request(&mut self, epoch: u64) -> VrResult<()> {
        if !self.pending {
            self.request
                .as_ref()
                .ok_or("OpenXR frame waiter stopped")?
                .send(epoch)
                .map_err(|_| "OpenXR frame waiter stopped")?;
            self.pending = true;
        }
        Ok(())
    }

    pub fn poll(&mut self, epoch: u64, present: bool) -> VrResult<Option<xr::FrameState>> {
        self.request(epoch)?;
        let result = if present {
            match self
                .result
                .recv_timeout(std::time::Duration::from_millis(30))
            {
                Ok(result) => Some(result),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(_) => return Err("OpenXR frame waiter stopped".into()),
            }
        } else {
            match self.result.try_recv() {
                Ok(result) => Some(result),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(_) => return Err("OpenXR frame waiter stopped".into()),
            }
        };
        let Some((generation, result)) = result else {
            return Ok(None);
        };
        self.pending = false;
        if generation != epoch {
            return Ok(None);
        }
        result.map(Some)
    }
}

impl Drop for FrameWait {
    fn drop(&mut self) {
        // An idle waiter must release its session/graphics references before
        // the caller destroys Vulkan or exits the process. Detaching it raced
        // Vulkan command-buffer cleanup during --vr-check and normal shutdown.
        self.request.take();
        if !self.pending
            || self
                .worker
                .as_ref()
                .is_some_and(|worker| worker.is_finished())
        {
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
        // A runtime sleeping inside xrWaitFrame cannot be cancelled. Its worker
        // retains BOTH queue and device; do not freeze the UI waiting for it.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn idle_waiter_releases_its_owner_before_shutdown_returns() {
        let released = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        struct Owner(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let owner = Owner(released.clone());
        let waiter = FrameWait::spawn(move || {
            let _retain = &owner;
            unreachable!("idle waiter must not call xrWaitFrame")
        });
        drop(waiter);
        assert!(released.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn next_headset_wait_can_overlap_application_work() {
        let (started, waiting) = mpsc::channel();
        let (wake, sleeping) = mpsc::channel();
        let mut waiter = FrameWait::spawn(move || {
            started.send(()).unwrap();
            sleeping.recv().unwrap();
            Ok(xr::FrameState {
                predicted_display_time: xr::Time::from_nanos(1),
                predicted_display_period: xr::Duration::from_nanos(8_333_333),
                should_render: true,
            })
        });
        // Request immediately after xrEndFrame, before the next app iteration.
        waiter.request(0).unwrap();
        waiting
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        assert!(waiter.poll(0, false).unwrap().is_none());
        wake.send(()).unwrap();
        assert!(waiter.poll(0, true).unwrap().unwrap().should_render);
        assert!(waiting.try_recv().is_err()); // polling didn't schedule another wait
    }

    #[test]
    fn sleeping_headset_wait_does_not_block_desktop_or_reuse_old_session_frames() {
        let (wake, sleeping) = mpsc::channel();
        let mut waiter = FrameWait::spawn(move || {
            sleeping.recv().unwrap();
            Ok(xr::FrameState {
                predicted_display_time: xr::Time::from_nanos(1),
                predicted_display_period: xr::Duration::from_nanos(11_111_111),
                should_render: true,
            })
        });
        assert!(waiter.poll(0, false).unwrap().is_none());
        // Several desktop frames can proceed while the runtime remains asleep.
        for _ in 0..10 {
            assert!(waiter.poll(0, false).unwrap().is_none());
        }
        wake.send(()).unwrap();
        assert!(waiter.poll(1, true).unwrap().is_none());
        // A new session generation must request its own frame.
        wake.send(()).unwrap();
        assert!(waiter.poll(1, true).unwrap().unwrap().should_render);
    }
}

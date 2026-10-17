//! Keep sleeping-headset frame waits and empty submissions off the desktop UI thread.

use super::VrResult;
use openxr as xr;
use std::sync::{mpsc, Arc, Mutex};

enum FrameRequest {
    Wait,
    Empty(xr::FrameState),
}

pub(super) struct FrameWait {
    request: Option<mpsc::Sender<(u64, FrameRequest)>>,
    result: mpsc::Receiver<(u64, VrResult<Option<xr::FrameState>>)>,
    pending: bool,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl FrameWait {
    pub fn new(
        waiter: xr::FrameWaiter,
        stream: Arc<Mutex<xr::FrameStream<xr::Vulkan>>>,
        queue: wgpu::Queue,
        device: wgpu::Device,
    ) -> Self {
        // Release the OpenXR handles before the retained graphics owners.
        let mut owner = (waiter, stream, queue, device);
        Self::spawn(move |request| {
            // Borrow the whole tuple so Rust retains the queue/device too,
            // rather than capturing only the OpenXR fields used below.
            let owner = &mut owner;
            match request {
                FrameRequest::Wait => owner
                    .0
                    .wait()
                    .map(Some)
                    .map_err(|e| format!("OpenXR wait frame: {e}")),
                FrameRequest::Empty(frame) => {
                    let mut stream = owner.1.lock().unwrap();
                    stream
                        .begin()
                        .map_err(|e| format!("OpenXR idle begin frame: {e}"))?;
                    stream
                        .end(
                            frame.predicted_display_time,
                            xr::EnvironmentBlendMode::OPAQUE,
                            &[],
                        )
                        .map_err(|e| format!("OpenXR idle end frame: {e}"))?;
                    Ok(None)
                }
            }
        })
    }

    fn spawn(
        mut execute: impl FnMut(FrameRequest) -> VrResult<Option<xr::FrameState>> + Send + 'static,
    ) -> Self {
        let (request, requests) = mpsc::channel();
        let (results, result) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("OpenXR frame wait".into())
            .spawn(move || {
                while let Ok((epoch, request)) = requests.recv() {
                    if results.send((epoch, execute(request))).is_err() {
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
            self.send(epoch, FrameRequest::Wait)?;
        }
        Ok(())
    }

    fn send(&mut self, epoch: u64, request: FrameRequest) -> VrResult<()> {
        self.request
            .as_ref()
            .ok_or("OpenXR frame waiter stopped")?
            .send((epoch, request))
            .map_err(|_| "OpenXR frame waiter stopped")?;
        self.pending = true;
        Ok(())
    }

    /// Complete a waited frame without layers. No stream lock is taken by the
    /// caller: a runtime throttling xrBeginFrame/xrEndFrame cannot stall desktop.
    pub fn submit_empty(&mut self, epoch: u64, frame: xr::FrameState) -> VrResult<()> {
        if self.pending {
            return Err("OpenXR frame operation already pending".into());
        }
        self.send(epoch, FrameRequest::Empty(frame))
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
        let frame = result?;
        if frame.is_none() {
            // Only start the next wait after the empty frame has ended. The
            // main thread cannot begin a rendered frame while submission runs.
            self.request(epoch)?;
        }
        Ok(frame)
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
        // A runtime sleeping inside a frame call cannot be cancelled. Its worker
        // retains BOTH queue and device; do not freeze the UI waiting for it.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_wait(wait: impl FnMut() -> VrResult<xr::FrameState> + Send + 'static) -> FrameWait {
        let mut wait = wait;
        FrameWait::spawn(move |request| match request {
            FrameRequest::Wait => wait().map(Some),
            FrameRequest::Empty(_) => Ok(None),
        })
    }
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
        let waiter = spawn_wait(move || {
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
        let mut waiter = spawn_wait(move || {
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
        let mut waiter = spawn_wait(move || {
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

    #[test]
    fn blocked_empty_submission_leaves_desktop_free_and_serializes_resume() {
        let (started, operations) = mpsc::channel();
        let (wake, sleeping) = mpsc::channel();
        let frame = xr::FrameState {
            predicted_display_time: xr::Time::from_nanos(123),
            predicted_display_period: xr::Duration::from_nanos(8_333_333),
            should_render: true,
        };
        let mut waiter = FrameWait::spawn(move |request| match request {
            FrameRequest::Wait => {
                started.send("wait").unwrap();
                Ok(Some(frame))
            }
            FrameRequest::Empty(frame) => {
                assert_eq!(frame.predicted_display_time.as_nanos(), 123);
                started.send("empty begin").unwrap();
                sleeping.recv().unwrap();
                started.send("empty end").unwrap();
                Ok(None)
            }
        });
        let frame = waiter.poll(0, true).unwrap().unwrap();
        assert_eq!(operations.recv().unwrap(), "wait");
        waiter.submit_empty(0, frame).unwrap();
        assert_eq!(operations.recv().unwrap(), "empty begin");
        // Desktop frames must continue even if the runtime never finishes end.
        for _ in 0..20 {
            assert!(waiter.poll(0, false).unwrap().is_none());
        }
        assert!(operations.try_recv().is_err());
        assert!(waiter.submit_empty(0, frame).is_err());
        wake.send(()).unwrap();
        assert!(waiter.poll(0, true).unwrap().is_none());
        assert_eq!(operations.recv().unwrap(), "empty end");
        assert!(waiter.poll(0, true).unwrap().unwrap().should_render);
        assert_eq!(operations.recv().unwrap(), "wait");
    }

    #[test]
    fn empty_submission_errors_reach_the_caller() {
        let mut waiter = FrameWait::spawn(|request| match request {
            FrameRequest::Wait => unreachable!(),
            FrameRequest::Empty(_) => Err("idle submission failed".into()),
        });
        waiter
            .submit_empty(
                0,
                xr::FrameState {
                    predicted_display_time: xr::Time::from_nanos(1),
                    predicted_display_period: xr::Duration::from_nanos(8_333_333),
                    should_render: false,
                },
            )
            .unwrap();
        assert_eq!(waiter.poll(0, true).unwrap_err(), "idle submission failed");
    }

    #[test]
    fn shutdown_during_empty_submission_retains_owners_until_completion() {
        let (released, release_notice) = mpsc::channel();
        struct Owner(mpsc::Sender<()>);
        impl Drop for Owner {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let owner = Owner(released);
        let (started, running) = mpsc::channel();
        let (wake, sleeping) = mpsc::channel();
        let mut waiter = FrameWait::spawn(move |request| {
            let _retain = &owner;
            assert!(matches!(request, FrameRequest::Empty(_)));
            started.send(()).unwrap();
            sleeping.recv().unwrap();
            Ok(None)
        });
        waiter
            .submit_empty(
                0,
                xr::FrameState {
                    predicted_display_time: xr::Time::from_nanos(1),
                    predicted_display_period: xr::Duration::from_nanos(8_333_333),
                    should_render: false,
                },
            )
            .unwrap();
        running.recv().unwrap();
        drop(waiter);
        assert!(release_notice.try_recv().is_err());
        wake.send(()).unwrap();
        release_notice
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
    }
}

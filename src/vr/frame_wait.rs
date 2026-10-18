//! Keep sleeping-headset frame waits and empty submissions off the desktop UI thread.

use super::{GpuAccess, VrResult};
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
        gpu_access: GpuAccess,
    ) -> Self {
        // Release the OpenXR handles before the retained graphics owners.
        let mut owner = (waiter, stream, queue, device);
        Self::spawn(gpu_access, move |request| {
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
        gpu_access: GpuAccess,
        mut execute: impl FnMut(FrameRequest) -> VrResult<Option<xr::FrameState>> + Send + 'static,
    ) -> Self {
        let (request, requests) = mpsc::channel();
        let (results, result) = mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("OpenXR frame wait".into())
            .spawn(move || {
                while let Ok((epoch, request)) = requests.recv() {
                    // xrWaitFrame does not access VkQueue; idle begin/end may.
                    // Share this guard with desktop uploads, submits and presents.
                    let result = match request {
                        FrameRequest::Wait => execute(request),
                        FrameRequest::Empty(_) => {
                            let _gpu = gpu_access.lock();
                            execute(request)
                        }
                    };
                    if results.send((epoch, result)).is_err() {
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
    /// caller. Application rendering yields while the native queue is in use.
    pub fn submit_empty(&mut self, epoch: u64, frame: xr::FrameState) -> VrResult<()> {
        if self.pending {
            return Err("OpenXR frame operation already pending".into());
        }
        self.send(epoch, FrameRequest::Empty(frame))
    }

    pub fn poll(&mut self, epoch: u64, present: bool) -> VrResult<Option<xr::FrameState>> {
        self.poll_impl(epoch, present, true)
    }

    /// Finish an outstanding frame when unworn, without starting another wait.
    /// Presence events continue to be polled by the application thread.
    pub fn drain(&mut self, epoch: u64) -> VrResult<Option<xr::FrameState>> {
        self.poll_impl(epoch, false, false)
    }

    fn poll_impl(
        &mut self,
        epoch: u64,
        present: bool,
        request_next: bool,
    ) -> VrResult<Option<xr::FrameState>> {
        if request_next {
            self.request(epoch)?;
        }
        if !self.pending {
            return Ok(None);
        }
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
        if frame.is_none() && request_next {
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
        spawn_wait_on(GpuAccess::default(), wait)
    }

    fn spawn_wait_on(
        gpu_access: GpuAccess,
        wait: impl FnMut() -> VrResult<xr::FrameState> + Send + 'static,
    ) -> FrameWait {
        let mut wait = wait;
        FrameWait::spawn(gpu_access, move |request| match request {
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
        let gpu_access = GpuAccess::default();
        let mut waiter = spawn_wait_on(gpu_access.clone(), move || {
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
        // xrWaitFrame may sleep, but must leave the desktop GPU available.
        assert!(gpu_access.try_lock().is_some());
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
    fn idle_submission_excludes_desktop_gpu_work_but_keeps_event_polling_responsive() {
        let (started, operations) = mpsc::channel();
        let (wake, sleeping) = mpsc::channel();
        let frame = xr::FrameState {
            predicted_display_time: xr::Time::from_nanos(123),
            predicted_display_period: xr::Duration::from_nanos(8_333_333),
            should_render: true,
        };
        let gpu_access = GpuAccess::default();
        let mut waiter = FrameWait::spawn(gpu_access.clone(), move |request| match request {
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
        // Event polling remains nonblocking, while desktop GPU submissions must
        // yield until xrEndFrame releases the same native Vulkan queue.
        assert!(gpu_access.try_lock().is_none());
        for _ in 0..20 {
            assert!(waiter.poll(0, false).unwrap().is_none());
        }
        assert!(operations.try_recv().is_err());
        assert!(waiter.submit_empty(0, frame).is_err());
        wake.send(()).unwrap();
        assert!(waiter.poll(0, true).unwrap().is_none());
        assert_eq!(operations.recv().unwrap(), "empty end");
        assert!(gpu_access.try_lock().is_some());
        assert!(waiter.poll(0, true).unwrap().unwrap().should_render);
        assert_eq!(operations.recv().unwrap(), "wait");
    }

    #[test]
    fn repeated_unworn_transitions_drain_once_and_resume_with_fresh_frames() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let waits = Arc::new(AtomicUsize::new(0));
        let started = waits.clone();
        let mut waiter = FrameWait::spawn(GpuAccess::default(), move |request| match request {
            FrameRequest::Wait => {
                let number = started.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(Some(xr::FrameState {
                    predicted_display_time: xr::Time::from_nanos(number as i64),
                    predicted_display_period: xr::Duration::from_nanos(11_111_111),
                    should_render: true,
                }))
            }
            FrameRequest::Empty(_) => Ok(None),
        });
        for cycle in 1..=20 {
            let frame = waiter.poll(0, true).unwrap().unwrap();
            assert_eq!(frame.predicted_display_time.as_nanos(), cycle as i64);
            waiter.submit_empty(0, frame).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            while waiter.pending {
                assert!(std::time::Instant::now() < deadline);
                assert!(waiter.drain(0).unwrap().is_none());
                std::thread::yield_now();
            }
            for _ in 0..20 {
                assert!(waiter.drain(0).unwrap().is_none());
            }
            assert_eq!(waits.load(Ordering::SeqCst), cycle);
        }
    }

    #[test]
    #[ignore = "requires a connected OpenXR headset"]
    fn live_headset_idle_handoffs_preserve_desktop_staging_uploads() {
        use wgpu::util::DeviceExt;
        let instance = super::super::create_desktop_instance(&Default::default()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            super::super::request_desktop_device(&adapter, &Default::default()).unwrap();
        let gpu_access = GpuAccess::default();
        let live = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Live data across VR desktop handoffs"),
            contents: bytemuck::cast_slice(&[0u32; 4]),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let bootstrap = super::super::VrBootstrap::for_existing_device().unwrap();
        let mut vr = super::super::VrState::on_existing_device(
            bootstrap,
            &device,
            &queue,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            64,
            64,
            gpu_access.clone(),
        )
        .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !vr.running() {
            assert!(
                std::time::Instant::now() < deadline,
                "Headset session never became ready"
            );
            let _gpu = gpu_access.lock();
            vr.begin_frame().unwrap();
            vr.end_frame().unwrap();
        }
        let mut uploads = 0u32;
        for _ in 0..10 {
            // Exercise the real idle xrBeginFrame/xrEndFrame worker used on
            // removal while desktop uploads continue on the same device.
            vr.frame_waiter.request(vr.session_epoch).unwrap();
            let mut waited = false;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                assert!(
                    std::time::Instant::now() < deadline,
                    "Idle frame never completed"
                );
                if let Some(_gpu) = gpu_access.try_lock() {
                    if let Some(frame) = vr.frame_waiter.drain(vr.session_epoch).unwrap() {
                        assert!(!waited);
                        waited = true;
                        vr.frame_waiter
                            .submit_empty(vr.session_epoch, frame)
                            .unwrap();
                    }
                    uploads += 1;
                    {
                        let mut staging = queue
                            .write_buffer_with(&live, 0, std::num::NonZeroU64::new(16).unwrap())
                            .expect("Desktop staging allocation failed during VR handoff");
                        staging.copy_from_slice(bytemuck::cast_slice(&[uploads; 4]));
                    }
                    queue.submit([device.create_command_encoder(&Default::default()).finish()]);
                    if waited && !vr.frame_waiter.pending {
                        break;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            // Unworn desktop frames do not submit idle work indefinitely.
            assert!(vr.frame_waiter.drain(vr.session_epoch).unwrap().is_none());
            assert!(!vr.frame_waiter.pending);
        }
        let _gpu = gpu_access.lock();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&live, 0, &readback, 0, 16);
        queue.submit([encoder.finish()]);
        let (send, receive) = mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            send.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        receive.recv().unwrap().unwrap();
        assert_eq!(
            bytemuck::cast_slice::<u8, u32>(&readback.get_mapped_range(..)),
            &[uploads; 4]
        );
        drop(vr);
    }

    #[test]
    fn empty_submission_errors_reach_the_caller() {
        let mut waiter = FrameWait::spawn(GpuAccess::default(), |request| match request {
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
        let mut waiter = FrameWait::spawn(GpuAccess::default(), move |request| {
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

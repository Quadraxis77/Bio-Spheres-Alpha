//! GPU frame timing via wgpu timestamp queries.
//!
//! Splits a frame's GPU work into a handful of named segments and reports
//! each segment's GPU time in milliseconds for display in the performance
//! monitor. Uses a small ring of readback buffers so mapping never stalls
//! the GPU - results lag a few frames behind.

use std::sync::mpsc::Receiver;

/// Number of timed segments per frame.
pub const SEGMENT_COUNT: usize = 10;

/// Number of timestamp writes per frame (one per segment boundary).
const TIMESTAMP_COUNT: usize = SEGMENT_COUNT + 1;
const MAX_VIEWS: usize = 2;

/// Frames of readback latency, so mapping a buffer never stalls the GPU.
const FRAMES_IN_FLIGHT: usize = 3;

/// wgpu requires QUERY_RESOLVE destination offsets aligned to this.
const RESOLVE_ALIGNMENT: u64 = 256;

/// Human-readable labels for each timed segment, in order.
///
/// Physics setup, cached signal evaluation, repeated fixed steps, and
/// once-per-render-frame physics maintenance are separated so catch-up cost
/// cannot hide label/scaffold/copy work (or vice versa).
pub const SEGMENT_LABELS: [&str; SEGMENT_COUNT] = [
    "Physics Setup",
    "Topology Repair",
    "Signal Processing",
    "Physics/Lifecycle Steps",
    "Physics Frame Maintenance",
    "Instance Build & Culling",
    "Opaque Render",
    "Skins & Water Mesh",
    "Particles & Fog",
    "Post-Process",
];

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    map_receiver: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    view_count: usize,
}

/// Tracks per-segment GPU timings using `wgpu::QuerySet` timestamp queries.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    slots: [ReadbackSlot; FRAMES_IN_FLIGHT],
    period_ns: f32,
    frame_index: usize,
    last_segments_ms: [f32; SEGMENT_COUNT],
    resolved_this_frame: bool,
    last_completed_at: Option<std::time::Instant>,
    view_count: usize,
    view_index: usize,
}

impl GpuTimer {
    /// Create a new GPU timer, or `None` if the device doesn't support
    /// timestamp queries between passes.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let required =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(required) {
            return None;
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Frame Timer Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: (TIMESTAMP_COUNT * MAX_VIEWS * FRAMES_IN_FLIGHT) as u32,
        });

        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Frame Timer Resolve Buffer"),
            size: FRAMES_IN_FLIGHT as u64 * RESOLVE_ALIGNMENT,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let slots = std::array::from_fn(|_| ReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU Frame Timer Readback Buffer"),
                size: (TIMESTAMP_COUNT * MAX_VIEWS * std::mem::size_of::<u64>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            map_receiver: None,
            view_count: 1,
        });

        Some(Self {
            query_set,
            resolve_buffer,
            slots,
            period_ns: queue.get_timestamp_period(),
            frame_index: 0,
            last_segments_ms: [0.0; SEGMENT_COUNT],
            resolved_this_frame: false,
            last_completed_at: None,
            view_count: 1,
            view_index: 0,
        })
    }

    /// Write a timestamp at segment boundary `boundary` (0..=SEGMENT_COUNT) for the
    /// current frame's slot. Boundary 0 is the start of the frame; boundary
    /// `SEGMENT_COUNT` is the end. Segment `i` spans boundaries `i` to `i + 1`.
    pub fn write_timestamp(&self, encoder: &mut wgpu::CommandEncoder, boundary: usize) {
        debug_assert!(boundary < TIMESTAMP_COUNT);
        let index = (self.frame_index * TIMESTAMP_COUNT * MAX_VIEWS
            + self.view_index * TIMESTAMP_COUNT
            + boundary) as u32;
        encoder.write_timestamp(&self.query_set, index);
    }

    /// Resolve this frame's timestamps into the readback buffer. Must be called
    /// once, after all `write_timestamp` calls for the frame, before `queue.submit`.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.resolved_this_frame = false;
        if self.view_index + 1 < self.view_count {
            return;
        }
        let slot = &self.slots[self.frame_index];
        // If the previous readback for this slot hasn't completed yet, skip -
        // we'll catch up next time this slot comes around.
        if slot.map_receiver.is_some() {
            return;
        }

        let first = (self.frame_index * TIMESTAMP_COUNT * MAX_VIEWS) as u32;
        let last = first + (TIMESTAMP_COUNT * self.view_count) as u32;
        let resolve_offset = self.frame_index as u64 * RESOLVE_ALIGNMENT;

        encoder.resolve_query_set(
            &self.query_set,
            first..last,
            &self.resolve_buffer,
            resolve_offset,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            resolve_offset,
            &slot.buffer,
            0,
            (TIMESTAMP_COUNT * self.view_count * std::mem::size_of::<u64>()) as u64,
        );
        self.slots[self.frame_index].view_count = self.view_count;
        self.resolved_this_frame = true;
    }

    /// Age of the most recently received sample; absent until a readback completes.
    pub fn sample_age_ms(&self) -> Option<f64> {
        self.last_completed_at
            .map(|at| at.elapsed().as_secs_f64() * 1000.0)
    }

    /// Call after `queue.submit`. Kicks off async mapping for the slot just
    /// resolved and polls all slots for completed readbacks.
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        if self.view_index + 1 < self.view_count {
            return;
        }
        if self.resolved_this_frame && self.slots[self.frame_index].map_receiver.is_none() {
            let slot = &mut self.slots[self.frame_index];
            let (sender, receiver) = std::sync::mpsc::channel();
            slot.buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).ok();
                });
            slot.map_receiver = Some(receiver);
        }

        let _ = device.poll(wgpu::PollType::Poll);

        for slot in &mut self.slots {
            let completed = match slot.map_receiver.as_ref().map(|rx| rx.try_recv()) {
                Some(Ok(Ok(()))) => true,
                Some(Ok(Err(error))) => {
                    log::warn!("GPU timing readback failed, recycling slot: {error}");
                    // A failed map leaves the buffer unmapped.
                    slot.map_receiver = None;
                    false
                }
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    // A failed map leaves the buffer unmapped.
                    slot.map_receiver = None;
                    false
                }
                _ => false,
            };

            if completed {
                slot.map_receiver = None;

                {
                    let view = slot.buffer.slice(..).get_mapped_range();
                    let timestamps: &[u64] = bytemuck::cast_slice(&view);
                    self.last_segments_ms.fill(0.0);
                    for view in 0..slot.view_count {
                        let base = view * TIMESTAMP_COUNT;
                        for i in 0..SEGMENT_COUNT {
                            let delta_ticks =
                                timestamps[base + i + 1].saturating_sub(timestamps[base + i]);
                            self.last_segments_ms[i] +=
                                delta_ticks as f32 * self.period_ns / 1_000_000.0;
                        }
                    }
                }
                slot.buffer.unmap();
                self.last_completed_at = Some(std::time::Instant::now());
            }
        }

        self.resolved_this_frame = false;
        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
    }

    pub fn set_view_count(&mut self, count: usize) {
        assert!((1..=MAX_VIEWS).contains(&count));
        self.view_count = count;
        self.view_index = 0;
    }

    pub fn begin_view(&mut self, advance_world: bool) {
        self.view_index = if self.view_count == 2 && !advance_world {
            1
        } else {
            0
        };
    }

    /// GPU time per segment (ms) from the most recently completed readback.
    pub fn segment_times_ms(&self) -> [f32; SEGMENT_COUNT] {
        self.last_segments_ms
    }

    /// Total GPU time across all segments (ms).
    pub fn total_ms(&self) -> f32 {
        self.last_segments_ms.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readback_recovers_and_only_maps_resolved_frames() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .unwrap();
            let features =
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            if !adapter.features().contains(features) {
                return;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features: features,
                    ..Default::default()
                })
                .await
                .unwrap();
            let mut timer = GpuTimer::new(&device, &queue).unwrap();
            timer.after_submit(&device);
            assert!(timer.slots.iter().all(|slot| slot.map_receiver.is_none()));
            assert!(timer.sample_age_ms().is_none());

            let (sender, receiver) = std::sync::mpsc::channel();
            sender.send(Err(wgpu::BufferAsyncError)).unwrap();
            timer.slots[0].map_receiver = Some(receiver);
            timer.after_submit(&device);
            assert!(timer.slots[0].map_receiver.is_none());

            for _ in 0..8 {
                let mut encoder = device.create_command_encoder(&Default::default());
                for boundary in 0..=SEGMENT_COUNT {
                    timer.write_timestamp(&mut encoder, boundary);
                }
                timer.resolve(&mut encoder);
                queue.submit([encoder.finish()]);
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .unwrap();
                timer.after_submit(&device);
            }
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            timer.after_submit(&device);
            assert!(timer.sample_age_ms().is_some());

            // A stereo sample must wait for both eyes. The first eye alone must
            // not map a query range that the second eye will still write.
            timer.last_completed_at = None;
            timer.set_view_count(2);
            for eye in 0..2 {
                timer.begin_view(eye == 0);
                let mut encoder = device.create_command_encoder(&Default::default());
                for boundary in 0..=SEGMENT_COUNT {
                    timer.write_timestamp(&mut encoder, boundary);
                }
                timer.resolve(&mut encoder);
                queue.submit([encoder.finish()]);
                timer.after_submit(&device);
                if eye == 0 {
                    assert!(timer.sample_age_ms().is_none());
                    assert!(!timer.resolved_this_frame);
                }
            }
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            timer.after_submit(&device);
            assert!(timer.sample_age_ms().is_some());
            assert!(timer
                .segment_times_ms()
                .iter()
                .all(|ms| ms.is_finite() && *ms >= 0.0));
            timer.set_view_count(1);
        });
    }
}

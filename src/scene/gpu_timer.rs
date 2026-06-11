//! GPU frame timing via wgpu timestamp queries.
//!
//! Splits a frame's GPU work into a handful of named segments and reports
//! each segment's GPU time in milliseconds for display in the performance
//! monitor. Uses a small ring of readback buffers so mapping never stalls
//! the GPU - results lag a few frames behind.

use std::sync::mpsc::Receiver;

/// Number of timed segments per frame.
pub const SEGMENT_COUNT: usize = 5;

/// Number of timestamp writes per frame (one per segment boundary).
const TIMESTAMP_COUNT: usize = SEGMENT_COUNT + 1;

/// Frames of readback latency, so mapping a buffer never stalls the GPU.
const FRAMES_IN_FLIGHT: usize = 3;

/// wgpu requires QUERY_RESOLVE destination offsets aligned to this.
const RESOLVE_ALIGNMENT: u64 = 256;

/// Human-readable labels for each timed segment, in order.
pub const SEGMENT_LABELS: [&str; SEGMENT_COUNT] = [
    "Physics & Compute",
    "Instance Build & Culling",
    "Opaque Render",
    "Skins & Effects",
    "Post-Process",
];

struct ReadbackSlot {
    buffer: wgpu::Buffer,
    map_receiver: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

/// Tracks per-segment GPU timings using `wgpu::QuerySet` timestamp queries.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    slots: [ReadbackSlot; FRAMES_IN_FLIGHT],
    period_ns: f32,
    frame_index: usize,
    last_segments_ms: [f32; SEGMENT_COUNT],
}

impl GpuTimer {
    /// Create a new GPU timer, or `None` if the device doesn't support
    /// timestamp queries between passes.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let required = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(required) {
            return None;
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Frame Timer Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: (TIMESTAMP_COUNT * FRAMES_IN_FLIGHT) as u32,
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
                size: (TIMESTAMP_COUNT * std::mem::size_of::<u64>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            map_receiver: None,
        });

        Some(Self {
            query_set,
            resolve_buffer,
            slots,
            period_ns: queue.get_timestamp_period(),
            frame_index: 0,
            last_segments_ms: [0.0; SEGMENT_COUNT],
        })
    }

    /// Write a timestamp at segment boundary `boundary` (0..=SEGMENT_COUNT) for the
    /// current frame's slot. Boundary 0 is the start of the frame; boundary
    /// `SEGMENT_COUNT` is the end. Segment `i` spans boundaries `i` to `i + 1`.
    pub fn write_timestamp(&self, encoder: &mut wgpu::CommandEncoder, boundary: usize) {
        debug_assert!(boundary < TIMESTAMP_COUNT);
        let index = (self.frame_index * TIMESTAMP_COUNT + boundary) as u32;
        encoder.write_timestamp(&self.query_set, index);
    }

    /// Resolve this frame's timestamps into the readback buffer. Must be called
    /// once, after all `write_timestamp` calls for the frame, before `queue.submit`.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let slot = &self.slots[self.frame_index];
        // If the previous readback for this slot hasn't completed yet, skip -
        // we'll catch up next time this slot comes around.
        if slot.map_receiver.is_some() {
            return;
        }

        let first = (self.frame_index * TIMESTAMP_COUNT) as u32;
        let last = first + TIMESTAMP_COUNT as u32;
        let resolve_offset = self.frame_index as u64 * RESOLVE_ALIGNMENT;

        encoder.resolve_query_set(&self.query_set, first..last, &self.resolve_buffer, resolve_offset);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            resolve_offset,
            &slot.buffer,
            0,
            (TIMESTAMP_COUNT * std::mem::size_of::<u64>()) as u64,
        );
    }

    /// Call after `queue.submit`. Kicks off async mapping for the slot just
    /// resolved and polls all slots for completed readbacks.
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        if self.slots[self.frame_index].map_receiver.is_none() {
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
            let completed = matches!(
                slot.map_receiver.as_ref().map(|rx| rx.try_recv()),
                Some(Ok(Ok(())))
            );

            if completed {
                slot.map_receiver = None;

                {
                    let view = slot.buffer.slice(..).get_mapped_range();
                    let timestamps: &[u64] = bytemuck::cast_slice(&view);
                    for i in 0..SEGMENT_COUNT {
                        let delta_ticks = timestamps[i + 1].saturating_sub(timestamps[i]);
                        self.last_segments_ms[i] =
                            delta_ticks as f32 * self.period_ns / 1_000_000.0;
                    }
                }
                slot.buffer.unmap();
            }
        }

        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
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

//! Uploads that must take effect between dispatches in one submission.
use wgpu::util::DeviceExt;

pub(crate) fn encode_buffer_write(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    destination: &wgpu::Buffer,
    contents: &[u8],
) {
    // Queue::write_buffer runs before the entire submission. An encoded copy
    // preserves each step's values when several steps share a uniform buffer.
    let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Simulation step parameter upload"),
        contents,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_buffer(&staging, 0, destination, 0, contents.len() as u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_up_steps_keep_distinct_times_in_one_submission() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("GPU adapter required for upload ordering regression test");
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .expect("test device");
            let params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shared step parameters"),
                size: 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Per-step parameter snapshots"),
                size: 80,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            for step in 0..10u32 {
                let values = [step, (step as f32 / 64.0).to_bits()];
                encode_buffer_write(
                    &device,
                    &mut encoder,
                    &params,
                    bytemuck::cast_slice(&values),
                );
                // Snapshot what this step would read before encoding the next update.
                encoder.copy_buffer_to_buffer(&params, 0, &readback, step as u64 * 8, 8);
            }
            queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap();
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .unwrap();
            receiver.recv().unwrap().unwrap();
            let mapped = readback.slice(..).get_mapped_range();
            let snapshots: &[u32] = bytemuck::cast_slice(&mapped);
            for step in 0..10usize {
                assert_eq!(snapshots[step * 2], step as u32);
                assert_eq!(f32::from_bits(snapshots[step * 2 + 1]), step as f32 / 64.0);
            }
            drop(mapped);
            readback.unmap();
        });
    }
}

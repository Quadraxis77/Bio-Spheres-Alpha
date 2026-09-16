//! GPU centroid reduction; only the final 32-byte camera target crosses to CPU.
use super::GpuTripleBufferSystem;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FollowResult {
    pub center: [f32; 4],
    pub root: u32,
    pub count: u32,
    _pad: [u32; 2],
}
pub struct OrganismFollow {
    gather: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    bindings: [wgpu::BindGroup; 3],
    params: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    groups: u32,
    copied: bool,
    receiver: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}
impl OrganismFollow {
    pub fn new(device: &wgpu::Device, b: &GpuTripleBufferSystem, labels: &wgpu::Buffer) -> Self {
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let groups = b.capacity.div_ceil(256).max(1);
        let params = buffer(
            "Follow parameters",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let partials = buffer(
            "Follow partial sums",
            groups as u64 * 16,
            wgpu::BufferUsages::STORAGE,
        );
        let output = buffer(
            "Follow result",
            32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = buffer(
            "Follow camera readback",
            32,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Organism follow reduction"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/organism_follow.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Follow reduction"),
            entries: &(0..6)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 3 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 3,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let bindings = std::array::from_fn(|i| {
            let buffers = [
                &b.position_and_mass[i],
                labels,
                &b.cell_count_buffer,
                &params,
                &partials,
                &output,
            ];
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Follow reduction"),
                layout: &layout,
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        Self {
            gather: pipeline("gather"),
            finish: pipeline("finish"),
            bindings,
            params,
            output,
            staging,
            groups,
            copied: false,
            receiver: None,
        }
    }
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        index: usize,
        cell: u32,
    ) {
        if self.copied || self.receiver.is_some() {
            return;
        }
        crate::simulation::gpu_upload::encode_buffer_write(
            device,
            encoder,
            &self.params,
            bytemuck::cast_slice(&[cell, 0, 0, 0]),
        );
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &self.bindings[index], &[]);
            pass.set_pipeline(&self.gather);
            pass.dispatch_workgroups(self.groups, 1, 1);
            pass.set_pipeline(&self.finish);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, 32);
        self.copied = true;
    }
    pub fn after_submit(&mut self) {
        if !self.copied {
            return;
        }
        self.copied = false;
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.receiver = Some(rx);
    }
    pub fn poll(&mut self, device: &wgpu::Device) -> Option<FollowResult> {
        let rx = self.receiver.as_ref()?;
        let _ = device.poll(wgpu::PollType::Poll);
        match rx.try_recv() {
            Ok(Ok(())) => {
                let result =
                    bytemuck::pod_read_unaligned(&self.staging.slice(..).get_mapped_range());
                self.staging.unmap();
                self.receiver = None;
                Some(result)
            }
            Ok(Err(_)) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.staging.unmap();
                self.receiver = None;
                None
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;
    #[test]
    fn follow_reduction_matches_cpu_across_groups_deaths_and_slot_growth() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&Default::default());
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .unwrap();
            let b = GpuTripleBufferSystem::with_mode_capacity(&device, 777, 16);
            let mut labels = vec![0u32; 777];
            let positions: Vec<[f32; 4]> = (0..777)
                .map(|i| {
                    labels[i] = if i % 3 == 0 { 0 } else { 1 };
                    [
                        i as f32 - 300.,
                        i as f32 * 0.25,
                        -(i as f32),
                        if i % 7 == 0 { 0. } else { 1. },
                    ]
                })
                .collect();
            let label_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&labels),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let mut follow = OrganismFollow::new(&device, &b, &label_buffer);
            assert_eq!(follow.staging.size(), 32);
            for (index, slots, cell) in [(0, 250, 0), (1, 777, 1), (2, 777, 0), (0, 0, 0)] {
                queue.write_buffer(
                    &b.position_and_mass[index],
                    0,
                    bytemuck::cast_slice(&positions),
                );
                queue.write_buffer(
                    &b.cell_count_buffer,
                    0,
                    bytemuck::cast_slice(&[slots as u32, slots as u32]),
                );
                let mut encoder = device.create_command_encoder(&Default::default());
                follow.encode(&device, &mut encoder, index, cell);
                // Duplicate scheduling before/after map must not overwrite staging.
                follow.encode(&device, &mut encoder, index, cell);
                queue.submit([encoder.finish()]);
                follow.after_submit();
                follow.after_submit();
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .unwrap();
                let actual = follow.poll(&device).expect("completed camera target");
                let root = if slots == 0 {
                    u32::MAX
                } else {
                    labels[cell as usize]
                };
                let expected: Vec<_> = positions
                    .iter()
                    .zip(&labels)
                    .take(slots)
                    .filter(|(p, label)| **label == root && p[3] > 0.)
                    .map(|(p, _)| p)
                    .collect();
                assert_eq!(actual.root, root);
                assert_eq!(actual.count as usize, expected.len());
                if !expected.is_empty() {
                    for axis in 0..3 {
                        let avg =
                            expected.iter().map(|p| p[axis]).sum::<f32>() / expected.len() as f32;
                        assert!((actual.center[axis] - avg).abs() < 0.001);
                    }
                }
                assert!(follow.poll(&device).is_none());
            }
        });
    }
}

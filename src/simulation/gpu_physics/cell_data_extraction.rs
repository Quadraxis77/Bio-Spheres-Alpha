//! GPU Cell Data Extraction System
//!
//! Provides GPU-based cell data extraction for cell inspection without CPU state management.
//! Uses bounds validation and calculates derived values like age.

use super::{AdhesionBuffers, GpuTripleBufferSystem};

/// Cell extraction parameters for GPU cell data extraction compute shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellExtractionParams {
    pub cell_index: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Inspected cell data structure matching the WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InspectedCellData {
    // Cell position (12 bytes + 4 padding = 16 bytes)
    pub position: [f32; 3],
    pub _pad0: f32,

    // Cell velocity (12 bytes + 4 padding = 16 bytes)
    pub velocity: [f32; 3],
    pub _pad1: f32,

    // Cell physics properties (16 bytes)
    pub mass: f32,
    pub radius: f32,
    pub birth_time: f32,
    pub age: f32,

    // Cell division properties (16 bytes)
    pub nutrient_threshold: f32,
    pub split_interval: f32,
    pub split_count: u32,
    pub max_splits: u32,

    // Cell genome and mode info (16 bytes)
    pub genome_id: u32,
    pub mode_index: u32,
    pub cell_id: u32,
    pub cell_slot_index: u32,

    // Cell state properties (16 bytes)
    pub nutrient_gain_rate: f32,
    pub max_cell_size: f32,
    pub stiffness: f32,
    pub is_valid: u32,

    // Additional properties (16 bytes)
    pub nutrients: f32,
    pub cell_type: u32,
    pub adhesion_count: u32,
    pub is_dead: u32,

    // Organism identity (16 bytes)
    pub organism_id: u32, // min cell index in connected component; 0xFFFFFFFF = dead/isolated
    pub reserve: u32, // embryocyte reserve (0-65535); also used by non-embryocytes as head-start buffer
    pub _pad3: u32,
    pub _pad4: u32,
}

impl Default for InspectedCellData {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            _pad0: 0.0,
            velocity: [0.0; 3],
            _pad1: 0.0,
            mass: 0.0,
            radius: 0.0,
            birth_time: 0.0,
            age: 0.0,
            nutrient_threshold: 0.0,
            split_interval: 0.0,
            split_count: 0,
            max_splits: 0,
            genome_id: 0,
            mode_index: 0,
            cell_id: 0,
            cell_slot_index: 0,
            nutrient_gain_rate: 0.0,
            max_cell_size: 0.0,
            stiffness: 0.0,
            is_valid: 0,
            nutrients: 0.0,
            cell_type: 0,
            adhesion_count: 0,
            is_dead: 0,
            organism_id: u32::MAX,
            reserve: 0,
            _pad3: 0,
            _pad4: 0,
        }
    }
}

impl InspectedCellData {
    /// Check if the extracted data is valid
    pub fn is_valid(&self) -> bool {
        self.is_valid != 0
    }

    /// Get position as glam::Vec3
    pub fn position_vec3(&self) -> glam::Vec3 {
        glam::Vec3::new(self.position[0], self.position[1], self.position[2])
    }

    /// Get velocity as glam::Vec3
    pub fn velocity_vec3(&self) -> glam::Vec3 {
        glam::Vec3::new(self.velocity[0], self.velocity[1], self.velocity[2])
    }
}

/// GPU cell data extraction system for real-time cell inspection
pub struct GpuCellDataExtraction {
    /// Cell data extraction compute pipeline
    pipeline: wgpu::ComputePipeline,

    /// Cell extraction parameters uniform buffer
    params_buffer: wgpu::Buffer,

    /// Output buffer for extracted cell data (GPU-side)
    output_buffer: wgpu::Buffer,

    /// Staging buffer for CPU readback
    readback_buffer: wgpu::Buffer,

    /// Bind groups for cell data extraction
    physics_bind_group: wgpu::BindGroup,
    params_bind_group: wgpu::BindGroup,
    state_bind_group: wgpu::BindGroup,
    output_bind_group: wgpu::BindGroup,

    /// Cached extracted data (from last successful readback)
    cached_data: Option<InspectedCellData>,

    /// Pending map_async receiver - Some means a mapping is in flight
    map_receiver: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

impl GpuCellDataExtraction {
    /// Create a new GPU cell data extraction system
    pub fn new(
        device: &wgpu::Device,
        pipeline: wgpu::ComputePipeline,
        physics_layout: &wgpu::BindGroupLayout,
        params_layout: &wgpu::BindGroupLayout,
        state_layout: &wgpu::BindGroupLayout,
        output_layout: &wgpu::BindGroupLayout,
        buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &AdhesionBuffers,
        label_buffer: &wgpu::Buffer,
        buffer_index: usize,
    ) -> Self {
        let data_size = std::mem::size_of::<InspectedCellData>() as u64;

        // Create parameters uniform buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Extraction Params Buffer"),
            size: std::mem::size_of::<CellExtractionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffer (GPU-side)
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Extraction Output Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for CPU readback
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Extraction Readback Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create physics bind group (Group 0)
        let physics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Extraction Physics Bind Group"),
            layout: physics_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.position_and_mass[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.velocity[buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.position_and_mass[(buffer_index + 1) % 3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.velocity[(buffer_index + 1) % 3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        });

        // Create params bind group (Group 1)
        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Extraction Params Bind Group"),
            layout: params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        // Create state bind group (Group 2) - read-only access to cell state
        let state_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Extraction State Bind Group"),
            layout: state_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.birth_times.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.split_intervals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.split_nutrient_thresholds.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.split_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.split_ready_frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.max_splits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.genome_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.cell_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.nutrient_gain_rates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.max_cell_sizes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.stiffnesses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.nutrients_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.cell_types.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: buffers.death_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: label_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: buffers.embryocyte_reserve_buffer.as_entire_binding(),
                },
            ],
        });

        // Create output bind group (Group 3)
        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Extraction Output Bind Group"),
            layout: output_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            params_buffer,
            output_buffer,
            readback_buffer,
            physics_bind_group,
            params_bind_group,
            state_bind_group,
            output_bind_group,
            cached_data: None,
            map_receiver: None,
        }
    }

    /// Extract cell data from GPU buffers using compute shader
    ///
    /// This method uploads the cell index and dispatches the compute shader
    /// to extract all cell properties and calculate derived values.
    pub fn extract_cell_data(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        cell_index: u32,
    ) {
        // Create cell extraction parameters
        let params = CellExtractionParams {
            cell_index,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        // Upload parameters to GPU
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Dispatch compute shader (single workgroup as per requirements)
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cell Data Extraction Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.physics_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.params_bind_group, &[]);
        compute_pass.set_bind_group(2, &self.state_bind_group, &[]);
        compute_pass.set_bind_group(3, &self.output_bind_group, &[]);

        // Single workgroup dispatch (1,1,1) for single cell extraction as per requirements
        compute_pass.dispatch_workgroups(1, 1, 1);

        drop(compute_pass);

        // Copy output buffer to staging buffer for CPU readback
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.readback_buffer,
            0,
            std::mem::size_of::<InspectedCellData>() as u64,
        );
    }

    /// Poll for extraction completion and return extracted data if available
    ///
    /// This method should be called each frame to check for completed async readbacks.
    /// Returns Some(data) if extraction is complete, None if still in progress.
    pub fn poll_extraction(&mut self, device: &wgpu::Device) -> Option<InspectedCellData> {
        if self.map_receiver.is_none() {
            // No mapping in flight - start one
            let slice = self.readback_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).ok();
            });
            self.map_receiver = Some(receiver);
        }

        // Poll device to advance async work
        let _ = device.poll(wgpu::PollType::Poll);

        // Check if the in-flight mapping completed
        let completed = if let Some(ref rx) = self.map_receiver {
            matches!(rx.try_recv(), Ok(Ok(())))
        } else {
            false
        };

        if completed {
            self.map_receiver = None;

            let slice = self.readback_buffer.slice(..);
            let view = slice.get_mapped_range();
            let data_bytes: &[u8] = &view;

            if data_bytes.len() >= std::mem::size_of::<InspectedCellData>() {
                let data: InspectedCellData =
                    *bytemuck::from_bytes(&data_bytes[..std::mem::size_of::<InspectedCellData>()]);
                drop(view);
                self.readback_buffer.unmap();
                self.cached_data = Some(data);
                return Some(data);
            } else {
                drop(view);
                self.readback_buffer.unmap();
            }
        }

        None
    }

    /// Get cached extracted data from the last successful extraction
    pub fn get_cached_data(&self) -> Option<&InspectedCellData> {
        self.cached_data.as_ref()
    }

    /// Get reference to the output buffer for async readback operations
    pub fn get_output_buffer(&self) -> &wgpu::Buffer {
        &self.output_buffer
    }

    /// Clear cached data
    pub fn clear_cache(&mut self) {
        self.cached_data = None;
    }
}

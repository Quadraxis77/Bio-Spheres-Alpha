use crate::genome::{CellAddressSelector, Genome};
use crate::simulation::gpu_physics::{AdhesionBuffers, GpuTripleBufferSystem};
use wgpu::util::DeviceExt;

const MAX_GPU_SCAFFOLD_RULES: usize = 4096;

const SELECTOR_ANY: u32 = 0;
const SELECTOR_MODE: u32 = 1;
const SELECTOR_LINEAGE: u32 = 2;
const SELECTOR_LINEAGE_OR_MODE: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuScaffoldRule {
    id: u32,
    genome_id: u32,
    endpoint_a_kind: u32,
    endpoint_b_kind: u32,
    endpoint_a_mode: u32,
    endpoint_b_mode: u32,
    endpoint_a_hash_lo: u32,
    endpoint_a_hash_hi: u32,
    endpoint_b_hash_lo: u32,
    endpoint_b_hash_hi: u32,
    rest_length_bits: u32,
    max_range_bits: u32,
    endpoint_a_branch_slot: u32,
    endpoint_b_branch_slot: u32,
    preferred_generation_delta: i32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuScaffoldParams {
    rule_count: u32,
    cell_slots: u32,
    pass_index: u32,
    _pad0: u32,
}

pub struct GpuScaffoldSystem {
    pipeline: wgpu::ComputePipeline,
    physics_layout: wgpu::BindGroupLayout,
    cell_layout: wgpu::BindGroupLayout,
    adhesion_layout: wgpu::BindGroupLayout,
    rule_layout: wgpu::BindGroupLayout,
    rules_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    rule_count: u32,
}

impl GpuScaffoldSystem {
    pub fn new(device: &wgpu::Device) -> Self {
        let physics_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Scaffold Physics Layout"),
            entries: &[
                buffer_entry(0, true, false, wgpu::ShaderStages::COMPUTE),
                buffer_entry(1, true, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(2, true, true, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let cell_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Scaffold Cell Layout"),
            entries: &[
                buffer_entry(0, true, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(1, true, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(2, true, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(3, true, true, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let adhesion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Scaffold Adhesion Layout"),
            entries: &[
                buffer_entry(0, false, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(1, false, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(2, false, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(3, false, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(4, false, true, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let rule_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Scaffold Rule Layout"),
            entries: &[
                buffer_entry(0, true, true, wgpu::ShaderStages::COMPUTE),
                buffer_entry(1, true, false, wgpu::ShaderStages::COMPUTE),
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Scaffold Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/gpu_scaffold_resolve.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU Scaffold Pipeline Layout"),
            bind_group_layouts: &[
                &physics_layout,
                &cell_layout,
                &adhesion_layout,
                &rule_layout,
            ],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GPU Scaffold Resolve Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let rules_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Scaffold Rules"),
            size: (MAX_GPU_SCAFFOLD_RULES * std::mem::size_of::<GpuScaffoldRule>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = GpuScaffoldParams {
            rule_count: 0,
            cell_slots: 0,
            pass_index: 0,
            _pad0: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Scaffold Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            physics_layout,
            cell_layout,
            adhesion_layout,
            rule_layout,
            rules_buffer,
            params_buffer,
            rule_count: 0,
        }
    }

    pub fn sync_genomes(&mut self, queue: &wgpu::Queue, genomes: &[Genome]) {
        let mut rules = Vec::new();
        let mut mode_offset = 0usize;
        for (genome_id, genome) in genomes.iter().enumerate() {
            for rule in &genome.scaffold_rules {
                if rules.len() >= MAX_GPU_SCAFFOLD_RULES {
                    break;
                }
                rules.push(GpuScaffoldRule {
                    id: rule.id,
                    genome_id: genome_id as u32,
                    endpoint_a_kind: selector_kind(&rule.endpoint_a),
                    endpoint_b_kind: selector_kind(&rule.endpoint_b),
                    endpoint_a_mode: selector_mode(&rule.endpoint_a, mode_offset),
                    endpoint_b_mode: selector_mode(&rule.endpoint_b, mode_offset),
                    endpoint_a_hash_lo: selector_hash(&rule.endpoint_a) as u32,
                    endpoint_a_hash_hi: (selector_hash(&rule.endpoint_a) >> 32) as u32,
                    endpoint_b_hash_lo: selector_hash(&rule.endpoint_b) as u32,
                    endpoint_b_hash_hi: (selector_hash(&rule.endpoint_b) >> 32) as u32,
                    rest_length_bits: rule.rest_length.max(0.001).to_bits(),
                    max_range_bits: rule.max_formation_range.max(0.0).to_bits(),
                    endpoint_a_branch_slot: selector_branch_slot(&rule.endpoint_a),
                    endpoint_b_branch_slot: selector_branch_slot(&rule.endpoint_b),
                    preferred_generation_delta: rule.preferred_generation_delta as i32,
                    _pad1: 0,
                });
            }
            mode_offset += genome.modes.len();
        }

        self.rule_count = rules.len() as u32;
        if !rules.is_empty() {
            queue.write_buffer(&self.rules_buffer, 0, bytemuck::cast_slice(&rules));
        }
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        buffers: &GpuTripleBufferSystem,
        adhesion_buffers: &AdhesionBuffers,
        position_buffer_index: usize,
        cell_slots: u32,
    ) {
        if self.rule_count == 0 || cell_slots == 0 {
            return;
        }

        let physics_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Scaffold Physics BG"),
            layout: &self.physics_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.physics_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.position_and_mass[position_buffer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cell_count_buffer.as_entire_binding(),
                },
            ],
        });
        let cell_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Scaffold Cell BG"),
            layout: &self.cell_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.mode_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.genome_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.development_addresses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.death_flags.as_entire_binding(),
                },
            ],
        });
        let adhesion_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Scaffold Adhesion BG"),
            layout: &self.adhesion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: adhesion_buffers.adhesion_connections.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adhesion_buffers.cell_adhesion_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: adhesion_buffers.next_adhesion_id.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adhesion_buffers.free_adhesion_slots.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adhesion_buffers.adhesion_counts.as_entire_binding(),
                },
            ],
        });
        let rule_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Scaffold Rule BG"),
            layout: &self.rule_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.rules_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups = (cell_slots + 127) / 128;
        for pass_index in 0..2 {
            let params = GpuScaffoldParams {
                rule_count: self.rule_count,
                cell_slots,
                pass_index,
                _pad0: 0,
            };
            queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU Scaffold Resolve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &physics_bg, &[]);
            pass.set_bind_group(1, &cell_bg, &[]);
            pass.set_bind_group(2, &adhesion_bg, &[]);
            pass.set_bind_group(3, &rule_bg, &[]);
            pass.dispatch_workgroups(workgroups, self.rule_count, 1);
        }

        encoder.copy_buffer_to_buffer(
            &adhesion_buffers.next_adhesion_id,
            0,
            &adhesion_buffers.adhesion_counts,
            0,
            4,
        );
    }
}

fn buffer_entry(
    binding: u32,
    read_only: bool,
    storage: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: if storage {
                wgpu::BufferBindingType::Storage { read_only }
            } else {
                wgpu::BufferBindingType::Uniform
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn selector_kind(selector: &CellAddressSelector) -> u32 {
    match selector {
        CellAddressSelector::AnyCell => SELECTOR_ANY,
        CellAddressSelector::ByModeIndex(_) => SELECTOR_MODE,
        CellAddressSelector::ByMorphologyHash(_) | CellAddressSelector::ByLineageHash(_) => {
            SELECTOR_LINEAGE
        }
        CellAddressSelector::ByLineageHashOrMode { .. } => SELECTOR_LINEAGE_OR_MODE,
    }
}

fn selector_mode(selector: &CellAddressSelector, mode_offset: usize) -> u32 {
    match selector {
        CellAddressSelector::ByModeIndex(mode_index)
        | CellAddressSelector::ByLineageHashOrMode { mode_index, .. } => {
            (mode_offset + *mode_index) as u32
        }
        _ => u32::MAX,
    }
}

fn selector_hash(selector: &CellAddressSelector) -> u64 {
    match selector {
        CellAddressSelector::ByMorphologyHash(hash)
        | CellAddressSelector::ByLineageHash(hash)
        | CellAddressSelector::ByLineageHashOrMode {
            lineage_hash: hash, ..
        } => *hash,
        _ => 0,
    }
}

fn selector_branch_slot(selector: &CellAddressSelector) -> u32 {
    match selector {
        CellAddressSelector::ByLineageHashOrMode {
            preferred_branch_slot,
            ..
        } => *preferred_branch_slot as u32,
        _ => 0,
    }
}

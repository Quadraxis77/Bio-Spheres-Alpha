//! GPU-only fusion: locked pair -> genome allocation -> crossover -> newborn.
//! No readback, CPU genome registration, or per-offspring queue writes.
use super::{
    AdhesionBuffers, GpuCellInsertion, GpuPhysicsPipelines, GpuTripleBufferSystem, MutationSystem,
};
use wgpu::util::DeviceExt;

fn pipeline(device: &wgpu::Device, source: String, entry: &str) -> wgpu::ComputePipeline {
    pipeline_with_layout(device, source, entry, None)
}
fn pipeline_with_layout(
    device: &wgpu::Device,
    source: String,
    entry: &str,
    layout: Option<&wgpu::PipelineLayout>,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
fn bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("GPU Fusion Bindings"),
        layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}

/// Single inventory used by crossover and exact GPU-genome snapshot persistence.
pub(super) fn genome_mode_buffers<'a>(
    b: &'a GpuTripleBufferSystem,
    a: &'a AdhesionBuffers,
    colors: &'a wgpu::Buffer,
    emissive: &'a wgpu::Buffer,
) -> Vec<(&'static str, &'a wgpu::Buffer, u32)> {
    vec![
        ("mode_properties_v0", &b.mode_properties_v0, 4),
        ("mode_properties_v1", &b.mode_properties_v1, 4),
        ("mode_properties_v2", &b.mode_properties_v2, 4),
        ("mode_properties_v3", &b.mode_properties_v3, 4),
        ("mode_properties_v4", &b.mode_properties_v4, 4),
        ("mode_properties_v5", &b.mode_properties_v5, 4),
        ("mode_properties_v6", &b.mode_properties_v6, 4),
        ("mode_properties_v7", &b.mode_properties_v7, 4),
        ("mode_properties_v8", &b.mode_properties_v8, 4),
        ("mode_properties_v9", &b.mode_properties_v9, 4),
        ("mode_properties_v10", &b.mode_properties_v10, 4),
        ("mode_properties_v11", &b.mode_properties_v11, 4),
        ("mode_properties_v12", &b.mode_properties_v12, 4),
        ("mode_properties_v13", &b.mode_properties_v13, 4),
        ("mode_properties_v14", &b.mode_properties_v14, 4),
        ("mode_properties_v15", &b.mode_properties_v15, 4),
        ("genome_mode_data_v0", &b.genome_mode_data_v0, 4),
        ("genome_mode_data_v1", &b.genome_mode_data_v1, 4),
        ("genome_mode_data_v2", &b.genome_mode_data_v2, 4),
        ("genome_mode_data_v3", &b.genome_mode_data_v3, 4),
        ("genome_mode_data_v4", &b.genome_mode_data_v4, 4),
        ("signal_settings_v0", &b.signal_settings_v0, 4),
        ("signal_settings_v1", &b.signal_settings_v1, 4),
        ("signal_settings_v2", &b.signal_settings_v2, 4),
        ("signal_settings_v3", &b.signal_settings_v3, 4),
        ("signal_settings_v4", &b.signal_settings_v4, 4),
        ("child_mode_indices", &b.child_mode_indices, 2),
        ("is_initial_mode", &b.is_initial_mode, 1),
        (
            "parent_make_adhesion_flags",
            &b.parent_make_adhesion_flags,
            1,
        ),
        (
            "child_a_keep_adhesion_flags",
            &b.child_a_keep_adhesion_flags,
            1,
        ),
        (
            "child_b_keep_adhesion_flags",
            &b.child_b_keep_adhesion_flags,
            1,
        ),
        (
            "child_a_after_split_keep_adhesion_flags",
            &b.child_a_after_split_keep_adhesion_flags,
            1,
        ),
        (
            "child_b_after_split_keep_adhesion_flags",
            &b.child_b_after_split_keep_adhesion_flags,
            1,
        ),
        (
            "glueocyte_env_adhesion_flags",
            &b.glueocyte_env_adhesion_flags,
            1,
        ),
        (
            "glueocyte_boulder_adhesion_flags",
            &b.glueocyte_boulder_adhesion_flags,
            1,
        ),
        (
            "glueocyte_cell_adhesion_flags",
            &b.glueocyte_cell_adhesion_flags,
            4,
        ),
        ("oculocyte_params", &b.oculocyte_params, 4),
        ("oculocyte_signal_values", &b.oculocyte_signal_values, 1),
        ("oculocyte_light_filters", &b.oculocyte_light_filters, 4),
        ("regulation_params", &b.regulation_params, 4),
        ("adhesion_settings_v0", &a.adhesion_settings_v0, 4),
        ("adhesion_settings_v1", &a.adhesion_settings_v1, 4),
        ("adhesion_settings_v2", &a.adhesion_settings_v2, 4),
        ("colors", colors, 4),
        ("emissive", emissive, 4),
        ("mode_cell_types", &b.mode_cell_types, 1),
        ("embryocyte_defaults_v9", &b.embryocyte_defaults_v9, 4),
        ("embryocyte_defaults_v10", &b.embryocyte_defaults_v10, 4),
    ]
}

pub struct GpuFusion {
    allocate: wgpu::ComputePipeline,
    allocate_bindings: wgpu::BindGroup,
    copies: Vec<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    resets: Vec<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    spawn: wgpu::ComputePipeline,
    spawn_bindings: wgpu::BindGroup,
    /// GPU-owned ancestry. Read only on explicit inspection/snapshot, never to birth cells.
    pub parentage: wgpu::Buffer,
}

impl GpuFusion {
    pub fn new(
        device: &wgpu::Device,
        b: &GpuTripleBufferSystem,
        a: &AdhesionBuffers,
        m: &MutationSystem,
        physics: &GpuPhysicsPipelines,
        events: &wgpu::Buffer,
        colors: &wgpu::Buffer,
        emissive: &wgpu::Buffer,
    ) -> Self {
        let plans = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Fusion Plans"),
            size: 64 * 32,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let parentage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Fusion Parentage"),
            size: super::mutation::GENOME_RING_CAPACITY as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let limits = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Fusion Limits"),
            contents: bytemuck::cast_slice(&[
                b.mode_pool_capacity as u32,
                super::mutation::GENOME_RING_CAPACITY,
                0,
                0,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let allocate = pipeline(
            device,
            include_str!("../../../shaders/fusion_allocate.wgsl").into(),
            "main",
        );
        let allocate_bindings = bind(
            device,
            &allocate.get_bind_group_layout(0),
            &[
                events,
                &plans,
                m.genome_meta_buffer(),
                &m.genome_ref_counts_buffer,
                &m.genome_ring_state_buffer,
                &m.genome_free_ring_buffer,
                &b.death_flags,
                &limits,
                &parentage,
                &m.genome_initial_orientations,
            ],
        );
        // Copy every per-mode property, not just the subset currently mutated.
        // Split into small passes to stay below storage-binding limits. All passes
        // use the same deterministic per-mode parent choice.
        let fields: Vec<_> = genome_mode_buffers(b, a, colors, emissive)
            .into_iter()
            .filter(|(name, _, _)| {
                *name != "mode_cell_types" && !name.starts_with("embryocyte_defaults")
            })
            .collect();
        let mut copies = Vec::new();
        // 20 + 6 bindings, also fitting devices with 31 storage buffers/stage.
        for (chunk_index, chunk) in fields.chunks(20).enumerate() {
            let mut source = include_str!("../../../shaders/fusion_copy_common.wgsl").to_owned();
            for (i, (_, _, _)) in chunk.iter().enumerate() {
                source += &format!(
                    "\n@group(0) @binding({}) var<storage, read_write> field{i}: array<u32>;\n",
                    i + 6
                );
            }
            source += r#"
@compute @workgroup_size(64)
fn copy_modes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let event = gid.y;
    if (event >= min(events[0], 64u)) { return; }
    let p = plans[event];
    let local = gid.x;
    if (p.destination.x == 0xffffffffu || local >= p.destination.z) { return; }
    let e = 1u + event * 8u;
    let choose_b = local >= p.parents.y || (local < p.parents.w &&
        (hash(ids[events[e]] ^ hash(ids[events[e + 1u]]) ^ hash(local)) & 1u) != 0u);
    let base = select(p.parents.x, p.parents.z, choose_b);
    let src = base + local;
    let dst = p.destination.y + local;
    let delta = i32(p.destination.y) - i32(base);
    let source_type = types[src];
"#;
            for (i, (name, _, width)) in chunk.iter().enumerate() {
                for component in 0..*width {
                    let value = format!("field{i}[src * {width}u + {component}u]");
                    let value = match (*name, component) {
                        ("child_mode_indices", _) => format!("remap_int({value}, delta)"),
                        ("mode_properties_v4", 1 | 2)
                        | ("signal_settings_v2", 0 | 1)
                        | ("signal_settings_v3", 0 | 1)
                        | ("signal_settings_v4", 0) => format!("remap_float({value}, delta)"),
                        ("mode_properties_v9", 2 | 3) | ("mode_properties_v10", 0..=2) => format!(
                            "select({value}, remap_float({value}, delta), source_type == 19u)"
                        ),
                        ("is_initial_mode", _) => "select(0u, 1u, local == p.destination.w)".into(),
                        _ => value,
                    };
                    let value = match *name {
                        "mode_properties_v9" => format!("select({value}, embryo9[src][{component}], local == p.destination.w && source_type == 19u)"),
                        "mode_properties_v10" => format!("select({value}, embryo10[src][{component}], local == p.destination.w && source_type == 19u)"),
                        _ => value,
                    };
                    source += &format!("    field{i}[dst * {width}u + {component}u] = {value};\n");
                }
                if *name == "colors" {
                    source += &format!(
                        r#"
    if (local < min(p.parents.y, p.parents.w)) {{
        for (var c = 0u; c < 3u; c++) {{
            field{i}[dst * 4u + c] = bitcast<u32>((bitcast<f32>(field{i}[(p.parents.x + local) * 4u + c])
                + bitcast<f32>(field{i}[(p.parents.z + local) * 4u + c])) * 0.5);
        }}
    }}
"#
                    );
                }
            }
            if chunk_index == 0 {
                source += "    embryo9[dst] = embryo9[src];\n    embryo10[dst] = embryo10[src];\n";
            }
            source += "}\n";
            let copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fusion Mode Copy Layout"),
                entries: &(0..chunk.len() + 6)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding: binding as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: binding < 4,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Fusion Mode Copy Pipeline Layout"),
                bind_group_layouts: &[&copy_layout],
                push_constant_ranges: &[],
            });
            let copy = pipeline_with_layout(device, source, "copy_modes", Some(&layout));
            let mut buffers = vec![
                &plans,
                events,
                &b.cell_ids,
                &b.mode_cell_types,
                &b.embryocyte_defaults_v9,
                &b.embryocyte_defaults_v10,
            ];
            buffers.extend(chunk.iter().map(|(_, buffer, _)| *buffer));
            let bindings = bind(device, &copy.get_bind_group_layout(0), &buffers);
            copies.push((copy, bindings));
        }
        // Types are committed after the copy passes, so stemocyte remapping above
        // always reads the original parent type. The initial mode is embryonic.
        let type_source = include_str!("../../../shaders/fusion_copy_common.wgsl").replace(
            "var<storage, read> types:",
            "var<storage, read_write> types:",
        ) + include_str!("../../../shaders/fusion_copy_types.wgsl");
        let types = pipeline(device, type_source, "copy_types");
        let types_bindings = bind(
            device,
            &types.get_bind_group_layout(0),
            &[&plans, events, &b.cell_ids, &b.mode_cell_types],
        );
        copies.push((types, types_bindings));

        let spawn_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fusion Newborn Initialization"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../../../shaders/cell_insertion.wgsl"),
                    include_str!("../../../shaders/fusion_spawn.wgsl")
                )
                .into(),
            ),
        });
        let spawn_extra_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fusion Newborn Genome Layout"),
                entries: &(0..6)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fusion Newborn Layout"),
            bind_group_layouts: &[
                &physics.cell_insertion_physics_layout,
                &physics.cell_insertion_params_layout,
                &physics.cell_insertion_state_layout,
                &spawn_extra_layout,
            ],
            push_constant_ranges: &[],
        });
        let spawn = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fusion Newborn"),
            layout: Some(&layout),
            module: &spawn_module,
            entry_point: Some("spawn"),
            compilation_options: Default::default(),
            cache: None,
        });
        let spawn_bindings = bind(
            device,
            &spawn_extra_layout,
            &[
                &plans,
                events,
                &b.mode_properties_v0,
                &b.mode_properties_v1,
                &b.mode_properties_v2,
                &m.genome_initial_orientations,
            ],
        );
        // A reused parent slot must not retain forces, anchors, or public signals.
        let reset_fields: Vec<(&wgpu::Buffer, u32)> = vec![
            (&b.prev_accelerations, 4),
            (&b.env_anchor_buffer, 4),
            (&b.muscle_contraction_buffer, 1),
            (&b.cell_grip_buffer, 1),
            (&b.cell_prev_muscle_contraction, 1),
            (&b.cell_water_delta, 1),
            (&b.cell_heat_delta, 1),
            (&b.mass_deltas_buffer, 1),
            (&a.force_accum_x, 1),
            (&a.force_accum_y, 1),
            (&a.force_accum_z, 1),
            (&a.torque_accum_x, 1),
            (&a.torque_accum_y, 1),
            (&a.torque_accum_z, 1),
            (&a.signal_flags, 16),
        ];
        let mut source = "struct FusionPlan { destination: vec4<u32>, parents: vec4<u32> }\n@group(0) @binding(0) var<storage, read> plans: array<FusionPlan>;\n@group(0) @binding(1) var<storage, read> events: array<u32>;\n".to_string();
        for i in 0..reset_fields.len() {
            source += &format!(
                "@group(0) @binding({}) var<storage, read_write> field{i}: array<u32>;\n",
                i + 2
            );
        }
        source += "@compute @workgroup_size(64) fn reset(@builtin(global_invocation_id) gid: vec3<u32>) { let e = gid.x; if (e >= min(events[0], 64u)) { return; } if (plans[e].destination.x == 0xffffffffu) { return; } let slot = events[1u + e * 8u];\n";
        for (i, (_, width)) in reset_fields.iter().enumerate() {
            source += &format!(
                "for (var j = 0u; j < {width}u; j++) {{ field{i}[slot * {width}u + j] = 0u; }}\n"
            );
        }
        source += "}";
        let reset = pipeline(device, source, "reset");
        let mut reset_buffers = vec![&plans, events];
        reset_buffers.extend(reset_fields.iter().map(|(buffer, _)| *buffer));
        let reset_bindings = bind(device, &reset.get_bind_group_layout(0), &reset_buffers);
        Self {
            allocate,
            allocate_bindings,
            copies,
            resets: vec![(reset, reset_bindings)],
            spawn,
            spawn_bindings,
            parentage,
        }
    }

    /// Complete birth within the same command stream as contact detection.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        insertion: &GpuCellInsertion,
        physics: &GpuPhysicsPipelines,
        cached: &super::CachedBindGroups,
        physics_index: usize,
        adhesion_slots: u32,
    ) {
        self.prepare(encoder);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fusion Parent Adhesion Cleanup"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&physics.adhesion_cleanup);
            pass.set_bind_group(0, &cached.physics[physics_index], &[]);
            pass.set_bind_group(1, &cached.lifecycle, &[]);
            pass.set_bind_group(2, &cached.lifecycle_adhesion, &[]);
            pass.dispatch_workgroups(adhesion_slots.div_ceil(256), 1, 1);
        }
        self.finish(encoder, insertion);
    }

    fn prepare(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fusion Allocate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.allocate);
        pass.set_bind_group(0, &self.allocate_bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        for (pipeline, bindings) in &self.copies {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fusion Crossover"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bindings, &[]);
            pass.dispatch_workgroups(2, 64, 1);
        }
    }

    /// Call after adhesion cleanup has removed the locked parents' old bonds.
    fn finish(&self, encoder: &mut wgpu::CommandEncoder, insertion: &GpuCellInsertion) {
        for (pipeline, bindings) in &self.resets {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fusion Reset Cell"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fusion Birth"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.spawn);
        pass.set_bind_group(0, &insertion.physics_bind_group, &[]);
        pass.set_bind_group(1, &insertion.params_bind_group, &[]);
        pass.set_bind_group(2, &insertion.state_bind_group, &[]);
        pass.set_bind_group(3, &self.spawn_bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::gpu_physics::gametocyte_merge::GametocyteMergeSystem;

    fn read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: u64,
        bytes: u64,
    ) -> Vec<u32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, bytes);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
    }

    #[test]
    fn gpu_fusion_birth_and_exhaustion_without_cpu_offspring() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&Default::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .unwrap();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .unwrap();
            let b = GpuTripleBufferSystem::with_mode_capacity(&device, 4, 32);
            let a = AdhesionBuffers::with_mode_capacity(&device, 4, 32);
            let mut m = MutationSystem::new(&device, &queue, 4);
            m.sync_genome_metadata(&queue, &vec![crate::genome::Genome::default(); 2]);
            m.rebuild_gc_bind_group(&device, &queue);
            m.rebuild_ref_count_sync_bind_group(&device, &queue, &b.genome_ids, &b.death_flags);
            let physics = GpuPhysicsPipelines::new(&device);
            let insertion = GpuCellInsertion::new(
                &device,
                physics.cell_insertion.clone(),
                &physics.cell_insertion_physics_layout,
                &physics.cell_insertion_params_layout,
                &physics.cell_insertion_state_layout,
                &b,
                &a,
            );
            let cached = physics.create_cached_bind_groups(&device, &b, &a, None, None, None);
            let detector = GametocyteMergeSystem::new(&device);
            let colors = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32 * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let emissive = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 32 * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let fusion = GpuFusion::new(
                &device,
                &b,
                &a,
                &m,
                &physics,
                &detector.merge_events_buffer,
                &colors,
                &emissive,
            );
            // Parents have different lengths; the longer parent's dormant tail
            // and absolute developmental links must survive the crossover.
            queue.write_buffer(
                m.genome_meta_buffer(),
                0,
                bytemuck::cast_slice(&[2u32, 0, 0, u32::MAX, 3, 2, 0, u32::MAX]),
            );
            queue.write_buffer(
                &m.genome_ref_counts_buffer,
                0,
                bytemuck::cast_slice(&[u32::MAX; 2]),
            );
            queue.write_buffer(
                &m.genome_ring_state_buffer,
                0,
                bytemuck::cast_slice(&[0u32, 0, 2, 8, 8]),
            );
            queue.write_buffer(
                &m.genome_initial_orientations,
                0,
                bytemuck::cast_slice(&[0f32, 0., 0., 1., 0., 1., 0., 0.]),
            );
            queue.write_buffer(
                &b.cell_count_buffer,
                0,
                bytemuck::cast_slice(&[4u32, 4, u32::MAX, 0]),
            );
            queue.write_buffer(&b.cell_ids, 0, bytemuck::cast_slice(&[41u32, 42, 43, 44]));
            queue.write_buffer(&b.next_cell_id, 0, bytemuck::bytes_of(&100u32));
            queue.write_buffer(
                &b.mode_cell_types,
                0,
                bytemuck::cast_slice(&[13u32, 19, 13, 19, 9]),
            );
            queue.write_buffer(
                &b.child_mode_indices,
                0,
                bytemuck::cast_slice(&[1i32, 0, 0, 1, 3, 2, 2, 3, 4, -1]),
            );
            queue.write_buffer(
                &b.mode_properties_v0,
                0,
                bytemuck::cast_slice(&[[0.2f32, 2., 50., 10.]; 5]),
            );
            queue.write_buffer(
                &b.mode_properties_v1,
                0,
                bytemuck::cast_slice(&[[2f32, 0., 0., 0.]; 5]),
            );
            queue.write_buffer(
                &b.mode_properties_v2,
                0,
                bytemuck::cast_slice(&[[-1f32, 0.5, 0., 0.]; 5]),
            );
            queue.write_buffer(
                &b.mode_properties_v9,
                0,
                bytemuck::cast_slice(&[
                    [0f32, 0., -1., -2.],
                    [8., 0., 0., 1.],
                    [0., 0., -1., -2.],
                    [8., 0., 2., 3.],
                    [0.; 4],
                ]),
            );
            queue.write_buffer(
                &b.mode_properties_v13,
                0,
                bytemuck::cast_slice(&[[7f32, 0., 0., 0.]; 5]),
            );
            queue.write_buffer(
                &b.mode_properties_v15,
                0,
                bytemuck::cast_slice(&[[11f32, 12., 13., 14.]; 5]),
            );
            queue.write_buffer(
                &colors,
                0,
                bytemuck::cast_slice(&[
                    [1f32, 0., 0., 1.],
                    [1., 0., 0., 1.],
                    [0., 0., 1., 1.],
                    [0., 0., 1., 1.],
                    [0., 1., 0., 1.],
                ]),
            );
            let mut params = [0u32; 16];
            params[1] = 3.5f32.to_bits();
            params[12] = 4;
            queue.write_buffer(&b.physics_params, 0, bytemuck::cast_slice(&params));
            queue.write_buffer(&a.signal_flags, 0, bytemuck::cast_slice(&[999u32; 64]));
            // Detection output: locked pair 0+1, midpoint (1,2,3), zero reserve.
            // No CPU genome is created or synced anywhere after this point.
            let events = [
                1u32,
                0,
                1,
                0,
                1,
                1f32.to_bits(),
                2f32.to_bits(),
                3f32.to_bits(),
                0,
            ];
            queue.write_buffer(
                &detector.merge_events_buffer,
                0,
                bytemuck::cast_slice(&events),
            );
            queue.write_buffer(&b.death_flags, 0, bytemuck::cast_slice(&[2u32, 2, 0, 0]));
            let mut bond = [0u32; 28];
            bond[1] = 3;
            bond[3] = 1;
            queue.write_buffer(&a.adhesion_connections, 0, bytemuck::cast_slice(&bond));
            queue.write_buffer(
                &a.adhesion_counts,
                0,
                bytemuck::cast_slice(&[1u32, 1, 0, 0]),
            );
            let mut indices = [-1i32; 80];
            indices[0] = 0;
            indices[60] = 0;
            queue.write_buffer(&a.cell_adhesion_indices, 0, bytemuck::cast_slice(&indices));
            let mut encoder = device.create_command_encoder(&Default::default());
            fusion.encode(
                &mut encoder,
                &insertion,
                &physics,
                &cached,
                0,
                a.max_connections,
            );
            queue.submit([encoder.finish()]);
            assert_eq!(read(&device, &queue, &a.adhesion_counts, 4, 8), [0, 1]);
            assert_eq!(
                read(&device, &queue, &a.cell_adhesion_indices, 60 * 4, 4),
                [u32::MAX],
                "former neighbor must not remain bonded to the offspring"
            );
            assert_eq!(read(&device, &queue, &b.genome_ids, 0, 4), [2]);
            assert_eq!(read(&device, &queue, &b.mode_indices, 0, 4), [8]);
            assert_eq!(read(&device, &queue, &b.cell_types, 0, 4), [10]);
            assert_eq!(read(&device, &queue, &b.death_flags, 0, 8), [0, 2]);
            assert_eq!(
                read(&device, &queue, &b.embryocyte_reserve_buffer, 0, 4),
                [0]
            );
            assert_eq!(read(&device, &queue, &b.cell_ids, 0, 4), [100]);
            assert_eq!(
                read(&device, &queue, &b.cell_count_buffer, 0, 8),
                [4, 4],
                "birth reuses a slot at full capacity; death scan later removes parent B"
            );
            for slot in 0..3 {
                assert_eq!(
                    read(&device, &queue, &b.position_and_mass[slot], 0, 16),
                    [
                        1f32.to_bits(),
                        2f32.to_bits(),
                        3f32.to_bits(),
                        1f32.to_bits()
                    ]
                );
                assert_eq!(
                    read(&device, &queue, &b.rotations[slot], 0, 16),
                    [0, 0, 0, 1f32.to_bits()]
                );
            }
            assert_eq!(
                read(&device, &queue, &b.child_mode_indices, 10 * 8, 8),
                [10, u32::MAX]
            );
            assert_eq!(
                read(&device, &queue, &b.mode_properties_v9, 9 * 16 + 8, 8),
                [8f32.to_bits(), 9f32.to_bits()]
            );
            assert_eq!(
                read(&device, &queue, &b.mode_properties_v13, 10 * 16, 4),
                [7f32.to_bits()]
            );
            assert_eq!(
                read(&device, &queue, &b.mode_properties_v15, 10 * 16, 4),
                [11f32.to_bits()]
            );
            assert_eq!(
                read(&device, &queue, &b.is_initial_mode, 8 * 4, 12),
                [1, 0, 0]
            );
            assert_eq!(
                read(&device, &queue, &colors, 8 * 16, 12),
                [0.5f32.to_bits(), 0, 0.5f32.to_bits()]
            );
            assert_eq!(read(&device, &queue, &a.signal_flags, 0, 64), vec![0; 16]);
            assert_eq!(read(&device, &queue, &fusion.parentage, 2 * 8, 8), [0, 1]);
            // Reproduce with the GPU-born genome: compatibility must not depend
            // on the two CPU-authored genomes, and birth must complete in one submit.
            queue.write_buffer(&b.cell_types, 0, bytemuck::cast_slice(&[13u32, 0, 13, 0]));
            queue.write_buffer(&b.genome_ids, 0, bytemuck::cast_slice(&[2u32, 0, 1, 0]));
            queue.write_buffer(&b.mode_indices, 0, bytemuck::cast_slice(&[9u32, 0, 4, 0]));
            queue.write_buffer(
                &b.development_addresses,
                2 * 16,
                bytemuck::cast_slice(&[300u32, 1, 1, 0]),
            );
            for slot in 0..3 {
                queue.write_buffer(
                    &b.position_and_mass[slot],
                    2 * 16,
                    bytemuck::cast_slice(&[1.2f32, 2., 3., 1.]),
                );
            }
            params[4] = 100f32.to_bits();
            params[8] = 1;
            params[10] = 16;
            queue.write_buffer(&b.physics_params, 0, bytemuck::cast_slice(&params));
            queue.write_buffer(&b.spatial_grid_counts, 0, bytemuck::bytes_of(&4u32));
            queue.write_buffer(
                &b.spatial_grid_cells,
                0,
                bytemuck::cast_slice(&[0u32, 1, 2, 3]),
            );
            let detector_physics = detector.create_physics_bind_group(
                &device,
                &b.physics_params,
                &b.position_and_mass[0],
                &b.velocity[0],
                &b.position_and_mass[0],
                &b.velocity[0],
                &b.cell_count_buffer,
            );
            let detector_cells = detector.create_cell_data_bind_group(
                &device,
                &b.cell_types,
                &b.death_flags,
                &b.development_addresses,
                &b.genome_ids,
                &b.mode_indices,
                &b.mode_properties_v13,
                &b.embryocyte_reserve_buffer,
                m.genome_meta_buffer(),
                &b.mode_cell_types,
            );
            let detector_grid = detector.create_spatial_bind_group(
                &device,
                &b.spatial_grid_counts,
                &b.spatial_grid_cells,
                &b.cell_grid_indices,
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            detector.clear_events(&mut encoder);
            detector.run(
                &mut encoder,
                &detector_physics,
                &detector_cells,
                &detector_grid,
                4,
            );
            fusion.encode(
                &mut encoder,
                &insertion,
                &physics,
                &cached,
                0,
                a.max_connections,
            );
            queue.submit([encoder.finish()]);
            assert_eq!(read(&device, &queue, &b.genome_ids, 0, 4), [3]);
            assert_eq!(read(&device, &queue, &b.cell_ids, 0, 4), [101]);
            assert_eq!(read(&device, &queue, &fusion.parentage, 3 * 8, 8), [2, 1]);
            assert_eq!(read(&device, &queue, &b.death_flags, 0, 12), [0, 2, 2]);

            // Allocation failures restore both parents, without changing genome,
            // reserves, or slot counts. Test both mode and genome exhaustion.
            for ring in [
                [0u32, 0, 4, 32, 32],
                [0, 0, super::super::mutation::GENOME_RING_CAPACITY, 14, 14],
            ] {
                queue.write_buffer(&m.genome_ring_state_buffer, 0, bytemuck::cast_slice(&ring));
                queue.write_buffer(
                    &detector.merge_events_buffer,
                    0,
                    bytemuck::cast_slice(&[1u32, 0, 2, 3, 1, 0, 0, 0, 0]),
                );
                queue.write_buffer(&b.death_flags, 0, bytemuck::cast_slice(&[2u32, 2, 2]));
                let mut encoder = device.create_command_encoder(&Default::default());
                fusion.encode(
                    &mut encoder,
                    &insertion,
                    &physics,
                    &cached,
                    0,
                    a.max_connections,
                );
                queue.submit([encoder.finish()]);
                assert_eq!(read(&device, &queue, &b.death_flags, 0, 12), [0, 2, 0]);
                assert_eq!(read(&device, &queue, &b.genome_ids, 0, 4), [3]);
                assert_eq!(read(&device, &queue, &b.cell_ids, 0, 4), [101]);
            }
            // Reclamation runs with radiation disabled and keeps authored genomes
            // immortal even when they currently have no living cells.
            m.set_radiation_level(0.0);
            queue.write_buffer(&b.death_flags, 0, bytemuck::cast_slice(&[0u32, 2, 2, 2]));
            let mut encoder = device.create_command_encoder(&Default::default());
            m.maintain_genomes(&device, &mut encoder);
            queue.submit([encoder.finish()]);
            assert_eq!(read(&device, &queue, m.genome_meta_buffer(), 0, 4), [2]);
            assert_eq!(read(&device, &queue, m.genome_meta_buffer(), 16, 4), [3]);
            assert_eq!(
                read(&device, &queue, m.genome_meta_buffer(), 2 * 16, 4),
                [0]
            );
            assert_eq!(
                read(&device, &queue, m.genome_meta_buffer(), 3 * 16, 4),
                [3]
            );
            // A free genome block is reusable even when the bump allocator is full.
            queue.write_buffer(
                &m.genome_ring_state_buffer,
                0,
                bytemuck::cast_slice(&[0u32, 1, 4, 32, 32]),
            );
            queue.write_buffer(&m.genome_free_ring_buffer, 0, bytemuck::bytes_of(&2u32));
            queue.write_buffer(&b.death_flags, 0, bytemuck::cast_slice(&[2u32, 2, 2, 2]));
            let mut encoder = device.create_command_encoder(&Default::default());
            fusion.encode(
                &mut encoder,
                &insertion,
                &physics,
                &cached,
                0,
                a.max_connections,
            );
            queue.submit([encoder.finish()]);
            assert_eq!(read(&device, &queue, &b.genome_ids, 0, 4), [2]);
            assert_eq!(read(&device, &queue, &b.mode_indices, 0, 4), [8]);
            assert_eq!(read(&device, &queue, &b.cell_ids, 0, 4), [102]);
            assert_eq!(read(&device, &queue, &b.death_flags, 0, 12), [0, 2, 2]);

            // World persistence retains raw GPU genes instead of reconstructing
            // an incomplete CPU Genome. Round-trip every inherited buffer.
            use crate::simulation::gpu_physics::genome_snapshot;
            let saved = genome_snapshot::capture(
                &device,
                &queue,
                &b,
                &a,
                &m,
                &colors,
                &emissive,
                [2, 2],
                2,
            )
            .unwrap();
            assert_eq!(saved.len(), 1);
            let encoded = ron::to_string(&saved).unwrap();
            let saved: Vec<genome_snapshot::SavedGpuGenome> = ron::from_str(&encoded).unwrap();
            for (_, buffer, width) in genome_mode_buffers(&b, &a, &colors, &emissive) {
                queue.write_buffer(
                    buffer,
                    8 * width as u64 * 4,
                    bytemuck::cast_slice(&vec![0u32; 3 * width as usize]),
                );
            }
            queue.write_buffer(
                m.genome_meta_buffer(),
                2 * 16,
                bytemuck::cast_slice(&[0u32; 4]),
            );
            genome_snapshot::restore(&queue, &b, &a, &mut m, &colors, &emissive, &saved, 2, 5)
                .unwrap();
            let restored =
                genome_snapshot::capture(&device, &queue, &b, &a, &m, &colors, &emissive, [2], 2)
                    .unwrap();
            assert_eq!(saved, restored);
            queue.write_buffer(&b.cell_types, 0, bytemuck::bytes_of(&0u32));
            genome_snapshot::restore_cell_properties(&device, &queue, &b, 4);
            assert_eq!(read(&device, &queue, &b.cell_types, 0, 4), [10]);
            assert_eq!(
                read(&device, &queue, &b.stiffnesses, 0, 4),
                [50f32.to_bits()]
            );
            let mut invalid = saved.clone();
            invalid[0].modes.remove("mode_properties_v13");
            assert!(genome_snapshot::restore(
                &queue, &b, &a, &mut m, &colors, &emissive, &invalid, 2, 5
            )
            .is_err());
        });
    }
}

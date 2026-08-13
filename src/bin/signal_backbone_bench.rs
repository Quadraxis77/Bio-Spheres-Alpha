use std::num::NonZeroU64;

use bio_spheres::simulation::signal_backbone_bench::{
    populate_workload, synthetic_forest, BoundaryInputKind, Channels, GpuCellTopology,
    GpuMicrotreeTopology, MacroStrategy, MicrotreeSchedule, SignalWorkload, SyntheticShape,
    CHANNEL_COUNT, TOPOLOGY_GENERATION_INITIAL,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use bio_spheres::simulation::gpu_physics::{
    GpuSignalProcessorConfig, GpuSignalProcessorState, GpuSignalSourceMeta,
    SignalBackboneValuePipeline,
};

const WORKGROUP_SIZE: u32 = 256;
const PARAM_ALIGNMENT: u64 = 256;
const PARITY_TOLERANCE: f32 = 0.05;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    cell_count: u32,
    block_count: u32,
    stride: u32,
    channel_group: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeneralTopologyParams {
    cell_count: u32,
    microtree_count: u32,
    generation: u32,
    block_size: u32,
    node_list_count: u32,
    child_list_count: u32,
    depth_offset_count: u32,
    depth_microtree_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeneralPropagationParams {
    cell_count: u32,
    microtree_count: u32,
    microtree_start: u32,
    microtree_dispatch_count: u32,
    channel_group: u32,
    block_size: u32,
    generation: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReductionParams {
    chunk_count: u32,
    input_kind: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProcessorParams {
    cell_count: u32,
    operation: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuReductionChunk {
    input_offset: u32,
    input_count: u32,
    output_slot: u32,
    target_parent_cell: u32,
    final_output: u32,
    padding: [u32; 3],
}

struct Args {
    cells: usize,
    warmup: usize,
    samples: usize,
    shape: SyntheticShape,
    workload: SignalWorkload,
    block_size: u32,
    strategy: Strategy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Strategy {
    Specialized,
    General,
    Integrated,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        cells: 200_000,
        warmup: 5,
        samples: 30,
        shape: SyntheticShape::Chain,
        workload: SignalWorkload::AllChannelsSparse,
        block_size: 64,
        strategy: Strategy::Specialized,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let value = |iter: &mut std::iter::Skip<std::env::Args>, name: &str| {
            iter.next()
                .ok_or_else(|| format!("missing value for {name}"))?
                .parse::<usize>()
                .map_err(|error| format!("invalid {name}: {error}"))
        };
        match arg.as_str() {
            "--cells" => args.cells = value(&mut iter, "--cells")?,
            "--warmup" => args.warmup = value(&mut iter, "--warmup")?,
            "--samples" => args.samples = value(&mut iter, "--samples")?,
            "--block-size" => {
                let parsed = value(&mut iter, "--block-size")? as u32;
                if !matches!(parsed, 64 | 96 | 128) {
                    return Err("--block-size must be 64, 96, or 128".into());
                }
                args.block_size = parsed;
            }
            "--shape" => {
                args.shape = match iter.next().as_deref() {
                    Some("chain") => SyntheticShape::Chain,
                    Some("star") => SyntheticShape::Star,
                    Some("balanced") => SyntheticShape::BalancedBinary,
                    Some("gameplay-mixed") => SyntheticShape::GameplayMixed,
                    Some("many-pairs") => SyntheticShape::ManyPairs,
                    Some("dense-mechanical") => SyntheticShape::DenseMechanicalSparseBackbone,
                    Some(value) => return Err(format!("unsupported linear shape: {value}")),
                    None => return Err("missing value for --shape".into()),
                }
            }
            "--strategy" => {
                args.strategy = match iter.next().as_deref() {
                    Some("specialized") => Strategy::Specialized,
                    Some("general") => Strategy::General,
                    Some("integrated") => Strategy::Integrated,
                    Some(value) => return Err(format!("unsupported strategy: {value}")),
                    None => return Err("missing value for --strategy".into()),
                }
            }
            "--workload" => {
                args.workload = match iter.next().as_deref() {
                    Some("silent") => SignalWorkload::SilentWithInvertedListener,
                    Some("one-source") => SignalWorkload::OneChannelOneSource,
                    Some("vec4") => SignalWorkload::OneVec4Group,
                    Some("all-sparse") => SignalWorkload::AllChannelsSparse,
                    Some("every-cell") => SignalWorkload::EveryCellEmits,
                    Some("cognocytes") => SignalWorkload::EveryCellCognocyte,
                    Some("memorocytes") => SignalWorkload::EveryCellMemorocyte,
                    Some("saturated") => SignalWorkload::SaturatedFanIn,
                    Some("cancellation") => SignalWorkload::SignedCancellationFanIn,
                    Some("oscillators") => SignalWorkload::ContinuousOscillators,
                    Some("heat") => SignalWorkload::HeatScreamAllChannels,
                    Some(value) => return Err(format!("unsupported workload: {value}")),
                    None => return Err("missing value for --workload".into()),
                }
            }
            "--help" | "-h" => {
                println!("signal-backbone-bench [--strategy specialized|general|integrated] [--cells N] [--warmup N] [--samples N] [--block-size 64|96|128] [--shape chain|star|balanced|many-pairs|gameplay-mixed|dense-mechanical] [--workload silent|one-source|vec4|all-sparse|every-cell|cognocytes|memorocytes|saturated|cancellation|oscillators|heat]");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    if args.cells == 0 || args.samples == 0 {
        return Err("--cells and --samples must be positive".into());
    }
    Ok(args)
}

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("signal backbone benchmark failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = parse_args()?;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|error| format!("no high-performance adapter: {error}"))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let timestamp_features =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    if !adapter.features().contains(timestamp_features) {
        return Err("adapter does not expose TIMESTAMP_QUERY".into());
    }

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Signal Backbone Phase 1 Benchmark Device"),
            required_features: timestamp_features,
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: limits
                    .max_storage_buffer_binding_size
                    .min(512 * 1024 * 1024),
                max_buffer_size: limits.max_buffer_size.min(512 * 1024 * 1024),
                ..wgpu::Limits::default()
            },
            memory_hints: Default::default(),
            trace: Default::default(),
            experimental_features: Default::default(),
        })
        .await
        .map_err(|error| format!("request_device failed: {error}"))?;

    let mut is_star = args.shape == SyntheticShape::Star && args.strategy == Strategy::Specialized;
    let mut is_balanced =
        args.shape == SyntheticShape::BalancedBinary && args.strategy == Strategy::Specialized;
    let mut is_gameplay_mixed =
        args.shape == SyntheticShape::GameplayMixed && args.strategy == Strategy::Specialized;
    println!(
        "phase=1 candidate={}",
        if args.strategy == Strategy::Integrated {
            "phase3_integrated_value_pipeline"
        } else if args.strategy == Strategy::General {
            "topology_general_microtree"
        } else if is_star {
            "hierarchical_star_reduction"
        } else if is_balanced {
            "balanced_depth_buckets"
        } else if is_gameplay_mixed {
            "gameplay_mixed_local_microtrees"
        } else {
            "blocked_bidirectional_chain"
        }
    );
    println!(
        "adapter_name={:?} backend={:?} device_type={:?} driver={:?} driver_info={:?}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
    println!(
        "vendor=0x{:04x} device=0x{:04x} max_buffer_size={} max_storage_binding={} max_workgroup_storage={}",
        info.vendor,
        info.device,
        limits.max_buffer_size,
        limits.max_storage_buffer_binding_size,
        limits.max_compute_workgroup_storage_size
    );
    println!(
        "rustc={} profile={} cells={} warmup={} samples={} shape={:?} workload={:?} block_size={}",
        option_env!("RUSTC_VERSION").unwrap_or("unknown"),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        args.cells,
        args.warmup,
        args.samples,
        args.shape,
        args.workload,
        args.block_size
    );

    let mut forest = synthetic_forest(args.shape, args.cells);
    populate_workload(&mut forest.sources, args.workload, 77);
    let cached = forest
        .cache()
        .map_err(|error| format!("cache: {error:?}"))?;
    let mut general_uses_optimized_kernel = false;
    if args.strategy == Strategy::General || args.strategy == Strategy::Integrated {
        validate_general_topology(&device, &queue, &cached, args.block_size).await?;
        let schedule = MicrotreeSchedule::build(&cached, args.block_size as usize);
        let macro_schedule = schedule.macro_schedule();
        let general_uses_path_scan = macro_schedule.strategy == MacroStrategy::PointerJumpingPath
            && macro_schedule.depth_buckets.len() > 1;
        if general_uses_path_scan {
            if cached.preorder.iter().copied().ne(0..args.cells as u32)
                || cached.parent.iter().enumerate().any(|(cell, &parent)| {
                    (cell == 0 && parent != u32::MAX) || (cell > 0 && parent != (cell - 1) as u32)
                })
            {
                return Err(
                    "pointer-jumping path requires topology-order remap for this fixture".into(),
                );
            }
            println!("general_macro_strategy=pointer_jumping_path topology_order=identity");
            general_uses_optimized_kernel = true;
        } else if macro_schedule.depth_buckets.len() == 2
            && macro_schedule.maximum_children > args.block_size
        {
            let canonical_star = cached.parent.first() == Some(&u32::MAX)
                && cached.parent.iter().skip(1).all(|parent| *parent == 0);
            if !canonical_star {
                return Err("high-degree reduction fixture is not canonical star order".into());
            }
            is_star = true;
            general_uses_optimized_kernel = true;
            println!("general_macro_strategy=hierarchical_high_degree_reduction");
        } else if macro_schedule.depth_buckets.len() > 1 {
            let canonical_heap = cached.parent.first() == Some(&u32::MAX)
                && cached
                    .parent
                    .iter()
                    .enumerate()
                    .skip(1)
                    .all(|(cell, parent)| *parent == ((cell - 1) / 2) as u32);
            if !canonical_heap {
                return Err("branching depth-bucket fixture is not canonical heap order".into());
            }
            is_balanced = true;
            general_uses_optimized_kernel = true;
            println!("general_macro_strategy=branching_depth_buckets");
        } else {
            let maximum_nodes = schedule
                .microtrees
                .iter()
                .map(|microtree| microtree.nodes.len())
                .max()
                .unwrap_or(0);
            if maximum_nodes <= 2 {
                general_uses_optimized_kernel = true;
                println!("general_macro_strategy=segmented_pair_blocks");
            } else if maximum_nodes <= 37 {
                is_gameplay_mixed = true;
                general_uses_optimized_kernel = true;
                println!("general_macro_strategy=bounded_shallow_local_trees");
            }
        }
    }
    let cpu_start = std::time::Instant::now();
    let expected = cached
        .propagate(&forest.sources)
        .map_err(|error| format!("CPU oracle: {error:?}"))?;
    println!(
        "cpu_oracle_ms={:.3}",
        cpu_start.elapsed().as_secs_f64() * 1000.0
    );

    if args.strategy == Strategy::Integrated {
        let integrated_expected = if args.workload == SignalWorkload::HeatScreamAllChannels {
            cached.propagate_with_local_overlay(&forest.sources, Some(&forest.sources))
                .map_err(|error| format!("heat CPU oracle: {error:?}"))?
        } else { expected };
        return run_integrated_value_pipeline(&device, &queue, &args, &forest.sources, &cached, &integrated_expected).await;
    }

    if args.strategy == Strategy::General && !general_uses_optimized_kernel {
        return run_general_propagation(
            &device,
            &queue,
            &args,
            &forest.sources,
            &cached,
            &expected,
        )
        .await;
    }

    let source_vec4 = flatten_sources(&forest.sources);
    let mut retention = vec![0.0; args.cells];
    for edge in &forest.edges {
        if edge.active
            && edge.bond_class
                == bio_spheres::simulation::signal_backbone_bench::BondClass::Backbone
        {
            let (low, high) = if edge.a < edge.b {
                (edge.a as usize, edge.b as usize)
            } else {
                (edge.b as usize, edge.a as usize)
            };
            if high == low + 1 {
                retention[high] = edge.edge_class.retention();
            }
        }
    }
    let cell_count = args.cells as u32;
    let block_count = cell_count.div_ceil(args.block_size);
    let macro_workgroups = block_count.div_ceil(WORKGROUP_SIZE);
    let rounds = if block_count <= 1 {
        0
    } else {
        u32::BITS - (block_count - 1).leading_zeros()
    };
    let star_partial_count = cell_count.div_ceil(WORKGROUP_SIZE);
    let mut star_reduce_inputs = Vec::new();
    let mut star_input_count = star_partial_count;
    while star_input_count > 1 {
        star_reduce_inputs.push(star_input_count);
        star_input_count = star_input_count.div_ceil(WORKGROUP_SIZE);
    }
    let mut balanced_levels = Vec::<(u32, u32)>::new();
    let mut level_start = 0u32;
    let mut level_width = 1u32;
    while level_start < cell_count {
        let count = level_width.min(cell_count - level_start);
        balanced_levels.push((level_start, count));
        level_start += count;
        level_width = level_width.saturating_mul(2);
    }
    let active_groups: &[usize] = match args.workload {
        SignalWorkload::SilentWithInvertedListener
        | SignalWorkload::OneChannelOneSource
        | SignalWorkload::OneVec4Group
        | SignalWorkload::ContinuousOscillators => &[0],
        _ => &[0, 1, 2, 3],
    };
    let active_group_mask = active_groups
        .iter()
        .fold(0u32, |mask, group| mask | (1u32 << group));
    // Dispatch the four active vec4 groups in the Z dimension for balanced
    // forests.  Each group has disjoint scratch (see BALANCED_GROUPED below),
    // which removes 111 tiny depth dispatches at 200k without changing the
    // deterministic per-group accumulation order.
    let balanced_all_groups = is_balanced && active_groups.len() == 4;
    let processor_dispatches = u32::from(matches!(
        args.workload,
        SignalWorkload::EveryCellCognocyte | SignalWorkload::EveryCellMemorocyte
    ));
    let dispatch_group_count = if balanced_all_groups {
        1
    } else {
        active_groups.len() as u32
    };
    let dispatches = dispatch_group_count
        * if is_star {
            star_reduce_inputs.len() as u32 + 2
        } else if is_balanced {
            1 + balanced_levels.len() as u32 * 2
        } else if is_gameplay_mixed {
            1
        } else {
            rounds + 2
        }
        + processor_dispatches;

    let source_bytes = (args.cells * 4 * std::mem::size_of::<[f32; 4]>()) as u64;
    let output_bytes = source_bytes;
    let retention_bytes = (args.cells * std::mem::size_of::<f32>()) as u64;
    let coeff_bytes = (block_count as usize * std::mem::size_of::<[f32; 2]>()) as u64;
    let bias_bytes = (block_count as usize * std::mem::size_of::<[f32; 8]>()) as u64;
    let tree_down_bytes = if is_balanced
        && (balanced_all_groups || args.workload == SignalWorkload::ContinuousOscillators)
    {
        source_bytes
    } else {
        (args.cells * std::mem::size_of::<[f32; 4]>()) as u64
    };

    let source_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Chain Bench Sources"),
        contents: bytemuck::cast_slice(&source_vec4),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let retention_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Chain Bench Retention"),
        contents: bytemuck::cast_slice(&retention),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let storage = |label: &'static str, size: u64| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let coeff_a = storage("Chain Bench Coeff A", coeff_bytes);
    let coeff_b = storage("Chain Bench Coeff B", coeff_bytes);
    let bias_a = storage("Chain Bench Bias A", bias_bytes);
    let bias_b = storage("Chain Bench Bias B", bias_bytes);
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Chain Bench Finalized"),
        size: output_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let tree_down_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Balanced Tree Down Messages / Processor State"),
        size: tree_down_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let external_processor = args.strategy == Strategy::General
        && is_balanced
        && args.workload == SignalWorkload::EveryCellMemorocyte;
    let external_processor_state = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Optimized Persistent Processor State"),
        size: tree_down_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params_per_group = (rounds as usize)
        .max(star_reduce_inputs.len())
        .max(balanced_levels.len() * 2)
        .saturating_add(2);
    let mut params_bytes = vec![0u8; 4 * params_per_group * PARAM_ALIGNMENT as usize];
    let param_offset = |group: usize, phase: usize| {
        ((group * params_per_group + phase) as u64 * PARAM_ALIGNMENT) as usize
    };
    for group in 0..4 {
        let write_params = |bytes: &mut [u8], phase: usize, stride: u32, phase_block_count: u32| {
            let params = Params {
                cell_count,
                block_count: phase_block_count,
                stride,
                channel_group: if balanced_all_groups && group == 0 {
                    u32::MAX
                } else {
                    group as u32
                },
            };
            let offset = param_offset(group, phase);
            bytes[offset..offset + std::mem::size_of::<Params>()]
                .copy_from_slice(bytemuck::bytes_of(&params));
        };
        write_params(&mut params_bytes, 0, 0, block_count);
        if is_star {
            for (round, &input_count) in star_reduce_inputs.iter().enumerate() {
                write_params(&mut params_bytes, round + 1, 0, input_count);
            }
            write_params(&mut params_bytes, star_reduce_inputs.len() + 1, 0, 1);
        } else if is_balanced {
            for (phase, &(start, count)) in balanced_levels.iter().rev().enumerate() {
                write_params(&mut params_bytes, phase + 1, count, start);
            }
            for (level, &(start, count)) in balanced_levels.iter().enumerate() {
                write_params(
                    &mut params_bytes,
                    balanced_levels.len() + level + 1,
                    count,
                    start,
                );
            }
        } else {
            for round in 0..rounds as usize {
                write_params(&mut params_bytes, round + 1, 1u32 << round, block_count);
            }
            write_params(&mut params_bytes, rounds as usize + 1, 0, block_count);
        }
    }
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Chain Bench Dynamic Params"),
        contents: &params_bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Chain Bench Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                },
                count: None,
            },
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
            storage_entry(5, true),
            storage_entry(6, false),
            storage_entry(7, false),
            storage_entry(8, false),
        ],
    });
    let make_bind_group = |label: &'static str,
                           coeff_in: &wgpu::Buffer,
                           coeff_out: &wgpu::Buffer,
                           bias_in: &wgpu::Buffer,
                           bias_out: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: retention_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: coeff_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: coeff_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bias_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bias_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: tree_down_buffer.as_entire_binding(),
                },
            ],
        })
    };
    let bind_ab = make_bind_group("Chain Bench A to B", &coeff_a, &coeff_b, &bias_a, &bias_b);
    let bind_ba = make_bind_group("Chain Bench B to A", &coeff_b, &coeff_a, &bias_b, &bias_a);

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let block_size_text = args.block_size.to_string();
    let shader_source = include_str!("../../shaders/signal_backbone_chain_bench.wgsl")
        .replace(
            "const BLOCK_SIZE: u32 = 128u;",
            &format!("const BLOCK_SIZE: u32 = {}u;", args.block_size),
        )
        .replace("array<f32, 128>", &format!("array<f32, {block_size_text}>"))
        .replace(
            "array<vec4<f32>, 128>",
            &format!("array<vec4<f32>, {block_size_text}>"),
        )
        .replace(
            "@workgroup_size(128)",
            &format!("@workgroup_size({block_size_text})"),
        )
        .replace(
            "const BALANCED_COMPENSATED: bool = true;",
            if args.workload == SignalWorkload::ContinuousOscillators {
                "const BALANCED_COMPENSATED: bool = true;"
            } else {
                "const BALANCED_COMPENSATED: bool = false;"
            },
        )
        .replace(
            "const BALANCED_GROUPED: bool = false;",
            if balanced_all_groups {
                "const BALANCED_GROUPED: bool = true;"
            } else {
                "const BALANCED_GROUPED: bool = false;"
            },
        );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Signal Backbone Chain Benchmark Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Signal Backbone Chain Benchmark Pipeline Layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = |entry_point: &'static str, label: &'static str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let initialize = pipeline("summarize_block", "Chain Bench Summarize Blocks");
    let scan = pipeline("scan_macro", "Chain Bench Scan Macro Forest");
    let finalize = pipeline("finalize_block", "Chain Bench Finalize Blocks");
    let star_partial = pipeline("star_partial", "Star Bench Partial Reduction");
    let star_reduce = pipeline("star_reduce", "Star Bench Hierarchical Reduction");
    let star_finalize = pipeline("star_finalize", "Star Bench Finalize");
    let balanced_initialize = pipeline("balanced_initialize", "Balanced Bench Initialize");
    let balanced_up = pipeline("balanced_up", "Balanced Bench Upward Depth");
    let balanced_down = pipeline("balanced_down", "Balanced Bench Downward Depth");
    let gameplay_solve = pipeline("gameplay_solve", "Gameplay Mixed Local Solve");
    let cognocyte_bench = pipeline("cognocyte_bench", "Every Cell Cognocyte");
    let memorocyte_bench = pipeline("memorocyte_bench", "Every Cell Memorocyte");
    let external_processor_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Optimized External Processor Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ProcessorParams>() as u64
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
    let external_processor_params = ProcessorParams {
        cell_count,
        operation: 0,
        padding: [0; 2],
    };
    let external_processor_params_buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Optimized External Processor Params"),
            contents: bytemuck::bytes_of(&external_processor_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let external_processor_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Optimized External Processor Bind"),
        layout: &external_processor_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: external_processor_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: external_processor_state.as_entire_binding(),
            },
        ],
    });
    let external_processor_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Optimized External Processor Shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/signal_backbone_processor_bench.wgsl").into(),
        ),
    });
    let external_processor_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Optimized External Processor Pipeline Layout"),
            bind_group_layouts: &[&external_processor_layout],
            push_constant_ranges: &[],
        });
    let external_processor_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Optimized External Processor Pipeline"),
            layout: Some(&external_processor_pipeline_layout),
            module: &external_processor_shader,
            entry_point: Some("process_cells"),
            compilation_options: Default::default(),
            cache: None,
        });
    if let Some(error) = device.pop_error_scope().await {
        return Err(format!("WGSL/pipeline validation failed: {error}"));
    }
    println!("wgsl_pipeline_creation=pass");

    macro_rules! dispatch_processor {
        ($pass:expr, $group:expr) => {
            if $group == 0 {
                let processor_offset = param_offset(0, 0) as u32;
                match args.workload {
                    SignalWorkload::EveryCellCognocyte => {
                        $pass.set_pipeline(&cognocyte_bench);
                        $pass.set_bind_group(0, &bind_ab, &[processor_offset]);
                        $pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
                    }
                    SignalWorkload::EveryCellMemorocyte => {
                        if !external_processor {
                            $pass.set_pipeline(&memorocyte_bench);
                            $pass.set_bind_group(0, &bind_ab, &[processor_offset]);
                            $pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
                        }
                    }
                    _ => {}
                }
            }
        };
    }

    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Chain Bench Timestamp Query"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let query_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Chain Bench Query Resolve"),
        size: 256,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let query_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Chain Bench Query Readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let encode_tick = |encoder: &mut wgpu::CommandEncoder, timed: bool| {
        let timestamp_writes = timed.then_some(wgpu::ComputePassTimestampWrites {
            query_set: &query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Signal Backbone Chain Tick"),
            timestamp_writes,
        });
        for &group in active_groups {
            if is_gameplay_mixed {
                let offset = param_offset(group, 0) as u32;
                pass.set_pipeline(&gameplay_solve);
                pass.set_bind_group(0, &bind_ab, &[offset]);
                pass.dispatch_workgroups(cell_count.div_ceil(37), 1, 1);
                dispatch_processor!(pass, group);
                continue;
            }
            if is_balanced {
                if balanced_all_groups && group != 0 {
                    continue;
                }
                let dispatch_z = if balanced_all_groups { 4 } else { 1 };
                let offset = param_offset(group, 0) as u32;
                pass.set_pipeline(&balanced_initialize);
                pass.set_bind_group(0, &bind_ab, &[offset]);
                pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, dispatch_z);

                for (phase, &(_, count)) in balanced_levels.iter().rev().enumerate() {
                    let phase_offset = param_offset(group, phase + 1) as u32;
                    pass.set_pipeline(&balanced_up);
                    pass.set_bind_group(0, &bind_ab, &[phase_offset]);
                    pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, dispatch_z);
                }
                for (level, &(_, count)) in balanced_levels.iter().enumerate() {
                    let phase_offset =
                        param_offset(group, balanced_levels.len() + level + 1) as u32;
                    pass.set_pipeline(&balanced_down);
                    pass.set_bind_group(0, &bind_ab, &[phase_offset]);
                    pass.dispatch_workgroups(count.div_ceil(WORKGROUP_SIZE), 1, dispatch_z);
                }
                dispatch_processor!(pass, group);
                continue;
            }
            if is_star {
                let offset = param_offset(group, 0) as u32;
                pass.set_pipeline(&star_partial);
                pass.set_bind_group(0, &bind_ba, &[offset]);
                pass.dispatch_workgroups(star_partial_count, 1, 1);

                for (round, &input_count) in star_reduce_inputs.iter().enumerate() {
                    let round_offset = param_offset(group, round + 1) as u32;
                    pass.set_pipeline(&star_reduce);
                    if round % 2 == 0 {
                        pass.set_bind_group(0, &bind_ab, &[round_offset]);
                    } else {
                        pass.set_bind_group(0, &bind_ba, &[round_offset]);
                    }
                    pass.dispatch_workgroups(input_count.div_ceil(WORKGROUP_SIZE), 1, 1);
                }

                let final_offset = param_offset(group, star_reduce_inputs.len() + 1) as u32;
                pass.set_pipeline(&star_finalize);
                if star_reduce_inputs.len() % 2 == 0 {
                    pass.set_bind_group(0, &bind_ab, &[final_offset]);
                } else {
                    pass.set_bind_group(0, &bind_ba, &[final_offset]);
                }
                pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
                dispatch_processor!(pass, group);
                continue;
            }
            let offset = param_offset(group, 0) as u32;
            pass.set_pipeline(&initialize);
            pass.set_bind_group(0, &bind_ba, &[offset]);
            pass.dispatch_workgroups(block_count, 1, 1);

            for round in 0..rounds as usize {
                let round_offset = param_offset(group, round + 1) as u32;
                pass.set_pipeline(&scan);
                if round % 2 == 0 {
                    pass.set_bind_group(0, &bind_ab, &[round_offset]);
                } else {
                    pass.set_bind_group(0, &bind_ba, &[round_offset]);
                }
                pass.dispatch_workgroups(macro_workgroups, 1, 1);
            }

            let final_offset = param_offset(group, rounds as usize + 1) as u32;
            pass.set_pipeline(&finalize);
            if rounds % 2 == 0 {
                pass.set_bind_group(0, &bind_ab, &[final_offset]);
            } else {
                pass.set_bind_group(0, &bind_ba, &[final_offset]);
            }
            pass.dispatch_workgroups(block_count, 1, 1);
            dispatch_processor!(pass, group);
        }
        if external_processor {
            pass.set_pipeline(&external_processor_pipeline);
            pass.set_bind_group(0, &external_processor_bind, &[]);
            pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
    };

    for _ in 0..args.warmup {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Chain Bench Warmup Encoder"),
        });
        encode_tick(&mut encoder, false);
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| format!("warmup wait failed: {error:?}"))?;
    }

    let timestamp_period = queue.get_timestamp_period() as f64;
    let mut timings_ms = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Chain Bench Timed Encoder"),
        });
        encode_tick(&mut encoder, true);
        encoder.resolve_query_set(&query_set, 0..2, &query_resolve, 0);
        encoder.copy_buffer_to_buffer(&query_resolve, 0, &query_readback, 0, 16);
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| format!("timed wait failed: {error:?}"))?;
        let timestamps: Vec<u64> = map_read(&device, &query_readback)?;
        let elapsed = timestamps[1].saturating_sub(timestamps[0]) as f64 * timestamp_period;
        timings_ms.push(elapsed / 1_000_000.0);
    }

    let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Chain Bench Value Readback"),
        size: output_bytes.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Chain Bench Value Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &output_readback, 0, output_bytes);
    let submission = queue.submit(std::iter::once(encoder.finish()));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map_err(|error| format!("value readback wait failed: {error:?}"))?;
    let gpu_vec4: Vec<[f32; 4]> = map_read(&device, &output_readback)?;
    let (max_error, mismatches) = compare_results(&expected, &gpu_vec4);
    let published_mismatches = compare_published_results(&expected, &gpu_vec4);
    let mut processor_mismatches = 0_usize;
    if matches!(
        args.workload,
        SignalWorkload::EveryCellCognocyte | SignalWorkload::EveryCellMemorocyte
    ) {
        let processor_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Path Processor Readback"),
            size: tree_down_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Path Processor Readback Encoder"),
        });
        let processor_source = if external_processor {
            &external_processor_state
        } else {
            &tree_down_buffer
        };
        encoder.copy_buffer_to_buffer(processor_source, 0, &processor_readback, 0, tree_down_bytes);
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| format!("path processor readback wait failed: {error:?}"))?;
        let processor_actual = map_read::<[f32; 4]>(&device, &processor_readback)?;
        let mut processor_max_error = 0.0_f32;
        for cell in 0..args.cells {
            let expected_output = if args.workload == SignalWorkload::EveryCellCognocyte {
                (expected[cell][0] * expected[cell][1] / 1000.0).clamp(-1000.0, 1000.0)
            } else {
                let effective_rate = 1.0 - 0.5_f32.powf(1.0 / 15.0);
                let ticks = args.warmup + args.samples;
                expected[cell][0] * (1.0 - (1.0 - effective_rate).powi(ticks as i32))
            };
            let actual_output = if args.workload == SignalWorkload::EveryCellCognocyte {
                processor_actual[cell][1]
            } else {
                processor_actual[cell][0]
            };
            let error = (actual_output - expected_output).abs();
            processor_max_error = processor_max_error.max(error);
            if !actual_output.is_finite() || error > PARITY_TOLERANCE {
                processor_mismatches += 1;
            }
        }
        println!(
            "processor_parity_tolerance={} processor_max_abs_error={:.6} processor_mismatches={}",
            PARITY_TOLERANCE, processor_max_error, processor_mismatches
        );
    }

    timings_ms.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        let index = ((timings_ms.len() - 1) as f64 * p).round() as usize;
        timings_ms[index]
    };
    let workspace_bytes = source_bytes
        + output_bytes
        + retention_bytes
        + 2 * coeff_bytes
        + 2 * bias_bytes
        + tree_down_bytes
        + u64::from(external_processor) * tree_down_bytes;
    println!(
        "block_size={} microtrees={} macro_rounds={} active_group_mask=0x{:x} workspace_bytes={} workspace_mib={:.3} benchmark_readback_bytes={} dispatches={}",
        args.block_size,
        block_count,
        rounds,
        active_group_mask,
        workspace_bytes,
        workspace_bytes as f64 / 1_048_576.0,
        output_bytes + 16,
        dispatches
    );
    println!(
        "gpu_ms_p50={:.4} gpu_ms_p95={:.4} gpu_ms_worst={:.4}",
        percentile(0.50),
        percentile(0.95),
        timings_ms[timings_ms.len() - 1]
    );
    println!(
        "parity_tolerance={} parity_max_abs_error={:.6} parity_mismatches={}",
        PARITY_TOLERANCE, max_error, mismatches
    );
    println!("published_integer_mismatches={published_mismatches}");
    let memory_pass = workspace_bytes <= 64 * 1024 * 1024;
    let correctness_pass = mismatches == 0 && processor_mismatches == 0;
    let discrete_timing_pass = percentile(0.95) < 2.0;
    println!(
        "gate_memory={} gate_correctness={} gate_discrete_gpu_p95={} phase1_gate=INCOMPLETE_MATRIX",
        pass_fail(memory_pass),
        pass_fail(correctness_pass),
        pass_fail(discrete_timing_pass),
    );

    if !memory_pass || !correctness_pass {
        return Err("candidate failed a non-timing acceptance gate".into());
    }
    Ok(())
}

fn flatten_sources(sources: &[Channels]) -> Vec<[f32; 4]> {
    let mut flattened = Vec::with_capacity(sources.len() * 4);
    for channels in sources {
        for group in 0..4 {
            flattened.push([
                channels[group * 4],
                channels[group * 4 + 1],
                channels[group * 4 + 2],
                channels[group * 4 + 3],
            ]);
        }
    }
    flattened
}

fn compare_results(expected: &[Channels], actual: &[[f32; 4]]) -> (f32, usize) {
    let mut max_error = 0.0f32;
    let mut mismatches = 0usize;
    for (cell, expected_channels) in expected.iter().enumerate() {
        for channel in 0..CHANNEL_COUNT {
            let actual_value = actual[cell * 4 + channel / 4][channel % 4];
            let error = (actual_value - expected_channels[channel]).abs();
            max_error = max_error.max(error);
            if !actual_value.is_finite() || error > PARITY_TOLERANCE {
                mismatches += 1;
            }
        }
    }
    (max_error, mismatches)
}

fn compare_published_results(expected: &[Channels], actual: &[[f32; 4]]) -> usize {
    let mut mismatches = 0;
    for (cell, expected_channels) in expected.iter().enumerate() {
        for channel in 0..CHANNEL_COUNT {
            let expected_packed = expected_channels[channel].round().clamp(-1000.0, 1000.0) as i32;
            let actual_value = actual[cell * 4 + channel / 4][channel % 4];
            let actual_packed = actual_value.round().clamp(-1000.0, 1000.0) as i32;
            if !actual_value.is_finite() || actual_packed != expected_packed {
                mismatches += 1;
            }
        }
    }
    mismatches
}

fn active_groups(workload: SignalWorkload) -> &'static [usize] {
    match workload {
        SignalWorkload::SilentWithInvertedListener
        | SignalWorkload::OneChannelOneSource
        | SignalWorkload::OneVec4Group
        | SignalWorkload::ContinuousOscillators => &[0],
        _ => &[0, 1, 2, 3],
    }
}

async fn run_integrated_value_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    args: &Args,
    sources: &[Channels],
    cached: &bio_spheres::simulation::signal_backbone_bench::CachedForest,
    expected: &[Channels],
) -> Result<(), String> {
    let cell_count = args.cells as u32;
    let field_bytes = args.cells as u64 * 4 * 16;
    let nutrients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Integrated Signal Nutrients"),
        contents: bytemuck::cast_slice(&vec![100_000_000i32; args.cells]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let packed_public = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Integrated Signal Public Field"),
        size: args.cells as u64 * CHANNEL_COUNT as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut pipeline = SignalBackboneValuePipeline::new(device, cell_count, &nutrients, &packed_public);
    pipeline.set_static_forest(device, cached);

    let requested = flatten_sources(sources);
    let metadata = sources.iter().enumerate().map(|(cell, channels)| GpuSignalSourceMeta {
        identity: cell as u32,
        flags: 1 | if args.workload == SignalWorkload::HeatScreamAllChannels { 2 } else { 0 },
        requested_absolute: channels.iter().map(|value| value.abs()).sum(),
        _padding: 0,
    }).collect::<Vec<_>>();
    let configs = match args.workload {
        SignalWorkload::EveryCellCognocyte => (0..args.cells).map(|cell| {
            GpuSignalProcessorConfig::new(1, 0, 0, 1, (cell % 16) as u32, 1, 0, 0.0, 0.0, 0.0)
        }).collect(),
        SignalWorkload::EveryCellMemorocyte => (0..args.cells).map(|cell| {
            GpuSignalProcessorConfig::new(2, 0, 0, 0, (cell % 16) as u32, 1, 0, 0.5, 0.0, 0.0)
        }).collect(),
        _ => vec![GpuSignalProcessorConfig::default(); args.cells],
    };
    queue.write_buffer(&pipeline.source_field, 0, bytemuck::cast_slice(&requested));
    queue.write_buffer(&pipeline.source_meta, 0, bytemuck::cast_slice(&metadata));
    queue.write_buffer(&pipeline.processor_config, 0, bytemuck::cast_slice(&configs));
    let active_group_mask = active_groups(args.workload)
        .iter().fold(0u32, |mask, group| mask | (1 << group));
    pipeline.write_params(queue, 0, cell_count, 77, active_group_mask);

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Integrated Signal Stage Timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 4,
    });
    let query_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Integrated Signal Query Resolve"), size: 256,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let query_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Integrated Signal Query Readback"), size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Integrated Signal Output Readback"), size: field_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let processor_state_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Integrated Signal Prior Processor State Readback"),
        size: args.cells as u64 * std::mem::size_of::<GpuSignalProcessorState>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    if let Some(error) = device.pop_error_scope().await {
        return Err(format!("integrated WGSL/pipeline validation failed: {error}"));
    }
    println!("integrated_wgsl_pipeline_creation=pass");

    let encode_tick = |encoder: &mut wgpu::CommandEncoder, timed: bool| {
        if timed { encoder.write_timestamp(&query_set, 0); }
        pipeline.encode_sources(encoder, 0, cell_count);
        if timed { encoder.write_timestamp(&query_set, 1); }
        pipeline.encode_propagation(encoder, active_group_mask, 0);
        if timed { encoder.write_timestamp(&query_set, 2); }
        pipeline.encode_finalize_and_processors(
            encoder,
            0,
            cell_count,
            matches!(args.workload, SignalWorkload::EveryCellCognocyte | SignalWorkload::EveryCellMemorocyte),
        );
        if timed { encoder.write_timestamp(&query_set, 3); }
    };
    for _ in 0..args.warmup {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Integrated Signal Warmup") });
        encode_tick(&mut encoder, false);
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
            .map_err(|error| format!("integrated warmup wait: {error:?}"))?;
    }

    let timestamp_period = queue.get_timestamp_period() as f64;
    let mut source_ms = Vec::with_capacity(args.samples);
    let mut propagation_ms = Vec::with_capacity(args.samples);
    let mut processor_ms = Vec::with_capacity(args.samples);
    let mut total_ms = Vec::with_capacity(args.samples);
    for sample in 0..args.samples {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Integrated Signal Timed") });
        if sample + 1 == args.samples && matches!(args.workload, SignalWorkload::EveryCellCognocyte | SignalWorkload::EveryCellMemorocyte) {
            pipeline.copy_processor_state_to(&mut encoder, &processor_state_readback, cell_count);
        }
        encode_tick(&mut encoder, true);
        encoder.resolve_query_set(&query_set, 0..4, &query_resolve, 0);
        encoder.copy_buffer_to_buffer(&query_resolve, 0, &query_readback, 0, 32);
        if sample + 1 == args.samples {
            encoder.copy_buffer_to_buffer(&pipeline.finalized_field, 0, &output_readback, 0, field_bytes);
        }
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
            .map_err(|error| format!("integrated timed wait: {error:?}"))?;
        let timestamp = map_read::<u64>(device, &query_readback)?;
        let elapsed = |from: usize, to: usize| {
            timestamp[to].saturating_sub(timestamp[from]) as f64 * timestamp_period / 1_000_000.0
        };
        source_ms.push(elapsed(0, 1));
        propagation_ms.push(elapsed(1, 2));
        processor_ms.push(elapsed(2, 3));
        total_ms.push(elapsed(0, 3));
    }
    for timings in [&mut source_ms, &mut propagation_ms, &mut processor_ms, &mut total_ms] {
        timings.sort_by(f64::total_cmp);
    }
    let percentile = |values: &[f64], fraction: f64| {
        // Standard nearest-rank percentile (1-based rank). For 20 samples,
        // p95 is the 19th ordered sample rather than the maximum.
        values[((values.len() as f64 * fraction).ceil() as usize).saturating_sub(1)]
    };
    println!(
        "integrated_source_ms_p50={:.4} p95={:.4} propagation_ms_p50={:.4} p95={:.4} processor_publication_ms_p50={:.4} p95={:.4}",
        percentile(&source_ms, 0.5), percentile(&source_ms, 0.95),
        percentile(&propagation_ms, 0.5), percentile(&propagation_ms, 0.95),
        percentile(&processor_ms, 0.5), percentile(&processor_ms, 0.95),
    );
    println!(
        "gpu_ms_p50={:.4} gpu_ms_p95={:.4} gpu_ms_worst={:.4}",
        percentile(&total_ms, 0.5), percentile(&total_ms, 0.95), total_ms[total_ms.len() - 1]
    );

    let actual = map_read::<[f32; 4]>(device, &output_readback)?;
    // Processor workloads are intentionally stateful. Use the state captured
    // immediately before the final timed tick to build its exact CPU source
    // snapshot; this diagnostic copy is outside the timestamp interval and is
    // not part of the gameplay command path.
    let temporal_expected;
    let expected = if matches!(args.workload, SignalWorkload::EveryCellCognocyte | SignalWorkload::EveryCellMemorocyte) {
        let prior = map_read::<GpuSignalProcessorState>(device, &processor_state_readback)?;
        let mut tick_sources = sources.to_vec();
        for (cell, state) in prior.iter().enumerate() {
            let channel = state.output_channel.min(15) as usize;
            tick_sources[cell][channel] =
                (tick_sources[cell][channel] + state.output).clamp(-1000.0, 1000.0);
        }
        temporal_expected = cached.propagate(&tick_sources)
            .map_err(|error| format!("temporal CPU oracle: {error:?}"))?;
        &temporal_expected
    } else {
        expected
    };
    let mut max_error = 0.0f32;
    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, usize, f32, f32)> = None;
    for cell in 0..args.cells {
        for channel in 0..CHANNEL_COUNT {
            let error = (actual[cell * 4 + channel / 4][channel % 4] - expected[cell][channel]).abs();
            max_error = max_error.max(error);
            if error > PARITY_TOLERANCE {
                first_mismatch.get_or_insert((cell, channel, expected[cell][channel], actual[cell * 4 + channel / 4][channel % 4]));
                mismatches += 1;
            }
        }
    }
    let workspace = pipeline.allocated_bytes();
    let dispatches = pipeline.tick_dispatch_count(matches!(
        args.workload,
        SignalWorkload::EveryCellCognocyte | SignalWorkload::EveryCellMemorocyte
    ));
    println!(
        "integrated_workspace_bytes={} workspace_mib={:.3} dispatches={} queue_submissions_per_tick=1 cpu_readbacks_per_tick=0",
        workspace, workspace as f64 / (1024.0 * 1024.0), dispatches,
    );
    println!(
        "parity_tolerance={} parity_max_abs_error={:.6} parity_mismatches={}",
        PARITY_TOLERANCE, max_error, mismatches,
    );
    if let Some((cell, channel, cpu, gpu)) = first_mismatch {
        println!("parity_first_mismatch_cell={cell} channel={channel} cpu={cpu:.6} gpu={gpu:.6}");
    }
    let memory_pass = workspace <= 64 * 1024 * 1024;
    let timing_pass = percentile(&total_ms, 0.95) < 2.0;
    let correctness_pass = mismatches == 0;
    println!(
        "gate_memory={} gate_correctness={} gate_discrete_gpu_p95={} phase3_integrated_gate={}",
        if memory_pass { "PASS" } else { "FAIL" },
        if correctness_pass { "PASS" } else { "FAIL" },
        if timing_pass { "PASS" } else { "FAIL" },
        if memory_pass && correctness_pass && timing_pass { "PASS" } else { "FAIL" },
    );
    Ok(())
}

async fn run_general_propagation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    args: &Args,
    sources: &[Channels],
    cache: &bio_spheres::simulation::signal_backbone_bench::CachedForest,
    expected: &[Channels],
) -> Result<(), String> {
    let schedule = MicrotreeSchedule::build(cache, args.block_size as usize);
    let macro_schedule = schedule.macro_schedule();
    if macro_schedule.strategy == MacroStrategy::PointerJumpingPath
        && macro_schedule.depth_buckets.len() > 1
    {
        return Err("general pointer-jumping path solve is not connected yet".into());
    }
    let upload = schedule.flatten_for_gpu(cache, TOPOLOGY_GENERATION_INITIAL);
    let reductions = schedule.boundary_reduction_schedule(cache);
    let source_vec4 = flatten_sources(sources);
    let cell_count = sources.len() as u32;
    let microtree_count = upload.microtrees.len() as u32;
    let value_bytes = sources.len() as u64 * std::mem::size_of::<[f32; 4]>() as u64;
    let field_bytes = value_bytes * 4;
    let microtree_value_bytes = microtree_count as u64 * std::mem::size_of::<[f32; 4]>() as u64;

    let init_buffer = |label: &'static str, contents: &[u8], usage: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    };
    let storage = |label: &'static str, size: u64, extra: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    };
    let source_buffer = init_buffer(
        "General Sources",
        bytemuck::cast_slice(&source_vec4),
        wgpu::BufferUsages::STORAGE,
    );
    let cells_buffer = init_buffer(
        "General Cells",
        bytemuck::cast_slice::<GpuCellTopology, u8>(&upload.cells),
        wgpu::BufferUsages::STORAGE,
    );
    let microtrees_buffer = init_buffer(
        "General Microtrees",
        bytemuck::cast_slice::<GpuMicrotreeTopology, u8>(&upload.microtrees),
        wgpu::BufferUsages::STORAGE,
    );
    let node_list_buffer = init_buffer(
        "General Node List",
        bytemuck::cast_slice(&upload.node_list),
        wgpu::BufferUsages::STORAGE,
    );
    let boundary_buffer = storage(
        "General Boundary Or Down",
        value_bytes,
        wgpu::BufferUsages::COPY_DST,
    );
    let subtree_buffer = storage("General Subtree", value_bytes, wgpu::BufferUsages::empty());
    let microtree_up_buffer = storage(
        "General Microtree Up",
        microtree_value_bytes,
        wgpu::BufferUsages::empty(),
    );
    let finalized_buffer = storage(
        "General Finalized",
        field_bytes,
        wgpu::BufferUsages::COPY_SRC,
    );
    let has_processor = matches!(
        args.workload,
        SignalWorkload::EveryCellCognocyte | SignalWorkload::EveryCellMemorocyte
    );
    let processor_state_buffer = storage(
        "General Processor State",
        value_bytes,
        wgpu::BufferUsages::COPY_SRC,
    );

    let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let main_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("General Propagation Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        std::mem::size_of::<GeneralPropagationParams>() as u64,
                    ),
                },
                count: None,
            },
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, false),
            storage_entry(6, false),
            storage_entry(7, false),
            storage_entry(8, false),
        ],
    });

    let make_main_bind = |params: GeneralPropagationParams| {
        let params_buffer = init_buffer(
            "General Propagation Params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
        );
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("General Propagation Bind Group"),
            layout: &main_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cells_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: microtrees_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: node_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: boundary_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: subtree_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: microtree_up_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: finalized_buffer.as_entire_binding(),
                },
            ],
        })
    };

    let mut depth_ranges = Vec::with_capacity(macro_schedule.depth_buckets.len());
    for bucket in &macro_schedule.depth_buckets {
        let start = bucket.first().copied().unwrap_or(0);
        depth_ranges.push((start, bucket.len() as u32));
    }
    let mut main_binds = Vec::with_capacity(active_groups(args.workload).len());
    for &group in active_groups(args.workload) {
        let mut group_binds = Vec::with_capacity(depth_ranges.len());
        for &(start, count) in &depth_ranges {
            group_binds.push(make_main_bind(GeneralPropagationParams {
                cell_count,
                microtree_count,
                microtree_start: start,
                microtree_dispatch_count: count,
                channel_group: group as u32,
                block_size: args.block_size,
                generation: TOPOLOGY_GENERATION_INITIAL,
                padding: 0,
            }));
        }
        main_binds.push(group_binds);
    }

    let max_partial_count = reductions
        .iter()
        .flat_map(|depth| depth.passes.iter())
        .map(|pass| pass.scratch_output_count)
        .max()
        .unwrap_or(0) as u64;
    let partial_bytes = max_partial_count.max(1) * std::mem::size_of::<[f32; 4]>() as u64;
    let partial_a = storage(
        "General Reduction Partial A",
        partial_bytes,
        wgpu::BufferUsages::empty(),
    );
    let partial_b = storage(
        "General Reduction Partial B",
        partial_bytes,
        wgpu::BufferUsages::empty(),
    );
    let reduction_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("General Boundary Reduction Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ReductionParams>() as u64
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
    struct ReductionDispatch {
        bind_group: wgpu::BindGroup,
        chunks: u32,
    }
    let mut reduction_dispatches = Vec::<Vec<ReductionDispatch>>::new();
    let dummy_u32 = [0_u32];
    for depth in &reductions {
        let mut passes = Vec::new();
        for (pass_index, pass) in depth.passes.iter().enumerate() {
            let params = ReductionParams {
                chunk_count: pass.chunks.len() as u32,
                input_kind: u32::from(pass.input_kind == BoundaryInputKind::PreviousPassPartial),
                padding: [0; 2],
            };
            let gpu_chunks = pass
                .chunks
                .iter()
                .map(|chunk| GpuReductionChunk {
                    input_offset: chunk.input_offset,
                    input_count: chunk.input_count,
                    output_slot: chunk.output_slot,
                    target_parent_cell: chunk.target_parent_cell,
                    final_output: u32::from(chunk.final_output),
                    padding: [0; 3],
                })
                .collect::<Vec<_>>();
            let params_buffer = init_buffer(
                "Boundary Reduction Params",
                bytemuck::bytes_of(&params),
                wgpu::BufferUsages::UNIFORM,
            );
            let inputs_buffer = init_buffer(
                "Boundary Reduction Inputs",
                bytemuck::cast_slice(if pass.inputs.is_empty() {
                    &dummy_u32
                } else {
                    &pass.inputs
                }),
                wgpu::BufferUsages::STORAGE,
            );
            let chunks_buffer = init_buffer(
                "Boundary Reduction Chunks",
                bytemuck::cast_slice(&gpu_chunks),
                wgpu::BufferUsages::STORAGE,
            );
            let (previous, next) = if pass_index % 2 == 0 {
                (&partial_b, &partial_a)
            } else {
                (&partial_a, &partial_b)
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Boundary Reduction Bind Group"),
                layout: &reduction_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: inputs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: chunks_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: microtree_up_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: previous.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: next.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: boundary_buffer.as_entire_binding(),
                    },
                ],
            });
            passes.push(ReductionDispatch {
                bind_group,
                chunks: params.chunk_count,
            });
        }
        reduction_dispatches.push(passes);
    }

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let block_text = args.block_size.to_string();
    let propagation_source = include_str!("../../shaders/signal_backbone_general_propagate.wgsl")
        .replace(
            "const BLOCK_SIZE: u32 = 64u;",
            &format!("const BLOCK_SIZE: u32 = {block_text}u;"),
        )
        .replace(
            "array<vec4<f32>, 64>",
            &format!("array<vec4<f32>, {block_text}>"),
        )
        .replace(
            "@workgroup_size(64)",
            &format!("@workgroup_size({block_text})"),
        );
    let propagation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("General Propagation Shader"),
        source: wgpu::ShaderSource::Wgsl(propagation_source.into()),
    });
    let main_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("General Propagation Pipeline Layout"),
        bind_group_layouts: &[&main_layout],
        push_constant_ranges: &[],
    });
    let make_pipeline = |entry: &'static str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&main_pipeline_layout),
            module: &propagation_shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let local_up = make_pipeline("local_up");
    let local_down = make_pipeline("local_down");
    let write_child_down = make_pipeline("write_child_down");
    let reduction_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("General Boundary Reduction Shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/signal_backbone_boundary_reduce.wgsl").into(),
        ),
    });
    let reduction_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("General Boundary Reduction Pipeline Layout"),
            bind_group_layouts: &[&reduction_layout],
            push_constant_ranges: &[],
        });
    let reduce_boundaries = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("General Boundary Reduction"),
        layout: Some(&reduction_pipeline_layout),
        module: &reduction_shader,
        entry_point: Some("reduce_boundaries"),
        compilation_options: Default::default(),
        cache: None,
    });
    let processor_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("General Processor Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ProcessorParams>() as u64
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
    let processor_params = ProcessorParams {
        cell_count,
        operation: u32::from(args.workload == SignalWorkload::EveryCellCognocyte),
        padding: [0; 2],
    };
    let processor_params_buffer = init_buffer(
        "General Processor Params",
        bytemuck::bytes_of(&processor_params),
        wgpu::BufferUsages::UNIFORM,
    );
    let processor_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("General Processor Bind Group"),
        layout: &processor_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: processor_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: finalized_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: processor_state_buffer.as_entire_binding(),
            },
        ],
    });
    let processor_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("General Processor Shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/signal_backbone_processor_bench.wgsl").into(),
        ),
    });
    let processor_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("General Processor Pipeline Layout"),
            bind_group_layouts: &[&processor_layout],
            push_constant_ranges: &[],
        });
    let processor_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("General Processor Pipeline"),
        layout: Some(&processor_pipeline_layout),
        module: &processor_shader,
        entry_point: Some("process_cells"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(error) = device.pop_error_scope().await {
        return Err(format!(
            "general propagation WGSL/pipeline validation failed: {error}"
        ));
    }
    println!("general_propagation_pipeline_creation=pass");

    let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("General Propagation Timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let query_resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("General Propagation Query Resolve"),
        size: 256,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let query_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("General Propagation Query Readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let encode_tick = |encoder: &mut wgpu::CommandEncoder, timed: bool| {
        if timed {
            encoder.write_timestamp(&query_set, 0);
        }
        for (group_index, _) in active_groups(args.workload).iter().enumerate() {
            encoder.clear_buffer(&boundary_buffer, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("General Upward Propagation"),
                    timestamp_writes: None,
                });
                for depth in (0..depth_ranges.len()).rev() {
                    pass.set_pipeline(&local_up);
                    pass.set_bind_group(0, &main_binds[group_index][depth], &[]);
                    pass.dispatch_workgroups(depth_ranges[depth].1, 1, 1);
                    if depth > 0 {
                        pass.set_pipeline(&reduce_boundaries);
                        for reduction in &reduction_dispatches[depth - 1] {
                            pass.set_bind_group(0, &reduction.bind_group, &[]);
                            pass.dispatch_workgroups(reduction.chunks, 1, 1);
                        }
                    }
                }
            }
            encoder.clear_buffer(&boundary_buffer, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("General Downward Propagation"),
                    timestamp_writes: None,
                });
                for depth in 0..depth_ranges.len() {
                    pass.set_pipeline(&local_down);
                    pass.set_bind_group(0, &main_binds[group_index][depth], &[]);
                    pass.dispatch_workgroups(depth_ranges[depth].1, 1, 1);
                    if depth + 1 < depth_ranges.len() {
                        pass.set_pipeline(&write_child_down);
                        pass.set_bind_group(0, &main_binds[group_index][depth + 1], &[]);
                        pass.dispatch_workgroups(
                            depth_ranges[depth + 1].1.div_ceil(WORKGROUP_SIZE),
                            1,
                            1,
                        );
                    }
                }
            }
        }
        if has_processor {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("General Processor Evaluation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&processor_pipeline);
            pass.set_bind_group(0, &processor_bind, &[]);
            pass.dispatch_workgroups(cell_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        if timed {
            encoder.write_timestamp(&query_set, 1);
        }
    };

    for _ in 0..args.warmup {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("General Propagation Warmup"),
        });
        encode_tick(&mut encoder, false);
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| format!("general warmup wait failed: {error:?}"))?;
    }

    let timestamp_period = queue.get_timestamp_period() as f64;
    let mut timings_ms = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("General Propagation Timed"),
        });
        encode_tick(&mut encoder, true);
        encoder.resolve_query_set(&query_set, 0..2, &query_resolve, 0);
        encoder.copy_buffer_to_buffer(&query_resolve, 0, &query_readback, 0, 16);
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| format!("general sample wait failed: {error:?}"))?;
        let timestamps = map_read::<u64>(device, &query_readback)?;
        timings_ms.push((timestamps[1] - timestamps[0]) as f64 * timestamp_period / 1_000_000.0);
    }

    let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("General Output Readback"),
        size: field_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("General Output Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(&finalized_buffer, 0, &output_readback, 0, field_bytes);
    let submission = queue.submit(std::iter::once(encoder.finish()));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map_err(|error| format!("general output wait failed: {error:?}"))?;
    let actual = map_read::<[f32; 4]>(device, &output_readback)?;
    let (max_error, mismatches) = compare_results(expected, &actual);
    let mut processor_max_error = 0.0_f32;
    let mut processor_mismatches = 0_usize;
    if has_processor {
        let processor_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("General Processor Readback"),
            size: value_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("General Processor Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &processor_state_buffer,
            0,
            &processor_readback,
            0,
            value_bytes,
        );
        let submission = queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| format!("processor output wait failed: {error:?}"))?;
        let processor_actual = map_read::<[f32; 4]>(device, &processor_readback)?;
        for cell in 0..args.cells {
            let input = expected[cell];
            let expected_output = if args.workload == SignalWorkload::EveryCellCognocyte {
                (input[0] * input[1] / 1000.0).clamp(-1000.0, 1000.0)
            } else {
                let effective_rate = 1.0 - 0.5_f32.powf(1.0 / 15.0);
                let ticks = args.warmup + args.samples;
                input[0] * (1.0 - (1.0 - effective_rate).powi(ticks as i32))
            };
            let error = (processor_actual[cell][0] - expected_output).abs();
            processor_max_error = processor_max_error.max(error);
            if !processor_actual[cell][0].is_finite() || error > PARITY_TOLERANCE {
                processor_mismatches += 1;
            }
        }
        println!(
            "processor_parity_tolerance={} processor_max_abs_error={:.6} processor_mismatches={}",
            PARITY_TOLERANCE, processor_max_error, processor_mismatches
        );
    }
    timings_ms.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((timings_ms.len() - 1) as f64 * fraction).ceil() as usize;
        timings_ms[index]
    };
    let schedule_bytes = reductions
        .iter()
        .flat_map(|depth| depth.passes.iter())
        .map(|pass| {
            pass.inputs.len() as u64 * 4
                + pass.chunks.len() as u64 * std::mem::size_of::<GpuReductionChunk>() as u64
        })
        .sum::<u64>();
    let workspace_bytes = source_vec4.len() as u64 * 16
        + field_bytes
        + upload.cells.len() as u64 * std::mem::size_of::<GpuCellTopology>() as u64
        + upload.microtrees.len() as u64 * std::mem::size_of::<GpuMicrotreeTopology>() as u64
        + upload.node_list.len() as u64 * 4
        + value_bytes * 2
        + microtree_value_bytes
        + partial_bytes * 2
        + u64::from(has_processor) * value_bytes
        + schedule_bytes;
    let per_group_dispatches = depth_ranges.len() as u32 * 2
        + reduction_dispatches
            .iter()
            .flat_map(|passes| passes.iter())
            .count() as u32
        + depth_ranges.len().saturating_sub(1) as u32;
    let dispatches =
        per_group_dispatches * active_groups(args.workload).len() as u32 + u32::from(has_processor);
    println!(
        "general_strategy=depth_microtrees active_group_mask=0x{:x} workspace_bytes={} workspace_mib={:.3} dispatches={}",
        active_groups(args.workload).iter().fold(0_u32, |mask, group| mask | (1 << group)),
        workspace_bytes,
        workspace_bytes as f64 / (1024.0 * 1024.0),
        dispatches,
    );
    println!(
        "gpu_ms_p50={:.4} gpu_ms_p95={:.4} gpu_ms_worst={:.4}",
        percentile(0.50),
        percentile(0.95),
        timings_ms[timings_ms.len() - 1]
    );
    println!(
        "parity_tolerance={} parity_max_abs_error={:.6} parity_mismatches={}",
        PARITY_TOLERANCE, max_error, mismatches
    );
    let memory_pass = workspace_bytes <= 64 * 1024 * 1024;
    let correctness_pass = mismatches == 0 && processor_mismatches == 0;
    let timing_pass = percentile(0.95) <= 2.0;
    println!(
        "gate_memory={} gate_correctness={} gate_discrete_gpu_p95={} phase1_gate=INCOMPLETE_MATRIX",
        pass_fail(memory_pass),
        pass_fail(correctness_pass),
        pass_fail(timing_pass)
    );
    if !memory_pass || !correctness_pass || !timing_pass {
        return Err("general propagation acceptance row failed".into());
    }
    Ok(())
}

async fn validate_general_topology(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &bio_spheres::simulation::signal_backbone_bench::CachedForest,
    block_size: u32,
) -> Result<(), String> {
    let schedule = MicrotreeSchedule::build(cache, block_size as usize);
    let upload = schedule.flatten_for_gpu(cache, TOPOLOGY_GENERATION_INITIAL);
    let params = GeneralTopologyParams {
        cell_count: upload.cells.len() as u32,
        microtree_count: upload.microtrees.len() as u32,
        generation: TOPOLOGY_GENERATION_INITIAL,
        block_size,
        node_list_count: upload.node_list.len() as u32,
        child_list_count: upload.child_microtrees.len() as u32,
        depth_offset_count: upload.depth_offsets.len() as u32,
        depth_microtree_count: upload.depth_microtrees.len() as u32,
    };
    let buffer = |label: &'static str, contents: &[u8], usage: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    };
    let params_buffer = buffer(
        "General Topology Params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    );
    let cells_buffer = buffer(
        "General Cell Topology",
        bytemuck::cast_slice::<GpuCellTopology, u8>(&upload.cells),
        wgpu::BufferUsages::STORAGE,
    );
    let microtrees_buffer = buffer(
        "General Microtree Topology",
        bytemuck::cast_slice::<GpuMicrotreeTopology, u8>(&upload.microtrees),
        wgpu::BufferUsages::STORAGE,
    );
    let node_list_buffer = buffer(
        "General Node List",
        bytemuck::cast_slice(&upload.node_list),
        wgpu::BufferUsages::STORAGE,
    );
    let empty_child = [u32::MAX];
    let child_data = if upload.child_microtrees.is_empty() {
        &empty_child[..]
    } else {
        &upload.child_microtrees
    };
    let child_buffer = buffer(
        "General Child Microtrees",
        bytemuck::cast_slice(child_data),
        wgpu::BufferUsages::STORAGE,
    );
    let depth_offsets_buffer = buffer(
        "General Depth Offsets",
        bytemuck::cast_slice(&upload.depth_offsets),
        wgpu::BufferUsages::STORAGE,
    );
    let depth_microtrees_buffer = buffer(
        "General Depth Microtrees",
        bytemuck::cast_slice(&upload.depth_microtrees),
        wgpu::BufferUsages::STORAGE,
    );
    let diagnostics_buffer = buffer(
        "General Topology Diagnostics",
        bytemuck::cast_slice(&[0_u32; 4]),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("General Topology Diagnostics Readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("General Topology Validation Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(
                        std::mem::size_of::<GeneralTopologyParams>() as u64
                    ),
                },
                count: None,
            },
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, true),
            storage_entry(6, true),
            storage_entry(7, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("General Topology Validation Bind Group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cells_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: microtrees_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: node_list_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: child_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: depth_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: depth_microtrees_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: diagnostics_buffer.as_entire_binding(),
            },
        ],
    });

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("General Signal Backbone Topology Shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../shaders/signal_backbone_general_bench.wgsl").into(),
        ),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("General Topology Validation Pipeline Layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = |entry_point: &'static str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let validate_cells = pipeline("validate_cells");
    let validate_microtrees = pipeline("validate_microtrees");
    let validate_depths = pipeline("validate_depths");
    if let Some(error) = device.pop_error_scope().await {
        return Err(format!(
            "general topology WGSL/pipeline validation failed: {error}"
        ));
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("General Topology Validation Encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("General Topology Validation Pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&validate_cells);
        pass.dispatch_workgroups(params.node_list_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        pass.set_pipeline(&validate_microtrees);
        pass.dispatch_workgroups(params.microtree_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        pass.set_pipeline(&validate_depths);
        pass.dispatch_workgroups(params.depth_microtree_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&diagnostics_buffer, 0, &readback, 0, 16);
    let submission = queue.submit(std::iter::once(encoder.finish()));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map_err(|error| format!("general topology validation wait failed: {error:?}"))?;
    let diagnostics = map_read::<u32>(device, &readback)?;
    if diagnostics.len() < 4 || diagnostics[0..3] != [0, 0, 0] {
        return Err(format!(
            "general topology GPU validation failed: {diagnostics:?}"
        ));
    }
    if diagnostics[3] != params.node_list_count {
        return Err(format!(
            "general topology GPU validation checked {} of {} records",
            diagnostics[3], params.node_list_count
        ));
    }
    println!(
        "general_topology_pipeline_creation=pass cells={} microtrees={} max_children={} macro_strategy={:?} topology_bytes={} topology_mib={:.3} diagnostics={:?}",
        params.cell_count,
        params.microtree_count,
        upload.maximum_children,
        upload.macro_strategy,
        upload.allocated_bytes(),
        upload.allocated_bytes() as f64 / (1024.0 * 1024.0),
        diagnostics,
    );
    Ok(())
}

fn map_read<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<T>, String> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|error| format!("map poll failed: {error:?}"))?;
    receiver
        .recv()
        .map_err(|error| format!("map callback failed: {error}"))?
        .map_err(|error| format!("map failed: {error}"))?;
    let view = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    buffer.unmap();
    Ok(values)
}

fn pass_fail(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}

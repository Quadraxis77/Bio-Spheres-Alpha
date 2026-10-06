use bio_spheres::genome::Genome;
use bio_spheres::scene::preview_state::InitialState;
use bio_spheres::simulation::gpu_physics::{GpuPhysicsPipelines, GpuScaffoldSystem};
use bio_spheres::simulation::physics_config::PhysicsConfig;
use bio_spheres::simulation::preview_physics::physics_step_with_genome;
use std::collections::HashSet;
use std::path::PathBuf;

fn genome_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("genomes")
        .join("luca_pelagic_signal_foundation.genome")
}

#[test]
fn triskelion_genome_has_a_coherent_rotational_body_plan() {
    let genome = Genome::load_from_file(&genome_path()).expect("LUCA genome must load");

    assert_eq!(genome.name, "LUCA Triskelion Ray");
    assert_eq!(genome.initial_mode, 8);
    assert_eq!(genome.modes.len(), 9);
    assert_eq!(genome.modes[8].cell_type, 10);
    assert_eq!(genome.modes[1].max_splits, 3);
    assert_eq!(genome.modes[1].child_a.mode_number, 2);
    assert_eq!(genome.modes[1].child_b.mode_number, 1);
    assert_eq!(genome.modes[1].mode_b_after_splits, 3);
    assert!(genome.modes[4].flagellocyte_use_signal);
    assert_eq!(genome.scaffold_rules.len(), 1);
    assert_eq!(genome.scaffold_rules[0].rest_length, 2.15);
}

#[test]
fn gpu_embryocyte_hatch_assigns_separate_development_scopes() {
    let division_shader = include_str!("../shaders/lifecycle_division_execute_ring.wgsl");
    let scaffold_shader = include_str!("../shaders/gpu_scaffold_resolve.wgsl");
    let spatial_clear_shader = include_str!("../shaders/spatial_grid_clear.wgsl");
    let spatial_build_shader = include_str!("../shaders/spatial_grid_build.wgsl");
    let collision_shader = include_str!("../shaders/collision_detection.wgsl");
    let nutrient_shader = include_str!("../shaders/nutrient_transport.wgsl");
    let gpu_scene = include_str!("../src/scene/gpu_scene.rs");
    let gpu_timer = include_str!("../src/scene/gpu_timer.rs");
    let gpu_integration = include_str!("../src/simulation/gpu_physics/gpu_scene_integration.rs");

    assert!(division_shader.contains(
        "let detached_embryocyte_hatch = parent_is_embryocyte && !create_sibling_adhesion;"
    ));
    assert!(division_shader.contains("let child_a_is_new_org = detached_embryocyte_hatch"));
    assert!(division_shader.contains("let child_b_is_new_org = detached_embryocyte_hatch"));
    assert!(scaffold_shader
        .contains("@group(1) @binding(9) var<storage, read> organism_labels: array<u32>;"));
    assert!(scaffold_shader.contains("&& organism_labels[cell_idx] == component_label;"));
    assert!(scaffold_shader
        .contains("@group(0) @binding(3) var<storage, read> spatial_grid_counts: array<u32>;"));
    assert!(scaffold_shader
        .contains("let candidate = neighborhood_candidate(source, ordinal, max_range);"));
    assert!(
        scaffold_shader.contains("if (!is_within_formation_range(source, candidate, max_range))")
    );
    assert!(scaffold_shader.contains("const SCAFFOLD_SOURCE_PHASE_COUNT: u32 = 8u;"));
    assert!(scaffold_shader
        .contains("(source % SCAFFOLD_SOURCE_PHASE_COUNT) != scaffold_params.source_phase"));
    assert!(
        !scaffold_shader.contains("for (var candidate = 0u; candidate < live_slots"),
        "scaffold matching regressed to an all-live-cell candidate scan"
    );
    assert!(gpu_scene
        .contains("Scaffold matching must run after the developmental-component label pass."));
    assert!(gpu_scene.contains("self.total_cell_slots.max(1)"));
    let scaffold_system = include_str!("../src/simulation/gpu_physics/scaffold.rs");
    assert!(scaffold_system.contains("const GPU_SCAFFOLD_PHASE_COUNT: u32 = 8;"));
    assert!(spatial_clear_shader.contains("occupied_grid_cells[occupied_idx]"));
    assert!(gpu_integration.contains("Some(active_scalar_bytes)"));
    assert!(gpu_integration.contains("reachable_adhesion_slots"));
    assert!(gpu_scene.contains("final_encoded_step"));
    assert!(gpu_timer.contains("Physics Frame Maintenance"));
    assert!(gpu_scene.contains("ASYNC_NEWBORN_SLOT_GUARD"));
    assert!(gpu_scene.contains("fn populate_nutrients_for_physics_step("));
    assert!(!gpu_scene.contains("Nutrient population moved into run_physics()"));
    assert!(
        collision_shader.contains("dispatch_idx < cell_count_buffer[0] && live_cell(dispatch_idx)")
    );
    assert!(!collision_shader.contains("!live_cell(a_idx) || !live_cell(b_idx)"));
    assert!(
        spatial_build_shader
            .find("if (death_flags[cell_idx]")
            .unwrap()
            < spatial_build_shader
                .find("let grid_idx = world_pos_to_grid_index")
                .unwrap()
    );
    assert!(nutrient_shader.contains("if (!has_transport_connection)"));
}

#[test]
fn gpu_scaffold_component_scope_shader_compiles() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("a GPU adapter is required to validate scaffold scope");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Scaffold Scope Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            })
            .await
            .expect("scaffold scope test device");

        let _system = GpuScaffoldSystem::new(&device);
        let _pipelines = GpuPhysicsPipelines::new(&device);
    });
}

#[test]
fn triskelion_reproduces_before_thirty_seconds() {
    let genome = Genome::load_from_file(&genome_path()).expect("LUCA genome must load");
    let config = PhysicsConfig::default();
    let mut state = InitialState::from_genome(&genome, 256, &config).to_canonical_state();

    let mut time = 0.0;
    while time < 28.0 && state.cell_count < state.capacity {
        time += config.fixed_timestep;
        physics_step_with_genome(&mut state, &genome, &config, time, None);
    }

    let live_modes: HashSet<usize> = state.mode_indices[..state.cell_count]
        .iter()
        .copied()
        .collect();
    let solar_rays = state.mode_indices[..state.cell_count]
        .iter()
        .filter(|&&mode| mode == 2)
        .count();
    let brood_cores = state.mode_indices[..state.cell_count]
        .iter()
        .filter(|&&mode| mode == 3)
        .count();
    let later_founders = state.mode_indices[..state.cell_count]
        .iter()
        .filter(|&&mode| mode == 0)
        .count();
    let scaffold_bonds = state.adhesion_connections.scaffold_rule_id
        [..state.adhesion_connections.active_count]
        .iter()
        .filter(|&&rule_id| rule_id == 1)
        .count();

    assert!(
        state.cell_count >= 12,
        "expected reproduction before 30 seconds, got {} cells",
        state.cell_count
    );
    assert!(live_modes.contains(&3), "brood core never matured");
    assert!(live_modes.contains(&4), "axial motor never developed");
    assert!(solar_rays >= 3, "the three-ray crown never developed");
    assert!(brood_cores >= 1, "no brood core survived");
    assert!(
        later_founders >= 2,
        "no second-generation founders hatched before 30 seconds"
    );
    assert!(scaffold_bonds > 0, "the ray-to-core scaffold never formed");
}

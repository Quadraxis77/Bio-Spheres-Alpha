use bio_spheres::genome::Genome;
use bio_spheres::simulation::preview_physics::{
    check_embryocyte_release_triggers, transport_nutrients_through_adhesions,
    update_embryocyte_reserve_burn, update_nutrient_growth,
};
use bio_spheres::simulation::CanonicalState;
use glam::{Quat, Vec3};

#[test]
fn nutrient_transport_shader_validates() {
    let source = include_str!("../shaders/nutrient_transport.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("nutrient transport shader must validate");
}

fn attached_gamete() -> (Genome, CanonicalState) {
    let mut genome = Genome::default();
    genome.modes.truncate(1);
    genome.modes.push(genome.modes[0].clone());
    genome.modes[0].cell_type = 13;
    genome.modes[0].embryocyte_use_timer = false;
    genome.modes[0].embryocyte_use_threshold = false;
    genome.modes[0].embryocyte_use_signal = false;
    genome.modes[1].cell_type = 0;
    let mut state = CanonicalState::new(4);
    for mode in 0..2 {
        state.add_cell(
            Vec3::X * mode as f32,
            Vec3::ZERO,
            Quat::IDENTITY,
            Quat::IDENTITY,
            Vec3::ZERO,
            100.0,
            0,
            mode,
            0.0,
            60.0,
            100.0,
            1.0,
        );
    }
    state
        .adhesion_manager
        .add_adhesion_with_directions(
            &mut state.adhesion_connections,
            0,
            1,
            0,
            Vec3::X,
            -Vec3::X,
            Vec3::X,
            Vec3::X,
            Quat::IDENTITY,
            Quat::IDENTITY,
            0.5,
            0.5,
            0.0,
        )
        .expect("fixture bond");
    (genome, state)
}

fn assert_attached(state: &CanonicalState, attached: bool) {
    for cell in 0..2 {
        assert_eq!(
            state
                .adhesion_manager
                .count_active_adhesions(cell, &state.adhesion_connections,),
            usize::from(attached)
        );
    }
}

#[test]
fn timer_releases_gamete_at_deadline() {
    let (mut genome, mut state) = attached_gamete();
    genome.modes[0].embryocyte_use_timer = true;
    genome.modes[0].embryocyte_release_timer = 1.0;
    update_embryocyte_reserve_burn(&mut state, &genome, 0.5);
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, true);
    update_embryocyte_reserve_burn(&mut state, &genome, 0.5);
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, false);
    assert_eq!(state.embryocyte_timers[0], 0.0);
}

#[test]
fn feeding_fills_gamete_reserve_and_triggers_release() {
    let (mut genome, mut state) = attached_gamete();
    genome.modes[0].embryocyte_use_threshold = true;
    genome.modes[0].embryocyte_threshold_value = 1;
    state.reserves[0] = 0;
    state.nutrients[1] = 100.0;
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, true);
    transport_nutrients_through_adhesions(&mut state, &genome, 0.1);
    assert!(state.reserves[0] >= 1000);
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, false);
}

#[test]
fn signal_releases_only_when_threshold_is_met() {
    let (mut genome, mut state) = attached_gamete();
    genome.modes[0].embryocyte_use_signal = true;
    genome.modes[0].embryocyte_signal_channel = 8;
    genome.modes[0].embryocyte_signal_value = 0.5;
    state.signal_channels[8] = Some(0.25);
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, true);
    state.signal_channels[8] = Some(0.75);
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, false);
}

#[test]
fn release_requires_all_enabled_triggers() {
    let (mut genome, mut state) = attached_gamete();
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, true);
    let mode = &mut genome.modes[0];
    mode.embryocyte_use_timer = true;
    mode.embryocyte_release_timer = 1.0;
    mode.embryocyte_use_threshold = true;
    mode.embryocyte_threshold_value = 1;
    mode.embryocyte_use_signal = true;
    mode.embryocyte_signal_channel = 8;
    mode.embryocyte_signal_value = 0.5;
    state.embryocyte_timers[0] = 1.0;
    state.reserves[0] = 1000;
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, true);
    state.signal_channels[8] = Some(0.75);
    check_embryocyte_release_triggers(&mut state, &genome);
    assert_attached(&state, false);
}

#[test]
fn gamete_metabolism_is_slow_both_attached_and_free() {
    let (mut genome, mut state) = attached_gamete();
    state.reserves[0] = 20_000;
    update_nutrient_growth(&mut state, &genome, 1.0);
    update_embryocyte_reserve_burn(&mut state, &genome, 1.0);
    assert_eq!(state.reserves[0], 19_900);
    genome.modes[0].embryocyte_use_timer = true;
    genome.modes[0].embryocyte_release_timer = 1.0;
    check_embryocyte_release_triggers(&mut state, &genome);
    update_embryocyte_reserve_burn(&mut state, &genome, 1.0);
    assert_eq!(state.reserves[0], 19_800);
}

#[test]
fn reserve_loading_preserves_parent_food_across_multiple_receivers() {
    for reserve_type in [10, 13] {
        for parent_food in [5.0f32, 10.0, 12.0, 100.0] {
            let (mut genome, mut state) = attached_gamete();
            genome.modes[0].cell_type = reserve_type;
            genome.modes[0].nutrient_priority = 20.0;
            genome.modes[1].prioritize_when_low = false;
            state.nutrients[1] = parent_food;
            state.reserves[0] = 0;
            state.add_cell(
                Vec3::Y, Vec3::ZERO, Quat::IDENTITY, Quat::IDENTITY,
                Vec3::ZERO, 100.0, 0, 0, 0.0, 60.0, 100.0, 1.0,
            );
            state.reserves[2] = 0;
            // Opposite endpoint order from the first bond: both directions must
            // preserve the same parent's food budget.
            state.adhesion_manager.add_adhesion_with_directions(
                &mut state.adhesion_connections, 1, 2, 0,
                Vec3::Y, -Vec3::Y, Vec3::X, Vec3::X,
                Quat::IDENTITY, Quat::IDENTITY, 0.5, 0.5, 0.0,
            ).unwrap();
            for _ in 0..100 {
                transport_nutrients_through_adhesions(&mut state, &genome, 0.1);
                assert!(state.nutrients[1] >= parent_food.min(10.0) - 0.001);
                let stored = (state.reserves[0] + state.reserves[2]) as f32 / 1000.0;
                assert!((state.nutrients[1] + stored - parent_food).abs() < 0.01,
                    "multiple receivers must conserve their donor's food");
            }
            assert!((state.nutrients[1] - parent_food.min(10.0)).abs() < 0.001);
        }
    }
}

#[test]
fn reserve_intake_scales_with_priority_above_and_below_one() {
    for (cell_type, baseline) in [(13, 10.0f32), (10, 100.0)] {
        for priority in [0.1f32, 0.5, 1.0, 2.0, 10.0] {
            let (mut genome, mut state) = attached_gamete();
            genome.modes[0].cell_type = cell_type;
            genome.modes[0].nutrient_priority = priority;
            state.reserves[0] = 0;
            state.nutrients[1] = 1000.0;
            transport_nutrients_through_adhesions(&mut state, &genome, 0.1);
            let expected = baseline * priority * 0.1;
            assert!((state.reserves[0] as f32 / 1000.0 - expected).abs() < 0.002);
            assert!((state.nutrients[1] - (1000.0 - expected)).abs() < 0.002);
        }
    }
}

#[test]
fn gamete_burn_saturates_and_embryo_metabolism_is_unchanged() {
    let (mut genome, mut state) = attached_gamete();
    state.reserves[0] = 50;
    update_embryocyte_reserve_burn(&mut state, &genome, 1.0);
    assert_eq!(state.reserves[0], 0);
    genome.modes[0].cell_type = 10;
    state.reserves[0] = 20_000;
    update_embryocyte_reserve_burn(&mut state, &genome, 1.0);
    assert_eq!(state.reserves[0], 20_000);
    state.adhesion_manager.remove_all_connections_for_cell(&mut state.adhesion_connections, 0);
    update_embryocyte_reserve_burn(&mut state, &genome, 1.0);
    assert_eq!(state.reserves[0], 10_000);
}

#[test]
fn lifecycle_shader_validates_after_gamete_metabolism_change() {
    let source = include_str!("../shaders/lifecycle_unified.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    ).validate(&module).expect("lifecycle shader must validate");
}

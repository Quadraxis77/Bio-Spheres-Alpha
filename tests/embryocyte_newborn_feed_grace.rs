use bio_spheres::genome::Genome;
use bio_spheres::simulation::preview_physics::physics_step_with_genome;
use bio_spheres::simulation::{CanonicalState, PhysicsConfig};
use glam::{Quat, Vec3};

fn zero_reserve_embryocyte_state() -> (Genome, PhysicsConfig, CanonicalState) {
    let mut genome = Genome::default();
    genome.modes.truncate(1);
    genome.modes[0].cell_type = 10;
    genome.modes[0].split_interval = 60.0;

    let config = PhysicsConfig::default();
    let mut state = CanonicalState::new(4);
    let split_threshold = (genome.modes[0].split_mass - 1.0) * 100.0;
    state.add_cell(
        Vec3::ZERO,
        Vec3::ZERO,
        Quat::IDENTITY,
        Quat::IDENTITY,
        Vec3::ZERO,
        100.0,
        0,
        0,
        0.0,
        genome.modes[0].split_interval,
        split_threshold,
        genome.modes[0].membrane_stiffness,
    );

    (genome, config, state)
}

#[test]
fn empty_newborn_embryocyte_survives_brief_feed_grace() {
    let (genome, config, mut state) = zero_reserve_embryocyte_state();

    physics_step_with_genome(&mut state, &genome, &config, 0.1, None);

    assert_eq!(
        state.cell_count, 1,
        "empty newborn embryocytes need a short chance to receive reserve"
    );
}

#[test]
fn empty_embryocyte_dies_after_feed_grace_if_still_unfed() {
    let (genome, config, mut state) = zero_reserve_embryocyte_state();

    physics_step_with_genome(&mut state, &genome, &config, 0.25, None);

    assert_eq!(
        state.cell_count, 0,
        "empty embryocytes should still die if they remain unfed after grace"
    );
}

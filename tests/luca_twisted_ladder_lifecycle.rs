use bio_spheres::genome::Genome;
use bio_spheres::scene::preview_state::InitialState;
use bio_spheres::simulation::physics_config::PhysicsConfig;
use bio_spheres::simulation::preview_physics::physics_step_with_genome;
use std::collections::HashSet;
use std::path::PathBuf;

const JUVENILE_MODE: usize = 0;
const MOTOR_MODE: usize = 9;
const BROOD_MODE: usize = 10;
const EMBRYO_MODE: usize = 11;
const EMBRYO_THRESHOLD: u32 = 160_000;

fn genome_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("genomes")
        .join("LUCA Twisted Ladder.genome")
}

fn adhesion_count(
    state: &bio_spheres::simulation::canonical_state::CanonicalState,
    cell: usize,
) -> usize {
    state
        .adhesion_manager
        .count_active_adhesions(cell, &state.adhesion_connections)
}

#[test]
fn ladder_genome_has_nursed_one_egg_lifecycle_and_redundant_rungs() {
    let genome = Genome::load_from_file(&genome_path()).expect("twisted ladder genome must load");

    assert_eq!(genome.name, "LUCA Twisted Ladder");
    assert_eq!(genome.initial_mode, EMBRYO_MODE as i32);
    assert_eq!(genome.modes.len(), 12);
    assert_eq!(genome.scaffold_rules.len(), 3);
    assert!(genome.modes[EMBRYO_MODE].embryocyte_use_timer);
    assert!(genome.modes[EMBRYO_MODE].embryocyte_use_threshold);
    assert_eq!(
        genome.modes[EMBRYO_MODE].embryocyte_threshold_value,
        EMBRYO_THRESHOLD / 1000
    );
    assert!(genome.modes[0].adhesion_settings.creates_backbone);
    assert!(genome.modes[1].adhesion_settings.creates_backbone);
    assert!(genome.modes[2].adhesion_settings.creates_backbone);
    assert!(genome.modes[5].adhesion_settings.creates_backbone);
    assert!(genome.modes[6].adhesion_settings.creates_backbone);
    assert_eq!(genome.modes[BROOD_MODE].max_adhesions, 4);
    assert!(genome.modes[BROOD_MODE].child_a.keep_adhesion);
}

#[test]
fn threshold_nursed_embryo_releases_and_yields_foraging_juveniles_within_forty_seconds() {
    let genome = Genome::load_from_file(&genome_path()).expect("twisted ladder genome must load");
    let config = PhysicsConfig::default();
    let mut state = InitialState::from_genome(&genome, 512, &config).to_canonical_state();

    // Remove the enormous bootstrap reserve after the founding embryo hatches. This makes
    // the adult egg prove that it can be filled by living feeder tissue rather than by an
    // inherited launch subsidy, approximating an imperfect post-bootstrap generation.
    let mut founding_hatched = false;
    let mut saw_adult = false;
    let mut saw_attached_empty_egg = false;
    let mut saw_threshold_ready_attached_egg = false;
    let mut first_viable_release = None;
    let mut released_juvenile_reserve = 0;
    let mut live_rung_ids = HashSet::new();
    let mut time = 0.0;

    while time < 40.0 && state.cell_count < state.capacity {
        time += config.fixed_timestep;
        physics_step_with_genome(&mut state, &genome, &config, time, None);

        if !founding_hatched
            && state.mode_indices[..state.cell_count]
                .iter()
                .any(|&mode| mode == JUVENILE_MODE)
        {
            founding_hatched = true;
            for reserve in &mut state.reserves[..state.cell_count] {
                *reserve = 0;
            }
        }

        saw_adult |= state.mode_indices[..state.cell_count]
            .iter()
            .any(|&mode| mode == MOTOR_MODE)
            && state.mode_indices[..state.cell_count]
                .iter()
                .any(|&mode| mode == BROOD_MODE);

        live_rung_ids.clear();
        for bond in 0..state.adhesion_connections.active_count {
            if state.adhesion_connections.is_active[bond] != 0 {
                let rule_id = state.adhesion_connections.scaffold_rule_id[bond];
                if rule_id != 0 {
                    live_rung_ids.insert(rule_id);
                }
            }
        }

        for cell in 0..state.cell_count {
            if state.mode_indices[cell] != EMBRYO_MODE || adhesion_count(&state, cell) == 0 {
                continue;
            }
            if state.reserves[cell] < EMBRYO_THRESHOLD {
                saw_attached_empty_egg = true;
            } else {
                saw_threshold_ready_attached_egg = true;
            }
        }

        let juvenile_count = state.mode_indices[..state.cell_count]
            .iter()
            .filter(|&&mode| mode == JUVENILE_MODE)
            .count();
        if founding_hatched && saw_threshold_ready_attached_egg && juvenile_count >= 2 {
            // The founding juveniles have already differentiated by this point; a renewed
            // mode-0 population therefore records hatchlings from the nursed adult egg.
            first_viable_release = Some(time);
            released_juvenile_reserve = (0..state.cell_count)
                .filter(|&cell| state.mode_indices[cell] == JUVENILE_MODE)
                .map(|cell| state.reserves[cell])
                .min()
                .unwrap_or(0);
            break;
        }
    }

    assert!(founding_hatched, "founding embryo never hatched");
    assert!(
        saw_adult,
        "ladder never formed its motor and brood endpoints"
    );
    assert!(
        saw_attached_empty_egg,
        "adult embryo did not remain attached while under its nutrient threshold"
    );
    assert!(
        saw_threshold_ready_attached_egg,
        "adult embryo never accumulated its 160-unit release reserve"
    );
    let release_time = first_viable_release.expect("no viable juvenile release by 40 seconds");
    eprintln!(
        "twisted ladder viable release: {release_time:.2}s; juvenile reserve: {:.1}; live authored rungs: {:?}",
        released_juvenile_reserve as f32 / 1000.0,
        live_rung_ids
    );
    assert!(
        (10.0..=40.0).contains(&release_time),
        "viable release occurred outside the 10-40 second design window: {release_time:.2}s"
    );
    assert!(
        released_juvenile_reserve >= 75_000,
        "hatchlings received too little reserve to bridge into foraging: {:.1}",
        released_juvenile_reserve as f32 / 1000.0
    );
    assert_eq!(
        live_rung_ids,
        HashSet::from([1, 2, 3]),
        "the mature ladder did not retain all three authored rungs"
    );
}

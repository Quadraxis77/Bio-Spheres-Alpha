//! Procedural genome naming - generates biologically plausible names from
//! Latin/Greek roots based on the genome's actual cell-type composition.
//!
//! `generate_unique` keeps rerolling until it finds a name not already in use.

use crate::genome::Genome;

/// Generate a name that is not in `used_names`, rerolling up to 200 times.
/// `variation` is the user-facing "regenerate" counter - each click increments it.
pub fn generate_unique(genome: &Genome, variation: u64, used_names: &[String]) -> String {
    for attempt in 0u64..200 {
        let name = generate(
            genome,
            variation.wrapping_add(attempt.wrapping_mul(0x9e3779b97f4a7c15)),
        );
        if !used_names.contains(&name.to_lowercase()) {
            return name;
        }
    }
    // Absolute fallback: append the variation number
    format!(
        "{} {}",
        generate(genome, variation),
        variation.wrapping_add(1)
    )
}

/// Generate a procedural name for a genome.
/// `variation` changes which name is produced for the same genome.
pub fn generate(genome: &Genome, variation: u64) -> String {
    let modes = &genome.modes;

    // -- Trait detection -------------------------------------------------------
    let has = |id: i32| modes.iter().any(|m| m.cell_type == id);
    let has_photo = has(12);
    let has_devour = has(9);
    let has_phage = has(2);
    let has_flag = has(3);
    let has_myo = has(8);
    let has_cilia = has(6);
    let has_oculo = has(7);
    let has_glue = has(5);
    let has_buoy = has(4);
    let has_embryo = has(10);
    let has_vasc = has(11);
    let has_lipo = has(1);
    let has_signal = modes.iter().any(|m| m.regulation_emit_channel >= 0);
    let mode_count = modes.len();

    // Seed mixes genome content + variation so every click gives a new result
    let mut seed: u64 = variation.wrapping_mul(0x9e3779b97f4a7c15);
    seed = seed.wrapping_add((mode_count as u64).wrapping_mul(0x6c62272e07bb0142));
    for m in modes {
        seed ^= (m.cell_type as u64).wrapping_mul(0xff51afd7ed558ccd);
        seed = seed.rotate_left(17);
    }
    seed ^= seed >> 33;
    seed = seed.wrapping_mul(0xbf58476d1ce4e5b9);
    seed ^= seed >> 27;
    let mut rng = Rng(seed);

    // -- Trophic root ----------------------------------------------------------
    let trophic: &str = match (has_photo, has_devour, has_phage) {
        (true, true, true) => rng.pick(&[
            "Pantotroph",
            "Omnivore",
            "Polytroph",
            "Euryphage",
            "Pantophage",
            "Holotroph",
            "Panivore",
            "Omnitroph",
            "Polyphage",
            "Universotroph",
        ]),
        (true, true, false) => rng.pick(&[
            "Photocarnivore",
            "Mixotroph",
            "Photopredator",
            "Heliophage",
            "Photovore",
            "Solarcarnivore",
            "Photobiophage",
            "Heliovore",
            "Photoraptor",
            "Lumivore",
            "Soliphage",
            "Photohunter",
        ]),
        (true, false, true) => rng.pick(&[
            "Photolithotroph",
            "Mixotroph",
            "Photoabsorber",
            "Heliotroph",
            "Photoheterotroph",
            "Solartroph",
            "Photoorganotroph",
            "Lumotroph",
            "Helioabsorber",
            "Photosaprotroph",
            "Soliphage",
            "Photofeeder",
        ]),
        (true, false, false) => rng.pick(&[
            "Phototroph",
            "Autotroph",
            "Photoautotroph",
            "Heliotroph",
            "Photosynthete",
            "Chlorotroph",
            "Solartroph",
            "Lumotroph",
            "Radiotroph",
            "Photolit",
            "Heliotroph",
            "Solarvore",
            "Lumivore",
            "Actinotroph",
            "Photolit",
        ]),
        (false, true, true) => rng.pick(&[
            "Omnivore",
            "Euryphage",
            "Necrophage",
            "Saprophage",
            "Polyphage",
            "Scavenger",
            "Opportunivore",
            "Necrovore",
            "Saprovore",
            "Detritivore",
            "Omniphage",
            "Generalist",
        ]),
        (false, true, false) => rng.pick(&[
            "Carnivore",
            "Predator",
            "Zoophage",
            "Biophage",
            "Devorator",
            "Raptor",
            "Zoovore",
            "Biovore",
            "Sarcophage",
            "Predaphage",
            "Carniphage",
            "Zymovore",
            "Haematophage",
            "Cytophage",
            "Preyivore",
        ]),
        (false, false, true) => rng.pick(&[
            "Osmotroph",
            "Absorber",
            "Heterotroph",
            "Saprotroph",
            "Phagotroph",
            "Organotroph",
            "Osmovore",
            "Absorptroph",
            "Chemoorganotroph",
            "Detritivore",
            "Lysotroph",
            "Endotroph",
            "Absorbiont",
            "Nutritroph",
            "Saprophyte",
        ]),
        (false, false, false) => rng.pick(&[
            "Lithotroph",
            "Chemotroph",
            "Protobiont",
            "Microbiont",
            "Nanotroph",
            "Chemolit",
            "Abiotroph",
            "Nullitroph",
            "Cryptobiont",
            "Endobiont",
            "Prototroph",
            "Archivore",
        ]),
    };

    // -- Mobility prefix -------------------------------------------------------
    let mobility: Option<&str> = if has_flag && has_myo {
        Some(rng.pick(&[
            "Natato", "Kineto", "Motile ", "Agile ", "Veloci", "Tachyo", "Cursori", "Rapidio",
            "Dynamo", "Impeto",
        ]))
    } else if has_flag {
        Some(rng.pick(&[
            "Flagello",
            "Natato",
            "Plankto",
            "Necton",
            "Motile ",
            "Necto",
            "Flagelli",
            "Undulo",
            "Vibrio",
            "Pelagic ",
            "Drifting ",
            "Gliding ",
        ]))
    } else if has_myo {
        Some(rng.pick(&[
            "Myo",
            "Sarco",
            "Kineto",
            "Pulsatile ",
            "Contractile ",
            "Musculo",
            "Fibro",
            "Tensio",
            "Spasmo",
            "Rhythmo",
        ]))
    } else if has_cilia {
        Some(rng.pick(&[
            "Cilio",
            "Tricho",
            "Ciliated ",
            "Vibrio",
            "Flagelli",
            "Trichoid ",
            "Cilioform ",
            "Waving ",
            "Sweeping ",
            "Beating ",
        ]))
    } else if has_buoy {
        Some(rng.pick(&[
            "Plankto",
            "Pelagic ",
            "Buoyant ",
            "Necto",
            "Floato",
            "Drifting ",
            "Pelago",
            "Mesopelagic ",
            "Neusto",
            "Epipelagic ",
        ]))
    } else if has_glue {
        Some(rng.pick(&[
            "Sessile ",
            "Bento",
            "Adheso",
            "Fixo",
            "Benthic ",
            "Anchored ",
            "Attached ",
            "Sedentary ",
            "Rooted ",
            "Immobile ",
        ]))
    } else {
        None
    };

    // -- Structural suffix -----------------------------------------------------
    let structure: Option<&str> = if has_oculo && has_signal {
        Some(rng.pick(&[
            " Coordinator",
            " Signaller",
            " Networker",
            " Synapse",
            " Ganglion",
            " Nexus",
            " Relay",
            " Conductor",
            " Integrator",
            " Plexus",
            " Arbiter",
            " Sentinel",
        ]))
    } else if has_oculo {
        Some(rng.pick(&[
            " Sensor",
            " Scout",
            " Oculus",
            " Watcher",
            " Receptor",
            " Observer",
            " Detector",
            " Vigilant",
            " Perceiver",
            " Lookout",
            " Tracker",
            " Seeker",
        ]))
    } else if has_vasc && has_embryo {
        Some(rng.pick(&[
            " Colony",
            " Zooid",
            " Polyp",
            " Cormus",
            " Siphon",
            " Consortium",
            " Commune",
            " Symbiont",
            " Collective",
            " Aggregate",
            " Cluster",
            " Bloom",
        ]))
    } else if has_embryo {
        Some(rng.pick(&[
            " Propagator",
            " Spawner",
            " Gametophyte",
            " Zygote",
            " Breeder",
            " Progenitor",
            " Germinator",
            " Seeder",
            " Reproducer",
            " Proliferator",
            " Budder",
            " Divider",
        ]))
    } else if has_glue && has_devour {
        Some(rng.pick(&[
            " Trapper",
            " Snare",
            " Ambusher",
            " Lurer",
            " Tentacle",
            " Ensnarer",
            " Captor",
            " Stalker",
            " Ambush",
            " Grappler",
            " Seizer",
            " Clutcher",
        ]))
    } else if has_lipo {
        Some(rng.pick(&[
            " Reservoir",
            " Cache",
            " Adipocyte",
            " Hoarder",
            " Lipovore",
            " Storer",
            " Accumulator",
            " Depot",
            " Vessel",
            " Cistern",
            " Sac",
            " Vacuole",
        ]))
    } else if has_signal {
        Some(rng.pick(&[
            " Emitter",
            " Broadcaster",
            " Transmitter",
            " Signaller",
            " Pheromone",
            " Beacon",
            " Pulse",
            " Radiator",
        ]))
    } else if has_vasc {
        Some(rng.pick(&[
            " Network",
            " Web",
            " Lattice",
            " Mesh",
            " Conduit",
            " Pipeline",
            " Circuit",
            " Plexus",
        ]))
    } else if mode_count >= 24 {
        Some(rng.pick(&[
            " Complex",
            " Elaborate",
            " Metazoan",
            " Macroorganism",
            " Superorganism",
            " Megabiont",
            " Macroform",
            " Archetype",
        ]))
    } else if mode_count <= 3 {
        Some(rng.pick(&[
            " Primitive",
            " Progenitor",
            " Protocell",
            " Rudiment",
            " Archaic",
            " Primordial",
            " Vestige",
            " Precursor",
        ]))
    } else {
        None
    };

    // -- Optional habitat/size adjective (adds variety on ~40% of names) -------
    let habitat: Option<&str> = if rng.chance(40) {
        if has_buoy || has_flag {
            Some(rng.pick(&[
                "Pelagic ",
                "Planktonic ",
                "Nektonic ",
                "Oceanic ",
                "Aquatic ",
                "Abyssal ",
                "Mesopelagic ",
                "Epipelagic ",
            ]))
        } else if has_glue {
            Some(rng.pick(&[
                "Benthic ",
                "Littoral ",
                "Epilithic ",
                "Saxicolous ",
                "Rupestral ",
                "Endolithic ",
                "Cryptic ",
                "Interstitial ",
            ]))
        } else if has_photo {
            Some(rng.pick(&[
                "Photic ",
                "Euphotic ",
                "Epiphytic ",
                "Aerial ",
                "Radiant ",
                "Luminous ",
                "Solar ",
                "Heliac ",
            ]))
        } else {
            Some(rng.pick(&[
                "Micro",
                "Nano",
                "Macro",
                "Giant ",
                "Dwarf ",
                "Cryptic ",
                "Vestigial ",
                "Emergent ",
            ]))
        }
    } else {
        None
    };

    // -- Assemble --------------------------------------------------------------
    // Pick one of several assembly patterns for variety
    let name = match rng.u8() % 4 {
        0 => {
            // habitat + mobility + trophic
            let h = habitat.unwrap_or("");
            let m = mobility.unwrap_or("");
            format!("{}{}{}", h, m, trophic)
        }
        1 => {
            // mobility + trophic + structure
            let m = mobility.unwrap_or("");
            let s = structure.unwrap_or("");
            format!("{}{}{}", m, trophic, s)
        }
        2 => {
            // habitat + trophic + structure
            let h = habitat.unwrap_or("");
            let s = structure.unwrap_or("");
            format!("{}{}{}", h, trophic, s)
        }
        _ => {
            // mobility + trophic (simplest, always valid)
            let m = mobility.unwrap_or("");
            format!("{}{}", m, trophic)
        }
    };

    // Trim and capitalise first letter
    let name = name.trim().to_string();
    let mut chars = name.chars();
    match chars.next() {
        None => "Microbiont".to_string(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Returns true if the given name is a default/unnamed placeholder.
pub fn is_default_name(name: &str) -> bool {
    let t = name.trim();
    t.is_empty()
        || t.eq_ignore_ascii_case("untitled genome")
        || t.eq_ignore_ascii_case("unnamed genome")
}

// -- Minimal seeded RNG --------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn step(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn pick<'a>(&mut self, slice: &[&'a str]) -> &'a str {
        let idx = (self.step() as usize) % slice.len();
        slice[idx]
    }
    fn u8(&mut self) -> u8 {
        (self.step() & 0xFF) as u8
    }
    /// Returns true with `percent`% probability (0-100).
    fn chance(&mut self, percent: u64) -> bool {
        (self.step() % 100) < percent
    }
}

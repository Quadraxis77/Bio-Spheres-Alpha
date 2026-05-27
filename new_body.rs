ST 1: ASSIG ROLSO NDOMMOD STS
       // Rlese funtional idnitiMode indiesare jus stoge slos.
       // The generator pckwhi lot achrole lvs i, th wires hem
       // ogetrwith signal hannels. Nthg is lockdto a ixd indxShufflethe first N mde slots to assign roles unpreditably.
        // We use 8 rles inimum; cmplex creaures use up to 12.
        let num_roles: usze = rng.i32_range(8, 13) as usize;
        let mut slts: Vec<usize> = (0..um_roles).collect();
       // Fisher-Yaes shuffle
        for i in (1..slots.len()).rev() {
            let j = rng.u32(i as u32 + 1) as usize;
            slots.swap(i, j);
        }

        // Role → mode index mapping (named for clarit, not for fixed ositions)
        ltr_zygote:     usize = slots[0];
        let r_stem:       usize = slots[1];
        let r_struct_a:   usize = slots[2]; // primary structural / growth cell
        let r_struct_b:   usize = slots[3]; // secondary structural (bilateral mirror)
        let r_feeder:     usize = slots[4]; // nutrient producer
        let r_loco:       usize = slots[5]; // locomotion specialist
        let r_specialist: usize = slots[6]; // terminal body function
        let r_gonad:      usize = slots[7]; // reproductive organ, emits maturity
 Optional roles (only present if num_roles > 8)
        let r_sensor:     Option<usize> = if num_roles > 8  { Some(slots[8])  } else { None };
        let r_adult_struct: Option<usize> = if num_roles > 9  { Some(slots[9])  } else { None };
        let r_extra_spec: Option<usize> = if num_roles > 10 {Some(slots[1]) } else { None };
        let r_anchor:     Option<usize>  if num_roles > 11 { Some(slots[11]) } else { None };

        geme.iitial_mode = r_zygot as i32;

        // ══════════════════════════════════════════════════════════════════════
        // STEP 2: INDEPENDENT TRAIT SELECTION
        // ══════════════════════════════════════════════════════════════════════

        //Locomotion:  5=none_cell*pick(&[12, 1, , 3, 4]:_cell Specialistterminalfunctin(attack/ore/achor/sne)t pec_ce_type: i32 =*rng.pick(&[232, 4, 6, 8, 9, 11]);

        // Bdytffes ..  :howthes(deresrm fo
geometry: xis, symmytyldg040pral
Bs— child rienion is thpimry spitoladgbdgpratist312tista23:s →

        // ══════════════════════════════════════════════════════════════════════
        // STEP 3: SIGNAL CHANNEL ASSIGNMENTS
        //
        // Channels 8–15 are regulation (any cell can emit/receive).
        // Channels 0–7 are oculocyte sensing only.
        //
        // We assign channels from a shuffled pool so different creatures use
        // different channels for the same roles — preventing cross-talk if
        // multiple creatures share a world.
        // ══════════════════════════════════════════════════════════════════════
        let mut ch_pool: Vec<i32> = (8..=15).collect();
        for i in (1..ch_pool.len()).rev() {
            let j = rng.u32(i as u32 + 1) as usize;
            ch_pool.swap(i, j);
        }
        let ch_anterior: i32 = ch_pool[0]; // emitted by stem, marks head end
        let ch_lateral:  i32 = ch_pool[1]; // emitted by spine, marks axis distance
        let ch_feeder:   i32 = ch_pool[2]; // emitted by feeder, gates stem division
        let ch_maturity: i32 = ch_pool[3]; // emitted by gonad, triggers grow→adult
        let ch_repro:    i32 = ch_pool[4]; // emitted by gonad, triggers egg shedding

        // Oculocyte sensing channel (0–7): sensor fires on this, loco reads it
        let ch_sense: i32 = rng.i32_range(0, 8);

        // ══════════════════════════════════════════════════════════════════════
        // STEP 4: HELPER CLOSURES
        // ══════════════════════════════════════════════════════════════════════

        // Apply shared adhesion parameters
        fn apply_adhesion(m: &mut ModeSettings, rest: f32, lin: f32, ang: f32, flex: bool) {
            m.adhesion_settings.rest_length                  = rest;
            m.adhesion_settings.linear_spring_stiffness      = lin;
            m.adhesion_settings.orientation_spring_stiffness = ang;
            m.adhesion_settings.linear_spring_damping        = lin * 0.03;
            m.adhesion_settings.orientation_spring_damping   = ang * 0.01;
            m.adhesion_settings.break_force                  = 1000.0;
            m.adhesion_settings.enable_twist_constraint      = !flex;
        }

        // Apply cell-type-specific behaviour parameters
        fn apply_type(m: &mut ModeSettings, cell_type: i32, ch_sense: i32, rng: &mut Rng) {
            m.cell_type = cell_type;
            match cell_type {
                1 => { // Flagellocyte: signal-gated speed
                    m.swim_force                 = rng.f32(0.8, 2.5);
                    m.flagellocyte_use_signal    = true;
                    m.flagellocyte_signal_channel = ch_sense;
                    m.flagellocyte_speed_a       = rng.f32(0.2, 0.8); // slow: no signal
                    m.flagellocyte_speed_b       = rng.f32(1.0, 2.5); // fast: signal detected
                    m.flagellocyte_threshold_c   = 1.0;
                }
                2 => {} // Phagocyte — no extra params
                3 => {} // Photocyte — no extra params
                4 => { m.nutrient_priority = rng.f32(1.5, 3.0); } // Lipocyte
                5 => { m.buoyancy_force    = rng.f32(0.3, 0.8); } // Buoyocyte
                6 => { // Glueocyte
                    m.glueocyte_env_adhesion  = true;
                    m.glueocyte_cell_adhesion = false;
                }
                8 => { // Ciliocyte: signal-gated speed
                    m.cilia_use_signal     = true;
                    m.cilia_signal_channel = ch_sense;
                    m.cilia_speed_below    = rng.f32(0.2, 0.6);
                    m.cilia_speed_above    = rng.f32(0.7, 1.0);
                    m.cilia_threshold      = 1.0;
                    m.cilia_push_bonded    = false;
                    m.cilia_attract_force  = rng.f32(0.0, 0.3);
                }
                9 => { // Myocyte: signal-gated contraction
                    m.myocyte_use_signal        = true;
                    m.myocyte_signal_channel    = ch_sense;
                    m.myocyte_contraction_above = rng.f32(0.4, 0.8);
                    m.myocyte_contraction_below = rng.f32(0.0, 0.2);
                    m.myocyte_threshold         = 1.0;
                    m.myocyte_pulse_rate        = rng.f32(0.5, 2.0);
                    m.myocyte_pulse_phase       = rng.u32(2) as i32;
                }
                10 => { // Embryocyte
                    m.embryocyte_use_timer     = true;
                    m.embryocyte_release_timer = rng.f32(3.0, 8.0);
                }
                11 => { // Devorocyte
                    m.devorocyte_consume_range = rng.f32(1.5, 3.0);
                    m.devorocyte_consume_rate  = rng.f32(30.0, 80.0);
                }
                12 => { // Vasculocyte
                    m.vascular_outlet   = true;
                    m.nutrient_priority = 0.5;
                }
                _ => {}
            }
        }


        // ══════════════════════════════════════════════════════════════════════
        // STEP 5: WIRE THE MODES
        //
        // Each mode is configured by its role. The mode index (r_stem, r_feeder,
        // etc.) is whatever the shuffle assigned — the wiring uses those indices
        // directly, so the body plan is fully determined by the signal graph,
        // not by slot positions.
        // ══════════════════════════════════════════════════════════════════════

        // ── ZYGOTE ────────────────────────────────────────────────────────────
        // Embryocyte. Incubates and releases the stem as a free organism.
        // Self-renews so the zygote keeps producing offspring.
        {
            let m = &mut genome.modes[r_zygote];
            m.name                             = "Zygote".to_string();
            m.max_cell_size                    = stem_size;
            m.nutrient_gain_rate               = 25.0;
            m.nutrient_priority                = 3.0;
            m.split_mass                       = stem_mass * 1.2;
            m.split_interval                   = stem_ivl;
            m.parent_make_adhesion             = false;
            m.max_splits                       = -1; // infinite: keeps releasing
            m.enable_parent_angle_snapping     = false;
            m.membrane_stiffness               = membrane;
            apply_type(m, 10, ch_sense, &mut rng); // Embryocyte
            m.embryocyte_use_timer             = true;
            m.embryocyte_release_timer         = rng.f32(5.0, 12.0);
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            // Child A = released stem (free, no adhesion)
            m.child_a.mode_number              = r_stem as i32;
            m.child_a.keep_adhesion            = false;
            // Child B = zygote self-renews
            m.child_b.mode_number              = r_zygote as i32;
            m.child_b.keep_adhesion            = false;
            m.mode_a_after_splits              = r_stem as i32;
            m.mode_b_after_splits              = r_zygote as i32;
            m.child_a_after_split_keep_adhesion = false;
            m.child_b_after_split_keep_adhesion = false;
        }

        // ── STEM ──────────────────────────────────────────────────────────────
        // Self-renewing growth engine. Emits ch_anterior (head gradient).
        // Division gated on ch_feeder so it only grows when fed.
        // Spawns struct_a (child A) and renews itself (child B).
        // After spine_segs splits → both children become gonad.
        // On ch_maturity → switches to feeder role (stem becomes a feeder
        // in the adult body, contributing to nutrient supply).
        {
            let m = &mut genome.modes[r_stem];
            m.name                             = "Stem".to_string();
            m.cell_type                        = 2; // Phagocyte
            m.max_cell_size                    = stem_size;
            m.nutrient_priority                = 2.5;
            m.split_mass                       = stem_mass;
            m.split_interval                   = stem_ivl;
            m.parent_make_adhesion             = true;
            m.max_splits                       = spine_segs;
            m.enable_parent_angle_snapping     = false;
            m.parent_split_direction           = Vec2::new(spine_pitch, 0.0);
            m.split_ratio                      = 0.5;
            m.membrane_stiffness               = membrane * 1.1;
            // Gate division: only divide when feeder is supplying nutrients
            m.division_signal_channel          = ch_feeder;
            m.division_signal_threshold        = 0.5;
            m.division_signal_invert           = false;
            // Emit anterior gradient
            m.regulation_emit_channel          = ch_anterior;
            m.regulation_emit_value            = 50.0;
            m.regulation_emit_hops             = 20;
            // On maturity: stem transitions to feeder role in adult body
            m.mode_switch_signal_channel       = ch_maturity;
            m.mode_switch_signal_threshold     = 1.0;
            m.mode_switch_target               = r_feeder as i32;
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            m.child_a.mode_number              = r_struct_a as i32;
            m.child_a.keep_adhesion            = true;
            m.child_b.mode_number              = r_stem as i32;
            m.child_b.keep_adhesion            = true;
            // After spine_segs: both become gonad
            m.mode_a_after_splits              = r_gonad as i32;
            m.mode_b_after_splits              = r_gonad as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }


        // ── STRUCT_A (primary structural / growth cell) ───────────────────────
        // Grows along the body axis. Emits ch_lateral (axis distance gradient).
        // Spawns struct_b (child A, branching direction) and continues as struct_a.
        // After branch_segs splits → both children become specialist.
        // On ch_maturity → switches to locomotion role.
        {
            let m = &mut genome.modes[r_struct_a];
            m.name                             = "Struct-A".to_string();
            m.cell_type                        = 2; // Phagocyte structural
            m.max_cell_size                    = spine_size;
            m.nutrient_priority                = 1.5;
            m.split_mass                       = spine_mass;
            m.split_interval                   = spine_ivl;
            m.parent_make_adhesion             = true;
            m.max_splits                       = branch_segs;
            m.enable_parent_angle_snapping     = false;
            m.parent_split_direction           = Vec2::new(spine_pitch, 0.0);
            m.split_ratio                      = 0.5;
            m.membrane_stiffness               = membrane;
            // Emit lateral gradient
            m.regulation_emit_channel          = ch_lateral;
            m.regulation_emit_value            = 30.0;
            m.regulation_emit_hops             = 10;
            // On maturity: structural cells activate locomotion
            m.mode_switch_signal_channel       = ch_maturity;
            m.mode_switch_signal_threshold     = 1.0;
            m.mode_switch_target               = r_loco as i32;
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            // Child A = struct_b (branches off with symmetry-dependent orientation)
            m.child_a.mode_number              = r_struct_b as i32;
            m.child_a.keep_adhesion            = true;
            m.child_a.orientation              = match symmetry {
                0 => branch_quat_a,      // bilateral: A-side
                1 => branch_quat_a,      // radial: same direction
                _ => branch_quat_spiral, // spiral: twisted
            };
            // Child B = struct_a continues the axis
            m.child_b.mode_number              = r_struct_a as i32;
            m.child_b.keep_adhesion            = true;
            // After branch_segs: both become specialist
            m.mode_a_after_splits              = r_specialist as i32;
            m.mode_b_after_splits              = r_specialist as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }

        // ── STRUCT_B (secondary structural / bilateral mirror) ────────────────
        // Branches off struct_a. Self-extends branch_ext times.
        // Mirror orientation of struct_a's branch for bilateral symmetry.
        // On ch_maturity → switches to specialist role.
        {
            let m = &mut genome.modes[r_struct_b];
            m.name                             = "Struct-B".to_string();
            m.cell_type                        = 2; // Phagocyte structural
            m.max_cell_size                    = branch_size;
            m.nutrient_priority                = 1.0;
            m.split_mass                       = branch_mass;
            m.split_interval                   = branch_ivl;
            m.parent_make_adhesion             = true;
            m.max_splits                       = branch_ext;
            m.enable_parent_angle_snapping     = false;
            m.parent_split_direction           = Vec2::new(branch_deg * 0.5, 0.0);
            m.split_ratio                      = rng.f32(0.4, 0.6);
            m.membrane_stiffness               = membrane * 0.85;
            m.max_adhesions                    = rng.i32_range(4, 10);
            // On maturity: branch cells become specialists
            m.mode_switch_signal_channel       = ch_maturity;
            m.mode_switch_signal_threshold     = 1.0;
            m.mode_switch_target               = r_specialist as i32;
            apply_adhesion(m, adh_rest * 1.05, adh_lin * 0.75, adh_ang * 0.65, flex);
            // Child A = struct_b continues (self-extension)
            m.child_a.mode_number              = r_struct_b as i32;
            m.child_a.keep_adhesion            = true;
            m.child_a.orientation              = match symmetry {
                0 => branch_quat_b,      // bilateral: B-side (mirror of A)
                1 => branch_quat_a,      // radial: same as A
                _ => branch_quat_spiral, // spiral: same twist
            };
            // Child B = struct_b continues
            m.child_b.mode_number              = r_struct_b as i32;
            m.child_b.keep_adhesion            = true;
            // After branch_ext: both become specialist
            m.mode_a_after_splits              = r_specialist as i32;
            m.mode_b_after_splits              = r_specialist as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }


        // ── FEEDER ────────────────────────────────────────────────────────────
        // Nutrient producer. Emits ch_feeder continuously so the stem knows
        // food is available and can divide. Terminal — does not divide further
        // once mature. The stem transitions into this role on ch_maturity,
        // so the adult body has more feeders than the juvenile.
        {
            let m = &mut genome.modes[r_feeder];
            m.name                             = "Feeder".to_string();
            m.max_cell_size                    = branch_size;
            m.nutrient_priority                = 1.8;
            m.split_mass                       = spec_mass;
            m.split_interval                   = spec_ivl;
            m.parent_make_adhesion             = true;
            m.max_splits                       = rng.i32_range(1, 3); // small cluster
            m.enable_parent_angle_snapping     = false;
            m.split_ratio                      = 0.5;
            m.membrane_stiffness               = membrane * 0.9;
            // Emit feeder abundance signal continuously
            m.regulation_emit_channel          = ch_feeder;
            m.regulation_emit_value            = 20.0;
            m.regulation_emit_hops             = 15;
            apply_adhesion(m, adh_rest, adh_lin * 0.8, adh_ang * 0.7, flex);
            apply_type(m, feed_cell_type, ch_sense, &mut rng);
            // Feeder self-renews after max_splits
            m.child_a.mode_number              = r_feeder as i32;
            m.child_a.keep_adhesion            = true;
            m.child_b.mode_number              = r_feeder as i32;
            m.child_b.keep_adhesion            = true;
            m.mode_a_after_splits              = r_feeder as i32;
            m.mode_b_after_splits              = r_feeder as i32;
            m.child_a_after_split_keep_adhesion = true;
            m.child_b_after_split_keep_adhesion = true;
        }

        // ── LOCOMOTION ────────────────────────────────────────────────────────
        // Active movement cell. Reads ch_sense so it speeds up when the sensor
        // detects food or a target. Structural cells transition into this role
        // when ch_maturity floods the body — the whole body activates movement.
        // Terminal: does not divide.
        {
            let m = &mut genome.modes[r_loco];
            m.name                             = "Loco".to_string();
            m.max_cell_size                    = spec_size * 1.2;
            m.nutrient_priority                = rng.f32(1.0, 2.0);
            m.split_mass                       = spec_mass;
            m.split_interval                   = spec_ivl;
            m.parent_make_adhesion             = false;
            m.max_splits                       = 0; // terminal
            m.enable_parent_angle_snapping     = false;
            m.membrane_stiffness               = membrane * rng.f32(0.8, 1.1);
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            // Buoyocyte (type 5) doesn't use ch_sense — it just floats
            let effective_loco = if loco_cell_type == 5 { 5 } else { loco_cell_type };
            apply_type(m, effective_loco, ch_sense, &mut rng);
            m.child_a.mode_number              = r_loco as i32;
            m.child_b.mode_number              = r_loco as i32;
            m.mode_a_after_splits              = r_loco as i32;
            m.mode_b_after_splits              = r_loco as i32;
        }

        // ── SPECIALIST ────────────────────────────────────────────────────────
        // Terminal body function. Branch tips and spine caps transition here
        // after exhausting their growth splits. Also activated by ch_maturity
        // in any remaining structural cells.
        {
            let m = &mut genome.modes[r_specialist];
            m.name                             = "Specialist".to_string();
            m.max_cell_size                    = spec_size;
            m.nutrient_priority                = rng.f32(0.8, 1.6);
            m.split_mass                       = spec_mass;
            m.split_interval                   = spec_ivl;
            m.parent_make_adhesion             = false;
            m.max_splits                       = 0; // terminal
            m.enable_parent_angle_snapping     = false;
            m.membrane_stiffness               = membrane * rng.f32(0.7, 1.1);
            apply_adhesion(m, adh_rest, adh_lin * 0.9, adh_ang * 0.9, flex);
            apply_type(m, spec_cell_type, ch_sense, &mut rng);
            m.child_a.mode_number              = r_specialist as i32;
            m.child_b.mode_number              = r_specialist as i32;
            m.mode_a_after_splits              = r_specialist as i32;
            m.mode_b_after_splits              = r_specialist as i32;
        }


        // ── GONAD ─────────────────────────────────────────────────────────────
        // Reproductive organ. The stem transitions here after spine_segs splits.
        // Emits ch_maturity (floods the whole body, triggering all grow→adult
        // mode switches). Also emits ch_repro to trigger egg shedding.
        // Sheds num_eggs free zygotes, then reverts to stem to rebuild.
        {
            let m = &mut genome.modes[r_gonad];
            m.name                             = "Gonad".to_string();
            m.cell_type                        = 2; // Phagocyte — auto-gains mass
            m.max_cell_size                    = stem_size;
            m.nutrient_priority                = 2.0;
            m.nutrient_gain_rate               = 20.0;
            m.split_mass                       = stem_mass;
            m.split_interval                   = stem_ivl;
            m.parent_make_adhesion             = false; // eggs detach freely
            m.max_splits                       = num_eggs;
            m.enable_parent_angle_snapping     = false;
            m.split_ratio                      = 0.5;
            m.membrane_stiffness               = membrane;
            // Emit maturity signal: floods body, triggers all grow→adult switches
            m.regulation_emit_channel          = ch_maturity;
            m.regulation_emit_value            = 50.0;
            m.regulation_emit_hops             = 20;
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            // Child A = free zygote (detaches, starts new organism)
            m.child_a.mode_number              = r_zygote as i32;
            m.child_a.keep_adhesion            = false;
            // Child B = gonad continues shedding
            m.child_b.mode_number              = r_gonad as i32;
            m.child_b.keep_adhesion            = true;
            // After num_eggs: gonad reverts to stem → organism rebuilds
            m.mode_a_after_splits              = r_zygote as i32;
            m.mode_b_after_splits              = r_stem as i32;
            m.child_a_after_split_keep_adhesion = false;
            m.child_b_after_split_keep_adhesion = true;
        }

        // ── SENSOR (optional) ─────────────────────────────────────────────────
        // Oculocyte. Senses food or cells along its forward axis and fires
        // ch_sense into the adhesion network. Locomotion cells read ch_sense
        // and speed up. Only present if the creature has locomotion.
        if let Some(idx) = r_sensor {
            let m = &mut genome.modes[idx];
            m.name                             = "Sensor".to_string();
            m.cell_type                        = 7; // Oculocyte
            m.max_cell_size                    = spec_size;
            m.nutrient_priority                = 2.0;
            m.split_mass                       = spec_mass;
            m.split_interval                   = spec_ivl;
            m.parent_make_adhesion             = false;
            m.max_splits                       = 0; // terminal
            m.enable_parent_angle_snapping     = false;
            m.membrane_stiffness               = membrane;
            // Sense food for phagocytes/photocytes, cells for devorocytes
            m.oculocyte_sense_type             = if feed_cell_type == 11 { 1 << 1 } else { 1 << 0 }; // Food=bit1, Cell=bit0
            m.oculocyte_signal_channel         = ch_sense;
            m.oculocyte_signal_value           = rng.f32(5.0, 20.0);
            m.oculocyte_signal_hops            = rng.i32_range(3, 12);
            m.oculocyte_ray_length             = rng.f32(10.0, 40.0);
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            m.child_a.mode_number              = idx as i32;
            m.child_b.mode_number              = idx as i32;
            m.mode_a_after_splits              = idx as i32;
            m.mode_b_after_splits              = idx as i32;
        }

        // ── ADULT STRUCT (optional) ───────────────────────────────────────────
        // An alternative adult form for structural cells. When present, struct_a
        // switches to this instead of r_loco on ch_maturity — giving the creature
        // a distinct adult structural identity (e.g. vasculocyte transport network).
        if let Some(idx) = r_adult_struct {
            let m = &mut genome.modes[idx];
            m.name                             = "Adult-Struct".to_string();
            m.cell_type                        = 12; // Vasculocyte: nutrient transport
            m.max_cell_size                    = spine_size;
            m.nutrient_priority                = 0.5;
            m.split_mass                       = spec_mass;
            m.split_interval                   = spec_ivl;
            m.parent_make_adhesion             = false;
            m.max_splits                       = 0; // terminal
            m.enable_parent_angle_snapping     = false;
            m.membrane_stiffness               = membrane;
            m.vascular_outlet                  = rng.bool(0.4); // some are outlets
            apply_adhesion(m, adh_rest, adh_lin, adh_ang, flex);
            m.child_a.mode_number              = idx as i32;
            m.child_b.mode_number              = idx as i32;
            m.mode_a_after_splits              = idx as i32;
            m.mode_b_after_splits              = idx as i32;
            // Redirect struct_a's maturity switch to this adult form
            genome.modes[r_struct_a].mode_switch_target = idx as i32;
        }

        // ── EXTRA SPECIALIST (optional) ───────────────────────────────────────
        // A second specialist type. Struct_b switches to this on ch_maturity
        // instead of r_specialist, giving branch tips a different function
        // from spine tips. Creates functional differentiation along the axis.
        if let Some(idx) = r_extra_spec {
            let extra_type: i32 = *rng.pick(&[1i32, 3, 5, 8, 9, 11]);
            let m = &mut genome.modes[idx];
            m.name                             = "Extra-Spec".to_string();
            m.max_cell_size                    = spec_size;
            m.nutrient_priority                = rng.f32(0.6, 1.4);
            m.split_mass                       = spec_mass;
            m.split_interval                   = spec_ivl;
            m.parent_make_adhesion             = false;
            m.max_splits                       = 0; // terminal
            m.enable_parent_angle_snapping     = false;
            m.membrane_stiffness               = membrane * 0.85;
            apply_adhesion(m, adh_rest, adh_lin * 0.9, adh_ang * 0.9, flex);
            apply_type(m, extra_type, ch_sense, &mut rng);
            m.child_a.mode_number              = idx as i32;
            m.child_b.mode_number              = idx as i32;
            m.mode_a_after_splits              = idx as i32;
            m.mode_b_after_splits              = idx as i32;
            // Redirect struct_b's maturity switch to this extra specialist
            genome.modes[r_struct_b].mode_switch_target = idx as i32;
        }

        // ── ANCHOR (optional) ─────────────────────────────────────────────────
        // Glueocyte. Present in sessile creatures (no locomotion). Struct_b
        // switches to this on ch_maturity, anchoring branch tips to the
        // environment. The creature grows, then locks itself in place.
        if let Some(idx) = r_anchor {
            let m = &mut genome.modes[idx];
            m.name                             = "Anchor".to_string();
            m.cell_type                        = 6; // Glueocyte
            m.max_cell_size                    = spec_size;
            m.nutrient_priority                = 1.5;
            m.split_mass                       = spec_mass;
            m.split_interval                   = spec_ivl;
            m.parent_make_adhesion             = false;
            m.max_splits                       = 0; // terminal
            m.enable_parent_angle_snapping     = false;
            m.membrane_stiffness               = membrane * 1.2;
            m.glueocyte_env_adhesion           = true;
            m.glueocyte_cell_adhesion          = false;
            apply_adhesion(m, adh_rest, adh_lin * 1.2, adh_ang * 1.2, false); // always rigid
            m.child_a.mode_number              = idx as i32;
            m.child_b.mode_number              = idx as i32;
            m.mode_a_after_splits              = idx as i32;
            m.mode_b_after_splits              = idx as i32;
            // Redirect struct_b's maturity switch to anchor
            genome.modes[r_struct_b].mode_switch_target = idx as i32;
        }

        genome
    }
}


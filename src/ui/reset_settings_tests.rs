use super::*;
use crate::ui::types::{SimulationMode, UiTheme};

fn frame(
    ctx: &Context,
    state: &mut GlobalUiState,
    events: Vec<egui::Event>,
) -> (bool, Vec<(String, egui::Pos2)>) {
    let mut active = false;
    let output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 700.0),
            )),
            events,
            ..Default::default()
        },
        |ui| {
            ui.set_width(44.0);
            let mut editor = crate::ui::GenomeEditorState::factory_defaults();
            let mut dock = crate::ui::DockManager::new();
            crate::ui::ui_system::render_side_rail(ui, state, &mut editor, &mut dock, None);
            let rect = editor.panel_rects["ResetSettings"];
            active = render_dialogs(ui.ctx(), state, Some(rect), None);
        },
    );
    fn texts(shape: &egui::epaint::Shape, result: &mut Vec<(String, egui::Pos2)>) {
        match shape {
            egui::epaint::Shape::Text(text) => {
                result.push((text.galley.text().to_string(), text.pos))
            }
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    texts(shape, result);
                }
            }
            _ => {}
        }
    }
    let mut labels = Vec::new();
    for shape in output.shapes {
        texts(&shape.shape, &mut labels);
    }
    (active, labels)
}

fn click(ctx: &Context, state: &mut GlobalUiState, label: &str) {
    frame(ctx, state, vec![]);
    let (_, labels) = frame(ctx, state, vec![]);
    let position = labels
        .iter()
        .find(|(text, _)| text == label)
        .unwrap_or_else(|| panic!("missing button {label}: {labels:?}"))
        .1
        + egui::vec2(5.0, 5.0);
    for pressed in [true, false] {
        frame(
            ctx,
            state,
            vec![
                egui::Event::PointerMoved(position),
                egui::Event::PointerButton {
                    pos: position,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
    }
}

#[test]
fn reset_intro_appears_once_in_each_scene_and_survives_save_and_reset() {
    let ctx = Context::default();
    let mut state = GlobalUiState {
        main_scene_quick_start_seen: true,
        ..Default::default()
    };
    for mode in [SimulationMode::Preview, SimulationMode::Gpu] {
        state.current_mode = mode;
        assert!(frame(&ctx, &mut state, vec![]).0);
        click(&ctx, &mut state, "Got it");
        assert!(!frame(&ctx, &mut state, vec![]).0);
        state = ron::from_str(&ron::to_string(&state).unwrap()).unwrap();
        assert!(!frame(&ctx, &mut state, vec![]).0);
        state.restore_scene_settings();
        assert!(!frame(&ctx, &mut state, vec![]).0);
    }
    assert_eq!(state.reset_intro_seen, [true, true]);
}

#[test]
fn reset_requires_confirmation_and_cancel_keeps_preferences() {
    let ctx = Context::default();
    let mut state = GlobalUiState {
        reset_intro_seen: [true; 2],
        ui_scale: 3.0,
        ..Default::default()
    };
    click(&ctx, &mut state, "⟲");
    assert!(state.show_settings_reset_dialog);
    assert!(!state.reset_scene_settings_requested);
    click(&ctx, &mut state, "Cancel");
    assert!(!state.show_settings_reset_dialog);
    assert_eq!(state.ui_scale, 3.0);
    click(&ctx, &mut state, "⟲");
    click(&ctx, &mut state, "Reset scene settings");
    assert!(state.reset_scene_settings_requested);
}

#[test]
fn main_scene_guide_follows_reset_intro_and_covers_water_cells_and_brightness_once() {
    let ctx = Context::default();
    let mut state = GlobalUiState {
        current_mode: SimulationMode::Gpu,
        ..Default::default()
    };
    let (_, labels) = frame(&ctx, &mut state, vec![]);
    assert!(!labels.iter().any(|(text, _)| text.contains("Quick start:")));
    click(&ctx, &mut state, "Got it");
    frame(&ctx, &mut state, vec![]);
    let (active, labels) = frame(&ctx, &mut state, vec![]);
    assert!(active);
    for instruction in [
        "small amount",
        "Hold Alt",
        "select Add",
        "Left-click",
        "0–5",
        "around 3 is temperate",
    ] {
        assert!(
            labels.iter().any(|(text, _)| text.contains(instruction)),
            "missing {instruction}"
        );
    }
    assert!(!labels
        .iter()
        .any(|(text, _)| text.contains("Panels →") || text == "Show Brightness"));
    click(&ctx, &mut state, "Let's play");
    assert!(state.main_scene_quick_start_seen);
    assert!(!frame(&ctx, &mut state, vec![]).0);
    state = ron::from_str(&ron::to_string(&state).unwrap()).unwrap();
    state.restore_scene_settings();
    assert!(!frame(&ctx, &mut state, vec![]).0);
}

#[test]
fn brightness_pointer_tracks_visible_slider_then_falls_back_to_the_lighting_tab() {
    let tab = Rect::from_min_size(egui::pos2(700.0, 40.0), egui::vec2(90.0, 24.0));
    let slider = Rect::from_min_size(egui::pos2(700.0, 120.0), egui::vec2(160.0, 24.0));
    let mut rects = std::collections::HashMap::new();
    rects.insert("LightingTab".to_string(), tab);
    assert_eq!(brightness_target(&rects), Some(tab));
    rects.insert("SunBrightness".to_string(), slider);
    assert_eq!(brightness_target(&rects), Some(slider));
    rects.remove("SunBrightness");
    assert_eq!(brightness_target(&rects), Some(tab));
    rects.clear();
    assert_eq!(brightness_target(&rects), None);
}

#[test]
fn guide_restores_a_closed_lighting_tab_without_switching_the_active_tab() {
    use crate::ui::Panel;
    let state = GlobalUiState {
        current_mode: SimulationMode::Gpu,
        reset_intro_seen: [true; 2],
        ..Default::default()
    };
    let mut dock = crate::ui::DockManager::new();
    *dock.current_tree_mut() =
        egui_dock::DockState::new(vec![Panel::Viewport, Panel::WorldSettings]);
    prepare_quick_start(&state, &mut dock);
    let (surface, node, _) = dock.current_tree().find_tab(&Panel::LightSettings).unwrap();
    let leaf = dock.current_tree()[surface][node].get_leaf().unwrap();
    assert_eq!(leaf.tabs[leaf.active.0], Panel::Viewport);
}

#[test]
fn quick_start_allows_clicking_the_rail_while_the_guide_is_open() {
    let ctx = Context::default();
    let mut state = GlobalUiState {
        current_mode: SimulationMode::Gpu,
        reset_intro_seen: [true; 2],
        ..Default::default()
    };
    click(&ctx, &mut state, "⟲");
    assert!(state.show_settings_reset_dialog);
    assert!(!state.main_scene_quick_start_seen);
}

#[test]
#[allow(deprecated)]
fn reset_icon_shares_the_rail_center_and_stays_at_the_bottom() {
    let ctx = Context::default();
    for mode in [SimulationMode::Preview, SimulationMode::Gpu] {
        let mut state = GlobalUiState {
            current_mode: mode,
            ..Default::default()
        };
        let mut editor = crate::ui::GenomeEditorState::factory_defaults();
        let mut dock = crate::ui::DockManager::new();
        let mut bottom = 0.0;
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 700.0),
                )),
                ..Default::default()
            },
            |ctx| {
                #[allow(deprecated)]
                egui::Panel::left("alignment_test_rail")
                    .exact_width(40.0)
                    .frame(
                        egui::Frame::new()
                            .inner_margin(egui::Margin {
                                left: 4,
                                right: 4,
                                top: 8,
                                bottom: 8,
                            })
                            .stroke(egui::Stroke::new(1.0, Color32::GRAY)),
                    )
                    .show(ctx, |ui| {
                        bottom = ui.max_rect().bottom();
                        crate::ui::ui_system::render_side_rail(
                            ui,
                            &mut state,
                            &mut editor,
                            &mut dock,
                            None,
                        );
                    });
            },
        );
        let reset = editor.panel_rects["ResetSettings"];
        assert_eq!(reset.size(), egui::vec2(32.0, 32.0));
        assert!((reset.bottom() - (bottom - 4.0)).abs() < 0.1);
        let top_icon = if mode == SimulationMode::Gpu {
            "⚙"
        } else {
            "⤢"
        };
        let top = output
            .shapes
            .iter()
            .find_map(|shape| match &shape.shape {
                egui::epaint::Shape::Text(text) if text.galley.text() == top_icon => {
                    Some(text.pos + text.galley.size() * 0.5)
                }
                _ => None,
            })
            .expect("top rail icon must be rendered");
        assert!(
            (top.x - reset.center().x).abs() < 0.1,
            "reset center {} must match top icon {}",
            reset.center().x,
            top.x
        );
    }
}

#[test]
fn reset_restores_active_scene_preferences_and_preserves_other_scene_and_shared_settings() {
    let initial: GlobalUiState = ron::from_str(include_str!("../../default_ui_state.ron")).unwrap();
    for mode in [SimulationMode::Preview, SimulationMode::Gpu] {
        let other = if mode == SimulationMode::Preview {
            SimulationMode::Gpu
        } else {
            SimulationMode::Preview
        };
        let mut state = GlobalUiState {
            current_mode: mode,
            ..Default::default()
        };
        state.reset_intro_seen = [true; 2];
        state.main_scene_quick_start_seen = true;
        state.tutorial.ever_shown = true;
        state.selected_theme = UiTheme::NeonToxic;
        state.ui_scale = 4.0;
        state.music_volume = 0.99;
        state.desktop_render_fps = 30;
        state.camera_scroll_sensitivity = 2.0;
        state.vr_screen.distance = 8.0;
        state.world_settings.gravity = -99.0;
        state.world_settings.world_radius = 999.0;
        state.world_settings.cell_capacity = 123456;
        state.world_diameter = 1998.0;
        state.fluid_settings.ice.alpha = 0.22;
        for config in state.scene_configs.values_mut() {
            config.locked_windows.insert("custom".to_string());
        }
        let before = state.clone();
        state.restore_scene_settings();
        assert_eq!(state.current_mode, mode);
        assert_eq!(state.scene_configs[&mode], initial.scene_configs[&mode]);
        assert_eq!(state.scene_configs[&other], before.scene_configs[&other]);
        assert_eq!(state.selected_theme, before.selected_theme);
        assert_eq!(state.ui_scale, before.ui_scale);
        assert_eq!(state.music_volume, before.music_volume);
        assert_eq!(state.desktop_render_fps, before.desktop_render_fps);
        assert_eq!(
            state.camera_scroll_sensitivity,
            before.camera_scroll_sensitivity
        );
        assert_eq!(state.vr_screen, before.vr_screen);
        assert_eq!(state.tutorial, before.tutorial);
        assert_eq!(state.reset_intro_seen, before.reset_intro_seen);
        assert_eq!(
            state.main_scene_quick_start_seen,
            before.main_scene_quick_start_seen
        );
        assert_eq!(state.world_diameter, before.world_diameter);
        if mode == SimulationMode::Gpu {
            assert_eq!(state.world_settings, initial.world_settings);
            assert_eq!(state.fluid_settings, initial.fluid_settings);
        } else {
            assert_eq!(state.world_settings, before.world_settings);
            assert_eq!(state.fluid_settings, before.fluid_settings);
        }
    }
}

#[test]
fn reset_preserves_live_editor_state_and_only_restores_the_active_scenes_controls() {
    use crate::ui::panel_context::GenomeEditorState;
    for mode in [SimulationMode::Preview, SimulationMode::Gpu] {
        let mut state = GenomeEditorState::factory_defaults();
        let defaults = GenomeEditorState::factory_defaults();
        state.qball_snapping = false;
        state.enable_snapping = false;
        state.selected_mode_index = 4;
        state.selected_mode_indices = vec![2, 4];
        state.time_value = 23.0;
        state.resim_display_time = 23.0;
        state.max_preview_duration = 120.0;
        state.cave_seed = 123456;
        state.cave_density = 0.123;
        state.geothermal_count = 49;
        state.fluid_water_count = 2468;
        state.fluid_water_mass = 1234.0;
        state.fluid_continuous_spawn = true;
        state.fluid_static_water_world = true;
        state.moss_growth_rate = 999.0;
        state.fluid_alpha = 0.01;
        state.sun_color = [0.0; 3];
        state.nutrient_density = 0.0;
        state.cell_outline_width = 0.15;
        state.reset_scene_settings(mode);
        assert_eq!(state.selected_mode_index, 4);
        assert_eq!(state.selected_mode_indices, vec![2, 4]);
        assert_eq!(state.time_value, 23.0);
        assert_eq!(state.resim_display_time, 23.0);
        assert_eq!(state.max_preview_duration, 120.0);
        assert_eq!(state.fluid_water_count, 2468);
        assert_eq!(state.fluid_water_mass, 1234.0);
        assert_eq!(state.cell_outline_width, 0.15);
        if mode == SimulationMode::Gpu {
            assert_eq!(state.cave_seed, defaults.cave_seed);
            assert_eq!(state.cave_density, defaults.cave_density);
            assert_eq!(state.geothermal_count, defaults.geothermal_count);
            assert_eq!(
                state.fluid_continuous_spawn,
                defaults.fluid_continuous_spawn
            );
            assert_eq!(
                state.fluid_static_water_world,
                defaults.fluid_static_water_world
            );
            assert!(!state.qball_snapping && !state.enable_snapping);
            assert_eq!(state.moss_growth_rate, defaults.moss_growth_rate);
            assert_eq!(state.fluid_alpha, defaults.fluid_alpha);
            assert_eq!(state.sun_color, defaults.sun_color);
            assert_eq!(state.nutrient_density, defaults.nutrient_density);
            assert!(state.light_params_dirty && state.cave_params_dirty);
        } else {
            assert_eq!(state.cave_seed, 123456);
            assert_eq!(state.cave_density, 0.123);
            assert_eq!(state.geothermal_count, 49);
            assert!(state.fluid_continuous_spawn && state.fluid_static_water_world);
            assert!(state.qball_snapping && state.enable_snapping);
            assert_eq!(state.moss_growth_rate, 999.0);
            assert_eq!(state.fluid_alpha, 0.01);
            assert_eq!(state.sun_color, [0.0; 3]);
            assert_eq!(state.nutrient_density, 0.0);
        }
    }
}

#[test]
fn main_scene_reset_uses_the_captured_personal_preset_including_seed_and_lighting() {
    let mut editor = crate::ui::GenomeEditorState::factory_defaults();
    assert_eq!(editor.cave_seed, 0);
    assert_eq!(editor.cave_density, 0.99); // UI density is 1 - stored density.
    assert!(!editor.cave_flat_ground_enabled);
    assert!(!editor.fluid_continuous_spawn);
    assert!(!editor.fluid_static_water_world);
    assert_eq!(editor.sun_intensity, 2.0);
    assert_eq!(editor.fluid_alpha, 0.25);
    let visuals = crate::cell::types::CellTypeVisualsStore::bundled_defaults();
    assert_eq!(
        serde_json::to_value(&editor.cell_type_visuals).unwrap(),
        serde_json::to_value(&visuals.visuals).unwrap()
    );
    editor.cave_seed = 9;
    editor.cave_density = 0.2;
    editor.cave_flat_ground_enabled = true;
    editor.fluid_continuous_spawn = true;
    editor.fluid_static_water_world = true;
    editor.sun_intensity = 0.0;
    editor.fluid_alpha = 1.0;
    editor.reset_scene_settings(SimulationMode::Gpu);
    assert_eq!(editor.cave_seed, 0);
    assert_eq!(editor.cave_density, 0.99);
    assert!(!editor.cave_flat_ground_enabled);
    assert!(!editor.fluid_continuous_spawn);
    assert!(!editor.fluid_static_water_world);
    assert_eq!(editor.sun_intensity, 2.0);
    assert_eq!(editor.fluid_alpha, 0.25);

    let defaults: GlobalUiState =
        ron::from_str(include_str!("../../default_ui_state.ron")).unwrap();
    let snapshot: GlobalUiState = ron::from_str(include_str!(
        "../../assets/defaults/main-scene/ui_state.ron"
    ))
    .unwrap();
    assert_eq!(defaults, snapshot);
    assert!(!defaults.tutorial.ever_shown);
    assert_eq!(defaults.reset_intro_seen, [false; 2]);
    assert!(!defaults.main_scene_quick_start_seen);
}

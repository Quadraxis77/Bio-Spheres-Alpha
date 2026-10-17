//! The shared reset safety net and its once-per-scene introduction.

use super::types::GlobalUiState;
use egui::{Color32, Context, Id, Rect, RichText, Ui};

const ACCENT: Color32 = Color32::from_rgb(255, 196, 83);

pub fn rail_button(ui: &mut Ui, state: &mut GlobalUiState) -> Rect {
    // Match the exact allocation and centered painting used by the other rail icons.
    let (rect, response) = ui.allocate_exact_size(egui::vec2(32.0, 32.0), egui::Sense::click());
    let response =
        response.on_hover_text("Restore this scene's settings and layout to initial defaults");
    ui.painter()
        .rect_filled(rect, 3.0, Color32::from_rgb(57, 43, 21));
    ui.painter().rect_stroke(
        rect,
        3.0,
        egui::Stroke::new(1.5, ACCENT),
        egui::StrokeKind::Inside,
    );
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "⟲",
        egui::FontId::proportional(15.0),
        ACCENT,
    );
    if response.clicked() {
        state.show_settings_reset_dialog = true;
    }
    rect
}

/// Ensure the guide can point at a Lighting tab without selecting it for the player.
pub fn prepare_quick_start(state: &GlobalUiState, dock: &mut super::dock::DockManager) {
    use super::panel::Panel;
    if state.current_mode != super::types::SimulationMode::Gpu
        || state.main_scene_quick_start_seen
        || !state.reset_intro_seen[state.reset_intro_index()]
        || dock
            .current_tree()
            .find_tab(&Panel::LightSettings)
            .is_some()
    {
        return;
    }
    let tree = dock.current_tree_mut();
    let anchor = [Panel::WorldSettings, Panel::FluidSettings, Panel::Viewport]
        .iter()
        .find_map(|panel| tree.find_tab(panel));
    if let Some(location) = anchor {
        let active = tree[location.0][location.1].get_leaf().unwrap().active;
        let focused = tree.focused_leaf();
        tree.set_focused_node_and_surface((location.0, location.1));
        tree.push_to_focused_leaf(Panel::LightSettings);
        tree.set_active_tab((location.0, location.1, active));
        if let Some(focused) = focused {
            tree.set_focused_node_and_surface(focused);
        }
    } else {
        tree.add_window(vec![Panel::LightSettings]);
    }
    dock.mark_dirty();
}

pub fn brightness_target(rects: &std::collections::HashMap<String, Rect>) -> Option<Rect> {
    rects
        .get("SunBrightness")
        .or_else(|| rects.get("LightingTab"))
        .copied()
}

/// Returns true while onboarding has priority over the genome tutorial.
pub fn render_dialogs(
    ctx: &Context,
    state: &mut GlobalUiState,
    reset_rect: Option<Rect>,
    brightness_rect: Option<Rect>,
) -> bool {
    if state.hide_ui {
        return false;
    }
    if let Some(error) = state.settings_reset_error.clone() {
        egui::Modal::new(Id::new("settings_reset_save_error")).show(ctx, |ui| {
            ui.set_width(360.0_f32.min(ctx.content_rect().width() - 40.0).max(160.0));
            ui.heading("Defaults restored, but saving failed");
            ui.label("The settings are reset for this session. Some changes may return after restarting because these files could not be saved:");
            ui.label(error);
            if ui.button("OK").clicked() {
                state.settings_reset_error = None;
            }
        });
        return true;
    }
    let intro_index = state.reset_intro_index();
    let intro = !state.reset_intro_seen[intro_index];
    let quick_start = state.current_mode == super::types::SimulationMode::Gpu
        && !state.main_scene_quick_start_seen;
    if !intro && !quick_start && !state.show_settings_reset_dialog {
        return false;
    }

    if state.show_settings_reset_dialog {
        egui::Modal::new(Id::new("reset_scene_settings_confirmation")).show(ctx, |ui| {
            ui.set_width(360.0_f32.min(ctx.content_rect().width() - 40.0).max(160.0));
            ui.heading(format!(
                "Reset {} settings?",
                state.current_mode.display_name()
            ));
            ui.label("Restore settings and layout for this scene only.");
            ui.add_space(8.0);
            ui.label(
                "Your live world, cells, water, simulation time, and current genome are kept.",
            );
            ui.label("The other scene and shared preferences stay as they are.");
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    state.show_settings_reset_dialog = false;
                }
                if ui
                    .button(RichText::new("Reset scene settings").color(ACCENT))
                    .clicked()
                {
                    state.show_settings_reset_dialog = false;
                    state.reset_scene_settings_requested = true;
                }
            });
        });
    } else if intro {
        egui::Modal::new(Id::new(("reset_settings_intro", intro_index))).show(ctx, |ui| {
            ui.set_width(360.0_f32.min(ctx.content_rect().width() - 40.0).max(160.0));
            ui.heading("Experiment with confidence");
            ui.label(format!("Welcome to {}.", state.current_mode.display_name()));
            ui.add_space(8.0);
            ui.label("The highlighted ⟲ Reset button at the bottom of the left rail restores settings and layout for this scene only.");
            ui.label("Your live world and current genome are kept. The other scene and shared preferences stay as they are.");
            ui.add_space(12.0);
            if ui.button("Got it").clicked() {
                state.reset_intro_seen[intro_index] = true;
            }
        });
    } else {
        let screen = ctx.content_rect();
        let width = 380.0_f32.min(screen.width() - 80.0).max(160.0);
        let x = if brightness_rect.is_some_and(|rect| rect.center().x < screen.center().x) {
            screen.right() - width - 24.0
        } else {
            screen.left() + 64.0
        };
        let guide = egui::Window::new("Quick start: bring your world to life")
            .id(Id::new("main_scene_quick_start"))
            .fixed_pos(egui::pos2(x, screen.top() + 64.0))
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.set_width(width);
                ui.add_space(8.0);
                ui.label(RichText::new("1. Add a little water").strong());
                ui.label("Click 🌊 Water Fill on the left rail. Let a small amount flow in, then click it again to stop filling. Start with a shallow pool.");
                ui.add_space(8.0);
                ui.label(RichText::new("2. Drop in your cells").strong());
                ui.label("Build or load an organism in the Genome Editor, then return to Live Simulation. Hold Alt, select Add from the tool wheel, and release Alt. Left-click in the world to place cells from your current Genome Editor genome.");
                ui.add_space(8.0);
                ui.label(RichText::new("3. Set the climate with Brightness").strong());
                ui.label("Brightness ranges from 0–5: 0 is dark and cold, around 3 is temperate, and 5 brings extreme heat. It also affects photocyte energy.");
                ui.add_space(12.0);
                if ui.button("Let's play").clicked() {
                    state.main_scene_quick_start_seen = true;
                }
            });
        if let (Some(guide), Some(target)) = (guide, brightness_rect) {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Tooltip,
                Id::new("quick_start_brightness_pointer"),
            ));
            painter.rect_stroke(
                target.expand(3.0),
                4.0,
                egui::Stroke::new(2.0, super::ui_system::palette().accent_primary),
                egui::StrokeKind::Outside,
            );
            super::tutorial::draw_schematic_pointer(
                &painter,
                guide.response.rect,
                target.clamp(guide.response.rect.center()),
                ctx,
                true,
            );
            ctx.request_repaint();
        }
    }

    if let Some(rect) = reset_rect.filter(|_| intro || state.show_settings_reset_dialog) {
        ctx.layer_painter(egui::LayerId::new(
            egui::Order::Tooltip,
            Id::new("reset_button_highlight"),
        ))
        .rect_stroke(
            rect.expand(4.0),
            5.0,
            egui::Stroke::new(3.0, ACCENT),
            egui::StrokeKind::Outside,
        );
    }
    true
}

#[cfg(test)]
#[path = "reset_settings_tests.rs"]
mod tests;

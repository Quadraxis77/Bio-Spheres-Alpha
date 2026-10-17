//! Shared slider layouts for docked settings and authoring panels.

use egui::{Slider, Ui};

#[cfg(test)]
#[path = "scene_settings_tests.rs"]
mod tests;

/// Labels get their own line; every value field has the same width and right edge.
/// Reserve the value field before sizing the track, including in narrow dock panels.
pub(super) fn setting(ui: &mut Ui, label: &str, slider: Slider<'_>) -> egui::Response {
    ui.label(label);
    slider_control(ui, slider)
}

pub(super) fn slider_control(ui: &mut Ui, slider: Slider<'_>) -> egui::Response {
    // Opening a panel must preserve saved precision and animated values.
    slider_row(ui, |ui| {
        ui.add(slider.clamping(egui::SliderClamping::Edits))
    })
}

fn reserve_value_width(ui: &mut Ui) {
    let font = ui.style().drag_value_text_style.resolve(ui.style());
    let number_width = ui.fonts_mut(|fonts| {
        fonts
            .layout_no_wrap("0000.00 s".into(), font, egui::Color32::WHITE)
            .size()
            .x
    }) + 2.0 * ui.spacing().button_padding.x;
    let value_width = number_width
        .max(72.0)
        .min((ui.available_width() - ui.spacing().item_spacing.x).max(0.0));
    ui.spacing_mut().interact_size.x = value_width;
    ui.spacing_mut().slider_width =
        (ui.available_width() - value_width - ui.spacing().item_spacing.x).max(0.0);
}

fn slider_row(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> egui::Response) -> egui::Response {
    ui.scope(|ui| {
        reserve_value_width(ui);
        add(ui)
    })
    .inner
}

/// Special authoring rows retain their custom parsers, sentinel values and tooltips.
pub(super) fn paired_row<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> egui::InnerResponse<R> {
    ui.scope(|ui| {
        reserve_value_width(ui);
        ui.horizontal(add)
    })
    .inner
}

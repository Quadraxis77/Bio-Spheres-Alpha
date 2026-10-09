//! Readability of the ordinary application panels projected into VR.
//! The wrist console uses its own context and rendering path.

use crate::ui::ui_system::ActivePalette;
use egui::{epaint::PathStroke, Color32, Shape, Stroke};

/// Applied to the main context before beginning a frame. Rasterized outlines
/// retain the original font metrics and work for custom labels and fallbacks.
pub fn set_text_weight(ctx: &egui::Context, presenting: bool) {
    ctx.global_style_mut(|style| {
        style.visuals.text_options.glyph_stroke_width = if presenting { 1.2 } else { 0.0 };
    });
}

/// Adjust the headset's copy of the paint output, once before drawing both eyes.
/// This also covers graph edges, separators, icons, and other custom painters
/// that do not use egui's widget style. Images and scene previews stay intact.
pub fn strengthen_shapes(shapes: &mut [egui::epaint::ClippedShape], palette: ActivePalette) {
    for clipped in shapes {
        strengthen_shape(&mut clipped.shape, palette);
    }
}

fn line_width(width: f32) -> f32 {
    (width * 1.6).max(1.5).min(width + 1.5)
}

fn border_color(color: Color32, palette: ActivePalette) -> Color32 {
    // Keep accent/status colors and opacity. Lift subdued theme borders toward
    // the theme's foreground, which works for both dark and light palettes.
    let rgba = color.to_srgba_unmultiplied();
    let subdued = [
        palette.border_subtle,
        palette.border_normal,
        palette.topbar_border,
    ];
    if !subdued
        .iter()
        .any(|base| base.to_srgba_unmultiplied()[..3] == rgba[..3])
    {
        return color;
    }
    let mix = |a: u8, b: u8| (a as f32 * 0.65 + b as f32 * 0.35).round() as u8;
    Color32::from_rgba_unmultiplied(
        mix(rgba[0], palette.text_primary.r()),
        mix(rgba[1], palette.text_primary.g()),
        mix(rgba[2], palette.text_primary.b()),
        rgba[3],
    )
}

fn stroke(stroke: &mut Stroke, palette: ActivePalette) {
    if !stroke.is_empty() {
        stroke.width = line_width(stroke.width);
        stroke.color = border_color(stroke.color, palette);
    }
}

fn path_stroke(stroke: &mut PathStroke, palette: ActivePalette) {
    if !stroke.is_empty() {
        stroke.width = line_width(stroke.width);
        if let egui::epaint::ColorMode::Solid(color) = &mut stroke.color {
            *color = border_color(*color, palette);
        }
    }
}

fn text_color(color: Color32, palette: ActivePalette) -> Color32 {
    let rgba = color.to_srgba_unmultiplied();
    let is_subdued = [palette.text_dim, palette.text_secondary]
        .iter()
        .any(|base| base.to_srgba_unmultiplied()[..3] == rgba[..3]);
    if !is_subdued || rgba[3] == 0 {
        return color;
    }
    // Theme-aware contrast for secondary labels and custom-painted readouts.
    // Keep the shape's separate opacity factor so animated fades still work.
    let mix = |a: u8, b: u8| (a as f32 * 0.2 + b as f32 * 0.8).round() as u8;
    Color32::from_rgba_unmultiplied(
        mix(rgba[0], palette.text_primary.r()),
        mix(rgba[1], palette.text_primary.g()),
        mix(rgba[2], palette.text_primary.b()),
        rgba[3].max(230),
    )
}

fn strengthen_shape(shape: &mut Shape, palette: ActivePalette) {
    match shape {
        Shape::Vec(shapes) => {
            for shape in shapes {
                strengthen_shape(shape, palette);
            }
        }
        Shape::Circle(shape) => stroke(&mut shape.stroke, palette),
        Shape::Ellipse(shape) => stroke(&mut shape.stroke, palette),
        Shape::LineSegment { stroke: line, .. } => stroke(line, palette),
        Shape::Path(shape) => path_stroke(&mut shape.stroke, palette),
        Shape::QuadraticBezier(shape) => path_stroke(&mut shape.stroke, palette),
        Shape::CubicBezier(shape) => path_stroke(&mut shape.stroke, palette),
        Shape::Rect(shape) => {
            stroke(&mut shape.stroke, palette);
            // Some separators and carets are filled hairline rectangles.
            if shape.brush.is_none()
                && shape.blur_width == 0.0
                && shape.fill != Color32::TRANSPARENT
            {
                let size = shape.rect.size();
                if size.min_elem() > 0.0 && size.min_elem() <= 1.5 && size.max_elem() >= 8.0 {
                    let extra = (line_width(size.min_elem()) - size.min_elem()) * 0.5;
                    shape.rect = shape.rect.expand2(if size.x < size.y {
                        egui::vec2(extra, 0.0)
                    } else {
                        egui::vec2(0.0, extra)
                    });
                    shape.fill = border_color(shape.fill, palette);
                }
            }
        }
        Shape::Text(shape) => {
            stroke(&mut shape.underline, palette);
            shape.fallback_color = text_color(shape.fallback_color, palette);
            if let Some(color) = shape.override_text_color {
                shape.override_text_color = Some(text_color(color, palette));
            } else if let Some(first) = shape.galley.job.sections.first() {
                // Only override uniformly colored text. Preserve multicolored
                // warnings, syntax, and status spans within the same galley.
                let color = first.format.color;
                let readable = text_color(color, palette);
                if readable != color
                    && shape
                        .galley
                        .job
                        .sections
                        .iter()
                        .all(|s| s.format.color == color)
                {
                    shape.override_text_color = Some(readable);
                }
            }
        }
        Shape::Noop | Shape::Mesh(_) | Shape::Callback(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdued_text_gains_contrast_in_dark_and_light_themes() {
        let dark = ActivePalette::default();
        for palette in [
            dark,
            ActivePalette {
                text_dim: Color32::from_gray(180),
                text_secondary: Color32::from_gray(140),
                text_primary: Color32::from_gray(25),
                ..dark
            },
        ] {
            for color in [palette.text_dim, palette.text_secondary] {
                let readable = text_color(color, palette);
                assert!(
                    readable.r().abs_diff(palette.text_primary.r())
                        < color.r().abs_diff(palette.text_primary.r())
                );
            }
            assert_eq!(text_color(palette.status_err, palette), palette.status_err);
        }
    }

    #[test]
    fn heavier_glyphs_preserve_layout_at_each_dpi_and_reset_after_vr() {
        for dpi in [1.0, 2.0, 3.0] {
            let ctx = egui::Context::default();
            let wrist = egui::Context::default();
            let measure = |presenting| {
                set_text_weight(&ctx, presenting);
                let mut input = egui::RawInput::default();
                input
                    .viewports
                    .get_mut(&input.viewport_id)
                    .unwrap()
                    .native_pixels_per_point = Some(dpi);
                ctx.begin_pass(input);
                let sizes = ctx.fonts_mut(|fonts| {
                    [
                        egui::FontId::proportional(12.0),
                        egui::FontId::monospace(11.0),
                    ]
                    .map(|font| {
                        fonts
                            .layout_no_wrap(
                                "Air 23.5 °C | Water 18.2 °C → Settings".into(),
                                font,
                                Color32::WHITE,
                            )
                            .size()
                    })
                });
                let coverage: u64 = ctx.fonts(|fonts| {
                    fonts
                        .image()
                        .pixels
                        .iter()
                        .map(|pixel| u64::from(pixel.a()))
                        .sum()
                });
                let _ = ctx.end_pass();
                (sizes, coverage)
            };
            let desktop = measure(false);
            let vr = measure(true);
            assert_eq!(
                desktop.0, vr.0,
                "weight must not shift labels or hit targets"
            );
            assert!(
                vr.1 > desktop.1,
                "glyphs must gain visible coverage at {dpi}x"
            );
            assert_eq!(
                desktop,
                measure(false),
                "leaving VR restores the original raster"
            );
            assert_eq!(
                wrist.global_style().visuals.text_options.glyph_stroke_width,
                0.0
            );
        }
    }

    #[test]
    fn panel_edges_gain_contrast_without_changing_images_or_hidden_strokes() {
        let dark = ActivePalette::default();
        for palette in [
            dark,
            ActivePalette {
                border_subtle: Color32::from_gray(210),
                text_primary: Color32::from_gray(20),
                ..dark
            },
        ] {
            let image = Shape::image(
                egui::TextureId::User(42),
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(100.0, 60.0)),
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
            let mut shape = Shape::Vec(vec![
                Shape::line_segment(
                    [egui::Pos2::ZERO, egui::pos2(100.0, 0.0)],
                    Stroke::new(0.5, palette.border_subtle),
                ),
                Shape::circle_filled(egui::pos2(20.0, 20.0), 5.0, palette.accent_primary),
                image.clone(),
            ]);
            strengthen_shape(&mut shape, palette);
            let Shape::Vec(shapes) = shape else { panic!() };
            let Shape::LineSegment { stroke, .. } = &shapes[0] else {
                panic!()
            };
            assert!(stroke.width >= 1.5);
            let distance = |color: Color32| color.r().abs_diff(palette.text_primary.r());
            assert!(distance(stroke.color) < distance(palette.border_subtle));
            let Shape::Circle(circle) = &shapes[1] else {
                panic!()
            };
            assert_eq!(circle.stroke, Stroke::NONE);
            assert_eq!(circle.fill, palette.accent_primary);
            assert_eq!(shapes[2], image);
        }
    }
}

#[cfg(test)]
#[path = "ui_readability_visual_tests.rs"]
mod visual_tests;

//! Vector styling for the Bio-Spheres wrist console, using the active UI palette.
use crate::ui::ui_system::ActivePalette;
use egui::{Color32, Painter, Pos2, Stroke, Vec2};

pub fn alpha(color: Color32, opacity: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), opacity)
}
fn mix(base: Color32, tint: Color32, weight: f32, opacity: u8) -> Color32 {
    let blend = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * weight) as u8;
    Color32::from_rgba_unmultiplied(
        blend(base.r(), tint.r()),
        blend(base.g(), tint.g()),
        blend(base.b(), tint.b()),
        opacity,
    )
}
pub fn chassis(p: &Painter, center: Pos2, palette: ActivePalette) {
    p.circle_filled(center, 279.0, alpha(palette.bg_panel, 100));
    for (radius, color, width) in [
        (278.0, alpha(palette.border_bright, 220), 2.0),
        (274.0, alpha(palette.bg_darkest, 245), 4.0),
        (97.0, alpha(palette.bg_darkest, 245), 5.0),
        (92.0, alpha(palette.border_bright, 180), 1.0),
    ] {
        p.circle_stroke(center, radius, Stroke::new(width, color));
    }
    p.circle_filled(center, 90.0, alpha(palette.bg_panel, 235));
}
#[allow(clippy::too_many_arguments)]
pub fn sector(
    p: &Painter,
    center: Pos2,
    inner: f32,
    outer: f32,
    start: f32,
    span: f32,
    tint: Color32,
    enabled: bool,
    active: bool,
    hovered: bool,
    press: f32,
    group_edges: (bool, bool),
    palette: ActivePalette,
) -> Pos2 {
    // Only category boundaries have broad engraved channels. Buttons within a
    // category share a surface separated by a thin seam.
    let gap = |deep: bool| (if deep { 0.035f32 } else { 0.006f32 }).min(span * 0.08);
    let a0 = start + gap(group_edges.0);
    let sweep = span - gap(group_edges.0) - gap(group_edges.1);
    let shift = Vec2::splat(2.5 * press);
    let point = |a: f32, r: f32| center + Vec2::new(a.cos(), a.sin()) * r;
    let arc = |r: f32, offset: Vec2| {
        (0..=32)
            .map(|i| point(a0 + sweep * i as f32 / 32.0, r) + offset)
            .collect::<Vec<_>>()
    };
    let mut well = egui::Mesh::default();
    for i in 0..=32 {
        let a = a0 + sweep * i as f32 / 32.0;
        for r in [inner + 1.0, outer - 1.0] {
            well.colored_vertex(point(a, r), alpha(palette.bg_darkest, 100));
        }
        if i > 0 {
            let b = i * 2;
            well.add_triangle(b - 2, b - 1, b);
            well.add_triangle(b, b - 1, b + 1);
        }
    }
    p.add(egui::Shape::mesh(well));
    let base = if !enabled {
        palette.bg_panel
    } else if press > 0.0 {
        palette.bg_active
    } else if hovered {
        palette.bg_hover
    } else if active {
        palette.bg_selected
    } else {
        palette.bg_widget
    };
    let mut face = egui::Mesh::default();
    for i in 0..=32 {
        let a = a0 + sweep * i as f32 / 32.0;
        let lit = (Vec2::new(a.cos(), a.sin()).dot(Vec2::new(-0.6, -0.8)) + 1.0) * 0.5;
        for (r, weight) in [
            (inner + 6.0, 0.02),
            (outer - 10.0, 0.07),
            (outer - 6.0, 0.12),
        ] {
            face.colored_vertex(
                point(a, r) + shift,
                mix(
                    base,
                    tint,
                    if enabled { weight + lit * 0.035 } else { 0.0 },
                    195,
                ),
            );
        }
        if i > 0 {
            let b = i * 3;
            for level in 0..2 {
                face.add_triangle(b - 3 + level, b - 2 + level, b + level);
                face.add_triangle(b + level, b - 2 + level, b + level + 1);
            }
        }
    }
    p.add(egui::Shape::mesh(face));
    let upper = alpha(
        if press > 0.0 {
            palette.bg_darkest
        } else {
            palette.border_bright
        },
        230,
    );
    let lower = alpha(
        if press > 0.0 {
            palette.border_bright
        } else {
            palette.bg_darkest
        },
        245,
    );
    p.add(egui::Shape::line(
        arc(outer - 6.0, shift),
        Stroke::new(2.0, upper),
    ));
    p.add(egui::Shape::line(
        arc(inner + 6.0, shift),
        Stroke::new(2.0, lower),
    ));
    for (a, deep) in [(a0, group_edges.0), (a0 + sweep, group_edges.1)] {
        let edge = [point(a, inner + 6.0) + shift, point(a, outer - 6.0) + shift];
        if deep {
            p.line_segment(edge, Stroke::new(4.0, alpha(palette.bg_darkest, 245)));
            p.line_segment(edge, Stroke::new(1.0, alpha(palette.border_bright, 180)));
        } else {
            p.line_segment(edge, Stroke::new(0.7, alpha(palette.border_subtle, 120)));
        }
    }
    let midpoint = a0 + sweep * 0.5;
    let lamp = point(midpoint, outer - 22.0) + shift;
    if enabled && (press > 0.0 || hovered || active) {
        p.circle_filled(lamp, 6.0, alpha(tint, 40));
        p.circle_filled(lamp, 2.5, tint);
    } else {
        p.circle_stroke(lamp, 2.0, Stroke::new(1.0, alpha(tint, 150)));
    }
    point(midpoint, (inner + outer) * 0.5) + shift
}
pub fn footer(
    p: &Painter,
    rect: egui::Rect,
    label: &str,
    hovered: bool,
    pressed: bool,
    palette: ActivePalette,
) {
    p.rect_filled(rect.expand(2.0), 8.0, alpha(palette.bg_darkest, 245));
    let face = rect.translate(if pressed {
        Vec2::splat(2.0)
    } else {
        Vec2::ZERO
    });
    let base = if pressed {
        palette.bg_active
    } else if hovered {
        palette.bg_hover
    } else {
        palette.bg_widget
    };
    p.rect_filled(face, 7.0, alpha(base, 235));
    p.rect_stroke(
        face,
        7.0,
        Stroke::new(
            1.0,
            alpha(
                if pressed {
                    palette.border_normal
                } else {
                    palette.border_bright
                },
                230,
            ),
        ),
        egui::StrokeKind::Inside,
    );
    p.text(
        face.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(13.0),
        palette.text_primary,
    );
}

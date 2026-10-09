//! Readable specimen pages on the left-hand console.
use super::controls::Choice;
use crate::{simulation::gpu_physics::InspectedCellData, ui::ui_system::ActivePalette};
#[derive(Default)]
pub struct View {
    pub data: Option<InspectedCellData>,
    pub dead: bool,
    pub genome_name: String,
    pub modes: usize,
    pub loadable: bool,
}
pub fn buttons() -> Vec<(egui::Rect, Choice, &'static str)> {
    let mut result = vec![];
    for (i, (choice, label)) in [
        (Choice::CellOverview, "Identity"),
        (Choice::CellBiology, "Biology"),
        (Choice::CellPhysics, "Physics"),
        (Choice::CellSignals, "Signals"),
    ]
    .into_iter()
    .enumerate()
    {
        result.push((
            egui::Rect::from_center_size(
                egui::pos2(198.0 + i as f32 * 108.0, 185.0),
                egui::vec2(102.0, 32.0),
            ),
            choice,
            label,
        ));
    }
    result.extend([
        (
            egui::Rect::from_center_size(egui::pos2(360.0, 550.0), egui::vec2(260.0, 38.0)),
            Choice::LoadInspectedGenome,
            "Load genome into Preview",
        ),
        (
            egui::Rect::from_center_size(egui::pos2(292.0, 600.0), egui::vec2(124.0, 30.0)),
            Choice::Back,
            "Back / B",
        ),
        (
            egui::Rect::from_center_size(egui::pos2(428.0, 600.0), egui::vec2(124.0, 30.0)),
            Choice::Close,
            "Close / X",
        ),
    ]);
    result
}
pub fn hit(pixel: glam::Vec2) -> Option<Choice> {
    buttons()
        .into_iter()
        .find(|(r, _, _)| r.contains(egui::pos2(pixel.x, pixel.y)))
        .map(|(_, c, _)| c)
}
pub fn draw(
    p: &egui::Painter,
    view: &View,
    tab: Choice,
    hover: Option<Choice>,
    palette: ActivePalette,
) {
    let text = |y: f32, label: String, size: f32, color: egui::Color32| {
        p.text(
            egui::pos2(360.0, y),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(size),
            color,
        );
    };
    text(99.0, "CELL INSPECTOR".into(), 18.0, palette.accent_primary);
    if let Some(d) = view.data {
        let name = crate::cell::types::CellType::names()
            .get(d.cell_type as usize)
            .copied()
            .unwrap_or("Unknown");
        text(
            130.0,
            format!("{}  #{}", name, d.cell_id),
            25.0,
            palette.text_primary,
        );
        text(
            155.0,
            if view.dead {
                "CELL DIED — GENOME RETAINED"
            } else {
                "LIVE READINGS"
            }
            .into(),
            14.0,
            if view.dead {
                palette.status_err
            } else {
                palette.status_ok
            },
        );
        let rows: Vec<(&str, String)> = match tab {
            Choice::CellBiology => vec![
                ("Age", format!("{:.1} s", d.age)),
                ("Nutrients", format!("{:.2}", d.nutrients)),
                (
                    "Net nutrient gain",
                    format!("{:+.2} / s", d.nutrient_gain_rate),
                ),
                ("Division threshold", format!("{:.2}", d.nutrient_threshold)),
                ("Divisions", format!("{} / {}", d.split_count, d.max_splits)),
                ("Division interval", format!("{:.2} s", d.split_interval)),
                ("Reserve", format!("{}", d.reserve)),
                ("Water", format!("{:.2}", d.cell_water)),
                (
                    "Temperature",
                    format!("{:.1} °C", d.cell_cached_temperature / 255.0 * 200.0 - 50.0),
                ),
                (
                    "Thermal state",
                    match d.cell_thermal_state {
                        0 => "Deep frozen",
                        1 => "Frozen",
                        2 => "Chilled",
                        3 => "Cool",
                        4 => "Ideal",
                        5 => "Warm",
                        6 => "Hot safe",
                        7 => "Overheated",
                        8 => "Heat shock",
                        9 => "Critical",
                        _ => "Unknown",
                    }
                    .into(),
                ),
            ],
            Choice::CellPhysics => vec![
                (
                    "Position",
                    format!(
                        "{:.1}, {:.1}, {:.1}",
                        d.position[0], d.position[1], d.position[2]
                    ),
                ),
                (
                    "Velocity",
                    format!(
                        "{:.1}, {:.1}, {:.1}",
                        d.velocity[0], d.velocity[1], d.velocity[2]
                    ),
                ),
                ("Speed", format!("{:.2}", d.velocity_vec3().length())),
                ("Mass", format!("{:.3}", d.mass)),
                ("Radius", format!("{:.3}", d.radius)),
                ("Maximum size", format!("{:.3}", d.max_cell_size)),
                ("Stiffness", format!("{:.3}", d.stiffness)),
                ("Adhesions", format!("{}", d.adhesion_count)),
                ("Heat energy", format!("{:.2}", d.cell_heat_energy)),
            ],
            Choice::CellSignals => (0..8)
                .map(|i| {
                    let a = d.signal_value(i);
                    let b = d.signal_value(i + 8);
                    (
                        "",
                        format!("CH {:02}  {:+5}     |     CH {:02}  {:+5}", i, a, i + 8, b),
                    )
                })
                .collect(),
            _ => vec![
                (
                    "Genome",
                    if view.genome_name.is_empty() {
                        format!("Genome {}", d.genome_id)
                    } else {
                        view.genome_name.clone()
                    },
                ),
                ("Genome modes", format!("{}", view.modes)),
                ("Mode", format!("GPU mode {}", d.mode_index)),
                ("Cell ID", format!("{}", d.cell_id)),
                ("GPU slot", format!("{}", d.cell_slot_index)),
                (
                    "Organism",
                    if d.organism_id == u32::MAX {
                        "Isolated".into()
                    } else {
                        format!("{}", d.organism_id)
                    },
                ),
                ("Age", format!("{:.1} s", d.age)),
                ("Adhesions", format!("{}", d.adhesion_count)),
            ],
        };
        for (i, (label, value)) in rows.into_iter().enumerate() {
            let y = 229.0 + i as f32 * 27.0;
            if label.is_empty() {
                text(y, value, 18.0, palette.text_primary);
            } else {
                p.text(
                    egui::pos2(165.0, y),
                    egui::Align2::LEFT_CENTER,
                    label,
                    egui::FontId::proportional(17.0),
                    palette.text_secondary,
                );
                let mut job = egui::text::LayoutJob::simple(
                    value,
                    egui::FontId::proportional(17.0),
                    palette.text_primary,
                    220.0,
                );
                job.wrap.max_rows = 1;
                job.wrap.break_anywhere = true;
                let galley = p.layout_job(job);
                p.galley(
                    egui::pos2(555.0 - galley.size().x, y - galley.size().y / 2.0),
                    galley,
                    palette.text_primary,
                );
            }
            p.line_segment(
                [egui::pos2(162.0, y + 13.0), egui::pos2(558.0, y + 13.0)],
                egui::Stroke::new(1.0, super::theme::alpha(palette.border_normal, 100)),
            );
        }
        text(
            515.0,
            if view.dead {
                "Last readings at death"
            } else if view.loadable {
                "Genome saved with this selection"
            } else {
                "Genome unavailable"
            }
            .into(),
            14.0,
            palette.text_dim,
        );
    } else {
        text(
            340.0,
            "Point at a cell with the Inspect tool".into(),
            21.0,
            palette.text_primary,
        );
    }
    for (rect, choice, label) in buttons() {
        let enabled = choice != Choice::LoadInspectedGenome || view.loadable;
        super::theme::footer(
            p,
            rect,
            if enabled { label } else { "Genome unavailable" },
            enabled && hover == Some(choice),
            choice == tab,
            palette,
        );
    }
}

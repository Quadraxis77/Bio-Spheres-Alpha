use super::render_world_controls;
use crate::ui::{menu_visibility::MenuAccess, types::WorldSettings};

fn labels(shape: &egui::Shape, output: &mut Vec<String>) {
    match shape {
        egui::Shape::Text(text) => output.push(text.galley.text().to_owned()),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                labels(shape, output);
            }
        }
        _ => {}
    }
}

#[test]
fn rendered_world_controls_preserve_gravity_names_and_gate_developer_settings() {
    for access in [
        MenuAccess::default(),
        MenuAccess {
            advanced: true,
            developer: false,
        },
        MenuAccess {
            advanced: false,
            developer: true,
        },
    ] {
        let ctx = egui::Context::default();
        let mut world = WorldSettings::default();
        world.acceleration_damping = 0.93214;
        world.water_viscosity = 0.62387;
        let before = serde_json::to_value(&world).unwrap();
        let radius = world.world_radius;
        let capacity = world.cell_capacity;
        let output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(640.0, 8000.0),
                )),
                ..Default::default()
            },
            |ui| render_world_controls(ui, &mut world, access, radius, capacity),
        );
        let mut text = Vec::new();
        for shape in output.shapes {
            labels(&shape.shape, &mut text);
        }
        for expected in [
            "Gravity:",
            "Gravity Axis:",
            "X",
            "Y",
            "Z",
            "Radial",
            "Radiation Level:",
            "Gene Pool:",
            "World Radius:",
            "Cell Capacity:",
        ] {
            assert!(
                text.iter().any(|label| label == expected),
                "Missing {expected} at {access:?}"
            );
        }
        for developer_label in [
            "Velocity Damping:",
            "Constraint Iterations:",
            "Water Viscosity:",
            "Solo cell metabolism penalty",
            "Subtle mutations",
        ] {
            assert_eq!(
                text.iter().any(|label| label == developer_label),
                access.developer,
                "Incorrect access for {developer_label} at {access:?}"
            );
        }
        assert_eq!(
            serde_json::to_value(&world).unwrap(),
            before,
            "Rendering or access must not alter settings"
        );
    }
}

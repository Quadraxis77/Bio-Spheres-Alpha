//! Node graph visualization for genome mode connections.
//!
//! Uses egui-snarl to display modes as nodes with connections
//! showing child_a and child_b mode transitions.

use egui_snarl::ui::{PinInfo, SnarlViewer};
use egui_snarl::{InPin, NodeId, OutPin, Snarl};
use serde::{Deserialize, Serialize};

/// Calculate contrasting text color (black or white) based on background luminance
fn contrasting_text_color(bg: egui::Color32) -> egui::Color32 {
    let luminance = 0.299 * bg.r() as f32 + 0.587 * bg.g() as f32 + 0.114 * bg.b() as f32;
    if luminance > 128.0 {
        egui::Color32::BLACK
    } else {
        egui::Color32::WHITE
    }
}

/// Convert Vec3 color to egui Color32
fn vec3_to_color32(c: glam::Vec3) -> egui::Color32 {
    egui::Color32::from_rgb(
        (c.x * 255.0) as u8,
        (c.y * 255.0) as u8,
        (c.z * 255.0) as u8,
    )
}

/// A node in the mode graph representing a single mode.
#[derive(Clone, Serialize, Deserialize)]
pub struct ModeNode {
    /// Index into the genome's modes array
    pub mode_index: usize,
    /// Cached name for display
    pub name: String,
    /// Cached color for display (as RGB array for serde)
    pub color: [f32; 3],
}

impl ModeNode {
    pub fn new(mode_index: usize, name: String, color: glam::Vec3) -> Self {
        Self {
            mode_index,
            name,
            color: [color.x, color.y, color.z],
        }
    }
    
    pub fn color_vec3(&self) -> glam::Vec3 {
        glam::Vec3::new(self.color[0], self.color[1], self.color[2])
    }
}

/// State for the mode graph panel.
#[derive(Serialize, Deserialize)]
pub struct ModeGraphState {
    /// The snarl graph containing mode nodes
    #[serde(skip)]
    pub snarl: Snarl<ModeNode>,
    /// Selected mode to add from dropdown
    pub selected_add_mode: usize,
    /// Counter for positioning new nodes in a column
    pub node_add_count: usize,
}

impl Default for ModeGraphState {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ModeGraphState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModeGraphState")
            .field("selected_add_mode", &self.selected_add_mode)
            .field("node_add_count", &self.node_add_count)
            .finish()
    }
}

impl Clone for ModeGraphState {
    fn clone(&self) -> Self {
        Self {
            snarl: Snarl::new(),
            selected_add_mode: self.selected_add_mode,
            node_add_count: self.node_add_count,
        }
    }
}

impl ModeGraphState {
    pub fn new() -> Self {
        Self {
            snarl: Snarl::new(),
            selected_add_mode: 0,
            node_add_count: 0,
        }
    }
    
    /// Get the next position for a new node, offset in a column
    pub fn next_node_position(&mut self) -> egui::Pos2 {
        let x = 50.0;
        let y = 50.0 + (self.node_add_count as f32 * 120.0);
        self.node_add_count += 1;
        egui::pos2(x, y)
    }
}

/// Viewer implementation for the mode graph.
pub struct ModeGraphViewer<'a> {
    /// Reference to genome modes for reading names/colors
    pub mode_names: &'a [String],
    pub mode_colors: &'a [glam::Vec3],
    /// Mutable reference to child_a mode numbers
    pub child_a_modes: &'a mut [i32],
    /// Mutable reference to child_b mode numbers  
    pub child_b_modes: &'a mut [i32],
    /// Reference to full mode settings for showing changes
    pub mode_settings: &'a [crate::genome::ModeSettings],
}

#[allow(refining_impl_trait)]
impl<'a> SnarlViewer<ModeNode> for ModeGraphViewer<'a> {
    fn title(&mut self, node: &ModeNode) -> String {
        if node.mode_index < self.mode_names.len() {
            self.mode_names[node.mode_index].clone()
        } else {
            node.name.clone()
        }
    }

    fn outputs(&mut self, _node: &ModeNode) -> usize {
        2 // Child A and Child B
    }

    fn inputs(&mut self, _node: &ModeNode) -> usize {
        1 // Parent connection
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<ModeNode>,
    ) -> PinInfo {
        let node = &snarl[pin.id.node];
        let color = if node.mode_index < self.mode_colors.len() {
            vec3_to_color32(self.mode_colors[node.mode_index])
        } else {
            egui::Color32::GRAY
        };

        ui.label("Parent");
        PinInfo::circle().with_fill(color)
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<ModeNode>,
    ) -> PinInfo {
        let node = &snarl[pin.id.node];
        let color = if node.mode_index < self.mode_colors.len() {
            vec3_to_color32(self.mode_colors[node.mode_index])
        } else {
            egui::Color32::GRAY
        };

        let label = match pin.id.output {
            0 => "Child A",
            1 => "Child B",
            _ => "?",
        };
        ui.label(label);
        PinInfo::circle().with_fill(color)
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<ModeNode>) -> bool {
        false
    }

    fn show_graph_menu(
        &mut self,
        _pos: egui::Pos2,
        _ui: &mut egui::Ui,
        _snarl: &mut Snarl<ModeNode>,
    ) {
        // No right-click menu - use toolbar instead
    }

    fn has_node_menu(&mut self, _node: &ModeNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<ModeNode>,
    ) {
        if ui.button("Remove").clicked() {
            snarl.remove_node(node_id);
            ui.close();
        }
    }

    fn connect(
        &mut self,
        from: &OutPin,
        to: &InPin,
        snarl: &mut Snarl<ModeNode>,
    ) {
        // Check if this output pin (Child A or Child B) already has a connection
        // Each output pin can only have one connection
        let from_pin_id = from.id;
        
        // Get existing connections from this output pin
        let existing_remotes: Vec<_> = snarl.out_pin(from_pin_id).remotes.iter().copied().collect();
        
        // Remove existing connections from this output pin
        for remote_in_pin in existing_remotes {
            snarl.disconnect(from_pin_id, remote_in_pin);
            
            // The child mode will be updated below, so no need to reset here
        }
        
        let from_node = &snarl[from.id.node];
        let to_node = &snarl[to.id.node];
        
        let from_mode = from_node.mode_index;
        let to_mode = to_node.mode_index as i32;
        
        // Update the child mode based on which output pin
        match from.id.output {
            0 => {
                if from_mode < self.child_a_modes.len() {
                    self.child_a_modes[from_mode] = to_mode;
                }
            }
            1 => {
                if from_mode < self.child_b_modes.len() {
                    self.child_b_modes[from_mode] = to_mode;
                }
            }
            _ => {}
        }

        // Add the wire to the graph
        snarl.connect(from.id, to.id);
    }

    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<ModeNode>) {
        let from_node = &snarl[from.id.node];
        let from_mode = from_node.mode_index;
        
        // Reset child mode to self when disconnected
        match from.id.output {
            0 => {
                if from_mode < self.child_a_modes.len() {
                    self.child_a_modes[from_mode] = from_mode as i32;
                }
            }
            1 => {
                if from_mode < self.child_b_modes.len() {
                    self.child_b_modes[from_mode] = from_mode as i32;
                }
            }
            _ => {}
        }

        snarl.disconnect(from.id, to.id);
    }

    fn header_frame(
        &mut self,
        frame: egui::Frame,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        snarl: &Snarl<ModeNode>,
    ) -> egui::Frame {
        let node = &snarl[node_id];
        let color = if node.mode_index < self.mode_colors.len() {
            vec3_to_color32(self.mode_colors[node.mode_index])
        } else {
            egui::Color32::GRAY
        };

        frame.fill(color)
    }

    fn show_header(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<ModeNode>,
    ) {
        let node = &snarl[node_id];
        let color = if node.mode_index < self.mode_colors.len() {
            vec3_to_color32(self.mode_colors[node.mode_index])
        } else {
            egui::Color32::GRAY
        };
        let text_color = contrasting_text_color(color);
        
        let title = if node.mode_index < self.mode_names.len() {
            self.mode_names[node.mode_index].clone()
        } else {
            node.name.clone()
        };
        
        ui.label(egui::RichText::new(title).color(text_color).strong());
    }

    fn has_body(&mut self, _node: &ModeNode) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node_id: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<ModeNode>,
    ) {
        let node = &snarl[node_id];
        let mode_idx = node.mode_index;
        
        if mode_idx >= self.mode_settings.len() {
            return;
        }
        
        let mode = &self.mode_settings[mode_idx];
        let default = crate::genome::ModeSettings::default();
        
        // Collect changes from default
        let mut changes = Vec::new();
        
        if mode.cell_type != default.cell_type {
            changes.push(format!("Type:{}", mode.cell_type));
        }
        if mode.parent_make_adhesion != default.parent_make_adhesion {
            changes.push("Adhesion".to_string());
        }
        if (mode.split_mass - default.split_mass).abs() > 0.01 {
            changes.push(format!("Mass:{:.1}", mode.split_mass));
        }
        if (mode.split_interval - default.split_interval).abs() > 0.01 {
            changes.push(format!("Int:{:.1}s", mode.split_interval));
        }
        if (mode.nutrient_gain_rate - default.nutrient_gain_rate).abs() > 0.01 {
            changes.push(format!("Gain:{:.2}", mode.nutrient_gain_rate));
        }
        if (mode.split_ratio - default.split_ratio).abs() > 0.01 {
            changes.push(format!("Ratio:{:.0}%", mode.split_ratio * 100.0));
        }
        if mode.max_splits != default.max_splits {
            if mode.max_splits == -1 {
                changes.push("Splits:∞".to_string());
            } else {
                changes.push(format!("Splits:{}", mode.max_splits));
            }
        }
        if mode.min_adhesions != default.min_adhesions {
            changes.push(format!("MinAdh:{}", mode.min_adhesions));
        }
        if mode.max_adhesions != default.max_adhesions {
            changes.push(format!("MaxAdh:{}", mode.max_adhesions));
        }
        if (mode.opacity - default.opacity).abs() > 0.01 {
            changes.push(format!("Opacity:{:.0}%", mode.opacity * 100.0));
        }
        if (mode.emissive - default.emissive).abs() > 0.01 {
            changes.push(format!("Glow:{:.1}", mode.emissive));
        }
        if (mode.swim_force - default.swim_force).abs() > 0.01 {
            changes.push(format!("Swim:{:.1}", mode.swim_force));
        }
        
        if changes.is_empty() {
            ui.label(egui::RichText::new("(default)").weak().small());
        } else {
            // Show changes in a compact format
            for change in changes.iter().take(4) {
                ui.label(egui::RichText::new(change).small());
            }
            if changes.len() > 4 {
                ui.label(egui::RichText::new(format!("+{} more", changes.len() - 4)).weak().small());
            }
        }
    }
}

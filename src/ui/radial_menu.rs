//! Radial menu for GPU scene tools.
//!
//! Provides a radial menu that opens when holding Alt, allowing quick tool selection
//! for cell manipulation in the GPU simulation scene.

use egui::{Color32, Pos2, Stroke, Vec2};

/// Tools available in the radial menu for GPU scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RadialTool {
    /// No tool selected / default state
    #[default]
    None,
    /// Insert new cells
    Insert,
    /// Inspect cell properties
    Inspect,
    /// Drag cells around
    Drag,
    /// Apply force boost to cells
    Boost,
    /// Remove cells
    Remove,
}

impl RadialTool {
    /// Get display name for the tool.
    pub fn display_name(&self) -> &'static str {
        match self {
            RadialTool::None => "None",
            RadialTool::Insert => "Insert",
            RadialTool::Inspect => "Inspect",
            RadialTool::Drag => "Drag",
            RadialTool::Boost => "Boost",
            RadialTool::Remove => "Remove",
        }
    }

    /// Get icon/emoji for the tool.
    pub fn icon(&self) -> &'static str {
        match self {
            RadialTool::None => "⊘",
            RadialTool::Insert => "➕",
            RadialTool::Inspect => "🔍",
            RadialTool::Drag => "✋",
            RadialTool::Boost => "⚡",
            RadialTool::Remove => "➖",
        }
    }

    /// Get all tools for the radial menu (excluding None).
    pub fn all_tools() -> &'static [RadialTool] {
        &[
            RadialTool::Insert,
            RadialTool::Inspect,
            RadialTool::Drag,
            RadialTool::Boost,
            RadialTool::Remove,
        ]
    }
}

/// State for the radial menu.
#[derive(Debug, Clone, Default)]
pub struct RadialMenuState {
    /// Whether the menu is currently visible
    pub visible: bool,
    /// Center position of the menu (where Alt was pressed)
    pub center: Pos2,
    /// Currently hovered segment index (None if center or outside)
    pub hovered_segment: Option<usize>,
    /// Currently selected/active tool
    pub active_tool: RadialTool,
    /// Whether Alt key is currently held
    pub alt_held: bool,
    /// Cell index being dragged (for Drag tool)
    pub dragging_cell: Option<usize>,
    /// Cell index being inspected (for Inspect tool)
    pub inspected_cell: Option<usize>,
}

impl RadialMenuState {
    /// Create a new radial menu state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open the menu at the given position.
    pub fn open(&mut self, position: Pos2) {
        self.visible = true;
        self.center = position;
        self.hovered_segment = None;
        self.alt_held = true;
    }

    /// Close the menu and optionally select the hovered tool.
    pub fn close(&mut self, select_hovered: bool) {
        if select_hovered {
            if let Some(idx) = self.hovered_segment {
                let tools = RadialTool::all_tools();
                if idx < tools.len() {
                    self.active_tool = tools[idx];
                }
            }
            // If no segment hovered (center), clear the active tool
            else {
                self.active_tool = RadialTool::None;
            }
        }
        self.visible = false;
        self.alt_held = false;
        self.hovered_segment = None;
    }

    /// Update hovered segment based on mouse position.
    pub fn update_hover(&mut self, mouse_pos: Pos2) {
        if !self.visible {
            return;
        }

        let config = RadialMenuConfig::default();
        let delta = mouse_pos - self.center;
        let distance = delta.length();

        // Dead zone in center (use inner radius from config)
        if distance < config.inner_radius {
            self.hovered_segment = None;
            return;
        }

        let tools = RadialTool::all_tools();
        let segment_count = tools.len();
        let segment_angle = std::f32::consts::TAU / segment_count as f32;

        // atan2 returns angle from -PI to PI, with 0 pointing right (+X)
        // We want 0 to be at the top (-Y in screen coords), so rotate by PI/2
        let angle = delta.y.atan2(delta.x);
        
        // Normalize to 0..TAU range, with 0 at top
        // Add PI/2 to rotate so top is 0, then normalize
        let normalized = (angle + std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU) 
            % std::f32::consts::TAU;
        
        // Determine which segment this angle falls into
        let segment_idx = (normalized / segment_angle) as usize % segment_count;
        self.hovered_segment = Some(segment_idx);
    }
}

/// Configuration for radial menu appearance.
pub struct RadialMenuConfig {
    pub inner_radius: f32,
    pub outer_radius: f32,
    pub segment_gap: f32,
    pub bg_color: Color32,
    pub hover_color: Color32,
    pub active_color: Color32,
    pub text_color: Color32,
    pub border_color: Color32,
    pub border_width: f32,
}

impl Default for RadialMenuConfig {
    fn default() -> Self {
        Self {
            inner_radius: 40.0,
            outer_radius: 120.0,
            segment_gap: 0.03, // radians
            bg_color: Color32::from_rgba_unmultiplied(30, 30, 40, 220),
            hover_color: Color32::from_rgba_unmultiplied(70, 130, 180, 240),
            active_color: Color32::from_rgba_unmultiplied(100, 180, 100, 240),
            text_color: Color32::WHITE,
            border_color: Color32::from_rgba_unmultiplied(100, 100, 120, 200),
            border_width: 2.0,
        }
    }
}

/// Render the radial menu.
pub fn show_radial_menu(ctx: &egui::Context, state: &mut RadialMenuState) {
    if !state.visible {
        return;
    }

    let config = RadialMenuConfig::default();
    let tools = RadialTool::all_tools();
    let segment_count = tools.len();
    let segment_angle = std::f32::consts::TAU / segment_count as f32;

    // Update hover based on current pointer position
    if let Some(pos) = ctx.pointer_hover_pos() {
        state.update_hover(pos);
    }

    // Draw using egui Area for overlay
    egui::Area::new(egui::Id::new("radial_menu"))
        .fixed_pos(Pos2::ZERO)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            let painter = ui.painter();
            let center = state.center;

            // Draw segments - first segment starts at top (angle 0 after rotation)
            for (i, tool) in tools.iter().enumerate() {
                // Start angle for this segment (0 = top, going clockwise)
                // In screen coords: -PI/2 is top, angles increase clockwise
                let start_angle = -std::f32::consts::FRAC_PI_2 
                    + (i as f32 * segment_angle) 
                    + config.segment_gap / 2.0;
                let end_angle = start_angle + segment_angle - config.segment_gap;

                let is_hovered = state.hovered_segment == Some(i);
                let is_active = state.active_tool == *tool;

                let fill_color = if is_hovered {
                    config.hover_color
                } else if is_active {
                    config.active_color
                } else {
                    config.bg_color
                };

                // Draw segment arc
                draw_arc_segment(
                    painter,
                    center,
                    config.inner_radius,
                    config.outer_radius,
                    start_angle,
                    end_angle,
                    fill_color,
                    Stroke::new(config.border_width, config.border_color),
                );

                // Draw icon and label at segment center
                let mid_angle = (start_angle + end_angle) / 2.0;
                let label_radius = (config.inner_radius + config.outer_radius) / 2.0;
                let label_pos = center + Vec2::new(
                    mid_angle.cos() * label_radius,
                    mid_angle.sin() * label_radius,
                );

                // Icon
                painter.text(
                    label_pos - Vec2::new(0.0, 8.0),
                    egui::Align2::CENTER_CENTER,
                    tool.icon(),
                    egui::FontId::proportional(20.0),
                    config.text_color,
                );

                // Label (only show on hover for cleaner look)
                if is_hovered {
                    painter.text(
                        label_pos + Vec2::new(0.0, 12.0),
                        egui::Align2::CENTER_CENTER,
                        tool.display_name(),
                        egui::FontId::proportional(12.0),
                        config.text_color,
                    );
                }
            }

            // Draw center circle with current tool indicator
            painter.circle_filled(center, config.inner_radius - 5.0, config.bg_color);
            painter.circle_stroke(
                center,
                config.inner_radius - 5.0,
                Stroke::new(config.border_width, config.border_color),
            );

            // Show current active tool in center
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                state.active_tool.icon(),
                egui::FontId::proportional(24.0),
                config.text_color,
            );
        });
}

/// Draw an arc segment (pie slice).
fn draw_arc_segment(
    painter: &egui::Painter,
    center: Pos2,
    inner_radius: f32,
    outer_radius: f32,
    start_angle: f32,
    end_angle: f32,
    fill: Color32,
    stroke: Stroke,
) {
    const SEGMENTS: usize = 32;
    let angle_step = (end_angle - start_angle) / SEGMENTS as f32;

    let mut points = Vec::with_capacity(SEGMENTS * 2 + 2);

    // Outer arc (clockwise)
    for i in 0..=SEGMENTS {
        let angle = start_angle + angle_step * i as f32;
        points.push(center + Vec2::new(angle.cos() * outer_radius, angle.sin() * outer_radius));
    }

    // Inner arc (counter-clockwise)
    for i in (0..=SEGMENTS).rev() {
        let angle = start_angle + angle_step * i as f32;
        points.push(center + Vec2::new(angle.cos() * inner_radius, angle.sin() * inner_radius));
    }

    // Draw filled polygon
    painter.add(egui::Shape::convex_polygon(points.clone(), fill, stroke));
}


/// Render a custom cursor overlay showing the active tool icon.
/// Call this when a tool is active to display the tool icon near the cursor.
pub fn show_tool_cursor(ctx: &egui::Context, state: &RadialMenuState) {
    // Don't show cursor overlay if no tool is active or menu is visible
    if state.active_tool == RadialTool::None || state.visible {
        return;
    }

    if let Some(pos) = ctx.pointer_hover_pos() {
        // Use closed hand icon when actively dragging a cell
        let icon = if state.active_tool == RadialTool::Drag && state.dragging_cell.is_some() {
            "✊" // Closed hand when dragging
        } else {
            state.active_tool.icon()
        };
        
        egui::Area::new(egui::Id::new("tool_cursor"))
            .fixed_pos(pos - Vec2::new(12.0, 12.0)) // Center icon at cursor tip
            .order(egui::Order::Tooltip)
            .interactable(false)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(icon)
                        .size(24.0)
                        .color(Color32::WHITE),
                );
            });
    }
}

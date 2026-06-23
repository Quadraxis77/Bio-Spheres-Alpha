//! UI System for Bio-Spheres using egui-wgpu and egui-winit.
//!
//! This module provides the core UI rendering system that integrates egui
//! with the existing wgpu/winit application.

use egui_wgpu::ScreenDescriptor;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::ui::types::GlobalUiState;

/// The main UI system that manages egui rendering.
///
/// This struct coordinates between egui-winit for input handling and
/// egui-wgpu for GPU rendering.
// -- Bio-Spheres Biotech Theme Colors -----------------------------------------
pub mod theme {
    use egui::Color32;

    // Background layers
    pub const BG_DARKEST: Color32 = Color32::from_rgb(8, 12, 22); // window bg
    pub const BG_PANEL: Color32 = Color32::from_rgb(12, 17, 30); // panel bg
    pub const BG_WIDGET: Color32 = Color32::from_rgb(18, 25, 42); // widget bg
    pub const BG_HOVER: Color32 = Color32::from_rgb(24, 34, 58); // hovered widget
    pub const BG_ACTIVE: Color32 = Color32::from_rgb(30, 44, 74); // active/pressed
    pub const BG_SELECTED: Color32 = Color32::from_rgb(20, 50, 80); // selected item

    // Accent colors
    pub const ACCENT_TEAL: Color32 = Color32::from_rgb(0, 200, 160); // primary accent
    pub const ACCENT_CYAN: Color32 = Color32::from_rgb(40, 180, 220); // secondary accent
    pub const ACCENT_DIM: Color32 = Color32::from_rgb(0, 80, 64); // dimmed accent

    // Text
    pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(210, 220, 235);
    pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(130, 150, 175);
    pub const TEXT_DIM: Color32 = Color32::from_rgb(70, 90, 120);
    pub const TEXT_ACCENT: Color32 = Color32::from_rgb(0, 200, 160);

    // Borders / strokes
    pub const BORDER_SUBTLE: Color32 = Color32::from_rgb(30, 45, 70);
    pub const BORDER_NORMAL: Color32 = Color32::from_rgb(45, 65, 100);
    pub const BORDER_BRIGHT: Color32 = Color32::from_rgb(0, 140, 110);

    // Status colors
    pub const STATUS_GREEN: Color32 = Color32::from_rgb(60, 200, 100);
    pub const STATUS_YELLOW: Color32 = Color32::from_rgb(220, 180, 50);
    pub const STATUS_RED: Color32 = Color32::from_rgb(220, 70, 70);
    pub const STATUS_BLUE: Color32 = Color32::from_rgb(60, 140, 220);

    // Top bar
    pub const TOPBAR_BG: Color32 = Color32::from_rgb(6, 9, 18);
    pub const TOPBAR_BORDER: Color32 = Color32::from_rgb(0, 120, 95);

    // Live Simulation button
    pub const BTN_LIVE_BG: Color32 = Color32::from_rgb(180, 40, 40);
    pub const BTN_LIVE_HOVER: Color32 = Color32::from_rgb(210, 55, 55);
    pub const BTN_EDITOR_BG: Color32 = Color32::from_rgb(30, 120, 80);
    pub const BTN_EDITOR_HOVER: Color32 = Color32::from_rgb(40, 150, 100);
}

// -- Active theme palette ------------------------------------------------------
// Stores the full resolved color palette for the currently active theme.
// Updated by `apply_theme` every time the theme changes. All rendering code
// reads from here instead of the static `theme::*` constants so that every
// part of the UI - panels, dock, status bar, side rail - responds to theme
// switches without needing to re-derive colors from the enum each frame.
#[derive(Clone, Copy)]
pub struct ActivePalette {
    pub bg_darkest: egui::Color32,
    pub bg_panel: egui::Color32,
    pub bg_widget: egui::Color32,
    pub bg_hover: egui::Color32,
    pub bg_active: egui::Color32,
    pub bg_selected: egui::Color32,
    pub accent_primary: egui::Color32,
    pub accent_secondary: egui::Color32,
    pub text_primary: egui::Color32,
    pub text_secondary: egui::Color32,
    pub text_dim: egui::Color32,
    pub border_subtle: egui::Color32,
    pub border_normal: egui::Color32,
    pub border_bright: egui::Color32,
    pub topbar_bg: egui::Color32,
    pub topbar_border: egui::Color32,
    pub status_ok: egui::Color32,
    pub status_warn: egui::Color32,
    pub status_err: egui::Color32,
    pub status_info: egui::Color32,
    /// Icon color for inactive rail buttons - always readable against the dark rail bg.
    pub rail_icon: egui::Color32,
    /// Icon color for active/toggled rail buttons (drawn on accent_primary bg).
    pub rail_icon_active: egui::Color32,
    /// The active theme variant - available to any rendering code that needs per-theme logic.
    pub theme: crate::ui::types::UiTheme,
}

impl Default for ActivePalette {
    fn default() -> Self {
        // Biotech Dark defaults
        Self {
            bg_darkest: egui::Color32::from_rgb(8, 12, 22),
            bg_panel: egui::Color32::from_rgb(12, 17, 30),
            bg_widget: egui::Color32::from_rgb(18, 25, 42),
            bg_hover: egui::Color32::from_rgb(24, 34, 58),
            bg_active: egui::Color32::from_rgb(30, 44, 74),
            bg_selected: egui::Color32::from_rgb(20, 50, 80),
            accent_primary: egui::Color32::from_rgb(0, 200, 160),
            accent_secondary: egui::Color32::from_rgb(40, 180, 220),
            text_primary: egui::Color32::from_rgb(210, 220, 235),
            text_secondary: egui::Color32::from_rgb(130, 150, 175),
            text_dim: egui::Color32::from_rgb(70, 90, 120),
            border_subtle: egui::Color32::from_rgb(30, 45, 70),
            border_normal: egui::Color32::from_rgb(45, 65, 100),
            border_bright: egui::Color32::from_rgb(0, 140, 110),
            topbar_bg: egui::Color32::from_rgb(6, 9, 18),
            topbar_border: egui::Color32::from_rgb(0, 120, 95),
            status_ok: egui::Color32::from_rgb(60, 200, 100),
            status_warn: egui::Color32::from_rgb(220, 180, 50),
            status_err: egui::Color32::from_rgb(220, 70, 70),
            status_info: egui::Color32::from_rgb(60, 140, 220),
            rail_icon: egui::Color32::from_rgb(160, 185, 220),
            rail_icon_active: egui::Color32::WHITE,
            theme: crate::ui::types::UiTheme::BiotechDark,
        }
    }
}

std::thread_local! {
    static ACTIVE_PALETTE: std::cell::RefCell<ActivePalette> =
        std::cell::RefCell::new(ActivePalette::default());
}

/// Read the current active palette. Call this at the top of any rendering
/// function that needs theme-aware colors.
pub fn palette() -> ActivePalette {
    ACTIVE_PALETTE.with(|cell| *cell.borrow())
}

pub struct UiSystem {
    /// egui context for immediate mode UI
    pub ctx: egui::Context,
    /// egui-winit state for input handling
    pub winit_state: egui_winit::State,
    /// egui-wgpu renderer
    pub renderer: egui_wgpu::Renderer,
    /// Global UI state
    pub state: GlobalUiState,
    /// Viewport rectangle for mouse filtering (set during rendering)
    pub viewport_rect: Option<egui::Rect>,
    /// Last applied UI scale for change detection
    last_scale: f32,
    /// Original style values for scaling
    original_spacing: Option<egui::style::Spacing>,
    original_text_styles: Option<std::collections::BTreeMap<egui::TextStyle, egui::FontId>>,
    /// Timer for auto-save functionality
    save_timer: std::time::Instant,
    /// Whether the UI state has changed since last save
    ui_state_dirty: bool,
    /// Whether the biotech theme has been applied
    theme_applied: bool,
    /// Last applied theme for change detection
    last_theme: crate::ui::types::UiTheme,
    /// Last applied custom theme palette for change detection
    last_custom_theme: crate::ui::types::CustomThemePalette,
    /// Genome browser window state - lives here (not on GlobalUiState) so it
    /// is never cloned and texture handles survive across frames.
    pub genome_browser: crate::ui::genome_browser::GenomeBrowserState,
    /// Toast notification queue - short messages shown in the bottom-right corner.
    /// Also lives here to avoid being wiped by the GlobalUiState clone.
    pub toasts: Vec<crate::ui::toast::Toast>,
    /// App icon texture used as the brand glyph in the top bar.
    app_icon: Option<egui::TextureHandle>,
    /// Loading animation GIF frames (user-provided, shown during GIF/save operations).
    pub loading_gif_frames: Vec<egui::TextureHandle>,
    /// Current frame index for the loading animation.
    pub loading_gif_frame: usize,
    /// Timer for loading animation frame advance.
    pub loading_gif_timer: f32,
    /// Backend-only report stream controller. UI panels only read its bounded history.
    pub field_report_director: crate::field_report::FieldReportDirector,
    /// Last lineage scan frame submitted to the report director.
    last_report_scan_frame: Option<i32>,
}

impl UiSystem {
    /// Create a new UI system.
    ///
    /// # Arguments
    /// * `device` - The wgpu device for creating GPU resources
    /// * `surface_format` - The texture format of the render surface
    /// * `window` - The winit window for input handling
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &Window,
    ) -> Self {
        // Create egui context
        let ctx = egui::Context::default();

        // Install fallback fonts so every Unicode symbol, arrow, and emoji renders.
        //
        // Priority order (egui tries each font in order until a glyph is found):
        //   1. Hack (egui built-in) - ASCII + common programming glyphs
        //   2. NotoSans-Regular    - broad Unicode: arrows, math, box-drawing, etc.
        //   3. Segoe UI Symbol     - Windows symbol font: geometric shapes, misc symbols
        //   4. Segoe UI Emoji      - full colour emoji fallback
        {
            let mut fonts = egui::FontDefinitions::default();

            fonts.font_data.insert(
                "NotoSans".to_owned(),
                egui::FontData::from_static(include_bytes!(
                    "../../assets/fonts/NotoSans-Regular.ttf"
                ))
                .into(),
            );
            fonts.font_data.insert(
                "SegoeSymbol".to_owned(),
                egui::FontData::from_static(include_bytes!("../../assets/fonts/seguisym.ttf"))
                    .into(),
            );
            fonts.font_data.insert(
                "SegoeEmoji".to_owned(),
                egui::FontData::from_static(include_bytes!("../../assets/fonts/seguiemj.ttf"))
                    .into(),
            );

            // Append fallbacks to both Proportional and Monospace families so
            // every text style benefits from the extended coverage.
            for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                let list = fonts.families.entry(family).or_default();
                list.push("NotoSans".to_owned());
                list.push("SegoeSymbol".to_owned());
                list.push("SegoeEmoji".to_owned());
            }

            ctx.set_fonts(fonts);
        }

        // Create egui-winit state for input handling
        let viewport_id = egui::ViewportId::ROOT;
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );

        // Create egui-wgpu renderer
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        // Create default UI state
        let state = GlobalUiState::load();

        // Load the app icon as the top-bar brand glyph. Embedded so it always
        // ships with the binary regardless of working directory.
        let app_icon = load_app_icon_texture(&ctx);

        // Try to load a user-provided loading animation GIF from assets/.
        let loading_gif_frames = load_loading_gif_frames(&ctx);

        Self {
            ctx,
            winit_state,
            renderer,
            state,
            viewport_rect: None,
            last_scale: 1.0,
            original_spacing: None,
            original_text_styles: None,
            save_timer: std::time::Instant::now(),
            ui_state_dirty: false,
            theme_applied: false,
            last_theme: crate::ui::types::UiTheme::default(),
            last_custom_theme: crate::ui::types::CustomThemePalette::default(),
            genome_browser: crate::ui::genome_browser::GenomeBrowserState::default(),
            toasts: Vec::new(),
            app_icon,
            loading_gif_frames,
            loading_gif_frame: 0,
            loading_gif_timer: 0.0,
            field_report_director: crate::field_report::FieldReportDirector::default(),
            last_report_scan_frame: None,
        }
    }

    /// Handle a winit window event.
    ///
    /// Returns `true` if egui consumed the event (i.e., the event should not
    /// be passed to other systems like the camera controller).
    pub fn handle_event(
        &mut self,
        window: &Window,
        event: &WindowEvent,
    ) -> egui_winit::EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    /// Check if egui wants pointer (mouse) input.
    ///
    /// Returns `true` if the mouse is over an egui UI element (excluding the viewport)
    /// or if egui is actively using the pointer (e.g., dragging a slider).
    pub fn wants_pointer_input(&self) -> bool {
        // When UI is hidden, all pointer input passes straight to the scene.
        if self.state.hide_ui {
            return false;
        }

        // Check if pointer is over viewport - if so, camera should get input
        if let Some(viewport) = self.viewport_rect {
            if let Some(pos) = self.ctx.pointer_hover_pos() {
                if viewport.contains(pos) {
                    return false;
                }
            }
        } else {
            // No viewport rect yet (e.g. first frame after menu transition).
            // Don't block camera input - the whole window is the viewport.
            return false;
        }

        // Otherwise, check if egui wants the pointer
        self.ctx.egui_wants_pointer_input() || self.ctx.is_pointer_over_egui()
    }

    /// Returns `true` if scroll/wheel input should be consumed by the UI rather
    /// than passed to the camera. This is a superset of `wants_pointer_input` -
    /// it also blocks scroll when a floating window (e.g. genome browser) is open,
    /// because egui floating windows sit on top of the viewport rect but the
    /// pointer-over check doesn't catch them reliably for scroll events.
    pub fn wants_scroll_input(&self) -> bool {
        // If the genome browser is open, always consume scroll.
        if self.genome_browser.open {
            return true;
        }
        self.wants_pointer_input()
    }

    /// Check if egui wants keyboard input.
    ///
    /// Returns `true` if egui has keyboard focus (e.g., a text field is active).
    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.egui_wants_keyboard_input()
    }

    /// Begin a new egui frame.
    ///
    /// Call this at the start of each frame before rendering UI.
    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.winit_state.take_egui_input(window);
        self.ctx.begin_pass(raw_input);

        // Clear the viewport rect every frame so it only ever holds the
        // rect from the current frame's Viewport tab render. This prevents
        // a stale rect from a different scene mode being used for brackets.
        self.viewport_rect = None;
    }

    /// Apply UI scale to the egui context style.
    ///
    /// This method scales spacing, text sizes, and other UI elements based on
    /// the current ui_scale factor.
    fn apply_ui_scale(&mut self) {
        let scale = self.state.ui_scale;

        self.ctx.global_style_mut(|style| {
            // Store original values on first run
            if self.original_spacing.is_none() {
                self.original_spacing = Some(style.spacing.clone());
                self.original_text_styles = Some(style.text_styles.clone());
            }

            // Apply scale from original values (not multiplicatively)
            if let Some(ref original_spacing) = self.original_spacing {
                style.spacing.item_spacing = original_spacing.item_spacing * scale;
                style.spacing.button_padding = original_spacing.button_padding * scale;
                style.spacing.menu_margin = original_spacing.menu_margin * scale;
                style.spacing.indent = original_spacing.indent * scale;
                style.spacing.interact_size = original_spacing.interact_size * scale;
                style.spacing.slider_width = original_spacing.slider_width * scale;
                style.spacing.combo_width = original_spacing.combo_width * scale;
                style.spacing.text_edit_width = original_spacing.text_edit_width * scale;
                style.spacing.icon_width = original_spacing.icon_width * scale;
                style.spacing.icon_width_inner = original_spacing.icon_width_inner * scale;
                style.spacing.icon_spacing = original_spacing.icon_spacing * scale;
                style.spacing.tooltip_width = original_spacing.tooltip_width * scale;
                style.spacing.menu_width = original_spacing.menu_width * scale;
                style.spacing.combo_height = original_spacing.combo_height * scale;
            }

            // Scale text sizes from original values
            if let Some(ref original_text_styles) = self.original_text_styles {
                for (text_style, font_id) in style.text_styles.iter_mut() {
                    if let Some(original_font) = original_text_styles.get(text_style) {
                        font_id.size = original_font.size * scale;
                    }
                }
            }
        });

        log::debug!("Applied UI scale: {:.2}x", scale);
    }

    /// Apply the selected UI theme to the egui context.
    ///
    /// Dispatches to per-theme color sets. Called once at startup and whenever
    /// `state.selected_theme` changes.
    fn apply_theme(&self, theme_choice: crate::ui::types::UiTheme) {
        use crate::ui::types::UiTheme;

        // -- Per-theme color palette -------------------------------------------
        // Each tuple: (bg_darkest, bg_panel, bg_widget, bg_hover, bg_active,
        //              bg_selected, accent_primary, accent_secondary,
        //              text_primary, text_secondary, text_dim,
        //              border_subtle, border_normal, border_bright,
        //              topbar_bg, topbar_border,
        //              status_ok, status_warn, status_err, status_info)
        #[allow(clippy::type_complexity)]
        let (
            bg_darkest,
            bg_panel,
            bg_widget,
            bg_hover,
            bg_active,
            bg_selected,
            accent_primary,
            accent_secondary,
            text_primary,
            text_secondary,
            text_dim,
            border_subtle,
            border_normal,
            border_bright,
            topbar_bg,
            topbar_border,
            status_ok,
            status_warn,
            status_err,
            status_info,
        ): (
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
            egui::Color32,
        ) = match theme_choice {
            // -- BIOTECH DARK -------------------------------------------------
            // Original. Navy backgrounds, bright teal accent, high-contrast white text.
            UiTheme::BiotechDark => (
                egui::Color32::from_rgb(6, 9, 18),      // bg_darkest
                egui::Color32::from_rgb(12, 17, 32),    // bg_panel
                egui::Color32::from_rgb(22, 30, 52),    // bg_widget
                egui::Color32::from_rgb(32, 44, 72),    // bg_hover
                egui::Color32::from_rgb(42, 58, 95),    // bg_active
                egui::Color32::from_rgb(18, 55, 85),    // bg_selected
                egui::Color32::from_rgb(0, 220, 175),   // accent_primary  - bright teal
                egui::Color32::from_rgb(60, 195, 240),  // accent_secondary - cyan
                egui::Color32::from_rgb(225, 235, 255), // text_primary    - bright white-blue
                egui::Color32::from_rgb(155, 175, 210), // text_secondary
                egui::Color32::from_rgb(80, 100, 145),  // text_dim
                egui::Color32::from_rgb(28, 42, 72),    // border_subtle
                egui::Color32::from_rgb(50, 72, 115),   // border_normal
                egui::Color32::from_rgb(0, 180, 140),   // border_bright
                egui::Color32::from_rgb(4, 6, 14),      // topbar_bg
                egui::Color32::from_rgb(0, 160, 125),   // topbar_border
                egui::Color32::from_rgb(60, 210, 100),  // status_ok
                egui::Color32::from_rgb(220, 185, 50),  // status_warn
                egui::Color32::from_rgb(225, 70, 70),   // status_err
                egui::Color32::from_rgb(60, 150, 230),  // status_info
            ),
            // -- ARCTIC - light, dark status colors for readability on white --
            UiTheme::Arctic => (
                egui::Color32::from_rgb(255, 255, 255),
                egui::Color32::from_rgb(240, 245, 255),
                egui::Color32::from_rgb(218, 228, 245),
                egui::Color32::from_rgb(195, 210, 238),
                egui::Color32::from_rgb(170, 192, 230),
                egui::Color32::from_rgb(185, 215, 255),
                egui::Color32::from_rgb(10, 90, 200),
                egui::Color32::from_rgb(0, 155, 185),
                egui::Color32::from_rgb(8, 15, 35),
                egui::Color32::from_rgb(45, 70, 120),
                egui::Color32::from_rgb(110, 135, 175),
                egui::Color32::from_rgb(185, 200, 225),
                egui::Color32::from_rgb(140, 165, 205),
                egui::Color32::from_rgb(10, 90, 200),
                egui::Color32::from_rgb(220, 230, 248),
                egui::Color32::from_rgb(10, 90, 200),
                egui::Color32::from_rgb(20, 140, 55), // status_ok   - dark green
                egui::Color32::from_rgb(155, 95, 0),  // status_warn - dark amber (no yellow)
                egui::Color32::from_rgb(185, 30, 30), // status_err  - dark red
                egui::Color32::from_rgb(15, 90, 175), // status_info - dark blue
            ),
            // -- PARCHMENT - light, earthy dark status colors -----------------
            UiTheme::Parchment => (
                egui::Color32::from_rgb(255, 252, 238),
                egui::Color32::from_rgb(245, 235, 210),
                egui::Color32::from_rgb(228, 212, 178),
                egui::Color32::from_rgb(208, 188, 148),
                egui::Color32::from_rgb(188, 165, 118),
                egui::Color32::from_rgb(215, 185, 130),
                egui::Color32::from_rgb(165, 55, 15),
                egui::Color32::from_rgb(120, 75, 20),
                egui::Color32::from_rgb(28, 14, 4),
                egui::Color32::from_rgb(85, 50, 18),
                egui::Color32::from_rgb(148, 112, 65),
                egui::Color32::from_rgb(205, 185, 148),
                egui::Color32::from_rgb(172, 145, 100),
                egui::Color32::from_rgb(165, 55, 15),
                egui::Color32::from_rgb(235, 222, 192),
                egui::Color32::from_rgb(145, 48, 12),
                egui::Color32::from_rgb(25, 120, 45), // status_ok   - forest green
                egui::Color32::from_rgb(145, 80, 0),  // status_warn - dark amber
                egui::Color32::from_rgb(170, 30, 20), // status_err  - dark red
                egui::Color32::from_rgb(20, 80, 160), // status_info - dark blue
            ),
            // -- BLOSSOM - light, deep jewel-tone status colors ---------------
            UiTheme::Blossom => (
                egui::Color32::from_rgb(255, 235, 248), // bg_darkest  - hot pink tint
                egui::Color32::from_rgb(252, 215, 238), // bg_panel    - vivid pink-white
                egui::Color32::from_rgb(242, 185, 222), // bg_widget   - medium pink
                egui::Color32::from_rgb(228, 155, 205), // bg_hover    - deeper pink
                egui::Color32::from_rgb(212, 120, 185), // bg_active   - strong pink
                egui::Color32::from_rgb(245, 160, 215), // bg_selected - bright pink highlight
                egui::Color32::from_rgb(185, 10, 105),  // accent_primary  - deep magenta-rose
                egui::Color32::from_rgb(110, 15, 165),  // accent_secondary - deep violet
                egui::Color32::from_rgb(55, 5, 35),     // text_primary    - near-black plum
                egui::Color32::from_rgb(120, 20, 80),   // text_secondary  - dark rose
                egui::Color32::from_rgb(175, 80, 140),  // text_dim        - medium rose
                egui::Color32::from_rgb(225, 160, 205), // border_subtle
                egui::Color32::from_rgb(195, 110, 168), // border_normal
                egui::Color32::from_rgb(185, 10, 105),  // border_bright   - deep rose
                egui::Color32::from_rgb(248, 200, 232), // topbar_bg
                egui::Color32::from_rgb(165, 8, 92),    // topbar_border
                egui::Color32::from_rgb(15, 115, 45),   // status_ok   - deep green
                egui::Color32::from_rgb(130, 65, 0),    // status_warn - dark amber
                egui::Color32::from_rgb(160, 15, 40),   // status_err  - deep crimson
                egui::Color32::from_rgb(45, 20, 155),   // status_info - deep indigo
            ),
            // -- CRIMSON - dark, bright status colors -------------------------
            UiTheme::Crimson => (
                egui::Color32::from_rgb(10, 2, 5),
                egui::Color32::from_rgb(28, 8, 15),
                egui::Color32::from_rgb(50, 15, 25),
                egui::Color32::from_rgb(72, 22, 35),
                egui::Color32::from_rgb(98, 30, 48),
                egui::Color32::from_rgb(88, 18, 12),
                egui::Color32::from_rgb(225, 178, 35),
                egui::Color32::from_rgb(255, 215, 80),
                egui::Color32::from_rgb(252, 238, 215),
                egui::Color32::from_rgb(205, 165, 130),
                egui::Color32::from_rgb(130, 85, 72),
                egui::Color32::from_rgb(55, 18, 28),
                egui::Color32::from_rgb(95, 32, 48),
                egui::Color32::from_rgb(225, 178, 35),
                egui::Color32::from_rgb(8, 2, 4),
                egui::Color32::from_rgb(195, 152, 28),
                egui::Color32::from_rgb(80, 210, 110), // status_ok
                egui::Color32::from_rgb(225, 185, 45), // status_warn
                egui::Color32::from_rgb(230, 75, 75),  // status_err
                egui::Color32::from_rgb(80, 160, 235), // status_info
            ),
            // -- NEON SYNTHWAVE -----------------------------------------------
            UiTheme::NeonSynthwave => (
                egui::Color32::from_rgb(5, 2, 12),
                egui::Color32::from_rgb(12, 6, 24),
                egui::Color32::from_rgb(22, 10, 42),
                egui::Color32::from_rgb(35, 15, 62),
                egui::Color32::from_rgb(50, 20, 85),
                egui::Color32::from_rgb(55, 8, 55),
                egui::Color32::from_rgb(255, 20, 175),
                egui::Color32::from_rgb(0, 245, 255),
                egui::Color32::from_rgb(248, 228, 255),
                egui::Color32::from_rgb(188, 148, 228),
                egui::Color32::from_rgb(105, 68, 148),
                egui::Color32::from_rgb(28, 12, 55),
                egui::Color32::from_rgb(75, 18, 98),
                egui::Color32::from_rgb(255, 20, 175),
                egui::Color32::from_rgb(3, 1, 8),
                egui::Color32::from_rgb(210, 15, 148),
                egui::Color32::from_rgb(50, 255, 120), // status_ok
                egui::Color32::from_rgb(255, 210, 30), // status_warn - amber not yellow
                egui::Color32::from_rgb(255, 50, 80),  // status_err
                egui::Color32::from_rgb(0, 245, 255),  // status_info
            ),
            // -- NEON TOXIC ---------------------------------------------------
            UiTheme::NeonToxic => (
                egui::Color32::from_rgb(1, 5, 1),
                egui::Color32::from_rgb(3, 12, 3),
                egui::Color32::from_rgb(5, 22, 5),
                egui::Color32::from_rgb(8, 35, 8),
                egui::Color32::from_rgb(12, 50, 12),
                egui::Color32::from_rgb(8, 48, 4),
                egui::Color32::from_rgb(40, 255, 40),
                egui::Color32::from_rgb(215, 255, 0),
                egui::Color32::from_rgb(210, 255, 210),
                egui::Color32::from_rgb(105, 195, 105),
                egui::Color32::from_rgb(42, 105, 42),
                egui::Color32::from_rgb(6, 28, 6),
                egui::Color32::from_rgb(18, 75, 18),
                egui::Color32::from_rgb(40, 255, 40),
                egui::Color32::from_rgb(1, 4, 1),
                egui::Color32::from_rgb(28, 210, 28),
                egui::Color32::from_rgb(40, 255, 40), // status_ok
                egui::Color32::from_rgb(215, 255, 0), // status_warn - electric yellow (fine on black)
                egui::Color32::from_rgb(255, 60, 60), // status_err
                egui::Color32::from_rgb(60, 200, 255), // status_info
            ),
            // -- NEON ULTRAVIOLET ---------------------------------------------
            UiTheme::NeonUltraviolet => (
                egui::Color32::from_rgb(4, 1, 10),
                egui::Color32::from_rgb(8, 3, 20),
                egui::Color32::from_rgb(16, 5, 38),
                egui::Color32::from_rgb(26, 8, 58),
                egui::Color32::from_rgb(38, 10, 80),
                egui::Color32::from_rgb(42, 4, 62),
                egui::Color32::from_rgb(175, 25, 255),
                egui::Color32::from_rgb(255, 25, 195),
                egui::Color32::from_rgb(242, 222, 255),
                egui::Color32::from_rgb(172, 128, 222),
                egui::Color32::from_rgb(88, 48, 135),
                egui::Color32::from_rgb(20, 6, 48),
                egui::Color32::from_rgb(62, 12, 108),
                egui::Color32::from_rgb(175, 25, 255),
                egui::Color32::from_rgb(2, 0, 7),
                egui::Color32::from_rgb(145, 18, 215),
                egui::Color32::from_rgb(60, 255, 140), // status_ok
                egui::Color32::from_rgb(255, 200, 30), // status_warn - amber not yellow
                egui::Color32::from_rgb(255, 40, 100), // status_err
                egui::Color32::from_rgb(255, 25, 195), // status_info
            ),
            // -- HIGH CONTRAST ------------------------------------------------
            UiTheme::HighContrast => (
                egui::Color32::from_rgb(0, 0, 0),
                egui::Color32::from_rgb(10, 10, 10),
                egui::Color32::from_rgb(25, 25, 25),
                egui::Color32::from_rgb(45, 45, 45),
                egui::Color32::from_rgb(65, 65, 65),
                egui::Color32::from_rgb(50, 45, 0),
                egui::Color32::from_rgb(255, 230, 0),
                egui::Color32::from_rgb(0, 200, 255),
                egui::Color32::from_rgb(255, 255, 255),
                egui::Color32::from_rgb(195, 195, 195),
                egui::Color32::from_rgb(115, 115, 115),
                egui::Color32::from_rgb(45, 45, 45),
                egui::Color32::from_rgb(95, 95, 95),
                egui::Color32::from_rgb(255, 230, 0),
                egui::Color32::from_rgb(0, 0, 0),
                egui::Color32::from_rgb(200, 180, 0),
                egui::Color32::from_rgb(50, 255, 50), // status_ok
                egui::Color32::from_rgb(255, 230, 0), // status_warn - yellow fine on black
                egui::Color32::from_rgb(255, 60, 60), // status_err
                egui::Color32::from_rgb(0, 200, 255), // status_info
            ),
            // -- CUSTOM -------------------------------------------------------
            UiTheme::Custom => {
                let c = &self.state.custom_theme;
                (
                    c.bg_darkest.to_egui(),
                    c.bg_panel.to_egui(),
                    c.bg_widget.to_egui(),
                    c.bg_hover.to_egui(),
                    c.bg_active.to_egui(),
                    c.bg_selected.to_egui(),
                    c.accent_primary.to_egui(),
                    c.accent_secondary.to_egui(),
                    c.text_primary.to_egui(),
                    c.text_secondary.to_egui(),
                    c.text_dim.to_egui(),
                    c.border_subtle.to_egui(),
                    c.border_normal.to_egui(),
                    c.border_bright.to_egui(),
                    c.topbar_bg.to_egui(),
                    c.topbar_border.to_egui(),
                    c.status_ok.to_egui(),
                    c.status_warn.to_egui(),
                    c.status_err.to_egui(),
                    c.status_info.to_egui(),
                )
            }
        };

        self.ctx.global_style_mut(|style| {
            let v = &mut style.visuals;
            // Light themes need dark_mode = false so egui renders
            // things like tooltips and popups with light backgrounds.
            v.dark_mode = matches!(
                theme_choice,
                UiTheme::BiotechDark
                    | UiTheme::Crimson
                    | UiTheme::NeonSynthwave
                    | UiTheme::NeonToxic
                    | UiTheme::NeonUltraviolet
                    | UiTheme::HighContrast
            ) || (theme_choice == UiTheme::Custom
                && self.state.custom_theme.dark_mode);

            v.window_fill = bg_panel;
            v.panel_fill = bg_panel;
            v.faint_bg_color = bg_widget;
            v.extreme_bg_color = bg_darkest;
            v.code_bg_color = bg_widget;

            v.window_stroke = egui::Stroke::new(1.0, border_normal);

            v.selection.bg_fill = bg_selected;
            v.selection.stroke = egui::Stroke::new(1.0, accent_primary);

            v.hyperlink_color = accent_secondary;

            v.widgets.noninteractive.bg_fill = bg_panel;
            v.widgets.noninteractive.weak_bg_fill = bg_widget;
            v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, border_subtle);
            v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, text_secondary);
            v.widgets.noninteractive.corner_radius = egui::CornerRadius::same(3);
            v.widgets.noninteractive.expansion = 0.0;

            v.widgets.inactive.bg_fill = bg_widget;
            v.widgets.inactive.weak_bg_fill = bg_widget;
            v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, border_normal);
            v.widgets.inactive.fg_stroke = egui::Stroke::new(1.5, text_secondary);
            v.widgets.inactive.corner_radius = egui::CornerRadius::same(3);
            v.widgets.inactive.expansion = 0.0;

            v.widgets.hovered.bg_fill = bg_hover;
            v.widgets.hovered.weak_bg_fill = bg_hover;
            v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, border_bright);
            v.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, text_primary);
            v.widgets.hovered.corner_radius = egui::CornerRadius::same(3);
            v.widgets.hovered.expansion = 1.0;

            v.widgets.active.bg_fill = bg_active;
            v.widgets.active.weak_bg_fill = bg_active;
            v.widgets.active.bg_stroke = egui::Stroke::new(1.5, accent_primary);
            v.widgets.active.fg_stroke = egui::Stroke::new(2.0, accent_primary);
            v.widgets.active.corner_radius = egui::CornerRadius::same(3);
            v.widgets.active.expansion = 1.0;

            v.widgets.open.bg_fill = bg_active;
            v.widgets.open.weak_bg_fill = bg_active;
            v.widgets.open.bg_stroke = egui::Stroke::new(1.0, accent_primary);
            v.widgets.open.fg_stroke = egui::Stroke::new(1.5, accent_primary);
            v.widgets.open.corner_radius = egui::CornerRadius::same(3);
            v.widgets.open.expansion = 0.0;

            v.override_text_color = Some(text_primary);
            v.slider_trailing_fill = true;

            v.window_shadow = egui::Shadow {
                offset: [0, 4],
                blur: 16,
                spread: 0,
                color: egui::Color32::from_black_alpha(120),
            };
            v.popup_shadow = egui::Shadow {
                offset: [0, 2],
                blur: 8,
                spread: 0,
                color: egui::Color32::from_black_alpha(100),
            };

            style.spacing.item_spacing = egui::vec2(6.0, 4.0);
            style.spacing.button_padding = egui::vec2(8.0, 4.0);
            style.spacing.indent = 14.0;
            style.spacing.menu_margin = egui::Margin::same(6);
            style.spacing.slider_width = 200.0;

            use egui::{FontFamily, FontId, TextStyle};
            style.text_styles.insert(
                TextStyle::Small,
                FontId::new(10.0, FontFamily::Proportional),
            );
            style
                .text_styles
                .insert(TextStyle::Body, FontId::new(12.0, FontFamily::Proportional));
            style.text_styles.insert(
                TextStyle::Button,
                FontId::new(12.0, FontFamily::Proportional),
            );
            style.text_styles.insert(
                TextStyle::Heading,
                FontId::new(13.0, FontFamily::Proportional),
            );
            style.text_styles.insert(
                TextStyle::Monospace,
                FontId::new(11.0, FontFamily::Monospace),
            );
        });

        // Store the full palette in the thread-local so all rendering code
        // can read it without re-deriving colors from the theme enum.
        ACTIVE_PALETTE.with(|cell| {
            *cell.borrow_mut() = ActivePalette {
                bg_darkest: bg_darkest,
                bg_panel: bg_panel,
                bg_widget: bg_widget,
                bg_hover: bg_hover,
                bg_active: bg_active,
                bg_selected: bg_selected,
                accent_primary: accent_primary,
                accent_secondary: accent_secondary,
                text_primary: text_primary,
                text_secondary: text_secondary,
                text_dim: text_dim,
                border_subtle: border_subtle,
                border_normal: border_normal,
                border_bright: border_bright,
                topbar_bg: topbar_bg,
                topbar_border: topbar_border,
                status_ok: status_ok,
                status_warn: status_warn,
                status_err: status_err,
                status_info: status_info,
                // Rail icon: always bright against the dark rail background.
                // Use the accent color tinted toward white for inactive icons,
                // and pure white for active (drawn on accent bg).
                rail_icon: {
                    // Blend accent_primary toward white at 60% for a bright but themed icon color.
                    let a = accent_primary;
                    egui::Color32::from_rgb(
                        ((a.r() as u16 * 60 + 255 * 40) / 100) as u8,
                        ((a.g() as u16 * 60 + 255 * 40) / 100) as u8,
                        ((a.b() as u16 * 60 + 255 * 40) / 100) as u8,
                    )
                },
                rail_icon_active: {
                    // Choose black or white based on accent luminance so the icon
                    // is always readable on the accent-coloured active button background.
                    // Bright accents (yellow, neon green) need black; dark accents need white.
                    let a = accent_primary;
                    let luminance = 0.2126 * (a.r() as f32 / 255.0)
                        + 0.7152 * (a.g() as f32 / 255.0)
                        + 0.0722 * (a.b() as f32 / 255.0);
                    if luminance > 0.45 {
                        egui::Color32::BLACK
                    } else {
                        egui::Color32::WHITE
                    }
                },
                theme: theme_choice,
            };
        });
    }

    /// Mark UI state as dirty (needs saving).
    pub fn mark_ui_state_dirty(&mut self) {
        self.ui_state_dirty = true;
    }

    /// Auto-save UI state if needed.
    pub fn auto_save_ui_state(&mut self) {
        const AUTO_SAVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

        if self.ui_state_dirty && self.save_timer.elapsed() >= AUTO_SAVE_INTERVAL {
            if let Err(e) = self.state.save() {
                log::warn!("Failed to auto-save UI state: {}", e);
            } else {
                log::debug!("Auto-saved UI state");
                self.ui_state_dirty = false;
                self.save_timer = std::time::Instant::now();
            }
        }
    }

    /// Save UI state immediately.
    pub fn save_ui_state(&mut self) {
        if let Err(e) = self.state.save() {
            log::warn!("Failed to save UI state: {}", e);
        } else {
            log::info!("Saved UI state");
            self.ui_state_dirty = false;
        }
    }

    /// End the egui frame and get the output.
    ///
    /// Call this after all UI rendering is complete.
    pub fn end_frame(
        &mut self,
        dock_manager: &mut crate::ui::dock::DockManager,
        genome: &mut crate::genome::Genome,
        editor_state: &mut crate::ui::panel_context::GenomeEditorState,
        scene_manager: &mut crate::scene::SceneManager,
        camera: &mut crate::ui::camera::CameraController,
        scene_request: &mut crate::ui::panel_context::SceneModeRequest,
        performance: &crate::ui::performance::PerformanceMetrics,
    ) -> egui::FullOutput {
        if let Some(gpu_scene) = scene_manager.gpu_scene() {
            if let Some(scan_frame) = gpu_scene.lineage_archive.last_scan_frame {
                if self.last_report_scan_frame != Some(scan_frame)
                    && gpu_scene
                        .lineage_archive
                        .nodes
                        .iter()
                        .any(|node| !node.telemetry_history.is_empty())
                {
                    if self.state.field_reports_enabled {
                        let _ = self
                            .field_report_director
                            .update(&gpu_scene.lineage_archive);
                    }
                    self.last_report_scan_frame = Some(scan_frame);
                }
            }
        }
        // Apply UI scale only when it changes
        let scale_changed = (self.last_scale - self.state.ui_scale).abs() > 0.001;
        if scale_changed {
            self.apply_ui_scale();
            self.last_scale = self.state.ui_scale;
        }

        // Apply biotech theme once at startup, and re-apply when theme changes
        let theme_changed = self.state.selected_theme != self.last_theme;
        // Also re-apply when custom theme colors change (compare via selected_theme == Custom)
        let custom_changed = self.state.selected_theme == crate::ui::types::UiTheme::Custom
            && self.state.custom_theme != self.last_custom_theme;
        if !self.theme_applied || theme_changed || custom_changed {
            self.apply_theme(self.state.selected_theme);
            self.last_theme = self.state.selected_theme;
            self.last_custom_theme = self.state.custom_theme.clone();
            self.theme_applied = true;
            // Reset scale tracking so spacing is re-applied on top of the new theme
            self.last_scale = -1.0;
        }

        // Auto-save UI state periodically
        self.auto_save_ui_state();

        // Show branded top bar
        let mut ui_state_copy = self.state.clone();

        // Pending mutations from inside egui closures (can't borrow self inside them).
        let mut pending_toasts: Vec<crate::ui::toast::Toast> = Vec::new();
        let mut pending_browser_open_load = false;
        let mut pending_browser_refresh = false;
        let mut pending_gif_capture: Option<std::path::PathBuf> = None;

        // Read the active theme palette once - used throughout this frame.
        let p = palette();
        let (
            tb_bg,
            tb_border,
            tb_accent,
            tb_text_primary,
            tb_text_secondary,
            tb_text_dim,
            tb_border_normal,
        ) = (
            p.topbar_bg,
            p.topbar_border,
            p.accent_primary,
            p.text_primary,
            p.text_secondary,
            p.text_dim,
            p.border_normal,
        );

        if !ui_state_copy.hide_ui {
            #[allow(deprecated)]
        egui::Panel::top("top_bar")
            .frame(
                egui::Frame::none()
                    .fill(tb_bg)
                    .inner_margin(egui::Margin { left: 12, right: 12, top: 6, bottom: 6 })
                    .stroke(egui::Stroke::new(1.0, tb_border)),
            )
            .show(&self.ctx, |ui| {
                // Top bar layout uses fixed pixel sizes throughout, so keep its
                // spacing unscaled regardless of ui_scale to avoid the bar's
                // height/widths drifting out of sync with those fixed values.
                if let Some(ref orig) = self.original_spacing {
                    *ui.spacing_mut() = orig.clone();
                }

                let bar_rect = ui.available_rect_before_wrap();
                ui.spacing_mut().item_spacing.x = 6.0;

                // Right side: fixed-width slot at the right edge.
                let right_width = 490.0_f32;
                let right_rect = egui::Rect::from_min_size(
                    egui::pos2(bar_rect.right() - right_width, bar_rect.top()),
                    egui::vec2(right_width, bar_rect.height()),
                );
                ui.allocate_ui_at_rect(right_rect, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;

                        // Mode-switch button - far right (first in right-to-left)
                        let (btn_text, btn_bg) = if ui_state_copy.current_mode == crate::ui::types::SimulationMode::Preview {
                            ("▶  LIVE SIMULATION", theme::BTN_LIVE_BG)
                        } else {
                            ("⬡  GENOME EDITOR", theme::BTN_EDITOR_BG)
                        };
                        if ui.add(egui::Button::new(egui::RichText::new(btn_text).strong().size(11.5).color(egui::Color32::WHITE))
                            .fill(btn_bg).stroke(egui::Stroke::new(1.0, egui::Color32::from_white_alpha(50)))
                            .corner_radius(egui::CornerRadius::same(3)).min_size(egui::vec2(150.0, 24.0))).clicked() {
                            ui_state_copy.mode_request = Some(if ui_state_copy.current_mode == crate::ui::types::SimulationMode::Preview {
                                crate::ui::types::SimulationMode::Gpu } else { crate::ui::types::SimulationMode::Preview });
                        }

                        ui.add_space(8.0); topbar_divider(ui); ui.add_space(8.0);

                        let tut_label = if ui_state_copy.tutorial.active { "🎓 Tutorial ●" } else { "🎓 Tutorial" };
                        let tut_color = if ui_state_copy.tutorial.active { tb_accent } else { tb_text_secondary };
                        let tut_enabled = ui_state_copy.current_mode == crate::ui::types::SimulationMode::Preview;
                        if ui.add_enabled(
                            tut_enabled,
                            egui::Button::new(egui::RichText::new(tut_label).size(11.5).color(if tut_enabled { tut_color } else { tb_text_dim }))
                                .frame(false)
                        ).clicked() {
                            if ui_state_copy.tutorial.active { ui_state_copy.tutorial.close(); } else { ui_state_copy.tutorial.start(); }
                        }

                        ui.add_space(8.0); topbar_divider(ui); ui.add_space(8.0);

                        if ui.add(egui::Button::new(egui::RichText::new("Help").size(11.5).color(tb_text_secondary)).frame(false)).clicked() {
                            let p = crate::ui::panel::Panel::Help;
                            if is_panel_open(dock_manager.current_tree(), &p) { close_panel(dock_manager.current_tree_mut(), &p); }
                            else { open_panel(dock_manager.current_tree_mut(), &p); }
                        }

                        ui.add_space(8.0); topbar_divider(ui); ui.add_space(8.0);

                        // Themes button - opens theme picker popup
                        let themes_resp = ui.add(
                            egui::Button::new(egui::RichText::new("🎨 Themes").size(11.5).color(tb_text_primary))
                                .frame(false)
                        );
                        egui::Popup::menu(&themes_resp)
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                            .show(|ui| {
                                ui.set_min_width(180.0);
                                show_themes_menu(ui, &mut ui_state_copy, dock_manager);
                            });

                        ui.add_space(8.0); topbar_divider(ui); ui.add_space(8.0);

                        // Panels button - frameless, same style as Tutorial/Help
                        let panels_resp = ui.add(
                            egui::Button::new(egui::RichText::new("Panels").size(11.5).color(tb_text_primary))
                                .frame(false)
                        );
                        egui::Popup::menu(&panels_resp)
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                            .show(|ui| {
                                ui.set_min_width(200.0);
                                show_windows_menu(
                                    ui,
                                    &mut ui_state_copy,
                                    dock_manager,
                                    scene_manager,
                                );
                            });                    });
                });

                // Left side: icon + genome controls.
                let left_rect = egui::Rect::from_min_max(
                    bar_rect.left_top(),
                    egui::pos2(bar_rect.right() - right_width, bar_rect.bottom()),
                );
                ui.allocate_ui_at_rect(left_rect, |ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;

                        let glyph_size = egui::vec2(22.0, 22.0);
                        if let Some(ref tex) = self.app_icon {
                            ui.add(egui::Image::new((tex.id(), glyph_size)).fit_to_exact_size(glyph_size));
                        } else {
                            let (r, _) = ui.allocate_exact_size(glyph_size, egui::Sense::hover());
                            draw_logo_glyph(ui.painter(), r, tb_accent);
                        }

                        // GPU mode: Save/Load sphere on the far left
                        if ui_state_copy.current_mode == crate::ui::types::SimulationMode::Gpu {
                            ui.add_space(6.0); topbar_divider(ui); ui.add_space(6.0);
                            ui.menu_button(egui::RichText::new("Save / Load").size(11.5).color(tb_text_primary), |ui| {
                                show_save_load_menu(ui, &mut ui_state_copy);
                            });
                        }

                        if ui_state_copy.current_mode == crate::ui::types::SimulationMode::Preview {
                            ui.add_space(6.0); topbar_divider(ui); ui.add_space(6.0);

                            let btn = |label: &str| egui::Button::new(egui::RichText::new(label).size(11.0).color(tb_text_primary))
                                .fill(p.bg_widget).stroke(egui::Stroke::new(1.0, tb_border_normal)).corner_radius(egui::CornerRadius::same(3));

                            if ui.add(btn("⬆ Save")).on_hover_text("Save genome to disk using the current name").clicked() {
                                let name = genome.name.trim().to_string();
                                let is_default = crate::genome::procedural_name::is_default_name(&name);                                if is_default {
                                    // Collect existing genome names to avoid clashes.
                                    let used: Vec<String> = crate::genome::Genome::list_genomes_dir()
                                        .into_iter()
                                        .filter_map(|p| p.file_stem()
                                            .and_then(|s| s.to_str())
                                            .map(|s| s.to_lowercase()))
                                        .collect();
                                    editor_state.show_name_dialog = true;
                                    editor_state.name_dialog_buffer = String::new();
                                    editor_state.name_dialog_seed = 0;
                                    editor_state.name_dialog_focused = false;
                                    editor_state.name_dialog_used_names = used;
                                } else {
                                    let path = crate::app_dirs::genomes_dir()
                                        .join(format!("{}.genome", crate::app_dirs::sanitize_filename(&name)));
                                    match genome.save_to_file(&path) {
                                        Ok(()) => {
                                            pending_toasts.push(crate::ui::toast::Toast::success(
                                                format!("✓ Saved — {}.genome", name)
                                            ));
                                            pending_browser_refresh = true;
                                            pending_gif_capture = Some(path.clone());
                                        }
                                        Err(e) => {
                                            pending_toasts.push(crate::ui::toast::Toast::error(
                                                format!("Save failed: {}", e)
                                            ));
                                        }
                                    }
                                }
                            }
                            if ui.add(btn("⬇ Load")).on_hover_text("Browse and load a saved genome").clicked() {
                                pending_browser_open_load = true;
                            }
                            if ui.add(btn("✦ New")).on_hover_text("Create a new blank genome").clicked() {
                                editor_state.confirm_new_genome = true;
                            }

                            ui.add_space(6.0); topbar_divider(ui); ui.add_space(6.0);

                            // Name field - red tint if error
                            let name_color = if editor_state.name_field_error {
                                egui::Color32::from_rgb(255, 100, 100)
                            } else {
                                tb_text_primary
                            };
                            let name_resp = ui.add(
                                egui::TextEdit::singleline(&mut genome.name)
                                    .desired_width(130.0)
                                    .hint_text("Genome Name")
                                    .font(egui::FontId::proportional(11.5))
                                    .text_color(name_color),
                            );
                            // Only check for overwrite when the field loses focus or on hover,
                            // not every frame - avoids blocking disk reads while typing.
                            if name_resp.hovered() && !name_resp.has_focus() {
                                let current_name = genome.name.trim().to_string();
                                if !current_name.is_empty() {
                                    let path = crate::app_dirs::genomes_dir()
                                        .join(format!("{}.genome", crate::app_dirs::sanitize_filename(&current_name)));
                                    if path.exists() {
                                        egui::show_tooltip_text(ui.ctx(), ui.layer_id(), egui::Id::new("name_overwrite_tip"),
                                            "⚠ A genome with this name already exists — saving will overwrite it");
                                    }
                                }
                            }
                            // Tick error timer
                            if editor_state.name_field_error {
                                editor_state.name_field_error_timer -= 1.0 / 60.0;
                                if editor_state.name_field_error_timer <= 0.0 {
                                    editor_state.name_field_error = false;
                                }
                            }
                        }
                    });
                });
            });

            // Show bottom status bar
            let cell_count = scene_manager
                .gpu_scene()
                .map(|s| s.current_cell_count)
                .unwrap_or_else(|| scene_manager.active_scene().cell_count() as u32);
            let cell_capacity = scene_manager.gpu_scene().map(|s| s.capacity()).unwrap_or(0);
            let sim_time = scene_manager.active_scene().current_time();
            let is_paused = scene_manager.active_scene().is_paused();
            let mem_used_gb = performance.memory_used() as f64 / (1024.0 * 1024.0 * 1024.0);
            let fps = performance.fps();
            let cpu_usage = performance.cpu_usage_total();

            #[allow(deprecated)]
            egui::Panel::bottom("status_bar")
                .frame(
                    egui::Frame::none()
                        .fill(p.topbar_bg)
                        .inner_margin(egui::Margin {
                            left: 12,
                            right: 12,
                            top: 4,
                            bottom: 4,
                        })
                        .stroke(egui::Stroke::new(1.0, p.border_subtle)),
                )
                .show(&self.ctx, |ui| {
                    // Status bar layout uses fixed pixel sizes throughout, so keep
                    // its spacing unscaled regardless of ui_scale (see top bar).
                    if let Some(ref orig) = self.original_spacing {
                        *ui.spacing_mut() = orig.clone();
                    }

                    let bar_rect = ui.max_rect();

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;

                        // Status indicator
                        let (status_label, status_color) = if ui_state_copy.current_mode
                            == crate::ui::types::SimulationMode::Preview
                        {
                            ("PREVIEW", p.accent_secondary)
                        } else if is_paused {
                            ("PAUSED", p.status_warn)
                        } else if cell_count > 0 {
                            ("RUNNING", p.status_ok)
                        } else {
                            ("IDLE", p.text_secondary)
                        };

                        status_field(ui, "SIMULATION STATUS", &|ui| {
                            let (dot_rect, _) =
                                ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                            let t = ui.input(|i| i.time) as f32;
                            let pulse = (t * 2.5).sin() * 0.4 + 0.6;
                            let dot_color = egui::Color32::from_rgba_premultiplied(
                                (status_color.r() as f32 * pulse) as u8,
                                (status_color.g() as f32 * pulse) as u8,
                                (status_color.b() as f32 * pulse) as u8,
                                255,
                            );
                            ui.painter()
                                .circle_filled(dot_rect.center(), 3.5, dot_color);
                            ui.label(
                                egui::RichText::new(status_label)
                                    .strong()
                                    .size(11.5)
                                    .color(status_color),
                            );
                            ui.ctx().request_repaint();
                        });

                        status_separator(ui);

                        // Sim Time
                        let total_seconds = sim_time as u64;
                        let h = total_seconds / 3600;
                        let m = (total_seconds % 3600) / 60;
                        let s = total_seconds % 60;
                        let time_str = if h > 0 {
                            format!("{}h {:02}m {:02}s", h, m, s)
                        } else {
                            format!("{:.1}s", sim_time)
                        };
                        status_field(ui, "SIM TIME", &|ui| {
                            ui.label(
                                egui::RichText::new(time_str.clone())
                                    .strong()
                                    .size(11.5)
                                    .color(p.text_primary),
                            );
                        });

                        status_separator(ui);

                        // Cells
                        let cells_str = if cell_capacity > 0 {
                            format!("{} / {}k", cell_count, cell_capacity / 1000)
                        } else {
                            format!("{}", cell_count)
                        };
                        status_field(ui, "CELLS", &|ui| {
                            ui.label(
                                egui::RichText::new(cells_str.clone())
                                    .strong()
                                    .size(11.5)
                                    .color(p.text_primary),
                            );
                        });

                        status_separator(ui);

                        // System load (CPU usage)
                        let cpu_color = if cpu_usage > 80.0 {
                            p.status_err
                        } else if cpu_usage > 50.0 {
                            p.status_warn
                        } else {
                            p.status_ok
                        };
                        status_field(ui, "SYSTEM LOAD", &|ui| {
                            ui.label(
                                egui::RichText::new(format!("{:.0}%", cpu_usage))
                                    .strong()
                                    .size(11.5)
                                    .color(cpu_color),
                            );
                            let (bar_rect, _) =
                                ui.allocate_exact_size(egui::vec2(40.0, 6.0), egui::Sense::hover());
                            ui.painter().rect_filled(bar_rect, 1.0, p.bg_darkest);
                            let fill_w = (cpu_usage / 100.0).clamp(0.0, 1.0) * bar_rect.width();
                            let fill_rect = egui::Rect::from_min_size(
                                bar_rect.min,
                                egui::vec2(fill_w, bar_rect.height()),
                            );
                            ui.painter().rect_filled(fill_rect, 1.0, cpu_color);
                        });

                        status_separator(ui);

                        // Process memory (RSS)
                        let mem_str = if mem_used_gb >= 1.0 {
                            format!("{:.1} GB", mem_used_gb)
                        } else {
                            format!("{:.0} MB", mem_used_gb * 1024.0)
                        };
                        let mem_color = if mem_used_gb >= 3.5 {
                            p.status_err
                        } else if mem_used_gb >= 2.0 {
                            p.status_warn
                        } else {
                            p.status_info
                        };
                        status_field(ui, "MEMORY", &|ui| {
                            ui.label(
                                egui::RichText::new(mem_str.clone())
                                    .strong()
                                    .size(11.5)
                                    .color(p.text_primary),
                            );
                            let (bar_rect, _) =
                                ui.allocate_exact_size(egui::vec2(40.0, 6.0), egui::Sense::hover());
                            ui.painter().rect_filled(bar_rect, 1.0, p.bg_darkest);
                            // Bar scaled to 4 GB reference
                            let mem_pct = (mem_used_gb / 4.0).clamp(0.0, 1.0) as f32;
                            let fill_w = mem_pct * bar_rect.width();
                            let fill_rect = egui::Rect::from_min_size(
                                bar_rect.min,
                                egui::vec2(fill_w, bar_rect.height()),
                            );
                            ui.painter().rect_filled(fill_rect, 1.0, mem_color);
                        });

                        // FPS on the far right
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let fps_color = if fps >= 50.0 {
                                p.status_ok
                            } else if fps >= 30.0 {
                                p.status_warn
                            } else {
                                p.status_err
                            };
                            status_field(ui, "FPS", &|ui| {
                                ui.label(
                                    egui::RichText::new(format!("{:.0}", fps))
                                        .strong()
                                        .size(13.0)
                                        .color(fps_color),
                                );
                            });
                        });
                    });

                    // Centered water/air rolling-average temperature readout with a
                    // F/C unit toggle between the two values. Drawn as an overlay so
                    // it stays centered on the bar regardless of how much space the
                    // left- and right-flowing status fields consume.
                    let (live_avg_water_temp_c, live_avg_air_temp_c) = scene_manager
                        .gpu_scene()
                        .and_then(|s| s.fluid_simulator.as_ref())
                        .map(|sim| (sim.avg_water_temp_c(), sim.avg_air_temp_c()))
                        .unwrap_or((0.0, 0.0));

                    let group_size = egui::vec2(220.0, bar_rect.height());
                    let group_rect = egui::Rect::from_center_size(
                        egui::pos2(bar_rect.center().x, bar_rect.center().y),
                        group_size,
                    );
                    ui.put(group_rect, |ui: &mut egui::Ui| -> egui::Response {
                        ui.with_layout(
                            egui::Layout::left_to_right(egui::Align::Center)
                                .with_main_align(egui::Align::Center),
                            |ui| {
                                ui.spacing_mut().item_spacing.x = 8.0;

                                let fahrenheit = editor_state.temp_display_fahrenheit;
                                let unit_label = if fahrenheit { "°F" } else { "°C" };
                                let format_temp = |celsius: f32| -> String {
                                    if fahrenheit {
                                        format!("{:.0}{}", celsius * 9.0 / 5.0 + 32.0, unit_label)
                                    } else {
                                        format!("{:.0}{}", celsius, unit_label)
                                    }
                                };

                                // Fixed-width containers with label AND value each
                                // center-aligned (status_field left-aligns the value
                                // under its label, which made the group read
                                // off-center), so the toggle button sits exactly
                                // between the two numbers.
                                let field_width = 84.0;
                                ui.allocate_ui_with_layout(
                                    egui::vec2(field_width, ui.available_height()),
                                    egui::Layout::top_down(egui::Align::Center),
                                    |ui| {
                                        ui.spacing_mut().item_spacing.y = 1.0;
                                        ui.label(
                                            egui::RichText::new("AVG WATER TEMP")
                                                .size(8.5)
                                                .color(theme::TEXT_DIM),
                                        );
                                        ui.label(
                                            egui::RichText::new(format_temp(live_avg_water_temp_c))
                                                .strong()
                                                .size(11.5)
                                                .color(p.status_info),
                                        );
                                    },
                                );

                                if ui
                                    .add(
                                        egui::Button::new(
                                            egui::RichText::new(if fahrenheit { "F" } else { "C" })
                                                .strong()
                                                .size(11.0),
                                        )
                                        .small(),
                                    )
                                    .on_hover_text("Toggle temperature units (°F / °C)")
                                    .clicked()
                                {
                                    editor_state.temp_display_fahrenheit = !fahrenheit;
                                }

                                ui.allocate_ui_with_layout(
                                    egui::vec2(field_width, ui.available_height()),
                                    egui::Layout::top_down(egui::Align::Center),
                                    |ui| {
                                        ui.spacing_mut().item_spacing.y = 1.0;
                                        ui.label(
                                            egui::RichText::new("AVG AIR TEMP")
                                                .size(8.5)
                                                .color(theme::TEXT_DIM),
                                        );
                                        ui.label(
                                            egui::RichText::new(format_temp(live_avg_air_temp_c))
                                                .strong()
                                                .size(11.5)
                                                .color(p.text_primary),
                                        );
                                    },
                                );
                            },
                        )
                        .response
                    });
                });
        } // end if !ui_state_copy.hide_ui (top bar and status bar)

        // Apply pending mutations collected from inside egui closures.
        self.toasts.extend(pending_toasts);
        if pending_browser_open_load {
            self.genome_browser.open_load();
        }
        if pending_browser_refresh {
            self.genome_browser.needs_refresh = true;
            self.genome_browser.force_full_reload = true;
        }
        if let Some(ref save_path) = pending_gif_capture {
            if editor_state.gif_capture.is_none() {
                editor_state.request_gif_capture = true;
                editor_state.gif_capture_save_path = Some(save_path.clone());
                crate::ui::toast::upsert_progress_toast(&mut self.toasts, "Preparing GIF…", 0.0);
            }
        }

        // Side rail always renders - it contains the hide_ui toggle itself,
        // so hiding it would trap the user with no way to restore the UI.
        #[allow(deprecated)]
        egui::Panel::left("side_rail")
            .resizable(false)
            .exact_width(40.0)
            .frame(
                egui::Frame::none()
                    .fill(p.topbar_bg)
                    .inner_margin(egui::Margin {
                        left: 4,
                        right: 4,
                        top: 8,
                        bottom: 8,
                    })
                    .stroke(egui::Stroke::new(1.0, p.border_subtle)),
            )
            .show(&self.ctx, |ui| {
                render_side_rail(ui, &mut ui_state_copy, editor_state, dock_manager);
            });

        {
            // Show dock area in remaining space
            let mut style = egui_dock::Style::from_egui(self.ctx.global_style().as_ref());
            style.separator.extra = 75.0;

            const DOCK_GUTTER: i8 = 5;
            style.dock_area_padding = Some(egui::Margin::same(DOCK_GUTTER));
            style.main_surface_border_stroke = egui::Stroke::NONE;

            // Tab bar
            style.tab_bar.bg_fill = p.bg_panel;
            style.tab_bar.height = 26.0;
            style.tab_bar.hline_color = p.border_subtle;
            style.tab_bar.corner_radius = egui::CornerRadius::ZERO;
            style.tab_bar.fill_tab_bar = false;
            style.tab_bar.inner_margin = egui::Margin::symmetric(2, 0);

            style.tab.hline_below_active_tab_name = false;
            style.tab.spacing = 4.0;

            let tab_fill = p.bg_panel;
            // Active tab gets a slightly lighter background tint so it stands out
            // from inactive tabs even when the glow is subtle.
            let active_tab_fill = egui::Color32::from_rgba_unmultiplied(
                p.accent_primary.r().saturating_add(8),
                p.accent_primary.g().saturating_add(8),
                p.accent_primary.b().saturating_add(8),
                18,
            );
            let active_tab_bg = egui::Color32::from_rgba_unmultiplied(
                (p.bg_panel.r() as u16 + active_tab_fill.r() as u16).min(255) as u8,
                (p.bg_panel.g() as u16 + active_tab_fill.g() as u16).min(255) as u8,
                (p.bg_panel.b() as u16 + active_tab_fill.b() as u16).min(255) as u8,
                255,
            );
            // Glow oval: full accent color at high opacity so it reads on any theme.
            let active_glow = egui::Color32::from_rgba_unmultiplied(
                p.accent_primary.r(),
                p.accent_primary.g(),
                p.accent_primary.b(),
                200,
            );
            let hover_glow = egui::Color32::from_rgba_unmultiplied(
                p.accent_primary.r(),
                p.accent_primary.g(),
                p.accent_primary.b(),
                80,
            );
            const GLOW_ASPECT: f32 = 0.28;

            // Active
            style.tab.active.bg_fill = active_tab_bg;
            style.tab.active.text_color = p.accent_primary;
            style.tab.active.outline_color = egui::Color32::TRANSPARENT;
            style.tab.active.corner_radius = egui::CornerRadius::same(3);
            style.tab.active.glow_color = active_glow;
            style.tab.active.glow_radius_factor = 0.85;
            style.tab.active.glow_aspect = GLOW_ASPECT;

            // Focused
            style.tab.focused.bg_fill = active_tab_bg;
            style.tab.focused.text_color = p.accent_primary;
            style.tab.focused.outline_color = egui::Color32::TRANSPARENT;
            style.tab.focused.corner_radius = egui::CornerRadius::same(3);
            style.tab.focused.glow_color = active_glow;
            style.tab.focused.glow_radius_factor = 0.85;
            style.tab.focused.glow_aspect = GLOW_ASPECT;

            // Inactive
            style.tab.inactive.bg_fill = tab_fill;
            style.tab.inactive.text_color = p.text_dim;
            style.tab.inactive.outline_color = egui::Color32::TRANSPARENT;
            style.tab.inactive.corner_radius = egui::CornerRadius::same(3);
            style.tab.inactive.glow_color = egui::Color32::TRANSPARENT;

            // Hovered
            style.tab.hovered.bg_fill = tab_fill;
            style.tab.hovered.text_color = p.text_primary;
            style.tab.hovered.outline_color = egui::Color32::TRANSPARENT;
            style.tab.hovered.corner_radius = egui::CornerRadius::same(3);
            style.tab.hovered.glow_color = hover_glow;
            style.tab.hovered.glow_radius_factor = 0.85;
            style.tab.hovered.glow_aspect = GLOW_ASPECT;

            // KB-focus variants
            style.tab.active_with_kb_focus.bg_fill = active_tab_bg;
            style.tab.active_with_kb_focus.text_color = p.accent_primary;
            style.tab.active_with_kb_focus.outline_color = egui::Color32::TRANSPARENT;
            style.tab.active_with_kb_focus.corner_radius = egui::CornerRadius::same(3);
            style.tab.active_with_kb_focus.glow_color = active_glow;
            style.tab.active_with_kb_focus.glow_radius_factor = 0.85;
            style.tab.active_with_kb_focus.glow_aspect = GLOW_ASPECT;

            style.tab.inactive_with_kb_focus.bg_fill = tab_fill;
            style.tab.inactive_with_kb_focus.text_color = p.text_secondary;
            style.tab.inactive_with_kb_focus.outline_color = egui::Color32::TRANSPARENT;
            style.tab.inactive_with_kb_focus.corner_radius = egui::CornerRadius::same(3);
            style.tab.inactive_with_kb_focus.glow_color = egui::Color32::TRANSPARENT;

            style.tab.focused_with_kb_focus.bg_fill = active_tab_bg;
            style.tab.focused_with_kb_focus.text_color = p.accent_primary;
            style.tab.focused_with_kb_focus.outline_color = egui::Color32::TRANSPARENT;
            style.tab.focused_with_kb_focus.corner_radius = egui::CornerRadius::same(3);
            style.tab.focused_with_kb_focus.glow_color = active_glow;
            style.tab.focused_with_kb_focus.glow_radius_factor = 0.85;
            style.tab.focused_with_kb_focus.glow_aspect = GLOW_ASPECT;

            // Tab bar background matches tab fill
            style.tab_bar.bg_fill = tab_fill;
            style.tab_bar.hline_color = egui::Color32::TRANSPARENT;

            // Tab body
            style.tab.tab_body.bg_fill = tab_fill;
            style.tab.tab_body.stroke = egui::Stroke::NONE;
            style.tab.tab_body.corner_radius = egui::CornerRadius::ZERO;
            style.tab.tab_body.inner_margin = egui::Margin {
                left: 6,
                right: 4,
                top: 4,
                bottom: 4,
            };

            // Buttons
            style.buttons.close_tab_color = p.text_dim;
            style.buttons.close_tab_active_color = p.status_err;
            style.buttons.close_tab_bg_fill = egui::Color32::TRANSPARENT;
            style.buttons.add_tab_color = p.text_secondary;
            style.buttons.add_tab_active_color = p.accent_primary;
            style.buttons.add_tab_bg_fill = egui::Color32::TRANSPARENT;
            style.buttons.add_tab_border_color = p.border_subtle;

            // Separator
            style.separator.color_idle = p.bg_darkest;
            style.separator.color_hovered = p.accent_primary;
            style.separator.color_dragged = p.accent_primary;
            style.separator.width = DOCK_GUTTER as f32;

            // Drop overlay
            style.overlay.selection_color = p.accent_primary.linear_multiply(0.35);
            style.overlay.button_color = p.bg_widget;
            style.overlay.button_border_stroke = egui::Stroke::new(1.0, p.accent_primary);

            // Apply lock settings to hide tab bar height if locked
            if ui_state_copy.lock_tab_bar {
                style.tab_bar.height = 0.0;
            }

            // When hide_ui is active, make the dock chrome invisible so only the
            // viewport content shows. The dock layout is preserved - no panels are
            // closed or moved.
            if ui_state_copy.hide_ui {
                let transparent = egui::Color32::TRANSPARENT;
                style.tab_bar.bg_fill = transparent;
                style.tab_bar.hline_color = transparent;
                style.tab_bar.height = 0.0;
                style.tab.active.bg_fill = transparent;
                style.tab.active.text_color = transparent;
                style.tab.active.outline_color = transparent;
                style.tab.active.glow_color = transparent;
                style.tab.focused.bg_fill = transparent;
                style.tab.focused.text_color = transparent;
                style.tab.focused.outline_color = transparent;
                style.tab.focused.glow_color = transparent;
                style.tab.inactive.bg_fill = transparent;
                style.tab.inactive.text_color = transparent;
                style.tab.inactive.outline_color = transparent;
                style.tab.inactive.glow_color = transparent;
                style.tab.hovered.bg_fill = transparent;
                style.tab.hovered.text_color = transparent;
                style.tab.hovered.outline_color = transparent;
                style.tab.hovered.glow_color = transparent;
                style.tab.active_with_kb_focus.bg_fill = transparent;
                style.tab.active_with_kb_focus.text_color = transparent;
                style.tab.active_with_kb_focus.outline_color = transparent;
                style.tab.active_with_kb_focus.glow_color = transparent;
                style.tab.inactive_with_kb_focus.bg_fill = transparent;
                style.tab.inactive_with_kb_focus.text_color = transparent;
                style.tab.inactive_with_kb_focus.outline_color = transparent;
                style.tab.inactive_with_kb_focus.glow_color = transparent;
                style.tab.focused_with_kb_focus.bg_fill = transparent;
                style.tab.focused_with_kb_focus.text_color = transparent;
                style.tab.focused_with_kb_focus.outline_color = transparent;
                style.tab.focused_with_kb_focus.glow_color = transparent;
                style.tab.tab_body.bg_fill = transparent;
                style.tab.tab_body.stroke = egui::Stroke::NONE;
                style.separator.color_idle = transparent;
                style.separator.color_hovered = transparent;
                style.separator.color_dragged = transparent;
                style.separator.width = 0.0;
                style.overlay.selection_color = transparent;
                style.overlay.button_color = transparent;
                style.overlay.button_border_stroke = egui::Stroke::NONE;
                style.dock_area_padding = Some(egui::Margin::same(0));
                style.buttons.close_tab_color = transparent;
                style.buttons.add_tab_color = transparent;
            }

            // Create panel context for PanelTabViewer
            let current_mode = ui_state_copy.current_mode;
            let hide_ui = ui_state_copy.hide_ui;

            // Reset headless mode each frame so it's only active when the
            // lineage panel is the visible tab and re-enables automatically
            // when switching away without closing.
            ui_state_copy.gpu_headless_mode = false;

            // Show dock area (scoped to release borrows after)
            {
                let mut panel_context = crate::ui::panel_context::PanelContext::new(
                    genome,
                    editor_state,
                    scene_manager,
                    camera,
                    scene_request,
                    current_mode,
                    performance,
                    &mut self.field_report_director,
                    ui_state_copy.hide_ui,
                );

                let mut dock_area = egui_dock::DockArea::new(dock_manager.current_tree_mut())
                    .style(style)
                    .show_leaf_collapse_buttons(false)
                    .show_leaf_close_all_buttons(false)
                    .draggable_tabs(true)
                    .window_bounds(self.ctx.content_rect());

                // Apply lock settings for tabs and close buttons
                if ui_state_copy.lock_tabs {
                    dock_area = dock_area
                        .show_tab_name_on_hover(false)
                        .draggable_tabs(false);
                }

                if ui_state_copy.lock_close_buttons {
                    dock_area = dock_area.show_close_buttons(false);
                }

                let mut tab_viewer = crate::ui::tab_viewer::PanelTabViewer::new(
                    &mut ui_state_copy,
                    &mut panel_context,
                    &mut self.viewport_rect,
                );

                // Host the dock inside a *transparent* CentralPanel so the 3D
                // scene rendered behind egui shows through any tabs that opt out
                // of background clearing (the Viewport).
                //
                // We then paint a black border around the dock's inset region
                // (the area defined by `dock_area_padding`) by hand - this gives
                // us the black gutter from the concept art around the panel
                // cluster without obscuring the viewport.
                const GUTTER: f32 = DOCK_GUTTER as f32;
                #[allow(deprecated)]
                egui::CentralPanel::default()
                    .frame(
                        egui::Frame::none()
                            .fill(egui::Color32::TRANSPARENT)
                            .inner_margin(egui::Margin::ZERO),
                    )
                    .show(&self.ctx, |ui| {
                        let area = ui.max_rect();
                        let gutter_color = p.bg_darkest;

                        // Render the dock first so its panel/tab body fills paint
                        // their backgrounds, then overlay the black gutter strips
                        // on top. Painting the gutters last guarantees they cover
                        // every pixel of the outer 8px frame regardless of how
                        // the dock has clipped or rounded its contents.
                        dock_area.show_inside(ui, &mut tab_viewer);

                        // Copy the viewport rect out of the tab_viewer before
                        // the gutter/bracket paint so we have a clean local copy.
                        let vp_rect = *tab_viewer.viewport_rect;

                        let painter = ui.painter();
                        // Top gutter strip
                        painter.rect_filled(
                            egui::Rect::from_min_max(
                                area.left_top(),
                                egui::pos2(area.right(), area.top() + GUTTER),
                            ),
                            0.0,
                            gutter_color,
                        );
                        // Bottom gutter strip
                        painter.rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(area.left(), area.bottom() - GUTTER),
                                area.right_bottom(),
                            ),
                            0.0,
                            gutter_color,
                        );
                        // Left gutter strip
                        painter.rect_filled(
                            egui::Rect::from_min_max(
                                area.left_top(),
                                egui::pos2(area.left() + GUTTER, area.bottom()),
                            ),
                            0.0,
                            gutter_color,
                        );
                        // Right gutter strip
                        painter.rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(area.right() - GUTTER, area.top()),
                                area.right_bottom(),
                            ),
                            0.0,
                            gutter_color,
                        );

                        // Paint the corner brackets on the GPU viewport AFTER the
                        // dock has rendered AND the gutters are down so they sit
                        // on top of everything. Use a foreground layer painter
                        // so the brackets are never clipped by any panel's clip rect.
                        // Fall back to the previous frame's rect on the first frame
                        // after switching to GPU mode (before the Viewport tab renders).
                        if current_mode == crate::ui::types::SimulationMode::Gpu && !hide_ui {
                            // Only paint brackets when we have a rect from THIS frame's
                            // Viewport tab render. Never fall back to a stale rect from
                            // a different scene - one missed frame is fine.
                            if let Some(viewport_rect) = vp_rect {
                                let bracket_painter = ui.ctx().layer_painter(egui::LayerId::new(
                                    egui::Order::Middle,
                                    egui::Id::new("viewport_brackets"),
                                ));
                                paint_viewport_brackets(&bracket_painter, viewport_rect);
                            }
                        }

                        // Paint rounded corners on the Preview viewport - black
                        // quarter-circle fills at each corner make the viewport
                        // appear to have rounded corners against the black gutter.
                        // Hidden when UI is hidden so the full viewport is unobstructed.
                        if current_mode == crate::ui::types::SimulationMode::Preview && !hide_ui {
                            if let Some(viewport_rect) = vp_rect {
                                let panel_rect = viewport_rect.expand2(egui::vec2(6.0, 4.0));
                                paint_viewport_rounded_corners(
                                    ui.painter(),
                                    panel_rect,
                                    40.0,
                                    gutter_color,
                                );
                            }
                        }
                    });
            }
        }

        // -- Genome browser window ---------------------------------------------
        // Floats above the dock. Renders in Preview mode only.
        if ui_state_copy.current_mode == crate::ui::types::SimulationMode::Preview {
            // Process open requests from panels/buttons
            if editor_state.open_genome_browser_save {
                editor_state.open_genome_browser_save = false;
                let name = genome.name.trim().to_string();
                if !name.is_empty() {
                    let path = crate::app_dirs::genomes_dir().join(format!(
                        "{}.genome",
                        crate::app_dirs::sanitize_filename(&name)
                    ));
                    match genome.save_to_file(&path) {
                        Ok(()) => {
                            self.toasts.push(crate::ui::toast::Toast::success(format!(
                                "✓ Saved — {}.genome",
                                name
                            )));
                            self.genome_browser.needs_refresh = true;
                            self.genome_browser.force_full_reload = true;
                            if editor_state.gif_capture.is_none() {
                                editor_state.request_gif_capture = true;
                                editor_state.gif_capture_save_path = Some(path.clone());
                                crate::ui::toast::upsert_progress_toast(
                                    &mut self.toasts,
                                    "Preparing GIF…",
                                    0.0,
                                );
                            }
                        }
                        Err(e) => {
                            self.toasts.push(crate::ui::toast::Toast::error(format!(
                                "Save failed: {}",
                                e
                            )));
                        }
                    }
                }
            }
            if editor_state.open_genome_browser_load {
                editor_state.open_genome_browser_load = false;
                self.genome_browser.open_load();
            }

            let dt_for_browser = 1.0 / 60.0; // approximate; good enough for animation
            crate::ui::genome_browser::render_genome_browser(
                &self.ctx,
                &mut self.genome_browser,
                genome,
                editor_state,
                dt_for_browser,
            );
        }

        // -- Panel close confirmation dialog -----------------------------------
        // Shown when the user clicks the x on any panel tab.
        // Keeps the panel open until confirmed so accidental clicks don't lose layout.
        if let Some(panel_to_close) = ui_state_copy.pending_close_panel {
            let p = palette();
            let mut confirmed = false;
            let mut cancelled = false;

            let win_frame = egui::Frame::new()
                .fill(p.bg_darkest)
                .stroke(egui::Stroke::new(1.0, p.border_bright))
                .corner_radius(egui::CornerRadius::same(8))
                .inner_margin(egui::Margin::same(20));

            egui::Window::new("Close Panel")
                .id(egui::Id::new("panel_close_confirm"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .frame(win_frame)
                .title_bar(false)
                .show(&self.ctx, |ui| {
                    ui.set_width(260.0);
                    ui.vertical_centered(|ui| {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new("Close Panel?")
                                .size(15.0)
                                .color(p.text_primary)
                                .strong(),
                        );
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "Close \"{}\"?",
                                panel_to_close.display_name()
                            ))
                            .size(12.0)
                            .color(p.text_secondary),
                        );
                        ui.label(
                            egui::RichText::new("You can reopen it from the Windows menu.")
                                .size(11.0)
                                .color(p.text_dim),
                        );
                        ui.add_space(14.0);
                        ui.horizontal(|ui| {
                            let btn_w = 110.0;
                            let spacing = ui.available_width() - btn_w * 2.0;
                            if spacing > 0.0 {
                                ui.add_space(spacing * 0.5);
                            }
                            if ui
                                .add_sized(
                                    [btn_w, 30.0],
                                    egui::Button::new(
                                        egui::RichText::new("Close")
                                            .color(egui::Color32::from_rgb(220, 80, 80)),
                                    )
                                    .fill(egui::Color32::from_rgba_unmultiplied(180, 40, 40, 30))
                                    .stroke(egui::Stroke::new(
                                        1.0,
                                        egui::Color32::from_rgb(140, 40, 40),
                                    )),
                                )
                                .clicked()
                            {
                                confirmed = true;
                            }
                            ui.add_space(8.0);
                            if ui
                                .add_sized(
                                    [btn_w, 30.0],
                                    egui::Button::new(
                                        egui::RichText::new("Keep Open").color(p.accent_primary),
                                    )
                                    .fill(egui::Color32::from_rgba_unmultiplied(
                                        p.accent_primary.r(),
                                        p.accent_primary.g(),
                                        p.accent_primary.b(),
                                        25,
                                    ))
                                    .stroke(egui::Stroke::new(1.0, p.border_bright)),
                                )
                                .clicked()
                            {
                                cancelled = true;
                            }
                        });
                        ui.add_space(4.0);
                    });
                });

            if confirmed {
                // Actually remove the tab from the dock tree.
                let panel_name = format!("{:?}", panel_to_close);
                ui_state_copy.set_panel_visible(&panel_name, false);
                if let Some(loc) = dock_manager.current_tree().find_tab(&panel_to_close) {
                    dock_manager.current_tree_mut().remove_tab(loc);
                }
                ui_state_copy.pending_close_panel = None;
            } else if cancelled {
                ui_state_copy.pending_close_panel = None;
            }
        }

        // -- New Genome confirmation dialog ------------------------------------
        if editor_state.confirm_new_genome {
            let p = palette();
            let mut do_new = false;
            let mut cancel_new = false;

            let win_frame = egui::Frame::new()
                .fill(p.bg_darkest)
                .stroke(egui::Stroke::new(1.5, p.accent_primary))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(20));

            egui::Window::new("confirm_new_genome")
                .frame(win_frame)
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(&self.ctx, |ui| {
                    ui.set_min_width(300.0);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new("New Genome")
                                .size(14.0)
                                .strong()
                                .color(p.text_primary),
                        );
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("Unsaved changes will be lost.")
                                .size(12.0)
                                .color(p.text_primary),
                        );
                        ui.label(
                            egui::RichText::new("Create a new blank genome?")
                                .size(11.0)
                                .color(p.text_dim),
                        );
                        ui.add_space(14.0);
                        ui.horizontal(|ui| {
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new("Cancel")
                                            .size(12.0)
                                            .color(p.text_secondary),
                                    )
                                    .fill(p.bg_widget)
                                    .stroke(egui::Stroke::new(1.0, p.border_normal))
                                    .min_size(egui::Vec2::new(90.0, 28.0))
                                    .corner_radius(egui::CornerRadius::same(4)),
                                )
                                .clicked()
                            {
                                cancel_new = true;
                            }
                            ui.add_space(8.0);
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new("✦ New Genome")
                                            .size(12.0)
                                            .strong()
                                            .color(p.bg_darkest),
                                    )
                                    .fill(p.accent_primary)
                                    .stroke(egui::Stroke::new(1.0, p.accent_primary))
                                    .min_size(egui::Vec2::new(110.0, 28.0))
                                    .corner_radius(egui::CornerRadius::same(4)),
                                )
                                .clicked()
                            {
                                do_new = true;
                            }
                        });
                    });
                });

            if do_new {
                *genome = crate::genome::Genome::new_with_random_colors();
                editor_state.selected_mode_index = 0;
                editor_state.selected_mode_indices = vec![0];
                editor_state.genome_just_loaded = true;
                editor_state.confirm_new_genome = false;
                self.toasts.push(crate::ui::toast::Toast::info(
                    "New genome created".to_string(),
                ));
            }
            if cancel_new {
                editor_state.confirm_new_genome = false;
            }
        }

        // -- Name dialog (shown when saving with an empty name) ----------------
        if editor_state.show_name_dialog {
            let p = palette();
            // Re-read from disk every frame so deleted genomes free up their names immediately.
            let used: Vec<String> = crate::genome::Genome::list_genomes_dir()
                .into_iter()
                .filter_map(|p| {
                    p.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_lowercase())
                })
                .collect();
            let procedural = crate::genome::procedural_name::generate_unique(
                genome,
                editor_state.name_dialog_seed,
                &used,
            );
            let mut do_save_name: Option<String> = None;
            let mut cancel = false;
            let mut regenerate = false;

            let win_frame = egui::Frame::new()
                .fill(p.bg_darkest)
                .stroke(egui::Stroke::new(1.5, p.accent_primary))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(20));

            egui::Window::new("name_genome_dialog")
                .frame(win_frame)
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(&self.ctx, |ui| {
                    ui.set_min_width(340.0);

                    ui.label(
                        egui::RichText::new("Name Your Genome")
                            .size(14.0)
                            .strong()
                            .color(p.text_primary),
                    );
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("Enter a custom name or use the suggested one.")
                            .size(11.0)
                            .color(p.text_dim),
                    );
                    ui.add_space(12.0);

                    // Procedural suggestion
                    egui::Frame::new()
                        .fill(p.bg_widget)
                        .stroke(egui::Stroke::new(1.0, p.border_subtle))
                        .corner_radius(egui::CornerRadius::same(4))
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("Suggested:")
                                        .size(10.0)
                                        .color(p.text_dim),
                                );
                                ui.add_space(4.0);
                                ui.label(
                                    egui::RichText::new(&procedural)
                                        .size(12.0)
                                        .color(p.accent_primary),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .add(
                                                egui::Button::new(
                                                    egui::RichText::new("🔀")
                                                        .size(11.0)
                                                        .color(p.text_secondary),
                                                )
                                                .fill(egui::Color32::TRANSPARENT)
                                                .stroke(egui::Stroke::NONE)
                                                .min_size(egui::Vec2::new(22.0, 22.0)),
                                            )
                                            .on_hover_text("Generate a different name")
                                            .clicked()
                                        {
                                            regenerate = true;
                                        }
                                    },
                                );
                            });
                        });
                    ui.add_space(8.0);

                    // Custom name field
                    ui.label(
                        egui::RichText::new("Custom name:")
                            .size(10.0)
                            .color(p.text_dim),
                    );
                    ui.add_space(2.0);
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut editor_state.name_dialog_buffer)
                            .desired_width(300.0)
                            .hint_text("Leave blank to use suggested name…")
                            .font(egui::FontId::proportional(12.0))
                            .text_color(p.text_primary),
                    );
                    // Auto-focus the text field on the first frame the dialog opens.
                    if !editor_state.name_dialog_focused {
                        resp.request_focus();
                        editor_state.name_dialog_focused = true;
                    }

                    // Enter key confirms
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        let chosen = if editor_state.name_dialog_buffer.trim().is_empty() {
                            procedural.clone()
                        } else {
                            editor_state.name_dialog_buffer.trim().to_string()
                        };
                        do_save_name = Some(chosen);
                    }

                    ui.add_space(14.0);
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("Cancel")
                                        .size(12.0)
                                        .color(p.text_secondary),
                                )
                                .fill(p.bg_widget)
                                .stroke(egui::Stroke::new(1.0, p.border_normal))
                                .min_size(egui::Vec2::new(80.0, 28.0))
                                .corner_radius(egui::CornerRadius::same(4)),
                            )
                            .clicked()
                        {
                            cancel = true;
                        }
                        ui.add_space(8.0);
                        // Save with custom name (or suggested if blank)
                        let custom = editor_state.name_dialog_buffer.trim().to_string();
                        let save_label = if custom.is_empty() {
                            "✨ Use Suggested".to_string()
                        } else {
                            format!(
                                "⬆ Save as \"{}\"",
                                if custom.len() > 18 {
                                    format!("{}…", &custom[..18])
                                } else {
                                    custom.clone()
                                }
                            )
                        };
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(&save_label)
                                        .size(12.0)
                                        .strong()
                                        .color(p.bg_darkest),
                                )
                                .fill(p.accent_primary)
                                .stroke(egui::Stroke::new(1.0, p.accent_primary))
                                .min_size(egui::Vec2::new(120.0, 28.0))
                                .corner_radius(egui::CornerRadius::same(4)),
                            )
                            .clicked()
                        {
                            let chosen = if custom.is_empty() {
                                procedural.clone()
                            } else {
                                custom
                            };
                            do_save_name = Some(chosen);
                        }
                    });
                });

            if regenerate {
                editor_state.name_dialog_seed = editor_state.name_dialog_seed.wrapping_add(1);
            }
            if let Some(name) = do_save_name {
                editor_state.show_name_dialog = false;
                genome.name = name.clone();
                let path = crate::app_dirs::genomes_dir().join(format!(
                    "{}.genome",
                    crate::app_dirs::sanitize_filename(&name)
                ));
                match genome.save_to_file(&path) {
                    Ok(()) => {
                        self.toasts.push(crate::ui::toast::Toast::success(format!(
                            "✓ Saved — {}.genome",
                            name
                        )));
                        self.genome_browser.needs_refresh = true;
                        self.genome_browser.force_full_reload = true;
                        if editor_state.gif_capture.is_none() {
                            crate::ui::toast::upsert_progress_toast(
                                &mut self.toasts,
                                "Preparing GIF…",
                                0.0,
                            );
                            editor_state.request_gif_capture = true;
                            editor_state.gif_capture_save_path = Some(path.clone());
                        }
                    }
                    Err(e) => {
                        self.toasts.push(crate::ui::toast::Toast::error(format!(
                            "Save failed: {}",
                            e
                        )));
                    }
                }
            }
            if cancel {
                editor_state.show_name_dialog = false;
                editor_state.name_dialog_focused = false;
            }
        }

        // Handle mode graph panel toggle request
        if editor_state.toggle_mode_graph_panel {
            editor_state.toggle_mode_graph_panel = false;
            let panel = crate::ui::panel::Panel::ModeGraph;
            if let Some(location) = dock_manager.current_tree().find_tab(&panel) {
                // Panel is open - store its location and close it
                editor_state.mode_graph_panel_location = Some(location);
                dock_manager.current_tree_mut().remove_tab(location);
            } else {
                // Panel is closed - restore to original location if available
                if let Some((surface_index, node_index, tab_index)) =
                    editor_state.mode_graph_panel_location
                {
                    // Try to restore to the original location
                    let dock_state = dock_manager.current_tree_mut();

                    // Check if the surface still exists and has the node
                    let can_restore = dock_state
                        .get_surface(surface_index)
                        .map(|surface| match surface {
                            egui_dock::Surface::Main(tree)
                            | egui_dock::Surface::Window(tree, _) => {
                                node_index.0 < tree.len() && tree[node_index].is_leaf()
                            }
                            egui_dock::Surface::Empty => false,
                        })
                        .unwrap_or(false);

                    if can_restore {
                        // Restore to the original location
                        dock_state[surface_index][node_index].insert_tab(tab_index, panel);
                        editor_state.mode_graph_panel_location = None;
                    } else {
                        // Original location no longer valid, create as floating window
                        let _surface_index = dock_state.add_window(vec![panel]);
                        editor_state.mode_graph_panel_location = None;
                    }
                } else {
                    // No stored location, create as floating window
                    let _surface_index = dock_manager.current_tree_mut().add_window(vec![panel]);
                }
            }
        }

        // Handle procedural genome request (Preview mode rail button)
        if let Some(seed) = editor_state.procedural_genome_seed.take() {
            *genome = crate::genome::Genome::generate_procedural(seed);
            editor_state.selected_mode_index = 0;
        }

        // Handle water fill toggle request (GPU mode rail button)
        if editor_state.request_toggle_water {
            editor_state.request_toggle_water = false;
            editor_state.fluid_continuous_spawn = !editor_state.fluid_continuous_spawn;
            if let Some(gpu_scene) = scene_manager.gpu_scene_mut() {
                if let Some(ref mut simulator) = gpu_scene.fluid_simulator {
                    simulator.set_continuous_spawn(editor_state.fluid_continuous_spawn);
                }
            }
            editor_state.save_fluid_settings();
        }

        // Render radial menu overlay (GPU mode only)
        // Now editor_state is no longer borrowed by panel_context
        if ui_state_copy.current_mode == crate::ui::types::SimulationMode::Gpu
            && !ui_state_copy.gpu_headless_mode
        {
            crate::ui::radial_menu::show_radial_menu(&self.ctx, &mut editor_state.radial_menu);
            // Only show the tool cursor icon when the pointer is over the viewport,
            // not over a panel - over panels the system cursor is visible instead.
            if !self.wants_pointer_input() {
                crate::ui::radial_menu::show_tool_cursor(&self.ctx, &editor_state.radial_menu);
            }

            // Check for low FPS and show warning dialog
            let fps = performance.fps();
            let sim_speed = scene_manager
                .gpu_scene()
                .map(|s| s.time_scale)
                .unwrap_or(1.0);

            // Only show dialog if FPS < 15 and speed > 1x and dialog not already shown and not suppressed
            if fps < 15.0
                && sim_speed > 1.0
                && !ui_state_copy.show_low_fps_dialog
                && !ui_state_copy.suppress_low_fps_dialog
            {
                ui_state_copy.show_low_fps_dialog = true;
                // Pause simulation while dialog is shown
                *scene_request = crate::ui::panel_context::SceneModeRequest::TogglePause;
            }

            // Render low FPS dialog
            if ui_state_copy.show_low_fps_dialog {
                egui::Window::new("⚠ Low Frame Rate")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(&self.ctx, |ui| {
                        ui.label(format!(
                            "Frame rate dropped to {:.0} FPS at {:.1}x speed.",
                            fps, sim_speed
                        ));
                        ui.add_space(8.0);
                        ui.label("Simulation paused. What would you like to do?");
                        ui.add_space(12.0);

                        ui.horizontal(|ui| {
                            if ui.button("Set to 1x & Resume").clicked() {
                                *scene_request =
                                    crate::ui::panel_context::SceneModeRequest::SetSpeedAndUnpause(
                                        1.0,
                                    );
                                ui_state_copy.show_low_fps_dialog = false;
                            }
                            if ui.button("Resume at Current Speed").clicked() {
                                *scene_request =
                                    crate::ui::panel_context::SceneModeRequest::TogglePause;
                                ui_state_copy.show_low_fps_dialog = false;
                            }
                        });
                        ui.add_space(4.0);
                        ui.checkbox(
                            &mut ui_state_copy.suppress_low_fps_dialog,
                            "Don't ask again this session",
                        );
                    });
            }

            // Render reset confirmation dialog
            if ui_state_copy.show_reset_dialog {
                egui::Window::new("⟲ Reset Simulation")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(&self.ctx, |ui| {
                        ui.label("What would you like to reset?");
                        ui.add_space(12.0);

                        ui.vertical(|ui| {
                            if ui
                                .button("Reset Everything")
                                .on_hover_text("Remove all cells and clear water/fluid")
                                .clicked()
                            {
                                *scene_request = crate::ui::panel_context::SceneModeRequest::Reset;
                                ui_state_copy.show_reset_dialog = false;
                            }
                            ui.add_space(4.0);
                            if ui
                                .button("Reset Cells Only")
                                .on_hover_text("Remove all cells but keep water/fluid")
                                .clicked()
                            {
                                *scene_request =
                                    crate::ui::panel_context::SceneModeRequest::ResetCellsOnly;
                                ui_state_copy.show_reset_dialog = false;
                            }
                            ui.add_space(4.0);
                            if ui.button("Cancel").clicked() {
                                ui_state_copy.show_reset_dialog = false;
                            }
                        });
                    });
            }

            // -- Saving popup --------------------------------------------------
            // Frame 1: popup renders, pending_save_ready is false - no request yet.
            // Frame 2: popup still visible, pending_save_ready becomes true - fire request.
            // This guarantees the user sees the overlay before the blocking GPU work.
            if ui_state_copy.show_saving_popup {
                egui::Window::new("💾 Saving Sphere…")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(&self.ctx, |ui| {
                        ui.add_space(4.0);
                        ui.label("Reading simulation state from GPU…");
                        ui.add_space(8.0);
                        ui.label("This may take a moment for large simulations.");
                        ui.add_space(4.0);
                    });

                if ui_state_copy.pending_save_ready {
                    // Popup has already been painted once - now fire the work.
                    *scene_request = crate::ui::panel_context::SceneModeRequest::SaveSnapshot;
                } else {
                    // First frame: mark ready so the request fires next frame.
                    ui_state_copy.pending_save_ready = true;
                }
            }

            // -- Loading popup -------------------------------------------------
            // The file path was already chosen in the menu item click.
            // Frame 1: popup renders with path stored in pending_load_path.
            // Frame 2: path is taken and LoadSnapshot request is fired.
            if ui_state_copy.show_loading_popup {
                egui::Window::new("📂 Loading Sphere…")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(&self.ctx, |ui| {
                        ui.add_space(4.0);
                        ui.label("Restoring simulation state…");
                        ui.add_space(8.0);
                        ui.label("Uploading cell and fluid data to GPU.");
                        ui.add_space(4.0);
                    });

                if let Some(path) = ui_state_copy.pending_load_path.take() {
                    // Path present - fire the restore request.
                    *scene_request = crate::ui::panel_context::SceneModeRequest::LoadSnapshot(path);
                }
                // app.rs clears show_loading_popup once the work completes.
            }
        }

        // -- Toast notifications -----------------------------------------------
        crate::ui::toast::tick_toasts(&mut self.toasts, 1.0 / 60.0);
        crate::ui::toast::render_toasts(&self.ctx, &self.toasts);

        // -- Loading GIF overlay (shown during GIF capture) --------------------
        if editor_state.gif_capture.is_some() && !self.loading_gif_frames.is_empty() {
            // Advance loading animation at 20fps
            self.loading_gif_timer += 1.0 / 60.0;
            if self.loading_gif_timer >= 1.0 / 20.0 {
                self.loading_gif_timer = 0.0;
                self.loading_gif_frame =
                    (self.loading_gif_frame + 1) % self.loading_gif_frames.len();
            }
            #[allow(deprecated)]
            let screen = self.ctx.screen_rect();
            let size = 64.0_f32;
            let margin = 16.0 + crate::ui::toast::TOAST_H_PUB + crate::ui::toast::TOAST_GAP_PUB;
            let pos = egui::pos2(screen.max.x - 16.0 - size, screen.max.y - margin - size);
            let rect = egui::Rect::from_min_size(pos, egui::vec2(size, size));
            let painter = self.ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("loading_gif_overlay"),
            ));
            painter.image(
                self.loading_gif_frames[self.loading_gif_frame].id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            self.ctx.request_repaint();
        } else if editor_state.gif_capture.is_none() {
            self.loading_gif_frame = 0;
            self.loading_gif_timer = 0.0;
        }

        // -- Tutorial overlay --------------------------------------------------
        // Auto-launch the tutorial the very first time the player opens the
        // Genome Editor, before they've ever seen it.
        if !ui_state_copy.tutorial.ever_shown
            && ui_state_copy.current_mode == crate::ui::types::SimulationMode::Preview
        {
            ui_state_copy.tutorial.start();
        }

        // Render the tutorial dialogue + schematic pointer line.
        crate::ui::tutorial::render_tutorial(
            &self.ctx,
            &mut ui_state_copy.tutorial,
            &editor_state.panel_rects,
            genome,
            editor_state.selected_mode_index,
        );

        // Apply any changes back to the original state
        let state_changed = self.state != ui_state_copy;
        self.state = ui_state_copy;

        // Mark UI state as dirty if it changed
        if state_changed {
            self.mark_ui_state_dirty();
        }

        // Handle global click to clear label text selection
        if self.ctx.input(|i| i.pointer.any_click()) {
            let plugin = self
                .ctx
                .plugin::<egui::text_selection::LabelSelectionState>();
            plugin.lock().clear_selection();
        }

        // Also clear text selection if Escape is pressed
        if self.ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            let plugin = self
                .ctx
                .plugin::<egui::text_selection::LabelSelectionState>();
            plugin.lock().clear_selection();
        }

        self.ctx.end_pass()
    }

    /// Render egui output to the screen.
    ///
    /// This method handles texture updates, buffer uploads, and the actual
    /// rendering of egui primitives.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        screen_descriptor: ScreenDescriptor,
        output: egui::FullOutput,
    ) {
        // Handle platform output (clipboard, cursor, etc.)
        // Note: We don't have access to window here, so platform output
        // should be handled separately if needed

        // Process texture updates
        for (id, image_delta) in &output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }

        // Tessellate shapes into primitives
        let paint_jobs = self.ctx.tessellate(output.shapes, output.pixels_per_point);

        // Update buffers and get any callback command buffers
        let _command_buffers =
            self.renderer
                .update_buffers(device, queue, encoder, &paint_jobs, &screen_descriptor);

        // Create render pass and render egui
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear - render on top of 3D scene
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Render egui - need to forget lifetime for wgpu render pass
            self.renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        // Free textures that are no longer needed
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }

    /// Get the egui context for rendering UI.
    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }

    /// Set the viewport rectangle.
    ///
    /// This should be called when rendering the viewport panel to track
    /// where the 3D scene is displayed.
    pub fn set_viewport_rect(&mut self, rect: egui::Rect) {
        self.viewport_rect = Some(rect);
    }

    /// Get the current viewport rectangle.
    pub fn get_viewport_rect(&self) -> Option<egui::Rect> {
        self.viewport_rect
    }
}

/// Show the Themes menu for selecting the active UI color theme.
fn show_themes_menu(
    ui: &mut egui::Ui,
    state: &mut GlobalUiState,
    dock_manager: &mut crate::ui::dock::DockManager,
) {
    use crate::ui::panel::Panel;
    use crate::ui::types::UiTheme;

    ui.label("Color Theme:");
    ui.add_space(4.0);

    for &theme_choice in UiTheme::all() {
        let is_selected = state.selected_theme == theme_choice;
        let accent = theme_choice.accent_color();

        ui.horizontal(|ui| {
            // Colored swatch dot
            let (dot_rect, _) =
                ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            ui.painter().circle_filled(dot_rect.center(), 4.5, accent);

            // Selectable label - clicking sets the theme
            if ui
                .selectable_label(is_selected, theme_choice.display_name())
                .clicked()
            {
                state.selected_theme = theme_choice;
                // When Custom is selected, open the Theme Editor panel automatically.
                if theme_choice == UiTheme::Custom {
                    let panel = Panel::ThemeEditor;
                    if !is_panel_open(dock_manager.current_tree(), &panel) {
                        open_panel(dock_manager.current_tree_mut(), &panel);
                    }
                }
                ui.close_kind(egui::UiKind::Menu);
            }
        });
    }
}

/// Show the Windows menu for panel visibility toggles.
fn show_windows_menu(
    ui: &mut egui::Ui,
    state: &mut GlobalUiState,
    dock_manager: &mut crate::ui::dock::DockManager,
    scene_manager: &mut crate::scene::SceneManager,
) {
    use crate::ui::panel::Panel;

    // UI Scale radio buttons
    ui.label("UI Scale:");
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            if ui.radio(state.ui_scale == 0.5, "0.5x").clicked() {
                state.ui_scale = 0.5;
            }
            if ui.radio(state.ui_scale == 1.0, "1.0x").clicked() {
                state.ui_scale = 1.0;
            }
            if ui.radio(state.ui_scale == 1.5, "1.5x").clicked() {
                state.ui_scale = 1.5;
            }
            if ui.radio(state.ui_scale == 3.0, "3.0x").clicked() {
                state.ui_scale = 3.0;
            }
        });
        ui.vertical(|ui| {
            if ui.radio(state.ui_scale == 0.75, "0.75x").clicked() {
                state.ui_scale = 0.75;
            }
            if ui.radio(state.ui_scale == 1.25, "1.25x").clicked() {
                state.ui_scale = 1.25;
            }
            if ui.radio(state.ui_scale == 2.0, "2.0x").clicked() {
                state.ui_scale = 2.0;
            }
            if ui.radio(state.ui_scale == 4.0, "4.0x").clicked() {
                state.ui_scale = 4.0;
            }
        });
    });

    ui.add_space(6.0);
    ui.label("Horizontal FOV:")
        .on_hover_text("Horizontal camera field of view in degrees. Higher values show more of the scene with stronger perspective.");
    let fov_response = ui.add(
        egui::Slider::new(
            &mut state.horizontal_fov_degrees,
            crate::ui::camera::MIN_HORIZONTAL_FOV_DEGREES
                ..=crate::ui::camera::MAX_HORIZONTAL_FOV_DEGREES,
        )
        .suffix(" deg"),
    );
    if fov_response.changed() {
        scene_manager
            .active_scene_mut()
            .camera_mut()
            .horizontal_fov_degrees = state.horizontal_fov_degrees;
    }

    ui.separator();

    // List of genome editor panels that can be toggled (only show in Preview mode)
    let genome_editor_panels = [
        Panel::Modes,
        Panel::ModeGraph,
        Panel::AdhesionSettings,
        Panel::ParentSettings,
        Panel::CircleSliders,
        Panel::QuaternionBall,
        Panel::TimeSlider,
        Panel::CellTypeVisuals,
    ];

    // Only show genome editor windows in Preview mode
    if state.current_mode == crate::ui::types::SimulationMode::Preview {
        ui.label("Genome Editor:");
        for panel in &genome_editor_panels {
            let is_open = is_panel_open(dock_manager.current_tree(), panel);
            let panel_name = format!("{:?}", panel);
            let is_locked = state.is_panel_locked(&panel_name);

            ui.horizontal(|ui| {
                // Window toggle button
                if ui
                    .selectable_label(is_open, format!("  {}", panel.display_name()))
                    .clicked()
                {
                    if is_open {
                        close_panel(dock_manager.current_tree_mut(), panel);
                    } else {
                        open_panel(dock_manager.current_tree_mut(), panel);
                    }
                }

                // Lock/Unlock button
                let lock_icon = if is_locked { "🔒" } else { "🔓" };
                if ui.small_button(lock_icon).clicked() {
                    state.set_panel_locked(&panel_name, !is_locked);
                }
            });
        }

        ui.separator();
    }

    // Layout Panels
    ui.label("Layout Panels:");

    let layout_panels = [
        Panel::LeftPanel,
        Panel::RightPanel,
        Panel::BottomPanel,
        Panel::Viewport,
        Panel::GizmoSettings,
    ];

    for panel in &layout_panels {
        let is_open = is_panel_open(dock_manager.current_tree(), panel);
        let panel_name = format!("{:?}", panel);
        let is_locked = state.is_panel_locked(&panel_name);

        ui.horizontal(|ui| {
            // Window toggle button
            if ui
                .selectable_label(is_open, format!("  {}", panel.display_name()))
                .clicked()
            {
                if is_open {
                    close_panel(dock_manager.current_tree_mut(), panel);
                } else {
                    open_panel(dock_manager.current_tree_mut(), panel);
                }
            }

            // Lock/Unlock button
            let lock_icon = if is_locked { "🔒" } else { "🔓" };
            if ui.small_button(lock_icon).clicked() {
                state.set_panel_locked(&panel_name, !is_locked);
            }
        });
    }

    ui.separator();

    // Other Windows
    ui.label("Other Windows:");

    // Scene Manager
    let scene_manager_open = is_panel_open(dock_manager.current_tree(), &Panel::SceneManager);
    let scene_manager_name = format!("{:?}", Panel::SceneManager);
    let scene_manager_locked = state.is_panel_locked(&scene_manager_name);

    ui.horizontal(|ui| {
        if ui
            .selectable_label(scene_manager_open, "  Scene Manager")
            .clicked()
        {
            if scene_manager_open {
                close_panel(dock_manager.current_tree_mut(), &Panel::SceneManager);
            } else {
                open_panel(dock_manager.current_tree_mut(), &Panel::SceneManager);
            }
        }

        let lock_icon = if scene_manager_locked { "🔒" } else { "🔓" };
        if ui.small_button(lock_icon).clicked() {
            state.set_panel_locked(&scene_manager_name, !scene_manager_locked);
        }
    });

    // Performance Monitor
    let perf_open = is_panel_open(dock_manager.current_tree(), &Panel::PerformanceMonitor);
    let perf_name = format!("{:?}", Panel::PerformanceMonitor);
    let perf_locked = state.is_panel_locked(&perf_name);

    ui.horizontal(|ui| {
        if ui.selectable_label(perf_open, "  Performance").clicked() {
            if perf_open {
                close_panel(dock_manager.current_tree_mut(), &Panel::PerformanceMonitor);
            } else {
                open_panel(dock_manager.current_tree_mut(), &Panel::PerformanceMonitor);
            }
        }

        let lock_icon = if perf_locked { "🔒" } else { "🔓" };
        if ui.small_button(lock_icon).clicked() {
            state.set_panel_locked(&perf_name, !perf_locked);
        }
    });

    // Cell Inspector
    let inspector_open = is_panel_open(dock_manager.current_tree(), &Panel::CellInspector);
    let inspector_name = format!("{:?}", Panel::CellInspector);
    let inspector_locked = state.is_panel_locked(&inspector_name);

    ui.horizontal(|ui| {
        if ui
            .selectable_label(inspector_open, "  Cell Inspector")
            .clicked()
        {
            if inspector_open {
                close_panel(dock_manager.current_tree_mut(), &Panel::CellInspector);
            } else {
                open_panel(dock_manager.current_tree_mut(), &Panel::CellInspector);
            }
        }

        let lock_icon = if inspector_locked { "🔒" } else { "🔓" };
        if ui.small_button(lock_icon).clicked() {
            state.set_panel_locked(&inspector_name, !inspector_locked);
        }
    });

    // Cave System (GPU mode only)
    if state.current_mode == crate::ui::types::SimulationMode::Gpu {
        // Lineage Viewer (GPU mode only)
        let lineage_open = is_panel_open(dock_manager.current_tree(), &Panel::LineageViewer);
        let lineage_name = format!("{:?}", Panel::LineageViewer);
        let lineage_locked = state.is_panel_locked(&lineage_name);

        ui.horizontal(|ui| {
            if ui
                .selectable_label(lineage_open, "  Lineage Viewer")
                .clicked()
            {
                if lineage_open {
                    close_panel(dock_manager.current_tree_mut(), &Panel::LineageViewer);
                } else {
                    open_panel_docked_to_viewport(
                        dock_manager.current_tree_mut(),
                        &Panel::LineageViewer,
                    );
                }
            }

            let lock_icon = if lineage_locked {
                "ðŸ”’"
            } else {
                "ðŸ”“"
            };
            if ui.small_button(lock_icon).clicked() {
                state.set_panel_locked(&lineage_name, !lineage_locked);
            }
        });

        let cave_open = is_panel_open(dock_manager.current_tree(), &Panel::CaveSystem);
        let cave_name = format!("{:?}", Panel::CaveSystem);
        let cave_locked = state.is_panel_locked(&cave_name);

        ui.horizontal(|ui| {
            if ui.selectable_label(cave_open, "  Cave System").clicked() {
                if cave_open {
                    close_panel(dock_manager.current_tree_mut(), &Panel::CaveSystem);
                } else {
                    open_panel(dock_manager.current_tree_mut(), &Panel::CaveSystem);
                }
            }

            let lock_icon = if cave_locked { "🔒" } else { "🔓" };
            if ui.small_button(lock_icon).clicked() {
                state.set_panel_locked(&cave_name, !cave_locked);
            }
        });

        // Fluid Settings (GPU mode only)
        let fluid_open = is_panel_open(dock_manager.current_tree(), &Panel::FluidSettings);
        let fluid_name = format!("{:?}", Panel::FluidSettings);
        let fluid_locked = state.is_panel_locked(&fluid_name);

        ui.horizontal(|ui| {
            if ui.selectable_label(fluid_open, "  Fluid System").clicked() {
                if fluid_open {
                    close_panel(dock_manager.current_tree_mut(), &Panel::FluidSettings);
                } else {
                    open_panel(dock_manager.current_tree_mut(), &Panel::FluidSettings);
                }
            }

            let lock_icon = if fluid_locked { "🔒" } else { "🔓" };
            if ui.small_button(lock_icon).clicked() {
                state.set_panel_locked(&fluid_name, !fluid_locked);
            }
        });

        // World Settings (GPU mode only)
        let world_open = is_panel_open(dock_manager.current_tree(), &Panel::WorldSettings);
        let world_name = format!("{:?}", Panel::WorldSettings);
        let world_locked = state.is_panel_locked(&world_name);

        ui.horizontal(|ui| {
            if ui
                .selectable_label(world_open, "  World Settings")
                .clicked()
            {
                if world_open {
                    close_panel(dock_manager.current_tree_mut(), &Panel::WorldSettings);
                } else {
                    open_panel(dock_manager.current_tree_mut(), &Panel::WorldSettings);
                }
            }

            let lock_icon = if world_locked { "🔒" } else { "🔓" };
            if ui.small_button(lock_icon).clicked() {
                state.set_panel_locked(&world_name, !world_locked);
            }
        });

        // Lighting Settings (GPU mode only)
        let light_open = is_panel_open(dock_manager.current_tree(), &Panel::LightSettings);
        let light_name = format!("{:?}", Panel::LightSettings);
        let light_locked = state.is_panel_locked(&light_name);

        ui.horizontal(|ui| {
            if ui.selectable_label(light_open, "  Lighting").clicked() {
                if light_open {
                    close_panel(dock_manager.current_tree_mut(), &Panel::LightSettings);
                } else {
                    open_panel(dock_manager.current_tree_mut(), &Panel::LightSettings);
                }
            }

            let lock_icon = if light_locked { "🔒" } else { "🔓" };
            if ui.small_button(lock_icon).clicked() {
                state.set_panel_locked(&light_name, !light_locked);
            }
        });
    }

    ui.separator();

    // Layout Management
    ui.label("Layout Management:");

    // Reset to default layout button
    if ui.button("🔄 Reset to Default").clicked() {
        dock_manager.reset_current_to_default();
    }

    ui.separator();

    // Layout file information
    ui.label("Layout Files:");
    ui.small("Current layout files:");

    let current_mode = dock_manager.current_mode();
    let layout_file = format!("dock_state_{}.ron", current_mode.dock_file_suffix());
    let default_file = format!("default_dock_state_{}.ron", current_mode.dock_file_suffix());

    ui.horizontal(|ui| {
        ui.small("• Active:");
        ui.small(&layout_file);
    });

    ui.horizontal(|ui| {
        ui.small("• Default:");
        if std::path::Path::new(&default_file).exists() {
            ui.small(&default_file);
        } else {
            ui.small("(using hardcoded default)");
        }
    });
}

/// Check if a panel is currently open in the dock tree
fn is_panel_open(
    tree: &egui_dock::DockState<crate::ui::panel::Panel>,
    panel: &crate::ui::panel::Panel,
) -> bool {
    tree.iter_all_tabs().any(|(_, tab)| tab == panel)
}

/// Close a panel in the dock tree
fn close_panel(
    tree: &mut egui_dock::DockState<crate::ui::panel::Panel>,
    panel: &crate::ui::panel::Panel,
) {
    // Simple approach: just remove all instances of this panel
    tree.retain_tabs(|tab| tab != panel);
}

/// Open a panel in the dock tree as a floating window
fn open_panel(
    tree: &mut egui_dock::DockState<crate::ui::panel::Panel>,
    panel: &crate::ui::panel::Panel,
) {
    // Create a new floating window with the panel
    // Ensure we always pass a non-empty vector to prevent crashes
    let _surface_index = tree.add_window(vec![*panel]);
}

/// Open a panel as a tab in the same dock leaf as the main viewport, then focus it.
fn open_panel_docked_to_viewport(
    tree: &mut egui_dock::DockState<crate::ui::panel::Panel>,
    panel: &crate::ui::panel::Panel,
) {
    if let Some(location) = tree.find_tab(panel) {
        tree.set_active_tab(location);
        tree.set_focused_node_and_surface((location.0, location.1));
        return;
    }

    if let Some((surface, node, _tab)) = tree.find_tab(&crate::ui::panel::Panel::Viewport) {
        tree.set_focused_node_and_surface((surface, node));
        tree.push_to_focused_leaf(*panel);
        if let Some(location) = tree.find_tab(panel) {
            tree.set_active_tab(location);
            tree.set_focused_node_and_surface((location.0, location.1));
        }
    } else {
        tree.push_to_first_leaf(*panel);
        if let Some(location) = tree.find_tab(panel) {
            tree.set_active_tab(location);
            tree.set_focused_node_and_surface((location.0, location.1));
        }
    }
}

/// Show the Save / Load menu.
///
/// For **save**: sets `show_saving_popup = true`.  The popup renders for one
/// frame, then `app.rs` fires the GPU readback + file dialog on the next frame.
///
/// For **load**: opens the native file picker immediately (it's a quick dialog,
/// not blocking GPU work), stores the chosen path in `pending_load_path`, and
/// sets `show_loading_popup = true`.  The popup renders for one frame, then
/// `app.rs` fires the restore work on the next frame.
fn show_save_load_menu(ui: &mut egui::Ui, ui_state: &mut GlobalUiState) {
    if ui
        .button("💾  Save Sphere…")
        .on_hover_text("Save the current simulation state to a .sphere file")
        .clicked()
    {
        ui.close_kind(egui::UiKind::Menu);
        ui_state.show_saving_popup = true;
        ui_state.pending_save_ready = false;
    }

    if ui
        .button("📂  Load Sphere…")
        .on_hover_text("Restore a previously saved simulation state from a .sphere file")
        .clicked()
    {
        ui.close_kind(egui::UiKind::Menu);
        // Open the file picker here - it's a quick OS dialog and doesn't
        // involve any GPU work, so blocking is fine at this point.
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Load Sphere")
            .add_filter("Bio-Spheres Sphere", &["sphere"])
            .set_directory("genomes")
            .pick_file()
        {
            // Store the path and show the popup.  The actual restore work
            // fires on the next frame once the popup has been painted.
            ui_state.pending_load_path = Some(path);
            ui_state.show_loading_popup = true;
        }
    }
}

// -----------------------------------------------------------------------------
// Top bar / status bar helpers
// -----------------------------------------------------------------------------

/// Draw the Bio-Spheres logo glyph - a hexagon with a small inner ring.
fn draw_logo_glyph(painter: &egui::Painter, rect: egui::Rect, color: egui::Color32) {
    let center = rect.center();
    let radius = rect.width() * 0.45;

    // Outer hexagon (pointy-top)
    let mut points = Vec::with_capacity(6);
    for i in 0..6 {
        let angle = (i as f32) * std::f32::consts::TAU / 6.0 - std::f32::consts::FRAC_PI_2;
        points.push(egui::pos2(
            center.x + angle.cos() * radius,
            center.y + angle.sin() * radius,
        ));
    }
    // Hexagon stroke
    for i in 0..6 {
        painter.line_segment(
            [points[i], points[(i + 1) % 6]],
            egui::Stroke::new(1.5, color),
        );
    }
    // Inner ring
    painter.circle_stroke(center, radius * 0.45, egui::Stroke::new(1.0, color));
    // Center dot
    painter.circle_filled(center, 1.4, color);
}

/// Render a labelled status field for the bottom status bar.
///
/// Layout:  `LABEL` (small, dim, uppercase) above value content.
/// The closure paints whatever value/widgets the field needs.
fn status_field(ui: &mut egui::Ui, label: &str, content: &dyn Fn(&mut egui::Ui)) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        ui.label(egui::RichText::new(label).size(8.5).color(theme::TEXT_DIM));
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            content(ui);
        });
    });
}

/// Render a thin vertical separator between status bar fields.
fn status_separator(ui: &mut egui::Ui) {
    ui.add_space(10.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 24.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 0.0, theme::BORDER_SUBTLE);
    ui.add_space(10.0);
}

/// Render a thin vertical divider for the top bar.
///
/// Slightly shorter than the status-bar variant and uses the brighter
/// top-bar border tone so the separation reads cleanly against the
/// darker top-bar fill.
fn topbar_divider(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, 18.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 0.0, theme::BORDER_NORMAL);
}

pub(crate) fn headless_metric(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    let p = palette();
    ui.vertical(|ui| {
        ui.label(
            egui::RichText::new(label.to_ascii_uppercase())
                .size(10.5)
                .color(p.text_dim),
        );
        ui.label(egui::RichText::new(value).size(24.0).strong().color(color));
    });
}

pub(crate) fn stat_label(ui: &mut egui::Ui, label: &str, value: &str) {
    let p = palette();
    ui.label(
        egui::RichText::new(label)
            .size(11.0)
            .color(p.text_secondary),
    );
    ui.label(egui::RichText::new(value).size(11.0).color(p.text_primary));
}

pub(crate) fn average_fps_for_last_samples(frame_times_ms: &[f32], samples: usize) -> f32 {
    if frame_times_ms.is_empty() {
        return 0.0;
    }
    let start = frame_times_ms.len().saturating_sub(samples);
    let slice = &frame_times_ms[start..];
    let avg_ms = slice.iter().sum::<f32>() / slice.len() as f32;
    if avg_ms > 0.0 {
        1000.0 / avg_ms
    } else {
        0.0
    }
}

pub(crate) fn fps_color(fps: f32, p: ActivePalette) -> egui::Color32 {
    if fps >= 50.0 {
        p.status_ok
    } else if fps >= 30.0 {
        p.status_warn
    } else {
        p.status_err
    }
}

pub(crate) fn draw_headless_fps_graph(
    ui: &mut egui::Ui,
    frame_times_ms: &[f32],
    height: f32,
    target_fps: f32,
) {
    let p = palette();
    let (response, painter) = ui.allocate_painter(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let rect = response.rect;
    painter.rect_filled(rect, egui::CornerRadius::same(6), p.bg_panel);
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(6),
        egui::Stroke::new(1.0, p.border_subtle),
        egui::StrokeKind::Inside,
    );

    if frame_times_ms.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "waiting for samples",
            egui::FontId::proportional(13.0),
            p.text_dim,
        );
        return;
    }

    let max_fps = 90.0_f32;
    for guide in [30.0_f32, 60.0_f32, target_fps] {
        let y = rect.bottom() - (guide / max_fps).clamp(0.0, 1.0) * rect.height();
        let color = if (guide - target_fps).abs() < 0.1 {
            p.accent_secondary
        } else {
            p.border_normal
        };
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0, color.linear_multiply(0.7)),
        );
        painter.text(
            egui::pos2(rect.left() + 8.0, y - 2.0),
            egui::Align2::LEFT_BOTTOM,
            format!("{:.0}", guide),
            egui::FontId::proportional(10.0),
            color,
        );
    }

    let bar_width = (rect.width() / frame_times_ms.len() as f32).max(1.0);
    for (i, &time) in frame_times_ms.iter().enumerate() {
        let fps_val = if time > 0.0 { 1000.0 / time } else { 0.0 };
        let x = rect.left() + i as f32 * bar_width;
        let h = (fps_val / max_fps).clamp(0.0, 1.0) * rect.height();
        let y = rect.bottom() - h;
        let color = fps_color(fps_val, p);
        painter.rect_filled(
            egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(bar_width, h)),
            0.0,
            color,
        );
    }
}

// -----------------------------------------------------------------------------
// Left side rail - quick access to mode-specific panels
// -----------------------------------------------------------------------------

/// Render the left side rail with stacked icon buttons.
fn render_side_rail(
    ui: &mut egui::Ui,
    state: &mut GlobalUiState,
    editor_state: &mut crate::ui::panel_context::GenomeEditorState,
    dock_manager: &mut crate::ui::dock::DockManager,
) {
    let p = palette();
    ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);

    match state.current_mode {
        crate::ui::types::SimulationMode::Preview => {
            // Adhesion expansion toggle
            let expand_active = state.adhesion_expansion_active;
            if rail_button_toggle(ui, "⤢", "Expand Adhesions (toggle)", expand_active, &p) {
                state.adhesion_expansion_active = !expand_active;
            }

            // Hide UI toggle
            let hide_active = state.hide_ui;
            if rail_button_toggle(ui, "👁", "Hide UI (toggle)", hide_active, &p) {
                state.hide_ui = !hide_active;
            }

            // Procedural genome
            if rail_button(ui, "🎲", "Generate Procedural Genome", &p) {
                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0xdeadbeef_cafebabe);
                editor_state.procedural_genome_seed = Some(seed);
            }

            // Angle snapping toggle
            let snap_active = editor_state.enable_snapping;
            if rail_button_toggle(ui, "📐", "Snap Angles to 15° (toggle)", snap_active, &p) {
                editor_state.enable_snapping = !snap_active;
                // Keep qball snapping in sync
                editor_state.qball_snapping = !snap_active;
            }

            // Mode graph toggle
            let graph_open = is_panel_open(
                dock_manager.current_tree(),
                &crate::ui::panel::Panel::ModeGraph,
            );
            if rail_button_toggle(ui, "🕸", "Toggle Mode Graph", graph_open, &p) {
                editor_state.toggle_mode_graph_panel = true;
            }

            // Screenshot
            if rail_button(ui, "📷", "Take Screenshot", &p) {
                editor_state.request_screenshot = true;
            }

            // Open biospheres folder
            if rail_button(ui, "📁", "Open Bio-Spheres Folder", &p) {
                let dir = crate::app_dirs::biospheres_dir();
                #[cfg(target_os = "windows")]
                let _ = std::process::Command::new("explorer").arg(&dir).spawn();
                #[cfg(target_os = "macos")]
                let _ = std::process::Command::new("open").arg(&dir).spawn();
                #[cfg(target_os = "linux")]
                let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
            }
        }
        crate::ui::types::SimulationMode::Gpu => {
            // Advanced Options toggle
            let adv_active = state.show_advanced_options;
            if rail_button_toggle(
                ui,
                "⚙",
                "Advanced Options (toggle fine-tuning sliders)",
                adv_active,
                &p,
            ) {
                state.show_advanced_options = !adv_active;
            }

            // Hide UI toggle
            let hide_active = state.hide_ui;
            if rail_button_toggle(ui, "👁", "Hide UI (toggle)", hide_active, &p) {
                state.hide_ui = !hide_active;
            }

            // Lineage viewer
            let lineage_open = is_panel_open(
                dock_manager.current_tree(),
                &crate::ui::panel::Panel::LineageViewer,
            );
            if rail_button_toggle(ui, "🧬", "Lineage Viewer", lineage_open, &p) {
                if lineage_open {
                    close_panel(
                        dock_manager.current_tree_mut(),
                        &crate::ui::panel::Panel::LineageViewer,
                    );
                } else {
                    open_panel_docked_to_viewport(
                        dock_manager.current_tree_mut(),
                        &crate::ui::panel::Panel::LineageViewer,
                    );
                }
            }

            // Water fill toggle
            let water_active = editor_state.fluid_continuous_spawn;
            if rail_button_toggle(ui, "🌊", "Toggle Water Fill", water_active, &p) {
                editor_state.request_toggle_water = true;
            }

            // Screenshot
            if rail_button(ui, "📷", "Take Screenshot", &p) {
                editor_state.request_screenshot = true;
            }

            // Open biospheres folder
            if rail_button(ui, "📁", "Open Bio-Spheres Folder", &p) {
                let dir = crate::app_dirs::biospheres_dir();
                #[cfg(target_os = "windows")]
                let _ = std::process::Command::new("explorer").arg(&dir).spawn();
                #[cfg(target_os = "macos")]
                let _ = std::process::Command::new("open").arg(&dir).spawn();
                #[cfg(target_os = "linux")]
                let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
            }
        }
    }

    ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 4.0);
        ui.add_space(4.0);
        if rail_button(ui, "⟲", "Reset Layout", &p) {
            dock_manager.reset_current_to_default();
        }
    });
}

/// Render a single icon button on the side rail.
fn rail_button(ui: &mut egui::Ui, icon: &str, tooltip: &str, p: &ActivePalette) -> bool {
    let size = egui::vec2(32.0, 32.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let resp = resp.on_hover_text(tooltip);

    let painter = ui.painter();
    painter.rect_filled(rect, egui::CornerRadius::same(3), p.bg_widget);
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(3),
        egui::Stroke::new(1.0, p.border_subtle),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(15.0),
        p.rail_icon,
    );

    resp.clicked()
}

/// Render a toggleable icon button on the side rail.
/// When `active` is true the button is highlighted with the accent colour.
fn rail_button_toggle(
    ui: &mut egui::Ui,
    icon: &str,
    tooltip: &str,
    active: bool,
    p: &ActivePalette,
) -> bool {
    let size = egui::vec2(32.0, 32.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let resp = resp.on_hover_text(tooltip);

    let painter = ui.painter();
    let bg_color = if active {
        p.accent_primary
    } else {
        p.bg_widget
    };
    let border_color = if active {
        p.accent_primary
    } else {
        p.border_subtle
    };
    let text_color = if active {
        p.rail_icon_active
    } else {
        p.rail_icon
    };

    painter.rect_filled(rect, egui::CornerRadius::same(3), bg_color);
    painter.rect_stroke(
        rect,
        egui::CornerRadius::same(3),
        egui::Stroke::new(1.0, border_color),
        egui::StrokeKind::Inside,
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon,
        egui::FontId::proportional(15.0),
        text_color,
    );

    resp.clicked()
}

/// Load the embedded app icon PNG into an egui texture handle.
///
/// The PNG bytes are embedded at compile time so the icon always ships with
/// the binary regardless of the runtime working directory. Returns `None` if
/// decoding fails - callers should fall back to a procedural glyph.
fn load_app_icon_texture(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    const ICON_BYTES: &[u8] = include_bytes!("../../assets/icon.png");

    let img = match image::load_from_memory(ICON_BYTES) {
        Ok(img) => img.to_rgba8(),
        Err(e) => {
            log::warn!("Failed to decode embedded app icon: {}", e);
            return None;
        }
    };
    let (w, h) = img.dimensions();
    let pixels = img.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &pixels);

    Some(ctx.load_texture(
        "bio_spheres_app_icon",
        color_image,
        egui::TextureOptions::LINEAR,
    ))
}

/// Load the user-provided loading animation GIF from `assets/loading.gif`.
/// Returns an empty Vec if the file doesn't exist - callers fall back to a
/// text spinner. The file is loaded at runtime (not embedded) so the user can
/// drop in their own GIF without recompiling.
fn load_loading_gif_frames(ctx: &egui::Context) -> Vec<egui::TextureHandle> {
    let gif_path = std::path::Path::new("assets/loading.gif");
    if !gif_path.exists() {
        return Vec::new();
    }

    let data = match std::fs::read(gif_path) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("Could not read assets/loading.gif: {}", e);
            return Vec::new();
        }
    };

    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::RGBA);
    let mut decoder = match opts.read_info(std::io::Cursor::new(&data)) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("Could not decode assets/loading.gif: {}", e);
            return Vec::new();
        }
    };

    let w = decoder.width() as usize;
    let h = decoder.height() as usize;
    let mut frames = Vec::new();

    while let Ok(Some(frame)) = decoder.read_next_frame() {
        let pixels = frame.buffer.to_vec();
        if pixels.len() == w * h * 4 {
            let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &pixels);
            let handle = ctx.load_texture(
                format!("loading_gif_{}", frames.len()),
                img,
                egui::TextureOptions::LINEAR,
            );
            frames.push(handle);
        }
    }

    if !frames.is_empty() {
        log::info!(
            "Loaded loading animation: {} frames ({}×{})",
            frames.len(),
            w,
            h
        );
    }
    frames
}

/// Paint L-shaped teal corner brackets around the GPU-mode viewport rect.
fn paint_viewport_brackets(painter: &egui::Painter, viewport_rect: egui::Rect) {
    if viewport_rect.width() <= 0.0 || viewport_rect.height() <= 0.0 {
        return;
    }

    // Inset from the viewport edge so the brackets sit clearly inside.
    let inset = 16.0;
    let r = viewport_rect.shrink(inset);
    if r.width() <= 0.0 || r.height() <= 0.0 {
        return;
    }

    // Arm length: 4% of the smaller dimension, clamped 20-80px.
    let arm = (r.width().min(r.height()) * 0.04).clamp(20.0, 80.0);

    let color = theme::ACCENT_TEAL;
    let stroke = egui::Stroke::new(1.5, color);

    // Each bracket is a single connected polyline (3 points forming an L).
    // Drawing as a polyline ensures the corner pixel is shared and the join
    // looks clean - no gap, no overlap, no double-painted corner.
    let brackets: [&[egui::Pos2; 3]; 4] = [
        // Top-left
        &[
            egui::pos2(r.left(), r.top() + arm),
            egui::pos2(r.left(), r.top()),
            egui::pos2(r.left() + arm, r.top()),
        ],
        // Top-right
        &[
            egui::pos2(r.right() - arm, r.top()),
            egui::pos2(r.right(), r.top()),
            egui::pos2(r.right(), r.top() + arm),
        ],
        // Bottom-left
        &[
            egui::pos2(r.left(), r.bottom() - arm),
            egui::pos2(r.left(), r.bottom()),
            egui::pos2(r.left() + arm, r.bottom()),
        ],
        // Bottom-right
        &[
            egui::pos2(r.right() - arm, r.bottom()),
            egui::pos2(r.right(), r.bottom()),
            egui::pos2(r.right(), r.bottom() - arm),
        ],
    ];

    for pts in brackets {
        painter.line_segment([pts[0], pts[1]], stroke);
        painter.line_segment([pts[1], pts[2]], stroke);
    }
}

/// Paint black corner pieces that make the viewport appear to have rounded
/// corners - like the reference image showing a rounded rectangle.
///
/// At each corner we paint a black shape that fills the gap between the
/// viewport's straight bounding-box corner and the rounded corner curve.
/// The shape is: a `radius x radius` square with a quarter-circle notch
/// cut from the inner corner (toward the viewport centre).
///
/// Built as a polygon fan: corner -> edge_x -> arc[0..N] -> edge_y -> corner
fn paint_viewport_rounded_corners(
    painter: &egui::Painter,
    viewport_rect: egui::Rect,
    radius: f32,
    color: egui::Color32,
) {
    const SEGMENTS: usize = 20;

    let vert = |pos: egui::Pos2| egui::epaint::Vertex {
        pos,
        uv: egui::epaint::WHITE_UV,
        color,
    };

    // For each corner:
    //   corner   = the actual viewport corner (bounding-box corner)
    //   edge_x   = point `radius` along the horizontal edge inward
    //   edge_y   = point `radius` along the vertical edge inward
    //   arc_cx   = centre of the quarter-circle arc = (edge_x.x, edge_y.y)
    //              i.e. `radius` inward from the corner diagonally
    //   arc sweeps from edge_x back to edge_y going through the corner
    //   (the arc bulges toward the corner, creating the concave notch)
    let r = viewport_rect;
    let corners: [(egui::Pos2, egui::Pos2, egui::Pos2, f32, f32); 4] = [
        // top-left: arc centre (left+r, top+r), sweep 180 deg->270 deg (points toward corner)
        (
            r.left_top(),
            egui::pos2(r.left() + radius, r.top()), // along top edge
            egui::pos2(r.left(), r.top() + radius), // along left edge
            std::f32::consts::PI,
            std::f32::consts::PI * 1.5,
        ),
        // top-right: arc centre (right-r, top+r), sweep 270 deg->360 deg
        (
            r.right_top(),
            egui::pos2(r.right(), r.top() + radius), // along right edge
            egui::pos2(r.right() - radius, r.top()), // along top edge
            std::f32::consts::PI * 1.5,
            std::f32::consts::TAU,
        ),
        // bottom-right: arc centre (right-r, bottom-r), sweep 0 deg->90 deg
        (
            r.right_bottom(),
            egui::pos2(r.right() - radius, r.bottom()), // along bottom edge
            egui::pos2(r.right(), r.bottom() - radius), // along right edge
            0.0_f32,
            std::f32::consts::FRAC_PI_2,
        ),
        // bottom-left: arc centre (left+r, bottom-r), sweep 90 deg->180 deg
        (
            r.left_bottom(),
            egui::pos2(r.left(), r.bottom() - radius), // along left edge
            egui::pos2(r.left() + radius, r.bottom()), // along bottom edge
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
        ),
    ];

    for (corner, edge_a, edge_b, start_angle, end_angle) in corners {
        // Arc centre = radius inward from corner on both axes
        let arc_cx = egui::pos2(
            corner.x + (edge_a.x - corner.x) + (edge_b.x - corner.x),
            corner.y + (edge_a.y - corner.y) + (edge_b.y - corner.y),
        );

        let mut mesh = egui::Mesh::default();

        // Fan centre = the bounding-box corner.
        // The arc endpoints land exactly on edge_a and edge_b, so we don't
        // need explicit spoke vertices - the arc itself closes the shape.
        mesh.vertices.push(vert(corner));

        // Arc sweeping from start_angle to end_angle
        for i in 0..=SEGMENTS {
            let t = i as f32 / SEGMENTS as f32;
            let angle = start_angle + t * (end_angle - start_angle);
            mesh.vertices.push(vert(egui::pos2(
                arc_cx.x + angle.cos() * radius,
                arc_cx.y + angle.sin() * radius,
            )));
        }

        // Triangle fan from corner (0) to each consecutive arc pair
        let n = mesh.vertices.len() as u32;
        for i in 1..(n - 1) {
            mesh.indices.extend_from_slice(&[0, i, i + 1]);
        }

        painter.add(egui::Shape::mesh(mesh));
    }
}

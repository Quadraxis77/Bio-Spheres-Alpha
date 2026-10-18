const fs = require('fs');
const file='src/ui/ui_system.rs'; let source=fs.readFileSync(file,'utf8');
function get(name) { const a=source.indexOf('fn '+name+'('); let b=source.indexOf('\nfn ',a+4);if(b<0)b=source.length;const s=source.slice(a,b);return s.slice(0,s.lastIndexOf('\n}')+2); }
let menu=get('show_windows_menu');
const prefix=menu.slice(0,menu.indexOf('    // List of genome editor panels'));
const speedStart=prefix.indexOf('    ui.label("Speed 1:")');
const speedEnd=prefix.lastIndexOf('    ui.separator();');
if(speedStart<0||speedEnd<speedStart)throw Error('Camera shortcut settings');
let patched=prefix.slice(0,speedStart)+'    if access.allows(15) {\n'+prefix.slice(speedStart,speedEnd)+'    }\n'+prefix.slice(speedEnd);
patched=patched.replace('    use crate::ui::panel::Panel;','    use crate::ui::panel::Panel;\n    let access = state.menu_access();\n    if access.developer { ui.weak("Developer settings (F11)"); }');
patched+=`    if access.allows(18) {
        if state.current_mode == crate::ui::types::SimulationMode::Preview {
            ui.label("Genome Editor:");
            for panel in [Panel::Modes, Panel::ModeGraph, Panel::AdhesionSettings, Panel::ParentSettings,
                Panel::CircleSliders, Panel::QuaternionBall, Panel::TimeSlider, Panel::CellTypeVisuals] {
                reviewed_panel_toggle(ui, state, dock_manager, panel);
            }
            ui.separator();
        }
        ui.label("Layout Panels:");
        for panel in [Panel::LeftPanel, Panel::RightPanel, Panel::BottomPanel, Panel::Viewport, Panel::GizmoSettings] {
            reviewed_panel_toggle(ui, state, dock_manager, panel);
        }
        ui.separator();
        ui.label("Other Windows:");
        for panel in [Panel::SceneManager, Panel::PerformanceMonitor, Panel::CellInspector, Panel::LineageViewer,
            Panel::CaveSystem, Panel::FluidSettings, Panel::WorldSettings, Panel::LightSettings,
            Panel::CameraSettings, Panel::RenderingControls, Panel::ThemeEditor, Panel::TimeScrubber,
            Panel::GenomeEditor, Panel::Help] {
            if state.current_mode != crate::ui::types::SimulationMode::Gpu
                && matches!(panel, Panel::CaveSystem | Panel::FluidSettings | Panel::WorldSettings | Panel::LightSettings) { continue; }
            reviewed_panel_toggle(ui, state, dock_manager, panel);
        }
    }
    ui.separator();
    ui.label("Layout Management:");
    if ui.button("🔄 Reset to Default").clicked() {
        dock_manager.reset_current_to_default();
    }
    if access.allows(21) {
        ui.separator();
        ui.label("Layout Files:");
        ui.small("Current layout files:");
        let current_mode = dock_manager.current_mode();
        let layout_file = format!("dock_state_{}.ron", current_mode.dock_file_suffix());
        let default_file = format!("default_dock_state_{}.ron", current_mode.dock_file_suffix());
        ui.horizontal(|ui| { ui.small("• Active:"); ui.small(&layout_file); });
        ui.horizontal(|ui| {
            ui.small("• Default:");
            ui.small(if std::path::Path::new(&default_file).exists() { &default_file } else { "(using hardcoded default)" });
        });
    }
}

fn reviewed_panel_toggle(ui: &mut egui::Ui, state: &mut GlobalUiState,
    dock_manager: &mut crate::ui::dock::DockManager, panel: crate::ui::panel::Panel) {
    if !panel.available_in_mode(state.current_mode) || !state.menu_access().allows_panel(panel) { return; }
    let open = is_panel_open(dock_manager.current_tree(), &panel);
    let name = format!("{panel:?}");
    let locked = state.is_panel_locked(&name);
    ui.horizontal(|ui| {
        if ui.selectable_label(open, format!("  {}", panel.display_name())).clicked() {
            if open { close_panel(dock_manager.current_tree_mut(), &panel); }
            else if panel == crate::ui::panel::Panel::LineageViewer {
                open_panel_docked_to_viewport(dock_manager.current_tree_mut(), &panel);
            } else { open_panel(dock_manager.current_tree_mut(), &panel); }
        }
        if state.menu_access().allows(19) && ui.small_button(if locked { "🔒" } else { "🔓" }).clicked() {
            state.set_panel_locked(&name, !locked);
        }
    });
}`;
source=source.replace(menu,patched);
source=source.replace('if ui_state_copy.cosmetic_options_visible() {','if ui_state_copy.menu_access().allows(4) {');
const marker='            // Show dock area (scoped to release borrows after)';
source=source.replace(marker,'            dock_manager.apply_menu_visibility(ui_state_copy.menu_access());\n\n'+marker);
// Advanced is required by some reviewed editor panels as well as the live scene.
let rail=get('render_side_rail');
const a=rail.indexOf('            // Advanced Options toggle'),b=rail.indexOf('            // Hide UI toggle',a);
if(a<0||b<0)throw Error('Advanced rail control');
const advanced=rail.slice(a,b);
rail=rail.slice(0,a)+rail.slice(b);
rail=rail.replace('    match state.current_mode {',advanced+'\n    match state.current_mode {');
source=source.replace(get('render_side_rail'),rail);
fs.writeFileSync(file,source);

const tabs='src/ui/tab_viewer.rs';let tab=fs.readFileSync(tabs,'utf8');
const ha=tab.indexOf('fn render_headless_section('),hb=tab.indexOf('\nfn ',ha+4);
let headless=tab.slice(ha,hb);
headless=headless.replace('    let p = palette();','    let access = state.menu_access();\n    let p = palette();');
const control=headless.lastIndexOf('        ui.horizontal(|ui| {',headless.indexOf('egui::RichText::new("Simulation speed")'));
const end=headless.lastIndexOf('    });');
if(control<0||end<control)throw Error('Headless controls');
headless=headless.slice(0,control)+'        if access.allows(390) {\n'+headless.slice(control,end)+'        }\n'+headless.slice(end);
tab=tab.slice(0,ha)+headless+tab.slice(hb);
fs.writeFileSync(tabs,tab);
console.log('Applied top-bar, menu, panel, rail and headless visibility.');

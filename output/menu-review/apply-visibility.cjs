const fs = require('fs');
const file = 'src/ui/tab_viewer.rs';
let source = fs.readFileSync(file, 'utf8');
function range(name) {
  const a = source.indexOf('fn ' + name + '(');
  if (a < 0) throw Error(name);
  let b = source.indexOf('\nfn ', a + 4);
  if (b < 0) b = source.length;
  const chunk = source.slice(a, b);
  return [a, a + chunk.lastIndexOf('\n}') + 2];
}
function edit(name, transform) {
  const [a,b] = range(name);
  source = source.slice(0,a) + transform(source.slice(a,b)) + source.slice(b);
}
function wrap(text, start, end, id) {
  const a = text.indexOf(start), b = end === null ? text.lastIndexOf('\n}') : text.indexOf(end, a + start.length);
  if (a < 0 || b < a) throw Error('Missing section: ' + start + ' → ' + end);
  return text.slice(0,a) + `if access.allows(${id}) {\n` + text.slice(a,b) + '\n}\n' + text.slice(b);
}
function init(text, state='state') {
  const a = text.indexOf('{');
  return text.slice(0,a+1) + `\n    let access = ${state}.menu_access();` + text.slice(a+1);
}

edit('render_cave_system', text => {
  text = init(text);
  const ids = [61,74,111,119,127,142]; let index=0;
  text = text.replaceAll('state.show_advanced_options', () => `access.allows(${ids[index++]})`);
  if (index !== ids.length) throw Error('Cave advanced blocks: '+index);
  text = text.replace('state.cosmetic_options_visible()', 'access.allows(69)');
  text = wrap(text, '            ui.label("Scale:")', '            if access.allows(61)',60);
  text = wrap(text, '            ui.label(if lava_tubes_selected {\n                "Lava Tube Colors"', '            if access.allows(74)',70);
  text = wrap(text, '                ui.label("Length:")', '            if access.allows(119)',108);
  // The vent block must close before the existing geothermal-enabled branch closes.
  // Move our new closing brace inside that branch (debug readouts are outside it).
  text = text.replace(/(\n\s*\}\s*\n)\n\}\n(\s*if access\.allows\(119\))/, '\n}\n$1$2');
  text = wrap(text, '    ui.heading("🌿 Moss");', '    ui.heading("🪨 Mossrocks");',118);
  text = wrap(text, '            // Spawn frequency', '            if access.allows(142)',139);
  return text;
});

edit('render_fluid_settings', text => {
  text = init(text);
  const ids=[149,151,152,157,165,168,174];let n=0;
  text=text.replaceAll('state.show_advanced_options',()=>`access.allows(${ids[n++]})`);
  if(n!==ids.length)throw Error('Fluid advanced blocks: '+n);
  text=text.replace('    if !state.cosmetic_options_visible() { return; }','');
  text=wrap(text,'    let static_water_active =','    // === Nutrients ===',192);
  text=wrap(text,'    ui.label("Epoch Duration:")','    if access.allows(149)',147);
  text=wrap(text,'    ui.heading("Fluid Physics");','    // Condensation and vaporization',151);
  text=wrap(text,'    // === Ice Appearance ===','    if access.allows(174)',158);
  text=text.replace('if access.allows(174)', 'if access.allows(185)');
  text=wrap(text,'        ui.heading("Cubemap Reflection");','        ui.heading("Underwater Light");',174);
  return text;
});

edit('render_light_settings_organized', text => {
  text=init(text);
  text=text.replace('state.cosmetic_options_visible()', 'access.allows(207)');
  text=text.replace('state.cosmetic_options_visible() && context.editor_state.sun_rotation_enabled', 'access.allows(209) && context.editor_state.sun_rotation_enabled');
  text=text.replace('state.cosmetic_options_visible()', 'access.allows(211)');
  text=text.replace('if state.show_advanced_options {\n                    let (mut orbit_pitch', 'if access.allows(214) {\n                    let (mut orbit_pitch');
  text=wrap(text,'                    ui.label("Path Yaw");','                    if orbit_changed {',215);
  text=text.replace('} else {\n                changed |= render_sun_direction_control', '} else if access.allows(216) {\n                changed |= render_sun_direction_control');
  text=text.replace('state.show_advanced_options && context.editor_state.show_sun','access.allows(217) && context.editor_state.show_sun');
  text=wrap(text,'            ui.label("Thermal Inertia (0-5)");','            ui.add_space(6.0);\n            changed |= ui\n                .checkbox(&mut context.editor_state.sun_cycle_enabled',220);
  text=wrap(text,'    egui::CollapsingHeader::new("Luminocyte Lighting")','    // Everything below',224);
  const a=text.indexOf('    if !state.show_advanced_options {');
  const b=text.indexOf('    egui::CollapsingHeader::new("Surfaces & Shadows")',a);
  if(a<0||b<0)throw Error('Lighting early return');
  text=text.slice(0,a)+text.slice(b);
  text=text.replaceAll('state.show_advanced_options','access.allows(228)');
  text=wrap(text,'    egui::CollapsingHeader::new("Surfaces & Shadows")','    egui::CollapsingHeader::new("Luminocyte Glow")',225);
  text=wrap(text,'    egui::CollapsingHeader::new("Luminocyte Glow")','    if access.allows(228) {\n        ui.separator();\n\n        egui::CollapsingHeader::new("Camera Response")',246);
  text=text.replace('if access.allows(228) {\n        ui.separator();\n\n        egui::CollapsingHeader::new("Camera Response")','if access.allows(248) {\n        ui.separator();\n\n        egui::CollapsingHeader::new("Camera Response")');
  text=wrap(text,'                ui.label(egui::RichText::new("Contrast & Eye Adaptation")','            });\n    }\n\n    // Performance section',249);
  text=wrap(text,'    // Performance section','    if changed || sun_changed {',258);
  return text;
});

edit('render_world_settings', text => {
  text=init(text);
  text=text.replaceAll('state.show_advanced_options','access.allows(265)');
  text=wrap(text,'    // Global velocity damping','    ui.add_space(12.0);\n\n    // Biology section',264);
  text=wrap(text,'    // Biology section','    // Mutation section',268);
  text=wrap(text,'    ui.checkbox(&mut world.subtle_mutations','    ui.add_space(6.0);\n    ui.horizontal(|ui| {\n        ui.label("Gene Pool:")',271);
  return text;
});

edit('render_camera_settings', text => {
  text=init(text,'ui_state');
  return wrap(text,'    ui.label("Base Movement Speed:")',null,275);
});

edit('render_performance_monitor', text => {
  text=init(text);
  text=text.replace('state.cosmetic_options_visible()','access.allows(387)');
  text=wrap(text,'    // GPU Readbacks toggle at the top','    // Culling section',383);
  text=wrap(text,'        // Culling toggles','        let (total, visible, frustum_culled, occluded)',385);
  return text;
});

// Previously blanket cosmetic gating must not override approved Show choices.
source=source.replace('if tab.is_cosmetic() && !self.state.cosmetic_options_visible() {\n            ui.label("Enable Advanced Options to edit appearance settings.");','if !self.state.menu_access().allows_panel(*tab) {\n            ui.label("Enable the required settings level to access this panel.");');
source=source.replace('if self.state.cosmetic_options_visible() {\n                    render_organism_skin_settings', 'if self.state.menu_access().allows(193) {\n                    render_organism_skin_settings');
fs.writeFileSync(file,source);
console.log('Applied reviewed visibility to scene settings without changing controls or values.');

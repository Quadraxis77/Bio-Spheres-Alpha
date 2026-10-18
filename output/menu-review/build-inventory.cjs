const fs = require('fs');
const path = require('path');
const sourcePath = path.resolve('src/ui/tab_viewer.rs');
const source = fs.readFileSync(sourcePath, 'utf8');
const groups = [];
function body(name) {
  const start = source.indexOf('fn ' + name + '(');
  if (start < 0) throw new Error(name);
  let end = source.indexOf('\nfn ', start + 4);
  if (end < 0) end = source.length;
  const text = source.slice(start, end);
  return {text: text.slice(0, text.lastIndexOf('\n}') + 2), line: source.slice(0, start).split('\n').length};
}
function group(name, items, defaultChoice = 'advanced') {
  const result = {name, sections: [{name: 'Controls', items: items.map(label => ({label, choice: defaultChoice}))}]};
  groups.push(result);
  return result;
}
function scan(name, functionName, defaultChoice = 'advanced') {
  const fn = body(functionName);
  const events = [];
  function collect(re, type, transform) {
    for (const m of fn.text.matchAll(re)) events.push({at: m.index, type, ...transform(m)});
  }
  collect(/(?:ui\.heading|CollapsingHeader::new)\(\s*"([^"\n]+)"/g, 'section', m => ({label: m[1]}));
  collect(/ui\.label\(egui::RichText::new\("([^"\n]+)"\)\.strong\(\)\)/g, 'section', m => ({label: m[1]}));
  collect(/ui\.label\(\s*(?:egui::RichText::new\()?"([^"\n]+)"/g, 'label', m => ({label: m[1]}));
  collect(/ui\.label\(if[^\{\n]*\{\s*"([^"\n]+)"\s*\}\s*else\s*\{\s*"([^"\n]+)"\s*\}\)/g, 'label', m => ({label: m[1] + ' / ' + m[2]}));
  collect(/Slider::new\(\s*&mut\s+([^,\n]+),/g, 'control', m => ({field: m[1].trim()}));
  collect(/DragValue::new\(\s*&mut\s+([^\)\n]+)\)/g, 'control', m => ({field: m[1].trim()}));
  collect(/\.checkbox\(\s*&mut\s+([^,\n]+),\s*"([^"\n]+)"/g, 'control', m => ({field: m[1].trim(), label: m[2]}));
  collect(/Checkbox::new\(\s*&mut\s+([^,\n]+),\s*"([^"\n]+)"/g, 'control', m => ({field: m[1].trim(), label: m[2]}));
  collect(/TextEdit::singleline\(\s*&mut\s+([^\)\n]+)\)/g, 'control', m => ({field: m[1].trim()}));
  collect(/color_edit_button_(?:rgb|rgba|srgba|srgb)\(\s*&mut\s+([^\)\n]+)\)/g, 'control', m => ({field: m[1].trim()}));
  collect(/ComboBox::from_id_salt\("([^"\n]+)"\)/g, 'control', m => ({field: 'picker:' + m[1]}));
  collect(/(?:\.|::)setting\(ui,\s*"([^"\n]+)"/g, 'label', m => ({label: m[1]}));
  collect(/color_row!\(\s*"([^"\n]+)"/g, 'control', m => ({field: 'theme:' + m[1], label: m[1]}));
  collect(/ui\.(?:small_button|button)\(\s*"([^"\n]+)"/g, 'control', m => ({field: 'button:' + m[1], label: m[1]}));
  collect(/sun_orbit_position_slider\(/g, 'control', () => ({field: 'context.editor_state.sun_orbit_angle', label: 'Position'}));
  collect(/render_sun_direction_control\(/g, 'control', () => ({field: 'context.editor_state.light_dir', label: 'Light Direction'}));
  const result = {name, sections: []};
  let section = {name: 'Controls', items: []};
  result.sections.push(section);
  let label = '';
  const seen = new Set();
  for (const event of events.sort((a, b) => a.at - b.at || (a.type === 'section' ? -1 : 1))) {
    if (event.type === 'section') {
      section = result.sections.find(s => s.name === event.label);
      if (!section) { section = {name: event.label, items: []}; result.sections.push(section); }
      label = event.label;
    } else if (event.type === 'label') label = event.label;
    else {
      const itemLabel = event.label || label;
      if (!itemLabel || itemLabel.length > 110 || event.field.includes('$')) continue;
      const key = section.name + '|' + event.field + '|' + itemLabel;
      if (seen.has(key)) continue;
      seen.add(key);
      section.items.push({label: itemLabel, field: event.field, choice: defaultChoice,
        source: 'src/ui/tab_viewer.rs:' + (fn.line + fn.text.slice(0, event.at).split('\n').length - 1)});
    }
  }
  result.sections = result.sections.filter(s => s.items.length);
  groups.push(result);
  return result;
}

// Names below follow the current UI. This catalog does not alter game files.
group('Top bar and general menus', ['Scene switch', 'Tutorial', 'Help', 'Themes', 'Panels', 'Save Sphere…', 'Load Sphere…', 'Save genome', 'Load genome', 'New genome', 'Genome name', 'UI Scale', 'Frame rate limit', 'Horizontal FOV', 'Speed 1', 'Speed 2', 'Scroll Sensitivity', 'Panel visibility toggles', 'Panel Lock/Unlock', 'Reset to Default (layout)', 'Layout Files: Active / Default'], 'show');
group('Left button rail', ['Generate Procedural Genome', 'Take Screenshot', 'Open Bio-Spheres Folder', 'Advanced Options', 'Lineage Viewer', 'Toggle Water Fill', 'Static Water World (fill world, disable dynamic water physics)', 'Flat Ground (fill lower third with sandy ocean floor)', 'Reset scene settings'], 'show');
group('Panels and windows', ['Viewport', 'Scene Manager', 'Cell Inspector', 'Lineage Viewer', 'Genome Editor', 'Performance', 'Rendering', 'Time Scrubber', 'Theme Editor', 'Camera', 'Lighting', 'Gizmo Rendering', 'Cell Visuals', 'Cave System', 'Fluid System', 'World Settings', 'Modes', 'Adhesion Settings', 'Parent Settings', 'Circle Sliders', 'Child Rotation', 'Time Slider', 'Mode Graph', 'Help', 'Left Panel', 'Right Panel', 'Bottom Panel'], 'show');
scan('Cave System', 'render_cave_system');
scan('Fluid System', 'render_fluid_settings');
scan('Organism Skins', 'render_organism_skin_settings');
scan('Lighting', 'render_light_settings_organized');
scan('World Settings', 'render_world_settings');
scan('Camera', 'render_camera_settings', 'show');
scan('Cell Visuals', 'render_cell_type_visuals', 'show');
scan('Theme Editor', 'render_theme_editor');
scan('Gizmo Rendering', 'render_gizmo_settings');
scan('Performance', 'render_performance_monitor');
scan('Lineage Viewer — Headless Simulation', 'render_headless_section');
group('Genome Editor — authoring panels', ['Modes: cell type, color, mode name and mode selection', 'Adhesion Settings: breaking, spring and connection properties', 'Parent Settings: cell functions, signaling and division rules', 'Circle Sliders: split angles', 'Child Rotation: quaternion/orientation controls', 'Time Slider: preview time and resimulation progress', 'Mode Graph: nodes and mode transitions'], 'show');

function add(panel, section, label, field) {
  const g = groups.find(g => g.name === panel);
  let s = g.sections.find(s => s.name === section);
  if (!s) { s = {name: section, items: []}; g.sections.push(s); }
  if (!s.items.some(i => i.field === field)) s.items.push({label, field, choice: 'advanced'});
}
add('World Settings', 'Physics', 'Gravity Axis: X / Y / Z / Radial', 'world.gravity_mode');
add('World Settings', 'Mutation', 'Gene Pool: cell type selection', 'world.mutation_gene_pool_mask');
add('Lighting', 'Performance', 'Light Field Interval: Every /2 /4 /8', 'world.light_field_update_interval');
add('Lighting', 'Performance', 'Physics Rate: 32 / 48 / 64 Hz', 'world.physics_hz');
add('Fluid System', 'Fluid Spawning', 'Water Fill: ON / OFF', 'editor.fluid_continuous_spawn');
add('Fluid System', 'Fluid Spawning', 'Static Water World: ON / OFF', 'editor.fluid_static_water_world');
add('Performance', 'Controls', 'Debug Colors', 'state.lod_debug_colors');
add('Theme Editor', 'Custom theme', 'Reset to Defaults', 'button:Reset to Defaults');

// Context for compact inputs whose visible Min/Max labels repeat in the source.
const contextualLabels = {
  rmin: 'Radius Range: Min', rmax: 'Radius Range: Max',
  mmin: 'Moss Store Range (nutrients): Min', mmax: 'Moss Store Range (nutrients): Max',
  buoyancy: 'Buoyancy in Water:'
};
for (const g of groups) for (const s of g.sections) for (const i of s.items) {
  if (g.name === 'Cave System' && contextualLabels[i.field]) i.label = contextualLabels[i.field];
}
// The color-row macro definition is an implementation detail, not a visible control.
groups.find(g => g.name === 'Theme Editor').sections = groups.find(g => g.name === 'Theme Editor').sections.filter(s => s.name !== 'Mode');
const cellVisuals = groups.find(g => g.name === 'Cell Visuals');
const cellActions = cellVisuals.sections.flatMap(s => s.items.filter(i => ['Save', 'Reset Type', 'Reset All'].includes(i.label)));
for (const s of cellVisuals.sections) s.items = s.items.filter(i => !cellActions.includes(i));
cellVisuals.sections.push({name: 'Save / Reset', items: cellActions});

// Preselection is a recommendation about UI visibility only, not parameter values.
for (const g of groups) for (const s of g.sections) for (const i of s.items) {
  const f = (i.field || '').toLowerCase(), label = i.label.toLowerCase();
  if (g.name === 'Top bar and general menus') {
    if (/speed 1|speed 2|scroll sensitivity|panel lock|themes/.test(label)) i.choice = 'advanced';
    if (/layout files/.test(label)) i.choice = 'hide';
  } else if (g.name === 'Left button rail') {
    if (/static water|flat ground|open bio/.test(label)) i.choice = 'advanced';
  } else if (g.name === 'Panels and windows') {
    if (/^(rendering|left panel|right panel|bottom panel)$/.test(label)) i.choice = 'hide';
    else if (/theme editor|gizmo rendering/.test(label)) i.choice = 'advanced';
  } else if (g.name === 'Cave System') {
    if (/^(density|scale|seed_text)$/.test(f) || /button:.*random/.test(f)) i.choice = 'show';
    if (/show_moss|show_boulders/.test(f)) i.choice = 'show';
    if (/geothermal_enabled/.test(f)) i.choice = 'show';
    if (/cave_resolution|cave_octaves|mesh_smoothing|isolated_chunk|bump|noise|frequency|lacunarity|grain|parallax/.test(f)) i.choice = 'hide';
    if (/picker:cave_appearance/.test(f)) i.choice = 'advanced';
  } else if (g.name === 'Fluid System') {
    if (/nutrient_density/.test(f) || /water fill/.test(label)) i.choice = 'show';
    if (/vorticity|pressure_iterations|lateral_flow|iso_level|noise_octaves|noise_lacunarity|noise_persistence/.test(f)) i.choice = 'hide';
  } else if (g.name === 'Lighting') {
    if (/sun_intensity|sun_rotation_enabled/.test(f)) i.choice = 'show';
    if (/sun_night_ratio|sun_cycle_period/.test(f)) i.choice = 'show';
    if (/max_steps|step_size|absorption_solid|absorption_cell|fog_steps|fog_composite_blur/.test(f)) i.choice = 'hide';
    if (/^(world\.light_field_update_interval|world\.physics_hz)$/.test(f)) i.choice = 'hide';
  } else if (g.name === 'World Settings') {
    if (/world\.gravity$|world\.gravity_mode$|world\.radiation_level$/.test(f) || label === 'radiation level:') i.choice = 'show';
    if (/constraint_iterations/.test(f)) i.choice = 'hide';
  } else if (g.name === 'Camera') {
    if (/move_speed|camera_sprint_multiplier|camera_alternate_speed_multiplier/.test(f)) i.choice = 'advanced';
  } else if (g.name === 'Cell Visuals') {
    // Preserve the detailed appearance editor the user explicitly asked to retain.
    i.choice = /tail_segments|specular_power/.test(f) ? 'advanced' : 'show';
  } else if (g.name === 'Performance') {
    if (/lod_debug_colors/.test(f)) i.choice = 'hide';
  }
}

// Avoid confusing duplicate labels caused by sliders and value inputs sharing a value.
for (const g of groups) for (const s of g.sections) {
  const labels = new Set();
  s.items = s.items.filter(i => { if (labels.has(i.label)) return false; labels.add(i.label); return true; });
}
let count = 0;
for (const g of groups) for (const s of g.sections) for (const i of s.items) i.id = 'menu-' + String(++count).padStart(4, '0');
const inventory = {groups, total: count};
fs.writeFileSync(path.join(__dirname, 'inventory.json'), JSON.stringify(inventory, null, 2));
for (const g of groups) {
  console.log('\n' + g.name);
  for (const s of g.sections) console.log('  ' + s.name + ': ' + s.items.map(i => i.label + ' [' + i.choice + ']').join(' | '));
}
console.log('\nTOTAL', count);

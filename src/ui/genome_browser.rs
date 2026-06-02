//! Genome browser - a styled floating window for loading genomes.

use std::path::PathBuf;
use egui::{Color32, Rect, Sense, Vec2};
use crate::ui::ui_system::palette;

const CARD_W: f32 = 180.0;
/// Thumbnail is square - matches the 256x256 GIF aspect ratio exactly.
const CARD_THUMB_H: f32 = CARD_W;
/// Info area below the thumbnail: name (14px) + mode count (12px) + tags row (18px) + padding.
const CARD_INFO_H: f32 = 60.0;
const CARD_H: f32 = CARD_THUMB_H + CARD_INFO_H;
const CARD_GAP: f32 = 10.0;
const GIF_FPS: f32 = 20.0;

// -- Thumbnail -----------------------------------------------------------------

pub struct GenomeThumbnail {
    pub frames: Vec<egui::TextureHandle>,
    pub frame_idx: usize,
    pub frame_timer: f32,
    /// Frame to display when not hovered (from the .gif.meta sidecar).
    pub static_frame: usize,
}

impl GenomeThumbnail {
    pub fn load(ctx: &egui::Context, gif_path: &PathBuf) -> Option<Self> {
        let data = std::fs::read(gif_path).ok()?;
        let mut opts = gif::DecodeOptions::new();
        opts.set_color_output(gif::ColorOutput::RGBA);
        let mut decoder = opts.read_info(std::io::Cursor::new(&data)).ok()?;
        let w = decoder.width() as usize;
        let h = decoder.height() as usize;
        let mut frames = Vec::new();
        while let Ok(Some(frame)) = decoder.read_next_frame() {
            let pixels = frame.buffer.to_vec();
            if pixels.len() == w * h * 4 {
                let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &pixels);
                frames.push(ctx.load_texture(
                    format!("gt_{}_{}", gif_path.display(), frames.len()),
                    img,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }
        if frames.is_empty() { return None; }

        // Read the static frame index from the .gif.meta sidecar if it exists.
        let meta_path = gif_path.with_extension("gif.meta");
        let static_frame = if meta_path.exists() {
            std::fs::read_to_string(&meta_path).ok()
                .and_then(|s| {
                    // Parse {"static_frame":N}
                    s.split(':').nth(1)
                        .and_then(|v| v.trim_matches(|c: char| !c.is_ascii_digit()).parse::<usize>().ok())
                })
                .unwrap_or(0)
                .min(frames.len().saturating_sub(1))
        } else {
            0
        };

        Some(Self { frames, frame_idx: static_frame, frame_timer: 0.0, static_frame })
    }

    pub fn advance(&mut self, dt: f32) {
        self.frame_timer += dt;
        let last = self.frames.len().saturating_sub(1);
        // Last frame holds for 1 second before looping back to frame 0.
        let d = if self.frame_idx == last { 1.0 } else { 1.0 / GIF_FPS };
        while self.frame_timer >= d {
            self.frame_timer -= d;
            if self.frame_idx == last {
                self.frame_idx = 0;
            } else {
                self.frame_idx += 1;
            }
        }
    }

    pub fn reset(&mut self) { self.frame_idx = self.static_frame; self.frame_timer = 0.0; }
    pub fn current_tex(&self) -> egui::TextureId { self.frames[self.frame_idx].id() }
}

// -- Entry ---------------------------------------------------------------------

/// Pre-computed stats derived from the genome, shown on the card.
pub struct GenomeStats {
    pub mode_count: usize,
    /// Compact tag strings shown as pills on the card.
    pub tags: Vec<(&'static str, [u8; 3])>,
}

impl GenomeStats {
    pub fn compute(genome: &crate::genome::Genome) -> Self {
        let modes = &genome.modes;
        let mode_count = modes.len();

        // Cell type IDs (from cell/types.rs): 0=Test,1=Lipocyte,2=Phagocyte,3=Flagellocyte,
        // 4=Buoyocyte,5=Glueocyte,6=Ciliocyte,7=Oculocyte,8=Myocyte,9=Devorocyte,
        // 10=Embryocyte,11=Vasculocyte,12=Photocyte
        let has = |id: i32| modes.iter().any(|m| m.cell_type == id);
        let any_bool = |f: fn(&crate::genome::ModeSettings) -> bool| modes.iter().any(|m| f(m));

        let has_photo   = has(12);
        let has_devour  = has(9);
        let has_phage   = has(2);

        let mut tags: Vec<(&'static str, [u8; 3])> = Vec::new();

        // -- Diet classification (always first) -------------------------------
        // All 8 combinations of photo / devour / phage:
        let diet_tag: (&'static str, [u8; 3]) = match (has_photo, has_devour, has_phage) {
            // photo + devour + phage
            (true,  true,  true)  => ("🌿🦷🌾 Apex",       [200, 220,  60]),
            // photo + devour, no phage
            (true,  true,  false) => ("🌿🦷 Photovore",    [180, 220,  80]),
            // photo + phage, no devour
            (true,  false, true)  => ("🌿🌾 Mixotroph",    [100, 220, 140]),
            // photo only
            (true,  false, false) => ("🌿 Autotroph",      [100, 220, 100]),
            // devour + phage, no photo
            (false, true,  true)  => ("🍖 Omnivore",       [220, 160,  60]),
            // devour only
            (false, true,  false) => ("🦷 Carnivore",      [220,  70,  70]),
            // phage only
            (false, false, true)  => ("🌾 Herbivore",      [140, 200, 100]),
            // none of the three
            (false, false, false) => ("🔬 Microbe",        [160, 160, 200]),
        };
        tags.push(diet_tag);

        // -- Trait tags -------------------------------------------------------
        if has(10)  { tags.push(("🥚 Repro",    [100, 200, 120])); }
        if has(7)   { tags.push(("👁 Sense",    [120, 180, 240])); }
        if has(3)   { tags.push(("🏊 Swim",     [80,  160, 220])); }
        if has(8)   { tags.push(("💪 Muscle",   [220, 140,  60])); }
        if has(6)   { tags.push(("〰 Cilia",    [160, 200, 160])); }
        if has(5)   { tags.push(("🔗 Glue",     [200, 160, 100])); }
        if has(4)   { tags.push(("🫧 Buoy",     [100, 180, 220])); }
        if has(11)  { tags.push(("🩸 Vascular", [180,  80, 120])); }
        if has(1)   { tags.push(("🫙 Lipocyte", [180, 160, 100])); }
        if any_bool(|m| m.regulation_emit_channel >= 0) {
            tags.push(("📡 Signal", [160, 120, 220]));
        }

        Self { mode_count, tags }
    }
}

pub struct GenomeEntry {
    pub path: PathBuf,
    pub name: String,
    pub stats: GenomeStats,
    pub thumbnail: Option<GenomeThumbnail>,
    /// File modification time for "Recent" sort (seconds since UNIX epoch, 0 if unavailable).
    pub modified_secs: u64,
}

impl GenomeEntry {
    pub fn load(ctx: &egui::Context, path: PathBuf) -> Self {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Unknown").to_string();
        let stats = match crate::genome::Genome::load_from_file(&path) {
            Ok(g) => GenomeStats::compute(&g),
            Err(_) => GenomeStats { mode_count: 0, tags: vec![] },
        };
        let modified_secs = std::fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let gif_path = path.with_extension("gif");
        let thumbnail = if gif_path.exists() { GenomeThumbnail::load(ctx, &gif_path) } else { None };
        Self { path, name, stats, thumbnail, modified_secs }
    }
}

// -- State ---------------------------------------------------------------------

pub struct GenomeBrowserState {
    pub open: bool,
    pub entries: Vec<GenomeEntry>,
    pub selected: Option<usize>,
    pub search: String,
    pub needs_refresh: bool,
    pub force_full_reload: bool,
    pub confirm_delete: bool,
    pub status_msg: Option<(String, bool)>,
    pub status_timer: f32,
    /// Current sort order.
    pub sort_mode: BrowserSort,
    /// Active tag filter - empty string means no tag filter.
    pub tag_filter: String,
    /// Paths queued for incremental loading (one per frame).
    pending_load: std::collections::VecDeque<PathBuf>,
    /// True while incremental loading is in progress.
    pub is_loading: bool,
    /// Whether to show the "New Genome" confirmation dialog.
    pub confirm_new: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrowserSort {
    /// Alphabetical A->Z (default).
    NameAsc,
    /// Alphabetical Z->A.
    NameDesc,
    /// Most recently modified first.
    Recent,
}

impl Default for BrowserSort {
    fn default() -> Self { BrowserSort::NameAsc }
}

impl Default for GenomeBrowserState {
    fn default() -> Self {
        Self { open: false, entries: Vec::new(), selected: None, search: String::new(),
               needs_refresh: true, force_full_reload: false, confirm_delete: false,
               status_msg: None, status_timer: 0.0,
               sort_mode: BrowserSort::NameAsc, tag_filter: String::new(),
               pending_load: std::collections::VecDeque::new(), is_loading: false,
               confirm_new: false }
    }
}

impl Clone for GenomeBrowserState {
    fn clone(&self) -> Self {
        Self { open: self.open, entries: Vec::new(), selected: self.selected,
               search: self.search.clone(), needs_refresh: true, force_full_reload: false,
               confirm_delete: false,
               status_msg: self.status_msg.clone(), status_timer: self.status_timer,
               sort_mode: self.sort_mode.clone(), tag_filter: self.tag_filter.clone(),
               pending_load: std::collections::VecDeque::new(), is_loading: false,
               confirm_new: false }
    }
}

impl std::fmt::Debug for GenomeBrowserState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenomeBrowserState").field("open", &self.open).finish()
    }
}

impl PartialEq for GenomeBrowserState {
    fn eq(&self, _: &Self) -> bool { false }
}

impl GenomeBrowserState {
    pub fn open_load(&mut self) {
        self.open = true;
        self.needs_refresh = true; // always refresh on open so GIFs are current
        self.selected = None;
        self.search.clear();
        self.tag_filter.clear();
    }

    pub fn refresh(&mut self, _ctx: &egui::Context) {
        self.needs_refresh = false;
        let paths = crate::genome::Genome::list_genomes_dir();
        // Clear existing entries and queue all paths for incremental loading.
        self.entries.clear();
        self.pending_load = paths.into_iter().collect();
        self.is_loading = !self.pending_load.is_empty();
    }

    /// Load one pending entry per call. Call once per frame while `is_loading` is true.
    pub fn tick_load(&mut self, ctx: &egui::Context) {
        if let Some(path) = self.pending_load.pop_front() {
            let mut entry = GenomeEntry::load(ctx, path.clone());
            let gp = path.with_extension("gif");
            if gp.exists() {
                entry.thumbnail = GenomeThumbnail::load(ctx, &gp);
            }
            self.entries.push(entry);
        }
        self.is_loading = !self.pending_load.is_empty();
    }

    pub fn set_status(&mut self, msg: impl Into<String>, is_error: bool) {
        self.status_msg = Some((msg.into(), is_error));
        self.status_timer = 3.0;
    }

    pub fn tick(&mut self, dt: f32) {
        if self.status_timer > 0.0 {
            self.status_timer -= dt;
            if self.status_timer <= 0.0 { self.status_msg = None; }
        }
    }
}

// -- Render --------------------------------------------------------------------

pub fn render_genome_browser(
    ctx: &egui::Context,
    state: &mut GenomeBrowserState,
    genome: &mut crate::genome::Genome,
    editor_state: &mut crate::ui::panel_context::GenomeEditorState,
    dt: f32,
) -> Option<PathBuf> {
    if !state.open { return None; }
    state.tick(dt);
    if state.needs_refresh { state.refresh(ctx); }
    if state.is_loading { state.tick_load(ctx); ctx.request_repaint(); }

    let p = palette();
    let mut result: Option<PathBuf> = None;
    let mut close = false;

    let win_frame = egui::Frame::new()
        .fill(p.bg_darkest)
        .stroke(egui::Stroke::new(1.5, p.accent_primary))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::same(0));

    let screen = ctx.content_rect();
    let max_w = (screen.width() - 40.0).max(500.0);
    let max_h = (screen.height() - 40.0).max(350.0);
    let def_w = max_w.min(900.0);
    let def_h = max_h.min(600.0);

    egui::Window::new("genome_browser")
        .frame(win_frame)
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .default_size([def_w, def_h])
        .min_size([500.0, 350.0])
        .max_size([max_w, max_h])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.set_min_size(ui.available_size());

            // Header
            egui::Frame::new()
                .fill(p.bg_panel)
                .inner_margin(egui::Margin { left: 16, right: 12, top: 10, bottom: 10 })
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("GENOME BROWSER").size(15.0).strong().color(p.text_primary));
                        ui.add_space(6.0);
                        let count_label = if state.is_loading {
                            format!("— loading… ({}/{})", state.entries.len(), state.entries.len() + state.pending_load.len())
                        } else {
                            format!("— {} genomes", state.entries.len())
                        };
                        ui.label(egui::RichText::new(count_label).size(11.0).color(p.text_dim));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.add(egui::Button::new(egui::RichText::new("✕").size(13.0).color(p.text_secondary))
                                .fill(Color32::TRANSPARENT).stroke(egui::Stroke::NONE)
                                .min_size(Vec2::new(24.0, 24.0))).clicked() { close = true; }
                        });
                    });
                });

            ui.add(egui::Separator::default().spacing(0.0));

            // Search + sort/filter bar
            egui::Frame::new()
                .fill(p.bg_panel)
                .inner_margin(egui::Margin { left: 12, right: 12, top: 6, bottom: 6 })
                .show(ui, |ui| {
                    // Row 1: search field
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("🔍").size(12.0).color(p.text_dim));
                        ui.add_space(4.0);
                        let r = ui.add(egui::TextEdit::singleline(&mut state.search)
                            .desired_width(260.0).hint_text("Filter by name or tag…")
                            .font(egui::FontId::proportional(12.0)).text_color(p.text_primary).frame(false));
                        if r.changed() { state.selected = None; }
                        if !state.search.is_empty() {
                            if ui.add(egui::Button::new(egui::RichText::new("✕").size(10.0).color(p.text_dim))
                                .fill(Color32::TRANSPARENT).stroke(egui::Stroke::NONE)).clicked() {
                                state.search.clear(); state.selected = None;
                            }
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.add(egui::Button::new(egui::RichText::new("⟳ Refresh").size(11.0).color(p.text_secondary))
                                .fill(p.bg_widget).stroke(egui::Stroke::new(1.0, p.border_subtle))
                                .corner_radius(egui::CornerRadius::same(3))).clicked() {
                                state.needs_refresh = true;
                            }
                        });
                    });

                    ui.add_space(4.0);

                    // Row 2: sort buttons + tag filter chips
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);

                        // Sort buttons
                        let sorts: &[(&str, BrowserSort, &str)] = &[
                            ("A→Z",    BrowserSort::NameAsc,  "Sort alphabetically A to Z"),
                            ("Z→A",    BrowserSort::NameDesc, "Sort alphabetically Z to A"),
                            ("Recent", BrowserSort::Recent,   "Sort by most recently saved"),
                        ];
                        for (label, mode, tip) in sorts {
                            let active = &state.sort_mode == mode;
                            let btn = egui::Button::new(egui::RichText::new(*label).size(10.5)
                                .color(if active { p.bg_darkest } else { p.text_secondary }))
                                .fill(if active { p.accent_primary } else { p.bg_widget })
                                .stroke(egui::Stroke::new(1.0, if active { p.accent_primary } else { p.border_subtle }))
                                .corner_radius(egui::CornerRadius::same(3))
                                .min_size(Vec2::new(0.0, 18.0));
                            if ui.add(btn).on_hover_text(*tip).clicked() {
                                state.sort_mode = mode.clone();
                            }
                        }

                        ui.add(egui::Separator::default().vertical().spacing(4.0));

                        // Collect unique diet tags across all entries for quick-filter chips
                        // Exclude tags that aren't useful as standalone filters.
                        const EXCLUDED_TAGS: &[&str] = &["mixotroph", "herbivore"];
                        let diet_tags: Vec<String> = {
                            let mut seen = std::collections::HashSet::new();
                            state.entries.iter()
                                .filter_map(|e| e.stats.tags.first())
                                .map(|(label, _)| {
                                    label.trim_start_matches(|c: char| !c.is_alphabetic()).trim().to_string()
                                })
                                .filter(|t| {
                                    let lower = t.to_lowercase();
                                    !EXCLUDED_TAGS.contains(&lower.as_str()) && seen.insert(t.clone())
                                })
                                .collect()
                        };

                        for tag in &diet_tags {
                            let active = state.tag_filter == *tag;
                            let btn = egui::Button::new(egui::RichText::new(tag).size(10.0)
                                .color(if active { p.bg_darkest } else { p.text_dim }))
                                .fill(if active { p.accent_primary } else { Color32::TRANSPARENT })
                                .stroke(egui::Stroke::new(1.0, if active { p.accent_primary } else { p.border_subtle }))
                                .corner_radius(egui::CornerRadius::same(3))
                                .min_size(Vec2::new(0.0, 18.0));
                            if ui.add(btn).on_hover_text(format!("Show only {} genomes", tag)).clicked() {
                                if active {
                                    state.tag_filter.clear();
                                } else {
                                    state.tag_filter = tag.clone();
                                }
                                state.selected = None;
                            }
                        }

                        // Clear tag filter chip
                        if !state.tag_filter.is_empty() {
                            if ui.add(egui::Button::new(egui::RichText::new("✕ Clear").size(10.0).color(p.text_dim))
                                .fill(Color32::TRANSPARENT).stroke(egui::Stroke::NONE)).clicked() {
                                state.tag_filter.clear();
                                state.selected = None;
                            }
                        }
                    });
                });

            ui.add(egui::Separator::default().spacing(0.0));

            // Card grid - filter then sort
            let search_lower = state.search.to_lowercase();
            let tag_lower = state.tag_filter.to_lowercase();
            let mut vis: Vec<usize> = state.entries.iter().enumerate()
                .filter(|(_, e)| {
                    // Text search
                    if !search_lower.is_empty() {
                        let name_match = e.name.to_lowercase().contains(&search_lower);
                        let tag_match = e.stats.tags.iter().any(|(label, _)| {
                            label.trim_start_matches(|c: char| !c.is_alphabetic()).to_lowercase().contains(&search_lower)
                        });
                        if !name_match && !tag_match { return false; }
                    }
                    // Tag filter chip
                    if !tag_lower.is_empty() {
                        let has_tag = e.stats.tags.iter().any(|(label, _)| {
                            label.trim_start_matches(|c: char| !c.is_alphabetic()).to_lowercase() == tag_lower
                        });
                        if !has_tag { return false; }
                    }
                    true
                })
                .map(|(i, _)| i).collect();

            // Apply sort
            match state.sort_mode {
                BrowserSort::NameAsc  => vis.sort_by(|&a, &b| state.entries[a].name.to_lowercase().cmp(&state.entries[b].name.to_lowercase())),
                BrowserSort::NameDesc => vis.sort_by(|&a, &b| state.entries[b].name.to_lowercase().cmp(&state.entries[a].name.to_lowercase())),
                BrowserSort::Recent   => vis.sort_by(|&a, &b| state.entries[b].modified_secs.cmp(&state.entries[a].modified_secs)),
            }

            let strip_h = 52.0;
            let grid_h = ui.available_height() - strip_h;

            egui::ScrollArea::vertical()
                .id_salt("gbscroll")
                .max_height(grid_h)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if vis.is_empty() {
                        ui.add_space(40.0);
                        ui.vertical_centered(|ui| {
                            ui.label(egui::RichText::new(if state.search.is_empty() {
                                "No genomes found.\nSave a genome to see it here."
                            } else { "No genomes match your search." }).size(13.0).color(p.text_dim));
                        });
                        return;
                    }

                    let pad = 14.0;
                    let cols = ((ui.available_width() - pad * 2.0 + CARD_GAP) / (CARD_W + CARD_GAP)).floor().max(1.0) as usize;
                    let mut clicked: Option<usize> = None;
                    let mut dbl: Option<usize> = None;
                    let mut hovered: Vec<usize> = Vec::new();
                    let delete_idx: Option<usize> = None;
                    let _ = &delete_idx; // reserved for future delete action

                    ui.add_space(pad);
                    for chunk in vis.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.add_space(pad);
                            for &idx in chunk {
                                let sel = state.selected == Some(idx);
                                let (r, resp) = ui.allocate_exact_size(Vec2::new(CARD_W, CARD_H), Sense::click());
                                let is_hovered = resp.hovered();
                                if is_hovered { hovered.push(idx); }
                                if resp.clicked() { clicked = Some(idx); }
                                if resp.double_clicked() { dbl = Some(idx); }
                                draw_card(ui, r, &state.entries[idx], sel, is_hovered, p);

                                ui.add_space(CARD_GAP);
                            }
                        });
                        ui.add_space(CARD_GAP);
                    }
                    ui.add_space(pad);

                    for &i in &hovered { if let Some(t) = state.entries[i].thumbnail.as_mut() { t.advance(dt); } }
                    for (i, e) in state.entries.iter_mut().enumerate() {
                        if !hovered.contains(&i) { if let Some(t) = e.thumbnail.as_mut() { t.reset(); } }
                    }
                    if !hovered.is_empty() { ui.ctx().request_repaint(); }
                    if let Some(i) = clicked { state.selected = Some(i); }
                    if let Some(i) = dbl { state.selected = Some(i); result = Some(state.entries[i].path.clone()); close = true; }

                    // Process delete after the loop (can't mutate entries while iterating)
                    if let Some(i) = delete_idx {
                        let path = state.entries[i].path.clone();
                        // Delete .genome, .gif, and .gif.meta sidecars
                        let _ = std::fs::remove_file(&path);
                        let _ = std::fs::remove_file(path.with_extension("gif"));
                        let _ = std::fs::remove_file(path.with_extension("gif.meta"));
                        // Clear selection if we deleted the selected entry
                        if state.selected == Some(i) { state.selected = None; }
                        state.needs_refresh = true;
                        state.force_full_reload = true;
                    }
                });

            // Action strip
            ui.add(egui::Separator::default().spacing(0.0));
            egui::Frame::new()
                .fill(p.bg_panel)
                .inner_margin(egui::Margin { left: 16, right: 16, top: 10, bottom: 10 })
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if let Some((msg, is_err)) = &state.status_msg {
                            let c = if *is_err { p.status_err } else { p.status_ok };
                            let a = (state.status_timer.min(1.0) * 255.0) as u8;
                            ui.label(egui::RichText::new(msg).size(11.0)
                                .color(Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)));
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.add(egui::Button::new(egui::RichText::new("Cancel").size(12.0).color(p.text_secondary))
                                .fill(p.bg_widget).stroke(egui::Stroke::new(1.0, p.border_normal))
                                .min_size(Vec2::new(80.0, 28.0)).corner_radius(egui::CornerRadius::same(4))).clicked() {
                                close = true;
                            }
                            ui.add_space(8.0);
                            let can = state.selected.is_some();
                            if ui.add_enabled(can, egui::Button::new(
                                egui::RichText::new("📂  Load").size(12.0).strong()
                                    .color(if can { p.bg_darkest } else { p.text_dim }))
                                .fill(if can { p.accent_primary } else { p.bg_widget })
                                .stroke(egui::Stroke::new(1.0, if can { p.accent_primary } else { p.border_subtle }))
                                .min_size(Vec2::new(90.0, 28.0)).corner_radius(egui::CornerRadius::same(4))).clicked() {
                                if let Some(idx) = state.selected {
                                    let path = state.entries[idx].path.clone();
                                    match crate::genome::Genome::load_from_file(&path) {
                                        Ok(loaded) => {
                                            *genome = loaded;
                                            editor_state.selected_mode_index = 0;
                                            editor_state.selected_mode_indices = vec![0];
                                            editor_state.genome_just_loaded = true;
                                            result = Some(path);
                                            close = true;
                                        }
                                        Err(e) => state.set_status(format!("Load failed: {}", e), true),
                                    }
                                }
                            }
                            ui.add_space(8.0);
                            // Delete button - left side of action strip (right-to-left layout, so add last)
                            if ui.add_enabled(can, egui::Button::new(
                                egui::RichText::new("🗑  Delete").size(12.0)
                                    .color(if can { Color32::from_rgb(220, 80, 80) } else { p.text_dim }))
                                .fill(p.bg_widget)
                                .stroke(egui::Stroke::new(1.0, if can { Color32::from_rgb(180, 60, 60) } else { p.border_subtle }))
                                .min_size(Vec2::new(90.0, 28.0)).corner_radius(egui::CornerRadius::same(4)))
                                .on_hover_text("Delete the selected genome and its thumbnail")
                                .clicked() {
                                state.confirm_delete = true;
                            }
                        });
                    });
                });
        });

    if close { state.open = false; }

    // Confirmation popup for delete
    if state.confirm_delete {
        if let Some(idx) = state.selected {
            let name = state.entries.get(idx).map(|e| e.name.as_str()).unwrap_or("this genome");
            let mut do_delete = false;
            let mut cancel = false;

            let popup_frame = egui::Frame::new()
                .fill(p.bg_darkest)
                .stroke(egui::Stroke::new(1.5, Color32::from_rgb(180, 60, 60)))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(20));

            egui::Window::new("confirm_delete_genome")
                .frame(popup_frame)
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new("Delete Genome?").size(14.0).strong().color(Color32::from_rgb(220, 80, 80)));
                        ui.add_space(8.0);
                        ui.label(egui::RichText::new(format!("\"{}\" will be permanently deleted.", name))
                            .size(12.0).color(p.text_primary));
                        ui.label(egui::RichText::new("This cannot be undone.")
                            .size(11.0).color(p.text_dim));
                        ui.add_space(14.0);
                        ui.horizontal(|ui| {
                            if ui.add(egui::Button::new(egui::RichText::new("Cancel").size(12.0).color(p.text_secondary))
                                .fill(p.bg_widget).stroke(egui::Stroke::new(1.0, p.border_normal))
                                .min_size(Vec2::new(80.0, 28.0)).corner_radius(egui::CornerRadius::same(4))).clicked() {
                                cancel = true;
                            }
                            ui.add_space(8.0);
                            if ui.add(egui::Button::new(egui::RichText::new("🗑  Delete").size(12.0).strong().color(Color32::WHITE))
                                .fill(Color32::from_rgb(180, 40, 40))
                                .stroke(egui::Stroke::new(1.0, Color32::from_rgb(220, 60, 60)))
                                .min_size(Vec2::new(90.0, 28.0)).corner_radius(egui::CornerRadius::same(4))).clicked() {
                                do_delete = true;
                            }
                        });
                    });
                });

            if do_delete {
                let path = state.entries[idx].path.clone();
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(path.with_extension("gif"));
                let _ = std::fs::remove_file(path.with_extension("gif.meta"));
                if state.selected == Some(idx) { state.selected = None; }
                state.needs_refresh = true;
                state.force_full_reload = true;
                state.confirm_delete = false;
            }
            if cancel { state.confirm_delete = false; }
        } else {
            state.confirm_delete = false;
        }
    }

    result
}

// -- Card drawing --------------------------------------------------------------

fn draw_card(ui: &mut egui::Ui, rect: Rect, entry: &GenomeEntry, selected: bool, hovered: bool, p: crate::ui::ui_system::ActivePalette) {
    let painter = ui.painter();

    let bg = if selected { p.bg_selected } else if hovered { p.bg_hover } else { p.bg_panel };
    painter.rect_filled(rect, egui::CornerRadius::same(6), bg);

    let (bc, bw) = if selected { (p.accent_primary, 1.5) } else if hovered { (p.border_normal, 1.0) } else { (p.border_subtle, 1.0) };
    painter.rect_stroke(rect, egui::CornerRadius::same(6), egui::Stroke::new(bw, bc), egui::StrokeKind::Inside);

    if selected {
        let glow = Color32::from_rgba_unmultiplied(p.accent_primary.r(), p.accent_primary.g(), p.accent_primary.b(), 30);
        painter.rect_stroke(rect.expand(2.0), egui::CornerRadius::same(8), egui::Stroke::new(2.0, glow), egui::StrokeKind::Outside);
    }

    // Thumbnail - square, fills the top of the card edge-to-edge (1px inset for border).
    let thumb = Rect::from_min_size(
        rect.min + Vec2::new(1.0, 1.0),
        Vec2::splat(CARD_THUMB_H - 2.0),
    );
    if let Some(t) = &entry.thumbnail {
        painter.image(
            t.current_tex(),
            thumb,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        painter.rect_filled(thumb, egui::CornerRadius::same(5), p.bg_widget);
        painter.text(thumb.center(), egui::Align2::CENTER_CENTER, "🧬", egui::FontId::proportional(32.0), p.text_dim);
    }

    // Divider.
    let div_y = rect.min.y + CARD_THUMB_H;
    painter.line_segment(
        [egui::pos2(rect.left() + 8.0, div_y), egui::pos2(rect.right() - 8.0, div_y)],
        egui::Stroke::new(1.0, p.border_subtle),
    );

    // Name.
    let iy = div_y + 7.0;
    painter.text(
        egui::pos2(rect.left() + 10.0, iy),
        egui::Align2::LEFT_TOP,
        &entry.name,
        egui::FontId::proportional(12.0),
        if selected { p.accent_primary } else { p.text_primary },
    );

    // Mode count - right-aligned on the same row as the name.
    let mode_str = format!("{} modes", entry.stats.mode_count);
    painter.text(
        egui::pos2(rect.right() - 8.0, iy),
        egui::Align2::RIGHT_TOP,
        &mode_str,
        egui::FontId::proportional(9.0),
        p.text_dim,
    );

    // Tag pills - wrap across up to 2 rows.
    let tag_font = egui::FontId::proportional(8.5);
    let pill_h = 13.0;
    let pill_pad_x = 4.0;
    let pill_gap = 3.0;
    let tag_x0 = rect.left() + 8.0;
    let tag_max_x = rect.right() - 8.0;
    let mut tx = tag_x0;
    let mut ty = iy + 18.0;
    let max_rows = 2;
    let mut row = 0;

    for (label, rgb) in &entry.stats.tags {
        // Estimate pill width from character count (monospace approximation).
        let char_w = 7.0_f32;
        let pill_w = label.chars().count() as f32 * char_w * 0.62 + pill_pad_x * 2.0 + 2.0;

        if tx + pill_w > tag_max_x {
            row += 1;
            if row >= max_rows { break; }
            tx = tag_x0;
            ty += pill_h + pill_gap;
        }

        let pill_rect = Rect::from_min_size(
            egui::pos2(tx, ty),
            Vec2::new(pill_w, pill_h),
        );
        let pill_bg = Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], 35);
        let pill_border = Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], 120);
        painter.rect_filled(pill_rect, egui::CornerRadius::same(3), pill_bg);
        painter.rect_stroke(pill_rect, egui::CornerRadius::same(3), egui::Stroke::new(0.8, pill_border), egui::StrokeKind::Inside);
        painter.text(
            pill_rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            tag_font.clone(),
            Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
        );

        tx += pill_w + pill_gap;
    }
}

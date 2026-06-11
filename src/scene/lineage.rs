//! Ecosystem lineage archive.
//!
//! This is the lightweight, save-file-friendly history layer for the future
//! lineage viewer. It intentionally stores compact species metadata and bounded
//! bookmarks instead of full simulation state. Full world state still belongs to
//! `GpuSceneSnapshot`.

use crate::genome::Genome;
use serde::{Deserialize, Serialize};

pub type LineageId = u64;
pub type LineageEventId = u64;

const CURRENT_VERSION: u32 = 1;
pub const LINEAGE_CAPTURE_INTERVAL_SECONDS: f32 = 30.0;
const DEFAULT_MAX_NODES: usize = 2_000;
const DEFAULT_MAX_EVENTS: usize = 8_000;
const DEFAULT_MAX_BOOKMARKS: usize = 256;
const DEFAULT_MAX_TIME_GAPS: usize = 1_024;
const DEFAULT_MAX_MINOR_VARIANT_STATS: usize = 512;

/// Fraction change in total live cells below which a scan interval is considered stable.
pub const STABLE_LIVE_CELL_CHANGE_THRESHOLD: f32 = 0.12;

/// Resolution of the GPU scene thumbnail captured for each lineage node.
///
/// 16:9 aspect ratio so the image fills the specimen panel (which is also
/// constrained to 16:9) without any cropping or UV tricks.  The same
/// texture is displayed cover-cropped in the small timeline cards.
pub const LINEAGE_THUMBNAIL_WIDTH: u32 = 448;
pub const LINEAGE_THUMBNAIL_HEIGHT: u32 = 252;

/// Size policy for lineage data embedded in `.sphere` biosphere saves.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageRetentionPolicy {
    /// Maximum lineage/species nodes retained in detail.
    pub max_nodes: usize,
    /// Maximum timeline events retained.
    pub max_events: usize,
    /// Maximum full genome payloads retained for loading into Preview.
    ///
    /// The lineage tree can keep older cards and stats, but full YAML genome
    /// payloads roll off from the earliest snapshots first to keep saves small.
    pub max_bookmarks: usize,
    /// Never prune living branches at or above this peak population unless the
    /// archive is still over budget after all expendable branches are removed.
    pub protect_peak_population_at_least: u32,
    /// Maximum number of time-gap records retained.
    pub max_time_gaps: usize,
}

impl Default for LineageRetentionPolicy {
    fn default() -> Self {
        Self {
            max_nodes: DEFAULT_MAX_NODES,
            max_events: DEFAULT_MAX_EVENTS,
            max_bookmarks: DEFAULT_MAX_BOOKMARKS,
            protect_peak_population_at_least: 10,
            max_time_gaps: DEFAULT_MAX_TIME_GAPS,
        }
    }
}

/// Where a lineage came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LineageOrigin {
    UserInserted,
    ProceduralSeed,
    Mutation {
        parent: LineageId,
        mutation_event: Option<LineageEventId>,
    },
    Hybrid {
        parent_a: LineageId,
        parent_b: LineageId,
        similarity: f32,
    },
    Unknown,
}

impl Default for LineageOrigin {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Compact trait tags used by the dossier UI and for search/filtering.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageTraitSummary {
    pub mode_count: u32,
    pub has_phagocyte: bool,
    pub has_photocyte: bool,
    pub has_devorocyte: bool,
    pub has_flagellocyte: bool,
    pub has_ciliocyte: bool,
    pub has_myocyte: bool,
    pub has_glueocyte: bool,
    pub has_buoyocyte: bool,
    pub has_vasculocyte: bool,
    pub has_gametocyte: bool,
    pub has_cognocyte: bool,
    pub has_memorocyte: bool,
    pub uses_signals: bool,
    pub uses_scaffolds: bool,
}

impl LineageTraitSummary {
    pub fn from_genome(genome: &Genome) -> Self {
        let mut summary = Self {
            mode_count: genome.modes.len() as u32,
            uses_scaffolds: !genome.scaffold_rules.is_empty(),
            ..Self::default()
        };

        for mode in &genome.modes {
            match crate::cell::types::CellType::from_index(mode.cell_type as u32) {
                Some(crate::cell::types::CellType::Phagocyte) => summary.has_phagocyte = true,
                Some(crate::cell::types::CellType::Photocyte) => summary.has_photocyte = true,
                Some(crate::cell::types::CellType::Devorocyte) => summary.has_devorocyte = true,
                Some(crate::cell::types::CellType::Flagellocyte) => summary.has_flagellocyte = true,
                Some(crate::cell::types::CellType::Ciliocyte) => summary.has_ciliocyte = true,
                Some(crate::cell::types::CellType::Myocyte) => summary.has_myocyte = true,
                Some(crate::cell::types::CellType::Glueocyte) => summary.has_glueocyte = true,
                Some(crate::cell::types::CellType::Buoyocyte) => summary.has_buoyocyte = true,
                Some(crate::cell::types::CellType::Vasculocyte) => summary.has_vasculocyte = true,
                Some(crate::cell::types::CellType::Gametocyte) => summary.has_gametocyte = true,
                Some(crate::cell::types::CellType::Cognocyte) => summary.has_cognocyte = true,
                Some(crate::cell::types::CellType::Memorocyte) => summary.has_memorocyte = true,
                _ => {}
            }

            summary.uses_signals |= mode.regulation_emit_channel >= 0
                || mode.division_signal_channel >= 0
                || mode.apoptosis_signal_channel >= 0
                || mode.mode_switch_signal_channel >= 0
                || mode.signal_child_a_channel >= 0
                || mode.signal_child_b_channel >= 0
                || mode.photocyte_emit_enabled
                || mode.lipocyte_emit_enabled;
        }

        summary
    }
}

/// Compact cell pose for an adult-stage organism snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageAdultCellSnapshot {
    pub position: [f32; 3],
    pub radius: f32,
    pub mode_index: u16,
    pub cell_type: u8,
    pub color: [f32; 3],
    pub emissive: f32,
}

impl Default for LineageAdultCellSnapshot {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            radius: 1.0,
            mode_index: 0,
            cell_type: 0,
            color: [0.35, 0.75, 0.85],
            emissive: 0.0,
        }
    }
}

/// Save-friendly morphology capture of a genome's adult/reproductive pose.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageAdultSnapshot {
    pub genome_hash: u64,
    pub captured_time: f32,
    /// Simulation frame this snapshot was captured at (0 for legacy saves).
    pub captured_frame: i32,
    pub captured_before_division: bool,
    pub world_center: [f32; 3],
    pub world_radius: f32,
    /// Compressed RGBA thumbnail captured for the lineage UI.
    ///
    /// Older saves may not have this. The UI falls back to the compact cell
    /// pose data when absent.
    pub scene_thumbnail_png: Option<Vec<u8>>,
    pub cells: Vec<LineageAdultCellSnapshot>,
    pub bonds: Vec<[u16; 2]>,
}

impl Default for LineageAdultSnapshot {
    fn default() -> Self {
        Self {
            genome_hash: 0,
            captured_time: 0.0,
            captured_frame: 0,
            captured_before_division: false,
            world_center: [0.0; 3],
            world_radius: 1.0,
            scene_thumbnail_png: None,
            cells: Vec::new(),
            bonds: Vec::new(),
        }
    }
}

/// One species/genome branch in the ecosystem tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageNode {
    pub id: LineageId,
    pub genome_id: u32,
    pub display_name: String,
    pub origin: LineageOrigin,
    pub parent_a: Option<LineageId>,
    pub parent_b: Option<LineageId>,
    pub first_frame: i32,
    pub last_seen_frame: i32,
    pub extinct_frame: Option<i32>,
    pub current_cells: u32,
    pub current_organisms: u32,
    pub peak_cells: u32,
    pub peak_organisms: u32,
    pub total_birth_events: u32,
    pub total_death_events: u32,
    pub mutation_count: u32,
    pub pinned: bool,
    pub noteworthy_score: f32,
    pub traits: LineageTraitSummary,
    /// Legacy single-snapshot field kept for deserializing old saves.
    /// New code should use `snapshots` instead.
    #[serde(default)]
    pub adult_snapshot: Option<LineageAdultSnapshot>,
    /// All captured snapshots for this lineage, sorted by `captured_frame`.
    #[serde(default)]
    pub snapshots: Vec<LineageAdultSnapshot>,
}

impl Default for LineageNode {
    fn default() -> Self {
        Self {
            id: 0,
            genome_id: 0,
            display_name: String::new(),
            origin: LineageOrigin::Unknown,
            parent_a: None,
            parent_b: None,
            first_frame: 0,
            last_seen_frame: 0,
            extinct_frame: None,
            current_cells: 0,
            current_organisms: 0,
            peak_cells: 0,
            peak_organisms: 0,
            total_birth_events: 0,
            total_death_events: 0,
            mutation_count: 0,
            pinned: false,
            noteworthy_score: 0.0,
            traits: LineageTraitSummary::default(),
            adult_snapshot: None,
            snapshots: Vec::new(),
        }
    }
}

impl LineageNode {
    /// Returns the snapshot closest to `frame`, or None if no snapshots exist.
    pub fn snapshot_near_frame(&self, frame: i32) -> Option<&LineageAdultSnapshot> {
        if self.snapshots.is_empty() {
            return self.adult_snapshot.as_ref();
        }
        self.snapshots
            .iter()
            .min_by_key(|s| (s.captured_frame - frame).abs())
    }

    /// Returns the most recent snapshot.
    pub fn latest_snapshot(&self) -> Option<&LineageAdultSnapshot> {
        if self.snapshots.is_empty() {
            return self.adult_snapshot.as_ref();
        }
        self.snapshots.last()
    }
}

/// Timeline event for the bestiary/evolution graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageEvent {
    pub id: LineageEventId,
    pub frame: i32,
    pub lineage_id: LineageId,
    pub kind: LineageEventKind,
    pub title: String,
    pub detail: String,
    pub impact_score: f32,
    pub noteworthy: bool,
}

impl Default for LineageEvent {
    fn default() -> Self {
        Self {
            id: 0,
            frame: 0,
            lineage_id: 0,
            kind: LineageEventKind::Note,
            title: String::new(),
            detail: String::new(),
            impact_score: 0.0,
            noteworthy: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineageEventKind {
    Inserted,
    Mutation,
    Hybridization,
    PopulationBoom,
    NearExtinction,
    Extinction,
    Dominance,
    Bookmark,
    Note,
}

/// Compact population aggregate for one lineage. Produced by a future
/// throttled GPU aggregation pass, not by scanning every cell on the CPU.
#[derive(Debug, Clone, Copy, Default)]
pub struct LineagePopulationSample {
    pub lineage_id: LineageId,
    pub cells: u32,
    pub organisms: u32,
    pub frame: i32,
}

/// Aggregated telemetry for mutation genomes that were culled instead of
/// promoted into lineage nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageMinorVariantStats {
    pub parent_genome_id: Option<u32>,
    pub first_seen_frame: i32,
    pub last_seen_frame: i32,
    /// Number of culled minor variant observations in this bucket. This is not
    /// deduplicated by genome id because those ids are intentionally discarded.
    pub variant_observations: u32,
    /// Number of scan observations accumulated for this bucket.
    pub observations: u32,
    /// Live cells seen in the most recent scan for this bucket.
    pub current_cells: u32,
    /// Living organisms seen in the most recent scan for this bucket.
    pub current_organisms: u32,
    pub peak_cells: u32,
    pub peak_organisms: u32,
    pub total_cell_observations: u64,
    pub total_organism_observations: u64,
}

impl LineageMinorVariantStats {
    fn record(&mut self, parent_genome_id: Option<u32>, cells: u32, organisms: u32, frame: i32) {
        self.parent_genome_id = parent_genome_id.or(self.parent_genome_id);
        if self.observations == 0 {
            self.first_seen_frame = frame;
        }
        if self.last_seen_frame != frame {
            self.current_cells = 0;
            self.current_organisms = 0;
        }
        self.last_seen_frame = frame;
        self.variant_observations = self.variant_observations.saturating_add(1);
        self.observations = self.observations.saturating_add(1);
        self.current_cells = self.current_cells.saturating_add(cells);
        self.current_organisms = self.current_organisms.saturating_add(organisms);
        self.peak_cells = self.peak_cells.max(cells);
        self.peak_organisms = self.peak_organisms.max(organisms);
        self.total_cell_observations = self.total_cell_observations.saturating_add(cells as u64);
        self.total_organism_observations = self
            .total_organism_observations
            .saturating_add(organisms as u64);
    }
}

/// Optional full-genome record for a lineage node.
///
/// Most nodes should reference `GpuSceneSnapshot::genomes_yaml[genome_id]`.
/// `genome_yaml` is only for promoted/pinned future cases where the genome is
/// no longer present in the scene genome table or has been reconstructed from a
/// GPU-native mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageGenomeBookmark {
    pub lineage_id: LineageId,
    pub genome_id: u32,
    pub saved_frame: i32,
    pub display_name: String,
    pub reason: String,
    pub genome_yaml: Option<String>,
}

impl Default for LineageGenomeBookmark {
    fn default() -> Self {
        Self {
            lineage_id: 0,
            genome_id: 0,
            saved_frame: 0,
            display_name: String::new(),
            reason: String::new(),
            genome_yaml: None,
        }
    }
}

impl LineageGenomeBookmark {
    pub fn is_loadable(&self) -> bool {
        self.genome_yaml.is_some()
    }
}

/// A compressed/skipped time region on the evolution timeline.
///
/// Recorded when consecutive scans show no meaningful population changes —
/// no new lineages, no extinctions, no noteworthy events, and live-cell count
/// within `STABLE_LIVE_CELL_CHANGE_THRESHOLD` of the prior scan.  The UI renders
/// these as narrow hatched "gap" bands instead of proportional timeline space.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageTimeGap {
    /// First frame of the stable/compressed period.
    pub start_frame: i32,
    /// Exclusive end frame of the stable/compressed period.
    pub end_frame: i32,
    /// Simulation time (seconds) at the start of the gap.
    pub start_time: f32,
    /// Simulation time (seconds) at the end of the gap.
    pub end_time: f32,
    /// Approximate live-cell count during the gap (snapshot at gap start).
    pub live_cells: u32,
}

impl Default for LineageTimeGap {
    fn default() -> Self {
        Self {
            start_frame: 0,
            end_frame: 0,
            start_time: 0.0,
            end_time: 0.0,
            live_cells: 0,
        }
    }
}

impl LineageTimeGap {
    pub fn duration_seconds(&self) -> f32 {
        (self.end_time - self.start_time).max(0.0)
    }

    pub fn frame_span(&self) -> i32 {
        (self.end_frame - self.start_frame).max(0)
    }
}

/// Serializable lineage state embedded into biosphere saves.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EcosystemLineageArchive {
    pub version: u32,
    pub next_lineage_id: LineageId,
    pub next_event_id: LineageEventId,
    /// Frame of the last explicit bestiary scan. `None` means population data
    /// has not been captured yet.
    pub last_scan_frame: Option<i32>,
    /// Simulation time in seconds for the last explicit bestiary scan.
    pub last_scan_time: f32,
    /// Authoritative live-cell count from the GPU cell counter at last scan.
    pub last_scan_live_cells: u32,
    /// Live cells that could be assigned to a lineage node at last scan.
    pub last_scan_tracked_cells: u32,
    /// Live cells skipped because their genome id was invalid or otherwise
    /// unavailable in the lineage archive at last scan.
    pub last_scan_untracked_cells: u32,
    /// Whether organism/body counts came from usable connected-component labels.
    pub last_scan_organism_counts_reliable: bool,
    pub retention: LineageRetentionPolicy,
    pub nodes: Vec<LineageNode>,
    pub events: Vec<LineageEvent>,
    pub genome_bookmarks: Vec<LineageGenomeBookmark>,
    pub minor_variant_stats: Vec<LineageMinorVariantStats>,
    /// Compressed time periods with no meaningful ecosystem changes.
    /// Used by the timeline UI to collapse stable intervals into narrow gap bands.
    pub time_gaps: Vec<LineageTimeGap>,
}

impl Default for EcosystemLineageArchive {
    fn default() -> Self {
        Self {
            version: CURRENT_VERSION,
            next_lineage_id: 1,
            next_event_id: 1,
            last_scan_frame: None,
            last_scan_time: 0.0,
            last_scan_live_cells: 0,
            last_scan_tracked_cells: 0,
            last_scan_untracked_cells: 0,
            last_scan_organism_counts_reliable: false,
            retention: LineageRetentionPolicy::default(),
            nodes: Vec::new(),
            events: Vec::new(),
            genome_bookmarks: Vec::new(),
            minor_variant_stats: Vec::new(),
            time_gaps: Vec::new(),
        }
    }
}

impl EcosystemLineageArchive {
    pub fn lineage_for_genome_id(&self, genome_id: u32) -> Option<LineageId> {
        self.nodes
            .iter()
            .find(|n| n.genome_id == genome_id)
            .map(|n| n.id)
    }

    pub fn ensure_user_lineage(
        &mut self,
        genome_id: u32,
        genome: &Genome,
        frame: i32,
    ) -> LineageId {
        if let Some(node) = self
            .nodes
            .iter()
            .find(|n| n.genome_id == genome_id && matches!(n.origin, LineageOrigin::UserInserted))
        {
            return node.id;
        }

        let id = self.allocate_lineage_id();
        self.nodes.push(LineageNode {
            id,
            genome_id,
            display_name: genome.name.clone(),
            origin: LineageOrigin::UserInserted,
            first_frame: frame,
            last_seen_frame: frame,
            traits: LineageTraitSummary::from_genome(genome),
            ..LineageNode::default()
        });
        self.push_event(
            frame,
            id,
            LineageEventKind::Inserted,
            "Genome inserted",
            format!("{} entered the biosphere.", genome.name),
            0.1,
            false,
        );
        self.enforce_retention();
        id
    }

    /// Push a snapshot for the given genome at `frame`, preserving history.
    /// Deduplicates by frame; keeps the most recent 32 snapshots per node.
    pub fn push_adult_snapshot_for_genome(
        &mut self,
        genome_id: u32,
        frame: i32,
        mut snapshot: LineageAdultSnapshot,
    ) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.genome_id == genome_id) {
            snapshot.captured_frame = frame;
            node.snapshots.retain(|s| s.captured_frame != frame);
            node.snapshots.push(snapshot);
            node.snapshots.sort_by_key(|s| s.captured_frame);
            if node.snapshots.len() > 32 {
                node.snapshots.drain(..node.snapshots.len() - 32);
            }
            node.adult_snapshot = None; // no longer used for new data
        }
    }

    /// Migrate legacy `adult_snapshot` fields into `snapshots` for old saves.
    pub fn migrate_legacy_snapshots(&mut self) {
        for node in &mut self.nodes {
            if node.snapshots.is_empty() {
                if let Some(snap) = node.adult_snapshot.take() {
                    node.snapshots.push(snap);
                }
            } else {
                node.adult_snapshot = None;
            }
        }
    }

    pub fn register_hybrid_lineage(
        &mut self,
        genome_id: u32,
        genome: &Genome,
        parent_a: LineageId,
        parent_b: LineageId,
        similarity: f32,
        frame: i32,
    ) -> LineageId {
        if let Some(node) = self.nodes.iter().find(|n| {
            n.genome_id == genome_id
                && matches!(n.origin, LineageOrigin::Hybrid { .. })
                && n.parent_a == Some(parent_a)
                && n.parent_b == Some(parent_b)
        }) {
            return node.id;
        }

        let id = self.allocate_lineage_id();
        self.nodes.push(LineageNode {
            id,
            genome_id,
            display_name: genome.name.clone(),
            origin: LineageOrigin::Hybrid {
                parent_a,
                parent_b,
                similarity,
            },
            parent_a: Some(parent_a),
            parent_b: Some(parent_b),
            first_frame: frame,
            last_seen_frame: frame,
            noteworthy_score: 1.0,
            traits: LineageTraitSummary::from_genome(genome),
            ..LineageNode::default()
        });
        self.push_event(
            frame,
            id,
            LineageEventKind::Hybridization,
            "Hybrid lineage",
            format!(
                "{} emerged from a gametocyte merge ({:.0}% similarity).",
                genome.name,
                similarity * 100.0
            ),
            1.0,
            true,
        );
        self.enforce_retention();
        id
    }

    pub fn apply_population_samples(&mut self, samples: &[LineagePopulationSample], frame: i32) {
        let sampled: std::collections::HashSet<_> =
            samples.iter().map(|sample| sample.lineage_id).collect();

        for sample in samples {
            if let Some(node) = self.nodes.iter_mut().find(|n| n.id == sample.lineage_id) {
                let was_alive = node.current_cells > 0 || node.current_organisms > 0;
                node.current_cells = sample.cells;
                node.current_organisms = sample.organisms;
                node.peak_cells = node.peak_cells.max(sample.cells);
                node.peak_organisms = node.peak_organisms.max(sample.organisms);
                node.last_seen_frame = sample.frame.max(frame);

                let is_alive = sample.cells > 0 || sample.organisms > 0;
                if is_alive {
                    node.extinct_frame = None;
                } else if was_alive && node.extinct_frame.is_none() {
                    node.extinct_frame = Some(frame);
                    self.push_event(
                        frame,
                        sample.lineage_id,
                        LineageEventKind::Extinction,
                        "Lineage extinct",
                        "No living cells remain for this lineage.",
                        0.7,
                        true,
                    );
                }
            }
        }

        let mut extinct_events = Vec::new();
        for node in &mut self.nodes {
            if sampled.contains(&node.id) {
                continue;
            }

            let was_alive = node.current_cells > 0 || node.current_organisms > 0;
            node.current_cells = 0;
            node.current_organisms = 0;
            if was_alive && node.extinct_frame.is_none() {
                node.extinct_frame = Some(frame);
                extinct_events.push(node.id);
            }
        }

        for lineage_id in extinct_events {
            self.push_event(
                frame,
                lineage_id,
                LineageEventKind::Extinction,
                "Lineage extinct",
                "No living cells remain for this lineage.",
                0.7,
                true,
            );
        }
        self.enforce_retention();
    }

    pub fn record_scan(&mut self, frame: i32, time_seconds: f32) {
        self.record_scan_population(frame, time_seconds, 0, 0, 0, false);
    }

    pub fn record_scan_population(
        &mut self,
        frame: i32,
        time_seconds: f32,
        live_cells: u32,
        tracked_cells: u32,
        untracked_cells: u32,
        organism_counts_reliable: bool,
    ) {
        self.last_scan_frame = Some(frame);
        self.last_scan_time = time_seconds;
        self.last_scan_live_cells = live_cells;
        self.last_scan_tracked_cells = tracked_cells;
        self.last_scan_untracked_cells = untracked_cells;
        self.last_scan_organism_counts_reliable = organism_counts_reliable;
    }

    /// Ensure every scene genome has at least a root lineage. This is mainly
    /// used when restoring older saves that predate the lineage archive.
    pub fn ensure_scene_genomes(&mut self, genomes: &[Genome], frame: i32) {
        for (genome_id, genome) in genomes.iter().enumerate() {
            self.ensure_user_lineage(genome_id as u32, genome, frame);
        }
    }

    pub fn ensure_sampled_mutation_lineage(
        &mut self,
        genome_id: u32,
        parent_lineage: Option<LineageId>,
        frame: i32,
    ) -> LineageId {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.genome_id == genome_id) {
            if node.parent_a.is_none() {
                if let Some(parent) = parent_lineage {
                    node.parent_a = Some(parent);
                    node.origin = LineageOrigin::Mutation {
                        parent,
                        mutation_event: None,
                    };
                }
            }
            return node.id;
        }

        let id = self.allocate_lineage_id();
        let parent = parent_lineage.unwrap_or(0);
        self.nodes.push(LineageNode {
            id,
            genome_id,
            display_name: format!("GPU Genome #{genome_id}"),
            origin: LineageOrigin::Mutation {
                parent,
                mutation_event: None,
            },
            parent_a: parent_lineage,
            first_frame: frame,
            last_seen_frame: frame,
            mutation_count: 1,
            noteworthy_score: 0.35,
            ..LineageNode::default()
        });
        self.push_event(
            frame,
            id,
            LineageEventKind::Mutation,
            "Mutation branch sampled",
            "A living GPU-native genome branch was sampled during a lineage interval snapshot.",
            0.35,
            false,
        );
        self.enforce_retention();
        id
    }

    pub fn record_minor_variant_observation(
        &mut self,
        _genome_id: u32,
        parent_genome_id: Option<u32>,
        cells: u32,
        organisms: u32,
        frame: i32,
    ) {
        if let Some(stats) = self
            .minor_variant_stats
            .iter_mut()
            .find(|stats| stats.parent_genome_id == parent_genome_id)
        {
            stats.record(parent_genome_id, cells, organisms, frame);
        } else {
            let mut stats = LineageMinorVariantStats {
                parent_genome_id,
                ..LineageMinorVariantStats::default()
            };
            stats.record(parent_genome_id, cells, organisms, frame);
            self.minor_variant_stats.push(stats);
        }
        self.enforce_retention();
    }

    pub fn minor_variant_totals(&self) -> (u32, u32, u32) {
        self.minor_variant_stats.iter().fold(
            (0u32, 0u32, 0u32),
            |(variants, cells, organisms), stats| {
                (
                    variants.saturating_add(stats.variant_observations),
                    cells.saturating_add(stats.current_cells),
                    organisms.saturating_add(stats.current_organisms),
                )
            },
        )
    }

    pub fn loadable_genome_count(&self) -> usize {
        self.genome_bookmarks
            .iter()
            .filter(|bookmark| bookmark.is_loadable())
            .count()
    }

    pub fn loadable_bookmark_for_lineage(
        &self,
        lineage_id: LineageId,
    ) -> Option<&LineageGenomeBookmark> {
        self.genome_bookmarks
            .iter()
            .find(|bookmark| bookmark.lineage_id == lineage_id && bookmark.is_loadable())
    }

    pub fn upsert_loadable_genome(
        &mut self,
        lineage_id: LineageId,
        genome_id: u32,
        display_name: impl Into<String>,
        reason: impl Into<String>,
        frame: i32,
        genome_yaml: String,
    ) {
        let display_name = display_name.into();
        let reason = reason.into();
        if let Some(bookmark) = self
            .genome_bookmarks
            .iter_mut()
            .find(|bookmark| bookmark.lineage_id == lineage_id && bookmark.genome_id == genome_id)
        {
            bookmark.saved_frame = frame;
            bookmark.display_name = display_name;
            bookmark.reason = reason;
            bookmark.genome_yaml = Some(genome_yaml);
        } else {
            self.genome_bookmarks.push(LineageGenomeBookmark {
                lineage_id,
                genome_id,
                saved_frame: frame,
                display_name,
                reason,
                genome_yaml: Some(genome_yaml),
            });
        }
        self.enforce_retention();
    }

    pub fn bookmark_scene_genome(
        &mut self,
        lineage_id: LineageId,
        genome_id: u32,
        display_name: impl Into<String>,
        reason: impl Into<String>,
        frame: i32,
    ) {
        if self
            .genome_bookmarks
            .iter()
            .any(|b| b.lineage_id == lineage_id && b.genome_id == genome_id)
        {
            return;
        }

        let display_name = display_name.into();
        self.genome_bookmarks.push(LineageGenomeBookmark {
            lineage_id,
            genome_id,
            saved_frame: frame,
            display_name: display_name.clone(),
            reason: reason.into(),
            genome_yaml: None,
        });
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == lineage_id) {
            node.pinned = true;
        }
        self.push_event(
            frame,
            lineage_id,
            LineageEventKind::Bookmark,
            "Genome bookmarked",
            format!("{display_name} can be loaded from this biosphere save."),
            0.5,
            true,
        );
        self.enforce_retention();
    }

    /// Record a stable period where no meaningful ecosystem changes occurred.
    ///
    /// Consecutive contiguous stable periods are merged into a single gap entry.
    /// Call this after a lineage scan that passed the stability threshold.
    pub fn record_stable_period(
        &mut self,
        start_frame: i32,
        end_frame: i32,
        start_time: f32,
        end_time: f32,
        live_cells: u32,
    ) {
        if end_frame <= start_frame {
            return;
        }
        if let Some(last) = self.time_gaps.last_mut() {
            if last.end_frame == start_frame {
                last.end_frame = end_frame;
                last.end_time = end_time;
                return;
            }
        }
        self.time_gaps.push(LineageTimeGap {
            start_frame,
            end_frame,
            start_time,
            end_time,
            live_cells,
        });
        // Keep bounded without a full sort: oldest gaps are at the front.
        let max_gaps = self.retention.max_time_gaps.max(1);
        if self.time_gaps.len() > max_gaps {
            self.time_gaps.drain(..self.time_gaps.len() - max_gaps);
        }
    }

    /// Returns true if the given frame falls inside any recorded time gap.
    pub fn frame_in_gap(&self, frame: i32) -> bool {
        self.time_gaps
            .iter()
            .any(|g| frame >= g.start_frame && frame < g.end_frame)
    }

    pub fn prepare_for_save(&mut self) {
        self.enforce_retention();
    }

    pub fn generation_for_lineage(&self, lineage_id: LineageId) -> u32 {
        let mut memo = std::collections::HashMap::new();
        self.generation_for_lineage_inner(lineage_id, &mut memo)
    }

    fn allocate_lineage_id(&mut self) -> LineageId {
        let id = self.next_lineage_id.max(1);
        self.next_lineage_id = id.saturating_add(1);
        id
    }

    fn push_event(
        &mut self,
        frame: i32,
        lineage_id: LineageId,
        kind: LineageEventKind,
        title: impl Into<String>,
        detail: impl Into<String>,
        impact_score: f32,
        noteworthy: bool,
    ) {
        let id = self.next_event_id.max(1);
        self.next_event_id = id.saturating_add(1);
        self.events.push(LineageEvent {
            id,
            frame,
            lineage_id,
            kind,
            title: title.into(),
            detail: detail.into(),
            impact_score,
            noteworthy,
        });
    }

    fn enforce_retention(&mut self) {
        let max_gaps = self.retention.max_time_gaps.max(1);
        if self.time_gaps.len() > max_gaps {
            self.time_gaps.drain(..self.time_gaps.len() - max_gaps);
        }

        let max_events = self.retention.max_events.max(1);
        if self.events.len() > max_events {
            self.events.sort_by(|a, b| {
                b.noteworthy
                    .cmp(&a.noteworthy)
                    .then_with(|| b.impact_score.total_cmp(&a.impact_score))
                    .then_with(|| b.frame.cmp(&a.frame))
            });
            self.events.truncate(max_events);
            self.events.sort_by_key(|e| (e.frame, e.id));
        }

        self.enforce_genome_bookmark_retention();

        if self.minor_variant_stats.len() > DEFAULT_MAX_MINOR_VARIANT_STATS {
            self.minor_variant_stats.sort_by(|a, b| {
                b.peak_cells
                    .cmp(&a.peak_cells)
                    .then_with(|| b.peak_organisms.cmp(&a.peak_organisms))
                    .then_with(|| b.variant_observations.cmp(&a.variant_observations))
                    .then_with(|| b.last_seen_frame.cmp(&a.last_seen_frame))
            });
            self.minor_variant_stats
                .truncate(DEFAULT_MAX_MINOR_VARIANT_STATS);
            self.minor_variant_stats
                .sort_by_key(|stats| (stats.first_seen_frame, stats.parent_genome_id));
        }

        let max_nodes = self.retention.max_nodes.max(1);
        if self.nodes.len() > max_nodes {
            self.nodes.sort_by(|a, b| {
                b.pinned
                    .cmp(&a.pinned)
                    .then_with(|| {
                        let bp = b.peak_organisms.max(b.peak_cells);
                        let ap = a.peak_organisms.max(a.peak_cells);
                        bp.cmp(&ap)
                    })
                    .then_with(|| b.noteworthy_score.total_cmp(&a.noteworthy_score))
                    .then_with(|| b.last_seen_frame.cmp(&a.last_seen_frame))
            });
            self.nodes.truncate(max_nodes);
            self.nodes.sort_by_key(|n| (n.first_frame, n.id));
        }
    }

    fn enforce_genome_bookmark_retention(&mut self) {
        let max_bookmarks = self.retention.max_bookmarks.max(1);
        if self.genome_bookmarks.len() <= max_bookmarks {
            return;
        }

        self.genome_bookmarks.sort_by(|a, b| {
            b.is_loadable()
                .cmp(&a.is_loadable())
                .then_with(|| b.saved_frame.cmp(&a.saved_frame))
                .then_with(|| b.lineage_id.cmp(&a.lineage_id))
                .then_with(|| b.genome_id.cmp(&a.genome_id))
        });
        self.genome_bookmarks.truncate(max_bookmarks);
        self.genome_bookmarks
            .sort_by_key(|bookmark| (bookmark.saved_frame, bookmark.lineage_id));
    }

    fn generation_for_lineage_inner(
        &self,
        lineage_id: LineageId,
        memo: &mut std::collections::HashMap<LineageId, u32>,
    ) -> u32 {
        if let Some(generation) = memo.get(&lineage_id) {
            return *generation;
        }

        let generation = self
            .nodes
            .iter()
            .find(|node| node.id == lineage_id)
            .map(|node| {
                [node.parent_a, node.parent_b]
                    .into_iter()
                    .flatten()
                    .map(|parent| {
                        self.generation_for_lineage_inner(parent, memo)
                            .saturating_add(1)
                    })
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        memo.insert(lineage_id, generation);
        generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retention_keeps_pinned_and_recent_events_under_caps() {
        let mut archive = EcosystemLineageArchive {
            retention: LineageRetentionPolicy {
                max_nodes: 2,
                max_events: 3,
                max_bookmarks: 1,
                protect_peak_population_at_least: 10,
                max_time_gaps: DEFAULT_MAX_TIME_GAPS,
            },
            ..EcosystemLineageArchive::default()
        };

        archive.nodes.push(LineageNode {
            id: 1,
            pinned: false,
            peak_cells: 1,
            last_seen_frame: 1,
            ..LineageNode::default()
        });
        archive.nodes.push(LineageNode {
            id: 2,
            pinned: true,
            peak_cells: 1,
            last_seen_frame: 2,
            ..LineageNode::default()
        });
        archive.nodes.push(LineageNode {
            id: 3,
            pinned: false,
            peak_cells: 100,
            last_seen_frame: 3,
            ..LineageNode::default()
        });

        for i in 0..5 {
            archive.push_event(
                i,
                1,
                LineageEventKind::Note,
                format!("event {i}"),
                "",
                i as f32,
                i == 0,
            );
        }

        archive.genome_bookmarks.push(LineageGenomeBookmark {
            lineage_id: 1,
            saved_frame: 1,
            ..LineageGenomeBookmark::default()
        });
        archive.genome_bookmarks.push(LineageGenomeBookmark {
            lineage_id: 2,
            saved_frame: 2,
            ..LineageGenomeBookmark::default()
        });

        archive.enforce_retention();

        assert_eq!(archive.nodes.len(), 2);
        assert!(archive.nodes.iter().any(|n| n.id == 2));
        assert!(archive.nodes.iter().any(|n| n.id == 3));
        assert_eq!(archive.events.len(), 3);
        assert_eq!(archive.genome_bookmarks.len(), 1);
        assert_eq!(archive.genome_bookmarks[0].lineage_id, 2);
    }

    #[test]
    fn loadable_genomes_roll_off_from_earliest_snapshots() {
        let mut archive = EcosystemLineageArchive {
            retention: LineageRetentionPolicy {
                max_nodes: 10,
                max_events: 10,
                max_bookmarks: 2,
                protect_peak_population_at_least: 10,
                max_time_gaps: DEFAULT_MAX_TIME_GAPS,
            },
            ..EcosystemLineageArchive::default()
        };

        archive.upsert_loadable_genome(1, 101, "A", "test", 10, "genome-a".to_string());
        archive.upsert_loadable_genome(2, 102, "B", "test", 20, "genome-b".to_string());
        archive.upsert_loadable_genome(3, 103, "C", "test", 30, "genome-c".to_string());

        assert_eq!(archive.loadable_genome_count(), 2);
        assert!(archive.loadable_bookmark_for_lineage(1).is_none());
        assert!(archive.loadable_bookmark_for_lineage(2).is_some());
        assert!(archive.loadable_bookmark_for_lineage(3).is_some());
    }

    #[test]
    fn minor_variants_aggregate_by_parent_without_nodes() {
        let mut archive = EcosystemLineageArchive::default();

        archive.record_minor_variant_observation(101, Some(1), 3, 1, 10);
        archive.record_minor_variant_observation(102, Some(1), 5, 2, 10);
        archive.record_minor_variant_observation(103, Some(2), 7, 1, 11);

        assert!(archive.nodes.is_empty());
        assert_eq!(archive.minor_variant_stats.len(), 2);

        let parent_one = archive
            .minor_variant_stats
            .iter()
            .find(|stats| stats.parent_genome_id == Some(1))
            .unwrap();
        assert_eq!(parent_one.variant_observations, 2);
        assert_eq!(parent_one.current_cells, 8);
        assert_eq!(parent_one.current_organisms, 3);
        assert_eq!(parent_one.peak_cells, 5);

        let (observations, current_cells, current_organisms) = archive.minor_variant_totals();
        assert_eq!(observations, 3);
        assert_eq!(current_cells, 15);
        assert_eq!(current_organisms, 4);
    }

    #[test]
    fn lineage_generation_follows_deepest_parent_branch() {
        let mut archive = EcosystemLineageArchive::default();
        archive.nodes.push(LineageNode {
            id: 1,
            ..LineageNode::default()
        });
        archive.nodes.push(LineageNode {
            id: 2,
            parent_a: Some(1),
            ..LineageNode::default()
        });
        archive.nodes.push(LineageNode {
            id: 3,
            parent_a: Some(1),
            ..LineageNode::default()
        });
        archive.nodes.push(LineageNode {
            id: 4,
            parent_a: Some(2),
            parent_b: Some(3),
            ..LineageNode::default()
        });
        archive.nodes.push(LineageNode {
            id: 5,
            parent_a: Some(4),
            ..LineageNode::default()
        });

        assert_eq!(archive.generation_for_lineage(1), 0);
        assert_eq!(archive.generation_for_lineage(2), 1);
        assert_eq!(archive.generation_for_lineage(4), 2);
        assert_eq!(archive.generation_for_lineage(5), 3);
    }
}

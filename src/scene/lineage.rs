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
const DEFAULT_MAX_TELEMETRY_SAMPLES_PER_LINEAGE: usize = 32;

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
    /// Bounded, low-frequency telemetry captured by explicit lineage scans.
    ///
    /// This deliberately stores compact aggregates rather than per-cell data.
    #[serde(default)]
    pub telemetry_history: Vec<LineageTelemetrySample>,
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
            telemetry_history: Vec::new(),
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

    pub fn latest_telemetry(&self) -> Option<&LineageTelemetrySample> {
        self.telemetry_history.last()
    }

    pub fn telemetry_at_least_seconds_ago(&self, seconds: f32) -> Option<&LineageTelemetrySample> {
        let latest = self.latest_telemetry()?;
        let target = latest.time_seconds - seconds.max(0.0);
        self.telemetry_history
            .iter()
            .rev()
            .find(|sample| sample.time_seconds <= target)
            .or_else(|| self.telemetry_history.first())
    }

    pub fn consecutive_growth_windows(&self) -> u32 {
        self.telemetry_history
            .iter()
            .rev()
            .take_while(|sample| sample.cell_delta > 0)
            .count()
            .min(u32::MAX as usize) as u32
    }

    pub fn consecutive_decline_windows(&self) -> u32 {
        self.telemetry_history
            .iter()
            .rev()
            .take_while(|sample| sample.cell_delta < 0)
            .count()
            .min(u32::MAX as usize) as u32
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
    Recovery,
    NewPopulationPeak,
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
    pub time_seconds: f32,
    pub avg_nutrient: f32,
    pub nutrient_positive_fraction: f32,
    pub division_ready_fraction: f32,
    pub starvation_risk_fraction: f32,
    pub average_age: f32,
    pub dominant_cell_type: u32,
    pub dominant_cell_type_fraction: f32,
    pub active_mode_count: u32,
    pub center: [f32; 3],
    pub bounding_radius: f32,
}

/// Compact report-oriented telemetry for one lineage at one scan.
///
/// Samples are captured only by the explicit blocking lineage scan and retained
/// in a small ring-like history. Trends and report events are derived on CPU.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageTelemetrySample {
    pub frame: i32,
    pub time_seconds: f32,
    pub cells: u32,
    pub organisms: u32,
    pub cell_delta: i32,
    pub organism_delta: i32,
    pub avg_nutrient: f32,
    pub nutrient_positive_fraction: f32,
    pub division_ready_fraction: f32,
    pub starvation_risk_fraction: f32,
    pub average_age: f32,
    pub dominant_cell_type: u32,
    pub dominant_cell_type_fraction: f32,
    pub active_mode_count: u32,
    pub center: [f32; 3],
    pub bounding_radius: f32,
}

impl LineageTelemetrySample {
    pub fn growth_rate_per_second(&self, previous: &Self) -> f32 {
        let elapsed = (self.time_seconds - previous.time_seconds).max(f32::EPSILON);
        self.cell_delta as f32 / elapsed
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EcosystemTelemetrySummary {
    pub active_lineages: u32,
    pub total_cells: u32,
    pub largest_lineage_fraction: f32,
    pub diversity_score: f32,
    pub evenness_score: f32,
    pub rare_lineages: u32,
    pub growing_lineages: u32,
    pub declining_lineages: u32,
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
    pub fn ecosystem_telemetry(&self) -> EcosystemTelemetrySummary {
        let mut active_lineages = 0usize;
        let mut total_cells = 0u32;
        for node in &self.nodes {
            if node.current_cells > 0 {
                active_lineages += 1;
                total_cells = total_cells.saturating_add(node.current_cells);
            }
        }
        if active_lineages == 0 || total_cells == 0 {
            return EcosystemTelemetrySummary::default();
        }

        let mut entropy = 0.0f32;
        let mut largest_fraction = 0.0f32;
        let mut rare_lineages = 0u32;
        let mut growing_lineages = 0u32;
        let mut declining_lineages = 0u32;
        for node in self.nodes.iter().filter(|node| node.current_cells > 0) {
            let fraction = node.current_cells as f32 / total_cells as f32;
            largest_fraction = largest_fraction.max(fraction);
            entropy -= fraction * fraction.ln();
            rare_lineages += u32::from(fraction <= 0.02);
            if let Some(latest) = node.latest_telemetry() {
                growing_lineages += u32::from(latest.cell_delta > 0);
                declining_lineages += u32::from(latest.cell_delta < 0);
            }
        }

        let max_entropy = (active_lineages as f32).ln();
        let evenness = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            1.0
        };

        EcosystemTelemetrySummary {
            active_lineages: active_lineages.min(u32::MAX as usize) as u32,
            total_cells,
            largest_lineage_fraction: largest_fraction,
            diversity_score: entropy,
            evenness_score: evenness,
            rare_lineages,
            growing_lineages,
            declining_lineages,
        }
    }

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
                let had_previous_telemetry = !node.telemetry_history.is_empty();
                let previous_cells = node.current_cells;
                let previous_organisms = node.current_organisms;
                let previous_peak = node.peak_cells;
                let previous_near_extinction = previous_cells > 0 && previous_cells <= 5;
                node.current_cells = sample.cells;
                node.current_organisms = sample.organisms;
                node.peak_cells = node.peak_cells.max(sample.cells);
                node.peak_organisms = node.peak_organisms.max(sample.organisms);
                node.last_seen_frame = sample.frame.max(frame);

                let is_alive = sample.cells > 0 || sample.organisms > 0;
                let telemetry = LineageTelemetrySample {
                    frame: sample.frame,
                    time_seconds: sample.time_seconds,
                    cells: sample.cells,
                    organisms: sample.organisms,
                    cell_delta: (sample.cells as i64 - previous_cells as i64)
                        .clamp(i32::MIN as i64, i32::MAX as i64)
                        as i32,
                    organism_delta: (sample.organisms as i64 - previous_organisms as i64)
                        .clamp(i32::MIN as i64, i32::MAX as i64)
                        as i32,
                    avg_nutrient: sample.avg_nutrient,
                    nutrient_positive_fraction: sample.nutrient_positive_fraction,
                    division_ready_fraction: sample.division_ready_fraction,
                    starvation_risk_fraction: sample.starvation_risk_fraction,
                    average_age: sample.average_age,
                    dominant_cell_type: sample.dominant_cell_type,
                    dominant_cell_type_fraction: sample.dominant_cell_type_fraction,
                    active_mode_count: sample.active_mode_count,
                    center: sample.center,
                    bounding_radius: sample.bounding_radius,
                };
                node.telemetry_history.push(telemetry);
                if node.telemetry_history.len() > DEFAULT_MAX_TELEMETRY_SAMPLES_PER_LINEAGE {
                    let excess =
                        node.telemetry_history.len() - DEFAULT_MAX_TELEMETRY_SAMPLES_PER_LINEAGE;
                    node.telemetry_history.drain(..excess);
                }

                let growth = sample.cells.saturating_sub(previous_cells);
                let decline = previous_cells.saturating_sub(sample.cells);
                let boom_threshold = (previous_cells / 4).max(10);
                let crashed_to_near_extinction =
                    sample.cells > 0 && sample.cells <= 5 && previous_cells > 5;
                let recovered = previous_near_extinction && sample.cells > 5;
                let new_peak = sample.cells > previous_peak && previous_peak > 0;

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

                if had_previous_telemetry && growth >= boom_threshold {
                    self.push_event(
                        frame,
                        sample.lineage_id,
                        LineageEventKind::PopulationBoom,
                        "Population boom",
                        format!(
                            "Population increased by {} cells to {}.",
                            growth, sample.cells
                        ),
                        0.55,
                        true,
                    );
                } else if had_previous_telemetry && crashed_to_near_extinction {
                    self.push_event(
                        frame,
                        sample.lineage_id,
                        LineageEventKind::NearExtinction,
                        "Near extinction",
                        format!(
                            "Population fell by {} cells; only {} remain.",
                            decline, sample.cells
                        ),
                        0.75,
                        true,
                    );
                } else if had_previous_telemetry && recovered {
                    self.push_event(
                        frame,
                        sample.lineage_id,
                        LineageEventKind::Recovery,
                        "Population recovery",
                        format!("Population recovered to {} cells.", sample.cells),
                        0.5,
                        true,
                    );
                }

                if had_previous_telemetry && new_peak {
                    self.push_event(
                        frame,
                        sample.lineage_id,
                        LineageEventKind::NewPopulationPeak,
                        "New population peak",
                        format!("Population reached a new peak of {} cells.", sample.cells),
                        0.35,
                        false,
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

    #[test]
    fn telemetry_history_is_bounded_and_tracks_trends() {
        let mut archive = EcosystemLineageArchive::default();
        archive.nodes.push(LineageNode {
            id: 1,
            ..LineageNode::default()
        });

        for scan in 0..40 {
            archive.apply_population_samples(
                &[LineagePopulationSample {
                    lineage_id: 1,
                    cells: 10 + scan,
                    organisms: 1,
                    frame: scan as i32,
                    time_seconds: scan as f32 * 30.0,
                    avg_nutrient: 50.0,
                    ..LineagePopulationSample::default()
                }],
                scan as i32,
            );
        }

        let node = &archive.nodes[0];
        assert_eq!(
            node.telemetry_history.len(),
            DEFAULT_MAX_TELEMETRY_SAMPLES_PER_LINEAGE
        );
        assert_eq!(node.latest_telemetry().unwrap().cells, 49);
        assert_eq!(node.consecutive_growth_windows(), 32);
        assert_eq!(node.consecutive_decline_windows(), 0);
        assert_eq!(node.telemetry_at_least_seconds_ago(60.0).unwrap().cells, 47);
    }

    #[test]
    fn population_changes_generate_report_events_after_baseline() {
        let mut archive = EcosystemLineageArchive::default();
        archive.nodes.push(LineageNode {
            id: 1,
            ..LineageNode::default()
        });

        archive.apply_population_samples(
            &[LineagePopulationSample {
                lineage_id: 1,
                cells: 20,
                frame: 1,
                time_seconds: 30.0,
                ..LineagePopulationSample::default()
            }],
            1,
        );
        assert!(archive.events.is_empty());

        archive.apply_population_samples(
            &[LineagePopulationSample {
                lineage_id: 1,
                cells: 35,
                frame: 2,
                time_seconds: 60.0,
                ..LineagePopulationSample::default()
            }],
            2,
        );

        assert!(archive
            .events
            .iter()
            .any(|event| matches!(event.kind, LineageEventKind::PopulationBoom)));
        assert!(archive
            .events
            .iter()
            .any(|event| matches!(event.kind, LineageEventKind::NewPopulationPeak)));
    }

    #[test]
    fn ecosystem_summary_reports_dominance_evenness_and_direction() {
        let mut archive = EcosystemLineageArchive::default();
        archive.nodes.push(LineageNode {
            id: 1,
            current_cells: 75,
            telemetry_history: vec![LineageTelemetrySample {
                cells: 75,
                cell_delta: 5,
                ..LineageTelemetrySample::default()
            }],
            ..LineageNode::default()
        });
        archive.nodes.push(LineageNode {
            id: 2,
            current_cells: 25,
            telemetry_history: vec![LineageTelemetrySample {
                cells: 25,
                cell_delta: -2,
                ..LineageTelemetrySample::default()
            }],
            ..LineageNode::default()
        });

        let summary = archive.ecosystem_telemetry();
        assert_eq!(summary.active_lineages, 2);
        assert_eq!(summary.total_cells, 100);
        assert!((summary.largest_lineage_fraction - 0.75).abs() < 0.001);
        assert!(summary.evenness_score > 0.0 && summary.evenness_score < 1.0);
        assert_eq!(summary.growing_lineages, 1);
        assert_eq!(summary.declining_lineages, 1);
    }
}

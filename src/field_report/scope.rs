use crate::field_report::{
    validate_report, ClaimConfidence, ClaimKey, FieldReportSeverity, FieldReportTag,
    RenderedFieldReport, RenderedSentence, ReportTheme, SentenceRole, ToneFamily, ToneProfile,
};
use crate::scene::lineage::LineageId;

pub type OrganismId = u32;
pub type CellId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReportScope {
    Ecosystem,
    Lineage(LineageId),
    Organism(OrganismId),
    Specimen(CellId),
    LineageSnapshot {
        lineage_id: LineageId,
        snapshot_frame: u64,
    },
}

#[derive(Debug, Clone)]
pub struct SpecimenReportSnapshot {
    pub cell_id: CellId,
    pub cell_type_name: String,
    pub lineage_id: Option<LineageId>,
    pub lineage_name: Option<String>,
    pub organism_id: Option<OrganismId>,
    pub alive: bool,
    pub nutrient_level: f32,
    pub nutrient_gain_rate: f32,
    pub thermal_state: u32,
    pub adhesion_count: u32,
    pub active_signal_channels: u32,
}

#[derive(Debug, Clone)]
pub struct LineageSnapshotReportSnapshot {
    pub lineage_id: LineageId,
    pub lineage_name: String,
    pub snapshot_frame: u64,
    pub captured_time: f32,
    pub morphology_cells: u32,
    pub current_cells_at_snapshot: Option<u32>,
    pub peak_cells: u32,
    pub dominant_cell_type_name: Option<String>,
    pub active_modes: Option<u32>,
    pub territory_radius: f32,
}

pub fn render_specimen_report(
    snapshot: &SpecimenReportSnapshot,
    requested_tone: &ToneProfile,
) -> Option<RenderedFieldReport> {
    let severity = specimen_severity(snapshot);
    let tone = crate::field_report::resolve_tone(*requested_tone, severity);
    let subject = snapshot
        .lineage_name
        .as_deref()
        .unwrap_or("an unclassified lineage");
    let scope_label = if let Some(organism_id) = snapshot.organism_id {
        format!(
            "Cell {} · Organism {} · {}",
            snapshot.cell_id, organism_id, subject
        )
    } else {
        format!("Cell {} · isolated · {}", snapshot.cell_id, subject)
    };
    let nutrient_state = if snapshot.nutrient_gain_rate > 0.01 {
        "nutrient-secure"
    } else if snapshot.nutrient_gain_rate < -0.01 {
        "nutrient-negative"
    } else {
        "nutrient-stable"
    };
    let thermal_state = thermal_label(snapshot.thermal_state);
    let integrated = if snapshot.adhesion_count > 0 {
        "embedded in bonded tissue"
    } else {
        "not currently bonded to neighboring cells"
    };

    let lead = match tone.family() {
        ToneFamily::Formal => format!(
            "Selected cell {} is a {} associated with {}.",
            snapshot.cell_id, snapshot.cell_type_name, subject
        ),
        ToneFamily::Naturalist => format!(
            "Cell {} is a {} within {}.",
            snapshot.cell_id, snapshot.cell_type_name, subject
        ),
        ToneFamily::Living => format!(
            "This {} belongs to {} and remains part of the living sample.",
            snapshot.cell_type_name, subject
        ),
        ToneFamily::Alert => format!(
            "Cell {} specimen status: {}.",
            snapshot.cell_id, snapshot.cell_type_name
        ),
        ToneFamily::Any => unreachable!(),
    };
    let evidence = format!(
        "It is {}, thermally {}, and has {} active bond{}.",
        nutrient_state,
        thermal_state,
        snapshot.adhesion_count,
        if snapshot.adhesion_count == 1 {
            ""
        } else {
            "s"
        }
    );
    let interpretation = if snapshot.organism_id.is_some() {
        format!(
            "The specimen is {}; organism-level composition has not been inferred.",
            integrated
        )
    } else {
        format!("The specimen is isolated and {}.", integrated)
    };

    let lineage_claim = snapshot
        .lineage_id
        .map(ClaimKey::DominantCellType)
        .into_iter()
        .collect();
    let sentences = vec![
        RenderedSentence {
            text: lead,
            role: SentenceRole::Lead,
            confidence: ClaimConfidence::Observed,
            claim_keys: lineage_claim,
            fragment_id: "specimen_identity",
            numeric_anchor: true,
            metaphor_cost: 0,
        },
        RenderedSentence {
            text: evidence,
            role: SentenceRole::Evidence,
            confidence: ClaimConfidence::Observed,
            claim_keys: Vec::new(),
            fragment_id: "specimen_condition",
            numeric_anchor: true,
            metaphor_cost: 0,
        },
        RenderedSentence {
            text: interpretation,
            role: SentenceRole::Interpretation,
            confidence: ClaimConfidence::Derived,
            claim_keys: Vec::new(),
            fragment_id: "specimen_integration",
            numeric_anchor: false,
            metaphor_cost: 0,
        },
    ];
    let tags = specimen_tags(snapshot);
    let report = RenderedFieldReport {
        title: if snapshot.organism_id.is_some() {
            "Organism Specimen".to_string()
        } else {
            "Selected Specimen".to_string()
        },
        body: format!("{}\n{}", scope_label, sentence_body(&sentences)),
        sentences,
        theme: ReportTheme::CompositionShift,
        severity,
        tags,
        confidence: ClaimConfidence::Derived,
        involved_lineages: snapshot.lineage_id.into_iter().collect(),
    };
    validate_report(&report, &tone).is_empty().then_some(report)
}

pub fn render_lineage_snapshot_report(
    snapshot: &LineageSnapshotReportSnapshot,
    requested_tone: &ToneProfile,
) -> Option<RenderedFieldReport> {
    let tone = crate::field_report::resolve_tone(*requested_tone, FieldReportSeverity::Routine);
    let phase = match snapshot.current_cells_at_snapshot {
        Some(cells) if cells == snapshot.peak_cells && snapshot.peak_cells > 0 => {
            "near its retained population peak"
        }
        Some(cells) if cells.saturating_mul(2) < snapshot.peak_cells => {
            "below half of its retained population peak"
        }
        Some(_) => "within its recorded population history",
        None => "at a morphology capture without a matching population sample",
    };
    let lead = match tone.family() {
        ToneFamily::Formal => format!("This snapshot records {} {}.", snapshot.lineage_name, phase),
        ToneFamily::Naturalist => format!(
            "This field image catches {} {}.",
            snapshot.lineage_name, phase
        ),
        ToneFamily::Living => format!("{} is held here {}.", snapshot.lineage_name, phase),
        ToneFamily::Alert => format!("{} snapshot status recorded.", snapshot.lineage_name),
        ToneFamily::Any => unreachable!(),
    };
    let mut evidence = format!(
        "At {}, the snapshot contains {} morphology cells within an approximate radius of {:.1} units.",
        format_simulation_time(snapshot.captured_time as f64),
        snapshot.morphology_cells,
        snapshot.territory_radius
    );
    if let (Some(cell_type), Some(modes)) = (
        snapshot.dominant_cell_type_name.as_deref(),
        snapshot.active_modes,
    ) {
        evidence.push_str(&format!(
            " {} is the dominant sampled cell type across {} active modes.",
            cell_type, modes
        ));
    }
    let sentences = vec![
        RenderedSentence {
            text: lead,
            role: SentenceRole::Lead,
            confidence: ClaimConfidence::Derived,
            claim_keys: vec![ClaimKey::Territory(snapshot.lineage_id)],
            fragment_id: "lineage_snapshot_phase",
            numeric_anchor: false,
            metaphor_cost: u8::from(matches!(tone.family(), ToneFamily::Living)),
        },
        RenderedSentence {
            text: evidence,
            role: SentenceRole::DossierDetail,
            confidence: ClaimConfidence::Observed,
            claim_keys: Vec::new(),
            fragment_id: "lineage_snapshot_evidence",
            numeric_anchor: true,
            metaphor_cost: 0,
        },
    ];
    let report = RenderedFieldReport {
        title: format!(
            "Snapshot Report · {}",
            format_simulation_time(snapshot.captured_time as f64)
        ),
        body: sentence_body(&sentences),
        sentences,
        theme: ReportTheme::TerritoryObservation,
        severity: FieldReportSeverity::Routine,
        tags: vec![FieldReportTag::Territory, FieldReportTag::Composition],
        confidence: ClaimConfidence::Derived,
        involved_lineages: vec![snapshot.lineage_id],
    };
    validate_report(&report, &tone).is_empty().then_some(report)
}

pub fn format_simulation_time(seconds: f64) -> String {
    let total_seconds = seconds.max(0.0).round() as u64;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn sentence_body(sentences: &[RenderedSentence]) -> String {
    sentences
        .iter()
        .map(|sentence| sentence.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn specimen_severity(snapshot: &SpecimenReportSnapshot) -> FieldReportSeverity {
    if !snapshot.alive || matches!(snapshot.thermal_state, 0 | 1 | 8 | 9) {
        FieldReportSeverity::Warning
    } else if snapshot.nutrient_gain_rate < -0.01 || snapshot.nutrient_level <= 10.0 {
        FieldReportSeverity::Notable
    } else {
        FieldReportSeverity::Routine
    }
}

fn specimen_tags(snapshot: &SpecimenReportSnapshot) -> Vec<FieldReportTag> {
    let mut tags = vec![FieldReportTag::Composition];
    if snapshot.nutrient_gain_rate < -0.01 || snapshot.nutrient_level <= 10.0 {
        tags.push(FieldReportTag::StarvationRisk);
    }
    if snapshot.adhesion_count > 0 {
        tags.push(FieldReportTag::Territory);
    }
    tags
}

fn thermal_label(state: u32) -> &'static str {
    match state {
        0 => "deep-frozen",
        1 => "frozen",
        2 => "cold-stressed",
        3 | 4 | 5 | 6 => "stable",
        7 => "warm-stressed",
        8 => "in heat shock",
        9 => "critical",
        _ => "unclassified",
    }
}

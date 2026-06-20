use std::collections::HashSet;

use crate::field_report::facts::{FieldReportTag, ReportFact, ReportFactKind};
use crate::scene::lineage::LineageId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReportTheme {
    StarvingExpansion,
    PopulationBoom,
    SustainedDecline,
    Recovery,
    NewPopulationPeak,
    NearExtinction,
    ReproductivePulse,
    CompositionShift,
    TerritoryObservation,
    EcosystemDominance,
    EcosystemBalance,
    SingleLineageProfile,
    SceneComposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FieldReportSeverity {
    Routine,
    Notable,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RhetoricalShape {
    Observation,
    ObservationEvidence,
    ObservationEvidenceWatch,
    AlertEvidenceAction,
}

/// Tone-neutral semantic plan. It contains meaning, never final prose.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportPlan {
    pub theme: ReportTheme,
    pub subject_lineage: Option<LineageId>,
    pub supporting_lineages: Vec<LineageId>,
    pub lead_fact: ReportFact,
    pub supporting_facts: Vec<ReportFact>,
    pub severity: FieldReportSeverity,
    pub tags: Vec<FieldReportTag>,
    pub rhetorical_shape: RhetoricalShape,
    pub require_numeric_anchor: bool,
}

pub fn build_report_plans(facts: &[ReportFact]) -> Vec<ReportPlan> {
    let mut plans = Vec::new();
    let lineage_ids: HashSet<_> = facts
        .iter()
        .filter_map(ReportFact::subject_lineage)
        .collect();

    for lineage_id in lineage_ids {
        let lineage_facts: Vec<_> = facts
            .iter()
            .filter(|fact| fact.subject_lineage() == Some(lineage_id))
            .cloned()
            .collect();
        if lineage_facts.is_empty() {
            continue;
        }

        let has = |kind| lineage_facts.iter().any(|fact| fact.kind() == kind);
        let population_delta = lineage_facts.iter().find_map(|fact| match fact {
            ReportFact::PopulationChange { delta_cells, .. } => Some(*delta_cells),
            _ => None,
        });
        let (theme, severity, lead_kind, rhetorical_shape) = if has(ReportFactKind::NearExtinction)
        {
            (
                ReportTheme::NearExtinction,
                FieldReportSeverity::Critical,
                ReportFactKind::NearExtinction,
                RhetoricalShape::AlertEvidenceAction,
            )
        } else if has(ReportFactKind::Recovery) {
            (
                ReportTheme::Recovery,
                FieldReportSeverity::Notable,
                ReportFactKind::Recovery,
                RhetoricalShape::ObservationEvidenceWatch,
            )
        } else if has(ReportFactKind::SustainedGrowth) && has(ReportFactKind::ResourcePressure) {
            (
                ReportTheme::StarvingExpansion,
                FieldReportSeverity::Warning,
                ReportFactKind::SustainedGrowth,
                RhetoricalShape::ObservationEvidenceWatch,
            )
        } else if has(ReportFactKind::SustainedDecline) {
            (
                ReportTheme::SustainedDecline,
                FieldReportSeverity::Warning,
                ReportFactKind::SustainedDecline,
                RhetoricalShape::ObservationEvidenceWatch,
            )
        } else if has(ReportFactKind::PopulationPeak) {
            (
                ReportTheme::NewPopulationPeak,
                FieldReportSeverity::Notable,
                ReportFactKind::PopulationPeak,
                RhetoricalShape::ObservationEvidence,
            )
        } else if has(ReportFactKind::ReproductiveReadiness) {
            (
                ReportTheme::ReproductivePulse,
                FieldReportSeverity::Notable,
                ReportFactKind::ReproductiveReadiness,
                RhetoricalShape::ObservationEvidence,
            )
        } else if population_delta.is_some_and(|delta| delta > 0) {
            (
                ReportTheme::PopulationBoom,
                FieldReportSeverity::Notable,
                ReportFactKind::PopulationChange,
                RhetoricalShape::ObservationEvidence,
            )
        } else if population_delta.is_some_and(|delta| delta < 0) {
            (
                ReportTheme::SustainedDecline,
                FieldReportSeverity::Notable,
                ReportFactKind::PopulationChange,
                RhetoricalShape::ObservationEvidenceWatch,
            )
        } else if has(ReportFactKind::SingleLineageProfile) {
            (
                ReportTheme::SingleLineageProfile,
                FieldReportSeverity::Notable,
                ReportFactKind::SingleLineageProfile,
                RhetoricalShape::ObservationEvidence,
            )
        } else if has(ReportFactKind::PopulationComposition) {
            (
                ReportTheme::CompositionShift,
                FieldReportSeverity::Routine,
                ReportFactKind::PopulationComposition,
                RhetoricalShape::Observation,
            )
        } else {
            (
                ReportTheme::TerritoryObservation,
                FieldReportSeverity::Routine,
                ReportFactKind::TerritoryExtent,
                RhetoricalShape::Observation,
            )
        };

        let lead_index = lineage_facts
            .iter()
            .position(|fact| fact.kind() == lead_kind)
            .unwrap_or(0);
        let lead_fact = lineage_facts[lead_index].clone();
        let supporting_facts = lineage_facts
            .into_iter()
            .enumerate()
            .filter_map(|(index, fact)| (index != lead_index).then_some(fact))
            .take(4)
            .collect::<Vec<_>>();
        let mut tags = Vec::new();
        for tag in lead_fact
            .tags()
            .iter()
            .chain(supporting_facts.iter().flat_map(|fact| fact.tags()))
        {
            if !tags.contains(tag) {
                tags.push(*tag);
            }
        }

        plans.push(ReportPlan {
            theme,
            subject_lineage: Some(lineage_id),
            supporting_lineages: Vec::new(),
            lead_fact,
            supporting_facts,
            severity,
            tags,
            rhetorical_shape,
            require_numeric_anchor: severity >= FieldReportSeverity::Notable,
        });
    }

    for fact in facts.iter().filter(|fact| fact.subject_lineage().is_none()) {
        let (theme, severity) = match fact {
            ReportFact::SceneComposition { .. } => (
                ReportTheme::SceneComposition,
                FieldReportSeverity::Notable,
            ),
            ReportFact::EcosystemDominance { .. } => (
                ReportTheme::EcosystemDominance,
                FieldReportSeverity::Warning,
            ),
            _ => (ReportTheme::EcosystemBalance, FieldReportSeverity::Routine),
        };
        plans.push(ReportPlan {
            theme,
            subject_lineage: None,
            supporting_lineages: Vec::new(),
            lead_fact: fact.clone(),
            supporting_facts: facts
                .iter()
                .filter(|candidate| {
                    candidate.subject_lineage().is_none()
                        && candidate.kind() != fact.kind()
                })
                .cloned()
                .collect(),
            severity,
            tags: fact.tags().to_vec(),
            rhetorical_shape: RhetoricalShape::ObservationEvidence,
            require_numeric_anchor: severity >= FieldReportSeverity::Notable,
        });
    }

    plans.sort_by(|a, b| b.severity.cmp(&a.severity));
    plans
}

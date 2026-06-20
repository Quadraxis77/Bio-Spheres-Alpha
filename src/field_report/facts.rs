use crate::field_report::context::FieldReportContext;
use crate::scene::lineage::LineageId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClaimConfidence {
    Observed,
    Derived,
    Tentative,
}

pub type ReportConfidence = ClaimConfidence;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldReportTag {
    PopulationGrowth,
    PopulationDecline,
    PopulationBoom,
    NearExtinction,
    Recovery,
    StarvationRisk,
    Reproduction,
    Composition,
    Territory,
    Dominance,
    Diversity,
    Stability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReportFactKind {
    PopulationChange,
    SustainedGrowth,
    SustainedDecline,
    Recovery,
    PopulationPeak,
    ResourcePressure,
    ReproductiveReadiness,
    PopulationComposition,
    TerritoryExtent,
    NearExtinction,
    EcosystemDominance,
    EcosystemBalance,
}

/// Objective semantic facts. No wording or tone belongs here.
#[derive(Debug, Clone, PartialEq)]
pub enum ReportFact {
    PopulationChange {
        lineage_id: LineageId,
        current_cells: u32,
        delta_cells: i32,
    },
    SustainedGrowth {
        lineage_id: LineageId,
        windows: u32,
    },
    SustainedDecline {
        lineage_id: LineageId,
        windows: u32,
    },
    Recovery {
        lineage_id: LineageId,
        current_cells: u32,
        previous_cells: u32,
    },
    PopulationPeak {
        lineage_id: LineageId,
        peak_cells: u32,
    },
    ResourcePressure {
        lineage_id: LineageId,
        avg_nutrient: f32,
        starvation_risk: f32,
        nutrient_positive: f32,
    },
    ReproductiveReadiness {
        lineage_id: LineageId,
        division_ready: f32,
        average_age: f32,
    },
    PopulationComposition {
        lineage_id: LineageId,
        dominant_cell_type: u32,
        dominant_fraction: f32,
        active_modes: u32,
    },
    TerritoryExtent {
        lineage_id: LineageId,
        center: [f32; 3],
        radius: f32,
    },
    NearExtinction {
        lineage_id: LineageId,
        remaining_cells: u32,
    },
    EcosystemDominance {
        largest_lineage_fraction: f32,
        active_lineages: u32,
    },
    EcosystemBalance {
        diversity_score: f32,
        evenness_score: f32,
        growing_lineages: u32,
        declining_lineages: u32,
    },
}

impl ReportFact {
    pub fn kind(&self) -> ReportFactKind {
        match self {
            Self::PopulationChange { .. } => ReportFactKind::PopulationChange,
            Self::SustainedGrowth { .. } => ReportFactKind::SustainedGrowth,
            Self::SustainedDecline { .. } => ReportFactKind::SustainedDecline,
            Self::Recovery { .. } => ReportFactKind::Recovery,
            Self::PopulationPeak { .. } => ReportFactKind::PopulationPeak,
            Self::ResourcePressure { .. } => ReportFactKind::ResourcePressure,
            Self::ReproductiveReadiness { .. } => ReportFactKind::ReproductiveReadiness,
            Self::PopulationComposition { .. } => ReportFactKind::PopulationComposition,
            Self::TerritoryExtent { .. } => ReportFactKind::TerritoryExtent,
            Self::NearExtinction { .. } => ReportFactKind::NearExtinction,
            Self::EcosystemDominance { .. } => ReportFactKind::EcosystemDominance,
            Self::EcosystemBalance { .. } => ReportFactKind::EcosystemBalance,
        }
    }

    pub fn subject_lineage(&self) -> Option<LineageId> {
        match self {
            Self::PopulationChange { lineage_id, .. }
            | Self::SustainedGrowth { lineage_id, .. }
            | Self::SustainedDecline { lineage_id, .. }
            | Self::Recovery { lineage_id, .. }
            | Self::PopulationPeak { lineage_id, .. }
            | Self::ResourcePressure { lineage_id, .. }
            | Self::ReproductiveReadiness { lineage_id, .. }
            | Self::PopulationComposition { lineage_id, .. }
            | Self::TerritoryExtent { lineage_id, .. }
            | Self::NearExtinction { lineage_id, .. } => Some(*lineage_id),
            Self::EcosystemDominance { .. } | Self::EcosystemBalance { .. } => None,
        }
    }

    pub fn confidence(&self) -> ClaimConfidence {
        match self {
            Self::PopulationChange { .. }
            | Self::Recovery { .. }
            | Self::PopulationPeak { .. }
            | Self::PopulationComposition { .. }
            | Self::TerritoryExtent { .. } => ClaimConfidence::Observed,
            _ => ClaimConfidence::Derived,
        }
    }

    pub fn tags(&self) -> &'static [FieldReportTag] {
        match self {
            Self::PopulationChange { delta_cells, .. } if *delta_cells > 0 => {
                &[FieldReportTag::PopulationGrowth]
            }
            Self::PopulationChange { .. } => &[FieldReportTag::PopulationDecline],
            Self::SustainedGrowth { .. } => &[FieldReportTag::PopulationGrowth],
            Self::SustainedDecline { .. } => &[FieldReportTag::PopulationDecline],
            Self::Recovery { .. } => &[FieldReportTag::Recovery],
            Self::PopulationPeak { .. } => &[FieldReportTag::PopulationBoom],
            Self::ResourcePressure { .. } => &[FieldReportTag::StarvationRisk],
            Self::ReproductiveReadiness { .. } => &[FieldReportTag::Reproduction],
            Self::PopulationComposition { .. } => &[FieldReportTag::Composition],
            Self::TerritoryExtent { .. } => &[FieldReportTag::Territory],
            Self::NearExtinction { .. } => &[FieldReportTag::NearExtinction],
            Self::EcosystemDominance { .. } => &[FieldReportTag::Dominance],
            Self::EcosystemBalance { .. } => &[FieldReportTag::Diversity],
        }
    }
}

pub fn extract_facts(context: &FieldReportContext) -> Vec<ReportFact> {
    let mut facts = Vec::new();

    for lineage in &context.lineages {
        let sample = lineage.current;
        if sample.cell_delta != 0 {
            facts.push(ReportFact::PopulationChange {
                lineage_id: lineage.lineage_id,
                current_cells: sample.cells,
                delta_cells: sample.cell_delta,
            });
        }
        if lineage.consecutive_growth_windows >= 2 {
            facts.push(ReportFact::SustainedGrowth {
                lineage_id: lineage.lineage_id,
                windows: lineage.consecutive_growth_windows,
            });
        }
        if lineage.consecutive_decline_windows >= 2 {
            facts.push(ReportFact::SustainedDecline {
                lineage_id: lineage.lineage_id,
                windows: lineage.consecutive_decline_windows,
            });
        }
        if let Some(previous) = lineage.previous {
            if previous.cells > 0 && previous.cells <= 5 && sample.cells > 5 {
                facts.push(ReportFact::Recovery {
                    lineage_id: lineage.lineage_id,
                    current_cells: sample.cells,
                    previous_cells: previous.cells,
                });
            }
        }
        if sample.cells > 0
            && sample.cells == lineage.peak_cells
            && lineage
                .previous
                .is_some_and(|previous| previous.cells < sample.cells)
        {
            facts.push(ReportFact::PopulationPeak {
                lineage_id: lineage.lineage_id,
                peak_cells: sample.cells,
            });
        }
        if sample.cells > 0 && sample.cells <= 5 {
            facts.push(ReportFact::NearExtinction {
                lineage_id: lineage.lineage_id,
                remaining_cells: sample.cells,
            });
        }
        if sample.starvation_risk_fraction >= 0.2 || sample.nutrient_positive_fraction <= 0.35 {
            facts.push(ReportFact::ResourcePressure {
                lineage_id: lineage.lineage_id,
                avg_nutrient: sample.avg_nutrient,
                starvation_risk: sample.starvation_risk_fraction,
                nutrient_positive: sample.nutrient_positive_fraction,
            });
        }
        if sample.division_ready_fraction >= 0.15 {
            facts.push(ReportFact::ReproductiveReadiness {
                lineage_id: lineage.lineage_id,
                division_ready: sample.division_ready_fraction,
                average_age: sample.average_age,
            });
        }
        if sample.dominant_cell_type_fraction >= 0.4 {
            facts.push(ReportFact::PopulationComposition {
                lineage_id: lineage.lineage_id,
                dominant_cell_type: sample.dominant_cell_type,
                dominant_fraction: sample.dominant_cell_type_fraction,
                active_modes: sample.active_mode_count,
            });
        }
        if sample.bounding_radius > 0.0 {
            facts.push(ReportFact::TerritoryExtent {
                lineage_id: lineage.lineage_id,
                center: sample.center,
                radius: sample.bounding_radius,
            });
        }
    }

    if context.ecosystem.active_lineages > 0 {
        if context.ecosystem.largest_lineage_fraction >= 0.5 {
            facts.push(ReportFact::EcosystemDominance {
                largest_lineage_fraction: context.ecosystem.largest_lineage_fraction,
                active_lineages: context.ecosystem.active_lineages,
            });
        }
        facts.push(ReportFact::EcosystemBalance {
            diversity_score: context.ecosystem.diversity_score,
            evenness_score: context.ecosystem.evenness_score,
            growing_lineages: context.ecosystem.growing_lineages,
            declining_lineages: context.ecosystem.declining_lineages,
        });
    }

    facts
}

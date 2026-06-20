use crate::field_report::plan::{ReportPlan, ReportTheme};

use super::{ToneFamily, ToneProfile};

pub fn render_title(plan: &ReportPlan, tone: &ToneProfile) -> String {
    match (plan.theme, tone.family()) {
        (ReportTheme::StarvingExpansion, ToneFamily::Formal) => {
            "Expansion Under Resource Pressure"
        }
        (ReportTheme::StarvingExpansion, ToneFamily::Naturalist) => "A Strained Bloom",
        (ReportTheme::StarvingExpansion, ToneFamily::Living) => "Hungry Bloom",
        (ReportTheme::StarvingExpansion, ToneFamily::Alert) => "Bloom Instability",
        (ReportTheme::NearExtinction, ToneFamily::Formal) => "Near-Extinction State",
        (ReportTheme::NearExtinction, ToneFamily::Naturalist) => "A Thin Remnant",
        (ReportTheme::NearExtinction, ToneFamily::Living) => "Holding at the Edge",
        (ReportTheme::NearExtinction, ToneFamily::Alert) => "Near-Extinction Warning",
        (ReportTheme::Recovery, ToneFamily::Formal) => "Population Recovery",
        (ReportTheme::Recovery, ToneFamily::Naturalist) => "Recovery at the Edge",
        (ReportTheme::Recovery, ToneFamily::Living) => "Holding On",
        (ReportTheme::Recovery, ToneFamily::Alert) => "Recovery Detected",
        (ReportTheme::NewPopulationPeak, ToneFamily::Formal) => "New Population Peak",
        (ReportTheme::NewPopulationPeak, ToneFamily::Naturalist) => "A New High",
        (ReportTheme::NewPopulationPeak, ToneFamily::Living) => "The Bloom Widens",
        (ReportTheme::NewPopulationPeak, ToneFamily::Alert) => "Population Peak Detected",
        (ReportTheme::SustainedDecline, ToneFamily::Formal) => "Sustained Population Decline",
        (ReportTheme::SustainedDecline, ToneFamily::Naturalist) => "A Continuing Retreat",
        (ReportTheme::SustainedDecline, ToneFamily::Living) => "The Lineage Thins",
        (ReportTheme::SustainedDecline, ToneFamily::Alert) => "Decline Continues",
        (ReportTheme::ReproductivePulse, ToneFamily::Formal) => "Reproductive Readiness",
        (ReportTheme::ReproductivePulse, ToneFamily::Naturalist) => "A Reproductive Pulse",
        (ReportTheme::ReproductivePulse, ToneFamily::Living) => "Ready to Divide",
        (ReportTheme::ReproductivePulse, ToneFamily::Alert) => "Division Readiness Elevated",
        (ReportTheme::EcosystemDominance, ToneFamily::Formal) => "Ecosystem Dominance Shift",
        (ReportTheme::EcosystemDominance, ToneFamily::Naturalist) => "One Lineage Takes Hold",
        (ReportTheme::EcosystemDominance, ToneFamily::Living) => "One Bloom Spreads Wide",
        (ReportTheme::EcosystemDominance, ToneFamily::Alert) => "Dominance Threshold Exceeded",
        (ReportTheme::EcosystemBalance, ToneFamily::Formal) => "Ecosystem Balance",
        (ReportTheme::EcosystemBalance, ToneFamily::Naturalist) => "Balance Across Lineages",
        (ReportTheme::EcosystemBalance, ToneFamily::Living) => "Many Lineages Holding",
        (ReportTheme::EcosystemBalance, ToneFamily::Alert) => "Diversity Status",
        (ReportTheme::PopulationBoom, ToneFamily::Formal) => "Population Expansion",
        (ReportTheme::PopulationBoom, ToneFamily::Naturalist) => "A New Foothold",
        (ReportTheme::PopulationBoom, ToneFamily::Living) => "The Bloom Spreads",
        (ReportTheme::PopulationBoom, ToneFamily::Alert) => "Population Increase Detected",
        (ReportTheme::CompositionShift, ToneFamily::Formal) => "Population Composition",
        (ReportTheme::CompositionShift, ToneFamily::Naturalist) => "The Surviving Form",
        (ReportTheme::CompositionShift, ToneFamily::Living) => "What the Bloom Has Become",
        (ReportTheme::CompositionShift, ToneFamily::Alert) => "Composition Status",
        (ReportTheme::TerritoryObservation, ToneFamily::Formal) => "Territory Observation",
        (ReportTheme::TerritoryObservation, ToneFamily::Naturalist) => "Range and Foothold",
        (ReportTheme::TerritoryObservation, ToneFamily::Living) => "Where It Holds",
        (ReportTheme::TerritoryObservation, ToneFamily::Alert) => "Territory Status",
        (_, ToneFamily::Any) => unreachable!("profiles always resolve to a concrete family"),
    }
    .to_string()
}

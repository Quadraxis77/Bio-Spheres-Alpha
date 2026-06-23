use crate::field_report::plan::{ReportPlan, ReportTheme};

use super::{ToneFamily, ToneProfile};

pub fn render_title(plan: &ReportPlan, tone: &ToneProfile) -> String {
    render_title_with_variation(plan, tone, 0)
}

pub fn render_title_with_variation(
    plan: &ReportPlan,
    tone: &ToneProfile,
    variation: usize,
) -> String {
    if plan.theme == ReportTheme::EcosystemDominance {
        let titles = match tone.family() {
            ToneFamily::Formal => [
                "Ecosystem Dominance Shift",
                "Lineage Concentration",
                "Reduced Ecosystem Evenness",
            ],
            ToneFamily::Naturalist => [
                "One Lineage Takes Hold",
                "An Uneven Living Field",
                "Growth Beneath the Majority",
            ],
            ToneFamily::Living => [
                "One Bloom Spreads Wide",
                "The Living Field Narrows",
                "What Moves Beneath the Bloom",
            ],
            ToneFamily::Alert => [
                "Dominance Threshold Exceeded",
                "Lineage Concentration Elevated",
                "Ecosystem Balance Degraded",
            ],
            ToneFamily::Any => unreachable!("profiles always resolve to a concrete family"),
        };
        return titles[variation % titles.len()].to_string();
    }
    if plan.theme == ReportTheme::SingleLineageProfile {
        let titles = match tone.family() {
            ToneFamily::Formal => [
                "Population Structure",
                "Metabolic Condition",
                "Reproductive Posture",
                "Spatial Occupation",
                "Population Stability",
            ],
            ToneFamily::Naturalist => [
                "The Shape Within",
                "How the Lineage Is Living",
                "The Next Generation",
                "The Reach of the Lineage",
                "Holding Steady",
            ],
            ToneFamily::Living => [
                "Inside the Bloom",
                "The Bloom's Condition",
                "Ready to Renew",
                "The Bloom's Reach",
                "The Rhythm of the Bloom",
            ],
            ToneFamily::Alert => [
                "Structural Assessment",
                "Resource Assessment",
                "Reproduction Assessment",
                "Territory Assessment",
                "Stability Assessment",
            ],
            ToneFamily::Any => unreachable!("profiles always resolve to a concrete family"),
        };
        return titles[variation % titles.len()].to_string();
    }
    if plan.theme == ReportTheme::SceneComposition {
        let titles = match tone.family() {
            ToneFamily::Formal => [
                "Functional Cell Composition",
                "Resource-Acquisition Profile",
                "Motility and Sensing Profile",
                "Structure and Reproduction Profile",
            ],
            ToneFamily::Naturalist => [
                "The Work of the Living Field",
                "How the Scene Feeds",
                "How the Scene Moves and Senses",
                "Holding Together, Making More",
            ],
            ToneFamily::Living => [
                "Many Cells, Many Tasks",
                "How the Field Finds Energy",
                "The Field in Motion",
                "Building and Becoming",
            ],
            ToneFamily::Alert => [
                "Cell Composition Update",
                "Resource Role Assessment",
                "Response Capacity Assessment",
                "Structural Investment Assessment",
            ],
            ToneFamily::Any => unreachable!("profiles always resolve to a concrete family"),
        };
        return titles[variation % titles.len()].to_string();
    }

    match (plan.theme, tone.family()) {
        (ReportTheme::StarvingExpansion, ToneFamily::Formal) => "Expansion Under Resource Pressure",
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
        (ReportTheme::EcosystemDominance, _) => unreachable!("handled above"),
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
        (ReportTheme::SingleLineageProfile, _) => unreachable!("handled above"),
        (ReportTheme::SceneComposition, _) => unreachable!("handled above"),
        (_, ToneFamily::Any) => unreachable!("profiles always resolve to a concrete family"),
    }
    .to_string()
}

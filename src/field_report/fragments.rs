use crate::field_report::facts::{ClaimConfidence, FieldReportTag};
use crate::field_report::grammar::SentenceRole;
use crate::field_report::plan::ReportTheme;
use crate::field_report::style::{NumberDensity, ToneFamily};

#[derive(Debug, Clone, Copy)]
pub struct ToneVariant {
    pub tone_family: ToneFamily,
    pub template: &'static str,
    pub weight: f32,
    pub number_density: NumberDensity,
}

#[derive(Debug, Clone, Copy)]
pub struct FragmentConcept {
    pub id: &'static str,
    pub role: SentenceRole,
    pub themes: &'static [ReportTheme],
    pub required_tags: &'static [FieldReportTag],
    pub forbidden_tags: &'static [FieldReportTag],
    pub confidence: ClaimConfidence,
    pub metaphor_cost: u8,
    pub variants: &'static [ToneVariant],
}

impl FragmentConcept {
    pub fn variant_for(&self, family: ToneFamily) -> Option<&ToneVariant> {
        self.variants
            .iter()
            .find(|variant| variant.tone_family == family)
            .or_else(|| {
                self.variants
                    .iter()
                    .find(|variant| variant.tone_family == ToneFamily::Any)
            })
    }
}

const fn variant(
    tone_family: ToneFamily,
    template: &'static str,
    number_density: NumberDensity,
) -> ToneVariant {
    ToneVariant {
        tone_family,
        template,
        weight: 1.0,
        number_density,
    }
}

static STARVING_EXPANSION_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "{lineage_ref} is expanding under resource pressure.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} is spreading quickly, but the growth looks strained.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} is blooming hard, but hunger is rising underneath it.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} expansion stress detected.",
        NumberDensity::Sparse,
    ),
];

static NEAR_EXTINCTION_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "{lineage_ref} has entered a near-extinction state.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} persists as a thin remnant.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} is holding at the edge of disappearance.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} near-extinction warning.",
        NumberDensity::Sparse,
    ),
];

static RECOVERY_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "{lineage_ref} is recovering from its recent population low.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} is finding room again after a narrow survival.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} is holding on and beginning to spread again.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} recovery detected.",
        NumberDensity::Sparse,
    ),
];

static DECLINE_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "{lineage_ref} remains in sustained decline.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} continues to thin across successive scans.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} keeps fading, scan after scan.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} decline continues.",
        NumberDensity::Sparse,
    ),
];

static GENERIC_GROWTH_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "{lineage_ref} is expanding.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} is establishing a broader foothold.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} is spreading into new room.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} population increase detected.",
        NumberDensity::Sparse,
    ),
];

static PEAK_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "{lineage_ref} has reached a new observed population peak.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} has reached its broadest observed population.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} has grown wider than at any earlier scan.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} new population peak detected.",
        NumberDensity::Sparse,
    ),
];

static REPRODUCTION_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "{lineage_ref} shows elevated reproductive readiness.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} appears to be entering a reproductive pulse.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} is gathering toward another round of division.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} division readiness elevated.",
        NumberDensity::Sparse,
    ),
];

static DOMINANCE_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "One lineage now holds a majority of active cells.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "A single lineage now occupies most of the living population.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "One bloom has spread across most of the living system.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "Ecosystem dominance threshold exceeded.",
        NumberDensity::Sparse,
    ),
];

static BALANCE_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "The current scan records the ecosystem's lineage balance.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "The current scan shows how evenly the living lineages are holding.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "The living system is being shared across several lineages.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "Ecosystem diversity status updated.",
        NumberDensity::Sparse,
    ),
];

static DOSSIER_VARIANTS: [ToneVariant; 4] = [
    variant(
        ToneFamily::Formal,
        "{lineage_ref} is currently defined by its sampled population structure.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} is settling into a recognizable form.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} has taken on a clear living shape.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} composition status recorded.",
        NumberDensity::Sparse,
    ),
];

pub static STARVING_EXPANSION_LEAD: FragmentConcept = FragmentConcept {
    id: "starving_expansion_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::StarvingExpansion],
    required_tags: &[
        FieldReportTag::PopulationGrowth,
        FieldReportTag::StarvationRisk,
    ],
    forbidden_tags: &[FieldReportTag::NearExtinction],
    confidence: ClaimConfidence::Derived,
    metaphor_cost: 1,
    variants: &STARVING_EXPANSION_VARIANTS,
};

static NEAR_EXTINCTION_LEAD: FragmentConcept = FragmentConcept {
    id: "near_extinction_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::NearExtinction],
    required_tags: &[FieldReportTag::NearExtinction],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Observed,
    metaphor_cost: 1,
    variants: &NEAR_EXTINCTION_VARIANTS,
};

static RECOVERY_LEAD: FragmentConcept = FragmentConcept {
    id: "recovery_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::Recovery],
    required_tags: &[FieldReportTag::Recovery],
    forbidden_tags: &[FieldReportTag::NearExtinction],
    confidence: ClaimConfidence::Observed,
    metaphor_cost: 1,
    variants: &RECOVERY_VARIANTS,
};

static DECLINE_LEAD: FragmentConcept = FragmentConcept {
    id: "sustained_decline_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::SustainedDecline],
    required_tags: &[FieldReportTag::PopulationDecline],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Derived,
    metaphor_cost: 1,
    variants: &DECLINE_VARIANTS,
};

static GROWTH_LEAD: FragmentConcept = FragmentConcept {
    id: "population_growth_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::PopulationBoom],
    required_tags: &[FieldReportTag::PopulationGrowth],
    forbidden_tags: &[FieldReportTag::NearExtinction],
    confidence: ClaimConfidence::Observed,
    metaphor_cost: 1,
    variants: &GENERIC_GROWTH_VARIANTS,
};

static PEAK_LEAD: FragmentConcept = FragmentConcept {
    id: "population_peak_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::NewPopulationPeak],
    required_tags: &[FieldReportTag::PopulationBoom],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Observed,
    metaphor_cost: 1,
    variants: &PEAK_VARIANTS,
};

static REPRODUCTION_LEAD: FragmentConcept = FragmentConcept {
    id: "reproductive_pulse_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::ReproductivePulse],
    required_tags: &[FieldReportTag::Reproduction],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Derived,
    metaphor_cost: 1,
    variants: &REPRODUCTION_VARIANTS,
};

static DOMINANCE_LEAD: FragmentConcept = FragmentConcept {
    id: "ecosystem_dominance_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::EcosystemDominance],
    required_tags: &[FieldReportTag::Dominance],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Derived,
    metaphor_cost: 1,
    variants: &DOMINANCE_VARIANTS,
};

static BALANCE_LEAD: FragmentConcept = FragmentConcept {
    id: "ecosystem_balance_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::EcosystemBalance],
    required_tags: &[FieldReportTag::Diversity],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Observed,
    metaphor_cost: 0,
    variants: &BALANCE_VARIANTS,
};

static DOSSIER_LEAD: FragmentConcept = FragmentConcept {
    id: "lineage_dossier_lead",
    role: SentenceRole::Lead,
    themes: &[
        ReportTheme::CompositionShift,
        ReportTheme::TerritoryObservation,
    ],
    required_tags: &[],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Observed,
    metaphor_cost: 1,
    variants: &DOSSIER_VARIANTS,
};

pub fn lead_concept(theme: ReportTheme) -> &'static FragmentConcept {
    match theme {
        ReportTheme::StarvingExpansion => &STARVING_EXPANSION_LEAD,
        ReportTheme::NearExtinction => &NEAR_EXTINCTION_LEAD,
        ReportTheme::Recovery => &RECOVERY_LEAD,
        ReportTheme::SustainedDecline => &DECLINE_LEAD,
        ReportTheme::PopulationBoom => &GROWTH_LEAD,
        ReportTheme::NewPopulationPeak => &PEAK_LEAD,
        ReportTheme::ReproductivePulse => &REPRODUCTION_LEAD,
        ReportTheme::EcosystemDominance => &DOMINANCE_LEAD,
        ReportTheme::EcosystemBalance => &BALANCE_LEAD,
        ReportTheme::CompositionShift | ReportTheme::TerritoryObservation => &DOSSIER_LEAD,
    }
}

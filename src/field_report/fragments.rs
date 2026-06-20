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
    pub fn variant_for(&self, family: ToneFamily, variation: usize) -> Option<&ToneVariant> {
        let matching: Vec<_> = self
            .variants
            .iter()
            .filter(|variant| {
                variant.tone_family == family || variant.tone_family == ToneFamily::Any
            })
            .collect();
        (!matching.is_empty()).then(|| matching[variation % matching.len()])
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

static STARVING_EXPANSION_VARIANTS: [ToneVariant; 8] = [
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
    variant(
        ToneFamily::Formal,
        "{lineage_ref} continues to grow while resource indicators weaken.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} is still gaining ground, though fewer cells are carrying the growth well.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} keeps spreading, but the new growth is running thin.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} growth persists under declining resource health.",
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

static DECLINE_VARIANTS: [ToneVariant; 8] = [
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
    variant(
        ToneFamily::Formal,
        "{lineage_ref} has not recovered from its recent population losses.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} has yet to find a stable floor.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} is still searching for a place to hold.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} population stabilization has not occurred.",
        NumberDensity::Sparse,
    ),
];

static GENERIC_GROWTH_VARIANTS: [ToneVariant; 8] = [
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
    variant(
        ToneFamily::Formal,
        "{lineage_ref} has extended its recent growth trend.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "{lineage} continues to establish itself across the biosphere.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "{lineage} is finding more room to hold.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "{lineage} sustained growth confirmed.",
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

static DOMINANCE_VARIANTS: [ToneVariant; 12] = [
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
    variant(
        ToneFamily::Formal,
        "The living population is becoming less evenly distributed among lineages.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "The living field has become markedly uneven.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "The wider bloom is leaving less room among its neighbors.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "Ecosystem evenness has fallen under concentrated lineage growth.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Formal,
        "The majority lineage now defines the ecosystem's broader trajectory.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Naturalist,
        "The majority is clear, but the smaller lineages reveal whether that hold is still tightening.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Living,
        "One bloom fills the view; the motion beneath it tells whether the field is narrowing further.",
        NumberDensity::Sparse,
    ),
    variant(
        ToneFamily::Alert,
        "Minor-lineage trends now determine whether dominance is stabilizing or intensifying.",
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

static SINGLE_LINEAGE_VARIANTS: [ToneVariant; 20] = [
    variant(ToneFamily::Formal, "{lineage_ref} is being assessed by population structure.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "{lineage} is alone here; its internal structure is now the more useful story.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "{lineage} fills the living field, so the shape inside the bloom matters most.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "{lineage} single-lineage structural assessment.", NumberDensity::Sparse),
    variant(ToneFamily::Formal, "{lineage_ref} is being assessed by metabolic condition.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "With no rival lineage present, {lineage}'s resource condition becomes the sharper measure.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "The field belongs to {lineage}; the question is how comfortably it is living.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "{lineage} single-lineage resource assessment.", NumberDensity::Sparse),
    variant(ToneFamily::Formal, "{lineage_ref} is being assessed by reproductive posture.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "{lineage}'s next change is more likely to come from within than from competition.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "Nothing contests the bloom; its readiness to renew itself is the movement to watch.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "{lineage} single-lineage reproduction assessment.", NumberDensity::Sparse),
    variant(ToneFamily::Formal, "{lineage_ref} is being assessed by spatial occupation.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "{lineage}'s reach now says more than its uncontested rank.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "The bloom has no rival edge, but it still has a shape and a reach.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "{lineage} single-lineage territory assessment.", NumberDensity::Sparse),
    variant(ToneFamily::Formal, "{lineage_ref} is being assessed by short-term population stability.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "The useful question is no longer who leads, but whether {lineage} is holding steady.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "The bloom is alone; now its rhythm of gain and loss carries the story.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "{lineage} single-lineage stability assessment.", NumberDensity::Sparse),
];

static SCENE_COMPOSITION_VARIANTS: [ToneVariant; 16] = [
    variant(ToneFamily::Formal, "The scene's functional cell composition has been sampled.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "The latest scan shows what kinds of work the living cells are built to perform.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "The living field has a new balance of gatherers, movers, sensors, and builders.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "Scene cell-type composition updated.", NumberDensity::Sparse),
    variant(ToneFamily::Formal, "The scene's resource-acquisition profile has been assessed.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "The balance of feeding, photosynthetic, and storage cells reveals how the scene is provisioning itself.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "The field's way of finding and keeping energy is coming into focus.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "Scene resource-role composition assessed.", NumberDensity::Sparse),
    variant(ToneFamily::Formal, "The scene's movement and sensing profile has been assessed.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "The mix of movers and sensing cells suggests how actively the population can search and respond.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "The field's capacity to move, feel, and react is shifting.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "Scene motility and sensing composition assessed.", NumberDensity::Sparse),
    variant(ToneFamily::Formal, "The scene's structural and reproductive profile has been assessed.", NumberDensity::Sparse),
    variant(ToneFamily::Naturalist, "Support cells and reproductive cells show how the population is investing beyond immediate survival.", NumberDensity::Sparse),
    variant(ToneFamily::Living, "The field is dividing its effort between holding together and making what comes next.", NumberDensity::Sparse),
    variant(ToneFamily::Alert, "Scene structure and reproduction composition assessed.", NumberDensity::Sparse),
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

static SINGLE_LINEAGE_LEAD: FragmentConcept = FragmentConcept {
    id: "single_lineage_profile_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::SingleLineageProfile],
    required_tags: &[FieldReportTag::Stability],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Observed,
    metaphor_cost: 1,
    variants: &SINGLE_LINEAGE_VARIANTS,
};

static SCENE_COMPOSITION_LEAD: FragmentConcept = FragmentConcept {
    id: "scene_composition_lead",
    role: SentenceRole::Lead,
    themes: &[ReportTheme::SceneComposition],
    required_tags: &[FieldReportTag::Composition],
    forbidden_tags: &[],
    confidence: ClaimConfidence::Observed,
    metaphor_cost: 1,
    variants: &SCENE_COMPOSITION_VARIANTS,
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
        ReportTheme::SingleLineageProfile => &SINGLE_LINEAGE_LEAD,
        ReportTheme::SceneComposition => &SCENE_COMPOSITION_LEAD,
        ReportTheme::CompositionShift | ReportTheme::TerritoryObservation => &DOSSIER_LEAD,
    }
}

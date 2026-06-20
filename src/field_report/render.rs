use std::collections::HashSet;

use crate::field_report::context::FieldReportContext;
use crate::field_report::facts::{ClaimConfidence, FieldReportTag, ReportFact};
use crate::field_report::fragments::{lead_concept, FragmentConcept};
use crate::field_report::grammar::{role_allows_supporting_detail, ClaimKey, SentenceRole};
use crate::field_report::plan::{
    FieldReportSeverity, ReportPlan, ReportTheme, RhetoricalShape,
};
use crate::field_report::style::{
    format_cell_count, format_delta_cells, format_percent, render_title, resolve_tone,
    NumberDensity, ToneFamily, ToneId, ToneLexicon, ToneProfile,
};
use crate::scene::lineage::LineageId;

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedSentence {
    pub text: String,
    pub role: SentenceRole,
    pub confidence: crate::field_report::facts::ReportConfidence,
    pub claim_keys: Vec<ClaimKey>,
    pub fragment_id: &'static str,
    pub numeric_anchor: bool,
    pub metaphor_cost: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedFieldReport {
    pub title: String,
    pub body: String,
    pub sentences: Vec<RenderedSentence>,
    pub theme: ReportTheme,
    pub severity: FieldReportSeverity,
    pub tags: Vec<FieldReportTag>,
    pub confidence: crate::field_report::facts::ReportConfidence,
    pub involved_lineages: Vec<LineageId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoherenceError {
    EmptyTitle,
    EmptyBody,
    MissingNumericAnchor,
    MetaphorBudgetExceeded,
    UnsupportedCausality,
    ExtinctionWordingForNearExtinction,
    DuplicateClaim(ClaimKey),
    TooManySentences,
    MysticalLanguage,
}

pub fn render_report(
    context: &FieldReportContext,
    plan: &ReportPlan,
    requested_tone: &ToneProfile,
) -> Option<RenderedFieldReport> {
    let tone = resolve_tone(*requested_tone, plan.severity);
    let subject = plan
        .subject_lineage
        .and_then(|lineage_id| context.lineage(lineage_id));
    if plan.subject_lineage.is_some() && subject.is_none() {
        return None;
    }

    let lineage_name = subject
        .map(|lineage| lineage.display_name.as_str())
        .unwrap_or("the ecosystem");
    let lexicon = ToneLexicon::for_tone(&tone);
    let lineage_ref = format!("{}{}", lexicon.lineage_prefix, lineage_name);

    let mut sentences = Vec::new();
    let concept = lead_concept(plan.theme);
    if concept_allowed(concept, plan) {
        let variant = concept.variant_for(tone.family())?;
        let text = fill_template(
            variant.template,
            &[("lineage", lineage_name), ("lineage_ref", &lineage_ref)],
        );
        sentences.push(RenderedSentence {
            numeric_anchor: contains_numeric_anchor(&text),
            text,
            role: concept.role,
            confidence: concept.confidence,
            claim_keys: lead_claims(plan),
            fragment_id: concept.id,
            metaphor_cost: metaphor_cost(concept, tone.family()),
        });
    }

    if let Some(evidence) = render_evidence(plan, &tone, lineage_name) {
        push_coherent(&mut sentences, evidence);
    }

    if matches!(plan.theme, ReportTheme::StarvingExpansion) {
        if let Some(contrast) = render_starving_contrast(plan, &tone, lineage_name) {
            push_coherent(&mut sentences, contrast);
        }
    } else if matches!(
        plan.rhetorical_shape,
        RhetoricalShape::ObservationEvidenceWatch | RhetoricalShape::AlertEvidenceAction
    ) {
        if let Some(interpretation) = render_interpretation(plan, &tone, lineage_name) {
            push_coherent(&mut sentences, interpretation);
        }
    }

    if matches!(
        plan.rhetorical_shape,
        RhetoricalShape::ObservationEvidenceWatch | RhetoricalShape::AlertEvidenceAction
    ) {
        push_coherent(&mut sentences, render_watch(plan, &tone, lineage_name));
    }

    if plan.require_numeric_anchor && !sentences.iter().any(|sentence| sentence.numeric_anchor) {
        let anchor = fallback_numeric_anchor(plan, &tone, lineage_name)?;
        push_coherent(&mut sentences, anchor);
    }

    let max_sentences = target_sentence_count(plan.rhetorical_shape);
    sentences.truncate(max_sentences);
    let body = sentences
        .iter()
        .map(|sentence| sentence.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let confidence = report_confidence(&sentences);
    let mut involved_lineages = Vec::new();
    if let Some(lineage_id) = plan.subject_lineage {
        involved_lineages.push(lineage_id);
    }
    for lineage_id in &plan.supporting_lineages {
        if !involved_lineages.contains(lineage_id) {
            involved_lineages.push(*lineage_id);
        }
    }

    let report = RenderedFieldReport {
        title: render_title(plan, &tone),
        body,
        sentences,
        theme: plan.theme,
        severity: plan.severity,
        tags: plan.tags.clone(),
        confidence,
        involved_lineages,
    };

    validate_report(&report, &tone).is_empty().then_some(report)
}

pub fn render_report_all_tones(
    context: &FieldReportContext,
    plan: &ReportPlan,
) -> Vec<(ToneId, RenderedFieldReport)> {
    [
        ToneProfile::formal_scientific(),
        ToneProfile::naturalist_field_journal(),
        ToneProfile::living_ecosystem(),
        ToneProfile::alert_monitor(),
    ]
    .into_iter()
    .filter_map(|tone| render_report(context, plan, &tone).map(|report| (tone.id, report)))
    .collect()
}

pub fn validate_report(
    report: &RenderedFieldReport,
    tone: &ToneProfile,
) -> Vec<CoherenceError> {
    let mut errors = Vec::new();
    if report.title.trim().is_empty() {
        errors.push(CoherenceError::EmptyTitle);
    }
    if report.body.trim().is_empty() {
        errors.push(CoherenceError::EmptyBody);
    }
    if report.severity >= FieldReportSeverity::Notable
        && !report
            .sentences
            .iter()
            .any(|sentence| sentence.numeric_anchor)
    {
        errors.push(CoherenceError::MissingNumericAnchor);
    }
    let metaphor_total: u8 = report
        .sentences
        .iter()
        .map(|sentence| sentence.metaphor_cost)
        .sum();
    if metaphor_total > tone.metaphor_budget {
        errors.push(CoherenceError::MetaphorBudgetExceeded);
    }
    let lower = report.body.to_ascii_lowercase();
    if lower.contains(" because ") {
        errors.push(CoherenceError::UnsupportedCausality);
    }
    if report.theme == ReportTheme::NearExtinction
        && (lower.contains("is extinct") || lower.contains("has vanished"))
    {
        errors.push(CoherenceError::ExtinctionWordingForNearExtinction);
    }
    if ["spirit", "destiny", "life force", "nature whispers"]
        .iter()
        .any(|forbidden| lower.contains(forbidden))
    {
        errors.push(CoherenceError::MysticalLanguage);
    }
    if report.sentences.len() > 4 {
        errors.push(CoherenceError::TooManySentences);
    }

    let mut used = HashSet::new();
    for sentence in &report.sentences {
        for claim in &sentence.claim_keys {
            if !used.insert(*claim) && !role_allows_supporting_detail(sentence.role) {
                errors.push(CoherenceError::DuplicateClaim(*claim));
            }
        }
    }
    errors
}

fn concept_allowed(concept: &FragmentConcept, plan: &ReportPlan) -> bool {
    concept.themes.contains(&plan.theme)
        && concept
            .required_tags
            .iter()
            .all(|tag| plan.tags.contains(tag))
        && concept
            .forbidden_tags
            .iter()
            .all(|tag| !plan.tags.contains(tag))
}

fn metaphor_cost(concept: &FragmentConcept, family: ToneFamily) -> u8 {
    if matches!(family, ToneFamily::Naturalist | ToneFamily::Living) {
        concept.metaphor_cost
    } else {
        0
    }
}

fn lead_claims(plan: &ReportPlan) -> Vec<ClaimKey> {
    let lineage_id = plan.subject_lineage;
    match (plan.theme, lineage_id) {
        (ReportTheme::StarvingExpansion, Some(id)) => {
            vec![ClaimKey::PopulationIncrease(id), ClaimKey::NutrientPressure(id)]
        }
        (ReportTheme::PopulationBoom, Some(id)) => vec![ClaimKey::PopulationIncrease(id)],
        (ReportTheme::SustainedDecline, Some(id)) => vec![ClaimKey::PopulationDecline(id)],
        (ReportTheme::Recovery, Some(id)) => vec![ClaimKey::Recovery(id)],
        (ReportTheme::NewPopulationPeak, Some(id)) => vec![ClaimKey::PopulationPeak(id)],
        (ReportTheme::NearExtinction, Some(id)) => vec![ClaimKey::NearExtinction(id)],
        (ReportTheme::ReproductivePulse, Some(id)) => vec![ClaimKey::DivisionReadiness(id)],
        (ReportTheme::CompositionShift, Some(id)) => vec![ClaimKey::DominantCellType(id)],
        (ReportTheme::TerritoryObservation, Some(id)) => vec![ClaimKey::Territory(id)],
        (ReportTheme::EcosystemDominance, _) => vec![ClaimKey::EcosystemDominance],
        (ReportTheme::EcosystemBalance, _) => vec![ClaimKey::EcosystemDiversity],
        _ => Vec::new(),
    }
}

fn render_evidence(
    plan: &ReportPlan,
    tone: &ToneProfile,
    lineage: &str,
) -> Option<RenderedSentence> {
    let mut facts = std::iter::once(&plan.lead_fact).chain(plan.supporting_facts.iter());
    let population = facts.clone().find_map(|fact| match fact {
        ReportFact::PopulationChange {
            current_cells,
            delta_cells,
            ..
        } => Some((*current_cells, *delta_cells)),
        _ => None,
    });
    let resources = facts.clone().find_map(|fact| match fact {
        ReportFact::ResourcePressure {
            avg_nutrient,
            starvation_risk,
            nutrient_positive,
            ..
        } => Some((*avg_nutrient, *starvation_risk, *nutrient_positive)),
        _ => None,
    });

    let (text, claims, confidence, id) = match plan.theme {
        ReportTheme::StarvingExpansion => {
            let (current, delta) = population?;
            let (_, starvation, positive) = resources?;
            (
                format!(
                    "The population changed by {} cells to {}, while starvation risk reached {} and {} of cells remained nutrient-positive.",
                    format_delta_cells(delta, tone, NumberDensity::Precise),
                    format_cell_count(current, tone, NumberDensity::Precise),
                    format_percent(starvation, tone, NumberDensity::Precise),
                    format_percent(positive, tone, NumberDensity::Precise),
                ),
                vec![
                    ClaimKey::PopulationIncrease(plan.subject_lineage?),
                    ClaimKey::NutrientPressure(plan.subject_lineage?),
                ],
                ClaimConfidence::Observed,
                "starving_expansion_evidence",
            )
        }
        ReportTheme::NearExtinction => {
            let remaining = facts.find_map(|fact| match fact {
                ReportFact::NearExtinction {
                    remaining_cells, ..
                } => Some(*remaining_cells),
                _ => None,
            })?;
            (
                format!(
                    "The latest scan found {} living cells remaining.",
                    format_cell_count(remaining, tone, NumberDensity::Precise)
                ),
                vec![ClaimKey::NearExtinction(plan.subject_lineage?)],
                ClaimConfidence::Observed,
                "near_extinction_evidence",
            )
        }
        ReportTheme::Recovery => {
            let (current, previous) = facts.find_map(|fact| match fact {
                ReportFact::Recovery {
                    current_cells,
                    previous_cells,
                    ..
                } => Some((*current_cells, *previous_cells)),
                _ => None,
            })?;
            (
                format!(
                    "The sampled population rose from {} to {} cells.",
                    format_cell_count(previous, tone, NumberDensity::Precise),
                    format_cell_count(current, tone, NumberDensity::Precise),
                ),
                vec![ClaimKey::Recovery(plan.subject_lineage?)],
                ClaimConfidence::Observed,
                "recovery_evidence",
            )
        }
        ReportTheme::SustainedDecline => {
            let windows = facts
                .clone()
                .find_map(|fact| match fact {
                    ReportFact::SustainedDecline { windows, .. } => Some(*windows),
                    _ => None,
                })
                .unwrap_or(1);
            let delta = population.map(|(_, delta)| delta).unwrap_or_default();
            (
                format!(
                    "Decline has continued for {} scan windows, with a latest change of {} cells.",
                    windows,
                    format_delta_cells(delta, tone, NumberDensity::Precise)
                ),
                vec![ClaimKey::PopulationDecline(plan.subject_lineage?)],
                ClaimConfidence::Derived,
                "sustained_decline_evidence",
            )
        }
        ReportTheme::NewPopulationPeak => {
            let peak = facts.find_map(|fact| match fact {
                ReportFact::PopulationPeak { peak_cells, .. } => Some(*peak_cells),
                _ => None,
            })?;
            (
                format!(
                    "The latest scan recorded {} cells, the highest retained value for this lineage.",
                    format_cell_count(peak, tone, NumberDensity::Precise)
                ),
                vec![ClaimKey::PopulationPeak(plan.subject_lineage?)],
                ClaimConfidence::Observed,
                "population_peak_evidence",
            )
        }
        ReportTheme::ReproductivePulse => {
            let (ready, age) = facts.find_map(|fact| match fact {
                ReportFact::ReproductiveReadiness {
                    division_ready,
                    average_age,
                    ..
                } => Some((*division_ready, *average_age)),
                _ => None,
            })?;
            (
                format!(
                    "{} of sampled cells are division-ready; average age is {:.1} seconds.",
                    format_percent(ready, tone, NumberDensity::Precise),
                    age
                ),
                vec![ClaimKey::DivisionReadiness(plan.subject_lineage?)],
                ClaimConfidence::Observed,
                "reproductive_pulse_evidence",
            )
        }
        ReportTheme::PopulationBoom => {
            let (current, delta) = population?;
            (
                format!(
                    "{} gained {} cells and now contains {}.",
                    lineage,
                    format_cell_count(delta.unsigned_abs(), tone, NumberDensity::Precise),
                    format_cell_count(current, tone, NumberDensity::Precise)
                ),
                vec![ClaimKey::PopulationIncrease(plan.subject_lineage?)],
                ClaimConfidence::Observed,
                "population_growth_evidence",
            )
        }
        ReportTheme::CompositionShift => {
            let (cell_type, fraction, modes) = facts.find_map(|fact| match fact {
                ReportFact::PopulationComposition {
                    dominant_cell_type,
                    dominant_fraction,
                    active_modes,
                    ..
                } => Some((*dominant_cell_type, *dominant_fraction, *active_modes)),
                _ => None,
            })?;
            let type_name = crate::cell::types::CellType::names()
                .get(cell_type as usize)
                .copied()
                .unwrap_or("Unknown");
            (
                format!(
                    "{} cells account for {} of the sample across {} active modes.",
                    type_name,
                    format_percent(fraction, tone, NumberDensity::Precise),
                    modes
                ),
                vec![ClaimKey::DominantCellType(plan.subject_lineage?)],
                ClaimConfidence::Observed,
                "composition_evidence",
            )
        }
        ReportTheme::TerritoryObservation => {
            let radius = facts.find_map(|fact| match fact {
                ReportFact::TerritoryExtent { radius, .. } => Some(*radius),
                _ => None,
            })?;
            (
                format!("The sampled territory has an approximate radius of {radius:.1} units."),
                vec![ClaimKey::Territory(plan.subject_lineage?)],
                ClaimConfidence::Observed,
                "territory_evidence",
            )
        }
        ReportTheme::EcosystemDominance => {
            let (fraction, active) = facts.find_map(|fact| match fact {
                ReportFact::EcosystemDominance {
                    largest_lineage_fraction,
                    active_lineages,
                } => Some((*largest_lineage_fraction, *active_lineages)),
                _ => None,
            })?;
            (
                format!(
                    "The largest lineage holds {} of active cells across {} living lineages.",
                    format_percent(fraction, tone, NumberDensity::Precise),
                    active
                ),
                vec![ClaimKey::EcosystemDominance],
                ClaimConfidence::Observed,
                "ecosystem_dominance_evidence",
            )
        }
        ReportTheme::EcosystemBalance => {
            let (evenness, growing, declining) = facts.find_map(|fact| match fact {
                ReportFact::EcosystemBalance {
                    evenness_score,
                    growing_lineages,
                    declining_lineages,
                    ..
                } => Some((*evenness_score, *growing_lineages, *declining_lineages)),
                _ => None,
            })?;
            (
                format!(
                    "Evenness is {}, with {} lineages growing and {} declining.",
                    format_percent(evenness, tone, NumberDensity::Precise),
                    growing,
                    declining
                ),
                vec![ClaimKey::EcosystemDiversity],
                ClaimConfidence::Observed,
                "ecosystem_balance_evidence",
            )
        }
    };

    Some(RenderedSentence {
        numeric_anchor: true,
        text,
        role: SentenceRole::Evidence,
        confidence,
        claim_keys: claims,
        fragment_id: id,
        metaphor_cost: 0,
    })
}

fn render_starving_contrast(
    plan: &ReportPlan,
    tone: &ToneProfile,
    lineage: &str,
) -> Option<RenderedSentence> {
    let resource = plan
        .supporting_facts
        .iter()
        .chain(std::iter::once(&plan.lead_fact))
        .find_map(|fact| match fact {
            ReportFact::ResourcePressure {
                starvation_risk,
                nutrient_positive,
                ..
            } => Some((*starvation_risk, *nutrient_positive)),
            _ => None,
        })?;
    let text = match tone.family() {
        ToneFamily::Formal => format!(
            "The expansion is not yet evidence of durable growth: starvation risk is {}.",
            format_percent(resource.0, tone, NumberDensity::Rounded)
        ),
        ToneFamily::Naturalist => format!(
            "The new growth is not carrying comfortably; only {} of cells are gaining nutrients.",
            format_percent(resource.1, tone, NumberDensity::Rounded)
        ),
        ToneFamily::Living => format!(
            "The bloom is widening, but only {} of its cells are gaining nutrients.",
            format_percent(resource.1, tone, NumberDensity::Precise)
        ),
        ToneFamily::Alert => format!("{lineage} growth remains resource-limited."),
        ToneFamily::Any => unreachable!(),
    };
    Some(RenderedSentence {
        numeric_anchor: contains_numeric_anchor(&text),
        text,
        role: SentenceRole::Contrast,
        confidence: ClaimConfidence::Derived,
        claim_keys: vec![ClaimKey::NutrientPressure(plan.subject_lineage?)],
        fragment_id: "starving_expansion_contrast",
        metaphor_cost: u8::from(matches!(tone.family(), ToneFamily::Living)),
    })
}

fn render_interpretation(
    plan: &ReportPlan,
    tone: &ToneProfile,
    lineage: &str,
) -> Option<RenderedSentence> {
    let text = match plan.theme {
        ReportTheme::NearExtinction => match tone.family() {
            ToneFamily::Formal => {
                "Current evidence indicates severe population fragility, not confirmed extinction."
                    .to_string()
            }
            ToneFamily::Naturalist => {
                "The lineage remains present, but its foothold is extremely narrow.".to_string()
            }
            ToneFamily::Living => {
                "A few cells still hold on; the lineage has not disappeared.".to_string()
            }
            ToneFamily::Alert => "Extinction is not confirmed; survival remains critical.".to_string(),
            ToneFamily::Any => unreachable!(),
        },
        ReportTheme::SustainedDecline => {
            format!("The retained pattern suggests that {lineage} has not yet stabilized.")
        }
        ReportTheme::Recovery => {
            format!("The signs point toward recovery, though another scan is needed to confirm it.")
        }
        _ => return None,
    };
    Some(RenderedSentence {
        numeric_anchor: false,
        text,
        role: SentenceRole::Interpretation,
        confidence: ClaimConfidence::Tentative,
        claim_keys: Vec::new(),
        fragment_id: "cautious_interpretation",
        metaphor_cost: 0,
    })
}

fn render_watch(plan: &ReportPlan, tone: &ToneProfile, lineage: &str) -> RenderedSentence {
    let lexicon = ToneLexicon::for_tone(tone);
    let subject = if plan.subject_lineage.is_some() {
        lineage
    } else {
        "the ecosystem"
    };
    let focus = match plan.theme {
        ReportTheme::StarvingExpansion => "resource pressure and division readiness",
        ReportTheme::NearExtinction => "survival or disappearance in the next scan",
        ReportTheme::SustainedDecline => "continued population loss",
        ReportTheme::Recovery => "whether the recovery persists",
        ReportTheme::EcosystemDominance => "further concentration in one lineage",
        _ => "the next telemetry window",
    };
    RenderedSentence {
        text: format!("{} {subject} for {focus}.", lexicon.watch),
        role: SentenceRole::Watch,
        confidence: ClaimConfidence::Tentative,
        claim_keys: Vec::new(),
        fragment_id: "report_watch",
        numeric_anchor: false,
        metaphor_cost: 0,
    }
}

fn fallback_numeric_anchor(
    plan: &ReportPlan,
    tone: &ToneProfile,
    lineage: &str,
) -> Option<RenderedSentence> {
    let fact = std::iter::once(&plan.lead_fact)
        .chain(plan.supporting_facts.iter())
        .find(|fact| fact.subject_lineage() == plan.subject_lineage)?;
    let text = match fact {
        ReportFact::PopulationChange {
            current_cells,
            delta_cells,
            ..
        } => format!(
            "{lineage} changed by {} cells to {}.",
            format_delta_cells(*delta_cells, tone, NumberDensity::Precise),
            format_cell_count(*current_cells, tone, NumberDensity::Precise)
        ),
        ReportFact::NearExtinction {
            remaining_cells, ..
        } => format!("{remaining_cells} living cells remain in the latest scan."),
        ReportFact::PopulationPeak { peak_cells, .. } => {
            format!("The retained population peak is {peak_cells} cells.")
        }
        _ => return None,
    };
    Some(RenderedSentence {
        text,
        role: SentenceRole::Evidence,
        confidence: fact.confidence(),
        claim_keys: Vec::new(),
        fragment_id: "numeric_precision_anchor",
        numeric_anchor: true,
        metaphor_cost: 0,
    })
}

fn push_coherent(sentences: &mut Vec<RenderedSentence>, candidate: RenderedSentence) {
    let duplicate_non_detail = candidate.claim_keys.iter().any(|claim| {
        sentences
            .iter()
            .flat_map(|sentence| sentence.claim_keys.iter())
            .any(|used| used == claim)
    }) && !role_allows_supporting_detail(candidate.role);
    if !duplicate_non_detail && !sentences.iter().any(|sentence| sentence.text == candidate.text) {
        sentences.push(candidate);
    }
}

fn target_sentence_count(shape: RhetoricalShape) -> usize {
    match shape {
        RhetoricalShape::Observation => 2,
        RhetoricalShape::ObservationEvidence => 3,
        RhetoricalShape::ObservationEvidenceWatch | RhetoricalShape::AlertEvidenceAction => 4,
    }
}

fn report_confidence(sentences: &[RenderedSentence]) -> ClaimConfidence {
    if sentences
        .iter()
        .any(|sentence| sentence.confidence == ClaimConfidence::Tentative)
    {
        ClaimConfidence::Tentative
    } else if sentences
        .iter()
        .any(|sentence| sentence.confidence == ClaimConfidence::Derived)
    {
        ClaimConfidence::Derived
    } else {
        ClaimConfidence::Observed
    }
}

fn fill_template(template: &str, replacements: &[(&str, &str)]) -> String {
    replacements.iter().fold(template.to_string(), |text, (key, value)| {
        text.replace(&format!("{{{key}}}"), value)
    })
}

fn contains_numeric_anchor(text: &str) -> bool {
    text.chars().any(|character| character.is_ascii_digit())
}

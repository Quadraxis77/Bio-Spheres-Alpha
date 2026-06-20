use std::collections::HashSet;

use crate::field_report::context::FieldReportContext;
use crate::field_report::facts::{ClaimConfidence, FieldReportTag, ReportFact};
use crate::field_report::grammar::{role_allows_supporting_detail, ClaimKey, SentenceRole};
use crate::field_report::plan::{
    FieldReportSeverity, ReportPlan, ReportTheme, RhetoricalShape,
};
use crate::field_report::style::{
    format_cell_count, format_delta_cells, format_percent, render_title_with_variation,
    resolve_tone, NumberDensity, ToneFamily, ToneId, ToneLexicon, ToneProfile,
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
    render_report_with_variation(context, plan, requested_tone, 0)
}

pub fn render_report_with_variation(
    context: &FieldReportContext,
    plan: &ReportPlan,
    requested_tone: &ToneProfile,
    variation: usize,
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
    let mut sentences = Vec::new();

    if let Some(evidence) = render_evidence(plan, &tone, lineage_name, variation) {
        push_coherent(&mut sentences, evidence);
    }

    if matches!(plan.theme, ReportTheme::StarvingExpansion) {
        if let Some(contrast) =
            render_starving_contrast(plan, &tone, lineage_name, variation)
        {
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
        title: render_title_with_variation(plan, &tone, variation),
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

fn render_evidence(
    plan: &ReportPlan,
    tone: &ToneProfile,
    lineage: &str,
    variation: usize,
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
            let text = if variation % 3 == 0 {
                format!(
                    "The population changed by {} cells to {}, while starvation risk reached {}.",
                    format_delta_cells(delta, tone, NumberDensity::Precise),
                    format_cell_count(current, tone, NumberDensity::Precise),
                    format_percent(starvation, tone, NumberDensity::Precise),
                )
            } else if variation % 3 == 1 {
                format!(
                    "Only {} of sampled cells remain nutrient-secure, despite a population of {}.",
                    format_percent(positive, tone, NumberDensity::Precise),
                    format_cell_count(current, tone, NumberDensity::Precise),
                )
            } else {
                let readiness = facts.clone().find_map(|fact| match fact {
                    ReportFact::ReproductiveReadiness { division_ready, .. } => {
                        Some(*division_ready)
                    }
                    _ => None,
                });
                if let Some(readiness) = readiness {
                    format!(
                        "Population rose by {} cells, but division readiness is {} and starvation risk is {}.",
                        format_delta_cells(delta, tone, NumberDensity::Precise),
                        format_percent(readiness, tone, NumberDensity::Precise),
                        format_percent(starvation, tone, NumberDensity::Precise),
                    )
                } else {
                    format!(
                        "Population rose by {} cells while the nutrient-secure share fell to {}.",
                        format_delta_cells(delta, tone, NumberDensity::Precise),
                        format_percent(positive, tone, NumberDensity::Precise),
                    )
                }
            };
            (
                text,
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
        ReportTheme::SingleLineageProfile => {
            let (
                current,
                delta,
                avg_nutrient,
                nutrient_positive,
                starvation,
                division_ready,
                average_age,
                dominant_cell_type,
                dominant_fraction,
                active_modes,
                radius,
                growth_windows,
                decline_windows,
            ) = facts.find_map(|fact| match fact {
                ReportFact::SingleLineageProfile {
                    current_cells,
                    delta_cells,
                    avg_nutrient,
                    nutrient_positive,
                    starvation_risk,
                    division_ready,
                    average_age,
                    dominant_cell_type,
                    dominant_fraction,
                    active_modes,
                    territory_radius,
                    growth_windows,
                    decline_windows,
                    ..
                } => Some((
                    *current_cells,
                    *delta_cells,
                    *avg_nutrient,
                    *nutrient_positive,
                    *starvation_risk,
                    *division_ready,
                    *average_age,
                    *dominant_cell_type,
                    *dominant_fraction,
                    *active_modes,
                    *territory_radius,
                    *growth_windows,
                    *decline_windows,
                )),
                _ => None,
            })?;
            let type_name = crate::cell::types::CellType::names()
                .get(dominant_cell_type as usize)
                .copied()
                .unwrap_or("Unknown");
            let text = match variation % 5 {
                0 => format!(
                    "{type_name} cells make up {} of the {}-cell population across {} active modes.",
                    format_percent(dominant_fraction, tone, NumberDensity::Precise),
                    format_cell_count(current, tone, NumberDensity::Precise),
                    active_modes,
                ),
                1 => format!(
                    "{} of cells are nutrient-secure; average nutrient is {:.1}, with {} at starvation risk.",
                    format_percent(nutrient_positive, tone, NumberDensity::Precise),
                    avg_nutrient,
                    format_percent(starvation, tone, NumberDensity::Precise),
                ),
                2 => format!(
                    "{} of cells are division-ready, and their average age is {:.1} seconds.",
                    format_percent(division_ready, tone, NumberDensity::Precise),
                    average_age,
                ),
                3 => format!(
                    "The {}-cell population occupies a sampled radius of {:.1} units.",
                    format_cell_count(current, tone, NumberDensity::Precise),
                    radius,
                ),
                _ => {
                    let trend = if growth_windows >= 2 {
                        format!("growth has continued for {growth_windows} scan windows")
                    } else if decline_windows >= 2 {
                        format!("decline has continued for {decline_windows} scan windows")
                    } else if delta == 0 {
                        "the latest scan shows no population change".to_string()
                    } else {
                        format!(
                            "the latest population change was {} cells",
                            format_delta_cells(delta, tone, NumberDensity::Precise)
                        )
                    };
                    format!(
                        "The lineage contains {} cells, and {trend}.",
                        format_cell_count(current, tone, NumberDensity::Precise),
                    )
                }
            };
            (
                text,
                vec![ClaimKey::LineageProfile(plan.subject_lineage?)],
                ClaimConfidence::Observed,
                "single_lineage_profile_evidence",
            )
        }
        ReportTheme::SceneComposition => {
            let (total, active_types, counts, deltas) = facts.find_map(|fact| match fact {
                ReportFact::SceneComposition {
                    total_cells,
                    active_types,
                    counts,
                    deltas,
                } => Some((*total_cells, *active_types, *counts, *deltas)),
                _ => None,
            })?;
            let text = render_scene_composition_evidence(
                total,
                active_types,
                &counts,
                &deltas,
                tone,
                variation,
            );
            (
                text,
                vec![ClaimKey::SceneComposition],
                ClaimConfidence::Observed,
                "scene_composition_evidence",
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
            let balance = std::iter::once(&plan.lead_fact)
                .chain(plan.supporting_facts.iter())
                .find_map(|fact| match fact {
                    ReportFact::EcosystemBalance {
                        diversity_score,
                        evenness_score,
                        growing_lineages,
                        declining_lineages,
                    } => Some((
                        *diversity_score,
                        *evenness_score,
                        *growing_lineages,
                        *declining_lineages,
                    )),
                    _ => None,
                });
            let text = match (variation % 3, balance) {
                (1, Some((diversity, evenness, _, _))) => format!(
                    "Evenness is {} and diversity is {}, while the largest lineage holds {} of active cells.",
                    format_percent(evenness, tone, NumberDensity::Precise),
                    format_percent(diversity, tone, NumberDensity::Precise),
                    format_percent(fraction, tone, NumberDensity::Precise),
                ),
                (2, Some((_, _, growing, declining))) => format!(
                    "Across {} living lineages, {} are growing and {} are declining beneath a {} majority.",
                    active,
                    growing,
                    declining,
                    format_percent(fraction, tone, NumberDensity::Precise),
                ),
                _ => format!(
                    "The largest lineage holds {} of active cells across {} living lineages.",
                    format_percent(fraction, tone, NumberDensity::Precise),
                    active
                ),
            };
            (
                text,
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

fn render_scene_composition_evidence(
    total: u32,
    active_types: u32,
    counts: &[u32; crate::cell::types::CellType::MAX_TYPES],
    deltas: &[i32; crate::cell::types::CellType::MAX_TYPES],
    tone: &ToneProfile,
    variation: usize,
) -> String {
    use crate::cell::types::CellType;

    let fraction = |count: u32| count as f32 / total.max(1) as f32;
    let category = |types: &[CellType]| {
        types
            .iter()
            .map(|cell_type| counts[*cell_type as usize])
            .sum::<u32>()
    };
    let category_delta = |types: &[CellType]| {
        types
            .iter()
            .map(|cell_type| deltas[*cell_type as usize] as i64)
            .sum::<i64>()
    };
    let implication = |share: f32, high: &'static str, low: &'static str| {
        if share >= 0.35 {
            high
        } else if share <= 0.1 {
            low
        } else {
            "The allocation is mixed rather than strongly specialized."
        }
    };

    match variation % 4 {
        0 => {
            let mut ranked: Vec<_> = counts
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, count)| *count > 0)
                .collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let names = crate::cell::types::CellType::names();
            let summary = ranked
                .iter()
                .take(3)
                .map(|(index, count)| {
                    format!(
                        "{} {}",
                        names.get(*index).copied().unwrap_or("Unknown"),
                        format_percent(fraction(*count), tone, NumberDensity::Precise)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let strongest_change = deltas
                .iter()
                .copied()
                .enumerate()
                .max_by_key(|(index, delta)| (delta.unsigned_abs(), std::cmp::Reverse(*index)))
                .filter(|(_, delta)| *delta != 0)
                .map(|(index, delta)| {
                    format!(
                        " The largest compositional movement is {} at {} cells.",
                        names.get(index).copied().unwrap_or("Unknown"),
                        format_delta_cells(delta, tone, NumberDensity::Precise)
                    )
                })
                .unwrap_or_default();
            format!(
                "{} cells span {} active types; the largest shares are {}.{}",
                format_cell_count(total, tone, NumberDensity::Precise),
                active_types,
                summary,
                strongest_change,
            )
        }
        1 => {
            let acquisition = category(&[
                CellType::Test,
                CellType::Phagocyte,
                CellType::Photocyte,
                CellType::Devorocyte,
            ]);
            let storage = category(&[CellType::Lipocyte]);
            let share = fraction(acquisition.saturating_add(storage));
            format!(
                "Resource acquisition and storage account for {} of cells (change: {:+}); {}",
                format_percent(share, tone, NumberDensity::Precise),
                category_delta(&[
                    CellType::Test,
                    CellType::Phagocyte,
                    CellType::Photocyte,
                    CellType::Devorocyte,
                    CellType::Lipocyte,
                ]),
                implication(
                    share,
                    "The scene is heavily invested in acquiring or buffering nutrients.",
                    "Resource work is thinly represented, which may limit resilience if conditions worsen.",
                )
            )
        }
        2 => {
            let movement = category(&[
                CellType::Flagellocyte,
                CellType::Buoyocyte,
                CellType::Ciliocyte,
                CellType::Myocyte,
                CellType::Plumocyte,
            ]);
            let sensing = category(&[
                CellType::Oculocyte,
                CellType::Cognocyte,
                CellType::Memorocyte,
                CellType::Luminocyte,
            ]);
            let share = fraction(movement.saturating_add(sensing));
            format!(
                "Movement cells make up {} and sensing or control cells {} of the scene; {}",
                format_percent(fraction(movement), tone, NumberDensity::Precise),
                format_percent(fraction(sensing), tone, NumberDensity::Precise),
                implication(
                    share,
                    "A large share of the population is equipped to move, detect, or coordinate responses.",
                    "The scene appears weighted toward passive or locally constrained strategies.",
                )
            )
        }
        _ => {
            let structure = category(&[
                CellType::Glueocyte,
                CellType::Vasculocyte,
                CellType::Siphonocyte,
            ]);
            let reproduction = category(&[
                CellType::Embryocyte,
                CellType::Gametocyte,
                CellType::Stemocyte,
            ]);
            format!(
                "Structural support accounts for {} and reproductive or developmental cells {} of the population. {}",
                format_percent(fraction(structure), tone, NumberDensity::Precise),
                format_percent(fraction(reproduction), tone, NumberDensity::Precise),
                if reproduction > structure {
                    "Current allocation leans more toward renewal and developmental change than physical support."
                } else if structure > reproduction.saturating_mul(2) {
                    "Current allocation favors cohesion and transport over producing the next generation."
                } else {
                    "Investment in support and renewal is comparatively balanced."
                }
            )
        }
    }
}

fn render_starving_contrast(
    plan: &ReportPlan,
    tone: &ToneProfile,
    lineage: &str,
    variation: usize,
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
    let text = if variation % 2 == 0 {
        match tone.family() {
        ToneFamily::Formal => format!(
            "The expansion is not yet evidence of durable growth: starvation risk is {}.",
            format_percent(resource.0, tone, NumberDensity::Rounded)
        ),
        ToneFamily::Naturalist => format!(
            "The new growth is not carrying comfortably; only {} of cells have reserves above the starvation threshold.",
            format_percent(resource.1, tone, NumberDensity::Rounded)
        ),
        ToneFamily::Living => format!(
            "The bloom is widening, but only {} of its cells have reserves above the starvation threshold.",
            format_percent(resource.1, tone, NumberDensity::Precise)
        ),
        ToneFamily::Alert => format!("{lineage} growth remains resource-limited."),
        ToneFamily::Any => unreachable!(),
        }
    } else {
        match tone.family() {
            ToneFamily::Formal => {
                "Current abundance therefore overstates the lineage's resource security."
                    .to_string()
            }
            ToneFamily::Naturalist => {
                "The population is larger, but its reserves of easy growth appear narrower."
                    .to_string()
            }
            ToneFamily::Living => {
                "There is more of the bloom now, but less comfort inside it.".to_string()
            }
            ToneFamily::Alert => "Population size exceeds current resource health.".to_string(),
            ToneFamily::Any => unreachable!(),
        }
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

fn contains_numeric_anchor(text: &str) -> bool {
    text.chars().any(|character| character.is_ascii_digit())
}

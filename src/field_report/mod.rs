//! Tone-independent field-report reasoning.
//!
//! This module stops before prose or UI. It converts compact simulation
//! telemetry into objective facts and semantic report plans. A separate style
//! model describes possible voices without changing those facts.

pub mod context;
pub mod facts;
pub mod fragments;
pub mod grammar;
pub mod history;
pub mod plan;
pub mod render;
pub mod scope;
pub mod style;
pub mod director;

pub use context::{FieldReportContext, LineageReportSnapshot, SceneCompositionSnapshot};
pub use facts::{
    extract_facts, ClaimConfidence, FieldReportTag, ReportConfidence, ReportFact, ReportFactKind,
};
pub use grammar::{ClaimKey, SentenceRole};
pub use history::{
    ArchivedFieldReport, ContinuityFact, FieldReportHistory, FieldReportId,
};
pub use plan::{build_report_plans, FieldReportSeverity, ReportPlan, ReportTheme, RhetoricalShape};
pub use director::{
    debug_run_report_sequence_all_tones, rendered_from, DebugReportSequence, FieldReportDirector,
    ReportCooldownRules, ReportDecision, SuppressionReason,
};
pub use render::{
    render_report, render_report_all_tones, render_report_with_variation, validate_report,
    CoherenceError, RenderedFieldReport, RenderedSentence,
};
pub use scope::{
    format_simulation_time, render_lineage_snapshot_report, render_specimen_report, CellId,
    LineageSnapshotReportSnapshot, OrganismId, ReportScope, SpecimenReportSnapshot,
};
pub use style::{
    resolve_tone, NumberDensity, NumberPolicy, SentenceLengthStyle, ToneFamily, ToneId,
    ToneProfile, UncertaintyStyle,
};

use crate::scene::lineage::EcosystemLineageArchive;

/// Complete semantic analysis result before style, fragments, or prose.
#[derive(Debug, Clone, Default)]
pub struct FieldReportAnalysis {
    pub context: FieldReportContext,
    pub facts: Vec<ReportFact>,
    pub plans: Vec<ReportPlan>,
}

/// Build report-ready meaning from the latest retained telemetry.
///
/// This is intentionally not called by rendering code and performs no GPU
/// reads. Callers decide when analysis is valuable enough to run.
pub fn analyze_archive(archive: &EcosystemLineageArchive) -> FieldReportAnalysis {
    let context = FieldReportContext::from_archive(archive);
    let facts = extract_facts(&context);
    let plans = build_report_plans(&facts);
    FieldReportAnalysis {
        context,
        facts,
        plans,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::lineage::{EcosystemTelemetrySummary, LineageTelemetrySample};

    fn strained_growth_context() -> FieldReportContext {
        FieldReportContext {
            frame: 120,
            time_seconds: 60.0,
            lineages: vec![LineageReportSnapshot {
                lineage_id: 8,
                display_name: "Veyra-9".to_string(),
                current: LineageTelemetrySample {
                    frame: 120,
                    time_seconds: 60.0,
                    cells: 940,
                    cell_delta: 184,
                    avg_nutrient: 22.0,
                    nutrient_positive_fraction: 0.31,
                    division_ready_fraction: 0.12,
                    starvation_risk_fraction: 0.42,
                    dominant_cell_type: 3,
                    dominant_cell_type_fraction: 0.55,
                    active_mode_count: 4,
                    bounding_radius: 18.0,
                    ..LineageTelemetrySample::default()
                },
                consecutive_growth_windows: 3,
                peak_cells: 940,
                ..LineageReportSnapshot::default()
            }],
            ecosystem: EcosystemTelemetrySummary {
                active_lineages: 3,
                total_cells: 1_500,
                largest_lineage_fraction: 0.63,
                diversity_score: 0.8,
                evenness_score: 0.72,
                growing_lineages: 2,
                declining_lineages: 1,
                ..EcosystemTelemetrySummary::default()
            },
            composition: SceneCompositionSnapshot::default(),
        }
    }

    fn stable_single_lineage_context(frame: i32) -> FieldReportContext {
        let mut cell_type_counts = [0; crate::cell::types::CellType::MAX_TYPES];
        cell_type_counts[crate::cell::types::CellType::Photocyte as usize] = 210;
        cell_type_counts[crate::cell::types::CellType::Flagellocyte as usize] = 90;
        cell_type_counts[crate::cell::types::CellType::Lipocyte as usize] = 60;
        cell_type_counts[crate::cell::types::CellType::Stemocyte as usize] = 40;
        cell_type_counts[crate::cell::types::CellType::Glueocyte as usize] = 20;
        FieldReportContext {
            frame,
            time_seconds: frame as f32 / 60.0,
            lineages: vec![LineageReportSnapshot {
                lineage_id: 4,
                display_name: "Solum-4".to_string(),
                current: LineageTelemetrySample {
                    frame,
                    time_seconds: frame as f32 / 60.0,
                    cells: 420,
                    cell_delta: 0,
                    avg_nutrient: 61.0,
                    nutrient_positive_fraction: 0.74,
                    division_ready_fraction: 0.11,
                    starvation_risk_fraction: 0.06,
                    average_age: 18.5,
                    dominant_cell_type: 3,
                    dominant_cell_type_fraction: 0.58,
                    cell_type_counts,
                    active_mode_count: 5,
                    bounding_radius: 24.0,
                    ..LineageTelemetrySample::default()
                },
                peak_cells: 440,
                ..LineageReportSnapshot::default()
            }],
            ecosystem: EcosystemTelemetrySummary {
                active_lineages: 1,
                total_cells: 420,
                largest_lineage_fraction: 1.0,
                diversity_score: 0.0,
                evenness_score: 1.0,
                ..EcosystemTelemetrySummary::default()
            },
            composition: SceneCompositionSnapshot {
                current_counts: cell_type_counts,
                previous_counts: {
                    let mut previous = cell_type_counts;
                    previous[crate::cell::types::CellType::Photocyte as usize] -= 15;
                    previous[crate::cell::types::CellType::Stemocyte as usize] += 5;
                    previous
                },
            },
        }
    }

    #[test]
    fn facts_and_plans_are_independent_of_tone() {
        let facts = extract_facts(&strained_growth_context());
        let plans = build_report_plans(&facts);
        let lineage_plan = plans
            .iter()
            .find(|plan| plan.subject_lineage == Some(8))
            .unwrap();

        assert_eq!(lineage_plan.theme, ReportTheme::StarvingExpansion);
        assert_eq!(lineage_plan.severity, FieldReportSeverity::Warning);
        assert!(lineage_plan.require_numeric_anchor);

        let formal = resolve_tone(ToneProfile::formal_scientific(), lineage_plan.severity);
        let organic = resolve_tone(ToneProfile::living_ecosystem(), lineage_plan.severity);
        assert_ne!(formal.id, organic.id);
        assert_eq!(lineage_plan.theme, ReportTheme::StarvingExpansion);
        assert!(matches!(
            lineage_plan.lead_fact,
            ReportFact::SustainedGrowth { lineage_id: 8, .. }
        ));
    }

    #[test]
    fn critical_tone_override_reduces_poetry_without_rewriting_voice() {
        let base = ToneProfile::living_ecosystem();
        let resolved = resolve_tone(base, FieldReportSeverity::Critical);

        assert_eq!(resolved.id, ToneId::LivingEcosystem);
        assert!(resolved.urgency >= 0.8);
        assert!(resolved.scientific_precision >= 0.55);
        assert!(resolved.poetic_density < base.poetic_density);
        assert!(resolved.metaphor_budget <= 1);
    }

    #[test]
    fn extractor_keeps_observations_objective() {
        let facts = extract_facts(&strained_growth_context());

        assert!(facts.iter().any(|fact| matches!(
            fact,
            ReportFact::ResourcePressure {
                lineage_id: 8,
                starvation_risk,
                ..
            } if (*starvation_risk - 0.42).abs() < 0.001
        )));
        assert!(facts
            .iter()
            .all(|fact| !matches!(fact.confidence(), ClaimConfidence::Tentative)));
    }

    #[test]
    fn analysis_entry_point_requires_no_tone_or_renderer() {
        let mut archive = crate::scene::lineage::EcosystemLineageArchive::default();
        archive.nodes.push(crate::scene::lineage::LineageNode {
            id: 8,
            display_name: "Veyra-9".to_string(),
            current_cells: 10,
            telemetry_history: vec![LineageTelemetrySample {
                frame: 30,
                time_seconds: 30.0,
                cells: 10,
                cell_delta: 10,
                dominant_cell_type_fraction: 1.0,
                ..LineageTelemetrySample::default()
            }],
            ..crate::scene::lineage::LineageNode::default()
        });
        archive.last_scan_frame = Some(30);
        archive.last_scan_time = 30.0;

        let analysis = analyze_archive(&archive);
        assert_eq!(analysis.context.lineages.len(), 1);
        assert!(!analysis.facts.is_empty());
        assert!(!analysis.plans.is_empty());
    }

    #[test]
    fn one_plan_renders_in_all_four_tones_without_changing_meaning() {
        let context = strained_growth_context();
        let facts = extract_facts(&context);
        let plans = build_report_plans(&facts);
        let plan = plans
            .iter()
            .find(|plan| plan.subject_lineage == Some(8))
            .unwrap();

        let reports = render_report_all_tones(&context, plan);
        assert_eq!(reports.len(), 4);
        for (_, report) in &reports {
            assert_eq!(report.theme, ReportTheme::StarvingExpansion);
            assert_eq!(report.severity, FieldReportSeverity::Warning);
            assert!(!report.title.is_empty());
            assert!(!report.body.is_empty());
            assert!((2..=4).contains(&report.sentences.len()));
            assert!(report
                .sentences
                .iter()
                .any(|sentence| sentence.numeric_anchor));
            assert!(report.tags.contains(&FieldReportTag::PopulationGrowth));
            assert!(report.tags.contains(&FieldReportTag::StarvationRisk));
            assert!(!report.body.to_ascii_lowercase().contains(" because "));
        }

        let bodies: std::collections::HashSet<_> =
            reports.iter().map(|(_, report)| &report.body).collect();
        assert_eq!(bodies.len(), 4);
    }

    #[test]
    fn rendered_reports_pass_coherence_validation() {
        let context = strained_growth_context();
        let facts = extract_facts(&context);
        let plan = build_report_plans(&facts)
            .into_iter()
            .find(|plan| plan.subject_lineage == Some(8))
            .unwrap();

        for tone in [
            ToneProfile::formal_scientific(),
            ToneProfile::naturalist_field_journal(),
            ToneProfile::living_ecosystem(),
            ToneProfile::alert_monitor(),
        ] {
            let resolved = resolve_tone(tone, plan.severity);
            let report = render_report(&context, &plan, &tone).unwrap();
            assert_eq!(validate_report(&report, &resolved), Vec::new());
            let lower = report.body.to_ascii_lowercase();
            for forbidden in ["spirit", "destiny", "life force", "nature whispers"] {
                assert!(!lower.contains(forbidden));
            }
        }
    }

    #[test]
    fn near_extinction_is_not_rendered_as_extinction() {
        let context = FieldReportContext {
            frame: 90,
            time_seconds: 45.0,
            lineages: vec![LineageReportSnapshot {
                lineage_id: 2,
                display_name: "Rill-2".to_string(),
                current: LineageTelemetrySample {
                    frame: 90,
                    time_seconds: 45.0,
                    cells: 3,
                    cell_delta: -7,
                    ..LineageTelemetrySample::default()
                },
                previous: Some(LineageTelemetrySample {
                    frame: 60,
                    time_seconds: 30.0,
                    cells: 10,
                    ..LineageTelemetrySample::default()
                }),
                consecutive_decline_windows: 2,
                peak_cells: 20,
                ..LineageReportSnapshot::default()
            }],
            ..FieldReportContext::default()
        };
        let facts = extract_facts(&context);
        let plan = build_report_plans(&facts).remove(0);
        assert_eq!(plan.theme, ReportTheme::NearExtinction);

        for (_, report) in render_report_all_tones(&context, &plan) {
            let lower = report.body.to_ascii_lowercase();
            assert!(!lower.contains("is extinct"));
            assert!(!lower.contains("has vanished"));
            assert!(report
                .sentences
                .iter()
                .any(|sentence| sentence.numeric_anchor));
        }
    }

    #[test]
    fn recovery_and_peak_are_distinct_semantic_themes() {
        let recovery_context = FieldReportContext {
            lineages: vec![LineageReportSnapshot {
                lineage_id: 3,
                display_name: "Sera-3".to_string(),
                current: LineageTelemetrySample {
                    frame: 60,
                    time_seconds: 60.0,
                    cells: 12,
                    cell_delta: 8,
                    ..LineageTelemetrySample::default()
                },
                previous: Some(LineageTelemetrySample {
                    frame: 30,
                    time_seconds: 30.0,
                    cells: 4,
                    ..LineageTelemetrySample::default()
                }),
                consecutive_growth_windows: 1,
                peak_cells: 12,
                ..LineageReportSnapshot::default()
            }],
            ..FieldReportContext::default()
        };
        let recovery_plan = build_report_plans(&extract_facts(&recovery_context)).remove(0);
        assert_eq!(recovery_plan.theme, ReportTheme::Recovery);

        let report = render_report(
            &recovery_context,
            &recovery_plan,
            &ToneProfile::naturalist_field_journal(),
        )
        .unwrap();
        assert!(report.body.contains("from 4 to 12"));
    }

    fn population_context(
        frame: i32,
        lineage_id: u64,
        name: &str,
        previous_cells: u32,
        current_cells: u32,
        growth_windows: u32,
        decline_windows: u32,
        starvation_risk: f32,
    ) -> FieldReportContext {
        FieldReportContext {
            frame,
            time_seconds: frame as f32 / 10.0,
            lineages: vec![LineageReportSnapshot {
                lineage_id,
                display_name: name.to_string(),
                current: LineageTelemetrySample {
                    frame,
                    time_seconds: frame as f32 / 10.0,
                    cells: current_cells,
                    cell_delta: current_cells as i32 - previous_cells as i32,
                    avg_nutrient: if starvation_risk > 0.0 { 20.0 } else { 60.0 },
                    nutrient_positive_fraction: if starvation_risk > 0.0 {
                        0.3
                    } else {
                        0.7
                    },
                    starvation_risk_fraction: starvation_risk,
                    dominant_cell_type_fraction: 0.2,
                    ..LineageTelemetrySample::default()
                },
                previous: Some(LineageTelemetrySample {
                    frame: frame - 30,
                    time_seconds: (frame - 30) as f32 / 10.0,
                    cells: previous_cells,
                    ..LineageTelemetrySample::default()
                }),
                consecutive_growth_windows: growth_windows,
                consecutive_decline_windows: decline_windows,
                peak_cells: current_cells.saturating_add(100),
                ..LineageReportSnapshot::default()
            }],
            ..FieldReportContext::default()
        }
    }

    fn analysis_from_context(context: FieldReportContext) -> FieldReportAnalysis {
        let facts = extract_facts(&context);
        let plans = build_report_plans(&facts);
        FieldReportAnalysis {
            context,
            facts,
            plans,
        }
    }

    #[test]
    fn report_director_emits_once_then_suppresses_repetition() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 100,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let first = analysis_from_context(population_context(
            1_000, 1, "Veyra", 100, 130, 1, 0, 0.0,
        ));
        let repeated = analysis_from_context(population_context(
            1_150, 1, "Veyra", 130, 155, 1, 0, 0.0,
        ));

        assert!(matches!(
            director.decide_analysis(&first, ToneId::FormalScientific),
            ReportDecision::Emit(_)
        ));
        assert!(matches!(
            director.decide_analysis(&repeated, ToneId::FormalScientific),
            ReportDecision::Suppress(SuppressionReason::LineageRecentlyReported {
                lineage_id: 1
            })
        ));
    }

    #[test]
    fn critical_near_extinction_bypasses_active_cooldowns() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 10_000,
                allow_critical_bypass: true,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let growth = analysis_from_context(population_context(
            1_000, 1, "Veyra", 100, 130, 1, 0, 0.0,
        ));
        let critical =
            analysis_from_context(population_context(1_001, 1, "Veyra", 130, 3, 0, 2, 0.5));

        assert!(matches!(
            director.decide_analysis(&growth, ToneId::FormalScientific),
            ReportDecision::Emit(_)
        ));
        assert!(matches!(
            director.decide_analysis(&critical, ToneId::FormalScientific),
            ReportDecision::Emit(_)
        ));
    }

    #[test]
    fn different_lineages_can_emit_in_sequence_when_theme_is_not_repeated() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 0,
                lineage_cooldown_reports: 2,
                theme_cooldown_reports: 1,
                claim_cooldown_reports: 1,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let boom = analysis_from_context(population_context(
            1_000, 1, "Veyra", 100, 130, 1, 0, 0.0,
        ));
        let decline =
            analysis_from_context(population_context(1_001, 2, "Rill", 100, 70, 0, 2, 0.0));

        assert!(matches!(
            director.decide_analysis(&boom, ToneId::NaturalistFieldJournal),
            ReportDecision::Emit(_)
        ));
        assert!(matches!(
            director.decide_analysis(&decline, ToneId::NaturalistFieldJournal),
            ReportDecision::Emit(_)
        ));
    }

    #[test]
    fn repeated_theme_returns_specific_suppression_reason() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 0,
                lineage_cooldown_reports: 0,
                theme_cooldown_reports: 2,
                claim_cooldown_reports: 0,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let first = analysis_from_context(population_context(
            1_000, 1, "Veyra", 100, 130, 1, 0, 0.0,
        ));
        let second = analysis_from_context(population_context(
            1_001, 2, "Sera", 200, 250, 1, 0, 0.0,
        ));
        director.decide_analysis(&first, ToneId::FormalScientific);

        assert!(matches!(
            director.decide_analysis(&second, ToneId::FormalScientific),
            ReportDecision::Suppress(SuppressionReason::ThemeRecentlyReported {
                theme: ReportTheme::PopulationBoom
            })
        ));
    }

    #[test]
    fn repeated_ecosystem_claim_returns_claim_suppression_reason() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 0,
                lineage_cooldown_reports: 0,
                theme_cooldown_reports: 0,
                claim_cooldown_reports: 2,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let ecosystem_analysis = |frame| {
            analysis_from_context(FieldReportContext {
                frame,
                ecosystem: EcosystemTelemetrySummary {
                    active_lineages: 3,
                    total_cells: 100,
                    largest_lineage_fraction: 0.7,
                    diversity_score: 0.7,
                    evenness_score: 0.6,
                    ..EcosystemTelemetrySummary::default()
                },
                ..FieldReportContext::default()
            })
        };

        assert!(matches!(
            director.decide_analysis(&ecosystem_analysis(1_000), ToneId::AlertMonitor),
            ReportDecision::Emit(_)
        ));
        assert!(matches!(
            director.decide_analysis(&ecosystem_analysis(1_001), ToneId::AlertMonitor),
            ReportDecision::Suppress(SuppressionReason::ClaimRecentlyReported {
                claim: ClaimKey::EcosystemDominance
            })
        ));
    }

    #[test]
    fn recovery_bypasses_cooldown_and_records_reversal_continuity() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 10_000,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let near_extinction =
            analysis_from_context(population_context(1_000, 4, "Rill", 20, 3, 0, 2, 0.4));
        let recovery =
            analysis_from_context(population_context(1_001, 4, "Rill", 3, 12, 1, 0, 0.0));

        assert!(matches!(
            director.decide_analysis(&near_extinction, ToneId::LivingEcosystem),
            ReportDecision::Emit(_)
        ));
        let ReportDecision::Emit(report) =
            director.decide_analysis(&recovery, ToneId::LivingEcosystem)
        else {
            panic!("recovery should bypass cooldown");
        };
        assert_eq!(report.rendered.theme, ReportTheme::Recovery);
        assert!(report.continuity.iter().any(|fact| matches!(
            fact,
            ContinuityFact::ConditionReversed {
                lineage_id: 4,
                from: FieldReportTag::NearExtinction,
                to: FieldReportTag::Recovery,
            }
        )));
    }

    #[test]
    fn report_history_is_bounded_and_ids_are_monotonic() {
        let mut director = FieldReportDirector {
            history: FieldReportHistory::new(2),
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 0,
                lineage_cooldown_reports: 0,
                theme_cooldown_reports: 0,
                claim_cooldown_reports: 0,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let mut ids = Vec::new();
        for (index, lineage_id) in [1, 2, 3].into_iter().enumerate() {
            let analysis = analysis_from_context(population_context(
                1_000 + index as i32,
                lineage_id,
                "Line",
                100,
                130,
                1,
                0,
                0.0,
            ));
            let ReportDecision::Emit(report) =
                director.decide_analysis(&analysis, ToneId::AlertMonitor)
            else {
                panic!("report should emit");
            };
            ids.push(report.id);
        }
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(director.history.reports.len(), 2);
        assert_eq!(director.history.reports.front().unwrap().id, 2);
    }

    #[test]
    fn uninteresting_analysis_reports_below_threshold() {
        let context = FieldReportContext {
            frame: 100,
            lineages: vec![LineageReportSnapshot {
                lineage_id: 1,
                display_name: "Quiet".to_string(),
                current: LineageTelemetrySample {
                    cells: 10,
                    dominant_cell_type: 2,
                    dominant_cell_type_fraction: 0.8,
                    ..LineageTelemetrySample::default()
                },
                peak_cells: 10,
                ..LineageReportSnapshot::default()
            }],
            ..FieldReportContext::default()
        };
        let analysis = analysis_from_context(context);
        let mut director = FieldReportDirector::default();
        assert!(matches!(
            director.decide_analysis(&analysis, ToneId::FormalScientific),
            ReportDecision::Suppress(SuppressionReason::BelowInterestThreshold)
        ));
    }

    #[test]
    fn all_tone_sequence_helper_preserves_decision_count() {
        let analyses = vec![
            analysis_from_context(population_context(
                1_000, 1, "Veyra", 100, 130, 1, 0, 0.0,
            )),
            analysis_from_context(population_context(
                1_200, 1, "Veyra", 130, 160, 1, 0, 0.0,
            )),
        ];
        let sequences =
            debug_run_report_sequence_all_tones(&analyses, ReportCooldownRules::default());
        assert_eq!(sequences.len(), 4);
        assert!(sequences
            .iter()
            .all(|sequence| sequence.decisions.len() == analyses.len()));
    }

    #[test]
    fn specimen_scope_only_claims_known_organism_context() {
        let isolated = SpecimenReportSnapshot {
            cell_id: 42,
            cell_type_name: "Photocyte".to_string(),
            lineage_id: Some(3),
            lineage_name: Some("Veyra-9".to_string()),
            organism_id: None,
            alive: true,
            nutrient_level: 60.0,
            nutrient_gain_rate: 0.5,
            thermal_state: 4,
            adhesion_count: 0,
            active_signal_channels: 1,
        };
        let report =
            render_specimen_report(&isolated, &ToneProfile::naturalist_field_journal()).unwrap();
        assert!(report.body.contains("isolated"));
        assert!(!report.body.contains("Organism "));
        assert!(!report.body.contains("composition is"));

        let integrated = SpecimenReportSnapshot {
            organism_id: Some(82),
            adhesion_count: 5,
            ..isolated
        };
        let report =
            render_specimen_report(&integrated, &ToneProfile::naturalist_field_journal()).unwrap();
        assert!(report.body.contains("Organism 82"));
        assert!(report
            .body
            .contains("organism-level composition has not been inferred"));
    }

    #[test]
    fn lineage_snapshot_scope_is_archival_and_numeric() {
        let snapshot = LineageSnapshotReportSnapshot {
            lineage_id: 9,
            lineage_name: "Aurex-4".to_string(),
            snapshot_frame: 82_400,
            captured_time: 1_200.0,
            morphology_cells: 36,
            current_cells_at_snapshot: Some(2_840),
            peak_cells: 2_840,
            dominant_cell_type_name: Some("Photocyte".to_string()),
            active_modes: Some(6),
            territory_radius: 18.5,
        };
        let report = render_lineage_snapshot_report(
            &snapshot,
            &ToneProfile::naturalist_field_journal(),
        )
        .unwrap();
        assert!(report.title.contains("20:00"));
        assert!(report.body.contains("36 morphology cells"));
        assert!(report.body.contains("6 active modes"));
        assert_eq!(report.severity, FieldReportSeverity::Routine);
    }

    #[test]
    fn report_scope_variants_remain_distinct() {
        assert_ne!(
            ReportScope::Ecosystem,
            ReportScope::Specimen(1)
        );
        assert_ne!(
            ReportScope::Lineage(1),
            ReportScope::LineageSnapshot {
                lineage_id: 1,
                snapshot_frame: 10,
            }
        );
    }

    #[test]
    fn repeated_lineage_can_emit_again_after_frame_cooldown_expires() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 100,
                lineage_cooldown_reports: 2,
                theme_cooldown_reports: 2,
                claim_cooldown_reports: 2,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let first = analysis_from_context(population_context(
            1_000, 1, "Veyra", 100, 130, 1, 0, 0.0,
        ));
        let later = analysis_from_context(population_context(
            1_250, 1, "Veyra", 130, 170, 1, 0, 0.0,
        ));
        assert!(matches!(
            director.decide_analysis(&first, ToneId::FormalScientific),
            ReportDecision::Emit(_)
        ));
        assert!(matches!(
            director.decide_analysis(&later, ToneId::FormalScientific),
            ReportDecision::Emit(_)
        ));
    }

    #[test]
    fn recurring_condition_rotates_analysis_instead_of_repeating_report() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 100,
                lineage_cooldown_reports: 2,
                theme_cooldown_reports: 2,
                claim_cooldown_reports: 2,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let first = analysis_from_context(population_context(
            1_000, 1, "Veyra", 100, 140, 3, 0, 0.42,
        ));
        let later = analysis_from_context(population_context(
            1_250, 1, "Veyra", 140, 180, 4, 0, 0.45,
        ));

        let ReportDecision::Emit(first_report) =
            director.decide_analysis(&first, ToneId::NaturalistFieldJournal)
        else {
            panic!("first report should emit");
        };
        let ReportDecision::Emit(second_report) =
            director.decide_analysis(&later, ToneId::NaturalistFieldJournal)
        else {
            panic!("recurring condition should emit after cooldown");
        };

        assert_eq!(
            first_report.rendered.theme,
            second_report.rendered.theme
        );
        assert_ne!(first_report.rendered.body, second_report.rendered.body);
        assert!(first_report.rendered.body.contains("starvation risk"));
        assert!(second_report.rendered.body.contains("nutrient-secure"));
    }

    #[test]
    fn explicit_variations_preserve_facts_but_change_analytical_emphasis() {
        let context = strained_growth_context();
        let facts = extract_facts(&context);
        let plan = build_report_plans(&facts)
            .into_iter()
            .find(|plan| plan.subject_lineage == Some(8))
            .unwrap();
        let tone = ToneProfile::formal_scientific();
        let growth_focus = render_report_with_variation(&context, &plan, &tone, 0).unwrap();
        let resource_focus = render_report_with_variation(&context, &plan, &tone, 1).unwrap();

        assert_eq!(growth_focus.theme, resource_focus.theme);
        assert_eq!(growth_focus.tags, resource_focus.tags);
        assert_ne!(growth_focus.body, resource_focus.body);
        assert!(growth_focus.body.contains("starvation risk"));
        assert!(resource_focus.body.contains("nutrient-secure"));
    }

    #[test]
    fn recurring_dominance_reports_rotate_ecosystem_analysis() {
        let context = strained_growth_context();
        let facts = extract_facts(&context);
        let plan = build_report_plans(&facts)
            .into_iter()
            .find(|plan| plan.theme == ReportTheme::EcosystemDominance)
            .unwrap();
        let tone = ToneProfile::naturalist_field_journal();
        let concentration = render_report_with_variation(&context, &plan, &tone, 0).unwrap();
        let balance = render_report_with_variation(&context, &plan, &tone, 1).unwrap();
        let trajectory = render_report_with_variation(&context, &plan, &tone, 2).unwrap();

        assert_eq!(concentration.title, "One Lineage Takes Hold");
        assert_eq!(balance.title, "An Uneven Living Field");
        assert_eq!(trajectory.title, "Growth Beneath the Majority");
        assert!(concentration.body.contains("largest lineage"));
        assert!(balance.body.contains("Evenness"));
        assert!(trajectory.body.contains("growing"));
        assert_ne!(concentration.body, balance.body);
        assert_ne!(balance.body, trajectory.body);
    }

    #[test]
    fn a_single_lineage_is_profiled_instead_of_called_dominant() {
        let context = stable_single_lineage_context(1_000);
        let facts = extract_facts(&context);
        assert!(!facts
            .iter()
            .any(|fact| matches!(fact, ReportFact::EcosystemDominance { .. })));

        let plans = build_report_plans(&facts);
        let profile = plans
            .iter()
            .find(|plan| plan.theme == ReportTheme::SingleLineageProfile)
            .expect("stable single-lineage systems should produce a profile");
        assert_eq!(profile.subject_lineage, Some(4));
        assert_eq!(profile.severity, FieldReportSeverity::Notable);
    }

    #[test]
    fn single_lineage_reports_rotate_across_five_useful_perspectives() {
        let context = stable_single_lineage_context(1_000);
        let facts = extract_facts(&context);
        let plan = build_report_plans(&facts)
            .into_iter()
            .find(|plan| plan.theme == ReportTheme::SingleLineageProfile)
            .unwrap();
        let tone = ToneProfile::naturalist_field_journal();
        let reports: Vec<_> = (0..5)
            .map(|variation| {
                render_report_with_variation(&context, &plan, &tone, variation).unwrap()
            })
            .collect();

        assert_eq!(reports[0].title, "The Shape Within");
        assert_eq!(reports[1].title, "How the Lineage Is Living");
        assert_eq!(reports[2].title, "The Next Generation");
        assert_eq!(reports[3].title, "The Reach of the Lineage");
        assert_eq!(reports[4].title, "Holding Steady");
        assert!(reports[0].body.contains("active modes"));
        assert!(reports[1].body.contains("nutrient-secure"));
        assert!(reports[2].body.contains("division-ready"));
        assert!(reports[3].body.contains("radius"));
        assert!(reports[4].body.contains("no population change"));
    }

    #[test]
    fn scene_composition_reports_rotate_roles_and_implications() {
        let context = stable_single_lineage_context(1_000);
        let facts = extract_facts(&context);
        let plan = build_report_plans(&facts)
            .into_iter()
            .find(|plan| plan.theme == ReportTheme::SceneComposition)
            .expect("composition telemetry should produce a periodic report plan");
        let tone = ToneProfile::naturalist_field_journal();
        let reports: Vec<_> = (0..4)
            .map(|variation| {
                render_report_with_variation(&context, &plan, &tone, variation).unwrap()
            })
            .collect();

        assert_eq!(reports[0].title, "The Work of the Living Field");
        assert_eq!(reports[1].title, "How the Scene Feeds");
        assert_eq!(reports[2].title, "How the Scene Moves and Senses");
        assert_eq!(reports[3].title, "Holding Together, Making More");
        assert!(reports[0].body.contains("Photocyte"));
        assert!(reports[1].body.contains("acquiring or buffering nutrients"));
        assert!(reports[2].body.contains("Movement cells"));
        assert!(reports[3].body.contains("support"));
    }

    #[test]
    fn report_body_starts_with_evidence_instead_of_repeating_the_title() {
        let context = stable_single_lineage_context(1_000);
        let facts = extract_facts(&context);
        let plan = build_report_plans(&facts)
            .into_iter()
            .find(|plan| plan.theme == ReportTheme::SceneComposition)
            .unwrap();
        let report = render_report_with_variation(
            &context,
            &plan,
            &ToneProfile::naturalist_field_journal(),
            0,
        )
        .unwrap();

        assert_eq!(report.sentences[0].role, SentenceRole::Evidence);
        assert!(report.body.starts_with("420 cells span"));
        assert!(!report.body.starts_with(&report.title));
    }

    #[test]
    fn scene_composition_returns_periodically_without_crowding_every_report() {
        let mut director = FieldReportDirector {
            cooldowns: ReportCooldownRules {
                min_frames_between_reports: 100,
                lineage_cooldown_reports: 1,
                theme_cooldown_reports: 1,
                claim_cooldown_reports: 1,
                ..ReportCooldownRules::default()
            },
            ..FieldReportDirector::default()
        };
        let frames = [1_000, 1_300, 1_600, 1_900];
        let themes: Vec<_> = frames
            .into_iter()
            .filter_map(|frame| {
                let analysis = analysis_from_context(stable_single_lineage_context(frame));
                match director.decide_analysis(&analysis, ToneId::NaturalistFieldJournal) {
                    ReportDecision::Emit(report) => Some(report.rendered.theme),
                    ReportDecision::Suppress(_) => None,
                }
            })
            .collect();

        assert_eq!(themes.first(), Some(&ReportTheme::SceneComposition));
        assert!(themes.contains(&ReportTheme::SingleLineageProfile));
        assert!(themes
            .iter()
            .filter(|&&theme| theme == ReportTheme::SceneComposition)
            .count()
            >= 2);
    }

    #[test]
    fn simulation_time_formats_as_clock_time() {
        assert_eq!(format_simulation_time(5.0), "0:05");
        assert_eq!(format_simulation_time(125.0), "2:05");
        assert_eq!(format_simulation_time(3_725.0), "1:02:05");
    }

    #[test]
    fn extinct_remnant_is_not_treated_as_a_current_near_extinction() {
        let mut archive = EcosystemLineageArchive::default();
        archive.last_scan_frame = Some(2_000);
        archive.last_scan_time = 120.0;
        archive.nodes.push(crate::scene::lineage::LineageNode {
            id: 1,
            display_name: "Lost Remnant".to_string(),
            current_cells: 0,
            extinct_frame: Some(1_900),
            telemetry_history: vec![LineageTelemetrySample {
                frame: 1_800,
                time_seconds: 90.0,
                cells: 2,
                ..LineageTelemetrySample::default()
            }],
            ..crate::scene::lineage::LineageNode::default()
        });
        archive.nodes.push(crate::scene::lineage::LineageNode {
            id: 2,
            display_name: "Living Line".to_string(),
            current_cells: 120,
            peak_cells: 120,
            telemetry_history: vec![
                LineageTelemetrySample {
                    frame: 1_900,
                    time_seconds: 90.0,
                    cells: 100,
                    ..LineageTelemetrySample::default()
                },
                LineageTelemetrySample {
                    frame: 2_000,
                    time_seconds: 120.0,
                    cells: 120,
                    cell_delta: 20,
                    ..LineageTelemetrySample::default()
                },
            ],
            ..crate::scene::lineage::LineageNode::default()
        });

        let analysis = analyze_archive(&archive);
        assert!(analysis.context.lineage(1).is_none());
        assert!(analysis.context.lineage(2).is_some());
        assert!(!analysis.facts.iter().any(|fact| {
            fact.subject_lineage() == Some(1)
                && matches!(fact, ReportFact::NearExtinction { .. })
        }));
        assert!(analysis
            .plans
            .iter()
            .any(|plan| plan.subject_lineage == Some(2)));
    }
}

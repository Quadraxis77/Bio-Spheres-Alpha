use crate::field_report::history::{ArchivedFieldReport, FieldReportHistory, FieldReportId};
use crate::field_report::{
    analyze_archive, render_report_with_variation, ClaimKey, FieldReportAnalysis,
    FieldReportSeverity, RenderedFieldReport, ReportPlan, ReportTheme, ToneId,
};
use crate::scene::lineage::{EcosystemLineageArchive, LineageId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReportCooldownRules {
    pub min_frames_between_reports: u64,
    pub lineage_cooldown_reports: usize,
    pub theme_cooldown_reports: usize,
    pub claim_cooldown_reports: usize,
    pub allow_warning_bypass: bool,
    pub allow_critical_bypass: bool,
}

impl Default for ReportCooldownRules {
    fn default() -> Self {
        Self {
            min_frames_between_reports: 300,
            lineage_cooldown_reports: 2,
            theme_cooldown_reports: 2,
            claim_cooldown_reports: 2,
            allow_warning_bypass: false,
            allow_critical_bypass: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuppressionReason {
    NoPlanAvailable,
    BelowInterestThreshold,
    ReportCooldown { remaining_frames: u64 },
    ThemeRecentlyReported { theme: ReportTheme },
    LineageRecentlyReported { lineage_id: LineageId },
    ClaimRecentlyReported { claim: ClaimKey },
    RenderingFailed,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReportDecision {
    Emit(ArchivedFieldReport),
    Suppress(SuppressionReason),
}

#[derive(Debug, Clone)]
pub struct FieldReportDirector {
    pub history: FieldReportHistory,
    pub cooldowns: ReportCooldownRules,
    pub preferred_tone: ToneId,
    pub(crate) next_report_id: FieldReportId,
}

impl Default for FieldReportDirector {
    fn default() -> Self {
        Self {
            history: FieldReportHistory::default(),
            cooldowns: ReportCooldownRules::default(),
            preferred_tone: ToneId::NaturalistFieldJournal,
            next_report_id: 1,
        }
    }
}

impl FieldReportDirector {
    pub fn update(&mut self, archive: &EcosystemLineageArchive) -> ReportDecision {
        self.update_with_tone(archive, self.preferred_tone)
    }

    pub fn update_with_tone(
        &mut self,
        archive: &EcosystemLineageArchive,
        tone: ToneId,
    ) -> ReportDecision {
        let analysis = analyze_archive(archive);
        self.decide_analysis(&analysis, tone)
    }

    pub fn decide_analysis(
        &mut self,
        analysis: &FieldReportAnalysis,
        tone: ToneId,
    ) -> ReportDecision {
        if analysis.plans.is_empty() {
            return ReportDecision::Suppress(SuppressionReason::NoPlanAvailable);
        }
        let Some(plan) = self.select_report_plan_with_history(analysis) else {
            return ReportDecision::Suppress(SuppressionReason::BelowInterestThreshold);
        };
        if let Some(reason) = self.suppression_reason(plan, analysis.context.frame.max(0) as u64) {
            return ReportDecision::Suppress(reason);
        }

        let variation = self.history.prior_reports_for_plan(plan);
        let Some(rendered) =
            render_report_with_variation(&analysis.context, plan, &tone.profile(), variation)
        else {
            return ReportDecision::Suppress(SuppressionReason::RenderingFailed);
        };
        let continuity = self.history.continuity_for(plan);
        let archived = ArchivedFieldReport {
            id: self.allocate_id(),
            created_frame: analysis.context.frame.max(0) as u64,
            created_time: analysis.context.time_seconds.max(0.0) as f64,
            rendered,
            continuity,
        };
        self.history.record(archived.clone());
        ReportDecision::Emit(archived)
    }

    fn select_report_plan_with_history<'a>(
        &self,
        analysis: &'a FieldReportAnalysis,
    ) -> Option<&'a ReportPlan> {
        analysis
            .plans
            .iter()
            .filter(|plan| plan.severity >= FieldReportSeverity::Notable)
            .max_by(|a, b| {
                plan_score(
                    a,
                    &self.history,
                    analysis.context.frame.max(0) as u64,
                    &self.cooldowns,
                )
                .cmp(&plan_score(
                    b,
                    &self.history,
                    analysis.context.frame.max(0) as u64,
                    &self.cooldowns,
                ))
                .then_with(|| b.subject_lineage.cmp(&a.subject_lineage))
            })
    }

    fn suppression_reason(
        &self,
        plan: &ReportPlan,
        current_frame: u64,
    ) -> Option<SuppressionReason> {
        if bypasses_cooldowns(plan, &self.cooldowns) {
            return None;
        }
        if let Some(last_frame) = self.history.last_report_frame {
            let elapsed = current_frame.saturating_sub(last_frame);
            if elapsed < self.cooldowns.min_frames_between_reports {
                return Some(SuppressionReason::ReportCooldown {
                    remaining_frames: self.cooldowns.min_frames_between_reports - elapsed,
                });
            }
        }
        if let Some(lineage_id) = plan.subject_lineage {
            if self.history.recent_report_for_lineage(
                lineage_id,
                self.cooldowns.lineage_cooldown_reports,
                current_frame,
                cooldown_window_frames(
                    self.cooldowns.min_frames_between_reports,
                    self.cooldowns.lineage_cooldown_reports,
                ),
            ) {
                return Some(SuppressionReason::LineageRecentlyReported { lineage_id });
            }
        }
        if self.history.recent_theme(
            plan.theme,
            self.cooldowns.theme_cooldown_reports,
            current_frame,
            cooldown_window_frames(
                self.cooldowns.min_frames_between_reports,
                self.cooldowns.theme_cooldown_reports,
            ),
        ) {
            return Some(SuppressionReason::ThemeRecentlyReported { theme: plan.theme });
        }
        let claims = plan_claims(plan);
        if self.history.recent_claim(
            &claims,
            self.cooldowns.claim_cooldown_reports,
            current_frame,
            cooldown_window_frames(
                self.cooldowns.min_frames_between_reports,
                self.cooldowns.claim_cooldown_reports,
            ),
        ) {
            return claims
                .into_iter()
                .find(|claim| {
                    self.history.recent_claim(
                        &[*claim],
                        self.cooldowns.claim_cooldown_reports,
                        current_frame,
                        cooldown_window_frames(
                            self.cooldowns.min_frames_between_reports,
                            self.cooldowns.claim_cooldown_reports,
                        ),
                    )
                })
                .map(|claim| SuppressionReason::ClaimRecentlyReported { claim });
        }
        None
    }

    fn allocate_id(&mut self) -> FieldReportId {
        let id = self.next_report_id.max(1);
        self.next_report_id = id.saturating_add(1);
        id
    }
}

fn plan_score(
    plan: &ReportPlan,
    history: &FieldReportHistory,
    current_frame: u64,
    cooldowns: &ReportCooldownRules,
) -> i32 {
    let severity = match plan.severity {
        FieldReportSeverity::Routine => 0,
        FieldReportSeverity::Notable => 100,
        FieldReportSeverity::Warning => 200,
        FieldReportSeverity::Critical => 300,
    };
    let event_bonus = match plan.theme {
        ReportTheme::Recovery | ReportTheme::NewPopulationPeak => 50,
        ReportTheme::NearExtinction => 75,
        ReportTheme::SceneComposition => {
            if history
                .reports
                .iter()
                .rev()
                .take(2)
                .any(|report| report.rendered.theme == ReportTheme::SceneComposition)
            {
                -15
            } else {
                25
            }
        }
        _ => 0,
    };
    let novelty = if history.recent_theme(
        plan.theme,
        4,
        current_frame,
        cooldown_window_frames(cooldowns.min_frames_between_reports, 4),
    ) {
        -30
    } else {
        20
    };
    severity + event_bonus + novelty
}

fn cooldown_window_frames(min_frames: u64, report_window: usize) -> u64 {
    if report_window == 0 {
        return 0;
    }
    min_frames
        .max(1)
        .saturating_mul(report_window.min(u64::MAX as usize) as u64)
}

fn bypasses_cooldowns(plan: &ReportPlan, rules: &ReportCooldownRules) -> bool {
    if plan.severity == FieldReportSeverity::Critical && rules.allow_critical_bypass {
        return true;
    }
    if plan.severity == FieldReportSeverity::Warning && rules.allow_warning_bypass {
        return true;
    }
    matches!(
        plan.theme,
        ReportTheme::Recovery | ReportTheme::NewPopulationPeak
    )
}

fn plan_claims(plan: &ReportPlan) -> Vec<ClaimKey> {
    let mut claims = Vec::new();
    if let Some(id) = plan.subject_lineage {
        match plan.theme {
            ReportTheme::StarvingExpansion => {
                claims.push(ClaimKey::PopulationIncrease(id));
                claims.push(ClaimKey::NutrientPressure(id));
            }
            ReportTheme::PopulationBoom => claims.push(ClaimKey::PopulationIncrease(id)),
            ReportTheme::SustainedDecline => claims.push(ClaimKey::PopulationDecline(id)),
            ReportTheme::Recovery => claims.push(ClaimKey::Recovery(id)),
            ReportTheme::NewPopulationPeak => claims.push(ClaimKey::PopulationPeak(id)),
            ReportTheme::NearExtinction => claims.push(ClaimKey::NearExtinction(id)),
            ReportTheme::ReproductivePulse => claims.push(ClaimKey::DivisionReadiness(id)),
            ReportTheme::CompositionShift => claims.push(ClaimKey::DominantCellType(id)),
            ReportTheme::TerritoryObservation => claims.push(ClaimKey::Territory(id)),
            ReportTheme::SingleLineageProfile => claims.push(ClaimKey::LineageProfile(id)),
            ReportTheme::SceneComposition
            | ReportTheme::EcosystemDominance
            | ReportTheme::EcosystemBalance => {}
        }
    } else {
        match plan.theme {
            ReportTheme::SceneComposition => claims.push(ClaimKey::SceneComposition),
            ReportTheme::EcosystemDominance => claims.push(ClaimKey::EcosystemDominance),
            ReportTheme::EcosystemBalance => claims.push(ClaimKey::EcosystemDiversity),
            _ => {}
        }
    }
    claims
}

#[derive(Debug, Clone)]
pub struct DebugReportSequence {
    pub tone: ToneId,
    pub decisions: Vec<ReportDecision>,
}

pub fn debug_run_report_sequence_all_tones(
    analyses: &[FieldReportAnalysis],
    cooldowns: ReportCooldownRules,
) -> Vec<DebugReportSequence> {
    [
        ToneId::FormalScientific,
        ToneId::NaturalistFieldJournal,
        ToneId::LivingEcosystem,
        ToneId::AlertMonitor,
    ]
    .into_iter()
    .map(|tone| {
        let mut director = FieldReportDirector {
            cooldowns,
            preferred_tone: tone,
            ..FieldReportDirector::default()
        };
        let decisions = analyses
            .iter()
            .map(|analysis| director.decide_analysis(analysis, tone))
            .collect();
        DebugReportSequence { tone, decisions }
    })
    .collect()
}

pub fn rendered_from(decision: &ReportDecision) -> Option<&RenderedFieldReport> {
    match decision {
        ReportDecision::Emit(report) => Some(&report.rendered),
        ReportDecision::Suppress(_) => None,
    }
}

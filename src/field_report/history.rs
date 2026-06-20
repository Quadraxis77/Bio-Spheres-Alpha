use std::collections::VecDeque;

use crate::field_report::{
    ClaimKey, FieldReportSeverity, FieldReportTag, RenderedFieldReport, ReportPlan, ReportTheme,
};
use crate::scene::lineage::LineageId;

pub type FieldReportId = u64;

#[derive(Debug, Clone, PartialEq)]
pub enum ContinuityFact {
    PreviouslyReportedLineage {
        lineage_id: LineageId,
        reports_ago: usize,
        previous_theme: ReportTheme,
    },
    ConditionPersisted {
        lineage_id: LineageId,
        tag: FieldReportTag,
        report_count: u32,
    },
    ConditionResolved {
        lineage_id: LineageId,
        tag: FieldReportTag,
    },
    ConditionWorsened {
        lineage_id: LineageId,
        tag: FieldReportTag,
    },
    ConditionReversed {
        lineage_id: LineageId,
        from: FieldReportTag,
        to: FieldReportTag,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArchivedFieldReport {
    pub id: FieldReportId,
    pub created_frame: u64,
    pub created_time: f64,
    pub rendered: RenderedFieldReport,
    pub continuity: Vec<ContinuityFact>,
}

#[derive(Debug, Clone)]
pub struct FieldReportHistory {
    pub reports: VecDeque<ArchivedFieldReport>,
    pub max_reports: usize,
    pub recent_themes: VecDeque<ReportTheme>,
    pub recent_lineages: VecDeque<LineageId>,
    pub recent_tags: VecDeque<FieldReportTag>,
    pub recent_claims: VecDeque<ClaimKey>,
    pub last_report_frame: Option<u64>,
    pub last_report_time: Option<f64>,
}

impl Default for FieldReportHistory {
    fn default() -> Self {
        Self::new(128)
    }
}

impl FieldReportHistory {
    pub fn new(max_reports: usize) -> Self {
        Self {
            reports: VecDeque::new(),
            max_reports: max_reports.max(1),
            recent_themes: VecDeque::new(),
            recent_lineages: VecDeque::new(),
            recent_tags: VecDeque::new(),
            recent_claims: VecDeque::new(),
            last_report_frame: None,
            last_report_time: None,
        }
    }

    pub fn record(&mut self, report: ArchivedFieldReport) {
        self.last_report_frame = Some(report.created_frame);
        self.last_report_time = Some(report.created_time);
        self.recent_themes.push_back(report.rendered.theme);
        self.recent_tags
            .extend(report.rendered.tags.iter().copied());
        self.recent_lineages
            .extend(report.rendered.involved_lineages.iter().copied());
        self.recent_claims.extend(
            report
                .rendered
                .sentences
                .iter()
                .flat_map(|sentence| sentence.claim_keys.iter().copied()),
        );
        self.reports.push_back(report);
        self.trim();
    }

    pub fn continuity_for(&self, plan: &ReportPlan) -> Vec<ContinuityFact> {
        let Some(lineage_id) = plan.subject_lineage else {
            return Vec::new();
        };
        let Some((reports_ago, previous)) = self
            .reports
            .iter()
            .rev()
            .enumerate()
            .find(|(_, report)| report.rendered.involved_lineages.contains(&lineage_id))
            .map(|(index, report)| (index + 1, report))
        else {
            return Vec::new();
        };

        let mut continuity = vec![ContinuityFact::PreviouslyReportedLineage {
            lineage_id,
            reports_ago,
            previous_theme: previous.rendered.theme,
        }];

        if previous.rendered.tags.contains(&FieldReportTag::NearExtinction)
            && plan.tags.contains(&FieldReportTag::Recovery)
        {
            continuity.push(ContinuityFact::ConditionReversed {
                lineage_id,
                from: FieldReportTag::NearExtinction,
                to: FieldReportTag::Recovery,
            });
        } else if previous
            .rendered
            .tags
            .contains(&FieldReportTag::PopulationDecline)
            && plan.tags.contains(&FieldReportTag::PopulationGrowth)
        {
            continuity.push(ContinuityFact::ConditionReversed {
                lineage_id,
                from: FieldReportTag::PopulationDecline,
                to: FieldReportTag::PopulationGrowth,
            });
        }

        for tag in important_condition_tags() {
            let had = previous.rendered.tags.contains(tag);
            let has = plan.tags.contains(tag);
            if had && has {
                let report_count = self
                    .reports
                    .iter()
                    .rev()
                    .take_while(|report| {
                        report.rendered.involved_lineages.contains(&lineage_id)
                            && report.rendered.tags.contains(tag)
                    })
                    .count()
                    .saturating_add(1)
                    .min(u32::MAX as usize) as u32;
                if plan.severity > previous.rendered.severity {
                    continuity.push(ContinuityFact::ConditionWorsened {
                        lineage_id,
                        tag: *tag,
                    });
                } else {
                    continuity.push(ContinuityFact::ConditionPersisted {
                        lineage_id,
                        tag: *tag,
                        report_count,
                    });
                }
            } else if had && !has {
                continuity.push(ContinuityFact::ConditionResolved {
                    lineage_id,
                    tag: *tag,
                });
            }
        }
        continuity
    }

    pub fn recent_report_for_lineage(
        &self,
        lineage_id: LineageId,
        within_reports: usize,
        current_frame: u64,
        within_frames: u64,
    ) -> bool {
        self.reports
            .iter()
            .rev()
            .take(within_reports)
            .any(|report| {
                current_frame.saturating_sub(report.created_frame) < within_frames
                    && report.rendered.involved_lineages.contains(&lineage_id)
            })
    }

    pub fn recent_theme(
        &self,
        theme: ReportTheme,
        within_reports: usize,
        current_frame: u64,
        within_frames: u64,
    ) -> bool {
        self.reports
            .iter()
            .rev()
            .take(within_reports)
            .any(|report| {
                current_frame.saturating_sub(report.created_frame) < within_frames
                    && report.rendered.theme == theme
            })
    }

    pub fn recent_claim(
        &self,
        claims: &[ClaimKey],
        within_reports: usize,
        current_frame: u64,
        within_frames: u64,
    ) -> bool {
        self.reports
            .iter()
            .rev()
            .take(within_reports)
            .filter(|report| current_frame.saturating_sub(report.created_frame) < within_frames)
            .flat_map(|report| report.rendered.sentences.iter())
            .flat_map(|sentence| sentence.claim_keys.iter())
            .any(|claim| claims.contains(claim))
    }

    pub fn last_severity_for_lineage(
        &self,
        lineage_id: LineageId,
    ) -> Option<FieldReportSeverity> {
        self.reports
            .iter()
            .rev()
            .find(|report| report.rendered.involved_lineages.contains(&lineage_id))
            .map(|report| report.rendered.severity)
    }

    pub fn prior_reports_for_plan(&self, plan: &ReportPlan) -> usize {
        self.reports
            .iter()
            .filter(|report| {
                report.rendered.theme == plan.theme
                    && match plan.subject_lineage {
                        Some(lineage_id) => {
                            report.rendered.involved_lineages.contains(&lineage_id)
                        }
                        None => report.rendered.involved_lineages.is_empty(),
                    }
            })
            .count()
    }

    fn trim(&mut self) {
        while self.reports.len() > self.max_reports {
            self.reports.pop_front();
        }
        trim_deque(&mut self.recent_themes, self.max_reports);
        trim_deque(&mut self.recent_lineages, self.max_reports * 4);
        trim_deque(&mut self.recent_tags, self.max_reports * 8);
        trim_deque(&mut self.recent_claims, self.max_reports * 16);
    }
}

fn trim_deque<T>(deque: &mut VecDeque<T>, max_len: usize) {
    while deque.len() > max_len.max(1) {
        deque.pop_front();
    }
}

fn important_condition_tags() -> &'static [FieldReportTag] {
    &[
        FieldReportTag::StarvationRisk,
        FieldReportTag::NearExtinction,
        FieldReportTag::PopulationDecline,
        FieldReportTag::PopulationGrowth,
    ]
}

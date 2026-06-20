use crate::scene::lineage::{
    EcosystemLineageArchive, EcosystemTelemetrySummary, LineageId, LineageTelemetrySample,
};

/// Immutable input to fact extraction.
#[derive(Debug, Clone, Default)]
pub struct FieldReportContext {
    pub frame: i32,
    pub time_seconds: f32,
    pub lineages: Vec<LineageReportSnapshot>,
    pub ecosystem: EcosystemTelemetrySummary,
    pub composition: SceneCompositionSnapshot,
}

#[derive(Debug, Clone, Default)]
pub struct SceneCompositionSnapshot {
    pub current_counts: [u32; crate::cell::types::CellType::MAX_TYPES],
    pub previous_counts: [u32; crate::cell::types::CellType::MAX_TYPES],
}

/// Compact lineage state used by the report brain.
///
/// This owns no per-cell data and can be cheaply retained in report history.
#[derive(Debug, Clone, Default)]
pub struct LineageReportSnapshot {
    pub lineage_id: LineageId,
    pub display_name: String,
    pub current: LineageTelemetrySample,
    pub previous: Option<LineageTelemetrySample>,
    pub cells_30s_ago: Option<u32>,
    pub cells_2m_ago: Option<u32>,
    pub consecutive_growth_windows: u32,
    pub consecutive_decline_windows: u32,
    pub peak_cells: u32,
    pub extinct: bool,
}

impl FieldReportContext {
    pub fn from_archive(archive: &EcosystemLineageArchive) -> Self {
        let lineages = archive
            .nodes
            .iter()
            .filter_map(|node| {
                let current = *node.latest_telemetry()?;
                let previous = node
                    .telemetry_history
                    .get(node.telemetry_history.len().saturating_sub(2))
                    .copied()
                    .filter(|sample| sample.frame != current.frame);
                Some(LineageReportSnapshot {
                    lineage_id: node.id,
                    display_name: node.display_name.clone(),
                    current,
                    previous,
                    cells_30s_ago: node
                        .telemetry_at_least_seconds_ago(30.0)
                        .map(|sample| sample.cells),
                    cells_2m_ago: node
                        .telemetry_at_least_seconds_ago(120.0)
                        .map(|sample| sample.cells),
                    consecutive_growth_windows: node.consecutive_growth_windows(),
                    consecutive_decline_windows: node.consecutive_decline_windows(),
                    peak_cells: node.peak_cells,
                    extinct: node.extinct_frame.is_some(),
                })
            })
            .collect();

        let mut composition = SceneCompositionSnapshot::default();
        for node in archive.nodes.iter().filter(|node| node.current_cells > 0) {
            if let Some(current) = node.latest_telemetry() {
                for (target, count) in composition
                    .current_counts
                    .iter_mut()
                    .zip(current.cell_type_counts)
                {
                    *target = target.saturating_add(count);
                }
            }
            if let Some(previous) = node
                .telemetry_history
                .get(node.telemetry_history.len().saturating_sub(2))
            {
                for (target, count) in composition
                    .previous_counts
                    .iter_mut()
                    .zip(previous.cell_type_counts)
                {
                    *target = target.saturating_add(count);
                }
            }
        }

        Self {
            frame: archive.last_scan_frame.unwrap_or_default(),
            time_seconds: archive.last_scan_time,
            lineages,
            ecosystem: archive.ecosystem_telemetry(),
            composition,
        }
    }

    pub fn lineage(&self, lineage_id: LineageId) -> Option<&LineageReportSnapshot> {
        self.lineages
            .iter()
            .find(|lineage| lineage.lineage_id == lineage_id)
    }
}

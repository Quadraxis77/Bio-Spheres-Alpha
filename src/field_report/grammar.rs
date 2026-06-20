use crate::scene::lineage::LineageId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SentenceRole {
    Lead,
    Evidence,
    Contrast,
    Interpretation,
    Watch,
    DossierDetail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClaimKey {
    PopulationIncrease(LineageId),
    PopulationDecline(LineageId),
    PopulationPeak(LineageId),
    NearExtinction(LineageId),
    Recovery(LineageId),
    NutrientPressure(LineageId),
    DivisionReadiness(LineageId),
    DominantCellType(LineageId),
    Territory(LineageId),
    EcosystemDiversity,
    EcosystemDominance,
}

pub fn role_allows_supporting_detail(role: SentenceRole) -> bool {
    matches!(
        role,
        SentenceRole::Evidence | SentenceRole::Contrast | SentenceRole::DossierDetail
    )
}

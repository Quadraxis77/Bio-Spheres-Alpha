use serde::{Deserialize, Serialize};

use crate::field_report::plan::FieldReportSeverity;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToneId {
    FormalScientific,
    NaturalistFieldJournal,
    LivingEcosystem,
    AlertMonitor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToneFamily {
    Formal,
    Naturalist,
    Living,
    Alert,
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UncertaintyStyle {
    Explicit,
    Soft,
    Intuitive,
    Direct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SentenceLengthStyle {
    Short,
    Medium,
    MediumLong,
    Varied,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumberPolicy {
    Precise,
    Rounded,
    Sparse,
    Qualitative,
}

/// Style configuration applied after a semantic `ReportPlan` exists.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ToneProfile {
    pub id: ToneId,
    pub formality: f32,
    pub warmth: f32,
    pub organic_language: f32,
    pub scientific_precision: f32,
    pub urgency: f32,
    pub poetic_density: f32,
    pub uncertainty_style: UncertaintyStyle,
    pub sentence_length: SentenceLengthStyle,
    pub number_policy: NumberPolicy,
    pub metaphor_budget: u8,
}

impl ToneProfile {
    pub const fn family(&self) -> ToneFamily {
        match self.id {
            ToneId::FormalScientific => ToneFamily::Formal,
            ToneId::NaturalistFieldJournal => ToneFamily::Naturalist,
            ToneId::LivingEcosystem => ToneFamily::Living,
            ToneId::AlertMonitor => ToneFamily::Alert,
        }
    }
    pub const fn formal_scientific() -> Self {
        Self {
            id: ToneId::FormalScientific,
            formality: 0.9,
            warmth: 0.2,
            organic_language: 0.2,
            scientific_precision: 0.9,
            urgency: 0.4,
            poetic_density: 0.1,
            uncertainty_style: UncertaintyStyle::Explicit,
            sentence_length: SentenceLengthStyle::Medium,
            number_policy: NumberPolicy::Precise,
            metaphor_budget: 0,
        }
    }

    pub const fn naturalist_field_journal() -> Self {
        Self {
            id: ToneId::NaturalistFieldJournal,
            formality: 0.55,
            warmth: 0.55,
            organic_language: 0.75,
            scientific_precision: 0.65,
            urgency: 0.35,
            poetic_density: 0.45,
            uncertainty_style: UncertaintyStyle::Soft,
            sentence_length: SentenceLengthStyle::MediumLong,
            number_policy: NumberPolicy::Rounded,
            metaphor_budget: 1,
        }
    }

    pub const fn living_ecosystem() -> Self {
        Self {
            id: ToneId::LivingEcosystem,
            formality: 0.25,
            warmth: 0.65,
            organic_language: 0.95,
            scientific_precision: 0.45,
            urgency: 0.4,
            poetic_density: 0.65,
            uncertainty_style: UncertaintyStyle::Intuitive,
            sentence_length: SentenceLengthStyle::Varied,
            number_policy: NumberPolicy::Sparse,
            metaphor_budget: 2,
        }
    }

    pub const fn alert_monitor() -> Self {
        Self {
            id: ToneId::AlertMonitor,
            formality: 0.75,
            warmth: 0.1,
            organic_language: 0.2,
            scientific_precision: 0.8,
            urgency: 0.9,
            poetic_density: 0.0,
            uncertainty_style: UncertaintyStyle::Direct,
            sentence_length: SentenceLengthStyle::Short,
            number_policy: NumberPolicy::Rounded,
            metaphor_budget: 0,
        }
    }
}

impl ToneId {
    pub const fn profile(self) -> ToneProfile {
        match self {
            Self::FormalScientific => ToneProfile::formal_scientific(),
            Self::NaturalistFieldJournal => ToneProfile::naturalist_field_journal(),
            Self::LivingEcosystem => ToneProfile::living_ecosystem(),
            Self::AlertMonitor => ToneProfile::alert_monitor(),
        }
    }
}

/// Applies severity constraints without changing the selected voice identity.
pub fn resolve_tone(mut tone: ToneProfile, severity: FieldReportSeverity) -> ToneProfile {
    match severity {
        FieldReportSeverity::Critical => {
            tone.urgency = tone.urgency.max(0.8);
            tone.scientific_precision = tone.scientific_precision.max(0.55);
            tone.poetic_density *= 0.6;
            tone.metaphor_budget = tone.metaphor_budget.min(1);
            if matches!(tone.number_policy, NumberPolicy::Qualitative) {
                tone.number_policy = NumberPolicy::Rounded;
            }
        }
        FieldReportSeverity::Warning => {
            tone.urgency = tone.urgency.max(0.55);
        }
        FieldReportSeverity::Routine | FieldReportSeverity::Notable => {}
    }
    tone
}

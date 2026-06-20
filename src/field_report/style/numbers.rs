use super::{NumberPolicy, ToneProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberDensity {
    None,
    Sparse,
    Rounded,
    Precise,
}

fn effective_density(tone: &ToneProfile, density: NumberDensity) -> NumberDensity {
    if !matches!(density, NumberDensity::Sparse) {
        return density;
    }
    match tone.number_policy {
        NumberPolicy::Precise => NumberDensity::Precise,
        NumberPolicy::Rounded => NumberDensity::Rounded,
        NumberPolicy::Sparse | NumberPolicy::Qualitative => NumberDensity::Sparse,
    }
}

pub fn format_percent(value: f32, tone: &ToneProfile, density: NumberDensity) -> String {
    let percent = (value.clamp(0.0, 1.0) * 100.0).round() as i32;
    match effective_density(tone, density) {
        NumberDensity::None => qualitative_percent(value).to_string(),
        NumberDensity::Sparse if matches!(tone.number_policy, NumberPolicy::Sparse) => {
            qualitative_percent(value).to_string()
        }
        NumberDensity::Rounded => format!("about {}%", ((percent + 5) / 10) * 10),
        NumberDensity::Sparse | NumberDensity::Precise => format!("{percent}%"),
    }
}

pub fn format_cell_count(value: u32, tone: &ToneProfile, density: NumberDensity) -> String {
    match effective_density(tone, density) {
        NumberDensity::None => "the sampled population".to_string(),
        NumberDensity::Rounded if value >= 100 => format!("about {}", ((value + 5) / 10) * 10),
        NumberDensity::Sparse if matches!(tone.number_policy, NumberPolicy::Sparse) && value >= 1000 => {
            format!("{:.1}k", value as f32 / 1000.0)
        }
        _ => value.to_string(),
    }
}

pub fn format_delta_cells(value: i32, tone: &ToneProfile, density: NumberDensity) -> String {
    let magnitude = value.unsigned_abs();
    let formatted = format_cell_count(magnitude, tone, density);
    if value >= 0 {
        format!("+{formatted}")
    } else {
        format!("-{formatted}")
    }
}

fn qualitative_percent(value: f32) -> &'static str {
    if value >= 0.75 {
        "very high"
    } else if value >= 0.5 {
        "high"
    } else if value >= 0.25 {
        "moderate"
    } else if value > 0.0 {
        "low"
    } else {
        "none"
    }
}

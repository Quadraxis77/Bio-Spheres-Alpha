use super::{ToneFamily, ToneProfile};

#[derive(Debug, Clone, Copy)]
pub struct ToneLexicon {
    pub lineage_prefix: &'static str,
    pub expanding: &'static str,
    pub declining: &'static str,
    pub starvation_pressure: &'static str,
    pub recovery: &'static str,
    pub near_extinction: &'static str,
    pub stable_growth: &'static str,
    pub watch: &'static str,
}

impl ToneLexicon {
    pub fn for_tone(tone: &ToneProfile) -> Self {
        match tone.family() {
            ToneFamily::Formal => Self {
                lineage_prefix: "Lineage ",
                expanding: "expanding",
                declining: "declining",
                starvation_pressure: "resource pressure",
                recovery: "recovery",
                near_extinction: "near-extinction",
                stable_growth: "stable growth",
                watch: "Monitor",
            },
            ToneFamily::Naturalist => Self {
                lineage_prefix: "",
                expanding: "spreading",
                declining: "thinning",
                starvation_pressure: "strained resources",
                recovery: "recovery",
                near_extinction: "a thin remnant",
                stable_growth: "a new foothold",
                watch: "Watch",
            },
            ToneFamily::Living => Self {
                lineage_prefix: "",
                expanding: "blooming",
                declining: "fading",
                starvation_pressure: "hunger",
                recovery: "holding on",
                near_extinction: "close to disappearing",
                stable_growth: "a steady foothold",
                watch: "Watch",
            },
            ToneFamily::Alert => Self {
                lineage_prefix: "",
                expanding: "increasing",
                declining: "decreasing",
                starvation_pressure: "resource stress",
                recovery: "recovery detected",
                near_extinction: "near-extinction warning",
                stable_growth: "stable",
                watch: "Monitor",
            },
            ToneFamily::Any => unreachable!("profiles always resolve to a concrete family"),
        }
    }
}

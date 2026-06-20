//! Presentation policy for future field-report renderers.
//!
//! Nothing in this module extracts facts or chooses report themes.

mod lexicon;
mod numbers;
mod titles;
mod tone;

pub use lexicon::ToneLexicon;
pub use numbers::{
    format_cell_count, format_delta_cells, format_percent, NumberDensity,
};
pub use titles::render_title;
pub use tone::{
    resolve_tone, NumberPolicy, SentenceLengthStyle, ToneFamily, ToneId, ToneProfile,
    UncertaintyStyle,
};

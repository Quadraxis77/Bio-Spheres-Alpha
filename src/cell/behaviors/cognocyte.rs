//! Cognocyte Cell Behavior
//!
//! Cognocytes are signal-processing cells. Each reads signals from two input
//! channels, applies a configurable operation, and emits the result on an output
//! channel. If either required input channel carries no signal, the cell emits
//! nothing - misconfigured circuits go dark visibly rather than producing
//! plausible-looking garbage.
//!
//! # Operations
//!
//! ## Arithmetic
//! - **Add** (0): `A + B`
//! - **Subtract** (1): `A - B`
//! - **Multiply** (2): `A * B`
//! - **Divide** (3): `A / B` (safe: returns 0 if B == 0)
//! - **Min** (4): `min(A, B)`
//! - **Max** (5): `max(A, B)`
//! - **Average** (6): `(A + B) / 2`
//!
//! ## Comparison - output 1.0 (true) or 0.0 (false)
//! - **GreaterThan** (7): `A > B`
//! - **LessThan** (8): `A < B`
//! - **Equal** (9): `|A - B| < epsilon`
//!
//! ## Boolean - truth test: value > 0
//! - **AND** (10): `A > 0 && B > 0`
//! - **OR** (11): `A > 0 || B > 0`
//! - **NOT** (12): `!(A > 0)` - unary, uses A only
//!
//! ## Control flow
//! - **Select** (13): `if A > 0 { B } else { 0.0 }` - gate / mux
//!
//! # Missing inputs
//!
//! If any required input channel has no signal, the cell emits nothing.
//! NOT only requires channel A. All other operations require both A and B.
//! This is enforced at the call site in `signal_system::process_cognocytes`.
//!
//! # Composability
//!
//! Clean 1.0/0.0 outputs from comparisons and boolean ops feed naturally into
//! downstream arithmetic or boolean cells. A chain like:
//!
//! ```text
//! Oculocyte  ->  GreaterThan(food, hunger)  ->  AND(mature, hungry)  ->  Embryocyte
//! ```
//!
//! lets organisms build threshold detectors, timers, state machines, and
//! multi-condition decision trees out of a uniform set of primitives.

use crate::genome::ModeSettings;
use super::{CellBehavior, TypeSpecificInstanceData};

/// Operation codes stored in `ModeSettings::cognocyte_operation`.
pub const OP_ADD: i32          = 0;
pub const OP_SUBTRACT: i32     = 1;
pub const OP_MULTIPLY: i32     = 2;
pub const OP_DIVIDE: i32       = 3;
pub const OP_MIN: i32          = 4;
pub const OP_MAX: i32          = 5;
pub const OP_AVERAGE: i32      = 6;
pub const OP_GREATER_THAN: i32 = 7;
pub const OP_LESS_THAN: i32    = 8;
pub const OP_EQUAL: i32        = 9;
pub const OP_AND: i32          = 10;
pub const OP_OR: i32           = 11;
pub const OP_NOT: i32          = 12;
pub const OP_SELECT: i32       = 13;

/// Number of defined operations.
pub const OP_COUNT: i32 = 14;

/// Tolerance used by the Equal operation.
const EQUAL_EPSILON: f32 = 1e-4;

/// Behavior implementation for Cognocyte (signal-processing) cells.
pub struct CognocyteBehavior;

impl CellBehavior for CognocyteBehavior {
    fn build_instance_data(&self, _mode_settings: &ModeSettings) -> TypeSpecificInstanceData {
        TypeSpecificInstanceData::empty()
    }
}

/// Evaluate a cognocyte operation.
///
/// Both `a` and `b` are already-resolved signal values. The caller is
/// responsible for not calling this function when a required input is absent -
/// in that case no emission should occur at all.
///
/// For Select: `A > 0 -> B`, `A <= 0 -> 0.0`.
pub fn evaluate(op: i32, a: f32, b: f32) -> f32 {
    match op {
        OP_ADD          => a + b,
        OP_SUBTRACT     => a - b,
        OP_MULTIPLY     => a * b,
        OP_DIVIDE       => if b.abs() < EQUAL_EPSILON { 0.0 } else { a / b },
        OP_MIN          => a.min(b),
        OP_MAX          => a.max(b),
        OP_AVERAGE      => (a + b) * 0.5,
        OP_GREATER_THAN => if a > b       { 1.0 } else { 0.0 },
        OP_LESS_THAN    => if a < b       { 1.0 } else { 0.0 },
        OP_EQUAL        => if (a - b).abs() < EQUAL_EPSILON { 1.0 } else { 0.0 },
        OP_AND          => if a > 0.0 && b > 0.0 { 1.0 } else { 0.0 },
        OP_OR           => if a > 0.0 || b > 0.0 { 1.0 } else { 0.0 },
        OP_NOT          => if a > 0.0 { 0.0 } else { 1.0 }, // b ignored
        OP_SELECT       => if a > 0.0 { b } else { 0.0 },
        _               => 0.0,
    }
}

/// Human-readable name for a cognocyte operation code. Used by the UI.
pub fn op_name(op: i32) -> &'static str {
    match op {
        OP_ADD          => "Add",
        OP_SUBTRACT     => "Subtract",
        OP_MULTIPLY     => "Multiply",
        OP_DIVIDE       => "Divide",
        OP_MIN          => "Min",
        OP_MAX          => "Max",
        OP_AVERAGE      => "Average",
        OP_GREATER_THAN => "Greater Than",
        OP_LESS_THAN    => "Less Than",
        OP_EQUAL        => "Equal",
        OP_AND          => "AND",
        OP_OR           => "OR",
        OP_NOT          => "NOT",
        OP_SELECT       => "Select",
        _               => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_ops() {
        assert_eq!(evaluate(OP_ADD,      3.0, 2.0), 5.0);
        assert_eq!(evaluate(OP_SUBTRACT, 3.0, 2.0), 1.0);
        assert_eq!(evaluate(OP_MULTIPLY, 3.0, 2.0), 6.0);
        assert_eq!(evaluate(OP_DIVIDE,   6.0, 2.0), 3.0);
        assert_eq!(evaluate(OP_MIN,      3.0, 2.0), 2.0);
        assert_eq!(evaluate(OP_MAX,      3.0, 2.0), 3.0);
        assert_eq!(evaluate(OP_AVERAGE,  3.0, 1.0), 2.0);
    }

    #[test]
    fn divide_by_zero_returns_zero() {
        assert_eq!(evaluate(OP_DIVIDE, 5.0, 0.0), 0.0);
    }

    #[test]
    fn comparison_ops() {
        assert_eq!(evaluate(OP_GREATER_THAN, 3.0, 2.0), 1.0);
        assert_eq!(evaluate(OP_GREATER_THAN, 1.0, 2.0), 0.0);
        assert_eq!(evaluate(OP_LESS_THAN,    1.0, 2.0), 1.0);
        assert_eq!(evaluate(OP_LESS_THAN,    3.0, 2.0), 0.0);
        assert_eq!(evaluate(OP_EQUAL,        2.0, 2.0), 1.0);
        assert_eq!(evaluate(OP_EQUAL,        2.0, 3.0), 0.0);
    }

    #[test]
    fn boolean_ops() {
        assert_eq!(evaluate(OP_AND, 1.0, 1.0), 1.0);
        assert_eq!(evaluate(OP_AND, 1.0, 0.0), 0.0);
        assert_eq!(evaluate(OP_OR,  0.0, 1.0), 1.0);
        assert_eq!(evaluate(OP_OR,  0.0, 0.0), 0.0);
        assert_eq!(evaluate(OP_NOT, 1.0, 0.0), 0.0); // b ignored
        assert_eq!(evaluate(OP_NOT, 0.0, 9.0), 1.0); // b ignored
    }

    #[test]
    fn select_op() {
        assert_eq!(evaluate(OP_SELECT, 1.0, 7.0), 7.0); // A true -> B
        assert_eq!(evaluate(OP_SELECT, 0.0, 7.0), 0.0); // A false -> 0
    }

    #[test]
    fn cognocyte_behavior_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CognocyteBehavior>();
    }
}

use glam::Vec3;

/// Adhesion zone classification for division inheritance
///
/// Zones determine which child cell inherits an adhesion connection during division:
/// - Zone A: Adhesions pointing opposite to split direction -> inherit to child B
/// - Zone B: Adhesions pointing same as split direction -> inherit to child A
/// - Zone C: Adhesions in equatorial band (90 deg  threshold) -> inherit to both children
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdhesionZone {
    ZoneA = 0, // Green in visualization
    ZoneB = 1, // Blue in visualization
    ZoneC = 2, // Red in visualization (equatorial)
}

/// Base equatorial threshold in degrees (used when split_ratio = 0.5)
pub const EQUATORIAL_THRESHOLD_DEGREES_MIN: f32 = 3.0;

/// Max equatorial threshold in degrees (used when split_ratio = 0.3 or 0.7)
pub const EQUATORIAL_THRESHOLD_DEGREES_MAX: f32 = 22.0;

/// Legacy constant kept for backward compatibility
pub const EQUATORIAL_THRESHOLD_DEGREES: f32 = 3.0;

/// Compute the dynamic equatorial zone width in degrees based on split_ratio.
/// Returns 60 deg when split_ratio = 0.5 (120 deg total Zone C arc for equilateral triangles),
/// linearly increasing to 75 deg at split_ratio = 0.3 or 0.7.
pub fn compute_equatorial_degrees(split_ratio: f32) -> f32 {
    let deviation = (split_ratio - 0.5).abs();
    // Linear interpolation: 60 deg at deviation=0, 75 deg at deviation=0.2
    let t = (deviation / 0.2).min(1.0);
    EQUATORIAL_THRESHOLD_DEGREES_MIN
        + (EQUATORIAL_THRESHOLD_DEGREES_MAX - EQUATORIAL_THRESHOLD_DEGREES_MIN) * t
}

/// Compute the ratio shift that moves the split plane away from the equator.
/// At split_ratio=0.5, shift=0. At 0.7, shift=0.4. At 0.3, shift=-0.4.
pub fn compute_ratio_shift(split_ratio: f32) -> f32 {
    2.0 * split_ratio - 1.0
}

/// Classify adhesion bond direction relative to split direction with dynamic equatorial zone.
///
/// The adhesion classification plane is shifted by `ratio_shift = 2*split_ratio - 1`
/// to bias which child inherits bonds without changing other divided resources.
/// The equatorial band width is 60 deg at split_ratio=0.5 (giving a 120 deg total Zone C arc,
/// wide enough for equilateral triangle formation), scaling to 75 deg at 0.3/0.7.
///
/// # Arguments
/// * `bond_direction` - Direction of the adhesion bond (normalized)
/// * `split_direction` - Direction of cell division (normalized)
/// * `split_ratio` - Adhesion distribution ratio (0.0-1.0)
///
/// # Returns
/// The zone classification for this adhesion
pub fn classify_bond_direction(
    bond_direction: Vec3,
    split_direction: Vec3,
    split_ratio: f32,
) -> AdhesionZone {
    let dot_product = bond_direction.normalize().dot(split_direction.normalize());

    // Shift the split plane based on split_ratio (reference: ratio_shift = 2*split_ratio - 1)
    let ratio_shift = compute_ratio_shift(split_ratio);
    let shifted_dot = dot_product - ratio_shift;

    // Dynamic equatorial threshold based on split_ratio
    let equatorial_degrees = compute_equatorial_degrees(split_ratio);
    let equatorial_threshold = equatorial_degrees.to_radians().sin();

    // Zone classification using shifted dot product
    if shifted_dot.abs() <= equatorial_threshold {
        AdhesionZone::ZoneC // Equatorial - near the (shifted) split plane
    } else if shifted_dot > 0.0 {
        AdhesionZone::ZoneB // Positive shifted dot (same direction as split)
    } else {
        AdhesionZone::ZoneA // Negative shifted dot (opposite to split)
    }
}

/// Get zone color for visualization (matches GPU shader)
/// Returns [R, G, B, A] in 0.0-1.0 range
pub fn get_zone_color(zone: AdhesionZone) -> [f32; 4] {
    match zone {
        AdhesionZone::ZoneA => [0.0, 1.0, 0.0, 1.0], // Green
        AdhesionZone::ZoneB => [0.0, 0.0, 1.0, 1.0], // Blue
        AdhesionZone::ZoneC => [1.0, 0.0, 0.0, 1.0], // Red
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equatorial_degrees() {
        // At split_ratio=0.5, equatorial zone = 3 deg
        let deg_50 = compute_equatorial_degrees(0.5);
        assert!((deg_50 - 3.0).abs() < 0.01, "Expected 3.0, got {}", deg_50);

        // At split_ratio=0.7, equatorial zone = 22 deg
        let deg_70 = compute_equatorial_degrees(0.7);
        assert!(
            (deg_70 - 22.0).abs() < 0.01,
            "Expected 22.0, got {}",
            deg_70
        );

        // At split_ratio=0.3, equatorial zone = 22 deg
        let deg_30 = compute_equatorial_degrees(0.3);
        assert!(
            (deg_30 - 22.0).abs() < 0.01,
            "Expected 22.0, got {}",
            deg_30
        );

        // At split_ratio=0.6, should be halfway: 12.5 deg
        let deg_60 = compute_equatorial_degrees(0.6);
        assert!(
            (deg_60 - 12.5).abs() < 0.01,
            "Expected 12.5, got {}",
            deg_60
        );
    }

    #[test]
    fn test_ratio_shift() {
        assert!((compute_ratio_shift(0.5) - 0.0).abs() < 0.001);
        assert!((compute_ratio_shift(0.7) - 0.4).abs() < 0.001);
        assert!((compute_ratio_shift(0.3) - (-0.4)).abs() < 0.001);
    }

    #[test]
    fn test_zone_classification_equal_split() {
        let split_dir = Vec3::Y;
        let ratio = 0.5; // Equal split, 3 deg zone, no shift
                         // threshold = sin(3 deg) ~= 0.052

        // Bond pointing straight up (along split): dot=1.0, |shifted|=1.0 > 0.052 -> Zone B
        let bond_b = Vec3::new(0.0, 1.0, 0.0).normalize();
        assert_eq!(
            classify_bond_direction(bond_b, split_dir, ratio),
            AdhesionZone::ZoneB
        );

        // Bond pointing straight down (opposite split): dot=-1.0, |shifted|=1.0 > 0.052 -> Zone A
        let bond_a = Vec3::new(0.0, -1.0, 0.0).normalize();
        assert_eq!(
            classify_bond_direction(bond_a, split_dir, ratio),
            AdhesionZone::ZoneA
        );

        // Bond perpendicular to split: dot=0.0, |shifted|=0.0 <= 0.052 -> Zone C
        let bond_c = Vec3::new(1.0, 0.0, 0.0).normalize();
        assert_eq!(
            classify_bond_direction(bond_c, split_dir, ratio),
            AdhesionZone::ZoneC
        );

        // Bond at 2 deg from equator: dot=sin(2 deg)~=0.035, |shifted|~=0.035 <= 0.052 -> Zone C
        let bond_2deg = Vec3::new(0.0, 0.035, 1.0).normalize();
        assert_eq!(
            classify_bond_direction(bond_2deg, split_dir, ratio),
            AdhesionZone::ZoneC
        );

        // Bond at 5 deg from equator: dot=sin(5 deg)~=0.087, |shifted|~=0.087 > 0.052 -> Zone B
        let bond_5deg = Vec3::new(0.0, 0.087, 1.0).normalize();
        assert_eq!(
            classify_bond_direction(bond_5deg, split_dir, ratio),
            AdhesionZone::ZoneB
        );
    }

    #[test]
    fn test_zone_classification_unequal_split() {
        let split_dir = Vec3::Y;
        let ratio = 0.7; // Unequal split, 22 deg zone, shift=0.4
                         // threshold = sin(22 deg) ~= 0.375

        // Bond pointing straight up: dot=1.0, shifted=1.0-0.4=0.6, |0.6| > 0.375 -> Zone B
        let bond_up = Vec3::new(0.0, 1.0, 0.0).normalize();
        assert_eq!(
            classify_bond_direction(bond_up, split_dir, ratio),
            AdhesionZone::ZoneB
        );

        // Bond pointing straight down: dot=-1.0, shifted=-1.0-0.4=-1.4, |-1.4| > 0.375 -> Zone A
        let bond_down = Vec3::new(0.0, -1.0, 0.0).normalize();
        assert_eq!(
            classify_bond_direction(bond_down, split_dir, ratio),
            AdhesionZone::ZoneA
        );

        // Bond perpendicular: dot=0.0, shifted=0.0-0.4=-0.4, |-0.4| > 0.375 -> Zone A
        let bond_perp = Vec3::new(1.0, 0.0, 0.0).normalize();
        assert_eq!(
            classify_bond_direction(bond_perp, split_dir, ratio),
            AdhesionZone::ZoneA
        );
    }

    #[test]
    fn test_zone_colors() {
        // Just verify colors are distinct
        let color_a = get_zone_color(AdhesionZone::ZoneA);
        let color_b = get_zone_color(AdhesionZone::ZoneB);
        let color_c = get_zone_color(AdhesionZone::ZoneC);

        assert_ne!(color_a, color_b);
        assert_ne!(color_b, color_c);
        assert_ne!(color_a, color_c);
    }
}

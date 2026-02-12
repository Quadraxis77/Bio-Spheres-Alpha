use glam::Vec3;

/// Adhesion zone classification for division inheritance
/// 
/// Zones determine which child cell inherits an adhesion connection during division:
/// - Zone A: Adhesions pointing opposite to split direction → inherit to child B
/// - Zone B: Adhesions pointing same as split direction → inherit to child A
/// - Zone C: Adhesions in equatorial band (90° ± threshold) → inherit to both children
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdhesionZone {
    ZoneA = 0,  // Green in visualization
    ZoneB = 1,  // Blue in visualization
    ZoneC = 2,  // Red in visualization (equatorial)
}

/// Equatorial threshold in degrees (±2° from 90°, matching Biospheres-Master)
pub const EQUATORIAL_THRESHOLD_DEGREES: f32 = 2.0;

/// Classify adhesion bond direction relative to split direction
/// 
/// Uses dot product threshold: sin(2°) ≈ 0.0349, scaled by split ratio symmetry.
/// - Zone C: abs(shifted_dot) <= threshold (equatorial band)
/// - Zone B: shifted_dot > 0 (pointing same direction as split) → inherit to child A
/// - Zone A: shifted_dot < 0 (pointing opposite to split) → inherit to child B
/// 
/// When `split_ratio` != 0.5, the split plane shifts so the larger daughter
/// inherits more bonds, and the equatorial band narrows at extreme ratios.
/// 
/// # Arguments
/// * `bond_direction` - Direction of the adhesion bond (normalized)
/// * `split_direction` - Direction of cell division (normalized)
/// * `split_ratio` - Fraction of mass going to child A (0.0 to 1.0, 0.5 = symmetric)
/// 
/// # Returns
/// The zone classification for this adhesion
pub fn classify_bond_direction(bond_direction: Vec3, split_direction: Vec3, split_ratio: f32) -> AdhesionZone {
    let dot_product = bond_direction.normalize().dot(split_direction.normalize());
    
    // Shift the split plane based on split_ratio so the larger daughter inherits more bonds
    // At 0.5 (symmetric): ratio_shift = 0.0 (no shift)
    // At 0.7: ratio_shift = 0.4 (plane shifts toward child B side)
    let ratio_shift = 2.0 * split_ratio - 1.0;
    let shifted_dot = dot_product - ratio_shift;
    
    // Scale equatorial threshold by ratio symmetry — narrows at extreme ratios
    // At 0.5: ratio_symmetry = 1.0 (full width)
    // At 0.9: ratio_symmetry = 0.2 (very narrow)
    let ratio_symmetry = 2.0 * split_ratio.min(1.0 - split_ratio);
    let base_threshold = EQUATORIAL_THRESHOLD_DEGREES.to_radians().sin();
    let equatorial_threshold = base_threshold * ratio_symmetry;
    
    if shifted_dot.abs() <= equatorial_threshold {
        AdhesionZone::ZoneC // Equatorial - near the (shifted) split plane
    } else if shifted_dot > 0.0 {
        AdhesionZone::ZoneB // Same direction as split → inherit to child A
    } else {
        AdhesionZone::ZoneA // Opposite to split → inherit to child B
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
    fn test_zone_classification_symmetric() {
        let split_dir = Vec3::Y; // Split along Y axis
        let ratio = 0.5; // Symmetric split
        
        // Test Zone A (opposite to split direction)
        let bond_a = Vec3::new(0.0, -1.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_a, split_dir, ratio), AdhesionZone::ZoneA);
        
        // Test Zone B (same as split direction)
        let bond_b = Vec3::new(0.0, 1.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_b, split_dir, ratio), AdhesionZone::ZoneB);
        
        // Test Zone C (equatorial - perpendicular to split)
        // sin(2°) ≈ 0.0349, so dot product must be <= 0.0349 for Zone C
        let bond_c = Vec3::new(1.0, 0.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_c, split_dir, ratio), AdhesionZone::ZoneC);
        
        // Test Zone C (equatorial - another perpendicular direction)
        let bond_c2 = Vec3::new(0.0, 0.0, 1.0).normalize();
        assert_eq!(classify_bond_direction(bond_c2, split_dir, ratio), AdhesionZone::ZoneC);
        
        // Test near-equatorial (should be Zone C with 2° threshold)
        // dot = sin(2°) ≈ 0.0349, so y component of 0.034 at x=1 gives dot ≈ 0.034
        let bond_near_eq = Vec3::new(1.0, 0.034, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_near_eq, split_dir, ratio), AdhesionZone::ZoneC);
        
        // Test just outside equatorial (should be Zone B with 2° threshold)
        // y component of 0.05 at x=1 gives dot ≈ 0.05 which is > sin(2°) ≈ 0.0349
        let bond_outside_eq = Vec3::new(1.0, 0.05, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_outside_eq, split_dir, ratio), AdhesionZone::ZoneB);
    }
    
    #[test]
    fn test_zone_classification_asymmetric() {
        let split_dir = Vec3::Y;
        
        // With ratio 0.7, ratio_shift = 0.4, so the split plane shifts toward child B.
        // A bond at dot=0.0 (perpendicular) gets shifted_dot = 0.0 - 0.4 = -0.4 → Zone A
        // This means more bonds go to child A (the larger daughter).
        let bond_perp = Vec3::new(1.0, 0.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_perp, split_dir, 0.7), AdhesionZone::ZoneA);
        
        // A bond pointing slightly toward split dir (dot ≈ 0.3) gets shifted_dot = 0.3 - 0.4 = -0.1 → Zone A
        let bond_slight = Vec3::new(1.0, 0.3, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_slight, split_dir, 0.7), AdhesionZone::ZoneA);
        
        // A bond pointing strongly toward split dir (dot ≈ 0.9) gets shifted_dot = 0.9 - 0.4 = 0.5 → Zone B
        let bond_strong = Vec3::new(0.0, 1.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_strong, split_dir, 0.7), AdhesionZone::ZoneB);
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

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
/// This matches the reference implementation (Biospheres-Master) EXACTLY:
/// - Uses dot product threshold: sin(2°) ≈ 0.0349
/// - Zone C: abs(dot) <= threshold (almost perpendicular to split direction)
/// - Zone B: dot > 0 (pointing same direction as split) → inherit to child A
/// - Zone A: dot < 0 (pointing opposite to split) → inherit to child B
/// 
/// # Arguments
/// * `bond_direction` - Direction of the adhesion bond (normalized)
/// * `split_direction` - Direction of cell division (normalized)
/// 
/// # Returns
/// The zone classification for this adhesion
pub fn classify_bond_direction(bond_direction: Vec3, split_direction: Vec3) -> AdhesionZone {
    let dot_product = bond_direction.normalize().dot(split_direction.normalize());
    
    // Reference uses: sin(radians(2.0)) as threshold
    // sin(2°) ≈ 0.0349
    let equatorial_threshold = EQUATORIAL_THRESHOLD_DEGREES.to_radians().sin();
    
    // Zone classification based on dot product (matches reference exactly)
    if dot_product.abs() <= equatorial_threshold {
        AdhesionZone::ZoneC // Equatorial - almost perpendicular to split direction
    } else if dot_product > 0.0 {
        AdhesionZone::ZoneB // Positive dot product (same direction as split)
    } else {
        AdhesionZone::ZoneA // Negative dot product (opposite to split)
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
    fn test_zone_classification() {
        let split_dir = Vec3::Y; // Split along Y axis
        
        // Test Zone A (opposite to split direction)
        let bond_a = Vec3::new(0.0, -1.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_a, split_dir), AdhesionZone::ZoneA);
        
        // Test Zone B (same as split direction)
        let bond_b = Vec3::new(0.0, 1.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_b, split_dir), AdhesionZone::ZoneB);
        
        // Test Zone C (equatorial - perpendicular to split)
        // sin(2°) ≈ 0.0349, so dot product must be <= 0.0349 for Zone C
        let bond_c = Vec3::new(1.0, 0.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_c, split_dir), AdhesionZone::ZoneC);
        
        // Test Zone C (equatorial - another perpendicular direction)
        let bond_c2 = Vec3::new(0.0, 0.0, 1.0).normalize();
        assert_eq!(classify_bond_direction(bond_c2, split_dir), AdhesionZone::ZoneC);
        
        // Test near-equatorial (should be Zone C with 2° threshold)
        // dot = sin(2°) ≈ 0.0349, so y component of 0.034 at x=1 gives dot ≈ 0.034
        let bond_near_eq = Vec3::new(1.0, 0.034, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_near_eq, split_dir), AdhesionZone::ZoneC);
        
        // Test just outside equatorial (should be Zone B with 2° threshold)
        // y component of 0.05 at x=1 gives dot ≈ 0.05 which is > sin(2°) ≈ 0.0349
        let bond_outside_eq = Vec3::new(1.0, 0.05, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_outside_eq, split_dir), AdhesionZone::ZoneB);
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

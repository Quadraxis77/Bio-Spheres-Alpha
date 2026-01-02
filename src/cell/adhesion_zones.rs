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

/// Equatorial threshold in degrees (±4° from 90°)
pub const EQUATORIAL_THRESHOLD_DEGREES: f32 = 4.0;

/// Classify adhesion bond direction relative to split direction
/// 
/// This matches the GPU implementation exactly:
/// - Zone A: dot < 0 (pointing opposite to split direction)
/// - Zone B: dot > 0 and not equatorial (pointing same as split direction)
/// - Zone C: angle ≈ 90° from split direction (equatorial band ±4°)
/// 
/// # Arguments
/// * `bond_direction` - Direction of the adhesion bond (normalized)
/// * `split_direction` - Direction of cell division (normalized)
/// 
/// # Returns
/// The zone classification for this adhesion
pub fn classify_bond_direction(bond_direction: Vec3, split_direction: Vec3) -> AdhesionZone {
    let dot_product = bond_direction.dot(split_direction);
    let angle = dot_product.clamp(-1.0, 1.0).acos().to_degrees();
    let half_width = EQUATORIAL_THRESHOLD_DEGREES;
    let equatorial_angle = 90.0;
    
    // Check if within equatorial threshold (90° ± 4°)
    if (angle - equatorial_angle).abs() <= half_width {
        AdhesionZone::ZoneC // Equatorial band
    }
    // Classify based on which side relative to split direction
    else if dot_product > 0.0 {
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
        let bond_c = Vec3::new(1.0, 0.0, 0.0).normalize();
        assert_eq!(classify_bond_direction(bond_c, split_dir), AdhesionZone::ZoneC);
        
        // Test Zone C (equatorial - another perpendicular direction)
        let bond_c2 = Vec3::new(0.0, 0.0, 1.0).normalize();
        assert_eq!(classify_bond_direction(bond_c2, split_dir), AdhesionZone::ZoneC);
        
        // Test near-equatorial (should be Zone C)
        let bond_near_eq = Vec3::new(1.0, 0.07, 0.0).normalize(); // ~86° from Y (within 4° threshold)
        assert_eq!(classify_bond_direction(bond_near_eq, split_dir), AdhesionZone::ZoneC);
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

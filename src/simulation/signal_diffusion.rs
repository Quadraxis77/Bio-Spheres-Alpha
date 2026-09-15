//! Conservative, finite-speed diffusion in equal unit cell volumes.
use super::signal_system::SIGNAL_CHANNELS;

#[derive(Clone, Copy, Debug)]
pub struct DiffusionSettings {
    /// Symmetric quantity transfer rate per ordinary connection, per second.
    pub conductance: f32,
    /// First-order degradation rate per second.
    pub decay_rate: f32,
    /// Authored positive amplitudes represent quantity/second at scale one.
    pub production_scale: f32,
}
impl Default for DiffusionSettings {
    fn default() -> Self {
        Self {
            conductance: 0.5,
            decay_rate: 0.25,
            production_scale: 1.0,
        }
    }
}
impl DiffusionSettings {
    pub fn validate(self, dt: f32, max_degree: usize) -> Result<(), &'static str> {
        if !dt.is_finite()
            || dt <= 0.0
            || [self.conductance, self.decay_rate, self.production_scale]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(
                "diffusion rates must be finite and nonnegative; timestep must be positive",
            );
        }
        if dt * self.conductance * max_degree as f32 > 0.9 {
            return Err("diffusion export weight exceeds the 0.9 stability margin");
        }
        Ok(())
    }
}

/// CPU edge-scatter reference: transport reads exclusively from `current`.
/// The GPU implements the equivalent per-cell gather, with no neighboring writes.
pub fn step(
    current: &[[f32; SIGNAL_CHANNELS]],
    production: &[[f32; SIGNAL_CHANNELS]],
    edges: &[(usize, usize)],
    settings: DiffusionSettings,
    dt: f32,
) -> Result<Vec<[f32; SIGNAL_CHANNELS]>, &'static str> {
    if current.len() != production.len() {
        return Err("field lengths differ");
    }
    if current
        .iter()
        .chain(production)
        .flatten()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("concentrations and production must be finite and nonnegative");
    }
    let mut degree = vec![0; current.len()];
    for &(a, b) in edges {
        if a >= current.len() || b >= current.len() || a == b {
            return Err("invalid edge");
        }
        degree[a] += 1;
        degree[b] += 1;
    }
    settings.validate(dt, degree.into_iter().max().unwrap_or(0))?;
    let mut next = current.to_vec();
    for &(a, b) in edges {
        for ch in 0..SIGNAL_CHANNELS {
            let quantity = dt * settings.conductance * (current[a][ch] - current[b][ch]);
            next[a][ch] -= quantity;
            next[b][ch] += quantity;
        }
    }
    let retention = (-dt * settings.decay_rate).exp();
    for (value, source) in next.iter_mut().zip(production) {
        for ch in 0..SIGNAL_CHANNELS {
            value[ch] = value[ch] * retention + dt * settings.production_scale * source[ch];
        }
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    const DT: f32 = 1.0 / 15.0;
    fn total(field: &[[f32; 16]]) -> f32 {
        field.iter().map(|v| v[0]).sum()
    }
    #[test]
    fn conservation_cycles_parallel_edges_and_high_degree() {
        let settings = DiffusionSettings {
            decay_rate: 0.0,
            ..Default::default()
        };
        let mut edges: Vec<_> = (1..20).map(|i| (0, i)).collect();
        edges.extend([(1, 2), (2, 3), (3, 1), (0, 1)]);
        let mut field = vec![[0.0; 16]; 20];
        field[0][0] = 1000.0;
        let source = vec![[0.0; 16]; 20];
        for _ in 0..1000 {
            field = step(&field, &source, &edges, settings, DT).unwrap();
            assert!(field.iter().flatten().all(|v| *v >= 0.0));
            assert!((total(&field) - 1000.0).abs() < 0.02);
        }
        assert!(DiffusionSettings {
            conductance: 1.0,
            ..settings
        }
        .validate(DT, 20)
        .is_err());
    }
    #[test]
    fn gradient_superposition_and_shutdown() {
        let edges: Vec<_> = (0..9).map(|i| (i, i + 1)).collect();
        let settings = DiffusionSettings::default();
        let mut sources = vec![vec![[0.0; 16]; 10]; 3];
        sources[0][0][0] = 100.0;
        sources[1][9][0] = 100.0;
        sources[2][0][0] = 100.0;
        sources[2][9][0] = 100.0;
        let mut fields = vec![vec![[0.0; 16]; 10]; 3];
        for _ in 0..3000 {
            for i in 0..3 {
                fields[i] = step(&fields[i], &sources[i], &edges, settings, DT).unwrap();
            }
        }
        for i in 0..9 {
            assert!(fields[0][i][0] > fields[0][i + 1][0]);
        }
        for i in 0..10 {
            assert!((fields[2][i][0] - fields[0][i][0] - fields[1][i][0]).abs() < 0.01);
        }
        let before = total(&fields[2]);
        let mut faded = fields[2].clone();
        for _ in 0..300 {
            faded = step(&faded, &vec![[0.0; 16]; 10], &edges, settings, DT).unwrap();
        }
        assert!((total(&faded) - before * (-settings.decay_rate * DT * 300.0).exp()).abs() < 0.02);
    }
    #[test]
    fn transport_waits_for_next_tick_and_new_connections_use_retained_field() {
        let mut field = vec![[0.0; 16]; 3];
        let mut source = field.clone();
        source[0][0] = 150.0;
        let settings = DiffusionSettings {
            decay_rate: 0.0,
            ..Default::default()
        };
        field = step(&field, &source, &[(0, 1), (1, 2)], settings, DT).unwrap();
        assert!((field[0][0] - 10.0).abs() < 0.00001);
        assert_eq!(field[1][0], 0.0);
        source[0][0] = 0.0;
        field = step(&field, &source, &[], settings, DT).unwrap();
        assert!((field[0][0] - 10.0).abs() < 0.00001);
        field = step(&field, &source, &[(0, 2)], settings, DT).unwrap();
        assert!(field[2][0] > 0.0);
        assert_eq!(field[1][0], 0.0);
    }
}

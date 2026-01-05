//! Performance monitoring and metrics collection.
//!
//! Tracks FPS, frame times, CPU usage, and GPU information for the
//! performance monitor panel. Also detects and logs performance spikes.

use std::collections::VecDeque;
use sysinfo::System;

/// Number of frame time samples to keep for averaging.
const FRAME_TIME_SAMPLES: usize = 120;

/// How often to refresh system info (in seconds).
/// Set to 5 seconds to reduce CPU overhead from sysinfo calls.
const SYSTEM_REFRESH_INTERVAL: f32 = 5.0;

/// Performance spike detection configuration.
const SPIKE_THRESHOLD_MULTIPLIER: f32 = 2.5; // Spike if frame time > 2.5x average
const SPIKE_MIN_THRESHOLD_MS: f32 = 16.67; // Always consider >16.67ms (60fps) a potential spike
const SPIKE_COOLDOWN_SECONDS: f32 = 2.0; // Don't log spikes more often than every 2 seconds
const BASELINE_SAMPLES: usize = 30; // Number of samples to establish baseline

/// Performance spike detector and logger.
struct PerformanceSpikeDetector {
    /// Baseline frame times for spike detection (in seconds).
    baseline_times: VecDeque<f32>,
    /// Time since last spike was logged.
    time_since_last_spike: f32,
    /// Current baseline average (in seconds).
    baseline_average: f32,
}

impl PerformanceSpikeDetector {
    /// Create a new performance spike detector.
    fn new() -> Self {
        Self {
            baseline_times: VecDeque::with_capacity(BASELINE_SAMPLES),
            time_since_last_spike: SPIKE_COOLDOWN_SECONDS, // Start ready to log
            baseline_average: 0.0,
        }
    }
}

/// Performance metrics tracker.
pub struct PerformanceMetrics {
    /// Recent frame times for averaging (in seconds).
    frame_times: VecDeque<f32>,
    /// System info for CPU metrics.
    system: System,
    /// Time since last system refresh.
    time_since_refresh: f32,
    /// Time since last culling stats refresh.
    time_since_culling_refresh: f32,
    /// Cached CPU core count.
    cpu_core_count: usize,
    /// Cached per-core CPU usage (0-100%).
    cpu_usage_per_core: Vec<f32>,
    /// Cached total CPU usage (0-100%).
    cpu_usage_total: f32,
    /// Process memory usage in bytes.
    memory_used: u64,
    /// Total system memory in bytes.
    memory_total: u64,
    /// Culling statistics (total, visible, frustum culled, occluded)
    culling_stats: (u32, u32, u32, u32),
    /// Frame counter for periodic operations.
    frame_count: u64,
    /// Performance spike detection
    spike_detector: PerformanceSpikeDetector,
}

impl PerformanceMetrics {
    /// Create a new performance metrics tracker.
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_cpu_all();
        system.refresh_memory();
        
        let cpu_core_count = system.cpus().len();
        let memory_total = system.total_memory();
        
        Self {
            frame_times: VecDeque::with_capacity(FRAME_TIME_SAMPLES),
            system,
            time_since_refresh: 0.0,
            time_since_culling_refresh: 0.0,
            cpu_core_count,
            cpu_usage_per_core: vec![0.0; cpu_core_count],
            cpu_usage_total: 0.0,
            memory_used: 0,
            memory_total,
            culling_stats: (0, 0, 0, 0),
            frame_count: 0,
            spike_detector: PerformanceSpikeDetector::new(),
        }
    }
    
    /// Update metrics with a new frame time.
    pub fn update(&mut self, dt: f32) {
        // Track frame time
        if self.frame_times.len() >= FRAME_TIME_SAMPLES {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(dt);
        
        // Increment frame counter
        self.frame_count = self.frame_count.wrapping_add(1);
        
        // Track time for culling stats refresh
        self.time_since_culling_refresh += dt;
        
        // Refresh system info periodically
        self.time_since_refresh += dt;
        if self.time_since_refresh >= SYSTEM_REFRESH_INTERVAL {
            self.time_since_refresh = 0.0;
            self.refresh_system_info();
        }
        
        // Check for performance spikes (after other updates)
        self.check_for_performance_spike(dt);
    }
    
    /// Check if culling stats should be refreshed (once per second).
    pub fn should_refresh_culling_stats(&mut self) -> bool {
        if self.time_since_culling_refresh >= 1.0 {
            self.time_since_culling_refresh = 0.0;
            true
        } else {
            false
        }
    }
    
    /// Refresh CPU and memory usage from system.
    fn refresh_system_info(&mut self) {
        self.system.refresh_cpu_all();
        self.system.refresh_memory();
        
        let cpus = self.system.cpus();
        self.cpu_usage_per_core.clear();
        
        let mut total = 0.0;
        for cpu in cpus {
            let usage = cpu.cpu_usage();
            self.cpu_usage_per_core.push(usage);
            total += usage;
        }
        
        if !cpus.is_empty() {
            self.cpu_usage_total = total / cpus.len() as f32;
        }
        
        // Update memory usage
        self.memory_used = self.system.used_memory();
    }
    
    /// Get current FPS (frames per second).
    pub fn fps(&self) -> f32 {
        let avg_frame_time = self.average_frame_time();
        if avg_frame_time > 0.0 {
            1.0 / avg_frame_time
        } else {
            0.0
        }
    }
    
    /// Get average frame time in milliseconds.
    pub fn average_frame_time_ms(&self) -> f32 {
        self.average_frame_time() * 1000.0
    }
    
    /// Get average frame time in seconds.
    fn average_frame_time(&self) -> f32 {
        if self.frame_times.is_empty() {
            return 0.0;
        }
        self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32
    }
    
    /// Get min frame time in milliseconds.
    pub fn min_frame_time_ms(&self) -> f32 {
        self.frame_times.iter().cloned().fold(f32::INFINITY, f32::min) * 1000.0
    }
    
    /// Get max frame time in milliseconds.
    pub fn max_frame_time_ms(&self) -> f32 {
        self.frame_times.iter().cloned().fold(0.0, f32::max) * 1000.0
    }
    
    /// Get CPU core count.
    pub fn cpu_core_count(&self) -> usize {
        self.cpu_core_count
    }
    
    /// Get total CPU usage (0-100%).
    pub fn cpu_usage_total(&self) -> f32 {
        self.cpu_usage_total
    }
    
    /// Get per-core CPU usage.
    pub fn cpu_usage_per_core(&self) -> &[f32] {
        &self.cpu_usage_per_core
    }
    
    /// Get memory used in bytes.
    pub fn memory_used(&self) -> u64 {
        self.memory_used
    }
    
    /// Get total system memory in bytes.
    pub fn memory_total(&self) -> u64 {
        self.memory_total
    }
    
    /// Get memory usage as percentage (0-100).
    pub fn memory_usage_percent(&self) -> f32 {
        if self.memory_total > 0 {
            (self.memory_used as f64 / self.memory_total as f64 * 100.0) as f32
        } else {
            0.0
        }
    }
    
    /// Get frame time history for plotting.
    pub fn frame_time_history(&self) -> impl Iterator<Item = f32> + '_ {
        self.frame_times.iter().map(|&t| t * 1000.0)
    }
    
    /// Update culling statistics.
    pub fn set_culling_stats(&mut self, total: u32, visible: u32, frustum_culled: u32, occluded: u32) {
        self.culling_stats = (total, visible, frustum_culled, occluded);
    }
    
    /// Get culling statistics: (total, visible, frustum_culled, occluded).
    pub fn culling_stats(&self) -> (u32, u32, u32, u32) {
        self.culling_stats
    }
    
    /// Get the current frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
    
    /// Manually trigger a performance spike log for testing purposes.
    /// This bypasses the normal spike detection logic.
    pub fn log_test_spike(&self, frame_time_ms: f32, reason: &str) {
        log::warn!("🧪 TEST PERFORMANCE SPIKE - Frame: {:.2}ms, Reason: {}", frame_time_ms, reason);
        
        // Log system context
        log::warn!(
            "📊 System Context - CPU: {:.1}%, Memory: {:.1}% ({:.1}MB/{:.1}MB), FPS: {:.1}",
            self.cpu_usage_total(),
            self.memory_usage_percent(),
            self.memory_used() as f64 / 1024.0 / 1024.0,
            self.memory_total() as f64 / 1024.0 / 1024.0,
            self.fps()
        );
        
        // Log culling stats if available
        let (total, visible, frustum_culled, occluded) = self.culling_stats();
        if total > 0 {
            log::warn!(
                "🎯 Culling Stats - Total: {}, Visible: {}, Frustum Culled: {}, Occluded: {}",
                total, visible, frustum_culled, occluded
            );
        }
        
        log::warn!("🔬 This was a test spike to verify the logging system is working correctly");
    }
    
    /// Check for performance spikes and log them if detected.
    fn check_for_performance_spike(&mut self, dt: f32) {
        // Update cooldown timer
        self.spike_detector.time_since_last_spike += dt;
        
        // Build baseline from recent good frame times
        let dt_ms = dt * 1000.0;
        if dt_ms < SPIKE_MIN_THRESHOLD_MS * 1.5 {
            if self.spike_detector.baseline_times.len() >= BASELINE_SAMPLES {
                self.spike_detector.baseline_times.pop_front();
            }
            self.spike_detector.baseline_times.push_back(dt);
            
            // Recalculate baseline average
            if !self.spike_detector.baseline_times.is_empty() {
                self.spike_detector.baseline_average = self.spike_detector.baseline_times.iter().sum::<f32>() / self.spike_detector.baseline_times.len() as f32;
            }
        }
        
        // Skip spike detection if we don't have enough baseline data
        if self.spike_detector.baseline_times.len() < BASELINE_SAMPLES / 2 {
            return;
        }
        
        // Check if this frame time qualifies as a spike
        let baseline_ms = self.spike_detector.baseline_average * 1000.0;
        let threshold_ms = (baseline_ms * SPIKE_THRESHOLD_MULTIPLIER).max(SPIKE_MIN_THRESHOLD_MS);
        
        let is_spike = dt_ms > threshold_ms && self.spike_detector.time_since_last_spike >= SPIKE_COOLDOWN_SECONDS;
        
        if is_spike {
            self.log_performance_spike(dt_ms, baseline_ms, threshold_ms);
            self.spike_detector.time_since_last_spike = 0.0;
        }
    }
    
    /// Log detailed information about a performance spike.
    fn log_performance_spike(&self, spike_ms: f32, baseline_ms: f32, threshold_ms: f32) {
        let severity = if spike_ms > threshold_ms * 2.0 {
            "SEVERE"
        } else if spike_ms > threshold_ms * 1.5 {
            "MAJOR"
        } else {
            "MINOR"
        };
        
        log::warn!(
            "🚨 PERFORMANCE SPIKE DETECTED [{}] - Frame: {:.2}ms (baseline: {:.2}ms, threshold: {:.2}ms)",
            severity, spike_ms, baseline_ms, threshold_ms
        );
        
        // Log system context
        log::warn!(
            "📊 System Context - CPU: {:.1}%, Memory: {:.1}% ({:.1}MB/{:.1}MB), FPS: {:.1}",
            self.cpu_usage_total(),
            self.memory_usage_percent(),
            self.memory_used() as f64 / 1024.0 / 1024.0,
            self.memory_total() as f64 / 1024.0 / 1024.0,
            self.fps()
        );
        
        // Log culling stats if available
        let (total, visible, frustum_culled, occluded) = self.culling_stats();
        if total > 0 {
            log::warn!(
                "🎯 Culling Stats - Total: {}, Visible: {}, Frustum Culled: {}, Occluded: {}",
                total, visible, frustum_culled, occluded
            );
        }
        
        // Analyze potential causes
        self.analyze_spike_causes(spike_ms, baseline_ms);
    }
    
    /// Analyze and suggest potential causes for the performance spike.
    fn analyze_spike_causes(&self, spike_ms: f32, baseline_ms: f32) {
        let mut causes = Vec::new();
        
        // CPU-related causes
        if self.cpu_usage_total() > 80.0 {
            causes.push("High CPU usage (>80%)");
        }
        
        // Memory-related causes
        if self.memory_usage_percent() > 85.0 {
            causes.push("High memory usage (>85%)");
        }
        
        // Frame time analysis
        let spike_ratio = spike_ms / baseline_ms;
        if spike_ratio > 5.0 {
            causes.push("Extreme frame time spike (>5x baseline) - possible GC or blocking operation");
        } else if spike_ratio > 3.0 {
            causes.push("Large frame time spike (>3x baseline) - possible resource loading or heavy computation");
        }
        
        // GPU-related causes (inferred from culling stats)
        let (total, visible, _, _) = self.culling_stats();
        if total > 0 {
            let visibility_ratio = visible as f32 / total as f32;
            if visibility_ratio > 0.8 && total > 10000 {
                causes.push("High cell visibility with large cell count - possible GPU bottleneck");
            }
        }
        
        // Specific frame time thresholds
        if spike_ms > 100.0 {
            causes.push("Frame time >100ms - likely blocking I/O or major allocation");
        } else if spike_ms > 50.0 {
            causes.push("Frame time >50ms - possible shader compilation or large buffer update");
        } else if spike_ms > 33.33 {
            causes.push("Frame time >33ms - dropped below 30fps, check render complexity");
        }
        
        if causes.is_empty() {
            log::warn!("🔍 Spike Analysis - No obvious cause identified, may be system-level interference");
        } else {
            log::warn!("🔍 Potential Causes - {}", causes.join(", "));
        }
        
        // Suggestions based on spike characteristics and Bio-Spheres architecture
        if spike_ms > 50.0 {
            log::warn!("💡 Suggestions - Check for: shader compilation, large texture uploads, blocking file I/O, or garbage collection");
            log::warn!("🔬 Bio-Spheres Specific - May be caused by: GPU scene mode switch, large cell buffer updates, or occlusion query readback");
        } else if spike_ms > 25.0 {
            log::warn!("💡 Suggestions - Check for: buffer updates, draw call batching, or CPU-GPU synchronization stalls");
            log::warn!("🔬 Bio-Spheres Specific - May be caused by: cell instance buffer updates, adhesion line rendering, or UI layout changes");
        } else {
            log::warn!("💡 Suggestions - Check for: frame pacing issues, vsync conflicts, or minor resource contention");
            log::warn!("🔬 Bio-Spheres Specific - May be caused by: camera updates, gizmo rendering, or split ring calculations");
        }
        
        // Additional Bio-Spheres specific analysis
        self.analyze_biospheres_specific_causes(spike_ms, total, visible);
    }
    
    /// Analyze Bio-Spheres specific potential causes for performance spikes.
    fn analyze_biospheres_specific_causes(&self, spike_ms: f32, total_cells: u32, visible_cells: u32) {
        let mut bio_causes = Vec::new();
        
        // Cell count analysis
        if total_cells > 50000 {
            bio_causes.push("Very high cell count (>50k) - consider GPU compute mode");
        } else if total_cells > 10000 {
            bio_causes.push("High cell count (>10k) - may stress CPU physics or rendering");
        }
        
        // Visibility analysis
        if total_cells > 0 {
            let visibility_ratio = visible_cells as f32 / total_cells as f32;
            if visibility_ratio > 0.9 {
                bio_causes.push("Most cells visible (>90%) - culling not effective");
            } else if visibility_ratio > 0.7 {
                bio_causes.push("Many cells visible (>70%) - high render load");
            }
        }
        
        // Spike timing analysis for Bio-Spheres operations
        if spike_ms > 100.0 {
            bio_causes.push("Possible causes: Scene mode switch, genome loading, or large simulation reset");
        } else if spike_ms > 50.0 {
            bio_causes.push("Possible causes: Cell buffer reallocation, shader recompilation, or UI dock layout change");
        } else if spike_ms > 25.0 {
            bio_causes.push("Possible causes: Physics step with many collisions, adhesion line updates, or egui heavy frame");
        }
        
        if !bio_causes.is_empty() {
            log::warn!("🧬 Bio-Spheres Analysis - {}", bio_causes.join(", "));
        }
        
        // Render pipeline specific suggestions
        log::warn!("🎨 Render Pipeline Check - Verify: Clear pass, Skybox, Cell instancing, Adhesion lines, Split rings, Debug overlays, UI rendering");
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

//! Performance monitoring and metrics collection.
//!
//! Tracks FPS, frame times, CPU usage, and GPU information for the
//! performance monitor panel.

use std::collections::VecDeque;
use sysinfo::System;

/// Number of frame time samples to keep for averaging.
const FRAME_TIME_SAMPLES: usize = 120;

/// How often to refresh system info (in seconds).
const SYSTEM_REFRESH_INTERVAL: f32 = 1.0;

/// Performance metrics tracker.
pub struct PerformanceMetrics {
    /// Recent frame times for averaging (in seconds).
    frame_times: VecDeque<f32>,
    /// System info for CPU metrics.
    system: System,
    /// Time since last system refresh.
    time_since_refresh: f32,
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
            cpu_core_count,
            cpu_usage_per_core: vec![0.0; cpu_core_count],
            cpu_usage_total: 0.0,
            memory_used: 0,
            memory_total,
        }
    }
    
    /// Update metrics with a new frame time.
    pub fn update(&mut self, dt: f32) {
        // Track frame time
        if self.frame_times.len() >= FRAME_TIME_SAMPLES {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(dt);
        
        // Refresh system info periodically
        self.time_since_refresh += dt;
        if self.time_since_refresh >= SYSTEM_REFRESH_INTERVAL {
            self.time_since_refresh = 0.0;
            self.refresh_system_info();
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
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

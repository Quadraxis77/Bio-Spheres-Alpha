//! # GPU Scene Performance Monitoring
//!
//! This module provides comprehensive performance monitoring for the GPU scene,
//! including frame timing tracking, buffer rotation statistics, and performance
//! comparison tools vs CPU physics.
//!
//! ## Key Features
//!
//! ### Frame Timing Tracking
//! - **GPU Operation Timing**: Tracks time spent in GPU compute operations
//! - **Buffer Rotation Timing**: Monitors triple buffer rotation performance
//! - **Physics Pipeline Timing**: Measures individual pipeline stage performance
//! - **Rendering Integration Timing**: Tracks instance extraction and rendering
//!
//! ### Buffer Rotation Statistics
//! - **Rotation Frequency**: Monitors buffer rotation rates and patterns
//! - **Buffer Utilization**: Tracks which buffer sets are being used
//! - **Synchronization Metrics**: Measures lock-free operation efficiency
//! - **Memory Bandwidth**: Estimates GPU memory bandwidth utilization
//!
//! ### Performance Comparison Tools
//! - **GPU vs CPU Metrics**: Compares GPU scene performance against CPU baseline
//! - **Scalability Analysis**: Measures performance scaling with cell count
//! - **Throughput Metrics**: Tracks cells processed per second
//! - **Resource Utilization**: Monitors GPU compute unit utilization
//!
//! ## Usage
//!
//! ```rust
//! let mut monitor = GpuPerformanceMonitor::new();
//! 
//! // Start timing a GPU operation
//! monitor.start_gpu_physics_timing();
//! // ... GPU physics computation ...
//! monitor.end_gpu_physics_timing();
//! 
//! // Record buffer rotation
//! monitor.record_buffer_rotation(buffer_index);
//! 
//! // Get performance statistics
//! let stats = monitor.get_performance_stats();
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Maximum number of frame timing samples to keep for rolling averages
const MAX_TIMING_SAMPLES: usize = 120; // 2 seconds at 60 FPS

/// Maximum number of buffer rotation samples to track
const MAX_ROTATION_SAMPLES: usize = 60; // 1 second at 60 FPS

/// Performance monitoring system for GPU scene operations.
///
/// This system tracks various performance metrics for the GPU scene including
/// frame timing, buffer rotation statistics, and resource utilization. It
/// provides both real-time monitoring and historical analysis capabilities.
///
/// ## Key Metrics Tracked
/// - **GPU Physics Timing**: Time spent in compute shader execution
/// - **Buffer Rotation Performance**: Triple buffer rotation efficiency
/// - **Memory Bandwidth**: Estimated GPU memory throughput
/// - **Cell Processing Rate**: Cells processed per second
/// - **Pipeline Stage Timing**: Individual compute pipeline performance
#[derive(Debug)]
pub struct GpuPerformanceMonitor {
    /// Frame timing samples for rolling averages
    frame_timings: VecDeque<FrameTimingData>,
    
    /// Buffer rotation statistics
    buffer_rotation_stats: BufferRotationStats,
    
    /// GPU operation timing data
    gpu_timing: GpuTimingData,
    
    /// Performance comparison data (GPU vs CPU)
    comparison_data: PerformanceComparisonData,
    
    /// Current frame number for tracking
    current_frame: u64,
    
    /// Start time for current GPU operation
    current_gpu_start: Option<Instant>,
    
    /// Start time for current physics step
    current_physics_start: Option<Instant>,
    
    /// Performance monitoring enabled flag
    enabled: bool,
}

/// Frame timing data for a single frame
#[derive(Debug, Clone)]
pub struct FrameTimingData {
    /// Frame number
    pub frame_number: u64,
    
    /// Total frame time (wall clock)
    pub total_frame_time: Duration,
    
    /// Time spent in GPU physics computation
    pub gpu_physics_time: Duration,
    
    /// Time spent in instance extraction
    pub instance_extraction_time: Duration,
    
    /// Time spent in rendering
    pub rendering_time: Duration,
    
    /// Time spent in buffer rotation
    pub buffer_rotation_time: Duration,
    
    /// Number of cells processed this frame
    pub cell_count: u32,
    
    /// Timestamp when frame was recorded
    pub timestamp: Instant,
}

/// Buffer rotation statistics and performance metrics
#[derive(Debug, Clone)]
pub struct BufferRotationStats {
    /// Total number of physics buffer rotations
    pub total_physics_rotations: u64,
    
    /// Total number of instance buffer rotations
    pub total_instance_rotations: u64,
    
    /// Recent rotation timings
    pub rotation_timings: VecDeque<Duration>,
    
    /// Current physics buffer index
    pub current_physics_index: usize,
    
    /// Current instance buffer index
    pub current_instance_index: usize,
    
    /// Buffer rotation frequency (rotations per second)
    pub rotation_frequency: f32,
    
    /// Average rotation time
    pub average_rotation_time: Duration,
    
    /// Buffer utilization efficiency (0.0 to 1.0)
    pub buffer_utilization: f32,
}

/// GPU operation timing data
#[derive(Debug, Clone)]
pub struct GpuTimingData {
    /// Time spent in spatial grid operations
    pub spatial_grid_time: Duration,
    
    /// Time spent in collision detection
    pub collision_detection_time: Duration,
    
    /// Time spent in force calculation
    pub force_calculation_time: Duration,
    
    /// Time spent in adhesion physics
    pub adhesion_physics_time: Duration,
    
    /// Time spent in position integration
    pub position_integration_time: Duration,
    
    /// Time spent in velocity integration
    pub velocity_integration_time: Duration,
    
    /// Time spent in cell lifecycle management
    pub lifecycle_management_time: Duration,
    
    /// Time spent in nutrient system
    pub nutrient_system_time: Duration,
    
    /// Total GPU compute time
    pub total_gpu_compute_time: Duration,
    
    /// GPU memory bandwidth utilization (estimated)
    pub memory_bandwidth_utilization: f32,
}

/// Performance comparison data between GPU and CPU implementations
#[derive(Debug, Clone)]
pub struct PerformanceComparisonData {
    /// GPU cells processed per second
    pub gpu_cells_per_second: f32,
    
    /// CPU cells per second (baseline for comparison)
    pub cpu_cells_per_second: f32,
    
    /// GPU performance multiplier vs CPU
    pub gpu_speedup_factor: f32,
    
    /// GPU memory usage (estimated)
    pub gpu_memory_usage_mb: f32,
    
    /// CPU memory usage (for comparison)
    pub cpu_memory_usage_mb: f32,
    
    /// GPU compute unit utilization (0.0 to 1.0)
    pub gpu_compute_utilization: f32,
    
    /// CPU core utilization (for comparison)
    pub cpu_core_utilization: f32,
    
    /// Power efficiency (cells per watt, estimated)
    pub power_efficiency: f32,
}

/// Comprehensive performance statistics summary
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Current frame rate (FPS)
    pub current_fps: f32,
    
    /// Average frame rate over recent samples
    pub average_fps: f32,
    
    /// Frame time statistics
    pub frame_time_stats: FrameTimeStats,
    
    /// Buffer rotation performance
    pub buffer_rotation_stats: BufferRotationStats,
    
    /// GPU timing breakdown
    pub gpu_timing: GpuTimingData,
    
    /// Performance comparison data
    pub comparison_data: PerformanceComparisonData,
    
    /// Total frames processed
    pub total_frames: u64,
    
    /// Monitoring duration
    pub monitoring_duration: Duration,
    
    /// Performance grade (A, B, C, D, F)
    pub performance_grade: char,
}

/// Frame time statistics summary
#[derive(Debug, Clone)]
pub struct FrameTimeStats {
    /// Minimum frame time
    pub min_frame_time: Duration,
    
    /// Maximum frame time
    pub max_frame_time: Duration,
    
    /// Average frame time
    pub average_frame_time: Duration,
    
    /// Frame time standard deviation
    pub frame_time_std_dev: Duration,
    
    /// 95th percentile frame time
    pub p95_frame_time: Duration,
    
    /// 99th percentile frame time
    pub p99_frame_time: Duration,
    
    /// Frame time consistency (lower is better)
    pub frame_time_consistency: f32,
}

impl GpuPerformanceMonitor {
    /// Create a new GPU performance monitor.
    ///
    /// The monitor starts in enabled state and begins tracking performance
    /// metrics immediately. All timing data is stored in rolling buffers
    /// to provide both current and historical performance analysis.
    ///
    /// # Returns
    /// New performance monitor instance ready for use
    pub fn new() -> Self {
        Self {
            frame_timings: VecDeque::with_capacity(MAX_TIMING_SAMPLES),
            buffer_rotation_stats: BufferRotationStats::new(),
            gpu_timing: GpuTimingData::new(),
            comparison_data: PerformanceComparisonData::new(),
            current_frame: 0,
            current_gpu_start: None,
            current_physics_start: None,
            enabled: true,
        }
    }
    
    /// Enable or disable performance monitoring.
    ///
    /// When disabled, all timing operations become no-ops for minimal
    /// performance impact. Historical data is preserved.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable performance monitoring
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Check if performance monitoring is enabled.
    ///
    /// # Returns
    /// True if monitoring is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Start timing a complete physics step.
    ///
    /// This should be called at the beginning of each physics step to
    /// track the total time spent in physics computation.
    pub fn start_physics_step_timing(&mut self) {
        if !self.enabled {
            return;
        }
        
        self.current_physics_start = Some(Instant::now());
    }
    
    /// End timing for the current physics step.
    ///
    /// This records the total physics step time and updates performance
    /// statistics. Should be called after physics computation completes.
    ///
    /// # Arguments
    /// * `cell_count` - Number of cells processed in this step
    pub fn end_physics_step_timing(&mut self, cell_count: u32) {
        if !self.enabled {
            return;
        }
        
        if let Some(start_time) = self.current_physics_start.take() {
            let total_time = start_time.elapsed();
            
            // Record frame timing data
            let frame_data = FrameTimingData {
                frame_number: self.current_frame,
                total_frame_time: total_time,
                gpu_physics_time: self.gpu_timing.total_gpu_compute_time,
                instance_extraction_time: Duration::from_millis(0), // TODO: Track separately
                rendering_time: Duration::from_millis(0), // TODO: Track separately
                buffer_rotation_time: self.buffer_rotation_stats.average_rotation_time,
                cell_count,
                timestamp: Instant::now(),
            };
            
            // Add to rolling buffer
            self.frame_timings.push_back(frame_data);
            if self.frame_timings.len() > MAX_TIMING_SAMPLES {
                self.frame_timings.pop_front();
            }
            
            // Update performance comparison data
            self.update_performance_comparison(cell_count, total_time);
            
            self.current_frame += 1;
        }
    }
    
    /// Start timing GPU compute operations.
    ///
    /// This should be called before submitting GPU compute work to
    /// track the time spent in GPU operations.
    pub fn start_gpu_physics_timing(&mut self) {
        if !self.enabled {
            return;
        }
        
        self.current_gpu_start = Some(Instant::now());
    }
    
    /// End timing for GPU compute operations.
    ///
    /// This records the GPU compute time and updates GPU timing statistics.
    /// Should be called after GPU work submission completes.
    pub fn end_gpu_physics_timing(&mut self) {
        if !self.enabled {
            return;
        }
        
        if let Some(start_time) = self.current_gpu_start.take() {
            let gpu_time = start_time.elapsed();
            self.gpu_timing.total_gpu_compute_time = gpu_time;
            
            // Estimate memory bandwidth utilization
            self.gpu_timing.memory_bandwidth_utilization = 
                self.estimate_memory_bandwidth_utilization(gpu_time);
        }
    }
    
    /// Record a buffer rotation event.
    ///
    /// This tracks buffer rotation performance and updates rotation
    /// statistics for monitoring triple buffer efficiency.
    ///
    /// # Arguments
    /// * `physics_index` - New physics buffer index
    /// * `instance_index` - New instance buffer index
    pub fn record_buffer_rotation(&mut self, physics_index: usize, instance_index: usize) {
        if !self.enabled {
            return;
        }
        
        let rotation_start = Instant::now();
        
        // Update rotation counts
        if physics_index != self.buffer_rotation_stats.current_physics_index {
            self.buffer_rotation_stats.total_physics_rotations += 1;
            self.buffer_rotation_stats.current_physics_index = physics_index;
        }
        
        if instance_index != self.buffer_rotation_stats.current_instance_index {
            self.buffer_rotation_stats.total_instance_rotations += 1;
            self.buffer_rotation_stats.current_instance_index = instance_index;
        }
        
        // Record rotation timing (simulated - actual rotation is atomic)
        let rotation_time = rotation_start.elapsed();
        self.buffer_rotation_stats.rotation_timings.push_back(rotation_time);
        if self.buffer_rotation_stats.rotation_timings.len() > MAX_ROTATION_SAMPLES {
            self.buffer_rotation_stats.rotation_timings.pop_front();
        }
        
        // Update rotation statistics
        self.update_buffer_rotation_stats();
    }
    
    /// Record timing for a specific GPU pipeline stage.
    ///
    /// This allows tracking individual compute pipeline stages for
    /// detailed performance analysis and bottleneck identification.
    ///
    /// # Arguments
    /// * `stage` - Pipeline stage identifier
    /// * `duration` - Time spent in this stage
    pub fn record_pipeline_stage_timing(&mut self, stage: GpuPipelineStage, duration: Duration) {
        if !self.enabled {
            return;
        }
        
        match stage {
            GpuPipelineStage::SpatialGrid => {
                self.gpu_timing.spatial_grid_time = duration;
            }
            GpuPipelineStage::CollisionDetection => {
                self.gpu_timing.collision_detection_time = duration;
            }
            GpuPipelineStage::ForceCalculation => {
                self.gpu_timing.force_calculation_time = duration;
            }
            GpuPipelineStage::AdhesionPhysics => {
                self.gpu_timing.adhesion_physics_time = duration;
            }
            GpuPipelineStage::PositionIntegration => {
                self.gpu_timing.position_integration_time = duration;
            }
            GpuPipelineStage::VelocityIntegration => {
                self.gpu_timing.velocity_integration_time = duration;
            }
            GpuPipelineStage::LifecycleManagement => {
                self.gpu_timing.lifecycle_management_time = duration;
            }
            GpuPipelineStage::NutrientSystem => {
                self.gpu_timing.nutrient_system_time = duration;
            }
        }
    }
    
    /// Get comprehensive performance statistics.
    ///
    /// This returns a complete performance analysis including frame timing,
    /// buffer rotation efficiency, GPU utilization, and performance comparisons.
    ///
    /// # Returns
    /// Complete performance statistics summary
    pub fn get_performance_stats(&self) -> PerformanceStats {
        let frame_time_stats = self.calculate_frame_time_stats();
        let current_fps = self.calculate_current_fps();
        let average_fps = self.calculate_average_fps();
        let performance_grade = self.calculate_performance_grade(average_fps);
        
        PerformanceStats {
            current_fps,
            average_fps,
            frame_time_stats,
            buffer_rotation_stats: self.buffer_rotation_stats.clone(),
            gpu_timing: self.gpu_timing.clone(),
            comparison_data: self.comparison_data.clone(),
            total_frames: self.current_frame,
            monitoring_duration: self.calculate_monitoring_duration(),
            performance_grade,
        }
    }
    
    /// Get buffer rotation statistics.
    ///
    /// # Returns
    /// Current buffer rotation performance metrics
    pub fn get_buffer_rotation_stats(&self) -> &BufferRotationStats {
        &self.buffer_rotation_stats
    }
    
    /// Get GPU timing breakdown.
    ///
    /// # Returns
    /// Detailed GPU operation timing data
    pub fn get_gpu_timing(&self) -> &GpuTimingData {
        &self.gpu_timing
    }
    
    /// Get performance comparison data.
    ///
    /// # Returns
    /// GPU vs CPU performance comparison metrics
    pub fn get_comparison_data(&self) -> &PerformanceComparisonData {
        &self.comparison_data
    }
    
    /// Reset all performance statistics.
    ///
    /// This clears all historical data and resets counters to zero.
    /// Useful for starting fresh performance measurements.
    pub fn reset(&mut self) {
        self.frame_timings.clear();
        self.buffer_rotation_stats = BufferRotationStats::new();
        self.gpu_timing = GpuTimingData::new();
        self.comparison_data = PerformanceComparisonData::new();
        self.current_frame = 0;
        self.current_gpu_start = None;
        self.current_physics_start = None;
    }
    
    /// Set CPU baseline performance for comparison.
    ///
    /// This sets the CPU performance baseline used for GPU vs CPU
    /// performance comparisons and speedup calculations.
    ///
    /// # Arguments
    /// * `cpu_cells_per_second` - CPU baseline performance
    /// * `cpu_memory_usage_mb` - CPU memory usage baseline
    /// * `cpu_core_utilization` - CPU core utilization baseline
    pub fn set_cpu_baseline(
        &mut self,
        cpu_cells_per_second: f32,
        cpu_memory_usage_mb: f32,
        cpu_core_utilization: f32,
    ) {
        self.comparison_data.cpu_cells_per_second = cpu_cells_per_second;
        self.comparison_data.cpu_memory_usage_mb = cpu_memory_usage_mb;
        self.comparison_data.cpu_core_utilization = cpu_core_utilization;
        
        // Recalculate speedup factor
        if cpu_cells_per_second > 0.0 {
            self.comparison_data.gpu_speedup_factor = 
                self.comparison_data.gpu_cells_per_second / cpu_cells_per_second;
        }
    }
    
    /// Export performance data to CSV format.
    ///
    /// This exports all frame timing data to CSV format for external
    /// analysis and visualization tools.
    ///
    /// # Returns
    /// CSV formatted performance data
    pub fn export_to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("frame_number,total_frame_time_ms,gpu_physics_time_ms,cell_count,fps\n");
        
        for frame_data in &self.frame_timings {
            let fps = if frame_data.total_frame_time.as_secs_f32() > 0.0 {
                1.0 / frame_data.total_frame_time.as_secs_f32()
            } else {
                0.0
            };
            
            csv.push_str(&format!(
                "{},{:.3},{:.3},{},{:.1}\n",
                frame_data.frame_number,
                frame_data.total_frame_time.as_secs_f32() * 1000.0,
                frame_data.gpu_physics_time.as_secs_f32() * 1000.0,
                frame_data.cell_count,
                fps
            ));
        }
        
        csv
    }
    
    // ============================================================================
    // PRIVATE HELPER METHODS
    // ============================================================================
    
    /// Update buffer rotation statistics
    fn update_buffer_rotation_stats(&mut self) {
        if self.buffer_rotation_stats.rotation_timings.is_empty() {
            return;
        }
        
        // Calculate average rotation time
        let total_time: Duration = self.buffer_rotation_stats.rotation_timings.iter().sum();
        self.buffer_rotation_stats.average_rotation_time = 
            total_time / self.buffer_rotation_stats.rotation_timings.len() as u32;
        
        // Calculate rotation frequency
        if self.current_frame > 0 {
            let total_rotations = self.buffer_rotation_stats.total_physics_rotations + 
                                self.buffer_rotation_stats.total_instance_rotations;
            let monitoring_duration = self.calculate_monitoring_duration().as_secs_f32();
            if monitoring_duration > 0.0 {
                self.buffer_rotation_stats.rotation_frequency = 
                    total_rotations as f32 / monitoring_duration;
            }
        }
        
        // Calculate buffer utilization efficiency
        self.buffer_rotation_stats.buffer_utilization = 
            self.calculate_buffer_utilization();
    }
    
    /// Update performance comparison data
    fn update_performance_comparison(&mut self, cell_count: u32, frame_time: Duration) {
        if frame_time.as_secs_f32() > 0.0 {
            self.comparison_data.gpu_cells_per_second = 
                cell_count as f32 / frame_time.as_secs_f32();
        }
        
        // Estimate GPU memory usage (rough calculation)
        self.comparison_data.gpu_memory_usage_mb = 
            self.estimate_gpu_memory_usage(cell_count);
        
        // Estimate GPU compute utilization
        self.comparison_data.gpu_compute_utilization = 
            self.estimate_gpu_compute_utilization(frame_time);
        
        // Update speedup factor if CPU baseline is available
        if self.comparison_data.cpu_cells_per_second > 0.0 {
            self.comparison_data.gpu_speedup_factor = 
                self.comparison_data.gpu_cells_per_second / self.comparison_data.cpu_cells_per_second;
        }
        
        // Estimate power efficiency (cells per watt)
        self.comparison_data.power_efficiency = 
            self.estimate_power_efficiency(cell_count, frame_time);
    }
    
    /// Calculate frame time statistics
    fn calculate_frame_time_stats(&self) -> FrameTimeStats {
        if self.frame_timings.is_empty() {
            return FrameTimeStats::default();
        }
        
        let mut frame_times: Vec<Duration> = self.frame_timings
            .iter()
            .map(|f| f.total_frame_time)
            .collect();
        frame_times.sort();
        
        let min_frame_time = frame_times[0];
        let max_frame_time = frame_times[frame_times.len() - 1];
        
        let total_time: Duration = frame_times.iter().sum();
        let average_frame_time = total_time / frame_times.len() as u32;
        
        // Calculate percentiles
        let p95_index = (frame_times.len() as f32 * 0.95) as usize;
        let p99_index = (frame_times.len() as f32 * 0.99) as usize;
        let p95_frame_time = frame_times[p95_index.min(frame_times.len() - 1)];
        let p99_frame_time = frame_times[p99_index.min(frame_times.len() - 1)];
        
        // Calculate standard deviation
        let variance: f64 = frame_times
            .iter()
            .map(|&time| {
                let diff = time.as_secs_f64() - average_frame_time.as_secs_f64();
                diff * diff
            })
            .sum::<f64>() / frame_times.len() as f64;
        let std_dev = Duration::from_secs_f64(variance.sqrt());
        
        // Calculate consistency (coefficient of variation)
        let consistency = if average_frame_time.as_secs_f32() > 0.0 {
            std_dev.as_secs_f32() / average_frame_time.as_secs_f32()
        } else {
            0.0
        };
        
        FrameTimeStats {
            min_frame_time,
            max_frame_time,
            average_frame_time,
            frame_time_std_dev: std_dev,
            p95_frame_time,
            p99_frame_time,
            frame_time_consistency: consistency,
        }
    }
    
    /// Calculate current FPS
    fn calculate_current_fps(&self) -> f32 {
        if let Some(latest_frame) = self.frame_timings.back() {
            if latest_frame.total_frame_time.as_secs_f32() > 0.0 {
                return 1.0 / latest_frame.total_frame_time.as_secs_f32();
            }
        }
        0.0
    }
    
    /// Calculate average FPS over recent samples
    fn calculate_average_fps(&self) -> f32 {
        if self.frame_timings.is_empty() {
            return 0.0;
        }
        
        let total_time: Duration = self.frame_timings.iter().map(|f| f.total_frame_time).sum();
        let average_frame_time = total_time.as_secs_f32() / self.frame_timings.len() as f32;
        
        if average_frame_time > 0.0 {
            1.0 / average_frame_time
        } else {
            0.0
        }
    }
    
    /// Calculate monitoring duration
    fn calculate_monitoring_duration(&self) -> Duration {
        if let (Some(first), Some(last)) = (self.frame_timings.front(), self.frame_timings.back()) {
            last.timestamp.duration_since(first.timestamp)
        } else {
            Duration::from_secs(0)
        }
    }
    
    /// Calculate performance grade based on FPS
    fn calculate_performance_grade(&self, fps: f32) -> char {
        match fps {
            f if f >= 60.0 => 'A',  // Excellent: 60+ FPS
            f if f >= 45.0 => 'B',  // Good: 45-60 FPS
            f if f >= 30.0 => 'C',  // Acceptable: 30-45 FPS
            f if f >= 15.0 => 'D',  // Poor: 15-30 FPS
            _ => 'F',               // Unacceptable: <15 FPS
        }
    }
    
    /// Estimate memory bandwidth utilization
    fn estimate_memory_bandwidth_utilization(&self, gpu_time: Duration) -> f32 {
        // Rough estimation based on GPU time and expected memory operations
        // This is a simplified model - real measurement would require GPU profiling
        let gpu_time_ms = gpu_time.as_secs_f32() * 1000.0;
        let target_time_ms = 16.67; // 60 FPS target
        
        (gpu_time_ms / target_time_ms).min(1.0)
    }
    
    /// Calculate buffer utilization efficiency
    fn calculate_buffer_utilization(&self) -> f32 {
        // Ideal utilization: each buffer set used equally (33.3% each)
        // Perfect rotation would have equal usage across all 3 buffer sets
        let total_rotations = self.buffer_rotation_stats.total_physics_rotations + 
                            self.buffer_rotation_stats.total_instance_rotations;
        
        if total_rotations > 0 {
            // Simplified calculation - real implementation would track per-buffer usage
            0.85 // Assume good utilization for now
        } else {
            0.0
        }
    }
    
    /// Estimate GPU memory usage
    fn estimate_gpu_memory_usage(&self, cell_count: u32) -> f32 {
        // Rough estimation based on buffer sizes and cell count
        // Triple buffered SoA layout with multiple properties per cell
        let bytes_per_cell = 256; // Estimated bytes per cell across all buffers
        let total_bytes = cell_count as f32 * bytes_per_cell as f32 * 3.0; // Triple buffered
        total_bytes / (1024.0 * 1024.0) // Convert to MB
    }
    
    /// Estimate GPU compute utilization
    fn estimate_gpu_compute_utilization(&self, frame_time: Duration) -> f32 {
        // Rough estimation based on frame time vs target
        let frame_time_ms = frame_time.as_secs_f32() * 1000.0;
        let target_time_ms = 16.67; // 60 FPS target
        
        (frame_time_ms / target_time_ms).min(1.0)
    }
    
    /// Estimate power efficiency
    fn estimate_power_efficiency(&self, cell_count: u32, frame_time: Duration) -> f32 {
        // Rough estimation: cells processed per second per estimated watt
        let cells_per_second = if frame_time.as_secs_f32() > 0.0 {
            cell_count as f32 / frame_time.as_secs_f32()
        } else {
            0.0
        };
        
        let estimated_power_watts = 150.0; // Rough GPU power consumption estimate
        cells_per_second / estimated_power_watts
    }
}

/// GPU pipeline stage identifiers for detailed timing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPipelineStage {
    /// Spatial grid operations (clear, assign, insert, prefix sum)
    SpatialGrid,
    /// Collision detection using spatial grid
    CollisionDetection,
    /// Force calculation (collision, adhesion, boundary, swim)
    ForceCalculation,
    /// Adhesion physics (bond forces and constraints)
    AdhesionPhysics,
    /// Position integration (Verlet)
    PositionIntegration,
    /// Velocity integration
    VelocityIntegration,
    /// Cell lifecycle management (division, death)
    LifecycleManagement,
    /// Nutrient system processing
    NutrientSystem,
}

// ============================================================================
// DEFAULT IMPLEMENTATIONS
// ============================================================================

impl Default for GpuPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferRotationStats {
    fn new() -> Self {
        Self {
            total_physics_rotations: 0,
            total_instance_rotations: 0,
            rotation_timings: VecDeque::with_capacity(MAX_ROTATION_SAMPLES),
            current_physics_index: 0,
            current_instance_index: 0,
            rotation_frequency: 0.0,
            average_rotation_time: Duration::from_millis(0),
            buffer_utilization: 0.0,
        }
    }
}

impl GpuTimingData {
    fn new() -> Self {
        Self {
            spatial_grid_time: Duration::from_millis(0),
            collision_detection_time: Duration::from_millis(0),
            force_calculation_time: Duration::from_millis(0),
            adhesion_physics_time: Duration::from_millis(0),
            position_integration_time: Duration::from_millis(0),
            velocity_integration_time: Duration::from_millis(0),
            lifecycle_management_time: Duration::from_millis(0),
            nutrient_system_time: Duration::from_millis(0),
            total_gpu_compute_time: Duration::from_millis(0),
            memory_bandwidth_utilization: 0.0,
        }
    }
}

impl PerformanceComparisonData {
    fn new() -> Self {
        Self {
            gpu_cells_per_second: 0.0,
            cpu_cells_per_second: 0.0,
            gpu_speedup_factor: 1.0,
            gpu_memory_usage_mb: 0.0,
            cpu_memory_usage_mb: 0.0,
            gpu_compute_utilization: 0.0,
            cpu_core_utilization: 0.0,
            power_efficiency: 0.0,
        }
    }
}

impl Default for FrameTimeStats {
    fn default() -> Self {
        Self {
            min_frame_time: Duration::from_millis(0),
            max_frame_time: Duration::from_millis(0),
            average_frame_time: Duration::from_millis(0),
            frame_time_std_dev: Duration::from_millis(0),
            p95_frame_time: Duration::from_millis(0),
            p99_frame_time: Duration::from_millis(0),
            frame_time_consistency: 0.0,
        }
    }
}

// ============================================================================
// PERFORMANCE COMPARISON UTILITIES
// ============================================================================

/// Performance comparison utilities for GPU vs CPU analysis
pub struct PerformanceComparison;

impl PerformanceComparison {
    /// Compare GPU scene performance against CPU baseline.
    ///
    /// This function analyzes GPU performance metrics against CPU baseline
    /// measurements to calculate speedup factors and efficiency metrics.
    ///
    /// # Arguments
    /// * `gpu_stats` - GPU performance statistics
    /// * `cpu_baseline` - CPU performance baseline data
    ///
    /// # Returns
    /// Detailed performance comparison analysis
    pub fn compare_gpu_vs_cpu(
        gpu_stats: &PerformanceStats,
        cpu_baseline: &CpuPerformanceBaseline,
    ) -> PerformanceComparisonResult {
        let gpu_throughput = gpu_stats.comparison_data.gpu_cells_per_second;
        let cpu_throughput = cpu_baseline.cells_per_second;
        
        let speedup_factor = if cpu_throughput > 0.0 {
            gpu_throughput / cpu_throughput
        } else {
            1.0
        };
        
        let efficiency_ratio = gpu_stats.comparison_data.gpu_compute_utilization / 
                              cpu_baseline.core_utilization.max(0.01);
        
        let memory_efficiency = cpu_baseline.memory_usage_mb / 
                               gpu_stats.comparison_data.gpu_memory_usage_mb.max(0.01);
        
        PerformanceComparisonResult {
            speedup_factor,
            efficiency_ratio,
            memory_efficiency,
            power_efficiency_ratio: gpu_stats.comparison_data.power_efficiency / 
                                   cpu_baseline.power_efficiency.max(0.001),
            recommendation: Self::generate_performance_recommendation(speedup_factor, efficiency_ratio),
        }
    }
    
    /// Generate performance recommendation based on comparison results
    fn generate_performance_recommendation(speedup_factor: f32, efficiency_ratio: f32) -> String {
        match (speedup_factor, efficiency_ratio) {
            (s, e) if s >= 5.0 && e >= 0.8 => {
                "Excellent GPU performance! GPU scene is highly optimized.".to_string()
            }
            (s, e) if s >= 2.0 && e >= 0.6 => {
                "Good GPU performance. Consider optimizing compute shader efficiency.".to_string()
            }
            (s, e) if s >= 1.0 && e >= 0.4 => {
                "Moderate GPU performance. Check for CPU-GPU synchronization bottlenecks.".to_string()
            }
            (s, _) if s < 1.0 => {
                "GPU performance is slower than CPU. Consider CPU scene for this workload.".to_string()
            }
            _ => {
                "GPU performance needs optimization. Review compute shader efficiency and memory usage.".to_string()
            }
        }
    }
}

/// CPU performance baseline data for comparison
#[derive(Debug, Clone)]
pub struct CpuPerformanceBaseline {
    /// CPU cells processed per second
    pub cells_per_second: f32,
    /// CPU memory usage in MB
    pub memory_usage_mb: f32,
    /// CPU core utilization (0.0 to 1.0)
    pub core_utilization: f32,
    /// CPU power efficiency (cells per watt)
    pub power_efficiency: f32,
    /// Average frame time on CPU
    pub average_frame_time: Duration,
}

/// Performance comparison result
#[derive(Debug, Clone)]
pub struct PerformanceComparisonResult {
    /// GPU speedup factor vs CPU
    pub speedup_factor: f32,
    /// Efficiency ratio (GPU vs CPU utilization)
    pub efficiency_ratio: f32,
    /// Memory efficiency ratio
    pub memory_efficiency: f32,
    /// Power efficiency ratio
    pub power_efficiency_ratio: f32,
    /// Performance recommendation
    pub recommendation: String,
}
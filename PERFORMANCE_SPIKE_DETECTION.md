# Performance Spike Detection System

## Overview

Bio-Spheres now includes an intelligent performance spike detection system that automatically monitors frame times and logs detailed information when performance issues occur. This helps identify and diagnose performance problems during development and testing.

## How It Works

### Automatic Detection
- **Baseline Tracking**: The system maintains a rolling average of recent "good" frame times (30 samples)
- **Spike Threshold**: A frame is considered a spike if it exceeds 2.5x the baseline average OR 16.67ms (60fps threshold)
- **Cooldown Period**: Spikes are only logged once every 2 seconds to avoid spam
- **Smart Filtering**: Only stable frame times are used to build the baseline, excluding spikes

### Detection Configuration
```rust
const SPIKE_THRESHOLD_MULTIPLIER: f32 = 2.5; // Spike if frame time > 2.5x average
const SPIKE_MIN_THRESHOLD_MS: f32 = 16.67;   // Always consider >16.67ms (60fps) a potential spike
const SPIKE_COOLDOWN_SECONDS: f32 = 2.0;     // Don't log spikes more often than every 2 seconds
const BASELINE_SAMPLES: usize = 30;          // Number of samples to establish baseline
```

## Spike Severity Levels

- **MINOR**: Frame time > threshold but < 1.5x threshold
- **MAJOR**: Frame time > 1.5x threshold but < 2x threshold  
- **SEVERE**: Frame time > 2x threshold

## Information Logged

When a spike is detected, the system logs:

### 1. Basic Spike Info
```
🚨 PERFORMANCE SPIKE DETECTED [MAJOR] - Frame: 45.23ms (baseline: 16.67ms, threshold: 25.00ms)
```

### 2. System Context
```
📊 System Context - CPU: 65.2%, Memory: 42.1% (2.1GB/8.0GB), FPS: 22.1
```

### 3. Culling Statistics (if available)
```
🎯 Culling Stats - Total: 15000, Visible: 8500, Frustum Culled: 4200, Occluded: 2300
```

### 4. Potential Causes Analysis
```
🔍 Potential Causes - Large frame time spike (>3x baseline) - possible resource loading or heavy computation
```

### 5. General Suggestions
```
💡 Suggestions - Check for: buffer updates, draw call batching, or CPU-GPU synchronization stalls
```

### 6. Bio-Spheres Specific Analysis
```
🔬 Bio-Spheres Specific - May be caused by: cell instance buffer updates, adhesion line rendering, or UI layout changes
🧬 Bio-Spheres Analysis - High cell count (>10k) - may stress CPU physics or rendering
🎨 Render Pipeline Check - Verify: Clear pass, Skybox, Cell instancing, Adhesion lines, Split rings, Debug overlays, UI rendering
```

## Bio-Spheres Specific Causes

The system includes specialized analysis for Bio-Spheres operations:

### Cell Count Analysis
- **>50k cells**: Very high count, suggests GPU compute mode
- **>10k cells**: High count, may stress CPU physics or rendering

### Visibility Analysis
- **>90% visible**: Culling not effective
- **>70% visible**: High render load

### Timing-Based Analysis
- **>100ms**: Scene mode switch, genome loading, simulation reset
- **>50ms**: Cell buffer reallocation, shader recompilation, UI dock changes
- **>25ms**: Physics with many collisions, adhesion updates, heavy egui frame

### Render Pipeline Phases
The system suggests checking specific render phases:
1. Clear pass
2. Skybox rendering
3. Cell instancing
4. Adhesion lines
5. Split rings
6. Debug overlays
7. UI rendering

## Testing the System

### Manual Test Trigger
Press **F12** to trigger a test spike log and verify the system is working:
```
🧪 TEST PERFORMANCE SPIKE - Frame: 75.50ms, Reason: F12 key pressed - testing spike detection system
```

### Programmatic Testing
```rust
// In your code, you can manually trigger a test spike
performance_metrics.log_test_spike(42.5, "Testing custom spike detection");
```

## Integration

The spike detection is automatically integrated into the main render loop in `src/app.rs`:

```rust
// Update performance metrics (includes spike detection)
self.performance.update(dt);
```

The detection logic is implemented in `src/ui/performance.rs` as part of the `PerformanceMetrics` struct.

## Configuration

To adjust spike detection sensitivity, modify the constants in `src/ui/performance.rs`:

- Increase `SPIKE_THRESHOLD_MULTIPLIER` to reduce sensitivity
- Decrease `SPIKE_MIN_THRESHOLD_MS` to catch smaller spikes
- Adjust `SPIKE_COOLDOWN_SECONDS` to change logging frequency
- Modify `BASELINE_SAMPLES` to change baseline stability

## Benefits

1. **Automatic Monitoring**: No manual intervention required
2. **Detailed Context**: Comprehensive system and application state
3. **Smart Analysis**: Bio-Spheres specific cause identification
4. **Actionable Insights**: Specific suggestions for investigation
5. **Non-Intrusive**: Minimal performance overhead
6. **Configurable**: Easy to adjust sensitivity and behavior

This system helps developers quickly identify and resolve performance issues, making Bio-Spheres run smoother and more efficiently.
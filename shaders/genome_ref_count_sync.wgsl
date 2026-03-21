// Genome Reference Count Synchronization Shader
// Scans all active cells and rebuilds genome_ref_counts from scratch
// based on which genomes are actually in use.
//
// This ensures ref_counts are always accurate, accounting for:
// - Cells that reuse parent genomes (no mutation)
// - Cells that get new genomes (mutation)
// - Dead cells (no longer count toward ref_count)
//
// Run this BEFORE genome GC to ensure accurate recycling decisions.

@group(0) @binding(0)
var<uniform> params: SyncParams;

@group(0) @binding(1)
var<storage, read> genome_ids: array<u32>;

@group(0) @binding(2)
var<storage, read> death_flags: array<u32>;

@group(0) @binding(3)
var<storage, read_write> genome_ref_counts: array<atomic<u32>>;

struct SyncParams {
    cell_capacity: u32,
    genome_capacity: u32,
    _pad0: u32,
    _pad1: u32,
}

// Stage 1: Clear all ref_counts
@compute @workgroup_size(64)
fn clear_ref_counts(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let genome_id = global_id.x;
    
    if (genome_id >= params.genome_capacity) {
        return;
    }
    
    // Reset ref_count to 0
    // User genomes (set to u32::MAX during init) will be reset here,
    // but we'll recount them in the next stage
    atomicStore(&genome_ref_counts[genome_id], 0u);
}

// Stage 2: Count active cells per genome
@compute @workgroup_size(64)
fn count_genome_usage(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    
    if (cell_idx >= params.cell_capacity) {
        return;
    }
    
    // Skip dead cells
    if (death_flags[cell_idx] == 1u) {
        return;
    }
    
    // Increment ref_count for this cell's genome
    let genome_id = genome_ids[cell_idx];
    if (genome_id < params.genome_capacity) {
        atomicAdd(&genome_ref_counts[genome_id], 1u);
    }
}

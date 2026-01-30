//! GPU-side genome compaction compute shader
//! 
//! This shader compacts genome buffer groups by removing unused genomes
//! and reorganizing the remaining ones to eliminate fragmentation.

@group(0) @binding(0)
var<storage, read> genome_reference_counts: array<u32>;

@group(0) @binding(1)
var<storage, read> genome_marked_for_deletion: array<u32>;

@group(0) @binding(2)
var<storage, read_write> genome_compaction_map: array<u32>; // old_id -> new_id mapping

@group(0) @binding(3)
var<storage, read_write> compacted_genome_count: array<u32>;

@group(0) @binding(4)
var<storage, read> max_genomes: array<u32>;

struct CompactionResult {
    old_genome_id: u32,
    new_genome_id: u32,
    should_keep: u32,
}

@group(0) @binding(5)
var<storage, read_write> compaction_results: array<CompactionResult>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let max_genome_count = max_genomes[0];
    
    if idx >= max_genome_count {
        return;
    }
    
    let ref_count = genome_reference_counts[idx];
    let marked_for_deletion = genome_marked_for_deletion[idx];
    
    // Determine if this genome should be kept
    let should_keep = u32(ref_count > 0u && marked_for_deletion == 0u);
    
    // Initialize compaction map
    genome_compaction_map[idx] = idx; // Default: no change
    
    // Store compaction result
    compaction_results[idx] = CompactionResult(
        idx,           // old_genome_id
        idx,           // new_genome_id (will be updated by prefix sum)
        should_keep    // should_keep
    );
    
    // Count genomes that should be kept (using atomic add for parallel reduction)
    if should_keep == 1u {
        let old_count = atomicAdd(&compacted_genome_count[0], 1u);
        genome_compaction_map[idx] = old_count; // Temporary assignment
    }
}

// Second pass: create final mapping using prefix sum
@compute @workgroup_size(64)
fn finalize_compaction(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let max_genome_count = max_genomes[0];
    
    if idx >= max_genome_count {
        return;
    }
    
    let result = compaction_results[idx];
    
    if result.should_keep == 1u {
        // This genome is kept, assign final compacted ID
        genome_compaction_map[result.old_genome_id] = result.new_genome_id;
    } else {
        // This genome is removed, map to invalid value
        genome_compaction_map[result.old_genome_id] = 0xFFFFFFFFu;
    }
}

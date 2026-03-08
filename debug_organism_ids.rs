// Debug script to read organism label buffer
use std::path::Path;

fn main() {
    println!("=== Organism ID Debug Analysis ===");
    
    println!("ISSUE IDENTIFIED:");
    println!("The organism label buffer is created without initial data (mapped_at_creation: false)");
    println!("This means on first frame, the buffer contains garbage/uninitialized values.");
    println!("The organism labeling system only triggers when topology changes occur.");
    println!("On initial startup or when cells are isolated, no topology change = no labeling = garbage data.");
    
    println!("\nROOT CAUSE:");
    println!("1. label_buffer created with uninitialized data in organism_labels.rs:81-86");
    println!("2. Controller only triggers init when cell_count or bond_count changes");
    println!("3. Single cells or static organisms don't trigger topology changes");
    println!("4. extract_cell_data.wgsl reads garbage values from label_buffer");
    
    println!("\nSOLUTION:");
    println!("Initialize label_buffer with proper default values (0xFFFFFFFF for dead/isolated)");
    println!("or force an initial labeling pass on startup.");
}

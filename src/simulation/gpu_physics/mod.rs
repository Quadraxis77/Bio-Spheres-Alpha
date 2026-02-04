//! GPU Physics Module
//! 
//! Contains the GPU compute physics pipeline for Bio-Spheres simulation.

pub mod adhesion;
pub mod adhesion_buffers;
pub mod adhesion_integration;
pub mod cell_data_extraction;
pub mod cell_insertion;
pub mod compute_pipelines;
pub mod dynamic_buffers;
pub mod genome_buffers;
pub mod genome_compaction;
pub mod gpu_cell_inspector;
pub mod gpu_scene_integration;
pub mod gpu_tool_operations;
pub mod phagocyte_consumption;
pub mod triple_buffer;

pub use adhesion_buffers::AdhesionBuffers;
pub use cell_data_extraction::{CellExtractionParams, GpuCellDataExtraction, InspectedCellData};
pub use cell_insertion::GpuCellInsertion;
pub use compute_pipelines::{CachedBindGroups, CellBoostParams, CellInsertionParams, CellRemovalParams, GpuPhysicsPipelines, PositionUpdateParams, SpatialQueryParams, SpatialQueryResult, CellDataExtractionLayouts};
pub use dynamic_buffers::{DynamicBuffer, DynamicGenomeBufferManager};
pub use genome_buffers::{GenomeBufferGroup, GenomeBufferManager, MAX_GENOMES};
pub use gpu_cell_inspector::{AsyncReadbackManager, GpuCellInspector, ReadbackId, ReadbackResult, ReadbackStats};
pub use gpu_scene_integration::{execute_gpu_physics_step, execute_lifecycle_pipeline};
pub use gpu_tool_operations::GpuToolOperations;
pub use phagocyte_consumption::PhagocyteConsumptionSystem;
pub use triple_buffer::GpuTripleBufferSystem;
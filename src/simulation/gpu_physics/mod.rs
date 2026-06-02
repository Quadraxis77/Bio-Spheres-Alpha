//! GPU Physics Module
//!
//! Contains the GPU compute physics pipeline for Bio-Spheres simulation.

pub mod adhesion;
pub mod adhesion_buffers;
pub mod adhesion_integration;
pub mod boulder_buffers;
pub mod boulder_system;
pub mod cell_data_extraction;
pub mod cell_insertion;
pub mod compute_pipelines;
pub mod devorocyte_consumption;
pub mod dynamic_buffers;
pub mod gametocyte_merge;
pub mod genome_buffers;
pub mod genome_compaction;
pub mod gpu_cell_inspector;
pub mod gpu_scene_integration;
pub mod gpu_tool_operations;
pub mod light_field;
pub mod moss;
pub mod mutation;
pub mod organism_labels;
pub mod phagocyte_consumption;
pub mod triple_buffer;

pub use adhesion_buffers::AdhesionBuffers;
pub use boulder_buffers::{BoulderBuffers, BoulderSpawnRequest, GpuBoulder, MAX_BOULDERS};
pub use boulder_system::BoulderSystem;
pub use cell_data_extraction::{CellExtractionParams, GpuCellDataExtraction, InspectedCellData};
pub use cell_insertion::GpuCellInsertion;
pub use compute_pipelines::{
    CachedBindGroups, CellBoostParams, CellDataExtractionLayouts, CellInsertionParams,
    CellRemovalParams, GpuPhysicsPipelines, PositionUpdateParams, SpatialQueryParams,
    SpatialQueryResult,
};
pub use devorocyte_consumption::DevorocyteConsumptionSystem;
pub use dynamic_buffers::{DynamicBuffer, DynamicGenomeBufferManager};
pub use gametocyte_merge::{GameteMergeEvent, GametocyteMergeSystem};
pub use genome_buffers::{GenomeBufferGroup, GenomeBufferManager, MAX_GENOMES};
pub use gpu_cell_inspector::{
    AsyncReadbackManager, GpuCellInspector, ReadbackId, ReadbackResult, ReadbackStats,
};
pub use gpu_scene_integration::{
    execute_gpu_mechanics_step, execute_gpu_physics_step, execute_lifecycle_pipeline,
    execute_signal_system, PhysicsFeatureFlags,
};
pub use gpu_tool_operations::GpuToolOperations;
pub use light_field::LightFieldSystem;
pub use moss::MossSystem;
pub use mutation::GenomeMeta;
pub use mutation::MutationSystem;
pub use organism_labels::OrganismLabelSystem;
pub use phagocyte_consumption::PhagocyteConsumptionSystem;
pub use triple_buffer::GpuTripleBufferSystem;

//! # Bio-Spheres Application Entry Point
//! 
//! This is the main entry point for the Bio-Spheres biological cell simulation.
//! The actual application logic is implemented in the `app` module.
//! 
//! ## Quick Start
//! 
//! Bio-Spheres is a GPU-accelerated biological cell simulation with two modes:
//! - **Preview Mode**: CPU-based physics for genome editing and small simulations
//! - **GPU Mode**: GPU compute shaders for large-scale simulations (10k+ cells)
//! 
//! The application uses:
//! - `wgpu` for GPU rendering and compute
//! - `egui` for the user interface
//! - `winit` for window management
//! 
//! ## Architecture Overview
//! 
//! The main components are:
//! - [`app::App`] - Central application coordinator with wgpu setup
//! - [`simulation::CanonicalState`] - Structure-of-Arrays simulation state
//! - [`scene::PreviewScene`] - CPU physics with GPU rendering
//! - [`scene::GpuScene`] - GPU compute physics (planned)
//! - [`genome::Genome`] - Node-based genome representation
//! 
//! See the `lib.rs` module documentation for detailed architecture information.

fn main() {
    // Initialize and run the Bio-Spheres application
    // All setup, event handling, and rendering is managed by the App struct
    bio_spheres::app::run();
}
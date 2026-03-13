//! AXIOM — Adaptive eXecution with Intelligent Operations Memory
//!
//! A sparse routing architecture for cost-efficient large language model inference.
//! Routes queries across three tiers (Surface, Reasoning, Deep) using a 128-dimensional
//! structural encoder and a hierarchical resolver with dynamic coalition formation
//! and non-local graph communication.
//!
//! No preference data. No GPU. No ML frameworks. Pure Rust.

pub mod encoder;
pub mod graph;
pub mod resolver;
pub mod types;

pub use encoder::StructuralEncoder;
pub use graph::ComputationGraph;
pub use resolver::HierarchicalResolver;
pub use types::*;

//! Sparse computation graph — dynamic routing through minimal computational paths.

pub mod edge;
pub mod engine;
pub mod node;

pub use edge::{ConditionalEdge, EdgeCondition, LateralCondition, LateralEdge};
pub use engine::{RouteResult, SparseGraph, TraceStep, TraversalDirection};
pub use node::{AnalyticalInit, ComputeNode, ContrastiveUpdateInfo, NodeOutput, NodeWeightsData, OrthogonalInit};

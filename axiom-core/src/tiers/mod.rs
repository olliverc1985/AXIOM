//! Hierarchical reasoning tiers — escalating compute from fast/cheap to deep.

pub mod feedback;
pub mod resolver;
pub mod tier;

pub use feedback::{FeedbackReason, FeedbackSignal};
pub use resolver::{HierarchicalResolver, TemporalBuffer, TemporalEntry};
pub use tier::{AxiomConfig, Tier, TierConfig};

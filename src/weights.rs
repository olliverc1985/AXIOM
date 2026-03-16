//! Weight serialisation — save and load trained models.
//!
//! All weight formats use JSON via serde. No custom binary formats.
//!
//! - [`SemanticEncoderData`]: Vocabulary + transformer weights
//! - [`AxiomRouterWeights`]: Full router (semantic encoder + classification head)
//! - [`AxiomConfig`]: Graph architecture configuration

pub use crate::router::AxiomRouterWeights;
pub use crate::semantic::encoder::SemanticEncoderData;
pub use crate::semantic::transformer::SemanticWeightsData;
pub use crate::tiers::AxiomConfig;

//! Transformer encoder components.
//!
//! Small, configurable transformer encoder with full forward and backward passes.
//! Supports multi-head self-attention, layer normalization, and configurable
//! activation functions (GELU, ReLU, SiLU).
//!
//! # Example
//!
//! ```
//! use axiom::transformer::{TransformerConfig, TransformerEncoder};
//!
//! let config = TransformerConfig {
//!     vocab_size: 8192,
//!     hidden_dim: 128,
//!     num_heads: 4,
//!     num_layers: 2,
//!     ff_dim: 512,
//!     max_seq_len: 128,
//!     pooling: "mean".to_string(),
//!     activation: "gelu".to_string(),
//! };
//!
//! let encoder = TransformerEncoder::new(config);
//! assert!(encoder.param_count() > 0);
//! ```

pub use crate::semantic::transformer::{
    AdamState, ForwardCache, LayerCache, LayerGradients, SemanticWeightsData, TransformerConfig,
    TransformerEncoder, TransformerGradients, TransformerLayer,
};

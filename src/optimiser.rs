//! Optimisers for training.
//!
//! Adam optimiser with configurable learning rate, betas, and epsilon.
//! Weight decay is applied externally (AdamW pattern) via the training loop.
//!
//! # Example
//!
//! ```
//! use axiom::optimiser::AdamState;
//!
//! // Create Adam state for parameter groups with sizes [128, 64]
//! let param_sizes = vec![128, 64];
//! let adam = AdamState::new(&param_sizes, 1e-3);
//! assert_eq!(adam.lr, 1e-3);
//! ```

pub use crate::semantic::transformer::AdamState;

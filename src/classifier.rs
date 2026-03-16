//! Classification head for N-class prediction.
//!
//! Linear classifier with softmax over concatenated encoder outputs.
//! Used by [`AxiomRouter`](crate::router::AxiomRouter) for 3-tier routing,
//! but can be used independently for any N-class classification task.
//!
//! # Example
//!
//! ```
//! use axiom::classifier::ClassificationHead;
//!
//! // 256 input dimensions, 3 output classes
//! let head = ClassificationHead::new(256, 42);
//! let input = vec![0.1f32; 256];
//! let (predicted_class, confidence) = head.classify(&input);
//! assert!(predicted_class < 3);
//! assert!((confidence.iter().sum::<f32>() - 1.0).abs() < 1e-5);
//! ```

pub use crate::router::{ClassificationHead, RoutingDecision};
pub use crate::tiers::Tier;

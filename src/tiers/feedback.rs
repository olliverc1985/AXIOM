//! Feedback signals — upward confidence adjustment from deeper tiers.
//!
//! When a Deep node resolves an input with high confidence (above 0.90),
//! it emits a FeedbackSignal upward. This is not backprop — it is
//! directional confidence nudging based on resolution outcomes.

use crate::tiers::Tier;
use serde::{Deserialize, Serialize};

/// A feedback signal emitted by a node to influence shallower tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSignal {
    /// Node that emitted this signal.
    pub from_node: String,
    /// Target tier to receive the signal.
    pub to_tier: Tier,
    /// Reason for the feedback.
    pub reason: FeedbackReason,
    /// Confidence adjustment delta (positive = increase, negative = decrease).
    pub confidence_delta: f32,
}

/// Reason a feedback signal was emitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackReason {
    /// A deep node resolved with high confidence what a shallower tier escalated.
    /// The shallower tier should lower its base confidence for similar inputs.
    LowConfidenceResolved,
    /// Contradictory signals detected between tiers.
    ContradictionDetected,
    /// A cached result was invalidated by deeper processing.
    CacheInvalidation,
}

impl std::fmt::Display for FeedbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeedbackReason::LowConfidenceResolved => write!(f, "LowConfidenceResolved"),
            FeedbackReason::ContradictionDetected => write!(f, "ContradictionDetected"),
            FeedbackReason::CacheInvalidation => write!(f, "CacheInvalidation"),
        }
    }
}

impl FeedbackSignal {
    /// Create a feedback signal for when Deep resolves what Reasoning escalated.
    pub fn low_confidence_resolved(from_node: &str, confidence_delta: f32) -> Self {
        Self {
            from_node: from_node.to_string(),
            to_tier: Tier::Reasoning,
            reason: FeedbackReason::LowConfidenceResolved,
            confidence_delta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_signal_creation() {
        let sig = FeedbackSignal::low_confidence_resolved("deep_standalone", -0.01);
        assert_eq!(sig.from_node, "deep_standalone");
        assert_eq!(sig.to_tier, Tier::Reasoning);
        assert!(sig.confidence_delta < 0.0);
    }

    #[test]
    fn test_feedback_reason_display() {
        assert_eq!(
            format!("{}", FeedbackReason::LowConfidenceResolved),
            "LowConfidenceResolved"
        );
        assert_eq!(
            format!("{}", FeedbackReason::ContradictionDetected),
            "ContradictionDetected"
        );
    }
}

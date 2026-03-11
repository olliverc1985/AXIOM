//! Conditional edges that control routing between compute nodes.

use crate::tiers::Tier;
use serde::{Deserialize, Serialize};

/// An edge connecting two nodes with an activation condition.
///
/// During graph routing, an edge is only traversed if its condition evaluates to true
/// given the current routing state (confidence, tier, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalEdge {
    /// Source node ID.
    pub from: String,
    /// Destination node ID.
    pub to: String,
    /// Condition that must be met for this edge to be traversed.
    pub condition: EdgeCondition,
}

/// Conditions that gate edge traversal during routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeCondition {
    /// Always traverse this edge.
    Always,
    /// Only traverse if current confidence is above the threshold.
    IfConfidenceAbove(f32),
    /// Only traverse if current confidence is below the threshold.
    IfConfidenceBelow(f32),
    /// Only traverse if currently operating at this tier.
    IfTier(Tier),
}

impl ConditionalEdge {
    /// Create an edge that always activates.
    pub fn always(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            condition: EdgeCondition::Always,
        }
    }

    /// Create an edge gated on confidence being below a threshold.
    pub fn if_confidence_below(
        from: impl Into<String>,
        to: impl Into<String>,
        threshold: f32,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            condition: EdgeCondition::IfConfidenceBelow(threshold),
        }
    }

    /// Create an edge gated on confidence being above a threshold.
    pub fn if_confidence_above(
        from: impl Into<String>,
        to: impl Into<String>,
        threshold: f32,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            condition: EdgeCondition::IfConfidenceAbove(threshold),
        }
    }

    /// Create an edge gated on being at a specific tier.
    pub fn if_tier(from: impl Into<String>, to: impl Into<String>, tier: Tier) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            condition: EdgeCondition::IfTier(tier),
        }
    }

    /// Evaluate whether this edge should be traversed given current state.
    pub fn should_traverse(&self, current_confidence: f32, current_tier: Tier) -> bool {
        match &self.condition {
            EdgeCondition::Always => true,
            EdgeCondition::IfConfidenceAbove(t) => current_confidence > *t,
            EdgeCondition::IfConfidenceBelow(t) => current_confidence < *t,
            EdgeCondition::IfTier(t) => current_tier == *t,
        }
    }
}

/// A lateral connection between two nodes at the same tier.
///
/// When a node produces low confidence, lateral edges allow neighbouring
/// nodes at the same tier to attempt the input before escalating.
/// Models cortical column behaviour — neighbours try before escalating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralEdge {
    /// Source node ID.
    pub from: String,
    /// Destination node ID (must be at the same tier).
    pub to: String,
    /// Weight controlling influence of the lateral signal.
    pub weight: f32,
    /// Condition for lateral activation.
    pub condition: LateralCondition,
}

/// Conditions for lateral edge activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LateralCondition {
    /// Fire lateral edge when source confidence is below this threshold.
    IfConfidenceBelow(f32),
    /// Always fire the lateral edge.
    Always,
}

impl LateralEdge {
    /// Create a lateral edge that fires when confidence is below a threshold.
    pub fn if_confidence_below(
        from: impl Into<String>,
        to: impl Into<String>,
        threshold: f32,
        weight: f32,
    ) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            weight,
            condition: LateralCondition::IfConfidenceBelow(threshold),
        }
    }

    /// Create a lateral edge that always fires.
    pub fn always(from: impl Into<String>, to: impl Into<String>, weight: f32) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            weight,
            condition: LateralCondition::Always,
        }
    }

    /// Evaluate whether this lateral edge should fire.
    pub fn should_fire(&self, current_confidence: f32) -> bool {
        match &self.condition {
            LateralCondition::IfConfidenceBelow(t) => current_confidence < *t,
            LateralCondition::Always => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_always_edge() {
        let edge = ConditionalEdge::always("a", "b");
        assert!(edge.should_traverse(0.5, Tier::Surface));
        assert!(edge.should_traverse(0.0, Tier::Deep));
    }

    #[test]
    fn test_confidence_below_edge() {
        let edge = ConditionalEdge::if_confidence_below("a", "b", 0.85);
        assert!(edge.should_traverse(0.5, Tier::Surface));
        assert!(!edge.should_traverse(0.9, Tier::Surface));
        assert!(!edge.should_traverse(0.85, Tier::Surface));
    }

    #[test]
    fn test_confidence_above_edge() {
        let edge = ConditionalEdge::if_confidence_above("a", "b", 0.5);
        assert!(edge.should_traverse(0.8, Tier::Surface));
        assert!(!edge.should_traverse(0.3, Tier::Surface));
    }

    #[test]
    fn test_tier_edge() {
        let edge = ConditionalEdge::if_tier("a", "b", Tier::Reasoning);
        assert!(edge.should_traverse(0.5, Tier::Reasoning));
        assert!(!edge.should_traverse(0.5, Tier::Surface));
    }

    #[test]
    fn test_lateral_edge_confidence_below() {
        let edge = LateralEdge::if_confidence_below("a", "b", 0.75, 1.0);
        assert!(edge.should_fire(0.60));
        assert!(!edge.should_fire(0.80));
    }

    #[test]
    fn test_lateral_edge_always() {
        let edge = LateralEdge::always("a", "b", 0.5);
        assert!(edge.should_fire(0.0));
        assert!(edge.should_fire(1.0));
    }
}

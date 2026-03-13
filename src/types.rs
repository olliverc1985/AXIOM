use serde::{Deserialize, Serialize};
use std::fmt;

/// Model routing tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    Surface,
    Reasoning,
    Deep,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tier::Surface => write!(f, "Surface"),
            Tier::Reasoning => write!(f, "Reasoning"),
            Tier::Deep => write!(f, "Deep"),
        }
    }
}

/// Four distinct traversal directions in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraversalDirection {
    Forward,
    Lateral,
    Feedback,
    Temporal,
}

/// A single step in a routing trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub node_id: String,
    pub tier: Tier,
    pub direction: TraversalDirection,
    pub confidence_in: f32,
    pub confidence_out: f32,
    pub was_cached: bool,
}

/// Operating mode controlling cache behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteMode {
    /// Cache disabled — used during training.
    Training,
    /// Cache enabled at cosine similarity threshold 0.92.
    Inference,
}

/// Result of a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResult {
    pub selected_tier: Tier,
    pub confidence: f32,
    pub resolved_by: String,
    pub trace: Vec<TraceStep>,
    pub lateral_count: usize,
    pub lateral_prevented_escalation: usize,
    pub coalition_members: Vec<String>,
    pub cross_tier_resolution: bool,
    pub was_cached: bool,
}

impl RouteResult {
    pub fn new(tier: Tier, confidence: f32, resolved_by: String) -> Self {
        Self {
            selected_tier: tier,
            confidence,
            resolved_by,
            trace: Vec::new(),
            lateral_count: 0,
            lateral_prevented_escalation: 0,
            coalition_members: Vec::new(),
            cross_tier_resolution: false,
            was_cached: false,
        }
    }
}

/// Conditions under which a conditional edge fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeCondition {
    Always,
    IfConfidenceAbove(f32),
    IfConfidenceBelow(f32),
    IfTier(Tier),
}

impl EdgeCondition {
    pub fn evaluate(&self, confidence: f32, current_tier: Tier) -> bool {
        match self {
            EdgeCondition::Always => true,
            EdgeCondition::IfConfidenceAbove(t) => confidence > *t,
            EdgeCondition::IfConfidenceBelow(t) => confidence < *t,
            EdgeCondition::IfTier(t) => current_tier == *t,
        }
    }
}

/// Condition for lateral edge activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LateralCondition {
    Always,
    IfConfidenceBelow(f32),
}

impl LateralCondition {
    pub fn evaluate(&self, confidence: f32) -> bool {
        match self {
            LateralCondition::Always => true,
            LateralCondition::IfConfidenceBelow(t) => confidence < *t,
        }
    }
}

/// Reason a feedback signal was emitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackReason {
    LowConfidenceResolved,
    ContradictionDetected,
    CacheInvalidation,
}

/// Feedback signal from deeper to shallower tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSignal {
    pub from_node: String,
    pub to_tier: Tier,
    pub reason: FeedbackReason,
    pub confidence_delta: f32,
}

/// Initialisation strategy for compute nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitStrategy {
    /// Mean direction of simple training examples — used for Surface nodes.
    AnalyticalInit,
    /// Near-orthogonal weight directions — used for Reasoning and Deep nodes.
    OrthogonalInit,
}

/// Cost model for a single tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierCostModel {
    pub tier: Tier,
    pub model_name: String,
    pub input_cost_per_million: f64,
    pub output_cost_per_million: f64,
    pub avg_tokens_in: usize,
    pub avg_tokens_out: usize,
}

impl TierCostModel {
    /// Cost for a single query at this tier.
    pub fn query_cost(&self) -> f64 {
        let input_cost = (self.avg_tokens_in as f64 / 1_000_000.0) * self.input_cost_per_million;
        let output_cost =
            (self.avg_tokens_out as f64 / 1_000_000.0) * self.output_cost_per_million;
        input_cost + output_cost
    }
}

/// Production cost model from Table 2.
pub fn default_cost_model() -> Vec<TierCostModel> {
    vec![
        TierCostModel {
            tier: Tier::Surface,
            model_name: "claude-haiku-4-5".into(),
            input_cost_per_million: 0.80,
            output_cost_per_million: 4.00,
            avg_tokens_in: 150,
            avg_tokens_out: 200,
        },
        TierCostModel {
            tier: Tier::Reasoning,
            model_name: "claude-sonnet-4-5".into(),
            input_cost_per_million: 3.00,
            output_cost_per_million: 15.00,
            avg_tokens_in: 300,
            avg_tokens_out: 500,
        },
        TierCostModel {
            tier: Tier::Deep,
            model_name: "claude-opus-4".into(),
            input_cost_per_million: 15.00,
            output_cost_per_million: 75.00,
            avg_tokens_in: 800,
            avg_tokens_out: 1500,
        },
    ]
}

/// AXIOM system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomConfig {
    pub mid_dim: usize,
    pub surface_node_count: usize,
    pub reasoning_node_count: usize,
    pub deep_node_count: usize,
    pub bid_threshold: f32,
    pub max_coalition_size: usize,
    pub temporal_buffer_capacity: usize,
    pub cache_similarity_threshold: f32,
    pub temporal_similarity_threshold: f32,
    pub temporal_blend_live: f32,
    pub temporal_blend_past: f32,
    pub chunk_escalation_threshold: f32,
    pub surface_escalation_threshold: f32,
    pub simple_mean_confidence: f32,
    pub complex_mean_confidence: f32,
}

impl Default for AxiomConfig {
    fn default() -> Self {
        Self {
            mid_dim: 128,
            surface_node_count: 5,
            reasoning_node_count: 15,
            deep_node_count: 15,
            bid_threshold: 0.10,
            max_coalition_size: 4,
            temporal_buffer_capacity: 16,
            cache_similarity_threshold: 0.92,
            temporal_similarity_threshold: 0.85,
            temporal_blend_live: 0.7,
            temporal_blend_past: 0.3,
            chunk_escalation_threshold: 0.40,
            surface_escalation_threshold: 0.5,
            simple_mean_confidence: 0.0,
            complex_mean_confidence: 0.0,
        }
    }
}

/// Persisted weight data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomWeights {
    pub g5_simple_mean_norm: f32,
    pub g5_complex_mean_norm: f32,
    pub node_weights: Vec<NodeWeightEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeWeightEntry {
    pub node_id: String,
    pub tier: Tier,
    pub weights: Vec<f32>,
    pub base_confidence: f32,
}

/// Embedding dimension constant.
pub const EMBEDDING_DIM: usize = 128;

// Feature group dimensions (must sum to EMBEDDING_DIM).
pub const G1_DIM: usize = 26;
pub const G2_DIM: usize = 36;
pub const G3_DIM: usize = 39;
pub const G4_DIM: usize = 15;
pub const G5_DIM: usize = 12;

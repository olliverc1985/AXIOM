//! Tier definitions and configuration for hierarchical reasoning.

use serde::{Deserialize, Serialize};

/// Reasoning tier — escalating levels of computation.
///
/// Most inputs should be resolved at Surface. Only those with low confidence
/// escalate to Reasoning, and only the hardest problems reach Deep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    /// Fast, cheap computation — runs always as the first pass.
    Surface,
    /// Medium computation — only if Surface confidence < surface_confidence_threshold.
    Reasoning,
    /// Deep computation — only if Reasoning confidence < reasoning_confidence_threshold.
    Deep,
}

impl Tier {
    /// Display name for this tier.
    pub fn name(&self) -> &'static str {
        match self {
            Tier::Surface => "Surface",
            Tier::Reasoning => "Reasoning",
            Tier::Deep => "Deep",
        }
    }

    /// Numeric level (0=Surface, 1=Reasoning, 2=Deep).
    pub fn level(&self) -> u8 {
        match self {
            Tier::Surface => 0,
            Tier::Reasoning => 1,
            Tier::Deep => 2,
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Configuration for tier escalation thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// If Surface confidence is below this, escalate to Reasoning (default 0.85).
    pub surface_confidence_threshold: f32,
    /// If Reasoning confidence is below this, escalate to Deep (default 0.70).
    pub reasoning_confidence_threshold: f32,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            surface_confidence_threshold: 0.85,
            reasoning_confidence_threshold: 0.70,
        }
    }
}

/// Full AXIOM configuration — written by the auto-tuner, read by the resolver.
///
/// Persisted to `axiom_config.json`. When present, overrides defaults on startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomConfig {
    /// Surface confidence escalation threshold.
    pub surface_confidence_threshold: f32,
    /// Reasoning confidence escalation threshold.
    pub reasoning_confidence_threshold: f32,
    /// Base confidence for the reasoning standalone node.
    pub reasoning_base_confidence: f32,
    /// Cosine similarity threshold for embedding cache hits.
    pub cache_similarity_threshold: f32,
    /// G5 structural feature norm of the simple corpus mean (Phase 14).
    /// Used for Surface confidence magnitude penalty.
    #[serde(default)]
    pub g5_simple_mean_norm: f32,
    /// G5 structural feature norm of the complex corpus mean (Phase 14).
    /// Used for Surface confidence magnitude penalty.
    #[serde(default)]
    pub g5_complex_mean_norm: f32,
    /// Human-readable explanation of why these values were chosen.
    pub rationale: String,
}

impl Default for AxiomConfig {
    fn default() -> Self {
        Self {
            surface_confidence_threshold: 0.85,
            reasoning_confidence_threshold: 0.70,
            reasoning_base_confidence: 0.72,
            cache_similarity_threshold: 0.92,
            g5_simple_mean_norm: 0.0,
            g5_complex_mean_norm: 0.0,
            rationale: "defaults".to_string(),
        }
    }
}

impl AxiomConfig {
    /// Load from `axiom_config.json` if it exists, otherwise return defaults.
    pub fn load_or_default() -> Self {
        Self::load_from("axiom_config.json").unwrap_or_default()
    }

    /// Load from a specific file path.
    pub fn load_from(path: &str) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Save to `axiom_config.json`.
    pub fn save(&self) -> std::io::Result<()> {
        self.save_to("axiom_config.json")
    }

    /// Save to a specific file path.
    pub fn save_to(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_ordering() {
        assert!(Tier::Surface.level() < Tier::Reasoning.level());
        assert!(Tier::Reasoning.level() < Tier::Deep.level());
    }

    #[test]
    fn test_default_config() {
        let config = TierConfig::default();
        assert!((config.surface_confidence_threshold - 0.85).abs() < 1e-6);
        assert!((config.reasoning_confidence_threshold - 0.70).abs() < 1e-6);
    }
}

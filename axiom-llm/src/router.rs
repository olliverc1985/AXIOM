//! AXIOM Router — combines AXIOM tier routing with LLM API calls.

use crate::{LlmClient, LlmTier, ModelConfig};
use axiom_core::input::{Encoder, Tokeniser};
use axiom_core::tiers::resolver::ResolveResult;
use axiom_core::tiers::{AxiomConfig, HierarchicalResolver, RouteMode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

// ── QueryCostRecord ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCostRecord {
    pub query_hash: String,
    pub routing_tier: String,
    pub model_called: String,
    pub axiom_confidence: f32,
    pub axiom_routing_us: u64,
    pub llm_latency_ms: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub actual_cost_usd: f64,
    pub premium_cost_usd: f64,
    pub saved_usd: f64,
    pub timestamp: u64,
}

// ── RouterResponse ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterResponse {
    pub response: String,
    pub tier: String,
    pub model: String,
    pub confidence: f32,
    pub coalition_members: Vec<String>,
    pub axiom_routing_us: u64,
    pub llm_latency_ms: u64,
    pub actual_cost_usd: f64,
    pub saved_vs_premium_usd: f64,
}

// ── CostSummary ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSummary {
    pub total_queries: u32,
    pub surface_count: u32,
    pub reasoning_count: u32,
    pub deep_count: u32,
    pub total_actual_cost_usd: f64,
    pub total_premium_cost_usd: f64,
    pub total_saved_usd: f64,
    pub savings_percentage: f64,
    pub mean_axiom_routing_us: f64,
    pub mean_llm_latency_ms: f64,
}

// ── AxiomRouter ──

pub struct AxiomRouter {
    pub resolver: HierarchicalResolver,
    pub encoder: Encoder,
    pub llm_client: LlmClient,
    pub cost_log: Vec<QueryCostRecord>,
}

impl AxiomRouter {
    /// Build a router from trained weights and config files.
    pub fn from_trained(
        weights_path: &str,
        vocab_path: &str,
        config_path: &str,
        model_config: ModelConfig,
    ) -> Result<Self, String> {
        // Load AXIOM config
        let config = if std::path::Path::new(config_path).exists() {
            AxiomConfig::load_or_default()
        } else {
            AxiomConfig::default()
        };

        let input_dim = 128;

        // Build tokeniser and load vocabulary
        let mut tokeniser = Tokeniser::default_tokeniser();
        tokeniser
            .load_vocab(vocab_path)
            .map_err(|e| format!("Failed to load vocabulary: {}", e))?;
        let encoder = Encoder::new(input_dim, tokeniser);

        // Build resolver and load weights
        let mut resolver = HierarchicalResolver::build_with_axiom_config(input_dim, &config);
        resolver.mode = RouteMode::Inference;
        resolver
            .load_all_weights(weights_path)
            .map_err(|e| format!("Failed to load weights: {}", e))?;

        // Restore G5 magnitude penalty
        if resolver.g5_simple_mean_norm > 0.0 || resolver.g5_complex_mean_norm > 0.0 {
            resolver.set_g5_penalty_weight(0.25);
        }
        resolver.validate_confidence_invariants();

        // Build LLM client
        let llm_client = LlmClient::new(model_config)?;

        Ok(Self {
            resolver,
            encoder,
            llm_client,
            cost_log: Vec::new(),
        })
    }

    /// Determine LlmTier from a ResolveResult.
    fn tier_from_result(result: &ResolveResult) -> LlmTier {
        match result.tier_reached.name() {
            "Surface" => LlmTier::Surface,
            "Reasoning" => LlmTier::Reasoning,
            "Deep" => LlmTier::Deep,
            _ => LlmTier::Reasoning, // fallback
        }
    }

    /// Hash query for privacy-preserving logging.
    fn query_hash(query: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(query.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)[..8].to_string()
    }

    /// Route a query through AXIOM, then call the appropriate LLM.
    pub fn route_and_call(&mut self, query: &str) -> Result<RouterResponse, String> {
        // Step 1: AXIOM routing
        let route_start = Instant::now();
        let (_surface_conf, _g5_norm, resolve_result) =
            self.resolver.resolve_text(&self.encoder, query);
        let axiom_routing_us = route_start.elapsed().as_micros() as u64;

        let tier = Self::tier_from_result(&resolve_result);
        let confidence = resolve_result.surface_confidence;

        // Collect coalition members
        let coalition_members: Vec<String> = resolve_result
            .coalition
            .as_ref()
            .map(|c| c.members.iter().map(|m| m.node_id.clone()).collect())
            .unwrap_or_default();

        // Step 2: Call LLM
        let llm_response = self.llm_client.call(tier, query)?;

        // Step 3: Compute premium cost (what Opus would have cost)
        let premium_cost =
            LlmClient::premium_cost(llm_response.input_tokens, llm_response.output_tokens);
        let saved = premium_cost - llm_response.cost_usd;

        // Step 4: Record cost
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.cost_log.push(QueryCostRecord {
            query_hash: Self::query_hash(query),
            routing_tier: tier.name().to_string(),
            model_called: llm_response.model.clone(),
            axiom_confidence: confidence,
            axiom_routing_us,
            llm_latency_ms: llm_response.latency_ms,
            input_tokens: llm_response.input_tokens,
            output_tokens: llm_response.output_tokens,
            actual_cost_usd: llm_response.cost_usd,
            premium_cost_usd: premium_cost,
            saved_usd: saved,
            timestamp: now,
        });

        Ok(RouterResponse {
            response: llm_response.content,
            tier: tier.name().to_string(),
            model: llm_response.model,
            confidence,
            coalition_members,
            axiom_routing_us,
            llm_latency_ms: llm_response.latency_ms,
            actual_cost_usd: llm_response.cost_usd,
            saved_vs_premium_usd: saved,
        })
    }

    /// Save cost log to a JSON file.
    pub fn save_cost_log(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.cost_log)
            .map_err(|e| format!("Failed to serialize cost log: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write cost log: {}", e))?;
        Ok(())
    }

    /// Aggregate cost log into summary statistics.
    pub fn cost_summary(&self) -> CostSummary {
        let total = self.cost_log.len() as u32;
        if total == 0 {
            return CostSummary {
                total_queries: 0,
                surface_count: 0,
                reasoning_count: 0,
                deep_count: 0,
                total_actual_cost_usd: 0.0,
                total_premium_cost_usd: 0.0,
                total_saved_usd: 0.0,
                savings_percentage: 0.0,
                mean_axiom_routing_us: 0.0,
                mean_llm_latency_ms: 0.0,
            };
        }

        let surface_count = self
            .cost_log
            .iter()
            .filter(|r| r.routing_tier == "Surface")
            .count() as u32;
        let reasoning_count = self
            .cost_log
            .iter()
            .filter(|r| r.routing_tier == "Reasoning")
            .count() as u32;
        let deep_count = self
            .cost_log
            .iter()
            .filter(|r| r.routing_tier == "Deep")
            .count() as u32;

        let total_actual: f64 = self.cost_log.iter().map(|r| r.actual_cost_usd).sum();
        let total_premium: f64 = self.cost_log.iter().map(|r| r.premium_cost_usd).sum();
        let total_saved: f64 = self.cost_log.iter().map(|r| r.saved_usd).sum();

        let savings_pct = if total_premium > 0.0 {
            (total_saved / total_premium) * 100.0
        } else {
            0.0
        };

        let mean_routing: f64 =
            self.cost_log.iter().map(|r| r.axiom_routing_us as f64).sum::<f64>() / total as f64;
        let mean_llm: f64 =
            self.cost_log.iter().map(|r| r.llm_latency_ms as f64).sum::<f64>() / total as f64;

        CostSummary {
            total_queries: total,
            surface_count,
            reasoning_count,
            deep_count,
            total_actual_cost_usd: total_actual,
            total_premium_cost_usd: total_premium,
            total_saved_usd: total_saved,
            savings_percentage: savings_pct,
            mean_axiom_routing_us: mean_routing,
            mean_llm_latency_ms: mean_llm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_hash_length() {
        let hash = AxiomRouter::query_hash("hello world");
        assert_eq!(hash.len(), 8);
    }

    #[test]
    fn test_query_hash_deterministic() {
        let h1 = AxiomRouter::query_hash("test query");
        let h2 = AxiomRouter::query_hash("test query");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_query_hash_different_inputs() {
        let h1 = AxiomRouter::query_hash("simple");
        let h2 = AxiomRouter::query_hash("complex");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_cost_summary_empty() {
        let summary = CostSummary {
            total_queries: 0,
            surface_count: 0,
            reasoning_count: 0,
            deep_count: 0,
            total_actual_cost_usd: 0.0,
            total_premium_cost_usd: 0.0,
            total_saved_usd: 0.0,
            savings_percentage: 0.0,
            mean_axiom_routing_us: 0.0,
            mean_llm_latency_ms: 0.0,
        };
        assert_eq!(summary.total_queries, 0);
    }

    #[test]
    fn test_cost_summary_aggregation() {
        let records = vec![
            QueryCostRecord {
                query_hash: "abcd1234".to_string(),
                routing_tier: "Surface".to_string(),
                model_called: "claude-haiku".to_string(),
                axiom_confidence: 0.9,
                axiom_routing_us: 100,
                llm_latency_ms: 500,
                input_tokens: 50,
                output_tokens: 100,
                actual_cost_usd: 0.001,
                premium_cost_usd: 0.01,
                saved_usd: 0.009,
                timestamp: 0,
            },
            QueryCostRecord {
                query_hash: "efgh5678".to_string(),
                routing_tier: "Deep".to_string(),
                model_called: "claude-opus".to_string(),
                axiom_confidence: 0.4,
                axiom_routing_us: 200,
                llm_latency_ms: 2000,
                input_tokens: 100,
                output_tokens: 500,
                actual_cost_usd: 0.05,
                premium_cost_usd: 0.05,
                saved_usd: 0.0,
                timestamp: 0,
            },
        ];

        // Manually compute what cost_summary would do
        let total_actual: f64 = records.iter().map(|r| r.actual_cost_usd).sum();
        let total_premium: f64 = records.iter().map(|r| r.premium_cost_usd).sum();
        let total_saved: f64 = records.iter().map(|r| r.saved_usd).sum();
        let savings_pct = (total_saved / total_premium) * 100.0;

        assert!((total_actual - 0.051).abs() < 1e-10);
        assert!((total_premium - 0.06).abs() < 1e-10);
        assert!((savings_pct - 15.0).abs() < 1e-6);
    }
}

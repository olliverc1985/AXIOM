//! AXIOM LLM integration — route queries to Anthropic Claude models
//! based on AXIOM tier decisions.

pub mod router;

use serde::{Deserialize, Serialize};
use std::time::Instant;

// ── Model pricing per million tokens (USD) ──

pub const HAIKU_INPUT_PER_M: f64 = 0.80;
pub const HAIKU_OUTPUT_PER_M: f64 = 4.00;
pub const SONNET_INPUT_PER_M: f64 = 3.00;
pub const SONNET_OUTPUT_PER_M: f64 = 15.00;
pub const OPUS_INPUT_PER_M: f64 = 15.00;
pub const OPUS_OUTPUT_PER_M: f64 = 75.00;

// ── LlmTier ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmTier {
    Surface,   // claude-haiku-3
    Reasoning, // claude-sonnet-4-5
    Deep,      // claude-opus-4 (fallback to claude-sonnet-4-5)
}

impl LlmTier {
    pub fn name(&self) -> &'static str {
        match self {
            LlmTier::Surface => "Surface",
            LlmTier::Reasoning => "Reasoning",
            LlmTier::Deep => "Deep",
        }
    }
}

// ── ModelConfig ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub surface_model: String,
    pub reasoning_model: String,
    pub deep_model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            surface_model: "claude-haiku-4-5-20241022".to_string(),
            reasoning_model: "claude-sonnet-4-5-20250514".to_string(),
            deep_model: "claude-opus-4-20250514".to_string(),
            max_tokens: 1024,
            temperature: 0.7,
        }
    }
}

impl ModelConfig {
    pub fn model_for_tier(&self, tier: LlmTier) -> &str {
        match tier {
            LlmTier::Surface => &self.surface_model,
            LlmTier::Reasoning => &self.reasoning_model,
            LlmTier::Deep => &self.deep_model,
        }
    }

    pub fn input_price_per_m(model: &str) -> f64 {
        if model.contains("haiku") {
            HAIKU_INPUT_PER_M
        } else if model.contains("opus") {
            OPUS_INPUT_PER_M
        } else {
            SONNET_INPUT_PER_M
        }
    }

    pub fn output_price_per_m(model: &str) -> f64 {
        if model.contains("haiku") {
            HAIKU_OUTPUT_PER_M
        } else if model.contains("opus") {
            OPUS_OUTPUT_PER_M
        } else {
            SONNET_OUTPUT_PER_M
        }
    }
}

// ── LlmResponse ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub content: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: f64,
    pub latency_ms: u64,
}

// ── Anthropic API request/response types ──

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    model: String,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContentBlock {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
struct AnthropicError {
    error: AnthropicErrorDetail,
}

#[derive(Deserialize)]
struct AnthropicErrorDetail {
    message: String,
}

// ── LlmClient ──

pub struct LlmClient {
    pub config: ModelConfig,
    api_key: String,
    client: reqwest::blocking::Client,
}

impl LlmClient {
    pub fn new(config: ModelConfig) -> Result<Self, String> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY environment variable not set".to_string())?;

        if api_key.is_empty() {
            return Err("ANTHROPIC_API_KEY is empty".to_string());
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        Ok(Self {
            config,
            api_key,
            client,
        })
    }

    pub fn call(&self, tier: LlmTier, prompt: &str) -> Result<LlmResponse, String> {
        let model = self.config.model_for_tier(tier);
        self.call_anthropic(model, prompt)
    }

    fn call_anthropic(&self, model: &str, prompt: &str) -> Result<LlmResponse, String> {
        let request = AnthropicRequest {
            model: model.to_string(),
            max_tokens: self.config.max_tokens,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: Some(self.config.temperature),
        };

        let start = Instant::now();

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        let latency_ms = start.elapsed().as_millis() as u64;
        let status = response.status();
        let body = response
            .text()
            .map_err(|e| format!("Failed to read response body: {}", e))?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<AnthropicError>(&body) {
                return Err(format!(
                    "API error ({}): {}",
                    status.as_u16(),
                    err.error.message
                ));
            }
            return Err(format!("API error ({}): {}", status.as_u16(), body));
        }

        let api_response: AnthropicResponse = serde_json::from_str(&body)
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        let content = api_response
            .content
            .into_iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");

        let input_tokens = api_response.usage.input_tokens;
        let output_tokens = api_response.usage.output_tokens;

        let input_price = ModelConfig::input_price_per_m(&api_response.model);
        let output_price = ModelConfig::output_price_per_m(&api_response.model);
        let cost_usd = (input_tokens as f64 * input_price / 1_000_000.0)
            + (output_tokens as f64 * output_price / 1_000_000.0);

        Ok(LlmResponse {
            content,
            model: api_response.model,
            input_tokens,
            output_tokens,
            cost_usd,
            latency_ms,
        })
    }

    /// Compute what it would cost to call Opus for a given token count.
    pub fn premium_cost(input_tokens: u32, output_tokens: u32) -> f64 {
        (input_tokens as f64 * OPUS_INPUT_PER_M / 1_000_000.0)
            + (output_tokens as f64 * OPUS_OUTPUT_PER_M / 1_000_000.0)
    }
}

// ── Unit tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_names() {
        assert_eq!(LlmTier::Surface.name(), "Surface");
        assert_eq!(LlmTier::Reasoning.name(), "Reasoning");
        assert_eq!(LlmTier::Deep.name(), "Deep");
    }

    #[test]
    fn test_model_mapping() {
        let config = ModelConfig::default();
        assert!(config.model_for_tier(LlmTier::Surface).contains("haiku"));
        assert!(config.model_for_tier(LlmTier::Reasoning).contains("sonnet"));
        assert!(config.model_for_tier(LlmTier::Deep).contains("opus"));
    }

    #[test]
    fn test_cost_calculation() {
        // 1000 input tokens on Haiku: 1000 * 0.80 / 1_000_000 = 0.0008
        // 500 output tokens on Haiku: 500 * 4.00 / 1_000_000 = 0.002
        let input_price = ModelConfig::input_price_per_m("claude-haiku-4-5-20241022");
        let output_price = ModelConfig::output_price_per_m("claude-haiku-4-5-20241022");
        let cost = (1000.0 * input_price / 1_000_000.0) + (500.0 * output_price / 1_000_000.0);
        assert!((cost - 0.0028).abs() < 1e-10);
    }

    #[test]
    fn test_premium_cost() {
        // 1000 input + 500 output on Opus
        let cost = LlmClient::premium_cost(1000, 500);
        let expected = (1000.0 * 15.0 / 1_000_000.0) + (500.0 * 75.0 / 1_000_000.0);
        assert!((cost - expected).abs() < 1e-10);
    }

    #[test]
    fn test_savings_ratio() {
        let haiku_cost = (1000.0 * HAIKU_INPUT_PER_M / 1_000_000.0)
            + (500.0 * HAIKU_OUTPUT_PER_M / 1_000_000.0);
        let opus_cost = LlmClient::premium_cost(1000, 500);
        let savings = 1.0 - (haiku_cost / opus_cost);
        // Haiku is much cheaper than Opus — savings should be >90%
        assert!(savings > 0.90);
    }

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.max_tokens, 1024);
        assert!((config.temperature - 0.7).abs() < 1e-6);
    }
}

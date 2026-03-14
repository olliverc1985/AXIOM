//! AXIOM auto-tuner — reads bench logs and adjusts configuration parameters.
//!
//! Tuning rules:
//! - Deep % > 25% → raise reasoning_base_confidence by 0.02
//! - Deep % < 10% → lower reasoning_base_confidence by 0.02
//! - Cache hit rate > 60% → raise cache similarity threshold by 0.01
//! - Cache hit rate < 25% → lower cache similarity threshold by 0.01
//! - Surface % > 70% → raise surface_confidence_threshold by 0.02
//! - All values clamped to [0.50, 0.98]

use crate::tiers::AxiomConfig;
use serde::Deserialize;

/// A single bench log entry (matches BenchLogEntry from axiom-bench).
#[derive(Debug, Deserialize)]
pub struct LogEntry {
    pub tier_reached: String,
    pub confidence: f32,
    pub compute_cost: f32,
    pub cache_hits: u32,
    pub from_cache: bool,
}

/// Statistics extracted from a bench log.
#[derive(Debug)]
pub struct BenchStats {
    /// Total number of entries.
    pub total: usize,
    /// Percentage of Surface tier results.
    pub surface_pct: f32,
    /// Percentage of Reasoning tier results.
    pub reasoning_pct: f32,
    /// Percentage of Deep tier results.
    pub deep_pct: f32,
    /// Cache hit rate as percentage.
    pub cache_hit_pct: f32,
    /// Average confidence.
    pub avg_confidence: f32,
    /// Average compute cost.
    pub avg_cost: f32,
}

/// Compute statistics from a bench log file.
pub fn compute_stats(log_path: &str) -> Result<BenchStats, String> {
    let data =
        std::fs::read_to_string(log_path).map_err(|e| format!("Failed to read {}: {}", log_path, e))?;
    let entries: Vec<LogEntry> =
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse {}: {}", log_path, e))?;

    if entries.is_empty() {
        return Err("Log file is empty".to_string());
    }

    let total = entries.len();
    let surface = entries.iter().filter(|e| e.tier_reached == "Surface").count();
    let reasoning = entries.iter().filter(|e| e.tier_reached == "Reasoning").count();
    let deep = entries.iter().filter(|e| e.tier_reached == "Deep").count();
    let cache_hits = entries.iter().filter(|e| e.from_cache).count();
    let avg_confidence = entries.iter().map(|e| e.confidence).sum::<f32>() / total as f32;
    let avg_cost = entries.iter().map(|e| e.compute_cost).sum::<f32>() / total as f32;

    Ok(BenchStats {
        total,
        surface_pct: surface as f32 / total as f32 * 100.0,
        reasoning_pct: reasoning as f32 / total as f32 * 100.0,
        deep_pct: deep as f32 / total as f32 * 100.0,
        cache_hit_pct: cache_hits as f32 / total as f32 * 100.0,
        avg_confidence,
        avg_cost,
    })
}

/// Clamp a value to [0.50, 0.98].
fn clamp(v: f32) -> f32 {
    v.clamp(0.50, 0.98)
}

/// Generate a tuning recommendation from bench statistics.
///
/// Reads the current config (or defaults) and applies adjustment rules.
pub fn tune(stats: &BenchStats, current: &AxiomConfig) -> AxiomConfig {
    let mut config = current.clone();
    let mut reasons = Vec::new();

    // Deep % rules
    if stats.deep_pct > 25.0 {
        config.reasoning_base_confidence = clamp(config.reasoning_base_confidence + 0.02);
        reasons.push(format!(
            "Deep {:.1}% > 25% → raised reasoning_base_confidence to {:.2}",
            stats.deep_pct, config.reasoning_base_confidence
        ));
    } else if stats.deep_pct < 10.0 {
        config.reasoning_base_confidence = clamp(config.reasoning_base_confidence - 0.02);
        reasons.push(format!(
            "Deep {:.1}% < 10% → lowered reasoning_base_confidence to {:.2}",
            stats.deep_pct, config.reasoning_base_confidence
        ));
    } else {
        reasons.push(format!("Deep {:.1}% in range [10%, 25%] → no change", stats.deep_pct));
    }

    // Cache hit rate rules
    if stats.cache_hit_pct > 60.0 {
        config.cache_similarity_threshold = clamp(config.cache_similarity_threshold + 0.01);
        reasons.push(format!(
            "Cache {:.1}% > 60% → raised cache threshold to {:.2}",
            stats.cache_hit_pct, config.cache_similarity_threshold
        ));
    } else if stats.cache_hit_pct < 25.0 {
        config.cache_similarity_threshold = clamp(config.cache_similarity_threshold - 0.01);
        reasons.push(format!(
            "Cache {:.1}% < 25% → lowered cache threshold to {:.2}",
            stats.cache_hit_pct, config.cache_similarity_threshold
        ));
    } else {
        reasons.push(format!(
            "Cache {:.1}% in range [25%, 60%] → no change",
            stats.cache_hit_pct
        ));
    }

    // Surface % rules
    if stats.surface_pct > 70.0 {
        config.surface_confidence_threshold = clamp(config.surface_confidence_threshold + 0.02);
        reasons.push(format!(
            "Surface {:.1}% > 70% → raised surface threshold to {:.2}",
            stats.surface_pct, config.surface_confidence_threshold
        ));
    } else {
        reasons.push(format!(
            "Surface {:.1}% <= 70% → no change",
            stats.surface_pct
        ));
    }

    // Ensure reasoning_base_confidence can actually reach the reasoning threshold.
    // Max achievable confidence = base * 0.7 + 0.3 (when ratio = 1.0).
    // Must exceed reasoning_confidence_threshold.
    let min_base = (config.reasoning_confidence_threshold - 0.3) / 0.7;
    if config.reasoning_base_confidence < min_base + 0.02 {
        config.reasoning_base_confidence = min_base + 0.02;
        reasons.push("floor applied to reasoning_base_confidence to ensure reachability".to_string());
    }

    config.rationale = reasons.join("; ");
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tune_high_deep() {
        let stats = BenchStats {
            total: 100,
            surface_pct: 30.0,
            reasoning_pct: 40.0,
            deep_pct: 30.0,
            cache_hit_pct: 40.0,
            avg_confidence: 0.8,
            avg_cost: 0.5,
        };
        let current = AxiomConfig::default();
        let result = tune(&stats, &current);
        assert!(result.reasoning_base_confidence > current.reasoning_base_confidence);
    }

    #[test]
    fn test_tune_low_deep() {
        let stats = BenchStats {
            total: 100,
            surface_pct: 50.0,
            reasoning_pct: 45.0,
            deep_pct: 5.0,
            cache_hit_pct: 40.0,
            avg_confidence: 0.8,
            avg_cost: 0.5,
        };
        let current = AxiomConfig::default();
        let result = tune(&stats, &current);
        assert!(result.reasoning_base_confidence < current.reasoning_base_confidence);
    }

    #[test]
    fn test_tune_high_cache() {
        let stats = BenchStats {
            total: 100,
            surface_pct: 50.0,
            reasoning_pct: 40.0,
            deep_pct: 10.0,
            cache_hit_pct: 65.0,
            avg_confidence: 0.8,
            avg_cost: 0.5,
        };
        let current = AxiomConfig::default();
        let result = tune(&stats, &current);
        assert!(result.cache_similarity_threshold > current.cache_similarity_threshold);
    }

    #[test]
    fn test_tune_high_surface() {
        let stats = BenchStats {
            total: 100,
            surface_pct: 75.0,
            reasoning_pct: 15.0,
            deep_pct: 10.0,
            cache_hit_pct: 40.0,
            avg_confidence: 0.8,
            avg_cost: 0.5,
        };
        let current = AxiomConfig::default();
        let result = tune(&stats, &current);
        assert!(result.surface_confidence_threshold > current.surface_confidence_threshold);
    }

    #[test]
    fn test_clamp_bounds() {
        let stats = BenchStats {
            total: 100,
            surface_pct: 0.0,
            reasoning_pct: 0.0,
            deep_pct: 0.0,
            cache_hit_pct: 0.0,
            avg_confidence: 0.8,
            avg_cost: 0.5,
        };
        // Start with extreme values
        let mut current = AxiomConfig::default();
        current.reasoning_base_confidence = 0.50;
        current.cache_similarity_threshold = 0.50;
        let result = tune(&stats, &current);
        assert!(result.reasoning_base_confidence >= 0.50);
        assert!(result.cache_similarity_threshold >= 0.50);
    }
}

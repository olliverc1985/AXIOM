//! AXIOM Bench — Phase 4: weight learning + temporal geometry.
//!
//! Four-pass bench: synthetic (learning) → text → synthetic (learning) → text.
//! Hebbian weight learning active during synthetic passes (lr=0.001).
//! Weight drift logged every 100 iterations. Recalibrate between passes.
//!
//! Usage:
//!   cargo run --release -p axiom-bench              # terminal only
//!   cargo run --release -p axiom-bench -- --dashboard  # terminal + HTML dashboard on :8080

mod dashboard;

use axiom_core::graph::TraversalDirection;
use axiom_core::input::{Encoder, Tokeniser};
use axiom_core::tiers::{AxiomConfig, HierarchicalResolver};
use axiom_core::Tensor;
use dashboard::{DashboardState, TraceSnapshot};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// JSON log entry for a single bench iteration.
#[derive(Serialize, Deserialize)]
struct BenchLogEntry {
    input_hash: u64,
    execution_trace: Vec<String>,
    confidence: f32,
    compute_cost: f32,
    cache_hits: u32,
    tier_reached: String,
    timestamp: u64,
    iteration: usize,
    from_cache: bool,
    lateral_count: u32,
    lateral_prevented: u32,
    feedback_count: usize,
    forward_steps: usize,
    lateral_steps: usize,
    feedback_steps: usize,
    temporal_steps: usize,
    winning_path: String,
}

/// JSON log entry for a text input.
#[derive(Serialize, Deserialize)]
struct TextLogEntry {
    sentence: String,
    token_count: usize,
    input_hash: u64,
    execution_trace: Vec<String>,
    confidence: f32,
    compute_cost: f32,
    cache_hits: u32,
    tier_reached: String,
    from_cache: bool,
    lateral_count: u32,
    lateral_prevented: u32,
    feedback_count: usize,
    forward_steps: usize,
    lateral_steps: usize,
    feedback_steps: usize,
    temporal_steps: usize,
    winning_path: String,
}

/// Simple deterministic PRNG (xorshift32).
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Synthetic input generator — same three populations as Phase 1.
fn generate_input(iteration: usize, dim: usize) -> Tensor {
    let mut data = Vec::with_capacity(dim);
    let category = iteration % 10;

    match category {
        0..=3 => {
            let archetype = (iteration / 10) % 8;
            let mut rng = Rng::new((iteration as u32).wrapping_mul(2654435761));
            for j in 0..dim {
                let d1 = (archetype * 2) % dim;
                let d2 = (archetype * 2 + 1) % dim;
                let base = if j == d1 {
                    0.8
                } else if j == d2 {
                    0.6
                } else {
                    0.0
                };
                let noise = rng.next_f32() * 0.005;
                data.push(base + noise);
            }
        }
        4..=7 => {
            let mut rng =
                Rng::new((iteration as u32).wrapping_mul(1103515245).wrapping_add(12345));
            let n_active = 4 + (rng.next_u32() as usize % 5);
            let mut active_dims: Vec<usize> = Vec::new();
            while active_dims.len() < n_active && active_dims.len() < dim {
                let d = rng.next_u32() as usize % dim;
                if !active_dims.contains(&d) {
                    active_dims.push(d);
                }
            }
            for j in 0..dim {
                if active_dims.contains(&j) {
                    data.push(rng.next_f32() * 0.7);
                } else {
                    data.push(rng.next_f32() * 0.02);
                }
            }
        }
        _ => {
            let mut rng = Rng::new(
                (iteration as u32)
                    .wrapping_mul(2246822519)
                    .wrapping_add(iteration as u32 + 1),
            );
            for _ in 0..dim {
                data.push(rng.next_f32());
            }
        }
    }

    Tensor::from_vec(data)
}

/// Hardcoded test sentences spanning simple → complex.
fn test_sentences() -> Vec<(&'static str, &'static str)> {
    vec![
        // Simple
        ("the cat sat on the mat", "simple"),
        ("hello world", "simple"),
        ("the dog runs fast", "simple"),
        ("it is raining today", "simple"),
        ("she likes green apples", "simple"),
        ("the sun is bright", "simple"),
        ("birds fly south in winter", "simple"),
        ("water flows downhill", "simple"),
        ("he reads a book", "simple"),
        ("the sky is blue", "simple"),
        ("fish swim in the sea", "simple"),
        ("bread is made from flour", "simple"),
        ("trees grow tall", "simple"),
        ("snow falls in december", "simple"),
        ("the baby sleeps quietly", "simple"),
        // Moderate
        ("machine learning models require significant computational resources", "moderate"),
        ("the stock market fluctuates based on investor sentiment and economic indicators", "moderate"),
        ("neural networks approximate complex nonlinear functions through layered transformations", "moderate"),
        ("sustainable energy solutions demand innovative engineering approaches", "moderate"),
        ("distributed systems must handle network partitions and eventual consistency", "moderate"),
        ("the relationship between correlation and causation is frequently misunderstood", "moderate"),
        ("genetic algorithms evolve solutions through mutation crossover and selection", "moderate"),
        ("urban planning requires balancing economic growth with environmental preservation", "moderate"),
        ("cryptographic protocols ensure secure communication over untrusted channels", "moderate"),
        ("database indexing strategies significantly impact query performance", "moderate"),
        ("compiler optimisation transforms source code into efficient machine instructions", "moderate"),
        ("reinforcement learning agents maximise cumulative reward through exploration", "moderate"),
        ("microservice architectures introduce complexity in exchange for scalability", "moderate"),
        ("the efficiency of sorting algorithms depends on input distribution characteristics", "moderate"),
        ("parallel computing enables processing of large datasets across multiple cores", "moderate"),
        ("memory management in systems programming prevents resource leaks and corruption", "moderate"),
        ("functional programming emphasises immutability and compositional reasoning", "moderate"),
        ("type systems provide compile time guarantees about program behaviour", "moderate"),
        // Complex
        ("the recursive nature of self-referential systems creates emergent properties that resist reduction", "complex"),
        ("consciousness remains an unsolved problem at the intersection of neuroscience philosophy and computation", "complex"),
        ("the halting problem demonstrates fundamental limits of algorithmic decidability in formal systems", "complex"),
        ("quantum entanglement challenges classical notions of locality and suggests nonlocal correlations in nature", "complex"),
        ("goedel's incompleteness theorems reveal inherent limitations in any sufficiently powerful formal axiomatic system", "complex"),
        ("the boundary between deterministic chaos and true randomness has profound implications for predictability", "complex"),
        ("emergence in complex adaptive systems suggests that reductionist explanations are fundamentally insufficient", "complex"),
        ("the relationship between syntactic structure and semantic meaning in natural language defies simple formalisation", "complex"),
        ("computational irreducibility implies that some processes cannot be predicted without full simulation", "complex"),
        ("the tension between expressiveness and decidability shapes the design of every formal language and logic", "complex"),
        ("category theory provides a unifying framework for mathematical structures through abstract morphisms and functors", "complex"),
        ("the measurement problem in quantum mechanics exposes deep philosophical questions about the nature of observation and reality", "complex"),
        ("information theoretic entropy quantifies uncertainty and establishes fundamental limits on data compression and transmission", "complex"),
        ("the interplay between cooperation and competition in evolutionary game theory produces counterintuitive stable equilibria", "complex"),
        ("kolmogorov complexity offers an objective measure of randomness but is itself uncomputable in the general case", "complex"),
        ("strange attractors in dynamical systems exhibit sensitive dependence on initial conditions while maintaining bounded trajectories", "complex"),
        ("the church turing thesis equates mechanical computation with turing machine computation but remains an unproven conjecture", "complex"),
    ]
}

/// Count traversal direction steps from a resolve result.
fn count_directions(
    trace_steps: &[axiom_core::graph::TraceStep],
) -> (usize, usize, usize, usize) {
    let mut forward = 0;
    let mut lateral = 0;
    let mut feedback = 0;
    let mut temporal = 0;
    for step in trace_steps {
        match step.direction {
            TraversalDirection::Forward => forward += 1,
            TraversalDirection::Lateral => lateral += 1,
            TraversalDirection::Feedback => feedback += 1,
            TraversalDirection::Temporal => temporal += 1,
        }
    }
    (forward, lateral, feedback, temporal)
}

/// Weight drift snapshot at a point in time.
#[derive(Serialize, Deserialize, Clone)]
struct WeightDriftEntry {
    iteration: usize,
    weight_norm: f32,
}

/// Synthetic pass results.
struct SyntheticPassResult {
    entries: Vec<BenchLogEntry>,
    tier_counts: HashMap<String, usize>,
    total_cost: f32,
    total_confidence: f32,
    forward: usize,
    lateral: usize,
    feedback: usize,
    temporal: usize,
    lateral_count: u32,
    lateral_prevented: u32,
    feedback_signals: usize,
    weight_drift: Vec<WeightDriftEntry>,
}

/// Run a synthetic bench pass with optional Hebbian learning and weight drift tracking.
fn run_synthetic_pass(
    resolver: &mut HierarchicalResolver,
    input_dim: usize,
    iterations: usize,
    learning_rate: f32,
    progress: &ProgressBar,
    stats_bar: &ProgressBar,
    dashboard_mode: bool,
    state: &Arc<Mutex<DashboardState>>,
    bench_start: &Instant,
) -> SyntheticPassResult {
    let mut tier_counts: HashMap<String, usize> = HashMap::new();
    tier_counts.insert("Surface".to_string(), 0);
    tier_counts.insert("Reasoning".to_string(), 0);
    tier_counts.insert("Deep".to_string(), 0);
    let mut total_cost = 0.0f32;
    let mut total_confidence = 0.0f32;
    let mut entries = Vec::with_capacity(iterations);
    let mut total_forward = 0usize;
    let mut total_lateral_steps = 0usize;
    let mut total_feedback_steps = 0usize;
    let mut total_temporal_steps = 0usize;
    let mut total_lateral_count = 0u32;
    let mut total_lateral_prevented = 0u32;
    let mut total_feedback_signals = 0usize;
    let mut weight_drift: Vec<WeightDriftEntry> = Vec::new();

    // Initial weight norm
    weight_drift.push(WeightDriftEntry {
        iteration: 0,
        weight_norm: resolver.total_weight_norm(),
    });

    for i in 0..iterations {
        let input = generate_input(i, input_dim);
        let input_hash = input.content_hash();
        let result = resolver.resolve(&input);

        // Hebbian learning
        if learning_rate > 0.0 {
            resolver.learn(&input, &result, learning_rate);
        }

        let tier_name = result.tier_reached.name().to_string();
        *tier_counts.get_mut(&tier_name).unwrap() += 1;
        total_cost += result.route.total_compute_cost;
        total_confidence += result.route.confidence;
        total_lateral_count += result.route.lateral_count;
        total_lateral_prevented += result.route.lateral_prevented_escalation;
        total_feedback_signals += result.feedback_signals.len();

        let (fwd, lat, fb, tmp) = count_directions(&result.route.trace_steps);
        total_forward += fwd;
        total_lateral_steps += lat;
        total_feedback_steps += fb;
        total_temporal_steps += tmp;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        entries.push(BenchLogEntry {
            input_hash,
            execution_trace: result.route.execution_trace.clone(),
            confidence: result.route.confidence,
            compute_cost: result.route.total_compute_cost,
            cache_hits: result.route.cache_hits,
            tier_reached: tier_name.clone(),
            timestamp,
            iteration: i,
            from_cache: result.from_cache,
            lateral_count: result.route.lateral_count,
            lateral_prevented: result.route.lateral_prevented_escalation,
            feedback_count: result.feedback_signals.len(),
            forward_steps: fwd,
            lateral_steps: lat,
            feedback_steps: fb,
            temporal_steps: tmp,
            winning_path: result.winning_path.clone(),
        });

        // Weight drift logging every 100 iterations
        if learning_rate > 0.0 && (i + 1) % 100 == 0 {
            weight_drift.push(WeightDriftEntry {
                iteration: i + 1,
                weight_norm: resolver.total_weight_norm(),
            });
        }

        if dashboard_mode {
            let mut s = state.lock().unwrap();
            s.iteration = i + 1;
            s.surface_count = *tier_counts.get("Surface").unwrap();
            s.reasoning_count = *tier_counts.get("Reasoning").unwrap();
            s.deep_count = *tier_counts.get("Deep").unwrap();
            s.cache_hit_rate = resolver.cache_hit_rate() * 100.0;
            s.cache_size = resolver.cache_size();
            s.avg_compute_cost = total_cost / (i + 1) as f32;
            s.avg_confidence = total_confidence / (i + 1) as f32;
            s.elapsed_ms = bench_start.elapsed().as_millis() as u64;
            s.forward_steps = total_forward;
            s.lateral_steps = total_lateral_steps;
            s.feedback_steps = total_feedback_steps;
            s.temporal_steps = total_temporal_steps;
            s.lateral_prevented = total_lateral_prevented as usize;
            s.feedback_signals = total_feedback_signals;
            s.push_trace(TraceSnapshot {
                iteration: i,
                tier: tier_name.clone(),
                trace: result.route.execution_trace,
                confidence: result.route.confidence,
                cost: result.route.total_compute_cost,
                cached: result.from_cache,
            });
        }

        if i % 100 == 0 || i == iterations - 1 {
            let done = i + 1;
            let surface_pct =
                *tier_counts.get("Surface").unwrap() as f32 / done as f32 * 100.0;
            let reasoning_pct =
                *tier_counts.get("Reasoning").unwrap() as f32 / done as f32 * 100.0;
            let deep_pct = *tier_counts.get("Deep").unwrap() as f32 / done as f32 * 100.0;
            let cache_rate = resolver.cache_hit_rate() * 100.0;

            progress.set_position(done as u64);
            progress.set_message(format!("cache: {:.1}%", cache_rate));
            stats_bar.set_message(format!(
                "S: {:.1}% | R: {:.1}% | D: {:.1}% | Cache: {:.1}%",
                surface_pct, reasoning_pct, deep_pct, cache_rate
            ));
        }

        if dashboard_mode && i % 100 == 99 {
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    SyntheticPassResult {
        entries,
        tier_counts,
        total_cost,
        total_confidence,
        forward: total_forward,
        lateral: total_lateral_steps,
        feedback: total_feedback_steps,
        temporal: total_temporal_steps,
        lateral_count: total_lateral_count,
        lateral_prevented: total_lateral_prevented,
        feedback_signals: total_feedback_signals,
        weight_drift,
    }
}

/// Text pass results.
struct TextPassResult {
    entries: Vec<TextLogEntry>,
    tier_counts: HashMap<String, usize>,
    complexity_tiers: HashMap<String, HashMap<String, usize>>,
    forward: usize,
    lateral: usize,
    feedback: usize,
    temporal: usize,
    lateral_count: u32,
    lateral_prevented: u32,
    feedback_signals: usize,
}

/// Run a text pass and return structured results.
fn run_text_pass(
    resolver: &mut HierarchicalResolver,
    encoder: &Encoder,
    sentences: &[(&str, &str)],
    pass_label: &str,
) -> TextPassResult {
    let mut text_entries: Vec<TextLogEntry> = Vec::new();
    let mut text_tiers: HashMap<String, usize> = HashMap::new();
    text_tiers.insert("Surface".to_string(), 0);
    text_tiers.insert("Reasoning".to_string(), 0);
    text_tiers.insert("Deep".to_string(), 0);

    let mut complexity_tiers: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for label in &["simple", "moderate", "complex"] {
        let mut m = HashMap::new();
        m.insert("Surface".to_string(), 0);
        m.insert("Reasoning".to_string(), 0);
        m.insert("Deep".to_string(), 0);
        complexity_tiers.insert(label.to_string(), m);
    }

    let mut total_forward = 0usize;
    let mut total_lateral = 0usize;
    let mut total_feedback = 0usize;
    let mut total_temporal = 0usize;
    let mut total_lateral_count = 0u32;
    let mut total_lateral_prevented = 0u32;
    let mut total_feedback_signals = 0usize;

    for (sentence, complexity) in sentences {
        let tensor = encoder.encode_text_readonly(sentence);
        let token_ids = encoder.tokeniser.tokenise_readonly(sentence);
        let result = resolver.resolve(&tensor);

        let tier_name = result.tier_reached.name().to_string();
        *text_tiers.get_mut(&tier_name).unwrap() += 1;
        *complexity_tiers
            .get_mut(*complexity)
            .unwrap()
            .get_mut(&tier_name)
            .unwrap() += 1;

        let (fwd, lat, fb, tmp) = count_directions(&result.route.trace_steps);
        total_forward += fwd;
        total_lateral += lat;
        total_feedback += fb;
        total_temporal += tmp;
        total_lateral_count += result.route.lateral_count;
        total_lateral_prevented += result.route.lateral_prevented_escalation;
        total_feedback_signals += result.feedback_signals.len();

        text_entries.push(TextLogEntry {
            sentence: sentence.to_string(),
            token_count: token_ids.len(),
            input_hash: tensor.content_hash(),
            execution_trace: result.route.execution_trace,
            confidence: result.route.confidence,
            compute_cost: result.route.total_compute_cost,
            cache_hits: result.route.cache_hits,
            tier_reached: tier_name,
            from_cache: result.from_cache,
            lateral_count: result.route.lateral_count,
            lateral_prevented: result.route.lateral_prevented_escalation,
            feedback_count: result.feedback_signals.len(),
            forward_steps: fwd,
            lateral_steps: lat,
            feedback_steps: fb,
            temporal_steps: tmp,
            winning_path: result.winning_path.clone(),
        });
    }

    let _ = pass_label; // used for display outside

    TextPassResult {
        entries: text_entries,
        tier_counts: text_tiers,
        complexity_tiers,
        forward: total_forward,
        lateral: total_lateral,
        feedback: total_feedback,
        temporal: total_temporal,
        lateral_count: total_lateral_count,
        lateral_prevented: total_lateral_prevented,
        feedback_signals: total_feedback_signals,
    }
}

fn print_tier_summary(label: &str, tier_counts: &HashMap<String, usize>, total: usize) {
    println!("  {} Tier Distribution:", label);
    for tier in &["Surface", "Reasoning", "Deep"] {
        let count = tier_counts.get(*tier).copied().unwrap_or(0);
        println!(
            "    {:>10}: {:>4} ({:.1}%)",
            tier,
            count,
            count as f32 / total as f32 * 100.0
        );
    }
}

fn print_direction_summary(forward: usize, lateral: usize, feedback: usize, temporal: usize) {
    let total = (forward + lateral + feedback + temporal).max(1);
    println!("  Traversal Directions:");
    println!(
        "    Forward:  {:>5} ({:.1}%)",
        forward,
        forward as f32 / total as f32 * 100.0
    );
    println!(
        "    Lateral:  {:>5} ({:.1}%)",
        lateral,
        lateral as f32 / total as f32 * 100.0
    );
    println!(
        "    Feedback: {:>5} ({:.1}%)",
        feedback,
        feedback as f32 / total as f32 * 100.0
    );
    println!(
        "    Temporal: {:>5} ({:.1}%)",
        temporal,
        temporal as f32 / total as f32 * 100.0
    );
}

fn print_complexity_breakdown(complexity_tiers: &HashMap<String, HashMap<String, usize>>) {
    println!("  Per-complexity breakdown:");
    for label in &["simple", "moderate", "complex"] {
        let counts = &complexity_tiers[*label];
        let n: usize = counts.values().sum();
        if n == 0 {
            continue;
        }
        let s = counts["Surface"] as f32 / n as f32 * 100.0;
        let r = counts["Reasoning"] as f32 / n as f32 * 100.0;
        let d = counts["Deep"] as f32 / n as f32 * 100.0;
        println!(
            "    {:>10}: S {:.0}%  R {:.0}%  D {:.0}%  (n={})",
            label, s, r, d, n
        );
    }
}

fn print_synth_results(
    label: &str,
    result: &SyntheticPassResult,
    iterations: usize,
    cache_rate: f32,
    elapsed: Duration,
) {
    println!();
    println!("  {} results ({:.2?}):", label, elapsed);
    print_tier_summary(label, &result.tier_counts, iterations);
    println!(
        "  Cache hit rate: {:.1}%  Avg cost: {:.4}  Avg conf: {:.4}",
        cache_rate,
        result.total_cost / iterations as f32,
        result.total_confidence / iterations as f32,
    );
    print_direction_summary(result.forward, result.lateral, result.feedback, result.temporal);
    println!(
        "  Lateral: {} attempted, {} prevented escalation",
        result.lateral_count, result.lateral_prevented
    );
    println!("  Feedback signals: {}", result.feedback_signals);

    if !result.weight_drift.is_empty() {
        let first = result.weight_drift.first().unwrap().weight_norm;
        let last = result.weight_drift.last().unwrap().weight_norm;
        let drift = (last - first).abs();
        let drift_pct = if first > 0.0 { drift / first * 100.0 } else { 0.0 };
        println!(
            "  Weight drift: {:.4} → {:.4} (delta {:.4}, {:.2}%)",
            first, last, drift, drift_pct
        );
    }
}

fn print_text_results(label: &str, result: &TextPassResult, n_sentences: usize) {
    println!();
    print_tier_summary(label, &result.tier_counts, n_sentences);
    println!();
    print_complexity_breakdown(&result.complexity_tiers);
    println!();
    print_direction_summary(result.forward, result.lateral, result.feedback, result.temporal);
    println!(
        "  Lateral: {} attempted, {} prevented escalation",
        result.lateral_count, result.lateral_prevented
    );
    println!("  Feedback signals: {}", result.feedback_signals);
}

fn write_json_log(path: &str, entries: &impl Serialize) {
    let log_file = File::create(path).unwrap_or_else(|e| panic!("Failed to create {}: {}", path, e));
    let mut writer = BufWriter::new(log_file);
    let json = serde_json::to_string_pretty(entries).unwrap();
    writeln!(writer, "{}", json).unwrap();
    writer.flush().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dashboard_mode = args.contains(&"--dashboard".to_string());
    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║        AXIOM Phase 4 — Weight Learning Bench        ║");
    println!("║  Hebbian Learning + Temporal Geometry + Four Passes  ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    let input_dim = 64;
    let synth_iterations = 10_000;
    let learning_rate = 0.001;

    // Load or default config
    let config = AxiomConfig::load_or_default();
    let config_source = if std::path::Path::new("axiom_config.json").exists() {
        "axiom_config.json"
    } else {
        "defaults"
    };
    println!("  Config source: {}", config_source);
    println!(
        "    surface_threshold: {:.2}  reasoning_threshold: {:.2}",
        config.surface_confidence_threshold, config.reasoning_confidence_threshold
    );
    println!(
        "    reasoning_base_conf: {:.2}  cache_sim_threshold: {:.2}",
        config.reasoning_base_confidence, config.cache_similarity_threshold
    );
    if config_source != "defaults" {
        println!("    rationale: {}", config.rationale);
    }
    println!();
    println!("  Config: dim={}, synth_iterations={}, learning_rate={}", input_dim, synth_iterations, learning_rate);

    // Build encoder early (needed for text passes and recalibration)
    let sentences = test_sentences();
    let mut tokeniser = Tokeniser::default_tokeniser();
    for (sentence, _) in &sentences {
        tokeniser.tokenise(sentence);
    }
    tokeniser.save_vocab("axiom_vocab.json").ok();
    let encoder = Encoder::new(input_dim, tokeniser);
    println!("  Vocabulary: {} tokens", encoder.tokeniser.vocab_size());

    // Dashboard setup
    let state = Arc::new(Mutex::new(DashboardState::new(synth_iterations)));
    if dashboard_mode {
        dashboard::start_server(Arc::clone(&state), port);
        println!();
        println!("  Dashboard: http://localhost:{}", port);
        println!("  Open in your browser, then press Enter to start...");
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).ok();
    }

    // Build resolver from config (includes calibration pass)
    let mut resolver = HierarchicalResolver::build_with_axiom_config(input_dim, &config);
    let weight_count = resolver.total_weight_count();
    let initial_weight_norm = resolver.total_weight_norm();
    println!("  Total trainable parameters: {}", weight_count);
    println!(
        "  Calibrated thresholds: surface={:.4}  reasoning={:.4}",
        resolver.config.surface_confidence_threshold,
        resolver.config.reasoning_confidence_threshold,
    );
    println!("  Initial weight norm: {:.4}", initial_weight_norm);
    println!();

    // ═══════════════════════════════════════════════════════
    // PASS 1 — Synthetic (learning active)
    // ═══════════════════════════════════════════════════════
    println!("─── Pass 1: Synthetic ({} iterations, lr={}) ───", synth_iterations, learning_rate);

    let multi = MultiProgress::new();
    let progress = multi.add(ProgressBar::new(synth_iterations as u64));
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} [{elapsed_precise}] [{bar:40.green/dark_gray}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("█▓▒░  "),
    );
    let stats_bar = multi.add(ProgressBar::new_spinner());
    stats_bar.set_style(ProgressStyle::with_template("{spinner:.yellow} {msg}").unwrap());

    let bench_start = Instant::now();
    let pass1 = run_synthetic_pass(
        &mut resolver,
        input_dim,
        synth_iterations,
        learning_rate,
        &progress,
        &stats_bar,
        dashboard_mode,
        &state,
        &bench_start,
    );
    let pass1_elapsed = bench_start.elapsed();

    progress.finish_with_message("done");
    stats_bar.finish_and_clear();

    write_json_log("axiom_synth_pass1.json", &pass1.entries);
    print_synth_results("Synthetic Pass 1", &pass1, synth_iterations, resolver.cache_hit_rate() * 100.0, pass1_elapsed);
    println!();

    // ═══════════════════════════════════════════════════════
    // RECALIBRATE after Pass 1
    // ═══════════════════════════════════════════════════════
    println!("─── Recalibration after learning ───");
    resolver.calibrate(input_dim, 0.65, 0.35);
    resolver.rebuild_graph_edges();
    println!(
        "  New thresholds: surface={:.4}  reasoning={:.4}",
        resolver.config.surface_confidence_threshold,
        resolver.config.reasoning_confidence_threshold,
    );
    println!();

    // ═══════════════════════════════════════════════════════
    // PASS 2 — Text (no learning, evaluation only)
    // ═══════════════════════════════════════════════════════
    println!("─── Pass 2: Text ({} sentences, post-learning) ───", sentences.len());

    // Reset cache for text pass
    resolver.cache = axiom_core::cache::EmbeddingCache::new(256, config.cache_similarity_threshold);

    let pass2 = run_text_pass(&mut resolver, &encoder, &sentences, "Pass 2");
    write_json_log("axiom_text_pass2.json", &pass2.entries);
    print_text_results("Text Pass 2 (post-learning)", &pass2, sentences.len());
    println!();

    // ═══════════════════════════════════════════════════════
    // PASS 3 — Synthetic (learning active, second round)
    // ═══════════════════════════════════════════════════════
    println!("─── Pass 3: Synthetic ({} iterations, lr={}) ───", synth_iterations, learning_rate);

    // Reset cache for second synthetic pass
    resolver.cache = axiom_core::cache::EmbeddingCache::new(256, config.cache_similarity_threshold);

    let progress3 = multi.add(ProgressBar::new(synth_iterations as u64));
    progress3.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} [{elapsed_precise}] [{bar:40.green/dark_gray}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("█▓▒░  "),
    );
    let stats_bar3 = multi.add(ProgressBar::new_spinner());
    stats_bar3.set_style(ProgressStyle::with_template("{spinner:.yellow} {msg}").unwrap());

    let pass3_start = Instant::now();
    let pass3 = run_synthetic_pass(
        &mut resolver,
        input_dim,
        synth_iterations,
        learning_rate,
        &progress3,
        &stats_bar3,
        dashboard_mode,
        &state,
        &pass3_start,
    );
    let pass3_elapsed = pass3_start.elapsed();

    progress3.finish_with_message("done");
    stats_bar3.finish_and_clear();

    write_json_log("axiom_synth_pass3.json", &pass3.entries);
    print_synth_results("Synthetic Pass 3", &pass3, synth_iterations, resolver.cache_hit_rate() * 100.0, pass3_elapsed);
    println!();

    // ═══════════════════════════════════════════════════════
    // PASS 4 — Text (no learning, final evaluation)
    // ═══════════════════════════════════════════════════════
    println!("─── Pass 4: Text ({} sentences, post-learning x2) ───", sentences.len());

    // Reset cache for final text pass
    resolver.cache = axiom_core::cache::EmbeddingCache::new(256, config.cache_similarity_threshold);

    // Recalibrate again after Pass 3
    resolver.calibrate(input_dim, 0.65, 0.35);
    resolver.rebuild_graph_edges();
    println!(
        "  Final thresholds: surface={:.4}  reasoning={:.4}",
        resolver.config.surface_confidence_threshold,
        resolver.config.reasoning_confidence_threshold,
    );

    let pass4 = run_text_pass(&mut resolver, &encoder, &sentences, "Pass 4");
    write_json_log("axiom_text_pass4.json", &pass4.entries);
    print_text_results("Text Pass 4 (post-learning x2)", &pass4, sentences.len());
    println!();

    // ═══════════════════════════════════════════════════════
    // PASS 2 vs PASS 4 Comparison
    // ═══════════════════════════════════════════════════════
    println!("─── Text Comparison: Pass 2 → Pass 4 ───");
    for tier in &["Surface", "Reasoning", "Deep"] {
        let p2 = pass2.tier_counts.get(*tier).copied().unwrap_or(0);
        let p4 = pass4.tier_counts.get(*tier).copied().unwrap_or(0);
        let delta = p4 as i32 - p2 as i32;
        let arrow = if delta > 0 { "+" } else if delta < 0 { "" } else { " " };
        println!(
            "    {:>10}: {:>2} → {:>2} ({}{:>2})",
            tier, p2, p4, arrow, delta
        );
    }
    println!();

    // Per-complexity comparison
    println!("  Complexity ordering comparison:");
    for label in &["simple", "moderate", "complex"] {
        let p2_counts = &pass2.complexity_tiers[*label];
        let p4_counts = &pass4.complexity_tiers[*label];
        let p2_n: usize = p2_counts.values().sum();
        let p4_n: usize = p4_counts.values().sum();
        let p2_s = if p2_n > 0 { p2_counts["Surface"] as f32 / p2_n as f32 * 100.0 } else { 0.0 };
        let p4_s = if p4_n > 0 { p4_counts["Surface"] as f32 / p4_n as f32 * 100.0 } else { 0.0 };
        println!(
            "    {:>10}: S {:.0}% → S {:.0}%",
            label, p2_s, p4_s
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════
    // WEIGHT DRIFT SUMMARY
    // ═══════════════════════════════════════════════════════
    println!("─── Weight Drift Summary ───");
    let final_weight_norm = resolver.total_weight_norm();
    let total_drift = (final_weight_norm - initial_weight_norm).abs();
    let total_drift_pct = if initial_weight_norm > 0.0 { total_drift / initial_weight_norm * 100.0 } else { 0.0 };
    println!("  Initial:  {:.4}", initial_weight_norm);
    if let Some(last_p1) = pass1.weight_drift.last() {
        println!("  After P1: {:.4}", last_p1.weight_norm);
    }
    if let Some(last_p3) = pass3.weight_drift.last() {
        println!("  After P3: {:.4}", last_p3.weight_norm);
    }
    println!("  Final:    {:.4}", final_weight_norm);
    println!("  Total drift: {:.4} ({:.2}%)", total_drift, total_drift_pct);

    // Auto-tuner (runs on pass1 synthetic log for config persistence)
    println!();
    println!("─── Auto-Tuner ───");
    // Write pass1 entries as the bench log for the tuner
    write_json_log("axiom_bench_log.json", &pass1.entries);
    match axiom_tuner::compute_stats("axiom_bench_log.json") {
        Ok(stats) => {
            let recommended = axiom_tuner::tune(&stats, &config);
            println!("  Rationale: {}", recommended.rationale);
            match recommended.save() {
                Ok(()) => println!("  Written: axiom_config.json"),
                Err(e) => eprintln!("  Failed to write config: {}", e),
            }
        }
        Err(e) => {
            eprintln!("  Tuner error: {}", e);
        }
    }

    // Mark dashboard as done
    if dashboard_mode {
        state.lock().unwrap().done = true;
    }

    // ═══════════════════════════════════════════════════════
    // FINAL SUMMARY
    // ═══════════════════════════════════════════════════════
    println!();
    println!("═══════════════════ FINAL SUMMARY ═══════════════════");
    println!("  Total parameters:   {}", weight_count);
    println!("  Learning rate:      {}", learning_rate);
    println!("  Synth iterations:   {} x 2 = {}", synth_iterations, synth_iterations * 2);
    println!("  Text sentences:     {} x 2 = {}", sentences.len(), sentences.len() * 2);
    println!("  Weight drift:       {:.4} ({:.2}%)", total_drift, total_drift_pct);
    println!("  Logs:");
    println!("    axiom_synth_pass1.json");
    println!("    axiom_text_pass2.json");
    println!("    axiom_synth_pass3.json");
    println!("    axiom_text_pass4.json");
    println!(
        "  Vocabulary: axiom_vocab.json ({} tokens)",
        encoder.tokeniser.vocab_size()
    );
    println!("  Config: axiom_config.json");
    println!("═════════════════════════════════════════════════════");

    if dashboard_mode {
        println!();
        println!("  Dashboard still live at http://localhost:{}", port);
        println!("  Press Ctrl+C to exit.");
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

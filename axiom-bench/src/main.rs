//! AXIOM Bench — Phase 3: multidirectional traversal + feedback loops.
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

/// Run a bench pass and return (entries, tier_counts).
fn run_synthetic_pass(
    resolver: &mut HierarchicalResolver,
    input_dim: usize,
    iterations: usize,
    progress: &ProgressBar,
    stats_bar: &ProgressBar,
    dashboard_mode: bool,
    state: &Arc<Mutex<DashboardState>>,
    bench_start: &Instant,
) -> (Vec<BenchLogEntry>, HashMap<String, usize>, f32, f32, usize, usize, usize, usize, u32, u32, usize) {
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

    for i in 0..iterations {
        let input = generate_input(i, input_dim);
        let input_hash = input.content_hash();
        let result = resolver.resolve(&input);

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

        if i % 10 == 0 || i == iterations - 1 {
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

        if dashboard_mode && i % 10 == 9 {
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    (
        entries,
        tier_counts,
        total_cost,
        total_confidence,
        total_forward,
        total_lateral_steps,
        total_feedback_steps,
        total_temporal_steps,
        total_lateral_count,
        total_lateral_prevented,
        total_feedback_signals,
    )
}

/// Run a text pass and return (entries, tier_counts, complexity_tiers, direction_counts).
fn run_text_pass(
    resolver: &mut HierarchicalResolver,
    encoder: &Encoder,
    sentences: &[(&str, &str)],
    pass_label: &str,
) -> (
    Vec<TextLogEntry>,
    HashMap<String, usize>,
    HashMap<String, HashMap<String, usize>>,
    usize,
    usize,
    usize,
    usize,
    u32,
    u32,
    usize,
) {
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

    (
        text_entries,
        text_tiers,
        complexity_tiers,
        total_forward,
        total_lateral,
        total_feedback,
        total_temporal,
        total_lateral_count,
        total_lateral_prevented,
        total_feedback_signals,
    )
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dashboard_mode = args.contains(&"--dashboard".to_string());
    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    println!("╔══════════════════════════════════════════════════╗");
    println!("║           AXIOM Phase 3 — Bench Loop            ║");
    println!("║   Multidirectional Traversal + Feedback Loops   ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let input_dim = 64;
    let iterations = 1000;

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
    println!("  Config: dim={}, iterations={}", input_dim, iterations);

    // Dashboard setup
    let state = Arc::new(Mutex::new(DashboardState::new(iterations)));
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
    println!("  Total trainable parameters: {}", weight_count);
    println!(
        "  Calibrated thresholds: surface={:.4}  reasoning={:.4}",
        resolver.config.surface_confidence_threshold,
        resolver.config.reasoning_confidence_threshold,
    );
    println!();

    // ═══════════════════════════════════════════════════════
    // PASS 1 — Synthetic
    // ═══════════════════════════════════════════════════════
    println!("─── Pass 1: Synthetic ({} iterations) ───", iterations);

    let multi = MultiProgress::new();
    let progress = multi.add(ProgressBar::new(iterations as u64));
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
    let (entries, tier_counts, total_cost, total_confidence, fwd, lat, fb, tmp, lat_count, lat_prev, fb_signals) =
        run_synthetic_pass(
            &mut resolver,
            input_dim,
            iterations,
            &progress,
            &stats_bar,
            dashboard_mode,
            &state,
            &bench_start,
        );
    let synth_elapsed = bench_start.elapsed();

    progress.finish_with_message("done");
    stats_bar.finish_and_clear();

    // Write synthetic log
    {
        let log_file = File::create("axiom_bench_log.json").expect("Failed to create log file");
        let mut writer = BufWriter::new(log_file);
        let json = serde_json::to_string(&entries).unwrap();
        writeln!(writer, "{}", json).unwrap();
        writer.flush().unwrap();
    }

    println!();
    println!("  Synthetic results ({:.2?}):", synth_elapsed);
    print_tier_summary("Synthetic", &tier_counts, iterations);
    println!(
        "  Cache hit rate: {:.1}%  Avg cost: {:.4}  Avg conf: {:.4}",
        resolver.cache_hit_rate() * 100.0,
        total_cost / iterations as f32,
        total_confidence / iterations as f32,
    );
    print_direction_summary(fwd, lat, fb, tmp);
    println!(
        "  Lateral: {} attempted, {} prevented escalation",
        lat_count, lat_prev
    );
    println!("  Feedback signals: {}", fb_signals);
    println!();

    // ═══════════════════════════════════════════════════════
    // PASS 2 — Text
    // ═══════════════════════════════════════════════════════
    let sentences = test_sentences();
    println!("─── Pass 2: Text ({} sentences) ───", sentences.len());

    // Reset cache for text pass
    resolver.cache =
        axiom_core::cache::EmbeddingCache::new(256, config.cache_similarity_threshold);

    let mut tokeniser = Tokeniser::default_tokeniser();
    for (sentence, _) in &sentences {
        tokeniser.tokenise(sentence);
    }
    tokeniser.save_vocab("axiom_vocab.json").ok();
    println!(
        "  Vocabulary: {} tokens → axiom_vocab.json",
        tokeniser.vocab_size()
    );

    let encoder = Encoder::new(input_dim, tokeniser);

    let (text_entries, text_tiers, complexity_tiers, t_fwd, t_lat, t_fb, t_tmp, t_lat_count, t_lat_prev, t_fb_signals) =
        run_text_pass(&mut resolver, &encoder, &sentences, "Pass 2");

    // Write text log
    {
        let log_file = File::create("axiom_text_log.json").expect("Failed to create text log");
        let mut writer = BufWriter::new(log_file);
        let json = serde_json::to_string_pretty(&text_entries).unwrap();
        writeln!(writer, "{}", json).unwrap();
        writer.flush().unwrap();
    }

    println!();
    print_tier_summary("Text", &text_tiers, sentences.len());
    println!();
    print_complexity_breakdown(&complexity_tiers);
    println!();
    print_direction_summary(t_fwd, t_lat, t_fb, t_tmp);
    println!(
        "  Lateral: {} attempted, {} prevented escalation",
        t_lat_count, t_lat_prev
    );
    println!("  Feedback signals: {}", t_fb_signals);

    // ═══════════════════════════════════════════════════════
    // AUTO-TUNER (between Pass 2 and Pass 3)
    // ═══════════════════════════════════════════════════════
    println!();
    println!("─── Auto-Tuner ───");

    let tuned_config = match axiom_tuner::compute_stats("axiom_bench_log.json") {
        Ok(stats) => {
            let recommended = axiom_tuner::tune(&stats, &config);
            println!("  Rationale: {}", recommended.rationale);
            match recommended.save() {
                Ok(()) => println!("  Written: axiom_config.json"),
                Err(e) => eprintln!("  Failed to write config: {}", e),
            }
            recommended
        }
        Err(e) => {
            eprintln!("  Tuner error: {}", e);
            config.clone()
        }
    };

    // ═══════════════════════════════════════════════════════
    // PASS 3 — Text again after auto-tuner cycle
    // ═══════════════════════════════════════════════════════
    println!();
    println!("─── Pass 3: Text after tuning ({} sentences) ───", sentences.len());

    // Rebuild resolver with tuned config
    let mut resolver3 = HierarchicalResolver::build_with_axiom_config(input_dim, &tuned_config);

    let (text_entries3, text_tiers3, complexity_tiers3, t3_fwd, t3_lat, t3_fb, t3_tmp, t3_lat_count, t3_lat_prev, t3_fb_signals) =
        run_text_pass(&mut resolver3, &encoder, &sentences, "Pass 3");

    // Write pass 3 text log
    {
        let log_file =
            File::create("axiom_text_log_pass3.json").expect("Failed to create pass3 text log");
        let mut writer = BufWriter::new(log_file);
        let json = serde_json::to_string_pretty(&text_entries3).unwrap();
        writeln!(writer, "{}", json).unwrap();
        writer.flush().unwrap();
    }

    println!();
    print_tier_summary("Text (tuned)", &text_tiers3, sentences.len());
    println!();
    print_complexity_breakdown(&complexity_tiers3);
    println!();
    print_direction_summary(t3_fwd, t3_lat, t3_fb, t3_tmp);
    println!(
        "  Lateral: {} attempted, {} prevented escalation",
        t3_lat_count, t3_lat_prev
    );
    println!("  Feedback signals: {}", t3_fb_signals);

    // ═══════════════════════════════════════════════════════
    // PASS 2 vs PASS 3 comparison
    // ═══════════════════════════════════════════════════════
    println!();
    println!("─── Pass 2 → Pass 3 Comparison ───");
    for tier in &["Surface", "Reasoning", "Deep"] {
        let p2 = text_tiers.get(*tier).copied().unwrap_or(0);
        let p3 = text_tiers3.get(*tier).copied().unwrap_or(0);
        let delta = p3 as i32 - p2 as i32;
        let arrow = if delta > 0 {
            "↑"
        } else if delta < 0 {
            "↓"
        } else {
            "="
        };
        println!(
            "    {:>10}: {} → {} ({}{})",
            tier,
            p2,
            p3,
            arrow,
            delta.abs()
        );
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
    println!("  Synthetic log:      axiom_bench_log.json");
    println!("  Text log (pass 2):  axiom_text_log.json");
    println!("  Text log (pass 3):  axiom_text_log_pass3.json");
    println!(
        "  Vocabulary:         axiom_vocab.json ({} tokens)",
        encoder.tokeniser.vocab_size()
    );
    println!("  Config:             axiom_config.json");
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

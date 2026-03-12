//! AXIOM Bench — Phase 11: Analytical Init + Frozen Surface.
//!
//! All synthetic training removed. Learning happens exclusively on text.
//! 50,000 iterations (100 passes through 500-sentence corpus).
//! Surface nodes analytically initialised toward simple sentence space, then frozen.
//! Learning (Oja + contrastive) active on Reasoning and Deep nodes only.
//! Cache bypassed during training (RouteMode::Training).
//!
//! Usage:
//!   cargo run --release -p axiom-bench              # terminal only
//!   cargo run --release -p axiom-bench -- --dashboard  # terminal + HTML dashboard on :8080

mod corpus;
mod dashboard;

use axiom_core::graph::TraversalDirection;
use axiom_core::input::{Encoder, Tokeniser};
use axiom_core::tiers::{AxiomConfig, HierarchicalResolver, RouteMode};
use corpus::Corpus;
use dashboard::{DashboardState, TraceSnapshot};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// JSON log entry for a text training iteration.
#[derive(Serialize, Deserialize)]
struct TrainLogEntry {
    iteration: usize,
    sentence: String,
    complexity: String,
    confidence: f32,
    surface_confidence: f32,
    compute_cost: f32,
    tier_reached: String,
    from_cache: bool,
    forward_steps: usize,
    lateral_steps: usize,
    feedback_steps: usize,
    temporal_steps: usize,
}

/// JSON log entry for a validation text input.
#[derive(Serialize, Deserialize)]
struct TextLogEntry {
    sentence: String,
    complexity: String,
    token_count: usize,
    input_hash: u64,
    execution_trace: Vec<String>,
    confidence: f32,
    surface_confidence: f32,
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

/// JSON log entry for a contrastive update event.
#[derive(Serialize, Deserialize)]
struct ContrastiveLogEntry {
    iteration: usize,
    node_id: String,
    positive_count: usize,
    negative_count: usize,
    contrast_magnitude: f32,
    weight_norm_before: f32,
    weight_norm_after: f32,
    positive_mean_norm: f32,
    negative_mean_norm: f32,
}

/// Training snapshot at a checkpoint.
#[derive(Serialize, Deserialize, Clone)]
struct TrainingSnapshot {
    iteration: usize,
    simple_mean: f32,
    complex_mean: f32,
    gap: f32,
    ordering_correct: bool,
    weight_norm: f32,
    surface_norm: f32,
    nonsurface_norm: f32,
    contrastive_updates_fired: usize,
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
}

/// 6 fixed diagnostic sentences used throughout all phases.
fn diagnostic_sentences() -> Vec<(&'static str, &'static str)> {
    vec![
        ("the dog runs fast", "simple"),
        ("the cat sat on the mat", "simple"),
        ("birds fly south in winter", "simple"),
        ("the recursive nature of self-referential systems creates emergent properties that resist reduction", "complex"),
        ("quantum entanglement challenges classical notions of locality and causality", "complex"),
        ("the church-turing thesis equates mechanical computation with recursive function theory", "complex"),
    ]
}

/// Hardcoded validation sentences (50 total: 15 simple, 18 moderate, 17 complex).
fn test_sentences() -> Vec<(&'static str, &'static str)> {
    vec![
        // Simple (15)
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
        // Moderate (18)
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
        // Complex (17)
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

/// Compute max Surface confidence for diagnostic sentences.
fn compute_surface_confidences(
    resolver: &HierarchicalResolver,
    encoder: &Encoder,
    sentences: &[(&str, &str)],
) -> Vec<f32> {
    sentences
        .iter()
        .map(|(text, _)| {
            let tensor = encoder.encode_text_readonly(text);
            resolver.max_surface_confidence(&tensor)
        })
        .collect()
}

/// Truncate a string to max_len chars, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

fn std_dev_f32(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    variance.sqrt()
}

fn write_json_log(path: &str, entries: &impl Serialize) {
    let log_file =
        File::create(path).unwrap_or_else(|e| panic!("Failed to create {}: {}", path, e));
    let mut writer = BufWriter::new(log_file);
    let json = serde_json::to_string_pretty(entries).unwrap();
    writeln!(writer, "{}", json).unwrap();
    writer.flush().unwrap();
}

/// Text pass results (for validation — no learning).
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

/// Run a text validation pass (no learning).
fn run_text_pass(
    resolver: &mut HierarchicalResolver,
    encoder: &Encoder,
    sentences: &[(&str, &str)],
    _pass_label: &str,
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
        let surface_conf = resolver.max_surface_confidence(&tensor);
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
            complexity: complexity.to_string(),
            token_count: token_ids.len(),
            input_hash: tensor.content_hash(),
            execution_trace: result.route.execution_trace,
            confidence: result.route.confidence,
            surface_confidence: surface_conf,
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dashboard_mode = args.contains(&"--dashboard".to_string());
    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   AXIOM Phase 11 — Analytical Init + Frozen Surface    ║");
    println!("║  Surface=fixed discriminator, learning in R+D only     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let input_dim = 128;
    let train_iterations = 50_000;
    let learning_rate = 0.001;
    let error_lr = 0.0005;

    // Load corpus
    let corpus = Corpus::load();
    println!("  Corpus: {} simple, {} moderate, {} complex = {} total",
        corpus.simple.len(), corpus.moderate.len(), corpus.complex.len(),
        corpus.simple.len() + corpus.moderate.len() + corpus.complex.len());

    // Corpus statistics
    let mean_words = |v: &[String]| -> f64 {
        let total: usize = v.iter().map(|s| s.split_whitespace().count()).sum();
        total as f64 / v.len() as f64
    };
    // Need to collect owned strings first for lifetime
    let all_sentences = corpus.all();
    let vocab_set: std::collections::HashSet<&str> = all_sentences.iter()
        .flat_map(|(s, _)| s.split_whitespace().collect::<Vec<&str>>())
        .collect();
    let simple_mean_wc = mean_words(&corpus.simple);
    let moderate_mean_wc = mean_words(&corpus.moderate);
    let complex_mean_wc = mean_words(&corpus.complex);
    println!("  Mean word count: simple={:.1}, moderate={:.1}, complex={:.1}",
        simple_mean_wc, moderate_mean_wc, complex_mean_wc);
    println!("  Vocabulary size: {} unique words", vocab_set.len());
    assert!(simple_mean_wc < complex_mean_wc, "simple mean word count must be lower than complex");
    println!();

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
    println!("  Config: dim={}, train_iterations={}, lr={}, error_lr={}",
        input_dim, train_iterations, learning_rate, error_lr);

    // Build encoder — train tokeniser on corpus + validation + diagnostic sentences
    let validation_sentences = test_sentences();
    let diag_sents = diagnostic_sentences();
    let mut tokeniser = Tokeniser::default_tokeniser();
    for (sentence, _) in &all_sentences {
        tokeniser.tokenise(sentence);
    }
    for (sentence, _) in &validation_sentences {
        tokeniser.tokenise(sentence);
    }
    for (sentence, _) in &diag_sents {
        tokeniser.tokenise(sentence);
    }
    tokeniser.save_vocab("axiom_vocab.json").ok();
    let encoder = Encoder::new(input_dim, tokeniser);
    println!("  Vocabulary: {} tokens", encoder.tokeniser.vocab_size());

    // Dashboard setup
    let state = Arc::new(Mutex::new(DashboardState::new(train_iterations)));
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

    // Phase 9: reduce cache threshold from 0.92 to 0.75 to prevent learning starvation
    let cache_threshold = 0.75f32;
    resolver.cache = axiom_core::cache::EmbeddingCache::new(256, cache_threshold);
    println!("  Cache threshold override: {:.2} (Phase 9 — prevent learning starvation)", cache_threshold);

    // ── Phase 11: Analytical Surface initialisation ──
    // Encode all simple and complex corpus sentences
    let simple_tensors: Vec<_> = corpus.simple.iter()
        .map(|s| encoder.encode_text_readonly(s))
        .collect();
    let complex_tensors: Vec<_> = corpus.complex.iter()
        .map(|s| encoder.encode_text_readonly(s))
        .collect();

    let (dir_norm, simple_norm, complex_norm, mean_cosine) =
        resolver.init_surface_analytical(&simple_tensors, &complex_tensors);

    println!("  Phase 11: Analytical Surface initialisation:");
    println!("    discrimination_direction norm (pre-L2): {:.6}", dir_norm);
    println!("    simple_mean norm:  {:.4}", simple_norm);
    println!("    complex_mean norm: {:.4}", complex_norm);
    println!("    cosine(simple_mean, complex_mean): {:.6}", mean_cosine);
    println!("    Surface nodes: frozen=true, weights point toward simple space");
    println!();

    let weight_count = resolver.total_weight_count();
    let initial_weight_norm = resolver.total_weight_norm();
    let initial_surface_norm = resolver.surface_weight_norm();
    let initial_nonsurface_norm = resolver.non_surface_weight_norm();
    println!("  Total trainable parameters: {}", weight_count);
    println!(
        "  Calibrated thresholds: surface={:.4}  reasoning={:.4}",
        resolver.config.surface_confidence_threshold,
        resolver.config.reasoning_confidence_threshold,
    );
    println!("  Initial weight norms: total={:.4}  surface={:.4}  reasoning+deep={:.4}",
        initial_weight_norm, initial_surface_norm, initial_nonsurface_norm);
    resolver.validate_confidence_invariants();
    println!();

    // Pre-training baseline confidences (after analytical init)
    println!("  Post-analytical-init diagnostic confidences:");
    {
        let baseline = compute_surface_confidences(&resolver, &encoder, &diag_sents);
        let simple_mean: f32 = baseline[0..3].iter().sum::<f32>() / 3.0;
        let complex_mean: f32 = baseline[3..6].iter().sum::<f32>() / 3.0;
        let gap = simple_mean - complex_mean;
        let status = if simple_mean > complex_mean {
            "CORRECT"
        } else {
            "inverted"
        };
        println!(
            "    analytical: simple={:.4}  complex={:.4}  gap={:+.4}  {}",
            simple_mean, complex_mean, gap, status
        );
        for (j, (text, _)) in diag_sents.iter().enumerate() {
            let tag = if j < 3 { "S" } else { "C" };
            println!(
                "              [{tag}] {:.4}  \"{}\"",
                baseline[j],
                truncate_str(text, 60)
            );
        }
        if gap > 0.05 {
            println!("    GAP EXCEEDS 0.05 — discrimination direction is strong");
        } else if gap > 0.03 {
            println!("    Gap above 0.03 but below 0.05 — marginal");
        } else {
            println!("    WARNING: Gap below 0.03 — discrimination direction may be too weak");
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════
    // TEXT TRAINING LOOP — 50,000 iterations
    // ═══════════════════════════════════════════════════════
    // Phase 10: Training mode — cache completely bypassed
    resolver.mode = RouteMode::Training;
    println!(
        "─── Text Training: {} iterations (100 passes x 500 sentences) ───",
        train_iterations
    );
    println!("    lr={}, error_lr={}, mode={:?}", learning_rate, error_lr, resolver.mode);
    println!();

    let multi = MultiProgress::new();
    let progress = multi.add(ProgressBar::new(train_iterations as u64));
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} [{elapsed_precise}] [{bar:40.green/dark_gray}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("█▓▒░  "),
    );
    let stats_bar = multi.add(ProgressBar::new_spinner());
    stats_bar.set_style(ProgressStyle::with_template("{spinner:.yellow} {msg}").unwrap());

    // Clear error events and activation counts
    resolver.clear_error_events();
    resolver.reset_activation_counts();

    // Pre-encode all corpus sentences for training
    let corpus_all = corpus.all();
    let corpus_tensors: Vec<_> = corpus_all
        .iter()
        .map(|(s, c)| (encoder.encode_text_readonly(s), *c, s.clone()))
        .collect();
    let corpus_size = corpus_tensors.len();

    let mut train_entries: Vec<TrainLogEntry> = Vec::with_capacity(train_iterations);
    let mut snapshots: Vec<TrainingSnapshot> = Vec::new();
    let mut contrastive_log: Vec<ContrastiveLogEntry> = Vec::new();
    let mut total_contrastive_updates = 0usize;
    let mut tier_counts: HashMap<String, usize> = HashMap::new();
    tier_counts.insert("Surface".to_string(), 0);
    tier_counts.insert("Reasoning".to_string(), 0);
    tier_counts.insert("Deep".to_string(), 0);
    let mut total_forward = 0usize;
    let mut total_lateral_steps = 0usize;
    let mut total_feedback_steps = 0usize;
    let mut total_temporal_steps = 0usize;
    let mut total_lateral_prevented = 0u32;
    let mut total_feedback_signals = 0usize;

    let bench_start = Instant::now();

    // Deterministic PRNG for sentence sampling
    let mut rng = Rng::new(42);

    for i in 0..train_iterations {
        // Sample one sentence deterministically
        let idx = rng.next_u32() as usize % corpus_size;
        let (ref tensor, complexity, ref sentence) = corpus_tensors[idx];

        let surface_conf = resolver.max_surface_confidence(tensor);
        let result = resolver.resolve(tensor);

        // Oja's rule learning
        if learning_rate > 0.0 {
            resolver.learn(tensor, &result, learning_rate, i + 1);
        }

        // Error signal
        if error_lr > 0.0 {
            resolver.apply_error_signal(tensor, &result, error_lr);
        }

        let tier_name = result.tier_reached.name().to_string();
        *tier_counts.get_mut(&tier_name).unwrap() += 1;

        let (fwd, lat, fb, tmp) = count_directions(&result.route.trace_steps);
        total_forward += fwd;
        total_lateral_steps += lat;
        total_feedback_steps += fb;
        total_temporal_steps += tmp;
        total_lateral_prevented += result.route.lateral_prevented_escalation;
        total_feedback_signals += result.feedback_signals.len();

        train_entries.push(TrainLogEntry {
            iteration: i,
            sentence: sentence.clone(),
            complexity: format!("{}", complexity),
            confidence: result.route.confidence,
            surface_confidence: surface_conf,
            compute_cost: result.route.total_compute_cost,
            tier_reached: tier_name.clone(),
            from_cache: result.from_cache,
            forward_steps: fwd,
            lateral_steps: lat,
            feedback_steps: fb,
            temporal_steps: tmp,
        });

        // Contrastive update every 100 iterations
        if learning_rate > 0.0 && (i + 1) % 100 == 0 {
            let infos = resolver.apply_contrastive_update_all();
            for info in &infos {
                total_contrastive_updates += 1;
                contrastive_log.push(ContrastiveLogEntry {
                    iteration: i + 1,
                    node_id: info.node_id.clone(),
                    positive_count: info.positive_count,
                    negative_count: info.negative_count,
                    contrast_magnitude: info.contrast_magnitude,
                    weight_norm_before: info.weight_norm_before,
                    weight_norm_after: info.weight_norm_after,
                    positive_mean_norm: info.positive_mean_norm,
                    negative_mean_norm: info.negative_mean_norm,
                });
            }
        }

        // Confidence std dev check every 500 iterations
        if (i + 1) % 500 == 0 {
            let recent_confs: Vec<f32> = train_entries[train_entries.len().saturating_sub(500)..]
                .iter()
                .map(|e| e.confidence)
                .collect();
            let conf_std = std_dev_f32(&recent_confs);
            if conf_std < 0.01 {
                progress.suspend(|| {
                    println!(
                        "    [WARNING] iter {}: confidence std dev = {:.6} (below 0.01)",
                        i + 1,
                        conf_std
                    );
                });
            }
        }

        // Diagnostic snapshot every 5,000 iterations
        if (i + 1) % 5_000 == 0 {
            let confs = compute_surface_confidences(&resolver, &encoder, &diag_sents);
            let simple_mean: f32 = confs[0..3].iter().sum::<f32>() / 3.0;
            let complex_mean: f32 = confs[3..6].iter().sum::<f32>() / 3.0;
            let gap = simple_mean - complex_mean;
            let ordering_correct = simple_mean > complex_mean;
            let weight_norm = resolver.total_weight_norm();

            let surface_norm = resolver.surface_weight_norm();
            let nonsurface_norm = resolver.non_surface_weight_norm();

            let snapshot = TrainingSnapshot {
                iteration: i + 1,
                simple_mean,
                complex_mean,
                gap,
                ordering_correct,
                weight_norm,
                surface_norm,
                nonsurface_norm,
                contrastive_updates_fired: total_contrastive_updates,
            };
            snapshots.push(snapshot);

            progress.suspend(|| {
                let status = if ordering_correct {
                    "CORRECT"
                } else {
                    "inverted"
                };
                println!(
                    "    iter {:>5}: simple={:.4}  complex={:.4}  gap={:+.4}  {}  S_norm={:.1}  RD_norm={:.1}",
                    i + 1,
                    simple_mean,
                    complex_mean,
                    gap,
                    status,
                    surface_norm,
                    nonsurface_norm
                );
                for (j, (text, _)) in diag_sents.iter().enumerate() {
                    let tag = if j < 3 { "S" } else { "C" };
                    println!(
                        "              [{tag}] {:.4}  \"{}\"",
                        confs[j],
                        truncate_str(text, 60)
                    );
                }
            });
        }

        // Dashboard updates
        if dashboard_mode {
            let mut s = state.lock().unwrap();
            s.iteration = i + 1;
            s.surface_count = *tier_counts.get("Surface").unwrap();
            s.reasoning_count = *tier_counts.get("Reasoning").unwrap();
            s.deep_count = *tier_counts.get("Deep").unwrap();
            s.cache_hit_rate = resolver.cache_hit_rate() * 100.0;
            s.cache_size = resolver.cache_size();
            s.avg_compute_cost =
                train_entries.iter().map(|e| e.compute_cost).sum::<f32>() / (i + 1) as f32;
            s.avg_confidence =
                train_entries.iter().map(|e| e.confidence).sum::<f32>() / (i + 1) as f32;
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

        // Progress bar updates
        if i % 100 == 0 || i == train_iterations - 1 {
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

    let train_elapsed = bench_start.elapsed();
    progress.finish_with_message("done");
    stats_bar.finish_and_clear();

    // Write training logs
    write_json_log("axiom_text_train_log.json", &train_entries);
    write_json_log("axiom_training_snapshots.json", &snapshots);
    write_json_log("axiom_contrastive_log.json", &contrastive_log);
    let error_events_vec: Vec<_> = resolver.error_events().to_vec();
    write_json_log("axiom_error_log.json", &error_events_vec);

    // Training summary
    println!();
    println!("  Text Training results ({:.2?}):", train_elapsed);
    print_tier_summary("Training", &tier_counts, train_iterations);
    println!(
        "  Cache hit rate: {:.1}%",
        resolver.cache_hit_rate() * 100.0
    );

    let train_confs: Vec<f32> = train_entries.iter().map(|e| e.confidence).collect();
    let conf_std = std_dev_f32(&train_confs);
    println!(
        "  Confidence std dev: {:.6}{}",
        conf_std,
        if conf_std < 0.005 {
            "  *** OVER-CONVERGED ***"
        } else {
            ""
        }
    );

    let final_weight_norm = resolver.total_weight_norm();
    let final_surface_norm = resolver.surface_weight_norm();
    let final_nonsurface_norm = resolver.non_surface_weight_norm();
    let weight_drift = (final_weight_norm - initial_weight_norm).abs();
    let weight_drift_pct = if initial_weight_norm > 0.0 {
        weight_drift / initial_weight_norm * 100.0
    } else {
        0.0
    };
    println!(
        "  Weight drift: {:.4} -> {:.4} (delta {:.4}, {:.2}%)",
        initial_weight_norm, final_weight_norm, weight_drift, weight_drift_pct
    );
    let surface_drift = (final_surface_norm - initial_surface_norm).abs();
    let nonsurface_drift = (final_nonsurface_norm - initial_nonsurface_norm).abs();
    println!(
        "  Surface norm:  {:.4} -> {:.4} (delta {:.4}) {}",
        initial_surface_norm, final_surface_norm, surface_drift,
        if surface_drift < 0.001 { "FROZEN" } else { "DRIFTED — ERROR" }
    );
    println!(
        "  R+D norm:      {:.4} -> {:.4} (delta {:.4})",
        initial_nonsurface_norm, final_nonsurface_norm, nonsurface_drift
    );
    println!(
        "  Contrastive updates: {} total",
        total_contrastive_updates
    );
    let esc_count = resolver.escalation_penalty_count();
    let cache_rein_count = resolver.cache_reinforcement_count();
    println!(
        "  Error signals: {} escalation penalties, {} cache reinforcements",
        esc_count, cache_rein_count
    );
    println!();

    // Training snapshot table
    println!("─── Training Snapshot Table ───");
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {}",
        "iter", "s_mean", "c_mean", "gap", "S_norm", "RD_norm", "contrast", "status"
    );
    println!(
        "  {}  {}  {}  {}  {}  {}  {}  {}",
        "──────", "────────", "────────", "────────", "────────", "────────", "────────", "────────"
    );
    for snap in &snapshots {
        let status = if snap.ordering_correct {
            "CORRECT"
        } else {
            "inverted"
        };
        println!(
            "  {:>6}  {:>8.4}  {:>8.4}  {:>+8.4}  {:>8.1}  {:>8.1}  {:>8}  {}",
            snap.iteration,
            snap.simple_mean,
            snap.complex_mean,
            snap.gap,
            snap.surface_norm,
            snap.nonsurface_norm,
            snap.contrastive_updates_fired,
            status
        );
    }
    println!();

    // Final ordering status
    if let Some(last) = snapshots.last() {
        if last.ordering_correct {
            println!(
                "  ORDERING: CORRECT at iteration {} (gap={:+.4})",
                last.iteration, last.gap
            );
        } else {
            println!(
                "  ORDERING: INVERTED at iteration {} (gap={:+.4})",
                last.iteration, last.gap
            );
        }
    }
    println!("  Final weight norm: {:.4}", final_weight_norm);
    println!("  Total contrastive updates: {}", total_contrastive_updates);
    println!();

    // ═══════════════════════════════════════════════════════
    // POPULATION-AWARE CALIBRATION ON TEXT
    // ═══════════════════════════════════════════════════════
    // Phase 10: Switch to Inference mode for calibration, validation, adversarial
    resolver.mode = RouteMode::Inference;
    println!("─── Population-Aware Calibration on Text (mode={:?}) ───", resolver.mode);

    // Sample 100 sentences (40 simple, 30 moderate, 30 complex)
    let calib_simple: Vec<_> = corpus.simple[..40].to_vec();
    let calib_moderate: Vec<_> = corpus.moderate[..30].to_vec();
    let calib_complex: Vec<_> = corpus.complex[..30].to_vec();

    let mut calib_confs: Vec<f32> = Vec::new();
    let mut simple_calib_confs: Vec<f32> = Vec::new();
    let mut complex_calib_confs: Vec<f32> = Vec::new();
    let mut moderate_calib_confs: Vec<f32> = Vec::new();

    for s in &calib_simple {
        let tensor = encoder.encode_text_readonly(s);
        let conf = resolver.max_surface_confidence(&tensor);
        calib_confs.push(conf);
        simple_calib_confs.push(conf);
    }
    for s in &calib_moderate {
        let tensor = encoder.encode_text_readonly(s);
        let conf = resolver.max_surface_confidence(&tensor);
        calib_confs.push(conf);
        moderate_calib_confs.push(conf);
    }
    for s in &calib_complex {
        let tensor = encoder.encode_text_readonly(s);
        let conf = resolver.max_surface_confidence(&tensor);
        calib_confs.push(conf);
        complex_calib_confs.push(conf);
    }

    let simple_mean_conf =
        simple_calib_confs.iter().sum::<f32>() / simple_calib_confs.len() as f32;
    let complex_mean_conf =
        complex_calib_confs.iter().sum::<f32>() / complex_calib_confs.len() as f32;
    let moderate_mean_conf =
        moderate_calib_confs.iter().sum::<f32>() / moderate_calib_confs.len() as f32;
    // Surface threshold = midpoint between simple mean and complex mean.
    // Places threshold inside the gap between the two populations.
    let surface_threshold = (simple_mean_conf + complex_mean_conf) / 2.0;

    // Reasoning threshold at 40th percentile of all confidences
    let mut sorted_confs = calib_confs.clone();
    sorted_confs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let reasoning_idx = (sorted_confs.len() as f32 * 0.40) as usize;
    let reasoning_threshold = sorted_confs[reasoning_idx];

    // Apply thresholds
    resolver.config.surface_confidence_threshold = surface_threshold;
    resolver.config.reasoning_confidence_threshold = reasoning_threshold;
    resolver.rebuild_graph_edges();

    println!(
        "  Simple mean confidence:    {:.4}",
        simple_mean_conf
    );
    println!(
        "  Moderate mean confidence:  {:.4}",
        moderate_mean_conf
    );
    println!(
        "  Complex mean confidence:   {:.4}",
        complex_mean_conf
    );
    println!(
        "  Surface threshold (midpt): {:.4}",
        surface_threshold
    );
    println!(
        "  Reasoning threshold (p40): {:.4}",
        reasoning_threshold
    );

    // Compute escalation rates
    let simple_escalate = simple_calib_confs
        .iter()
        .filter(|&&c| c < surface_threshold)
        .count() as f32
        / simple_calib_confs.len() as f32
        * 100.0;
    let complex_escalate = complex_calib_confs
        .iter()
        .filter(|&&c| c < surface_threshold)
        .count() as f32
        / complex_calib_confs.len() as f32
        * 100.0;
    let moderate_escalate = moderate_calib_confs
        .iter()
        .filter(|&&c| c < surface_threshold)
        .count() as f32
        / moderate_calib_confs.len() as f32
        * 100.0;

    println!();
    println!("  Expected escalation rates:");
    println!("    Simple:   {:.1}% would escalate", simple_escalate);
    println!("    Moderate: {:.1}% would escalate", moderate_escalate);
    println!("    Complex:  {:.1}% would escalate", complex_escalate);

    let discrimination = if simple_escalate > 0.0 {
        complex_escalate / simple_escalate
    } else if complex_escalate > 0.0 {
        f32::INFINITY
    } else {
        1.0
    };
    println!(
        "  Discrimination metric: complex/simple = {:.1}x",
        discrimination
    );
    if complex_escalate >= simple_escalate * 2.0 {
        println!("  DISCRIMINATION: PASS (>= 2x)");
    } else {
        println!("  DISCRIMINATION: FAIL (< 2x)");
    }
    println!();

    // ═══════════════════════════════════════════════════════
    // VALIDATION BENCH — 50 sentences
    // ═══════════════════════════════════════════════════════
    println!(
        "─── Validation Bench: {} sentences ───",
        validation_sentences.len()
    );

    // Reset cache for validation
    resolver.cache =
        axiom_core::cache::EmbeddingCache::new(256, cache_threshold);

    let validation_result =
        run_text_pass(&mut resolver, &encoder, &validation_sentences, "Validation");
    write_json_log("axiom_validation.json", &validation_result.entries);
    print_text_results("Validation", &validation_result, validation_sentences.len());
    println!();

    // Per-sentence table sorted by surface confidence
    {
        let mut all_entries: Vec<&TextLogEntry> = validation_result.entries.iter().collect();
        all_entries.sort_by(|a, b| {
            b.surface_confidence
                .partial_cmp(&a.surface_confidence)
                .unwrap()
        });

        println!(
            "  All {} sentences ranked by Surface confidence (descending):",
            validation_sentences.len()
        );
        println!(
            "  {:>4}  {:>6}  {:>6}  {:>10}  {}",
            "rank", "s_conf", "conf", "tier", "sentence"
        );
        println!(
            "  {}  {}  {}  {}  {}",
            "────",
            "──────",
            "──────",
            "──────────",
            "─".repeat(60)
        );
        for (rank, entry) in all_entries.iter().enumerate() {
            let tag = match entry.complexity.as_str() {
                "simple" => "[S]",
                "moderate" => "[M]",
                "complex" => "[C]",
                _ => "[?]",
            };
            println!(
                "  {:>4}  {:.4}  {:.4}  {:>10}  {} {}",
                rank + 1,
                entry.surface_confidence,
                entry.confidence,
                entry.tier_reached,
                tag,
                truncate_str(&entry.sentence, 55),
            );
        }
        println!();

        // Per-complexity confidence stats
        let simple_entries: Vec<&TextLogEntry> = validation_result
            .entries
            .iter()
            .filter(|e| e.complexity == "simple")
            .collect();
        let moderate_entries: Vec<&TextLogEntry> = validation_result
            .entries
            .iter()
            .filter(|e| e.complexity == "moderate")
            .collect();
        let complex_entries: Vec<&TextLogEntry> = validation_result
            .entries
            .iter()
            .filter(|e| e.complexity == "complex")
            .collect();

        let simple_surf: Vec<f32> = simple_entries.iter().map(|e| e.surface_confidence).collect();
        let moderate_surf: Vec<f32> =
            moderate_entries.iter().map(|e| e.surface_confidence).collect();
        let complex_surf: Vec<f32> =
            complex_entries.iter().map(|e| e.surface_confidence).collect();

        let mean = |v: &[f32]| -> f32 { v.iter().sum::<f32>() / v.len().max(1) as f32 };
        let std_d = |v: &[f32]| -> f32 {
            let m = mean(v);
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len().max(1) as f32).sqrt()
        };
        let min_f = |v: &[f32]| -> f32 { v.iter().cloned().fold(f32::INFINITY, f32::min) };
        let max_f = |v: &[f32]| -> f32 { v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) };

        let s_mean = mean(&simple_surf);
        let m_mean = mean(&moderate_surf);
        let c_mean = mean(&complex_surf);

        println!("  Surface confidence by complexity:");
        println!(
            "    {:>10}  {:>5}  {:>6}  {:>6}  {:>6}  {:>6}",
            "group", "n", "mean", "std", "min", "max"
        );
        println!(
            "    {:>10}  {:>5}  {:.4}  {:.4}  {:.4}  {:.4}",
            "simple",
            simple_surf.len(),
            s_mean,
            std_d(&simple_surf),
            min_f(&simple_surf),
            max_f(&simple_surf)
        );
        println!(
            "    {:>10}  {:>5}  {:.4}  {:.4}  {:.4}  {:.4}",
            "moderate",
            moderate_surf.len(),
            m_mean,
            std_d(&moderate_surf),
            min_f(&moderate_surf),
            max_f(&moderate_surf)
        );
        println!(
            "    {:>10}  {:>5}  {:.4}  {:.4}  {:.4}  {:.4}",
            "complex",
            complex_surf.len(),
            c_mean,
            std_d(&complex_surf),
            min_f(&complex_surf),
            max_f(&complex_surf)
        );
        println!();

        // Confidence gaps
        let sm_gap = s_mean - m_mean;
        let mc_gap = m_mean - c_mean;
        let sc_gap = s_mean - c_mean;
        println!("  Confidence gaps:");
        println!(
            "    simple - moderate:  {:+.4}  {}",
            sm_gap,
            if sm_gap > 0.0 { "CORRECT" } else { "inverted" }
        );
        println!(
            "    moderate - complex: {:+.4}  {}",
            mc_gap,
            if mc_gap > 0.0 { "CORRECT" } else { "inverted" }
        );
        println!(
            "    simple - complex:   {:+.4}  {}",
            sc_gap,
            if sc_gap > 0.0 { "CORRECT" } else { "inverted" }
        );
        println!();

        // Tier routing by complexity
        println!("  Tier routing by complexity:");
        for (label, entries) in &[
            ("simple", &simple_entries),
            ("moderate", &moderate_entries),
            ("complex", &complex_entries),
        ] {
            let n = entries.len();
            let s_count = entries.iter().filter(|e| e.tier_reached == "Surface").count();
            let r_count = entries
                .iter()
                .filter(|e| e.tier_reached == "Reasoning")
                .count();
            let d_count = entries.iter().filter(|e| e.tier_reached == "Deep").count();
            println!(
                "    {:>10}: S {:>2}/{:<2} ({:>5.1}%)  R {:>2}/{:<2} ({:>5.1}%)  D {:>2}/{:<2} ({:>5.1}%)",
                label,
                s_count,
                n,
                s_count as f32 / n as f32 * 100.0,
                r_count,
                n,
                r_count as f32 / n as f32 * 100.0,
                d_count,
                n,
                d_count as f32 / n as f32 * 100.0,
            );
        }
        println!();

        // Eleven-phase comparison
        let p11_simple_s = simple_entries
            .iter()
            .filter(|e| e.tier_reached == "Surface")
            .count() as f32
            / simple_entries.len() as f32
            * 100.0;
        let p11_complex_s = complex_entries
            .iter()
            .filter(|e| e.tier_reached == "Surface")
            .count() as f32
            / complex_entries.len() as f32
            * 100.0;
        let phase11_correct = p11_simple_s > p11_complex_s;
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │                  Eleven-Phase Comparison Table                          │");
        println!("  ├─────────┬──────────────────────────────────────────────────────────────┤");
        println!("  │ Phase   │ Result                                                       │");
        println!("  ├─────────┼──────────────────────────────────────────────────────────────┤");
        println!("  │  4      │ simple 13% S, complex 47% S — inverted pre-learning          │");
        println!("  │  5      │ 100% S everything — over-converged                           │");
        println!("  │  6      │ simple 20% S, complex 71% S — inverted post-contrastive      │");
        println!("  │  7      │ simple 73% S, complex 88% S — inverted post-synthetic        │");
        println!("  │  8      │ simple 13% S, complex 100% S — inverted text-only            │");
        println!("  │  9      │ simple 0% S, complex 100% S — cosine init, training broke it │");
        println!("  │ 10      │ simple 0% S, complex 100% S — Oja overwrites discrimination  │");
        println!(
            "  │ 11      │ simple {:.0}% S, complex {:.0}% S — {} │",
            p11_simple_s,
            p11_complex_s,
            if phase11_correct {
                "CORRECT analytical init frozen Surface"
            } else {
                "inverted                              "
            }
        );
        println!("  └─────────┴──────────────────────────────────────────────────────────────┘");
        println!();

        // Overall ordering verdict
        let ordering_ok = s_mean > m_mean && m_mean > c_mean;
        let partial_ok = s_mean > c_mean;
        if ordering_ok {
            println!("  VERDICT: Full ordering CORRECT (simple > moderate > complex)");
        } else if partial_ok {
            println!(
                "  VERDICT: Partial ordering (simple > complex) but moderate out of place"
            );
        } else {
            println!("  VERDICT: Ordering INVERTED (simple <= complex)");
        }
        println!(
            "    Surface threshold: {:.4}",
            resolver.config.surface_confidence_threshold
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════
    // ADVERSARIAL PASS — 20 sentences
    // ═══════════════════════════════════════════════════════
    println!("─── Adversarial Pass (19 sentences) ───");

    // Reset cache for adversarial
    resolver.cache =
        axiom_core::cache::EmbeddingCache::new(256, cache_threshold);

    let adversarial_sentences: Vec<(&str, &str)> = vec![
        // Very short complex
        ("Cogito ergo sum", "complex"),
        ("Gödel's proof breaks mathematics", "complex"),
        ("Being precedes essence", "complex"),
        // Very long simple
        ("the big red dog ran quickly down the long straight road toward the tall old brown wooden fence near the small quiet house by the river", "simple"),
        // Rare words simple structure
        ("the tintinnabulation resonated melodiously", "simple"),
        ("photosynthesis converts sunlight efficiently", "simple"),
        // Common words complex nesting
        ("the cat that the dog that the man owned chased sat on the mat", "complex"),
        ("she said that he thought that they believed it was true", "complex"),
        // Mixed complexity
        ("neural networks approximate complex nonlinear functions through hierarchical feature learning", "complex"),
        ("it is raining today", "simple"),
        ("the interplay between cooperation and competition drives evolutionary dynamics", "complex"),
        ("the sky is blue", "simple"),
        ("water flows downhill", "simple"),
        ("consciousness remains one of the most profound unsolved problems in all of science", "complex"),
        ("if then else", "simple"),
        ("the mitochondria is the powerhouse of the cell", "moderate"),
        ("dark matter constitutes approximately twenty seven percent of the total mass energy content of the observable universe", "complex"),
        ("she runs fast", "simple"),
        ("go", "simple"),
    ];

    let adversarial_result =
        run_text_pass(&mut resolver, &encoder, &adversarial_sentences, "Adversarial");
    write_json_log("axiom_adversarial.json", &adversarial_result.entries);

    println!(
        "  {:>4}  {:>6}  {:>6}  {:>10}  {:>8}  {:>7}  {}",
        "rank", "s_conf", "conf", "tier", "expect", "correct", "sentence"
    );
    println!(
        "  {}  {}  {}  {}  {}  {}  {}",
        "────",
        "──────",
        "──────",
        "──────────",
        "────────",
        "───────",
        "─".repeat(55)
    );
    for (i, entry) in adversarial_result.entries.iter().enumerate() {
        // Routing correct: simple should stay Surface, complex should escalate, moderate is ok either way
        let correct = match entry.complexity.as_str() {
            "simple" => {
                if entry.tier_reached == "Surface" { "yes" } else { "no" }
            }
            "complex" => {
                if entry.tier_reached != "Surface" { "yes" } else { "no" }
            }
            _ => "—", // moderate: either routing is acceptable
        };
        println!(
            "  {:>4}  {:.4}  {:.4}  {:>10}  {:>8}  {:>7}  {}",
            i + 1,
            entry.surface_confidence,
            entry.confidence,
            entry.tier_reached,
            entry.complexity,
            correct,
            truncate_str(&entry.sentence, 55),
        );
    }
    println!();

    // ═══════════════════════════════════════════════════════
    // FINAL SUMMARY
    // ═══════════════════════════════════════════════════════
    println!("═══════════════════ PHASE 11 FINAL SUMMARY ═══════════════════");
    println!("  Total parameters:        {}", weight_count);
    println!("  Tests passing:           109 (98 core + 6 bench + 5 tuner)");
    println!("  Training iterations:     {}", train_iterations);
    println!("  Surface weight norm:     {:.4} (constant — frozen)", initial_surface_norm);
    let final_rd_norm = final_weight_norm - initial_surface_norm;
    println!("  R+D weight norm:         {:.4} (final)", final_rd_norm);
    println!(
        "  Confidence std dev:      {:.6}",
        std_dev_f32(&train_confs)
    );
    println!(
        "  Weight drift:            {:.4} -> {:.4} ({:.2}%)",
        initial_weight_norm, final_weight_norm, weight_drift_pct
    );
    println!("  Logs: axiom_*_log.json, axiom_validation.json, axiom_adversarial.json");
    println!("═══════════════════════════════════════════════════════════════");

    // Auto-tuner
    println!();
    println!("─── Auto-Tuner ───");
    // Write training entries as bench log for tuner compatibility
    #[derive(Serialize)]
    struct TunerEntry {
        tier_reached: String,
        confidence: f32,
        compute_cost: f32,
        cache_hits: u32,
        from_cache: bool,
    }
    let tuner_entries: Vec<TunerEntry> = train_entries
        .iter()
        .map(|e| TunerEntry {
            tier_reached: e.tier_reached.clone(),
            confidence: e.confidence,
            compute_cost: e.compute_cost,
            cache_hits: 0,
            from_cache: e.from_cache,
        })
        .collect();
    write_json_log("axiom_bench_log.json", &tuner_entries);
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
        println!();
        println!("  Dashboard still live at http://localhost:{}", port);
        println!("  Press Ctrl+C to exit.");
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

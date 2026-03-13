use axiom::{AxiomConfig, HierarchicalResolver, RouteMode, Tier};
use std::time::Instant;

fn main() {
    println!("AXIOM — Adaptive eXecution with Intelligent Operations Memory");
    println!("==============================================================\n");

    let config = AxiomConfig::default();
    let mut resolver = HierarchicalResolver::new(config);

    // ── Training phase ──────────────────────────────────────────────────
    println!("Phase 1: Training");
    resolver.set_mode(RouteMode::Training);

    let simple_training = vec![
        "What time is it?",
        "Hello, how are you?",
        "What is your name?",
        "Can you help me?",
        "Thank you very much.",
        "Where is the nearest store?",
        "How do I reset my password?",
        "What are your hours?",
        "I need to cancel my order.",
        "Please send me the receipt.",
        "Is this available in blue?",
        "How much does shipping cost?",
        "Can I return this item?",
        "What is the refund policy?",
        "I forgot my username.",
    ];

    let complex_training = vec![
        "Although the phenomenon has been observed across multiple contexts, \
         the underlying mechanisms that drive it remain poorly understood, \
         which complicates efforts to develop interventions that would \
         effectively address the root causes while accounting for the \
         heterogeneous responses observed in different populations.",
        "Reconcile the epistemological tensions between Bayesian inference \
         and frequentist hypothesis testing in the context of reproducibility \
         concerns that have emerged across multiple scientific disciplines.",
        "The implementation requires careful consideration of the trade-offs \
         between computational efficiency and numerical stability, particularly \
         when dealing with ill-conditioned matrices that arise in practice \
         from correlated feature sets in high-dimensional spaces.",
        "Analyze the socioeconomic implications of algorithmic decision-making \
         systems that perpetuate historical biases while simultaneously \
         optimizing for efficiency metrics that may not align with equity \
         objectives across diverse demographic groups.",
        "Consider how the interaction between monetary policy transmission \
         mechanisms and fiscal multiplier effects varies across different \
         exchange rate regimes, particularly in small open economies \
         with significant external debt denominated in foreign currencies.",
    ];

    // Analytical initialization from simple examples
    let simple_embeddings: Vec<Vec<f32>> = simple_training
        .iter()
        .map(|t| resolver.encoder.encode(t))
        .collect();
    resolver.graph.analytical_init_surface(&simple_embeddings);
    resolver.graph.orthogonal_init_reasoning_deep(42);

    println!(
        "  Initialized {} nodes ({} parameters)",
        resolver.graph.nodes.len(),
        resolver.graph.total_parameters()
    );
    println!(
        "  R+D mean pairwise cosine: {:.4}",
        resolver.graph.rd_mean_pairwise_cosine()
    );

    // Train on examples
    for text in &simple_training {
        resolver.train_example(text, true);
    }
    for text in &complex_training {
        resolver.train_example(text, false);
    }

    resolver.finalize_training();
    println!(
        "  G5 simple mean norm: {:.4}",
        resolver.encoder.g5_simple_mean_norm
    );
    println!(
        "  G5 complex mean norm: {:.4}",
        resolver.encoder.g5_complex_mean_norm
    );

    // ── Inference phase ─────────────────────────────────────────────────
    println!("\nPhase 2: Inference");
    resolver.set_mode(RouteMode::Inference);

    let benchmark_simple = vec![
        "What is the weather today?",
        "How do I contact support?",
        "Can I change my email address?",
        "What payment methods do you accept?",
        "When will my package arrive?",
        "Is there a discount available?",
        "How do I log out?",
        "What is your phone number?",
        "Can I speak to a manager?",
        "Thank you for your help.",
    ];

    let benchmark_complex = vec![
        "Explain the implications of Gödel's incompleteness theorems \
         for the foundations of mathematics and their relationship to \
         Turing's halting problem.",
        "Analyze how the interaction between epigenetic modifications \
         and environmental stressors influences gene expression patterns \
         across multiple generations, with particular attention to \
         transgenerational inheritance mechanisms.",
        "Compare and contrast the effectiveness of attention-based \
         transformer architectures versus state-space models for \
         long-range sequence modeling, considering both computational \
         complexity and empirical performance on downstream tasks.",
        "Evaluate the philosophical tensions between consequentialist \
         and deontological frameworks when applied to autonomous vehicle \
         decision-making in unavoidable accident scenarios.",
        "Assess the macroeconomic implications of central bank digital \
         currencies on monetary policy transmission, financial stability, \
         and the existing commercial banking system's fractional reserve model.",
    ];

    let mut all_results = Vec::new();
    let mut simple_correct = 0;
    let mut complex_correct = 0;

    let start = Instant::now();

    for text in &benchmark_simple {
        let result = resolver.route(text);
        if result.selected_tier == Tier::Surface {
            simple_correct += 1;
        }
        all_results.push(result);
    }

    for text in &benchmark_complex {
        let result = resolver.route(text);
        if result.selected_tier == Tier::Reasoning || result.selected_tier == Tier::Deep {
            complex_correct += 1;
        }
        all_results.push(result);
    }

    let elapsed = start.elapsed();
    let mean_latency_us = elapsed.as_micros() as f64 / all_results.len() as f64;

    println!(
        "  Simple accuracy: {}/{} ({:.1}%)",
        simple_correct,
        benchmark_simple.len(),
        simple_correct as f64 / benchmark_simple.len() as f64 * 100.0
    );
    println!(
        "  Complex accuracy: {}/{} ({:.1}%)",
        complex_correct,
        benchmark_complex.len(),
        complex_correct as f64 / benchmark_complex.len() as f64 * 100.0
    );

    let total_correct = simple_correct + complex_correct;
    let total_queries = benchmark_simple.len() + benchmark_complex.len();
    println!(
        "  Overall accuracy: {}/{} ({:.1}%)",
        total_correct,
        total_queries,
        total_correct as f64 / total_queries as f64 * 100.0
    );

    let (surface_pct, reasoning_pct, deep_pct) = resolver.routing_distribution(&all_results);
    println!(
        "  Distribution: Surface {:.1}%, Reasoning {:.1}%, Deep {:.1}%",
        surface_pct * 100.0,
        reasoning_pct * 100.0,
        deep_pct * 100.0
    );

    let savings = HierarchicalResolver::cost_savings(&all_results);
    println!("  Cost savings vs all-Opus: {:.1}%", savings);
    println!("  Mean routing latency: {:.0} µs", mean_latency_us);

    // ── Detailed trace for one query ────────────────────────────────────
    println!("\nPhase 3: Trace Example");
    let trace_query = "Reconcile Kant's categorical imperative with utilitarian ethics.";
    let result = resolver.route(trace_query);
    println!("  Query: \"{}\"", trace_query);
    println!("  Routed to: {}", result.selected_tier);
    println!("  Confidence: {:.4}", result.confidence);
    println!("  Resolved by: {}", result.resolved_by);
    println!("  Coalition: {:?}", result.coalition_members);
    println!("  Cross-tier: {}", result.cross_tier_resolution);
    println!("  Trace steps:");
    for step in &result.trace {
        println!(
            "    [{:?}] {} → conf {:.4} → {:.4} (cached: {})",
            step.direction, step.node_id, step.confidence_in, step.confidence_out, step.was_cached
        );
    }

    // ── Export weights ───────────────────────────────────────────────────
    let weights = resolver.export_weights();
    let weights_json = serde_json::to_string_pretty(&weights).unwrap();
    let weights_size = weights_json.len();
    println!(
        "\nWeights export: {} bytes ({:.1} KB)",
        weights_size,
        weights_size as f64 / 1024.0
    );

    // Export config
    let config_json = serde_json::to_string_pretty(&resolver.config).unwrap();
    std::fs::write("axiom_config.json", &config_json).ok();
    std::fs::write("axiom_weights.json", &weights_json).ok();
    println!("Written axiom_config.json and axiom_weights.json");

    println!("\nDone.");
}

#!/bin/bash
# AXIOM Experiment Sweep
# Runs a series of experiments and produces a comparison table.
# Usage: ./experiments/sweep.sh

set -e

SWEEP_DIR="experiments/results/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  AXIOM Phase 15 — Full Parameter Sweep                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Sweep dir: $SWEEP_DIR"
echo ""

# Build once
cargo build --release -p axiom-bench 2>&1 | tail -1

# Run a single experiment and extract metrics
run_exp() {
    local name="$1"
    shift
    local dir="$SWEEP_DIR/$name"
    mkdir -p "$dir"

    echo ""
    echo "━━━ Running: $name ━━━"
    echo "  Env: $@"

    env "$@" cargo run --release -p axiom-bench 2>&1 | tee "$dir/output.txt"

    # Copy artifacts
    for f in axiom_training_snapshots.json axiom_validation.json axiom_contrastive_log.json axiom_weights.json axiom_config.json; do
        [ -f "$f" ] && cp "$f" "$dir/"
    done

    # Extract key metrics to summary file
    local gap=$(grep "simple - complex:" "$dir/output.txt" | tail -1 | grep -oP '[+-]?[0-9]+\.[0-9]+' | head -1)
    local adv=$(grep "Adversarial score:" "$dir/output.txt" | grep -oP '[0-9]+/[0-9]+' | head -1)
    local params=$(grep "Total trainable parameters:" "$dir/output.txt" | grep -oP '[0-9]+' | head -1)
    local drift=$(grep "Weight drift:" "$dir/output.txt" | head -1 | grep -oP '[0-9]+\.[0-9]+%' | head -1)
    local contrastive=$(grep "Contrastive updates:" "$dir/output.txt" | head -1 | grep -oP '[0-9]+' | head -1)
    local coal_size=$(grep "Mean coalition size:" "$dir/output.txt" | head -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    local runtime=$(grep "Text Training results" "$dir/output.txt" | grep -oP '\([^)]+\)' | head -1)

    echo "$name|$gap|$adv|$params|$drift|$contrastive|$coal_size|$runtime" >> "$SWEEP_DIR/summary.csv"
}

# Header
echo "name|gap|adversarial|params|drift|contrastive|coal_size|runtime" > "$SWEEP_DIR/summary.csv"

# ── Sweep configurations ──

# 1. Baseline: current defaults
run_exp "baseline" AXIOM_ITER=100000

# 2. Contrastive LR sweep
run_exp "clr_0001" AXIOM_ITER=100000 AXIOM_CONTRASTIVE_LR=0.0001
run_exp "clr_0005" AXIOM_ITER=100000 AXIOM_CONTRASTIVE_LR=0.0005
run_exp "clr_001"  AXIOM_ITER=100000 AXIOM_CONTRASTIVE_LR=0.001

# 3. G5 weight sweep
run_exp "g5_020" AXIOM_ITER=100000 AXIOM_G5_WEIGHT=0.20
run_exp "g5_050" AXIOM_ITER=100000 AXIOM_G5_WEIGHT=0.50

# 4. Confidence mix sweep
run_exp "mix_50_50" AXIOM_ITER=100000 AXIOM_CONF_MIX=0.5
run_exp "mix_40_60" AXIOM_ITER=100000 AXIOM_CONF_MIX=0.4

# 5. Coalition sweep
run_exp "coal_2" AXIOM_ITER=100000 AXIOM_COALITION_MAX=2
run_exp "coal_6" AXIOM_ITER=100000 AXIOM_COALITION_MAX=6

# 6. LR schedule
run_exp "cosine_lr" AXIOM_ITER=100000 AXIOM_LR_SCHEDULE=cosine

# 7. Phased training
run_exp "phased" AXIOM_ITER=100000 AXIOM_PHASED=true

# 8. Higher training iterations
run_exp "200k_iter" AXIOM_ITER=200000

# ── Results table ──
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  SWEEP RESULTS                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
column -t -s'|' "$SWEEP_DIR/summary.csv"
echo ""
echo "Results saved to: $SWEEP_DIR/"

#!/bin/bash
# AXIOM Experiment Runner
# Usage: ./experiments/run_experiment.sh <experiment_name> [ENV_VAR=value ...]
# Example: ./experiments/run_experiment.sh baseline
# Example: ./experiments/run_experiment.sh contrastive_lr_0005 AXIOM_CONTRASTIVE_LR=0.0005

set -e

EXPERIMENT_NAME="${1:?Usage: $0 <experiment_name> [ENV=val ...]}"
shift

RESULTS_DIR="experiments/results/${EXPERIMENT_NAME}"
mkdir -p "$RESULTS_DIR"

echo "=== AXIOM Experiment: ${EXPERIMENT_NAME} ==="
echo "  Results dir: ${RESULTS_DIR}"
echo "  Env overrides: $@"
echo ""

# Build once
cargo build --release -p axiom-bench 2>&1 | tail -1

# Run with env overrides, capture output
env "$@" cargo run --release -p axiom-bench 2>&1 | tee "${RESULTS_DIR}/output.txt"

# Copy result files
for f in axiom_training_snapshots.json axiom_validation.json axiom_contrastive_log.json axiom_weights.json axiom_config.json; do
    [ -f "$f" ] && cp "$f" "${RESULTS_DIR}/"
done

# Extract key metrics from output
echo ""
echo "=== Key Metrics ==="
grep -E "Confidence gap|Adversarial score|Total parameters|Weight drift|Surface norm|R\+D norm|Contrastive updates|Mean coalition" "${RESULTS_DIR}/output.txt" || true
echo "=== End ${EXPERIMENT_NAME} ==="

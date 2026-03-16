#!/usr/bin/env python3
"""Convert RouterBench pickle to JSONL for Rust consumption."""

import pickle
import pandas as pd
import json
import ast
import sys

INPUT = "data/routerbench/routerbench_5shot.pkl"
OUTPUT = "data/routerbench/routerbench.jsonl"

MODELS = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    "meta/code-llama-instruct-34b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "claude-instant-v1",
    "meta/llama-2-70b-chat",
    "gpt-3.5-turbo-1106",
    "claude-v1",
    "claude-v2",
    "gpt-4-1106-preview",
]

# Short names for Rust
SHORT = {
    "mistralai/mistral-7b-chat": "mistral-7b",
    "WizardLM/WizardLM-13B-V1.2": "wizardlm-13b",
    "mistralai/mixtral-8x7b-chat": "mixtral-8x7b",
    "meta/code-llama-instruct-34b-chat": "codellama-34b",
    "zero-one-ai/Yi-34B-Chat": "yi-34b",
    "claude-instant-v1": "claude-instant",
    "meta/llama-2-70b-chat": "llama-70b",
    "gpt-3.5-turbo-1106": "gpt-3.5",
    "claude-v1": "claude-v1",
    "claude-v2": "claude-v2",
    "gpt-4-1106-preview": "gpt-4",
}

def extract_prompt(raw):
    """Extract prompt text from RouterBench format."""
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw)
    s = str(raw)
    # Try parsing as Python list literal
    if s.startswith("["):
        try:
            parts = ast.literal_eval(s)
            if isinstance(parts, list):
                return " ".join(str(x) for x in parts)
        except:
            pass
    return s

def main():
    print(f"Loading {INPUT}...")
    df = pickle.load(open(INPUT, "rb"))
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Group eval_names into high-level datasets
    dataset_map = {}
    for name in df["eval_name"].unique():
        if name.startswith("mmlu"):
            dataset_map[name] = "mmlu"
        elif name.startswith("chinese") or name.startswith("Chinese"):
            dataset_map[name] = "chinese"
        elif name.startswith("mtbench"):
            dataset_map[name] = "mtbench"
        else:
            dataset_map[name] = name

    out = open(OUTPUT, "w")
    written = 0
    skipped = 0

    for _, row in df.iterrows():
        prompt = extract_prompt(row["prompt"])
        if not prompt or len(prompt) < 2:
            skipped += 1
            continue

        eval_name = row["eval_name"]
        dataset = dataset_map.get(eval_name, eval_name)

        scores = {}
        costs = {}
        for m in MODELS:
            short = SHORT[m]
            score = pd.to_numeric(row.get(m, 0), errors="coerce")
            cost = pd.to_numeric(row.get(f"{m}|total_cost", 0), errors="coerce")
            scores[short] = round(float(score) if pd.notna(score) else 0.0, 4)
            costs[short] = round(float(cost) if pd.notna(cost) else 0.0, 8)

        oracle = row.get("oracle_model_to_route_to", "")
        oracle_short = SHORT.get(oracle, oracle) if oracle else ""

        record = {
            "prompt": prompt,
            "eval_name": eval_name,
            "dataset": dataset,
            "scores": scores,
            "costs": costs,
            "oracle": oracle_short,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

    out.close()
    print(f"  Written {written} records to {OUTPUT} (skipped {skipped})")

    # Summary stats
    print(f"\nDataset distribution:")
    datasets = {}
    for _, row in df.iterrows():
        d = dataset_map.get(row["eval_name"], row["eval_name"])
        datasets[d] = datasets.get(d, 0) + 1
    for d, c in sorted(datasets.items(), key=lambda x: -x[1]):
        print(f"  {d}: {c}")

if __name__ == "__main__":
    main()

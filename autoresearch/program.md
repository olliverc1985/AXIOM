# AXIOM Encoder Autoresearch

You are an autonomous research agent improving AXIOM's structural encoder.
Run experiments forever. Never stop. Never ask a human for input.

## Context

AXIOM is a sparse routing architecture for cost-efficient LLM inference.
It routes queries to three tiers: Surface (cheap), Reasoning (mid), Deep (expensive).
The routing decision depends on a 128-dimensional structural encoder that extracts
syntactic features from text. The encoder is the binding constraint — it cannot
detect semantic complexity without syntactic markers.

**Current bottleneck**: Short philosophical queries ("Cogito ergo sum"),
domain-complex sentences with simple syntax, and garden-path sentences
all get misrouted to Surface because they lack syntactic complexity markers.

**Your goal**: Maximize `val_score` (higher = better), which weights
complex accuracy 70% and simple accuracy 30%, with a penalty if simple
accuracy drops below 95%.

## Setup

```bash
git checkout -b autoresearch/encoder-$(date +%Y%m%d-%H%M%S)
python3 train.py > run.log 2>&1
grep "^val_score:" run.log
```

Record the baseline val_score. Initialize results.tsv:
```
commit	val_score	simple_acc	complex_acc	status	description
```

## The Loop (run FOREVER)

1. Read `train.py` and think about what to try next
2. Modify `train.py` with your experimental idea
3. `git add train.py && git commit -m "experiment: <description>"`
4. Run: `python3 train.py > run.log 2>&1`
5. If crash: `tail -50 run.log`, fix trivially or revert
6. Extract: `grep "^val_score:\|^simple_accuracy:\|^complex_accuracy:" run.log`
7. Append result to results.tsv (do NOT commit results.tsv)
8. If val_score improved → KEEP the commit
9. If val_score equal or worse → `git reset HEAD~1 --hard`
10. Go to step 1. NEVER STOP.

## What you can modify

**ONLY `train.py`**. Never modify `prepare.py`.

You can change anything in train.py:
- Feature extraction functions (g1-g5)
- Add entirely new feature groups (g6, g7, ...)
- Change dimensionality (OUTPUT_DIM doesn't have to be 128)
- Change amplification factors
- Change the confidence formula
- Add new word lists, heuristics, or learned components
- Import any Python stdlib module (math, collections, statistics, re, etc.)
- Add information-theoretic features, compression-based features, etc.

## What you CANNOT do

- Do NOT modify `prepare.py`
- Do NOT add external dependencies (no numpy, no torch, no nltk, etc.)
- Do NOT change the output format (val_score must still be printed)
- Do NOT hardcode test sentences or overfit to specific examples
- Do NOT make the encoder take more than 5 seconds total

## Architecture of train.py

```
encode(text) → list[float]  (any number of dimensions)
  └─ tokenize(text) → words
  └─ g1_ngram_profile(words) → 26 dims (character n-gram hash buckets)
  └─ g2_syntactic_features(words, text) → 36 dims (syntactic proxies)
  └─ g3_token_signal(words) → 39 dims (position-weighted token hash)
  └─ g4_complexity_scalars(words, text) → 15 dims (complexity measures)
  └─ g5_structural_syntax(words) → 12 dims (dependency depth, constituent variance, fw entropy)

confidence(encoding, surface_weight, g5_stats) → float
  └─ base_conf = min(||encoding||, 1.0)
  └─ cosine = cosine_sim(encoding, surface_weight)
  └─ conf = base_conf * 0.7 + cosine * 0.3 - g5_penalty * 0.35

Routing: conf >= threshold → Surface (simple) | conf < threshold → escalate (complex)
```

## Ideas to explore (pick ONE per experiment)

### Quick wins (feature weights)
- Sweep G5_AMP from 1.0 to 10.0
- Sweep g5_penalty weight from 0.1 to 0.8
- Sweep confidence mix ratio (currently 0.7/0.3)
- Remove or reweight individual feature groups

### New features targeting known failure modes
- **Word rarity score**: Use character frequency distribution as proxy for
  word rarity (rare character combinations = rare words = likely complex).
  E.g., "kolmogorov" has unusual char bigrams.
- **Compression ratio proxy**: Complex text has higher entropy. Compute
  character-level entropy or unique-ngram-to-total ratio.
- **Abstract vs concrete word detection**: Function word ratio alone doesn't
  capture whether content words are abstract ("ontology", "epistemology")
  vs concrete ("dog", "cat"). Try word length + char frequency combo.
- **Semantic domain indicators**: Academic/philosophical words often have
  Latin/Greek roots. Detect common suffixes: -tion, -ment, -ical, -ology,
  -istic, -eous, -uous, -ive, -ence, -ance, -ism, -ist.
- **Interrogative depth**: "What" vs "How might one reconcile..." — detect
  implicit complexity in question structure.
- **Information density**: Characters per word × unique words per total =
  how much novel information is packed per unit.
- **Polysyllabic word ratio**: Count vowel groups per word, flag words with
  4+ syllables as complex vocabulary indicators.

### Confidence formula changes
- Use ONLY cosine similarity (remove base_confidence magnitude dependence)
- Add per-group penalties beyond G5
- Use distance-from-simple-centroid instead of cosine similarity
- Try Mahalanobis distance using simple/complex covariance
- Multi-threshold: different thresholds for different sentence lengths
- Two-stage: cheap check first, then expensive features only if borderline

### Structural changes
- Variable-length encoding (more dims for longer sentences)
- Sentence chunking: split on periods, encode chunks separately, aggregate
- Multiple surface directions instead of one mean direction
- Learned character embeddings (build small embedding table from training data)

## Key insight

The three failure categories are:
1. **Short complex**: "Cogito ergo sum" — 3 words, no syntactic markers.
   These need SEMANTIC features, not syntactic ones.
2. **Long simple**: "the big red dog ran..." — many words inflate features.
   These need BETTER LENGTH NORMALIZATION.
3. **Domain vocabulary**: "mitochondrial electron transport chain" — technical
   words but simple syntax. Need LEXICAL RARITY features.

Simplicity is a virtue. If removing code gives the same val_score, KEEP the
simpler version. Complexity must justify itself with measurable improvement.

## Timing

Each run takes < 0.1 seconds. You can run ~36,000 experiments in an hour.
Don't overthink — try things fast, keep what works, revert what doesn't.

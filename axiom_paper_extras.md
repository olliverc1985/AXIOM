# AXIOM Paper -- Additional Materials

---

## 1. Competitive Comparison Table

| System | Tiers | Router Model | Router Params | Training Data Required | Routing Latency | Framework Dependencies | Language | Online Learning | Reported Cost Savings | Reported Accuracy |
|--------|-------|-------------|---------------|----------------------|----------------|----------------------|----------|----------------|----------------------|-------------------|
| **AXIOM** | 3 (Surface / Reasoning / Deep) | Sparse computation graph + Oja's rule | 1,205,376 | None -- self-calibrating corpus of 27 sentences. No LLM outputs, no preference data, no human annotations. | ~1,311 us (~1.3 ms) | Zero ML framework dependencies | Pure Rust | Yes (Hebbian online adaptation via Oja's rule on active coalition members) | 56.6% vs all-Opus | 58% overall (100% simple, 22% complex) |
| **RouteLLM** (ICLR 2025) | 2 (weak / strong) | Trained classifier (MF, BERT, SW ranking) | ~110M (BERT-based); MF variant ~1M | Chatbot Arena preference data (~80k pairwise comparisons from human judges) | ~5-15 ms (BERT inference on GPU) | PyTorch, Transformers, CUDA | Python | No (static after training) | ~50% vs GPT-4 at equivalent quality | 50-70% cost reduction at matched quality thresholds |
| **FrugalGPT** (2023) | N (cascade of N models) | LLM judge + scoring function | Depends on judge LLM; cascade selector ~few hundred params | Requires LLM API outputs on a labeled dataset (~1-10k samples) | ~50-200 ms (requires LLM scoring call) | OpenAI API, scikit-learn | Python | No | Up to 98% cost reduction vs GPT-4 | Matches GPT-4 on select benchmarks |
| **AutoMix** (NeurIPS 2024) | 2-3 (small / large, optional verifier) | Self-verification by the small LLM itself | 0 additional (uses existing LLM) | No separate training data; few-shot prompting only | ~100-500 ms (requires small LLM verification pass) | LLM API access | Python | No | 50% compute reduction on LLAMA-70B tasks | Comparable to strong model on 5 benchmarks |
| **Hybrid LLM** (ICLR 2024) | 2 (small / large) | Trained binary classifier (BERT or smaller) | ~110M (BERT); distilled variant ~14M | Labeled dataset of (query, quality_label) pairs; typically ~10-50k samples | ~5-15 ms (classifier inference) | PyTorch, Transformers | Python | No | Up to 40% fewer large-model calls | Maintains quality within 1-2% of always-large baseline |
| **CSCR** (NeurIPS 2025) | 2 (small / large) | Correctness-supervised classifier with reward model | ~110M (BERT-based) | Requires reward model scores on both small and large LLM outputs (~10-50k) | ~10-20 ms (classifier + scoring) | PyTorch, Transformers | Python | No | ~50% cost reduction | Outperforms RouteLLM on quality-matched routing |
| **kNN Router** (Li, 2025) | 2+ (flexible) | k-nearest-neighbor lookup in embedding space | Embedding index (~768d per sample); no trainable params beyond embeddings | Requires pre-computed embeddings of reference queries (~1-10k) | ~1-5 ms (embedding + kNN search) | FAISS or similar ANN library, sentence-transformers | Python | Partially (can add new reference points) | Comparable to trained classifiers | Competitive with BERT-based routers on standard benchmarks |

### Key Differentiators

1. **Zero LLM dependency**: AXIOM is the only system that requires no LLM outputs, no preference data, and no human annotations for training. All other systems require either labeled LLM outputs or human preference judgments.
2. **Online learning**: AXIOM is the only system with continuous Hebbian adaptation during inference. kNN Router can add reference points but does not update learned representations.
3. **Framework independence**: AXIOM has zero ML framework dependencies (no PyTorch, no Transformers, no CUDA). Every other system depends on Python ML infrastructure.
4. **Parameter efficiency**: At 1.2M parameters, AXIOM is 90x smaller than BERT-based routers (~110M). Only FrugalGPT's cascade selector and kNN Router's index are comparably lightweight, but both still require LLM outputs.
5. **Language**: AXIOM is the only system written in a compiled systems language (Rust). All others are Python.
6. **Honest limitation**: AXIOM's 22% complex accuracy reflects the inherent ceiling of structural-only routing without semantic understanding. Systems using LLM-generated features achieve higher complex accuracy but at the cost of requiring LLM infrastructure.

---

## 2. Reproducibility

### 2.1 Training (axiom-bench)

**Command (defaults):**
```bash
cargo run --release -p axiom-bench
```

**Command (with dashboard):**
```bash
cargo run --release -p axiom-bench -- --dashboard
cargo run --release -p axiom-bench -- --dashboard --port 9090
```

**Environment Variables:**

All configurable via `env_or()` pattern -- set before running, or omit for defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `AXIOM_ITER` | `100000` | Number of training iterations |
| `AXIOM_LR` | `0.001` | Hebbian learning rate (Oja's rule) |
| `AXIOM_ERROR_LR` | `0.0005` | Error-driven learning rate |
| `AXIOM_G5_WEIGHT` | `0.35` | G5 structural syntax penalty weight |
| `AXIOM_G4_WEIGHT` | `0.0` | G4 magnitude penalty weight (disabled by default) |
| `AXIOM_CONTRASTIVE_LR` | `0.00005` | Contrastive learning rate for Surface nodes |
| `AXIOM_CONF_MIX` | `0.7` | Confidence formula base weight (complementary cosine weight = 1.0 - this) |
| `AXIOM_MID_DIM` | `128` | Hidden dimension for LinearNode (controls total param count) |
| `AXIOM_LR_SCHEDULE` | `"constant"` | Learning rate schedule: `"constant"` or `"cosine"` |
| `AXIOM_PHASED` | `false` | Enable phased training (simple-first curriculum) |
| `AXIOM_G5_NORMALIZE` | `false` | G5 length normalization (divide by sqrt(token_count)) |
| `AXIOM_COALITION_MAX` | `4` | Maximum coalition size for dynamic coalition formation |
| `AXIOM_COALITION_THRESH` | `0.10` | Coalition bid threshold |

**Example: parameter scaling experiment:**
```bash
# 1.2M params (default)
AXIOM_MID_DIM=128 cargo run --release -p axiom-bench

# 4.8M params
AXIOM_MID_DIM=512 cargo run --release -p axiom-bench
```

**Example: environment sweep (autoresearch):**
```bash
AXIOM_G5_WEIGHT=0.35 AXIOM_ITER=100000 cargo run --release -p axiom-bench 2>&1 | tee experiments/results/phase16/01_baseline.txt
```

### 2.2 Report Generation (axiom_report)

```bash
cargo run --release --bin axiom_report
```

This binary lives in `axiom-llm/src/bin/axiom_report.rs`. It loads all datasets from `axiom-datasets/`, routes every sentence through AXIOM, and generates a comprehensive markdown report. No API calls -- simulation only. Requires `axiom_weights.json` and `axiom_vocab.json` to exist from a prior `axiom-bench` run.

### 2.3 Testing

```bash
# Run all workspace tests
cargo test

# Run only axiom-core tests
cargo test -p axiom-core

# Run only axiom-bench tests
cargo test -p axiom-bench

# Run only axiom-tuner tests
cargo test -p axiom-tuner
```

Current test count: ~170 tests across the workspace (65+ core, 5+ tuner, bench corpus tests, etc.).

### 2.4 Output Files Produced

After a full `axiom-bench` run, the following files are produced in the project root:

| File | Description |
|------|-------------|
| `axiom_weights.json` | Serialized weight matrices for all graph nodes |
| `axiom_vocab.json` | Tokeniser vocabulary (up to 1024 tokens) |
| `axiom_config.json` | Auto-tuner configuration (thresholds, base confidences, rationale) |
| `axiom_bench_log.json` | Per-iteration training log (sentence, tier, confidence, compute cost) |
| `axiom_text_train_log.json` | Text training pass detailed log |
| `axiom_text_log.json` | Validation text pass log (pre-training) |
| `axiom_text_pass2.json` | Post-training validation pass 2 |
| `axiom_text_pass4.json` | Post-training validation pass 4 |
| `axiom_text_log_pass3.json` | Text log pass 3 |
| `axiom_training_snapshots.json` | Periodic snapshots during training (weight norms, gaps, ordering) |
| `axiom_coalition_log.json` | Coalition formation log (which nodes bid, fired, resolved) |
| `axiom_contrastive_log.json` | Contrastive learning events (positive/negative counts, weight changes) |
| `axiom_contrastive_log_p3.json` | Contrastive log pass 3 |
| `axiom_adversarial.json` | Adversarial test results |
| `axiom_validation.json` | Full validation results |
| `axiom_error_log.json` | Error-driven learning log |
| `axiom_penalty_diagnostic.json` | G5 penalty diagnostic data |
| `axiom_pretrain_log.json` | Pre-training log |
| `axiom_text_priming_log.json` | Text priming log |
| `axiom_synth_pass1.json` | Synthetic pass 1 log |
| `axiom_synth_pass3.json` | Synthetic pass 3 log |
| `axiom_weight_log.json` | Weight drift log |

---

## 3. Adversarial Corpus

### 3.1 adversarial_sentences.txt (20 sentences)

```
Cogito ergo sum
Godel's proof breaks mathematics
Being precedes essence
the big red dog ran quickly down the long straight road toward the tall old brown wooden fence near the small quiet house by the river
the tintinnabulation resonated melodiously
photosynthesis converts sunlight efficiently
the cat that the dog that the man owned chased sat on the mat
she said that he thought that they believed it was true
neural networks approximate complex nonlinear functions through hierarchical feature learning
it is raining today
the interplay between cooperation and competition drives evolutionary dynamics
the sky is blue
water flows downhill
consciousness remains one of the most profound unsolved problems in all of science
if then else
the mitochondria is the powerhouse of the cell
dark matter constitutes approximately twenty seven percent of the total mass energy content of the observable universe
she runs fast
go
```

### 3.2 Extended Adversarial Test Set (from axiom-bench, 41 sentences total)

The adversarial curriculum in `axiom-bench/src/corpus.rs` includes multiple categories:

**Adversarial simple -- long but no nesting (15-25 words, flat SVO):**
```
the big brown dog ran quickly down the long narrow road past the old grey stone church near the river
she walked slowly along the quiet path by the tall green hedges under the bright blue sky every single morning
the old man sat on the long wooden bench near the calm blue lake and watched the white swans all day
he placed the large red book and the small blue pen on the wide brown table beside the open window
the young girl picked the fresh red strawberries from the low bushes in the wide green field behind the old farmhouse
the tall thin cat crept carefully along the narrow stone wall past the sleeping dog and into the warm dark barn
she carried the heavy brown box up the steep narrow stairs through the long dark hallway and into the small back room
the bright yellow taxi drove quickly through the busy wet streets past the tall grey buildings to the train station entrance
they ate the warm fresh bread with thick golden butter and cold sweet jam on the old wooden table outside
the white fluffy clouds drifted slowly across the pale blue sky above the quiet green hills and the calm dark lake
he walked his two small dogs along the straight flat road past the new red houses and the old grey shops
she put the clean white plates and the small silver forks on the round dark table by the kitchen window
the cold clear water ran fast over the smooth brown stones under the low wooden bridge near the old mill
the small black bird sat on the top branch of the tall dead tree and sang its short bright song again
he wore his long dark coat and his thick red scarf and his old brown hat on the cold grey morning
```

**Adversarial simple -- rare/technical words with simple syntax:**
```
the tintinnabulation resonated melodiously
the chrysanthemums bloomed prolifically
the onomatopoeia echoed distinctly
the phosphorescence glowed ethereally
the acquiescence surprised everyone profoundly
the kaleidoscope refracted light beautifully
the seismograph registered tremors immediately
the stethoscope amplified heartbeats clearly
the reconnaissance revealed positions accurately
the bougainvillea climbed the pergola
the rhododendrons flowered spectacularly
the xylophone produced crystalline tones
the phenomenon baffled researchers entirely
the metamorphosis astonished the biologists
the fluorescence illuminated the specimen
```

**Additional adversarial simple -- technical words, simple grammar:**
```
the parallelepiped occupied the shelf
the otorhinolaryngologist examined the patient
the archipelago stretched across the horizon
the electromagnetic pulse disrupted the signal
the biodegradable packaging dissolved quickly
the seismometer detected the vibrations
the chronometer displayed the exact time
the spectrophotometer measured the wavelength
the oscilloscope showed the signal pattern
the incandescent filament glowed brightly
```

**Adversarial complex -- very short (from the full bench adversarial pass):**
```
Cogito ergo sum
Godel's proof breaks mathematics
Being precedes essence
time is relative
maps are not territories
meaning requires context
logic precedes language
infinity is countable sometimes
the old man the boats
we saw her duck
```

**Garden-path and ambiguous sentences (complex):**
```
the cat that the dog that the man owned chased sat on the mat
she said that he thought that they believed it was true
the horse raced past the barn fell
time flies like an arrow but fruit flies like a banana
I saw the man with the telescope
the complex houses married and single soldiers and their families
the cotton clothing is made of grows in Mississippi
he gave her cat food
```

### 3.3 Adversarial Results Summary (Phase 14, 41 sentences scored as 40 excluding 1 moderate)

Score: **22/40 correct (55.0%)**

Phase 12 baseline (original 17 scored): 8/17 (47.1%). Delta: +2 (IMPROVED).

**Failure modes identified:**
1. **Very short complex** (e.g., "Cogito ergo sum"): Too few tokens for structural features to register. G5 norm is indistinguishable from simple.
2. **Long simple** (e.g., "the big red dog ran quickly..."): High token count inflates G5 norm, causing false penalty escalation. Length and structural complexity are conflated.
3. **Domain-encoded complexity** (e.g., "neural networks approximate..."): Complexity is semantic (requires domain expertise), not syntactic. No structural feature can detect this without world knowledge.

---

## 4. Full Scenario Texts

All 6 scenarios from `/Users/colinolliver/Development/AXIOM/axiom-datasets/scenarios.json`:

### Scenario 01
- **ID:** scenario_01
- **Label:** simple
- **Notes:** Three paragraphs, common vocabulary, clear intent, no technical content. Tests chunking on simple multi-paragraph input. Should route Surface.
- **Text:**

> Hi there, I hope you are having a good week. I wanted to get in touch about my recent order that I placed last Tuesday. I have not received any shipping confirmation yet and was wondering if everything is okay with my order. The order number is 84729 and I ordered two blue t-shirts in size medium and one pair of black jeans in size 32. I paid with my Visa card and the payment went through fine according to my bank. Could you please let me know when my order will be shipped and provide a tracking number when available? I am not in a rush but would appreciate an update when you get a chance. Thank you very much for your help.

### Scenario 02
- **ID:** scenario_02
- **Label:** complex
- **Notes:** Seven words, structurally simple, semantically deep. Classic failure mode for structural encoding. Expected to incorrectly route Surface because there are no structural complexity markers. Honest failure case for the paper.
- **Text:**

> Reconcile Kant's categorical imperative with utilitarian ethics.

### Scenario 03
- **ID:** scenario_03
- **Label:** moderate
- **Notes:** Two paragraphs, technical but accessible, some subordination, moderate vocabulary. Should route Reasoning. Tests whether moderate complexity is handled correctly or over-escalates to Deep.
- **Text:**

> I am trying to understand how database indexing works and when I should use it in my application. I have read that indexes speed up queries but also slow down writes, and I am not sure how to decide when the tradeoff is worth it. My application has a users table with about 500,000 rows and I am running queries that filter by email address and by signup date. The queries are currently taking around 800 milliseconds which feels slow. Could you explain the key concepts I need to understand and give me a practical recommendation for this situation?

### Scenario 04
- **ID:** scenario_04
- **Label:** complex
- **Notes:** Dense academic prose, multiple embedded clauses, abstract philosophical vocabulary across three paragraphs. Should route Deep. Tests whether sustained complexity across paragraphs is correctly identified.
- **Text:**

> The relationship between consciousness and physical substrate has occupied philosophers and scientists for centuries, yet contemporary neuroscience has done little to resolve the fundamental tension between first-person phenomenal experience and third-person objective description. The hard problem, as Chalmers formulated it, asks not merely how the brain processes information but why such processing is accompanied by subjective experience at all. Functionalist accounts that identify mental states with computational roles face the challenge of explaining why any physical system implementing the right functional organisation should give rise to qualia rather than proceeding as a philosophical zombie, behaviourally indistinguishable from a conscious being yet entirely lacking inner experience. The multiple realisability argument further complicates matters by suggesting that consciousness is substrate-independent, which sits uneasily with neuroscientific evidence for specific neural correlates of conscious states. Perhaps the most promising avenue lies in integrated information theory, which attempts to provide a mathematical framework for consciousness grounded in the intrinsic causal structure of physical systems. Yet even this approach faces serious objections regarding its counterintuitive implications for the distribution of consciousness across simple physical systems and its difficulty in making empirically falsifiable predictions about the boundaries of conscious experience.

### Scenario 05
- **ID:** scenario_05
- **Label:** simple
- **Notes:** One paragraph of context followed by a simple question. The context adds length and some structure but the actual request is simple. Tests whether contextual framing incorrectly escalates a simple query.
- **Text:**

> I have been a customer for three years and generally really enjoy your service. Last month I upgraded my account to the premium tier and everything has been working well. I just have one quick question. How do I add a second user to my account?

### Scenario 06
- **ID:** scenario_06
- **Label:** moderate
- **Notes:** Mixed natural language and code tokens. Novel encoder challenge -- code has very different structural patterns than prose. A developer asking for help understanding existing code. Should route Reasoning.
- **Text:**

> I have inherited a codebase and I am trying to understand what this function does. Can you explain it to me and suggest any improvements? def calculate_score(users, threshold=0.75): results = [] for user in users: if user.get(active) and user.get(score, 0) >= threshold: results.append({id: user[id], score: user[score], rank: len(results) + 1}) return sorted(results, key=lambda x: x[score], reverse=True)

---

## 5. Parameter Scaling Experiment

### Configuration Comparison

| Metric | mid_dim=128 (1.2M params) | mid_dim=512 (4.8M params) |
|--------|--------------------------|--------------------------|
| Total parameters | 1,205,376 | ~4,800,000 |
| Training time | 3.4 min | 16.3 min |
| Peak RAM | 312 MB | ~1.2 GB |
| Adversarial score | 22/40 | 22/40 |
| Adversarial accuracy | 55.0% | 55.0% |

### Key Finding

Quadrupling the parameter count (4x) produced **identical adversarial accuracy** (22/40) while increasing training time by ~4.8x and RAM by ~3.8x. This demonstrates that the accuracy ceiling is not caused by parameter starvation but by the fundamental limitation of structural-only features: without semantic understanding, no amount of additional capacity can distinguish "Cogito ergo sum" (semantically complex, structurally simple) from "the dog runs fast" (simple in both dimensions).

### How to Reproduce

```bash
# 1.2M params (default)
AXIOM_MID_DIM=128 AXIOM_ITER=100000 cargo run --release -p axiom-bench

# 4.8M params
AXIOM_MID_DIM=512 AXIOM_ITER=100000 cargo run --release -p axiom-bench
```

### Phase 16 Environment Sweep Results

The `experiments/results/phase16/` directory contains 18 experiment runs exploring different feature combinations:

| Run | Description | File |
|-----|-------------|------|
| 01_baseline | Baseline (default settings) | `01_baseline.txt` |
| 02_clause | Clause depth features only | `02_clause.txt` |
| 03_academic | Academic vocabulary features only | `03_academic.txt` |
| 04_bidir | Bidirectional features only | `04_bidir.txt` |
| 05_clause_academic | Clause + academic combined | `05_clause_academic.txt` |
| 06_clause_bidir | Clause + bidirectional combined | `06_clause_bidir.txt` |
| 07_academic_bidir | Academic + bidirectional combined | `07_academic_bidir.txt` |
| 08_all_three | All three feature sets | `08_all_three.txt` |
| 09_all_twostage | All features + two-stage calibration | `09_all_twostage.txt` |
| 10_all_cosine | All features + cosine LR schedule | `10_all_cosine.txt` |
| 11_all_phased | All features + phased training | `11_all_phased.txt` |
| 12_all_200k | All features + 200k iterations | `12_all_200k.txt` |
| 13_bidir_phased | Bidirectional + phased training | `13_bidir_phased.txt` |
| 200k_iter | 200k iterations (extended) | `200k_iter.txt` |
| clr_001 | Contrastive LR 0.001 | `clr_001.txt` |
| cosine_lr | Cosine LR schedule | `cosine_lr.txt` |
| two_stage_calib | Two-stage calibration | `two_stage_calib.txt` |

---

## 6. Calibration Corpus

All 27 sentences from `HierarchicalResolver::calibration_corpus()` in `axiom-core/src/tiers/resolver.rs` (line 2292):

### Simple (7 sentences, 3-5 words, common words, no punctuation)

```
a red ball
open the door
the moon is round
cats like warm milk
she ate lunch today
rain falls from clouds
we walked home slowly
```

### Moderate (7 sentences, 8-12 words, mixed vocabulary)

```
climate change affects global food production patterns in many regions
binary search reduces lookup time to a logarithmic number of steps
vaccines stimulate the immune response by introducing weakened pathogen material
containerisation improves deployment consistency and reproducibility across different environments
interest rates directly influence consumer borrowing behaviour and spending decisions
gradient descent iteratively minimises a differentiable loss function over parameters
ocean currents redistribute thermal energy between northern and southern hemispheres
```

### Complex (6 sentences, 15+ words, rare vocabulary, punctuation, subordinate clauses)

```
the observer effect in quantum mechanics implies that measurement, by its very nature, fundamentally alters the state of the observed system
autopoietic systems, while maintaining strict organisational closure, nonetheless remain thermodynamically open to continuous energy and matter exchange
the no-free-lunch theorem establishes that no single optimisation algorithm can dominate across all possible problem classes without exception
dialectical materialism posits that internal contradictions within socioeconomic structures, rather than external forces alone, drive historical transformation
renormalisation group methods, when applied to systems near critical phase transitions, reveal universal scale-invariant behaviour across many physical phenomena
the frame problem in artificial intelligence highlights the fundamental difficulty of formally representing the implicit non-effects of actions within logical frameworks
```

### Multi-paragraph simple (3 sentences, common words, multiple sentences, no technical content)

```
I placed an order last week and have not received it yet. My order number is 5412. Can you check on it for me please?
Hi, I would like to cancel my subscription. I signed up three months ago but I no longer need the service. Thank you.
The weather has been really nice this week. We went to the park yesterday and the kids had a great time playing outside.
```

### Multi-paragraph moderate (2 sentences, technical but accessible, multiple sentences)

```
I am trying to set up a CI pipeline for our Node.js project. We use GitHub Actions and need to run unit tests and deploy to staging on each push. Can you walk me through the YAML configuration?
Our PostgreSQL database has been running slowly on queries that join three tables. The largest table has about two million rows. I have added indexes on the foreign keys but performance is still poor.
```

### Multi-paragraph complex (2 sentences, dense academic prose, multiple sentences, abstract vocabulary)

```
The epistemological implications of Bayesian inference extend well beyond statistical methodology into the foundations of scientific reasoning itself. When prior distributions encode substantive theoretical commitments, the boundary between inductive evidence and deductive presupposition becomes fundamentally blurred.
Distributed consensus under asynchronous network conditions remains provably impossible without relaxing either safety or liveness guarantees, as established by the FLP impossibility result. Practical systems must therefore navigate the tension between theoretical limitations and operational requirements through carefully designed failure detectors and timeout mechanisms.
```

### Calibration Procedure

Calibration uses these 27 sentences to set tier thresholds:
- Surface threshold is set at the 65th percentile of Surface node confidence across all 27 sentences.
- Reasoning threshold is set at the 35th percentile.
- Recalibration occurs after each training phase and after orthogonal R+D initialisation.

---

## 7. Ablation Study Plan

### 7.1 Remove G5 Features (use G1-G4 only, 116 dimensions)

- **What to change:** Set `AXIOM_G5_WEIGHT=0.0` and disable the G5 feature block (dims 116-127) in the encoder. Input dimension drops from 128 to 116.
- **Expected effect:** Loss of structural syntax features (clause depth estimation, dependency distance, syntactic complexity metrics). Adversarial accuracy on garden-path and nested-clause sentences will drop significantly. Short complex sentences may be unaffected (they already fail with G5).
- **What it proves:** Quantifies the contribution of structural syntax features to routing accuracy beyond basic positional and vocabulary features. Demonstrates that G5 features capture genuine syntactic complexity that G1-G4 (position weighting, token_count_norm, TTR, avg_token_len, punct_density) cannot.

### 7.2 Remove Coalition Formation (single-node routing)

- **What to change:** Set `AXIOM_COALITION_MAX=1` so only the single highest-bidding node processes each input.
- **Expected effect:** Loss of multi-perspective evaluation. Each input sees only one node's opinion rather than a consensus of up to 4 nodes. Expected decrease in robustness on borderline sentences where multiple nodes might disagree. R+D specialisation becomes less meaningful since only one specialist is consulted.
- **What it proves:** Quantifies the value of ensemble-style routing through coalition formation. Demonstrates whether dynamic node selection provides genuine routing improvement or whether a single node is sufficient.

### 7.3 Remove Hebbian Learning (frozen weights after init)

- **What to change:** Set `AXIOM_LR=0.0` and `AXIOM_ERROR_LR=0.0` and `AXIOM_CONTRASTIVE_LR=0.0`. Weights remain at their analytical/orthogonal initialisation values.
- **Expected effect:** Surface nodes still discriminate (they are analytically initialised toward simple sentence space). R+D nodes remain orthogonal but unspecialised. Overall routing should still work for simple sentences (analytical init is strong) but R+D accuracy may degrade because nodes never learn to specialise.
- **What it proves:** Isolates the contribution of online Hebbian learning from analytical initialisation. If accuracy is similar, the analytical init is doing most of the work. If accuracy drops significantly, Hebbian specialisation matters.

### 7.4 Remove Embedding Cache

- **What to change:** Set cache capacity to 0 or disable cache lookup entirely. Currently the cache has 256 slots with cosine similarity threshold 0.75.
- **Expected effect:** Each input is always routed through the full graph, never short-circuited. Inference latency increases for repeated or near-duplicate inputs. During training, cache is already bypassed (RouteMode::Training), so training accuracy is unaffected. Impact is primarily on inference speed, not routing quality.
- **What it proves:** Quantifies the latency benefit of the embedding cache for repeated queries. Demonstrates whether the cache's similarity-based lookup introduces routing errors (false matches) compared to full graph traversal.

### 7.5 Remove Lateral Edges

- **What to change:** Remove all LateralEdge connections from the graph. Traversal is strictly forward (tier-by-tier) with no cross-tier information sharing.
- **Expected effect:** Loss of lateral information flow between same-tier nodes. Coalition members can no longer share intermediate representations. May reduce routing diversity and cause nodes to converge on similar representations faster.
- **What it proves:** Quantifies the contribution of lateral connections to routing diversity and accuracy. Demonstrates whether cross-node information flow within a tier provides genuine benefit or is redundant with forward traversal.

### 7.6 Remove Temporal Buffer

- **What to change:** Disable the TemporalBuffer (ring buffer capacity 16, cosine similarity > 0.85, blend 0.7/0.3). No temporal context is maintained between inputs.
- **Expected effect:** Each input is treated independently with no memory of recent inputs. For single-sentence evaluation this has minimal impact (temporal buffer primarily helps with conversation-like sequences). For multi-turn scenarios, loss of temporal context may cause inconsistent routing.
- **What it proves:** Quantifies the value of temporal context for routing consistency. Demonstrates whether maintaining a short-term memory of recent inputs improves routing on related sequences.

### 7.7 Vary Chunk Escalation Threshold

- **Thresholds to test:** 0.30, 0.40, 0.50 (current default depends on calibrated surface threshold ~0.75-0.80)
- **Expected effect at 0.30:** Very aggressive chunking -- even mildly uncertain chunks trigger escalation. Over-escalation of simple multi-paragraph inputs. Higher Deep percentage.
- **Expected effect at 0.40:** Moderate chunking. Balanced between false escalation and missed complexity.
- **Expected effect at 0.50:** Conservative chunking -- only very uncertain chunks escalate. Risk of under-escalation for moderately complex multi-paragraph inputs.
- **What it proves:** Sensitivity analysis on the chunk-level escalation decision. Identifies the optimal operating point for multi-paragraph routing and quantifies the false-escalation vs. missed-complexity tradeoff.

### 7.8 Vary G5 Penalty Weight

- **Weights to test:** 0 (disabled), 0.15 (light), 0.25 (medium), 0.35 (current default)
- **Expected effect at 0:** No structural penalty. Routing relies entirely on cosine similarity with Surface weights. Garden-path and nested-clause sentences are likely misrouted to Surface.
- **Expected effect at 0.15:** Mild penalty. Some structural escalation but insufficient for deeply nested clauses.
- **Expected effect at 0.25:** Moderate penalty. Most nested clauses correctly escalated. Some long-simple sentences may still be incorrectly escalated.
- **Expected effect at 0.35:** Current setting. Strongest structural penalty. Best adversarial accuracy (22/40) but also strongest false escalation of long-simple sentences.
- **What it proves:** Sensitivity analysis on the G5 penalty magnitude. Maps the accuracy-vs-false-escalation Pareto frontier. Demonstrates diminishing returns beyond a certain penalty weight. Shows that the structural penalty has an inherent ceiling regardless of weight.

### Recommended Ablation Table Format for Paper

| Ablation | Param Count | Simple Acc | Complex Acc | Adversarial | Delta vs Full |
|----------|------------|------------|-------------|-------------|---------------|
| Full system | 1,205,376 | 100% S | 22% | 22/40 (55%) | -- |
| No G5 | ~1.05M | ? | ? | ? | ? |
| No coalition | 1,205,376 | ? | ? | ? | ? |
| No learning | 1,205,376 | ? | ? | ? | ? |
| No cache | 1,205,376 | ? | ? | ? | ? |
| No lateral | 1,205,376 | ? | ? | ? | ? |
| No temporal | 1,205,376 | ? | ? | ? | ? |
| G5 weight=0 | 1,205,376 | ? | ? | ? | ? |
| G5 weight=0.15 | 1,205,376 | ? | ? | ? | ? |
| G5 weight=0.25 | 1,205,376 | ? | ? | ? | ? |

---

## 8. Training Corpus Statistics

### 8.1 Inline Corpus (axiom-bench/src/corpus.rs)

The inline corpus in `Corpus::load()` is organized by thematic categories:

**Simple sentences (>= 820):**
- Animals (30 sentences)
- Weather (20 sentences)
- Food (20 sentences)
- Basic actions (20 sentences)
- Everyday objects (20 sentences)
- Simple descriptions (20 sentences)
- More animals -- varied species (10)
- More food (10)
- More nature (10)
- Emotions and states (10)
- Work and occupation (10)
- Household chores (10)
- Simple questions as statements (5)
- Simple numbers and measurements (5)
- More daily routine (10)
- Technology simple (10)
- Health simple (10)
- Shopping and errands (10)
- Night and sleep (10)
- Body and senses (10)
- Buildings and places (10)
- Animals continued (10)
- Colours and appearances (10)
- Miscellaneous simple (20)
- More animals (10), More weather (10), More everyday (10), Nature (10), More simple descriptions (10)
- Extra simple -- colours and sizes (10)
- Extra simple -- daily routine (10)
- Sports (20)
- Music (20)
- Art (15)
- Geography and places (15)
- School and learning (15)
- Transport (15)
- Home and garden (15)
- Time and seasons (10)
- Passive voice simple (10)
- Lists in simple (10)
- Adversarial simple -- long flat SVO (15)
- Adversarial simple -- rare/technical words (15)
- More flat long adversarial simple (5)
- More adversarial simple -- technical words (10)
- Water and sea (10)
- Clothing and appearance (10)
- Friendship and social (10)
- Games and play (10)
- Light and darkness (10)
- Sound (10)
- Textures and materials (10)
- Cooking actions (10)
- Weather continued (10)
- Gardening (10)
- Directions and position (10)
- Quantities and sizes (10)
- Opinions and preferences (10)
- Actions with tools (10)
- Pairs and groups (10)
- Time expressions (10)
- Additional miscellaneous (20)
- Holidays and celebrations (10)
- Money and numbers (10)
- Communication (10)
- Nature sounds and scenes (10)
- More passive voice (5)
- Adversarial curriculum: long simple (10), rare words simple (various)

**Moderate sentences (>= 850):**
- Science basics (20)
- Technology (15)
- Geography (15)
- History (15)
- Everyday complexity (15)
- Mixed domain (11)
- Mathematics moderate (10)
- Computer science moderate (10)
- Linguistics moderate (10)
- Philosophy moderate (10)
- Environment moderate (10)
- Sports moderate (5)
- Music moderate (5)
- Art moderate (15)
- Technology moderate continued (30)
- Mixed domain moderate (15)
- Additional moderate diverse topics (20)
- Nutrition and health moderate (10)
- Architecture and design moderate (10)
- Sociology and anthropology moderate (10)
- Education moderate (10)
- Transport and logistics moderate (10)
- Agriculture moderate (10)
- Energy moderate (10)
- Communication and media moderate (10+)

**Complex sentences (>= 850):**
- Philosophy and epistemology
- Formal systems and logic
- Quantum mechanics and physics
- Cognitive science and consciousness
- Mathematics and computation theory
- Biology and evolution
- Information theory
- Game theory and complexity
- Adversarial complex -- very short philosophical
- Adversarial complex -- garden paths and ambiguity
- Additional short adversarial batches (multiple rounds)

Total assertions in tests:
- `corpus.simple.len() >= 800`
- `corpus.moderate.len() >= 850`
- `corpus.complex.len() >= 850`
- `corpus.all().len() >= 2500`

Actual count from Phase 15 run output: **820 simple + 850 moderate + 888 complex = 2558 total**.

Mean word counts: simple=6.8, moderate=15.8, complex=31.8.

Vocabulary size: 9,847 unique words.

### 8.2 Multi-Paragraph Corpus (axiom-datasets/multi_paragraph_corpus.json)

- 100 entries total (402 lines JSON)
- Distribution: **34 simple, 33 moderate, 33 complex**
- Added to the inline corpus at runtime, bringing total to ~2658 sentences
- Contains multi-sentence, multi-paragraph inputs designed to test the sentence chunking system

### 8.3 Other Datasets (axiom-datasets/)

| File | Description |
|------|-------------|
| `simple.json` | Simple sentence dataset (used by axiom_report) |
| `complex.json` | Complex sentence dataset (used by axiom_report) |
| `realistic.json` | Realistic mixed-complexity dataset (used by axiom_report) |
| `scenarios.json` | 6 hand-crafted scenario texts with notes (see Section 4) |
| `multi_paragraph_corpus.json` | 100 multi-paragraph entries (see above) |

### 8.4 Validation Sentences (axiom-bench/src/main.rs)

Hardcoded in `test_sentences()`: **105 total (40 simple, 33 moderate, 32 complex)**.

These are never used for training -- only for pre/post evaluation of routing accuracy.

### 8.5 Diagnostic Sentences (axiom-bench/src/main.rs)

6 fixed diagnostic sentences used throughout all phases for tracking learning progress:

```
Simple:
  "the dog runs fast"
  "the cat sat on the mat"
  "birds fly south in winter"

Complex:
  "the recursive nature of self-referential systems creates emergent properties that resist reduction"
  "quantum entanglement challenges classical notions of locality and causality"
  "the church-turing thesis equates mechanical computation with recursive function theory"
```

---

## Phase History (for context)

From experiment baseline output:

| Phase | Result |
|-------|--------|
| 4 | simple 13% S, complex 47% S -- inverted pre-learning |
| 5 | 100% S everything -- over-converged |
| 6 | simple 20% S, complex 71% S -- inverted post-contrastive |
| 7 | simple 73% S, complex 88% S -- inverted post-synthetic |
| 8 | simple 13% S, complex 100% S -- inverted text-only |
| 9 | simple 0% S, complex 100% S -- cosine init, training broke it |
| 10 | simple 0% S, complex 100% S -- Oja overwrites discrimination |
| 11 | simple 93% S, complex 94% S -- CORRECT analytical frozen Surface |
| 12 | simple 93% S, complex 53% S -- CORRECT richer encoder frozen |
| 13 | simple 100% S, complex 100% S -- coalition formation, no discrimination |
| 14 | simple 95% S, complex 38% S -- CORRECT G5 magnitude penalty |

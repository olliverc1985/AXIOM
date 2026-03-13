# AXIOM Prior Art Survey: LLM Routing, Complexity Estimation, and Related Work

Compiled: 2026-03-13

---

## 1. LLM Routing / Cascading / Model Selection

### 1.1 FrugalGPT
- **Paper**: "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"
- **Authors**: Lingjiao Chen, Matei Zaharia, James Zou (Stanford)
- **Year/Venue**: 2023, arXiv:2305.05176; ICLR 2025
- **What it does**: Learns which combinations of LLMs to use for different queries via an LLM cascade that sequentially tries cheaper models first.
- **Routing**: Sequential cascade. Uses a DistilBERT-based generation scoring function g(q,a) to rate response reliability (0-1). If score exceeds threshold, stops; otherwise escalates to next model. Router learns optimal LLM ordering and thresholds via constrained optimization.
- **Training data**: Requires labeled training examples with LLM outputs to train the scoring function.
- **Routing latency**: Not reported (DistilBERT inference per step).
- **Implementation**: Python (implied).
- **Key differences from AXIOM**: Cascade (sequential) vs. AXIOM's direct routing; requires running cheaper models first and scoring their outputs; uses DistilBERT (a transformer) for scoring; needs LLM output labels for training.

### 1.2 RouteLLM
- **Paper**: "RouteLLM: Learning to Route LLMs with Preference Data"
- **Authors**: Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, Ion Stoica (LMSYS / UC Berkeley)
- **Year/Venue**: 2024, arXiv:2406.18665; ICLR 2025
- **What it does**: Trains router models to dynamically select between a stronger and weaker LLM, achieving 85%+ cost reduction on MT Bench while maintaining 95% of GPT-4 performance.
- **Routing**: Binary (strong vs. weak). Four router architectures compared: (1) Matrix factorization (best performer, recommendation-system inspired), (2) BERT classifier (DeBERTa [CLS] token + logistic regression), (3) Causal LLM classifier (Llama 3 8B), (4) Similarity-weighted ranking. Binary threshold determines routing.
- **Training data**: Human preference data from Chatbot Arena (80K+ battles). Data augmentation with golden-label datasets.
- **Routing latency**: Not explicitly reported. Matrix factorization is lightweight; BERT/LLM classifiers add transformer inference overhead.
- **Implementation**: Python, open-source framework.
- **Key differences from AXIOM**: Binary routing only (strong vs. weak); all routers use heavyweight ML models (BERT, LLMs, learned embeddings); requires massive human preference datasets; semantic embedding-based features rather than structural/syntactic.

### 1.3 Hybrid LLM
- **Paper**: "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing"
- **Authors**: Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Ruhle, Laks V.S. Lakshmanan, Ahmed Hassan Awadallah (Microsoft / UBC)
- **Year/Venue**: 2024, arXiv:2404.14618; ICLR 2024
- **What it does**: Routes queries between a small edge model (Llama-2 13B) and a large cloud model (GPT-3.5) based on predicted query difficulty.
- **Routing**: Binary. Uses DeBERTa-v3-large as the router, trained with binary cross-entropy to predict whether the small model can handle a query. Threshold is tunable at test time for dynamic quality-cost tradeoff.
- **Training data**: Requires paired outputs from both models on training queries to generate binary labels.
- **Routing latency**: DeBERTa-v3-large inference (~350M params). Not explicitly reported.
- **Implementation**: Python/PyTorch.
- **Key differences from AXIOM**: Binary only; uses a 350M-parameter transformer as the router (vs. AXIOM's ~1M params, zero ML framework); requires paired LLM outputs for training; semantic features only.

### 1.4 AutoMix
- **Paper**: "AutoMix: Automatically Mixing Language Models"
- **Authors**: Pranjal Aggarwal, Aman Madaan, Ankit Anand, et al.
- **Year/Venue**: 2023, arXiv:2310.12963; NeurIPS 2024
- **What it does**: Routes queries to appropriately-sized models using self-verification and a POMDP-based router.
- **Routing**: Multi-tier. First runs small model, then uses few-shot self-verification to estimate output reliability, then a POMDP router decides whether to escalate. Novel: routes "unsolvable" queries to no model.
- **Training data**: Few-shot prompting for self-verification (no extensive training). POMDP router trained on small dataset.
- **Routing latency**: Requires running the small model first + self-verification LLM call. High overhead.
- **Implementation**: Python.
- **Key differences from AXIOM**: Requires running small model + verification before routing decision; POMDP is online learning but needs LLM calls; much higher routing latency (multiple LLM calls vs. sub-millisecond).

### 1.5 ZOOTER (Routing to the Expert)
- **Paper**: "Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models"
- **Authors**: Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, Jingren Zhou
- **Year/Venue**: 2023, arXiv:2311.08692; NAACL 2024
- **What it does**: Distills reward model scores into a lightweight routing function that directs queries to the expert LLM.
- **Routing**: Multi-model. Uses reward model to score training queries across all candidate LLMs, then trains a routing function with tag-based label enhancement to mitigate reward noise.
- **Training data**: Requires reward model + LLM outputs on training queries.
- **Routing latency**: Lightweight routing function (minor overhead). Not precisely reported.
- **Implementation**: Python.
- **Key differences from AXIOM**: Depends on reward models (which are themselves large neural nets); uses semantic embeddings for routing; requires LLM outputs during training.

### 1.6 LLM-Blender
- **Paper**: "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion"
- **Authors**: Dongfu Jiang, Xiang Ren, Bill Yuchen Lin
- **Year/Venue**: 2023, arXiv:2306.02561; ACL 2023
- **What it does**: Ensembles multiple LLMs via PairRanker (pairwise comparison of outputs) and GenFuser (merging top candidates).
- **Routing**: Not a router per se -- runs ALL models, ranks outputs, then fuses. Cross-attention encoders compare candidate pairs.
- **Training data**: MixInstruct benchmark with oracle pairwise comparisons.
- **Routing latency**: Very high -- requires running all candidate LLMs first.
- **Implementation**: Python/PyTorch.
- **Key differences from AXIOM**: Ensemble (runs all models) vs. routing (picks one); extremely high latency and cost; opposite philosophy to AXIOM's lightweight pre-routing.

### 1.7 MetaLLM
- **Paper**: "MetaLLM: A High-performant and Cost-efficient Dynamic Framework for Wrapping LLMs"
- **Authors**: Quang H. Nguyen et al.
- **Year/Venue**: 2024, arXiv:2407.10834
- **What it does**: Uses multi-armed bandit algorithm to dynamically route queries to the least expensive LLM likely to give correct answers.
- **Routing**: Multi-model, online learning. Multi-armed bandit formulation balances exploration/exploitation. No reward model needed.
- **Training data**: Learns online from observed correctness. No pre-training dataset required.
- **Routing latency**: Bandit computation is lightweight.
- **Implementation**: Python.
- **Key differences from AXIOM**: Bandit-based (no learned features, purely reward-based); doesn't analyze query content/structure; no structural features; learns from LLM correctness feedback rather than input complexity.

### 1.8 EcoAssistant
- **Paper**: "EcoAssistant: Using LLM Assistant More Affordably and Accurately"
- **Authors**: Jieyu Zhang et al.
- **Year/Venue**: 2023, arXiv:2310.03046
- **What it does**: Hierarchical LLM cascade for code-driven QA with solution retrieval from past successes.
- **Routing**: Sequential cascade. Tries weaker/cheaper LLMs first, backs off to stronger ones. Retrieves past solutions as in-context demonstrations.
- **Training data**: No explicit training -- learns from past successful queries.
- **Routing latency**: Sequential (must run cheaper models first).
- **Implementation**: Python, built on AutoGen.
- **Key differences from AXIOM**: Sequential cascade; no input analysis; relies on code execution feedback; domain-specific (code generation).

### 1.9 Unified Routing and Cascading
- **Paper**: "A Unified Approach to Routing and Cascading for LLMs"
- **Authors**: Jasper Dekoninck, Maximilian Baader, Martin Vechev (ETH Zurich)
- **Year/Venue**: 2024, arXiv:2410.10347; ICLR 2025 / ICML 2025
- **What it does**: Derives theoretically optimal strategies for both routing and cascading, and proposes "cascade routing" that unifies both.
- **Routing**: Unified framework. Proves optimality conditions for when routing vs. cascading vs. combined is best.
- **Training data**: Theoretical framework; specific instantiations vary.
- **Key differences from AXIOM**: Theoretical framework rather than a system; doesn't specify router architecture; doesn't address the routing mechanism itself (features, learning).

### 1.10 MixLLM
- **Paper**: "MixLLM: Dynamic Routing in Mixed Large Language Models"
- **Authors**: Zhengzhang Chen et al.
- **Year/Venue**: 2025, arXiv:2502.18482; NAACL 2025
- **What it does**: Contextual-bandit-based routing with query tag enhancement, achieving 97% of GPT-4 quality at 24% cost.
- **Routing**: Multi-model. Uses query tags to enhance embeddings, lightweight prediction models for quality/cost estimation, and a meta-decision maker for assignment. Supports continual learning.
- **Training data**: Query-response pairs with quality labels.
- **Routing latency**: Lightweight prediction models. Not precisely reported.
- **Key differences from AXIOM**: Uses contextual bandits (not graph-based); semantic embeddings + tags rather than structural features; Python/ML framework dependent.

### 1.11 BEST-Route
- **Paper**: "BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute"
- **Authors**: Microsoft Research
- **Year/Venue**: 2025, arXiv:2506.22716; ICML 2025
- **What it does**: Routes and also decides how many responses to sample (best-of-n), achieving 60% cost reduction with <1% performance drop.
- **Routing**: Multi-model + multi-sample. Key insight: multiple responses from a cheap model + best-of-n selection can beat a single expensive model response.
- **Training data**: Requires quality labels on training queries.
- **Key differences from AXIOM**: Focus on sample-count optimization alongside routing; requires running models to evaluate; Python/ML framework.

### 1.12 RouterDC
- **Paper**: "RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models"
- **Authors**: Shuhao Chen et al.
- **Year/Venue**: 2024, arXiv:2409.19886; NeurIPS 2024
- **What it does**: Dual contrastive learning trains an encoder + LLM embeddings to route queries, outperforming individual LLMs by 2.76%.
- **Routing**: Multi-model. Encoder maps queries to embeddings; sample-sample contrastive loss clusters similar queries; sample-center contrastive loss maps clusters to best-performing LLMs.
- **Training data**: Requires LLM outputs on training queries for contrastive labels.
- **Key differences from AXIOM**: Contrastive learning on semantic embeddings; requires PyTorch; needs LLM output labels.

### 1.13 Cost-Spectrum Contrastive Routing (CSCR)
- **Paper**: "Cost-Aware Contrastive Routing for LLMs"
- **Authors**: Reza Shirkavand
- **Year/Venue**: 2025, arXiv:2508.12491; NeurIPS 2025 Spotlight
- **What it does**: Maps prompts and models into shared embedding space for fast, cost-sensitive selection.
- **Routing**: Multi-model. Uses logit footprints (open-source) or perplexity fingerprints (black-box APIs). Contrastive encoder trained to favor cheapest accurate expert. **Routing reduces to k-NN lookup via FAISS index at inference.**
- **Training data**: Model fingerprints + accuracy labels.
- **Routing latency**: **Microsecond latency** (k-NN lookup). Closest to AXIOM's sub-millisecond claim.
- **Key differences from AXIOM**: Uses FAISS (C++/Python library) for k-NN; requires model fingerprinting; semantic embedding-based; needs Python + FAISS dependency.

### 1.14 Router-R1
- **Paper**: "Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning"
- **Authors**: UIUC
- **Year/Venue**: 2025, arXiv:2506.09033; NeurIPS 2025
- **What it does**: Uses an LLM as the router itself, trained with RL to interleave reasoning ("think") with model invocation ("route").
- **Routing**: Multi-model, multi-round. The router is a full LLM that reasons about which model to call. RL reward includes format, outcome, and cost components.
- **Training data**: RL training on QA benchmarks with cost-aware reward.
- **Routing latency**: High (runs an LLM for routing decisions).
- **Key differences from AXIOM**: Router IS an LLM (huge overhead); multi-round routing; requires full ML stack.

### 1.15 C3PO
- **Paper**: "C3PO: Optimized Large Language Model Cascades with Probabilistic Cost Constraints for Reasoning"
- **Year/Venue**: 2025, arXiv:2511.07396; NeurIPS 2025
- **What it does**: Self-supervised cascade optimization with conformal prediction for cost control.
- **Routing**: Cascade with theoretical guarantees. Uses conformal prediction to bound cost-exceeding probability. Learns cascade rules from <1% of examples used by SOTA.
- **Training data**: Only unlabeled prompts needed (self-supervised).
- **Key differences from AXIOM**: Cascade (sequential), not direct routing; conformal prediction framework; Python/ML dependent.

### 1.16 Cascadia
- **Paper**: "Cascadia: An Efficient Cascade Serving System for Large Language Models"
- **Year/Venue**: 2025, arXiv:2506.04203
- **What it does**: Systems-level cascade framework using bi-level optimization for deployment + routing co-optimization.
- **Routing**: Cascade with system-level optimization. Mixed-integer linear programming for resource allocation + Chebyshev-guided routing optimization.
- **Key differences from AXIOM**: Systems/infrastructure focus; cascade not direct routing; heavy optimization framework.

### 1.17 Universal Model Routing (UniRoute)
- **Paper**: "Universal Model Routing for Efficient LLM Inference"
- **Authors**: Wittawat Jitkrittum et al. (Google DeepMind)
- **Year/Venue**: 2025, arXiv:2502.08773
- **What it does**: Represents LLMs as feature vectors from representative prompt predictions; enables routing to unseen models.
- **Routing**: Multi-model, generalizable. Cluster-based routing or learned cluster map. Can route to LLMs not seen during training.
- **Key differences from AXIOM**: Focuses on model representation (not query analysis); needs representative prompt evaluations; Python/ML framework.

### 1.18 EmbedLLM
- **Paper**: "EmbedLLM: Learning Compact Representations of Large Language Models"
- **Authors**: Richard Zhuang et al. (UC Berkeley)
- **Year/Venue**: 2024, arXiv:2410.02223
- **What it does**: Learns compact vector embeddings of LLMs for downstream routing and benchmark prediction.
- **Routing**: Encoder-decoder learns LLM embeddings; used for model routing and performance forecasting.
- **Key differences from AXIOM**: Focuses on representing models (not queries); complementary approach.

### 1.19 LLMRank
- **Paper**: "LLMRank: Understanding LLM Strengths for Model Routing"
- **Authors**: Shubham Agrawal (Zeno AI)
- **Year/Venue**: 2025, arXiv:2510.01234
- **What it does**: Prompt-aware routing using human-readable features including task type, reasoning patterns, complexity indicators, and syntactic cues.
- **Routing**: Multi-model. **Uses explicit features including complexity indicators and syntactic cues** -- closest feature-engineering approach to AXIOM. Neural ranking model on RouterBench. Achieves 89.2% of oracle utility.
- **Training data**: RouterBench (36K prompts, 11 LLMs).
- **Key differences from AXIOM**: Still uses neural ranking model (needs ML framework); features extracted by LLM calls; much larger model; doesn't learn online.

### 1.20 Routoo
- **Paper**: "Routoo: Learning to Route to Large Language Models Effectively"
- **Year/Venue**: 2024, arXiv:2401.13979
- **What it does**: Uses a lightweight LLM as performance predictor + cost-aware selector.
- **Routing**: Multi-model. LLM-based performance prediction without running candidate models.
- **Key differences from AXIOM**: Uses an LLM for prediction (heavyweight); Python-based.

### 1.21 Arch-Router
- **Paper**: "Arch-Router: Aligning LLM Routing with Human Preferences"
- **Year/Venue**: 2025, arXiv:2506.16655
- **What it does**: 1.5B parameter model that maps queries to domain-action preferences for routing.
- **Routing**: Maps queries to human-defined taxonomy of domains/actions. Decouples route policy from model mapping.
- **Key differences from AXIOM**: 1.5B param router model; preference-aligned rather than complexity-based.

### 1.22 ICL-Router
- **Paper**: "ICL-Router: In-Context Learned Model Representations for LLM Routing"
- **Year/Venue**: 2025, arXiv:2510.09719; AAAI 2026
- **What it does**: Uses in-context vectors to represent model capabilities; enables adding new models without retraining.
- **Key differences from AXIOM**: LLM-based router; in-context learning mechanism.

### 1.23 InferenceDynamics
- **Paper**: "InferenceDynamics: Efficient Routing Across LLMs through Structured Capability and Knowledge Profiling"
- **Year/Venue**: 2025, arXiv:2505.16303
- **What it does**: Routes via structured capability/knowledge profiling. Models capabilities and knowledge dimensions of both queries and models.
- **Key differences from AXIOM**: Score-based profiling (not graph-based); needs evaluation dataset per new model.

### 1.24 RACER
- **Paper**: "RACER: Risk-Aware Calibrated Efficient Routing for Large Language Models"
- **Year/Venue**: 2026, arXiv:2603.06616
- **What it does**: Extends base routers to output model SETS with misrouting risk control via concentration bounds.
- **Key differences from AXIOM**: Meta-router (augments existing routers); set-valued output rather than single selection.

### 1.25 Rerouting LLM Routers (Adversarial)
- **Paper**: "Rerouting LLM Routers"
- **Authors**: Avital Shafran et al. (Hebrew University)
- **Year/Venue**: 2025, arXiv:2501.01818; COLM 2025
- **What it does**: Demonstrates adversarial attacks on LLM routers using "confounder gadgets" -- query-independent token sequences that force routing to strong models.
- **Key differences from AXIOM**: AXIOM's structural features (not semantic embeddings) may be more robust to adversarial token injection, since routing is based on syntactic properties rather than learned token representations.

---

## 2. Adaptive Compute / Early Exit (Related but Distinct)

### 2.1 CALM (Confident Adaptive Language Modeling)
- **Paper**: "Confident Adaptive Language Modeling"
- **Authors**: Google Research
- **Year/Venue**: 2022, arXiv:2207.07061; NeurIPS 2022
- **What it does**: Dynamically allocates compute per token by exiting early from decoder layers when confidence is high.
- **Routing**: Per-token early exit within a single model (not between models). Confidence measures: softmax response, state propagation (cosine distance), early-exit classifier.
- **Key differences from AXIOM**: Intra-model compute allocation vs. inter-model routing; per-token vs. per-query; requires access to model internals.

### 2.2 Mixture-of-Depths (MoD)
- **Paper**: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
- **Year/Venue**: 2024, arXiv:2404.02258
- **What it does**: Caps the number of tokens processed at each layer; top-k routing decides which tokens get full computation.
- **Key differences from AXIOM**: Intra-model token-level routing; modifies transformer architecture; not inter-model routing.

### 2.3 Cascade-Aware Training
- **Paper**: "Cascade-Aware Training of Language Models"
- **Year/Venue**: 2024, arXiv:2406.00060
- **What it does**: Trains small LM with awareness of its cascade position, using large model predictions to define loss.
- **Key differences from AXIOM**: Modifies model training (not router training); requires access to both models during training.

---

## 3. Surveys

### 3.1 "Doing More with Less"
- **Paper**: "Doing More with Less: A Survey on Routing Strategies for Resource Optimisation in Large Language Model-Based Systems"
- **Authors**: Clovis Varangot-Reille et al.
- **Year/Venue**: 2025, arXiv:2502.00409
- **Coverage**: Comprehensive survey of routing strategies including similarity-based, supervised, RL-based, and generative methods. Formalizes routing as performance-cost optimization.

### 3.2 "Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey"
- **Year/Venue**: 2026, arXiv:2603.04445
- **Coverage**: Most recent survey. Characterizes routing systems along: when decisions are made, what information is used, how they are computed.

### 3.3 RouterBench
- **Paper**: "RouterBench: A Benchmark for Multi-LLM Routing System"
- **Authors**: Qitian Jason Hu et al.
- **Year/Venue**: 2024, arXiv:2403.12031
- **Coverage**: 405K+ inference outcomes from representative LLMs. Theoretical framework + benchmark for evaluating routers.

### 3.4 RouterArena
- **Paper**: "RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers"
- **Year/Venue**: 2025, arXiv:2510.00202
- **Coverage**: 8,400 queries across 9 domains, 44 categories, multiple difficulty levels. Live leaderboard for academic and commercial routers.

---

## 4. Hebbian Learning in Modern ML

### 4.1 SoftHebb
- **Paper**: "Hebbian Deep Learning Without Feedback"
- **Authors**: Adrien Journe, Hector Garcia Rodriguez, Qinghai Guo, Timoleon Moraitis
- **Year/Venue**: 2022, arXiv:2209.11883; ICLR 2023 (Notable Top-25%)
- **What it does**: Trains deep CNNs with pure Hebbian learning (no backpropagation, no feedback signals). Achieves 99.4% MNIST, 80.3% CIFAR-10.
- **Key differences from AXIOM**: Image classification; pure unsupervised Hebbian; no routing application; PyTorch-based.

### 4.2 Oja's Rule in Deep Learning
- **Paper**: "Oja's plasticity rule overcomes challenges of training neural networks under biological constraints"
- **Year/Venue**: 2024, arXiv:2408.08408
- **What it does**: Shows Oja's rule stabilizes deep network training, mitigating exploding/vanishing gradients and reducing need for batch normalization.
- **Relevance to AXIOM**: AXIOM uses Hebbian learning (and could use Oja's rule for weight normalization). This validates Hebbian/Oja as viable for weight updates in applied systems.

### 4.3 Cannistraci-Hebb Training (CHT)
- **Paper**: "Brain network science modelling of sparse neural networks enables Transformers and LLMs to perform as fully connected"
- **Year/Venue**: 2025, arXiv:2501.19107
- **What it does**: Gradient-free, topology-driven link regrowth for sparse neural networks. At 1% connectivity, outperforms fully connected networks. At 5% connectivity, outperforms fully connected in machine translation.
- **Relevance to AXIOM**: Most closely related to AXIOM's sparse computation graph with Hebbian learning. Both use sparse connectivity + Hebbian-inspired updates. CHT focuses on intra-model sparsity; AXIOM uses sparse graph for inter-model routing.

### 4.4 Hebbian Learning the Local Structure of Language
- **Paper**: "Hebbian learning the local structure of language"
- **Authors**: P. Myles Eugenio
- **Year/Venue**: 2025, arXiv:2503.02057
- **What it does**: Derives a Hebbian language model with hierarchical tokenization and embedding via unsupervised replay. Learns morphology without data.
- **Relevance to AXIOM**: Directly relevant -- Hebbian learning applied to language processing. Shows Hebbian approaches can learn linguistic structure.

### 4.5 Hebbian-Oscillatory Co-Learning (HOC-L)
- **Paper**: "Hebbian-Oscillatory Co-Learning"
- **Year/Venue**: 2026, arXiv:2603.08731
- **What it does**: Couples Hebbian structural plasticity with oscillatory phase synchronization in sparse architectures. Proves convergence via Lyapunov function.
- **Relevance to AXIOM**: Theoretical grounding for Hebbian learning in sparse architectures (like AXIOM's sparse graph).

### 4.6 Continual Learning with Hebbian Plasticity Survey
- **Paper**: "Continual Learning with Hebbian Plasticity in Sparse and Predictive Coding Networks: A Survey and Perspective"
- **Authors**: Ali Safa
- **Year/Venue**: 2024, arXiv:2407.17305
- **Coverage**: Surveys Hebbian/STDP plasticity for continual learning at the edge. Relevant to AXIOM's online learning philosophy.

### 4.7 Emergence of Hebbian Dynamics in Regularized Non-Local Learners
- **Year/Venue**: 2025, arXiv:2505.18069
- **What it does**: Shows theoretical/empirical connection between SGD with weight decay and Hebbian learning near convergence.

---

## 5. Commercial / Production LLM Routers

### 5.1 Martian
- **Company**: Martian (backed by NEA, General Catalyst, $9M funding)
- **What it does**: Production LLM router. "Google for LLMs" -- selects best model per request optimizing cost, quality, latency. Used by 300+ companies. Accenture strategic investment (2024).
- **Approach**: Proprietary. Predicts model behavior using model compression/quantization/distillation techniques.
- **Key differences from AXIOM**: Proprietary black-box; likely Python-based; much larger system; production-grade but not academic.

### 5.2 OpenRouter
- **What it does**: Unified API providing access to 500+ models from 60+ providers. Load-balances by price, supports latency/cost routing shortcuts.
- **Key differences from AXIOM**: Infrastructure/API layer, not learned routing; provider-level rather than query-complexity-level routing.

### 5.3 Not Diamond
- **What it does**: AI model router that determines best-suited LLM per query. Maintains RouterArena benchmark.
- **Key differences from AXIOM**: Proprietary; likely transformer-based router; production service.

### 5.4 Unify
- **What it does**: Routes based on quality, time-to-first-token, inter-token latency, and cost metrics.
- **Key differences from AXIOM**: Metric-configuration-based routing; infrastructure focus.

---

## 6. Key Differentiators of AXIOM (vs. All Prior Art)

### 6.1 What No Existing System Does
Based on this comprehensive survey, AXIOM is unique in the following combination:

1. **Structural/syntactic features only**: All learned routers (RouteLLM, Hybrid LLM, RouterDC, CSCR, etc.) use semantic embeddings from transformers (BERT, DeBERTa, Llama). AXIOM uses hand-engineered structural features (token count, TTR, avg token length, punctuation density, positional weighting). LLMRank is the closest, using "syntactic cues" among other features, but still relies on neural ranking models and LLM-extracted features.

2. **Zero ML framework dependencies**: Every existing router requires Python + at least one ML framework (PyTorch, TensorFlow, FAISS, etc.). AXIOM is pure Rust with zero external ML dependencies.

3. **Hebbian learning for routing**: No existing LLM router uses Hebbian or Oja learning rules. Hebbian learning exists in vision (SoftHebb), sparse training (CHT), and language modeling (Eugenio 2025), but never for LLM routing decisions.

4. **Sparse computation graph**: Existing routers use dense classifiers (BERT), matrix factorization, bandits, or k-NN. None use a learned sparse graph with conditional/lateral edges and hierarchical tiers.

5. **Multi-tier routing (3+ tiers)**: Most routers are binary (strong vs. weak). AutoMix and ZOOTER support multi-model, but they use very different mechanisms. AXIOM's three-tier (Surface/Reasoning/Deep) with calibration percentiles is unique.

6. **Sub-millisecond routing with a learned model**: CSCR achieves microsecond latency but via k-NN lookup (FAISS), not a learned graph traversal. AXIOM achieves sub-millisecond through native Rust sparse graph traversal.

7. **Dynamic coalition formation**: No existing router uses stochastic node selection / coalition voting for routing decisions.

8. **Online Hebbian adaptation**: AXIOM adapts weights continuously during operation. Most routers are fixed after training. MetaLLM (bandit) and MixLLM (continual learning) adapt online but through different mechanisms.

9. **Memory footprint**: ~1M parameters vs. 110M+ (DistilBERT), 350M+ (DeBERTa), 8B (Llama router). AXIOM is 100-8000x smaller.

### 6.2 Potential Weaknesses to Address in Paper
- No standardized benchmark comparison (RouterBench, MMLU, etc.)
- No empirical comparison with RouteLLM/Hybrid LLM on same data
- Structural features may miss semantic difficulty (e.g., short but hard math questions)
- Hebbian learning convergence concerns (weight explosion noted in Phase 4)

---

## 7. Summary Table: Router Feature Comparison

| System | Year | Routing Type | Router Size | Features | Training Data | ML Framework | Latency | Online Learning |
|--------|------|-------------|-------------|----------|---------------|-------------|---------|-----------------|
| FrugalGPT | 2023 | Cascade | ~66M (DistilBERT) | Semantic | LLM outputs | Python/PyTorch | ~ms | No |
| RouteLLM | 2024 | Binary | 110M-8B | Semantic embeddings | 80K preferences | Python/PyTorch | ~ms | No |
| Hybrid LLM | 2024 | Binary | 350M (DeBERTa) | Semantic | Paired outputs | Python/PyTorch | ~ms | No |
| AutoMix | 2023 | Multi-tier | LLM-based | Self-verification | Few-shot | Python | ~seconds | No |
| ZOOTER | 2023 | Multi-model | Lightweight | Reward-based | Reward model | Python | Low | No |
| MetaLLM | 2024 | Multi-model | Minimal | Bandit rewards | Online | Python | Low | Yes (bandit) |
| RouterDC | 2024 | Multi-model | Encoder | Contrastive emb. | LLM outputs | Python/PyTorch | ~ms | No |
| CSCR | 2025 | Multi-model | Encoder+FAISS | Logit fingerprints | Model fingerprints | Python/FAISS | **us** | No |
| Router-R1 | 2025 | Multi-model | LLM | Reasoning | RL reward | Python/PyTorch | High | RL |
| MixLLM | 2025 | Multi-model | Lightweight | Tags + embeddings | Quality labels | Python | Low | Yes |
| LLMRank | 2025 | Multi-model | Neural ranker | **Syntactic cues** | RouterBench | Python | ~ms | No |
| **AXIOM** | **2026** | **3-tier** | **~1M** | **Structural only** | **Self-calibrated** | **Rust/none** | **<1ms** | **Yes (Hebbian)** |

---

## 8. References (by arXiv ID)

- 2207.07061 - CALM (Schuster et al., 2022)
- 2305.05176 - FrugalGPT (Chen, Zaharia, Zou, 2023)
- 2306.02561 - LLM-Blender (Jiang, Ren, Lin, 2023)
- 2310.03046 - EcoAssistant (Zhang et al., 2023)
- 2310.12963 - AutoMix (Aggarwal, Madaan et al., 2023)
- 2311.08692 - ZOOTER (Lu et al., 2023)
- 2401.13979 - Routoo (2024)
- 2403.12031 - RouterBench (Hu et al., 2024)
- 2404.02258 - Mixture-of-Depths (2024)
- 2404.14618 - Hybrid LLM (Ding et al., 2024)
- 2406.00060 - Cascade-Aware Training (2024)
- 2406.18665 - RouteLLM (Ong et al., 2024)
- 2407.10834 - MetaLLM (Nguyen et al., 2024)
- 2407.17305 - Hebbian Continual Learning Survey (Safa, 2024)
- 2408.08408 - Oja's Rule in Deep Learning (2024)
- 2408.12320 - PolyRouter (2024)
- 2409.19886 - RouterDC (Chen et al., 2024)
- 2410.02223 - EmbedLLM (Zhuang et al., 2024)
- 2410.10347 - Unified Routing & Cascading (Dekoninck et al., 2024)
- 2501.01818 - Rerouting LLM Routers (Shafran et al., 2025)
- 2501.19107 - Cannistraci-Hebb Training (2025)
- 2502.00409 - "Doing More with Less" Survey (Varangot-Reille et al., 2025)
- 2502.08773 - UniRoute (Jitkrittum et al., 2025)
- 2502.11021 - Uncertainty-Based Routing (2025)
- 2502.18482 - MixLLM (Chen et al., 2025)
- 2503.02057 - Hebbian Language Structure (Eugenio, 2025)
- 2505.16303 - InferenceDynamics (2025)
- 2506.04203 - Cascadia (2025)
- 2506.09033 - Router-R1 (2025)
- 2506.16655 - Arch-Router (2025)
- 2506.22716 - BEST-Route (Microsoft, 2025)
- 2508.12491 - CSCR (Shirkavand, 2025)
- 2509.09782 - One Head Many Models (2025)
- 2510.00202 - RouterArena (2025)
- 2510.01234 - LLMRank (Agrawal, 2025)
- 2510.09719 - ICL-Router (Wang et al., 2025)
- 2511.07396 - C3PO (2025)
- 2603.04445 - Routing/Cascading Survey (2026)
- 2603.06616 - RACER (2026)
- 2603.08731 - Hebbian-Oscillatory Co-Learning (2026)
- 2209.11883 - SoftHebb (Journe et al., 2022)

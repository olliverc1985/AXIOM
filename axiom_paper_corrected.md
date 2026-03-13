# AXIOM Paper — Corrected Draft

Read the SKILL.md at /mnt/skills/public/docx/SKILL.md before starting. Then write and produce a professional arXiv-style Word document saved as axiom_paper.docx in /mnt/user-data/outputs/. Use the exact content below verbatim — do not paraphrase, shorten, or reorder. Format with: title block, author, abstract in a shaded box, numbered sections with bold headings, all tables formatted as proper Word tables, code blocks in Courier New 9pt, references section at end. The paper must look publication-ready.

TITLE: AXIOM: Adaptive eXecution with Intelligent Operations Memory — A Sparse Dynamic Routing Architecture for Cost-Efficient LLM Inference

AUTHOR: Colin Oliver, Independent Research

ABSTRACT: We present AXIOM (Adaptive eXecution with Intelligent Operations Memory), a novel sparse routing architecture for cost-efficient large language model inference. AXIOM routes incoming queries across three model tiers — Surface, Reasoning, and Deep — using a 128-dimensional structural encoder and a hierarchical resolver with dynamic coalition formation and non-local graph communication, requiring no preference data, no GPU infrastructure, and no ML frameworks. Implemented in pure Rust with 1,205,376 parameters, AXIOM achieves 100% routing accuracy on simple queries, 58.0% overall accuracy across 200 benchmark queries, and 56.6% cost reduction compared to routing all queries to a frontier model, with a mean routing latency of 1,311 microseconds. The primary architectural contribution is a sparse computation graph supporting four distinct traversal directions — forward, lateral, feedback, and temporal — enabling non-local communication between routing nodes that no existing LLM router provides. We identify and characterise the structural encoder ceiling and propose attention-based extensions as future work.

SECTION 1 — INTRODUCTION

Large language models vary substantially in cost and capability. Frontier models such as Claude Opus and GPT-4 provide the highest quality responses but incur significant inference costs, while smaller models are substantially cheaper but may produce inadequate responses for complex queries. For organisations processing high query volumes, routing every query to the most capable model is economically prohibitive, while routing all queries to the cheapest model degrades output quality.

LLM routing addresses this problem by predicting query complexity and dispatching each query to an appropriately capable model. Existing approaches — most notably RouteLLM (Ong et al., 2024) — achieve strong results but require training on large datasets of human preference judgements, Python runtimes with PyTorch dependencies, and binary strong/weak routing decisions. A systematic survey of 75+ routing and cascading systems (see Section 2) reveals that every existing router makes a single forward-pass decision: input enters a classifier, a score exits, a model is selected. No existing router has nodes that communicate with each other about the routing decision.

AXIOM takes a fundamentally different approach. Its sparse computation graph supports four distinct communication patterns — forward traversal, lateral traversal between same-tier nodes, feedback traversal from deeper to shallower tiers, and temporal traversal blending past routing decisions into present ones. This non-local graph communication is the primary architectural contribution and has no equivalent in the LLM routing literature.

Secondary contributions include a vocabulary-independent 128-dimensional structural encoder using G5 syntactic features, analytical initialisation with frozen surface-tier weights, dynamic coalition formation across routing tiers, and a training/inference mode split preventing embedding cache contamination.

SECTION 2 — RELATED WORK

LLM routing. RouteLLM (Ong et al., 2024) trains routers on Chatbot Arena preference data, achieving up to 85% cost reduction with 95% GPT-4 performance on MT Bench. Four architectures are evaluated: matrix factorisation, weighted Elo, BERT classifier, and causal LLM classifier. All are single-pass classifiers producing a scalar score. FrugalGPT (Chen et al., 2023) uses a sequential cascade — attempt a cheap model, score the output, escalate if insufficient. AutoMix (Madaan et al., 2023) uses a self-verification loop — ask the model if it is confident, escalate if not. Hybrid LLM (Ding et al., 2024) trains a binary complexity classifier. CSCR uses k-NN lookup in embedding space. None of these systems have inter-node communication. The routing decision is always made by a single component.

The closest architectural analogy in the broader literature is Tryage (Hu et al., 2023), described as a "brain-inspired" thalamic router. However, Tryage is a single neural network predicting model performance — it has no actual inter-node communication. A survey published March 2026 (arXiv:2603.04445) provides a systematic analysis of 75+ multi-LLM routing and cascading approaches, covering query difficulty, preference-based, clustering, uncertainty quantification, reinforcement learning, and cascading paradigms. No surveyed system uses graph-based multi-directional communication between routing nodes.

Predictive coding networks on arbitrary graph topologies (Salvatori et al., NeurIPS 2022) are the closest architectural precedent for AXIOM's graph communication model, though applied in a completely different context. AXIOM is the first system to apply non-local graph communication to the LLM routing problem.

Mixture of Experts. The MoE paradigm (Shazeer et al., 2017; Fedus et al., 2022) routes tokens to expert networks within a single model. AXIOM applies routing at the inter-model level — complete queries rather than tokens — without a jointly trained gating network.

Complexity classification. Readability metrics (Kincaid et al., 1975) and syntactic complexity measures (Gibson, 1998; Yngve, 1960) inform AXIOM's encoder design. AXIOM's G5 features draw on these traditions while optimised for routing rather than readability scoring.

SECTION 3 — ARCHITECTURE

3.1 Overview

AXIOM comprises four components: a structural encoder producing a 128-dimensional representation; a sparse computation graph of ComputeNode instances with conditional, lateral, and feedback edges; a hierarchical resolver orchestrating routing decisions; and an embedding cache for repeated inputs. The system operates in two modes: RouteMode::Training (cache disabled) and RouteMode::Inference (cache enabled at cosine similarity threshold 0.92).

3.2 Non-Local Graph Communication

The central architectural contribution is AXIOM's sparse computation graph, which supports four distinct traversal directions. Every routing decision is traceable through a sequence of TraceStep records:

TraversalDirection { Forward, Lateral, Feedback, Temporal }

TraceStep { node_id, tier, direction, confidence_in, confidence_out, was_cached }

Forward traversal (Surface → Reasoning → Deep) is conditional: a ConditionalEdge fires only when its EdgeCondition evaluates true for the current routing state. EdgeCondition variants are: Always, IfConfidenceAbove(f32), IfConfidenceBelow(f32), IfTier(Tier). Which edges fire depends on the input — this is not a fixed pipeline.

Lateral traversal models cortical column behaviour. When a Surface node produces low confidence, lateral edges activate other Surface nodes at the same tier before escalating. A LateralEdge connects two same-tier nodes with a weight and LateralCondition. The RouteResult tracks lateral_count (how many lateral attempts were made) and lateral_prevented_escalation (how many avoided escalation to an expensive tier). This creates graceful degradation — the system exhausts cheap options before escalating.

Feedback traversal runs upward: when a Deep node resolves an input with confidence above 0.90, it emits a FeedbackSignal to shallower tiers. FeedbackSignal carries: from_node, to_tier, reason (LowConfidenceResolved | ContradictionDetected | CacheInvalidation), and confidence_delta. This is not backpropagation — it is directional confidence nudging. If Deep repeatedly resolves inputs that Reasoning escalated unnecessarily, Reasoning's base confidence lowers, reducing future over-escalation. The system corrects its own routing mistakes without external supervision.

Temporal traversal gives AXIOM memory across routing decisions. A ring buffer of capacity 16 stores recent routing results. When a new input arrives with cosine similarity above 0.85 to a recent input, the past result blends into the current routing: current_output = 0.7 × live_output + 0.3 × temporal_match. A burst of complex queries influences routing of subsequent queries even if they appear simple in isolation.

The communication topology distinguishes AXIOM from every surveyed alternative:

RouteLLM: Input → [BERT] → score → model selection
FrugalGPT: Input → [Model1] → score → maybe [Model2] → score → maybe [Model3]
AXIOM: Input → [Surface1] ←lateral→ [Surface2] → (conditional edge) → [Reasoning3] ←coalition→ [Deep6] → (feedback signal upward) with temporal_buffer blending throughout

3.3 Structural Encoder

The encoder produces a 128-dimensional vector divided into five feature groups. G1 (26 dimensions): character n-gram profiles, amplified 3.0×. G2 (36 dimensions): syntactic proxy features including nested clause depth, pronoun density, and hapax ratio, amplified 3.0×. G3 (39 dimensions): position-weighted token signal. G4 (15 dimensions): scalar complexity measures including type-token ratio and punctuation density, amplified 2.0×. G5 (12 dimensions): structural syntax features — dependency depth proxy, constituent length variance, and function word position entropy, amplified 3.0×.

G5 drives the magnitude penalty applied to Surface confidence. The full confidence formula is: cosine_sim = clamp(dot(input, weight_direction) / (||input|| × ||weight_direction|| + 1e-8), 0, 1); g5_penalty = clamp((g5_norm − g5_simple_mean_norm) / (g5_complex_mean_norm − g5_simple_mean_norm), 0, 1); confidence = base_confidence × 0.7 + cosine_sim × 0.3 − g5_penalty × 0.35. Parameters g5_simple_mean_norm = 2.4596 and g5_complex_mean_norm = 3.3316 are persisted to axiom_weights.json. For multi-sentence inputs, sentence chunking splits on punctuation boundaries, encodes independently, and averages G5 norms — confirmed ratio 1.00× on chunked versus unchunked equivalent input.

3.4 Analytical Initialisation and Frozen Surface Weights

Surface-tier nodes are initialised analytically from the mean direction of simple training examples (AnalyticalInit) and frozen throughout training. This is the central training invariant. The rationale: the Surface tier needs to identify definitively simple inputs. Any gradient signal applied to Surface weights risks corrupting the geometric separation established at initialisation. Reasoning and Deep nodes are initialised with near-orthogonal weight directions (OrthogonalInit, mean pairwise cosine 0.0032) and updated via Oja's rule during training.

3.5 Dynamic Coalition Formation

When input escalates past Surface, AXIOM forms a temporary coalition. Every Reasoning and Deep node computes cosine similarity to the input. Nodes above bid_threshold = 0.10 enter the bidding pool. The coalition is assembled by weighted random sampling up to max_coalition_size = 4 (stochastic selection, proportional to similarity score). The highest-bidding node's output becomes the final routing decision, but all coalition members process the input and update weights via Hebbian competition. This produces specialisation without explicit labels — nodes win bids on inputs they handle best, and their weights reinforce that specialisation.

A typical coalition: [reasoning_standalone_4 (bid=0.943), reasoning_standalone_13 (bid=0.940), deep_standalone_6 (bid=0.943, RESOLVED), reasoning_standalone_12 (bid=0.937)] with resolved_by=deep_standalone_6, cross_tier=true. Cross-tier resolutions — where a Reasoning node outbids Deep nodes — are tracked separately. Post-training: 19 of 30 R+D nodes activate regularly. Deep tier handles 9.0% of routed queries. R+D pairwise cosine drifts from 0.0032 at initialisation toward input-specific principal components via Oja convergence.

SECTION 4 — TRAINING METHODOLOGY

4.1 Corpus

Training corpus: 2,558 sentences from three sources. Manually constructed simple/complex sentence pairs across academic, conversational, technical, and narrative domains. A 100-entry multi-paragraph corpus (34 simple, 33 moderate, 33 complex). An adversarial curriculum of 40 sentences targeting known failure modes including very short semantically complex queries and long simple inputs.

4.2 Training Procedure

For each training example: encoder produces 128-dimensional representation; hierarchical resolver routes the input and computes confidence; Oja's rule updates activated Reasoning and Deep node weight directions; G5 population statistics accumulate. Surface nodes receive no updates. After training: calibration computes simple_mean_confidence, complex_mean_confidence, and Surface escalation threshold from a held-out set of 27 sentences. Auto-tuner writes final configuration to axiom_config.json. G5 parameters and node weights write to axiom_weights.json (10.1 MB).

4.3 Production Configuration

Production model: mid_dim = 128, 1,205,376 total parameters. Training time: 206 seconds (3.4 minutes). Peak RAM: 312 MB. No GPU at any stage. A scaling experiment at mid_dim = 512 (~4,800,000 parameters) produced identical adversarial accuracy (22/40) with approximately 4.7× longer training time. The bottleneck is the 128-dimensional encoder input, not node capacity — selected production configuration is 1.2M parameters.

SECTION 5 — EVALUATION

5.1 Benchmark Datasets

Three datasets were constructed. Simple dataset: 50 single sentences, ground truth "simple", spanning customer support, basic instructions, conversational queries. Complex dataset: 50 single sentences, ground truth "complex", spanning academic prose, technical analysis, multi-clause arguments, domain vocabulary. Realistic dataset: 100 enterprise queries — 40 simple, 30 moderate, 30 complex — reflecting actual LLM deployment patterns.

5.2 Routing Accuracy — INSERT TABLE 1

Dataset | Queries | Correct | Accuracy
Simple | 50 | 50 | 100.0%
Complex | 50 | 11 | 22.0%
Realistic | 100 | 55 | 55.0%
Overall | 200 | 116 | 58.0%

Simple routing accuracy is 100% — every simple query stays at Surface. This is the commercially critical result: false escalations to expensive tiers drive unnecessary cost, and AXIOM eliminates them entirely on the benchmark. Complex accuracy of 22% reflects the structural encoder ceiling described in Section 5.4.

5.3 Cost Model — INSERT TABLE 2

Tier | Model | Input $/M | Output $/M | Avg tokens in | Avg tokens out
Surface | claude-haiku-4-5 | $0.80 | $4.00 | 150 | 200
Reasoning | claude-sonnet-4-5 | $3.00 | $15.00 | 300 | 500
Deep | claude-opus-4 | $15.00 | $75.00 | 800 | 1500

Measured routing distribution across all 200 benchmark queries: Surface 79.5%, Reasoning 11.5%, Deep 9.0%. At this distribution AXIOM achieves 56.6% cost reduction versus all-Opus routing.

INSERT TABLE 3:
Query volume | AXIOM cost | All-Opus cost | Saving
1,000 | $12.90 | $29.75 | 56.6%
10,000 | $129.02 | $297.49 | 56.6%
100,000 | $1,290.24 | $2,974.88 | 56.6%

At 100,000 queries per day, AXIOM saves approximately $1,685 per day versus all-Opus routing. Cost savings are linear in query volume.

5.4 Structural Encoder Ceiling

Complex routing accuracy of 22% reflects a fundamental limitation. Three failure categories are identified. Category 1 — semantic complexity without syntactic markers: short queries like "Reconcile Kant's categorical imperative with utilitarian ethics" (7 words) present no G5 signal. The encoder cannot distinguish them from simple short queries. Category 2 — length inflation on simple inputs: multi-sentence simple inputs can inflate G5 norms if they contain subordinating conjunctions in simple contexts. Sentence chunking mitigates this (confirmed ratio 1.00×) but does not eliminate it entirely. Category 3 — domain vocabulary without structure: queries using technical vocabulary without syntactic complexity markers route incorrectly to Surface. These categories define the agenda for future work: a learned embedding layer mapping tokens to semantic representations before structural analysis.

5.5 Routing Latency

Mean routing time: 1,311 microseconds across 200 benchmark queries. This includes encoder computation, graph traversal, confidence evaluation, and embedding cache lookup. No API calls are made during routing. At 1,311 µs, AXIOM adds under 2 milliseconds of overhead to any LLM call — negligible against LLM inference latency of hundreds of milliseconds to seconds.

5.6 Multi-Paragraph Routing

Six enterprise scenario inputs tested multi-paragraph routing using threshold strategy C (escalate if more than 40% of chunks fall below Surface threshold, constant AXIOM_CHUNK_ESCALATION_THRESHOLD = 0.40). Accuracy: 3/6 correct. Primary failure mode: confidence compression in full-text encoding, where averaging structural features across long mixed inputs dilutes per-sentence complexity signals.

SECTION 6 — COMPETITIVE ANALYSIS — INSERT TABLE 4

Dimension | AXIOM | RouteLLM
Routing tiers | 3 (Surface, Reasoning, Deep) | 2 (strong, weak)
Training data | Structural corpus, no LLM labels | Chatbot Arena preference votes
Inter-node communication | Forward + lateral + feedback + temporal | None
Coalition formation | Dynamic per-query | None
Runtime | Pure Rust, no frameworks | Python + PyTorch
Routing latency | 1,311 µs | Model-scale (BERT/LLM routers)
Interpretability | Full trace: nodes, edges, direction, confidence | Scalar score
Encoder type | Vocabulary-independent structural | Semantic embedding
Reported cost savings | 56.6% vs all-Opus | Up to 85% vs GPT-4

The 56.6% vs 85% comparison requires context. RouteLLM's figure is achieved with routers trained on millions of human preference votes. AXIOM's figure is achieved with no preference data and no LLM calls during training. They represent different points in the accuracy–cost–dependency tradeoff space.

AXIOM's principal advantages: framework independence, zero LLM-output training dependency, three-tier granularity, non-local graph communication, interpretable routing traces, and sub-two-millisecond latency. RouteLLM's principal advantage: higher routing accuracy on semantic complexity due to its semantic training signal.

SECTION 7 — LIMITATIONS

Seven limitations are identified. First, the structural encoder ceiling prevents reliable detection of semantic complexity without syntactic markers — the binding constraint on overall accuracy. Second, complex routing accuracy of 22% means AXIOM under-escalates complex queries at a high rate. Third, multi-paragraph routing accuracy of 50% reflects confidence compression in full-text encoding. Fourth, the encoder is trained on English text — performance on other languages is untested. Fifth, no mechanism exists for learning from routing errors in production — online learning is an important future direction. Sixth, the cost model assumes fixed token counts per tier — real response length varies substantially. Seventh, the moderate/complex boundary is less well-defined than the Surface boundary, producing the largest share of errors on the realistic dataset.

SECTION 8 — FUTURE WORK

Five directions are identified. First, a learned embedding layer mapping tokens to semantic representations before structural analysis — directly addressing the structural encoder ceiling. Second, an HTTP API wrapper exposing AXIOM as a drop-in OpenAI-compatible routing endpoint. Third, OpenAI model support for vendor-neutral routing across GPT-4o-mini, GPT-4o, and o1. Fourth, online learning from routing outcomes using downstream quality signals. Fifth, per-domain threshold calibration — separate thresholds for customer support, code generation, and analytical reasoning.

SECTION 9 — CONCLUSION

AXIOM demonstrates that cost-efficient LLM routing is achievable without preference data, ML frameworks, or GPU infrastructure. The primary contribution is architectural: a sparse computation graph with four distinct traversal directions — forward, lateral, feedback, and temporal — enabling non-local inter-node communication that has no equivalent in the LLM routing literature. A systematic survey of 75+ existing routing and cascading systems confirms this topology is novel. The structural encoder ceiling is identified and characterised as a fundamental limitation requiring semantic extension. The system achieves 56.6% cost reduction on realistic enterprise workloads, 100% simple routing accuracy, and 1,311 microsecond routing latency with 159 passing tests, 1.2M parameters, and 3.4-minute training time on commodity hardware.

REFERENCES

Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176.

Ding, B., et al. (2024). Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. arXiv:2404.14618.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR 23(120).

Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. Cognition 68(1), 1–76.

Hu, S., et al. (2023). Inference Routing for Efficient LLM Serving. Tryage system.

Kincaid, J.P., et al. (1975). Derivation of new readability formulas for Navy enlisted personnel. Research Branch Report 8-75, Naval Technical Training Command.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. ICML 2023.

Madaan, D., et al. (2023). AutoMix: Automatically Mixing Language Models. arXiv:2310.12963.

Oja, E. (1982). A simplified neuron model as a principal component analyser. Journal of Mathematical Biology 15(3), 267–273.

Ong, I., et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665.

Salvatori, T., et al. (2022). Predictive Coding beyond Gaussian Distributions. NeurIPS 2022.

Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. arXiv:1701.06538.

Survey arXiv:2603.04445 (2026). Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey.

Yngve, V.H. (1960). A model and an hypothesis for language structure. Proceedings of the American Philosophical Society 104(5), 444–466.

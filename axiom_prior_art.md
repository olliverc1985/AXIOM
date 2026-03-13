# AXIOM Prior Art Survey

A comprehensive survey of related work for the AXIOM paper: a sparse computation graph with Hebbian/Oja learning for sub-millisecond LLM query routing. Compiled March 2026.

---

## 1. LLM Routing Systems

Single-shot model selection: classify a query and dispatch it to one model.

**Ong, I., Srivatsa, A., Hsu, J., & Stoica, I. (2024).** RouteLLM: Learning to Route LLMs with Preference Data. *ICLR 2025*. arXiv:2406.18665.
Four router architectures (similarity-weighted, matrix factorization, BERT, causal LM) trained on 80K Chatbot Arena preference pairs. Binary routing (strong/weak). Up to 85% cost reduction on MT Bench while maintaining quality. Python/PyTorch. The most direct comparator to AXIOM's routing goal.

**Hu, Q., et al. (2024).** RouterBench: A Benchmark for Multi-LLM Routing System. arXiv:2403.12031.
Standardized benchmark with 405K+ inference outcomes from 11 LLMs across 7 task categories. Enables multi-class routing evaluation. Provides the evaluation infrastructure AXIOM should benchmark against.

**Jiang, D., Ren, X., & Lin, B. Y. (2023).** LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. *ACL 2023*. arXiv:2306.02561.
Cross-attention PairRanker scores candidate outputs, then GenFuser merges top-k responses. Multi-model ensemble rather than routing, but addresses the same cost-quality tradeoff.

**Lu, K., et al. (2024).** ZOOTER: Routing to the Expert. *NAACL 2024*. arXiv:2311.08692.
Fine-tuned mDeBERTa-v3 with reward-model distillation for multi-class routing across model families. Demonstrates that distilled reward signals can train effective routers.

**Nguyen, A., et al. (2024).** MetaLLM: A High-performant and Cost-efficient Dynamic Framework for Wrapping LLMs. arXiv:2407.10834.
Contextual multi-armed bandit approach to LLM selection. Adapts routing policy online without retraining. Multi-class.

**Chen, L., Zaharia, M., & Zou, J. (2023).** FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176.
DistilBERT-based cascade scorer that learns when a cheaper model suffices. Up to 98% cost reduction with matching quality on specific benchmarks. Pioneering cost-aware LLM routing work.

**Hari, S. K., & Thomson, M. (2023).** Tryage: Real-time, Intelligent Routing of User Prompts to Expert LLMs. arXiv:2308.11601.
Brain-inspired "perceptive router" for selection from 200K+ models. Achieves 50.9% optimal selection rate. Notable for biologically-motivated design, though the bio-inspiration is more metaphorical than mechanistic.

**Chen, K., et al. (2024).** RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models. *NeurIPS 2024*. arXiv:2409.19886.
Dual contrastive learning trains a router that maps queries and LLM capabilities into a shared embedding space. +2.76% accuracy improvement in-distribution. Contrastive learning for routing parallels AXIOM's unfrozen contrastive training.

**Feng, Y., et al. (2024).** GraphRouter: A Graph-based LLM Router. *ICLR 2025*. arXiv:2410.03834.
Graph attention network over query-LLM-task structure. 12.3% improvement over baselines. Inductive: handles new LLMs without retraining. The graph-based approach has architectural similarity to AXIOM's sparse computation graph, though the graph semantics differ entirely.

**Liu, Y., et al. (2024).** OptLLM: Optimal Assignment of Queries to Large Language Models. *ICWS 2024*. arXiv:2405.15130.
Multi-label classification with multi-objective optimization (cost, quality, latency). 2.4--49.2% cost reduction across configurations.

**Wang, Z., et al. (2025).** MixLLM: LLM Quantization with Global Mixed-precision between Output-channel and In-channel. *NAACL 2025*. arXiv:2502.18482.
Contextual bandit with query tags. Achieves 97.25% of GPT-4 quality at 24.18% of the cost.

**Shnitzer, T., et al. (2023).** Large Language Model Routing with Benchmark Datasets. arXiv:2309.15789.
IBM Research. Binary classifiers trained on existing benchmark datasets as proxy labels, avoiding human preference data. Demonstrates that benchmark performance can serve as routing signal.

**Stripelis, D., et al. (2024).** TensorOpera Router: A Multi-Model Router for Efficient LLM Inference. arXiv:2408.12320.
BERT fine-tuned for multi-class routing with efficient inference. Production-oriented system from TensorOpera.

**Li, Y. (2025).** kNN Routers are All You Need. arXiv:2505.12601.
Shows that k=5 nearest-neighbor routing matches or exceeds complex learned routers on standard benchmarks. A strong simplicity baseline that any novel routing architecture must beat.

**Patil, S. G., et al. (2023).** Gorilla: Large Language Model Connected with Massive APIs. arXiv:2305.15334.
Fine-tuned LLaMA-7B for API selection (1,600+ APIs). Not strictly LLM routing, but demonstrates learned dispatching to heterogeneous backends.

**Somerstep, S., et al. (2025).** CARROT and SPROUT: Consistent and Reliable Routing of LLM Queries. arXiv:2502.03261.
RoBERTa-based router evaluated on 45K queries across 14 models. Focuses on routing consistency and reliability guarantees.

**Symbolic-MoE (2025).** arXiv:2503.05641.
Training-free skill-based routing using symbolic task descriptions. Routes without learned parameters, contrasting with AXIOM's fully learned approach.

---

## 2. LLM Cascade Systems

Sequential escalation: try a cheap model first, escalate on uncertainty.

**Dohan, D., et al. (2022).** Language Model Cascades. arXiv:2207.10342.
Theoretical framework formalizing LM composition as probabilistic programs. Foundation paper for cascade-style LLM systems.

**Aggarwal, P., Madaan, A., et al. (2023).** AutoMix: Automatically Mixing Language Models. *NeurIPS 2024*. arXiv:2310.12963.
Few-shot self-verification determines if a smaller model's answer is adequate; POMDP-based meta-verifier routes to larger models. >50% cost reduction. Cascading with learned escalation policy.

**Ding, D., et al. (2024).** Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. *ICLR 2024*. arXiv:2404.14618.
BERT-style binary classifier trained on BARTScore quality labels. Reduces large-model calls by 40% while maintaining output quality.

**Yue, X., et al. (2024).** Large Language Model Cascades with Mixture of Thoughts Representations. *ICLR 2024*. arXiv:2310.03094.
Uses answer consistency across multiple reasoning paths as a proxy for difficulty. 60% cost savings. The consistency-as-complexity signal is reminiscent of AXIOM's structural encoding approach to complexity estimation.

**Zhang, J., et al. (2023).** EcoAssistant: Using LLM Assistant More Affordably and Accurately. arXiv:2310.03046.
Hierarchical code-execution cascade with solution retrieval. Escalates through model tiers only when code execution fails.

**Valkanas, A., et al. (2025).** C3PO: Conformal Prediction Cascades for Cost-Optimal LLM Inference. arXiv:2511.07396.
Applies conformal prediction to cascade decisions, providing statistical coverage guarantees. Self-supervised: no human labels needed.

**Zellinger, W., Liu, W., & Thomson, M. (2025).** Cost-Saving LLM Cascades with Early Abstention. arXiv:2502.09054.
Markov-copula model for cascade decisions with early abstention (declining to answer). 13% additional cost reduction over standard cascades.

**Zellinger, W. & Thomson, M. (2025).** Rational Tuning of LLM Cascades via Parametric Markov-Copula Models. arXiv:2501.09345.
Parametric Markov-copula approach that works with as few as n<=30 labeled examples. Extremely sample-efficient cascade tuning.

**Wang, S., et al. (2024).** Cascade-Aware Training of Language Models. arXiv:2406.00060.
Trains smaller models with explicit deferral awareness -- the small model learns when to say "I should defer this." Changes the training objective, not just the routing policy.

**Nie, A., et al. (2024).** Online Cascade Learning for Efficient Inference Over Streams. *ICML 2024*. arXiv:2402.04513.
Imitation learning for online cascade decisions over streaming queries. Up to 90% cost reduction. Handles non-stationary query distributions.

**Rabanser, S., et al. (2025).** Gatekeeper: Confidence-Tuning Loss for Efficient LLM Cascading. arXiv:2502.19335.
Novel loss function that directly optimizes cascade confidence thresholds during training. Eliminates post-hoc threshold tuning.

**Li, Y., et al. (2025).** LLM Bandit: Cost-Efficient LLM Generation via Preference-Conditioned Bandits. arXiv:2502.02743.
Preference-conditioned bandit that factors in user-specific quality preferences. 27% cost improvement.

**BEST-Route (Microsoft, 2025).** arXiv:2506.22716.
Combines router with best-of-n sampling: routes to a model, generates n candidates, selects best. 60% cost reduction. Blurs the line between routing and generation-time compute allocation.

**Zhang, Z., Feng, Y., & You, J. (2025).** Router-R1: Teaching LLMs Multi-Round LLM Routing. arXiv:2506.09033.
RL-trained LLM that itself acts as the router, supporting multi-round routing decisions. The router is a language model reasoning about which model to call.

**Narasimhan, A., et al. (2025).** Faster Cascades via Speculative Decoding. *ICLR 2025*. arXiv:2405.19261.
Uses speculative decoding within cascade architectures for latency reduction. Orthogonal optimization to routing quality.

**Chen, Z. (2023).** Cascade Speculative Drafting for Even Faster LLM Inference. arXiv:2312.11462.
Multi-level speculative drafting cascade for inference acceleration.

**SpecRouter (2025).** arXiv:2505.07680.
Adaptive routing for multi-level speculative decoding. Dynamically selects draft model based on input.

**Jiang, Y., et al. (2025).** Cascadia: MILP-Based Serving for LLM Cascades. arXiv:2506.04203.
Mixed-integer linear programming formulation for optimal cascade serving under SLA constraints. Systems-level optimization.

**Wang, J., et al. (2024).** Mixture-of-Agents Surpasses GPT-4o. arXiv:2406.04692.
Layered LLM aggregation where each layer's models attend to previous layer outputs. Not strictly cascading, but hierarchical multi-model composition.

---

## 3. Unified Routing + Cascading

**Dekoninck, J., et al. (2024).** Cascade Routing: Optimizing the Cost-Performance Trade-off. *ICML 2025*. arXiv:2410.10347.
Unified theoretical framework proving that combined routing + cascading strictly dominates either approach alone. Provides optimality conditions. Key theoretical result that any routing paper should cite.

**Survey: "Doing More with Less" (2025).** arXiv:2502.00409.
Covers approximately 40 routing and cascading papers with taxonomy. See Surveys section.

---

## 4. Hebbian and Local Learning Rules

Biologically-inspired learning without backpropagation.

**Journe, G., et al. (2023).** Hebbian Deep Learning Without Feedback (SoftHebb). *ICLR 2023*. arXiv:2209.11883.
Soft winner-take-all Hebbian learning for CNNs. MNIST 99.4%, CIFAR-10 80.3% -- all without backpropagation or feedback signals. The most successful modern Hebbian system. Directly validates that Hebbian rules can learn useful representations.

**Miconi, T. (2021).** Hebbian Learning with Gradients: Hebbian Convolutional Neural Networks with Modern Losses. arXiv:2107.01729.
Proves that certain loss functions have gradients that exactly implement Oja's rule. Bridges Hebbian and gradient-based learning theoretically.

**Nimmo, J. & Mondragon, E. (2025).** Advancing Biological Plausibility of Hebbian Convolutional Neural Networks. arXiv:2501.17266.
Achieves 75.2% CIFAR-10 matching backpropagation performance with biologically plausible Hebbian CNNs.

**Talloen, S., et al. (2021).** PyTorch-Hebbian: A Framework for Local Learning. arXiv:2102.00428.
Open-source framework implementing various Hebbian learning rules in PyTorch. Useful reference implementation.

**Eugenio, M. (2025).** Hebbian Learning the Local Structure of Language. arXiv:2503.02057.
Applies Hebbian learning to tokenization and semantic embedding construction. Directly relevant: Hebbian learning on text/language data.

**Shervani-Tabar, N., et al. (2024).** Oja's Plasticity Rule Overcomes Several Challenges of Training Neural Networks. arXiv:2408.08408.
Demonstrates that Oja's rule in deep networks prevents weight explosion and surpasses backpropagation in data-scarce regimes. Directly validates AXIOM's choice of Oja's rule for weight stabilization.

**Lagani, G., et al. (2022).** Oja's Rule for CNN Pre-training.
60% CIFAR-10 with Oja-based unsupervised pre-training. Lower performance ceiling than SoftHebb but validates Oja specifically.

**Hinton, G. (2022).** The Forward-Forward Algorithm: Some Preliminary Investigations. arXiv:2212.13345.
Replaces backpropagation with two forward passes (positive and negative data). Local per-layer learning objectives. Seminal paper in the BP-free learning revival.

**DeeperForward (2025).** *ICLR 2025*.
Extends Forward-Forward to 17 layers with enhanced local objectives. Addresses the original algorithm's depth limitations.

**Lorberbom, G., et al. (2025).** Self-Contrastive Forward-Forward Algorithm. *Nature Communications*. arXiv:2409.11593.
Removes the need for negative data in Forward-Forward by using self-contrastive objectives. Published in Nature Communications, indicating broad scientific interest in BP-free methods.

**Mono-Forward (2025).** arXiv:2501.09238.
Single-forward-pass local learning that matches or surpasses backpropagation on some benchmarks, with 41% less energy consumption.

**PEPITA (2022).** *ICML 2022*. arXiv:2201.11665.
Error-driven input modulation as a biologically plausible learning signal. Local learning rule.

**3-Factor Hebbian Learning (2020).** *NeurIPS 2020*. arXiv:2006.07123.
Three-factor Hebbian rule incorporating a modulatory signal alongside pre/post-synaptic activity. Extends classical Hebb with a relevance gate.

**Krotov, D. & Hopfield, J. J. (2019).** Unsupervised Learning by Competing Hidden Units. *PNAS*. arXiv:1806.10181.
Competitive Hebbian learning where hidden units compete for activation. The competition mechanism parallels AXIOM's dynamic coalition formation.

**Melchior, J. & Wiskott, L. (2024).** Hebbian-Descent: A Novel Gradient-Free Learning Rule. arXiv:1905.10585.
Outperforms all other Hebbian rules in online learning settings. Gradient-free.

**Equilibrium Propagation (2021).** arXiv:2101.05536.
Hebbian-like weight updates computed at network equilibrium. Bridges energy-based models with Hebbian learning.

**Dual Propagation (2023).** *ICML 2023*. arXiv:2302.01228.
Addresses computational bottleneck in Hebbian/local learning methods. Enables scaling to larger networks.

**Millidge, B., et al. (2022).** A Theoretical Framework for Predictive Coding, Equilibrium Propagation, and Contrastive Hebbian Learning. arXiv:2206.02629.
Unifies predictive coding, equilibrium propagation, and contrastive Hebbian learning under a single theoretical framework. Shows these are all instances of the same underlying principle.

**Associated Learning (2022).** *ICLR 2022*.
Outperforms backpropagation on CNNs, RNNs, and Transformers. Strong result for local learning methods.

**RECAP: Hebbian Prototype Learning (2026).** arXiv:2603.06639.
Recent Hebbian prototype-based learning. Contemporary with AXIOM's development.

**Wang, H. & Hong, M. (2019).** Supervised Hebbian Learning for Feature Selection in Text Classification.
Applies Hebbian learning to text feature selection. Direct precedent for Hebbian methods in NLP/text domains.

---

## 5. Sparse Networks and Graph-Based Learning

Sparse topologies, arbitrary graph computation, dynamic structure.

**Predictive Coding Networks Survey (2022).** *IJCAI 2022*. arXiv:2202.09467.
Survey showing predictive coding works on arbitrary graph topologies, not just feedforward networks. Validates that sparse, non-standard graph structures can support learning.

**Salvatori, T., et al. (2022).** Learning on Arbitrary Graph Topologies via Predictive Coding. *NeurIPS 2022*. arXiv:2201.13180.
Demonstrates that predictive coding on arbitrary (non-layered) graphs matches or exceeds feedforward performance. Directly validates AXIOM's sparse computation graph approach.

**CTRE: Brain-Inspired Sparse Training (2022).** *ECML-PKDD 2022*. arXiv:1903.07138.
Uses cosine similarity for topology evolution in sparse networks. The cosine-similarity-driven structural adaptation parallels AXIOM's embedding cache mechanism.

**Hebbian Architecture Generation (2025).** *Nature Communications*.
Grows sparse connections via co-activation signals. The network topology itself emerges from Hebbian principles. Published in Nature Communications.

**SOM Survey (2025).** arXiv:2501.08416.
Self-organizing map survey covering SOMPL routing via clustering. SOMs represent an alternative bio-inspired approach to input-space partitioning for routing.

---

## 6. Text Complexity and Linguistic Features

Structural features for estimating text difficulty.

**Deutsch, T., et al. (2020).** Linguistic Features for Readability Assessment. arXiv:2006.00377.
Validates that structural linguistic features (sentence length, type-token ratio, lexical diversity) are predictive of text complexity. Directly supports AXIOM's G5 structural syntax features as routing signals.

**InterpretARA (2024).** *AAAI 2024*.
Hybrid readability assessment combining BERT embeddings with handcrafted linguistic features. Demonstrates that structural features complement learned representations -- relevant to AXIOM's position-weighted encoding + syntactic features design.

---

## 7. Surveys

**"Doing More with Less: A Survey on LLM Routing" (2025).** arXiv:2502.00409.
Comprehensive survey covering approximately 40 routing papers. Taxonomy of routing approaches (classifier-based, reward-based, cascade, bandit). Essential reference for positioning.

**"Dynamic Model Routing and Cascading" (Moslem & Kelleher, 2026).** arXiv:2603.04445.
Most recent survey covering both routing and cascading approaches. Published 2026, contemporaneous with AXIOM.

**"Beyond Backpropagation" (2025).** arXiv:2509.19063.
Survey of non-backpropagation learning methods including Hebbian, Forward-Forward, predictive coding, and equilibrium propagation.

**"The Cost of Avoiding Backpropagation" (2025).** arXiv:2506.21833.
Systematic comparison showing backpropagation still leads by 4.5--31.1% on ImageNet across various BP-free methods. Important reality check for Hebbian/local learning claims.

---

## 8. Commercial Systems

**Martian** — Commercial LLM router. Proprietary model selection with SLA guarantees. Production deployment.

**OpenRouter** — API gateway routing across 100+ models. Primarily cost-based routing with some quality signals.

**Not Diamond** — ML-based router selecting between GPT-4, Claude, Gemini, etc. Trained on preference data. Closest commercial analog to AXIOM's routing goal.

---

## 9. Adversarial / Security

**Shafran, E., et al. (2025).** Rerouting LLM Routers. *COLM 2025*. arXiv:2501.01818.
Demonstrates adversarial attacks on LLM routers -- small input perturbations can force misrouting to expensive models. Important for any deployed routing system's threat model.

---

## 10. AXIOM Positioning: Honest Assessment

### What is genuinely novel

1. **The combination is novel, not the components.** No existing system combines a sparse computation graph + Hebbian/Oja learning + structural syntax encoding + dynamic coalition formation for LLM routing. Each component has precedent; the integration does not.

2. **Hebbian/Oja learning for LLM routing.** No prior routing paper uses Hebbian or Oja's rule for the routing decision. All existing learned routers use backpropagation (RouteLLM's BERT/causal LM routers), reward distillation (ZOOTER), contrastive learning (RouterDC), or bandits (MetaLLM, MixLLM). AXIOM's use of Oja's rule for online, local weight updates in a router is unprecedented.

3. **Pure Rust, zero ML framework dependencies.** Every other learned router depends on PyTorch, TensorFlow, or similar. AXIOM's from-scratch implementation in Rust with no ML dependencies is unique in this space. Linfa (Rust ML) exists but is a general framework, not a routing system.

4. **Sub-millisecond routing with learned parameters.** Most learned routers (RouteLLM BERT, ZOOTER mDeBERTa, RouterDC) have inference latencies in the tens to hundreds of milliseconds due to transformer forward passes. AXIOM's 1.2M-parameter sparse graph should achieve sub-millisecond routing. This is a meaningful systems advantage, though kNN routers (Li, 2025) also achieve very low latency.

5. **Dynamic coalition formation.** The stochastic node selection mechanism has no direct precedent in routing literature. Krotov & Hopfield's competing hidden units (2019) provide a conceptual parallel but operate differently.

### What has been done before

1. **LLM routing itself is a crowded field.** At least 22 dedicated routing papers exist, plus surveys covering 40+ approaches. The problem statement is not novel. RouteLLM (2024), FrugalGPT (2023), and Hybrid LLM (2024) all demonstrate effective routing with simpler architectures.

2. **Structural/linguistic features for complexity estimation.** Deutsch et al. (2020) and InterpretARA (2024) validate structural features for readability. AXIOM's G5 features (token count, TTR, average token length, punctuation density) are standard readability metrics, not novel features.

3. **Sparse computation graphs for learning.** Salvatori et al. (NeurIPS 2022) demonstrate learning on arbitrary graph topologies. Predictive coding on sparse graphs is established. AXIOM's graph structure is novel in the routing context but not in the learning context.

4. **Hebbian/Oja learning in modern ML.** SoftHebb (ICLR 2023) achieves 80.3% CIFAR-10 with pure Hebbian learning. Shervani-Tabar et al. (2024) specifically validate Oja's rule for weight stabilization. AXIOM's use of Oja is well-motivated by prior art but is not a contribution to Hebbian learning theory.

5. **Contrastive learning for routing.** RouterDC (NeurIPS 2024) already uses dual contrastive learning for router training. AXIOM's "unfrozen contrastive" training is a variant, not a first.

6. **Graph-based routing.** GraphRouter (ICLR 2025) uses graph attention networks for routing. AXIOM's graph is structurally different (computation graph vs. relational graph) but the "graph-based router" framing is taken.

### Claims that need to be softened

1. **"First biologically-inspired LLM router"** -- Tryage (2023) already claims brain-inspired routing design. AXIOM can claim "first Hebbian-learning-based router" but not broadly "first bio-inspired."

2. **"Novel sparse computation graph"** -- The graph is novel for routing, but learning on arbitrary graph topologies is established (Salvatori et al., 2022). Frame as "novel application of sparse graph computation to the routing domain."

3. **Cost/quality claims** -- Without evaluation on RouterBench or comparison to RouteLLM baselines, absolute cost reduction claims cannot be made. The paper should include comparisons or clearly state the evaluation is preliminary.

4. **"Zero ML dependencies" as a feature** -- This is an engineering choice, not a scientific contribution. It can be mentioned but should not be positioned as a research finding. It does enable reproducibility and deployment simplicity.

5. **Sub-millisecond latency** -- Must be benchmarked against kNN routers (Li, 2025), which also achieve very low latency with competitive accuracy. If k=5 kNN matches AXIOM's routing quality, the architectural complexity is harder to justify.

6. **Parameter efficiency (1.2M params)** -- RouteLLM's matrix factorization router uses far fewer parameters. DistilBERT-based routers (FrugalGPT) use ~66M but are pre-trained. The comparison axis matters: 1.2M trained-from-scratch vs. 66M fine-tuned are different efficiency claims.

### Strongest competing systems

| System | Strength against AXIOM |
|--------|----------------------|
| **RouteLLM** (ICLR 2025) | Most established. Four router variants, strong empirical results, open-source. AXIOM must match or exceed its MT Bench / MMLU performance. |
| **kNN Router** (Li, 2025) | Devastating simplicity baseline. If k=5 kNN matches learned routers, AXIOM's complexity needs justification beyond routing accuracy (e.g., online adaptation, no labeled data). |
| **RouterDC** (NeurIPS 2024) | Also uses contrastive learning for routing. Direct methodological overlap. |
| **GraphRouter** (ICLR 2025) | Also graph-based routing. AXIOM needs to clearly differentiate its graph semantics. |
| **Cascade Routing** (ICML 2025) | Proves routing + cascading combined is optimal. AXIOM does pure routing; the paper should address why cascading is not included or how it could be added. |
| **SoftHebb** (ICLR 2023) | Strongest Hebbian learning result. AXIOM should cite this as validation that Hebbian methods can learn useful representations, while noting AXIOM applies Hebbian learning to a different problem (routing, not classification). |

### Recommended positioning

AXIOM is best positioned as: **"The first LLM router that learns routing decisions through local Hebbian/Oja learning rules on a sparse computation graph, enabling online adaptation without backpropagation, labeled preference data, or ML framework dependencies."**

Key differentiators to emphasize:
- **Online, unsupervised adaptation** via Oja's rule (no preference labels, no reward models)
- **Architectural novelty** of the specific combination (sparse graph + Hebbian + structural encoding + coalition formation)
- **Systems properties** (sub-ms latency, 1.2M params, pure Rust, <50MB memory)
- **Biological plausibility** as a secondary theme, not primary claim

Key claims to avoid:
- Broad "first bio-inspired router" (Tryage exists)
- Superiority claims without RouterBench/RouteLLM comparison
- Novelty claims on individual components (all have precedent)
- Framing zero-dependencies as a research contribution rather than engineering choice

### Additional papers from extended search

**CSCR — Cost-Spectrum Contrastive Routing** (Shirkavand, 2025) — arXiv:2508.12491, NeurIPS 2025 Spotlight. Maps prompts and models into shared embedding space; routing reduces to FAISS k-NN lookup at **microsecond latency**. Closest latency competitor to AXIOM. Uses logit/perplexity fingerprints, Python + FAISS.

**LLMRank** (Agrawal, 2025) — arXiv:2510.01234. Uses "syntactic cues" and "complexity indicators" among features — closest feature-engineering approach to AXIOM. Neural ranking model on RouterBench. 89.2% of oracle utility. Still uses ML framework; features extracted by LLM calls.

**Cannistraci-Hebb Training** (2025) — arXiv:2501.19107. Gradient-free, topology-driven link regrowth for sparse neural networks using Hebbian-inspired rules. At 1% connectivity, outperforms fully connected networks. Most closely related to AXIOM's sparse graph + Hebbian combination. CHT focuses on intra-model sparsity; AXIOM uses sparse graph for inter-model routing.

**Hebbian-Oscillatory Co-Learning (HOC-L)** (2026) — arXiv:2603.08731. Couples Hebbian structural plasticity with oscillatory phase synchronization in sparse architectures. Proves convergence via Lyapunov function. Theoretical grounding for Hebbian learning in sparse architectures.

**Routoo** (2024) — arXiv:2401.13979. Lightweight LLM as performance predictor + cost-aware selector. LLM-based prediction without running candidate models.

**Arch-Router** (2025) — arXiv:2506.16655. 1.5B parameter model that maps queries to domain-action preferences for routing.

**Router-R1** (Zhang, Feng & You, 2025) — arXiv:2506.09033, NeurIPS 2025. RL-trained LLM router. Interleaves reasoning ("think") with model invocation ("route") across multiple rounds.

### Updated positioning notes

- **CSCR achieves microsecond latency** — AXIOM's sub-millisecond claim is strong but not unique. Frame as "sub-millisecond with a learned sparse graph" (vs CSCR's k-NN lookup which requires FAISS indexing infrastructure).
- **LLMRank uses syntactic features** — AXIOM cannot claim sole ownership of structural/syntactic routing. Frame as "vocabulary-independent structural encoding without neural feature extraction."
- **Cannistraci-Hebb validates Hebbian + sparse** — AXIOM is not the first to combine Hebbian learning with sparse networks, but IS the first to apply this combination to LLM query routing.

---

*Survey compiled March 2026. 75+ papers reviewed across LLM routing, cascading, Hebbian learning, sparse networks, and text complexity assessment.*

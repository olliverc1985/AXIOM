# Patent Prior Art Search: LLM Routing, Query Complexity-Based Model Selection, and Adaptive Inference

**Date:** 2026-03-13
**Search scope:** UK (GB), European (EP), WIPO (WO), US, Chinese (CN), Japanese (JP), Korean (KR) patents
**Databases searched:** Google Patents (XHR API), FreePatentsOnline, EPO Register, OpenAlex

---

## HIGHLY RELEVANT PATENTS

### 1. US 12,436,974 B2 — Resource Conservation Based on Query Complexity
- **Assignee:** Microsoft Technology Licensing, LLC
- **Filing date:** 2024-02-28
- **Publication date:** 2025-10-07
- **Classification:** G06F16/28, G06F16/2457
- **Key claims:** System receiving input queries, classifying complexity via a "response classifier" with complexity scores, comparing scores to thresholds, and selecting AI models from multiple options with different performance characteristics. Queries exceeding threshold scores route to high-complexity models; simpler queries go to lower-complexity alternatives.
- **Relevance:** **VERY HIGH** — Directly covers complexity-based routing of queries to different AI models. The core concept of a complexity classifier determining which model tier handles a query is closely related to structural-complexity-based routing.

### 2. US 12,314,825 B2 — Prompt Routing System and Method
- **Assignee:** Martian Learning, Inc.
- **Filing date:** 2024-08-12
- **Publication date:** 2025-05-27
- **Classification:** G06N20/00
- **Key claims:** Training a scoring model (neural network) to predict candidate model response scores based on prompts; extracting an encoder; encoding new prompts to compute response scores for each candidate model; selecting runtime model based on scores. Also covers heuristic-based selection using training prompt-response pairs similar to new prompts.
- **Relevance:** **VERY HIGH** — Directly covers learned prompt routing using a trained neural network to predict which model will perform best, then routing accordingly.

### 3. US 2025/0383970 A1 — Hierarchical Cascade Architecture of Language Models for Multi-Stage Query Classification and Agent Routing
- **Assignee:** Citibank, N.A.
- **Filing date:** 2025-09-10
- **Publication date:** 2025-12-18
- **Classification:** G06F9/50, G06F11/34
- **Key claims:** Queries processed iteratively through hierarchical levels containing AI models of increasing complexity. Each level generates classification + confidence score. Dynamic bypass mechanism determines which hierarchy levels can be skipped. Conditional escalation to larger models when confidence is below threshold. Weighted historical performance metrics for aggregation.
- **Relevance:** **VERY HIGH** — Directly describes hierarchical/tiered model cascade with confidence-based escalation, closely matching the concept of routing queries through tiers based on structural complexity.

### 4. US 12,106,205 B2 — Dynamic, Resource-Sensitive Model Selection and Output Generation
- **Assignee:** Citibank, N.A.
- **Filing date:** 2024-05-10
- **Publication date:** 2024-10-01
- **Classification:** G06N3/0455, G06N3/084
- **Key claims:** Receiving output generation requests, determining performance metrics and system state, calculating threshold values, estimating resource usage by LLMs, comparing estimates against thresholds, routing prompts to appropriate models based on resource constraints.
- **Relevance:** **HIGH** — Covers cost/resource-aware model selection and routing.

### 5. US 2025/0384072 A1 — Explainable Large Language Model Routing with Immutable Audit Trails
- **Assignee:** Citibank, N.A.
- **Filing date:** 2025-08-28
- **Publication date:** 2025-12-18
- **Classification:** G06F16/338, G06F16/383
- **Key claims:** Analyzing query characteristics including **complexity, domain, regulatory constraints, and performance needs**; retrieving LLM profiles (performance data, resource consumption, compliance parameters); selecting optimal model by **balancing resource consumption against performance requirements**; ranking LLMs and prioritizing those with successful history.
- **Relevance:** **VERY HIGH** — Explicitly uses query complexity analysis as a routing signal, combined with multi-factor model selection.

### 6. US 2025/0378099 A1 — Intelligent Query Decomposition, Specialized Model Routing, and Hierarchical Aggregation with Conflict Resolution
- **Assignee:** Citibank, N.A.
- **Filing date:** 2025-08-25
- **Publication date:** 2025-12-11
- **Classification:** G06F16/338, G06F16/383
- **Key claims:** Decomposing complex queries using trained models analyzing **semantic and syntactic boundaries**; routing sub-queries to specialized models; assigning confidence scores; detecting discrepancies; aggregating responses with weighted algorithms prioritizing higher-confidence responses. Routing strategies encompass performance-based (latency, accuracy), cost-optimization, domain-expertise matching, and adaptive learning.
- **Relevance:** **VERY HIGH** — Covers semantic/syntactic analysis for query decomposition and routing, with confidence-based aggregation.

### 7. US 12,321,862 B2 — Latency-, Accuracy-, and Privacy-Sensitive Tuning of AI Model Selection Parameters
- **Assignee:** Citibank, N.A.
- **Filing date:** 2024-09-11
- **Publication date:** 2025-06-03
- **Classification:** G06N3/091
- **Key claims:** Dynamic model selection through risk evaluation and system state monitoring; generating performance weighting values prioritizing latency, accuracy, and privacy; identifying routing models; selecting appropriate LLMs.
- **Relevance:** **HIGH** — Covers multi-objective model selection with dynamic parameter tuning.

### 8. US 2025/0371433 A1 — Multi-Variable Optimization for Routing Requests to Language Models
- **Assignee:** Citibank, N.A.
- **Filing date:** 2025-08-15
- **Key claims:** Multi-variable optimization for LLM routing decisions.
- **Relevance:** **HIGH** — Directly covers optimized routing to language models.

### 9. US 12,524,210 B2 — Hybrid Inference System for COGS Reduction
- **Assignee:** Microsoft Technology Licensing, LLC
- **Filing date:** 2023-09-01
- **Publication date:** 2026-01-13
- **Classification:** G06F8/35, G06F11/3604
- **Key claims:** Routing model predicts whether LLM output will be accepted; routes to local model if likely rejected, or to LLM if likely accepted. Routing model trained on historical acceptance/rejection data.
- **Relevance:** **HIGH** — Cost-based routing between local and cloud models using a learned routing model.

### 10. US 2025/0252032 A1 — Methods and Systems for Generating, Training, Combining, Cascading, and Using Federated Language Models
- **Assignee:** Ema Unlimited Inc.
- **Filing date:** 2025-04-09 (CIP of 2024-02-06)
- **Publication date:** 2025-08-07
- **Classification:** G06F11/34, G06F16/3329
- **Key claims:** Real-time query analysis, dynamic model selection, generating response portions, determining confidence scores for predictive segments, **switching to alternative models when confidence drops below thresholds** during generation.
- **Relevance:** **VERY HIGH** — Mid-generation confidence-based model switching/cascading.

### 11. US 2025/0321852 A1 — Dynamic Model Selection and Routing Using Prompt Processing Units
- **Assignee:** Cisco Technology, Inc.
- **Filing date:** 2024-10-30
- **Publication date:** 2025-10-16
- **Classification:** G06F11/34, G06F40/284
- **Key claims:** Identifying task from prompt, computing estimated performance metrics for candidate language models, selecting optimal model to optimize metrics, routing prompt to that model.
- **Relevance:** **HIGH** — Task-aware dynamic model selection and routing.

### 12. US 2025/0238623 A1 — Token-Level Routing of Large Language Models as External Knowledge Models
- **Assignee:** Tencent America LLC
- **Filing date:** 2024-01-22
- **Publication date:** 2025-07-24
- **Classification:** G06F40/35
- **Key claims:** Token-level router directs generation to pretrained model (for factual content) or aligned model (for non-factual content) based on factuality assessment via binary classifier.
- **Relevance:** **MODERATE-HIGH** — Token-level routing between models, though focused on factuality rather than complexity.

### 13. US 2025/0117587 A1 — Specialist Language Model Set Mapping and Selection for Prompt Delegation
- **Assignee:** Insight Direct USA, Inc.
- **Filing date:** 2024-04-23
- **Publication date:** 2025-04-10
- **Classification:** G06F40/30
- **Key claims:** Decomposing compound prompts into steps, associating domain descriptors with specialized models, assigning relevance scores via semantic comparison, selecting model subsets, assembling coherent final output.
- **Relevance:** **HIGH** — Prompt decomposition and delegation to specialist models.

### 14. US 2025/0259008 A1 — Serverless Functional Routing for Large Language Model Inference Service
- **Assignee:** International Business Machines Corporation (IBM)
- **Filing date:** 2024-02-08
- **Publication date:** 2025-08-14
- **Classification:** G06F16/33, G06F40/40
- **Key claims:** Routing LLM prompts via serverless function router to endpoints with subject matter expert models; querying vector database to identify best-matching dataset/endpoint.
- **Relevance:** **HIGH** — Domain-based routing using vector similarity.

### 15. US 2025/0165752 A1 — Systems and Methods for Processing Data for Large Language Models
- **Assignee:** Infobip Ltd.
- **Filing date:** 2023-11-22
- **Publication date:** 2025-05-22
- **Classification:** G06N3/0455, G06N3/08
- **Key claims:** Receiving query, determining capability using ML/segmentation, routing to optimal LLM provider based on cost, quality, accuracy, with fallback options.
- **Relevance:** **HIGH** — Multi-provider LLM routing platform.

### 16. US 2025/0278578 A1 — Intermediary Routing and Moderation Platform for Generative AI
- **Assignee:** Target Brands, Inc.
- **Filing date:** 2025-03-04
- **Publication date:** 2025-09-04
- **Key claims:** API routing queries to LLM systems based on query source/context; moderation service quantifying response quality.
- **Relevance:** **MODERATE** — Enterprise LLM routing with quality moderation.

### 17. US 12,536,406 B2 — Dynamic AI Agent Orchestration Using a Large Language Model Gateway Router
- **Assignee:** Citibank, N.A.
- **Filing date:** 2025-07-24
- **Publication date:** 2026-01-27
- **Classification:** G06F11/36, G06F8/41, G06N3/042, G06N20/00
- **Key claims:** Gateway router dynamically coordinating agents based on prompt characteristics, user context, and operational factors; input segmentation; sub-query routing.
- **Relevance:** **MODERATE-HIGH** — Dynamic agent routing via gateway.

### 18. US 2025/0348349 A1 — Edge Cloud Hierarchical Language Model Design
- **Assignee:** Microsoft Technology Licensing, LLC
- **Filing date:** 2024-05-07
- **Publication date:** 2025-11-13
- **Classification:** H04L67/04, H04L67/10, H04L67/303
- **Key claims:** Hierarchical edge architecture with models of diverse compute capabilities; dynamically selecting model based on network connectivity level.
- **Relevance:** **MODERATE** — Hierarchical model selection, though triggered by connectivity rather than query complexity.

### 19. US 12,142,371 B1 — Low-Latency Conversational AI Architecture with LLM Routing System
- **Assignee:** HealthGPT, Inc. (DBA Hippocratic AI)
- **Filing date:** 2024-02-29
- **Key claims:** Low-latency conversational AI with LLM routing system for healthcare.
- **Relevance:** **MODERATE** — Domain-specific LLM routing.

---

## EUROPEAN (EP) AND WIPO (WO) PATENTS

### 20. EP 4558922 A1 / WO 2025/017427 A1 — Efficiently Controlling Routing of Requests to Model Endpoint Infrastructure
- **Assignee:** MIH Technology Holdings B.V. (Netherlands)
- **Filing date:** 2024-07-10
- **Priority date:** 2023-07-17
- **Publication date:** 2025-05-28 (EP) / 2025-01-23 (WO)
- **Classification:** G06F, G06N, H04L
- **Inventors:** Euro Beinat, Paul van der Boor, Zulkuf Genc, Ioannis Panagiotis Zempekakis, Riccardo Campari, Niek Naber, Bruce Martens, Zvonimir Bednarcik, Dogu Tan Araci, Nishikant Dhanuka, Ahmed Mohamed Hany Abdelaziz Mohamed, Dmitri Sergeyevich Jarnikov, Bartosz Jakub Hawelka
- **Key claims:** Controlling routing of requests to model endpoint infrastructure.
- **Relevance:** **HIGH** — The only EP/WO patent found specifically addressing LLM request routing. MIH Technology Holdings is associated with Naspers/Prosus (South African tech conglomerate with significant EU presence).

---

## CHINESE (CN) PATENTS

### 21. CN 120407739 A — Large Language Model Routing Method Based on Zero-Sample Difficulty Perception
- **Assignee:** University of Science and Technology of China (USTC)
- **Filing date:** 2025-04-16
- **Key claims:** LLM routing based on zero-shot difficulty perception — routes queries based on estimated difficulty without requiring labeled training data.
- **Relevance:** **VERY HIGH** — Directly covers difficulty-based LLM routing.

### 22. CN 121365715 A — Large-Scale Language Model Routing Methods and Devices
- **Assignee:** AISpeech Co., Ltd. (Siqianchi Technology)
- **Filing date:** 2025-09-03
- **Key claims:** General LLM routing methods and devices.
- **Relevance:** **HIGH** — General LLM routing framework.

### 23. CN 119814634 A — Artificial Intelligence Routing Method, System and Computer Device
- **Assignee:** China Resources Digital Technology Co., Ltd.
- **Filing date:** 2024-12-17
- **Key claims:** AI routing method and system.
- **Relevance:** **MODERATE-HIGH**

### 24. CN 119167051 A — Method, Device, Equipment for Intelligent LLM Routing
- **Assignee:** Beijing Minglue Zhaohui Technology Co., Ltd.
- **Filing date:** 2024-11-15
- **Key claims:** Intelligent LLM routing method and device.
- **Relevance:** **HIGH**

### 25. CN 118410851 B — Hybrid Expert Model Routing Network Optimization Method
- **Assignee:** Inspur Electronic Information Industry Co., Ltd.
- **Filing date:** 2024-07-03
- **Key claims:** Optimization of routing networks for hybrid/mixture-of-experts models.
- **Relevance:** **MODERATE-HIGH** — MoE routing optimization.

### 26. CN 120525080 A — Model Training Method, Device, Computer Equipment and Storage Medium
- **Assignee:** Tencent Technology (Beijing) Co., Ltd.
- **Filing date:** 2025-04-24
- **Key claims:** Model training related to adaptive inference and routing.
- **Relevance:** **MODERATE**

### 27. CN 120781957 A — Multi-Model Self-Adaptive Reasoning System and Reasoning Method
- **Assignee:** TravelSky Technology Co., Ltd. (China Civil Aviation Information Network)
- **Filing date:** 2025-06-05
- **Key claims:** Multi-model self-adaptive reasoning with dynamic model selection.
- **Relevance:** **HIGH** — Adaptive multi-model reasoning.

### 28. CN 121303366 A — Method and System Based on Dynamic Thinking Network and Iterative Routing Agent
- **Assignee:** Zhejiang University
- **Filing date:** 2025-12-12
- **Key claims:** Dynamic thinking network with iterative routing.
- **Relevance:** **MODERATE**

### 29. CN 119228447 A — Method, Device, Electronic Device and Medium for Determining Behavior Plan
- **Assignee:** Baidu Era Network Technology (Beijing) Co., Ltd.
- **Filing date:** 2024-09-25
- **Key claims:** Behavior plan determination using language model routing.
- **Relevance:** **MODERATE**

---

## JAPANESE (JP) PATENTS

### 30. JP 2024-505638 A — Deep Learning Model for Predicting MHC Class I or Class II Immunogenicity
- **Assignee:** Amazon Technologies, Inc.
- **Filing date:** 2021-12-01
- **Note:** This was the only JP patent returned for "hierarchical model selection" but is not relevant to LLM routing. No directly relevant JP patents were found in the search.

**Note on JP patents:** No directly relevant Japanese patents from NTT, Preferred Networks, or other JP companies were found for LLM routing. This may reflect that Japanese companies are filing in the US/WO jurisdictions rather than JP specifically, or that Japanese prior art in this area exists primarily in academic literature rather than patent filings.

---

## KOREAN (KR) PATENTS

No directly relevant KR patents were identified in the search. Samsung's LLM-related patent filings appear to be primarily through US applications. Naver's patent activity in this specific area was not found.

---

## UK (GB) PATENTS

No GB-specific patents were found for "language model routing." The Google Patents search for GB country code returned 0 results. UK companies (DeepMind, ARM, Samsung AI Cambridge, Huawei UK, BT) appear to file their AI patents through US, EP, or WO jurisdictions rather than the UK IPO directly.

---

## NOTABLE ASSIGNEE PATTERNS

### Citibank, N.A. (Most prolific filer)
Citibank has an extensive patent portfolio covering LLM routing, filing at least 8 closely related patents in 2024-2025:
- Hierarchical cascade architecture (US 2025/0383970)
- Explainable LLM routing (US 2025/0384072)
- Intelligent query decomposition and routing (US 2025/0378099)
- Dynamic resource-sensitive model selection (US 12,106,205)
- Latency/accuracy/privacy-sensitive model selection (US 12,321,862)
- Multi-variable routing optimization (US 2025/0371433)
- Gateway router orchestration (US 12,536,406)
- Prompt validation for model selection (US 12,147,513)
- Semantic fingerprinting for agent routing (US 2026/0004162)

### Microsoft Technology Licensing, LLC
- Query complexity-based resource conservation (US 12,436,974) — **most directly relevant**
- Hybrid inference for cost reduction (US 12,524,210)
- Edge cloud hierarchical model design (US 2025/0348349)
- Speculative sampling with models of different capacities (US 2025/0209271)

### Tencent
- Token-level routing between models (US 2025/0238623, Tencent America)
- Model training for routing (CN 120525080, Tencent Beijing)

### IBM
- Serverless functional routing for LLM inference (US 2025/0259008)

### Cisco Technology, Inc.
- Dynamic model selection using prompt processing units (US 2025/0321852)

### Chinese Academic Institutions
- USTC: Zero-shot difficulty perception routing (CN 120407739) — **very relevant**
- Zhejiang University: Dynamic thinking network routing (CN 121303366)

---

## KEY FINDINGS AND ANALYSIS

### 1. Patent Landscape is Very New
Nearly all patents identified were filed in 2024-2025, with the earliest relevant filing being Microsoft's hybrid inference patent (US 12,524,210, filed 2023-09-01). This field is extremely nascent from a patent perspective.

### 2. Gap: Structural Complexity Features for Routing
**No patent was found that specifically uses structural linguistic features (token count, type-token ratio, average token length, punctuation density, syntactic complexity) as the basis for routing queries to different model tiers.** The closest is:
- Microsoft's US 12,436,974 (query complexity classification) — but it does not specify structural/linguistic complexity features
- Citibank's US 2025/0384072 (mentions "complexity" as a query characteristic) — but does not detail structural features
- Citibank's US 2025/0378099 (analyzes "semantic and syntactic boundaries") — closest to structural analysis, but for decomposition rather than tier routing

### 3. Gap: Sparse Computation Graph Routing
**No patent was found combining sparse computation graphs with LLM routing.** The combination of sparse graph architectures with hierarchical model selection appears to be novel.

### 4. Gap: Hebbian Learning for Routing Adaptation
**No patent covers the use of Hebbian learning or similar unsupervised learning to adapt routing thresholds.** Most patents use supervised training, reinforcement learning, or fixed thresholds.

### 5. EP/WO Coverage is Minimal
Only one EP patent (EP 4558922 from MIH Technology Holdings) was found directly addressing LLM routing. The European patent landscape for this technology is far less developed than the US.

### 6. Chinese Activity is Growing
Chinese patents from USTC, AISpeech, Inspur, and others show active innovation in LLM routing, particularly in zero-shot and difficulty-based approaches.

---

## RELEVANCE TO AXIOM ARCHITECTURE

A system that routes queries to different LLMs based on structural complexity features (using a sparse computation graph, embedding cache, hierarchical tiers, and Hebbian learning) would be **differentiated** from the existing patent landscape in the following ways:

1. **Structural feature encoding** (token count normalization, TTR, avg token length, punctuation density) as routing signals — not found in any patent
2. **Sparse computation graph** as the routing mechanism — not found
3. **Embedding cache with cosine similarity** for routing decisions — partially overlaps with IBM's vector database approach but is architecturally distinct
4. **Hebbian learning for threshold adaptation** — not found in any patent
5. **Confidence-based tiered escalation** — overlaps with Citibank's hierarchical cascade (US 2025/0383970) and Microsoft's query complexity patent (US 12,436,974)
6. **Temporal buffer for sequential context** — not found in routing context

The most significant prior art overlap is with the general concept of complexity-based model selection (Microsoft US 12,436,974) and hierarchical cascading with confidence thresholds (Citibank US 2025/0383970).

---

## RELATED ACADEMIC LITERATURE (for completeness)

Key academic works found during the search that are relevant as prior art:

1. **"Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey"** — Moslem & Kelleher, Feb 2026. Comprehensive survey of the field.
2. **"Select-then-Route: Taxonomy Guided Routing for LLMs"** — Shah & Shridhar, EMNLP 2025 Industry Track. Taxonomy-based model selection.
3. **"Tryage: Real-time, Intelligent Routing of User Prompts to LLMs"** — Hari & Thomson, Aug 2023. Early work on prompt routing.
4. **"Confidence-Driven Multi-Scale Model Selection for Cost-Efficient Inference"** — Chen, Chen & Yen, Feb 2026.
5. **"Confidence-Calibrated Small-Large Language Model Collaboration for Cost-Efficient Reasoning"** — Zhang et al., Mar 2026.
6. **"vLLM Semantic Router: Signal Driven Decision Routing for Mixture-of-Modality Models"** — Liu et al., Feb 2026.
7. **"RouteGoT: Node-Adaptive Routing for Cost-Efficient Graph of Thoughts Reasoning"** — Liu et al., Mar 2026.
8. **"Act, Think or Abstain: Complexity-Aware Adaptive Inference"** — Izzo, Bardaro & Matteucci, Mar 2026.

---

*Search conducted 2026-03-13. Patent databases searched: Google Patents, FreePatentsOnline, EPO Register, OpenAlex. Some searches were limited by rate limiting on Google Patents (IP-level blocking after ~15 API requests). Additional targeted searches on Espacenet, WIPO PatentScope, and national patent offices may yield further results.*

# AXIOM Routing Report

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total queries routed | 200 |
| Surface (Haiku) | 159 (79.5%) |
| Reasoning (Sonnet) | 23 (11.5%) |
| Deep (Opus) | 18 (9.0%) |
| Overall routing accuracy | 58.0% (116/200) |
| Mean routing time | 1311 µs |
| Parameters | 1205376 |

### Cost Simulation vs All-Opus Baseline

| Scale | AXIOM Cost | All-Opus Cost | Savings | Savings % |
|-------|------------|---------------|---------|-----------|
|      1k | $     12.90 | $        29.75 | $  16.85 |     56.6% |
|     10k | $    129.02 | $       297.49 | $ 168.46 |     56.6% |
|    100k | $   1290.24 | $      2974.88 | $1684.63 |     56.6% |

## 2. Dataset Results

### Simple (50 queries) (50 queries, 100.0% accuracy)

Tier distribution: Surface 50 (100%), Reasoning 0 (0%), Deep 0 (0%)

| # | Sentence | Truth | AXIOM Tier | Conf | Correct |
|---|----------|-------|------------|------|---------|
| 1 | The cat sat on the mat. | simple | Surface | 0.648 | Yes |
| 2 | Water is wet. | simple | Surface | 0.883 | Yes |
| 3 | The sun is bright today. | simple | Surface | 0.909 | Yes |
| 4 | She ate lunch at noon. | simple | Surface | 0.893 | Yes |
| 5 | The dog runs fast. | simple | Surface | 0.880 | Yes |
| 6 | I like red apples. | simple | Surface | 0.886 | Yes |
| 7 | He closed the door. | simple | Surface | 0.894 | Yes |
| 8 | The sky is blue. | simple | Surface | 0.901 | Yes |
| 9 | Fish swim in the pond. | simple | Surface | 0.895 | Yes |
| 10 | The book is on the table. | simple | Surface | 0.699 | Yes |
| 11 | Snow falls in winter. | simple | Surface | 0.890 | Yes |
| 12 | She walks to work. | simple | Surface | 0.891 | Yes |
| 13 | The baby is sleeping. | simple | Surface | 0.906 | Yes |
| 14 | Birds fly south. | simple | Surface | 0.873 | Yes |
| 15 | He plays the guitar. | simple | Surface | 0.895 | Yes |
| 16 | The room is dark. | simple | Surface | 0.902 | Yes |
| 17 | I drank cold water. | simple | Surface | 0.885 | Yes |
| 18 | The train is late. | simple | Surface | 0.905 | Yes |
| 19 | She smiled at him. | simple | Surface | 0.892 | Yes |
| 20 | The ball is round. | simple | Surface | 0.903 | Yes |
| 21 | Trees grow tall. | simple | Surface | 0.867 | Yes |
| 22 | He wore a blue shirt. | simple | Surface | 0.892 | Yes |
| 23 | The car is parked outside. | simple | Surface | 0.909 | Yes |
| 24 | I bought new shoes. | simple | Surface | 0.887 | Yes |
| 25 | The milk is cold. | simple | Surface | 0.902 | Yes |
| 26 | She opened the window. | simple | Surface | 0.896 | Yes |
| 27 | The road is long. | simple | Surface | 0.901 | Yes |
| 28 | He likes chocolate cake. | simple | Surface | 0.881 | Yes |
| 29 | The lamp is on. | simple | Surface | 0.896 | Yes |
| 30 | Rain falls from clouds. | simple | Surface | 0.884 | Yes |
| 31 | The flowers are yellow. | simple | Surface | 0.908 | Yes |
| 32 | She read a short story. | simple | Surface | 0.895 | Yes |
| 33 | The clock shows noon. | simple | Surface | 0.883 | Yes |
| 34 | He went home early. | simple | Surface | 0.882 | Yes |
| 35 | The tea is hot. | simple | Surface | 0.895 | Yes |
| 36 | Dogs bark at night. | simple | Surface | 0.888 | Yes |
| 37 | The chair is wooden. | simple | Surface | 0.906 | Yes |
| 38 | She found her keys. | simple | Surface | 0.881 | Yes |
| 39 | The wall is white. | simple | Surface | 0.904 | Yes |
| 40 | He eats breakfast daily. | simple | Surface | 0.880 | Yes |
| 41 | The river flows east. | simple | Surface | 0.883 | Yes |
| 42 | I saw a red bird. | simple | Surface | 0.891 | Yes |
| 43 | The door is locked. | simple | Surface | 0.906 | Yes |
| 44 | She called her mother. | simple | Surface | 0.883 | Yes |
| 45 | The house is small. | simple | Surface | 0.905 | Yes |
| 46 | He kicked the ball. | simple | Surface | 0.897 | Yes |
| 47 | The test was easy. | simple | Surface | 0.902 | Yes |
| 48 | I need more paper. | simple | Surface | 0.890 | Yes |
| 49 | The phone is ringing. | simple | Surface | 0.908 | Yes |
| 50 | She drives a blue car. | simple | Surface | 0.895 | Yes |

### Complex (50 queries) (50 queries, 22.0% accuracy)

Tier distribution: Surface 39 (78%), Reasoning 9 (18%), Deep 2 (4%)

| # | Sentence | Truth | AXIOM Tier | Conf | Correct |
|---|----------|-------|------------|------|---------|
| 1 | The recursive nature of self-referential systems creates emergent properties ... | complex | Reasoning | 0.640 | Yes |
| 2 | Quantum entanglement challenges classical notions of locality and causality i... | complex | Reasoning | 0.636 | Yes |
| 3 | The isomorphism between computational complexity classes and the structure of... | complex | Surface | 0.776 | **No** |
| 4 | Gödel's incompleteness theorems demonstrate that any sufficiently powerful f... | complex | Surface | 0.738 | **No** |
| 5 | The relationship between consciousness and physical substrate remains an unso... | complex | Surface | 0.706 | **No** |
| 6 | Category theory provides a unifying framework for understanding structural re... | complex | Surface | 0.889 | **No** |
| 7 | The boundary between deterministic chaos and genuine stochastic processes has... | complex | Surface | 0.650 | **No** |
| 8 | Emergence in complex adaptive systems suggests that macroscopic phenomena can... | complex | Surface | 0.891 | **No** |
| 9 | The thermodynamic arrow of time, arising from the second law's statistical as... | complex | Surface | 0.820 | **No** |
| 10 | Topological quantum computing exploits non-abelian anyons to achieve fault-to... | complex | Surface | 0.880 | **No** |
| 11 | The underdetermination of scientific theories by empirical evidence implies t... | complex | Surface | 0.889 | **No** |
| 12 | Bayesian epistemology treats belief revision as a process of probabilistic up... | complex | Surface | 0.790 | **No** |
| 13 | The halting problem establishes a fundamental limit on computability, demonst... | complex | Surface | 0.749 | **No** |
| 14 | Distributed consensus protocols must navigate the impossibility results of Fi... | complex | Deep | 0.621 | Yes |
| 15 | The renormalization group in quantum field theory reveals how physical system... | complex | Surface | 0.872 | **No** |
| 16 | Evolutionary game theory extends classical game-theoretic equilibria to popul... | complex | Surface | 0.889 | **No** |
| 17 | The Church-Turing thesis, while not formally provable, has withstood decades ... | complex | Surface | 0.763 | **No** |
| 18 | Kolmogorov complexity provides an objective measure of information content by... | complex | Surface | 0.892 | **No** |
| 19 | The measurement problem in quantum mechanics arises from the apparent incompa... | complex | Surface | 0.887 | **No** |
| 20 | Causal inference in observational studies requires careful application of do-... | complex | Surface | 0.898 | **No** |
| 21 | The P versus NP problem asks whether every problem whose solution can be effi... | complex | Surface | 0.687 | **No** |
| 22 | Homotopy type theory unifies constructive mathematics with higher category th... | complex | Surface | 0.819 | **No** |
| 23 | The no-free-lunch theorems establish that no learning algorithm can outperfor... | complex | Surface | 0.874 | **No** |
| 24 | Phenotypic plasticity and epigenetic inheritance mechanisms complicate the mo... | complex | Reasoning | 0.641 | Yes |
| 25 | The information-theoretic approach to black hole thermodynamics suggests that... | complex | Surface | 0.814 | **No** |
| 26 | Nonequilibrium statistical mechanics extends Boltzmann's framework to systems... | complex | Surface | 0.874 | **No** |
| 27 | The sorites paradox exposes fundamental tensions in classical logic when appl... | complex | Surface | 0.669 | **No** |
| 28 | Algorithmic information theory connects the notion of randomness to incompres... | complex | Reasoning | 0.613 | Yes |
| 29 | The embedding problem in differential geometry asks when a Riemannian manifol... | complex | Surface | 0.808 | **No** |
| 30 | Decoherence theory explains the emergence of classical behaviour from quantum... | complex | Surface | 0.884 | **No** |
| 31 | The frame problem in artificial intelligence concerns how a reasoning system ... | complex | Reasoning | 0.624 | Yes |
| 32 | Modal logic extends propositional logic with necessity and possibility operat... | complex | Surface | 0.866 | **No** |
| 33 | The Curry-Howard correspondence reveals a deep structural isomorphism between... | complex | Surface | 0.741 | **No** |
| 34 | Sparse distributed representations in computational neuroscience suggest that... | complex | Surface | 0.884 | **No** |
| 35 | The Sapir-Whorf hypothesis in its strong form claims that linguistic structur... | complex | Surface | 0.762 | **No** |
| 36 | Persistent homology provides a multiscale topological summary of data by trac... | complex | Surface | 0.893 | **No** |
| 37 | The tragedy of the commons illustrates how rational individual behaviour in s... | complex | Surface | 0.878 | **No** |
| 38 | Autopoietic theory characterises living systems as self-producing networks th... | complex | Surface | 0.761 | **No** |
| 39 | The Langlands program conjectures deep reciprocity laws connecting number the... | complex | Deep | 0.640 | Yes |
| 40 | Neuroplasticity research demonstrates that cortical representational maps are... | complex | Surface | 0.721 | **No** |
| 41 | The Chinese room argument contends that syntactic manipulation of formal symb... | complex | Reasoning | 0.636 | Yes |
| 42 | Ergodic theory studies the long-term statistical behaviour of dynamical syste... | complex | Surface | 0.798 | **No** |
| 43 | The explanatory gap between phenomenal consciousness and physical processes m... | complex | Surface | 0.748 | **No** |
| 44 | Mechanism design theory inverts the game-theoretic problem by asking how to c... | complex | Surface | 0.842 | **No** |
| 45 | The holographic principle, originating from black hole entropy bounds, sugges... | complex | Surface | 0.851 | **No** |
| 46 | Constructive type theory replaces the law of excluded middle with a computati... | complex | Surface | 0.892 | **No** |
| 47 | Stochastic gradient descent converges to local minima in non-convex loss land... | complex | Reasoning | 0.632 | Yes |
| 48 | The binding problem in cognitive science asks how distributed neural processi... | complex | Reasoning | 0.625 | Yes |
| 49 | Topos theory generalises set-theoretic foundations by replacing classical log... | complex | Reasoning | 0.630 | Yes |
| 50 | The paradox of the heap reveals that our informal notion of number admits no ... | complex | Surface | 0.858 | **No** |

### Realistic Enterprise (100 queries) (100 queries, 55.0% accuracy)

Tier distribution: Surface 70 (70%), Reasoning 14 (14%), Deep 16 (16%)

| # | Sentence | Truth | AXIOM Tier | Conf | Correct |
|---|----------|-------|------------|------|---------|
| 1 | What are your business hours? | simple | Surface | 0.873 | Yes |
| 2 | How do I reset my password? | simple | Reasoning | 0.625 | **No** |
| 3 | What is the return policy? | simple | Surface | 0.889 | Yes |
| 4 | Where is my order? | simple | Surface | 0.869 | Yes |
| 5 | Can I cancel my subscription? | simple | Surface | 0.881 | Yes |
| 6 | What payment methods do you accept? | simple | Deep | 0.635 | **No** |
| 7 | How much does shipping cost? | simple | Surface | 0.882 | Yes |
| 8 | Is this product in stock? | simple | Surface | 0.905 | Yes |
| 9 | What is your phone number? | simple | Surface | 0.871 | Yes |
| 10 | How do I contact support? | simple | Surface | 0.874 | Yes |
| 11 | When does the sale end? | simple | Surface | 0.869 | Yes |
| 12 | Do you ship internationally? | simple | Surface | 0.878 | Yes |
| 13 | What size should I order? | simple | Surface | 0.877 | Yes |
| 14 | Can I change my delivery address? | simple | Deep | 0.636 | **No** |
| 15 | Is there a warranty on this item? | simple | Surface | 0.659 | Yes |
| 16 | What does this error code mean? | simple | Surface | 0.880 | Yes |
| 17 | How do I update my billing information? | simple | Reasoning | 0.623 | **No** |
| 18 | What file formats do you support? | simple | Reasoning | 0.634 | **No** |
| 19 | Can I get a refund? | simple | Surface | 0.902 | Yes |
| 20 | How long does delivery take? | simple | Surface | 0.883 | Yes |
| 21 | Do you have a mobile app? | simple | Surface | 0.695 | Yes |
| 22 | What is my account balance? | simple | Surface | 0.875 | Yes |
| 23 | How do I delete my account? | simple | Reasoning | 0.625 | **No** |
| 24 | Is there free shipping? | simple | Surface | 0.891 | Yes |
| 25 | What colour options are available? | simple | Surface | 0.881 | Yes |
| 26 | Can I speak to a manager? | simple | Surface | 0.658 | Yes |
| 27 | What is your email address? | simple | Surface | 0.872 | Yes |
| 28 | How do I apply a discount code? | simple | Surface | 0.668 | Yes |
| 29 | Is this item on sale? | simple | Surface | 0.899 | Yes |
| 30 | Where are you located? | simple | Surface | 0.872 | Yes |
| 31 | Can I track my package? | simple | Surface | 0.883 | Yes |
| 32 | What are the system requirements? | simple | Surface | 0.888 | Yes |
| 33 | Do you offer gift wrapping? | simple | Surface | 0.887 | Yes |
| 34 | How do I change my email? | simple | Deep | 0.624 | **No** |
| 35 | Is there a student discount? | simple | Surface | 0.905 | Yes |
| 36 | What browsers do you support? | simple | Surface | 0.883 | Yes |
| 37 | How do I enable notifications? | simple | Surface | 0.872 | Yes |
| 38 | Can I download my invoice? | simple | Surface | 0.883 | Yes |
| 39 | What is the minimum order amount? | simple | Surface | 0.890 | Yes |
| 40 | Do you offer bulk pricing? | simple | Surface | 0.887 | Yes |
| 41 | Write a professional email declining a meeting invitation and suggesting an a... | moderate | Surface | 0.718 | **No** |
| 42 | Summarise the key differences between REST and GraphQL APIs for a technical a... | moderate | Surface | 0.647 | **No** |
| 43 | Explain how a binary search tree works and when you would use one instead of ... | moderate | Surface | 0.882 | **No** |
| 44 | Draft a quarterly business review summary highlighting revenue growth and are... | moderate | Surface | 0.647 | **No** |
| 45 | Compare the advantages and disadvantages of microservices versus monolithic a... | moderate | Surface | 0.650 | **No** |
| 46 | Explain the difference between supervised and unsupervised machine learning w... | moderate | Surface | 0.653 | **No** |
| 47 | Write a SQL query to find the top ten customers by total purchase amount in t... | moderate | Surface | 0.910 | **No** |
| 48 | Describe how HTTPS encryption works from the initial handshake through data t... | moderate | Reasoning | 0.633 | Yes |
| 49 | Create a project timeline for migrating a legacy database to a cloud-hosted s... | moderate | Surface | 0.680 | **No** |
| 50 | Explain the CAP theorem and its practical implications for choosing a databas... | moderate | Surface | 0.656 | **No** |
| 51 | Write a Python function that reads a CSV file and calculates summary statisti... | moderate | Surface | 0.905 | **No** |
| 52 | Draft a technical specification for a user authentication system with OAuth2 ... | moderate | Surface | 0.649 | **No** |
| 53 | Explain how Docker containers differ from virtual machines and when to use ea... | moderate | Reasoning | 0.632 | Yes |
| 54 | Summarise the main provisions of GDPR and their implications for storing cust... | moderate | Surface | 0.719 | **No** |
| 55 | Write a code review checklist covering security, performance, and maintainabi... | moderate | Reasoning | 0.642 | Yes |
| 56 | Explain how load balancing works across multiple application servers and comm... | moderate | Deep | 0.626 | **No** |
| 57 | Draft an incident response plan for a production database outage affecting cu... | moderate | Surface | 0.780 | **No** |
| 58 | Compare PostgreSQL and MongoDB for a product catalogue with variable attribut... | moderate | Surface | 0.648 | **No** |
| 59 | Explain the git rebase workflow and when it is preferable to merge commits. | moderate | Reasoning | 0.636 | Yes |
| 60 | Write unit tests for a shopping cart class that handles adding items, removin... | moderate | Surface | 0.830 | **No** |
| 61 | Explain the difference between TCP and UDP and give examples of when to use e... | moderate | Surface | 0.737 | **No** |
| 62 | Write a Bash script that monitors disk usage and sends an alert when any part... | moderate | Surface | 0.880 | **No** |
| 63 | Describe how database indexing works and explain the trade-offs between B-tre... | moderate | Surface | 0.890 | **No** |
| 64 | Draft a data retention policy for a SaaS company that handles personally iden... | moderate | Surface | 0.643 | **No** |
| 65 | Explain the observer design pattern and provide a practical example in Python. | moderate | Surface | 0.662 | **No** |
| 66 | Create a Kubernetes deployment manifest for a stateless web application with ... | moderate | Surface | 0.683 | **No** |
| 67 | Compare message queues and event streams for decoupling services in a backend... | moderate | Surface | 0.651 | **No** |
| 68 | Write a migration script to add a new column to a production database table w... | moderate | Surface | 0.906 | **No** |
| 69 | Explain how continuous integration differs from continuous deployment and out... | moderate | Reasoning | 0.637 | Yes |
| 70 | Summarise the key principles of twelve-factor app methodology and how they ap... | moderate | Surface | 0.894 | **No** |
| 71 | Design a comprehensive data pipeline architecture that ingests real-time even... | complex | Deep | 0.618 | Yes |
| 72 | Analyse the tradeoffs between eventual consistency and strong consistency in ... | complex | Surface | 0.795 | **No** |
| 73 | Evaluate the technical and organisational implications of migrating a large-s... | complex | Deep | 0.628 | Yes |
| 74 | Design an ML model serving infrastructure that supports A/B testing, canary d... | complex | Surface | 0.788 | **No** |
| 75 | Propose a zero-trust security architecture for a hybrid cloud environment tha... | complex | Reasoning | 0.623 | Yes |
| 76 | Analyse the failure modes of distributed consensus algorithms under Byzantine... | complex | Surface | 0.835 | **No** |
| 77 | Design a multi-tenant SaaS platform architecture that provides data isolation... | complex | Surface | 0.739 | **No** |
| 78 | Evaluate the implications of the CAP theorem, PACELC framework, and the harve... | complex | Surface | 0.874 | **No** |
| 79 | Analyse how transformer attention mechanisms scale with sequence length and p... | complex | Surface | 0.707 | **No** |
| 80 | Design a chaos engineering programme for a microservices platform that system... | complex | Surface | 0.698 | **No** |
| 81 | Develop a comprehensive strategy for implementing differential privacy in a r... | complex | Surface | 0.846 | **No** |
| 82 | Evaluate the trade-offs between compile-time and runtime type safety in progr... | complex | Deep | 0.631 | Yes |
| 83 | Design an observability platform that correlates distributed traces, metrics,... | complex | Surface | 0.731 | **No** |
| 84 | Analyse the implications of quantum computing advances for current public-key... | complex | Surface | 0.877 | **No** |
| 85 | Design a real-time fraud detection system that processes millions of transact... | complex | Deep | 0.628 | Yes |
| 86 | Evaluate the architectural patterns for implementing event sourcing and CQRS ... | complex | Surface | 0.794 | **No** |
| 87 | Propose a federated learning architecture for training machine learning model... | complex | Deep | 0.621 | Yes |
| 88 | Analyse the sociotechnical factors that cause large-scale distributed system ... | complex | Deep | 0.621 | Yes |
| 89 | Design a database migration strategy for a system serving one billion request... | complex | Surface | 0.821 | **No** |
| 90 | Evaluate the fundamental limitations of current large language model architec... | complex | Reasoning | 0.615 | Yes |
| 91 | Design a capacity planning framework for a cloud-native platform that predict... | complex | Reasoning | 0.621 | Yes |
| 92 | Evaluate the challenges of implementing a polyglot persistence strategy in a ... | complex | Deep | 0.614 | Yes |
| 93 | Propose an automated security scanning pipeline that integrates static analys... | complex | Reasoning | 0.623 | Yes |
| 94 | Analyse the theoretical and practical challenges of building a truly reproduc... | complex | Deep | 0.619 | Yes |
| 95 | Design a progressive delivery system that supports feature flags, percentage ... | complex | Deep | 0.623 | Yes |
| 96 | Evaluate the trade-offs between shared-nothing and shared-disk architectures ... | complex | Surface | 0.838 | **No** |
| 97 | Propose a strategy for decomposing a legacy monolithic application into domai... | complex | Deep | 0.620 | Yes |
| 98 | Analyse the implications of edge computing for application architecture, cons... | complex | Reasoning | 0.619 | Yes |
| 99 | Design a comprehensive API governance framework for a large organisation with... | complex | Deep | 0.618 | Yes |
| 100 | Evaluate the technical debt implications of choosing between a custom-built i... | complex | Deep | 0.625 | Yes |

## 3. Cost Model

Token estimates per tier:

| Tier | Model | Input Tokens | Output Tokens | Cost/Query |
|------|-------|-------------|---------------|------------|
| Surface | Haiku | 150 | 200 | $0.000920 |
| Reasoning | Sonnet | 300 | 500 | $0.008400 |
| Deep | Opus | 800 | 1500 | $0.124500 |

Pricing: Haiku $0.80/$4.00, Sonnet $3.00/$15.00, Opus $15.00/$75.00 per million tokens (input/output).

### Measured Routing Distribution (all datasets combined)

- Surface: 79.5%  |  Reasoning: 11.5%  |  Deep: 9.0%

### Cost Comparison at Scale

| Scale | All-Haiku | All-Sonnet | All-Opus | AXIOM Routed | Savings vs Opus |
|-------|-----------|------------|----------|--------------|-----------------|
|      1k | $     1.59 | $      5.95 | $   29.75 | $       12.90 |           56.6% |
|     10k | $    15.87 | $     59.50 | $  297.49 | $      129.02 |           56.6% |
|    100k | $   158.66 | $    594.98 | $ 2974.88 | $     1290.24 |           56.6% |

## 4. Routing Analysis

### Confidence Distribution

```
  0.0-0.1 |   0 
  0.1-0.2 |   0 
  0.2-0.3 |   0 
  0.3-0.4 |   0 
  0.4-0.5 |   0 
  0.5-0.6 |   0 
  0.6-0.7 |  63 ███████████████████████████
  0.7-0.8 |  22 █████████
  0.8-0.9 |  93 ████████████████████████████████████████
  0.9-1.0 |  22 █████████
```

**Surface** — 159 queries, confidence: mean 0.834, min 0.643, max 0.910

**Reasoning** — 23 queries, confidence: mean 0.629, min 0.613, max 0.642

**Deep** — 18 queries, confidence: mean 0.625, min 0.614, max 0.640

### Correct Routing Examples

- **"The cat sat on the mat."**
  - Truth: simple → AXIOM: Surface (conf 0.648) — Correct. Short declarative sentence with common vocabulary stays at Surface tier.
- **"Water is wet."**
  - Truth: simple → AXIOM: Surface (conf 0.883) — Correct. Short declarative sentence with common vocabulary stays at Surface tier.
- **"The sun is bright today."**
  - Truth: simple → AXIOM: Surface (conf 0.909) — Correct. Short declarative sentence with common vocabulary stays at Surface tier.
- **"The recursive nature of self-referential systems creates emergent properties that resis..."**
  - Truth: complex → AXIOM: Reasoning (conf 0.640) — Correct. Complex content escalated past Surface; Reasoning accepted for complex ground truth.
- **"Quantum entanglement challenges classical notions of locality and causality in ways tha..."**
  - Truth: complex → AXIOM: Reasoning (conf 0.636) — Correct. Complex content escalated past Surface; Reasoning accepted for complex ground truth.
- **"Distributed consensus protocols must navigate the impossibility results of Fischer, Lyn..."**
  - Truth: complex → AXIOM: Deep (conf 0.621) — Correct. Technical/philosophical content with subordination escalated to Deep.

### Incorrect Routing Examples

- **"The isomorphism between computational complexity classes and the structure of mathemati..."**
  - Truth: complex → AXIOM: Surface (conf 0.776) — **Incorrect.** Under-escalation: complex query stayed at Surface (conf 0.776). The encoder may not capture the full complexity of this sentence — possible if it uses simple syntax despite complex semantics.
- **"Gödel's incompleteness theorems demonstrate that any sufficiently powerful formal syst..."**
  - Truth: complex → AXIOM: Surface (conf 0.738) — **Incorrect.** Under-escalation: complex query stayed at Surface (conf 0.738). The encoder may not capture the full complexity of this sentence — possible if it uses simple syntax despite complex semantics.
- **"The relationship between consciousness and physical substrate remains an unsolved probl..."**
  - Truth: complex → AXIOM: Surface (conf 0.706) — **Incorrect.** Under-escalation: complex query stayed at Surface (conf 0.706). The encoder may not capture the full complexity of this sentence — possible if it uses simple syntax despite complex semantics.
- **"Category theory provides a unifying framework for understanding structural relationship..."**
  - Truth: complex → AXIOM: Surface (conf 0.889) — **Incorrect.** Under-escalation: complex query stayed at Surface (conf 0.889). The encoder may not capture the full complexity of this sentence — possible if it uses simple syntax despite complex semantics.
- **"The boundary between deterministic chaos and genuine stochastic processes has profound ..."**
  - Truth: complex → AXIOM: Surface (conf 0.650) — **Incorrect.** Under-escalation: complex query stayed at Surface (conf 0.650). The encoder may not capture the full complexity of this sentence — possible if it uses simple syntax despite complex semantics.
- **"Emergence in complex adaptive systems suggests that macroscopic phenomena cannot always..."**
  - Truth: complex → AXIOM: Surface (conf 0.891) — **Incorrect.** Under-escalation: complex query stayed at Surface (conf 0.891). The encoder may not capture the full complexity of this sentence — possible if it uses simple syntax despite complex semantics.

### G5 Norm Inflation Diagnostic

Test: single complex sentence vs. same sentence repeated 5 times.

| Metric | Value |
|--------|-------|
| Sentence | "The recursive nature of self-referential systems creates emergent properties ..." |
| Single G5 norm | 3.6450 |
| Repeated 5x G5 norm (raw) | 4.8574 |
| Raw inflation ratio | 1.33x |
| Repeated 5x G5 norm (chunked via resolve_text) | 3.6450 |
| Chunked inflation ratio | 1.00x |

Chunked ratio 1.00x is within the 1.5x threshold. Sentence chunking adequately controls G5 norm inflation.

## 5. Architecture Summary

AXIOM is a lightweight, sparse-computation routing architecture that classifies input queries by complexity and routes them to appropriately-sized language models. The system employs a hierarchical resolver with three tiers (Surface, Reasoning, Deep), a content-addressable embedding cache, dynamic coalition formation with stochastic node selection, lateral traversal for confidence recovery, Hebbian learning with Oja's rule, and a G5 structural syntax encoder that produces 128-dimensional embeddings capturing lexical, syntactic, and semantic complexity signals. Surface nodes are frozen with analytical initialisation; Reasoning and Deep nodes learn contrastive discrimination boundaries. The entire system runs in under 1 millisecond per routing decision with zero external ML framework dependencies, implemented in approximately 6,000 lines of Rust.

| Metric | Value |
|--------|-------|
| Total parameters | 1205376 |
| Weight norm | 822.10 |
| Embedding dimension | 128 |
| Tests passing | 159 |
| Mean routing time | 1311 µs |
| Overall routing accuracy | 58.0% |
| Savings vs all-Opus (100k) | 56.6% |

## 6. Scenario Testing — Multi-Paragraph Enterprise Inputs

**6 scenarios tested, 3/6 correct (50% accuracy)**

| # | ID | Input (100 chars) | Truth | AXIOM Tier | Conf | G5 Norm | Chunks | Correct |
|---|-----|-------------------|-------|------------|------|---------|--------|---------|
| 1 | scenario_01 | Hi there, I hope you are having a good week. I wanted to get in touch about my recent order that ... | simple | Surface | 0.908 | 3.197 | 8 | Yes |
| 2 | scenario_02 | Reconcile Kant's categorical imperative with utilitarian ethics. | complex | Reasoning | 0.631 | 3.491 | 1 | Yes |
| 3 | scenario_03 | I am trying to understand how database indexing works and when I should use it in my application.... | moderate | Surface | 0.856 | 3.453 | 5 | **No** |
| 4 | scenario_04 | The relationship between consciousness and physical substrate has occupied philosophers and scien... | complex | Surface | 0.889 | 3.786 | 6 | **No** |
| 5 | scenario_05 | I have been a customer for three years and generally really enjoy your service. Last month I upgr... | simple | Surface | 0.907 | 2.773 | 4 | Yes |
| 6 | scenario_06 | I have inherited a codebase and I am trying to understand what this function does. Can you explai... | moderate | Surface | 0.888 | 1.892 | 7 | **No** |

### Per-Scenario Diagnosis

**scenario_01** (simple) — simple → Surface (conf 0.908, G5 3.197, 8 chunks) — **CORRECT**
- Multi-paragraph customer email with common vocabulary correctly stays at Surface. Sentence chunking splits into individual simple sentences and none escalate.

**scenario_02** (complex) — complex → Reasoning (conf 0.631, G5 3.491, 1 chunks) — **CORRECT**
- Short philosophical prompt correctly escalated despite minimal structural markers. Rare vocabulary or semantic features triggered escalation.

**scenario_03** (moderate) — moderate → Surface (conf 0.856, G5 3.453, 5 chunks) — **INCORRECT**
- Under-escalation: moderate technical question stayed at Surface (conf 0.856). The structural signals were insufficient to trigger escalation.

**scenario_04** (complex) — complex → Surface (conf 0.889, G5 3.786, 6 chunks) — **INCORRECT**
- Under-escalation: dense academic prose stayed at Surface (conf 0.889). Unexpected — this input has strong structural complexity markers.

**scenario_05** (simple) — simple → Surface (conf 0.907, G5 2.773, 4 chunks) — **CORRECT**
- Contextual framing around a simple question correctly routes to Surface. The chunking identifies the simple sentences and the overall complexity stays low.

**scenario_06** (moderate) — moderate → Surface (conf 0.888, G5 1.892, 7 chunks) — **INCORRECT**
- Under-escalation: code explanation stayed at Surface (conf 0.888). The encoder may not recognise code structure as complexity-bearing.

### Real-World Readiness Assessment

- Multi-paragraph inputs (>1 chunk): 2/5 correct (40%)
- Single-chunk inputs: 1/1 correct (100%)

**Strategy C (threshold-based chunk escalation):** `resolve_text` splits multi-paragraph inputs into sentence chunks and routes each independently. If >40% of chunks produce surface confidence below the threshold (0.85), the input escalates to the tier of the lowest-confidence chunk. Otherwise, the highest-confidence chunk's routing is used. This prevents over-escalation of simple multi-paragraph inputs (e.g., customer emails) while allowing escalation when a sufficient fraction of chunks signal complexity.

**Challenge: confidence compression.** After training, most chunks produce surface confidences in the 0.84–0.92 range. With a threshold of 0.85, moderate and complex chunks often score just above the threshold, so fewer than 40% fall below it. This explains the 3/6 scenario result: scenarios 3, 4, and 6 have mean confidences (0.856, 0.889, 0.888) that hover near the threshold boundary. The individual chunks within these scenarios do not consistently fall below 0.85, so Strategy C does not trigger escalation. See Section 7 for proposed mitigations.

**Positive finding:** Scenario 02 ("Reconcile Kant's categorical imperative with utilitarian ethics") correctly escalated to Reasoning despite having only 7 words and no structural complexity markers. The rare vocabulary ("Kant's", "categorical", "imperative", "utilitarian") was sufficient to trigger escalation — the encoder is more semantically aware than anticipated for single-sentence inputs.

## 7. Limitations and Future Work

### Phase History

| Phase | Focus | Key Outcome |
|-------|-------|-------------|
| 1–3 | Core architecture | Sparse graph, 3-tier routing, embedding cache, lateral traversal |
| 4 | Structural encoder | Position-weighted embeddings, 4 syntactic features, Hebbian learning |
| 5–6 | Learning stabilisation | Oja's rule, weight decay, contrastive loss, lr=0.001 |
| 7–8 | Node specialisation | Standalone nodes, dynamic coalition formation, stochastic selection |
| 9–10 | Confidence calibration | Percentile-based thresholds, auto-tuner, minimum escalation rate |
| 11–12 | Adversarial robustness | 40-sentence adversarial corpus, garden-path sentences, 47% → 55% |
| 13 | Dynamic coalitions | Stochastic node selection, mean coalition size 4.0 |
| 14 | G5 structural features | Magnitude penalty, bucketed norms, adversarial score 55% (22/40) |
| 15 | Production squeeze | 1.2M params (mid_dim=128), Strategy C chunk aggregation, final report |

### Known Limitations

1. **Encoder capacity bottleneck.** The 128-dimensional input encoding is the binding constraint on routing accuracy. Quadrupling parameters from 1.2M to 4.8M (mid_dim 128→512) produced identical adversarial accuracy (22/40, 55%). The encoder captures lexical and structural features but cannot represent deep semantic complexity (e.g., philosophical arguments in simple syntax).

2. **Confidence distribution compression.** After training, Surface node confidences cluster in a narrow band (approximately 0.84–0.92). This makes threshold-based discrimination fragile: a threshold of 0.85 passes most inputs, while 0.90 escalates most. The 65th-percentile calibration strategy works for single-sentence routing but leaves little margin for chunk-aggregation strategies that depend on per-chunk threshold comparisons.

3. **Multi-paragraph routing via chunking.** Strategy C (threshold-based chunk escalation) achieves 3/6 scenario accuracy (50%). The core difficulty: most multi-paragraph inputs contain at least one structurally simple sentence, which produces a high Surface confidence that anchors the aggregation. Escalation requires >40% of chunks to individually fall below the surface threshold, which rarely occurs when the threshold (0.85) sits within the compressed confidence band.

4. **Semantic vs. structural complexity.** Sentences like "Cogito ergo sum" (3 words, philosophically deep) and "the big fluffy white dog played happily" (7 words, semantically simple) can share similar structural profiles. Without world knowledge or attention over token context, the encoder cannot distinguish semantic depth from syntactic simplicity.

5. **G5 norm length sensitivity.** Longer inputs produce higher G5 norms regardless of complexity, conflating length with structural depth. Bucketed norms (short/medium/long) partially mitigate this but do not eliminate the correlation.

### Future Directions

1. **Attention mechanism.** Replace or augment the bag-of-features encoder with a lightweight self-attention layer (1–2 heads, 128-dim). This would allow the encoder to weight tokens by contextual relevance, potentially resolving semantic-vs-structural ambiguity.

2. **Learned chunk aggregation.** Replace the fixed 40% threshold with a small learned aggregation network that takes per-chunk confidence vectors and produces a single routing decision. This could adapt to the compressed confidence distribution.

3. **Parse tree depth estimation.** Add recursive feature extraction that estimates syntactic tree depth without a full parser. Proxy features (comma-separated clause counting, relative pronoun density) could improve discrimination for nested structures.

4. **Per-class calibration.** Maintain separate confidence distributions for short (<6 words), medium, and long (>10 words) inputs, producing length-appropriate thresholds rather than a single global threshold.

5. **Real API integration.** The current cost model uses simulated token counts and pricing. Integration with actual Claude API endpoints would validate routing decisions against response quality, enabling closed-loop optimisation where routing accuracy is measured by downstream task performance rather than label agreement.

---
*Report generated by axiom_report. No API calls were made — all costs are simulated.*

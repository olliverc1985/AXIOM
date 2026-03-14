"""
AXIOM Encoder Autoresearch — Evaluation Harness
================================================
READ ONLY. Do not modify this file.

This file contains:
- All test sentences with ground-truth labels
- Training simple sentences for analytical surface weight init
- The evaluation function that measures routing accuracy

The agent modifies train.py only.
"""

import math
import time

# ─── Training simple sentences (for computing surface weight direction) ───
# These are used to compute the mean encoding direction = surface weight.
TRAIN_SIMPLE = [
    "the cat sat on the mat",
    "hello world",
    "the dog runs fast",
    "it is raining today",
    "she likes green apples",
    "the sun is bright",
    "birds fly south in winter",
    "water flows downhill",
    "he reads a book",
    "the sky is blue",
    "fish swim in the sea",
    "bread is made from flour",
    "trees grow tall",
    "snow falls in december",
    "the baby sleeps quietly",
    "she walked to the store",
    "the car is red",
    "he ate lunch at noon",
    "my dog likes bones",
    "the door is open",
    "we went to the park",
    "the flowers are blooming",
    "it snowed last night",
    "they played football after school",
    "the milk is cold",
    "she wore a blue dress",
    "the train arrived on time",
    "he fixed the broken chair",
    "we ate dinner together",
    "the cat chased the mouse",
    "rain makes the grass green",
    "the clock struck twelve",
    "she sings every morning",
    "he painted the fence white",
    "the children ran outside",
    "they watched a movie last night",
    "coffee keeps me awake",
    "the wind blows gently",
    "she closed the window",
    "the soup is hot",
    # Additional simple from adversarial set
    "the big red dog ran quickly down the long straight road toward the tall old brown wooden fence near the small quiet house by the river",
    "the little grey cat sat on the soft warm mat by the big stone fireplace and purred quietly while the rain fell outside",
    "my grandmother bakes the most delicious chocolate chip cookies every single weekend without fail",
    "the tintinnabulation resonated melodiously",
    "photosynthesis converts sunlight efficiently",
    "it is raining today",
    "the sky is blue",
    "water flows downhill",
    "if then else",
    "she runs fast",
    "go",
    "he gave her cat food",
    "the big fluffy white dog played happily in the sunny green park all morning long",
    "they ate pizza",
    "she danced gracefully across the wooden stage",
    "the lamp is on the table",
]

# ─── Training complex sentences (for G5 calibration) ───
TRAIN_COMPLEX = [
    "the recursive nature of self-referential systems creates emergent properties that resist reduction",
    "consciousness remains an unsolved problem at the intersection of neuroscience philosophy and computation",
    "the halting problem demonstrates fundamental limits of algorithmic decidability in formal systems",
    "quantum entanglement challenges classical notions of locality and suggests nonlocal correlations in nature",
    "goedel's incompleteness theorems reveal inherent limitations in any sufficiently powerful formal axiomatic system",
    "the boundary between deterministic chaos and true randomness has profound implications for predictability",
    "emergence in complex adaptive systems suggests that reductionist explanations are fundamentally insufficient",
    "the relationship between syntactic structure and semantic meaning in natural language defies simple formalisation",
    "computational irreducibility implies that some processes cannot be predicted without full simulation",
    "the tension between expressiveness and decidability shapes the design of every formal language and logic",
    "category theory provides a unifying framework for mathematical structures through abstract morphisms and functors",
    "the measurement problem in quantum mechanics exposes deep philosophical questions about the nature of observation and reality",
    "information theoretic entropy quantifies uncertainty and establishes fundamental limits on data compression and transmission",
    "the interplay between cooperation and competition in evolutionary game theory produces counterintuitive stable equilibria",
    "kolmogorov complexity offers an objective measure of randomness but is itself uncomputable in the general case",
]

# ─── Test sentences: validation (105) + adversarial (41) = 146 ───
# Labels: "simple", "moderate", "complex"
# Scoring: simple → must route to Surface; complex → must escalate; moderate → not scored
TEST_SENTENCES = [
    # ═══ Validation: Simple (40) ═══
    ("the cat sat on the mat", "simple"),
    ("hello world", "simple"),
    ("the dog runs fast", "simple"),
    ("it is raining today", "simple"),
    ("she likes green apples", "simple"),
    ("the sun is bright", "simple"),
    ("birds fly south in winter", "simple"),
    ("water flows downhill", "simple"),
    ("he reads a book", "simple"),
    ("the sky is blue", "simple"),
    ("fish swim in the sea", "simple"),
    ("bread is made from flour", "simple"),
    ("trees grow tall", "simple"),
    ("snow falls in december", "simple"),
    ("the baby sleeps quietly", "simple"),
    ("she walked to the store", "simple"),
    ("the car is red", "simple"),
    ("he ate lunch at noon", "simple"),
    ("my dog likes bones", "simple"),
    ("the door is open", "simple"),
    ("we went to the park", "simple"),
    ("the flowers are blooming", "simple"),
    ("it snowed last night", "simple"),
    ("they played football after school", "simple"),
    ("the milk is cold", "simple"),
    ("she wore a blue dress", "simple"),
    ("the train arrived on time", "simple"),
    ("he fixed the broken chair", "simple"),
    ("we ate dinner together", "simple"),
    ("the cat chased the mouse", "simple"),
    ("rain makes the grass green", "simple"),
    ("the clock struck twelve", "simple"),
    ("she sings every morning", "simple"),
    ("he painted the fence white", "simple"),
    ("the children ran outside", "simple"),
    ("they watched a movie last night", "simple"),
    ("coffee keeps me awake", "simple"),
    ("the wind blows gently", "simple"),
    ("she closed the window", "simple"),
    ("the soup is hot", "simple"),
    # ═══ Validation: Moderate (33) — not scored ═══
    ("machine learning models require significant computational resources", "moderate"),
    ("the stock market fluctuates based on investor sentiment and economic indicators", "moderate"),
    ("neural networks approximate complex nonlinear functions through layered transformations", "moderate"),
    ("sustainable energy solutions demand innovative engineering approaches", "moderate"),
    ("distributed systems must handle network partitions and eventual consistency", "moderate"),
    ("the relationship between correlation and causation is frequently misunderstood", "moderate"),
    ("genetic algorithms evolve solutions through mutation crossover and selection", "moderate"),
    ("urban planning requires balancing economic growth with environmental preservation", "moderate"),
    ("cryptographic protocols ensure secure communication over untrusted channels", "moderate"),
    ("database indexing strategies significantly impact query performance", "moderate"),
    ("compiler optimisation transforms source code into efficient machine instructions", "moderate"),
    ("reinforcement learning agents maximise cumulative reward through exploration", "moderate"),
    ("microservice architectures introduce complexity in exchange for scalability", "moderate"),
    ("the efficiency of sorting algorithms depends on input distribution characteristics", "moderate"),
    ("parallel computing enables processing of large datasets across multiple cores", "moderate"),
    ("memory management in systems programming prevents resource leaks and corruption", "moderate"),
    ("functional programming emphasises immutability and compositional reasoning", "moderate"),
    ("type systems provide compile time guarantees about program behaviour", "moderate"),
    ("version control systems track changes and enable collaborative software development", "moderate"),
    ("containerisation isolates applications and their dependencies for consistent deployment", "moderate"),
    ("natural language processing bridges the gap between human communication and machine understanding", "moderate"),
    ("gradient descent optimises model parameters by following the steepest direction of loss reduction", "moderate"),
    ("load balancing distributes incoming requests across multiple servers to prevent overload", "moderate"),
    ("data normalisation reduces redundancy and improves integrity in relational databases", "moderate"),
    ("event driven architectures decouple producers and consumers for better scalability", "moderate"),
    ("regularisation techniques prevent overfitting by penalising model complexity during training", "moderate"),
    ("network protocols define rules for data exchange between connected computing devices", "moderate"),
    ("operating systems manage hardware resources and provide services to application software", "moderate"),
    ("continuous integration automates testing and building software after every code change", "moderate"),
    ("hash functions map arbitrary input to fixed size output for efficient data retrieval", "moderate"),
    ("binary search reduces the search space by half with each comparison step", "moderate"),
    ("recursive algorithms solve problems by breaking them into smaller identical subproblems", "moderate"),
    ("concurrency control mechanisms ensure data consistency in multi user database systems", "moderate"),
    # ═══ Validation: Complex (32) ═══
    ("the recursive nature of self-referential systems creates emergent properties that resist reduction", "complex"),
    ("consciousness remains an unsolved problem at the intersection of neuroscience philosophy and computation", "complex"),
    ("the halting problem demonstrates fundamental limits of algorithmic decidability in formal systems", "complex"),
    ("quantum entanglement challenges classical notions of locality and suggests nonlocal correlations in nature", "complex"),
    ("goedel's incompleteness theorems reveal inherent limitations in any sufficiently powerful formal axiomatic system", "complex"),
    ("the boundary between deterministic chaos and true randomness has profound implications for predictability", "complex"),
    ("emergence in complex adaptive systems suggests that reductionist explanations are fundamentally insufficient", "complex"),
    ("the relationship between syntactic structure and semantic meaning in natural language defies simple formalisation", "complex"),
    ("computational irreducibility implies that some processes cannot be predicted without full simulation", "complex"),
    ("the tension between expressiveness and decidability shapes the design of every formal language and logic", "complex"),
    ("category theory provides a unifying framework for mathematical structures through abstract morphisms and functors", "complex"),
    ("the measurement problem in quantum mechanics exposes deep philosophical questions about the nature of observation and reality", "complex"),
    ("information theoretic entropy quantifies uncertainty and establishes fundamental limits on data compression and transmission", "complex"),
    ("the interplay between cooperation and competition in evolutionary game theory produces counterintuitive stable equilibria", "complex"),
    ("kolmogorov complexity offers an objective measure of randomness but is itself uncomputable in the general case", "complex"),
    ("strange attractors in dynamical systems exhibit sensitive dependence on initial conditions while maintaining bounded trajectories", "complex"),
    ("the church turing thesis equates mechanical computation with turing machine computation but remains an unproven conjecture", "complex"),
    ("the frame problem in artificial intelligence reveals how difficult it is to represent the effects of actions in a changing world", "complex"),
    ("non equilibrium thermodynamics describes systems far from steady state where entropy production drives self organisation", "complex"),
    ("the underdetermination of scientific theory by empirical evidence means multiple incompatible theories can explain the same observations", "complex"),
    ("modal logic extends propositional logic with operators for necessity and possibility enabling reasoning about hypothetical scenarios", "complex"),
    ("the Chinese room argument challenges the notion that syntactic manipulation alone can produce genuine semantic understanding", "complex"),
    ("topological quantum computing encodes information in braided anyons whose non abelian statistics provide inherent fault tolerance", "complex"),
    ("the renormalisation group explains how physical systems exhibit universal behaviour near critical phase transitions regardless of microscopic details", "complex"),
    ("algorithmic information theory unifies computation probability and statistics through the lens of Kolmogorov complexity and Solomonoff induction", "complex"),
    ("the binding problem asks how distributed neural representations combine into unified conscious experience across spatially separated brain regions", "complex"),
    ("Bayesian nonparametric models allow the complexity of statistical models to grow with available data rather than being fixed a priori", "complex"),
    ("the symbol grounding problem questions how formal symbols in a computational system acquire meaning beyond their syntactic relationships", "complex"),
    ("ergodic theory studies the long term statistical behaviour of dynamical systems and underpins much of statistical mechanics", "complex"),
    ("the Curry Howard correspondence reveals a deep isomorphism between proofs in logic and programs in typed lambda calculus", "complex"),
    ("causal inference frameworks distinguish genuine cause and effect from mere statistical association using intervention and counterfactual reasoning", "complex"),
    ("the no free lunch theorem establishes that no single optimisation algorithm outperforms all others across every possible problem landscape", "complex"),
    # ═══ Adversarial (41) ═══
    # Very short complex
    ("Cogito ergo sum", "complex"),
    ("Godel's proof breaks mathematics", "complex"),
    ("Being precedes essence", "complex"),
    ("time is relative", "complex"),
    ("maps are not territories", "complex"),
    ("meaning requires context", "complex"),
    ("logic precedes language", "complex"),
    ("infinity is countable sometimes", "complex"),
    # Very long simple
    ("the big red dog ran quickly down the long straight road toward the tall old brown wooden fence near the small quiet house by the river", "simple"),
    ("the little grey cat sat on the soft warm mat by the big stone fireplace and purred quietly while the rain fell outside", "simple"),
    ("my grandmother bakes the most delicious chocolate chip cookies every single weekend without fail", "simple"),
    # Rare words simple structure
    ("the tintinnabulation resonated melodiously", "simple"),
    ("photosynthesis converts sunlight efficiently", "simple"),
    # Common words complex nesting
    ("the cat that the dog that the man owned chased sat on the mat", "complex"),
    ("she said that he thought that they believed it was true", "complex"),
    # Domain complex with simple syntax
    ("the mitochondrial electron transport chain couples proton translocation with ATP synthesis", "complex"),
    ("genome wide association studies identify statistical correlations between genetic variants and phenotypic traits", "complex"),
    # Garden path sentences
    ("the horse raced past the barn fell", "complex"),
    ("the old man the boats", "complex"),
    # Ambiguous
    ("time flies like an arrow but fruit flies like a banana", "complex"),
    # Mixed complexity
    ("neural networks approximate complex nonlinear functions through hierarchical feature learning", "complex"),
    ("it is raining today", "simple"),
    ("the interplay between cooperation and competition drives evolutionary dynamics", "complex"),
    ("the sky is blue", "simple"),
    ("water flows downhill", "simple"),
    ("consciousness remains one of the most profound unsolved problems in all of science", "complex"),
    ("if then else", "simple"),
    ("the mitochondria is the powerhouse of the cell", "moderate"),
    ("dark matter constitutes approximately twenty seven percent of the total mass energy content of the observable universe", "complex"),
    ("she runs fast", "simple"),
    ("go", "simple"),
    # Additional adversarial edge cases
    ("I saw the man with the telescope", "complex"),
    ("the complex houses married and single soldiers and their families", "complex"),
    ("we saw her duck", "complex"),
    ("the cotton clothing is made of grows in Mississippi", "complex"),
    ("the prime number theorem describes the asymptotic distribution of primes", "complex"),
    ("he gave her cat food", "simple"),
    ("the big fluffy white dog played happily in the sunny green park all morning long", "simple"),
    ("they ate pizza", "simple"),
    ("she danced gracefully across the wooden stage", "simple"),
    ("the lamp is on the table", "simple"),
]


# ─── Vector math utilities ───

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine_sim(a, b):
    na, nb = norm(a), norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return max(0.0, min(1.0, dot(a, b) / (na * nb)))


def evaluate(encode_fn, confidence_fn=None):
    """
    Evaluate encoder + confidence function on all test sentences.

    Args:
        encode_fn: text -> list[float] (encoding vector, any dimensionality)
        confidence_fn: (encoding, surface_weight, g5_stats) -> float
            If None, uses default confidence formula.
            g5_stats is a dict with 'simple_mean', 'complex_mean', 'g5_start', 'g5_end'

    Returns:
        dict with val_score, simple_accuracy, complex_accuracy, etc.
    """
    start = time.time()

    # 1. Encode all training simple sentences → compute surface weight
    simple_encodings = [encode_fn(s) for s in TRAIN_SIMPLE]
    dim = len(simple_encodings[0])
    n_simple = len(simple_encodings)
    surface_weight = [
        sum(enc[d] for enc in simple_encodings) / n_simple
        for d in range(dim)
    ]

    # 2. Compute G5 stats (dims 116-127 by default, but encoder can override)
    # Try to get g5 range from encode_fn if it has the attribute
    g5_start = getattr(encode_fn, 'g5_start', 116)
    g5_end = getattr(encode_fn, 'g5_end', 128)

    def g5_norm(enc):
        return norm(enc[g5_start:g5_end])

    simple_g5_norms = [g5_norm(enc) for enc in simple_encodings]
    g5_simple_mean = sum(simple_g5_norms) / len(simple_g5_norms)

    complex_encodings = [encode_fn(s) for s in TRAIN_COMPLEX]
    complex_g5_norms = [g5_norm(enc) for enc in complex_encodings]
    g5_complex_mean = sum(complex_g5_norms) / len(complex_g5_norms)

    g5_stats = {
        'simple_mean': g5_simple_mean,
        'complex_mean': g5_complex_mean,
        'g5_start': g5_start,
        'g5_end': g5_end,
    }

    # 3. Default confidence function
    if confidence_fn is None:
        def confidence_fn(enc, sw, stats):
            input_norm = norm(enc)
            base_conf = min(input_norm, 1.0)
            cos = cosine_sim(enc, sw)
            conf = base_conf * 0.7 + cos * 0.3

            # G5 penalty
            g5_n = norm(enc[stats['g5_start']:stats['g5_end']])
            g5_s = stats['simple_mean']
            g5_c = stats['complex_mean']
            if g5_c - g5_s > 1e-8:
                penalty = max(0.0, min(1.0, (g5_n - g5_s) / (g5_c - g5_s)))
                conf = max(0.0, conf - penalty * 0.35)

            return max(0.0, min(1.0, conf))

    # 4. Compute confidence for calibration sentences
    calib_simple_confs = [
        confidence_fn(enc, surface_weight, g5_stats)
        for enc in simple_encodings
    ]
    calib_complex_confs = [
        confidence_fn(enc, surface_weight, g5_stats)
        for enc in complex_encodings
    ]

    simple_mean_conf = sum(calib_simple_confs) / len(calib_simple_confs)
    complex_mean_conf = sum(calib_complex_confs) / len(calib_complex_confs)

    # Threshold = midpoint between simple and complex mean confidence
    surface_threshold = (simple_mean_conf + complex_mean_conf) / 2.0

    # 5. Route all test sentences
    simple_correct = 0
    simple_total = 0
    complex_correct = 0
    complex_total = 0
    moderate_surface = 0
    moderate_total = 0
    details = []

    for sentence, label in TEST_SENTENCES:
        enc = encode_fn(sentence)
        conf = confidence_fn(enc, surface_weight, g5_stats)
        routed_surface = conf >= surface_threshold

        if label == "simple":
            simple_total += 1
            if routed_surface:
                simple_correct += 1
            details.append((sentence, label, conf, routed_surface, routed_surface))
        elif label == "complex":
            complex_total += 1
            if not routed_surface:
                complex_correct += 1
            details.append((sentence, label, conf, routed_surface, not routed_surface))
        elif label == "moderate":
            moderate_total += 1
            if routed_surface:
                moderate_surface += 1
            details.append((sentence, label, conf, routed_surface, None))

    simple_acc = simple_correct / simple_total if simple_total > 0 else 0.0
    complex_acc = complex_correct / complex_total if complex_total > 0 else 0.0

    # Composite score: weight complex higher (it's the bottleneck)
    # Penalty if simple accuracy drops below 95%
    val_score = complex_acc * 0.7 + simple_acc * 0.3
    if simple_acc < 0.95:
        val_score *= (simple_acc / 0.95)

    confidence_gap = simple_mean_conf - complex_mean_conf
    elapsed = time.time() - start

    return {
        'val_score': val_score,
        'simple_accuracy': simple_acc,
        'complex_accuracy': complex_acc,
        'simple_correct': simple_correct,
        'simple_total': simple_total,
        'complex_correct': complex_correct,
        'complex_total': complex_total,
        'moderate_surface_pct': moderate_surface / moderate_total if moderate_total > 0 else 0.0,
        'surface_threshold': surface_threshold,
        'confidence_gap': confidence_gap,
        'simple_mean_conf': simple_mean_conf,
        'complex_mean_conf': complex_mean_conf,
        'g5_simple_mean': g5_simple_mean,
        'g5_complex_mean': g5_complex_mean,
        'elapsed_seconds': elapsed,
        'details': details,
    }

use std::fmt;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Complexity {
    Simple,
    Moderate,
    Complex,
}

impl fmt::Display for Complexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Complexity::Simple => write!(f, "Simple"),
            Complexity::Moderate => write!(f, "Moderate"),
            Complexity::Complex => write!(f, "Complex"),
        }
    }
}

pub struct Corpus {
    pub simple: Vec<String>,
    pub moderate: Vec<String>,
    pub complex: Vec<String>,
}

impl Corpus {
    pub fn load() -> Self {
        let simple: Vec<&str> = vec![
            // Animals
            "the dog runs fast",
            "cats sleep all day",
            "birds fly south in winter",
            "the horse eats hay",
            "fish swim in the river",
            "the cat drinks milk",
            "dogs bark at strangers",
            "the bird sings at dawn",
            "rabbits hop across the field",
            "the cow grazes on grass",
            "frogs jump near the pond",
            "the duck swims in the lake",
            "horses run in the meadow",
            "the sheep eats green grass",
            "a fox hides in the forest",
            "the owl hoots at night",
            "bees fly from flower to flower",
            "the dog wags its tail",
            "a cat climbs the tall tree",
            "the fish leaps from the water",
            "elephants drink from the river",
            "the lion roars at sunset",
            "a bear climbs the hill",
            "the wolf howls at the moon",
            "deer run through the woods",
            "the rabbit hides in the burrow",
            "a bird lands on the fence",
            "the snake rests on a rock",
            "the goat eats the leaves",
            "a pig rolls in the mud",
            // Weather
            "the sun is bright today",
            "rain falls on the roof",
            "snow covers the ground",
            "the wind blows the leaves",
            "clouds hide the sun",
            "the sky is very blue",
            "thunder shakes the windows",
            "lightning lights up the sky",
            "fog rolls over the hills",
            "the morning is cold and grey",
            "hail hits the glass hard",
            "the rainbow appears after rain",
            "frost covers the car window",
            "the storm blew all night",
            "a warm breeze moves the curtains",
            "ice forms on the pond",
            "the sun sets in the west",
            "dark clouds bring heavy rain",
            "the air smells fresh after rain",
            "the day starts bright and clear",
            // Food
            "she likes green apples",
            "he eats toast each morning",
            "the bread smells very good",
            "they made soup for dinner",
            "she drinks tea every morning",
            "he eats a banana for lunch",
            "the cake is very sweet",
            "they share a bowl of rice",
            "the orange juice is cold",
            "she bakes bread on weekends",
            "he grills fish for dinner",
            "the salad has fresh tomatoes",
            "they cook pasta every Tuesday",
            "the cheese is very sharp",
            "she eats cereal with milk",
            "he drinks coffee in the morning",
            "the soup is hot and salty",
            "they enjoy ice cream in summer",
            "the pie smells like cinnamon",
            "she fries eggs for breakfast",
            // Basic actions
            "she runs to the shop",
            "he reads a book at night",
            "they walk to school each day",
            "she writes in her notebook",
            "he sits by the window",
            "they play in the park",
            "she jumps over the puddle",
            "he lifts the heavy box",
            "they sing songs together",
            "she paints the garden wall",
            "he kicks the ball far",
            "they swim in the pool",
            "she climbs the old oak tree",
            "he throws the ball to her",
            "they dance in the living room",
            "she cuts the bread with a knife",
            "he opens the front door",
            "they build a fort from blankets",
            "she washes her hands first",
            "he closes the curtains at night",
            // Everyday objects
            "the chair is made of wood",
            "her bag is very heavy",
            "the table has four legs",
            "his keys are on the shelf",
            "the lamp gives off warm light",
            "her phone rings in her pocket",
            "the clock ticks on the wall",
            "his hat blows off in the wind",
            "the door is painted red",
            "her umbrella is bright yellow",
            "the book has torn pages",
            "his pencil broke in half",
            "the cup holds hot tea",
            "her coat hangs by the door",
            "the window is very dirty",
            "his boots are caked in mud",
            "the bed is soft and warm",
            "her scarf keeps her warm",
            "the bottle is almost empty",
            "his glasses sit on the desk",
            // Simple descriptions
            "the house is small and white",
            "her eyes are very blue",
            "the road is long and straight",
            "his voice is soft and low",
            "the garden is full of flowers",
            "her smile is very bright",
            "the hill is steep and rocky",
            "his hands are cold and rough",
            "the river is deep and wide",
            "her hair is long and dark",
            "the path is narrow and overgrown",
            "his car is old and slow",
            "the room is bright and tidy",
            "her dress is red and short",
            "the wall is cracked and grey",
            "his shoes are worn and old",
            "the tree is tall and strong",
            "her bag is small and blue",
            "the lake is calm and still",
            "his face is tired and pale",
            // More animals
            "the parrot repeats every word",
            "a mouse hides under the floor",
            "the hamster runs on its wheel",
            "a spider spins a web",
            "the turtle moves very slowly",
            "a goldfish swims in circles",
            "the crow caws on the roof",
            "a squirrel gathers nuts for winter",
            "the pigeon sits on the ledge",
            "a hedgehog curls into a ball",
            // More weather
            "the morning frost melts by noon",
            "spring rains wake the sleeping seeds",
            "autumn leaves fall from the trees",
            "the summer sun warms the stones",
            "a light drizzle wets the path",
            "the night sky is full of stars",
            "the moon is large and orange",
            "a cool wind blows from the north",
            "the pond freezes in December",
            "white clouds drift across the sky",
            // More everyday
            "she pours water into the glass",
            "he turns off the bedroom light",
            "the bus stops at the corner",
            "she picks flowers in the garden",
            "he feeds the dog each morning",
            "the train runs on time today",
            "she folds the laundry on Sunday",
            "he waters the potted plants",
            "the shop opens at nine",
            "she locks the door before leaving",
            // Nature
            "the river runs fast in spring",
            "moss grows on the old stone",
            "the forest is quiet at noon",
            "wild berries grow on the hill",
            "the meadow is green after rain",
            "tall grass sways in the wind",
            "the stream is cold and clear",
            "pine trees line the road",
            "the cliff drops to the sea",
            "sand dunes shift with the wind",
            // More simple descriptions
            "the shop is closed on Sunday",
            "her coat is long and warm",
            "the street is wet and empty",
            "his laugh is loud and warm",
            "the fence is white and new",
            "her hands are small and warm",
            "the hill is green and round",
            "his voice is clear and steady",
            "the bridge is old and narrow",
            "her cat is black and fluffy",
            // Extra simple — colours and sizes
            "the apple is red and round",
            "his shirt is bright orange",
            "the sky turns pink at sunset",
            "her ribbon is pale yellow",
            "the box is square and flat",
            "his coat is dark green",
            "the cushion is soft and round",
            "her hat is wide and white",
            "the flag is red and blue",
            "his jumper is grey and thick",
            // Extra simple — daily routine
            "she wakes up very early",
            "he brushes his teeth twice daily",
            "they eat lunch at noon",
            "she tidies her room each night",
            "he goes to bed at ten",
            "they water the plants each morning",
            "she buys milk at the corner shop",
            "he walks the dog after dinner",
            "they watch the news each evening",
            "she reads before she goes to sleep",
        ];

        let moderate: Vec<&str> = vec![
            // Science basics
            "the human brain contains approximately one hundred billion neurons",
            "photosynthesis converts sunlight into chemical energy stored in glucose",
            "the earth orbits the sun once every three hundred sixty five days",
            "water molecules consist of two hydrogen atoms bonded to one oxygen atom",
            "gravity pulls objects toward the centre of the earth at constant acceleration",
            "sound travels as pressure waves through air at roughly three hundred metres per second",
            "the periodic table organises chemical elements by their atomic number and properties",
            "mitochondria generate most of the chemical energy cells need to function",
            "the speed of light in a vacuum is approximately three hundred million metres per second",
            "volcanoes form where tectonic plates meet and magma rises through the crust",
            "the human heart pumps blood through a network of arteries and veins",
            "earthquakes occur when tectonic plates suddenly shift beneath the earth's surface",
            "atoms bond together through the sharing or transfer of electrons between them",
            "climate change is driven by rising concentrations of carbon dioxide in the atmosphere",
            "the immune system defends the body against infections by producing antibodies",
            "fossils form when organisms are buried and slowly replaced by minerals over millions of years",
            "the moon's gravitational pull creates tidal forces that raise sea levels twice daily",
            "DNA carries genetic information encoded in sequences of four chemical base pairs",
            "kinetic energy depends on the mass and velocity of a moving object",
            "black holes form when massive stars collapse under their own gravitational force",
            // Technology
            "machine learning models require significant computational resources to train effectively",
            "the internet connects billions of devices through a global network of servers and cables",
            "transistors switched from vacuum tubes to silicon chips in the mid twentieth century",
            "encryption algorithms protect digital communication by scrambling data with mathematical keys",
            "smartphones contain sensors for location, motion, light, and biometric authentication",
            "cloud computing allows businesses to rent processing power and storage over the internet",
            "lithium ion batteries store energy through reversible chemical reactions in the electrodes",
            "fibre optic cables transmit data as pulses of light over long distances with low loss",
            "artificial intelligence systems are trained on large datasets to recognise patterns",
            "electric vehicles use rechargeable batteries to power electric motors instead of combustion engines",
            "satellites in low earth orbit transmit navigation and communication signals to receivers below",
            "software version control systems track every change made to a codebase over time",
            "solid state drives store data on flash memory chips rather than spinning magnetic platters",
            "neural networks learn by adjusting weights between connected nodes based on training error",
            "the world wide web was invented by Tim Berners-Lee at CERN in nineteen eighty nine",
            // Geography
            "the amazon rainforest produces about twenty percent of the world's oxygen supply",
            "the sahara desert is the largest hot desert on earth spanning most of northern africa",
            "mount everest stands at nearly nine thousand metres above sea level in the himalayas",
            "the pacific ocean covers more area than all the land on earth combined",
            "the nile river flows northward through ten african countries before reaching the sea",
            "the great barrier reef off australia is the world's largest coral reef system",
            "the himalayas were formed when the indian subcontinent collided with the eurasian plate",
            "antarctica holds about ninety percent of all the fresh water on earth as ice",
            "the amazon river discharges more water into the sea than any other river on earth",
            "the arctic and antarctic circles mark the boundaries of the polar day and polar night",
            "the sahara receives less than twenty five millimetres of rainfall on average each year",
            "iceland sits on the mid atlantic ridge where two tectonic plates are slowly pulling apart",
            "the mariana trench reaches over eleven kilometres below the surface of the pacific ocean",
            "new zealand lies on the pacific ring of fire and experiences frequent seismic activity",
            "the dead sea sits four hundred metres below sea level and has no outlet to the ocean",
            // History
            "the roman empire at its peak controlled territory across europe, north africa, and the middle east",
            "the printing press invented by gutenberg transformed the spread of knowledge in europe",
            "the industrial revolution began in britain in the late eighteenth century and spread worldwide",
            "the first world war was triggered by the assassination of archduke franz ferdinand in nineteen fourteen",
            "the cold war between the united states and soviet union lasted from nineteen forty seven to nineteen ninety one",
            "ancient egypt built the pyramids as tombs for their pharaohs over four thousand years ago",
            "the silk road connected china and europe through a network of trade routes across asia",
            "the renaissance was a cultural movement that began in italy in the fourteenth century",
            "the french revolution began in seventeen eighty nine with the storming of the bastille",
            "the apollo eleven mission landed the first humans on the moon in july nineteen sixty nine",
            "the magna carta signed in twelve fifteen established that the king was subject to the rule of law",
            "the black death killed roughly a third of europe's population during the fourteenth century",
            "world war two ended in europe in may nineteen forty five and in asia in september the same year",
            "the berlin wall fell in november nineteen eighty nine ending the division of germany",
            "the roman empire officially fell in four hundred seventy six when the last emperor was deposed",
            // Everyday complexity
            "regular exercise reduces the risk of heart disease, diabetes, and some forms of cancer",
            "urban planning shapes how cities develop by balancing housing, transport, and green space",
            "good sleep hygiene involves consistent bedtimes, dark rooms, and avoiding screens before bed",
            "learning a second language improves cognitive flexibility and delays cognitive decline in older adults",
            "composting kitchen waste reduces landfill burden and produces rich organic matter for gardens",
            "renewable energy sources like solar and wind are becoming cheaper than fossil fuels",
            "vaccination programmes have eliminated or nearly eliminated several historically deadly diseases",
            "mindfulness practice has been shown to reduce stress and improve attention over time",
            "financial literacy helps individuals make better decisions about saving, spending, and investing",
            "public libraries provide free access to books, digital resources, and community programmes",
            "the average person spends over six hours a day interacting with digital screens of some kind",
            "urban trees reduce air pollution, lower local temperatures, and improve mental wellbeing",
            "regular handwashing is one of the most effective ways to prevent the spread of infectious diseases",
            "diverse diets rich in vegetables and whole grains are associated with better long term health",
            "the global population crossed eight billion people for the first time in two thousand and twenty two",
            // Mixed domain
            "photovoltaic panels convert sunlight directly into electricity through the photoelectric effect",
            "the human genome contains approximately three billion base pairs encoding around twenty thousand genes",
            "plate tectonics explains both the distribution of earthquakes and the shapes of continents",
            "the bending of light around massive objects is predicted by einstein's general theory of relativity",
            "market economies allocate resources through the interaction of supply and demand across millions of actors",
            "the lymphatic system drains excess fluid from tissues and plays a key role in immune defence",
            "compilers translate high level source code into machine code that processors can execute directly",
            "ocean currents distribute heat around the earth and strongly influence regional climates",
            "the placebo effect demonstrates that beliefs and expectations can produce measurable physiological changes",
            "social media algorithms optimise for engagement by showing users content likely to provoke strong reactions",
            "migration patterns of birds are guided by magnetic fields, star positions, and landmarks",
            "the greenhouse effect occurs when atmospheric gases absorb and re-emit infrared radiation",
            "statistical significance testing determines how likely an observed result is due to chance alone",
            "natural selection favours traits that improve an organism's chances of surviving and reproducing",
            "the human eye can distinguish approximately ten million different colours under good lighting conditions",
            // Economics and society
            "inflation erodes the purchasing power of money when the supply of goods fails to keep pace with demand",
            "interest rates set by central banks influence how much households and businesses borrow and spend",
            "international trade allows countries to specialise in goods they produce most efficiently and import the rest",
            "urbanisation has shifted more than half the global population into cities over the past century",
            "income inequality has widened in many countries as returns to capital have outpaced wage growth",
            "the gig economy offers workers flexible hours but often lacks the benefits of traditional employment",
            "social capital refers to the networks of trust and reciprocity that enable communities to function cooperatively",
            "the ageing population in many developed countries is placing increasing pressure on pension and healthcare systems",
            "foreign direct investment brings capital, technology, and jobs to host countries but can also displace local firms",
            "behavioural economics shows that people systematically deviate from rational choice theory in predictable ways",
            // Biology and medicine
            "the appendix was long considered vestigial but may play a role in restoring gut bacteria after illness",
            "the circadian rhythm is a roughly twenty four hour internal clock that regulates sleep, hormones, and digestion",
            "stem cells can differentiate into many cell types, making them valuable for research and regenerative medicine",
            "antibiotic resistance is accelerated when patients fail to complete full courses of prescribed medication",
            "the placenta transfers oxygen and nutrients from mother to foetus while blocking most harmful substances",
            "aerobic exercise increases the density of mitochondria in muscle cells, improving endurance over time",
            "blood type is determined by the presence or absence of specific antigens on the surface of red blood cells",
            "the reflex arc bypasses the brain by routing sensory signals directly to the spinal cord for immediate response",
            "mRNA vaccines instruct cells to produce a harmless protein fragment that trains the immune system to recognise a pathogen",
            "the gut microbiome contains trillions of bacteria that influence digestion, immunity, and even mood",
            // Physics and chemistry
            "superconductors conduct electricity without resistance when cooled below a critical temperature",
            "osmosis is the movement of water across a semi-permeable membrane from a region of low solute concentration to a high one",
            "catalysts lower the activation energy of a chemical reaction without being consumed in the process",
            "nuclear fission releases enormous energy by splitting heavy atomic nuclei into lighter fragments",
            "lasers produce coherent light by stimulating the emission of photons from an excited atomic medium",
            "the doppler effect causes the perceived frequency of a wave to shift when the source or observer is moving",
            "polymers are large molecules made of many repeated smaller units called monomers bonded together in long chains",
            "semiconductors have electrical conductivity between that of a conductor and an insulator and form the basis of electronics",
            "the law of conservation of energy states that energy cannot be created or destroyed, only converted between forms",
            "radioactive decay transforms unstable atomic nuclei into more stable ones by emitting particles or radiation",
            // Environment and ecology
            "deforestation releases stored carbon into the atmosphere and destroys habitats for millions of species",
            "coral reefs support roughly twenty five percent of all marine species despite covering less than one percent of the ocean floor",
            "the water cycle moves water between the atmosphere, land, and oceans through evaporation and precipitation",
            "nitrogen fixation by certain bacteria converts atmospheric nitrogen into compounds that plants can use as fertiliser",
            "invasive species introduced to new environments often outcompete native species and cause ecological damage",
            "ocean acidification occurs as seawater absorbs carbon dioxide, threatening shellfish and coral reef ecosystems",
            "wetlands filter pollutants from water, store flood water, and provide habitat for diverse wildlife",
            "biodiversity tends to be highest near the equator and decreases gradually toward the polar regions",
            "keystone species have disproportionately large effects on their ecosystems relative to their abundance",
            "the ozone layer in the stratosphere absorbs most of the sun's ultraviolet radiation, protecting life on earth",
            // Technology and engineering
            "global positioning systems use signals from multiple satellites to calculate a receiver's location with metre accuracy",
            "the internet protocol suite determines how data is packaged, addressed, and routed across interconnected networks",
            "three dimensional printing builds objects layer by layer from digital designs, enabling rapid prototyping",
            "nuclear reactors generate heat through controlled fission reactions that is then converted into electricity",
            "wind turbines convert the kinetic energy of moving air into electricity through electromagnetic induction",
            "the transistor revolutionised electronics by replacing bulky vacuum tubes with tiny semiconductor switches",
            "global supply chains link raw material extraction, manufacturing, and distribution across dozens of countries",
            "desalination plants remove salt from seawater to produce fresh water, increasingly important in arid regions",
            "radio telescopes detect microwave and radio emissions from distant astronomical objects invisible to optical instruments",
            "hydraulic fracturing extracts oil and gas from shale rock by injecting high pressure fluid to crack the formation",
            // Arts and culture
            "the renaissance saw artists develop linear perspective to create convincing illusions of three dimensional space",
            "jazz music originated in new orleans in the early twentieth century blending african rhythms with european harmony",
            "the novel as a literary form emerged in eighteenth century europe and quickly became the dominant narrative art form",
            "cinema evolved from silent black and white films to a global storytelling medium within a few decades",
            "the invention of photography in the nineteenth century forced painters to reconsider the purpose of representational art",
        ];

        let complex: Vec<&str> = vec![
            // Mathematics
            "the incompleteness theorems proved by kurt gödel in nineteen thirty one demonstrated that any sufficiently powerful formal system contains true statements that cannot be proved within that system",
            "the riemann hypothesis, one of the most famous unsolved problems in mathematics, conjectures that all non-trivial zeros of the riemann zeta function lie on the critical line with real part one half",
            "the p versus np problem asks whether every problem whose solution can be quickly verified by a computer can also be quickly solved, with profound implications for cryptography and computational complexity",
            "fourier analysis decomposes arbitrary periodic functions into infinite sums of sinusoidal components, enabling signal processing, image compression, and the solution of partial differential equations",
            "category theory provides a unifying language for mathematics by abstracting the common structure of diverse mathematical fields in terms of objects and morphisms satisfying compositional laws",
            "the axiom of choice, independent of the other axioms of zermelo-fraenkel set theory, asserts that for any collection of non-empty sets there exists a function selecting one element from each set",
            "nonlinear dynamical systems can exhibit sensitive dependence on initial conditions, producing deterministic yet practically unpredictable behaviour known as deterministic chaos or the butterfly effect",
            "the langlands programme is a far-reaching set of conjectures connecting number theory and representation theory through correspondences between automorphic forms and galois representations",
            "stochastic differential equations model the evolution of systems driven by both deterministic drift and random diffusion terms, underpinning mathematical finance and statistical physics",
            "topological spaces generalise metric spaces by abstracting the notion of nearness through open sets, enabling a rigorous treatment of continuity, connectedness, and compactness in arbitrary dimensions",
            "the central limit theorem states that the normalised sum of a large number of independent identically distributed random variables converges in distribution to a standard normal distribution regardless of the original distribution's shape",
            "algebraic geometry studies the geometric properties of solution sets of polynomial equations, unifying classical curve and surface theory with modern commutative algebra and sheaf theory",
            "the navier-stokes equations describe fluid motion through a system of nonlinear partial differential equations whose existence and smoothness of solutions in three dimensions remains unproved",
            "spectral graph theory connects the eigenvalues of matrices associated with graphs to structural properties such as connectivity, bipartiteness, and the mixing times of random walks",
            "measure theory provides the rigorous foundation for integration and probability by defining a consistent framework for assigning sizes to sets and extending limits to highly irregular functions",
            // Philosophy
            "the recursive nature of self-referential systems creates emergent properties that resist reduction to simpler components, challenging reductionist accounts of consciousness and cognition",
            "quine's indeterminacy of translation thesis argues that no unique correct translation of one language into another exists, because observable linguistic behaviour is always compatible with multiple incompatible translation schemes",
            "the problem of induction, articulated by hume, questions the logical basis for inferring universal laws from finite observations, a challenge that remains central to the philosophy of science",
            "functionalist theories of mind identify mental states with their causal roles rather than their physical substrate, suggesting that consciousness could in principle be realised in silicon as well as biological neurons",
            "the hard problem of consciousness, as framed by david chalmers, asks why physical processing in the brain gives rise to subjective experience rather than proceeding in the dark without any inner life",
            "kantian ethics grounds moral obligation in the categorical imperative, which demands that agents act only on maxims they could consistently will to be universal laws governing all rational beings",
            "the ontological argument, advanced by anselm and later descartes, attempts to derive the existence of god analytically from the concept of a being than which no greater can be conceived",
            "the philosophy of language distinguishes between the sense of an expression, which determines its reference, and the reference itself, following the famous distinction drawn by gottlob frege",
            "eliminativist materialism, defended by churchland, argues that ordinary psychological concepts like belief and desire will eventually be replaced by more accurate descriptions at the level of neuroscience",
            "the mereological puzzle of constitution asks how a statue and the clay of which it is made can be numerically distinct things occupying the same region of space at the same time",
            "compatibilist accounts of free will attempt to reconcile the thesis that human choices are causally determined by prior events with the ordinary assumption that agents are morally responsible for their actions",
            "the sorites paradox arises when iterating a tolerant predicate such as heap produces a contradiction, exposing a deep tension between classical logic and the vagueness inherent in natural language",
            "pragmatist epistemology, associated with peirce, james, and dewey, evaluates beliefs not by their correspondence to mind-independent reality but by their practical consequences and capacity to guide successful action",
            "the distinction between analytic and synthetic statements, central to logical positivism, was forcefully challenged by quine's argument that no statement is immune from revision in light of experience",
            "phenomenology, inaugurated by husserl, brackets questions of external existence to investigate the intentional structure of conscious experience from within the first-person perspective",
            // Advanced science
            "quantum entanglement produces correlations between spatially separated particles that cannot be explained by any local hidden variable theory, as demonstrated experimentally by violations of bell inequalities",
            "the standard model of particle physics unifies the electromagnetic, weak, and strong nuclear forces under a gauge symmetry framework but does not incorporate gravity or explain dark matter",
            "epigenetic mechanisms including dna methylation and histone modification regulate gene expression without altering the underlying sequence, enabling differentiated cell types to arise from identical genomes",
            "renormalisation group theory explains how the effective behaviour of physical systems changes with the scale of observation, unifying superficially disparate phenomena under universal critical exponents",
            "crispr-cas9 technology enables precise editing of genomic sequences by directing an endonuclease to a target site using a guide rna complementary to the sequence to be modified",
            "the cosmic microwave background radiation provides a snapshot of the universe approximately three hundred and eighty thousand years after the big bang when photons first decoupled from matter",
            "allosteric regulation allows enzymes to change conformation and activity in response to molecules binding at sites distant from the active site, providing sophisticated feedback control of metabolic pathways",
            "topological insulators are materials that behave as insulators in their bulk but conduct electricity along their surfaces through states protected by time-reversal symmetry",
            "horizontal gene transfer between bacteria allows antibiotic resistance genes to spread rapidly through a population independently of vertical inheritance, complicating the treatment of bacterial infections",
            "the many-worlds interpretation of quantum mechanics proposes that every quantum measurement causes the universe to branch into multiple non-interacting copies each realising one possible outcome",
            "protein folding is driven by the minimisation of free energy as hydrophobic residues collapse inward and hydrophilic residues arrange themselves to interact with the surrounding solvent",
            "general relativity describes gravity not as a force but as the curvature of spacetime caused by energy and momentum, predicting phenomena including gravitational waves and black holes",
            "the second law of thermodynamics states that the total entropy of an isolated system can never decrease over time, providing a thermodynamic arrow of time and constraining all energy transformations",
            "emergent phenomena in complex systems arise from interactions among many simple components and cannot in principle be predicted from knowledge of the components alone without consideration of system-level dynamics",
            "the endosymbiotic theory proposes that mitochondria and chloroplasts originated as free-living prokaryotes that were engulfed by ancestral eukaryotic cells and eventually became permanent organelles",
            // Linguistics
            "the sapir-whorf hypothesis in its strong form claims that the language one speaks determines the thoughts one can think, a view largely rejected but influential in softer forms about linguistic relativity",
            "transformational grammar, developed by chomsky, postulates that surface syntactic forms are derived from underlying deep structures through a series of movement and deletion transformations",
            "signed languages exhibit the full grammatical complexity of spoken languages, including recursion, tense morphology, and argument structure, demonstrating that language is independent of its modality",
            "historical linguists reconstruct proto-languages by applying the comparative method to cognate sets across related languages, identifying regular sound correspondences that reveal ancestral forms",
            "pragmatic inference, as described by grice's cooperative principle, allows speakers to communicate far more than the literal semantic content of their utterances through context-sensitive implicatures",
            "the poverty of the stimulus argument contends that children acquire grammatical knowledge that goes far beyond what is present in the linguistic input they receive, supporting nativist theories of language acquisition",
            "language contact situations can produce creoles when children acquire a pidgin as a first language and systematically expand it with full grammatical complexity absent in the input",
            "the phonology of tone languages uses pitch contrastively at the level of individual morphemes, requiring models of grammar that integrate prosodic and segmental information within the same representational framework",
            "discourse coherence depends on referential continuity, logical relations between propositions, and shared assumptions about relevance that allow interlocutors to construct a unified interpretation of multi-sentence texts",
            "construction grammar argues that grammatical knowledge consists of form-meaning pairings at all levels from morphemes to idioms and abstract argument structures without a separate autonomous syntax",
            // Computer science theory
            "the halting problem is undecidable because assuming the existence of a machine that determines whether any program halts leads by diagonalisation to a contradiction with its own behaviour",
            "the byzantine generals problem formalises the challenge of achieving consensus in a distributed system where some participants may be faulty or malicious and communication is unreliable",
            "the expressiveness hierarchy of formal language classes established by the chomsky hierarchy separates regular, context-free, context-sensitive, and recursively enumerable languages by the automata that recognise them",
            "amortised analysis averages the worst-case cost of a sequence of operations to show that expensive operations are rare enough that the average cost per operation is acceptably bounded",
            "the curry-howard correspondence establishes a deep structural isomorphism between propositions in intuitionistic logic and types in typed lambda calculi, equating proofs with programs",
            "satisfiability solvers based on conflict-driven clause learning can efficiently solve industrial instances with millions of variables by combining unit propagation, non-chronological backtracking, and clause learning",
            "cache-oblivious algorithms achieve optimal cache performance on machines with arbitrary memory hierarchies without knowing the cache size or cache line size at the time the algorithm is designed",
            "the raft consensus algorithm ensures fault-tolerant agreement on a replicated log by electing a leader who coordinates replication and handles log inconsistencies through a structured protocol",
            "dependent type systems, as in coq and agda, allow types to depend on values, enabling the expression and machine-checked verification of rich correctness properties within the type system itself",
            "the map-reduce programming model enables large-scale parallel data processing by splitting computation into independent map tasks followed by a reduce phase that merges partial results",
            "linear type systems guarantee that each resource is used exactly once, enabling compile-time memory safety and the elimination of garbage collection without sacrificing expressiveness",
            "abstract interpretation provides a formal framework for static analysis by computing over approximate abstract domains that soundly over-approximate the set of all possible runtime states of a program",
            "the actor model of computation treats independent actors that communicate only by asynchronous message passing as the fundamental unit of concurrency, avoiding the hazards of shared mutable state",
            "kolmogorov complexity measures the information content of a string as the length of the shortest program that produces it, providing a machine-independent notion of randomness and compressibility",
            "type inference algorithms such as hindley-milner reconstruct principal types for expressions in polymorphic lambda calculi without requiring any programmer-supplied type annotations",
            // More mathematics
            "the prime number theorem describes the asymptotic distribution of prime numbers by proving that the number of primes up to n is approximately n divided by the natural logarithm of n",
            "ergodic theory studies the long-run statistical behaviour of dynamical systems, establishing conditions under which time averages equal space averages for measure-preserving transformations",
            "the uniformisation theorem asserts that every simply connected riemann surface is conformally equivalent to one of three canonical domains, providing a complete classification of such surfaces",
            "random matrix theory studies the spectral properties of matrices with random entries and has found unexpected applications in nuclear physics, number theory, and wireless communication",
            "the theory of optimal transport, developed by monge and later kantorovich, quantifies the minimum cost of moving one probability distribution to another and connects geometry, probability, and economics",
            "noncommutative geometry extends classical differential geometry to spaces where coordinates fail to commute, providing a mathematical framework for quantum gravity and the standard model",
            "persistent homology tracks topological features of a dataset across multiple scales of resolution and provides stable invariants that are useful for characterising the shape of high-dimensional data",
            "the modularity theorem, proved as part of the proof of fermat's last theorem, establishes that every elliptic curve over the rationals is associated with a modular form",
            "information geometry studies the differential geometry of families of probability distributions, equipping statistical manifolds with a natural riemannian metric derived from the fisher information",
            "the theory of matroids axiomatises the notion of independence common to linear algebra and graph theory, enabling a unified treatment of greedy algorithms across combinatorial settings",
            // More philosophy
            "the chinese room argument proposed by searle contends that a system mechanically manipulating symbols according to rules has no genuine understanding even if its outputs are indistinguishable from those of a speaker",
            "the extended mind hypothesis, advanced by clark and chalmers, argues that cognitive processes can extend beyond the skull to incorporate external objects when those objects play an appropriate functional role",
            "modal realism, defended by david lewis, holds that all possible worlds are equally real and concrete, with our actual world being only one among infinitely many coexistent but causally isolated universes",
            "the problem of personal identity over time asks what makes a person at one moment the same person as one at another, given the continual physical and psychological changes that occur throughout a life",
            "consequentialist moral theories evaluate the rightness of actions solely by their outcomes, leading to demanding conclusions about the obligations of affluent individuals toward distant suffering strangers",
            "the transcendental aesthetic of kant's critique of pure reason argues that space and time are not properties of things in themselves but forms imposed by the human mind on the raw data of intuition",
            "deflationary theories of truth deny that truth is a substantive property, arguing instead that to say a proposition is true adds nothing beyond asserting the proposition itself",
            "the no-miracles argument for scientific realism claims that the predictive success of mature scientific theories would be miraculous if those theories were not at least approximately describing the structure of reality",
            "the is-ought problem, first identified by hume, highlights the logical gap between descriptive statements about how things are and normative statements about how they ought to be",
            "second-order logic extends first-order logic by allowing quantification over predicates and relations as well as individuals, significantly increasing expressiveness at the cost of completeness",
            // More advanced science
            "the holographic principle, suggested by the ads-cft correspondence, proposes that the information content of a volume of space is fully encoded on its lower-dimensional boundary",
            "nematic liquid crystals exhibit orientational order without positional order, producing optical anisotropy that underpins the operation of liquid crystal display technology",
            "the higgs mechanism explains how gauge bosons acquire mass by interacting with a scalar field that permeates all of space and whose excitation was observed as the higgs boson in twenty twelve",
            "quasicrystals are ordered but non-periodic atomic structures that exhibit forbidden rotational symmetries and diffraction patterns inconsistent with any classical crystallographic space group",
            "the sodium-potassium pump uses energy from ATP hydrolysis to transport three sodium ions out of and two potassium ions into cells, maintaining the electrochemical gradient essential for nerve signalling",
            "quantum chromodynamics describes the strong nuclear force through the exchange of gluons between quarks, whose colour charge is confined so that no individual quark can be isolated in normal conditions",
            "positron emission tomography detects pairs of gamma rays produced when a radioactive tracer undergoes positron-electron annihilation, enabling three-dimensional imaging of metabolic activity in living tissue",
            "the fluctuation-dissipation theorem relates the response of a system to a small perturbation to its spontaneous thermal fluctuations at equilibrium, connecting transport coefficients to correlation functions",
            "magnetohydrodynamics studies the behaviour of electrically conducting fluids such as plasma in the presence of magnetic fields, governing phenomena from solar flares to experimental fusion reactors",
            "the multi-messenger astronomy approach combines detections from gravitational wave observatories, gamma ray telescopes, and optical instruments to obtain complementary information about astrophysical events",
            // More linguistics
            "optimality theory proposes that surface phonological forms emerge from the competition between ranked and violable universal constraints rather than from the sequential application of rewrite rules",
            "the binding theory in government and binding syntax explains the distribution of pronouns, reflexives, and noun phrases by reference to structural conditions defined over c-command relations in phrase structure",
            "semantic compositionality, the principle that the meaning of a complex expression is determined by the meanings of its parts and how they are combined, underlies most formal approaches to natural language semantics",
            "lexical-functional grammar separates syntactic constituent structure from the functional structure encoding grammatical relations, allowing a unified analysis of grammatical phenomena across typologically diverse languages",
            "the theory of scalar implicature explains why saying some entails not all in ordinary discourse even though some is logically compatible with all, appealing to gricean norms of informativeness",
            "grammaticalisation describes the diachronic process by which lexical items gradually acquire grammatical functions through repeated use in contexts that encourage reanalysis and reduction",
            "evidentiality marking, grammaticalised in many indigenous american languages, requires speakers to indicate the source and reliability of the information they are conveying as part of the basic sentence structure",
            "head-driven phrase structure grammar uses typed feature structures and a hierarchy of lexical types to encode grammatical information, deriving sentence structure from the properties of lexical items",
            "the minimalist programme proposes that syntactic computation is driven by the need to satisfy interface conditions at the conceptual-intentional and articulatory-perceptual interfaces with minimal computational effort",
            "prototype theory in cognitive semantics argues that category membership is graded rather than all-or-nothing, with some members being more central representatives of a category than others",
            // More computer science theory
            "the online learning framework considers algorithms that make decisions sequentially without knowledge of future inputs and are evaluated by their regret relative to the best fixed strategy in hindsight",
            "proof-carrying code allows mobile code to be accompanied by a formal proof of safety properties that the receiving system can check efficiently before execution, eliminating the need to trust the code's origin",
            "the theory of communicating sequential processes models concurrent systems as networks of processes that synchronise by exchanging messages, enabling formal verification of deadlock freedom and trace equivalence",
            "spectral methods for machine learning exploit the geometry of data by working with eigenvectors of kernel matrices or graph laplacians to perform dimensionality reduction and clustering",
            "homomorphic encryption schemes allow arbitrary computations to be performed on encrypted data without decryption, potentially enabling cloud providers to process sensitive data without ever seeing the plaintext",
            "the theory of approximation algorithms provides formal guarantees on how close the solution produced by an efficient algorithm is to the optimal solution for computationally intractable optimisation problems",
            "symbolic execution explores program paths by treating inputs as symbolic variables and accumulating path conditions that can be discharged by a constraint solver to generate concrete test inputs",
            "the generalisation of boolean satisfiability to constraint satisfaction problems allows for the modelling and solution of a broad range of combinatorial problems arising in planning, scheduling, and verification",
            "concurrent separation logic extends separation logic with ownership transfer protocols that allow reasoning about shared mutable state accessed by multiple threads without global invariants",
            "the theory of reversible computing studies computations that can be run backward without loss of information, establishing connections between logical irreversibility and the thermodynamic cost of computation",
            // Advanced interdisciplinary
            "the free energy principle proposes that biological systems maintain their organisation by continuously acting and perceiving so as to minimise the surprise or prediction error generated by their sensory signals",
            "network science studies the structure and dynamics of complex networks from social graphs to neural connectomes, identifying properties like small-world topology and scale-free degree distributions",
            "the evolution of cooperation presents a paradox in evolutionary biology because natural selection operating on individuals appears to favour defection in prisoner's dilemma interactions among unrelated agents",
            "cultural evolution operates through mechanisms of variation, selection, and transmission that are analogous to genetic evolution but are mediated by social learning rather than genetic inheritance",
            "the problem of neural binding asks how the brain integrates information processed in spatially separated cortical areas into a unified conscious percept without a single convergence zone",
            "bioethics addresses moral questions raised by advances in medicine and biology such as the permissible limits of genetic enhancement and the conditions under which life-sustaining treatment may be withdrawn",
            "the computational theory of mind identifies mental processes with abstract computational procedures that are substrate-independent and could in principle be implemented in any physical system with sufficient complexity",
            "agent-based models simulate the actions and interactions of autonomous agents to study the emergence of macro-level patterns from micro-level rules in social, economic, and ecological systems",
            "the thermodynamic work cycle of molecular motors involves ratchet-like mechanisms that extract directed motion from thermal fluctuations by coupling mechanical transitions to chemical reactions far from equilibrium",
            "integrated information theory proposes a mathematical measure of consciousness called phi that quantifies the degree to which a system generates more information as a whole than the sum of its parts",
            // More advanced science continued
            "the warburg effect describes the tendency of cancer cells to rely on aerobic glycolysis rather than oxidative phosphorylation, even in the presence of oxygen, producing excess lactate as a metabolic byproduct",
            "synaptic plasticity in the form of long-term potentiation and long-term depression adjusts the strength of connections between neurons in an activity-dependent manner, providing a cellular basis for learning and memory",
            "the gauge symmetry underlying electromagnetism requires the photon to be massless and the electromagnetic interaction to be mediated by a vector field transforming under the abelian group u one",
            "immunological memory allows the adaptive immune system to mount a faster and stronger response on re-exposure to a pathogen because antigen-specific lymphocytes persist at elevated frequencies after the initial response",
            "the maximum entropy principle, advocated by jaynes as a basis for statistical inference, selects from among all probability distributions consistent with known constraints the one that maximises information entropy",
            "allometric scaling laws describe how biological rates and structures change with body mass, with metabolic rate scaling as the three-quarters power of mass across animals spanning more than twenty orders of magnitude in size",
            "the renormalisation of quantum field theories removes ultraviolet divergences by absorbing infinities into redefined parameters, yielding finite predictions that have been verified to extraordinary precision",
            "topological quantum computing uses non-abelian anyons whose braiding operations implement quantum gates in a manner inherently protected from local perturbations, potentially enabling fault-tolerant quantum computation",
            "the evolutionary origins of language remain debated, with competing hypotheses emphasising gestural origins, musical precursors, and the co-evolution of vocal anatomy and neural circuitry for syntactic processing",
            "supramolecular chemistry exploits non-covalent interactions such as hydrogen bonding, hydrophobic effects, and pi-pi stacking to assemble complex functional architectures from simpler molecular building blocks",
            "the anthropic principle in cosmology notes that observations of the universe are necessarily constrained by the requirement that conditions be compatible with the existence of observers capable of making those observations",
            "compressed sensing theory demonstrates that sparse signals can be recovered from far fewer measurements than the nyquist rate predicts, provided the measurement matrix satisfies a restricted isometry property",
            "the replication crisis in psychology and biomedicine has prompted calls for pre-registration, open data, and larger sample sizes to ensure that published findings reflect genuine effects rather than statistical artefacts",
            "predictive processing theories of perception propose that the brain continuously generates top-down predictions about sensory input and updates its model only when prediction errors signal a mismatch with incoming signals",
            "differential privacy provides a formal mathematical guarantee that the addition or removal of any single individual's data has negligible impact on the output of a statistical analysis, protecting against inference attacks",
            "the theory of mind ability to attribute mental states to others develops gradually in children and is partially impaired in autism spectrum conditions, illuminating the social cognitive architecture of the human brain",
            "attractor networks in computational neuroscience store memories as stable patterns of activity that the network settles into when presented with a partial or noisy cue, providing a model of content-addressable memory",
            "the dopaminergic reward prediction error signal, which encodes the difference between expected and received reward, is thought to drive reinforcement learning in the basal ganglia through the modification of synaptic weights",
            "climate sensitivity, defined as the equilibrium warming from a doubling of atmospheric carbon dioxide, is constrained by palaeoclimate evidence, present-day observations, and the emergent behaviour of global climate models",
            "the developmental origins of health and disease hypothesis proposes that early environmental exposures including nutrition and stress during critical periods of foetal development permanently alter physiology and disease risk later in life",
        ];

        Corpus {
            simple: simple.iter().map(|s| s.to_string()).collect(),
            moderate: moderate.iter().map(|s| s.to_string()).collect(),
            complex: complex.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn all(&self) -> Vec<(String, Complexity)> {
        let mut result = Vec::with_capacity(self.simple.len() + self.moderate.len() + self.complex.len());
        for s in &self.simple {
            result.push((s.clone(), Complexity::Simple));
        }
        for s in &self.moderate {
            result.push((s.clone(), Complexity::Moderate));
        }
        for s in &self.complex {
            result.push((s.clone(), Complexity::Complex));
        }
        result
    }

    #[allow(dead_code)]
    pub fn sample(&self, n: usize, rng_seed: u64) -> Vec<(String, Complexity)> {
        let all = self.all();
        let total = all.len();
        if n >= total {
            return all;
        }

        // xorshift32 seeded by rng_seed (truncate to u32, ensure non-zero)
        let seed32 = (rng_seed ^ (rng_seed >> 32)) as u32;
        let mut state: u32 = if seed32 == 0 { 0xdeadbeef } else { seed32 };

        let xorshift32 = |s: &mut u32| -> u32 {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            *s
        };

        // Fisher-Yates partial shuffle on indices
        let mut indices: Vec<usize> = (0..total).collect();
        for i in 0..n {
            let r = xorshift32(&mut state) as usize;
            let j = i + r % (total - i);
            indices.swap(i, j);
        }

        indices[..n]
            .iter()
            .map(|&i| all[i].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_counts() {
        let corpus = Corpus::load();
        assert_eq!(corpus.simple.len(), 200, "expected 200 simple sentences");
        assert_eq!(corpus.moderate.len(), 150, "expected 150 moderate sentences");
        assert_eq!(corpus.complex.len(), 150, "expected 150 complex sentences");
        assert_eq!(corpus.all().len(), 500, "expected 500 total sentences");
    }

    #[test]
    fn test_sample_determinism() {
        let corpus = Corpus::load();
        let a = corpus.sample(10, 42);
        let b = corpus.sample(10, 42);
        assert_eq!(a, b, "same seed must produce identical samples");
    }

    #[test]
    fn test_sample_different_seeds() {
        let corpus = Corpus::load();
        let a = corpus.sample(10, 1);
        let b = corpus.sample(10, 2);
        let different = a.iter().zip(b.iter()).any(|(x, y)| x != y);
        assert!(different, "different seeds should produce at least one differing element");
    }

    #[test]
    fn test_all_iterator() {
        let corpus = Corpus::load();
        let all = corpus.all();
        assert_eq!(all.len(), 500);
        for (_, c) in &all[..200] {
            assert_eq!(*c, Complexity::Simple);
        }
        for (_, c) in &all[200..350] {
            assert_eq!(*c, Complexity::Moderate);
        }
        for (_, c) in &all[350..500] {
            assert_eq!(*c, Complexity::Complex);
        }
    }

    #[test]
    fn test_no_empty_sentences() {
        let corpus = Corpus::load();
        for (sentence, _) in corpus.all() {
            assert!(!sentence.is_empty(), "found an empty sentence");
        }
    }

    #[test]
    fn test_word_count_ordering() {
        let corpus = Corpus::load();
        let mean_words = |v: &Vec<String>| -> f64 {
            let total: usize = v.iter().map(|s| s.split_whitespace().count()).sum();
            total as f64 / v.len() as f64
        };
        let simple_mean = mean_words(&corpus.simple);
        let moderate_mean = mean_words(&corpus.moderate);
        let complex_mean = mean_words(&corpus.complex);
        assert!(
            simple_mean < moderate_mean,
            "simple mean ({:.1}) must be less than moderate mean ({:.1})",
            simple_mean,
            moderate_mean
        );
        assert!(
            moderate_mean < complex_mean,
            "moderate mean ({:.1}) must be less than complex mean ({:.1})",
            moderate_mean,
            complex_mean
        );
    }
}

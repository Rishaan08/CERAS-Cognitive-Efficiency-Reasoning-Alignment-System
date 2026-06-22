export const GOOD_EXAMPLES = [
  {
    id: 'gp1',
    title: 'Quantum Foundations',
    gradient: 'linear-gradient(135deg, #4c1d95 0%, #1e1b4b 100%)',
    text: `Analyze the epistemological foundations of quantum entanglement by integrating formal mathematical structure, experimental validation, and philosophical interpretation into a coherent explanatory framework. Begin by describing how tensor product Hilbert spaces allow composite quantum systems to exhibit non-factorizable state vectors, and clarify why separability fails under entangled configurations. Then examine Bell's inequalities, including the CHSH formulation, and explain how empirical violations observed in Aspect-type experiments undermine classical locality and deterministic realism. Extend the discussion toward decoherence theory, entropic correlations, and the role of measurement operators in collapsing superposed amplitudes. Contrast Copenhagen, Many-Worlds, and relational interpretations, focusing specifically on their ontological commitments and metaphysical implications. Additionally, evaluate how quantum information theory reframes entanglement as a computational resource enabling teleportation, superdense coding, and cryptographic security. Finally, synthesize these perspectives into a structured argument addressing whether entanglement necessitates nonlocal causation or instead demands a revision of classical intuitions regarding separability, causality, and physical realism.`,
  },
  {
    id: 'gp2',
    title: 'Photosynthesis Systems',
    gradient: 'linear-gradient(135deg, #065f46 0%, #064e3b 100%)',
    text: `Analyze the systems-level biochemical and thermodynamic foundations of photosynthesis by integrating molecular structure, energetic transfer mechanisms, and ecological macro-dynamics into a coherent explanatory framework. Begin by explaining how chloroplast ultrastructure and pigment absorption spectra determine quantum excitation states. Then examine the light-dependent reactions as an electron transport optimization problem, including photolysis, proton gradients, chemiosmotic coupling, and ATP synthase rotation mechanics, and describe how each component contributes to overall energy yield. Compare this with the Calvin-Benson cycle, analyzing carbon fixation kinetics, RuBisCO efficiency constraints, and NADPH reduction pathways, and determine which limiting factors most affect photosynthetic throughput. Evaluate photosynthesis as an entropy-management system that converts low-entropy solar radiation into high-order biochemical organization. Finally, synthesize its planetary-scale implications for atmospheric regulation, carbon sequestration feedback loops, and biospheric energy flow stability across different ecosystems.`,
  },
  {
    id: 'gp3',
    title: 'Printing Press Analysis',
    gradient: 'linear-gradient(135deg, #9f1239 0%, #4c0519 100%)',
    text: `Analyze the multi-layered historical and epistemological significance of the Gutenberg printing press by integrating technological innovation theory, sociopolitical restructuring, and cognitive-cultural transformation into a coherent explanatory framework. Begin by describing the mechanical engineering principles underlying movable type standardization and ink transfer reproducibility. Then examine how mass replication altered information diffusion velocity and network topology across Renaissance Europe, and compare this to prior manuscript-based knowledge systems. Evaluate its causal role in accelerating scientific method formalization, destabilizing ecclesiastical epistemic monopolies, and enabling vernacular linguistic codification, and explain why each shift mattered for institutional authority. Extend the analysis toward media ecology theory and distributed cognition, examining how print culture reshaped memory externalization and authority structures. Finally, synthesize how the printing press functioned as an epistemic amplifier that reconfigured knowledge production, institutional legitimacy, and political sovereignty across early modern Europe.`,
  },
  {
    id: 'gp4',
    title: 'ML Paradigm Theory',
    gradient: 'linear-gradient(135deg, #1e40af 0%, #172554 100%)',
    text: `Analyze and compare the mathematically grounded architectural foundations of supervised and unsupervised machine learning paradigms, emphasizing objective functions, representational geometry, and statistical inference principles. Begin by defining supervised learning as an empirical risk minimization framework over labeled distributions, and contrast it with unsupervised latent-variable modeling and manifold estimation. Then examine bias-variance trade-offs, generalization bounds, and overfitting dynamics under distributional shift, and determine how these factors influence model selection. Compare algorithmic mechanisms such as Support Vector Machines, ensemble-based decision forests, K-Means clustering, and Principal Component Analysis through the lens of optimization landscapes and feature-space transformations. Evaluate interpretability constraints, scalability limits, and robustness under adversarial perturbations. Finally, synthesize these paradigms into a structured framework that determines when hybrid semi-supervised or self-supervised approaches become epistemically advantageous in real-world deployment.`,
  },
];

export const BAD_EXAMPLES = [
  { id: 'bp1', label: 'AI Basic', text: 'Explain artificial intelligence in simple terms.' },
  { id: 'bp2', label: 'Computers', text: 'Describe how computers work.' },
  { id: 'bp3', label: 'WWII Summary', text: 'Give a summary of World War II.' },
  { id: 'bp4', label: 'Sky Simple', text: 'Explain why the sky is blue in a short answer.' },
];

export const GROQ_MODELS = [
  'llama-3.3-70b-versatile',
  'llama-3.1-8b-instant',
  'qwen/qwen3-32b',
  'groq/compound',
  'groq/compound-mini',
  'openai/gpt-oss-120b',
  'openai/gpt-oss-20b',
];

export const GEMINI_MODELS = [
  'gemini-3-pro-preview',
  'gemini-3-flash-preview',
  'gemini-2.5-pro',
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite',
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
  'gemini-flash-latest',
  'gemini-flash-lite-latest',
  'gemini-robotics-er-1.5-preview',
];

export const OPENAI_MODELS = [
  'gpt-5.2',
  'gpt-5-mini',
  'gpt-5-nano',
  'gpt-5.2-pro',
  'gpt-5',
  'gpt-4.1',
  'gpt-4o',
  'gpt-4-turbo',
  'gpt-4',
  'gpt-3.5-turbo',
  'gpt-oss-120b',
  'gpt-oss-20b',
];
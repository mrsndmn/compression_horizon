# How does one memory embedding control reconstruction?

## Hypothesis
Cramming finds a sequence-specific control vector for a frozen conditional
reconstructor. Addressing, transmitted values, and interference between token
constraints may contribute differently in Standard and SPS. A fixed initial
predict query in SPS may delay context-dependent memory addressing; decoupling
alone does not guarantee more independently controllable target constraints.

## Setup
Diagnostic functions: `scripts.analyze_sps_memory_routes.probe` and
`scripts.analyze_sps_margin_geometry.probe`. Their registries and shared
`scripts.sps_mechanism_common` define cohort selection, exact saved stages,
source runs, numerical settings and job submission. Models and saved memories
remain frozen except for diagnostic gradients and explicitly recorded local
perturbations. All forwards use a single sample.

The route probe uses exact converged memories for the same target prefix in
both models. It separates context/predict routes in SPS, patches projected
memory keys and values from a norm-matched donor, and keeps the original
memory residual trajectory. Self-patching is the identity control. Context
replacement measures the layerwise query dependence on ordinary input tokens.

The margin probe compares the models at the same target length: the first
failed SPS stage. It evaluates both the previous converged memory and the
optimized memory for that stage. The Jacobian concerns target-vs-strongest-
competitor margins, not cross-entropy gradient rank. A local projected update
protects all previous-token margins to first order; finite perturbations check
where that linear approximation breaks. Local rank or conditioning cannot
establish a global representational limit.

## Results
Pending implementation verification and submission.

## Conclusions
Use the paper's attention-causality distinction and its warning that trajectory
PCA measures a path rather than the solution set. Existing knockout results
locate sensitive computations but do not yet explain the Standard/SPS horizon
gap. Checkpoints differ in learned weights as well as prediction layout, so
architecture-only attribution requires additional intervention controls.

## Changelog
- 2026-10-22: Started mechanism-directed routing, payload and controllability probes.

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
Both diagnostics were submitted from code snapshot
`c155f7c67bda8fbd63d13319322d6f7e4cb2a2e9` (see each output's job.json for
the authoritative full commit). Initial status Pending:

- Routing/KV: `lm-mpi-job-4e14b8cd-b4d7-4ba5-afb8-4d49196cf7d7`;
  output `artifacts/sps_cramming/mechanism_routes_20261022/`.
- Margin geometry: `lm-mpi-job-c0489873-6780-4a56-861d-8305032b4ebd`;
  output `artifacts/sps_cramming/mechanism_margin_geometry_20261022/`.

Implementation passed 23 focused tests, independent adversarial review, both
executable dry launchers, synthetic plot rendering and saved-data checks.
A durable Arkhip group observes both jobs. Results pending completion.

## Conclusions
Use the paper's attention-causality distinction and its warning that trajectory
PCA measures a path rather than the solution set. Existing knockout results
locate sensitive computations but do not yet explain the Standard/SPS horizon
gap. Checkpoints differ in learned weights as well as prediction layout, so
architecture-only attribution requires additional intervention controls.

## Changelog
- 2026-10-22: Started mechanism-directed routing, payload and controllability probes.

The distinction between QK addressing and OV information transmission follows
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
Its attention-only examples motivate interventions here; they do not establish
a mechanism in our deeper models with MLPs. The original
[Cramming paper](https://arxiv.org/abs/2502.13063) motivates measuring uncertainty
reduction as well as raw token counts; our local hypotheses remain unproven.

- 2026-10-22: Submitted both diagnostic jobs and registered durable observation.

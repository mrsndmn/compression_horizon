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
Both jobs completed successfully and their durable watch resolved. Raw outputs
and graphs were inspected. Self-patching changes NLL by at most 0.00000370;
all-layer joint-KV patching reproduces the full donor-memory condition on scored
positions. Every saved route/KV curve was recomputed independently.

The equal-length route comparison shows a distinction in SPS: first-layer
memory-to-context knockout increases NLL by 2.034, versus 0.220 for the direct
memory-to-predict route (joint: 2.194). By layer eight the pattern reverses:
0.038 for context versus 0.857 for predict. The first predict query is exactly
context-independent, but that fact alone does not explain the first-layer effect:
the dominant early causal route is through ordinary context states.

Across all layers, donor value replacement changes NLL by 5.277 for Standard
and 4.834 for SPS, while donor key replacement gives 2.396 and 2.450. This
intervention is stronger on outgoing values in both models; it is not a count
of bits stored in V or an additive decomposition of contributions. Effects of
individual patches and removals are non-additive.

At the matched first-failed SPS lengths, mean new-token gradient fractions
orthogonal to previous-token constraint rows are 0.847 for Standard and 0.875
for SPS at warm start. Every measured previous-token Jacobian has full row rank.
This does not support local rank exhaustion as a simple explanation. The
median minimum previous-token margin is 0.147 versus 0.029; finite moves can
invalidate the first-order protection. SPS is sampled at its frontier and
Standard is usually below its own, so the smaller margins may be a symptom of
being near the boundary, not an architecture-specific cause. Some warm starts
already lose tiny-margin matches when evaluated in FP32 rather than training BF16.

A follow-up diagnostic is `scripts.analyze_sps_margin_rescue.probe`: restart
failed SPS memories without adding parameters, compare margin-aware constrained
steps with Adam controls, account for VJPs and forwards, and validate selected
solutions in the original numerical regime and free generation. Its source
registry defines all settings. Results pending verification and submission.

## Conclusions
Use the paper's attention-causality distinction and its warning that trajectory
PCA measures a path rather than the solution set. Existing knockout results
locate sensitive computations but do not yet explain the Standard/SPS horizon
gap. The new probes support a memory-generated layerwise K/V controller with
an early context-mediated route and later direct prediction reads in SPS.
Decoupling readout positions does not make ordinary hidden states independent
of the memory. The fixed-query hypothesis is not enough to explain the measured
early dependence, and the local first-order rank hypothesis is weakened.
What remains unresolved is how much of the boundary comes from the optimization
protocol versus global feasibility of the constraints. Checkpoints differ in
learned weights as well as prediction layout, so architecture-only attribution
requires additional intervention controls.

## Changelog
- 2026-10-22: Started mechanism-directed routing, payload and controllability probes.

The distinction between QK addressing and OV information transmission follows
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
Its attention-only examples motivate interventions here; they do not establish
a mechanism in our deeper models with MLPs. The original
[Cramming paper](https://arxiv.org/abs/2502.13063) motivates measuring uncertainty
reduction as well as raw token counts; our local hypotheses remain unproven.

- 2026-10-22: Submitted both diagnostic jobs and registered durable observation.

- 2026-10-22: Recorded completed routing and margin findings; started a direct optimization-versus-feasibility rescue control.

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
registry defines all settings. Implementation passed 18 focused tests and
independent adversarial review. Submitted as
`lm-mpi-job-4ba5786a-dd90-4cfc-96ac-b48ba81c4340` from snapshot
`157a04b98608905364dda5cd311a6c90ab7a3b21`; output
`artifacts/sps_cramming/mechanism_margin_rescue_20261022/`.
The job completed successfully. CE restarts at either tested learning rate
recover none of the four prefixes within the diagnostic budget. Squared-margin
Adam recovers all four in FP32 using 20–55 gradients; three also pass both
teacher forcing and free generation in the original BF16 regime. The constrained
controller recovers all four in FP32, but only one survives BF16 validation.
The unchanged checkpoint and memory dimension therefore admit solutions beyond
three original BF16 stopping points. These selected prefix rescues do not measure
full progressive horizons or show that the overall Standard/SPS gap disappears.
The figure is `mechanism_margin_rescue_20261022/margin_rescue.png`.

The follow-up functions are the objective suite in
`scripts.jobs.run_jobs_sps.main` and
`scripts.audit_sps_objective_gradients.probe`. The former separates reconstruction
loss from the stopping margin in full progressive runs of both checkpoints; the
latter measures objective preferences and gradient cancellation on saved rescue
vectors. Their source registries define every setting. The gradient audit completed:
every exact hinge vector has higher CE than its corresponding erroneous CE
vector, and at all four selected CE vectors plain CE descent locally worsens
the worst-token margin. Correct- and wrong-token gradient cosines are negative.
These are local SPS findings. The complete progressive comparison is now
available: hinge increases reconstruction lengths in both models, but preserves
the SPS disadvantage. See `sps_objective_factorial.md` for measurements and controls.

The subsequent user-proposed intervention changes the conditioning site:
`scripts.analyze_sps_predict_embedding.probe` optimizes one shared prediction
input and holds an existing ordinary BOS fixed. All three runs completed;
`sps_predict_embedding.md` records the matched comparisons and independent
saved-stage audit. CE mean teacher-forced length increases from 19.42 to 64.17;
strict hinge increases from 31.52 to 132.10 with native initialization and
130.23 with random initialization. Every text improves in teacher-forced length.
Mean initial correct free-generation prefixes increase from 30.56 to 110.23
and 118.65 in the hinge comparison, with 38/48 and 43/48 exact reconstructions
of the selected longer prefixes. Most hinge texts reach the total step budget.

The completed independent-schedule control is documented in
`sps_predict_fixed_schedule.md`. Restoring the original LR trajectory at the
extended budget recovers teacher horizon from 79.42 to 133.85, versus 132.10
originally. It improves 33 texts and ties 15. All 6342 paired stages before the
original budget boundary match exactly in vector, update counter and convergence.
The extended fixed-schedule run uses 11391.65 updates on average and stops on a
per-stage failure for every text. The regression from stretching the schedule is
therefore distinct from a representational capacity limit. Merely allowing more
total updates gives a small gain when the original LR path is retained.

A further saved-stage audit, `terminal_constraint_audit.json` under the fixed
control's merged output, counts violations of the recorded positive-margin
convergence criterion. Across all 48 terminal failures the median count is ten,
the range is two to 92, and the equal-text mean fraction satisfying the margin
is 87.08%. A separate Arrow-based count check verifies the histogram and that
every preceding stage converged and every terminal stage adds exactly one
target. Thus every failed stage also violates at least one previously satisfied
target constraint; the median lower bound is nine old-position violations.
These are margin violations, not necessarily wrong argmax predictions. The audit
does not localize individual positions or separate optimization-induced changes
from BF16 context-shape changes; either can affect old constraints.

## Conclusions
Use the paper's attention-causality distinction and its warning that trajectory
PCA measures a path rather than the solution set. Existing knockout results
locate sensitive computations but do not yet explain the Standard/SPS horizon
gap. The new probes support a memory-generated layerwise K/V controller with
an early context-mediated route and later direct prediction reads in SPS.
Decoupling readout positions does not make ordinary hidden states independent
of the memory. The fixed-query hypothesis is not enough to explain the measured
early dependence, and the local first-order rank hypothesis is weakened.
The rescue establishes that several observed stopping points are failures of the
optimization protocol rather than infeasibility at the same memory dimension.
The complete matched-cohort factorial confirms a substantial optimization
bottleneck in both models: strict-hinge mean lengths are 54.83 (Standard) and
31.52 (SPS), versus raw-CE lengths 32.31 and 19.42. The gap remains; changing
the loss alone does not explain why the SPS checkpoint reconstructs less. Checkpoints differ in
learned weights as well as prediction layout, so architecture-only attribution
requires additional intervention controls.

The supported mechanistic picture is a frozen conditional reconstructor
controlled by a memory-generated sequence of layerwise K/V vectors. The memory
position is causally autonomous; other positions read its outgoing K/V and can
relay the effect through ordinary context states. For Standard, the first
prediction also has a direct memory-residual readout, which is excluded from
scored routing/KV comparisons. In SPS, input/predict separation and hidden-state
independence are distinct: ordinary tokens have ordinary input embeddings but
their hidden states remain memory-dependent, while all token constraints still
share the same optimized memory vector. Decoupling does not imply independently
adjustable reconstruction constraints.

The shared-predict intervention materially changes this picture: substantial
reconstruction gains are possible while ordinary context states are independent
of the learned vector. Reusing a text-specific vector in every prediction slot
provides direct control of the prediction residual stream and queries, rather
than conditioning exclusively through a memory position's outgoing K/V. The
earlier ordinary-memory horizons therefore do not bound this checkpoint's
one-vector reconstruction ability under other conditioning protocols. This
supports prediction-stream conditioning as a useful mechanism, but does not
attribute the gain uniquely to residuals, queries, or the absence of context
interference. The vector has the same parameter count but is injected at
different computational locations.

The remaining distinction between learned checkpoint quality and architecture
is unresolved. Short-prefix causal routing does not by itself locate the
long-prefix feasibility boundary. A local full-row-rank Jacobian excludes one
simple local explanation, not a nonlinear global capacity limit. Exact generation
also depends on numerical margin: even strict stopping leaves a few BF16 failures.
The paper's uncertainty-reduction perspective suggests why a reconstruction
length is model/text/optimizer dependent rather than a direct stored-bit count.
These limits bound the findings; they are not evidence for a specific remaining
architecture-only cause.

The operational interpretation of shared-predict cramming is a text-specific
control vector for a frozen conditional reconstructor: logits at each target
depend on the same vector and the already reconstructed history. A common vector
can produce different outputs because the conditioning history and position
change. Exact reconstruction imposes simultaneous target-versus-rival margin
constraints. The objective, accessible control route, optimization trajectory
and numerical implementation all affect achieved feasibility. This does not
establish semantic transfer or a portable representation independent of the
decoder; neither has been measured for the new shared-predict vectors.

The strongest within-checkpoint result concerns the conditioning interface:
shared-predict input control beats ordinary-memory control with either objective
and with either tested initialization. It gives the optimized vector a direct
residual path at every prediction position, including query control and its own
temporary self K/V, while the ordinary context states stay independent of it at
zero window. This changes several causal routes together. Their individual
contributions and the conjecture that avoiding context modification reduces
interference remain unisolated. All prediction constraints still share the same
vector, and terminal failures involve old constraints as well as the added one.

At the original matched budget, Standard/ordinary-memory strict hinge reaches
54.83 tokens, SPS/ordinary-memory reaches 31.52, and SPS/shared-predict reaches
132.10. These demonstrate an advantage for the new SPS conditioning protocol.
They do not isolate an architectural advantage: the checkpoints have different
learned weights, Standard has not received an analogous repeatedly applied
readout-only control, and equal parameter count does not equate intervention
locations or forward computation. The old memory-route and local-Jacobian
diagnostics also do not automatically describe the new shared-predict frontier.

Two unresolved causal questions should be kept distinct. To identify the source
of the interface gain, compare an analogous readout-only intervention in Standard
and isolate query/residual/self-KV routes at matched lengths. To identify the
current stopping mechanism, inspect per-position margins at the first failed
shared-predict stage before and after optimization, including an arithmetic
control with the same quantized weights; then test whether preserving old
constraints or relaxing the local optimization limit rescues that same prefix.
A successful rescue would disprove infeasibility at that prefix; unsuccessful
finite-budget searches would not establish a global ceiling. These are proposed
distinguishing tests, not newly submitted experiments or registered job configs.

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

- 2026-10-22: Submitted matched-gradient-budget rescue controls; independently checked geometry projections from raw Gram matrices and inspected all completed figures.

- 2026-10-22: Recorded successful unchanged-dimension rescues; prepared full objective/convergence factorial and a read-only objective-gradient audit.

- 2026-10-22: Added completed read-only evidence that mean-CE optimization and exact token constraints conflict at the selected SPS boundaries.

- 2026-10-22: Integrated completed factorial, generation robustness and bounded mechanism conclusions.

- 2026-10-22: Integrated completed fixed-BOS/shared-predict intervention: large same-checkpoint gains, free-generation checks, initialization control and optimization-budget limits.

- 2026-10-23: Integrated the fixed-schedule control, exact trajectory replay and
  saved terminal-margin audit. Distinguished the shared-predict conditioning
  advantage, unresolved architecture attribution, old-constraint failures and
  protocol/numerical/feasibility limits.

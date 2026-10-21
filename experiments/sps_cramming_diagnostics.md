# Why does the SPS checkpoint cram less text?

## Hypothesis
The fixed prediction-token embedding does not guarantee easier control by a
single input memory vector. SPS may differ in optimization conditioning,
precision sensitivity, and the pathways from memory to prediction states.

## Setup
Training/diagnostic function: `scripts.analyze_sps_cramming.probe`.
The paired artifact audit is `scripts.analyze_sps_cramming.summary`.
The source file defines every probe, refinement variant, sample selection,
checkpoint, budget and job setting.

## Results
The original comparison favors Standard on all 48 matched texts. Mean exact
reconstruction horizons are 32.0625 versus 17.1667 tokens. No sample exhausted
the cumulative optimization budget. Each horizon is bounded by the first
non-converged stage, after which that sample's embedding was frozen.
Targeted checkpoint diagnostics were submitted from code commit
`89185b2` as `lm-mpi-job-d2c4be82-69e8-4ec7-8b11-1e18f7637a20`.
The job completed successfully; its durable wait
`398bfaff9e9b43e3a37830b99ab1ec05` resolved.
The artifact audit, paired plot and probe outputs live under
`artifacts/sps_cramming/diagnostics_20261021/`.

The saved best pretraining validation NLL slightly favors SPS (3.36768 versus
3.38784), so worse general language-model quality alone is not established as
an explanation. Under exact reconstruction, a typical first failure is one
wrong token. The completed restart probe recovered exact teacher-forcing
reconstruction for some saved first-failure prefixes:

| Diagnostic variant | Standard recovered | SPS recovered |
| --- | ---: | ---: |
| Original-rate BF16 restart | 5/8 | 4/8 |
| Reduced-rate BF16 restart | 5/8 | 2/8 |
| Increased-rate BF16 restart | 2/8 | 5/8 |
| Original-rate FP32 restart | 5/8 | 4/8 |

Across variants, six Standard and five SPS samples were recovered at least once.
These are saved failed prefixes, not fresh progressive runs; the two models'
failed prefix lengths differ. Optimizer moments were restarted and the learning
rate was constant. The historical probe used a multi-sample batch and predates
the user's single-sample policy. It must not be reported as the new remeasurement.

Natural-prefix NLL on the small matched diagnostic subset was 4.86496 for
Standard and 4.90159 for SPS; token accuracies were 0.21371 and 0.19758.
This small subset does not overturn the saved pretraining validation comparison.
At the shared initial memory probe, mean token-loss gradient norm was 32.42310
for Standard versus 23.29867 for SPS. Mean smallest singular values of the
per-token loss-gradient matrix were 5.10751 versus 2.53151. These local
measurements suggest different memory sensitivity, but do not establish the
cause of the optimized horizon gap or a limit on model capacity.

The generation audit confirms the interleaved zero-window implementation is
functionally equivalent to ordinary-only history plus one current prediction
slot, including memory gradients. Corrupting historical prediction slots does
not affect next-token logits. The two real-checkpoint decoding layouts produced
the same tokens in the audited sample.

Generic native decoding lacked suppression of control/unused vocabulary IDs;
this is now fixed in free-generation paths. All 96 original first-failure stages
were replayed with exactly matching accuracies: none of the wrong argmax tokens
were reserved IDs, and suppression rescued no failed prefix. This decoder bug
therefore does not explain the measured teacher-forcing horizon gap.

A saved SPS prefix was exact in the original teacher-forcing batch shape but
lost one token at batch size one in BF16. The relevant target-vs-competitor
margin changed from 0.0625 to a tie. Holding quantized weights and input vectors
fixed while computing in FP32 restored agreement across shapes (margin about
0.02346). These are teacher-forcing horizons, not guaranteed greedy horizons.
Evidence: `generation_audit.json` beside the diagnostic artifacts.

## Conclusions
The current comparison establishes an optimization-protocol-dependent gap;
it does not establish an architectural capacity limit. Recovering saved failed
prefixes demonstrates that the original stopping boundary can be an optimization
failure. Learning-rate changes affect the models differently; FP32 did not
increase the rescue count in this probe, so it does not support arithmetic
precision as the sole explanation. A fixed SPS prediction embedding does not
imply that the next-token mapping is equally easy to control with one memory
vector. The separate single-sample progressive pair is still pending; use its
teacher-forcing replay and greedy results to assess the batch-shape confound.

## Changelog
- 2026-10-21: Added paired first-failure audit and bounded checkpoint probes.
- 2026-10-21: Matched Adam controls after independent review, submitted diagnostics and registered durable observation.
- 2026-10-21: Verified ordinary-history decoding equivalence, fixed reserved-ID generation, and isolated BF16 shape sensitivity.

- 2026-10-21: Recorded completed restart and local-sensitivity diagnostics; kept historical batched results separate from pending single-sample remeasurement.

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
An Arkhip durable wait is registered as `398bfaff9e9b43e3a37830b99ab1ec05`.
The artifact audit, paired plot and probe outputs live under
`artifacts/sps_cramming/diagnostics_20261021/`.

The saved best pretraining validation NLL slightly favors SPS (3.36768 versus
3.38784), so worse general language-model quality alone is not established as
an explanation. Under exact reconstruction, a typical first failure is one
wrong token. The gradient/precision/restart probe is still pending.

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
it does not establish an architectural capacity limit. Test whether failed
prefixes are recoverable before attributing the gap to representational capacity.

## Changelog
- 2026-10-21: Added paired first-failure audit and bounded checkpoint probes.
- 2026-10-21: Matched Adam controls after independent review, submitted diagnostics and registered durable observation.
- 2026-10-21: Verified ordinary-history decoding equivalence, fixed reserved-ID generation, and isolated BF16 shape sensitivity.

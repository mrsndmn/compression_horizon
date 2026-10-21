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
Targeted checkpoint diagnostics are pending.

## Conclusions
The current comparison establishes an optimization-protocol-dependent gap;
it does not establish an architectural capacity limit. Test whether failed
prefixes are recoverable before attributing the gap to representational capacity.

## Changelog
- 2026-10-21: Added paired first-failure audit and bounded checkpoint probes.

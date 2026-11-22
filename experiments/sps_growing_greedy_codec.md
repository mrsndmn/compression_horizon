# Greedy reconstruction with a growing input

## Hypothesis

A larger target-versus-rival logit margin can make a saved shared-prediction
vector robust to the change of forward shape during greedy decoding.

## Setup

Refinement and acceptance use `scripts.sps_growing_codec.certify_growing`.
The experiment registry is in the same module. Export a saved stage with
`python -m scripts.sps_codec certify --growing SHARD SAMPLE_ID PACKET.pt`.
Read it with `python -m scripts.sps_codec decode PACKET.pt`. Both paths use
the project's existing greedy generator, including its vocabulary mask and
EOS stopping rule. A packet contains one prediction vector per text.

## Results

For the six texts in the first Qwen SPS-CE shard, the archived growing BF16
generator recovered zero complete prefixes. In a diagnostic run with the
original vectors, projecting the last BF16 hidden state through the output
head in FP32 recovered two. Upcasting the whole BF16 model to FP32 recovered
four. Full FP32 resolved every original first mismatch when given the correct
history, but on two texts blind generation diverged elsewhere.

With margin refinement, the growing BF16 decoder recovered all six full
prefixes (1,626 tokens). All six certified packets were saved, reloaded and
verified against the original IDs. They are available under
`artifacts/analysis/sps_growing_codec_qwen_ce_shard00_e04f01a/`.
A separate-process CLI round trip reproduced all 271 IDs of sample zero.
The cohort is unchanged from the fixed-shape comparison.

## Conclusions

Additional vector optimization can pay the cost of robustness at encoding time.
The deployed generator processes the ordinary growing input. The margin is a
training heuristic. Exact blind replay with the stored vector determines
acceptance. The guarantee is scoped to the certified prefix and deterministic
decoder configuration. The checksum and token count are checked on every read.

## Changelog

- 2026-11-23. Compare precision and margin refinement, and add growing-prefix packet certification.

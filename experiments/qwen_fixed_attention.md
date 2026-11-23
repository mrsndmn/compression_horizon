# Causal attention invariant to cached prefill boundaries

## Hypothesis

Fixing the order of attention reductions can make cached and full-prefill
outputs agree once tokenwise projections are invariant to the number of rows.

## Setup

The diagnostic entry point is `scripts.probe_qwen_fixed_attention.run`; its
experiment registry defines the source text and comparisons. Run from the
project environment with the Thinking Machines [reference implementation](https://github.com/thinking-machines-lab/batch_invariant_ops/tree/f22b1fbe534b1fe045080f44a970a441caaf3c4f)
on PYTHONPATH:

```sh
python -m scripts.probe_qwen_fixed_attention --output artifacts/analysis/qwen_fixed_attention/prefill.json
```

The Triton forward implements unpadded batch-one causal attention, including
grouped query attention and an existing KV cache. Absolute query tiles and
key blocks fix the sequence of softmax updates across splits. The test compares
against a dense double-precision reference and checks cached prefix and suffix
outputs at aligned and unaligned boundaries.

## Results

On the original frozen Qwen checkpoint, the fixed-order forward reproduced
every logit of its own full pass for all three tested cached splits. The
first attention layer also matched exactly on every position. The final 32
positions, processed one token at a time with the ground-truth history and a
growing cache, reproduced every full-pass logit. The default SDPA and math
backends changed 3,602,909 and 3,581,841 logits respectively in the same
one-token comparison, with no change of the winning token on those positions. Both
built-in attention backends had logit differences when split. The fixed-order
and default full passes differed in the winning token at 145 positions.

## Conclusions

In this probe, changing attention while keeping tokenwise kernels invariant
removed sensitivity to the tested prefill boundaries. This measures numerical
consistency of the original causal model. SPS visibility and training gradients
require separate implementations and validation.

## Changelog

- 2026-11-23. Compare the default and math SDPA backends with fixed-order Triton attention.

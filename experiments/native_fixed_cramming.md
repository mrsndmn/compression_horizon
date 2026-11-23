# End-to-end native Standard and SPS cramming with fixed-order attention

## Hypothesis

With invariant tokenwise operations and masked attention, a converged vector
should reproduce its prefix during ordinary growing-prefix greedy decoding.
Changing numerical kernels can also change the achieved optimization horizon.

## Setup

The registry and training function are in `scripts.run_native_fixed_cramming`.
The Standard arm learns an ordinary memory and the SPS arm learns its shared
prediction embedding. Both replay the historical initial vectors and native
target IDs. The fixed-order forward consumes the adapter's visibility mask.
The analytical backward uses FP32 softmax-attention gradients. Frozen model
weights are shared by training and replay.

Use the module's `--phase dry`, `submit` and `summary` entry points. Workers
write stage artifacts, reload the selected vector, compare a full pass to
growing true prefixes, and separately decode blindly. The pinned dependencies
on the shared volume are the Thinking Machines source checkout and GPT-2
tokenizer snapshot identified in the registry.

## Results

Both real-checkpoint A100 smokes trained a vector and recovered its selected
prefix after serialization. The CUDA regression checks native visibility,
gradient parity and growing-prefix forward parity, including padding.

## Conclusions

The paired cohort will measure horizon, exact greedy recovery and logit
agreement on the trained vectors. Native greedy replay recomputes the growing
prefix with the historical adapter.

## Changelog

- 2026-11-23. Implement and smoke-test matched Standard and SPS cramming.

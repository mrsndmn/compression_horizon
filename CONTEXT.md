# Glossary (CONTEXT.md)

Canonical domain terms for the Progressive Cramming project. Glossary only — no
implementation detail. See `docs/adr/` for decisions and `src/` for code.

## Full Cramming
Compressing a span of text into one (or a few) learnable **memory embedding(s)** by
gradient descent, optimizing the embedding(s) so the frozen language model reconstructs
the original tokens. The base, non-staged method.
_Avoid:_ "single-shot cramming", "direct cramming".

## Progressive Cramming
A staged variant of cramming that grows the compressed span incrementally: each stage
extends the sequence (or adds tokens) only after the current stage has converged to the
reconstruction threshold.
_Avoid:_ "incremental cramming", "curriculum cramming".

## Low-Dim Projection
A reparametrization of the memory embedding as the image of a low-dimensional
coefficient vector through a learned linear projection, optimizing in the low-dimensional
subspace instead of full hidden size. A specialization of Full Cramming.
_Avoid:_ "low-rank cramming", "subspace cramming".

## Memory Embedding
The optimized embedding vector(s) that stand in for a text span — the artifact every
cramming method produces.
_Avoid:_ "compression token", "mem token" (these name the slot, not the learned vector).

## Toy Quickstart
The fast, public-facing demonstration: each of the three methods run on a small model
over a few samples, printing its headline metric in minutes on a single GPU. Distinct
from the paper-scale cluster runs.
_Avoid:_ "smoke test" (that implies pass/fail only, not metric demonstration).

## Cross-Entropy Temperature
A scalar `T` applied to the reconstruction cross-entropy during cramming: the model's
logits are divided by `T` before the softmax, so `T > 1` softens (flattens) the predicted
distribution the loss is measured against and `T < 1` sharpens it. A training-time
optimization knob for the loss only; convergence is still judged on the (temperature-
invariant) argmax. `T = 1` is ordinary cross-entropy.
_Avoid:_ "sampling temperature", "generation temperature" (those scale logits at decode
time, not inside the training loss).

## Gradient-Compensated Temperature
The Hinton-distillation convention of multiplying the temperature-scaled cross-entropy by
`T^2` so the gradient magnitude stays ~constant as `T` varies, isolating the
distribution-shape effect from the effective step-size effect. The uncompensated ("raw")
form omits the `T^2` factor, so the gradient scales ~`1/T`. Both reduce to plain
cross-entropy at `T = 1`.
_Avoid:_ "temperature normalization".

## Public Repo (progressive_cramming)
The published reproducibility repository (FusionBrainLab/progressive_cramming). Houses
the renamed `progressive_cramming` package, README, LICENSE, and the existing
presentation deck. Distinct from the internal `compression_horizon` research repo.
_Avoid:_ "compression_horizon" when referring to the public artifact.

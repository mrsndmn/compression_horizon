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

## Public Repo (progressive_cramming)
The published reproducibility repository (FusionBrainLab/progressive_cramming). Houses
the renamed `progressive_cramming` package, README, LICENSE, and the existing
presentation deck. Distinct from the internal `compression_horizon` research repo.
_Avoid:_ "compression_horizon" when referring to the public artifact.

## Convergence Margin (ε)
The logit gap a token must clear to count as converged: a position matches only when
`logit[true] − max_{j≠true} logit[j] ≥ ε`. ε = 0 is legacy bare-argmax convergence; ε > 0
yields decode-robust reconstruction (survives kernel / forward-shape perturbations) at the
cost of fewer crammed tokens. The knob of the robustness-versus-compression trade-off.
_Avoid:_ "confidence threshold", "loss margin" (that names the CE-reweighting knob, not the
convergence criterion).

## Information-Gain Budget
The total bits a memory embedding saves on its span, `Σ_token Δsurprisal`, treated as a
roughly fixed per-model resource that is distributed across the crammed tokens. The premise
that standard cramming spends this budget unevenly (huge margins on easy tokens, near-zero on
others) motivates rebalancing it.
_Avoid:_ "capacity" (broader), "compression budget".

## Budget-Rebalancing Loss
A training objective that redistributes the Information-Gain Budget across tokens — reclaiming
budget from over-margined (easy) tokens — so that a larger number of tokens clear the
Convergence Margin, extending the crammable span at fixed budget. Distinct from the existing
margin-deficit CE reweighting (+LM), which only raises the floor and never reclaims over-margin.
_Avoid:_ "margin-weighted loss" / "+LM" (that is the floor-raising baseline, not a rebalancer).

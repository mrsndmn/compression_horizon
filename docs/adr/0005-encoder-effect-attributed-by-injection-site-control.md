# 5. Attribute the encoder effect with a within-model injection-site control, not an N-token sweep

Date: 2026-08-29

## Status
Accepted

## Context
The hypothesis under test is that cramming works better on an encoder-decoder model
"because of the encoder", and specifically that it preserves more semantics. The obvious
reading of that claim is about the encoder's **bidirectional self-attention**, and the
obvious way to test it is to vary the number of memory embeddings `N`.

Two facts make that reading a trap:

1. At `N = 1` the encoder's self-attention is degenerate. Softmax over a single key is
   `1`, so every encoder layer collapses to `residual + MLP`: the encoder becomes a deep
   per-vector feature extractor, and bidirectionality contributes exactly nothing. What
   remains is still a real mechanism — a deep nonlinear reparametrization of the memory
   plus a cross-attention read path available at every decoder layer — but it is *not*
   the mechanism the word "encoder" usually invokes.
2. The cross-model comparison cannot attribute anything on its own. T5Gemma was adapted
   from Gemma-2 with additional PrefixLM pretraining, so "T5Gemma beats Gemma-2" is
   equally well explained by that extra pretraining, or by the ~2x parameter count.

`N = 1` is nevertheless the config every existing result in this project uses, and moving
off it would make the new numbers incomparable to `tab:progressive_modifications`,
`tab:semantic_evaluation` and everything downstream of them.

## Decision
Keep `N = 1` — matching the rest of the project — and buy the attribution from a
**within-model control on the injection site** instead of from an `N` sweep:

- **[[Encoder-Input Inversion]]** — memory in the encoder's input space (the hypothesis).
- **[[Encoder-Output Inversion]]** — memory optimized directly in the encoder's output
  space, encoder never run (the control).

Both arms use identical weights, identical data, and an identical step budget, so the
only difference is whether the encoder stack is in the path. The decoder-only
`gemma-2-2b` arm remains as the external reference point.

## Consequences
- The headline claim the experiment can support is "the encoder **stack** helps", not
  "bidirectional encoding helps". Bidirectionality is untested by construction and must
  not be claimed; an `N > 1` follow-up is the only thing that could test it.
- If encoder-input and encoder-output arms come out equal, the "because of the encoder"
  framing is refuted even when T5Gemma beats Gemma-2 — the credit then belongs to
  cross-attention, to the adaptation pretraining, or to parameter count.
- The encoder-output arm is cheaper than the hypothesis arm (no encoder forward and no
  backward through it), so the control costs less than a third arm normally would.
- A fourth arm (T5Gemma's decoder without its encoder) would additionally separate
  "architecture" from "extra pretraining"; it was considered and dropped for cost. That
  confound therefore survives in the cross-model comparison and must be stated whenever
  the Gemma-2 baseline is contrasted with either T5Gemma arm.

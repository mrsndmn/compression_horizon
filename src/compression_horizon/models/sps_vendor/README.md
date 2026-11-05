Copied from `sps` commit `198b1b706f7e72f2100ccd793c67f199bfcf3fbd`:
`src/modeling/models/model.py` → `components.py`, `src/modeling/masking.py` → `masking.py`.
The masking import is relocated; repository formatting/linting removes unused
imports and an unused local variable. Computation is unchanged. See LICENSE for the upstream MIT notice.
The cramming adapter reuses Block, RMSNorm, MLP and complex RoPE; SDPA supplies
attention with the Standard or SPS visibility rule. SPS doubles each input into
`[input, predict]`, shares RoPE positions across the pair, and reads odd-slot logits.
At w=0 input keys persist and only the current predict key is visible.

With `inputs_embeds`, each row is treated as one document unless `document_ids`
is supplied. This matches the single-text cramming task; EOS does not implicitly
reset its memory. The token-ID path also supports upstream EOS document boundaries.
No KV cache, HF save_pretrained or AutoModel registration is provided. Use
`load_sps_checkpoint` and `load_sps_tokenizer` for native checkpoint inference.

`pretrained.py` is copied from `sps` commit
`c4bc39c141d9004b5f2697a50f90d1e7e63ba767`,
`src/modeling/models/sps/pretrained.py` (upstream source SHA256
`1ada96f9b11847827a93b9ae62c1677af3d5f585ebdba52a316c18a19471c7a8`).
Only repository Black formatting is applied; the parsed AST matches the pinned
upstream source. It is the independent reference for pretrained Llama/Qwen3 SPS checkpoints,
which use a separate prediction parameter and prediction position i+1. It is
separate from the native pilot implementation described above. The differentiable
embedding adapter is `models/pretrained_sps.py`; its tests compare this reference
and an ordinary-history/current-prediction forward.

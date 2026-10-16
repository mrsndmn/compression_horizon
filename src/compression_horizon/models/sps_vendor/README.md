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

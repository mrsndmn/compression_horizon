# Progressive cramming after extended Qwen SPS-CE adaptation

## Hypothesis

The requested final checkpoint may recover longer prefixes than the previously
audited Qwen SPS-CE checkpoint under the same cramming protocol. The primary
comparison is the paired change in teacher-forced horizon. Greedy decoding
tests how much of that prefix can be recovered from the saved vector alone.

## Setup

Training function: `scripts.pretrained_sps_worker.train_shard`.

1. Add the requested checkpoint as a separate entry in the executable registry
   in `scripts/pretrained_sps_experiments.py`. Give the run its own output
   location and target it explicitly in the launcher. Reuse the audited Qwen
   protocol and verify the ordered document hashes and token IDs against the
   archived Qwen cohort before submission.
2. Run a GPU preflight for this entry. Verify source and revision, load the SPS
   state strictly, and check reference logits and gradients with `parity`.
   Exercise optimization with frozen model weights, saving, replay and greedy
   decoding on one text. Proceed only if these checks pass.
3. Submit the full cohort through MLS and record the checkpoint identity, code
   revision and job IDs. Verify that completed shards cover each text once.
4. Run `scripts.analyze_pretrained_sps.audit_experiment` and verify exact
   teacher-forced replay of each selected prefix. Pair results by document with
   the existing Qwen SPS-CE audit. Report mean and median paired horizon change,
   mean initial correct greedy prefix, exact greedy reconstruction count and
   budget exhaustion. Use the existing original Qwen AR audit for context.

## Results

The checkpoint metadata matches the registered Qwen source and revision. The
ordered document hashes and digest of target IDs and attention masks match the
archived Qwen SPS-CE shard reports. The checkpoint-specific GPU preflight was
submitted from `e1c08f8` as `lm-mpi-job-f0e7c1a9-757e-415b-922f-43ed7d4e3e96`.
This preflight passed, with zero maximum logit error against the reference.
After the registry correction, the matching-code GPU preflight was submitted
from `2b5014b` as `lm-mpi-job-8323f42d-32bc-495f-81a0-54ccce0c4f3d`.
The matching-code preflight passed. MLS accepted the full cohort from
`f3704ef`. All eight jobs in Arkhip group `aa49f348d0af449bbb0be5dffe31ae7e`
completed. Their IDs follow in shard order.

- `lm-mpi-job-3104d6bd-89e3-41e9-bc25-59f260938cf9`
- `lm-mpi-job-a1a090ba-b499-4d49-8604-8de55a5d45cb`
- `lm-mpi-job-ad1da02f-f609-4fef-b0ef-bd8edb5dc43e`
- `lm-mpi-job-25fdea15-bcde-4a21-895c-5ce98dfba925`
- `lm-mpi-job-0438fa40-af45-4f77-93c6-fe505b491b5e`
- `lm-mpi-job-00e1ad6d-0606-4aab-ad3a-0aa80e8d1c84`
- `lm-mpi-job-7fc7d5c6-22f5-455e-ac43-91f6ff81223b`
- `lm-mpi-job-d1e09dab-f542-463a-a59b-af8eebd3491a`

The raw-stage audit covered all 48 documents exactly once. Every selected
prefix replayed exactly under teacher forcing. The mean horizon was 210.40
tokens (median 219), compared with 280.02 (279) for the previous SPS-CE run.
Matching by document hash gave a mean change of −69.625 tokens and a median
change of −52.5. The horizon fell on 47 documents. The total update limit
stopped 46 texts. The per-stage limit stopped texts 31 and 40 at horizons of
5 and 7 tokens, respectively.

The mean initial correct greedy prefix increased from 106.96 to 159.17 tokens,
with gains on 33 documents. Greedy decoding reproduced the entire selected
prefix on 30/48 texts, up from 5/48. Original Qwen AR reached a mean horizon
of 295.06 and a mean greedy prefix of 168.67, with 15/48 exact greedy prefixes.
The audit outputs and the document-level paired comparison are under
`analysis/932925bc7978/qwen_ce_50k/` in the pretrained cramming artifact root.

## Conclusions

The longer-horizon hypothesis failed on this cohort at the tested budget.
The 50k checkpoint had a shorter teacher-forced horizon and generated more
correct consecutive tokens on average than the previous SPS-CE checkpoint.
Each model decoded its own selected prefix, so exact greedy counts refer to
different target lengths. A fixed-length comparison would isolate greedy
recovery at equal target length.

## Changelog

- 2026-11-20: Plan the paired comparison and checkpoint-specific preflight.
- 2026-11-20: Verify paired inputs and submit the targeted GPU preflight.
- 2026-11-20: Audit all shards and record the paired CE comparison and original AR reference.

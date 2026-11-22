# Exact greedy recovery of pretrained SPS prefixes

## Hypothesis

Changing the forward shape can flip the winning token on an identical history.
Keeping the teacher-replay shape during blind decoding can recover those prefixes
using the original shared prediction vector.

## Setup

The source vectors come from `scripts.pretrained_sps_worker.train_shard`.
Use `python -m scripts.sps_codec probe SHARD SAMPLE_ID` to compare the first
archived mismatch with a growing prefix, a fixed BOS-filled suffix, and a full
teacher-forced input. The probe scores the same position and history in each.
Positions in the output are zero-based.

Run `python -m scripts.sps_codec certify SHARD SAMPLE_ID PACKET.pt` and then
`python -m scripts.sps_codec decode PACKET.pt` to print recovered token IDs.
Certification requires an exact blind replay of the selected saved stage. The
packet carries the original vector, the decoder and model identifiers, the
target length and a SHA-256 digest. Decoding needs the registered checkpoint.

## Results

On the six Qwen SPS-CE texts in the first archived shard, growing-shape
generation first diverged after 84, 109, 134, 31, 143 and 54 correct tokens,
respectively. On A100 in BF16 the growing-shape probe reproduced the archived
winner on all six identical histories. In every case the target and winning
competitor had equal logits, and the lower token ID won the argmax tie. With
a blind BOS suffix at the full replay shape, the target beat that competitor
by 0.1875, 0.0625, 0.125, 0.0625, 0.25 and 0.1875, respectively. The full
teacher inputs gave the same winners and the same six pairwise margins.

Blind fixed-shape greedy replay exactly recovered 6/6 selected prefixes with
their original vectors (1,626 tokens total, lengths 230–299). The archived
growing-shape decoder recovered 0/6 complete prefixes. This is one shard
of one checkpoint. The comparisons localize a shape-dependent numerical effect;
they do not isolate a particular GPU kernel or reduction.

## Conclusions

For a deterministic causal forward, the teacher-forced winner at position zero
is the first greedy token. If the first k greedy tokens agree, the scored row
sees the same history as teacher forcing, so its winner agrees as well, provided
the two computations give the same argmax. Fixed shape preserves the numerical
execution geometry; certification directly checks that condition along the
entire generated trajectory. Exact replay on the same deterministic runtime
follows from the same initial state and transition at every step. Verification
on read detects a changed token sequence after a runtime or checkpoint change.

## Changelog

- 2026-11-23. Add the fixed-shape decoder, first-error probe and certified export.

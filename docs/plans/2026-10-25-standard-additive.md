# Standard additive cramming implementation plan

**Goal:** test the user's input `[z, E(x1)+z, E(x2)+z, ...]` with one
trainable vector and the frozen native Standard checkpoint.

**Design:** specialize the existing progressive trainer and evaluation in a
separate experiment script. Keep ordinary causal attention. Reuse the completed
Standard strict-hinge control's settings, exact cached texts and saved initial
vectors, so sharding cannot change the paired initialization. No scaling or
additional learned parameters. Reuse the existing MLS launcher and storage.

**Alternatives:** a separate vector at each position changes the parameter
budget; adding a vector only at the output changes the requested intervention.
Neither is included. The matched original budget is the first comparison;
an extended schedule would introduce another experimental variable.

**Implementation:** `scripts/analyze_standard_additive.py` owns the trainer,
conditioning/evaluation functions, experiment registry, shard launch and merge.
`tests/test_standard_additive.py` exercises the real model/trainer and saved
artifact boundary. `experiments/standard_additive.md` records the hypothesis
and outcomes; design settings remain in Python.

- [x] Write failing tests for input addition, gradient sharing, teacher/greedy
  consistency, EOS accounting, frozen weights, paired initialization and offsets.
- [x] Implement the specialized trainer and evaluation; run the focused tests.
- [x] Test disjoint shard launch and merge integrity; implement the registry,
  snapshot submission and merged paired comparison.
- [x] Verify the complete suite and real-checkpoint preflight, commit, then
  invoke a fresh portable adversarial review and address findings.
- [x] Submit the authorized shards, preserve full job IDs and register a new
  durable wait.
- [ ] Merge, compare and plot when the group resolves.

**Review focus:** first position is z rather than 2z; each later input receives
z exactly once; generation reconstructs raw token embeddings at each step;
no cross-text gradients; offsets and initialization stay paired; partial or
mixed-snapshot shards cannot produce a completed merge.

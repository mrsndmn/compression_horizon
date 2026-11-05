# Token Cramming through Shared Prediction Embeddings

This directory contains a separate working preprint about the completed
Standard/SPS pilot and pretrained-adaptation experiments. It follows the existing `paper/` LaTeX layout and keeps
that article's source and results independent.

The main finding concerns the **conditioning interface**: one shared prediction
input vector substantially increases achieved reconstruction lengths in the same
frozen SPS checkpoint. The manuscript distinguishes this result from an
architecture-only advantage, a global capacity bound, and semantic transfer.

## Contents

- `main.tex`, `appendix.tex`, `references.bib`: manuscript and primary references.
- `styles/`, `.latexmkrc`: the existing article's ICML style and local build setup;
  the article uses the preprint option, without claiming venue acceptance.
- `reproduce.py`: evidence export, audited statistics, generated tables/figures,
  and commands for historical experiment replay.
- `pretrained.py`: full-cohort export, offline verification/assets and executable
  replay for the pretrained Qwen3/SmolLM2 follow-up.
- `data/`: versioned measurement snapshots, exact token IDs and provenance.
  These are measured outputs, not a second experiment-design registry.
- `tables/`, `figures/`: generated publication assets; edit their generator.
- `check.py`: checks local figure/input/bibliography references, finite table
  values and the separation of body text from generated tables.
- `build/main.pdf`: compiled article, ignored by Git.

## Build and regenerate

From the repository root, in the project's existing Python environment:

```sh
make sps-paper
```

The build first recomputes the tables and figures from `data/`, then checks the
manuscript and runs LaTeX/BibTeX. No checkpoint, GPU, network access or shared
artifact mount is needed for this path. Python requires NumPy and Matplotlib;
the PDF build additionally requires `latexmk`, `pdflatex`, BibTeX and the LaTeX
packages used by the vendored ICML style.

For a portable checkout with those commands on PATH:

```sh
make -C sps_paper PYTHON=python LATEXMK=latexmk
```

Individual stages:

```sh
python sps_paper/reproduce.py render
make sps-paper-check
make sps-paper-clean
```

Alternative data/output locations can be supplied to `render` with
`--data-root` and `--output-root`. This allows regenerating the assets in a
temporary directory and comparing them with the committed files.

## Refresh the measured evidence

On the research filesystem containing the completed runs:

```sh
python sps_paper/reproduce.py export \
  --artifacts-root /path/to/compression_horizon/artifacts/sps_cramming
python sps_paper/reproduce.py render
```

Export audits successful completion evidence, registered cohorts, exact target
IDs, per-sample metrics and saved stages before saving compact measurements.
The snapshot records source commits, full MLspace job UUIDs and integrity hashes.
Rendering checks the saved evidence and derives aggregate values from it; it
does not substitute handwritten numbers from the manuscript.

## Repeat the underlying experiments

Asset regeneration and model-level reproduction are separate operations. Print
the historical replay plan without executing training:

```sh
python sps_paper/reproduce.py plan > /tmp/replay_sps_experiments.sh
```

Inspect the generated script and supply its checkpoint and fresh-output
prerequisites before running it on GPU infrastructure. For example, from the
repository root:

```sh
export STANDARD_CHECKPOINT=/absolute/path/to/standard/ckpt_tokens_1000341504_final.pt
export SPS_CHECKPOINT=/absolute/path/to/sps/ckpt_tokens_1000341504_final.pt
export REPLAY_ROOT=/absolute/path/to/a/new/replay_directory
# After reviewing the script and preparing the archived Python dependencies:
bash /tmp/replay_sps_experiments.sh
```

The plan reconstructs the exact selected tokenized cohort from the bundled IDs,
without retokenizing. It pins original experiment source commits and uses their
Python registries rather than retyping training settings. The two original
single-sample baselines are included as attention-ablation prerequisites.
Fresh training outputs are linked under `REPLAY_ROOT/upstream/` using the
historical input registry names; diagnostics consume these outputs with the
supplied checkpoint and cache paths. Rescue runs before the dependent gradient
audit. Original optimized-vector artifacts are not required. The script does
not submit MLspace jobs. Historical
CE/random controls are included for reproducing reported results; the current
launch policy for new shared-predict experiments remains hinge/native and
single-sample cramming.

The native `.pt` checkpoints are external research inputs and are **not
distributed** in this directory. `data/cohort.json.gz` contains the exact selected
input token IDs; the original cache files are identified by provenance hashes.
The compact offline bundle preserves measurements and vector hashes, not the
full optimized-vector datasets or model weights. The replay plan reconstructs
the vectors by training the upstream experiments before running their probes.
Rebuilding publication assets and loading the fresh diagnostic inputs are
tested here; a fresh end-to-end GPU retraining of every historical experiment
is not claimed.

## Interpretation and editing

The matched full experiments use the same 48 texts. Diagnostic subsets and the
historical batch-shape audit have different scopes and are labeled separately.
Teacher-forced horizon, initial correct greedy prefix and exact reconstruction
of the selected prefix are distinct measurements. Bootstrap intervals quantify
variation across the measured texts at fixed checkpoints and seed. A failed
stage with insufficient positive margins is not a proof of infeasibility, and
its margin-violation count is not necessarily an argmax-error count.

All experiment selection, model paths, hyperparameters and job settings remain
in the original training/diagnostic Python source. The replay manifest records
historical provenance. Add new experiments to source before exporting their
completed evidence; do not edit generated tables or invent new results here.

## Completed pretrained SPS follow-up

The pretrained Qwen3/SmolLM2 CE/KD comparison is defined by
`scripts/pretrained_sps_experiments.py`, with execution/replay steps in
`docs/plans/2026-11-05-pretrained-sps-cramming.md`. It uses the native source
tokenizers and a separate pretrained adapter; historical GPT-2 cohort token IDs
are not inputs for these models. All six cohorts have completed and passed the
raw-stage audit. Their separate `data/pretrained.json.gz` snapshot contains exact
raw-document identities, family-native target IDs and masks, selected token-level
validation, scalar stages, and complete job/source provenance. Its checksum and
source-file hashes are in `data/pretrained_manifest.json`. Model weights and full
optimized-vector trajectories remain external.

The follow-up finds longer teacher-forced prefixes for both SmolLM SPS variants
than their original AR control, while Qwen favors its AR control. Greedy results
have a different ordering. Every text exhausts the total update budget; the
measurements do not establish capacity saturation or isolate architecture from
adaptation, insertion site and initialization. Compare native-token lengths within
families. CE/KD name the adaptation recipes; cramming uses the source-registry
objective for every arm.

From the repository root:

```sh
# Refresh only from all completed, audited external cohorts:
python sps_paper/pretrained.py export
# These commands require neither shared artifacts, weights nor GPU:
python sps_paper/pretrained.py verify
python sps_paper/pretrained.py render
# The normal article build also renders/verifies this evidence:
make sps-paper
```

`--data-root` and `--output-root` allow offline rendering in a temporary directory.
The historical pilot snapshots and `reproduce.py` routes remain independent.

Generate an executable **serial GPU replay** without submitting jobs:

```sh
python sps_paper/pretrained.py plan > /tmp/replay_pretrained_sps.sh
export REPLAY_ROOT=/absolute/path/to/a/new/pretrained_replay
export PRETRAINED_SOURCE_ROOT=/absolute/path/to/sps-artifacts
# Prepare archived dependencies, pinned native HF snapshots, original/adapted
# weights and a GPU, then inspect the plan before executing it:
bash /tmp/replay_pretrained_sps.sh
```

`PRETRAINED_SOURCE_ROOT` preserves the registry's checkpoint/cache subdirectory
layout. The plan archives original commits, restores both exact bundled token
caches, runs the family preflights and dispatches each archived `train_shard`.
It imports scientific settings from archived registries and relocates paths only.
`restore-data` refuses to overwrite an existing cache tree, and `replay` refuses
existing shard outputs. CPU tests verify restored IDs/masks and archived-worker
dispatch. A second full GPU training of all six cohorts is not claimed. This
direct replay creates fresh worker outputs without inventing new MLspace job IDs.

# Thesis reproduction

Reproduces every paper anchor from *Progressive Cramming: Reliable Token
Compression and What It Reveals* (ICML 2026 submission) on the
post-refactor library code, plus two additional experiments that test the
paper's manifold hypothesis (Section 4.3) directly.

Used as a **verification harness** for the thesis (ВКР): the refactor
preserves the paper's math iff every measured metric here lands within 2σ
of the corresponding paper value.

## Layout

```
scripts/thesis_reproduction/
├── README.md                        # this file
├── train.py                         # clean training entry-point
├── analyze.py                       # loads saved Dataset, compares to expected.json,
│                                    #   writes analysis_summary.json next to it
├── summarize_verification.py        # walks all analysis_summary.json files,
│                                    #   prints aggregate Markdown report
├── run_attention_hijacking.py
├── run_attention_knockout.py
├── run_downstream_eval.py
├── run_pca_reconstruction.py
├── run_trajectory.py
├── summarize_downstream_table10.py
├── expected.json                    # paper-anchor values per experiment key
└── experiments/
    ├── full_cramming/
    │   ├── pythia_160m.sh
    │   └── llama_3_2_1b.sh
    ├── progressive/
    │   ├── smollm2_135m.sh
    │   └── smollm2_1_7b.sh
    ├── progressive_alignment/
    │   └── smollm2_1_7b.sh
    ├── progressive_lowdim/                # Table 6 dim=256 (paper-faithful)
    │   └── smollm2_1_7b.sh
    ├── progressive_lowdim_global/         # additional: shared basis on whole corpus
    │   └── smollm2_1_7b.sh
    ├── progressive_lowdim_transfer/       # additional: frozen basis on disjoint slice
    │   └── smollm2_1_7b.sh
    ├── trajectory/
    │   └── smollm2_135m_progressive.sh
    ├── pca_reconstruction/
    │   └── smollm2_135m_progressive.sh
    ├── attention_hijacking/
    │   └── smollm2_135m_progressive.sh
    ├── attention_knockout/
    │   └── smollm2_135m_progressive.sh
    └── downstream/
        ├── smollm2_135m_{hellaswag,arc_easy,arc_challenge}.sh
        ├── smollm2_1_7b_{hellaswag,arc_easy,arc_challenge}.sh
        ├── run_all_smollm2_135m.sh
        └── run_all_smollm2_1_7b.sh
```

Each experiment has a stable key `<family>/<model>`, a paired entry in
`expected.json`, and writes its results to
`artifacts/thesis_reproduction/<family>/<model>/`.

## How a single experiment works

1. `experiments/<family>/<model>.sh` invokes `train.py` with paper-faithful
   parameters, then runs `analyze.py` against the saved Dataset.
2. `train.py` writes:
   * `progressive_prefixes/` or `compressed_prefixes/` — HF Dataset with
     all per-sample rows (the raw numerical results),
   * `events.out.tfevents.*` — TensorBoard scalars,
   * `low_dim_projection.pt` (low-dim experiments only) — trained Linear's
     state_dict.
3. `analyze.py` loads the Dataset, computes mean ± std per metric,
   compares to `expected.json[<key>]`, prints a 4-column table, and writes
   `analysis_summary.json` next to the raw data.
4. The shell script pipes everything through `tee` to `train.log` so the
   terminal-output is preserved even if the session closes.

`analyze.py` verdicts:
* `OK` — within 2σ of the paper mean (or zero deviation for fixed-budget metrics),
* `WARN` — within 3σ but outside 2σ,
* `FAIL` — beyond 3σ; the refactor needs investigation.

## How to verify the whole reproduction at once

After running individual experiments:

```bash
uv run python scripts/thesis_reproduction/summarize_verification.py
# → artifacts/thesis_reproduction/verification_summary.md
```

This walks all `analysis_summary.json` files and aggregates them into one
Markdown table with per-metric verdicts. Useful for thesis Q&A and as a
single artefact to point reviewers at.

## Currently included (18 experiment keys)

| Experiment key | Paper anchor | Trainer | Samples | A100 80GB time |
|---|---|---|---|---|
| `full_cramming/pythia_160m` | Table 11 (Appendix C) | full | 50 | ~5-10 min |
| `full_cramming/llama_3_2_1b` | Table 11 (Appendix C) | full | 10 | ~15-25 min |
| `progressive/smollm2_135m` | Table 13 (Appendix F) | progressive | 50 | ~15-30 min |
| `progressive/smollm2_1_7b` | Table 6 baseline | progressive | 50 | ~2-3 h |
| `progressive_alignment/smollm2_1_7b` | Table 6 + Section 4.2/5.4 | progressive | 50 | ~2-4 h |
| `progressive_lowdim/smollm2_1_7b` | Table 6 dim=256 (paper-faithful) | progressive | 50 | ~6-10 h |
| `progressive_lowdim_global/smollm2_1_7b` | Additional — shared basis (manifold hypothesis test) | progressive | 50 | ~2-4 h |
| `progressive_lowdim_transfer/smollm2_1_7b` | Additional — frozen basis transfer to disjoint slice | progressive | 50 | ~6-10 h |
| `trajectory/smollm2_135m_progressive` | Table 13, Trajectory Length + PCA 99% | post-hoc | — | ~5 min |
| `pca_reconstruction/smollm2_135m_progressive` | Section 5.3, Figure 5 | post-hoc | — | ~5 min |
| `attention_hijacking/smollm2_135m_progressive` | Table 3 (Section 5.5) | post-hoc | — | ~10-20 min |
| `attention_knockout/smollm2_135m_progressive` | Reviewer 1 W2 / Section 4.4 | post-hoc | — | ~20-30 min |
| `downstream/smollm2_135m_{hellaswag,arc_easy,arc_challenge}` | Table 10 perfect-subset | downstream eval | varies | ~20-40 min each |
| `downstream/smollm2_1_7b_{hellaswag,arc_easy,arc_challenge}` | Table 10 same-model | downstream eval | varies | ~30-60 min each |

The three low-dim experiments form a small experimental subgraph for the
defense (none of them goes into the thesis text — the text is a conversion
of the paper itself — but they together test the paper's central claim
cleanly):

1. **`progressive_lowdim/...`** — per-sample basis. Direct Table 6
   reproduction (bs=1 + `--low_dim_projection`, no global flag — matches
   pre-refactor `run_jobs_progressive.py:230`).
2. **`progressive_lowdim_global/...`** — one Linear shared across all 50
   PG19 samples (mini-batch averaged at bs=25). Tests whether a single
   256-dim basis suffices for the corpus.
3. **`progressive_lowdim_transfer/...`** — loads the basis from (2) with
   `requires_grad=False`, optimizes only `z_j` on PG19 [50, 100). Tests
   whether the basis transfers to unseen texts.

## How to run a single experiment

```bash
# (Optional) Install flash-attn for closest-possible match to paper. Without
# it, the trainer auto-falls-back to PyTorch sdpa (bf16-close, not identical).
uv pip install flash-attn --no-build-isolation

# Train + analyze in one shot.
bash scripts/thesis_reproduction/experiments/progressive/smollm2_1_7b.sh

# Or re-analyze an already-trained run (reads the saved Dataset off disk).
uv run python scripts/thesis_reproduction/analyze.py \
    --experiment progressive/smollm2_1_7b
```

## Adding a new experiment

1. Append a new entry to `expected.json` keyed `<family>/<model>` with
   either `expected.<metric>` (mean / std / kind) for the generic
   analyzer, or a custom `analyzer` field if it needs a specialized one.
2. Write `experiments/<family>/<model>.sh` following the existing pattern
   (set `EXPERIMENT`, `OUTPUT_DIR`, wrap with `tee` to `train.log`).
3. Run, verify the output of `analyze.py`, then re-run
   `summarize_verification.py` to refresh the aggregate report.

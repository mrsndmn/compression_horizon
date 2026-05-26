"""Launch Q-Former compression-head training (+ progressive eval) for the WIDTH ablation.

Thin companion to ``scripts/jobs/run_jobs_compression_head.py``: it reuses that
module's command rendering, payload, idempotency, and job-description helpers
verbatim (so output-dir names and eval seeding stay byte-identical) and only swaps
the experiment matrix to the three model-width checkpoints --
``SmolLM2-{135M,360M,1.7B}-firstlast4`` (first-4 + last-4 = 8 layers, hidden sizes
576 / 960 / 2048) -- each trained with the canonical Q-Former config
(num_queries=1, num_layers=3, num_heads=8), lr 1e-3, distill alpha 1.0 / beta 0.0,
fineweb-edu. Unlike the depth ablation, each width trains at its OWN sequence length
(135M @ 64, 360M @ 128, 1.7B @ 512): the eval_paraphrase comparison showed sequence
length is the dominant factor, and the small models need short prefixes. Per-device
batch is scaled per width to keep ~4096 tokens/device, and the global batch is held at
256 sequences (so ~5k optimizer steps); the only thing varying across widths is seq_len.

Because each width now trains at its own sequence length, the output-dir names carry
``seq_{64,128,512}`` and do NOT collide with the depth ablation's seq_1024 entries --
every head here is trained fresh (single-model and dual-model arms alike).

Each width is run in two arms -- a single-model head and a dual-model head
(``--separate_reconstructor_model``: separate compressor + reconstructor so their
gradients never overlap, saved under ``compressor/`` + ``reconstructor/`` and tagged
``_dualmodel``) -- so the matrix is 3 widths x 2 arms = 6 heads.

Two stages, identical semantics to the depth launcher:
* ``--stage train`` (default): 4-GPU compression-head training (NUM_GPUS overridden to 4;
  per-width per-device batch + derived grad_accum keep the global batch at 256 sequences).
* ``--stage eval``: 1-GPU progressive cramming of each finished head, seeded via
  ``--embedding_init_method compression_head_forward`` on the pg19 benchmark. Dual-model
  heads are auto-detected from their ``compressor/`` + ``reconstructor/`` layout.

Usage:
    python scripts/jobs/run_jobs_compression_head_width.py --dry
    python scripts/jobs/run_jobs_compression_head_width.py --stage train
    python scripts/jobs/run_jobs_compression_head_width.py --stage eval
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_compression_head as J  # noqa: E402
from mls.manager.job.utils import get_in_progress_jobs  # noqa: E402

# Re-export the shared helpers so watch_width_ablation.py can drive this arm the
# same way watch_compression_head_eval.py drives the depth arm.
render_ch_job = J.render_ch_job
render_eval_job = J.render_eval_job
build_job = J.build_job
checkpoint_ready = J.checkpoint_ready
ch_job_desc = J.ch_job_desc
eval_job_desc = J.eval_job_desc
make_client = J.make_client

# Run the WIDTH ablation on 4 GPUs (not the shared 8-GPU default) while preserving the global batch.
# build_job/render_ch_job read NUM_GPUS / PER_DEVICE_TRAIN_BATCH_SIZE / TARGET_GLOBAL_TOKENS as module
# globals on J, and compute_grad_accum derives gradient_accumulation_steps from them; halving NUM_GPUS
# (8 -> 4) doubles grad_accum (8 -> 16) so the global batch
# (per_device * grad_accum * num_gpus * seq_len == TARGET_GLOBAL_TOKENS) is unchanged. The override is
# process-local: other launchers that import J still see the 8-GPU default. build_job also requests the
# a100.4gpu instance from this same NUM_GPUS.
J.NUM_GPUS = 4

WIDTH_CHECKPOINT_ROOT = "artifacts/checkpoints"
# First-4 + last-4 (= 8 layers) checkpoints across model widths, each with its own TRAINING sequence
# length. Compression saturates differently by width, so wider models train at longer prefixes; the
# debug comparison against eval_paraphrase showed sequence length is the dominant factor (the earlier
# uniform seq=1024 runs were the wrong regime for the small models). Per-device batch is scaled so
# every run keeps the same ~4096-token-per-device footprint (validated by the prior seq=1024/bs=4 runs),
# and the global batch stays at 256 sequences (target_global_tokens = 256 * seq_len, grad_accum derived).
WIDTH_TRAIN: dict[str, dict] = {
    f"{WIDTH_CHECKPOINT_ROOT}/SmolLM2-135M-firstlast4": {"max_sequence_length": 64, "per_device_train_batch_size": 64},
    f"{WIDTH_CHECKPOINT_ROOT}/SmolLM2-360M-firstlast4": {"max_sequence_length": 128, "per_device_train_batch_size": 32},
    f"{WIDTH_CHECKPOINT_ROOT}/SmolLM2-1.7B-firstlast4": {"max_sequence_length": 512, "per_device_train_batch_size": 8},
}
WIDTH_CHECKPOINTS = list(WIDTH_TRAIN)


def _train_cfg(checkpoint: str) -> dict:
    """Per-width training overrides read by J.render_ch_job: seq_len, per-device batch, and a token
    target that pins the global batch to 256 sequences regardless of seq_len."""
    cfg = WIDTH_TRAIN[checkpoint]
    return {**cfg, "target_global_tokens": 256 * cfg["max_sequence_length"]}


# Q-Former head on each width checkpoint (same head config as the depth ablation), run in two arms per
# width: the single-model head, and a dual-model head (--separate_reconstructor_model: separate
# compressor + reconstructor so their gradients never overlap), tagged ``_dualmodel`` in the output dir.
_SINGLE_MODEL_EXPERIMENTS: list[dict] = [
    {"model_checkpoint": ck, "head_kind": "qformer", **J.QFORMER, **_train_cfg(ck)} for ck in WIDTH_CHECKPOINTS
]
EXPERIMENTS: list[dict] = [
    *_SINGLE_MODEL_EXPERIMENTS,
    *[{**exp, "separate_reconstructor_model": True} for exp in _SINGLE_MODEL_EXPERIMENTS],
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument(
        "--stage",
        choices=["train", "eval"],
        default="train",
        help="train: launch compression-head training (4 GPU). "
        "eval: launch 1-GPU progressive-cramming evaluation of finished heads.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        help="Filter experiments by model name (substring match against checkpoint path / model name).",
    )
    args = parser.parse_args()

    client, extra_options = make_client()
    in_progress_job_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    experiments = J._filter_experiments(EXPERIMENTS, args.model)
    if not experiments:
        print(f"\033[33mNo models matched the filter: {args.model}\033[0m")
        return 0

    for experiment in experiments:
        built = build_job(experiment, args.stage)
        if built is None:
            print(f"\033[33mCompression head not trained yet, skip eval:\033[0m {render_ch_job(experiment)[2]}")
            continue
        script, job_desc, out_dir_name, instance_type = built

        already_done = checkpoint_ready(out_dir_name) if args.stage == "train" else os.path.exists(out_dir_name)
        if already_done:
            print(f"{'Experiment' if args.stage == 'train' else 'Eval'} {out_dir_name} exists, skip.")
            continue
        if job_desc in in_progress_job_descs:
            print(f"\033[33mSkipping: job already in queue with description:\033[0m {job_desc}")
            continue

        payload = J._payload(script, job_desc, instance_type, extra_options["region"])
        print(f"\033[32m Would launch with description:\033[0m {job_desc}")
        print(f"\033[90m     Command: {script}\033[0m")
        print(f"\033[90m     Output dir: {out_dir_name}\033[0m")

        if args.dry:
            continue

        result = client.run_job(payload=payload)
        print(out_dir_name, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

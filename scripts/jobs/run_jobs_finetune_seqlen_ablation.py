"""Causal-LM finetuning SEQUENCE-LENGTH ablation (does the finetune seq_len affect compression?).

The width ablation's causal-LM arm finetunes ``SmolLM2-1.7B-firstlast4`` at seq 1024 / lr 1e-3
(``-ftw``) and then measures progressive-cramming compression on PG19 -- that run already exists as
``sl_4096_SmolLM2-1.7B-firstlast4-ftw_ds_pg19_1k_limit_50_lr_0.1`` (428.6 compressed tokens). This
ablation re-finetunes the SAME checkpoint with everything held fixed except the training sequence
length (and learning rate), then re-evaluates compression the same way, to isolate the effect of the
finetuning sequence length. The four variants (relative to the seq 1024 / lr 1e-3 baseline):

    seq 512  (1/2x), lr 5e-4 (1/2x)
    seq 512  (1/2x), lr 1e-3 (same)
    seq 2048 (2x),   lr 2e-3 (2x)
    seq 2048 (2x),   lr 1e-3 (same)

Everything else matches ``run_jobs_finetune_width.py`` (fineweb-edu, 5k steps, wd 0.01,
cosine-with-min-lr, warmup 500, bf16, 8 GPUs). The global batch is held at 256 sequences and the
per-device batch is scaled so every variant keeps the same ~8192-token-per-device footprint as the
baseline (per_device = 8192 // seq_len; grad_accum derived). Each variant finetunes into a distinct
``…-ftw-seq{S}-lr{L}`` checkpoint, then is progressively evaluated with the standard baseline init
(``run_jobs_width_ablation_ft`` machinery: random0.02, lr 0.1, cross_entropy, align 1), producing a
``sl_4096_…-ftw-seq{S}-lr{L}_ds_pg19_1k_limit_50_lr_0.1`` dir directly comparable to the baseline.

Two stages (idempotent; safe to re-run):
* ``--stage train`` (default): 8-GPU finetuning of the four variants.
* ``--stage eval``: 1-GPU progressive cramming of each finished variant checkpoint.

Usage:
    python scripts/jobs/run_jobs_finetune_seqlen_ablation.py --dry
    python scripts/jobs/run_jobs_finetune_seqlen_ablation.py --stage train
    python scripts/jobs/run_jobs_finetune_seqlen_ablation.py --stage eval
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_finetune_width as FW  # noqa: E402  (finetune payload + submit + make_client)
import run_jobs_width_ablation_ft as EW  # noqa: E402  (progressive-eval submit via run_jobs_progressive.render_job)
from mls.manager.job.utils import get_in_progress_jobs  # noqa: E402

# The ablation finetunes this one checkpoint (the width ablation's 1.7B arm) under several configs.
BASE_CHECKPOINT = "artifacts/checkpoints/SmolLM2-1.7B-firstlast4"
BASE_SEQ_LEN = FW.DEFAULTS["max_sequence_length"]  # 1024
# Baseline per-device token footprint (per_device 8 x seq 1024); kept constant across variants so
# memory use matches the baseline and only the effective seq_len/lr change.
TOKENS_PER_DEVICE = FW.DEFAULTS["per_device_train_batch_size"] * BASE_SEQ_LEN  # 8192

# Four variants relative to the seq 1024 / lr 1e-3 baseline (which already exists, so is not re-run).
VARIANTS = [
    {"max_sequence_length": 512, "learning_rate": 0.0005},  # 1/2x seq, 1/2x lr
    {"max_sequence_length": 512, "learning_rate": 0.001},  # 1/2x seq, same lr
    {"max_sequence_length": 2048, "learning_rate": 0.002},  # 2x seq, 2x lr
    {"max_sequence_length": 2048, "learning_rate": 0.001},  # 2x seq, same lr
]


def _lr_label(lr: float) -> str:
    """Filesystem-safe LR tag: 0.0005 -> '5em4', 0.001 -> '1em3', 0.002 -> '2em3'."""
    mantissa, exponent = f"{lr:.0e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}em{abs(int(exponent))}"


def variant_dir(variant: dict) -> str:
    """Distinct finetuned-checkpoint dir per (seq, lr) variant."""
    return f"{BASE_CHECKPOINT}-ftw-seq{variant['max_sequence_length']}-lr{_lr_label(variant['learning_rate'])}"


def variant_opts(variant: dict) -> dict:
    """Finetune opts = baseline defaults with this variant's seq_len + lr, and per_device scaled to
    hold the per-device token footprint (and thus the 256-sequence global batch) constant."""
    seq_len = variant["max_sequence_length"]
    per_device = TOKENS_PER_DEVICE // seq_len
    return {
        **FW.DEFAULTS,
        "max_sequence_length": seq_len,
        "learning_rate": variant["learning_rate"],
        "per_device_train_batch_size": per_device,
    }


def eval_experiment(variant: dict) -> dict:
    """Progressive-eval spec for a variant checkpoint (same baseline init as the width-ft arm)."""
    return {
        "model_checkpoint": variant_dir(variant),
        "learning_rate": 0.1,
        "loss_type": "cross_entropy",
        "num_alignment_layers": 1,
        "hybrid_alpha": None,
        "low_dim_projection": False,
        "low_dim_size": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry", action="store_true", help="Only print generated scripts, do not launch jobs.")
    parser.add_argument(
        "--stage",
        choices=["train", "eval"],
        default="train",
        help="train: 8-GPU finetuning of the four seq/lr variants. eval: 1-GPU progressive cramming of "
        "each finished variant checkpoint.",
    )
    parser.add_argument("--force", action="store_true", help="Resubmit even if the output dir already exists.")
    args = parser.parse_args()

    client, extra_options = FW.make_client()
    in_progress_descs = {job.get("job_desc", "") for job in get_in_progress_jobs()}

    for variant in VARIANTS:
        out_dir = variant_dir(variant)
        if args.stage == "train":
            FW.submit_experiment(
                BASE_CHECKPOINT,
                client,
                extra_options,
                variant_opts(variant),
                in_progress_descs,
                dry=args.dry,
                force=args.force,
                out_dir=out_dir,
                job_label=os.path.basename(out_dir),
            )
        else:  # eval
            EW.submit_experiment(
                eval_experiment(variant), client, extra_options, in_progress_descs, dry=args.dry, force=args.force
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

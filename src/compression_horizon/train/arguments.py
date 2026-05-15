from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from transformers import SchedulerType, TrainingArguments


def _parse_cli_dict(value: Any) -> dict[str, Any] | None:
    """
    Parse dict-like CLI values.

    Supported forms:
    - JSON object string: '{"min_lr": 0.0001}'
    - Comma-separated key=value: 'min_lr=0.0001,foo="bar"'
    - None / "" / "null" / "none" -> None
    - dict -> unchanged
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected dict or str for dict-like CLI arg, got {type(value)}")

    s = value.strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None

    # Prefer JSON for unambiguous typing.
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object for dict-like CLI arg, got {type(parsed)}")
        return parsed

    # Fallback: key=value[,key=value...] (best-effort typing via json.loads on values).
    result: dict[str, Any] = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid dict-like CLI arg chunk (expected key=value): {chunk!r}")
        key, raw_val = chunk.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if key == "":
            raise ValueError(f"Invalid dict-like CLI arg chunk (empty key): {chunk!r}")
        try:
            val = json.loads(raw_val)
        except json.JSONDecodeError:
            val = raw_val
        result[key] = val
    return result


@dataclass
class MyTrainingArguments(TrainingArguments):
    """Training arguments for tokens compression experiments."""

    # Core compression arguments
    model_checkpoint: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        metadata={"help": "Huggingface location for a model and a tokenizer."},
    )
    dataset_name: str = field(
        default="mrsndmn/pg19",
        metadata={"help": "Dataset name to use for training (e.g., 'mrsndmn/pg19')."},
    )
    low_dim_projection: bool = field(
        default=False,
        metadata={"help": "Low dim projection flag"},
    )
    low_dim_projection_global: bool = field(
        default=False,
        metadata={"help": "when to optimize projection"},
    )
    low_dim_size: int = field(
        default=32,
        metadata={"help": "Dimension of small space for embeddings regularization"},
    )
    low_dim_proj_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Path to checkpoint file to load low-dimensional projection state from."},
    )
    low_dim_proj_train: bool = field(
        default=True,
        metadata={"help": "Whether to optimize the low-dimensional projection (False to freeze it)."},
    )

    number_of_mem_tokens: int = field(
        default=1,
        metadata={"help": "Number of trainable [mem] tokens for a single sample."},
    )
    embedding_init_method: str = field(
        default="random",
        metadata={
            "help": 'Initialization method for compression embeddings: "random", "mvnormal", "pretrained_pca", "load_from_disk"'
        },
    )
    embedding_init_path: str = field(
        default="",
        metadata={
            "help": "Path to file containing initial compression embeddings (when embedding_init_method=load_from_disk). "
            "If empty, embeddings will be generated using load_from_disk_embedding_init_method and saved. "
            "File should contain a tensor of shape [num_tokens, hidden_size] or [1, num_tokens, hidden_size] or [batch_size, num_tokens, hidden_size]."
        },
    )
    load_from_disk_embedding_init_method: str = field(
        default="random",
        metadata={
            "help": "Initialization method to use when generating embeddings for load_from_disk (when embedding_init_path is empty). "
            'Can be any valid embedding_init_method value (e.g., "random", "random0.02", "zeros", etc.).'
        },
    )
    pretrained_pca_num_components: int = field(
        default=16,
        metadata={"help": "Number of PCA components to use when embedding_init_method=pretrained_pca."},
    )
    pretrained_pca_path: str = field(
        default="",
        metadata={
            "help": "Path to progressive_prefixes dataset for PCA initialization (when embedding_init_method=pretrained_pca)."
        },
    )
    fix_position_ids: bool = field(
        default=False,
        metadata={"help": "Whether position_ids should be adjusted relative to compression embeddings."},
    )
    max_optimization_steps_per_sample: int = field(
        default=1_000,
        metadata={"help": "Max optimization steps for training a single sample."},
    )
    random_seed: int | None = field(default=42, metadata={"help": "Random seed for reproducibility (None to skip)."})

    # Alignment arguments
    loss_type: str = field(
        default="l2",
        metadata={"help": "Loss type for activation alignment: l2, l1, or cosine."},
    )
    hybrid_alpha: float | None = field(
        default=None,
        metadata={
            "help": "Multiplier in the loss function for l2, l1, or cosine, hybrid loss is applied in training when specified."
        },
    )
    num_alignment_layers: int = field(default=0, metadata={"help": "Number of transformer layers to align (0 = all)."})
    inverted_alignment: bool = field(
        default=False,
        metadata={
            "help": "Direction of taking transformer layers, True - from depth to shallow, False - from shallow to depth."
        },
    )
    max_sequence_length: int = field(
        default=128,
        metadata={"help": "Max sequence length for compressing in training."},
    )
    max_optimization_steps_per_sample: int = field(
        default=1_000,
        metadata={"help": "Max optimization steps for training 1 sample."},
    )
    max_optimization_steps_per_token: int = field(
        default=1_000,
        metadata={"help": "Max optimization steps for training 1 token (only applicable for progressive training)."},
    )
    random_seed: int | None = field(default=42, metadata={"help": "Random seed for reproducibility (None to skip)."})
    fix_position_ids: bool = field(
        default=False,
    )
    generate_in_compute_loss: bool = field(
        default=False,
    )
    limit_dataset_items: int | None = field(default=1)
    offset_dataset_items: int | None = field(
        default=None,
        metadata={"help": "Offset for dataset items selection (applied before limit_dataset_items)."},
    )
    no_bos_token: bool = field(
        default=False,
        metadata={"help": "Disable BOS token insertion during dataset tokenization."},
    )

    # Overrides with changed defaults
    optim: str = field(
        default="adamw_torch",
    )

    # Overrides with changed defaults
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device accelerator core/CPU for training."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    dataloader_drop_last: bool = field(
        default=True,
        metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."},
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    learning_rate: float = field(default=1e-2, metadata={"help": "The initial learning rate for an optimizer."})
    adam_beta1: float = 0.9
    adam_beta2: float = 0.9
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for an optimizer if we apply some."},
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    lr_scheduler_type: SchedulerType | str = field(
        default="cosine_with_min_lr",
        metadata={"help": "The scheduler type to use."},
    )
    # NOTE: Keep this as `str` for CLI parsing. It is converted to `dict` in __post_init__.
    lr_scheduler_kwargs: str = field(
        default_factory=lambda: {"min_lr": 1e-3},
        metadata={
            "help": (
                "Additional keyword arguments to pass to the learning rate scheduler. "
                "Pass as JSON (recommended), e.g. --lr_scheduler_kwargs '{\"min_lr\": 0.0001}', "
                "or as key=value pairs, e.g. --lr_scheduler_kwargs 'min_lr=0.0001'."
            )
        },
    )
    ddp_find_unused_parameters: bool | None = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )

    # TODO
    low_dim_train: bool = field(default=False, metadata={"help": "Whether to use low_dim training."})
    # Progressive training control
    noop_train: bool = field(default=False, metadata={"help": "Whether to use noop training."})
    noop_convergence_threshold: float = field(
        default=1.0,
        metadata={"help": "Mean token-level match ratio required to mark a stage as converged."},
    )

    progressive_train: bool = field(default=False, metadata={"help": "Whether to use progressive training."})
    train_prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use PEFT prefix tuning training."})
    progressive_min_seq_len: int = field(
        default=1,
        metadata={"help": "Starting effective sequence length for progressive_train."},
    )
    progressive_step: int = field(
        default=1,
        metadata={"help": "Step size to increase effective sequence length between stages."},
    )
    progressive_convergence_threshold: float = field(
        default=1.0,
        metadata={"help": "Mean token-level match ratio required to mark a stage as converged."},
    )
    progressive_max_stages: int = field(
        default=0,
        metadata={"help": "Optional cap on number of progressive stages (0 = no cap)."},
    )
    progressive_reset_lr_scheduler_on_non_convergence: bool = field(
        default=False,
        metadata={"help": "If True, reset LR scheduler and continue training when convergence fails (only once per stage)."},
    )
    progressive_bucketed_compile: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use the bucketed + per-token-mask progressive curriculum that "
                "keeps the base LM input shape constant within a bucket (multiple of 64), "
                "advances a sticky per-position frontier on single-step argmax match, and "
                "computes cumulative cross-entropy over the unmasked prefix. Designed to "
                "play nicely with torch.compile (one compiled graph per bucket size). "
                "Default OFF — legacy seq_len-sliced path is unchanged. "
                "REQUIRES --loss_type cross_entropy (raises NotImplementedError otherwise)."
            )
        },
    )
    progressive_bucket_size: int = field(
        default=64,
        metadata={
            "help": (
                "Initial bucket size for the bucketed progressive curriculum. Must be a "
                "positive multiple of 64. Bucket grows linearly by +64 on each transition."
            )
        },
    )
    progressive_fused_linear_ce: bool = field(
        default=True,
        metadata={
            "help": (
                "If True (default), compute cross-entropy via LigerFusedLinearCrossEntropyLoss "
                "(fused lm_head matmul + CE in a single kernel) instead of materializing "
                "the [B, T, V] logits tensor. Bypasses the LM head in the compiled forward "
                "by calling model.model(...) directly; the LM head weight is applied inside "
                "the Liger kernel. Only effective when --progressive_bucketed_compile is also set. "
                "Pass --progressive_fused_linear_ce False to fall back to the eager logits path."
            )
        },
    )
    save_progressive_artifacts: bool = field(
        default=True,
        metadata={"help": "Whether to persist intermediate compression tokens for each stage."},
    )
    max_tokens_in_distribution: int = field(
        default=1,
        metadata={"help": "Number of top tokens to keep in the distribution target (for train_noop)."},
    )

    # Precision control
    dtype: str = field(
        default="bf16",
        metadata={
            "help": (
                "Torch dtype for model and training. "
                "One of: auto, float32|fp32, bfloat16|bf16, float16|fp16. "
                "This overrides the torch_dtype used to load the model."
            )
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable gradient checkpointing to reduce activation memory during training "
                "(trades memory for extra compute)."
            )
        },
    )

    # Compression head training (single-pass compression embed prediction)
    train_compression_head: bool = field(
        default=False,
        metadata={"help": "Train a compression head (no per-sample embedding optimization)."},
    )
    compression_head_distill_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Weight (alpha) for the compressed-input reconstruction loss in the compression-head training objective."
        },
    )
    compression_head_distill_beta: float = field(
        default=1.0,
        metadata={
            "help": "Weight (beta) for the base next-token loss in the compression-head training objective. Set <1 to deprioritize base_loss relative to after_loss."
        },
    )
    detect_anomaly: bool = field(
        default=False,
        metadata={
            "help": "If True, call torch.autograd.set_detect_anomaly(True) before training so the first NaN/Inf-producing op raises with a precise stack trace. Slow — debug only."
        },
    )
    compression_head_freeze_base_model: bool = field(
        default=True,
        metadata={"help": "Freeze base LM parameters and train only compression head parameters."},
    )

    def __post_init__(self):
        # Convert CLI-friendly forms into a real dict early, before base TrainingArguments validation.
        self.lr_scheduler_kwargs = _parse_cli_dict(getattr(self, "lr_scheduler_kwargs", None))  # type: ignore[assignment]
        super().__post_init__()
        if self.progressive_bucketed_compile:
            if self.progressive_bucket_size <= 0 or self.progressive_bucket_size % 64 != 0:
                raise ValueError(
                    f"progressive_bucket_size must be a positive multiple of 64, " f"got {self.progressive_bucket_size}"
                )

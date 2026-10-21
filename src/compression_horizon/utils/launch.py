import torch
from transformers import set_seed


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def set_launch_seed(seed: int | None) -> None:
    if seed is not None:
        set_seed(seed)


def freeze_model_parameters(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    s = (dtype_str or "").lower()
    if s in {"auto"}:
        return "auto"
    if s in {"float32", "fp32"}:
        return torch.float32
    if s in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if s in {"float16", "fp16"}:
        return torch.float16
    # Fallback to float32 for unknown values
    return torch.float32


def validate_cramming_batch_size(args) -> None:
    """Enforce single-sample memory optimization at the training CLI boundary.

    Compression-head training optimizes shared model parameters and is a separate
    training task. Low-level trainer APIs remain available for regression tests.
    """
    if not getattr(args, "train_compression_head", False) and args.per_device_train_batch_size != 1:
        raise ValueError(
            "Per-sample cramming requires per_device_train_batch_size=1; "
            "use --per_device_train_batch_size 1 to avoid BF16 batch-shape sensitivity."
        )

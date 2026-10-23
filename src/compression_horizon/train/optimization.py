from functools import partial
from typing import Any

import torch
from torch.optim import SGD, AdamW, Optimizer
from transformers import get_scheduler


def _clamped_schedule(step, *, schedule, last_step):
    return schedule(min(step, last_step))


def build_optimizer_and_scheduler(
    args,
    parameters: list[torch.nn.Parameter],
    num_training_steps: int | None = None,
    num_processes: int = 1,
) -> tuple[Optimizer, Any]:
    """Build the AdamW/SGD optimizer + LR scheduler from training args."""
    decay_steps = getattr(args, "lr_scheduler_decay_steps", None)
    if decay_steps is not None and num_training_steps is None:
        raise ValueError("lr_scheduler_decay_steps requires num_training_steps")
    if args.optim == "adamw_torch":
        optimizer = AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    elif args.optim == "sgd":
        optimizer = SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Only AdamW and SGD are supported!")

    lr_scheduler = None
    if num_training_steps is not None:
        if args.lr_scheduler_kwargs is not None:
            assert args.lr_scheduler_kwargs["min_lr"] < args.learning_rate, (
                f"min_lr must be lower than regular LR, " f"{args.lr_scheduler_kwargs['min_lr']} < {args.learning_rate}!"
            )
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps * num_processes,
            num_training_steps=(decay_steps if decay_steps is not None else num_training_steps) * num_processes,
            scheduler_specific_kwargs=args.lr_scheduler_kwargs,
        )
        if decay_steps is not None:
            # The HF cosine rises again after its horizon unless its input is clamped.
            # Keep the original LambdaLR's initialization and checkpoint semantics.
            lr_scheduler.lr_lambdas = [
                partial(_clamped_schedule, schedule=schedule, last_step=decay_steps * num_processes)
                for schedule in lr_scheduler.lr_lambdas
            ]
    return optimizer, lr_scheduler

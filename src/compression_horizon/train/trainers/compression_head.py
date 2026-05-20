"""Compression head trainer: trains a compression head model on full sequences."""

import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from compression_horizon.train.trainers.base import BaseTrainer
from compression_horizon.utils.launch import freeze_model_parameters


def _sample_prefix_lengths(attention_mask: torch.Tensor) -> torch.LongTensor:
    """Sample a per-sample random prefix length in [1, sequence_length]."""
    if attention_mask.ndim == 3 and attention_mask.shape[1] == 1:
        attention_mask = attention_mask.squeeze(1)
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask to be [B, T], got shape {tuple(attention_mask.shape)}")
    device = attention_mask.device
    lengths = attention_mask.sum(dim=1).to(torch.long).clamp_min(1)
    u = torch.rand(lengths.shape, device=device, dtype=torch.float32)
    prefix_lengths = (torch.floor(u * lengths.to(torch.float32)).to(torch.long) + 1).clamp_min(1)
    return torch.minimum(prefix_lengths, lengths).clamp_min(1)


def _build_compressed_inputs(
    *,
    compression_embeds: torch.Tensor,
    token_embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_lengths: torch.LongTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inputs of shape [B, 1+T, H] with a compression token followed by tokens shifted by prefix_lengths."""
    if attention_mask.ndim == 3 and attention_mask.shape[1] == 1:
        attention_mask = attention_mask.squeeze(1)
    if input_ids.ndim == 3 and input_ids.shape[1] == 1:
        input_ids = input_ids.squeeze(1)
    if attention_mask.ndim != 2 or input_ids.ndim != 2:
        raise ValueError(
            f"Expected input_ids/attention_mask to be [B, T], got {tuple(input_ids.shape)}/{tuple(attention_mask.shape)}"
        )
    device = token_embeddings.device
    bsz, seq_len, hidden = token_embeddings.shape

    lengths = attention_mask.sum(dim=1).to(torch.long).clamp_min(1)
    max_prefix = (lengths - 1).clamp_min(0)
    p = torch.minimum(prefix_lengths.to(device=device).to(torch.long).clamp(min=0), max_prefix)

    out_len = 1 + seq_len
    inputs_embeds_new = torch.zeros((bsz, out_len, hidden), device=device, dtype=token_embeddings.dtype)
    attention_mask_new = torch.zeros((bsz, out_len), device=device, dtype=attention_mask.dtype)
    labels_new = torch.full((bsz, out_len), fill_value=-100, device=device, dtype=input_ids.dtype)

    inputs_embeds_new[:, 0:1, :] = compression_embeds
    attention_mask_new[:, 0] = 1

    ar = torch.arange(seq_len, device=device, dtype=torch.long)
    src_idx = p.unsqueeze(1) + ar.unsqueeze(0)
    valid = src_idx < lengths.unsqueeze(1)
    src_idx_safe = torch.clamp(src_idx, max=seq_len - 1)

    gathered_embeds = token_embeddings.gather(1, src_idx_safe.unsqueeze(-1).expand(-1, -1, hidden))
    gathered_ids = input_ids.gather(1, src_idx_safe)

    if valid.dtype != torch.bool:
        valid = valid.to(torch.bool)

    inputs_embeds_new[:, 1:, :] = gathered_embeds * valid.unsqueeze(-1).to(dtype=token_embeddings.dtype)
    attention_mask_new[:, 1:] = valid.to(dtype=attention_mask.dtype)
    labels_new[:, 1:] = torch.where(valid, gathered_ids, torch.full_like(gathered_ids, -100))
    return inputs_embeds_new, attention_mask_new, labels_new


class CompressionHeadTrainer(BaseTrainer):
    """Trainer for compression head models (e.g. LlamaForCausalLMCompressionHead)."""

    def train(self) -> str | None:
        """Run compression head training. Returns output_dir."""
        return self._train_compression_head()

    def _train_compression_head(self) -> str:
        args = self.args
        model = self.model

        grad_accum = args.gradient_accumulation_steps
        assert grad_accum >= 1

        accelerator = self.accelerator
        device = accelerator.device
        print("device", device)

        if not accelerator.is_main_process and self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            finally:
                self.writer = None

        profile = os.environ.get("CH_PROFILE", "0") not in {"0", "", "false", "False"}
        profile_first = int(os.environ.get("CH_PROFILE_FIRST", "5"))
        profile_every = int(os.environ.get("CH_PROFILE_EVERY", "50"))

        def _sync():
            if device.type == "cuda":
                torch.cuda.synchronize()

        if args.compression_head_freeze_base_model:
            freeze_model_parameters(model)
            for p in getattr(model, "compression_head", nn.Module()).parameters():
                p.requires_grad = True

        model.train()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=args.dataloader_num_workers,
            drop_last=args.dataloader_drop_last,
        )

        total_update_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None
        num_epochs = int(getattr(args, "num_train_epochs", 1) or 1)
        if total_update_steps is None:
            print("train_loader", len(train_loader))
            print("accelerator.num_processes", accelerator.num_processes)
            micro_steps_per_epoch = len(train_loader)
            if accelerator.num_processes > 1:
                micro_steps_per_epoch = int(math.ceil(micro_steps_per_epoch / accelerator.num_processes))
            micro_steps_total = micro_steps_per_epoch * num_epochs
            total_update_steps = int(math.ceil(micro_steps_total / grad_accum))

        params = [p for p in model.parameters() if p.requires_grad]

        print("total_update_steps", total_update_steps)
        optimizer, scheduler = self._build_optimizer_and_scheduler(
            params, num_training_steps=total_update_steps, num_processes=accelerator.num_processes
        )

        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        print("train_loader after prepare", len(train_loader))

        unwrapped_model = accelerator.unwrap_model(model)
        params = [p for p in model.parameters() if p.requires_grad]

        if hasattr(unwrapped_model, "config") and unwrapped_model.config is not None:
            unwrapped_model.config.use_cache = False
        if hasattr(model, "config") and model.config is not None:
            model.config.use_cache = False

        if getattr(args, "gradient_checkpointing", False):
            if hasattr(unwrapped_model, "gradient_checkpointing_enable"):
                unwrapped_model.gradient_checkpointing_enable()
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        else:
            if hasattr(unwrapped_model, "gradient_checkpointing_disable"):
                unwrapped_model.gradient_checkpointing_disable()
            elif hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()

        update_step = 0
        micro_step = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(
            total=total_update_steps,
            desc="train_compression_head",
            disable=not accelerator.is_main_process,
        )

        prev_iter_end = time.perf_counter()
        while update_step < total_update_steps:
            for _epoch in range(num_epochs):
                for batch in train_loader:
                    if update_step >= total_update_steps:
                        break
                    t_batch_ready = time.perf_counter()
                    data_wait_s = t_batch_ready - prev_iter_end

                    t0 = time.perf_counter()
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch.get("labels", input_ids).to(device)
                    _sync()
                    h2d_s = time.perf_counter() - t0

                    if input_ids.ndim == 3 and input_ids.shape[1] == 1:
                        input_ids = input_ids.squeeze(1)
                    if attention_mask.ndim == 3 and attention_mask.shape[1] == 1:
                        attention_mask = attention_mask.squeeze(1)
                    if labels.ndim == 3 and labels.shape[1] == 1:
                        labels = labels.squeeze(1)
                    if input_ids.ndim != 2 or attention_mask.ndim != 2 or labels.ndim != 2:
                        raise ValueError(
                            f"Expected batch tensors to be [B, T], got input_ids={tuple(input_ids.shape)} "
                            f"attention_mask={tuple(attention_mask.shape)} labels={tuple(labels.shape)}"
                        )

                    t0 = time.perf_counter()
                    prefix_lengths = _sample_prefix_lengths(attention_mask)
                    prefix_s = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        prefix_lengths=prefix_lengths,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    _sync()
                    base_loss = out.loss

                    if base_loss is None:
                        raise RuntimeError("Model did not return loss (labels missing?)")
                    if torch.isnan(base_loss) or torch.isinf(base_loss):
                        print(f"DEBUG: NaN/Inf detected in base_loss at step {update_step}, " f"micro_step {micro_step}")
                        raise RuntimeError(f"NaN/Inf in base_loss: {base_loss.item()}")

                    if out.compression_embeds is None:
                        raise RuntimeError("Model did not return compression embeddings.")

                    if torch.isnan(out.compression_embeds).any() or torch.isinf(out.compression_embeds).any():
                        raise RuntimeError("NaN/Inf in compression_embeds")

                    compression_embeds = out.compression_embeds
                    del out

                    t0 = time.perf_counter()
                    with accelerator.autocast():
                        token_embeddings = unwrapped_model.get_input_embeddings()(input_ids)
                    _sync()
                    embed_s = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    inputs_embeds_new, attention_mask_new, labels_new = _build_compressed_inputs(
                        compression_embeds=compression_embeds,
                        token_embeddings=token_embeddings,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        prefix_lengths=prefix_lengths,
                    )
                    build_s = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    with accelerator.autocast():
                        out2 = model(
                            inputs_embeds=inputs_embeds_new,
                            attention_mask=attention_mask_new,
                            labels=labels_new,
                            use_cache=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    _sync()
                    fwd2_s = time.perf_counter() - t0
                    after_loss = out2.loss
                    if after_loss is None:
                        raise RuntimeError("Model did not return loss for compressed-input forward (labels missing?)")
                    del out2
                    del token_embeddings, inputs_embeds_new, attention_mask_new, labels_new, compression_embeds

                    alpha = args.compression_head_distill_alpha
                    loss = base_loss + after_loss * alpha

                    if torch.isnan(loss) or torch.isinf(loss):
                        raise RuntimeError(f"NaN/Inf in total loss: {loss.item()}")

                    opt_s = 0.0
                    did_step = False
                    t0 = time.perf_counter()
                    with accelerator.accumulate(model):
                        accelerator.backward(loss / grad_accum)
                        _sync()
                        bwd_s = time.perf_counter() - t0

                        micro_step += 1
                        if accelerator.sync_gradients:
                            t0 = time.perf_counter()
                            if getattr(args, "max_grad_norm", None) is not None and args.max_grad_norm > 0:
                                accelerator.clip_grad_norm_(params, args.max_grad_norm)

                            optimizer.step()

                            if scheduler is not None:
                                scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            _sync()
                            opt_s = time.perf_counter() - t0

                            did_step = True
                            update_step += 1

                            if accelerator.num_processes == 1:
                                loss_m = float(loss.detach().item())
                                base_m = float(base_loss.detach().item())
                                after_m = float(after_loss.detach().item())
                            else:
                                loss_m = float(accelerator.gather(loss.detach()).mean().item())
                                base_m = float(accelerator.gather(base_loss.detach()).mean().item())
                                after_m = float(accelerator.gather(after_loss.detach()).mean().item())

                            if accelerator.is_main_process:
                                if self.writer:
                                    self.writer.add_scalar("loss/total", loss_m, self.global_step)
                                    self.writer.add_scalar("loss/base", base_m, self.global_step)
                                    self.writer.add_scalar("loss/after_compression", after_m, self.global_step)
                                    current_lr = optimizer.param_groups[0]["lr"]
                                    self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)

                                self.global_step += 1

                                def safe_format_scalar(v: float) -> str:
                                    if math.isnan(v) or math.isinf(v):
                                        return "nan" if math.isnan(v) else "inf"
                                    return f"{v:.4f}"

                                pbar.set_postfix(
                                    {
                                        "loss": safe_format_scalar(loss_m),
                                        "base": safe_format_scalar(base_m),
                                        "after": safe_format_scalar(after_m),
                                    }
                                )
                                pbar.update(1)
                    if (
                        accelerator.is_main_process
                        and profile
                        and (update_step <= profile_first or (did_step and update_step % profile_every == 0))
                    ):
                        mem = ""
                        if device.type == "cuda":
                            alloc_gb = torch.cuda.memory_allocated() / (1024**3)
                            max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
                            mem = f" cuda_mem_gb={alloc_gb:.2f} max={max_alloc_gb:.2f}"
                        print(
                            "profile:"
                            f" upd={update_step}/{total_update_steps}"
                            f" micro={micro_step}"
                            f" data_wait={data_wait_s:.3f}s"
                            f" h2d={h2d_s:.3f}s"
                            f" prefix={prefix_s:.3f}s"
                            f" embed={embed_s:.3f}s"
                            f" build={build_s:.3f}s"
                            f" fwd2={fwd2_s:.3f}s"
                            f" bwd={bwd_s:.3f}s"
                            f" opt={opt_s:.3f}s"
                            f"{mem}"
                        )

                    prev_iter_end = time.perf_counter()
                    if update_step >= total_update_steps:
                        break
                if update_step >= total_update_steps:
                    break
            if update_step >= total_update_steps:
                break

        pbar.close()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            if not hasattr(unwrapped_model, "save_pretrained"):
                raise RuntimeError("Expected a Hugging Face PreTrainedModel with save_pretrained().")
            unwrapped_model.save_pretrained(args.output_dir)
            if self.processing_class is not None and hasattr(self.processing_class, "save_pretrained"):
                self.processing_class.save_pretrained(args.output_dir)

            try:
                args_dict = getattr(args, "to_dict", lambda: {})()
                with open(
                    os.path.join(args.output_dir, "training_args.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(args_dict, f, ensure_ascii=False, indent=2, sort_keys=True)
                    f.write("\n")
            except Exception:
                pass
        return args.output_dir

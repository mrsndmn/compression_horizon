"""Compression head trainer: trains a compression head model on full sequences."""

import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from compression_horizon.train.base import BaseTrainer
from compression_horizon.utils.launch import freeze_model_parameters


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
            # Sync only when profiling — otherwise we destroy CUDA stream overlap.
            if profile and device.type == "cuda":
                torch.cuda.synchronize()

        if args.compression_head_freeze_base_model:
            freeze_model_parameters(model)
            for p in getattr(model, "compression_head", nn.Module()).parameters():
                p.requires_grad = True

        model.train()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
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

        if getattr(args, "torch_compile", False):
            compile_mode = getattr(args, "torch_compile_mode", None) or "default"
            compile_backend = getattr(args, "torch_compile_backend", None) or "inductor"
            print(f"[torch.compile] enabling backend={compile_backend} mode={compile_mode}")
            model = torch.compile(model, mode=compile_mode, backend=compile_backend)

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
                    prefix_lengths = self._sample_prefix_lengths(attention_mask)
                    prefix_s = time.perf_counter() - t0

                    t0 = time.perf_counter()

                    extra_params = dict()
                    if args.compression_head_distill_beta != 0:
                        extra_params = {"labels": labels}

                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        prefix_lengths=prefix_lengths,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                        **extra_params,
                    )
                    _sync()
                    base_loss = out.loss
                    if base_loss is None:
                        base_loss = torch.tensor(0.0, device=input_ids.device)

                    # if base_loss is None:
                    #     raise RuntimeError("Model did not return loss (labels missing?)")
                    # if torch.isnan(base_loss) or torch.isinf(base_loss):
                    #     print(f"DEBUG: NaN/Inf detected in base_loss at step {update_step}, " f"micro_step {micro_step}")
                    #     raise RuntimeError(f"NaN/Inf in base_loss: {base_loss.item()}")

                    if out.compression_embeds is None:
                        raise RuntimeError("Model did not return compression embeddings.")

                    # if torch.isnan(out.compression_embeds).any() or torch.isinf(out.compression_embeds).any():
                    #     raise RuntimeError("NaN/Inf in compression_embeds")

                    compression_embeds = out.compression_embeds
                    # Track ||compression_embeds|| as a TB scalar so we can see if the
                    # compression-token magnitude drifts out of distribution over training.
                    with torch.no_grad():
                        compression_embeds_norm = compression_embeds.detach().float().norm(dim=-1).mean()
                    del out

                    t0 = time.perf_counter()
                    with accelerator.autocast():
                        token_embeddings = unwrapped_model.get_input_embeddings()(input_ids)
                    _sync()
                    embed_s = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    inputs_embeds_new, attention_mask_new, labels_new = self._build_compressed_inputs(
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
                    # Argmax match rate on the reconstructed prefix positions (labels != -100).
                    # Mirrors progressive cramming's convergence metric — a direct proxy for
                    # whether the compression head is learning prefix reconstruction.
                    with torch.no_grad():
                        shift_logits_match = out2.logits[:, :-1, :]
                        shift_labels_match = labels_new[:, 1:]
                        valid_mask_match = shift_labels_match.ne(-100)
                        n_valid = valid_mask_match.sum()
                        if n_valid.item() > 0:
                            preds_match = shift_logits_match.argmax(dim=-1)
                            correct_match = (preds_match == shift_labels_match) & valid_mask_match
                            after_match_rate = correct_match.sum().float() / n_valid.float()
                        else:
                            after_match_rate = torch.zeros((), device=after_loss.device)
                    del out2
                    del token_embeddings, inputs_embeds_new, attention_mask_new, labels_new, compression_embeds

                    alpha = args.compression_head_distill_alpha
                    beta = args.compression_head_distill_beta
                    loss = base_loss * beta + after_loss * alpha

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
                                grad_norm_t = accelerator.clip_grad_norm_(params, args.max_grad_norm)
                            else:
                                # Compute grad norm without clipping so we still log it.
                                with torch.no_grad():
                                    sq = torch.zeros((), device=accelerator.device)
                                    for p in params:
                                        if p.grad is not None:
                                            sq = sq + p.grad.detach().float().pow(2).sum()
                                    grad_norm_t = sq.sqrt()

                            optimizer.step()

                            if scheduler is not None:
                                scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            _sync()
                            opt_s = time.perf_counter() - t0

                            did_step = True
                            update_step += 1

                            grad_norm_d = (
                                grad_norm_t.detach().float()
                                if isinstance(grad_norm_t, torch.Tensor)
                                else torch.tensor(float(grad_norm_t), device=accelerator.device)
                            )
                            if accelerator.num_processes == 1:
                                loss_m = float(loss.detach().item())
                                base_m = float(base_loss.detach().item())
                                after_m = float(after_loss.detach().item())
                                match_m = float(after_match_rate.detach().item())
                                grad_norm_m = float(grad_norm_d.item())
                                ce_norm_m = float(compression_embeds_norm.item())
                            else:
                                loss_m = float(accelerator.gather(loss.detach()).mean().item())
                                base_m = float(accelerator.gather(base_loss.detach()).mean().item())
                                after_m = float(accelerator.gather(after_loss.detach()).mean().item())
                                match_m = float(accelerator.gather(after_match_rate.detach()).mean().item())
                                grad_norm_m = float(accelerator.gather(grad_norm_d).mean().item())
                                ce_norm_m = float(accelerator.gather(compression_embeds_norm).mean().item())

                            if accelerator.is_main_process:
                                if self.writer:
                                    self.writer.add_scalar("loss/total", loss_m, self.global_step)
                                    self.writer.add_scalar("loss/base", base_m, self.global_step)
                                    self.writer.add_scalar("loss/after_compression", after_m, self.global_step)
                                    self.writer.add_scalar("metric/after_compression_match_rate", match_m, self.global_step)
                                    self.writer.add_scalar("metric/compression_embeds_norm", ce_norm_m, self.global_step)
                                    self.writer.add_scalar("train/grad_norm", grad_norm_m, self.global_step)
                                    current_lr = optimizer.param_groups[0]["lr"]
                                    self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)
                                    if total_update_steps > 0:
                                        epoch_frac = update_step * num_epochs / total_update_steps
                                        self.writer.add_scalar("train/epoch", epoch_frac, self.global_step)

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

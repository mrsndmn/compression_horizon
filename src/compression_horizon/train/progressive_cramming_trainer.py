"""Progressive cramming trainer: progressive sequence-length stages."""

import math
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from compression_horizon.train.base import BaseTrainer
from compression_horizon.utils.launch import freeze_model_parameters, get_device, set_launch_seed


class ProgressiveCrammingTrainer(BaseTrainer):
    """Trainer for progressive cramming: increase sequence length in stages."""

    def train(self) -> str | None:
        """Run progressive training. Returns save path or None."""
        return self._train_progressive()

    def _train_progressive(self) -> str | None:
        device = get_device()
        set_launch_seed(self.args.random_seed)

        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._prepare_embedding_init(model)

        dataloader = self._create_dataloader()

        num_compression_tokens = self.args.number_of_mem_tokens
        threshold = self.args.progressive_convergence_threshold
        step_increment = self.args.progressive_step
        min_len = self.args.progressive_min_seq_len
        max_stages_cap = self.args.progressive_max_stages

        collected_rows = []
        sample_id_counter = 0

        low_dim_prjoection = None
        low_dim_optim = None
        if self.args.low_dim_projection and self.args.low_dim_projection_global:
            low_dim_prjoection, low_dim_optim, low_dim_scheduler = self._prepare_low_dim_proj(
                embedding_dim=model.get_input_embeddings().embedding_dim
            )
            low_dim_prjoection = low_dim_prjoection.to(device)

        for batch in tqdm(dataloader):
            batch_size = batch["input_ids"].shape[0]
            full_input_ids = batch.input_ids.squeeze(1).to(device)
            with torch.no_grad():
                full_model_token_embeddings = model.get_input_embeddings()(full_input_ids)
            full_attention_mask = batch.attention_mask.squeeze(1).to(device)

            target_hidden_full = self.compute_target_hidden(model, full_model_token_embeddings, full_attention_mask)

            hidden_size = full_model_token_embeddings.shape[-1]
            if self.args.low_dim_projection:
                hidden_size = self.args.low_dim_size

            device = full_model_token_embeddings.device

            if self.args.low_dim_projection and not self.args.low_dim_projection_global:
                low_dim_prjoection, low_dim_optim, low_dim_scheduler = self._prepare_low_dim_proj(
                    embedding_dim=model.get_input_embeddings().embedding_dim
                )
                low_dim_prjoection = low_dim_prjoection.to(device)
                print(
                    "low_dim_prjoection",
                    low_dim_prjoection,
                    "low_dim_optim",
                    low_dim_optim,
                )

            if init_method == "pretrained_pca":
                assert pca_components is not None
                assert pca_mean is not None

                pca_components_device = pca_components.to(device)
                pca_mean_device = pca_mean.to(device)

                flattened_dim = pca_mean_device.shape[0]
                expected_flattened_dim = num_compression_tokens * hidden_size
                if flattened_dim != expected_flattened_dim:
                    raise ValueError(
                        f"PCA dimension mismatch: pretrained has {flattened_dim}, "
                        f"but current needs {expected_flattened_dim} "
                        f"(num_tokens={num_compression_tokens}, hidden_size={hidden_size})"
                    )

                n_components = pca_components_device.shape[0]
                per_sample_pca_coefficients = [
                    torch.nn.Parameter(
                        torch.randn(
                            [1, n_components],
                            dtype=torch.float32,
                            device=device,
                        )
                        * 0.1
                    )
                    for _ in range(batch_size)
                ]

                pca_coefficients = torch.cat(per_sample_pca_coefficients, dim=0)
                reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                initialization_embeddings = (
                    reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size).detach().cpu()
                )

                per_sample_optimizers = []
                per_sample_schedulers = []
                for j in range(batch_size):
                    opt, sched = self._build_optimizer_and_scheduler([per_sample_pca_coefficients[j]])
                    per_sample_optimizers.append(opt)
                    per_sample_schedulers.append(sched)
            elif init_method == "compression_head_forward":
                # Initialize the per-sample compression token by running the model's compression_head
                # over each sample's full prefix. Requires a model with a compression_head module
                # (e.g. LlamaForCausalLMCompressionHead). Tests whether the pretrained head provides
                # a useful starting point for progressive cramming.
                if not hasattr(model, "compression_head"):
                    raise ValueError(
                        "embedding_init_method=compression_head_forward requires a model with a "
                        "compression_head attribute (use a LlamaForCausalLMCompressionHead checkpoint)."
                    )
                if num_compression_tokens != 1:
                    raise ValueError(
                        "embedding_init_method=compression_head_forward currently supports " "num_compression_tokens=1 only."
                    )
                lengths_t = full_attention_mask.sum(dim=1).to(device=device, dtype=torch.long)
                with torch.no_grad():
                    outs = model(
                        input_ids=full_input_ids,
                        attention_mask=full_attention_mask,
                        prefix_lengths=lengths_t,
                        use_cache=False,
                        return_dict=True,
                    )
                    ch_embeds = outs.compression_embeds.detach().to(torch.float32)  # [B, 1, H]
                initialization_embeddings = ch_embeds.detach().clone().cpu()
                per_sample_params = [torch.nn.Parameter(ch_embeds[j : j + 1].clone().to(device)) for j in range(batch_size)]
                per_sample_optimizers = []
                per_sample_schedulers = []
                for j in range(batch_size):
                    opt, sched = self._build_optimizer_and_scheduler(
                        [per_sample_params[j]],
                        num_training_steps=self.args.max_optimization_steps_per_sample,
                    )
                    per_sample_optimizers.append(opt)
                    per_sample_schedulers.append(sched)
            else:
                full_init = self._init_compression_tokens(
                    batch_size,
                    num_compression_tokens,
                    hidden_size,
                    init_method,
                    mvn_dist,
                    pca_components=pca_components,
                    pca_mean=pca_mean,
                    loaded_embeddings=loaded_embeddings,
                )
                initialization_embeddings = full_init.detach().clone().cpu()
                per_sample_params = [
                    torch.nn.Parameter(full_init.data[j : j + 1].clone().to(device)) for j in range(batch_size)
                ]

                per_sample_optimizers = []
                per_sample_schedulers = []
                for j in range(batch_size):
                    opt, sched = self._build_optimizer_and_scheduler(
                        [per_sample_params[j]],
                        num_training_steps=self.args.max_optimization_steps_per_sample,
                    )
                    per_sample_optimizers.append(opt)
                    per_sample_schedulers.append(sched)

            compression_tokens_attention_mask = torch.tensor([[1]], dtype=full_attention_mask.dtype, device=device).repeat(
                batch_size, num_compression_tokens
            )

            per_sample_lengths = full_attention_mask.sum(dim=1).tolist()
            max_len = int(max(per_sample_lengths)) if len(per_sample_lengths) > 0 else full_attention_mask.shape[1]
            seq_len = min(min_len, max_len)
            stage_index = 0

            converged_mask = [False] * batch_size
            skipped_mask = [False] * batch_size
            per_sample_steps_taken = [0] * batch_size

            while True:
                scheduler_reset_used = False
                input_ids = full_input_ids[:, :seq_len]
                inputs_embeds = full_model_token_embeddings[:, :seq_len, :]
                target_hidden = list(h[:, :seq_len] for h in target_hidden_full)
                attention_mask = full_attention_mask[:, :seq_len]

                pbar = tqdm(
                    range(self.args.max_optimization_steps_per_token),
                    total=self.args.max_optimization_steps_per_token,
                    leave=False,
                    disable=True,
                )
                pbar.set_description(f"Stage L={seq_len}")
                last_loss_val = None
                last_conv = None
                converged = False

                # Reset per-stage convergence (but not skipped_mask or steps_taken)
                converged_mask = [skipped_mask[j] for j in range(batch_size)]

                while True:
                    for i in pbar:
                        if init_method == "pretrained_pca":
                            pca_coefficients = torch.cat(per_sample_pca_coefficients, dim=0)
                            reconstructed_flat = torch.matmul(
                                pca_coefficients, pca_components_device
                            ) + pca_mean_device.unsqueeze(0)
                            compression_tokens = reconstructed_flat.reshape(
                                batch_size,
                                num_compression_tokens,
                                hidden_size,
                            )
                        else:
                            compression_tokens = torch.cat(per_sample_params, dim=0)

                        current_compression_tokens = compression_tokens.clone()
                        if self.args.low_dim_projection:
                            current_compression_tokens = low_dim_prjoection(compression_tokens)

                        model_tokens_with_compression_tokens = torch.cat(
                            [
                                current_compression_tokens.to(inputs_embeds.device).to(inputs_embeds.dtype),
                                inputs_embeds,
                            ],
                            dim=1,
                        )
                        attention_mask_with_compression_tokens = torch.cat(
                            [compression_tokens_attention_mask, attention_mask],
                            dim=1,
                        )
                        (
                            loss,
                            alignment_loss,
                            convergece_per_sample,
                            generated_text,
                            ground_truth_text,
                        ) = self.compute_loss(
                            model,
                            input_ids,
                            inputs_embeds,
                            attention_mask,
                            model_tokens_with_compression_tokens,
                            attention_mask_with_compression_tokens,
                            num_compression_tokens,
                            target_hidden=target_hidden,
                        )
                        loss.backward()
                        pbar.update(1)

                        if init_method == "pretrained_pca":
                            grad_norms = [
                                (
                                    per_sample_pca_coefficients[j].grad.norm(2).item()
                                    if per_sample_pca_coefficients[j].grad is not None
                                    else 0.0
                                )
                                for j in range(batch_size)
                            ]
                            grad_norm = sum(grad_norms) / len(grad_norms)
                            comp_mean = compression_tokens.mean().item()
                            comp_std = compression_tokens.std().item()
                        else:
                            grad_norms = [
                                per_sample_params[j].grad.norm(2).item() if per_sample_params[j].grad is not None else 0.0
                                for j in range(batch_size)
                            ]
                            grad_norm = sum(grad_norms) / len(grad_norms)
                            comp_mean = compression_tokens.mean().item()
                            comp_std = compression_tokens.std().item()

                        log_lr = self.args.learning_rate
                        active_scheduler = None
                        for j in range(batch_size):
                            if not skipped_mask[j] and not converged_mask[j]:
                                active_scheduler = per_sample_schedulers[j]
                                break
                        if active_scheduler is not None:
                            log_lr = active_scheduler.get_last_lr()[0]

                        pbar.set_postfix(
                            loss=loss.item(),
                            convergece_per_sample=convergece_per_sample.mean().item(),
                            compr_t_mean=comp_mean,
                            compr_t_std=comp_std,
                            grad=grad_norm,
                            lr=log_lr,
                        )

                        self._log_step(
                            loss,
                            alignment_loss,
                            convergece_per_sample,
                            compression_tokens,
                            active_scheduler,
                            generated_text,
                            ground_truth_text,
                        )

                        # Per-sample optimizer step
                        for j in range(batch_size):
                            if skipped_mask[j] or converged_mask[j]:
                                per_sample_optimizers[j].zero_grad(set_to_none=True)
                            else:
                                per_sample_optimizers[j].step()
                                if per_sample_schedulers[j] is not None:
                                    per_sample_schedulers[j].step()
                                per_sample_optimizers[j].zero_grad(set_to_none=True)
                                per_sample_steps_taken[j] += 1

                        if self.args.low_dim_projection and self.args.low_dim_proj_train and low_dim_optim is not None:
                            low_dim_optim.step()
                            low_dim_optim.zero_grad()
                            if low_dim_scheduler is not None:
                                low_dim_scheduler.step()

                        last_loss_val = float(loss.item())
                        last_conv = convergece_per_sample.detach().cpu()

                        # Per-sample convergence check
                        for j in range(batch_size):
                            if not skipped_mask[j] and not converged_mask[j]:
                                if convergece_per_sample[j].item() >= threshold:
                                    converged_mask[j] = True

                        all_active_converged = all(converged_mask[j] or skipped_mask[j] for j in range(batch_size))
                        if all_active_converged:
                            converged = True
                            break

                    if converged:
                        break

                    if (
                        not converged
                        and self.args.progressive_reset_lr_scheduler_on_non_convergence
                        and not scheduler_reset_used
                    ):
                        print(f"Not converged at seq_len={seq_len}, " "resetting LR schedulers for non-converged samples...")
                        for j in range(batch_size):
                            if not skipped_mask[j] and not converged_mask[j]:
                                if init_method == "pretrained_pca":
                                    opt, sched = self._build_optimizer_and_scheduler([per_sample_pca_coefficients[j]])
                                else:
                                    opt, sched = self._build_optimizer_and_scheduler(
                                        [per_sample_params[j]],
                                        num_training_steps=self.args.max_optimization_steps_per_token,
                                    )
                                per_sample_optimizers[j] = opt
                                per_sample_schedulers[j] = sched
                        scheduler_reset_used = True
                        pbar = tqdm(
                            range(self.args.max_optimization_steps_per_token),
                            total=self.args.max_optimization_steps_per_token,
                            leave=False,
                        )
                        pbar.set_description(f"Stage L={seq_len} (retry)")
                        continue
                    else:
                        break

                # Mark non-converged samples as permanently skipped
                if not converged:
                    for j in range(batch_size):
                        if not skipped_mask[j] and not converged_mask[j]:
                            skipped_mask[j] = True
                            print(f"Sample {j} failed to converge at seq_len={seq_len}, marking as skipped.")

                with torch.no_grad():
                    tokenizer = self.processing_class

                    if init_method == "pretrained_pca":
                        pca_coefficients = torch.cat(per_sample_pca_coefficients, dim=0)
                        reconstructed_flat = torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(
                            0
                        )
                        comp_tokens_gpu = reconstructed_flat.reshape(batch_size, num_compression_tokens, hidden_size)
                        comp_tokens_cpu = comp_tokens_gpu.detach().cpu()
                        orig_comp_tokens_gpu = comp_tokens_gpu
                        orig_comp_tokens_cpu = orig_comp_tokens_gpu.detach().cpu()
                    else:
                        compression_tokens = torch.cat(per_sample_params, dim=0)
                        if self.args.low_dim_projection:
                            comp_tokens_gpu = low_dim_prjoection(compression_tokens)
                        else:
                            comp_tokens_gpu = compression_tokens
                        comp_tokens_cpu = comp_tokens_gpu.detach().cpu()
                        orig_comp_tokens_gpu = compression_tokens
                        orig_comp_tokens_cpu = orig_comp_tokens_gpu.detach().cpu()

                    if init_method == "pretrained_pca":
                        final_compression_tokens_for_ig = (
                            torch.matmul(pca_coefficients, pca_components_device) + pca_mean_device.unsqueeze(0)
                        ).reshape(batch_size, num_compression_tokens, hidden_size)
                    else:
                        final_compression_tokens_for_ig = compression_tokens
                    if self.args.low_dim_projection:
                        final_compression_tokens_for_ig = low_dim_prjoection(final_compression_tokens_for_ig)

                    per_sample_info_gain = []
                    for j in range(batch_size):
                        sample_input_ids = input_ids[j : j + 1]
                        sample_attention_mask = attention_mask[j : j + 1]
                        sample_compression_tokens = final_compression_tokens_for_ig[j : j + 1]

                        sample_outputs_lm = model(
                            input_ids=sample_input_ids,
                            attention_mask=sample_attention_mask,
                        )
                        sample_logits_lm = sample_outputs_lm.logits

                        sample_shift_logits_lm = sample_logits_lm[:, :-1, :].contiguous()
                        sample_shift_labels_lm = sample_input_ids[:, 1:].contiguous()
                        sample_shift_mask_lm = sample_attention_mask[:, 1:].contiguous()

                        sample_shift_logits_lm_flat = sample_shift_logits_lm.view(-1, sample_shift_logits_lm.size(-1))
                        sample_shift_labels_lm_flat = sample_shift_labels_lm.view(-1)
                        sample_shift_mask_lm_flat = sample_shift_mask_lm.view(-1)

                        sample_valid_mask_lm = sample_shift_mask_lm_flat.bool()
                        if sample_valid_mask_lm.sum() > 0:
                            sample_ce_lm_sum = F.cross_entropy(
                                sample_shift_logits_lm_flat[sample_valid_mask_lm],
                                sample_shift_labels_lm_flat[sample_valid_mask_lm],
                                reduction="sum",
                            )
                            sample_H_LM_bits = sample_ce_lm_sum.item() / math.log(2)
                        else:
                            sample_H_LM_bits = 0.0

                        sample_inputs_embeds = inputs_embeds[j : j + 1]
                        sample_model_tokens_with_compression = torch.cat(
                            [
                                sample_compression_tokens.to(sample_inputs_embeds.device).to(sample_inputs_embeds.dtype),
                                sample_inputs_embeds,
                            ],
                            dim=1,
                        )
                        sample_compression_attention_mask = compression_tokens_attention_mask[j : j + 1]
                        sample_attention_mask_with_compression = torch.cat(
                            [
                                sample_compression_attention_mask,
                                sample_attention_mask,
                            ],
                            dim=1,
                        )

                        sample_outputs_mem = model(
                            inputs_embeds=sample_model_tokens_with_compression,
                            attention_mask=sample_attention_mask_with_compression,
                        )
                        sample_logits_mem = sample_outputs_mem.logits
                        sample_aligned_logits_mem = sample_logits_mem[:, num_compression_tokens:, :]
                        sample_shift_logits_mem = sample_aligned_logits_mem[:, :-1, :].contiguous()
                        sample_shift_labels_mem = sample_input_ids[:, 1:].contiguous()
                        sample_shift_mask_mem = sample_attention_mask[:, 1:].contiguous()

                        sample_shift_logits_mem_flat = sample_shift_logits_mem.view(-1, sample_shift_logits_mem.size(-1))
                        sample_shift_labels_mem_flat = sample_shift_labels_mem.view(-1)
                        sample_shift_mask_mem_flat = sample_shift_mask_mem.view(-1)

                        sample_valid_mask_mem = sample_shift_mask_mem_flat.bool()
                        if sample_valid_mask_mem.sum() > 0:
                            sample_ce_mem_sum = F.cross_entropy(
                                sample_shift_logits_mem_flat[sample_valid_mask_mem],
                                sample_shift_labels_mem_flat[sample_valid_mask_mem],
                                reduction="sum",
                            )
                            sample_H_LM_mem_bits = sample_ce_mem_sum.item() / math.log(2)
                        else:
                            sample_H_LM_mem_bits = 0.0

                        sample_info_gain = sample_H_LM_bits - sample_H_LM_mem_bits
                        per_sample_info_gain.append(sample_info_gain)

                    embeddings_dir = None
                    if self.args.output_dir:
                        embeddings_dir = os.path.join(self.args.output_dir, "embeddings")
                        os.makedirs(embeddings_dir, exist_ok=True)

                    for j in range(batch_size):
                        attn = attention_mask[j].bool()
                        ids = input_ids[j][attn]
                        text = tokenizer.decode(ids.tolist(), skip_special_tokens=True) if tokenizer is not None else ""
                        sample_id_val = int(sample_id_counter + j)

                        pca_coefficients_to_save = None
                        if init_method == "pretrained_pca":
                            pca_coefficients_to_save = (
                                per_sample_pca_coefficients[j].clone().detach().to(torch.float32).cpu().numpy().tolist()
                            )

                        if embeddings_dir is not None and stage_index % 500 == 0:
                            comp_tokens_bfloat = comp_tokens_gpu[j].to(torch.bfloat16).detach().cpu()
                            orig_comp_tokens_bfloat = orig_comp_tokens_gpu[j].to(torch.bfloat16).detach().cpu()
                            initialization_embedding_bfloat = initialization_embeddings[j].to(torch.bfloat16)

                            embedding_filename = f"embedding_sample_{sample_id_val}_stage_{stage_index}.pt"
                            orig_embedding_filename = f"orig_embedding_sample_{sample_id_val}_stage_{stage_index}.pt"
                            init_embedding_filename = f"initialization_embedding_sample_{sample_id_val}_stage_{stage_index}.pt"
                            low_dim_proj_filename = f"low_dim_proj_sample_{sample_id_val}_stage_{stage_index}.pt"

                            embedding_path = os.path.join(embeddings_dir, embedding_filename)
                            orig_embedding_path = os.path.join(embeddings_dir, orig_embedding_filename)
                            init_embedding_path = os.path.join(embeddings_dir, init_embedding_filename)
                            low_dim_proj_path = os.path.join(embeddings_dir, low_dim_proj_filename)

                            torch.save(comp_tokens_bfloat, embedding_path)
                            torch.save(orig_comp_tokens_bfloat, orig_embedding_path)
                            torch.save(
                                initialization_embedding_bfloat,
                                init_embedding_path,
                            )
                            if self.args.low_dim_projection:
                                torch.save(
                                    low_dim_prjoection.state_dict(),
                                    low_dim_proj_path,
                                )

                        embedding = comp_tokens_cpu[j].to(torch.float32).numpy().tolist()
                        orig_embedding = orig_comp_tokens_cpu[j].to(torch.float32).numpy().tolist()
                        initialization_embedding = initialization_embeddings[j].to(torch.float32).numpy().tolist()

                        collected_rows.append(
                            {
                                "sample_id": int(sample_id_counter + j),
                                "stage_index": int(stage_index),
                                "stage_seq_len": int(seq_len),
                                "text": text,
                                "embedding": embedding,
                                "orig_embedding": orig_embedding,
                                "pca_coefficients_to_save": pca_coefficients_to_save,
                                "initialization_embedding": initialization_embedding,
                                "final_loss": (float(last_loss_val) if last_loss_val is not None else None),
                                "final_convergence": (float(last_conv[j].item()) if last_conv is not None else None),
                                "num_input_tokens": int(attn.sum().item()),
                                "num_compression_tokens": int(num_compression_tokens),
                                "hidden_size": int(comp_tokens_cpu.shape[-1]),
                                "loss_type": getattr(self.args, "loss_type", "l2"),
                                "dtype": getattr(self.args, "dtype", "float32"),
                                "model_checkpoint": getattr(self.args, "model_checkpoint", ""),
                                "max_optimization_steps_per_sample": int(
                                    getattr(
                                        self.args,
                                        "max_optimization_steps_per_sample",
                                        0,
                                    )
                                ),
                                "convergence_threshold": float(threshold),
                                "steps_taken": int(per_sample_steps_taken[j]),
                                "information_gain_bits": float(per_sample_info_gain[j]),
                            }
                        )

                stage_index += 1
                if seq_len >= max_len:
                    break
                if max_stages_cap and stage_index >= max_stages_cap:
                    break

                all_skipped = all(skipped_mask)
                if all_skipped:
                    print("All samples skipped. Stopping at seq_len =", seq_len)
                    break

                seq_len = min(seq_len + step_increment, max_len)

            sample_id_counter += batch_size

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        save_path = self._save_artifacts(None, collected_rows, "progressive_prefixes")
        if save_path is not None:
            return save_path
        return None

import os
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import get_scheduler

from compression_horizon.inference.generation import generate_from_compression
from compression_horizon.utils.launch import freeze_model_parameters, get_device, set_launch_seed


class MyTrainer:
    def __init__(
        self,
        model=None,
        processing_class=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
    ):
        self.model = model
        self.processing_class = processing_class
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        # TensorBoard
        log_dir = self.args.logging_dir
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.global_step = 0

    def compute_loss(
        self,
        model,
        input_ids,
        token_embeddings,
        attention_mask,
        united_token_embeddings,
        united_attention_mask,
        num_compression_tokens,
    ):
        with torch.no_grad():
            # Hidden state: [batch, sequence, hidden]
            outputs = model(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Hidden state: [batch, mem + sequence, hidden]
        compression_outputs = model(
            inputs_embeds=united_token_embeddings,
            attention_mask=united_attention_mask,
            output_hidden_states=True,
        )

        # Number of hidden states: 1 (embedder) + number of transformer decoder layers
        total_layers = len(outputs.hidden_states)
        if self.args.num_alignment_layers > 0:
            num_layers = max(0, min(self.args.num_alignment_layers, total_layers))
            if self.args.inverted_alignment:
                alignment_layer_indices = range(total_layers - num_layers, total_layers)
            else:
                alignment_layer_indices = range(num_layers)
        else:
            alignment_layer_indices = range(total_layers)

        # Cross entropy loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        loss = F.cross_entropy(
            compression_outputs.logits[:, num_compression_tokens - 1 : -1].flatten(0, 1),
            labels.flatten(),
            reduction="mean",
        )

        # Activation alignment loss
        loss_type = self.args.loss_type.lower()
        hybrid_alpha = self.args.hybrid_alpha
        if hybrid_alpha is not None:
            alignment_loss = 0
            for i in alignment_layer_indices:
                compression_hidden_states = compression_outputs.hidden_states[i][
                    :, num_compression_tokens:
                ]  # [batch, sequence, hidden]
                target_hidden_states = outputs.hidden_states[i]  # [batch, sequence, hidden]
                if loss_type == "l2":
                    layer_alignment_loss = (
                        F.mse_loss(compression_hidden_states, target_hidden_states, reduction="none").sum(dim=-1).sqrt().mean()
                    )
                elif loss_type == "l1":
                    layer_alignment_loss = (
                        F.l1_loss(compression_hidden_states, target_hidden_states, reduction="none").sum(dim=-1).mean()
                    )
                elif loss_type == "cosine":
                    cosine = F.cosine_similarity(compression_hidden_states, target_hidden_states, dim=-1)
                    layer_alignment_loss = (1.0 - cosine).mean()
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")
                alignment_loss = alignment_loss + layer_alignment_loss
            loss = (1 - hybrid_alpha) * loss + hybrid_alpha * alignment_loss

        model.eval()
        with torch.no_grad():
            # Accuracy by logits
            convergence_numerator = (
                compression_outputs.logits[:, num_compression_tokens - 1 : -1].argmax(dim=-1) == input_ids
            ).sum(dim=-1)
            convergence_per_sample = convergence_numerator / attention_mask.sum(dim=-1)

            if self.global_step % 100 == 0:
                # Accuracy by autoregressive generation
                # Generate tokens from compressed trained embedding
                generated_text: Optional[list] = generate_from_compression(
                    model,
                    self.processing_class,
                    united_token_embeddings[:, :num_compression_tokens],
                    max_new_tokens=self.args.max_sequence_length,
                    num_return_sequences=1,
                )
                ground_truth_text: Optional[list] = self.processing_class.batch_decode(input_ids, skip_special_tokens=True)
            else:
                generated_text = None
                ground_truth_text = None
        model.train()

        return loss, convergence_per_sample, generated_text, ground_truth_text

    def _prepare_embedding_init(self, model):
        init_method = self.args.embedding_init_method
        mvn_dist = None
        if init_method == "mvnormal":
            with torch.no_grad():
                emb_weight = None
                try:
                    emb_weight = model.model.embed_tokens.weight
                except Exception:
                    sd = model.state_dict()
                    if "transformer.wte.weight" in sd:
                        emb_weight = sd["transformer.wte.weight"]
                    else:
                        for k in sd.keys():
                            if k.endswith("embed_tokens.weight") or k.endswith("wte.weight"):
                                emb_weight = sd[k]
                                break
                if emb_weight is not None:
                    pre_expansion_embeddings = emb_weight[:-3, :] if emb_weight.shape[0] > 3 else emb_weight
                    mvn_mu = pre_expansion_embeddings.mean(dim=0)
                    n = pre_expansion_embeddings.size(0)
                    centered = pre_expansion_embeddings - mvn_mu
                    sigma = (centered.T @ centered) / max(n, 1)
                    eps = 1e-6
                    sigma = sigma + eps * torch.eye(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
                    covariance = 1e-5 * sigma
                    try:
                        mvn_dist = torch.distributions.MultivariateNormal(mvn_mu, covariance_matrix=covariance)
                    except Exception:
                        diag_cov = torch.clamp(torch.diag(covariance), min=1e-8)
                        mvn_dist = torch.distributions.MultivariateNormal(mvn_mu, covariance_matrix=torch.diag(diag_cov))
                else:
                    init_method = "random"
        return init_method, mvn_dist

    def _create_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    @staticmethod
    def _init_compression_tokens(batch_size, num_tokens, hidden_size, init_method, mvn_dist):
        if init_method == "mvnormal" and mvn_dist is not None:
            samples = mvn_dist.sample((batch_size, num_tokens))
            trainable_embeddings = torch.nn.Parameter(samples)
        else:
            trainable_embeddings = torch.nn.Parameter(torch.rand([batch_size, num_tokens, hidden_size]))
        return trainable_embeddings

    def _build_optimizer_and_scheduler(self, compression_tokens_param):
        optimizer = AdamW([compression_tokens_param], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.max_optimization_steps_per_sample,
        )
        return optimizer, lr_scheduler

    def _log_step(
        self,
        loss,
        convergence_per_sample,
        compression_token_embeddings,
        lr_scheduler,
        generated_text: Optional[list[str]],
        ground_truth_text: Optional[list[str]],
    ):
        if self.writer is None:
            return
        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        self.writer.add_scalar("train/convergence", convergence_per_sample.mean().item(), self.global_step)
        self.writer.add_scalar(
            "compression_token_embeddings/mean", compression_token_embeddings.mean().item(), self.global_step
        )
        self.writer.add_scalar("compression_token_embeddings/std", compression_token_embeddings.std().item(), self.global_step)
        grad_norm = compression_token_embeddings.grad.norm(2).item() if compression_token_embeddings.grad is not None else 0.0
        self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
        lr_val = lr_scheduler.get_last_lr()[0]
        self.writer.add_scalar("train/lr", lr_val, self.global_step)
        if generated_text:
            self.writer.add_text("train/generated_text", " | ".join(generated_text), self.global_step)
        if ground_truth_text:
            self.writer.add_text("train/ground_truth_text", " | ".join(ground_truth_text), self.global_step)
        flush_steps = getattr(self.args, "logging_flush_steps", 100)
        if flush_steps and self.global_step % flush_steps == 0:
            self.writer.flush()
        self.global_step += 1

    def _save_artifacts(self, rows, subdir_name):
        output_dir = self.args.output_dir
        if output_dir and len(rows) > 0:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, subdir_name)
            ds = Dataset.from_list(rows)
            ds.save_to_disk(save_path)
            return save_path
        return None

    def train(self):
        set_launch_seed(self.args.random_seed)
        device = get_device()
        model = self.model.to(device)
        freeze_model_parameters(model)
        init_method, mvn_dist = self._prepare_embedding_init(model)
        num_compression_tokens = self.args.number_of_mem_tokens

        # Collect per-sample artifacts for optional saving
        collected_rows = []
        sample_id_counter = 0

        dataloader = self._create_dataloader()
        for batch in dataloader:
            model.train()
            input_ids = batch.input_ids.squeeze(1)  # [batch, sequence]
            attention_mask = batch.attention_mask.squeeze(1)  # [batch, sequence]
            batch_size = input_ids.shape[0]
            with torch.no_grad():
                token_embeddings = model.model.embed_tokens(input_ids)  # [batch, sequence, hidden]
                hidden_size = token_embeddings.shape[-1]

            # Trainable compression tokens per sample
            compression_token_embeddings = self._init_compression_tokens(
                batch_size, num_compression_tokens, hidden_size, init_method, mvn_dist
            )  # [batch, mem, hidden]
            compression_attention_mask = torch.tensor([1], dtype=attention_mask.dtype).repeat(
                batch_size, num_compression_tokens
            )  # [batch, mem]

            optimizer, lr_scheduler = self._build_optimizer_and_scheduler(compression_token_embeddings)

            loss, convergence_per_sample, generated_text, ground_truth_text = None, None, None, None
            progress_bar = tqdm(
                range(self.args.max_optimization_steps_per_sample), total=self.args.max_optimization_steps_per_sample
            )
            progress_bar.set_description("Training")
            for _ in progress_bar:
                # Rebuild concatenations each step to avoid reusing the same autograd graph
                united_token_embeddings = torch.cat(
                    [compression_token_embeddings, token_embeddings],
                    dim=1,
                )  # [batch, mem + sequence, hidden]
                united_attention_mask = torch.cat(
                    [compression_attention_mask, attention_mask],
                    dim=1,
                )  # [batch, mem + sequence]
                loss, convergence_per_sample, generated_text, ground_truth_text = self.compute_loss(
                    model,
                    input_ids,
                    token_embeddings,
                    attention_mask,
                    united_token_embeddings,
                    united_attention_mask,
                    num_compression_tokens,
                )
                # Calculate gradients and update compression embeddings
                loss.backward()
                optimizer.step()

                # Log current step progress
                with torch.no_grad():
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        loss=loss.item(),
                        convergece_per_sample=convergence_per_sample.mean().item(),
                        compression_tokens_mean=compression_token_embeddings.mean().item(),
                        compression_tokens_std=compression_token_embeddings.std().item(),
                        grad=compression_token_embeddings.grad.norm(2).item(),
                        lr=lr_scheduler.get_last_lr()[0],
                    )
                    self._log_step(
                        loss,
                        convergence_per_sample,
                        compression_token_embeddings,
                        lr_scheduler,
                        generated_text,
                        ground_truth_text,
                    )

                # Update learning rate
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            # After optimizing this batch's compression tokens, record artifacts per sample (once per sample)
            with torch.no_grad():
                tokenizer = self.processing_class
                last_loss = float(loss.item())
                last_convergence = convergence_per_sample.cpu()
                compression_token_embeddings_cpu = compression_token_embeddings.detach().cpu()
                for j in range(batch_size):
                    sample_attention_mask = attention_mask[j].bool()
                    sample_input_ids = input_ids[j][sample_attention_mask]
                    sample_text = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
                    embedding = compression_token_embeddings_cpu[j].to(torch.float32).numpy().tolist()
                    compression_token_embeddings_mean = float(compression_token_embeddings_cpu[j].mean().item())
                    compression_token_embeddings_std = float(compression_token_embeddings_cpu[j].std().item())
                    # TODO: Add more description data
                    collected_rows.append(
                        {
                            "sample_id": int(sample_id_counter),
                            "text": sample_text,
                            "embedding": embedding,  # [mem, hidden]
                            "final_loss": last_loss,
                            "final_convergence": float(last_convergence[j].item()),
                            "compression_tokens_mean": compression_token_embeddings_mean,
                            "compression_tokens_std": compression_token_embeddings_std,
                            "num_input_tokens": int(sample_attention_mask.sum().item()),
                            "num_compression_tokens": int(num_compression_tokens),
                            "hidden_size": hidden_size,
                            "loss_type": self.args.loss_type,
                            "model_checkpoint": self.args.model_checkpoint,
                            "max_optimization_steps_per_sample": self.args.max_optimization_steps_per_sample,
                        }
                    )
                    sample_id_counter += 1

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        # Persist artifacts
        save_path = self._save_artifacts(collected_rows, "compressed_prefixes")
        if save_path is not None:
            return save_path
        return None

    def progressive_train(self):
        device = get_device()
        self._set_seed_if_any()
        model = self.model.to(device)
        self._freeze_model_params(model)
        init_method, mvn_dist = self._prepare_embedding_init(model)

        dataloader = self._create_dataloader()

        num_compression_tokens = getattr(self.args, "number_of_mem_tokens", 1)
        threshold = getattr(self.args, "progressive_convergence_threshold", 0.99)
        step_increment = getattr(self.args, "progressive_step", 16)
        min_len = getattr(self.args, "progressive_min_seq_len", 16)
        max_stages_cap = getattr(self.args, "progressive_max_stages", 0)

        collected_rows = []
        sample_id_counter = 0

        for batch in dataloader:
            batch_size = batch["input_ids"].shape[0]
            full_input_ids = batch.input_ids.squeeze(1)
            with torch.no_grad():
                full_model_token_embeddings = model.model.embed_tokens(full_input_ids)
            full_attention_mask = batch.attention_mask.squeeze(1)

            hidden_size = full_model_token_embeddings.shape[-1]
            compression_tokens = self._init_compression_tokens(
                batch_size, num_compression_tokens, hidden_size, init_method, mvn_dist
            )
            compression_tokens_attention_mask = torch.tensor([[1]], dtype=full_attention_mask.dtype).repeat(
                batch_size, num_compression_tokens
            )

            optimizer, lr_scheduler = self._build_optimizer_and_scheduler(compression_tokens)

            # Determine maximum effective length present in this batch (exclude padding)
            per_sample_lengths = full_attention_mask.sum(dim=1).tolist()
            max_len = int(max(per_sample_lengths)) if len(per_sample_lengths) > 0 else full_attention_mask.shape[1]
            seq_len = min(min_len, max_len)
            stage_index = 0

            while True:
                # Slice to current effective sequence length
                input_ids = full_input_ids[:, :seq_len]
                inputs_embeds = full_model_token_embeddings[:, :seq_len, :]
                attention_mask = full_attention_mask[:, :seq_len]

                pbar = tqdm(
                    range(self.args.max_optimization_steps_per_sample),
                    total=self.args.max_optimization_steps_per_sample,
                )
                pbar.set_description(f"Stage L={seq_len}")
                last_loss_val = None
                last_conv = None
                steps_taken = 0

                for i in pbar:
                    model_tokens_with_compression_tokens = torch.cat([compression_tokens, inputs_embeds], dim=1)
                    attention_mask_with_compression_tokens = torch.cat(
                        [compression_tokens_attention_mask, attention_mask], dim=1
                    )
                    loss, convergece_per_sample = self.compute_loss(
                        model,
                        input_ids,
                        inputs_embeds,
                        attention_mask,
                        model_tokens_with_compression_tokens,
                        attention_mask_with_compression_tokens,
                        num_compression_tokens,
                    )
                    loss.backward()
                    steps_taken += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=loss.item(),
                        convergece_per_sample=convergece_per_sample.mean().item(),
                        compression_tokens_mean=compression_tokens.mean().item(),
                        compression_tokens_std=compression_tokens.std().item(),
                        grad=compression_tokens.grad.norm(2).item(),
                        lr=lr_scheduler.get_last_lr()[0],
                    )

                    self._log_step(loss, convergece_per_sample, compression_tokens, lr_scheduler)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    last_loss_val = float(loss.item())
                    last_conv = convergece_per_sample.detach().cpu()

                    if convergece_per_sample.mean().item() >= threshold:
                        break

                # Save snapshot for this stage
                with torch.no_grad():
                    tokenizer = self.processing_class
                    comp_tokens_cpu = compression_tokens.detach().cpu()
                    for j in range(batch_size):
                        attn = attention_mask[j].bool()
                        ids = input_ids[j][attn]
                        text = tokenizer.decode(ids.tolist(), skip_special_tokens=True) if tokenizer is not None else ""
                        embedding = comp_tokens_cpu[j].to(torch.float32).numpy().tolist()
                        collected_rows.append(
                            {
                                "sample_id": int(sample_id_counter + j),
                                "stage_index": int(stage_index),
                                "stage_seq_len": int(seq_len),
                                "text": text,
                                "embedding": embedding,
                                "final_loss": float(last_loss_val) if last_loss_val is not None else None,
                                "final_convergence": float(last_conv[j].item()) if last_conv is not None else None,
                                "num_input_tokens": int(attn.sum().item()),
                                "num_compression_tokens": int(num_compression_tokens),
                                "hidden_size": int(comp_tokens_cpu.shape[-1]),
                                "loss_type": getattr(self.args, "loss_type", "l2"),
                                "model_checkpoint": getattr(self.args, "model_checkpoint", ""),
                                "max_optimization_steps_per_sample": int(
                                    getattr(self.args, "max_optimization_steps_per_sample", 0)
                                ),
                                "convergence_threshold": float(threshold),
                                "steps_taken": int(steps_taken),
                            }
                        )

                stage_index += 1
                # Advance to next length or exit
                if seq_len >= max_len:
                    break
                if max_stages_cap and stage_index >= max_stages_cap:
                    break
                seq_len = min(seq_len + step_increment, max_len)

            sample_id_counter += batch_size

        # Close writer
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        save_path = self._save_artifacts(collected_rows, "progressive_prefixes")
        if save_path is not None:
            return save_path
        return None

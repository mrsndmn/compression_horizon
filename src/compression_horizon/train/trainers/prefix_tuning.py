"""Prefix tuning trainer: PEFT-based prefix tuning per sample."""

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from compression_horizon.train.loss import (
    compute_hybrid_cross_entropy_and_alignment_loss_no_prefix,
    token_argmax_match_rate,
)
from compression_horizon.train.trainers.base import BaseTrainer
from compression_horizon.utils.launch import freeze_model_parameters, get_device, set_launch_seed


def _find_prefix_embedding_parameter(peft_model: nn.Module, num_virtual_tokens: int) -> tuple[str, torch.nn.Parameter] | None:
    """Best-effort: locate PEFT prefix/prompt embedding parameter for logging/saving."""
    candidates: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in peft_model.named_parameters():
        if not param.requires_grad or param.ndim != 2 or param.shape[0] != num_virtual_tokens:
            continue
        priority = 0
        lname = name.lower()
        if "prompt" in lname or "prefix" in lname:
            priority += 2
        if "embed" in lname:
            priority += 1
        candidates.append((f"{priority:02d}:{name}", param))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_name = candidates[0][0].split(":", 1)[1]
    return best_name, candidates[0][1]


class PrefixTuningTrainer(BaseTrainer):
    """Trainer for PEFT prefix tuning: per-sample prefix embeddings."""

    def train(self) -> str | None:
        """Run prefix tuning training. Returns save path or None."""
        return self._train_prefix_tuning()

    def _train_prefix_tuning(self) -> str | None:
        set_launch_seed(self.args.random_seed)
        device = get_device()

        base_model = self.model.to(device)
        freeze_model_parameters(base_model)
        base_model.eval()

        try:
            from peft import PrefixTuningConfig, TaskType, get_peft_model
        except Exception as e:
            raise RuntimeError("peft is required for --train_prefix_tuning. Install it (e.g. `uv add peft`).") from e

        num_virtual_tokens = int(getattr(self.args, "number_of_mem_tokens", 1))
        if num_virtual_tokens < 1:
            raise ValueError(f"number_of_mem_tokens must be >= 1 for prefix tuning, got {num_virtual_tokens}")

        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
        )
        peft_model = get_peft_model(base_model, peft_config).to(device)

        dataloader = self._create_dataloader()

        collected_rows = []
        sample_id_counter = 0
        final_prefix_embeddings_cpu = None

        tokenizer = self.processing_class
        hidden_size = base_model.config.hidden_size

        for batch in tqdm(dataloader):
            input_ids_b = batch.input_ids.squeeze(1).to(device)
            attention_mask_b = batch.attention_mask.squeeze(1).to(device)
            batch_size = input_ids_b.shape[0]

            for j in range(batch_size):
                input_ids = input_ids_b[j].unsqueeze(0)
                attention_mask = attention_mask_b[j].unsqueeze(0)

                with torch.no_grad():
                    token_embeddings = base_model.get_input_embeddings()(input_ids)
                target_hidden_states = self.compute_hidden_states(base_model, token_embeddings, attention_mask)

                peft_model.train()

                trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
                if not trainable_params:
                    raise RuntimeError("No trainable parameters found in PEFT model for prefix tuning")

                with torch.no_grad():
                    for p in trainable_params:
                        if p.ndim == 0:
                            continue
                        nn.init.normal_(p, mean=0.0, std=0.02)

                optimizer, lr_scheduler = self._build_optimizer_and_scheduler(
                    trainable_params,
                    num_training_steps=self.args.max_optimization_steps_per_sample,
                )

                found = _find_prefix_embedding_parameter(peft_model, num_virtual_tokens)
                prefix_name, prefix_param = found if found is not None else (None, None)

                initialization_prefix_embedding = None
                if prefix_param is not None:
                    initialization_prefix_embedding = prefix_param.detach().clone().to(torch.float32).cpu()

                loss = None
                alignment_loss = None
                convergence_per_sample = None

                progress_bar = tqdm(
                    range(self.args.max_optimization_steps_per_sample),
                    total=self.args.max_optimization_steps_per_sample,
                    leave=False,
                )
                progress_bar.set_description("Prefix tuning")

                for step_i in progress_bar:
                    outputs = peft_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=(self.args.loss_type != "cross_entropy"),
                        use_cache=False,
                    )
                    loss, alignment_loss = compute_hybrid_cross_entropy_and_alignment_loss_no_prefix(
                        logits=outputs.logits,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        target_hidden_states=target_hidden_states,
                        compression_hidden_states=outputs.hidden_states,
                        num_alignment_layers=self.args.num_alignment_layers,
                        inverted_alignment=self.args.inverted_alignment,
                        loss_type=self.args.loss_type,
                        hybrid_alpha=self.args.hybrid_alpha,
                    )

                    loss.backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    with torch.no_grad():
                        convergence_per_sample = token_argmax_match_rate(
                            outputs.logits,
                            input_ids,
                            attention_mask,
                        )
                        log_lr = self.args.learning_rate
                        if lr_scheduler is not None:
                            log_lr = lr_scheduler.get_last_lr()[0]
                        progress_bar.set_postfix(
                            loss=float(loss.item()),
                            convergence=float(convergence_per_sample.mean().item()),
                            lr=float(log_lr),
                            prefix_param=(prefix_name or "n/a"),
                        )
                        self._log_step(
                            loss,
                            alignment_loss,
                            convergence_per_sample,
                            prefix_param,
                            lr_scheduler,
                            embedding_namespace="prefix_tuning",
                        )

                    if float(convergence_per_sample.mean().item()) >= 1.0:
                        break

                with torch.no_grad():
                    sample_attention_mask = attention_mask[0].bool()
                    sample_input_ids = input_ids[0][sample_attention_mask]
                    sample_text = tokenizer.decode(sample_input_ids, skip_special_tokens=True)

                    prefix_embedding_cpu = None
                    if prefix_param is not None:
                        prefix_embedding_cpu = prefix_param.detach().clone().to(torch.float32).cpu()
                        final_prefix_embeddings_cpu = prefix_embedding_cpu.unsqueeze(0)

                    collected_rows.append(
                        {
                            "sample_id": sample_id_counter,
                            "text": sample_text,
                            "prefix_embedding": (
                                prefix_embedding_cpu.numpy().tolist() if prefix_embedding_cpu is not None else None
                            ),
                            "initialization_prefix_embedding": (
                                initialization_prefix_embedding.numpy().tolist()
                                if initialization_prefix_embedding is not None
                                else None
                            ),
                            "final_loss": (float(loss.item()) if loss is not None else None),
                            "final_convergence": (
                                float(convergence_per_sample.item()) if convergence_per_sample is not None else None
                            ),
                            "num_virtual_tokens": int(num_virtual_tokens),
                            "hidden_size": int(hidden_size),
                            "loss_type": self.args.loss_type,
                            "hybrid_alpha": self.args.hybrid_alpha,
                            "dtype": self.args.dtype,
                            "num_alignment_layers": self.args.num_alignment_layers,
                            "model_checkpoint": self.args.model_checkpoint,
                            "max_optimization_steps_per_sample": self.args.max_optimization_steps_per_sample,
                        }
                    )
                    sample_id_counter += 1

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        return self._save_artifacts(
            collected_rows,
            tensor=final_prefix_embeddings_cpu,
            tensor_filename="prefix_tuning_embeddings.pt",
            subdir_name="prefix_tuning_prefixes",
        )

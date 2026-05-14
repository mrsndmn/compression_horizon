"""Base trainer class with shared methods for all compression trainers."""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datasets import Dataset
from sklearn.decomposition import PCA
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler

from compression_horizon.inference.generation import generate_from_compression
from compression_horizon.train.loss import (
    compute_hybrid_cross_entropy_and_alignment_loss,
    token_argmax_match_rate_with_prefix,
)


class BaseTrainer:
    """Base class for compression trainers. Subclasses implement train()."""

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
        log_dir = self.args.logging_dir

        mixed_precision = "no"
        ddp_kwargs = None
        if args.ddp_find_unused_parameters:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=bool(args.ddp_find_unused_parameters))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs] if ddp_kwargs is not None else None,
        )

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir and self.accelerator.is_main_process else None

        self.global_step = 0

    def train(self) -> str | None:
        """Run training. Subclasses must override. Returns artifact path or output_dir or None."""
        raise NotImplementedError("Subclasses must implement train()")

    def compute_loss(
        self,
        model,
        input_ids,
        token_embeddings,
        attention_mask,
        united_token_embeddings,
        united_attention_mask,
        num_compression_tokens,
        target_hidden=None,
    ):
        loss_type = self.args.loss_type.lower()

        if loss_type != "cross_entropy" and target_hidden is None:
            target_hidden = self.compute_target_hidden(model, token_embeddings, attention_mask)

        extra_kwargs = {}
        if self.args.fix_position_ids:
            position_ids = torch.arange(
                -num_compression_tokens,
                token_embeddings.size(1),
                device=token_embeddings.device,
            )
            position_ids[:num_compression_tokens] = 0
            position_ids = position_ids.repeat(token_embeddings.size(0), 1)
            extra_kwargs["position_ids"] = position_ids
        compression_outputs = model(
            inputs_embeds=united_token_embeddings,
            attention_mask=united_attention_mask,
            output_hidden_states=(loss_type != "cross_entropy"),
            **extra_kwargs,
        )

        hybrid_alpha = self.args.hybrid_alpha
        loss, alignment_loss = compute_hybrid_cross_entropy_and_alignment_loss(
            logits=compression_outputs.logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_prefix_tokens=num_compression_tokens,
            target_hidden_states=target_hidden,
            compression_hidden_states=compression_outputs.hidden_states,
            num_alignment_layers=self.args.num_alignment_layers,
            inverted_alignment=self.args.inverted_alignment,
            loss_type=loss_type,
            hybrid_alpha=hybrid_alpha,
        )

        model.eval()
        with torch.no_grad():
            convergence_per_sample = token_argmax_match_rate_with_prefix(
                compression_outputs.logits,
                input_ids,
                attention_mask,
                num_compression_tokens,
            )

            if self.global_step % 100 == 0 and self.args.generate_in_compute_loss:
                generated_text: Optional[list] = generate_from_compression(
                    model,
                    self.processing_class,
                    united_token_embeddings[:, :num_compression_tokens],
                    max_new_tokens=self.args.max_sequence_length,
                    num_return_sequences=1,
                )
                ground_truth_text: Optional[list[str]] = self.processing_class.batch_decode(input_ids, skip_special_tokens=True)
            else:
                generated_text = None
                ground_truth_text = None
        model.eval()

        return (
            loss,
            alignment_loss,
            convergence_per_sample,
            generated_text,
            ground_truth_text,
        )

    def _sample_prefix_lengths(self, attention_mask: torch.Tensor) -> torch.LongTensor:
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
        self,
        *,
        compression_embeds: torch.Tensor,
        token_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_lengths: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        p = prefix_lengths.to(device=device).to(torch.long)
        p = torch.clamp(p, min=0)
        p = torch.minimum(p, max_prefix)

        out_len = 1 + seq_len
        inputs_embeds_new = torch.zeros((bsz, out_len, hidden), device=device, dtype=token_embeddings.dtype)
        attention_mask_new = torch.zeros((bsz, out_len), device=device, dtype=attention_mask.dtype)
        labels_new = torch.full((bsz, out_len), fill_value=-100, device=device, dtype=input_ids.dtype)

        inputs_embeds_new[:, 0:1, :] = compression_embeds
        attention_mask_new[:, 0] = 1

        # Reconstruction objective: the new sequence is [compression_token, prefix_tokens],
        # and the model is trained to autoregressively predict the *prefix tokens themselves*
        # (matching what progressive cramming evaluates), NOT the post-prefix continuation.
        ar = torch.arange(seq_len, device=device, dtype=torch.long)
        valid = ar.unsqueeze(0) < p.unsqueeze(1)  # [B, T]: i < prefix_length[b]
        src_idx_safe = ar.unsqueeze(0).expand(bsz, -1).contiguous()

        gathered_embeds = token_embeddings.gather(1, src_idx_safe.unsqueeze(-1).expand(-1, -1, hidden))
        gathered_ids = input_ids.gather(1, src_idx_safe)

        if valid.dtype != torch.bool:
            valid = valid.to(torch.bool)

        inputs_embeds_new[:, 1:, :] = gathered_embeds * valid.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        attention_mask_new[:, 1:] = valid.to(dtype=attention_mask.dtype)
        labels_new[:, 1:] = torch.where(valid, gathered_ids, torch.full_like(gathered_ids, -100))

        return inputs_embeds_new, attention_mask_new, labels_new

    def _prepare_embedding_init(self, model):
        init_method = self.args.embedding_init_method
        mvn_dist = None
        pca_components = None
        pca_mean = None
        loaded_embeddings = None

        if init_method == "load_from_disk":
            if not self.args.embedding_init_path or not os.path.exists(self.args.embedding_init_path):
                if not self.args.embedding_init_path:
                    if self.args.output_dir:
                        os.makedirs(self.args.output_dir, exist_ok=True)
                        save_path = os.path.join(self.args.output_dir, "generated_compression_embeddings.pt")
                    else:
                        save_path = "generated_compression_embeddings.pt"
                else:
                    save_path = self.args.embedding_init_path
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)

                hidden_size = model.config.hidden_size
                num_compression_tokens = self.args.number_of_mem_tokens

                gen_init_method = self.args.load_from_disk_embedding_init_method
                gen_mvn_dist = None
                gen_pca_components = None
                gen_pca_mean = None
                gen_loaded_embeddings = None

                if gen_init_method == "mvnormal":
                    with torch.no_grad():
                        emb_weight = None
                        try:
                            emb_weight = model.get_input_embeddings().weight
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
                            pre_expansion_embeddings = (emb_weight[:-3, :] if emb_weight.shape[0] > 3 else emb_weight).cpu()
                            mvn_mu = pre_expansion_embeddings.mean(dim=0).to(torch.float32)
                            n = pre_expansion_embeddings.size(0)
                            centered = pre_expansion_embeddings.to(torch.float32) - mvn_mu
                            sigma = (centered.T @ centered) / max(n, 1)
                            eps = 1e-6
                            sigma = sigma + eps * torch.eye(sigma.shape[0], device=sigma.device, dtype=sigma.dtype)
                            covariance = 1e-5 * sigma
                            try:
                                gen_mvn_dist = torch.distributions.MultivariateNormal(mvn_mu, covariance_matrix=covariance)
                            except Exception:
                                diag_cov = torch.clamp(torch.diag(covariance), min=1e-8)
                                gen_mvn_dist = torch.distributions.MultivariateNormal(
                                    mvn_mu, covariance_matrix=torch.diag(diag_cov)
                                )
                        else:
                            raise ValueError("cant run mv normal initialization method")
                elif gen_init_method == "pretrained_pca":
                    if not self.args.pretrained_pca_path:
                        raise ValueError(
                            "pretrained_pca_path must be specified when using "
                            "load_from_disk_embedding_init_method=pretrained_pca"
                        )
                    if not os.path.exists(self.args.pretrained_pca_path):
                        raise ValueError(f"pretrained_pca_path does not exist: {self.args.pretrained_pca_path}")
                    progressive_ds = Dataset.load_from_disk(self.args.pretrained_pca_path)
                    all_embeddings = []
                    for i in range(len(progressive_ds)):
                        row = progressive_ds[i]
                        if int(row.get("sample_id", -1)) == 0:
                            embedding = row.get("embedding")
                            if embedding is not None:
                                if isinstance(embedding, list):
                                    emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                                else:
                                    emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                                emb_flat = emb_tensor.reshape(-1).to(torch.float32).detach().cpu().numpy()
                                all_embeddings.append(emb_flat)
                    if len(all_embeddings) == 0:
                        raise ValueError(f"No embeddings found for sample_id=0 in {self.args.pretrained_pca_path}")
                    X = np.stack(all_embeddings, axis=0)
                    n_components = min(self.args.pretrained_pca_num_components, X.shape[0] - 1, X.shape[1])
                    if n_components < 1:
                        raise ValueError(f"Cannot fit PCA: need at least 2 samples, got {X.shape[0]}")
                    pca = PCA(n_components=n_components, random_state=42)
                    pca.fit(X)
                    gen_pca_components = torch.tensor(pca.components_, dtype=torch.float32)
                    gen_pca_mean = torch.tensor(pca.mean_, dtype=torch.float32)

                generated_embeddings = self._init_compression_tokens(
                    1,
                    num_compression_tokens,
                    hidden_size,
                    gen_init_method,
                    gen_mvn_dist,
                    token_embeddings=None,
                    single_compressed_embeddings_initialization=None,
                    pca_components=gen_pca_components,
                    pca_mean=gen_pca_mean,
                    loaded_embeddings=gen_loaded_embeddings,
                )
                generated_embeddings_tensor = generated_embeddings.data.detach().clone().cpu()
                torch.save(generated_embeddings_tensor, save_path)
                print(
                    f"Generated embeddings using method '{gen_init_method}' and saved to {save_path}: "
                    f"shape {generated_embeddings_tensor.shape}"
                )
                loaded_embeddings = generated_embeddings_tensor
            else:
                loaded_embeddings = torch.load(self.args.embedding_init_path, map_location="cpu")
                if isinstance(loaded_embeddings, dict):
                    if "compression_embeddings" in loaded_embeddings:
                        loaded_embeddings = loaded_embeddings["compression_embeddings"]
                    elif "state_dict" in loaded_embeddings:
                        for key in loaded_embeddings["state_dict"].keys():
                            if "compression" in key.lower() or "embedding" in key.lower():
                                loaded_embeddings = loaded_embeddings["state_dict"][key]
                                break
                        else:
                            raise ValueError(
                                f"Could not find compression embeddings in state_dict at " f"{self.args.embedding_init_path}"
                            )
                    else:
                        loaded_embeddings = next(iter(loaded_embeddings.values()))
                if not isinstance(loaded_embeddings, torch.Tensor):
                    loaded_embeddings = torch.tensor(loaded_embeddings, dtype=torch.float32)
                loaded_embeddings = loaded_embeddings.to(torch.float32)
                print(f"Loaded embeddings from {self.args.embedding_init_path}: " f"shape {loaded_embeddings.shape}")

        elif init_method == "pretrained_pca":
            if not self.args.pretrained_pca_path:
                raise ValueError("pretrained_pca_path must be specified when using " "embedding_init_method=pretrained_pca")
            if not os.path.exists(self.args.pretrained_pca_path):
                raise ValueError(f"pretrained_pca_path does not exist: {self.args.pretrained_pca_path}")

            progressive_ds = Dataset.load_from_disk(self.args.pretrained_pca_path)

            all_embeddings = []
            for i in range(len(progressive_ds)):
                row = progressive_ds[i]
                embedding = row.get("embedding")
                if embedding is not None:
                    if isinstance(embedding, list):
                        emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                    else:
                        emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                    emb_flat = emb_tensor.reshape(-1).to(torch.float32).detach().cpu().numpy()
                    all_embeddings.append(emb_flat)

            if len(all_embeddings) == 0:
                raise ValueError(f"No embeddings found for sample_id=0 in {self.args.pretrained_pca_path}")

            X = np.stack(all_embeddings, axis=0)

            n_components = min(self.args.pretrained_pca_num_components, X.shape[0] - 1, X.shape[1])
            if n_components < 1:
                raise ValueError(f"Cannot fit PCA: need at least 2 samples, got {X.shape[0]}")

            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(X)

            pca_components = torch.tensor(pca.components_, dtype=torch.float32)
            pca_mean = torch.tensor(pca.mean_, dtype=torch.float32)
            print(
                f"Loaded PCA from {self.args.pretrained_pca_path}: {n_components} components, "
                f"explained variance: {pca.explained_variance_ratio_.sum():.4f}"
            )

        elif init_method == "mvnormal":
            with torch.no_grad():
                emb_weight = None
                try:
                    emb_weight = model.get_input_embeddings().weight
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
                        mvn_dist = torch.distributions.MultivariateNormal(
                            mvn_mu.to(torch.float32),
                            covariance_matrix=covariance.to(torch.float32),
                        )
                    except Exception:
                        diag_cov = torch.clamp(torch.diag(covariance), min=1e-8)
                        mvn_dist = torch.distributions.MultivariateNormal(
                            mvn_mu.to(torch.float32),
                            covariance_matrix=torch.diag(diag_cov).to(torch.float32),
                        )
                else:
                    raise ValueError("cant run mv normal initialization method")
        return init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings

    def _create_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    @staticmethod
    def _init_compression_tokens(
        batch_size,
        num_tokens,
        hidden_size,
        init_method,
        mvn_dist,
        token_embeddings=None,
        single_compressed_embeddings_initialization=None,
        pca_components=None,
        pca_mean=None,
        loaded_embeddings=None,
    ):
        if init_method == "mvnormal" and mvn_dist is not None:
            samples = mvn_dist.sample((batch_size, num_tokens))
            trainable_embeddings = torch.nn.Parameter(samples.to(dtype=torch.float32))
        elif init_method == "zeros":
            trainable_embeddings = torch.nn.Parameter(torch.zeros([batch_size, num_tokens, hidden_size], dtype=torch.float32))
        elif init_method == "single_random":
            if single_compressed_embeddings_initialization is not None:
                trainable_embeddings = torch.nn.Parameter(
                    single_compressed_embeddings_initialization.detach().clone().repeat(batch_size, 1, 1)
                )
            else:
                single_random_embedding = torch.rand([1, num_tokens, hidden_size], dtype=torch.float32)
                single_random_embedding = single_random_embedding.repeat(batch_size, 1, 1)
                trainable_embeddings = torch.nn.Parameter(single_random_embedding)
        elif init_method == "single_random0.02":
            if single_compressed_embeddings_initialization is not None:
                trainable_embeddings = torch.nn.Parameter(
                    single_compressed_embeddings_initialization.detach().clone().repeat(batch_size, 1, 1)
                )
            else:
                single_random_embedding = torch.rand([1, num_tokens, hidden_size], dtype=torch.float32)
                single_random_embedding = single_random_embedding.repeat(batch_size, 1, 1)
                trainable_embeddings = torch.nn.Parameter(single_random_embedding)
        elif init_method == "random":
            trainable_embeddings = torch.nn.Parameter(torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32))
        elif init_method == "random0.2":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.2
            )
        elif init_method == "random0.02":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.02
            )
        elif init_method == "random_norm":
            trainable_embeddings = torch.nn.Parameter(torch.randn([batch_size, num_tokens, hidden_size], dtype=torch.float32))
        elif init_method == "random_norm_0.2":
            trainable_embeddings = torch.nn.Parameter(
                torch.randn([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.2
            )
        elif init_method == "random_norm_0.02":
            trainable_embeddings = torch.nn.Parameter(
                torch.randn([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.02
            )
        elif init_method == "random0.002":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.002
            )
        elif init_method == "random0.0002":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 0.0002
            )
        elif init_method == "random5":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 5
            )
        elif init_method == "neg_random":
            trainable_embeddings = torch.nn.Parameter(
                torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 2 - 1
            )
        elif init_method == "neg_random0.2":
            trainable_embeddings = torch.nn.Parameter(
                (torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 2 - 1) * 0.2
            )
        elif init_method == "neg_random5":
            trainable_embeddings = torch.nn.Parameter(
                (torch.rand([batch_size, num_tokens, hidden_size], dtype=torch.float32) * 2 - 1) * 5
            )
        elif init_method == "mean_token_embeds":
            assert token_embeddings is not None
            trainable_embeddings = torch.nn.Parameter(token_embeddings.mean(1, keepdim=True).repeat(1, num_tokens, 1))
        elif init_method == "pretrained_pca":
            assert pca_components is not None
            assert pca_mean is not None

            flattened_dim = pca_mean.shape[0]
            expected_flattened_dim = num_tokens * hidden_size
            if flattened_dim != expected_flattened_dim:
                raise ValueError(
                    f"PCA dimension mismatch: pretrained has {flattened_dim} "
                    f"but current needs {expected_flattened_dim} "
                    f"(num_tokens={num_tokens}, hidden_size={hidden_size})"
                )

            n_components_to_use = min(pca_components.shape[0], num_tokens)
            pca_coeffs = torch.randn([batch_size, n_components_to_use], dtype=torch.float32) * 0.1
            reconstructed_flat = torch.matmul(pca_coeffs, pca_components[:n_components_to_use])
            reconstructed_flat = reconstructed_flat + pca_mean.unsqueeze(0)
            trainable_embeddings = torch.nn.Parameter(reconstructed_flat.reshape(batch_size, num_tokens, hidden_size))
        elif init_method == "load_from_disk":
            assert loaded_embeddings is not None
            if len(loaded_embeddings.shape) == 2:
                if loaded_embeddings.shape[0] != num_tokens or loaded_embeddings.shape[1] != hidden_size:
                    raise ValueError(
                        f"Loaded embeddings shape mismatch: got {loaded_embeddings.shape}, "
                        f"expected [{num_tokens}, {hidden_size}] or [1, {num_tokens}, {hidden_size}]"
                    )
                trainable_embeddings = torch.nn.Parameter(
                    loaded_embeddings.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.float32)
                )
            elif len(loaded_embeddings.shape) == 3:
                if loaded_embeddings.shape[1] != num_tokens or loaded_embeddings.shape[2] != hidden_size:
                    raise ValueError(
                        f"Loaded embeddings shape mismatch: got {loaded_embeddings.shape}, "
                        f"expected [1, {num_tokens}, {hidden_size}] or "
                        f"[{batch_size}, {num_tokens}, {hidden_size}]"
                    )
                if loaded_embeddings.shape[0] == 1:
                    trainable_embeddings = torch.nn.Parameter(loaded_embeddings.repeat(batch_size, 1, 1).to(torch.float32))
                elif loaded_embeddings.shape[0] == batch_size:
                    trainable_embeddings = torch.nn.Parameter(loaded_embeddings.to(torch.float32))
                else:
                    raise ValueError(
                        f"Loaded embeddings batch size mismatch: got {loaded_embeddings.shape[0]}, "
                        f"expected 1 or {batch_size}"
                    )
            else:
                raise ValueError(f"Loaded embeddings must be 2D or 3D tensor, got shape {loaded_embeddings.shape}")
        else:
            raise ValueError(f"unsupported init method: {init_method}")
        return trainable_embeddings

    def _build_optimizer_and_scheduler(self, params, num_training_steps=None, num_processes=1):
        print("number of optimized params:", sum(p.numel() for p in params))

        if self.args.optim == "adamw_torch":
            optimizer = AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
            )
        elif self.args.optim == "sgd":
            optimizer = SGD(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError("Only SGD and adamw_torch are supported")

        lr_scheduler = None
        if num_training_steps is not None:
            print("self.args.lr_scheduler_type", self.args.lr_scheduler_type)
            scheduler_kwargs = {
                "optimizer": optimizer,
                "num_warmup_steps": self.args.warmup_steps * num_processes,
                "num_training_steps": num_training_steps * num_processes,
            }

            if self.args.lr_scheduler_kwargs is not None:
                assert self.args.lr_scheduler_kwargs["min_lr"] < self.args.learning_rate, (
                    f"min_lr must be lower than regular LR, "
                    f"{self.args.lr_scheduler_kwargs['min_lr']} < {self.args.learning_rate}"
                )

            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                **scheduler_kwargs,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )

        return optimizer, lr_scheduler

    def _log_step(
        self,
        loss,
        alignment_loss,
        convergence_per_sample,
        compression_token_embeddings,
        lr_scheduler,
        generated_text: Optional[list[str]],
        ground_truth_text: Optional[list[str]],
    ):
        if self.writer is None:
            return
        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        if alignment_loss is not None:
            self.writer.add_scalar("train/alignment_loss", alignment_loss.item(), self.global_step)
        self.writer.add_scalar("train/convergence", convergence_per_sample.mean().item(), self.global_step)
        self.writer.add_scalar(
            "compression_token_embeddings/mean",
            compression_token_embeddings.mean().item(),
            self.global_step,
        )
        self.writer.add_scalar(
            "compression_token_embeddings/std",
            compression_token_embeddings.std().item(),
            self.global_step,
        )
        grad_norm = compression_token_embeddings.grad.norm(2).item() if compression_token_embeddings.grad is not None else 0.0
        self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
        if lr_scheduler is not None:
            lr_val = lr_scheduler.get_last_lr()[0]
            self.writer.add_scalar("train/lr", lr_val, self.global_step)
        if generated_text:
            self.writer.add_text("train/generated_text", " | ".join(generated_text), self.global_step)
        if ground_truth_text:
            self.writer.add_text(
                "train/ground_truth_text",
                " | ".join(ground_truth_text),
                self.global_step,
            )
        flush_steps = getattr(self.args, "logging_flush_steps", 100)
        if flush_steps and self.global_step % flush_steps == 0:
            self.writer.flush()
        self.global_step += 1

    def _save_artifacts(self, compression_token_embeddings: torch.Tensor, rows, subdir_name):
        output_dir = self.args.output_dir
        if output_dir and len(rows) > 0:
            os.makedirs(output_dir, exist_ok=True)
            if compression_token_embeddings is not None:
                save_path = os.path.join(output_dir, "compression_embeddings.pt")
                torch.save(compression_token_embeddings, save_path)
            save_path = os.path.join(output_dir, subdir_name)
            ds = Dataset.from_list(rows)
            ds.save_to_disk(save_path)
            return save_path
        return None

    def _save_prefix_tuning_artifacts(self, prefix_embeddings: torch.Tensor | None, rows, subdir_name: str):
        output_dir = self.args.output_dir
        if output_dir and len(rows) > 0:
            os.makedirs(output_dir, exist_ok=True)
            if prefix_embeddings is not None:
                save_path = os.path.join(output_dir, "prefix_tuning_embeddings.pt")
                torch.save(prefix_embeddings, save_path)
            save_path = os.path.join(output_dir, subdir_name)
            ds = Dataset.from_list(rows)
            ds.save_to_disk(save_path)
            return save_path
        return None

    @staticmethod
    def _find_prefix_embedding_parameter(
        peft_model: nn.Module, num_virtual_tokens: int
    ) -> tuple[str, torch.nn.Parameter] | None:
        """Best-effort: locate PEFT prefix/prompt embedding parameter for logging/saving."""
        candidates: list[tuple[str, torch.nn.Parameter]] = []
        for name, param in peft_model.named_parameters():
            if not isinstance(param, torch.nn.Parameter):
                continue
            if not param.requires_grad:
                continue
            if param.ndim != 2:
                continue
            if param.shape[0] != num_virtual_tokens:
                continue
            lname = name.lower()
            priority = 0
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

    def _log_step_prefix_tuning(
        self,
        loss: torch.Tensor,
        alignment_loss: torch.Tensor | None,
        convergence_per_sample: torch.Tensor,
        prefix_embedding_param: torch.nn.Parameter | None,
        lr_scheduler,
    ):
        if self.writer is None:
            return
        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        if alignment_loss is not None:
            self.writer.add_scalar("train/alignment_loss", alignment_loss.item(), self.global_step)
        self.writer.add_scalar("train/convergence", convergence_per_sample.mean().item(), self.global_step)
        if prefix_embedding_param is not None:
            with torch.no_grad():
                self.writer.add_scalar(
                    "prefix_tuning/emb_mean",
                    prefix_embedding_param.detach().mean().item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "prefix_tuning/emb_std",
                    prefix_embedding_param.detach().std().item(),
                    self.global_step,
                )
            grad_norm = prefix_embedding_param.grad.norm(2).item() if prefix_embedding_param.grad is not None else 0.0
            self.writer.add_scalar("prefix_tuning/grad_norm", grad_norm, self.global_step)
        if lr_scheduler is not None:
            lr_val = lr_scheduler.get_last_lr()[0]
            self.writer.add_scalar("train/lr", lr_val, self.global_step)
        flush_steps = getattr(self.args, "logging_flush_steps", 100)
        if flush_steps and self.global_step % flush_steps == 0:
            self.writer.flush()
        self.global_step += 1

    def compute_target_hidden(self, model, token_embeddings, attention_mask):
        with torch.no_grad():
            outputs = model(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            target_hidden = outputs.hidden_states
        return target_hidden

    def _prepare_low_dim_proj(self, embedding_dim):
        low_dim_prjoection = nn.Linear(self.args.low_dim_size, embedding_dim)

        if self.args.low_dim_proj_checkpoint is not None:
            if not os.path.exists(self.args.low_dim_proj_checkpoint):
                raise ValueError(f"low_dim_proj_checkpoint does not exist: {self.args.low_dim_proj_checkpoint}")
            checkpoint = torch.load(self.args.low_dim_proj_checkpoint, map_location="cpu")
            if isinstance(checkpoint, dict):
                if "low_dim_projection" in checkpoint:
                    low_dim_prjoection.load_state_dict(checkpoint["low_dim_projection"])
                elif "state_dict" in checkpoint:
                    low_dim_prjoection.load_state_dict(checkpoint["state_dict"])
                else:
                    low_dim_prjoection.load_state_dict(checkpoint)
            else:
                low_dim_prjoection.load_state_dict(checkpoint)
            print(
                f"Loaded low-dimensional projection state from {self.args.low_dim_proj_checkpoint}, "
                f"low dim size = {self.args.low_dim_size}"
            )

        if self.args.low_dim_proj_train:
            low_dim_optim = AdamW(
                low_dim_prjoection.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            scheduler_kwargs = {
                "optimizer": low_dim_optim,
                "num_warmup_steps": self.args.warmup_steps,
                "num_training_steps": self.args.max_optimization_steps_per_sample,
            }

            low_dim_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                **scheduler_kwargs,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
        else:
            for param in low_dim_prjoection.parameters():
                param.requires_grad = False
            low_dim_optim = None
            low_dim_scheduler = None

        return low_dim_prjoection, low_dim_optim, low_dim_scheduler

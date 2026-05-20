"""Progressive cramming trainer: progressively grow the target prefix until reconstruction fails."""

import os
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from compression_horizon.analysis import ProgressiveSampleStateMachine
from compression_horizon.analysis.information_gain import compute_information_gain
from compression_horizon.train.inputs import build_compression_attention_mask, build_united_input
from compression_horizon.train.parametrization import build_per_sample_parametrization
from compression_horizon.train.trainers.base import BaseTrainer


@dataclass
class _RunContext:
    """Run-level constants prepared once before the dataloader loop."""

    model: torch.nn.Module
    device: torch.device
    init_method: str
    mvn_dist: object
    pca_components: torch.Tensor | None
    pca_mean: torch.Tensor | None
    loaded_embeddings: torch.Tensor | None
    num_compression_tokens: int
    threshold: float
    step_increment: int
    min_len: int
    max_stages_cap: int
    # In ``--low_dim_projection_global`` mode the projection's ``nn.Linear``,
    # its AdamW state, and its LR scheduler all live for the entire run (as
    # in the pre-refactor implementation in commit a0d39f6). The same
    # ``nn.Linear`` object is handed to every batch's parametrization via
    # ``projection_module`` so that gradient steps and momentum/variance
    # accumulate continuously. All three fields are ``None`` when the
    # projection is per-batch (the default).
    shared_projection: torch.nn.Linear | None = None
    shared_optimizer: object = None
    shared_scheduler: object = None


@dataclass
class _BatchContext:
    """Per-batch tensors and per-sample optimization state."""

    input_ids: torch.Tensor  # [batch, full_sequence]
    attention_mask: torch.Tensor  # [batch, full_sequence]
    full_token_embeddings: torch.Tensor  # [batch, full_sequence, hidden]
    target_hidden_states_full: tuple[torch.Tensor, ...]
    compression_attention_mask: torch.Tensor  # [batch, compression]
    batch_size: int
    hidden_size: int  # Always the model's hidden size; the parametrization
    # handles low-dim coefficient space internally.
    max_len: int
    parametrization: object
    per_sample_optimizers: list
    per_sample_schedulers: list
    initialization_embeddings: torch.Tensor
    # Optimizer for parametrization-shared parameters (e.g. low-dim Linear
    # projection weights). ``None`` when the parametrization has no shared
    # parameters (Direct / pretrained-PCA mode).
    shared_optimizer: object
    shared_scheduler: object


@dataclass
class _StageContext:
    """Per-stage sliced tensors (one stage = one fixed seq_len target)."""

    seq_len: int
    stage_index: int
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    inputs_embeds: torch.Tensor  # [batch, seq_len, hidden]
    target_hidden_states: list[torch.Tensor]


class ProgressiveCrammingTrainer(BaseTrainer):
    """Trainer for progressive cramming: grow target prefix token-by-token until reconstruction fails."""

    def train(self) -> str | None:
        """Run progressive training. Returns save path or None."""
        ctx = self._build_run_context()
        collected_rows: list[dict] = []
        sample_id_counter = 0
        # The last batch's parametrization is exposed to ``_on_training_complete``
        # so the projection weights can be persisted in non-global mode (in
        # global mode we save ``ctx.shared_projection`` directly).
        final_parametrization = None

        dataloader = self._create_dataloader()
        for batch in tqdm(dataloader):
            batch_ctx = self._setup_batch(batch, ctx)
            collected_rows.extend(self._run_progressive_stages(batch_ctx, ctx, sample_id_counter))
            sample_id_counter += batch_ctx.batch_size
            final_parametrization = batch_ctx.parametrization

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        self._on_training_complete(final_parametrization, ctx)

        return self._save_artifacts(
            collected_rows,
            tensor=None,
            tensor_filename="compression_embeddings.pt",
            subdir_name="progressive_prefixes",
        )

    def _on_training_complete(self, parametrization, ctx: _RunContext) -> None:
        """Hook called once after all batches finish. Persists low-dim projection if any.

        Two cases:
        * ``--low_dim_projection_global``: ``ctx.shared_projection`` is the one
          ``nn.Linear`` that lived through the run — save its state_dict.
        * non-global: ``parametrization`` is the *last* batch's parametrization
          (per-batch projection), so its ``shared_state_dict()`` is the most
          recently trained weights.
        """
        if not self.args.low_dim_projection or not self.args.low_dim_projection_train:
            return
        if ctx.shared_projection is not None:
            shared_state = ctx.shared_projection.state_dict()
        elif parametrization is not None:
            shared_state = parametrization.shared_state_dict()
        else:
            return
        if shared_state is None:
            return
        output_dir = self.args.output_dir
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "low_dim_projection.pt")
        torch.save(
            {
                "low_dim_projection": shared_state,
                "low_dim_size": self.args.low_dim_size,
                "hidden_size": ctx.model.config.hidden_size,
            },
            save_path,
        )
        print(f"Saved low-dimensional projection weights to {save_path}")

    # ------------------------------------------------------------------
    # Run-level setup (once per train()).
    # ------------------------------------------------------------------

    def _build_run_context(self) -> _RunContext:
        """Seed RNG, freeze model, prepare embedding-init helpers.

        In ``--low_dim_projection_global`` mode we additionally construct the
        run-shared ``nn.Linear`` projection, its AdamW optimizer, and its LR
        scheduler — exactly once, before the data loop — so that they live
        across all batches (matching the pre-refactor implementation in
        a0d39f6, where global mode created these three objects above
        ``for batch in dataloader:``).
        """
        model, device, init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings = self._initialize_run()

        shared_projection: torch.nn.Linear | None = None
        shared_optimizer = None
        shared_scheduler = None
        if self.args.low_dim_projection and self.args.low_dim_projection_global:
            shared_projection = self._build_low_dim_projection_module(
                hidden_size=model.config.hidden_size,
                device=device,
            )
            if self.args.low_dim_projection_train:
                shared_optimizer, shared_scheduler = self._build_optimizer_and_scheduler(
                    list(shared_projection.parameters()),
                    num_training_steps=self.args.max_optimization_steps_per_sample,
                )

        return _RunContext(
            model=model,
            device=device,
            init_method=init_method,
            mvn_dist=mvn_dist,
            pca_components=pca_components,
            pca_mean=pca_mean,
            loaded_embeddings=loaded_embeddings,
            num_compression_tokens=self.args.number_of_mem_tokens,
            threshold=self.args.progressive_convergence_threshold,
            step_increment=self.args.progressive_step,
            min_len=self.args.progressive_min_seq_len,
            max_stages_cap=self.args.progressive_max_stages,
            shared_projection=shared_projection,
            shared_optimizer=shared_optimizer,
            shared_scheduler=shared_scheduler,
        )

    # ------------------------------------------------------------------
    # Per-batch setup.
    # ------------------------------------------------------------------

    def _setup_batch(self, batch, ctx: _RunContext) -> _BatchContext:
        """Move batch to device, compute target hidden states, build per-sample parametrization + optimizers."""
        input_ids = batch.input_ids.squeeze(1).to(ctx.device)  # [batch, sequence]
        attention_mask = batch.attention_mask.squeeze(1).to(ctx.device)  # [batch, sequence]
        batch_size = input_ids.shape[0]
        with torch.no_grad():
            full_token_embeddings = ctx.model.get_input_embeddings()(input_ids)  # [batch, sequence, hidden]
        target_hidden_states_full = self.compute_hidden_states(ctx.model, full_token_embeddings, attention_mask)
        hidden_size = full_token_embeddings.shape[-1]  # Always model's hidden size now.

        parametrization = self._build_parametrization(ctx, batch_size, hidden_size)
        per_sample_optimizers, per_sample_schedulers = self._build_per_sample_optimizers(parametrization, ctx)
        # Global mode: reuse the run-level optimizer/scheduler — they own the
        # AdamW state and LR-curve position accumulated across all previous
        # batches. Non-global mode: create a fresh per-batch pair.
        if ctx.shared_projection is not None:
            shared_optimizer, shared_scheduler = ctx.shared_optimizer, ctx.shared_scheduler
        else:
            shared_optimizer, shared_scheduler = self._build_shared_optimizer(parametrization)
        compression_attention_mask = build_compression_attention_mask(
            batch_size,
            ctx.num_compression_tokens,
            device=ctx.device,
            dtype=attention_mask.dtype,
        )

        per_sample_lengths = attention_mask.sum(dim=1).tolist()
        max_len = int(max(per_sample_lengths)) if per_sample_lengths else attention_mask.shape[1]

        return _BatchContext(
            input_ids=input_ids,
            attention_mask=attention_mask,
            full_token_embeddings=full_token_embeddings,
            target_hidden_states_full=target_hidden_states_full,
            compression_attention_mask=compression_attention_mask,
            batch_size=batch_size,
            hidden_size=hidden_size,
            max_len=max_len,
            parametrization=parametrization,
            per_sample_optimizers=per_sample_optimizers,
            per_sample_schedulers=per_sample_schedulers,
            initialization_embeddings=parametrization.initialization_snapshot(),
            shared_optimizer=shared_optimizer,
            shared_scheduler=shared_scheduler,
        )

    def _build_parametrization(self, ctx: _RunContext, batch_size: int, hidden_size: int):
        """Construct the per-sample parametrization (Direct, PretrainedPCA, or low-dim projected).

        Low-dim semantics — the trainer always builds the ``nn.Linear`` itself
        and hands it to the parametrization via ``projection_module``:
        * ``--low_dim_projection_global``: reuse ``ctx.shared_projection`` (one
          Linear for the whole run; AdamW/scheduler in ``ctx`` accumulate
          state across batches).
        * ``--low_dim_projection`` alone: build a **fresh** Linear here per
          batch (warm-started from ``--low_dim_projection_checkpoint`` if
          given, frozen when ``--low_dim_projection_train`` is False). This
          matches the pre-refactor behaviour (commit a0d39f6) where the
          projection constructor was invoked inside the data loop.
        """
        # Coefficient init lives in low-dim space when low-dim is on; otherwise
        # in model hidden space.
        init_dim = self.args.low_dim_size if self.args.low_dim_projection else hidden_size

        def _init_helper():
            return self._init_compression_tokens(
                batch_size,
                ctx.num_compression_tokens,
                init_dim,
                ctx.init_method,
                ctx.mvn_dist,
                pca_components=ctx.pca_components,
                pca_mean=ctx.pca_mean,
                loaded_embeddings=ctx.loaded_embeddings,
            )

        projection_module = None
        if self.args.low_dim_projection:
            projection_module = ctx.shared_projection or self._build_low_dim_projection_module(
                hidden_size=hidden_size,
                device=ctx.device,
            )

        return build_per_sample_parametrization(
            init_method=ctx.init_method,
            batch_size=batch_size,
            num_compression_tokens=ctx.num_compression_tokens,
            hidden_size=hidden_size,
            device=ctx.device,
            init_helper=_init_helper,
            pca_components=ctx.pca_components,
            pca_mean=ctx.pca_mean,
            low_dim_train=self.args.low_dim_projection,
            low_dim_size=self.args.low_dim_size,
            train_projection=self.args.low_dim_projection_train,
            projection_module=projection_module,
        )

    def _build_shared_optimizer(self, parametrization):
        """One optimizer for ``parametrization.shared_parameters`` (e.g. low-dim projection)."""
        if not parametrization.shared_parameters:
            return None, None
        return self._build_optimizer_and_scheduler(
            list(parametrization.shared_parameters),
            num_training_steps=self.args.max_optimization_steps_per_sample,
        )

    def _build_per_sample_optimizers(self, parametrization, ctx: _RunContext):
        """One optimizer/scheduler per sample. PCA path uses constant LR; direct path uses cosine over the full per-sample step budget."""
        per_sample_optimizers, per_sample_schedulers = [], []
        for parameter in parametrization.parameters:
            if ctx.init_method == "pretrained_pca":
                optimizer, scheduler = self._build_optimizer_and_scheduler([parameter])
            else:
                optimizer, scheduler = self._build_optimizer_and_scheduler(
                    [parameter],
                    num_training_steps=self.args.max_optimization_steps_per_sample,
                )
            per_sample_optimizers.append(optimizer)
            per_sample_schedulers.append(scheduler)
        return per_sample_optimizers, per_sample_schedulers

    # ------------------------------------------------------------------
    # Stage progression.
    # ------------------------------------------------------------------

    def _run_progressive_stages(self, batch_ctx: _BatchContext, ctx: _RunContext, sample_id_counter: int) -> list[dict]:
        """Outer stage-while loop: grow seq_len, run a stage, save rows, repeat until cap / all-skipped / max-len."""
        state = ProgressiveSampleStateMachine(batch_ctx.batch_size, ctx.threshold)
        seq_len = min(ctx.min_len, batch_ctx.max_len)
        stage_index = 0
        rows: list[dict] = []

        while True:
            stage_ctx = self._setup_stage(batch_ctx, seq_len, stage_index)
            last_loss, last_convergence = self._run_stage_loop(batch_ctx, stage_ctx, ctx, state)
            state.mark_skipped_if_not_converged(seq_len)
            rows.extend(
                self._collect_stage_rows(batch_ctx, stage_ctx, ctx, state, last_loss, last_convergence, sample_id_counter)
            )

            stage_index += 1
            if seq_len >= batch_ctx.max_len:
                break
            if ctx.max_stages_cap and stage_index >= ctx.max_stages_cap:
                break
            if state.all_skipped:
                print("All samples skipped. Stopping at seq_len =", seq_len)
                break
            seq_len = min(seq_len + ctx.step_increment, batch_ctx.max_len)

        return rows

    def _setup_stage(self, batch_ctx: _BatchContext, seq_len: int, stage_index: int) -> _StageContext:
        """Slice batch tensors to the current stage's seq_len."""
        return _StageContext(
            seq_len=seq_len,
            stage_index=stage_index,
            input_ids=batch_ctx.input_ids[:, :seq_len],
            attention_mask=batch_ctx.attention_mask[:, :seq_len],
            inputs_embeds=batch_ctx.full_token_embeddings[:, :seq_len, :],
            target_hidden_states=[h[:, :seq_len] for h in batch_ctx.target_hidden_states_full],
        )

    def _run_stage_loop(
        self, batch_ctx: _BatchContext, stage_ctx: _StageContext, ctx: _RunContext, state: ProgressiveSampleStateMachine
    ):
        """One stage with optional scheduler-reset retry on non-convergence. Returns (last_loss, last_convergence)."""
        state.reset_stage()
        scheduler_reset_used = False
        last_loss, last_convergence = None, None

        while True:
            last_loss, last_convergence, converged = self._run_steps(
                batch_ctx, stage_ctx, ctx, state, retry=scheduler_reset_used
            )
            if converged:
                return last_loss, last_convergence
            if not self.args.progressive_reset_lr_scheduler_on_non_convergence or scheduler_reset_used:
                return last_loss, last_convergence
            print(f"Not converged at seq_len={stage_ctx.seq_len}, resetting LR schedulers for non-converged samples...")
            self._reset_per_sample_optimizers(batch_ctx, ctx, state)
            scheduler_reset_used = True

    def _reset_per_sample_optimizers(
        self, batch_ctx: _BatchContext, ctx: _RunContext, state: ProgressiveSampleStateMachine
    ) -> None:
        """Rebuild optimizers/schedulers for samples still active in the current stage."""
        for j in range(batch_ctx.batch_size):
            if not state.is_active(j):
                continue
            parameter = batch_ctx.parametrization.parameters[j]
            if ctx.init_method == "pretrained_pca":
                optimizer, scheduler = self._build_optimizer_and_scheduler([parameter])
            else:
                optimizer, scheduler = self._build_optimizer_and_scheduler(
                    [parameter],
                    num_training_steps=self.args.max_optimization_steps_per_token,
                )
            batch_ctx.per_sample_optimizers[j] = optimizer
            batch_ctx.per_sample_schedulers[j] = scheduler

    def _run_steps(
        self,
        batch_ctx: _BatchContext,
        stage_ctx: _StageContext,
        ctx: _RunContext,
        state: ProgressiveSampleStateMachine,
        retry: bool,
    ):
        """Inner step loop within a stage. Returns (last_loss, last_convergence, converged_bool)."""
        progress_bar = tqdm(
            range(self.args.max_optimization_steps_per_token),
            total=self.args.max_optimization_steps_per_token,
            leave=False,
        )
        progress_bar.set_description(f"Stage L={stage_ctx.seq_len}" + (" (retry)" if retry else ""))

        last_loss, last_convergence = None, None

        for _ in progress_bar:
            # `materialize()` returns the hidden-size embedding already projected
            # through the parametrization (low-dim Linear if applicable).
            compression_token_embeddings = batch_ctx.parametrization.materialize().clone()

            united_token_embeddings, united_attention_mask = build_united_input(
                compression_token_embeddings,
                batch_ctx.compression_attention_mask,
                stage_ctx.inputs_embeds,
                stage_ctx.attention_mask,
            )
            (
                loss,
                alignment_loss,
                convergence_per_sample,
                generated_text,
                ground_truth_text,
            ) = self.forward_and_compute_loss(
                ctx.model,
                stage_ctx.input_ids,
                stage_ctx.inputs_embeds,
                stage_ctx.attention_mask,
                united_token_embeddings,
                united_attention_mask,
                ctx.num_compression_tokens,
                target_hidden_states=stage_ctx.target_hidden_states,
            )
            loss.backward()

            self._log_progress(
                progress_bar=progress_bar,
                loss=loss,
                alignment_loss=alignment_loss,
                convergence_per_sample=convergence_per_sample,
                batch_ctx=batch_ctx,
                state=state,
                compression_token_embeddings=compression_token_embeddings,
                generated_text=generated_text,
                ground_truth_text=ground_truth_text,
            )

            self._step_per_sample_optimizers(batch_ctx, state)
            self._step_shared_optimizer(batch_ctx)

            last_loss = float(loss.item())
            last_convergence = convergence_per_sample.detach().cpu()

            if state.update(convergence_per_sample):
                return last_loss, last_convergence, True

        return last_loss, last_convergence, False

    def _step_per_sample_optimizers(self, batch_ctx: _BatchContext, state: ProgressiveSampleStateMachine) -> None:
        """Step active samples' optimizers; zero grads for all (active + skipped + already-converged-in-stage)."""
        for j in range(batch_ctx.batch_size):
            if state.is_active(j):
                batch_ctx.per_sample_optimizers[j].step()
                if batch_ctx.per_sample_schedulers[j] is not None:
                    batch_ctx.per_sample_schedulers[j].step()
                state.increment_steps(j)
            batch_ctx.per_sample_optimizers[j].zero_grad(set_to_none=True)

    def _step_shared_optimizer(self, batch_ctx: _BatchContext) -> None:
        """Step the optimizer for parametrization-shared parameters (e.g. low-dim Linear)."""
        if batch_ctx.shared_optimizer is None:
            return
        batch_ctx.shared_optimizer.step()
        batch_ctx.shared_optimizer.zero_grad(set_to_none=True)
        if batch_ctx.shared_scheduler is not None:
            batch_ctx.shared_scheduler.step()

    def _log_progress(
        self,
        *,
        progress_bar,
        loss: torch.Tensor,
        alignment_loss: torch.Tensor | None,
        convergence_per_sample: torch.Tensor,
        batch_ctx: _BatchContext,
        state: ProgressiveSampleStateMachine,
        compression_token_embeddings: torch.Tensor,
        generated_text,
        ground_truth_text,
    ) -> None:
        """Update tqdm postfix and forward scalars to TensorBoard."""
        grad_norms = [
            param.grad.norm(2).item() if param.grad is not None else 0.0 for param in batch_ctx.parametrization.parameters
        ]
        grad_norm = sum(grad_norms) / len(grad_norms)

        active_scheduler = None
        for j in range(batch_ctx.batch_size):
            if state.is_active(j):
                active_scheduler = batch_ctx.per_sample_schedulers[j]
                break
        log_lr = active_scheduler.get_last_lr()[0] if active_scheduler is not None else self.args.learning_rate

        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=loss.item(),
            convergece_per_sample=convergence_per_sample.mean().item(),
            compr_t_mean=compression_token_embeddings.mean().item(),
            compr_t_std=compression_token_embeddings.std().item(),
            grad=grad_norm,
            lr=log_lr,
        )

        self._log_step(
            loss,
            alignment_loss,
            convergence_per_sample,
            compression_token_embeddings,
            active_scheduler,
            generated_text,
            ground_truth_text,
            leaf_grad_params=list(batch_ctx.parametrization.parameters) + list(batch_ctx.parametrization.shared_parameters),
        )

    # ------------------------------------------------------------------
    # Stage row collection.
    # ------------------------------------------------------------------

    def _collect_stage_rows(
        self,
        batch_ctx: _BatchContext,
        stage_ctx: _StageContext,
        ctx: _RunContext,
        state: ProgressiveSampleStateMachine,
        last_loss,
        last_convergence,
        sample_id_counter: int,
    ) -> list[dict]:
        """Compute Information Gain for the stage and assemble per-sample row dicts."""
        with torch.no_grad():
            # `materialize()` already lifts coefficients to hidden_size when a
            # projection is in use, so there is no separate "after projection"
            # tensor to keep track of in this trainer anymore.
            compression_token_embeddings = batch_ctx.parametrization.materialize()  # [batch, compression, hidden]
            comp_tokens_cpu = compression_token_embeddings.detach().cpu()
            # Per-sample parametrization extras: PCA / low-dim coefficients
            # (None for vanilla Direct parametrization).
            pca_coefficients_to_save = batch_ctx.parametrization.serialize_extras()

            per_sample_info_gain = compute_information_gain(
                model=ctx.model,
                input_ids=stage_ctx.input_ids,
                attention_mask=stage_ctx.attention_mask,
                token_embeddings=stage_ctx.inputs_embeds,
                compression_token_embeddings=compression_token_embeddings,
                compression_attention_mask=batch_ctx.compression_attention_mask,
            )

            embeddings_dir = self._prepare_embeddings_dir()
            tokenizer = self.processing_class
            rows = []
            for j in range(batch_ctx.batch_size):
                rows.append(
                    self._build_sample_row(
                        sample_index=j,
                        sample_id=sample_id_counter + j,
                        batch_ctx=batch_ctx,
                        stage_ctx=stage_ctx,
                        ctx=ctx,
                        state=state,
                        tokenizer=tokenizer,
                        comp_tokens_cpu=comp_tokens_cpu,
                        comp_tokens_gpu=compression_token_embeddings,
                        pca_coefficients_to_save=pca_coefficients_to_save,
                        last_loss=last_loss,
                        last_convergence=last_convergence,
                        per_sample_info_gain=per_sample_info_gain,
                        embeddings_dir=embeddings_dir,
                    )
                )
            return rows

    def _prepare_embeddings_dir(self) -> str | None:
        """Create artifacts/.../embeddings/ subdir used for periodic stage snapshots."""
        if not self.args.output_dir:
            return None
        embeddings_dir = os.path.join(self.args.output_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        return embeddings_dir

    def _build_sample_row(
        self,
        *,
        sample_index: int,
        sample_id: int,
        batch_ctx: _BatchContext,
        stage_ctx: _StageContext,
        ctx: _RunContext,
        state: ProgressiveSampleStateMachine,
        tokenizer,
        comp_tokens_cpu: torch.Tensor,
        comp_tokens_gpu: torch.Tensor,
        pca_coefficients_to_save,
        last_loss,
        last_convergence,
        per_sample_info_gain: list[float],
        embeddings_dir: str | None,
    ) -> dict:
        """One sample's row dict for this stage. Schema preserved for downstream eval scripts.

        Note: ``orig_embedding`` used to refer to the pre-projection low-dim
        coefficient tensor in the legacy implementation. Now the parametrization
        encapsulates the projection inside ``materialize()`` and ``serialize_extras()``
        already returns the low-dim coefficients via ``pca_coefficients_to_save``.
        We keep ``orig_embedding`` in the schema (set equal to ``embedding``) so
        downstream eval scripts that ``remove_columns([..., "orig_embedding"])``
        continue to work without modification.
        """
        sample_attention_mask = stage_ctx.attention_mask[sample_index].bool()
        sample_input_ids = stage_ctx.input_ids[sample_index][sample_attention_mask]
        sample_text = tokenizer.decode(sample_input_ids.tolist(), skip_special_tokens=True) if tokenizer is not None else ""

        if embeddings_dir is not None and stage_ctx.stage_index % 500 == 0:
            self._dump_stage_embedding(
                embeddings_dir=embeddings_dir,
                sample_id=sample_id,
                stage_index=stage_ctx.stage_index,
                comp_tokens=comp_tokens_gpu[sample_index],
                initialization_embedding=batch_ctx.initialization_embeddings[sample_index],
                shared_state=batch_ctx.parametrization.shared_state_dict(),
            )

        sample_pca_coefficients = pca_coefficients_to_save[sample_index] if pca_coefficients_to_save is not None else None
        embedding_as_list = comp_tokens_cpu[sample_index].to(torch.float32).numpy().tolist()
        return {
            "sample_id": int(sample_id),
            "stage_index": int(stage_ctx.stage_index),
            "stage_seq_len": int(stage_ctx.seq_len),
            "text": sample_text,
            "embedding": embedding_as_list,
            # Back-compat with legacy schema: kept identical to ``embedding`` —
            # the low-dim coefficients (the only meaningful "pre-projection"
            # tensor) now live under ``pca_coefficients_to_save``.
            "orig_embedding": embedding_as_list,
            "pca_coefficients_to_save": sample_pca_coefficients,
            "initialization_embedding": batch_ctx.initialization_embeddings[sample_index].to(torch.float32).numpy().tolist(),
            "final_loss": float(last_loss) if last_loss is not None else None,
            "final_convergence": float(last_convergence[sample_index].item()) if last_convergence is not None else None,
            "num_input_tokens": int(sample_attention_mask.sum().item()),
            "num_compression_tokens": int(ctx.num_compression_tokens),
            "hidden_size": int(comp_tokens_cpu.shape[-1]),
            "loss_type": self.args.loss_type,
            "dtype": self.args.dtype,
            "model_checkpoint": self.args.model_checkpoint,
            "max_optimization_steps_per_sample": int(self.args.max_optimization_steps_per_sample),
            "convergence_threshold": float(ctx.threshold),
            "steps_taken": int(state.steps_taken[sample_index]),
            "information_gain_bits": float(per_sample_info_gain[sample_index]),
        }

    def _dump_stage_embedding(
        self,
        *,
        embeddings_dir: str,
        sample_id: int,
        stage_index: int,
        comp_tokens: torch.Tensor,
        initialization_embedding: torch.Tensor,
        shared_state: dict | None,
    ) -> None:
        """Persist bf16 snapshots of compression / init embeddings (+ shared low-dim projection state) for the current stage."""
        embedding_path = os.path.join(embeddings_dir, f"embedding_sample_{sample_id}_stage_{stage_index}.pt")
        init_embedding_path = os.path.join(
            embeddings_dir, f"initialization_embedding_sample_{sample_id}_stage_{stage_index}.pt"
        )

        torch.save(comp_tokens.to(torch.bfloat16).detach().cpu(), embedding_path)
        torch.save(initialization_embedding.to(torch.bfloat16), init_embedding_path)
        if shared_state is not None:
            low_dim_proj_path = os.path.join(embeddings_dir, f"low_dim_proj_sample_{sample_id}_stage_{stage_index}.pt")
            torch.save(shared_state, low_dim_proj_path)

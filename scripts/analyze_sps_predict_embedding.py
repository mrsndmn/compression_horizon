"""Compare SPS per-text predict-embedding cramming with the memory-site controls.

One learned H-vector is shared across all prediction slots; the ordinary slot
zero is a fixed existing BOS. Output-head weights and ordinary embeddings remain
frozen. The specialized artifacts are deliberately called predict_prefixes.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scripts import analyze_sps_cramming as jobs
from scripts.activation_distillation import load_or_create_tokenized_dataset
from scripts.greedy_reconstruction_eval import select_rows
from scripts.jobs.run_jobs_sps import TRAINING_ARGS
from scripts.sps_mechanism_common import source_commit
from scripts.validation.validate_sps_batch1_run import check_sample
from transformers import DataCollatorForLanguageModeling

from compression_horizon.models.sps_causal_lm import load_sps_checkpoint, load_sps_tokenizer
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainers.progressive_cramming import ProgressiveCrammingTrainer

ROOT = jobs.ROOT / "predict_embedding_20261022"
CHECKPOINT = next(e["checkpoint"] for e in jobs.experiment_configs if e["name"] == "sps_w0")
experiment_configs = [
    dict(
        name="predict_ce_native",
        init="native",
        training_args=dict(learning_rate=0.3),
        baseline=jobs.ROOT / "sps_w0_pilot_1b_batch1_lr0.3_f59ca78c7aa7",
    ),
    dict(
        name="predict_hinge_native",
        init="native",
        training_args=dict(learning_rate=0.3, token_objective="squared_margin", loss_margin=0.05, convergence_margin=0.05),
        baseline=jobs.ROOT / "sps_w0_pilot_1b_batch1_hinge_strict_893c490bf406",
    ),
    dict(
        name="predict_hinge_random",
        init="random",
        training_args=dict(learning_rate=0.3, token_objective="squared_margin", loss_margin=0.05, convergence_margin=0.05),
        baseline=jobs.ROOT / "sps_w0_pilot_1b_batch1_hinge_strict_893c490bf406",
    ),
]
# Fresh runs: saved stage vectors do not include Adam/scheduler state, and a
# larger total budget also stretches the existing cosine scheduler horizon.
# Keep the per-stage limit and all other settings unchanged. The completed
# same-initialization predict runs, rather than ordinary-memory runs, are controls.
budget_experiment_configs = [
    dict(
        exp,
        name=exp["name"] + "_budget30000",
        training_args=dict(exp["training_args"], max_optimization_steps_per_sample=30000),
        baseline=ROOT / exp["name"],
    )
    for exp in experiment_configs
    if exp["training_args"].get("token_objective") == "squared_margin"
]
EXPERIMENT_SUITES = {"baseline": experiment_configs, "budget": budget_experiment_configs}


def fixed_bos(model, batch_size, bos_token_id):
    ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=model.device)
    return model.get_input_embeddings()(ids).detach()


class PredictEmbeddingTrainer(ProgressiveCrammingTrainer):
    def __init__(self, *args, bos_token_id, predict_init, **kwargs):
        super().__init__(*args, **kwargs)
        a = self.args
        if (
            not self.model.config.sps
            or self.model.config.window_size != 0
            or a.per_device_train_batch_size != 1
            or a.number_of_mem_tokens != 1
            or a.low_dim_projection
            or a.progressive_prefix_len
            or a.loss_type != "cross_entropy"
            or a.hybrid_alpha is not None
            or predict_init not in ("native", "random")
        ):
            raise ValueError("Predict experiment requires native SPS w=0, batch 1, one direct vector and token-only loss")
        if not 0 <= bos_token_id < self.model.config.vocab_size or bos_token_id == self.model.config.predict_token_id:
            raise ValueError("BOS must be an existing ordinary vocabulary ID")
        self.bos_token_id = bos_token_id
        self.predict_init = predict_init

    def _setup_batch(self, batch, ctx, *, sample_id_counter=0):
        result = super()._setup_batch(batch, ctx, sample_id_counter=sample_id_counter)
        if self.predict_init == "native":
            with torch.no_grad():
                initial = ctx.model.get_input_embeddings().weight[ctx.model.config.predict_token_id].view(1, 1, -1)
                result.parametrization.parameters[0].copy_(initial)
            result.initialization_embeddings = result.parametrization.materialize().detach().clone().cpu()
        return result

    def forward_and_compute_loss(
        self,
        model,
        input_ids,
        token_embeddings,
        attention_mask,
        united_token_embeddings,
        united_attention_mask,
        num_compression_tokens,
        target_hidden_states=None,
        prefix_len=0,
    ):
        if num_compression_tokens != 1 or prefix_len:
            raise ValueError("Predict training requires exactly one latent vector and no uncompressed prefix")
        predict = united_token_embeddings[:, :1]
        context = torch.cat((fixed_bos(model, input_ids.shape[0], self.bos_token_id), united_token_embeddings[:, 1:]), 1)
        logits = model(inputs_embeds=context, attention_mask=united_attention_mask, predict_embeddings=predict).logits
        loss, alignment = self.compute_loss(logits, input_ids, attention_mask, 1)
        convergence = self.compute_convergence(logits, input_ids, attention_mask, 1)
        return loss, alignment, convergence, None, None

    @torch.no_grad()
    def _collect_stage_rows(self, batch_ctx, stage_ctx, ctx, state, last_loss, last_convergence, sample_id_counter):
        predict = batch_ctx.parametrization.materialize()
        context = torch.cat((fixed_bos(ctx.decode_model, 1, self.bos_token_id), stage_ctx.inputs_embeds), 1)
        mask = torch.cat((batch_ctx.compression_attention_mask, stage_ctx.attention_mask), 1)
        logits = ctx.decode_model(inputs_embeds=context, attention_mask=mask, predict_embeddings=predict).logits[:, :-1].float()
        baseline = ctx.decode_model(inputs_embeds=context, attention_mask=mask).logits[:, :-1].float()
        ids = stage_ctx.input_ids[:, :, None]
        gain = (
            (logits.log_softmax(-1).gather(-1, ids) - baseline.log_softmax(-1).gather(-1, ids)).squeeze(-1)
            * stage_ctx.attention_mask
        ).sum(-1) / math.log(2)
        row = self._build_sample_row(
            sample_index=0,
            sample_id=sample_id_counter + (self.args.offset_dataset_items or 0),
            batch_ctx=batch_ctx,
            stage_ctx=stage_ctx,
            ctx=ctx,
            state=state,
            tokenizer=self.processing_class,
            comp_tokens_cpu=predict.detach().cpu(),
            comp_tokens_gpu=predict,
            pca_coefficients_to_save=None,
            last_loss=last_loss,
            last_convergence=last_convergence,
            per_sample_info_gain=gain.tolist(),
            embeddings_dir=self._prepare_embeddings_dir(),
        )
        row.update(
            embedding_role="shared_predict",
            fixed_bos_token_id=self.bos_token_id,
            predict_init=self.predict_init,
            information_gain_reference="fixed_bos_native_predict",
        )
        return [row]

    def _save_artifacts(self, rows, *, tensor, tensor_filename, subdir_name):
        return super()._save_artifacts(
            rows, tensor=tensor, tensor_filename="predict_embeddings.pt", subdir_name="predict_prefixes"
        )


def resolved_training_args(exp):
    settings = dict(TRAINING_ARGS, **exp["training_args"])
    # Importing the launcher inside a staged snapshot changes its REPO-derived
    # cache path; keep the established absolute shared cache instead.
    settings["tokenized_dataset_path"] = str(jobs.CACHE_ROOT / Path(TRAINING_ARGS["tokenized_dataset_path"]).name)
    return settings


def probe(exp, *, output_root=None):
    torch.set_num_threads(8)
    output = (ROOT if output_root is None else Path(output_root)) / exp["name"]
    settings = resolved_training_args(exp)
    args = MyTrainingArguments(
        **settings, model_checkpoint=jobs.cluster_path(CHECKPOINT), output_dir=str(output), logging_dir=str(output / "logs")
    )
    model = load_sps_checkpoint(args.model_checkpoint, torch_dtype=torch.bfloat16)
    tokenizer = load_sps_tokenizer()
    bos = tokenizer.bos_token_id
    if bos is None or bos != model.config.eos_token_id:
        raise ValueError("Expected verified GPT-2 BOS/EOS identity for these checkpoints")
    dataset = load_or_create_tokenized_dataset(
        cache_dir="artifacts/cache/tokenized_datasets",
        dataset_name=args.dataset_name,
        split="test",
        tokenizer=tokenizer,
        max_sequence_length=args.max_sequence_length,
        model_checkpoint=args.model_checkpoint,
        no_bos_token=args.no_bos_token,
        limit_dataset_items=args.limit_dataset_items,
        offset_dataset_items=args.offset_dataset_items,
        tokenized_dataset_path=jobs.cluster_path(args.tokenized_dataset_path),
    )
    trainer = PredictEmbeddingTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        bos_token_id=bos,
        predict_init=exp["init"],
    )
    path = trainer.train()
    rows = select_rows(load_from_disk(path), "progr")
    offset = args.offset_dataset_items or 0
    expected_ids = set(range(offset, offset + args.limit_dataset_items))
    if len(rows) != args.limit_dataset_items or {r["sample_id"] for r in rows} != expected_ids:
        raise ValueError("Incomplete predict sample set")
    samples = []
    for row in rows:
        if row["embedding_role"] != "shared_predict" or row["fixed_bos_token_id"] != bos:
            raise ValueError("Predict artifact role mismatch")
        sid = row["sample_id"]
        length = row["num_input_tokens"]
        ids = dataset[sid - offset]["input_ids"][:length][None].to(model.device)
        predict = torch.tensor([row["embedding"]], device=model.device, dtype=model.dtype)
        result = check_sample(model, tokenizer, ids, fixed_bos(model, 1, bos), predict_embeddings=predict)
        result.update(
            sample_id=sid,
            horizon=length if row["final_convergence"] >= row["convergence_threshold"] else 0,
            stored_convergence=row["final_convergence"],
        )
        samples.append(result)
    report = dict(
        source_commit=source_commit(),
        experiment_name=exp["name"],
        training_args=settings,
        embedding_role="shared_predict",
        fixed_bos_token_id=bos,
        predict_token_id=model.config.predict_token_id,
        trainable_scalars=model.config.hidden_size,
        n_samples=len(samples),
        mean_horizon=float(np.mean([s["horizon"] for s in samples])),
        greedy_exact_samples=sum(s["greedy_exact"] for s in samples),
        teacher_replay_exact_samples=sum(s["teacher_replay_correct_tokens"] == len(s["target_ids"]) for s in samples),
        samples=samples,
    )
    jobs.write_json(output / "single_sample_validation.json", report)
    (output / "completed").touch()
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


def submit(dry=False, *, suite="baseline"):
    for exp in EXPERIMENT_SUITES[suite]:
        jobs.OUTPUT = ROOT / exp["name"]
        if not dry and (jobs.OUTPUT / "job.json").exists():
            print("Existing submission:", jobs.OUTPUT)
            continue
        jobs.submit(
            dry=dry,
            entrypoint="scripts/analyze_sps_predict_embedding.py",
            job_kind="sps_" + exp["name"],
            probe_args=("--experiment", exp["name"], "--suite", suite),
        )


def summarize(*, suite="baseline"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for exp in EXPERIMENT_SUITES[suite]:
        output = ROOT / exp["name"]
        if not (output / "completed").exists():
            raise ValueError(f"Incomplete experiment: {output}")
        learned = json.loads((output / "single_sample_validation.json").read_text())
        baseline = json.loads((exp["baseline"] / "single_sample_validation.json").read_text())
        a = sorted(learned["samples"], key=lambda r: r["sample_id"])
        b = sorted(baseline["samples"], key=lambda r: r["sample_id"])
        if len(a) != len(b) or [r["sample_id"] for r in a] != [r["sample_id"] for r in b]:
            raise ValueError("Unaligned cohorts")
        for x, y in zip(a, b):
            n = min(len(x["target_ids"]), len(y["target_ids"]))
            if x["target_ids"][:n] != y["target_ids"][:n]:
                raise ValueError("Unaligned exact targets")
        from scripts.analyze_sps_objective_factorial import leading_correct, paired_interval

        difference = paired_interval([x["horizon"] - y["horizon"] for x, y in zip(a, b)])
        rows.append(
            dict(
                name=exp["name"],
                mean_horizon=learned["mean_horizon"],
                baseline_horizon=baseline["mean_horizon"],
                paired_difference=difference,
                greedy_exact_samples=learned["greedy_exact_samples"],
                mean_greedy_initial_prefix=float(np.mean([leading_correct(s["target_ids"], s["generated_ids"]) for s in a])),
            )
        )
    prefix = "" if suite == "baseline" else suite + "_"
    jobs.write_json(ROOT / (prefix + "summary.json"), dict(analysis_commit=source_commit(), suite=suite, rows=rows))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(rows))
    baseline_label = "Learned memory / fixed predict" if suite == "baseline" else "Original budget / shared predict"
    learned_label = "Fixed BOS / learned shared predict" if suite == "baseline" else "Larger budget / shared predict"
    ax.bar(x - 0.18, [r["baseline_horizon"] for r in rows], 0.36, label=baseline_label)
    ax.bar(x + 0.18, [r["mean_horizon"] for r in rows], 0.36, label=learned_label)
    ax.set_xticks(x, [r["name"] for r in rows], rotation=12)
    ax.set(
        ylabel="Mean teacher-forced reconstruction tokens",
        title="Same SPS checkpoint, texts and one-vector parameter budget; batch 1",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / (prefix + "predict_embedding_comparison.png"), dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["probe", "dry", "submit", "summary"], default="dry")
    parser.add_argument("--suite", choices=EXPERIMENT_SUITES, default="baseline")
    parser.add_argument("--experiment", choices=[e["name"] for suite in EXPERIMENT_SUITES.values() for e in suite])
    cli = parser.parse_args()
    if cli.phase == "probe":
        if cli.experiment is None:
            parser.error("--experiment is required for probe")
        selected = next((e for e in EXPERIMENT_SUITES[cli.suite] if e["name"] == cli.experiment), None)
        if selected is None:
            parser.error("--experiment must belong to --suite")
        probe(selected)
    else:
        {
            "dry": lambda: submit(True, suite=cli.suite),
            "submit": lambda: submit(suite=cli.suite),
            "summary": lambda: summarize(suite=cli.suite),
        }[cli.phase]()

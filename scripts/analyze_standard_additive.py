"""Standard cramming with [z, E(x1)+z, ...]; one shared per-text vector.

Design authority: experiment_configs and resolved_training_args below. Reuses
the completed ordinary-memory control and the established immutable MLS launcher.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_from_disk
from scripts import analyze_sps_cramming as jobs
from scripts.activation_distillation import load_or_create_tokenized_dataset
from scripts.analyze_sps_objective_factorial import leading_correct, paired_interval
from scripts.analyze_sps_predict_shards import checked_samples
from scripts.greedy_reconstruction_eval import select_rows
from scripts.jobs import run_jobs_sps as ordinary
from scripts.sps_mechanism_common import source_commit
from transformers import DataCollatorForLanguageModeling

from compression_horizon.inference.generation import mask_native_generation_logits
from compression_horizon.models.sps_causal_lm import (
    load_sps_checkpoint,
    load_sps_tokenizer,
)
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainers.progressive_cramming import (
    ProgressiveCrammingTrainer,
)
from compression_horizon.utils.launch import set_launch_seed

ROOT = jobs.ROOT / "standard_additive_20261025"
BASELINE = jobs.ROOT / "standard_pilot_1b_batch1_hinge_strict_feb2c4c51ffa"
CHECKPOINT = next(e["checkpoint"] for e in jobs.experiment_configs if e["name"] == "standard")
CONTROL = next(e for e in ordinary.objective_experiment_configs if e["name"] == "standard_pilot_1b_batch1_hinge_strict")
COHORT_SIZE = ordinary.TRAINING_ARGS["limit_dataset_items"]
SHARD_SIZE = 6
ROLE = "prefix_and_additive"
experiment_configs = [
    dict(
        name=f"standard_additive_hinge_shard{index:02d}",
        offset=offset,
        count=min(SHARD_SIZE, COHORT_SIZE - offset),
    )
    for index, offset in enumerate(range(0, COHORT_SIZE, SHARD_SIZE))
]


def resolved_training_args(exp):
    settings = dict(ordinary.TRAINING_ARGS, **CONTROL["training_args"])
    settings.update(offset_dataset_items=exp["offset"], limit_dataset_items=exp["count"])
    settings["tokenized_dataset_path"] = str(jobs.CACHE_ROOT / Path(settings["tokenized_dataset_path"]).name)
    return settings


def condition_inputs(memory, token_embeddings):
    """Keep position zero once; broadcast the same differentiable z to later inputs."""
    if (
        memory.ndim != 3
        or memory.shape[:2] != (1, 1)
        or token_embeddings.ndim != 3
        or token_embeddings.shape[0] != 1
        or memory.shape[-1] != token_embeddings.shape[-1]
    ):
        raise ValueError("Additive cramming requires batch one and one shared vector")
    memory = memory.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    return torch.cat((memory, token_embeddings + memory), dim=1)


class AdditiveCrammingTrainer(ProgressiveCrammingTrainer):
    def __init__(self, *args, initializations, **kwargs):
        super().__init__(*args, **kwargs)
        a = self.args
        if (
            self.model.config.sps
            or a.per_device_train_batch_size != 1
            or a.number_of_mem_tokens != 1
            or a.low_dim_projection
            or a.progressive_prefix_len
            or a.fix_position_ids
            or a.loss_type != "cross_entropy"
            or a.hybrid_alpha is not None
            or self.model_reconstructor is not None
        ):
            raise ValueError("Additive cramming requires native Standard, one direct vector, batch one and token loss")
        self.initializations = initializations

    def _setup_batch(self, batch, ctx, *, sample_id_counter=0):
        result = super()._setup_batch(batch, ctx, sample_id_counter=sample_id_counter)
        sid = sample_id_counter + (self.args.offset_dataset_items or 0)
        initial = torch.as_tensor(
            self.initializations[sid],
            device=ctx.device,
            dtype=result.parametrization.parameters[0].dtype,
        ).unsqueeze(0)
        if initial.shape != result.parametrization.parameters[0].shape or not torch.isfinite(initial).all():
            raise ValueError("Invalid paired control initialization")
        with torch.no_grad():
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
            raise ValueError("Expected one vector and no uncompressed prefix")
        inputs = condition_inputs(united_token_embeddings[:, :1], token_embeddings)
        logits = model(inputs_embeds=inputs, attention_mask=united_attention_mask).logits
        loss, alignment = self.compute_loss(logits, input_ids, attention_mask, 1)
        convergence = self.compute_convergence(logits, input_ids, attention_mask, 1)
        return loss, alignment, convergence, None, None

    @torch.no_grad()
    def _collect_stage_rows(
        self,
        batch_ctx,
        stage_ctx,
        ctx,
        state,
        last_loss,
        last_convergence,
        sample_id_counter,
    ):
        z = batch_ctx.parametrization.materialize()
        mask = torch.cat((batch_ctx.compression_attention_mask, stage_ctx.attention_mask), 1)
        logits = (
            ctx.decode_model(
                inputs_embeds=condition_inputs(z, stage_ctx.inputs_embeds),
                attention_mask=mask,
            )
            .logits[:, :-1]
            .float()
        )
        zero = (
            ctx.decode_model(
                inputs_embeds=condition_inputs(torch.zeros_like(z), stage_ctx.inputs_embeds),
                attention_mask=mask,
            )
            .logits[:, :-1]
            .float()
        )
        ids = stage_ctx.input_ids[:, :, None]
        gain = (
            (logits.log_softmax(-1).gather(-1, ids) - zero.log_softmax(-1).gather(-1, ids)).squeeze(-1)
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
            comp_tokens_cpu=z.detach().cpu(),
            comp_tokens_gpu=z,
            pca_coefficients_to_save=None,
            last_loss=last_loss,
            last_convergence=last_convergence,
            per_sample_info_gain=gain.tolist(),
            embeddings_dir=self._prepare_embeddings_dir(),
        )
        row.update(
            embedding_role=ROLE,
            initialization_source="paired_standard_hinge_control",
            information_gain_reference="zero_vector_same_interface",
        )
        return [row]

    def _save_artifacts(self, rows, *, tensor, tensor_filename, subdir_name):
        return super()._save_artifacts(
            rows,
            tensor=tensor,
            tensor_filename="additive_embeddings.pt",
            subdir_name="additive_prefixes",
        )


@torch.no_grad()
def check_sample(model, tokenizer, ids, memory):
    if model.config.sps or ids.shape[0] != 1:
        raise ValueError("Expected native Standard and batch one")
    ids = ids.to(model.device)
    embedding = model.get_input_embeddings()
    memory = memory.to(device=model.device, dtype=embedding.weight.dtype)
    inputs = condition_inputs(memory, embedding(ids))
    replay = (
        model(
            inputs_embeds=inputs,
            attention_mask=torch.ones_like(ids.new_ones(inputs.shape[:2])),
        )
        .logits[:, :-1]
        .argmax(-1)
    )
    replay_correct = int((replay == ids).sum())
    generated = ids.new_empty((1, 0))
    for _ in range(ids.shape[1]):
        context = condition_inputs(memory, embedding(generated))
        logits = model(inputs_embeds=context, attention_mask=ids.new_ones(context.shape[:2])).logits[:, -1]
        token = mask_native_generation_logits(model, tokenizer, logits).argmax(-1)
        generated = torch.cat((generated, token[:, None]), dim=1)
        if tokenizer.eos_token_id is not None and token.item() == tokenizer.eos_token_id:
            break
    correct = int((generated == ids[:, : generated.shape[1]]).sum())
    return dict(
        teacher_replay_correct_tokens=replay_correct,
        teacher_replay_accuracy=replay_correct / ids.numel(),
        greedy_accuracy=correct / ids.numel(),
        greedy_exact=generated.shape == ids.shape and correct == ids.numel(),
        target_ids=ids[0].tolist(),
        generated_ids=generated[0].tolist(),
    )


def control_initializations():
    if not (BASELINE / "completed").exists():
        raise ValueError("Missing completed paired Standard control")
    design = json.loads((BASELINE / "design.json").read_text())
    expected = dict(ordinary.TRAINING_ARGS, **CONTROL["training_args"])
    expected["tokenized_dataset_path"] = str(jobs.CACHE_ROOT / Path(expected["tokenized_dataset_path"]).name)
    for key, value in expected.items():
        actual = design["training_args"][key]
        if key == "tokenized_dataset_path":
            actual, value = jobs.cluster_path(actual), jobs.cluster_path(value)
        if actual != value:
            raise ValueError(f"Control setting mismatch: {key}")
    if jobs.cluster_path(design["experiment"]["checkpoint"]) != jobs.cluster_path(CHECKPOINT):
        raise ValueError("Control checkpoint mismatch")
    raw = load_from_disk(BASELINE / "progressive_prefixes")
    rows = [r for r in raw.select_columns(["sample_id", "stage_index", "initialization_embedding"]) if r["stage_index"] == 0]
    if sorted(r["sample_id"] for r in rows) != list(range(COHORT_SIZE)):
        raise ValueError("Incomplete paired control initializations")
    return {r["sample_id"]: r["initialization_embedding"] for r in rows}


def probe(exp):
    settings = resolved_training_args(exp)
    set_launch_seed(settings["random_seed"])
    torch.set_num_threads(8)
    output = ROOT / exp["name"]
    if (output / "completed").exists():
        raise ValueError("Refusing to overwrite a completed experiment")
    initializations = control_initializations()
    args = MyTrainingArguments(
        **settings,
        model_checkpoint=jobs.cluster_path(CHECKPOINT),
        output_dir=str(output),
        logging_dir=str(output / "logs"),
    )
    jobs.write_json(
        output / "design.json",
        dict(
            experiment=exp,
            training_args=settings,
            source_commit=source_commit(),
            initialization_source=str(BASELINE),
            embedding_role=ROLE,
        ),
    )
    model = load_sps_checkpoint(args.model_checkpoint, torch_dtype=torch.bfloat16)
    tokenizer = load_sps_tokenizer()
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
    trainer = AdditiveCrammingTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        initializations=initializations,
    )
    path = trainer.train()
    rows = select_rows(load_from_disk(path), "progr")
    if [r["sample_id"] for r in rows] != list(range(exp["offset"], exp["offset"] + exp["count"])):
        raise ValueError("Incomplete additive cohort")
    samples = []
    for row in rows:
        sid, length = row["sample_id"], row["num_input_tokens"]
        ids = dataset[sid - exp["offset"]]["input_ids"][:length][None]
        result = check_sample(model, tokenizer, ids, torch.tensor([row["embedding"]]))
        result.update(
            sample_id=sid,
            horizon=(length if row["final_convergence"] >= row["convergence_threshold"] else 0),
            stored_convergence=row["final_convergence"],
        )
        samples.append(result)
    report = dict(
        source_commit=source_commit(),
        experiment_name=exp["name"],
        training_args=settings,
        embedding_role=ROLE,
        trainable_scalars=model.config.hidden_size,
        initialization_source=str(BASELINE),
        n_samples=len(samples),
        mean_horizon=float(np.mean([s["horizon"] for s in samples])),
        greedy_exact_samples=sum(s["greedy_exact"] for s in samples),
        teacher_replay_exact_samples=sum(s["teacher_replay_correct_tokens"] == len(s["target_ids"]) for s in samples),
        samples=samples,
    )
    checked_samples(
        report,
        range(exp["offset"], exp["offset"] + exp["count"]),
        load_from_disk(jobs.cluster_path(settings["tokenized_dataset_path"])),
    )
    jobs.write_json(output / "single_sample_validation.json", report)
    (output / "completed").touch()
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


def submit(dry=False):
    commit = source_commit()
    prior = [ROOT / e["name"] / "job.json" for e in experiment_configs]
    commits = {json.loads(p.read_text())["commit"] for p in prior if p.exists()}
    if not dry and (len(commits) > 1 or (commits and commits != {commit} and not all(p.exists() for p in prior))):
        raise ValueError("Resume partial submissions from their immutable source snapshot")
    records = []
    for exp, path in zip(experiment_configs, prior):
        jobs.OUTPUT = ROOT / exp["name"]
        if not dry and path.exists():
            print("Existing submission:", jobs.OUTPUT, flush=True)
        else:
            jobs.submit(
                dry=dry,
                entrypoint="scripts/analyze_standard_additive.py",
                job_kind=exp["name"],
                probe_args=("--experiment", exp["name"]),
                snapshot_commit=commit,
            )
        if not dry:
            records.append(dict(experiment=exp["name"], **json.loads(path.read_text())))
    if not dry:
        jobs.write_json(ROOT / "submission_manifest.json", dict(shards=records))


def merge():
    output = ROOT / "merged"
    if (output / "completed").exists():
        return output
    if output.exists():
        raise ValueError("Refusing to replace an incomplete merge")
    if not all((ROOT / exp["name"] / "completed").exists() for exp in experiment_configs):
        raise ValueError("All additive shards must finish successfully")
    cache = load_from_disk(jobs.cluster_path(resolved_training_args(experiment_configs[0])["tokenized_dataset_path"]))
    initializations = control_initializations()
    control = json.loads((BASELINE / "single_sample_validation.json").read_text())
    controls = checked_samples(control, range(COHORT_SIZE), cache)
    samples, datasets, provenance = [], [], []
    commit = None
    seen_jobs = set()
    for exp in experiment_configs:
        path = ROOT / exp["name"]
        report = json.loads((path / "single_sample_validation.json").read_text())
        job = json.loads((path / "job.json").read_text())
        commit = commit or report["source_commit"]
        job_id = job["result"]["job_name"]
        if (
            not re.fullmatch(r"lm-mpi-job-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", job_id)
            or job_id in seen_jobs
            or not re.fullmatch(r"[0-9a-f]{40}", commit)
        ):
            raise ValueError("Missing full unique job UUID or source SHA")
        seen_jobs.add(job_id)
        if (
            report["source_commit"] != commit
            or job["commit"] != commit
            or report["experiment_name"] != exp["name"]
            or report["training_args"] != resolved_training_args(exp)
            or report["embedding_role"] != ROLE
            or report["initialization_source"] != str(BASELINE)
        ):
            raise ValueError("Shard settings, source snapshot or conditioning role mismatch")
        part = checked_samples(report, range(exp["offset"], exp["offset"] + exp["count"]), cache)
        raw = load_from_disk(path / "additive_prefixes")
        validate_raw(raw, part, exp, initializations, report["trainable_scalars"])
        samples.extend(part)
        datasets.append(raw)
        provenance.append(
            dict(
                experiment=exp["name"],
                job_id=job["result"]["job_name"],
                source_commit=commit,
            )
        )
    samples.sort(key=lambda s: s["sample_id"])
    if [s["sample_id"] for s in samples] != list(range(COHORT_SIZE)):
        raise ValueError("Merged cohort overlaps or is incomplete")
    prefixes = [leading_correct(s["target_ids"], s["generated_ids"]) for s in samples]
    base_prefixes = [leading_correct(s["target_ids"], s["generated_ids"]) for s in controls]
    report = dict(
        source_commit=commit,
        merge_commit=source_commit(),
        embedding_role=ROLE,
        trainable_scalars=report["trainable_scalars"],
        initialization_source=str(BASELINE),
        n_samples=len(samples),
        mean_horizon=float(np.mean([s["horizon"] for s in samples])),
        greedy_exact_samples=sum(s["greedy_exact"] for s in samples),
        teacher_replay_exact_samples=sum(s["teacher_replay_correct_tokens"] == len(s["target_ids"]) for s in samples),
        samples=samples,
    )
    summary = dict(
        mean_horizon=report["mean_horizon"],
        baseline_horizon=control["mean_horizon"],
        mean_greedy_initial_prefix=float(np.mean(prefixes)),
        baseline_greedy_initial_prefix=float(np.mean(base_prefixes)),
        greedy_exact_samples=report["greedy_exact_samples"],
        paired_horizon_gain=paired_interval([a["horizon"] - b["horizon"] for a, b in zip(samples, controls)]),
        paired_greedy_prefix_gain=paired_interval(np.asarray(prefixes) - base_prefixes),
    )
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=ROOT) as directory:
        staged = Path(directory) / "merged"
        staged.mkdir()
        concatenate_datasets(datasets).sort(["sample_id", "stage_index"]).save_to_disk(staged / "additive_prefixes")
        jobs.write_json(staged / "single_sample_validation.json", report)
        jobs.write_json(staged / "summary.json", summary)
        jobs.write_json(staged / "manifest.json", dict(shards=provenance, control=str(BASELINE)))
        plot(summary, staged / "comparison.png")
        (staged / "completed").touch()
        staged.rename(output)
    return output


def validate_raw(raw, samples, exp, initializations, hidden_size):
    args = MyTrainingArguments(**resolved_training_args(exp))
    by_id = {s["sample_id"]: s for s in samples}
    groups = {}
    for row in raw:
        groups.setdefault(row["sample_id"], []).append(row)
    if set(groups) != set(by_id):
        raise ValueError("Raw sample IDs do not match assigned cohort")
    for sid, rows in groups.items():
        steps = 0
        for index, row in enumerate(sorted(rows, key=lambda r: r["stage_index"])):
            if (
                row["stage_index"] != index
                or row["stage_seq_len"] != index + 1
                or row["num_input_tokens"] != index + 1
                or row["embedding_role"] != ROLE
                or row["num_compression_tokens"] != 1
                or row["hidden_size"] != hidden_size
                or jobs.cluster_path(row["model_checkpoint"]) != jobs.cluster_path(CHECKPOINT)
                or not 0 <= row["steps_taken"] - steps <= args.max_optimization_steps_per_token
                or row["steps_taken"] > args.max_optimization_steps_per_sample
                or not 0 <= row["final_convergence"] <= 1
            ):
                raise ValueError("Invalid raw stage metadata or counters")
            for key in (
                "token_objective",
                "loss_type",
                "loss_margin",
                "convergence_margin",
                "dtype",
                "max_optimization_steps_per_sample",
            ):
                if row[key] != getattr(args, key):
                    raise ValueError(f"Raw training setting mismatch: {key}")
            z, initial = np.asarray(row["embedding"]), np.asarray(row["initialization_embedding"])
            if (
                z.shape != (1, hidden_size)
                or not np.isfinite(z).all()
                or not np.array_equal(initial, initializations[sid])
                or row["convergence_threshold"] != args.progressive_convergence_threshold
            ):
                raise ValueError("Invalid saved vector or paired initialization")
            if index < len(rows) - 1 and row["final_convergence"] < row["convergence_threshold"]:
                raise ValueError("Stages after first failed prefix")
            steps = row["steps_taken"]
    for row in select_rows(raw, "progr"):
        sample = by_id[row["sample_id"]]
        horizon = row["num_input_tokens"] if row["final_convergence"] >= row["convergence_threshold"] else 0
        if (
            sample["horizon"] != horizon
            or len(sample["target_ids"]) != row["num_input_tokens"]
            or sample["stored_convergence"] != row["final_convergence"]
        ):
            raise ValueError("Selected stage differs from replay report")


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(2)
    ax.bar(
        x - 0.18,
        [summary["baseline_horizon"], summary["baseline_greedy_initial_prefix"]],
        0.36,
        label="Standard / prefix only",
    )
    ax.bar(
        x + 0.18,
        [summary["mean_horizon"], summary["mean_greedy_initial_prefix"]],
        0.36,
        label="Standard / prefix + additive",
    )
    ax.set_xticks(x, ["Teacher-forced horizon", "Initial correct greedy prefix"])
    ax.set(
        ylabel="Mean tokens",
        title="One shared vector / paired initialization / hinge / batch 1",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dry", "submit", "probe", "merge"), default="dry")
    parser.add_argument("--experiment", choices=[e["name"] for e in experiment_configs])
    cli = parser.parse_args()
    if cli.phase == "probe":
        if cli.experiment is None:
            parser.error("--experiment is required for probe")
        probe(next(e for e in experiment_configs if e["name"] == cli.experiment))
    else:
        {"dry": lambda: submit(dry=True), "submit": submit, "merge": merge}[cli.phase]()

"""Matched Standard-memory and SPS-predict cramming with fixed-order masked attention."""

import argparse
import json
import math
import statistics
import subprocess
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from scripts import analyze_sps_cramming as jobs
from scripts import analyze_sps_predict_embedding as predict
from scripts.analyze_sps_objective_factorial import leading_correct
from scripts.greedy_reconstruction_eval import select_rows
from scripts.native_fixed_attention import native_fixed_attention
from scripts.probe_sps_kv_replay import replay
from scripts.sps_mechanism_common import source_commit
from scripts.validation.validate_sps_batch1_run import check_sample
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from compression_horizon.models.sps_causal_lm import load_sps_checkpoint
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainers.progressive_cramming import (
    ProgressiveCrammingTrainer,
)

ROOT = jobs.ROOT / "fixed_attention_cramming_v1"
BI_REV = "f22b1fbe534b1fe045080f44a970a441caaf3c4f"
DEPENDENCIES = jobs.SHARED_PROJECT.parent / "sps-artifacts/dependencies"
GPT2_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SPS_DESIGN = next(e for e in predict.experiment_configs if e["name"] == "predict_hinge_native")
STANDARD = jobs.ROOT / "standard_pilot_1b_batch1_hinge_strict_feb2c4c51ffa"
SPS = predict.ROOT / SPS_DESIGN["name"]
experiment_configs = [
    dict(
        name=f"{family}_shard{shard:02d}",
        family=family,
        offset=12 * shard,
        count=12,
        checkpoint=str(source["checkpoint"]),
        baseline=str(baseline),
    )
    for family, source, baseline in [
        ("standard", jobs.experiment_configs[0], STANDARD),
        ("sps", jobs.experiment_configs[1], SPS),
    ]
    for shard in range(4)
]


wide_experiment_configs = [dict(exp, wide_sum=True) for exp in experiment_configs if exp["family"] == "sps"]
SUITES = {
    "original": (ROOT, experiment_configs),
    "wide": (jobs.ROOT / "sps_wide_cramming_v1", wide_experiment_configs),
}


@contextmanager
def fixed_backend(*, wide_sum=False):
    dep = DEPENDENCIES / BI_REV
    if subprocess.check_output(
        ["git", "-C", str(dep), "rev-parse", "HEAD"], text=True
    ).strip() != BI_REV or subprocess.check_output(["git", "-C", str(dep), "diff", "HEAD", "--", "batch_invariant_ops"]):
        raise ValueError("Wrong batch-invariant dependency revision")
    sys.path.insert(0, str(dep))
    from batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )

    enable_batch_invariant_mode()
    try:
        with patch.object(
            F,
            "scaled_dot_product_attention",
            partial(native_fixed_attention, wide_sum=wide_sum),
        ):
            yield
    finally:
        disable_batch_invariant_mode()
        sys.path.remove(str(dep))


class StandardTrainer(ProgressiveCrammingTrainer):
    def __init__(self, *args, initial_vectors, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_vectors = initial_vectors

    def _setup_batch(self, batch, ctx, *, sample_id_counter=0):
        result = super()._setup_batch(batch, ctx, sample_id_counter=sample_id_counter)
        vector = self.initial_vectors[self.args.offset_dataset_items + sample_id_counter]
        with torch.no_grad():
            result.parametrization.parameters[0].copy_(vector)
        result.initialization_embeddings = result.parametrization.materialize().detach().clone().cpu()
        return result

    def _run_progressive_stages(self, batch_ctx, ctx, sample_id_counter):
        return super()._run_progressive_stages(batch_ctx, ctx, sample_id_counter + self.args.offset_dataset_items)


def read_baseline(exp, ids):
    path = Path(exp["baseline"])
    if not (path / "completed").is_file():
        raise ValueError("Historical control incomplete")
    report = json.loads((path / "single_sample_validation.json").read_text())
    samples = {s["sample_id"]: s for s in report["samples"]}
    raw = load_from_disk(str(path / ("predict_prefixes" if exp["family"] == "sps" else "progressive_prefixes")))
    slim = raw.select_columns(["sample_id", "stage_index"])
    indexes = {r["sample_id"]: i for i, r in enumerate(slim) if r["sample_id"] in ids and r["stage_index"] == 0}
    if set(indexes) != set(ids):
        raise ValueError("Missing historical initial vectors")
    initial = {sid: torch.tensor(raw[i]["initialization_embedding"]) for sid, i in indexes.items()}
    return samples, initial


@torch.no_grad()
def growing_difference(model, tokens, memory, forward_kwargs):
    inputs = torch.cat((memory, model.get_input_embeddings()(tokens)), dim=1)

    def score(x):
        return model(
            inputs_embeds=x,
            attention_mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.long),
            **forward_kwargs,
        ).logits

    full = score(inputs)[0, :-1]
    unequal = 0
    maximum = 0.0
    for i in range(tokens.shape[1]):
        last = score(inputs[:, : i + 1].contiguous())[0, -1]
        unequal += int((last != full[i]).sum())
        delta = float((last.float() - full[i].float()).abs().max())
        if not math.isfinite(delta):
            raise ValueError("Nonfinite logit comparison")
        maximum = max(maximum, delta)
    return dict(unequal_logits=unequal, max_abs_delta=maximum)


def train(exp, output, smoke=False):
    output = Path(output)
    if (output / "completed").exists():
        raise FileExistsError(output)
    is_sps = exp["family"] == "sps"
    settings = predict.resolved_training_args(SPS_DESIGN)
    offset = exp["offset"]
    count = 1 if smoke else exp["count"]
    sample_ids = range(offset, offset + count)
    old, initial = read_baseline(exp, sample_ids)
    dataset = load_from_disk(settings["tokenized_dataset_path"]).select(sample_ids)
    for sid, row in zip(sample_ids, dataset, strict=True):
        old_ids = old[sid]["target_ids"]
        if row["input_ids"][: len(old_ids)] != old_ids:
            raise ValueError("Historical target mismatch")
    if smoke:
        dataset = dataset.map(lambda row: {k: row[k][:12] for k in ("input_ids", "attention_mask")})
        settings.update(
            max_sequence_length=12,
            max_optimization_steps_per_sample=160,
            max_optimization_steps_per_token=80,
            warmup_steps=1,
        )
    settings.update(
        limit_dataset_items=count,
        offset_dataset_items=offset,
        model_checkpoint=exp["checkpoint"],
        output_dir=str(output),
        logging_dir=str(output / "logs"),
    )
    model = load_sps_checkpoint(exp["checkpoint"], torch_dtype=torch.bfloat16)
    if model.config.sps != is_sps:
        raise ValueError("Checkpoint family mismatch")
    tokenizer = AutoTokenizer.from_pretrained(DEPENDENCIES / GPT2_REV, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.bos_token_id != model.config.eos_token_id:
        raise ValueError("Wrong tokenizer for native checkpoint")
    args = MyTrainingArguments(**settings)
    common = dict(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset.with_format("torch"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer = (
        predict.PredictEmbeddingTrainer(**common, bos_token_id=tokenizer.bos_token_id, predict_init="native")
        if is_sps
        else StandardTrainer(**common, initial_vectors=initial)
    )
    wide_sum = exp.get("wide_sum", False)
    with fixed_backend(wide_sum=wide_sum):
        path = trainer.train()
        chosen = select_rows(load_from_disk(path), "progr")
        if sorted(r["sample_id"] for r in chosen) != list(sample_ids):
            raise ValueError("Incomplete or duplicate shard samples")
        records = []
        for row in chosen:
            sid = row["sample_id"]
            if not torch.equal(torch.tensor(row["initialization_embedding"]), initial[sid]):
                raise ValueError("Initialization differs from the historical stage")
            n = row["num_input_tokens"]
            tokens = torch.tensor([dataset[sid - offset]["input_ids"][:n]], device=model.device)
            stored = torch.tensor([row["embedding"]], device=model.device, dtype=model.dtype)
            memory = predict.fixed_bos(model, 1, tokenizer.bos_token_id) if is_sps else stored
            forward_kwargs = {"predict_embeddings": stored} if is_sps else {}
            if wide_sum:
                target = tokens[0].tolist()
                result = replay(model, tokenizer, target, stored)
                cached = result["generated"]["cached"]
                sample = dict(
                    target_ids=target,
                    generated_ids=cached,
                    greedy_exact=cached == target,
                    uncached_exact=result["generated"]["uncached"] == target,
                    replay=result,
                    teacher_replay_correct_tokens=result["teacher_correct"],
                    unequal_logits=result["comparisons"]["cache_vs_full"]["unequal_logits"],
                )
            else:
                sample = check_sample(model, tokenizer, tokens, memory, **forward_kwargs)
                sample.update(growing_difference(model, tokens, memory, forward_kwargs))
            horizon = n if row["final_convergence"] >= row["convergence_threshold"] else 0
            if not wide_sum and horizon and sample["teacher_replay_correct_tokens"] != horizon:
                raise RuntimeError("Stored converged vector failed teacher replay")
            sample.update(
                sample_id=sid,
                horizon=horizon,
                steps=row["steps_taken"],
                old_horizon=old[sid]["horizon"],
                old_exact=old[sid]["greedy_exact"],
                old_leading=leading_correct(old[sid]["target_ids"], old[sid]["generated_ids"]),
            )
            records.append(sample)
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("Expected frozen checkpoints")
    jobs.write_json(
        output / "validation.json",
        dict(
            config=exp,
            training_args=settings,
            source_commit=source_commit(),
            batch_invariant_revision=BI_REV,
            samples=records,
        ),
    )
    (output / "completed").touch()


def summarize(root=ROOT, configs=experiment_configs):
    for family in dict.fromkeys(exp["family"] for exp in configs):
        rows = []
        for exp in configs:
            if exp["family"] == family:
                p = root / exp["name"]
                if not (p / "completed").exists():
                    raise RuntimeError(f"Incomplete shard: {p}")
                rows.extend(json.loads((p / "validation.json").read_text())["samples"])
        if sorted(s["sample_id"] for s in rows) != list(range(48)):
            raise ValueError("Incomplete panel")
        summary = dict(
            n=len(rows),
            new_mean_horizon=statistics.mean(r["horizon"] for r in rows),
            old_mean_horizon=statistics.mean(r["old_horizon"] for r in rows),
            converged=sum(r["horizon"] > 0 for r in rows),
            exact_and_converged=sum(r["horizon"] > 0 and r["greedy_exact"] for r in rows),
            samples_with_identical_logits=sum(r["unequal_logits"] == 0 for r in rows),
            new_exact=sum(r["greedy_exact"] for r in rows),
            old_exact=sum(r["old_exact"] for r in rows),
            new_leading=statistics.mean(leading_correct(r["target_ids"], r["generated_ids"]) for r in rows),
            old_leading=statistics.mean(r["old_leading"] for r in rows),
        )
        if "replay" in rows[0]:
            summary.update(
                teacher_exact=sum(r["teacher_replay_correct_tokens"] == len(r["target_ids"]) for r in rows),
                uncached_exact=sum(r["uncached_exact"] for r in rows),
                identical_blind_sequences=sum(r["generated_ids"] == r["replay"]["generated"]["uncached"] for r in rows),
                scored_positions=sum(len(r["target_ids"]) for r in rows),
                cache_vs_full_unequal_logits=sum(r["unequal_logits"] for r in rows),
            )
        jobs.write_json(root / (family + "_summary.json"), summary)
        print(family, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["submit", "dry", "probe", "smoke", "summary"], required=True)
    parser.add_argument("--experiment", choices=[e["name"] for e in experiment_configs])
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--suite", choices=SUITES, default="original")
    opts = parser.parse_args()
    root, configs = SUITES[opts.suite]
    if opts.phase in ("submit", "dry"):
        commit = source_commit()
        for exp in configs:
            jobs.OUTPUT = root / exp["name"]
            record = jobs.OUTPUT / "job.json"
            if opts.phase == "submit" and record.exists():
                if json.loads(record.read_text())["commit"] != commit:
                    raise RuntimeError("Resume submission from the original commit")
                continue
            jobs.submit(
                dry=opts.phase == "dry",
                entrypoint="scripts/run_native_fixed_cramming.py",
                job_kind=("wide_" if opts.suite == "wide" else "fixed_") + exp["name"],
                probe_args=("--experiment", exp["name"], "--suite", opts.suite),
                snapshot_commit=commit,
            )
    elif opts.phase == "summary":
        summarize(root, configs)
    else:
        exp = next((e for e in configs if e["name"] == opts.experiment), None)
        if exp is None:
            parser.error("Select an experiment from the requested suite")
        if opts.phase == "smoke" and opts.smoke_output is None:
            parser.error("--smoke-output required for smoke")
        train(
            exp,
            (opts.smoke_output / exp["name"] if opts.phase == "smoke" else root / exp["name"]),
            opts.phase == "smoke",
        )

"""Per-shard training and real-checkpoint preflight; designs live in the registry."""

import gc
import hashlib
import os
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from scripts.analyze_sps_predict_embedding import PredictEmbeddingTrainer, fixed_bos
from scripts.greedy_reconstruction_eval import select_rows
from scripts.pretrained_sps_experiments import (
    COHORT,
    PREFLIGHT,
    ROOT,
    TRAINING_ARGS,
    UPSTREAM_COMMIT,
    code_digest,
    experiment_configs,
    read_json,
    write_json,
)
from scripts.validation.validate_sps_batch1_run import check_sample
from transformers import DataCollatorForLanguageModeling

from compression_horizon.models.pretrained_sps import existing_bos, load_pretrained_sps, load_pretrained_tokenizer
from compression_horizon.models.sps_vendor.pretrained import PretrainedSPS
from compression_horizon.train.arguments import MyTrainingArguments
from compression_horizon.train.trainers.progressive_cramming import ProgressiveCrammingTrainer


class TokenOnlyTrainer(ProgressiveCrammingTrainer):
    def compute_hidden_states(self, model, token_embeddings, attention_mask):
        # Token-only loss never consumes alignment targets. Avoid the unused
        # full 4096-token/interleaved forward (especially Qwen's large logits).
        if self.args.hybrid_alpha is not None or self.args.num_alignment_layers:
            raise ValueError("This worker supports token-only reconstruction")
        return ()

    def _collect_stage_rows(self, *args, **kwargs):
        rows = super()._collect_stage_rows(*args, **kwargs)
        for row in rows:
            row["sample_id"] += self.args.offset_dataset_items or 0
        return rows


class PretrainedPredictTrainer(PredictEmbeddingTrainer, TokenOnlyTrainer):
    def _setup_batch(self, batch, ctx, *, sample_id_counter=0):
        result = ProgressiveCrammingTrainer._setup_batch(self, batch, ctx, sample_id_counter=sample_id_counter)
        with torch.no_grad():
            result.parametrization.parameters[0].copy_(ctx.model.prediction.view(1, 1, -1))
        result.initialization_embeddings = result.parametrization.materialize().detach().clone().cpu()
        return result


def model_and_tokenizer(exp, dtype=torch.bfloat16):
    model = load_pretrained_sps(
        exp["source"], exp["revision"], checkpoint=exp["checkpoint"], source_path=exp["source_path"], torch_dtype=dtype
    )
    tokenizer = load_pretrained_tokenizer(exp["source_path"], exp["revision"])
    existing_bos(model, tokenizer)
    return model, tokenizer


def prepare_data():
    """Pin raw texts once, then family-native targets once, outside snapshots."""
    dest = ROOT / "data"
    design = dict(
        cohort=COHORT,
        sequence_cap=TRAINING_ARGS["max_sequence_length"],
        dataset=TRAINING_ARGS["dataset_name"],
        sources={e["family"]: [e["source"], e["revision"]] for e in experiment_configs},
    )
    if (dest / "completed").exists():
        if read_json(dest / "design.json") != design:
            raise ValueError("Prepared data does not match the source registry")
        return
    if dest.exists():
        raise RuntimeError("Partial data preparation exists; inspect before retry")
    raw = load_dataset(TRAINING_ARGS["dataset_name"], split="train").select(range(COHORT))
    texts = [row["text"] for row in raw]
    cohorts = [dict(sample_id=i, text=text, sha256=hashlib.sha256(text.encode()).hexdigest()) for i, text in enumerate(texts)]
    write_json(dest / "design.json", design)
    write_json(dest / "raw.json", cohorts)
    for exp in experiment_configs:
        target = dest / exp["family"]
        if target.exists():
            continue
        tokenizer = load_pretrained_tokenizer(exp["source_path"], exp["revision"])
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=TRAINING_ARGS["max_sequence_length"],
        )
        Dataset.from_dict(dict(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])).save_to_disk(
            str(target)
        )
        write_json(
            dest / (exp["family"] + ".json"),
            dict(
                source=exp["source"],
                revision=exp["revision"],
                targets_sha256=hashlib.sha256(str(encoded["input_ids"]).encode()).hexdigest(),
                document_hashes=[r["sha256"] for r in cohorts],
            ),
        )
    (dest / "completed").touch()


def train_shard(exp, offset, count, output, *, smoke=False):
    started = time.monotonic()
    settings = dict(
        TRAINING_ARGS,
        model_checkpoint=exp["checkpoint"] or exp["source"],
        output_dir=str(output),
        logging_dir=str(output / "logs"),
        offset_dataset_items=offset,
        limit_dataset_items=count,
    )
    if smoke:
        settings.update(
            max_sequence_length=PREFLIGHT["tokens"],
            max_optimization_steps_per_sample=PREFLIGHT["steps"],
            max_optimization_steps_per_token=PREFLIGHT["steps"],
            warmup_steps=0,
        )
    model, tokenizer = model_and_tokenizer(exp)
    bos = existing_bos(model, tokenizer)
    data = load_from_disk(str(ROOT / "data" / exp["family"]))
    data_manifest = read_json(ROOT / "data" / (exp["family"] + ".json"))
    if hashlib.sha256(str(list(data["input_ids"])).encode()).hexdigest() != data_manifest["targets_sha256"]:
        raise ValueError("Pinned target cache changed after preparation")
    data = data.select(range(offset, offset + count))
    if smoke:
        data = Dataset.from_dict({k: [row[k][: PREFLIGHT["tokens"]] for row in data] for k in ("input_ids", "attention_mask")})
    args = MyTrainingArguments(**settings)
    cls = TokenOnlyTrainer if exp["arm"] == "ar" else PretrainedPredictTrainer
    extra = {} if exp["arm"] == "ar" else dict(bos_token_id=bos, predict_init="native")
    trainer = cls(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=data.with_format("torch"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        **extra,
    )
    path = trainer.train()
    selected = list(select_rows(load_from_disk(path), "progr"))
    if {row["sample_id"] for row in selected} != set(range(offset, offset + count)) or len(selected) != count:
        raise ValueError("Missing/duplicate selected sample")
    records = []
    for row in selected:
        sid, length = row["sample_id"], row["num_input_tokens"]
        ids = torch.tensor(data[sid - offset]["input_ids"][:length], device=model.device)[None]
        z = torch.tensor(row["embedding"], device=model.device, dtype=model.dtype)[None]
        memory = z if exp["arm"] == "ar" else fixed_bos(model, 1, bos)
        report = check_sample(model, tokenizer, ids, memory, **({} if exp["arm"] == "ar" else dict(predict_embeddings=z)))
        horizon = length if row["final_convergence"] >= row["convergence_threshold"] else 0
        if horizon and report["teacher_replay_correct_tokens"] != length:
            raise ValueError("Converged selected vector does not replay exactly")
        report.update(
            sample_id=sid,
            horizon=horizon,
            selected_length=length,
            steps_taken=row["steps_taken"],
            stored_convergence=row["final_convergence"],
        )
        records.append(report)
    write_json(
        output / "single_sample_validation.json",
        dict(
            experiment=exp,
            settings=settings,
            code_digest=code_digest(Path(__file__).resolve().parents[1]),
            source_commit=os.environ.get("RUN_SOURCE_COMMIT"),
            data_manifest=data_manifest,
            bos_token_id=bos,
            embedding_role="ordinary_memory" if exp["arm"] == "ar" else "shared_predict",
            elapsed_seconds=time.monotonic() - started,
            samples=records,
        ),
    )
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError("Model weights were not kept frozen")
    (output / "completed").touch()
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def parity(exp):
    """Reference input-ID logits/gradients and independent ordinary-history scoring."""
    model, tokenizer = model_and_tokenizer(exp)
    model.cuda()
    ids = torch.tensor(
        load_from_disk(str(ROOT / "data" / exp["family"]))[0]["input_ids"][: PREFLIGHT["tokens"]], device="cuda"
    )[None]
    x = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    z = model.prediction.detach().view(1, 1, -1).clone().requires_grad_(True)
    actual = model(inputs_embeds=x, **(dict(predict_embeddings=z) if model.sps else {})).logits
    if model.sps:
        reference = PretrainedSPS(model.backbone).eval()
        reference.prediction = torch.nn.Parameter(model.prediction.detach().clone())
        xr = x.detach().clone().requires_grad_(True)
        handle = reference.backbone.get_input_embeddings().register_forward_hook(lambda _m, _a, _o: xr)
        try:
            expected = reference(ids).logits
        finally:
            handle.remove()
        ga = torch.autograd.grad(actual.float().square().mean(), (x, z))
        gr = torch.autograd.grad(expected.float().square().mean(), (xr, reference.prediction))
        for a, b in zip(ga, gr):
            torch.testing.assert_close(a.flatten(), b.flatten(), atol=PREFLIGHT["atol"], rtol=PREFLIGHT["rtol"])
        # No earlier prediction positions in this independent forward.
        context = x.detach()
        current = torch.cat((context, z.detach()), 1)
        last = model.backbone.model(
            inputs_embeds=current, position_ids=torch.arange(current.shape[1], device="cuda")[None], use_cache=False
        ).last_hidden_state[:, -1:]
        one_pred = model.backbone.lm_head(last)
        torch.testing.assert_close(actual[:, -1:], one_pred, atol=PREFLIGHT["bf16_atol"], rtol=PREFLIGHT["bf16_rtol"])
        del reference
    else:
        expected = model.backbone(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=PREFLIGHT["atol"], rtol=PREFLIGHT["rtol"])
    result = dict(
        experiment=exp["name"],
        max_logit_error=float((actual - expected).abs().max()),
        upstream_commit=UPSTREAM_COMMIT,
        bos_token_id=existing_bos(model, tokenizer),
    )
    del model, actual, expected
    gc.collect()
    torch.cuda.empty_cache()
    return result


def preflight(family, output):
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("Preflight must execute on the submitted GPU")
    results = []
    for exp in [e for e in experiment_configs if e["family"] == family]:
        results.append(parity(exp))
        train_shard(exp, 0, 1, output / exp["name"], smoke=True)
    write_json(
        output / "preflight.json",
        dict(family=family, results=results, code_digest=code_digest(Path(__file__).resolve().parents[1])),
    )
    (output / "completed").touch()

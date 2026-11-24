"""Natural PG19 constraints at the saved SPS cramming boundary."""

import argparse
import json
import random

import torch
from datasets import load_from_disk
from scripts import run_native_fixed_cramming as fixed
from scripts.run_native_fixed_cramming import jobs, load_sps_checkpoint

ARCHIVE = jobs.ROOT / "natural_constraints_v1"
ARCHIVE_COMMIT = "0f4bbe7156b3cbb30bbda86acae5ffbea0794051"
ROOT = jobs.ROOT / "natural_constraints_v2"
experiment_configs = [
    dict(
        name=f"shard{i:02d}",
        source=e,
        steps=160,
        lr=0.03,
        margin=0.05,
        protected=(0, 10, 25, 50, "all"),
        seeds=(117, 311),
        caliper=1.0,
    )
    for i, e in enumerate(x for x in fixed.experiment_configs if x["family"] == "sps")
]


def evaluate(model, context, vector, ids):
    logits = (
        model(
            inputs_embeds=context,
            attention_mask=torch.ones_like(context[:, :, 0], dtype=torch.long),
            predict_embeddings=vector.to(context.dtype),
        )
        .logits[0, :-1]
        .float()
    )
    target = logits.gather(1, ids[:, None]).flatten()
    competitor = logits.scatter(1, ids[:, None], -torch.inf).amax(1)
    return target - competitor, logits.argmax(1).eq(ids)


def fit(model, context, start, ids, selected, cfg):
    vector = start.float().detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([vector], lr=cfg["lr"])
    first_lost = {}
    for step in range(cfg["steps"] + 1):
        margins, correct = evaluate(model, context, vector, ids)
        values = margins.detach().tolist()
        if step == 0:
            initial = values
        for pos, (before, now) in enumerate(zip(initial, values)):
            if before >= cfg["margin"] > now and pos not in first_lost:
                first_lost[pos] = step
        active = margins[selected]
        success = bool((active >= cfg["margin"]).all())
        if success or step == cfg["steps"]:
            return dict(
                selected=selected,
                success=success,
                steps=step,
                initial=initial,
                final=values,
                correct=correct.tolist(),
                first_lost=first_lost,
            )
        optimizer.zero_grad()
        (cfg["margin"] - active).clamp_min(0).square().mean().backward()
        optimizer.step()


def matched_pair(ids, margins, cfg):
    """Anchor a,b with shared ending, replace b by c with the same target as b."""
    candidates = []
    for a in range(1, len(ids)):
        for b in range(1, len(ids)):
            if ids[a] == ids[b] or ids[a - 1] != ids[b - 1]:
                continue
            suffix = 1
            while suffix < min(a, b) and ids[a - 1 - suffix] == ids[b - 1 - suffix]:
                suffix += 1
            for c in range(1, len(ids)):
                gap = abs(margins[b] - margins[c])
                if ids[c] != ids[b] or ids[c - 1] == ids[a - 1] or gap > cfg["caliper"]:
                    continue
                rank = (-suffix, gap, abs(c - b), a, b, c)
                candidates.append((rank, dict(a=a, b=b, c=c, suffix=suffix, gap=gap)))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def probe(cfg):
    source = cfg["source"]
    directory = fixed.ROOT / source["name"]
    output = ROOT / cfg["name"]
    report = json.loads((directory / "validation.json").read_text())
    dataset = load_from_disk(report["training_args"]["tokenized_dataset_path"])
    stages = load_from_disk(str(directory / "predict_prefixes"))
    index = {
        (r["sample_id"], r["num_input_tokens"]): i
        for i, r in enumerate(stages.select_columns(["sample_id", "num_input_tokens"]))
    }
    model = load_sps_checkpoint(source["checkpoint"], torch_dtype=torch.bfloat16).cuda()
    native = model.get_input_embeddings().weight[model.config.predict_token_id].view(1, 1, -1)
    with fixed.fixed_backend():
        for sample in report["samples"]:
            sid, h = sample["sample_id"], sample["horizon"]
            saved = stages[index[sid, h]]
            failed = stages[index[sid, h + 1]]
            tokens = dataset[sid]["input_ids"][: h + 1]
            if (
                tokens[:-1] != sample["target_ids"]
                or len(tokens) != h + 1
                or saved["final_convergence"] < saved["convergence_threshold"]
                or failed["final_convergence"] >= failed["convergence_threshold"]
            ):
                raise ValueError(f"Invalid boundary {sid}")
            ids = torch.tensor(tokens, device="cuda")
            inputs = torch.tensor([[model.config.eos_token_id, *tokens]], device="cuda")
            context = model.get_input_embeddings()(inputs)
            start = torch.tensor([saved["embedding"]], device="cuda", dtype=context.dtype)
            prior, _ = evaluate(model, context[:, :-1].contiguous(), start, ids[:-1])
            archive = ARCHIVE / cfg["name"]
            archived = None
            if (archive / "completed").is_file() and json.loads((archive / "job.json").read_text())["commit"] == ARCHIVE_COMMIT:
                archived = json.loads((archive / f"sample_{sid:02d}.json").read_text())
                if archived["tokens"] != tokens or archived["previous"] != prior.tolist():
                    raise ValueError(f"Boundary replay changed for {sid}")
                arms = archived["arms"]
            else:
                arms = {}
                for seed in cfg["seeds"]:
                    order = list(range(h))
                    random.Random(seed + sid).shuffle(order)
                    for k in cfg["protected"]:
                        if k in (0, "all") and seed != cfg["seeds"][0]:
                            continue
                        subset = order[:k] if isinstance(k, int) else order
                        arms[f"{seed}_{k}"] = fit(model, context, start, ids, sorted(subset) + [h], cfg)
            natural, _ = evaluate(model, context, native, ids)
            pair = matched_pair(tokens, natural.tolist(), cfg)
            paired = {}
            if pair and archived and archived["pair"] and all(archived["pair"][key] == pair[key] for key in "abc"):
                if archived["paired"]["a"]["initial"] != natural.tolist():
                    raise ValueError(f"Native replay changed for {sid}")
                paired = archived["paired"]
            elif pair:
                a, b, c = (pair[key] for key in "abc")
                for name, selected in dict(a=[a], b=[b], c=[c], shared=[a, b], control=[a, c]).items():
                    paired[name] = fit(model, context, native, ids, selected, cfg)
            jobs.write_json(
                output / f"sample_{sid:02d}.json",
                dict(
                    sid=sid,
                    tokens=tokens,
                    horizon=h,
                    total_budget=failed["steps_taken"] >= failed["max_optimization_steps_per_sample"],
                    previous=prior.tolist(),
                    arms=arms,
                    pair=pair,
                    paired=paired,
                ),
            )
            print(f"Finished {sid}", flush=True)
    (output / "completed").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dry", "submit", "probe"), required=True)
    parser.add_argument("--experiment", choices=[e["name"] for e in experiment_configs])
    args = parser.parse_args()
    for config in experiment_configs:
        if args.experiment and config["name"] != args.experiment:
            continue
        if args.phase == "probe":
            probe(config)
        else:
            jobs.OUTPUT = ROOT / config["name"]
            jobs.submit(
                dry=args.phase == "dry",
                entrypoint="scripts/run_sps_natural_constraints.py",
                job_kind="natural_constraints_" + config["name"],
                probe_args=("--experiment", config["name"]),
                snapshot_commit=fixed.source_commit(),
            )

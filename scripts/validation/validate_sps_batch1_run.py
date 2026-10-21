"""Audit a new single-sample native cramming run using exact cached target IDs.

Runs after training in the same staged working directory. It reports teacher-
forcing replay and greedy reconstruction separately, using one sample per call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scripts.activation_distillation import load_or_create_tokenized_dataset
from scripts.greedy_reconstruction_eval import select_rows

from compression_horizon.inference.generation import generate_from_compression
from compression_horizon.models.sps_causal_lm import load_sps_checkpoint, load_sps_tokenizer
from compression_horizon.utils.launch import resolve_torch_dtype


@torch.no_grad()
def check_sample(model, tokenizer, ids, memory):
    if ids.shape[0] != 1 or memory.shape[0] != 1:
        raise ValueError("Cramming validation requires batch_size=1")
    embeddings = model.get_input_embeddings()
    memory = memory.to(device=model.device, dtype=embeddings.weight.dtype)
    ids = ids.to(model.device)
    inputs = torch.cat((memory, embeddings(ids)), dim=1)
    offset = memory.shape[1]
    predicted = (
        model(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device))
        .logits[:, offset - 1 : -1]
        .argmax(-1)
    )
    replay = float((predicted == ids).float().mean())
    _, generated = generate_from_compression(model, tokenizer, memory, max_new_tokens=ids.shape[1], return_generated_ids=True)
    n = min(ids.shape[1], generated.shape[1])
    correct = int((generated[:, :n] == ids[:, :n]).sum())
    # Missing tokens after early EOS remain errors; never score just the short prefix.
    return dict(
        teacher_replay_accuracy=replay,
        greedy_accuracy=correct / ids.shape[1],
        greedy_exact=generated.shape[1] == ids.shape[1] and correct == ids.shape[1],
        target_ids=ids[0].tolist(),
        generated_ids=generated[0].tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads((args.run_dir / "design.json").read_text())
    settings = design["training_args"]
    if settings["per_device_train_batch_size"] != 1:
        raise ValueError("This audit requires a single-sample training run")
    rows = select_rows(load_from_disk(str(args.run_dir / "progressive_prefixes")), "progr")
    expected = settings["limit_dataset_items"]
    if len(rows) != expected or {r["sample_id"] for r in rows} != set(range(expected)):
        raise ValueError("Incomplete or mismatched sample set")
    checkpoint = rows[0]["model_checkpoint"]
    tokenizer = load_sps_tokenizer()
    model = load_sps_checkpoint(checkpoint, torch_dtype=resolve_torch_dtype(settings["dtype"])).cuda()
    pinned = settings.get("tokenized_dataset_path")
    if pinned and not Path(pinned).exists():
        pinned = pinned.replace("/mnt/virtual_ai0001053-00054_SR004-nfs2", "/workspace-SR004.nfs2")
    dataset = load_or_create_tokenized_dataset(
        cache_dir="artifacts/cache/tokenized_datasets",
        dataset_name=settings["dataset_name"],
        split="test",
        tokenizer=tokenizer,
        max_sequence_length=settings["max_sequence_length"],
        model_checkpoint=checkpoint,
        no_bos_token=settings["no_bos_token"],
        limit_dataset_items=expected,
        tokenized_dataset_path=pinned,
    )
    samples = []
    for row in rows:
        length = row["num_input_tokens"]
        sample = row["sample_id"]
        # Do not re-tokenize the decoded prefix: truncated UTF-8 can change IDs.
        ids = dataset[sample]["input_ids"][:length][None]
        memory = torch.tensor(row["embedding"], dtype=torch.float32)[None]
        result = check_sample(model, tokenizer, ids, memory)
        result.update(
            sample_id=sample,
            horizon=length if row["final_convergence"] >= row["convergence_threshold"] else 0,
            stored_teacher_accuracy=row["final_convergence"],
        )
        samples.append(result)
    result = dict(
        training_commit=design["commit"],
        model_checkpoint=checkpoint,
        n_samples=len(samples),
        mean_horizon=float(np.mean([r["horizon"] for r in samples])),
        median_horizon=float(np.median([r["horizon"] for r in samples])),
        greedy_exact_samples=sum(r["greedy_exact"] for r in samples),
        mean_greedy_accuracy=float(np.mean([r["greedy_accuracy"] for r in samples])),
        teacher_replay_exact_samples=sum(r["teacher_replay_accuracy"] == 1 for r in samples),
        samples=samples,
    )
    target = args.run_dir / "single_sample_validation.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(target)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()

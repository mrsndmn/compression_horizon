"""Replay the archived native SPS vectors with the FP64 row sum and a real KV cache."""

import json

import torch
from datasets import load_from_disk
from scripts.analyze_sps_predict_embedding import fixed_bos
from scripts.run_native_fixed_cramming import (
    DEPENDENCIES,
    GPT2_REV,
    ROOT,
    experiment_configs,
    fixed_backend,
)
from transformers import AutoTokenizer

from compression_horizon.inference.generation import generate_from_compression
from compression_horizon.models.sps_causal_lm import load_sps_checkpoint


@torch.inference_mode()
def replay(model, tokenizer, target, vector):
    ids = torch.tensor([target], device=model.device)
    memory = fixed_bos(model, 1, tokenizer.bos_token_id)
    inputs = torch.cat((memory, model.get_input_embeddings()(ids)), dim=1)
    kwargs = {"predict_embeddings": vector}
    full = model(inputs_embeds=inputs, **kwargs).logits
    comparisons = {
        name: dict(unequal_logits=0, top1_changes=0, max_abs=0.0)
        for name in ("cache_vs_growing", "cache_vs_full", "growing_vs_full")
    }
    past = None
    for i in range(len(target)):
        cached = model(
            inputs_embeds=inputs[:, i : i + 1],
            use_cache=True,
            past_key_values=past,
            **kwargs,
        )
        past = cached.past_key_values
        short = model(inputs_embeds=inputs[:, : i + 1].contiguous(), **kwargs).logits[0, -1]
        current = cached.logits[0, -1]
        for name, left, right in (
            ("cache_vs_growing", current, short),
            ("cache_vs_full", current, full[0, i]),
            ("growing_vs_full", short, full[0, i]),
        ):
            delta = (left.float() - right.float()).abs().max().item()
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError("Nonfinite logits")
            cmp = comparisons[name]
            cmp["unequal_logits"] += int((left != right).sum())
            cmp["top1_changes"] += int(left.argmax() != right.argmax())
            cmp["max_abs"] = max(cmp["max_abs"], delta)
    generated = {}
    for cached in (False, True):
        _, tokens = generate_from_compression(
            model,
            tokenizer,
            memory,
            len(target),
            return_generated_ids=True,
            use_cache=cached,
            **kwargs,
        )
        generated["cached" if cached else "uncached"] = tokens[0].tolist()
    return dict(
        target=target,
        generated=generated,
        comparisons=comparisons,
        vocab_size=full.size(-1),
        teacher_correct=int((full[0, :-1].argmax(-1) == ids[0]).sum()),
    )


def main():
    configs = [e for e in experiment_configs if e["family"] == "sps"]
    model = load_sps_checkpoint(configs[0]["checkpoint"], torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(DEPENDENCIES / GPT2_REV, local_files_only=True)
    records = []
    with fixed_backend(wide_sum=True):
        for exp in configs:
            directory = ROOT / exp["name"]
            ds = load_from_disk(str(directory / "predict_prefixes"))
            index = {}
            for i, row in enumerate(
                ds.select_columns(
                    [
                        "sample_id",
                        "num_input_tokens",
                        "final_convergence",
                        "convergence_threshold",
                        "stage_index",
                    ]
                )
            ):
                if row["final_convergence"] >= row["convergence_threshold"]:
                    key = (row["sample_id"], row["num_input_tokens"])
                    index[key] = max(index.get(key, (-1, -1)), (row["stage_index"], i))
            for sample in json.loads((directory / "validation.json").read_text())["samples"]:
                sid = sample["sample_id"]
                target = sample["target_ids"]
                row = ds[index[(sid, len(target))][1]]
                vector = torch.tensor([row["embedding"]], device=model.device, dtype=model.dtype)
                result = replay(model, tokenizer, target, vector)
                records.append(dict(sample_id=sid, **result))
                print(
                    sid,
                    len(target),
                    result["generated"]["cached"] == target,
                    result["comparisons"],
                    flush=True,
                )
    if sorted(r["sample_id"] for r in records) != list(range(48)):
        raise ValueError("Incomplete or duplicate cohort")
    (ROOT / "sps_kv_replay.json").write_text(json.dumps(records) + "\n")


if __name__ == "__main__":
    main()

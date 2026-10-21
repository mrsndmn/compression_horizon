"""Bounded audit of native decoding semantics and reserved-ID failures.

Checks the actual saved first-failure memories for both models and compares
SPS greedy decoding with an ordinary-only context plus one current predict.
Uses original cached IDs, avoiding decode/re-tokenization changes.
"""

from __future__ import annotations

import gc
import json

import torch

from compression_horizon.inference.generation import generate_from_compression, mask_native_generation_logits
from compression_horizon.models.sps_causal_lm import load_sps_checkpoint, load_sps_tokenizer


def ordinary_history_next_logits(model, context):
    """Independent w=0 reference with no historical prediction slots."""
    b, n, _ = context.shape
    predict = model.get_input_embeddings().weight[model.config.predict_token_id].view(1, 1, -1).expand(b, 1, -1)
    x = torch.cat((context, predict), dim=1)
    positions = torch.cat((torch.arange(n, device=x.device), torch.tensor([n - 1], device=x.device)))
    freqs = model.freqs_cis[positions].unsqueeze(0).expand(b, -1, -1)
    causal = torch.ones(n + 1, n + 1, dtype=torch.bool, device=x.device).tril()[None, None]
    for block in model.transformer.h:
        x, _ = block(x, freqs, attn_block_mask=causal)
    return model.lm_head(model.transformer.output_norm(x))[:, -1]


@torch.no_grad()
def validate():
    from scripts.analyze_sps_cramming import OUTPUT, experiment_configs, group_rows, load_rows, token_cache

    torch.set_num_threads(2)
    tokenizer = load_sps_tokenizer()
    tokens = token_cache()
    report = {}
    for exp in experiment_configs:
        model = load_sps_checkpoint(exp["checkpoint"], torch_dtype=torch.bfloat16).cuda()
        groups = group_rows(load_rows(exp))
        failed = [(sample, next(r for r in rows if r["final_convergence"] < 1)) for sample, rows in groups.items()]
        record = dict(
            sample_count=len(failed),
            raw_wrong_tokens=0,
            reserved_wrong_tokens=0,
            exact_after_filter=0,
            replay_accuracy_max_error=0.0,
        )
        for offset in range(0, len(failed), 8):
            batch = failed[offset : offset + 8]
            lengths = torch.tensor([r["stage_seq_len"] for _, r in batch], device="cuda")
            ids = torch.tensor([tokens[i]["input_ids"][: int(lengths.max())] for i, _ in batch], device="cuda")
            mask = torch.arange(ids.shape[1], device="cuda")[None] < lengths[:, None]
            memory = torch.tensor([r["embedding"] for _, r in batch], device="cuda", dtype=model.dtype)
            embeds = torch.cat((memory, model.get_input_embeddings()(ids)), dim=1)
            attention = torch.cat((torch.ones_like(mask[:, :1]), mask), dim=1)
            logits = model(inputs_embeds=embeds, attention_mask=attention).logits[:, :-1]
            raw = logits.argmax(-1)
            filtered = mask_native_generation_logits(model, tokenizer, logits).argmax(-1)
            raw_acc = ((raw == ids) & mask).sum(1).float() / lengths
            expected = torch.tensor([r["final_convergence"] for _, r in batch], device="cuda")
            record["replay_accuracy_max_error"] = max(
                record["replay_accuracy_max_error"], (raw_acc - expected).abs().max().item()
            )
            record["raw_wrong_tokens"] += int(((raw != ids) & mask).sum())
            record["reserved_wrong_tokens"] += int(((raw >= len(tokenizer)) & mask).sum())
            record["exact_after_filter"] += int((((filtered == ids) | ~mask).all(1)).sum())
        if model.config.sps:
            sample = 0
            row = max((r for r in groups[sample] if r["final_convergence"] == 1), key=lambda r: r["stage_seq_len"])
            memory = torch.tensor([row["embedding"]], device="cuda", dtype=model.dtype)
            length = row["stage_seq_len"]
            _, actual = generate_from_compression(model, tokenizer, memory, max_new_tokens=length, return_generated_ids=True)
            context = memory.clone()
            reference = []
            for _ in range(length):
                logits = ordinary_history_next_logits(model, context)
                token = mask_native_generation_logits(model, tokenizer, logits).argmax(-1)
                reference.append(token.item())
                context = torch.cat((context, model.get_input_embeddings()(token)[:, None]), dim=1)
                if token.item() == tokenizer.eos_token_id:
                    break
            record["greedy"] = dict(
                sample_id=sample,
                full_forward_ids=actual[0].tolist(),
                ordinary_history_ids=reference,
                target_ids=tokens[sample]["input_ids"][:length],
            )
            # Reproduce the training batch shape and the shortened decode shape.
            # Hold weights/inputs fixed when switching BF16 arithmetic to FP32.
            group_ids = list(range(4))
            saved = [next(r for r in groups[i] if r["stage_seq_len"] == length) for i in group_ids]
            batch_memory = torch.tensor([r["embedding"] for r in saved], device="cuda", dtype=model.dtype)
            batch_ids = torch.tensor([tokens[i]["input_ids"][:length] for i in group_ids], device="cuda")
            numerical = {}
            for arithmetic in ("bfloat16", "float32"):
                model.to(dtype=getattr(torch, arithmetic))
                mem = batch_memory.to(model.dtype)
                text = model.get_input_embeddings()(batch_ids)
                complete = torch.cat((mem, text), dim=1)
                b4 = model(inputs_embeds=complete).logits[0, :length]
                b1 = model(inputs_embeds=complete[:1]).logits[0, :length]
                short = model(inputs_embeds=complete[:1, :-1]).logits[0, -1]
                target = int(batch_ids[0, -1])
                other = int(actual[0, -1])
                numerical[arithmetic] = dict(
                    teacher_b4_ids=b4.argmax(-1).tolist(),
                    teacher_b1_ids=b1.argmax(-1).tolist(),
                    shortened_last_id=int(short.argmax()),
                    target_minus_greedy_margin=dict(
                        b4=float(b4[-1, target] - b4[-1, other]),
                        b1=float(b1[-1, target] - b1[-1, other]),
                        shortened=float(short[target] - short[other]),
                    ),
                    max_b4_b1_error=float((b4 - b1).abs().max()),
                    max_full_short_last_error=float((b1[-1] - short).abs().max()),
                )
            record["fixed_quantized_weights_arithmetic_check"] = numerical

        report[exp["name"]] = record
        print(exp["name"], json.dumps(record), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    path = OUTPUT / "generation_audit.json"
    path.write_text(json.dumps(report, indent=2))
    print(path, flush=True)


if __name__ == "__main__":
    validate()

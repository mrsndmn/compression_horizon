"""Certify saved SPS vectors for the original growing-prefix greedy generator."""

import torch
from scripts.sps_fixed_shape import token_digest

from compression_horizon.inference.generation import generate_from_compression, mask_native_generation_logits

CODEC = "sps-growing-greedy-v1"
experiment_configs = [
    dict(name="growing_margin_repair", learning_rate=0.008, max_steps=100, loss_margin=0.55, stop_margin=0.5),
]


def growing_ids(model, tokenizer, stored, length):
    if not model.config.sps or length < 1:
        raise ValueError("Expected a pretrained SPS checkpoint and positive length")
    embed = model.get_input_embeddings()
    bos = torch.tensor([[model.config.bos_token_id]], device=model.device)
    _, generated = generate_from_compression(
        model,
        tokenizer,
        embed(bos),
        max_new_tokens=length,
        return_generated_ids=True,
        predict_embeddings=stored.to(embed.weight),
    )
    return generated[0].tolist()


def certify_growing(model, tokenizer, prediction, target_ids):
    settings = experiment_configs[0]
    length = len(target_ids)
    if not length:
        raise ValueError("Empty target")
    if tokenizer.eos_token_id in target_ids[:-1]:
        raise ValueError("The growing generator stops at interior EOS")
    weight = model.get_input_embeddings().weight
    stored = prediction.detach().to(weight).cpu().clone()
    if growing_ids(model, tokenizer, stored, length) != target_ids:
        # Refine a fresh copy of the vector while retaining the checkpoint's
        # actual forward precision. Only the optimizer master copy uses FP32.
        vector = stored.to(device=model.device, dtype=torch.float32).clone().requires_grad_()
        optimizer = torch.optim.Adam([vector], lr=settings["learning_rate"])
        ids = torch.tensor([[model.config.bos_token_id, *target_ids]], device=model.device)
        target = ids[0, 1:, None]
        for _ in range(settings["max_steps"]):
            logits = (
                model(input_ids=ids, attention_mask=torch.ones_like(ids), predict_embeddings=vector.to(weight))
                .logits[0, :-1]
                .float()
            )
            eligible = mask_native_generation_logits(model, tokenizer, logits.detach())
            strongest = eligible.scatter_(1, target, -torch.inf).argmax(1, keepdim=True)
            margin = logits.gather(1, target) - logits.gather(1, strongest)
            if margin.min().item() >= settings["stop_margin"]:
                break
            loss = (settings["loss_margin"] - margin).clamp_min(0).square().mean()
            (vector.grad,) = torch.autograd.grad(loss, vector)
            optimizer.step()
        stored = vector.detach().to(weight).cpu().clone()
        if growing_ids(model, tokenizer, stored, length) != target_ids:
            raise ValueError("Growing greedy reconstruction differs after refinement")
    return dict(codec=CODEC, prediction=stored, length=length, sha256=token_digest(target_ids))


def decompress_growing(model, tokenizer, packet):
    if packet["codec"] != CODEC:
        raise ValueError("Unknown growing codec")
    ids = growing_ids(model, tokenizer, packet["prediction"], packet["length"])
    if token_digest(ids) != packet["sha256"]:
        raise ValueError("Decoded tokens failed the integrity check")
    return ids

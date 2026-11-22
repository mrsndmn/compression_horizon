"""Lossless acceptance for shared-prediction SPS with a fixed-shape greedy decoder."""

import hashlib
import json

import torch


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()


@torch.inference_mode()
def fixed_shape_ids(model, prediction, length):
    """Decode exactly length IDs, including any interior EOS, starting from BOS."""
    if not model.config.sps or length <= 0:
        raise ValueError("Expected an SPS checkpoint and a positive prefix length")
    model.eval()
    embeddings = model.get_input_embeddings()
    device = embeddings.weight.device
    prediction = prediction.to(embeddings.weight)
    ids = torch.full((1, length + 1), model.config.bos_token_id, dtype=torch.long, device=device)
    # Same shape and all-ones mask as the converged teacher replay. Future BOS
    # placeholders are invisible at the scored row under the causal SPS mask.
    mask = torch.ones_like(ids)
    for pos in range(length):
        logits = model(inputs_embeds=embeddings(ids), attention_mask=mask, predict_embeddings=prediction).logits
        ids[:, pos + 1] = logits[:, pos].argmax(dim=-1)
    return ids[0, 1:].tolist()


def certify(model, prediction, target_ids):
    """Return a packet only after blind reconstruction of the stored vector succeeds."""
    stored = prediction.detach().to(model.get_input_embeddings().weight).cpu().contiguous().clone()
    reconstructed = fixed_shape_ids(model, stored, len(target_ids))
    if reconstructed != target_ids:
        raise ValueError("Fixed-shape greedy reconstruction differs from the target")
    return dict(codec="sps-fixed-shape-v1", prediction=stored, length=len(target_ids), sha256=token_digest(target_ids))


def decompress(model, packet):
    if packet["codec"] != "sps-fixed-shape-v1":
        raise ValueError("Unknown SPS codec")
    result = fixed_shape_ids(model, packet["prediction"], packet["length"])
    if token_digest(result) != packet["sha256"]:
        raise ValueError("Decoded tokens failed the integrity check")
    return result

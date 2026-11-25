"""Compare full and cached causal prefill on the original frozen Qwen checkpoint.

Run as a module with the Thinking Machines batch_invariant_ops package on PYTHONPATH.
The replacement below is an inference probe for unpadded batch-one BF16 causal Qwen.
"""

import argparse
import json
import math
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from scripts.pretrained_sps_experiments import ROOT, default_experiments
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

experiment_configs = [
    dict(length=8192, splits=(64, 65, 8191), document_index=44, decode_steps=32),
]


@triton.jit
def _fixed_attention(
    Q,
    K,
    V,
    OUT,
    MASK,
    mask_q_stride,
    mask_k_stride,
    HAS_MASK: tl.constexpr,
    nq,
    nk,
    sqh,
    sqt,
    skh,
    skt,
    svh,
    svt,
    soh,
    sot,
    GROUP: tl.constexpr,
    SCALE: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    WIDE_SUM: tl.constexpr = False,
):
    head = tl.program_id(1)
    cached = nk - nq
    # Query tiles are anchored at absolute position zero, even across a split.
    tile = cached // BM + tl.program_id(0)
    qpos = tile * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    active = (qpos >= cached) & (qpos < nk)
    q = tl.load(
        Q + head * sqh + (qpos[:, None] - cached) * sqt + d[None, :],
        mask=active[:, None],
        other=0,
    )
    kvhead = head // GROUP
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, D), 0, tl.float32)
    for block in range(0, tl.cdiv((tile + 1) * BM, BN)):
        kpos = block * BN + tl.arange(0, BN)
        k = tl.load(
            K + kvhead * skh + kpos[:, None] * skt + d[None, :],
            mask=kpos[:, None] < nk,
            other=0,
        )
        v = tl.load(
            V + kvhead * svh + kpos[:, None] * svt + d[None, :],
            mask=kpos[:, None] < nk,
            other=0,
        )
        score = tl.dot(q, tl.trans(k)) * SCALE
        visible = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] < nk)
        if HAS_MASK:
            allowed = tl.load(
                MASK + (qpos[:, None] - cached) * mask_q_stride + kpos[None, :] * mask_k_stride,
                mask=active[:, None] & (kpos[None, :] < nk),
                other=0,
            )
            visible = visible & allowed
        score = tl.where(visible, score, -float("inf"))
        newmax = tl.maximum(maximum, tl.max(score, 1))
        if HAS_MASK:
            # A padding/document mask may hide the entire first key tile.
            newmax = tl.where(newmax == -float("inf"), -1.0e30, newmax)
        rescale = tl.exp(maximum - newmax)
        p = tl.exp(score - newmax[:, None])
        acc = acc * rescale[:, None] + tl.dot(p.to(tl.bfloat16), v)
        if WIDE_SUM:
            block_sum = tl.sum(p.to(tl.float64), 1).to(tl.float32)
        else:
            block_sum = tl.sum(p, 1)
        denom = denom * rescale + block_sum
        maximum = newmax
    if HAS_MASK:
        denom = tl.where(denom == 0, 1.0, denom)
    tl.store(
        OUT + head * soh + (qpos[:, None] - cached) * sot + d[None, :],
        acc / denom[:, None],
        mask=active[:, None],
    )


def fixed_causal(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise ValueError("Inference only")
    b, h, nq, d = q.shape
    nk = k.shape[2]
    if (
        b != 1
        or q.dtype != torch.bfloat16
        or not q.is_cuda
        or d != 128
        or k.shape != v.shape
        or k.shape[0] != 1
        or k.shape[-1] != d
        or nk < nq
        or nq < 1
        or h % k.shape[1]
        or (h != k.shape[1] and not enable_gqa)
        or dropout_p
        or any(t.stride(-1) != 1 or t.device != q.device or t.dtype != q.dtype for t in (q, k, v))
    ):
        raise ValueError("Unsupported Qwen attention input")
    if attn_mask is not None:
        if is_causal or attn_mask.dtype != torch.bool or attn_mask.shape != (1, 1, nq, nk):
            raise ValueError("Expected a batch-one boolean attention mask")
        pos = torch.arange(nk, device=q.device)
        if not torch.equal(attn_mask[0, 0], pos[None, :] <= pos[-nq:, None]):
            raise ValueError("Only right-aligned causal attention is supported")
    elif not ((is_causal and nq == nk) or (not is_causal and nq == 1)):
        raise ValueError("Missing causal visibility information")
    return fixed_forward(q, k, v, scale)


def fixed_forward(q, k, v, scale=None, visibility=None, wide_sum=False):
    """Shared launch after the caller has validated its attention contract."""
    _, h, nq, d = q.shape
    nk = k.shape[2]
    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    bm, bn = 32, 64
    grid = (triton.cdiv((nk - nq) % bm + nq, bm), h)
    _fixed_attention[grid](
        q,
        k,
        v,
        o,
        visibility if visibility is not None else q,
        visibility.stride(-2) if visibility is not None else 0,
        visibility.stride(-1) if visibility is not None else 0,
        visibility is not None,
        nq,
        nk,
        *q.stride()[1:3],
        *k.stride()[1:3],
        *v.stride()[1:3],
        *o.stride()[1:3],
        h // k.shape[1],
        scale if scale is not None else d**-0.5,
        d,
        bm,
        bn,
        wide_sum,
        num_warps=4,
        num_stages=2,
    )
    return o


def compare(a, b):
    if a.shape != b.shape:
        raise ValueError("Mismatched positions")
    diffs = flips = 0
    maximum = 0.0
    for x, y in zip(a.split(128), b.split(128), strict=True):
        diffs += int((x != y).sum())
        flips += int((x.argmax(-1) != y.argmax(-1)).sum())
        delta = (x.float() - y.float()).abs().max().item()
        if not math.isfinite(delta):
            raise ValueError("Non-finite output")
        maximum = max(maximum, delta)
    return dict(
        positions=len(a),
        unequal_logits=diffs,
        top1_changes=flips,
        max_abs_delta=maximum,
    )


@torch.inference_mode()
def run(output):

    cfg = experiment_configs[0]
    exp = next(e for e in default_experiments if e["name"] == "qwen_ar")
    location = exp["source_path"]
    doc = json.loads((ROOT / "data/raw.json").read_text())[cfg["document_index"]]
    tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
    ids = tokenizer.encode(doc["text"], add_special_tokens=False, verbose=False)[: cfg["length"]]
    if len(ids) != cfg["length"]:
        raise ValueError("Source text too short")
    tokens = torch.tensor([ids], device="cuda")
    model = (
        AutoModelForCausalLM.from_pretrained(
            location,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    reference = None
    report = dict(
        config=cfg,
        document_sha256=doc["sha256"],
        checkpoint=exp["revision"],
        torch=torch.__version__,
        triton=triton.__version__,
        transformers=version("transformers"),
        gpu=torch.cuda.get_device_name(),
        comparisons=[],
    )
    from batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    already_enabled = is_batch_invariant_mode_enabled()
    if not already_enabled:
        enable_batch_invariant_mode()
    try:
        for mode in ("sdpa", "math", "fixed"):
            context = {
                "math": sdpa_kernel(SDPBackend.MATH),
                "fixed": patch.object(F, "scaled_dot_product_attention", fixed_causal),
            }
            with context.get(mode, nullcontext()):
                captured = []
                handle = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(
                    lambda _m, args: captured.append(args[0][0].detach())
                )
                full = model(input_ids=tokens, use_cache=False).logits[0]
                first_layer_full = captured.pop()
                if reference is None:
                    reference = full
                cross = compare(full, reference)
                print(mode, "full_vs_sdpa", cross, flush=True)
                report.setdefault("full_vs_sdpa", {})[mode] = cross
                for s in cfg["splits"]:
                    first = model(input_ids=tokens[:, :s], use_cache=True)
                    second = model(
                        input_ids=tokens[:, s:],
                        past_key_values=first.past_key_values,
                        use_cache=True,
                    )
                    delta = compare(second.logits[0], full[s:])
                    entry = dict(
                        mode=mode,
                        split=s,
                        suffix=delta,
                        prefix=compare(first.logits[0], full[:s]),
                    )
                    print(entry, flush=True)
                    report["comparisons"].append(entry)
                    entry["first_attention_unequal_elements"] = int((torch.cat(captured) != first_layer_full).sum())
                    captured.clear()
                    del first, second
                handle.remove()
                start = len(ids) - cfg["decode_steps"]
                cache = model(input_ids=tokens[:, :start], use_cache=True, logits_to_keep=1).past_key_values
                decoded = []
                for i in range(start, len(ids)):
                    step = model(
                        input_ids=tokens[:, i : i + 1],
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = step.past_key_values
                    decoded.append(step.logits[0])
                report.setdefault("one_token_steps", {})[mode] = compare(torch.cat(decoded), full[start:])
                print(mode, "one_token_steps", report["one_token_steps"][mode], flush=True)
                del full, cache, step, decoded
    finally:
        if not already_enabled:
            disable_batch_invariant_mode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

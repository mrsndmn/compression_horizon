"""Compare a native checkpoint adapter with the original SPS GPU implementation.

The source checkout is needed only for this validation command, never by jobs.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

from compression_horizon.models.sps_causal_lm import load_sps_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--long-smoke", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_repo / "src"))
    from modeling.models.full_attention_model import Model, ModelConfig
    from modeling.models.sps import SPSConfig, SPSModel

    ckpt = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    is_sps = ckpt["config"]["model"]["_target_"] == "modeling.models.sps.SPSModel"
    cfg = (SPSConfig if is_sps else ModelConfig)(
        **{k: v for k, v in ckpt["model_args"].items() if k in (SPSConfig if is_sps else ModelConfig).__dataclass_fields__}
    )
    original = (SPSModel if is_sps else Model)(cfg).cuda().eval().requires_grad_(False)
    original.load_state_dict(ckpt["model"], strict=True)
    adapter = load_sps_checkpoint(args.checkpoint).cuda()
    # Training uses autocast with fp32 native weights; exercise the same original kernel.
    ids = torch.tensor([[15496, 995, 318, 257, 1332, 13, 383, 2068] * 4], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = original(ids) if is_sps else original(ids, torch.zeros_like(ids))[0]
        actual = adapter(input_ids=ids).logits
    diff = (reference.float() - actual.float()).abs()
    torch.testing.assert_close(actual, reference, atol=0.15, rtol=0.02)
    assert (actual.argmax(-1) == reference.argmax(-1)).all()
    # Compare gradients to the token table, which are also gradients to individual
    # input embeddings when input IDs are unique and exclude the predict token.
    unique_ids = torch.arange(100, 132, device="cuda")[None]
    original.transformer.wte.weight.requires_grad_(True)
    adapter.get_input_embeddings().weight.requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = original(unique_ids) if is_sps else original(unique_ids, torch.zeros_like(unique_ids))[0]
        ref[..., :64].float().square().mean().backward()
        pred = adapter(input_ids=unique_ids).logits
        pred[..., :64].float().square().mean().backward()
    grad_ref = original.transformer.wte.weight.grad[unique_ids].float()
    grad_actual = adapter.get_input_embeddings().weight.grad[unique_ids].float()
    cosine = torch.nn.functional.cosine_similarity(grad_ref.flatten(), grad_actual.flatten(), dim=0).item()
    assert cosine > 0.995, cosine
    report = dict(
        checkpoint=str(args.checkpoint),
        sps=is_sps,
        max_logit_error=diff.max().item(),
        mean_logit_error=diff.mean().item(),
        argmax_equal=True,
        gradient_cosine=cosine,
    )
    del original, ckpt
    adapter.requires_grad_(False)
    for p in adapter.parameters():
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    if args.long_smoke:
        adapter.to(dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        memory = torch.randn(4, 1, adapter.config.hidden_size, device="cuda", requires_grad=True)
        tokens = torch.randint(0, 50000, (4, 4096), device="cuda")
        embeds = adapter.get_input_embeddings()(tokens)
        combined = torch.cat((memory.to(embeds.dtype), embeds), dim=1)
        started = time.monotonic()
        logits = adapter(inputs_embeds=combined).logits
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tokens.reshape(-1))
        loss.backward()
        torch.cuda.synchronize()
        assert memory.grad.isfinite().all() and memory.grad.norm() > 0
        report.update(
            long_smoke_loss=loss.item(),
            long_smoke_seconds=time.monotonic() - started,
            peak_gpu_gb=torch.cuda.max_memory_allocated() / 1024**3,
        )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

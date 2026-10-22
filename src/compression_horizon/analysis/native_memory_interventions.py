"""Causal interventions for native Standard/SPS memory reading; eval only."""

from contextlib import contextmanager

import torch


@contextmanager
def route_knockout(model, layers, route="all"):
    """Block memory-key edges to ordinary/context or predict queries.

    Physical q=0 remains untouched. Context routes are q=2,4,... in SPS;
    prediction routes are q=1,3,... . Standard has only the ordinary route.
    """
    if route not in ("all", "context", "predict"):
        raise ValueError(route)
    if route == "predict" and not model.config.sps:
        raise ValueError("Standard has no separate predict stream")
    handles = []

    def change(module, args, kwargs):
        mask = kwargs["attn_block_mask"].clone()
        selected = slice(1, None)
        if model.config.sps and route != "all":
            selected = slice(2, None, 2) if route == "context" else slice(1, None, 2)
        mask[..., selected, 0] = False
        return args, dict(kwargs, attn_block_mask=mask)

    try:
        for i in layers:
            handles.append(model.transformer.h[i].attn.register_forward_pre_hook(change, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def capture_memory_kv(model, inputs):
    """Capture projected pre-RoPE key/value of the memory at every layer."""
    result = {}
    handles = []

    def capture(i):
        def hook(module, args, output):
            _, k, v = output.chunk(3, dim=-1)
            result[i] = dict(key=k[:, 0].clone(), value=v[:, 0].clone())

        return hook

    try:
        for i, block in enumerate(model.transformer.h):
            handles.append(block.attn.c_attn.register_forward_hook(capture(i)))
        model(inputs_embeds=inputs)
    finally:
        for handle in handles:
            handle.remove()
    return result


@contextmanager
def patch_memory_kv(model, layers, donor, kind):
    """Replace K0 and/or V0 read by other positions, preserving memory evolution.

    q=0 has only its self key in the original causal mask, so its head output
    equals the unmodified V0 exactly. Restoring that value before c_proj avoids
    changing the memory residual trajectory when replacing its outgoing values.
    Donors use the same memory position, hence the same RoPE rotation.
    """
    if kind not in ("key", "value", "both") or model.training:
        raise ValueError("KV patching requires an eval model and key/value/both")
    handles = []
    original_values = {}

    def patch(i):
        def hook(module, args, output):
            q, k, v = output.chunk(3, dim=-1)
            original_values[i] = v[:, 0].clone()
            k, v = k.clone(), v.clone()
            if kind in ("key", "both"):
                k[:, 0] = donor[i]["key"]
            if kind in ("value", "both"):
                v[:, 0] = donor[i]["value"]
            return torch.cat((q, k, v), dim=-1)

        return hook

    def restore_self(i):
        def hook(module, args):
            x = args[0].clone()
            x[:, 0] = original_values[i]
            return (x, *args[1:])

        return hook

    try:
        for i in layers:
            attn = model.transformer.h[i].attn
            handles.append(attn.c_attn.register_forward_hook(patch(i)))
            handles.append(attn.c_proj.register_forward_pre_hook(restore_self(i)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def capture_readout_queries(model, inputs):
    """Unrotated queries for positions predicting each next target (except unused last)."""
    result = {}
    handles = []

    def capture(i):
        def hook(module, args, output):
            q = output.chunk(3, dim=-1)[0]
            readout = q[:, 1::2] if model.config.sps else q
            result[i] = readout[:, :-1].clone()

        return hook

    try:
        for i, block in enumerate(model.transformer.h):
            handles.append(block.attn.c_attn.register_forward_hook(capture(i)))
        model(inputs_embeds=inputs)
    finally:
        for handle in handles:
            handle.remove()
    return result

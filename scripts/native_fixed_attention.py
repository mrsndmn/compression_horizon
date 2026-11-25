"""Fixed-order masked forward and analytical attention gradient for frozen native pilots."""

import torch
from scripts.probe_qwen_fixed_attention import fixed_forward


class _NativeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, visible, wide_sum):
        ctx.save_for_backward(q, k, v, visible)
        return fixed_forward(q, k, v, visibility=visible, wide_sum=wide_sum)

    @staticmethod
    def backward(ctx, grad_out):
        # Differentiate the underlying scaled softmax attention in FP32.
        # As with ordinary mixed-precision attention, gradients pass through casts.
        q, k, v, visible = ctx.saved_tensors
        qf, kf, vf = q.float(), k.float(), v.float()
        scale = q.shape[-1] ** -0.5
        logits = (qf @ kf.transpose(-2, -1)) * scale
        allowed = visible.tril(diagonal=k.shape[2] - q.shape[2])
        logits = logits.masked_fill(~allowed, -torch.inf)
        prob = logits.softmax(-1).nan_to_num(0)
        go = grad_out.float()
        dprob = go @ vf.transpose(-2, -1)
        ds = prob * (dprob - (prob * dprob).sum(-1, keepdim=True)) * scale
        return (
            (ds @ kf).to(q.dtype),
            (ds.transpose(-2, -1) @ qf).to(k.dtype),
            (prob.transpose(-2, -1) @ go).to(v.dtype),
            None,
            None,
        )


def native_fixed_attention(q, k, v, attn_mask=None, dropout_p=0.0, *, wide_sum=False):
    b, h, n, d = q.shape
    if (
        b != 1
        or d != 64
        or k.shape != v.shape
        or k.shape[:2] != (b, h)
        or k.shape[-1] != d
        or k.size(2) < n
        or not q.is_cuda
        or any(t.device != q.device or t.dtype != torch.bfloat16 or t.stride(-1) != 1 for t in (q, k, v))
        or attn_mask is None
        or attn_mask.shape != (1, 1, n, k.size(2))
        or attn_mask.dtype != torch.bool
        or attn_mask.device != q.device
        or dropout_p
    ):
        raise ValueError("Expected native batch-one BF16 attention with explicit visibility and zero dropout")
    return _NativeAttention.apply(q, k, v, attn_mask, wide_sum)

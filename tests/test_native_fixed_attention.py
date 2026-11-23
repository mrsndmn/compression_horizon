import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False), reason="Requires BF16 CUDA"
)
@pytest.mark.parametrize("sps", [False, True])
def test_native_fixed_forward_backward_and_growing_prefix(sps):
    pytest.importorskip("triton")
    from scripts.native_fixed_attention import native_fixed_attention

    from compression_horizon.models.sps_causal_lm import attention_visibility

    generator = torch.Generator(device="cuda").manual_seed(27)
    valid = torch.ones((1, 71), device="cuda", dtype=torch.bool)
    valid[:, :2] = False
    visible = attention_visibility(valid, sps=sps)
    n = visible.shape[-1]
    q, k, v = [
        torch.randn(1, 2, n, 64, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_() for _ in range(3)
    ]
    actual = native_fixed_attention(q, k, v, visible)
    with torch.no_grad():
        cut = 66 if sps else 33
        short = native_fixed_attention(q[:, :, :cut], k[:, :, :cut], v[:, :, :cut], visible[:, :, :cut, :cut])
        assert torch.equal(actual[:, :, :cut], short)
    cotangent = torch.randn(actual.shape, device="cuda", dtype=actual.dtype, generator=generator)
    actual_grad = torch.autograd.grad(actual, (q, k, v), cotangent)
    Q, K, V = [t.detach().double().requires_grad_() for t in (q, k, v)]
    logits = (Q @ K.transpose(-2, -1) / 8).masked_fill(~visible, -torch.inf)
    logits = torch.where(visible.any(-1, keepdim=True), logits, 0.0)
    ref = logits.softmax(-1).masked_fill(~visible, 0.0) @ V
    ref_grad = torch.autograd.grad(ref, (Q, K, V), cotangent.double())
    torch.testing.assert_close(actual.double(), ref, atol=0.02, rtol=0.02)
    for g, e in zip(actual_grad, ref_grad, strict=True):
        torch.testing.assert_close(g.double(), e, atol=0.01, rtol=0.015)

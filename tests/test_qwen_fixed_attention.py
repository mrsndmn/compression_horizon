import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False),
    reason="Requires a BF16-capable CUDA GPU",
)
@torch.inference_mode()
def test_fixed_attention_causal_gqa_and_cache():
    pytest.importorskip("triton")
    from scripts.probe_qwen_fixed_attention import fixed_causal

    generator = torch.Generator(device="cuda").manual_seed(17)

    def rand(heads):
        return torch.randn(1, heads, 139, 128, device="cuda", dtype=torch.bfloat16, generator=generator)

    q, k, v = rand(4), rand(2), rand(2)
    pos = torch.arange(139, device="cuda")
    visible = pos[:, None] >= pos[None, :]
    scores = q.double() @ k.repeat_interleave(2, dim=1).double().transpose(-2, -1) / 128**0.5
    expected = scores.masked_fill(~visible, -torch.inf).softmax(-1) @ v.repeat_interleave(2, dim=1).double()
    actual = fixed_causal(q, k, v, is_causal=True, enable_gqa=True)
    torch.testing.assert_close(actual.double(), expected, atol=0.016, rtol=0.015)
    for split in (1, 32, 63, 65, 138):
        prefix = fixed_causal(q[:, :, :split], k[:, :, :split], v[:, :, :split], is_causal=True, enable_gqa=True)
        suffix = fixed_causal(q[:, :, split:], k, v, attn_mask=visible[None, None, split:], enable_gqa=True)
        assert torch.equal(prefix, actual[:, :, :split])
        assert torch.equal(suffix, actual[:, :, split:])
    with pytest.raises(ValueError, match="causal"):
        fixed_causal(q, k, v, attn_mask=(~visible)[None, None], enable_gqa=True)

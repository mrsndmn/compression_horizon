"""Serialization-fidelity tests for progressive-cramming embedding artifacts.

The progressive trainer persists each converged compression embedding **twice** with
two different mechanisms (see ``ProgressiveCrammingTrainer``):

* Dataset path (``progressive_prefixes/``) -- the canonical artifact read by
  ``_load_artifact_init`` and every downstream eval/figure script.
  Save side (``_build_sample_row`` -> ``_save_artifacts``)::

      embedding_as_list = comp_tokens_cpu[j].to(torch.float32).numpy().tolist()
      Dataset.from_list(rows).save_to_disk(path)

  Load side (``_load_artifact_init``)::

      emb = torch.tensor(row["embedding"], dtype=torch.float32).reshape(n, h)

  i.e. ``float32 -> Python float (float64) list -> Arrow float64 -> float32``.

* ``torch.save`` path (``embeddings/*.pt`` stage dumps via ``_dump_stage_embedding``
  and the top-level ``compression_embeddings.pt``) -- a direct tensor pickle.

This module pins down the question: are the three representations -- the in-memory
float32 tensor, the HF-dataset round-trip, and the ``torch.save`` round-trip -- bit
identical? They should be: ``float32 -> float64`` widening is lossless and the
``float64 -> float32`` narrowing returns the original bits exactly, while
``torch.save`` is a verbatim pickle.
"""

import torch
from datasets import Dataset


def _make_embedding(num_compression_tokens: int = 2, hidden_size: int = 16, seed: int = 0) -> torch.Tensor:
    """A ``[num_compression_tokens, hidden]`` float32 embedding like ``materialize()`` produces."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(num_compression_tokens, hidden_size, generator=gen, dtype=torch.float32)


def _dataset_roundtrip(emb: torch.Tensor, tmp_path) -> torch.Tensor:
    """Mirror the trainer's HF-dataset save/load path exactly."""
    num_compression_tokens, hidden_size = emb.shape
    # --- save side: _build_sample_row + _save_artifacts ---
    embedding_as_list = emb.to(torch.float32).numpy().tolist()
    row = {
        "embedding": embedding_as_list,
        "num_compression_tokens": int(num_compression_tokens),
        "hidden_size": int(hidden_size),
    }
    save_path = str(tmp_path / "progressive_prefixes")
    Dataset.from_list([row]).save_to_disk(save_path)
    # --- load side: _load_artifact_init ---
    ds = Dataset.load_from_disk(save_path)
    loaded = ds[0]
    emb_flat = torch.tensor(loaded["embedding"], dtype=torch.float32)
    return emb_flat.reshape(loaded["num_compression_tokens"], loaded["hidden_size"])


def _torch_save_roundtrip(emb: torch.Tensor, tmp_path) -> torch.Tensor:
    """Mirror the trainer's ``torch.save`` artifact path (kept in float32)."""
    pt_path = str(tmp_path / "compression_embeddings.pt")
    torch.save(emb, pt_path)
    return torch.load(pt_path, weights_only=True)


def test_dataset_roundtrip_is_bit_exact_float32(tmp_path):
    """HF-dataset save/load returns the in-memory float32 tensor bit-for-bit."""
    emb = _make_embedding(seed=1)
    loaded = _dataset_roundtrip(emb, tmp_path)

    assert loaded.dtype == torch.float32
    assert loaded.shape == emb.shape
    assert torch.equal(loaded, emb), "dataset round-trip diverged from the in-memory float32 tensor"


def test_torch_save_roundtrip_is_bit_exact_float32(tmp_path):
    """``torch.save``/``torch.load`` returns the in-memory float32 tensor bit-for-bit."""
    emb = _make_embedding(seed=2)
    loaded = _torch_save_roundtrip(emb, tmp_path)

    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, emb), "torch.save round-trip diverged from the in-memory float32 tensor"


def test_dataset_and_torch_save_agree(tmp_path):
    """The two persistence paths produce identical tensors -> the dataset artifact and the .pt dump match."""
    emb = _make_embedding(seed=3)
    via_dataset = _dataset_roundtrip(emb, tmp_path)
    via_torch = _torch_save_roundtrip(emb, tmp_path)

    assert torch.equal(via_dataset, via_torch)
    assert torch.equal(via_dataset, emb)
    assert torch.equal(via_torch, emb)


def test_dataset_roundtrip_exact_on_adversarial_values(tmp_path):
    """Tricky float32 values (subnormals, tiny/huge magnitudes, exact decimals) still round-trip exactly.

    These probe whether the float32->float64->float32 detour ever loses a bit. It never
    should: every finite float32 is exactly representable in float64, and casting that
    same value back to float32 is the identity.
    """
    values = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,  # not exactly representable in binary; float32 value must be preserved
            1.0 / 3.0,
            torch.finfo(torch.float32).eps,
            torch.finfo(torch.float32).tiny,  # smallest normal
            torch.finfo(torch.float32).tiny / 2,  # a subnormal
            torch.finfo(torch.float32).max,
            -torch.finfo(torch.float32).max,
            3.1415927,
            1e-30,
            1e30,
        ],
        dtype=torch.float32,
    ).reshape(2, 7)

    loaded = _dataset_roundtrip(values, tmp_path)
    assert torch.equal(loaded, values)
    # And the raw int32 bit patterns match, the strictest possible check.
    assert torch.equal(loaded.view(torch.int32), values.view(torch.int32))


def test_dataset_path_vs_stage_dump_path(tmp_path):
    """Compare the two artifacts the trainer writes for the SAME embedding.

    The learnable compression embedding is a **float32** ``nn.Parameter`` (every init in
    ``embedding_init.create_compression_embedding`` is float32; AdamW optimizes it in
    float32). The bf16 *model* only casts it transiently inside its forward -- the stored
    parameter, and therefore ``parametrization.materialize()``, stays float32. So:

    * Dataset path (``progressive_prefixes/``): keeps the **full float32** value.
    * Stage-dump path (``embeddings/*.pt`` via ``_dump_stage_embedding``): stores
      ``comp_tokens.to(torch.bfloat16)`` -- the float32 value **truncated to bf16** (lossy).

    Hence the two artifacts are NOT bit-equal: the dump is exactly the dataset value rounded
    to bf16 (max relative error ~ the bf16 step 2**-8). Re-widening the dump bf16 -> float32
    is lossless but does NOT recover the discarded mantissa bits.
    """
    emb_f32 = _make_embedding(num_compression_tokens=1, hidden_size=576, seed=7)  # what materialize() returns

    # --- Dataset path: keeps full float32 ---
    via_dataset = _dataset_roundtrip(emb_f32, tmp_path)
    assert via_dataset.dtype == torch.float32
    assert torch.equal(via_dataset, emb_f32)  # lossless

    # --- Stage-dump path: _dump_stage_embedding writes bf16 ---
    dump_path = str(tmp_path / "embedding_sample_0_stage_0.pt")
    torch.save(emb_f32.to(torch.bfloat16).detach().cpu(), dump_path)
    via_dump = torch.load(dump_path, weights_only=True)
    assert via_dump.dtype == torch.bfloat16

    # The two artifacts differ: the dump is the lossy (bf16-rounded) version of the dataset.
    assert not torch.equal(via_dataset, via_dump.to(torch.float32))
    # ...and the difference is exactly bf16 rounding: dataset rounded to bf16 == the dump.
    assert torch.equal(via_dataset.to(torch.bfloat16), via_dump)
    # Re-widening the dump is lossless but recovers only the rounded value, not the original.
    assert torch.equal(via_dump, via_dump.to(torch.float32).to(torch.bfloat16))
    assert (via_dataset - via_dump.to(torch.float32)).abs().max().item() > 0.0


def test_bf16_to_float32_is_always_lossless(tmp_path):
    """bf16 -> float32 widening is exact for every value (bf16 == the top 16 bits of a float32).

    This is why the dataset path's ``.to(torch.float32)`` never loses anything even if the
    source were bf16, and why a bf16 dump re-widens exactly (only the original fp32 mantissa,
    already discarded at cast time, is unrecoverable).
    """
    # Exhaustive-ish sweep over bf16: all 256 exponent/highbyte combos x representative mantissas.
    hi = torch.arange(0, 256, dtype=torch.int32) << 8
    mant = torch.tensor([0, 1, 0x3F, 0x40, 0x7F], dtype=torch.int32)
    bits = (hi.unsqueeze(1) | mant.unsqueeze(0)).reshape(-1).to(torch.int16)
    bf16 = bits.view(torch.bfloat16)
    finite = torch.isfinite(bf16.to(torch.float32))  # skip inf/nan (equality is ill-defined)
    bf16 = bf16[finite]

    widened = bf16.to(torch.float32)
    # Widening then re-narrowing returns the identical bf16 bits -> the widening lost nothing.
    assert torch.equal(widened.to(torch.bfloat16), bf16)
    # And the float32 bit pattern is exactly the bf16 bits zero-padded in the low 16 bits.
    expected_f32_bits = bf16.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF) << 16
    assert torch.equal(widened.view(torch.int32), expected_f32_bits)


def test_bf16_dataset_path_loses_precision_vs_torch_save(tmp_path):
    """Contrast: the *stage-dump* path casts to bf16, so it is NOT bit-equal to the float32 source.

    ``_dump_stage_embedding`` writes ``comp_tokens.to(torch.bfloat16)`` via ``torch.save``.
    bf16 has 8 mantissa bits, so this is a genuine lossy step -- documented here so the
    contrast with the (lossless) float32 dataset path is explicit rather than assumed.
    The bf16 .pt and a bf16 dataset round-trip still agree *with each other*.
    """
    emb = _make_embedding(seed=4)
    emb_bf16 = emb.to(torch.bfloat16)

    # bf16 differs from the float32 source (lossy cast).
    assert not torch.equal(emb_bf16.to(torch.float32), emb)

    # torch.save of the bf16 tensor is exact for bf16.
    pt_path = str(tmp_path / "embedding_stage.pt")
    torch.save(emb_bf16, pt_path)
    loaded_pt = torch.load(pt_path, weights_only=True)
    assert loaded_pt.dtype == torch.bfloat16
    assert torch.equal(loaded_pt, emb_bf16)

    # A bf16 dataset round-trip (cast to float32 list, back to bf16) matches the .pt bf16 tensor.
    row = {"embedding": emb_bf16.to(torch.float32).numpy().tolist()}
    save_path = str(tmp_path / "bf16_prefixes")
    Dataset.from_list([row]).save_to_disk(save_path)
    loaded_ds = torch.tensor(Dataset.load_from_disk(save_path)[0]["embedding"], dtype=torch.float32).to(torch.bfloat16)
    assert torch.equal(loaded_ds, emb_bf16)

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config

from compression_horizon.models.pretrained_sps import PretrainedSPSCausalLM, load_pretrained_sps
from compression_horizon.models.sps_vendor.pretrained import PretrainedSPS


def tiny(family, sps=True):
    config = (LlamaConfig if family == "llama" else Qwen3Config)(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        bos_token_id=29,
        eos_token_id=29,
        pad_token_id=31,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    return PretrainedSPSCausalLM(AutoModelForCausalLM.from_config(config), sps=sps).eval().requires_grad_(False)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_reference_parity_positions_and_gradients(family):
    torch.manual_seed(42)
    model = tiny(family)
    reference = PretrainedSPS(model.backbone).eval()
    reference.prediction = torch.nn.Parameter(model.prediction.detach().clone())
    ids = torch.tensor([[29, 2, 3, 4]])
    z = model.prediction.detach().view(1, 1, -1).clone().requires_grad_(True)
    actual = model(input_ids=ids, predict_embeddings=z).logits
    expected = reference(ids).logits
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    a = torch.autograd.grad(actual.square().sum(), z)[0]
    b = torch.autograd.grad(expected.square().sum(), reference.prediction)[0]
    torch.testing.assert_close(a.flatten(), b.flatten(), atol=1e-7, rtol=1e-5)
    assert a.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_context_isolation_causality_and_independent_prediction(family):
    torch.manual_seed(12)
    model = tiny(family)
    ids = torch.tensor([[29, 2, 3, 4]])
    x = model.get_input_embeddings()(ids)
    z = torch.randn(1, 1, 16)
    outputs = []
    hook = model.backbone.model.layers[-1].register_forward_pre_hook(
        lambda _m, _a, kw: outputs.append(kw.get("hidden_states", _a[0] if _a else None)[:, ::2].detach().clone()),
        with_kwargs=True,
    )
    try:
        actual = model(inputs_embeds=x, predict_embeddings=z).logits
        changed = model(inputs_embeds=x, predict_embeddings=z + 3).logits
    finally:
        hook.remove()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert not torch.allclose(actual, changed)
    future = x.clone()
    future[:, 2:] += 10
    torch.testing.assert_close(actual[:, :2], model(inputs_embeds=future, predict_embeddings=z).logits[:, :2])
    for length in range(1, ids.shape[1] + 1):
        current = torch.cat((x[:, :length], z), 1)
        hidden = model.backbone.model(
            inputs_embeds=current, use_cache=False, position_ids=torch.arange(length + 1)[None]
        ).last_hidden_state[:, -1]
        expected = model.backbone.lm_head(hidden)
        torch.testing.assert_close(actual[:, length - 1], expected, atol=1e-6, rtol=1e-5)
    padded = torch.cat((x, torch.randn(1, 2, 16)), 1)
    result = model(inputs_embeds=padded, attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0]]), predict_embeddings=z)
    torch.testing.assert_close(actual, result.logits[:, :4])


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_native_ar_and_strict_checkpoint_loader(tmp_path, family):
    torch.manual_seed(14)
    model = tiny(family, sps=False)
    ids = torch.tensor([[29, 2, 3]])
    torch.testing.assert_close(model(input_ids=ids).logits, model.backbone(input_ids=ids).logits)
    model = tiny(family)
    model.config.save_pretrained(tmp_path)
    checkpoint = tmp_path / "final.pt"
    saved = dict(model=model.state_dict(), source="test/source", revision="pinned")
    torch.save(saved, checkpoint)
    loaded = load_pretrained_sps(
        "test/source", "pinned", source_path=tmp_path, checkpoint=checkpoint, torch_dtype=torch.float32
    )
    torch.testing.assert_close(loaded(input_ids=ids).logits, model(input_ids=ids).logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match="source/revision"):
        load_pretrained_sps("wrong/source", "pinned", source_path=tmp_path, checkpoint=checkpoint)
    del saved["model"]["prediction"]
    torch.save(saved, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_pretrained_sps("test/source", "pinned", source_path=tmp_path, checkpoint=checkpoint)


@pytest.mark.parametrize("sps", [False, True])
def test_progressive_worker_trainer_save_offset_and_freeze(tmp_path, monkeypatch, sps):
    from datasets import load_from_disk
    from scripts.pretrained_sps_worker import PretrainedPredictTrainer, TokenOnlyTrainer
    from tests.test_sps_generation_semantics import SmallTokenizer
    from tests.trainer_helpers import _collate_batch, _make_args

    monkeypatch.setattr("compression_horizon.train.trainers.base.get_device", lambda: torch.device("cpu"))
    model = tiny("qwen3", sps=sps)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    args = _make_args(
        output_dir=str(tmp_path),
        dtype="float32",
        progressive_train=True,
        num_alignment_layers=0,
        progressive_max_stages=1,
        offset_dataset_items=6,
        token_objective="squared_margin",
        loss_margin=0.05,
        convergence_margin=0.05,
    )
    trainer = (PretrainedPredictTrainer if sps else TokenOnlyTrainer)(
        model=model,
        processing_class=SmallTokenizer(),
        args=args,
        train_dataset=[dict(input_ids=torch.tensor([1, 2, 3]), attention_mask=torch.ones(3, dtype=torch.long))],
        data_collator=_collate_batch,
        **(dict(bos_token_id=29, predict_init="native") if sps else {}),
    )
    row = load_from_disk(trainer.train())[0]
    assert row["sample_id"] == 6
    if sps:
        assert row["embedding_role"] == "shared_predict"
        torch.testing.assert_close(torch.tensor(row["initialization_embedding"]), original["prediction"][None])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_registry_and_launcher_dry_are_side_effect_free(monkeypatch, capsys, tmp_path):
    from scripts import run_pretrained_sps_cramming as launcher
    from scripts.pretrained_sps_experiments import COHORT, SHARD_SIZE, TRAINING_ARGS, experiment_configs

    assert len(experiment_configs) == 6 and COHORT // SHARD_SIZE == 8
    assert TRAINING_ARGS["per_device_train_batch_size"] == 1
    assert TRAINING_ARGS["max_optimization_steps_per_sample"] == 10000
    monkeypatch.setattr(launcher, "ROOT", tmp_path / "absent")
    monkeypatch.setattr("sys.argv", ["launcher", "--phase", "dry", "--kind", "runs"])
    launcher.main()
    assert len(capsys.readouterr().out.splitlines()) == 48
    assert not (tmp_path / "absent").exists()


def test_submission_is_idempotent_and_preserves_full_uuid(monkeypatch, tmp_path):
    from scripts import run_pretrained_sps_cramming as launcher

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "stage_code", lambda *_: None)

    class Client:
        calls = 0

        def run_job(self, *, payload):
            self.calls += 1
            return {"job_name": "lm-mpi-job-12345678-1234-1234-1234-123456789abc"}

    client = Client()
    active = set()
    job = launcher.submit_one("preflight", "qwen", 0, client, {"region": "SR004"}, active)
    assert launcher.submit_one("preflight", "qwen", 0, client, {"region": "SR004"}, active) == job
    assert client.calls == 1 and launcher.JOB_ID.fullmatch(job)
    (launcher.output_path("preflight", "qwen") / "completed").touch()
    assert launcher.submit_one("preflight", "qwen", 0, client, {"region": "SR004"}, active) is None
    assert client.calls == 1


def test_main_submission_rejects_missing_preflight_before_api(monkeypatch, tmp_path):
    from scripts import run_pretrained_sps_cramming as launcher

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *_a, **_k: b"")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/completed").touch()
    with pytest.raises(RuntimeError, match="preflights required"):
        launcher.submit("advance")


def test_uncertain_submission_is_not_retried(monkeypatch, tmp_path):
    from scripts import run_pretrained_sps_cramming as launcher

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    output = launcher.output_path("preflight", "qwen")
    output.mkdir(parents=True)
    (output / "submission-intent.json").write_text("{}")
    with pytest.raises(RuntimeError, match="uncertain"):
        launcher.submit_one("preflight", "qwen", 0, None, {}, set())

import pytest
import torch
from tests.test_sps_causal_lm import tiny_args
from tests.test_sps_generation_semantics import SmallTokenizer

from compression_horizon.inference.generation import generate_from_compression, mask_native_generation_logits
from compression_horizon.models.sps_causal_lm import SPSCausalLM


def test_predict_override_identity_gradients_and_context_independence():
    torch.manual_seed(17)
    model = SPSCausalLM(tiny_args(), sps=True).eval().requires_grad_(False)
    ids = torch.tensor([[29, 1, 2, 3]])
    context = model.get_input_embeddings()(ids)
    native = model.get_input_embeddings().weight[30][None, None].detach().clone().requires_grad_(True)
    default = model(inputs_embeds=context).logits
    output = model(inputs_embeds=context, predict_embeddings=native).logits
    torch.testing.assert_close(default, output)
    output[:, :-1, 0].sum().backward()
    assert native.grad.norm() > 0 and all(p.grad is None for p in model.parameters())
    captures = []
    handles = [
        b.register_forward_pre_hook(lambda module, args: captures.append(args[0][:, ::2].detach().clone()))
        for b in model.transformer.h
    ]
    try:
        model(inputs_embeds=context, predict_embeddings=native)
        first = captures[:]
        captures.clear()
        changed = model(inputs_embeds=context, predict_embeddings=native + torch.randn_like(native)).logits
        assert not torch.allclose(changed, output)
        for a, b in zip(first, captures):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        for h in handles:
            h.remove()


def test_predict_generation_uses_ordinary_history_and_current_trained_predict():
    torch.manual_seed(81)
    model = SPSCausalLM(tiny_args(), sps=True).eval().requires_grad_(False)
    tokenizer = SmallTokenizer()
    bos = model.get_input_embeddings()(torch.tensor([[29]]))
    predict = torch.randn(1, 1, 16)
    _, actual = generate_from_compression(
        model, tokenizer, bos, max_new_tokens=4, return_generated_ids=True, predict_embeddings=predict
    )
    history = torch.empty(1, 0, dtype=torch.long)
    for _ in range(4):
        context = torch.cat((bos, model.get_input_embeddings()(history)), 1)
        x = torch.cat((context, predict), 1)
        n = context.shape[1]
        pos = torch.cat((torch.arange(n), torch.tensor([n - 1])))
        freqs = model.freqs_cis[pos][None]
        mask = torch.ones(n + 1, n + 1, dtype=torch.bool).tril()[None, None]
        for block in model.transformer.h:
            x, _ = block(x, freqs, attn_block_mask=mask)
        logits = model.lm_head(model.transformer.output_norm(x))[:, -1]
        next_id = mask_native_generation_logits(model, tokenizer, logits).argmax(-1)
        history = torch.cat((history, next_id[:, None]), 1)
    assert torch.equal(actual, history)


@pytest.mark.parametrize("sps,shape", [(False, (1, 1, 16)), (True, (1, 2, 16)), (True, (2, 1, 16))])
def test_invalid_predict_override_rejected(sps, shape):
    model = SPSCausalLM(tiny_args(), sps=sps)
    with pytest.raises(ValueError, match="predict_embeddings"):
        model(input_ids=torch.tensor([[1, 2]]), predict_embeddings=torch.randn(shape))


def test_progressive_predict_trainer_saves_predict_role_and_keeps_weights_fixed(tmp_path, monkeypatch):
    from datasets import load_from_disk
    from scripts.analyze_sps_predict_embedding import PredictEmbeddingTrainer
    from tests.trainer_helpers import _collate_batch, _make_args

    monkeypatch.setattr("compression_horizon.train.trainers.base.get_device", lambda: torch.device("cpu"))
    torch.manual_seed(34)
    model = SPSCausalLM(tiny_args(), sps=True).eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    args = _make_args(
        output_dir=str(tmp_path),
        progressive_train=True,
        dtype="float32",
        progressive_min_seq_len=2,
        progressive_max_stages=1,
        max_optimization_steps_per_sample=2,
        max_optimization_steps_per_token=2,
    )
    trainer = PredictEmbeddingTrainer(
        model=model,
        processing_class=SmallTokenizer(),
        args=args,
        train_dataset=[dict(input_ids=torch.tensor([1, 2, 3]), attention_mask=torch.ones(3, dtype=torch.long))],
        data_collator=_collate_batch,
        bos_token_id=29,
        predict_init="native",
    )
    path = trainer.train()
    assert path == str(tmp_path / "predict_prefixes")
    row = load_from_disk(path)[0]
    assert row["embedding_role"] == "shared_predict" and row["fixed_bos_token_id"] == 29
    assert row["num_input_tokens"] == 2 and row["information_gain_reference"] == "fixed_bos_native_predict"
    torch.testing.assert_close(torch.tensor(row["initialization_embedding"]), original["transformer.wte.weight"][30][None])
    assert not torch.allclose(torch.tensor(row["embedding"]), torch.tensor(row["initialization_embedding"]))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_predict_registry_preserves_batch_data_and_parameter_budget():
    from scripts.analyze_sps_predict_embedding import experiment_configs
    from scripts.jobs.run_jobs_sps import TRAINING_ARGS

    assert len(experiment_configs) == 3
    for exp in experiment_configs:
        settings = dict(TRAINING_ARGS, **exp["training_args"])
        assert settings["per_device_train_batch_size"] == 1 and settings["number_of_mem_tokens"] == 1
        assert settings["learning_rate"] == 0.3 and settings["limit_dataset_items"] == 48
    assert {(e["init"], e["training_args"].get("token_objective", "cross_entropy")) for e in experiment_configs} == {
        ("native", "cross_entropy"),
        ("native", "squared_margin"),
        ("random", "squared_margin"),
    }


def test_diagnostic_launcher_preserves_explicit_experiment_argument(monkeypatch, capsys):
    import json

    from scripts import analyze_sps_cramming as jobs

    monkeypatch.setattr(jobs.subprocess, "check_output", lambda args, **kwargs: "a" * 40 if "rev-parse" in args else b"")
    jobs.submit(
        dry=True, entrypoint="scripts/analyze_sps_predict_embedding.py", probe_args=("--experiment", "predict_ce_native")
    )
    payload = json.loads(capsys.readouterr().out)
    assert "--phase probe --experiment predict_ce_native" in payload["script"]

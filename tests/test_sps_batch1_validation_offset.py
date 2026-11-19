import json

import pytest
from datasets import Dataset
from scripts.validation import validate_sps_batch1_run as validator


@pytest.mark.parametrize("offset", [None, 2])
def test_memory_replay_uses_shard_target_and_reports_global_id(tmp_path, monkeypatch, offset):
    cache = tmp_path / "cache"
    Dataset.from_dict({"input_ids": [[1, 2], [3, 4], [5, 6]]}).save_to_disk(str(cache))
    Dataset.from_list(
        [
            dict(
                sample_id=0,
                stage_index=0,
                num_input_tokens=1,
                embedding=[[0.0]],
                model_checkpoint="checkpoint.pt",
                final_convergence=1.0,
                convergence_threshold=1.0,
            )
        ]
    ).save_to_disk(str(tmp_path / "progressive_prefixes"))
    settings = dict(
        per_device_train_batch_size=1,
        limit_dataset_items=1,
        dtype="bfloat16",
        dataset_name="test",
        max_sequence_length=2,
        no_bos_token=True,
        tokenized_dataset_path=str(cache),
    )
    if offset is not None:
        settings["offset_dataset_items"] = offset
    (tmp_path / "design.json").write_text(json.dumps(dict(training_args=settings, commit="test")))

    class Model:
        def cuda(self):
            return self

    monkeypatch.setattr(validator, "load_sps_checkpoint", lambda *a, **k: Model())
    monkeypatch.setattr(validator, "load_sps_tokenizer", lambda: None)

    def check_sample(model, tokenizer, ids, embedding):
        return dict(
            target_ids=ids[0].tolist(),
            greedy_exact=True,
            greedy_accuracy=1.0,
            teacher_replay_correct_tokens=1,
        )

    monkeypatch.setattr(validator, "check_sample", check_sample)
    validator.validate_run(tmp_path)
    report = json.loads((tmp_path / "single_sample_validation.json").read_text())
    assert report["samples"][0]["sample_id"] == (offset or 0)
    assert report["samples"][0]["target_ids"] == ([1] if offset is None else [5])

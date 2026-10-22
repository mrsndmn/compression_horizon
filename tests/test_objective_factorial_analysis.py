import pytest
from scripts.analyze_sps_objective_factorial import leading_correct, paired_interval, validate_samples


def test_greedy_prefix_stops_at_first_error_and_counts_missing_tokens():
    assert leading_correct([1, 2, 3, 4], [1, 9, 3, 4]) == 1
    assert leading_correct([1, 2, 3], [1, 2]) == 2
    assert leading_correct([1, 2], [1, 2]) == 2
    assert leading_correct([1], []) == 0


def test_text_paired_constant_contrast_has_exact_interval():
    assert paired_interval([3, 3, 3]) == dict(mean=3.0, ci95=[3.0, 3.0])
    assert paired_interval([-1, -1, -1])["mean"] == -1


def test_validation_rejects_reported_greedy_success_after_wrong_token():
    sample = dict(
        sample_id=0,
        target_ids=[1, 2],
        generated_ids=[1, 9],
        horizon=2,
        greedy_exact=True,
        greedy_accuracy=1,
        teacher_replay_accuracy=1,
    )
    report = dict(samples=[sample], n_samples=1, mean_horizon=2, greedy_exact_samples=1)
    with pytest.raises(ValueError, match="Greedy metrics"):
        validate_samples(report, 1)
    sample.update(greedy_exact=False, greedy_accuracy=0.5)
    report["greedy_exact_samples"] = 0
    assert validate_samples(report, 1)[0]["greedy_initial_prefix"] == 1

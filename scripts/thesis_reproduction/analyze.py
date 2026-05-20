"""Compare a saved compressed_prefixes Dataset against paper-expected values."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import Dataset


def _load_expected(experiment_name: str) -> dict:
    """Load the paper-expected entry for `experiment_name` from expected.json."""
    expected_path = Path(__file__).parent / "expected.json"
    with open(expected_path) as f:
        all_expected = json.load(f)
    if experiment_name not in all_expected:
        raise KeyError(f"Experiment {experiment_name!r} not found in {expected_path}")
    return all_expected[experiment_name]


def _load_dataset(output_dir: str, trainer_type: str) -> Dataset:
    """Load the saved compressed_prefixes / progressive_prefixes dataset."""
    subdir = "progressive_prefixes" if trainer_type == "progressive" else "compressed_prefixes"
    dataset_path = os.path.join(output_dir, subdir)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Did the training finish?")
    return Dataset.load_from_disk(dataset_path)


def _zscore(value: float, mean: float, std: float) -> float:
    """Normalized distance from `value` to `mean` in units of `std` (or NaN if std==0)."""
    return abs(value - mean) / std if std > 0 else float("nan")


def _verdict(z: float) -> str:
    """Map z-score distance to a 3-tier success indicator."""
    if np.isnan(z):
        return "n/a"
    if z <= 2.0:
        return "OK"
    if z <= 3.0:
        return "WARN"
    return "FAIL"


def _print_row(metric: str, ours: str, paper: str, delta: str, verdict: str) -> None:
    """Print a single comparison row in the OUR | PAPER | DELTA | VERDICT layout."""
    print(f"  {metric:<32}  {ours:>20}   {paper:>20}   {delta:>10}   {verdict}")


def _print_header() -> None:
    """Print the comparison-table header row."""
    _print_row("metric", "ours", "paper", "delta", "verdict")
    print(f"  {'-'*32}  {'-'*20}   {'-'*20}   {'-'*10}   -------")


def _compare_configuration(metric_name: str, ours_values: np.ndarray, expected_spec: dict) -> bool:
    """Sanity-check an input parameter (e.g. fixed token budget). Tautological in full cramming."""
    paper_value = expected_spec["value"]
    tolerance = expected_spec.get("tolerance", 0)
    ours_mean = float(ours_values.mean())
    delta = abs(ours_mean - paper_value)
    verdict = "OK" if delta <= tolerance else "FAIL"
    _print_row(metric_name, f"{ours_mean:.3f}", f"{paper_value}", f"{delta:+.3f}", verdict)
    return verdict == "OK"


def _compare_measured(metric_name: str, ours_values: np.ndarray, expected_spec: dict) -> tuple[bool, dict]:
    """Compare a measured metric (mean ± std vs paper).

    Returns ``(ok, record)`` where ``ok`` is True iff verdict is within ±2σ
    and ``record`` is a JSON-serializable dict of the comparison — used by
    :func:`analyze` to persist a machine-readable summary alongside stdout.
    """
    paper_mean = expected_spec["mean"]
    paper_std = expected_spec["std"]
    ours_mean = float(ours_values.mean())
    ours_std = float(ours_values.std())
    z = _zscore(ours_mean, paper_mean, paper_std)
    verdict = _verdict(z)
    _print_row(
        metric_name,
        f"{ours_mean:.3f} ± {ours_std:.3f}",
        f"{paper_mean:.3f} ± {paper_std:.3f}",
        f"z={z:.2f}",
        verdict,
    )
    record = {
        "metric": metric_name,
        "ours_mean": ours_mean,
        "ours_std": ours_std,
        "paper_mean": paper_mean,
        "paper_std": paper_std,
        "z_score": None if np.isnan(z) else float(z),
        "verdict": verdict,
        "n_samples": int(ours_values.size),
    }
    return verdict == "OK", record


def _build_metric_extractors(trainer_type: str):
    """Map metric name → callable extracting per-sample value from a row dict."""
    extractors = {
        "information_gain_bits": lambda r: r["information_gain_bits"],
        "final_convergence": lambda r: r["final_convergence"],
    }
    if trainer_type == "full":
        # In full cramming compressed_tokens equals the fixed budget (= max_sequence_length).
        extractors["compressed_tokens"] = lambda r: r["num_input_tokens"]
    elif trainer_type == "progressive":
        # In progressive cramming compressed_tokens is the achieved prefix length per sample.
        extractors["compressed_tokens"] = lambda r: r["stage_seq_len"]
    return extractors


def _aggregate_rows_per_sample(rows: list[dict], trainer_type: str) -> list[dict]:
    """Reduce per-stage rows to one row per sample (no-op for full cramming).

    Full cramming saves one row per sample already. Progressive cramming saves
    one row per (sample, stage) — we collapse to the row of the *final converged
    stage* per sample (i.e. the largest stage_seq_len with final_convergence == 1.0).
    Samples that never converge contribute the row with the largest reached
    stage_seq_len (kept for diagnostic visibility).
    """
    if trainer_type != "progressive":
        return rows
    by_sample: dict[int, list[dict]] = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_id"]), []).append(row)
    aggregated: list[dict] = []
    for sample_id in sorted(by_sample):
        sample_rows = by_sample[sample_id]
        converged = [r for r in sample_rows if r.get("final_convergence") == 1.0]
        candidates = converged if converged else sample_rows
        aggregated.append(max(candidates, key=lambda r: r["stage_seq_len"]))
    return aggregated


def _split_metrics(spec: dict) -> tuple[dict, dict]:
    """Partition expected.json metrics into (configuration, measured) by their `kind` flag."""
    configuration: dict = {}
    measured: dict = {}
    for metric_name, expected_spec in spec["expected"].items():
        kind = expected_spec.get("kind", "measured")
        target = configuration if kind == "configuration" else measured
        target[metric_name] = expected_spec
    return configuration, measured


def _analyze_attention_hijacking(experiment_name: str, spec: dict, output_dir: str) -> bool:
    """Compare a saved attention_hijacking.json against paper Table 3 qualitative values.

    Paper Section 5.5 reports compression-mass / BOS-mass / correlation for a
    different model size (SmolLM2-1.7B). We use it as a qualitative reference:
    the experiment passes if (a) compression_mass >= 30% (clear hijacking) and
    (b) correlation >= 0.5 (per-layer profile shape matches BOS pattern).
    """
    json_path = Path(output_dir) / "attention_hijacking.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Run scripts/thesis_reproduction/run_attention_hijacking.py first.")
    with open(json_path) as f:
        result = json.load(f)
    summary = result["summary"]
    expected_summary = spec["expected"]

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Samples:    {summary['num_samples']}  (paper: {spec['num_samples']})")
    print(f"Note:       paper reference row is {spec.get('reference_model', spec['model'])} — qualitative comparison only.")

    print()
    print("Table-3 statistics (qualitative paper comparison):")
    _print_header()

    def _row(metric_name: str, ours: dict, paper: dict) -> None:
        ours_str = f"{ours['mean']:.3f} ± {ours['std']:.3f}"
        paper_str = f"{paper['mean']:.3f} ± {paper['std']:.3f}"
        z = _zscore(ours["mean"], paper["mean"], paper["std"])
        verdict = _verdict(z)
        _print_row(metric_name, ours_str, paper_str, f"z={z:.2f}", verdict)

    _row(
        "compression_mass_pct",
        summary["compression_mass"],
        expected_summary["compression_mass"],
    )
    _row("bos_mass_pct", summary["bos_mass"], expected_summary["bos_mass"])
    _row("correlation", summary["correlation"], expected_summary["correlation"])

    qual = spec.get("qualitative", {"min_compression_mass": 30.0, "min_correlation": 0.5})
    min_mass = float(qual["min_compression_mass"])
    min_corr = float(qual["min_correlation"])
    mass_ok = summary["compression_mass"]["mean"] >= min_mass
    corr_ok = summary["correlation"]["mean"] >= min_corr
    print()
    print(
        f"Qualitative gate: compression_mass ≥ {min_mass:.1f}% "
        f"({'OK' if mass_ok else 'FAIL'} — got {summary['compression_mass']['mean']:.2f}%) "
        f"AND correlation ≥ {min_corr:.2f} "
        f"({'OK' if corr_ok else 'FAIL'} — got {summary['correlation']['mean']:.4f})"
    )
    passed = mass_ok and corr_ok
    print()
    print(
        "Summary:",
        ("attention-hijacking pattern confirmed" if passed else "attention-hijacking pattern NOT confirmed"),
    )
    return passed


def _analyze_trajectory(experiment_name: str, spec: dict, output_dir: str) -> bool:
    """Compare a saved trajectory.json against paper Table 13 (trajectory_length, PCA 99%)."""
    json_path = Path(output_dir) / "trajectory.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Run scripts/thesis_reproduction/run_trajectory.py first.")
    with open(json_path) as f:
        result = json.load(f)
    summary = result["summary"]
    expected_summary = spec["expected"]

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Samples:    {summary['num_samples']}  (paper: {spec['num_samples']})")
    if summary.get("num_pca_excluded", 0):
        print(f"Note:       {summary['num_pca_excluded']} sample(s) excluded from PCA aggregate (<2 stages)")

    print()
    print("Table-13 statistics:")
    _print_header()

    def _row(metric_name: str, ours: dict, paper: dict) -> bool:
        ours_str = f"{ours['mean']:.3f} ± {ours['std']:.3f}"
        paper_str = f"{paper['mean']:.3f} ± {paper['std']:.3f}"
        z = _zscore(ours["mean"], paper["mean"], paper["std"])
        verdict = _verdict(z)
        _print_row(metric_name, ours_str, paper_str, f"z={z:.2f}", verdict)
        return verdict == "OK"

    length_ok = _row(
        "trajectory_length",
        summary["trajectory_length"],
        expected_summary["trajectory_length"],
    )
    pca_ok = _row("pca_99", summary["pca_99"], expected_summary["pca_99"])

    print()
    all_ok = length_ok and pca_ok
    print(
        "Summary:",
        "all measured metrics OK" if all_ok else "some metrics drifted — investigate",
    )
    return all_ok


def _analyze_attention_knockout(experiment_name: str, spec: dict, output_dir: str) -> bool:
    """Asymmetry-ratio gate for attention_knockout.json.

    Paper Reviewer 1 W2: early-layer KO degrades reconstruction; late-layer KO
    does not. The pass criteria (configurable per experiment) are:
        - baseline reconstruction accuracy is high (>= min_baseline);
        - early-layers (first ``num_edge`` layers) mean per-layer KO drops
          accuracy by at least ``min_early_drop`` (absolute floor);
        - the early-vs-late asymmetry ratio is at least ``min_asymmetry``,
          i.e. early_drop / max(late_drop, eps) >= min_asymmetry.
    The ratio criterion replaces an absolute ``max_late_drop`` so the gate
    stays meaningful when secondary-attention bumps near the model tail
    inflate the late-window mean (common at small scales).
    """
    json_path = Path(output_dir) / "attention_knockout.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Run scripts/thesis_reproduction/run_attention_knockout.py first.")
    with open(json_path) as f:
        result = json.load(f)
    summary = result["summary"]

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Samples:    {summary['num_samples']}  (paper: {spec['num_samples']})")
    print(f"Note:       qualitative causality probe — paper has no per-model knockout table for {spec['model']}.")

    baseline_acc = summary["baseline"]["mean"]
    num_layers = summary["num_layers"]
    per_layer_means = summary["per_layer"]["mean"]

    qual = spec.get(
        "qualitative",
        {
            "min_baseline": 0.9,
            "min_early_drop": 0.3,
            "min_asymmetry": 3.0,
            "num_edge": 4,
        },
    )
    num_edge = int(qual["num_edge"])
    early_acc = sum(per_layer_means[:num_edge]) / num_edge
    late_acc = sum(per_layer_means[-num_edge:]) / num_edge
    early_drop = baseline_acc - early_acc
    late_drop = baseline_acc - late_acc
    eps = 0.01
    asymmetry = early_drop / max(late_drop, eps)

    print()
    print("Reconstruction accuracy under per-layer attention knockout:")
    print(f"  baseline (no KO)             : {baseline_acc:.4f}")
    print(f"  layer 0                      : {per_layer_means[0]:.4f}")
    print(f"  layer {num_layers - 1:<22}: {per_layer_means[-1]:.4f}")
    print(f"  first {num_edge} layers (mean)        : {early_acc:.4f}  (drop {early_drop:+.4f})")
    print(f"  last  {num_edge} layers (mean)        : {late_acc:.4f}  (drop {late_drop:+.4f})")
    print(f"  asymmetry ratio (early/late) : {asymmetry:.2f}×")

    if summary.get("cumulative") is not None:
        cumulative_means = summary["cumulative"]["mean"]
        print()
        print("Forward cumulative KO (mask layers 0..l):")
        print(f"  l=0                          : {cumulative_means[0]:.4f}")
        print(f"  l={num_layers - 1:<26}: {cumulative_means[-1]:.4f}  (no-compression-context floor)")
    if summary.get("reverse_cumulative") is not None:
        reverse_means = summary["reverse_cumulative"]["mean"]
        print()
        print("Reverse cumulative KO (mask layers l..L-1):")
        print(f"  l=0                          : {reverse_means[0]:.4f}  (no-compression-context floor)")
        print(f"  l={num_layers - 1:<26}: {reverse_means[-1]:.4f}")

    min_baseline = float(qual["min_baseline"])
    min_early_drop = float(qual["min_early_drop"])
    min_asymmetry = float(qual["min_asymmetry"])
    baseline_ok = baseline_acc >= min_baseline
    early_ok = early_drop >= min_early_drop
    asymmetry_ok = asymmetry >= min_asymmetry
    print()
    print(
        f"Qualitative gate: baseline ≥ {min_baseline:.2f} "
        f"({'OK' if baseline_ok else 'FAIL'} — got {baseline_acc:.4f}), "
        f"early drop ≥ {min_early_drop:.2f} "
        f"({'OK' if early_ok else 'FAIL'} — got {early_drop:+.4f}), "
        f"asymmetry ≥ {min_asymmetry:.1f}× "
        f"({'OK' if asymmetry_ok else 'FAIL'} — got {asymmetry:.2f}×)"
    )
    passed = baseline_ok and early_ok and asymmetry_ok
    print()
    print(
        "Summary:",
        ("early-layer attention causally drives reconstruction" if passed else "causality pattern NOT confirmed — investigate"),
    )
    return passed


_DOWNSTREAM_VARIANTS: tuple[str, ...] = (
    "baseline",
    "baseline_endings",
    "compression",
    "compression_edge",
    "compression_endings",
    "compression_only",
    "compression_only_edge",
    "compression_only_endings",
)


def _analyze_downstream_eval(experiment_name: str, spec: dict, output_dir: str) -> bool:
    """Compare a saved downstream_eval.json against paper Table 10 (or Table 5).

    The eval JSON now contains 8 PPL variants per sample. We print every
    variant alongside the paper-expected value (if present) and apply a
    qualitative gate on the two canonical variants:
        - ``baseline_endings`` (Table 5 "Base") must clear random + a margin;
        - ``compression_endings`` (Table 5 "Cram") must be ``min_drop`` below.
    """
    json_path = Path(output_dir) / "downstream_eval.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Run scripts/thesis_reproduction/run_downstream_eval.py first.")
    with open(json_path) as f:
        result = json.load(f)
    # `subset` field in expected.json picks which summary view to compare against:
    #   "summary"                          — legacy primary view (matches CLI --only_full_convergence)
    #   "summary_all_samples"              — Table-5 view (paper §5.6 main column)
    #   "summary_perfectly_reconstructed"  — Table-10 view (perfectly-reconstructed subset)
    subset_key = spec.get("subset", "summary")
    if subset_key not in result:
        # Backwards-compat fallback for JSONs produced before dual-summary was added.
        subset_key = "summary"
    summary = result[subset_key]
    expected = spec.get("expected", {})

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Subset:     {subset_key}")
    print(
        f"Samples:    {summary['num_samples_total']}  (paper: {spec['num_samples']}; "
        f"fully converged: {summary['num_full_convergence']}/{summary['num_samples_total']})"
    )
    print(f"Note:       paper reference is {spec.get('reference_model', spec['model'])} — " f"qualitative comparison.")

    print()
    print("PPL variants (Table 10 of the paper):")
    _print_header()
    for variant in _DOWNSTREAM_VARIANTS:
        stats = summary[variant]
        # Paper Table 10 reports *token-normalized* accuracy (caption: "Token-normalized
        # accuracy (%) ..."), not the raw sample-count accuracy. Match the right field.
        ours = stats.get("token_normalized_accuracy", stats["accuracy"])
        ours_str = f"{ours:.4f}"
        paper_spec = expected.get(variant)
        if paper_spec is None:
            _print_row(variant, ours_str, "-", "-", "INFO")
            continue
        paper_mean = float(paper_spec["mean"])
        paper_std = float(paper_spec.get("std", 0.0))
        z = _zscore(ours, paper_mean, paper_std) if paper_std > 0 else float("nan")
        verdict = _verdict(z) if not np.isnan(z) else "INFO"
        delta = ours - paper_mean
        _print_row(variant, ours_str, f"{paper_mean:.4f}", f"{delta:+.4f}", verdict)

    # Use token-normalized accuracy to stay consistent with the paper Table 10
    # comparison above (and with the gate thresholds tuned for that scale).
    base_acc = summary["baseline_endings"].get("token_normalized_accuracy", summary["baseline_endings"]["accuracy"])
    cram_acc = summary["compression_endings"].get("token_normalized_accuracy", summary["compression_endings"]["accuracy"])
    drop = base_acc - cram_acc

    qual = spec.get("qualitative", {"min_base": 0.28, "min_drop": 0.10})
    min_base = float(qual["min_base"])
    min_drop = float(qual["min_drop"])
    base_ok = base_acc >= min_base
    drop_ok = drop >= min_drop

    print()
    print(
        f"Qualitative gate (baseline_endings vs compression_endings): "
        f"base ≥ {min_base:.2f} "
        f"({'OK' if base_ok else 'FAIL'} — got {base_acc:.4f}), "
        f"drop ≥ {min_drop:.2f} "
        f"({'OK' if drop_ok else 'FAIL'} — got {drop:+.4f})"
    )
    passed = base_ok and drop_ok
    print()
    print(
        "Summary:",
        (
            "downstream capability collapses under compression — paper claim confirmed"
            if passed
            else "downstream-collapse pattern NOT confirmed — investigate"
        ),
    )
    return passed


def _analyze_pca_reconstruction(experiment_name: str, spec: dict, output_dir: str) -> bool:
    """Qualitative gate on the PCA reconstruction accuracy curve (paper §5.3 / Figure 5).

    Paper claim: PCA 99 % of trajectory *variance* (Table 13 column) is NOT
    sufficient for near-perfect teacher-forced *accuracy*; one needs more
    components. Pass criteria (configurable per experiment):
        - max_k_accuracy ≥ ``min_max_k_accuracy``: at the largest k, reach
          ~baseline reconstruction accuracy.
        - pca99_accuracy ≤ ``max_pca99_accuracy``: at k = paper's PCA-99
          component count, accuracy is still below near-perfect (paper claim).
        - monotone-ish: each k-step should not drop accuracy by more than
          ``monotone_tolerance`` (allows small fluctuations from bf16 noise).
    """
    json_path = Path(output_dir) / "pca_reconstruction.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Run scripts/thesis_reproduction/run_pca_reconstruction.py first.")
    with open(json_path) as f:
        result = json.load(f)
    summary = result["summary"]
    curve = summary["curve"]

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Samples:    {summary['num_samples']}  (paper: {spec['num_samples']})")
    print(f"Note:       paper reference is {spec.get('reference_model', spec['model'])} — qualitative comparison.")

    print()
    print("PCA reconstruction curve (k → mean accuracy ± std | cum. variance ratio | n_samples):")
    for point in curve:
        var_str = (
            f"{point['variance_ratio_mean']:.4f} ± {point['variance_ratio_std']:.4f}"
            if "variance_ratio_mean" in point
            else "          -          "
        )
        print(f"  k={point['k']:>4}  acc={point['mean']:.4f} ± {point['std']:.4f}  " f"var={var_str}  (n={point['n_samples']})")

    qual = spec.get(
        "qualitative",
        {
            "min_max_k_accuracy": 0.9,
            "max_pca99_accuracy": 0.95,
            "pca99_k": 11,
            "monotone_tolerance": 0.05,
        },
    )
    min_max_k = float(qual["min_max_k_accuracy"])
    max_pca99 = float(qual["max_pca99_accuracy"])
    pca99_k = int(qual["pca99_k"])
    monotone_tol = float(qual.get("monotone_tolerance", 0.05))

    max_k_point = max(curve, key=lambda p: p["k"])
    max_k_acc = max_k_point["mean"]
    max_k_ok = max_k_acc >= min_max_k

    pca99_acc = next((p["mean"] for p in curve if p["k"] == pca99_k), None)
    if pca99_acc is None:
        # Fallback: the closest k <= pca99_k (paper's claim is about "PCA 99 %").
        candidates = [p for p in curve if p["k"] <= pca99_k]
        pca99_acc = max(candidates, key=lambda p: p["k"])["mean"] if candidates else None
    pca99_ok = pca99_acc is not None and pca99_acc <= max_pca99

    means = [p["mean"] for p in curve]
    max_drop = 0.0 if len(means) < 2 else max(means[i] - means[i + 1] for i in range(len(means) - 1))
    monotone_ok = max_drop <= monotone_tol

    print()
    print(
        f"Qualitative gate:\n"
        f"  max-k accuracy ≥ {min_max_k:.2f} "
        f"({'OK' if max_k_ok else 'FAIL'} — got {max_k_acc:.4f} at k={max_k_point['k']})\n"
        f"  accuracy@PCA99 (k={pca99_k}) ≤ {max_pca99:.2f}, i.e. PCA-99 is insufficient "
        f"({'OK' if pca99_ok else 'FAIL'} — got {pca99_acc if pca99_acc is None else f'{pca99_acc:.4f}'})\n"
        f"  monotone non-decreasing (tolerance {monotone_tol:.2f}) "
        f"({'OK' if monotone_ok else 'FAIL'} — max drop {max_drop:+.4f})"
    )
    passed = max_k_ok and pca99_ok and monotone_ok
    print()
    print(
        "Summary:",
        (
            "PCA 99 % variance is NOT sufficient for full reconstruction — paper claim confirmed"
            if passed
            else "PCA reconstruction pattern NOT confirmed — investigate"
        ),
    )
    return passed


def analyze(experiment_name: str, output_dir: str | None = None) -> bool:
    """Print paper-vs-ours comparison for one experiment. Returns True iff every measured metric is OK."""
    spec = _load_expected(experiment_name)
    output_dir = output_dir or os.path.join("artifacts", "thesis_reproduction", experiment_name)

    if spec.get("analyzer") == "attention_hijacking":
        return _analyze_attention_hijacking(experiment_name, spec, output_dir)
    if spec.get("analyzer") == "trajectory":
        return _analyze_trajectory(experiment_name, spec, output_dir)
    if spec.get("analyzer") == "attention_knockout":
        return _analyze_attention_knockout(experiment_name, spec, output_dir)
    if spec.get("analyzer") == "downstream_eval":
        return _analyze_downstream_eval(experiment_name, spec, output_dir)
    if spec.get("analyzer") == "pca_reconstruction":
        return _analyze_pca_reconstruction(experiment_name, spec, output_dir)

    ds = _load_dataset(output_dir, spec["trainer_type"])
    rows = _aggregate_rows_per_sample(list(ds), spec["trainer_type"])
    extractors = _build_metric_extractors(spec["trainer_type"])
    configuration_metrics, measured_metrics = _split_metrics(spec)

    print(f"\nExperiment: {experiment_name}")
    print(f"Paper:      {spec['paper_section']}")
    print(f"Model:      {spec['model']}")
    print(f"Samples:    {len(rows)}  (paper: {spec['num_samples']})")

    if configuration_metrics:
        print()
        print("Configuration (input parameters; tautological match expected):")
        _print_header()
        for metric_name, expected_spec in configuration_metrics.items():
            extractor = extractors.get(metric_name)
            if extractor is None:
                print(f"  {metric_name:<32}  [no extractor]")
                continue
            values = np.array([extractor(r) for r in rows], dtype=np.float64)
            _compare_configuration(metric_name, values, expected_spec)

    print()
    print("Measured metrics (real outputs of the optimization):")
    _print_header()
    all_ok = True
    measured_records: list[dict] = []
    for metric_name, expected_spec in measured_metrics.items():
        extractor = extractors.get(metric_name)
        if extractor is None:
            print(f"  {metric_name:<32}  [no extractor]")
            continue
        values = np.array([extractor(r) for r in rows], dtype=np.float64)
        ok, record = _compare_measured(metric_name, values, expected_spec)
        measured_records.append(record)
        all_ok = all_ok and ok

    print()
    print(
        "Summary:",
        "all measured metrics OK" if all_ok else "some metrics drifted — investigate",
    )

    # Persist a machine-readable copy of the comparison so the numbers
    # survive after the terminal scrolls / closes — useful for thesis tables
    # and later writeup. Saved as ``analysis_summary.json`` in output_dir.
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "experiment": experiment_name,
                    "paper_section": spec.get("paper_section"),
                    "model": spec.get("model"),
                    "trainer_type": spec.get("trainer_type"),
                    "num_samples_ours": len(rows),
                    "num_samples_paper": spec.get("num_samples"),
                    "measured": measured_records,
                    "all_ok": all_ok,
                },
                f,
                indent=2,
            )
        print(f"Saved JSON summary to {summary_path}")
    except OSError as e:
        # Don't fail the verdict just because the disk is read-only.
        print(f"(Failed to save analysis summary to {summary_path}: {e})")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a saved cramming run against paper-expected values.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment key from expected.json (e.g. full_cramming/pythia_160m).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory (default: artifacts/thesis_reproduction/<experiment>).",
    )
    args = parser.parse_args()
    analyze(args.experiment, args.output_dir)


if __name__ == "__main__":
    main()

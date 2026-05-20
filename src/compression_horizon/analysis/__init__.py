"""Analysis utilities for compression embeddings."""

from compression_horizon.analysis.attention_hijacking import (
    compute_attention_mass_profile,
    compute_sample_profiles,
    pearson_correlation,
    summarize_hijacking,
)
from compression_horizon.analysis.convergence import (
    ConvergedSamplesGuard,
    ConvergenceTracker,
    ProgressiveSampleStateMachine,
)
from compression_horizon.analysis.downstream_eval import (
    PPL_VARIANT_KEYS,
    aggregate_variant_accuracy,
    compute_continuation_nll,
    compute_ppl_baseline_batch,
    compute_ppl_compression_batch,
    predict_best_continuation,
    summarize_downstream,
)
from compression_horizon.analysis.information_gain import compute_information_gain
from compression_horizon.analysis.pca_reconstruction import (
    cumulative_variance_ratio,
    fit_per_sample_pca,
    project_top_k,
    summarize_pca_curve,
)
from compression_horizon.analysis.trajectory import (
    compute_pca_99,
    compute_trajectory_length,
    summarize_trajectory,
)

__all__ = [
    "compute_information_gain",
    "ConvergedSamplesGuard",
    "ConvergenceTracker",
    "ProgressiveSampleStateMachine",
    "compute_attention_mass_profile",
    "compute_sample_profiles",
    "pearson_correlation",
    "summarize_hijacking",
    "compute_pca_99",
    "compute_trajectory_length",
    "summarize_trajectory",
    "compute_continuation_nll",
    "compute_ppl_baseline_batch",
    "compute_ppl_compression_batch",
    "predict_best_continuation",
    "summarize_downstream",
    "aggregate_variant_accuracy",
    "PPL_VARIANT_KEYS",
    "fit_per_sample_pca",
    "project_top_k",
    "summarize_pca_curve",
    "cumulative_variance_ratio",
]

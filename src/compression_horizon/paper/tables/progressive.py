import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.decomposition import PCA
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.utils import hlines_to_booktabs, to_mean_std_cell

# This experiments finished before information gain was computed during experiment traning. information gain computed with scripts/visualize_multiple_trajectories.py
PRECOMPUTED_INFO_GAIN = {
    "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_2/progressive_prefixes": {
        "mean": 4694.2772,
        "std": 340.5500,
    },
    "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_4/progressive_prefixes": {
        "mean": 4960.7790,
        "std": 603.0176,
    },
    "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_8/progressive_prefixes": {
        "mean": 5130.3678,
        "std": 247.0386,
    },
    "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_16/progressive_prefixes": {
        "mean": 4686.1440,
        "std": 276.0113,
    },
    "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_24/progressive_prefixes": {
        "mean": 2094.5047,
        "std": 246.6593,
    },
    "artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_ds_pg19_loss_cosine_hybrid_1.0_align_32/progressive_prefixes": {
        "mean": 614.0741,
        "std": 168.8271,
    },
}


def _normalize_path(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def apply_precomputed_info_gain(dataset_path: str, stats: Dict[str, Any]) -> bool:
    normalized_path = _normalize_path(dataset_path)
    for path, values in PRECOMPUTED_INFO_GAIN.items():
        if normalized_path == _normalize_path(path):
            stats["information_gain_from_dataset"] = {
                "mean": float(values["mean"]),
                "std": float(values["std"]),
                "count": None,
            }
            return True
    return False


def load_progressive_dataset(dataset_path: str) -> Dataset:
    """Load a progressive embeddings dataset from disk."""
    return Dataset.load_from_disk(dataset_path)


def flatten_embedding(row: Dict[str, Any]) -> np.ndarray:
    """Flatten embedding from a dataset row."""
    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    return emb.reshape(-1).detach().cpu().numpy()


# v3: stats dict gained ``converged_prefix_len`` and ``steps_to_converged``
# (used by tab:added_tokens_ablation). These are derived from per-stage data, not
# from the cache, so older cache files remain reusable as-is (see
# load_experiment_cache, which never rebuilds or deletes on a version mismatch).
CACHE_VERSION = 3
CACHE_FILENAME = "low_dimensional_cache.json"


def get_experiment_cache_file(dataset_path: str) -> str:
    return os.path.join(dataset_path, CACHE_FILENAME)


def load_experiment_cache(dataset_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[str], bool]:
    if not dataset_path:
        return {}, None, False
    cache_file = get_experiment_cache_file(dataset_path)
    if not os.path.exists(cache_file):
        return {}, cache_file, False
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
            if not isinstance(cache_data, dict):
                return {}, cache_file, False
            cache_version = cache_data.get("cache_version")
            # Cache formats are compatible supersets across versions (newer versions only
            # add fields), so reuse whatever is present regardless of version instead of
            # discarding it. cache_has_metrics() decides per call whether the required
            # metrics exist and recomputes only what is missing. We deliberately never
            # delete a cache file, so a cache written by any other generator version stays
            # on disk and remains available for reuse.
            if cache_version != CACHE_VERSION:
                print(
                    f"Note: cache {cache_file} has version {cache_version!r}; generator "
                    f"expects {CACHE_VERSION}. Reading it as-is (no rebuild, not deleted)."
                )
            return cache_data, cache_file, True
    except (json.JSONDecodeError, IOError, ValueError) as e:
        raise ValueError(f"Failed to load cache file {cache_file}: {e}") from e


def save_experiment_cache(cache_file: Optional[str], cache_data: Dict[str, Any]) -> None:
    if cache_file is None:
        return
    try:
        cache_data["cache_version"] = CACHE_VERSION
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    except IOError as e:
        print(f"Warning: Failed to save cache file {cache_file}: {e}")


def serialize_array(array: np.ndarray) -> List[Any]:
    return array.tolist()


def deserialize_array(values: List[Any]) -> np.ndarray:
    return np.array(values, dtype=float)


def get_cache_metrics(cache_data: Dict[str, Any]) -> Dict[str, Any]:
    metrics = cache_data.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    metrics = {}
    cache_data["metrics"] = metrics
    return metrics


def get_metric_map(cache_data: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    metrics = get_cache_metrics(cache_data)
    metric_map = metrics.get(metric_name)
    if isinstance(metric_map, dict):
        return metric_map
    metric_map = {}
    metrics[metric_name] = metric_map
    return metric_map


def get_metric_value(cache_data: Dict[str, Any], metric_name: str) -> Optional[Any]:
    metrics = get_cache_metrics(cache_data)
    return metrics.get(metric_name)


def set_metric_value(cache_data: Dict[str, Any], metric_name: str, value: Any) -> bool:
    metrics = get_cache_metrics(cache_data)
    if metrics.get(metric_name) == value:
        return False
    metrics[metric_name] = value
    return True


def set_metric_map_value(cache_data: Dict[str, Any], metric_name: str, key: Any, value: Any) -> bool:
    metric_map = get_metric_map(cache_data, metric_name)
    key_str = str(key)
    if metric_map.get(key_str) == value:
        return False
    metric_map[key_str] = value
    return True


def cache_has_metrics(
    cache_data: Dict[str, Any],
    sample_ids: List[int],
    require_random_proj: bool,
    require_info_gain: bool,
    require_embedding_stats: bool,
    allow_info_gain_from_dataset: bool,
) -> bool:
    if not sample_ids:
        return False
    metrics = get_cache_metrics(cache_data)
    required_per_sample = ["trajectory_length", "pca_99_var"]
    if require_random_proj:
        required_per_sample.append("random_proj_99_var")
    if require_info_gain and not allow_info_gain_from_dataset:
        required_per_sample.append("information_gain")

    for metric_name in required_per_sample:
        metric_map = metrics.get(metric_name, {})
        if not isinstance(metric_map, dict):
            return False
        for sid in sample_ids:
            if str(sid) not in metric_map:
                return False

    if "pca_99_var_all_embeds" not in metrics:
        return False
    if require_embedding_stats and "embedding_statistics" not in metrics:
        return False

    return True


def get_sample_ids(ds: Dataset) -> List[int]:
    if "sample_id" not in ds.column_names:
        return []
    try:
        sample_ids = ds.unique("sample_id")
    except Exception:
        sample_ids = []
        for i in range(len(ds)):
            try:
                sid = ds[i].get("sample_id")
            except Exception:
                continue
            if sid is not None:
                sample_ids.append(sid)
    return sorted({int(sid) for sid in sample_ids if sid is not None})


def get_final_stage(stages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(stages) == 0:
        return None

    def get_sort_key(stage: Dict[str, Any]) -> Tuple[int, int]:
        stage_idx = stage.get("stage_index")
        stage_seq_len = stage.get("stage_seq_len")
        if stage_idx is not None:
            return (int(stage_idx), int(stage_seq_len) if stage_seq_len is not None else -1)
        if stage_seq_len is not None:
            return (-1, int(stage_seq_len))
        return (-1, -1)

    return max(stages, key=get_sort_key)


def filter_records(
    ds: Dataset,
    sample_id: Optional[int] = None,
    stage_index: Optional[int] = None,
    dataset_path: Optional[str] = None,
    model_checkpoint: Optional[str] = None,
    check_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Filter dataset records by sample_id and/or stage_index.

    Args:
        ds: Dataset to filter
        sample_id: Optional sample_id to filter by
        stage_index: Optional stage_index to filter by
        dataset_path: Optional dataset path for cache checking
        model_checkpoint: Optional model checkpoint for cache checking
        check_cache: If True, check if all cache files exist and remove embedding column if they do

    Returns:
        List of filtered records
    """
    rows: List[Dict[str, Any]] = []

    ds = ds.remove_columns(["orig_embedding", "initialization_embedding"])
    if "low_dim_prjoection_b" in ds.column_names:
        ds = ds.remove_columns(["low_dim_prjoection_b"])
    if "low_dim_prjoection_w" in ds.column_names:
        ds = ds.remove_columns(["low_dim_prjoection_w"])

    # Check if we can remove embedding column (if all cache metrics exist)
    if check_cache and dataset_path is not None and "embedding" in ds.column_names:
        if model_checkpoint is None and len(ds) > 0:
            try:
                model_checkpoint = ds[0].get("model_checkpoint")
            except Exception:
                model_checkpoint = None
        sample_ids_list = get_sample_ids(ds)
        if model_checkpoint is not None and sample_ids_list:
            cache_data, _, cache_loaded = load_experiment_cache(dataset_path)
            require_random_proj = os.environ.get("VISUALIZE_MULTIPLE_TRAJECTORIES_COMPUTE_RAND_PROJ") == "1"
            require_info_gain = os.environ.get("VISUALIZE_MULTIPLE_TRAJECTORIES_COMPUTE_IG") == "1"
            require_emb_stats = os.environ.get("VISUALIZE_MULTIPLE_TRAJECTORIES_COMPUTE_EMB_STATS") == "1"
            allow_info_gain_from_dataset = "information_gain_bits" in ds.column_names
            if cache_loaded and cache_has_metrics(
                cache_data,
                sample_ids_list,
                require_random_proj=require_random_proj,
                require_info_gain=require_info_gain,
                require_embedding_stats=require_emb_stats,
                allow_info_gain_from_dataset=allow_info_gain_from_dataset,
            ):
                print("Drop embeddings")
                ds = ds.remove_columns(["embedding"])

    for i in tqdm(range(len(ds)), desc="Filtering records"):
        r = ds[i]
        if sample_id is not None and int(r.get("sample_id", -1)) != int(sample_id):
            continue
        if stage_index is not None and int(r.get("stage_index", -1)) != int(stage_index):
            continue
        rows.append(r)
    return rows


def collate_stages_by_sample(
    rows: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Group rows by sample_id and sort by stage_index."""
    by_sid: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = int(r.get("sample_id", -1))
        if sid not in by_sid:
            by_sid[sid] = []
        by_sid[sid].append(r)
    for sid in by_sid:
        by_sid[sid].sort(key=lambda x: int(x.get("stage_index", 0)))
    return by_sid


def compute_num_pca_explained_99_var(
    embeddings: List[np.ndarray],
    cache_data: Optional[Dict[str, Any]] = None,
    cache_key_suffix: Optional[str] = None,
) -> float:
    """Compute the number of PCA components needed for 99% variance.

    Args:
        embeddings: List of flattened embedding arrays
        cache_data: Optional cache dictionary for the experiment.
        cache_key_suffix: Optional suffix for per-sample cache key (e.g., sample_id).

    Returns:
        The minimum number of principal components whose cumulative explained-variance
        ratio reaches 0.99. Returns -1 if 0.99 is not reached with the fitted PCA, or
        NaN if not computable.
    """
    if len(embeddings) < 2:
        return float("nan")

    # Stack embeddings: [n_samples, n_features]
    X = np.stack(embeddings, axis=0)

    # Need at least 2 samples for PCA
    if X.shape[0] < 2:
        return float("nan")

    n_samples, n_features = X.shape

    if cache_data is not None and cache_key_suffix is not None:
        metric_map = get_metric_map(cache_data, "pca_99_var")
        cached_result = metric_map.get(str(cache_key_suffix))
        if cached_result is not None:
            return float(cached_result)

    # Fit PCA with as many components as the data supports.
    max_PCA_components = min(512, n_samples - 1, n_features)
    if max_PCA_components < 1:
        return float("nan")

    pca = PCA(n_components=max_PCA_components, random_state=42)
    pca.fit(X)
    explained_var_ratio = pca.explained_variance_ratio_

    cumulative_var = np.cumsum(explained_var_ratio)
    idx = int(np.searchsorted(cumulative_var, 0.99, side="left"))
    if idx >= len(cumulative_var):
        result = -1.0
    else:
        result = float(idx + 1)

    if cache_data is not None and cache_key_suffix is not None:
        set_metric_map_value(cache_data, "pca_99_var", cache_key_suffix, result)

    return result


def compute_num_random_projections_explained_99_var(
    embeddings: List[np.ndarray],
    n_projections: int = 1000,
    random_state: int = 42,
    cache_data: Optional[Dict[str, Any]] = None,
    cache_key_suffix: Optional[str] = None,
) -> float:
    """Compute how many random projections explain 99% of variation in embeddings path.

    Args:
        embeddings: List of flattened embedding arrays
        n_projections: Number of random projection directions to generate
        random_state: Random seed for reproducibility
        cache_data: Optional cache dictionary for the experiment
        cache_key_suffix: Optional suffix for per-sample cache key (e.g., sample_id)

    Returns:
        Number of random projections needed to explain 99% variance, or NaN if not computable
    """
    if len(embeddings) < 2:
        return float("nan")

    if cache_data is not None and cache_key_suffix is not None:
        metric_map = get_metric_map(cache_data, "random_proj_99_var")
        cached_result = metric_map.get(str(cache_key_suffix))
        if cached_result is not None:
            return float(cached_result)

    # Stack embeddings: [n_samples, n_features]
    X = np.stack(embeddings, axis=0)

    # Need at least 2 samples
    if X.shape[0] < 2:
        return float("nan")

    n_samples, n_features = X.shape

    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Generate random projection directions (unit vectors)
    rng = np.random.RandomState(random_state)
    random_directions = rng.randn(n_projections, n_features)
    # Normalize to unit vectors
    norms = np.linalg.norm(random_directions, axis=1, keepdims=True)
    random_directions = random_directions / (norms + 1e-12)

    # Project embeddings onto each random direction
    projections = X_centered @ random_directions.T  # [n_samples, n_projections]

    # Compute variance along each projection direction
    variances = np.var(projections, axis=0)  # [n_projections]

    # Sort by variance (descending)
    sorted_indices = np.argsort(variances)[::-1]
    sorted_variances = variances[sorted_indices]

    # Compute cumulative variance
    total_variance = np.sum(sorted_variances)
    if total_variance == 0:
        return float("nan")

    cumulative_var = np.cumsum(sorted_variances) / total_variance

    # Find number of projections needed for 99% variance
    num_projections = (cumulative_var < 0.99).sum() + 1
    if num_projections > n_projections:
        num_projections = -1

    result = float(num_projections)
    if cache_data is not None and cache_key_suffix is not None:
        set_metric_map_value(cache_data, "random_proj_99_var", cache_key_suffix, result)

    return result


def compute_trajectory_length(
    embeddings: List[np.ndarray],
    cache_data: Optional[Dict[str, Any]] = None,
    cache_key_suffix: Optional[str] = None,
) -> float:
    """Compute trajectory length (sum of L2 distances between consecutive embeddings).

    Args:
        embeddings: List of flattened embedding arrays
        cache_data: Optional cache dictionary for the experiment.
        cache_key_suffix: Optional suffix for per-sample cache key (e.g., sample_id).

    Returns:
        Trajectory length (sum of distances), or 0.0 if less than 2 embeddings
    """
    if len(embeddings) < 2:
        return 0.0

    # Stack embeddings: [n_samples, n_features]
    X = np.stack(embeddings, axis=0)
    n_samples, n_features = X.shape

    if cache_data is not None and cache_key_suffix is not None:
        metric_map = get_metric_map(cache_data, "trajectory_length")
        cached_result = metric_map.get(str(cache_key_suffix))
        if cached_result is not None:
            return float(cached_result)

    # Compute trajectory length
    trajectory_length = 0.0
    for i in range(len(embeddings) - 1):
        dist = np.linalg.norm(embeddings[i + 1] - embeddings[i])
        trajectory_length += dist

    result = float(trajectory_length)

    if cache_data is not None and cache_key_suffix is not None:
        set_metric_map_value(cache_data, "trajectory_length", cache_key_suffix, result)

    return result


def compute_information_gain(
    rows: List[Dict[str, Any]],
    model_checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
    cache_data: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Compute information gain (CE-reduction) for all samples in the dataset.

    Information Gain = H_LM - H_LM+[mem]
    where H_LM is cross-entropy without memory vector and H_LM+[mem] is with memory vector.

    Args:
        rows: List of dataset rows, each containing 'text', 'embedding', 'num_compression_tokens', etc.
        model_checkpoint: Model checkpoint path. If None, tries to extract from first row.
        device: Device to run computation on. If None, uses CUDA if available.
        cache_data: Optional cache dictionary for the experiment.

    Returns:
        List of information gain values (one per sample, using final stage embedding)
    """

    if os.environ.get("VISUALIZE_MULTIPLE_TRAJECTORIES_COMPUTE_IG") != "1":
        return []

    if len(rows) == 0:
        return []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model checkpoint from first row if not provided
    if model_checkpoint is None:
        model_checkpoint = rows[0].get("model_checkpoint")
        if not model_checkpoint:
            print("Warning: model_checkpoint not found in dataset, skipping information gain computation")
            return []

    sample_ids = sorted({int(row.get("sample_id", -1)) for row in rows if row.get("sample_id") is not None})
    if cache_data is not None and sample_ids:
        metric_map = get_metric_map(cache_data, "information_gain")
        if all(str(sid) in metric_map for sid in sample_ids):
            return [float(metric_map[str(sid)]) for sid in sample_ids]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Group rows by sample_id and get final stage for each sample
    by_sid = collate_stages_by_sample(rows)

    # For each sample, get the final stage (highest stage_index or highest stage_seq_len)
    information_gains = []

    for sid, stages in by_sid.items():
        if len(stages) == 0:
            continue

        final_stage = get_final_stage(stages)
        if final_stage is None:
            continue

        text = final_stage.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            continue

        embedding = final_stage.get("embedding")
        if embedding is None:
            continue

        num_compression_tokens = int(final_stage.get("num_compression_tokens", 1))

        # Tokenize text
        enc = tokenizer(text, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Compute H_LM: cross-entropy without memory vector
        with torch.no_grad():
            outputs_lm = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_lm = outputs_lm.logits  # [1, seq_len, vocab_size]

            # Compute cross-entropy: shift logits and labels for next-token prediction
            shift_logits_lm = logits_lm[:, :-1, :].contiguous()
            shift_labels_lm = input_ids[:, 1:].contiguous()
            shift_mask_lm = attention_mask[:, 1:].contiguous()

            # Flatten for cross-entropy
            shift_logits_lm_flat = shift_logits_lm.view(-1, shift_logits_lm.size(-1))
            shift_labels_lm_flat = shift_labels_lm.view(-1)
            shift_mask_lm_flat = shift_mask_lm.view(-1)

            # Mask out padding
            valid_mask = shift_mask_lm_flat.bool()
            if valid_mask.sum() == 0:
                continue

            ce_lm = F.cross_entropy(
                shift_logits_lm_flat[valid_mask],
                shift_labels_lm_flat[valid_mask],
                reduction="sum",
            )
            # Convert from nats to bits: divide by ln(2)
            H_LM = ce_lm.item() / math.log(2)

        # Compute H_LM+[mem]: cross-entropy with memory vector
        embedding_tensor = torch.tensor(embedding, dtype=torch.bfloat16, device=device)
        if embedding_tensor.ndim == 1:
            # Reshape if needed: assume [num_compression_tokens * hidden_size] -> [num_compression_tokens, hidden_size]
            hidden_size = model.config.hidden_size
            if embedding_tensor.shape[0] == num_compression_tokens * hidden_size:
                embedding_tensor = embedding_tensor.reshape(num_compression_tokens, hidden_size)
            else:
                embedding_tensor = embedding_tensor.unsqueeze(0)
        if embedding_tensor.ndim == 2:
            embedding_tensor = embedding_tensor.unsqueeze(0)  # [1, num_compression_tokens, hidden_size]

        # Get token embeddings
        token_embeddings = model.model.embed_tokens(input_ids)  # [1, seq_len, hidden_size]

        # Concatenate compression tokens with token embeddings
        compression_attention_mask = torch.ones((1, num_compression_tokens), device=device, dtype=attention_mask.dtype)
        united_token_embeddings = torch.cat([embedding_tensor, token_embeddings], dim=1)
        united_attention_mask = torch.cat([compression_attention_mask, attention_mask], dim=1)

        with torch.no_grad():
            outputs_mem = model(inputs_embeds=united_token_embeddings.to(torch.bfloat16), attention_mask=united_attention_mask)
            logits_mem = outputs_mem.logits  # [1, num_compression_tokens + seq_len, vocab_size]

            # Align logits: slice from num_compression_tokens-1 to -1, then shift for next-token prediction
            aligned_logits_mem = logits_mem[:, num_compression_tokens:, :]  # [1, seq_len, vocab_size]

            # Compute cross-entropy: shift for next-token prediction
            shift_logits_mem = aligned_logits_mem[:, :-1, :].contiguous()
            shift_labels_mem = input_ids[:, 1:].contiguous()
            shift_mask_mem = attention_mask[:, 1:].contiguous()

            # Flatten for cross-entropy
            shift_logits_mem_flat = shift_logits_mem.view(-1, shift_logits_mem.size(-1))
            shift_labels_mem_flat = shift_labels_mem.view(-1)
            shift_mask_mem_flat = shift_mask_mem.view(-1)

            # Mask out padding
            valid_mask = shift_mask_mem_flat.bool()
            if valid_mask.sum() == 0:
                continue

            ce_mem = F.cross_entropy(
                shift_logits_mem_flat[valid_mask],
                shift_labels_mem_flat[valid_mask],
                reduction="sum",
            )
            # Convert from nats to bits: divide by ln(2)
            H_LM_mem = ce_mem.item() / math.log(2)

        # Information gain = H_LM - H_LM+[mem]
        info_gain = H_LM - H_LM_mem
        information_gains.append(info_gain)
        if cache_data is not None:
            set_metric_map_value(cache_data, "information_gain", sid, info_gain)

    return information_gains


def extract_information_gain_from_dataset(rows: List[Dict[str, Any]]) -> List[float]:
    """Extract information gain values from dataset rows.

    Args:
        rows: List of dataset rows, each potentially containing 'information_gain_bits'

    Returns:
        List of information gain values (one per sample, using final stage embedding)
    """
    if len(rows) == 0:
        return []

    # Group rows by sample_id and get final stage for each sample
    by_sid = collate_stages_by_sample(rows)

    # For each sample, get the final stage (highest stage_index or highest stage_seq_len)
    information_gains = []

    for sid, stages in by_sid.items():
        if len(stages) == 0:
            continue

        final_stage = get_final_stage(stages)
        if final_stage is None:
            continue

        # Extract information_gain_bits from the dataset
        info_gain = final_stage.get("information_gain_bits")
        if info_gain is not None:
            try:
                information_gains.append(float(info_gain))
            except (ValueError, TypeError):
                continue

    return information_gains


def compute_embedding_statistics(
    rows: List[Dict[str, Any]],
    model_checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
    cache_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, float]]:
    """Compute norm statistics (mean ± std of L2 norms) for compression embeddings vs regular vocab tokens.

    Args:
        rows: List of dataset rows, each containing 'embedding', 'num_compression_tokens', etc.
        model_checkpoint: Model checkpoint path. If None, tries to extract from first row.
        device: Device to run computation on. If None, uses CUDA if available.
        cache_data: Optional cache dictionary for the experiment.

    Returns:
        Dict with comp/vocab mean and std, or None if not computable
    """
    if os.environ.get("VISUALIZE_MULTIPLE_TRAJECTORIES_COMPUTE_EMB_STATS") != "1":
        return None

    if len(rows) == 0:
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model checkpoint from first row if not provided
    if model_checkpoint is None:
        model_checkpoint = rows[0].get("model_checkpoint")
        if not model_checkpoint:
            return None

    if cache_data is not None:
        cached_result = get_metric_value(cache_data, "embedding_statistics")
        if isinstance(cached_result, dict):
            return cached_result

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"Warning: Failed to load model for embedding statistics: {e}")
        return None

    # Get all vocab token embeddings
    vocab_size = len(tokenizer)
    vocab_token_ids = torch.arange(vocab_size, device=device)
    with torch.no_grad():
        vocab_embeddings = model.model.embed_tokens(vocab_token_ids)  # [vocab_size, hidden_size]
        vocab_embeddings_np = vocab_embeddings.float().cpu().numpy()

    # Group rows by sample_id and get final stage for each sample
    by_sid = collate_stages_by_sample(rows)

    # Collect all compression token embeddings from final stages
    compression_token_embeddings = []

    for sid, stages in by_sid.items():
        if len(stages) == 0:
            continue

        final_stage = get_final_stage(stages)
        if final_stage is None:
            continue

        embedding = final_stage.get("embedding")
        if embedding is None:
            continue

        num_compression_tokens = int(final_stage.get("num_compression_tokens", 1))
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        hidden_size = model.config.hidden_size

        # Reshape embedding if needed
        if embedding_tensor.ndim == 1:
            if embedding_tensor.shape[0] == num_compression_tokens * hidden_size:
                embedding_tensor = embedding_tensor.reshape(num_compression_tokens, hidden_size)
            else:
                embedding_tensor = embedding_tensor.unsqueeze(0)
        elif embedding_tensor.ndim == 2:
            # Already in [num_compression_tokens, hidden_size] format
            pass
        else:
            continue

        # Add each compression token embedding separately (not flattened)
        compression_token_embeddings.append(embedding_tensor.numpy())

    if len(compression_token_embeddings) == 0:
        return None

    # Stack compression token embeddings: [total_tokens, hidden_size]
    compression_token_embeddings_np = np.vstack(compression_token_embeddings)  # [total_compression_tokens, hidden_size]

    # Compute L2 norms for compression token embeddings (one norm per token)
    compression_norms = np.linalg.norm(compression_token_embeddings_np, axis=1)  # [total_compression_tokens]
    comp_norm_avg = np.mean(compression_norms)
    comp_norm_std = np.std(compression_norms)

    # Compute L2 norms for vocab embeddings (one norm per token)
    vocab_norms = np.linalg.norm(vocab_embeddings_np, axis=1)  # [vocab_size]
    vocab_norm_avg = np.mean(vocab_norms)
    vocab_norm_std = np.std(vocab_norms)

    result = {
        "comp_norm_avg": float(comp_norm_avg),
        "comp_norm_std": float(comp_norm_std),
        "vocab_norm_avg": float(vocab_norm_avg),
        "vocab_norm_std": float(vocab_norm_std),
    }

    if cache_data is not None:
        set_metric_value(cache_data, "embedding_statistics", result)

    return result


def extract_trajectory(
    dataset_path: str,
    sample_id: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any], np.ndarray]:
    """Extract embedding trajectory from a dataset.

    Args:
        dataset_path: Path to the progressive embeddings dataset
        sample_id: Optional sample_id to filter. If None, uses first available sample for visualization,
                   but computes statistics across all samples in the dataset.

    Returns:
        Tuple of (embeddings array [n_stages, n_features], labels list, statistics dict, final_embedding)
        Statistics dict contains: 'num_embeddings', 'total_steps', 'trajectory_length', 'num_pca_for99_var',
        'num_random_projections_for99_var', etc. Each metric stores raw mean/std/count (no formatting).
        final_embedding is the last embedding in the trajectory (for the selected sample)
    """
    cache_data, cache_file, cache_loaded = load_experiment_cache(dataset_path)
    if cache_loaded:
        cached_stats = cache_data.get("stats")
        cached_traj = cache_data.get("trajectory")
        if not isinstance(cached_stats, dict) or not isinstance(cached_traj, dict):
            raise ValueError(f"Cache file {cache_file} is missing required fields. Delete it to rebuild.")
        cached_sample_id = cached_traj.get("sample_id")
        if sample_id is not None and cached_sample_id is not None and int(cached_sample_id) != int(sample_id):
            raise ValueError(f"Cache file {cache_file} was built for sample_id={cached_sample_id}. Delete it to rebuild.")
        cached_embeddings = cached_traj.get("embeddings")
        cached_labels = cached_traj.get("labels")
        cached_final_embedding = cached_traj.get("final_embedding")
        if not cached_embeddings or not cached_labels or cached_final_embedding is None:
            raise ValueError(f"Cache file {cache_file} is incomplete. Delete it to rebuild.")
        if apply_precomputed_info_gain(dataset_path, cached_stats):
            cache_data["stats"] = cached_stats
            save_experiment_cache(cache_file, cache_data)
        embeddings = deserialize_array(cached_embeddings)
        final_embedding = deserialize_array(cached_final_embedding)
        return embeddings, list(cached_labels), cached_stats, final_embedding

    ds = load_progressive_dataset(dataset_path)
    # Get model_checkpoint from dataset if available (for cache checking)
    model_checkpoint_for_cache = None
    if len(ds) > 0:
        try:
            first_row = ds[0]
            model_checkpoint_for_cache = first_row.get("model_checkpoint")
        except Exception:
            pass

    # Load all rows to compute statistics across all samples
    all_rows = filter_records(
        ds, sample_id=None, dataset_path=dataset_path, model_checkpoint=model_checkpoint_for_cache, check_cache=True
    )

    if not all_rows:
        raise ValueError(f"No records found in {dataset_path}")

    # Group all rows by sample_id
    all_by_sid = collate_stages_by_sample(all_rows)

    # Compute statistics for all samples
    all_num_embeddings = []
    all_max_prefix_len = []
    all_converged_prefix_len = []
    all_steps_to_converged = []
    all_total_steps = []
    all_trajectory_lengths = []
    all_num_pca_for99_var = []
    all_num_random_projections_for99_var = []
    all_prefix_surprisal = []  # per-sample base-LM prefix surprisal (bits/token); empty for no-prefix runs

    all_embeds = []

    # Extract information gain from dataset (if available)
    information_gains_from_dataset = extract_information_gain_from_dataset(all_rows)

    # Compute information gain for all samples (by loading model and computing)
    # Only compute if not available from dataset
    model_checkpoint = None
    if len(all_rows) > 0:
        model_checkpoint = all_rows[0].get("model_checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(information_gains_from_dataset) == 0:
        information_gains = compute_information_gain(
            all_rows, model_checkpoint=model_checkpoint, device=device, cache_data=cache_data
        )
    else:
        information_gains = []

    # Compute embedding statistics (compression vs vocab)
    embedding_statistics = compute_embedding_statistics(
        all_rows, model_checkpoint=model_checkpoint, device=device, cache_data=cache_data
    )

    for sid, stages in all_by_sid.items():
        # Extract embeddings for this sample (if available)
        sample_embeddings = []
        sample_total_steps = 0
        has_embeddings = True
        for stage in stages:
            if "embedding" not in stage or stage.get("embedding") is None:
                has_embeddings = False
                break
            emb = flatten_embedding(stage)
            sample_embeddings.append(emb)
            steps = int(stage.get("steps_taken", 0))
            sample_total_steps += steps

        # Compute metrics that don't require embeddings
        all_num_embeddings.append(len(stages))
        # Furthest prefix length attempted = the achieved reconstructed-prefix
        # length n. With the default progressive_step=1 / min_seq_len=1 schedule
        # this equals len(stages) (so existing tables are unchanged), but for a
        # step Δ>1 schedule it reflects actual tokens (≈ Δ·#stages) rather than
        # the stage count. Exposed as ``max_prefix_len`` for opt-in use by tables
        # that ablate the step (see TableSpec.compressed_tokens_key).
        stage_seq_lens = [int(s.get("stage_seq_len")) for s in stages if s.get("stage_seq_len") is not None]
        all_max_prefix_len.append(max(stage_seq_lens) if stage_seq_lens else len(stages))
        all_total_steps.append(sample_total_steps)

        # Prefix surprisal (bits/token) is constant across a sample's stages; take the first
        # non-null value. Absent for no-prefix runs -> the column renders blank for them.
        sample_prefix_surprisal = next(
            (s.get("prefix_surprisal_bits_per_token") for s in stages if s.get("prefix_surprisal_bits_per_token") is not None),
            None,
        )
        if sample_prefix_surprisal is not None:
            all_prefix_surprisal.append(float(sample_prefix_surprisal))

        # Largest *converged* prefix length and the cumulative steps spent to reach
        # it. Progressive cramming stops at the first stage that fails to converge,
        # so the final stage is a failure whose seq_len overshoots the achieved
        # length by the step Δ; restricting to stages that hit the convergence
        # threshold gives the true number of perfectly reconstructed tokens (and
        # avoids the +Δ inflation that ``max_prefix_len`` carries for Δ>1).
        converged_stages = [
            s
            for s in stages
            if s.get("stage_seq_len") is not None
            and s.get("final_convergence") is not None
            and s.get("convergence_threshold") is not None
            and s["final_convergence"] >= s["convergence_threshold"]
        ]
        if converged_stages:
            best_converged = max(converged_stages, key=lambda s: int(s["stage_seq_len"]))
            all_converged_prefix_len.append(int(best_converged["stage_seq_len"]))
            all_steps_to_converged.append(int(best_converged.get("steps_taken", 0)))
        else:
            all_converged_prefix_len.append(0)
            all_steps_to_converged.append(0)

        # Compute metrics that require embeddings (use cache if embeddings missing)
        if has_embeddings and len(sample_embeddings) > 0:
            trajectory_length = compute_trajectory_length(sample_embeddings, cache_data=cache_data, cache_key_suffix=sid)
            all_trajectory_lengths.append(trajectory_length)

            all_embeds.extend(sample_embeddings)
            num_pca_explained_99_var = compute_num_pca_explained_99_var(
                sample_embeddings, cache_data=cache_data, cache_key_suffix=sid
            )
            if not np.isnan(num_pca_explained_99_var):
                all_num_pca_for99_var.append(num_pca_explained_99_var)

            if os.environ.get("VISUALIZE_MULTIPLE_TRAJECTORIES_COMPUTE_RAND_PROJ") == "1":
                num_random_projections = compute_num_random_projections_explained_99_var(
                    sample_embeddings, cache_data=cache_data, cache_key_suffix=sid
                )
                if not np.isnan(num_random_projections):
                    all_num_random_projections_for99_var.append(num_random_projections)
        else:
            cached_traj = get_metric_map(cache_data, "trajectory_length").get(str(sid))
            if cached_traj is not None:
                all_trajectory_lengths.append(float(cached_traj))

            cached_pca = get_metric_map(cache_data, "pca_99_var").get(str(sid))
            if cached_pca is not None and not np.isnan(float(cached_pca)):
                all_num_pca_for99_var.append(float(cached_pca))

            if os.environ.get("VISUALIZE_MULTIPLE_TRAJECTORIES_COMPUTE_RAND_PROJ") == "1":
                cached_rand = get_metric_map(cache_data, "random_proj_99_var").get(str(sid))
                if cached_rand is not None and not np.isnan(float(cached_rand)):
                    all_num_random_projections_for99_var.append(float(cached_rand))

    # Compute PCA for all embeddings (use cache if embeddings missing)
    cached_all_embeds = get_metric_value(cache_data, "pca_99_var_all_embeds")
    if cached_all_embeds is not None:
        num_pca_explained_99_var_all_embeds = float(cached_all_embeds)
    elif len(all_embeds) > 0:
        num_pca_explained_99_var_all_embeds = compute_num_pca_explained_99_var(all_embeds)
        set_metric_value(cache_data, "pca_99_var_all_embeds", num_pca_explained_99_var_all_embeds)
    else:
        num_pca_explained_99_var_all_embeds = float("nan")

    def summarize_values(values: List[float]) -> Optional[Dict[str, Any]]:
        if len(values) == 0:
            return None
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": int(len(values)),
        }

    stats = {
        "num_embeddings": summarize_values(all_num_embeddings),
        "max_prefix_len": summarize_values(all_max_prefix_len),
        "converged_prefix_len": summarize_values(all_converged_prefix_len),
        "steps_to_converged": summarize_values(all_steps_to_converged),
        "total_steps": summarize_values(all_total_steps),
        "steps_taken": summarize_values(all_total_steps),
        "trajectory_length": summarize_values(all_trajectory_lengths),
        "num_pca_for99_var": summarize_values(all_num_pca_for99_var),
        "num_pca_for99_var_all_embeds": (
            float(num_pca_explained_99_var_all_embeds) if not np.isnan(num_pca_explained_99_var_all_embeds) else None
        ),
        "num_random_projections_for99_var": summarize_values(all_num_random_projections_for99_var),
        "information_gain": summarize_values(information_gains),
        "information_gain_from_dataset": summarize_values(information_gains_from_dataset),
        "prefix_surprisal": summarize_values(all_prefix_surprisal),
        "embedding_statistics": embedding_statistics,
    }
    apply_precomputed_info_gain(dataset_path, stats)

    # Now extract trajectory for visualization (use specified sample_id or first available)
    if sample_id is not None:
        if sample_id not in all_by_sid:
            raise ValueError(f"Sample {sample_id} not found in {dataset_path}")
        vis_sample_id = sample_id
    else:
        # Use first available sample
        first_sid = sorted(all_by_sid.keys())[0]
        vis_sample_id = first_sid

    # Reload dataset for visualization sample if embeddings were removed
    # Check if we have embeddings in the stages
    stages = all_by_sid[vis_sample_id]
    has_embeddings_for_vis = any("embedding" in stage and stage.get("embedding") is not None for stage in stages)

    if not has_embeddings_for_vis:
        # Reload dataset with embeddings for visualization
        ds_vis = load_progressive_dataset(dataset_path)
        vis_rows = filter_records(ds_vis, sample_id=vis_sample_id, check_cache=False)
        stages = collate_stages_by_sample(vis_rows).get(vis_sample_id, [])

    # Extract embeddings in order for visualization
    embeddings = []
    labels = []
    for stage in stages:
        if "embedding" not in stage or stage.get("embedding") is None:
            raise ValueError(f"Embeddings not available for sample {vis_sample_id} in {dataset_path}")
        emb = flatten_embedding(stage)
        embeddings.append(emb)
        stage_seq_len = int(stage.get("stage_seq_len", -1))
        labels.append(f"L{stage_seq_len}")

    if len(embeddings) == 0:
        raise ValueError(f"No embeddings found for sample {sample_id} in {dataset_path}")

    X = np.stack(embeddings, axis=0)
    final_embedding = embeddings[-1]  # Last embedding

    cache_data["stats"] = stats
    cache_data["trajectory"] = {
        "sample_id": int(vis_sample_id),
        "embeddings": serialize_array(X),
        "labels": labels,
        "final_embedding": serialize_array(final_embedding),
    }

    save_experiment_cache(cache_file, cache_data)

    return X, labels, stats, final_embedding


def plot_pca_trajectories(
    trajectories: List[np.ndarray],
    checkpoint_names: List[str],
    outfile: str,
    n_components: int = 2,
    show_labels: bool = False,
    labels_list: Optional[List[List[str]]] = None,
):
    """Plot multiple embedding trajectories on a single PCA plot.

    Args:
        trajectories: List of embedding arrays, each of shape [n_stages, n_features]
        checkpoint_names: List of names for each trajectory (for legend)
        outfile: Output file path
        n_components: Number of PCA components to use (2 or 4)
        show_labels: Whether to show stage labels on points
        labels_list: Optional list of label lists for each trajectory
    """
    if len(trajectories) == 0:
        raise ValueError("No trajectories provided")

    # Combine all embeddings to fit a single PCA
    all_embeddings = np.vstack(trajectories)
    n_samples, n_features = all_embeddings.shape

    if n_samples < 2 or n_features < 2:
        raise ValueError(f"Insufficient data: {n_samples} samples, {n_features} features")

    n_components = min(n_components, n_samples - 1, n_features)
    if n_components < 2:
        raise ValueError(f"Cannot compute {n_components} components")

    # Fit PCA on all embeddings
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(all_embeddings)
    explained_var = pca.explained_variance_ratio_

    # Transform each trajectory
    transformed_trajectories = []
    for traj in trajectories:
        traj_transformed = pca.transform(traj)
        transformed_trajectories.append(traj_transformed)

    # Create distinct colors for checkpoints
    # Use a predefined set of highly distinct colors with maximum hue separation
    distinct_colors = [
        "#E6194B",  # bright red
        "#3CB44B",  # bright green
        "#FFE119",  # bright yellow
        "#4363D8",  # bright blue
        "#F58231",  # bright orange
        "#911EB4",  # bright purple
        "#42D4F4",  # bright cyan
        "#F032E6",  # bright magenta
        "#BFEF45",  # lime green
        "#FABED4",  # light pink
        "#469990",  # teal
        "#DCBEFF",  # light purple
        "#9A6324",  # brown
        "#FFFAC8",  # beige
        "#800000",  # maroon
        "#000075",  # navy
        "#A9A9A9",  # gray
        "#000000",  # black
    ]
    # Cycle through distinct colors if we have more trajectories than colors
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(trajectories))]

    if n_components == 2:
        # Single 2D plot
        plt.figure(figsize=(10, 8))
        legend_handles = []
        for idx, (traj_transformed, name, color) in enumerate(zip(transformed_trajectories, checkpoint_names, colors)):
            x_data = traj_transformed[:, 0]
            y_data = traj_transformed[:, 1]

            # Plot trajectory line (without label)
            plt.plot(x_data, y_data, color=color, alpha=0.5, linewidth=1.5, linestyle="--")

            # Plot points
            plt.scatter(x_data, y_data, c=[color], s=60, alpha=0.7, edgecolors="black", linewidths=0.5)

            # Create legend handle with scatter marker
            legend_handles.append(plt.scatter([], [], c=color, s=60, alpha=0.7, edgecolors="black", linewidths=0.5, label=name))

            # Add labels if requested
            if show_labels and labels_list is not None and idx < len(labels_list):
                labels = labels_list[idx]
                labeled_positions = []
                for k, lab in enumerate(labels):
                    if k >= len(x_data):
                        continue
                    # Check if there's already a labeled point within distance < 0.5
                    should_label = True
                    for labeled_pos in labeled_positions:
                        dist = np.linalg.norm([x_data[k] - labeled_pos[0], y_data[k] - labeled_pos[1]])
                        if dist < 0.5:
                            should_label = False
                            break
                    if should_label:
                        plt.text(x_data[k], y_data[k], lab, fontsize=12, ha="left", va="bottom", color=color)
                        labeled_positions.append([x_data[k], y_data[k]])

            # Mark start and end points
            if len(x_data) > 0:
                plt.scatter(x_data[0], y_data[0], c=[color], s=150, marker="o", edgecolors="black", linewidths=2, zorder=5)
                plt.scatter(x_data[-1], y_data[-1], c=[color], s=150, marker="s", edgecolors="black", linewidths=2, zorder=5)

        plt.xlabel(f"PC1 ({explained_var[0]:.4f})", fontsize=18)
        plt.ylabel(f"PC2 ({explained_var[1]:.4f})", fontsize=18)
        plt.title(
            f"PCA Trajectories Comparison\nCumulative variance: {explained_var.sum():.4f}",
            fontsize=20,
        )
        plt.legend(handles=legend_handles, loc="best", fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Saved 2D PCA plot to: {outfile}")

    elif n_components == 4:
        # Multiple subplots for 4 components (similar to plot_pca_4_components)
        pairs = [(i, j) for i in range(n_components) for j in range(i + 1, n_components)]
        n_pairs = len(pairs)

        n_cols = 3
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        legend_handles = []
        for pair_idx, (i, j) in enumerate(pairs):
            ax = axes[pair_idx]

            for idx, (traj_transformed, name, color) in enumerate(zip(transformed_trajectories, checkpoint_names, colors)):
                x_data = traj_transformed[:, i]
                y_data = traj_transformed[:, j]

                # Plot trajectory line (without label)
                ax.plot(x_data, y_data, color=color, alpha=0.5, linewidth=1.5, linestyle="--")

                # Plot points
                ax.scatter(x_data, y_data, c=[color], s=60, alpha=0.7, edgecolors="black", linewidths=0.5)

                # Create legend handle with scatter marker (only for first subplot)
                if pair_idx == 0:
                    legend_handles.append(
                        ax.scatter([], [], c=color, s=60, alpha=0.7, edgecolors="black", linewidths=0.5, label=name)
                    )

                # Mark start and end points
                if len(x_data) > 0:
                    ax.scatter(x_data[0], y_data[0], c=[color], s=150, marker="o", edgecolors="black", linewidths=2, zorder=5)
                    ax.scatter(x_data[-1], y_data[-1], c=[color], s=150, marker="s", edgecolors="black", linewidths=2, zorder=5)

            ax.set_xlabel(f"PC{i+1} ({explained_var[i]:.3f})", fontsize=14)
            ax.set_ylabel(f"PC{j+1} ({explained_var[j]:.3f})", fontsize=14)
            ax.set_title(f"PC{i+1} vs PC{j+1}", fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            if pair_idx == 0:
                ax.legend(handles=legend_handles, loc="best", fontsize=16)

        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"PCA Trajectories Comparison (4 components, cumulative variance: {explained_var.sum():.4f})",
            fontsize=18,
        )
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Saved 4-component PCA plot to: {outfile}")
    else:
        raise ValueError(f"n_components must be 2 or 4, got {n_components}")


def compute_pairwise_distances(final_embeddings: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise distances between final embeddings.

    Args:
        final_embeddings: List of final embedding arrays

    Returns:
        Tuple of (l2_distances, l1_distances, cosine_distances) matrices
    """
    n = len(final_embeddings)
    if n < 2:
        return np.array([]), np.array([]), np.array([])

    # Stack embeddings
    X = np.stack(final_embeddings, axis=0)  # [n_experiments, n_features]

    # Compute L2 distances
    diffs = X[:, None, :] - X[None, :, :]
    l2_distances = np.linalg.norm(diffs, axis=-1)

    # Compute L1 distances
    l1_distances = np.linalg.norm(diffs, ord=1, axis=-1)

    # Compute cosine distances
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos_sim = (Xn @ Xn.T).clip(-1.0, 1.0)
    cosine_distances = 1.0 - cos_sim

    return l2_distances, l1_distances, cosine_distances


def format_mean_std_cell(
    stat: Optional[Dict[str, Any]],
    precision: int,
    tablefmt: str,
) -> str:
    if not stat:
        return "nan"
    mean_val = stat.get("mean")
    std_val = stat.get("std")
    if mean_val is None or std_val is None:
        return "nan"
    return to_mean_std_cell(
        mean_val,
        std_val,
        use_latex=(tablefmt == "latex"),
        float_precision=precision,
    )


def format_embedding_statistics(
    stat: Optional[Dict[str, Any]],
    precision: int,
    tablefmt: str,
) -> str:
    if not stat:
        return "nan"
    if (
        stat.get("comp_norm_avg") is None
        or stat.get("comp_norm_std") is None
        or stat.get("vocab_norm_avg") is None
        or stat.get("vocab_norm_std") is None
    ):
        return "nan"
    comp = to_mean_std_cell(
        stat.get("comp_norm_avg"),
        stat.get("comp_norm_std"),
        use_latex=(tablefmt == "latex"),
        float_precision=precision,
    )
    vocab = to_mean_std_cell(
        stat.get("vocab_norm_avg"),
        stat.get("vocab_norm_std"),
        use_latex=(tablefmt == "latex"),
        float_precision=precision,
    )
    if "nan" in (comp, vocab):
        return "nan"
    return f"{comp} / {vocab}"


def format_statistics_table(
    checkpoint_names: List[str],
    statistics: List[Dict[str, Any]],
    midrule_indicies,
    tablefmt: str = "grid",
    short: bool = False,
    compressed_tokens_key: str = "num_embeddings",
    steps_key: Optional[str] = None,
    show_prefix_surprisal: bool = False,
) -> str:
    """Build the formatted statistics table as a string.

    Args:
        checkpoint_names: List of experiment labels
        statistics: List of statistics dicts, each containing 'num_embeddings' and 'total_steps'
        short: If True, build the table without the last two columns
        compressed_tokens_key: stats key used for the "Compressed Tokens" column
            (default 'num_embeddings' = stage count). Tables that ablate the
            progressive step pass 'converged_prefix_len' so the column reports the
            largest perfectly-reconstructed prefix length (excluding the final
            non-converged stage).
        steps_key: if set (and not ``short``), append a "Steps to Converge" column
            sourced from this stats key (e.g. 'steps_to_converged'). Default None
            leaves the column set unchanged for every other table.
        show_prefix_surprisal: If True, append an "Avg Prefix Surprisal (bits/tok)" column. Rows
            without a ``prefix_surprisal`` stat (e.g. a no-prefix baseline) render ``--`` rather than
            ``nan`` so the paper lint stays green. Opt-in so other tables are unaffected.

    Returns:
        The fully post-processed table text, or an empty string if there is
        nothing to render.
    """
    if len(checkpoint_names) == 0 or len(statistics) == 0:
        return ""

    if short:
        headers = [
            "Model",
            "Cram Tokens",
            "Info Gain",
        ]
    else:
        headers = [
            "Model",
            "Compressed Tokens",
            "Information Gain",
        ]
    if not short:
        headers += [
            "Trajectory Length",
            "PCA 99%",
        ]
    if steps_key is not None and not short:
        headers += ["Steps to Converge"]
    if show_prefix_surprisal:
        headers += ["Prefix Surp. (bits/tok)" if short else "Avg Prefix Surprisal (bits/tok)"]

    # Prepare table data
    table_data = []
    i = 0
    for name, stats in zip(checkpoint_names, statistics):

        table_name = name
        table_name = table_name.replace("sl_4096_", "")
        table_name = table_name.replace("_ds_pg19_1k_limit_50", "")
        table_name = table_name.replace("_nobos", " \\bcancel{B}")
        table_name = table_name.replace("_lowproj", "")
        table_name = table_name.replace("Meta-", "")
        table_name = table_name.replace("_ds_pg19_loss_cosine", "")
        table_name = table_name.replace("_loss_cosine", "")
        table_name = re.sub(r"_hybrid_(\d+(\.?\d+)?)", r" {\\small $\\alpha=\1$}", table_name)
        table_name = re.sub(r"_align_(\d+)", r" {\\small $L=\1$}", table_name)
        table_name = re.sub(r"_lowdim_(\d+)", r" {\\small dim=\1}", table_name)
        table_name = re.sub(r"_lr_(\d+(\.?\d+)?)", r" {\\small lr=\1}", table_name)

        num_embeds_precision = 1
        if short:
            num_embeds_precision = 0

        compressed_tokens_stat = stats.get(compressed_tokens_key)
        if compressed_tokens_stat is None:
            compressed_tokens_stat = stats.get("num_embeddings")
        row = [
            table_name,
            format_mean_std_cell(compressed_tokens_stat, precision=num_embeds_precision, tablefmt=tablefmt),
            format_mean_std_cell(stats.get("information_gain_from_dataset"), precision=0, tablefmt=tablefmt),
        ]
        if not short:
            row += [
                format_mean_std_cell(stats.get("trajectory_length"), precision=0, tablefmt=tablefmt),
                # format_mean_std_cell(stats.get("steps_taken"), precision=2, tablefmt=tablefmt),
                format_mean_std_cell(stats.get("num_pca_for99_var"), precision=2, tablefmt=tablefmt),
                # stats.get("num_pca_for99_var_all_embeds", "nan"),
                # format_mean_std_cell(stats.get("num_random_projections_for99_var"), precision=1, tablefmt=tablefmt),
                # format_mean_std_cell(stats.get("information_gain"), precision=0, tablefmt=tablefmt),
                # format_embedding_statistics(stats.get("embedding_statistics"), precision=4, tablefmt=tablefmt),
            ]
        if steps_key is not None and not short:
            row += [format_mean_std_cell(stats.get(steps_key), precision=0, tablefmt=tablefmt)]
        if show_prefix_surprisal:
            prefix_stat = stats.get("prefix_surprisal")
            # Render "--" (not "nan") for rows that have no prefix, so paper lint stays green.
            row.append("--" if prefix_stat is None else format_mean_std_cell(prefix_stat, precision=2, tablefmt=tablefmt))
        table_data.append(row)

        if midrule_indicies is not None and i in midrule_indicies:
            table_data.append(["\\midrule REMOVE"] + [""] * (len(headers) - 1))

        i += 1

    result = tabulate(table_data, headers=headers, tablefmt=tablefmt, numalign="right", stralign="left")

    result = result.replace("\\textbackslash{}", "\\")
    result = result.replace("\\$", "$")
    result = result.replace("\\{", "{")
    result = result.replace("\\}", "}")
    result = result.replace("P-", "Pythia")
    result = result.replace("L3.2-", "Llama-3.2-")
    result = result.replace("L3.1-", "Llama-3.1-")

    result = re.sub(r"REMOVE.+", "", result)

    if tablefmt.startswith("latex"):
        result = hlines_to_booktabs(result)

    return result


def print_statistics_table(
    checkpoint_names: List[str],
    statistics: List[Dict[str, Any]],
    midrule_indicies,
    tablefmt: str = "grid",
    short: bool = False,
    compressed_tokens_key: str = "num_embeddings",
    steps_key: Optional[str] = None,
    show_prefix_surprisal: bool = False,
) -> None:
    """Print the formatted statistics table to stdout, framed by a banner."""
    result = format_statistics_table(
        checkpoint_names,
        statistics,
        midrule_indicies,
        tablefmt=tablefmt,
        short=short,
        compressed_tokens_key=compressed_tokens_key,
        steps_key=steps_key,
        show_prefix_surprisal=show_prefix_surprisal,
    )
    if not result:
        return

    print("\n" + "=" * 80)
    print("Progressive Embeddings Statistics")
    print("=" * 80)
    print(result)
    print("=" * 80 + "\n")


def print_pairwise_distances_table(
    checkpoint_names: List[str],
    l2_distances: np.ndarray,
    l1_distances: np.ndarray,
    cosine_distances: np.ndarray,
    tablefmt: str = "grid",
):
    """Print pairwise distances tables using tabulate.

    Args:
        checkpoint_names: List of experiment labels
        l2_distances: L2 distance matrix [n_experiments, n_experiments]
        l1_distances: L1 distance matrix [n_experiments, n_experiments]
        cosine_distances: Cosine distance matrix [n_experiments, n_experiments]
    """
    if len(checkpoint_names) < 2 or l2_distances.size == 0:
        return

    n = len(checkpoint_names)

    # L2 distances table
    print("\n" + "=" * 80)
    print("Pairwise L2 Distances Between Final Embeddings")
    print("=" * 80)
    l2_table_data = []
    for i in range(n):
        row = [checkpoint_names[i]]
        for j in range(n):
            if i == j:
                row.append("0.000")
            else:
                row.append(f"{l2_distances[i, j]:.4f}")
        l2_table_data.append(row)
    l2_headers = ["Experiment"] + checkpoint_names
    l2_table = tabulate(l2_table_data, headers=l2_headers, tablefmt=tablefmt, numalign="right", stralign="left")
    print(l2_table)

    # L1 distances table
    print("\n" + "=" * 80)
    print("Pairwise L1 Distances Between Final Embeddings")
    print("=" * 80)
    l1_table_data = []
    for i in range(n):
        row = [checkpoint_names[i]]
        for j in range(n):
            if i == j:
                row.append("0.000")
            else:
                row.append(f"{l1_distances[i, j]:.4f}")
        l1_table_data.append(row)
    l1_headers = ["Experiment"] + checkpoint_names
    l1_table = tabulate(l1_table_data, headers=l1_headers, tablefmt=tablefmt, numalign="right", stralign="left")
    print(l1_table)

    # Cosine distances table
    print("\n" + "=" * 80)
    print("Pairwise Cosine Distances Between Final Embeddings")
    print("=" * 80)
    cos_table_data = []
    for i in range(n):
        row = [checkpoint_names[i]]
        for j in range(n):
            if i == j:
                row.append("0.000")
            else:
                row.append(f"{cosine_distances[i, j]:.4f}")
        cos_table_data.append(row)
    cos_headers = ["Experiment"] + checkpoint_names
    cos_table = tabulate(cos_table_data, headers=cos_headers, tablefmt=tablefmt, numalign="right", stralign="left")
    print(cos_table)
    print("=" * 80 + "\n")

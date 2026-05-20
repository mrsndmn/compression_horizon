"""Post-hoc PCA-reconstruction ablation (paper Section 5.3, Figure 5).

For each Progressive sample we fit PCA on its full trajectory of stage
embeddings, reconstruct the final converged embedding from the top-k
components for each k in a grid, and evaluate teacher-forced reconstruction
accuracy on the same prefix the embedding was trained for.

Aggregates per-sample (k, accuracy) curves into a mean ± std curve and
writes ``pca_reconstruction.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.analysis import (
    cumulative_variance_ratio,
    fit_per_sample_pca,
    project_top_k,
    summarize_pca_curve,
)
from compression_horizon.utils.launch import (
    freeze_model_parameters,
    get_device,
    resolve_torch_dtype,
)


def _resolve_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _load_progressive_dataset(source_dir: str) -> Dataset:
    path = os.path.join(source_dir, "progressive_prefixes")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — PCA reconstruction needs a Progressive run.")
    return Dataset.load_from_disk(path)


def _select_final_converged(rows: list[dict]) -> dict:
    converged = [r for r in rows if r.get("final_convergence") == 1.0]
    candidates = converged if converged else rows
    return max(candidates, key=lambda r: r["stage_seq_len"])


def _build_trajectory(rows: list[dict]) -> torch.Tensor:
    """[n_stages, num_comp, hidden]; rows sorted by stage_index."""
    rows_sorted = sorted(rows, key=lambda r: int(r["stage_index"]))
    tensors = [torch.tensor(r["embedding"], dtype=torch.float32) for r in rows_sorted]
    if tensors[0].dim() == 1:
        tensors = [t.unsqueeze(0) for t in tensors]
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def _teacher_forced_accuracy(
    model,
    tokenizer,
    embedding: torch.Tensor,
    text: str,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Per-sample teacher-forced reconstruction accuracy on the prefix ``text``."""
    enc = tokenizer(text, truncation=True, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_emb = model.get_input_embeddings()(input_ids)
    comp = embedding.unsqueeze(0).to(token_emb.dtype).to(device)
    united_emb = torch.cat([comp, token_emb], dim=1)
    num_comp = embedding.shape[0]
    united_mask = torch.cat(
        [
            torch.ones((1, num_comp), dtype=attention_mask.dtype, device=device),
            attention_mask,
        ],
        dim=1,
    )
    outputs = model(inputs_embeds=united_emb, attention_mask=united_mask)
    seq_len = int(attention_mask.sum().item())
    pred = outputs.logits[0, num_comp - 1 : num_comp + seq_len - 1].argmax(dim=-1)
    target = input_ids[0, :seq_len]
    return float((pred == target).float().mean().item())


def _parse_k_grid(spec: str) -> list[int]:
    return sorted(set(int(x) for x in spec.split(",") if x.strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA reconstruction ablation (paper §5.3 / Figure 5).")
    parser.add_argument(
        "--source_dir",
        required=True,
        help="Artifacts dir from a Progressive train.py run.",
    )
    parser.add_argument("--model_checkpoint", required=True, help="HF model id used by the source run.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write pca_reconstruction.json (default: source_dir).",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Cap the number of samples evaluated.",
    )
    parser.add_argument(
        "--k_grid",
        default="1,2,4,8,12,16,24,32,48",
        help="Comma-separated list of #components to evaluate. The per-sample max_k (n_stages-1) is always added.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.source_dir
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    dtype = resolve_torch_dtype(args.dtype)
    attn_impl = _resolve_attn_implementation()
    print(f"Device: {device}; dtype: {dtype}; attn: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, dtype=dtype, attn_implementation=attn_impl)
    freeze_model_parameters(model)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = _load_progressive_dataset(args.source_dir)
    by_sample: dict[int, list[dict]] = defaultdict(list)
    for row in ds:
        by_sample[int(row["sample_id"])].append(row)
    sample_ids = sorted(by_sample.keys())
    if args.num_samples is not None:
        sample_ids = sample_ids[: args.num_samples]
    print(f"Analyzing {len(sample_ids)} samples")

    k_grid_user = _parse_k_grid(args.k_grid)

    per_sample_curves: list[dict] = []
    for sample_id in tqdm(sample_ids, desc="pca-recon"):
        rows = by_sample[sample_id]
        trajectory = _build_trajectory(rows)  # [n_stages, num_comp, hidden]
        n_stages = trajectory.shape[0]
        if n_stages < 2:
            continue  # nothing to fit
        max_k = n_stages - 1  # rank of centered matrix

        final_row = _select_final_converged(rows)
        e_star = torch.tensor(final_row["embedding"], dtype=torch.float32)
        if e_star.dim() == 1:
            e_star = e_star.unsqueeze(0)

        mean, components, singular = fit_per_sample_pca(trajectory)
        cum_variance = cumulative_variance_ratio(singular)  # [r], cum_variance[i] = variance covered by top-(i+1)

        # Evaluate at each k from the user grid (clipped by max_k) plus max_k itself.
        k_values: list[int] = []
        for k in k_grid_user:
            if 1 <= k <= max_k and k not in k_values:
                k_values.append(k)
        if max_k not in k_values:
            k_values.append(max_k)

        curve: list[dict] = []
        for k in sorted(k_values):
            e_k = project_top_k(e_star, mean, components, k)
            e_k_tensor = e_k.to(dtype).to(device)
            acc = _teacher_forced_accuracy(model, tokenizer, e_k_tensor, final_row["text"], device, dtype)
            # variance_ratio at k = cum_variance[k-1] (1-indexed → 0-indexed).
            var_ratio = float(cum_variance[k - 1].item()) if 1 <= k <= cum_variance.numel() else None
            curve.append({"k": int(k), "accuracy": acc, "variance_ratio": var_ratio})

        per_sample_curves.append(
            {
                "sample_id": int(sample_id),
                "n_stages": int(n_stages),
                "max_k": int(max_k),
                "final_stage_seq_len": int(final_row["stage_seq_len"]),
                "curve": curve,
            }
        )

    summary = summarize_pca_curve(per_sample_curves)

    output = {
        "config": {
            "source_dir": args.source_dir,
            "model_checkpoint": args.model_checkpoint,
            "num_samples": len(per_sample_curves),
            "k_grid": k_grid_user,
        },
        "summary": summary,
        "samples": per_sample_curves,
    }
    output_path = Path(output_dir) / "pca_reconstruction.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {output_path}")

    print()
    print("Curve (k → mean accuracy ± std | cum.variance ratio | n_samples):")
    for point in summary["curve"]:
        var_str = (
            f"{point['variance_ratio_mean']:.4f} ± {point['variance_ratio_std']:.4f}"
            if "variance_ratio_mean" in point
            else "          -          "
        )
        print(f"  k={point['k']:>3}  acc={point['mean']:.4f} ± {point['std']:.4f}  " f"var={var_str}  (n={point['n_samples']})")


if __name__ == "__main__":
    main()

"""Trajectory + accuracy-landscape figure (paper Section 5.1, Figure 3).

Reproduces the paper's Figure 3: a progressive-cramming trajectory projected
onto its first two PCA components, drawn over the *local accuracy landscape*
of that 2-D plane at several anchor prefix lengths.

Because PCA is a linear, invertible projection, every point ``(g1, g2)`` in
the plane maps back to a concrete embedding ``E = mean + g1*PC1 + g2*PC2``
whose teacher-forced reconstruction accuracy can be measured. Sampling a grid
of such points and evaluating accuracy yields the landscape; as the prefix
length grows the near-perfect-reconstruction basin shrinks.

This is PCA-specific: t-SNE / UMAP are not invertible, so they cannot carry an
accuracy landscape (use run_dimreduction.py for their scatter projections).

Outputs under ``--output_dir``:
    trajectory_landscape_sample<sid>.png   the figure
    trajectory_landscape.json              variance explained, anchor metadata
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from compression_horizon.analysis import (  # noqa: E402
    fit_per_sample_pca,
    plane_grid,
    reconstruct_from_plane,
)
from compression_horizon.utils.launch import (  # noqa: E402
    freeze_model_parameters,
    get_device,
    resolve_torch_dtype,
)

# Distinct hues for the per-anchor landscapes (paper uses ~5 anchors).
_ANCHOR_COLORS = ["#d73027", "#fc8d59", "#4575b4", "#91bfdb", "#542788", "#1a9850", "#999999"]
# Paper caps the landscape colour scale at 90 % accuracy.
_ACCURACY_CAP = 0.9


def _resolve_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _load_progressive_dataset(source_dir: str) -> Dataset:
    path = os.path.join(source_dir, "progressive_prefixes")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- the landscape figure needs a Progressive run.")
    return Dataset.load_from_disk(path)


def _stage_embedding(row: dict) -> torch.Tensor:
    """[num_comp, hidden] tensor for one stage."""
    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    return emb.unsqueeze(0) if emb.dim() == 1 else emb


def _parse_int_list(spec: str) -> list[int]:
    return sorted({int(x) for x in spec.split(",") if x.strip()})


@torch.no_grad()
def _teacher_forced_accuracy_batch(
    model,
    tokenizer,
    embeddings: torch.Tensor,  # [batch, num_comp, hidden]
    text: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    """Per-embedding teacher-forced reconstruction accuracy on ``text``."""
    enc = tokenizer(text, truncation=True, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    seq_len = int(enc["attention_mask"].sum().item())
    token_emb = model.get_input_embeddings()(input_ids).to(dtype)  # [1, L, hidden]
    target = input_ids[0, :seq_len]
    num_comp = embeddings.shape[1]

    accuracies: list[float] = []
    for start in range(0, embeddings.shape[0], batch_size):
        comp = embeddings[start : start + batch_size].to(dtype).to(device)
        b = comp.shape[0]
        united = torch.cat([comp, token_emb.expand(b, -1, -1)], dim=1)
        mask = torch.ones((b, num_comp + seq_len), dtype=torch.long, device=device)
        logits = model(inputs_embeds=united, attention_mask=mask).logits
        pred = logits[:, num_comp - 1 : num_comp + seq_len - 1].argmax(dim=-1)  # [b, L]
        correct = (pred == target.unsqueeze(0)).float().mean(dim=1)
        accuracies.extend(correct.detach().cpu().tolist())
    return np.asarray(accuracies, dtype=np.float64)


def _anchor_colormap(color: str) -> LinearSegmentedColormap:
    """Transparent -> opaque ramp in one hue, so saturation tracks accuracy."""
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list("anchor", [(r, g, b, 0.0), (r, g, b, 0.85)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory accuracy-landscape figure (paper Figure 3).")
    parser.add_argument("--source_dir", required=True, help="Artifacts dir from a Progressive train.py run.")
    parser.add_argument("--model_checkpoint", required=True, help="HF model id used by the source run.")
    parser.add_argument("--output_dir", default=None, help="Where to write outputs (default: source_dir).")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="Sample to render (default: the one with the most stages).",
    )
    parser.add_argument(
        "--prefix_lengths",
        default="100,200,400,800,1000",
        help="Anchor prefix lengths for the accuracy landscapes (paper Figure 3 defaults).",
    )
    parser.add_argument("--grid_resolution", type=int, default=50, help="Landscape grid is resolution x resolution.")
    parser.add_argument("--batch_size", type=int, default=16, help="Grid embeddings evaluated per forward pass.")
    parser.add_argument("--margin", type=float, default=0.15, help="Relative padding of the grid around the trajectory.")
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

    if args.sample_id is not None:
        if args.sample_id not in by_sample:
            raise ValueError(f"--sample_id {args.sample_id} not found in {sorted(by_sample)}")
        sample_id = args.sample_id
    else:
        sample_id = max(by_sample, key=lambda s: len(by_sample[s]))
    rows = sorted(by_sample[sample_id], key=lambda r: int(r["stage_index"]))
    n_stages = len(rows)
    if n_stages < 3:
        raise RuntimeError(f"Sample {sample_id} has {n_stages} stages; need >= 3 for a 2-D PCA plane.")
    print(f"Sample {sample_id}: {n_stages} stages")

    stages = torch.stack([_stage_embedding(r) for r in rows], dim=0)  # [n, num_comp, hidden]
    num_comp, hidden = stages.shape[1], stages.shape[2]

    mean, components, singular = fit_per_sample_pca(stages)
    basis_2 = components[:2]  # [2, flat_dim]
    variance = singular.to(torch.float64) ** 2
    var_explained_2 = float(variance[:2].sum().item() / variance.sum().item())
    print(f"First 2 PCA components explain {var_explained_2 * 100:.1f}% of trajectory variance")

    flat = stages.reshape(n_stages, -1).to(torch.float64)
    coords = ((flat - mean) @ basis_2.t()).numpy()  # [n, 2]
    grid_xy, extent = plane_grid(coords, resolution=args.grid_resolution, margin=args.margin)

    # Map each requested prefix length to the nearest available stage.
    stage_seq_lens = [int(r["stage_seq_len"]) for r in rows]
    anchors: list[dict] = []
    seen_stage_idx: set[int] = set()
    for target_len in _parse_int_list(args.prefix_lengths):
        nearest = min(range(n_stages), key=lambda i: abs(stage_seq_lens[i] - target_len))
        if nearest in seen_stage_idx:
            continue
        seen_stage_idx.add(nearest)
        anchors.append({"requested_len": target_len, "stage_index": nearest, "actual_len": stage_seq_lens[nearest]})
    anchors.sort(key=lambda a: a["actual_len"])
    print(f"Anchor prefix lengths (actual): {[a['actual_len'] for a in anchors]}")

    mean_np = mean.numpy()
    basis_np = basis_2.numpy()
    grid_embeddings = reconstruct_from_plane(grid_xy, mean_np, basis_np)  # [G, flat_dim]
    grid_embeddings = torch.tensor(grid_embeddings, dtype=torch.float32).reshape(-1, num_comp, hidden)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    legend_handles: list[Patch] = []
    for anchor, color in zip(anchors, _ANCHOR_COLORS):
        text = rows[anchor["stage_index"]]["text"]
        accuracy = _teacher_forced_accuracy_batch(model, tokenizer, grid_embeddings, text, device, dtype, args.batch_size)
        field = np.clip(accuracy, 0.0, _ACCURACY_CAP).reshape(args.grid_resolution, args.grid_resolution)
        anchor["max_accuracy"] = float(accuracy.max())
        anchor["basin_fraction"] = float((accuracy >= _ACCURACY_CAP).mean())
        ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=_anchor_colormap(color),
            vmin=0.0,
            vmax=_ACCURACY_CAP,
            aspect="auto",
            interpolation="bilinear",
        )
        legend_handles.append(Patch(facecolor=color, label=f"L = {anchor['actual_len']}"))
        print(
            f"  L={anchor['actual_len']}: max acc={accuracy.max():.3f}, "
            f"basin(>={_ACCURACY_CAP:.0%})={anchor['basin_fraction']:.3f}"
        )

    ax.plot(coords[:, 0], coords[:, 1], color="0.35", linewidth=0.7, zorder=3)
    ax.scatter(coords[:, 0], coords[:, 1], c="black", s=10, zorder=4)
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    ax.set_title(
        f"Траектория оптимизации и ландшафт точности\n" f"(первые 2 компоненты PCA, {var_explained_2 * 100:.1f}% дисперсии)"
    )
    ax.legend(handles=legend_handles, title="длина префикса", loc="best", framealpha=0.9)
    fig.tight_layout()
    figure_path = Path(output_dir) / f"trajectory_landscape_sample{sample_id}.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {figure_path}")

    output = {
        "config": {
            "source_dir": args.source_dir,
            "model_checkpoint": args.model_checkpoint,
            "sample_id": sample_id,
            "grid_resolution": args.grid_resolution,
            "accuracy_cap": _ACCURACY_CAP,
        },
        "n_stages": n_stages,
        "variance_explained_first_2": var_explained_2,
        "anchors": anchors,
    }
    output_path = Path(output_dir) / "trajectory_landscape.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

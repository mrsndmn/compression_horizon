import argparse
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")


def load_progressive_dataset(dataset_path: str) -> Dataset:
    return Dataset.load_from_disk(dataset_path)


def _infer_output_dir(dataset_path: str) -> str:
    exp_dir = os.path.dirname(dataset_path)
    return os.path.join(exp_dir, "visualizations")


def _savefig_with_pdf(outfile: str, dpi: int = 200) -> None:
    plt.savefig(outfile, dpi=dpi)
    if outfile.lower().endswith(".png"):
        plt.savefig(outfile[:-4] + ".pdf", dpi=dpi)


def _as_2d_embedding(row: Dict[str, Any]) -> torch.Tensor:
    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    if emb.ndim != 2:
        raise ValueError(f"Expected embedding to be 1D or 2D, got shape={tuple(emb.shape)}")
    return emb


def flatten_embedding(row: Dict[str, Any]) -> np.ndarray:
    emb = _as_2d_embedding(row)
    return emb.reshape(-1).detach().cpu().numpy()


def _filter_rows_for_sample(ds: Dataset, sample_id: int) -> List[Dict[str, Any]]:
    # Drop heavy columns if present (keeps runtime + memory lower)
    drop_cols = [c for c in ["orig_embedding", "initialization_embedding"] if c in ds.column_names]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    rows: List[Dict[str, Any]] = []
    for i in tqdm(range(len(ds)), desc=f"Filtering sample_id={sample_id}"):
        r = ds[i]
        if int(r.get("sample_id", -1)) != int(sample_id):
            continue
        rows.append(r)
    return rows


def _pick_reference_row(rows_sorted: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Reference: the longest sequence length (matches existing landscape logic)
    return max(rows_sorted, key=lambda r: int(r.get("stage_seq_len", -1)))


def _pick_text(rows_sorted: List[Dict[str, Any]]) -> str:
    # Prefer text from the longest stage; fallback to any non-empty text.
    ref = _pick_reference_row(rows_sorted)
    t = ref.get("text", "")
    if isinstance(t, str) and t.strip():
        return t
    for r in rows_sorted[::-1]:
        t = r.get("text", "")
        if isinstance(t, str) and t.strip():
            return t
    raise ValueError("No non-empty `text` found for this sample_id in the dataset.")


def _most_common_str(values: List[str]) -> Optional[str]:
    values = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _load_model_and_tokenizer(
    model_checkpoint: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    # Be tolerant to older transformers / models not supporting attn_implementation.
    kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if device.type == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint, **kwargs).to(device)
    except Exception:
        # Retry without attn_implementation if incompatible
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint, **kwargs).to(device)

    model.eval()
    tok = AutoTokenizer.from_pretrained(model_checkpoint)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return model, tok


def _compute_accuracy_batch(
    compression_embeddings_flat: torch.Tensor,
    original_shape: Tuple[int, int],
    model: AutoModelForCausalLM,
    device: torch.device,
    input_ids: torch.Tensor,
    input_text_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    accuracies: List[np.ndarray] = []
    num_embeddings = int(compression_embeddings_flat.shape[0])
    mem_tokens = int(original_shape[0])
    attn_bs0 = attention_mask
    denom = attn_bs0.sum(dim=-1).clamp_min(1).float()

    for batch_start in range(0, num_embeddings, batch_size):
        batch_end = min(batch_start + batch_size, num_embeddings)
        batch_embeddings = compression_embeddings_flat[batch_start:batch_end]

        with torch.no_grad():
            bs = int(batch_embeddings.shape[0])
            comp = batch_embeddings.reshape(bs, original_shape[0], original_shape[1]).to(device)
            text_embeds_bs = input_text_embeds.expand(bs, -1, -1)
            inputs_embeds = torch.cat([comp, text_embeds_bs], dim=1)

            comp_attention = torch.ones((bs, mem_tokens), device=device, dtype=attention_mask.dtype)
            attn_bs = attention_mask.expand(bs, -1)
            extended_attention_mask = torch.cat([comp_attention, attn_bs], dim=1)

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask, use_cache=False)
            pred_logits = outputs.logits[:, mem_tokens - 1 : -1]
            pred_tokens = pred_logits.argmax(dim=-1)

            correct = (pred_tokens == input_ids.expand(bs, -1)).to(torch.float32)
            masked_correct = correct * attn_bs.to(torch.float32)
            acc = masked_correct.sum(dim=-1) / denom.expand(bs)
            accuracies.append(acc.float().cpu().numpy())

    return np.concatenate(accuracies, axis=0)


def _expanded_bounds(vmin: float, vmax: float, padding: float) -> Tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    span = vmax - vmin
    pad = float(padding * span) if span > 0 else 1.0
    return float(vmin - pad), float(vmax + pad)


def _plot_surface(
    X_mesh: np.ndarray,
    Y_mesh: np.ndarray,
    Z_mesh: np.ndarray,
    coords_xy: np.ndarray,
    stage_labels: List[str],
    current_idx: int,
    title: str,
    colorbar_label: str,
    outfile: str,
    cmap: str,
    x_label: str = "PC1",
    y_label: str = "PC2",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(9.5, 7.5))
    im = plt.pcolormesh(X_mesh, Y_mesh, Z_mesh, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)

    # Overlay trajectory
    plt.plot(coords_xy[:, 0], coords_xy[:, 1], color="white", alpha=0.65, linewidth=1.5)
    plt.scatter(coords_xy[:, 0], coords_xy[:, 1], s=45, c="grey", alpha=0.9, edgecolors="black", linewidths=0.4)
    if coords_xy.shape[0] >= 1 and 0 <= current_idx < coords_xy.shape[0]:
        plt.scatter(
            coords_xy[current_idx, 0],
            coords_xy[current_idx, 1],
            s=220,
            marker="*",
            c="red",
            edgecolors="black",
            linewidths=1.0,
            zorder=20,
        )

    # Optional labels (kept light to avoid clutter)
    for i, lab in enumerate(stage_labels):
        if i == 0 or i == len(stage_labels) - 1 or i == current_idx:
            plt.text(coords_xy[i, 0], coords_xy[i, 1], lab, fontsize=10, color="black")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    _savefig_with_pdf(outfile, dpi=220)
    plt.close()


def _render_accuracy_frame(
    X_mesh: np.ndarray,
    Y_mesh: np.ndarray,
    Z_acc: np.ndarray,
    coords_xy: np.ndarray,
    current_idx: int,
    sample_id: int,
    x_label: str,
    y_label: str,
) -> np.ndarray:
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.5))

    im = ax.pcolormesh(X_mesh, Y_mesh, Z_acc, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Teacher-forced accuracy")
    ax.plot(coords_xy[:, 0], coords_xy[:, 1], color="white", alpha=0.65, linewidth=1.5)
    ax.scatter(coords_xy[:, 0], coords_xy[:, 1], s=35, c="grey", alpha=0.9, edgecolors="black", linewidths=0.4)
    if coords_xy.shape[0] >= 1 and 0 <= current_idx < coords_xy.shape[0]:
        ax.scatter(
            coords_xy[current_idx, 0],
            coords_xy[current_idx, 1],
            s=220,
            marker="*",
            c="red",
            edgecolors="black",
            linewidths=1.0,
            zorder=20,
        )
    ax.set_title("Accuracy")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axis("equal")

    fig.suptitle(f"Accuracy landscape (sample_id={sample_id})")
    fig.tight_layout()
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = rgba[:, :, :3].copy()
    else:
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        img = argb[:, :, 1:4].copy()
    plt.close(fig)
    return img


def _render_accuracy_grid_frame(
    pair_indices: List[Tuple[int, int]],
    grids_x: List[np.ndarray],
    grids_y: List[np.ndarray],
    acc_surfaces: List[np.ndarray],
    coords: np.ndarray,
    explained_variance_ratio: np.ndarray,
    current_idx: int,
    sample_id: int,
) -> np.ndarray:
    # 2x3 grid for up to 6 pairs; falls back gracefully for fewer pairs.
    n_pairs = int(len(pair_indices))
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")

    n_cols = 3 if n_pairs > 1 else 1
    n_rows = int((n_pairs + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20.5 if n_pairs > 1 else 9.5, 6.2 * n_rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    # Shared colorbar across subplots
    mappable = None
    for k, (i, j) in enumerate(pair_indices):
        ax = axes[k]
        X_mesh = grids_x[k]
        Y_mesh = grids_y[k]
        Z_acc = acc_surfaces[k]
        mappable = ax.pcolormesh(X_mesh, Y_mesh, Z_acc, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)

        coords_pair = coords[:, [i, j]]
        ax.plot(coords_pair[:, 0], coords_pair[:, 1], color="white", alpha=0.65, linewidth=1.2)
        ax.scatter(coords_pair[:, 0], coords_pair[:, 1], s=22, c="grey", alpha=0.9, edgecolors="black", linewidths=0.3)
        if coords_pair.shape[0] >= 1 and 0 <= current_idx < coords_pair.shape[0]:
            ax.scatter(
                coords_pair[current_idx, 0],
                coords_pair[current_idx, 1],
                s=180,
                marker="*",
                c="red",
                edgecolors="black",
                linewidths=0.9,
                zorder=20,
            )

        vx = float(explained_variance_ratio[i]) if i < len(explained_variance_ratio) else float("nan")
        vy = float(explained_variance_ratio[j]) if j < len(explained_variance_ratio) else float("nan")
        ax.set_title(f"PC{i+1} {{{vx:.3f}}} vs PC{j+1} {{{vy:.3f}}}", fontsize=12)
        ax.set_xlabel(f"PC{i+1} {{{vx:.3f}}}")
        ax.set_ylabel(f"PC{j+1} {{{vy:.3f}}}")
        ax.axis("equal")

    # Hide unused axes
    for k in range(n_pairs, len(axes)):
        axes[k].axis("off")

    if mappable is not None:
        fig.colorbar(mappable, ax=axes[:n_pairs].tolist(), label="Teacher-forced accuracy", shrink=0.9)

    fig.suptitle(f"Accuracy landscapes (sample_id={sample_id})")
    fig.tight_layout()
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = rgba[:, :, :3].copy()
    else:
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        img = argb[:, :, 1:4].copy()
    plt.close(fig)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw accuracy landscape over PC1-PC2 for one progressive sample trajectory")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to progressive_prefixes dataset directory")
    parser.add_argument("--sample_id", type=int, required=True, help="Single sample_id to visualize")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save figures/npz")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Override HF model checkpoint")
    parser.add_argument("--mesh_resolution", type=int, default=40, help="Grid resolution for PC1/PC2")
    parser.add_argument("--padding", type=float, default=0.15, help="Fractional padding added to PCA bounds")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model forward passes")
    parser.add_argument(
        "--pca4",
        default=False,
        action="store_true",
        help="If set, compute accuracy landscapes for all pairs among the first 4 PCs (up to 6 pairs).",
    )
    parser.add_argument(
        "--num-frames",
        "--num_frames",
        type=int,
        default=1,
        help="If 1 (default), save a single landscape image. If >1, save a GIF over uniformly sampled trajectory points.",
    )
    parser.add_argument(
        "--anchor-indices",
        "--anchor_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit anchor indices (trajectory indices). If provided, overrides uniform sampling. "
        "The number of indices must match --num-frames.",
    )
    parser.add_argument(
        "--neighborhood",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-frame neighborhood radius in PCA space. If provided, for each frame we compute the landscape "
        "only within a square [center-r, center+r] around the current anchor point (for each plotted PC pair). "
        "The number of radii must match --num-frames.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (GPU: bfloat16/float16 recommended; CPU: float32 recommended)",
    )
    parser.add_argument(
        "--save-npz-only",
        "--save_npz_only",
        dest="save_npz_only",
        action="store_true",
        help="Multi-frame only: write just the dense NPZ, skipping the per-frame PNG renders and "
        "the GIF (and the imageio import). Use for cluster bundle jobs whose output is merged/animated "
        "downstream -- avoids wasted rendering and an imageio dependency.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or _infer_output_dir(args.dataset_path)
    os.makedirs(out_dir, exist_ok=True)

    ds = load_progressive_dataset(args.dataset_path)
    rows = _filter_rows_for_sample(ds, sample_id=args.sample_id)
    if not rows:
        raise ValueError(f"No rows found for sample_id={args.sample_id} in {args.dataset_path}")

    # Sort trajectory
    rows_sorted = sorted(rows, key=lambda r: int(r.get("stage_index", 0)))
    stage_labels = [f"L{int(r.get('stage_seq_len', -1))}" for r in rows_sorted]

    # Infer model checkpoint
    inferred_model = _most_common_str([str(r.get("model_checkpoint", "")).strip() for r in rows_sorted])
    model_checkpoint = args.model_checkpoint or inferred_model
    if model_checkpoint is None:
        raise ValueError("Could not infer `model_checkpoint` from dataset; please pass --model_checkpoint")

    # Device/dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    if device.type != "cuda" and torch_dtype != torch.float32:
        print("Warning: non-float32 dtype on CPU may be slow or unsupported; consider --dtype float32")

    model, tok = _load_model_and_tokenizer(model_checkpoint=model_checkpoint, device=device, torch_dtype=torch_dtype)

    # Reference text + embeddings shape
    reference_text = _pick_text(rows_sorted)
    reference_row = _pick_reference_row(rows_sorted)
    reference_emb = _as_2d_embedding(reference_row)
    original_shape = (int(reference_emb.shape[0]), int(reference_emb.shape[1]))

    # PCA input matrix
    X = np.stack([flatten_embedding(r) for r in rows_sorted], axis=0)
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 stages for PCA/landscape.")

    n_components = int(min(X.shape[0] - 1, X.shape[1]))
    if n_components < 2:
        raise ValueError(f"PCA requires >=2 components; got n_components={n_components}")

    # # Important: always fit PCA with up to 4 components, even if we only plot PC1-PC2.
    # # Otherwise, when plotting only one pair (PC1,PC2), the per-frame anchor would be
    # # overwritten entirely by the mesh grid and the landscape would not change across frames.
    # # Fitting with >=3 components lets us keep the remaining PCs fixed to the current anchor.
    # pca_dim = int(min(4, n_components))
    pca_dim = n_components
    if pca_dim < 2:
        raise ValueError(f"Need at least 2 PCA components; got pca_dim={pca_dim}")

    pca = PCA(n_components=pca_dim, random_state=42)
    coords = pca.fit_transform(X)

    # Default anchor: penultimate stage embedding in the sorted trajectory
    penultimate_idx = max(0, len(rows_sorted) - 2)

    if bool(args.pca4):
        pair_indices = [(i, j) for i in range(pca_dim) for j in range(i + 1, pca_dim)]
    else:
        pair_indices = [(0, 1)]
    if not pair_indices:
        raise ValueError("No PCA pairs available to plot.")

    # Precompute tokenization once. Accuracy will be sliced per-frame to match
    # the starred stage's stage_seq_len.
    enc = tok(reference_text, truncation=True, padding=False, return_tensors="pt")
    input_ids_full = enc["input_ids"].to(device)
    attention_mask_full = enc["attention_mask"].to(device)
    input_embeddings_layer = model.get_input_embeddings()
    with torch.no_grad():
        input_text_embeds_full = input_embeddings_layer(input_ids_full).to(dtype=torch_dtype)

    def _slice_inputs_for_seq_len(seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use attention_mask_full to determine max available non-pad length.
        max_len = int(attention_mask_full.sum(dim=-1).max().item())
        use_len = int(min(max(int(seq_len), 1), max_len))
        return (
            input_ids_full[:, :use_len],
            input_text_embeds_full[:, :use_len, :],
            attention_mask_full[:, :use_len],
        )

    def _make_mesh_for_pair(i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        lo_i, hi_i = _expanded_bounds(float(coords[:, i].min()), float(coords[:, i].max()), float(args.padding))
        lo_j, hi_j = _expanded_bounds(float(coords[:, j].min()), float(coords[:, j].max()), float(args.padding))
        x_range = np.linspace(lo_i, hi_i, int(args.mesh_resolution))
        y_range = np.linspace(lo_j, hi_j, int(args.mesh_resolution))
        return np.meshgrid(x_range, y_range)

    def _make_mesh_for_pair_centered(anchor_coords: np.ndarray, i: int, j: int, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        r = float(radius)
        if not np.isfinite(r) or r <= 0:
            raise ValueError(f"--neighborhood radii must be positive finite floats, got {radius}")
        cx = float(anchor_coords[i])
        cy = float(anchor_coords[j])
        x_range = np.linspace(cx - r, cx + r, int(args.mesh_resolution))
        y_range = np.linspace(cy - r, cy + r, int(args.mesh_resolution))
        return np.meshgrid(x_range, y_range)

    def _compute_accuracy_surface_for_anchor_pair(
        anchor_coords: np.ndarray, seq_len: int, i: int, j: int, X_mesh: np.ndarray, Y_mesh: np.ndarray
    ) -> np.ndarray:
        mesh_points = np.repeat(anchor_coords.reshape(1, -1), X_mesh.size, axis=0)
        mesh_points[:, i] = X_mesh.reshape(-1)
        mesh_points[:, j] = Y_mesh.reshape(-1)
        reconstructed = pca.inverse_transform(mesh_points).astype(np.float32, copy=False)
        reconstructed_t = torch.tensor(reconstructed, dtype=torch.float32)

        input_ids, input_text_embeds, attention_mask = _slice_inputs_for_seq_len(seq_len)
        accuracies = _compute_accuracy_batch(
            compression_embeddings_flat=reconstructed_t.to(dtype=torch_dtype),
            original_shape=original_shape,
            model=model,
            device=device,
            input_ids=input_ids,
            input_text_embeds=input_text_embeds,
            attention_mask=attention_mask,
            batch_size=int(args.batch_size),
        )
        return accuracies.reshape(X_mesh.shape)

    num_frames = int(args.num_frames)
    if num_frames <= 1:
        if args.anchor_indices is not None:
            if len(args.anchor_indices) != 1:
                raise ValueError("--anchor-indices must have exactly 1 value when --num-frames <= 1")
            current_idx = int(args.anchor_indices[0])
        else:
            current_idx = int(penultimate_idx)
        radius_single = None
        if args.neighborhood is not None:
            if len(args.neighborhood) != 1:
                raise ValueError("--neighborhood must have exactly 1 value when --num-frames <= 1")
            radius_single = float(args.neighborhood[0])
        if current_idx < 0 or current_idx >= int(coords.shape[0]):
            raise ValueError(f"anchor index out of range: {current_idx} (valid: 0..{int(coords.shape[0]) - 1})")
        anchor_coords = coords[current_idx].copy()
        current_seq_len = int(rows_sorted[current_idx].get("stage_seq_len", 1))
        grids_x: List[np.ndarray] = []
        grids_y: List[np.ndarray] = []
        acc_surfaces: List[np.ndarray] = []

        for i, j in pair_indices:
            if radius_single is None:
                X_mesh, Y_mesh = _make_mesh_for_pair(i, j)
            else:
                X_mesh, Y_mesh = _make_mesh_for_pair_centered(anchor_coords, i, j, radius=radius_single)
            Z_acc = _compute_accuracy_surface_for_anchor_pair(
                anchor_coords, seq_len=current_seq_len, i=i, j=j, X_mesh=X_mesh, Y_mesh=Y_mesh
            )
            grids_x.append(X_mesh)
            grids_y.append(Y_mesh)
            acc_surfaces.append(Z_acc)

            coords_pair = coords[:, [i, j]]
            out_name = f"landscape_accuracy_pc{i+1}_pc{j+1}.png"
            acc_out = os.path.join(out_dir, out_name)
            vx = float(pca.explained_variance_ratio_[i]) if i < len(pca.explained_variance_ratio_) else float("nan")
            vy = float(pca.explained_variance_ratio_[j]) if j < len(pca.explained_variance_ratio_) else float("nan")
            _plot_surface(
                X_mesh=X_mesh,
                Y_mesh=Y_mesh,
                Z_mesh=Z_acc,
                coords_xy=coords_pair,
                stage_labels=stage_labels,
                current_idx=current_idx,
                title=f"Accuracy landscape on PC{i+1} {{{vx:.3f}}} - PC{j+1} {{{vy:.3f}}} (sample_id={args.sample_id})",
                colorbar_label="Teacher-forced accuracy",
                outfile=acc_out,
                cmap="magma",
                x_label=f"PC{i+1} {{{vx:.3f}}}",
                y_label=f"PC{j+1} {{{vy:.3f}}}",
                vmin=0.0,
                vmax=1.0,
            )
            print(f"Saved: {acc_out}")

        # Save NPZ (single frame)
        npz_path = os.path.join(out_dir, "landscape_pca_pairs.npz")
        np.savez_compressed(
            npz_path,
            pair_indices=np.array(pair_indices, dtype=np.int64),
            grid_x=np.stack(grids_x, axis=0),
            grid_y=np.stack(grids_y, axis=0),
            accuracy=np.stack(acc_surfaces, axis=0),
            coords=coords,
            stage_index=np.array([int(r.get("stage_index", 0)) for r in rows_sorted], dtype=np.int64),
            stage_seq_len=np.array([int(r.get("stage_seq_len", -1)) for r in rows_sorted], dtype=np.int64),
            current_idx=np.array([current_idx], dtype=np.int64),
            current_seq_len=np.array([current_seq_len], dtype=np.int64),
            anchor_coords=anchor_coords,
            neighborhood_radius=(
                np.array([radius_single], dtype=np.float32) if radius_single is not None else np.array([], dtype=np.float32)
            ),
            explained_variance_ratio=pca.explained_variance_ratio_,
            model_checkpoint=np.array([model_checkpoint]),
            dataset_path=np.array([args.dataset_path]),
            sample_id=np.array([int(args.sample_id)], dtype=np.int64),
            created_at=np.array([datetime.now().isoformat()]),
        )
        print(f"Saved NPZ: {npz_path}")
        return

    # Multi-frame GIF
    if not args.save_npz_only:
        import imageio.v2 as imageio

    n_traj = int(coords.shape[0])
    if args.anchor_indices is not None:
        if len(args.anchor_indices) != num_frames:
            raise ValueError(
                f"--anchor-indices must have exactly --num-frames values: got {len(args.anchor_indices)} vs {num_frames}"
            )
        sampled = np.array([int(x) for x in args.anchor_indices], dtype=int)
    else:
        sampled = np.linspace(0, max(n_traj - 1, 0), num=num_frames, dtype=int)
        sampled = np.unique(sampled)
        if sampled.size == 0:
            sampled = np.array([0], dtype=int)

    if sampled.size > 0:
        if int(sampled.min()) < 0 or int(sampled.max()) >= n_traj:
            raise ValueError(
                f"--anchor-indices out of range: min={int(sampled.min())}, max={int(sampled.max())}, n_traj={n_traj}"
            )

    radii = None
    if args.neighborhood is not None:
        if len(args.neighborhood) != num_frames:
            raise ValueError(
                f"--neighborhood must have exactly --num-frames values: got {len(args.neighborhood)} vs {num_frames}"
            )
        radii = np.array([float(x) for x in args.neighborhood], dtype=np.float32)
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError(f"--neighborhood radii must be positive finite floats, got {args.neighborhood}")

    anchors: List[np.ndarray] = []
    acc_stack_by_pair: Dict[Tuple[int, int], List[np.ndarray]] = {p: [] for p in pair_indices}
    frames: List[np.ndarray] = []

    sampled_seq_lens: List[int] = []
    grids_x: List[np.ndarray] = []
    grids_y: List[np.ndarray] = []
    grids_x_per_frame: List[np.ndarray] = []
    grids_y_per_frame: List[np.ndarray] = []
    if radii is None:
        for i, j in pair_indices:
            X_mesh, Y_mesh = _make_mesh_for_pair(i, j)
            grids_x.append(X_mesh)
            grids_y.append(Y_mesh)

    for frame_k, idx in enumerate(tqdm(sampled.tolist(), desc="Computing landscapes frames")):
        anchor_coords = coords[int(idx)].copy()
        seq_len = int(rows_sorted[int(idx)].get("stage_seq_len", 1))
        sampled_seq_lens.append(seq_len)
        anchors.append(anchor_coords)

        acc_surfaces: List[np.ndarray] = []
        grids_x_this: List[np.ndarray] = []
        grids_y_this: List[np.ndarray] = []
        radius_k = float(radii[frame_k]) if radii is not None else None
        for k, (i, j) in enumerate(pair_indices):
            if radius_k is None:
                X_mesh = grids_x[k]
                Y_mesh = grids_y[k]
            else:
                X_mesh, Y_mesh = _make_mesh_for_pair_centered(anchor_coords, i, j, radius=radius_k)
            grids_x_this.append(X_mesh)
            grids_y_this.append(Y_mesh)
            Z_acc = _compute_accuracy_surface_for_anchor_pair(
                anchor_coords, seq_len=seq_len, i=i, j=j, X_mesh=X_mesh, Y_mesh=Y_mesh
            )
            acc_stack_by_pair[(i, j)].append(Z_acc)
            acc_surfaces.append(Z_acc)
        if radius_k is not None:
            grids_x_per_frame.append(np.stack(grids_x_this, axis=0))
            grids_y_per_frame.append(np.stack(grids_y_this, axis=0))

        if args.save_npz_only:
            continue
        if len(pair_indices) == 1:
            (i0, j0) = pair_indices[0]
            vx = float(pca.explained_variance_ratio_[i0]) if i0 < len(pca.explained_variance_ratio_) else float("nan")
            vy = float(pca.explained_variance_ratio_[j0]) if j0 < len(pca.explained_variance_ratio_) else float("nan")
            frame = _render_accuracy_frame(
                X_mesh=grids_x_this[0],
                Y_mesh=grids_y_this[0],
                Z_acc=acc_surfaces[0],
                coords_xy=coords[:, [i0, j0]],
                current_idx=int(idx),
                sample_id=int(args.sample_id),
                x_label=f"PC{i0+1} {{{vx:.3f}}}",
                y_label=f"PC{j0+1} {{{vy:.3f}}}",
            )
        else:
            frame = _render_accuracy_grid_frame(
                pair_indices=pair_indices,
                grids_x=grids_x_this,
                grids_y=grids_y_this,
                acc_surfaces=acc_surfaces,
                coords=coords,
                explained_variance_ratio=pca.explained_variance_ratio_,
                current_idx=int(idx),
                sample_id=int(args.sample_id),
            )
        frames.append(frame)

    if not args.save_npz_only:
        if len(pair_indices) == 1:
            (i0, j0) = pair_indices[0]
            gif_path = os.path.join(out_dir, f"landscape_accuracy_pc{i0+1}_pc{j0+1}.gif")
        else:
            gif_path = os.path.join(out_dir, "landscape_accuracy_pca_pairs.gif")
        imageio.mimsave(gif_path, frames, duration=1000, loop=0)
        print(f"Saved GIF: {gif_path}")

    # Save NPZ (multi-frame)
    acc_stack = np.stack([np.stack(acc_stack_by_pair[p], axis=0) for p in pair_indices], axis=1)
    npz_path = os.path.join(out_dir, "landscape_pca_pairs.npz")
    if radii is None:
        grid_x_out = np.stack(grids_x, axis=0)
        grid_y_out = np.stack(grids_y, axis=0)
    else:
        grid_x_out = np.stack(grids_x_per_frame, axis=0)
        grid_y_out = np.stack(grids_y_per_frame, axis=0)
    np.savez_compressed(
        npz_path,
        pair_indices=np.array(pair_indices, dtype=np.int64),
        grid_x=grid_x_out,
        grid_y=grid_y_out,
        accuracy=acc_stack,
        coords=coords,
        stage_index=np.array([int(r.get("stage_index", 0)) for r in rows_sorted], dtype=np.int64),
        stage_seq_len=np.array([int(r.get("stage_seq_len", -1)) for r in rows_sorted], dtype=np.int64),
        sampled_indices=sampled.astype(np.int64),
        sampled_seq_len=np.array(sampled_seq_lens, dtype=np.int64),
        anchor_coords=np.stack(anchors, axis=0),
        neighborhood_radius=(radii if radii is not None else np.array([], dtype=np.float32)),
        explained_variance_ratio=pca.explained_variance_ratio_,
        model_checkpoint=np.array([model_checkpoint]),
        dataset_path=np.array([args.dataset_path]),
        sample_id=np.array([int(args.sample_id)], dtype=np.int64),
        created_at=np.array([datetime.now().isoformat()]),
    )
    print(f"Saved NPZ: {npz_path}")


if __name__ == "__main__":
    main()

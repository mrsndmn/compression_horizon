"""
Compute accuracy-basin area in PC1-PC2 space for multiple samples of a
progressive-cramming experiment, then plot basin area vs normalised stage.

Usage (GPU required for recomputation):
    python scripts/paper/plot_basin_area_vs_stage.py \
        --dataset_path artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/progressive_prefixes \
        --sample_ids 0 1 2 --num_anchors 8 --mesh_resolution 60 --batch_size 32 \
        --output artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/visualizations/basin_area_vs_stage_normalised.png

If a precomputed NPZ exists for a sample it is reused (pass --recompute to force).
"""

import argparse
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Helpers shared with visualize_landscale_2pca.py (inlined to be standalone)
# ---------------------------------------------------------------------------


def _load_dataset(dataset_path: str):
    from datasets import Dataset

    return Dataset.load_from_disk(dataset_path)


def _as_2d_embedding(row: Dict[str, Any]):
    import torch

    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    return emb


def _flatten_embedding(row: Dict[str, Any]) -> np.ndarray:
    return _as_2d_embedding(row).reshape(-1).detach().cpu().numpy()


def _filter_rows(ds, sample_id: int) -> List[Dict[str, Any]]:
    from tqdm.auto import tqdm

    drop = [c for c in ["orig_embedding", "initialization_embedding"] if c in ds.column_names]
    if drop:
        ds = ds.remove_columns(drop)
    rows = []
    for i in tqdm(range(len(ds)), desc=f"Filtering sample_id={sample_id}", leave=False):
        r = ds[i]
        if int(r.get("sample_id", -1)) == int(sample_id):
            rows.append(r)
    return sorted(rows, key=lambda r: int(r.get("stage_index", 0)))


def _pick_text(rows: List[Dict[str, Any]]) -> str:
    for r in reversed(rows):
        t = r.get("text", "")
        if isinstance(t, str) and t.strip():
            return t
    raise ValueError("No text found")


def _most_common_str(vals: List[str]) -> Optional[str]:
    vals = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
    return Counter(vals).most_common(1)[0][0] if vals else None


def _expanded_bounds(vmin, vmax, pad):
    span = max(vmax - vmin, 1e-9)
    return vmin - pad * span, vmax + pad * span


# ---------------------------------------------------------------------------
# GPU accuracy computation
# ---------------------------------------------------------------------------


def _compute_accuracy_batch(
    embs_flat, original_shape, model, device, input_ids, input_text_embeds, attention_mask, batch_size, torch_dtype
):
    import torch

    accs = []
    n = embs_flat.shape[0]
    mem_tokens = original_shape[0]
    denom = attention_mask.sum(dim=-1).clamp_min(1).float()
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch = embs_flat[s:e]
        with torch.no_grad():
            bs = batch.shape[0]
            comp = batch.reshape(bs, *original_shape).to(device)
            te = input_text_embeds.expand(bs, -1, -1)
            ie = torch.cat([comp, te], dim=1)
            ca = torch.ones((bs, mem_tokens), device=device, dtype=attention_mask.dtype)
            am = torch.cat([ca, attention_mask.expand(bs, -1)], dim=1)
            out = model(inputs_embeds=ie, attention_mask=am, use_cache=False)
            pred = out.logits[:, mem_tokens - 1 : -1].argmax(dim=-1)
            correct = (pred == input_ids.expand(bs, -1)).float()
            acc = (correct * am[:, mem_tokens:].float()).sum(dim=-1) / denom.expand(bs)
            accs.append(acc.cpu().numpy())
    return np.concatenate(accs)


def _compute_basin_areas_for_sample(
    ds,
    sample_id: int,
    num_anchors: int,
    mesh_resolution: int,
    batch_size: int,
    threshold: float,
    padding: float,
    model_checkpoint_override: Optional[str],
    torch_dtype_name: str,
    npz_cache_path: str,
    recompute: bool,
) -> Dict[str, Any]:
    """Return dict with keys: stages, seq_lens, areas, max_stage, explained_var_cumul."""
    import torch
    from sklearn.decomposition import PCA

    if os.path.isfile(npz_cache_path) and not recompute:
        print(f"  [sample {sample_id}] Loading cached {npz_cache_path}")
        return dict(np.load(npz_cache_path, allow_pickle=True))

    rows = _filter_rows(ds, sample_id)
    if len(rows) < 2:
        raise ValueError(f"Sample {sample_id}: fewer than 2 stages")

    max_stage = max(int(r.get("stage_index", 0)) for r in rows)
    anchor_indices = np.unique(np.linspace(0, len(rows) - 1, num=num_anchors, dtype=int))

    # PCA
    X = np.stack([_flatten_embedding(r) for r in rows])
    pca = PCA(n_components=min(X.shape[0] - 1, X.shape[1]), random_state=42)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    print(f"  [sample {sample_id}] {len(rows)} stages, max_stage_index={max_stage}, " f"PC1+PC2={100*(ev[0]+ev[1]):.1f}%")

    # Model
    ckpt = model_checkpoint_override or _most_common_str([str(r.get("model_checkpoint", "")) for r in rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[torch_dtype_name]
    kwargs = {"torch_dtype": torch_dtype}
    if device.type == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(ckpt, **kwargs).to(device).eval()
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ref_emb = _as_2d_embedding(rows[-1])
    original_shape = (int(ref_emb.shape[0]), int(ref_emb.shape[1]))
    text = _pick_text(rows)
    enc = tok(text, truncation=True, padding=False, return_tensors="pt")
    input_ids_full = enc["input_ids"].to(device)
    attn_full = enc["attention_mask"].to(device)
    with torch.no_grad():
        text_embeds_full = model.get_input_embeddings()(input_ids_full).to(dtype=torch_dtype)

    # Compute basin area at each anchor
    stages_out, seq_lens_out, areas_out = [], [], []
    from tqdm.auto import tqdm

    for ai in tqdm(anchor_indices, desc=f"  [sample {sample_id}] Computing landscapes"):
        anchor_coords = coords[int(ai)].copy()
        seq_len = int(rows[int(ai)].get("stage_seq_len", 1))
        stage_idx = int(rows[int(ai)].get("stage_index", 0))
        max_len = int(attn_full.sum(dim=-1).max().item())
        use_len = min(max(seq_len, 1), max_len)
        ids = input_ids_full[:, :use_len]
        te = text_embeds_full[:, :use_len, :]
        am = attn_full[:, :use_len]

        # Mesh over PC1-PC2
        lo0, hi0 = _expanded_bounds(float(coords[:, 0].min()), float(coords[:, 0].max()), padding)
        lo1, hi1 = _expanded_bounds(float(coords[:, 1].min()), float(coords[:, 1].max()), padding)
        xr = np.linspace(lo0, hi0, mesh_resolution)
        yr = np.linspace(lo1, hi1, mesh_resolution)
        Xm, Ym = np.meshgrid(xr, yr)
        mesh_pts = np.repeat(anchor_coords.reshape(1, -1), Xm.size, axis=0)
        mesh_pts[:, 0] = Xm.ravel()
        mesh_pts[:, 1] = Ym.ravel()
        recon = pca.inverse_transform(mesh_pts).astype(np.float32)
        recon_t = torch.tensor(recon, dtype=torch.float32).to(dtype=torch_dtype)

        acc = _compute_accuracy_batch(recon_t, original_shape, model, device, ids, te, am, batch_size, torch_dtype)
        acc_2d = acc.reshape(Xm.shape)

        dx = float(np.median(np.diff(xr)))
        dy = float(np.median(np.diff(yr)))
        cell_area = dx * dy
        area = float((acc_2d > threshold).sum()) * cell_area

        stages_out.append(stage_idx)
        seq_lens_out.append(seq_len)
        areas_out.append(area)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    result = {
        "sample_id": np.array([sample_id]),
        "stages": np.array(stages_out),
        "seq_lens": np.array(seq_lens_out),
        "areas": np.array(areas_out),
        "max_stage": np.array([max_stage]),
        "explained_variance_cumul_2": np.array([float(ev[0] + ev[1])]),
        "threshold": np.array([threshold]),
        "mesh_resolution": np.array([mesh_resolution]),
        "created_at": np.array([datetime.now().isoformat()]),
    }
    os.makedirs(os.path.dirname(npz_cache_path), exist_ok=True)
    np.savez_compressed(npz_cache_path, **result)
    print(f"  [sample {sample_id}] Saved: {npz_cache_path}")
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_normalised(all_results: List[Dict[str, Any]], threshold: float, output: str, dpi: int):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    palette = sns.color_palette("husl", n_colors=len(all_results))
    for k, res in enumerate(all_results):
        sid = int(res["sample_id"].ravel()[0])
        stages = res["stages"].ravel().astype(float)
        areas = res["areas"].ravel()
        max_stage = int(res["max_stage"].ravel()[0])

        normalised = stages / max_stage

        # Normalise area: fraction of cell count (resolution-independent)
        mesh_res = int(res["mesh_resolution"].ravel()[0])
        total_cells = mesh_res * mesh_res
        area_frac = areas / total_cells if total_cells > 0 else areas

        label = f"Sample {sid} (cap={max_stage})"
        ax.plot(normalised, area_frac, "o-", color=palette[k], markersize=5, linewidth=1.5, label=label, zorder=3, alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("Normalised stage (stage / max capacity)", fontsize=13)
    ax.set_ylabel(f"Basin fraction (cells with acc > {threshold}) / total cells", fontsize=13)
    ax.set_title("Accuracy basin size vs. normalised stage — Llama-3.1-8B", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(-0.02, 1.05)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.savefig(output.rsplit(".", 1)[0] + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Saved: {output.rsplit('.', 1)[0]}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--sample_ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num_anchors", type=int, default=8)
    parser.add_argument("--mesh_resolution", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    exp_dir = os.path.dirname(args.dataset_path)
    viz_dir = os.path.join(exp_dir, "visualizations")
    output = args.output or os.path.join(viz_dir, "basin_area_vs_stage_normalised.png")

    ds = _load_dataset(args.dataset_path)

    all_results = []
    for sid in args.sample_ids:
        cache_path = os.path.join(viz_dir, f"basin_area_sample_{sid}.npz")
        res = _compute_basin_areas_for_sample(
            ds=ds,
            sample_id=sid,
            num_anchors=args.num_anchors,
            mesh_resolution=args.mesh_resolution,
            batch_size=args.batch_size,
            threshold=args.threshold,
            padding=args.padding,
            model_checkpoint_override=args.model_checkpoint,
            torch_dtype_name=args.dtype,
            npz_cache_path=cache_path,
            recompute=args.recompute,
        )
        all_results.append(res)

    _plot_normalised(all_results, threshold=args.threshold, output=output, dpi=args.dpi)


if __name__ == "__main__":
    main()

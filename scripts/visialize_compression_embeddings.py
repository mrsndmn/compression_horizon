import argparse
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.decomposition import PCA


def _short_label_from_path(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    if base.strip() == "":
        base = path
    return base


def _clean_label(label: str) -> str:
    label = os.path.splitext(label)[0]
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    return label


def load_single_row(
    dataset_path: str,
    sample_id: Optional[int] = None,
    text_contains: Optional[str] = None,
    stage_index: Optional[int] = None,
) -> Dict[str, Any]:
    ds = Dataset.load_from_disk(dataset_path)
    candidates = list(range(len(ds)))
    if sample_id is not None:
        candidates = [i for i in candidates if int(ds[i].get("sample_id", -1)) == int(sample_id)]
    if text_contains is not None:
        text_sub = text_contains.lower()
        candidates = [i for i in candidates if text_sub in str(ds[i].get("text", "")).lower()]
    if stage_index is not None:
        candidates = [i for i in candidates if int(ds[i].get("stage_index", -1)) == int(stage_index)]
    if not candidates:
        raise ValueError(
            f"No matching rows found in '{dataset_path}' for filters: sample_id={sample_id}, text_contains={text_contains}, stage_index={stage_index}"
        )
    row = ds[candidates[0]]
    embedding = torch.tensor(row["embedding"], dtype=torch.float32)
    info = {
        "text": row.get("text", ""),
        "embedding": embedding,  # [num_compression_tokens, hidden]
        "num_compression_tokens": int(row.get("num_compression_tokens", embedding.shape[0])),
        "hidden_size": int(row.get("hidden_size", embedding.shape[1] if embedding.dim() == 2 else embedding.shape[-1])),
        "model_checkpoint": row.get("model_checkpoint", None),
        "label": _short_label_from_path(dataset_path),
        "final_loss": row.get("final_loss", None),
        "final_convergence": row.get("final_convergence", None),
        "stage_index": row.get("stage_index", None),
        "stage_seq_len": row.get("stage_seq_len", None),
    }
    return info


def compute_pairwise_metrics(embeddings: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    flat = [e.reshape(-1).detach().cpu().numpy() for e in embeddings]
    X = np.stack(flat, axis=0)
    # L2 distances
    diffs = X[:, None, :] - X[None, :, :]
    l2 = np.linalg.norm(diffs, axis=-1)
    # Cosine similarity -> distance
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = (Xn @ Xn.T).clip(-1.0, 1.0)
    cos_dist = 1.0 - cos
    return l2, cos_dist


def plot_pairwise_heatmap(matrix: np.ndarray, labels: List[str], title: str, outfile: str):
    plt.figure(figsize=(0.8 * max(4, len(labels)), 0.8 * max(4, len(labels))))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", annot=False, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_pca_scatter(embeddings: List[torch.Tensor], labels: List[str], outfile: str):
    X = np.stack([e.reshape(-1).detach().cpu().numpy() for e in embeddings], axis=0)
    if X.shape[0] < 2 or X.shape[1] < 2:
        return
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    for i, lab in enumerate(labels):
        plt.scatter(xy[i, 0], xy[i, 1], s=80, label=lab)
        plt.text(xy[i, 0], xy[i, 1], lab, fontsize=8, ha="left", va="bottom")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of compressed embeddings (flattened)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_per_token_l2_vs_baseline(embeddings: List[torch.Tensor], labels: List[str], baseline_index: int, outfile: str):
    base = embeddings[baseline_index]
    shapes_match = all(e.shape == base.shape for e in embeddings)
    if not shapes_match:
        return
    num_tokens, hidden = base.shape
    per_ckpt_dists = []
    for e in embeddings:
        d = torch.norm((e - base), p=2, dim=-1).detach().cpu().numpy()  # [num_tokens]
        per_ckpt_dists.append(d)
    arr = np.stack(per_ckpt_dists, axis=0)  # [num_ckpts, num_tokens]
    plt.figure(figsize=(max(6, arr.shape[1] * 0.3), 4 + 0.2 * len(labels)))
    sns.heatmap(arr, annot=False, cmap="magma", yticklabels=labels, xticklabels=[str(i) for i in range(num_tokens)])
    plt.xlabel("compression token index")
    plt.ylabel("checkpoint")
    plt.title("Per-token L2 distance to baseline")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def save_metrics_csv(output_dir: str, labels: List[str], l2: np.ndarray, cos_dist: np.ndarray):
    path = os.path.join(output_dir, "pairwise_metrics.csv")
    with open(path, "w") as f:
        f.write("i,j,label_i,label_j,l2,cosine_distance\n")
        n = len(labels)
        for i in range(n):
            for j in range(n):
                f.write(f"{i},{j},{labels[i]},{labels[j]},{l2[i, j]:.6f},{cos_dist[i, j]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare compressed embeddings from multiple checkpoints")
    parser.add_argument("--embedding_paths", type=str, nargs="+", required=True, help="Paths to saved datasets")
    parser.add_argument("--labels", type=str, nargs="*", default=None, help="Optional labels for each path")
    parser.add_argument("--sample_id", type=int, default=None, help="Filter: sample_id to select")
    parser.add_argument("--text_contains", type=str, default=None, help="Filter: substring present in the sample text")
    parser.add_argument("--stage_index", type=int, default=None, help="Filter for progressive datasets")
    parser.add_argument("--baseline", type=int, default=0, help="Index of baseline for per-token heatmap")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save figures and metrics")

    args = parser.parse_args()

    os.makedirs("artifacts/visualizations", exist_ok=True)
    out_dir = args.output_dir
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("artifacts/visualizations", f"compression_embeddings_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for p in args.embedding_paths:
        rec = load_single_row(
            dataset_path=p,
            sample_id=args.sample_id,
            text_contains=args.text_contains,
            stage_index=args.stage_index,
        )
        rec["source_path"] = p
        records.append(rec)

    labels: List[str] = []
    if args.labels is not None and len(args.labels) == len(records):
        labels = [str(x) for x in args.labels]
    else:
        for r in records:
            label = r.get("model_checkpoint") or r.get("label") or _short_label_from_path(r.get("source_path", ""))
            labels.append(_clean_label(str(label)))

    embeddings = [r["embedding"] for r in records]
    texts = [r.get("text", "") for r in records]

    # Basic sanity checks
    flat_dims = [int(e.numel()) for e in embeddings]
    if len(set(flat_dims)) != 1:
        print("Warning: embeddings have different sizes; distance metrics may be invalid or skipped.")

    # Pairwise metrics
    l2, cos_dist = compute_pairwise_metrics(embeddings)
    save_metrics_csv(out_dir, labels, l2, cos_dist)

    # Plots
    sns.set(style="whitegrid")
    plot_pairwise_heatmap(l2, labels, title="Pairwise L2 distance", outfile=os.path.join(out_dir, "pairwise_l2.png"))
    plot_pairwise_heatmap(
        cos_dist,
        labels,
        title="Pairwise cosine distance (1 - cos)",
        outfile=os.path.join(out_dir, "pairwise_cosine.png"),
    )
    plot_pca_scatter(embeddings, labels, outfile=os.path.join(out_dir, "pca_scatter.png"))
    baseline_index = max(0, min(int(args.baseline), len(embeddings) - 1))
    plot_per_token_l2_vs_baseline(
        embeddings, labels, baseline_index=baseline_index, outfile=os.path.join(out_dir, "per_token_l2_vs_baseline.png")
    )

    # Save a small text info file
    info_path = os.path.join(out_dir, "info.txt")
    with open(info_path, "w") as f:
        f.write("inputs:\n")
        for lab, rec in zip(labels, records):
            f.write(f"- {lab}: {rec.get('source_path')}\n")
        f.write("\ntexts (first 160 chars):\n")
        for lab, t in zip(labels, texts):
            short = (t or "")[:160].replace("\n", " ")
            f.write(f"- {lab}: {short}\n")
        f.write("\nshapes:\n")
        for lab, e in zip(labels, embeddings):
            f.write(f"- {lab}: {tuple(e.shape)}\n")

    print(f"Saved metrics and figures to: {out_dir}")


if __name__ == "__main__":
    main()

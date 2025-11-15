import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import torch
from datasets import Dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_progressive_dataset(dataset_path: str) -> Dataset:
    return Dataset.load_from_disk(dataset_path)


def filter_records(
    ds: Dataset,
    sample_id: Optional[int] = None,
    stage_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        r = ds[i]
        if sample_id is not None and int(r.get("sample_id", -1)) != int(sample_id):
            continue
        if stage_index is not None and int(r.get("stage_index", -1)) != int(stage_index):
            continue
        rows.append(r)
    return rows


def collate_stages_by_sample(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_sid: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = int(r.get("sample_id", -1))
        if sid not in by_sid:
            by_sid[sid] = []
        by_sid[sid].append(r)
    for sid in by_sid:
        by_sid[sid].sort(key=lambda x: int(x.get("stage_index", 0)))
    return by_sid


def flatten_embedding(row: Dict[str, Any]) -> np.ndarray:
    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    return emb.reshape(-1).detach().cpu().numpy()


def compute_pairwise_similarities(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diffs = X[:, None, :] - X[None, :, :]
    l2 = np.linalg.norm(diffs, axis=-1)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = (Xn @ Xn.T).clip(-1.0, 1.0)
    cos_dist = 1.0 - cos
    return l2, cos_dist


def plot_heatmap(matrix: np.ndarray, labels: List[str], title: str, outfile: str):
    plt.figure(figsize=(0.7 * max(4, len(labels)), 0.7 * max(4, len(labels))))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", annot=False, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_pca(X: np.ndarray, labels: List[str], outfile: str):
    if X.shape[0] < 2 or X.shape[1] < 2:
        return
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    for i, lab in enumerate(labels):
        plt.scatter(xy[i, 0], xy[i, 1], s=60)
        plt.text(xy[i, 0], xy[i, 1], lab, fontsize=8, ha="left", va="bottom")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of progressive embeddings (flattened)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_correlation(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, outfile: str):
    plt.figure(figsize=(6, 4))
    sns.regplot(x=x, y=y, scatter_kws={"s": 20}, line_kws={"color": "red"})
    corr = np.corrcoef(x, y)[0, 1] if x.size > 1 and y.size > 1 else np.nan
    plt.title(f"{title} (r={corr:.3f})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_norms_over_stages(
    labels: List[str], mean_vals: List[float], max_vals: List[float], ylabel: str, title: str, outfile: str
):
    if len(mean_vals) == 0:
        return
    plt.figure(figsize=(max(6, 0.6 * len(labels)), 4))
    x = np.arange(len(labels))
    plt.plot(x, mean_vals, marker="o", label="mean")
    if len(max_vals) == len(mean_vals):
        plt.plot(x, max_vals, marker="s", label="max")
    plt.xticks(x, labels, rotation=0)
    plt.xlabel("stages")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def estimate_token_perplexity(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    # logits: [B, T, V], labels: [B, T], mask: [B, T]
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = labels[:, 1:]
    m = mask[:, 1:].bool()
    nll = -log_probs.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
    nll = nll[m]
    if nll.numel() == 0:
        return float("nan")
    ppl = torch.exp(nll.mean()).item()
    return float(ppl)


def compute_ppl_for_text(model: AutoModelForCausalLM, tok: AutoTokenizer, device: torch.device, text: str) -> Tuple[int, float]:
    with torch.no_grad():
        enc = tok(text, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        ppl = estimate_token_perplexity(out.logits, input_ids, attn)
        seq_len = int(attn.sum().item())
    return seq_len, ppl


def compute_distance_metrics(X: np.ndarray) -> float:
    # Returns (initial_final_l2, trajectory_length_l2)
    if X.shape[0] < 2:
        return 0.0, 0.0
    init_final = float(np.linalg.norm(X[-1] - X[0]))
    diffs = X[1:, :] - X[:-1, :]
    traj_len = float(np.linalg.norm(diffs, axis=1).sum())
    return init_final, traj_len


def compute_token_norm_stats_from_row(row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (l1_per_token, l2_per_token) across all tokens in the embedding
    # Accepts embeddings of shape [..., hidden_dim]; flattens leading dims to tokens
    emb = torch.tensor(row["embedding"], dtype=torch.float32)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    hidden_dim = emb.shape[-1]
    emb2d = emb.reshape(-1, hidden_dim)
    l2 = torch.linalg.norm(emb2d, ord=2, dim=-1).detach().cpu().numpy()
    l1 = torch.linalg.norm(emb2d, ord=1, dim=-1).detach().cpu().numpy()
    return l1, l2


def maybe_compute_perplexity(
    rows: List[Dict[str, Any]],
    model_name: Optional[str],
    max_eval_samples: int,
) -> Tuple[List[int], List[float]]:
    if model_name is None or len(rows) == 0 or max_eval_samples <= 0:
        return [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tok = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
    except Exception:
        return [], []

    seq_lens: List[int] = []
    ppls: List[float] = []
    with torch.no_grad():
        for r in rows[:max_eval_samples]:
            text = r.get("text", "")
            if not isinstance(text, str) or text.strip() == "":
                continue
            enc = tok(text, truncation=True, padding=False, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            ppl = estimate_token_perplexity(out.logits, input_ids, attn)
            seq_lens.append(int(attn.sum().item()))
            ppls.append(float(ppl))
    return seq_lens, ppls


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze progressive_train artifacts")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to progressive_prefixes dataset")
    parser.add_argument("--sample_id", type=int, default=None, help="Optional sample_id filter")
    parser.add_argument("--stage_index", type=int, default=None, help="Optional stage filter")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save figures and metrics")
    parser.add_argument(
        "--perplexity_model",
        type=str,
        default=None,
        help="HF model name to compute token-level perplexity of sample texts",
    )
    parser.add_argument("--perplexity_max_samples", type=int, default=64, help="Max rows to use for perplexity estimation")

    args = parser.parse_args()

    os.makedirs("artifacts/visualizations", exist_ok=True)
    out_dir = args.output_dir
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("artifacts/visualizations", f"progressive_embeddings_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    ds = load_progressive_dataset(args.dataset_path)
    rows = filter_records(ds, sample_id=args.sample_id, stage_index=args.stage_index)
    if not rows:
        raise ValueError("No records found with given filters.")

    # Group by sample and build stage-wise matrices
    by_sid = collate_stages_by_sample(rows)

    # For each sample: compute pairwise distances between stages and PCA
    sns.set(style="whitegrid")
    summary_steps: List[int] = []
    summary_conv: List[float] = []
    summary_seq_len: List[int] = []

    # Prepare optional perplexity model once
    model_for_ppl: Optional[str] = args.perplexity_model
    if model_for_ppl is None:
        names = [str(r.get("model_checkpoint", "")).strip() for r in rows]
        names = [n for n in names if n]
        if names:
            uniq = {}
            for n in names:
                uniq[n] = uniq.get(n, 0) + 1
            model_for_ppl = max(uniq.items(), key=lambda kv: kv[1])[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tok = None
    if model_for_ppl is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_for_ppl).to(device)
            model.eval()
            tok = AutoTokenizer.from_pretrained(model_for_ppl)
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
        except Exception:
            model = None
            tok = None

    # Holders for cross-sample correlation analyses
    dist_init_final_all: List[float] = []
    ppl_all: List[float] = []
    seq_len_all: List[int] = []
    sid_all: List[int] = []

    for sid, stages in by_sid.items():
        labels = [f"L{int(s.get('stage_seq_len', -1))}" for s in stages]
        X = np.stack([flatten_embedding(s) for s in stages], axis=0)
        l2, cos_d = compute_pairwise_similarities(X)
        plot_heatmap(l2, labels, title=f"Sample {sid}: L2 by stage", outfile=os.path.join(out_dir, f"sid{sid}_l2.png"))
        plot_heatmap(
            cos_d,
            labels,
            title=f"Sample {sid}: cosine distance by stage",
            outfile=os.path.join(out_dir, f"sid{sid}_cosine.png"),
        )
        plot_pca(X, labels, outfile=os.path.join(out_dir, f"sid{sid}_pca.png"))

        # Collect per-stage stats
        for s in stages:
            steps = int(s.get("steps_taken", 0))
            conv = float(s.get("final_convergence", np.nan)) if s.get("final_convergence") is not None else np.nan
            seql = int(s.get("stage_seq_len", -1))
            summary_steps.append(steps)
            summary_conv.append(conv)
            summary_seq_len.append(seql)

        # Per-sample distance metrics
        for i in range(X.shape[0] - 1):
            dist_init_final_all.append(float(np.linalg.norm(X[i + 1] - X[i])))

        # Per-sample perplexity (optional)
        if model is not None and tok is not None:
            sample_text = None
            for s in stages:
                sample_text = s.get("text", None)
                if sample_text is not None:
                    seql, ppl = compute_ppl_for_text(model, tok, device, sample_text)
                    if math.isnan(ppl):
                        continue

                    seq_len_all.append(seql)
                    ppl_all.append(float(ppl))
                    sid_all.append(int(sid))

        # Per-sample token norm trajectories across stages
        mean_l2_by_stage: List[float] = []
        max_l2_by_stage: List[float] = []
        mean_l1_by_stage: List[float] = []
        max_l1_by_stage: List[float] = []
        for s in stages:
            try:
                l1_tok, l2_tok = compute_token_norm_stats_from_row(s)
                if l1_tok.size == 0 or l2_tok.size == 0:
                    mean_l1_by_stage.append(float("nan"))
                    max_l1_by_stage.append(float("nan"))
                    mean_l2_by_stage.append(float("nan"))
                    max_l2_by_stage.append(float("nan"))
                else:
                    mean_l1_by_stage.append(float(np.mean(l1_tok)))
                    max_l1_by_stage.append(float(np.max(l1_tok)))
                    mean_l2_by_stage.append(float(np.mean(l2_tok)))
                    max_l2_by_stage.append(float(np.max(l2_tok)))
            except Exception:
                mean_l1_by_stage.append(float("nan"))
                max_l1_by_stage.append(float("nan"))
                mean_l2_by_stage.append(float("nan"))
                max_l2_by_stage.append(float("nan"))

        # Plot L2 and L1 norm trajectories for this sample
        plot_norms_over_stages(
            labels,
            mean_l2_by_stage,
            max_l2_by_stage,
            ylabel="token L2 norm",
            title=f"Sample {sid}: token L2 norms across stages",
            outfile=os.path.join(out_dir, f"sid{sid}_token_norms_l2.png"),
        )
        plot_norms_over_stages(
            labels,
            mean_l1_by_stage,
            max_l1_by_stage,
            ylabel="token L1 norm",
            title=f"Sample {sid}: token L1 norms across stages",
            outfile=os.path.join(out_dir, f"sid{sid}_token_norms_l1.png"),
        )

    # Correlation plots across all stages
    if len(summary_steps) > 1 and len(summary_conv) == len(summary_steps):
        plot_correlation(
            np.array(summary_steps),
            np.array(summary_conv),
            xlabel="steps_taken",
            ylabel="final_convergence",
            title="Steps vs Convergence",
            outfile=os.path.join(out_dir, "steps_vs_convergence.png"),
        )
    if len(summary_seq_len) > 1 and len(summary_steps) == len(summary_seq_len):
        plot_correlation(
            np.array(summary_seq_len),
            np.array(summary_steps),
            xlabel="stage_seq_len",
            ylabel="steps_taken",
            title="Length vs Steps",
            outfile=os.path.join(out_dir, "length_vs_steps.png"),
        )

    if len(ppl_all) > 1:
        plot_correlation(
            np.array(ppl_all),
            np.array(summary_steps[1:]),
            xlabel="ppl",
            ylabel="steps_taken",
            title="PPL vs Steps",
            outfile=os.path.join(out_dir, "ppl_vs_steps.png"),
        )

    # Optional: plots leveraging per-sample perplexities (if available)
    if len(ppl_all) > 1 and len(ppl_all) == len(dist_init_final_all):
        plot_correlation(
            np.array(dist_init_final_all),
            np.array(ppl_all),
            xlabel="steps L2",
            ylabel="perplexity",
            title="Comp Embeddings L2 Distance vs Perplexity",
            outfile=os.path.join(out_dir, "l2_dist_vs_perplexity.png"),
        )
    if len(seq_len_all) > 1 and len(seq_len_all) == len(ppl_all):
        plot_correlation(
            np.array(seq_len_all),
            np.array(ppl_all),
            xlabel="sequence_length",
            ylabel="perplexity",
            title="Length vs Perplexity",
            outfile=os.path.join(out_dir, "length_vs_perplexity.png"),
        )

    # Save a summary CSV
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("sample_id,stage_index,stage_seq_len,steps_taken,final_convergence\n")
        for sid, stages in by_sid.items():
            for s in stages:
                f.write(
                    f"{sid},{int(s.get('stage_index', -1))},{int(s.get('stage_seq_len', -1))},{int(s.get('steps_taken', 0))},{float(s.get('final_convergence', np.nan))}\n"
                )

    print(f"Saved progressive figures and metrics to: {out_dir}")


if __name__ == "__main__":
    main()

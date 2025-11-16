import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.decomposition import PCA
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

    # Optional: token-level perplexity on sample texts
    model_for_ppl: Optional[str] = args.perplexity_model
    if model_for_ppl is None:
        # Try infer from dataset rows' model_checkpoint
        names = [str(r.get("model_checkpoint", "")).strip() for r in rows]
        names = [n for n in names if n]
        if names:
            # choose most frequent
            uniq = {}
            for n in names:
                uniq[n] = uniq.get(n, 0) + 1
            model_for_ppl = max(uniq.items(), key=lambda kv: kv[1])[0]
    seq_lens, ppls = maybe_compute_perplexity(rows, model_for_ppl, args.perplexity_max_samples)
    if len(seq_lens) > 1 and len(seq_lens) == len(ppls):
        plot_correlation(
            np.array(seq_lens),
            np.array(ppls),
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

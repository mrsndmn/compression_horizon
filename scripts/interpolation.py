import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def curve_length_from_points(points: torch.Tensor) -> float:
    """Compute polyline length given sampled points along a curve.

    points: [K, C, D] tensor of compression token embeddings along t.
    Returns total Euclidean arc length over flattened [C*D] space.
    """
    if points.dim() != 3 or points.size(0) < 2:
        return 0.0
    diffs = points[1:] - points[:-1]  # [K-1, C, D]
    diffs_flat = diffs.reshape(diffs.size(0), -1)  # [K-1, C*D]
    seg_lengths = torch.linalg.norm(diffs_flat, dim=1)  # [K-1]
    return float(seg_lengths.sum().item())


def load_progressive_dataset(dataset_path: str) -> Dataset:
    return Dataset.load_from_disk(dataset_path)


def filter_records(
    ds: Dataset,
    sample_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        r = ds[i]
        if sample_id is not None and int(r.get("sample_id", -1)) != int(sample_id):
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


def to_tensor_embedding(row: Dict[str, Any], device: torch.device) -> torch.Tensor:
    emb = torch.tensor(row["embedding"], dtype=torch.float32, device=device)
    return emb


def prepare_model(model_name: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def compute_convergence(
    model,
    compression_tokens: torch.Tensor,  # [B, C, D]
    inputs_embeds: torch.Tensor,  # [B, T, D]
    attention_mask: torch.Tensor,  # [B, T]
    input_ids: torch.Tensor,  # [B, T]
) -> float:
    attn_ct = torch.ones(
        (compression_tokens.size(0), compression_tokens.size(1)),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    inputs_embeds_with_ct = torch.cat([compression_tokens, inputs_embeds], dim=1)
    attention_mask_with_ct = torch.cat([attn_ct, attention_mask], dim=1)
    outputs = model(inputs_embeds=inputs_embeds_with_ct, attention_mask=attention_mask_with_ct)
    preds = outputs.logits[:, 0:-1].argmax(dim=-1)
    conv_numerator = (preds == input_ids[:, :]).sum(dim=-1)
    denom = attention_mask.sum(dim=-1).clamp(min=1)
    conv = (conv_numerator / denom).mean().item()
    return float(conv)


def cross_entropy_loss_for_batch(
    logits: torch.Tensor,  # [B, L, V]
    input_ids: torch.Tensor,  # [B, T]
    attention_mask: torch.Tensor,  # [B, T]
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels.flatten(), reduction="mean")
    return loss


def tokenize_text(tok, text: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tok(text, truncation=True, padding=False, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def embed_tokens(model, input_ids: torch.Tensor) -> torch.Tensor:
    return model.model.embed_tokens(input_ids)


def evaluate_linear_curve(
    model,
    e0: torch.Tensor,  # [C, D]
    e1: torch.Tensor,  # [C, D]
    inputs_embeds: torch.Tensor,  # [1, T, D]
    attention_mask: torch.Tensor,  # [1, T]
    input_ids: torch.Tensor,  # [1, T]
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    accs: List[float] = []
    for t in tqdm(ts, desc="Evaluating linear curve"):
        ct = ((1.0 - float(t)) * e0 + float(t) * e1).unsqueeze(0)  # [1, C, D]
        acc = compute_convergence(model, ct, inputs_embeds, attention_mask, input_ids)
        accs.append(acc)
    return ts, np.array(accs, dtype=np.float32)


def learn_bezier_and_evaluate(
    model,
    e0: torch.Tensor,  # [C, D]
    e1: torch.Tensor,  # [C, D]
    inputs_embeds: torch.Tensor,  # [1, T, D]
    attention_mask: torch.Tensor,  # [1, T]
    input_ids: torch.Tensor,  # [1, T]
    num_points: int,
    bezier_order: int = 2,
    weight_decay: float = 0.0,
    steps: int = 1000,
    lr: float = 1e-2,
    batch_t: int = 16,
    seed: int = 42,
    evaluate_every: int = 100,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, float]:
    torch.manual_seed(int(seed))
    device = e0.device
    C, D = e0.shape
    n = max(2, int(bezier_order))
    # Learn internal control points P1..P(n-1); endpoints P0=e0, Pn=e1 are fixed
    control_params = torch.nn.ParameterList()
    for k in range(1, n):
        if k == n:
            break
        alpha = k / n
        init_k = (1.0 - alpha) * e0 + alpha * e1 + 0.01 * torch.randn(C, D, device=device)
        control_params.append(torch.nn.Parameter(init_k))
    opt = torch.optim.AdamW(control_params.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def _bezier_points(ts: torch.Tensor) -> torch.Tensor:
        t = ts.view(-1, 1, 1)
        # Build control point list P0..Pn
        points: List[torch.Tensor] = [e0] + [p for p in control_params] + [e1]
        P = torch.stack(points, dim=0)  # [n+1, C, D]
        B = t.shape[0]
        # Compute Bernstein coefficients for each i=0..n
        # coeffs[b, i] = comb(n,i) * (1-t_b)^{n-i} * t_b^{i}
        ts_flat = ts.view(-1)
        one_minus_t = 1.0 - ts_flat
        coeffs = []
        for i in range(n + 1):
            binom = float(math.comb(n, i))
            coeffs.append(binom * (one_minus_t ** (n - i)) * (ts_flat**i))
        coeffs_t = torch.stack(coeffs, dim=1).to(device)  # [B, n+1]
        ct = (coeffs_t.view(B, n + 1, 1, 1) * P.view(1, n + 1, C, D)).sum(dim=1)  # [B, C, D]
        return ct

    def run_step(ts: torch.Tensor) -> torch.Tensor:
        ct = _bezier_points(ts)
        B = ct.shape[0]
        inputs_b = inputs_embeds.expand(B, -1, -1)
        attn_b = attention_mask.expand(B, -1)
        ids_b = input_ids.expand(B, -1)
        attn_ct = torch.ones((B, C), dtype=attn_b.dtype, device=device)
        x = torch.cat([ct, inputs_b], dim=1)
        m = torch.cat([attn_ct, attn_b], dim=1)
        out = model(inputs_embeds=x, attention_mask=m)
        return cross_entropy_loss_for_batch(out.logits, ids_b, attn_b)

    for iter_i in tqdm(range(int(steps)), desc="Optimizing Bezier control point"):
        # beta-distribution with alpha=0.5 and beta=0.5
        ts = torch.rand(min(batch_t, num_points), device=device)
        # ts = distributions.Beta(0.5, 0.5).sample(sample_shape=torch.Size([min(batch_t, num_points)])).to(device)

        loss = run_step(ts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lr_scheduler.step()

        if iter_i % evaluate_every == 0:
            ts_np = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
            accs: List[float] = []
            with torch.no_grad():
                for tval in tqdm(ts_np, desc="Evaluating Bezier curve"):
                    t_t = torch.tensor([tval], device=device, dtype=torch.float32)
                    ct = _bezier_points(t_t)
                    acc = compute_convergence(model, ct, inputs_embeds, attention_mask, input_ids)
                    accs.append(acc)
            print(
                f"Iteration {iter_i}, mean accuracy: {torch.tensor(accs).mean().item()}, min accuracy: {torch.tensor(accs).min().item()}, max accuracy: {torch.tensor(accs).max().item()}"
            )
            plt.plot(ts_np, accs, label="Bezier (learned)", linewidth=2)
            plt.xlabel("t")
            plt.ylabel("convergence accuracy")
            plt.title(f"Interpolation Accuracy (iteration {iter_i})")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(f"/tmp/interpolation_accuracy_iteration{iter_i}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

    ts_np = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    accs: List[float] = []
    curve_pts: List[torch.Tensor] = []
    with torch.no_grad():
        for tval in tqdm(ts_np, desc="Evaluating Bezier curve"):
            t_t = torch.tensor([tval], device=device, dtype=torch.float32)
            ct = _bezier_points(t_t)
            acc = compute_convergence(model, ct, inputs_embeds, attention_mask, input_ids)
            accs.append(acc)
            curve_pts.append(ct.squeeze(0))  # [C, D]
    bezier_points = torch.stack(curve_pts, dim=0)  # [K, C, D]
    bezier_length = curve_length_from_points(bezier_points)
    # Stack learned control points into [n-1, C, D]
    learned = (
        torch.stack([p.detach().clone() for p in control_params], dim=0) if len(control_params) > 0 else torch.empty(0, C, D)
    )
    return learned, ts_np, np.array(accs, dtype=np.float32), bezier_length


def pick_model_name(rows: List[Dict[str, Any]]) -> Optional[str]:
    names = [str(r.get("model_checkpoint", "")).strip() for r in rows]
    names = [n for n in names if n]
    if not names:
        return None
    counts: Dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def main():
    parser = argparse.ArgumentParser(description="Interpolate compression embeddings and evaluate accuracies")
    parser.add_argument("--dataset_path1", type=str, required=True, help="Path to progressive_prefixes dataset")
    parser.add_argument("--dataset_path2", type=str, required=True, help="Path to progressive_prefixes dataset")
    parser.add_argument("--sample_id", type=int, default=None, help="Optional sample_id filter")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="HF model name; inferred if omitted")
    parser.add_argument("--num_points", type=int, default=300, help="Number of evaluation points along t  [0,1]")
    parser.add_argument("--bezier_steps", type=int, default=5000, help="Optimization steps for Bezier control point")
    parser.add_argument("--bezier_lr", type=float, default=1e-2, help="Learning rate for Bezier control point")
    parser.add_argument("--bezier_batch_t", type=int, default=32, help="Number of t samples per optimization step")
    parser.add_argument("--bezier_order", type=int, default=2, help="Bezier curve order (>=2)")
    parser.add_argument("--bezier_weight_decay", type=float, default=0.0, help="Weight decay for Bezier control point")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="/tmp", help="Where to save plots and parameters")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.manual_seed(int(args.seed))

    ds1 = load_progressive_dataset(args.dataset_path1)
    ds2 = load_progressive_dataset(args.dataset_path2)
    rows1 = filter_records(ds1, sample_id=args.sample_id)
    rows2 = filter_records(ds2, sample_id=args.sample_id)
    if not rows1 or not rows2:
        raise ValueError("No records found with given filters.")

    rows = rows1 + rows2
    model_name = args.model_checkpoint or pick_model_name(rows)
    if not model_name:
        raise ValueError("Could not infer model checkpoint from dataset; please pass --model_checkpoint")
    model, tok = prepare_model(model_name, device)
    # Freeze model weights; we only optimize Bezier control points
    for p in model.parameters():
        p.requires_grad_(False)

    by_sid = collate_stages_by_sample(rows)

    all_ts: Optional[np.ndarray] = None
    all_lin_accs: List[np.ndarray] = []
    all_bez_accs: List[np.ndarray] = []
    all_lin_lengths: List[float] = []
    all_bez_lengths: List[float] = []

    for sid, stages in by_sid.items():
        first = stages[0]
        last = stages[-1]
        text_eval = str(last.get("text", ""))
        if text_eval.strip() == "":
            for s in reversed(stages):
                t_ = str(s.get("text", ""))
                if t_.strip() != "":
                    text_eval = t_
                    break
        if text_eval.strip() == "":
            continue

        input_ids, attention_mask = tokenize_text(tok, text_eval, device)
        # Compute token embeddings once without tracking graph; reuse safely across steps
        with torch.no_grad():
            inputs_embeds = embed_tokens(model, input_ids)
        inputs_embeds = inputs_embeds.detach()

        e0 = to_tensor_embedding(first, device)
        e1 = to_tensor_embedding(last, device)

        ts_lin, accs_lin = evaluate_linear_curve(
            model, e0, e1, inputs_embeds, attention_mask, input_ids, num_points=int(args.num_points)
        )
        # Exact for linear interpolation even with discretization at uniform t
        linear_length = float(torch.linalg.norm((e1 - e0).reshape(-1)).item())

        learned_ctrl, ts_bez, accs_bez, bezier_length = learn_bezier_and_evaluate(
            model,
            e0,
            e1,
            inputs_embeds,
            attention_mask,
            input_ids,
            bezier_order=int(args.bezier_order),
            weight_decay=float(args.bezier_weight_decay),
            num_points=int(args.num_points),
            steps=int(args.bezier_steps),
            lr=float(args.bezier_lr),
            batch_t=int(args.bezier_batch_t),
            seed=int(args.seed),
        )

        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(ts_lin, accs_lin, label=f"Linear (L={linear_length:.2f})", linewidth=2)
        plt.plot(ts_bez, accs_bez, label=f"Bezier (learned, L={bezier_length:.2f})", linewidth=2)
        plt.xlabel("t")
        plt.ylabel("convergence accuracy")
        plt.title(f"Interpolation Accuracy (sample {sid})")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"interpolation_accuracy_sid{sid}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        params_path = os.path.join(args.output_dir, f"bezier_params_sid{sid}.pt")
        torch.save(
            {
                "text_eval": text_eval,
                "bezier_order": int(args.bezier_order),
                "control_points": learned_ctrl.cpu(),  # [order-1, C, D]
                "control_point": learned_ctrl.cpu()[0] if learned_ctrl.numel() > 0 else None,
                "num_compression_tokens": int(e0.shape[0]),
                "hidden_size": int(e0.shape[1]),
                "endpoints": {
                    "e0": e0.detach().cpu(),
                    "e1": e1.detach().cpu(),
                },
                "model_checkpoint": model_name,
            },
            params_path,
        )

        if all_ts is None:
            all_ts = ts_lin
        all_lin_accs.append(accs_lin)
        all_bez_accs.append(accs_bez)
        all_lin_lengths.append(linear_length)
        all_bez_lengths.append(bezier_length)

    if all_ts is not None and len(all_lin_accs) > 0 and len(all_bez_accs) > 0:
        import matplotlib.pyplot as plt

        lin_stack = np.stack(all_lin_accs, axis=0)
        bez_stack = np.stack(all_bez_accs, axis=0)
        lin_mean = lin_stack.mean(axis=0)
        lin_std = lin_stack.std(axis=0)
        bez_mean = bez_stack.mean(axis=0)
        bez_std = bez_stack.std(axis=0)

        mean_lin_len = float(np.mean(all_lin_lengths)) if len(all_lin_lengths) > 0 else 0.0
        mean_bez_len = float(np.mean(all_bez_lengths)) if len(all_bez_lengths) > 0 else 0.0

        plt.figure(figsize=(7, 4))
        plt.plot(all_ts, lin_mean, label=f"Linear (mean, L={mean_lin_len:.2f})", color="C0", linewidth=2)
        plt.fill_between(all_ts, lin_mean - lin_std, lin_mean + lin_std, color="C0", alpha=0.2)
        plt.plot(all_ts, bez_mean, label=f"Bezier (mean, L={mean_bez_len:.2f})", color="C1", linewidth=2)
        plt.fill_between(all_ts, bez_mean - bez_std, bez_mean + bez_std, color="C1", alpha=0.2)
        plt.xlabel("t")
        plt.ylabel("convergence accuracy")
        plt.title("Interpolation Accuracy (aggregate)")
        plt.legend()
        plt.tight_layout()
        agg_plot_path = os.path.join(args.output_dir, "interpolation_accuracy_aggregate.png")
        plt.savefig(agg_plot_path, dpi=150)
        plt.close()

    # Normalize e1, e1 cooridinates to 0, 1 and bezier control point coordinates too

    print(f"Saved interpolation results to: {args.output_dir}")


if __name__ == "__main__":
    main()

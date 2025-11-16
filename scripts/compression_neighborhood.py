import argparse
import csv
import math
import os
import random
import string
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n))


def l2_unit_direction_like(t: torch.Tensor) -> torch.Tensor:
    g = torch.randn_like(t)
    g_norm = torch.norm(g.flatten(), p=2) + 1e-12
    return g / g_norm


def l1_unit_direction_like(t: torch.Tensor) -> torch.Tensor:
    # Sample from Laplace-like direction and normalize to L1
    u = torch.empty_like(t).exponential_()
    signs = torch.sign(torch.randn_like(t))
    vec = u * signs
    denom = torch.norm(vec.flatten(), p=1) + 1e-12
    return vec / denom


def linf_direction_like(t: torch.Tensor, radius: float) -> torch.Tensor:
    return (2.0 * torch.rand_like(t) - 1.0) * radius


def sample_perturbations(
    base: torch.Tensor,
    num_points: int,
    norms: List[str],
    l2_radius_max: float,
    l1_radius_max: float,
    linf_radius_max: float,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, List[str], np.ndarray]:
    if generator is None:
        generator = torch.Generator(device=base.device)
        generator.manual_seed(42)

    points_per_norm = [num_points // len(norms)] * len(norms)
    for i in range(num_points % len(norms)):
        points_per_norm[i] += 1

    deltas: List[torch.Tensor] = []
    norm_types: List[str] = []
    radii: List[float] = []

    for norm_name, k in zip(norms, points_per_norm):
        if k == 0:
            continue
        if norm_name.lower() in {"l2", "2"}:
            for _ in range(k):
                radius = float(torch.rand(1, device=base.device).item() * l2_radius_max)
                direction = l2_unit_direction_like(base)
                deltas.append(direction * radius)
                norm_types.append("l2")
                radii.append(radius)
        elif norm_name.lower() in {"l1", "1"}:
            for _ in range(k):
                radius = float(torch.rand(1, device=base.device).item() * l1_radius_max)
                direction = l1_unit_direction_like(base)
                deltas.append(direction * radius)
                norm_types.append("l1")
                radii.append(radius)
        elif norm_name.lower() in {"linf", "inf", "infty"}:
            for _ in range(k):
                radius = float(torch.rand(1, device=base.device).item() * linf_radius_max)
                deltas.append(linf_direction_like(base, radius))
                norm_types.append("linf")
                radii.append(radius)
        else:
            raise ValueError(f"Unsupported norm type: {norm_name}")

    all_deltas = torch.stack(deltas, dim=0)
    return all_deltas, norm_types, np.asarray(radii, dtype=np.float64)


@torch.inference_mode()
def compute_convergence_for_points(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_embedding: torch.Tensor,
    perturbations: torch.Tensor,
    text: str,
    max_sequence_length: int,
    batch_size: int = 64,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # [1, seq_len, hidden]
    token_embeds = model.model.embed_tokens(input_ids)

    comp = base_embedding.to(device)
    if comp.dim() == 1:
        comp = comp.unsqueeze(0)
    # [num_comp, hidden]
    num_comp, hidden = comp.shape

    # Ensure perturbations match shape [N, num_comp, hidden]
    if perturbations.dim() == 2:
        perturbations = perturbations.unsqueeze(1)
    if perturbations.shape[1:] != (num_comp, hidden):
        raise ValueError(f"Perturbations must have shape [N, {num_comp}, {hidden}], got {tuple(perturbations.shape)}")

    num_points = perturbations.shape[0]
    results: List[float] = []

    for start in tqdm(range(0, num_points, batch_size)):
        end = min(start + batch_size, num_points)
        batch_delta = perturbations[start:end].to(device)
        comp_batch = comp.unsqueeze(0).expand(end - start, -1, -1) + batch_delta

        model_tokens_with_comp = torch.cat([comp_batch, token_embeds.expand(end - start, -1, -1)], dim=1)
        comp_mask = torch.ones((end - start, num_comp), dtype=attention_mask.dtype, device=device)
        attn_with_comp = torch.cat([comp_mask, attention_mask.expand(end - start, -1)], dim=1)

        outputs = model(
            inputs_embeds=model_tokens_with_comp,
            attention_mask=attn_with_comp,
            output_hidden_states=False,
        )
        logits = outputs.logits
        preds = logits[:, 0:-1].argmax(dim=-1)
        # Compare to input ids across the text portion
        conv_numerator = (preds == input_ids.expand(end - start, -1)).sum(dim=-1)
        conv_denominator = attention_mask.sum(dim=-1).expand(end - start)
        conv = (conv_numerator.to(torch.float32) / conv_denominator.to(torch.float32)).detach().cpu()
        results.extend(conv.tolist())

    return np.asarray(results, dtype=np.float32)


def make_plots(
    save_dir: str,
    radii: np.ndarray,
    norm_types: List[str],
    convergence: np.ndarray,
    perturbations: torch.Tensor,
):
    os.makedirs(save_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Scatter: radius vs convergence, colored by norm
    plt.figure(figsize=(7, 5))
    palette = {"l2": "tab:blue", "l1": "tab:green", "linf": "tab:red"}
    for norm_name in sorted(set(norm_types)):
        idx = [i for i, n in enumerate(norm_types) if n == norm_name]
        plt.scatter(radii[idx], convergence[idx], s=12, alpha=0.6, label=norm_name, c=palette.get(norm_name, None))
    plt.xlabel("radius")
    plt.ylabel("convergence")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "radius_vs_convergence.png"), dpi=150)
    plt.close()

    # Binned mean curve per norm
    plt.figure(figsize=(7, 5))
    bins = 20
    for norm_name in sorted(set(norm_types)):
        idx = np.array([i for i, n in enumerate(norm_types) if n == norm_name], dtype=np.int64)
        if idx.size == 0:
            continue
        r = radii[idx]
        c = convergence[idx]
        edges = np.linspace(r.min(), r.max() if r.max() > 0 else 1e-8, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        means = []
        for b0, b1 in zip(edges[:-1], edges[1:]):
            sel = (r >= b0) & (r < b1)
            means.append(c[sel].mean() if sel.any() else np.nan)
        means = np.asarray(means)
        plt.plot(centers, means, marker="o", label=norm_name)
    plt.xlabel("radius (binned)")
    plt.ylabel("mean convergence")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "binned_mean_convergence.png"), dpi=150)
    plt.close()

    # Histogram of convergence by norm
    plt.figure(figsize=(7, 5))
    for norm_name in sorted(set(norm_types)):
        idx = [i for i, n in enumerate(norm_types) if n == norm_name]
        plt.hist(
            convergence[idx],
            bins=20,
            alpha=0.4,
            label=norm_name,
            range=(0.0, 1.0),
        )
    plt.xlabel("convergence")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "convergence_hist.png"), dpi=150)
    plt.close()

    # PCA of perturbations colored by convergence
    pert_np = perturbations.detach().cpu().numpy().reshape(perturbations.shape[0], -1)
    if pert_np.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(pert_np)
        for scale in [0.01, 0.1, 0.5, 1.5, 2.5]:
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(xy[:, 0], xy[:, 1], c=convergence, cmap="jet", s=10, alpha=0.2)
            plt.xlim(-scale, scale)
            plt.ylim(-scale, scale)
            plt.colorbar(sc, label="convergence")
            plt.title("PCA of perturbations")
            plt.tight_layout()
            figure_file = os.path.join(save_dir, f"pca_convergence_scale_{scale}.png")
            plt.savefig(figure_file, dpi=150)
            print("Saved to", figure_file)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Explore convergence around a compressed embedding neighborhood")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to compressed embedding file")
    parser.add_argument("--text", type=str, default=None, help="Text to evaluate against")
    parser.add_argument("--text_file", type=str, default=None, help="Path to a text file to evaluate")
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--norm_types", type=str, nargs="+", default=["l2", "linf", "l1"], help="List of norms: l2 linf l1")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Radii overrides; if None, derive from embedding scale
    parser.add_argument("--l2_radius_max", type=float, default=None)
    parser.add_argument("--l1_radius_max", type=float, default=None)
    parser.add_argument("--linf_radius_max", type=float, default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset.load_from_disk(args.embedding_path)
    assert len(dataset) == 1
    dataset_item = dataset[0]
    embedding = torch.tensor(dataset_item["embedding"], dtype=torch.float32)
    text = dataset_item["text"]
    print("embedding", embedding.shape)
    num_comp = embedding.shape[0]
    hidden = embedding.shape[1]

    model_checkpoint = dataset_item["model_checkpoint"]

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Derive default radii from embedding scale
    scale = float(embedding.std().item()) if embedding.numel() > 1 else 1.0
    if args.l2_radius_max is None:
        args.l2_radius_max = 3.0 * scale * math.sqrt(float(num_comp * hidden))
    if args.l1_radius_max is None:
        args.l1_radius_max = 3.0 * scale * float(num_comp * hidden)
    if args.linf_radius_max is None:
        args.linf_radius_max = 3.0 * scale

    # Sample perturbations
    norms = [n.lower() for n in args.norm_types]
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    perturbations, norm_types, radii = sample_perturbations(
        base=embedding,
        num_points=args.num_points,
        norms=norms,
        l2_radius_max=float(args.l2_radius_max),
        l1_radius_max=float(args.l1_radius_max),
        linf_radius_max=float(args.linf_radius_max),
        generator=generator,
    )

    # Compute convergence
    convergence = compute_convergence_for_points(
        model=model,
        tokenizer=tokenizer,
        base_embedding=embedding,
        perturbations=perturbations,
        text=text,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
    )

    # Prepare outputs
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = "/tmp/compression_neighburhood"
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "norm", "radius", "convergence"])
        for i, (n, r, c) in enumerate(zip(norm_types, radii.tolist(), convergence.tolist())):
            writer.writerow([i, n, r, c])

    # Save plots
    make_plots(save_dir=out_dir, radii=radii, norm_types=norm_types, convergence=convergence, perturbations=perturbations)

    print(f"Saved results and plots to: {out_dir}")


if __name__ == "__main__":
    main()

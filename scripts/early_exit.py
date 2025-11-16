import argparse
import csv
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def _prepare_inputs_with_compression(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    compressed_embedding: torch.Tensor,
    max_sequence_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure model is on the same device as input indices for embedding lookup
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

    comp = compressed_embedding.to(device)
    if comp.dim() == 1:
        comp = comp.unsqueeze(0)
    # [num_comp, hidden]
    num_comp, hidden = comp.shape

    comp_batch = comp.unsqueeze(0)  # [1, num_comp, hidden]
    inputs_embeds = torch.cat([comp_batch, token_embeds], dim=1)  # [1, num_comp+seq_len, hidden]

    comp_mask = torch.ones((1, num_comp), dtype=attention_mask.dtype, device=device)
    attention_with_comp = torch.cat([comp_mask, attention_mask], dim=1)  # [1, num_comp+seq_len]

    return inputs_embeds, attention_with_comp, num_comp


def _get_final_norm_layer(model: AutoModelForCausalLM):
    # Try common locations for the final normalization layer used before lm_head
    # LLaMA/Mistral-like
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    # GPT2-like
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    # Fallback to identity
    return torch.nn.Identity()


@torch.inference_mode()
def compute_layerwise_predictions(
    model: AutoModelForCausalLM,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    device = inputs_embeds.device
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # len = n_layers+1 (embeddings + each layer)
    norm_layer = _get_final_norm_layer(model).to(device)

    # Compute logits per hidden state via final norm + lm_head
    layer_logits: List[torch.Tensor] = []
    for hs in hidden_states:
        # hs: [B, T, H]
        normed = norm_layer(hs)
        logits = model.lm_head(normed)  # [B, T, V]
        layer_logits.append(logits)

    return list(hidden_states), layer_logits


def evaluate_early_exit(
    layer_logits: List[torch.Tensor],
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    comp_len: int,
    pad_token_id: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray]:
    # We evaluate next-token prediction on the text region only.
    # Valid next-token labels are for positions where the next token exists and is not padding.
    # input_ids: [1, seq_len]
    # attention_mask: [1, comp_len + seq_len]
    # device = attention_mask.device
    seq_len = input_ids.shape[1]

    # For next-token prediction, we compare logits at positions comp_len + t to labels input_ids[t+1]
    valid_next_mask = input_ids.new_zeros((seq_len - 1,), dtype=torch.bool)
    # Valid if target (t+1) is not padding
    valid_next_mask = (input_ids[:, 1:] != pad_token_id).squeeze(0)
    # But we also need the model attention mask to have those target positions true
    valid_next_mask = valid_next_mask & (attention_mask[:, comp_len + 1 : comp_len + seq_len].squeeze(0) == 1)

    target_labels = input_ids[:, 1:].squeeze(0)[valid_next_mask]  # [N]

    per_layer_acc: List[float] = []
    earliest_correct_layer: List[int] = []  # per valid target position

    # Collect per-layer predictions for each valid position
    per_layer_correct_matrix: List[np.ndarray] = []

    for logits in layer_logits:
        # logits: [1, comp_len + seq_len, V]
        pos_logits = logits[:, comp_len : comp_len + seq_len - 1, :]  # [1, seq_len-1, V], positions predicting next tokens
        pos_logits = pos_logits.squeeze(0)[valid_next_mask]  # [N, V]
        preds = pos_logits.argmax(dim=-1)  # [N]
        correct = (preds == target_labels.to(preds.device)).detach().cpu().numpy().astype(np.bool_)
        per_layer_correct_matrix.append(correct)

    # Stack: [num_layers, N]
    correct_stack = np.stack(per_layer_correct_matrix, axis=0)
    # Per-layer accuracy across positions
    per_layer_acc = (
        correct_stack.mean(axis=1) if correct_stack.shape[1] > 0 else np.zeros(correct_stack.shape[0], dtype=np.float64)
    )

    # Earliest correct layer per position (first index where correct is True), else -1
    for j in range(correct_stack.shape[1]):
        idxs = np.nonzero(correct_stack[:, j])[0]
        earliest_correct_layer.append(int(idxs[0]) if idxs.size > 0 else -1)

    # Also return indices of valid target positions relative to text positions [0..seq_len-2]
    valid_indices = np.nonzero(valid_next_mask.detach().cpu().numpy())[0]

    return per_layer_acc, correct_stack, earliest_correct_layer, valid_indices


def save_outputs(
    out_dir: str,
    hidden_states: List[torch.Tensor],
    per_layer_acc: np.ndarray,
    earliest_correct_layer: List[int],
    correct_matrix: np.ndarray,
    layer_names: List[str],
    tokenizer: AutoTokenizer,
    target_token_ids_filtered: List[int],
):
    os.makedirs(out_dir, exist_ok=True)

    # Save hidden states tensor list
    torch.save(
        {
            "hidden_states": [hs.detach().cpu() for hs in hidden_states],
        },
        os.path.join(out_dir, "hidden_states.pt"),
    )

    # Save per-layer accuracies
    with open(os.path.join(out_dir, "layer_accuracies.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_index", "layer_name", "accuracy"])
        for i, (name, acc) in enumerate(zip(layer_names, per_layer_acc.tolist())):
            writer.writerow([i, name, acc])

    # Save earliest-correct per token position (next-token targets)
    # Provide target token string for interpretability
    with open(os.path.join(out_dir, "earliest_correct_per_token.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text_position", "target_token_id", "target_token", "earliest_correct_layer"])
        for pos, (tid, layer_idx) in enumerate(zip(target_token_ids_filtered, earliest_correct_layer)):
            token_str = tokenizer.decode([tid])
            writer.writerow([pos + 1, tid, token_str, layer_idx])

    # Save correctness matrix as numpy (layers x positions)
    np.save(os.path.join(out_dir, "correctness_matrix.npy"), correct_matrix)

    # Visualizations
    sns.set(style="whitegrid")
    saved_paths = []

    # 1) Per-layer accuracy line plot
    plt.figure(figsize=(8, 4))
    x = np.arange(len(per_layer_acc))
    plt.plot(x, per_layer_acc, marker="o")
    plt.xticks(x, layer_names, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.xlabel("layer")
    plt.ylabel("next-token accuracy")
    plt.title("Per-layer next-token accuracy")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "per_layer_accuracy.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    saved_paths.append(p1)

    # 2) Earliest-correct layer histogram and CDF
    valid_ecl = [e for e in earliest_correct_layer if e >= 0]
    if len(valid_ecl) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(valid_ecl, bins=np.arange(-0.5, len(layer_names) + 0.5, 1), rwidth=0.8)
        plt.xlabel("earliest correct layer index")
        plt.ylabel("count")
        plt.title("Histogram of earliest-correct layers")
        plt.tight_layout()
        p2 = os.path.join(out_dir, "earliest_correct_hist.png")
        plt.savefig(p2, dpi=150)
        plt.close()
        saved_paths.append(p2)

        # CDF
        plt.figure(figsize=(6, 4))
        counts, bin_edges = np.histogram(valid_ecl, bins=np.arange(-0.5, len(layer_names) + 0.5, 1))
        cdf = np.cumsum(counts) / max(1, np.sum(counts))
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(centers, cdf, marker="o")
        plt.ylim(0.0, 1.0)
        plt.xlabel("earliest correct layer index")
        plt.ylabel("cumulative fraction")
        plt.title("CDF of earliest-correct layer")
        plt.tight_layout()
        p3 = os.path.join(out_dir, "earliest_correct_cdf.png")
        plt.savefig(p3, dpi=150)
        plt.close()
        saved_paths.append(p3)

    # 3) Correctness matrix heatmap (layers x positions)
    if correct_matrix.size > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(correct_matrix.astype(np.float32), cmap="Greens", cbar=True, vmin=0.0, vmax=1.0)
        plt.xlabel("position (valid next-token targets)")
        plt.ylabel("layer index")
        plt.title("Correctness matrix (1=correct)")
        plt.tight_layout()
        p4 = os.path.join(out_dir, "correctness_matrix.png")
        plt.savefig(p4, dpi=150)
        plt.close()
        saved_paths.append(p4)

    # Print saved paths for convenience
    for p in saved_paths:
        print("Saved plot:", p)


def main():
    parser = argparse.ArgumentParser(description="Early-exit analysis via LM head over intermediate hidden states")
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="Path to compressed embedding dataset (HF Dataset.load_from_disk)",
    )
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    # Repro
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load compressed embedding and metadata
    dataset = Dataset.load_from_disk(args.embedding_path)
    assert len(dataset) == 1
    item = dataset[0]
    embedding = torch.tensor(item["embedding"], dtype=torch.float32)
    text = item["text"]
    model_checkpoint = item["model_checkpoint"]

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare inputs (embeddings + compression tokens)
    inputs_embeds, attention_mask, comp_len = _prepare_inputs_with_compression(
        model=model,
        tokenizer=tokenizer,
        text=text,
        compressed_embedding=embedding,
        max_sequence_length=args.max_sequence_length,
    )

    inputs_embeds = inputs_embeds.to(device)
    attention_mask = attention_mask.to(device)

    # Also keep token ids for evaluation
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=args.max_sequence_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)

    # Forward and collect hidden states + layerwise logits via lm_head
    hidden_states, layer_logits = compute_layerwise_predictions(
        model=model.to(device),
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )

    # Evaluate early-exit behavior
    per_layer_acc, correct_matrix, earliest_correct_layer, valid_indices = evaluate_early_exit(
        layer_logits=layer_logits,
        attention_mask=attention_mask,
        input_ids=input_ids,
        comp_len=comp_len,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Prepare layer names (embedding + each block)
    num_layers = len(layer_logits) - 1
    layer_names = ["embeddings"] + [f"layer_{i + 1}" for i in range(num_layers)]

    out_dir = args.output_dir or "/tmp/early_exit_analysis"
    # Build filtered target token ids corresponding to valid positions (returned by evaluate_early_exit)
    target_token_ids_filtered = input_ids[0, 1:].detach().cpu().numpy()[valid_indices]

    save_outputs(
        out_dir=out_dir,
        hidden_states=hidden_states,
        per_layer_acc=per_layer_acc,
        earliest_correct_layer=earliest_correct_layer,
        correct_matrix=correct_matrix,
        layer_names=layer_names,
        tokenizer=tokenizer,
        target_token_ids_filtered=target_token_ids_filtered.tolist(),
    )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()

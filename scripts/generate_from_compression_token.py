import argparse
import os
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def load_single_row(
    dataset_path: str,
    sample_id: Optional[int] = None,
    row_index: int = 0,
) -> Dict[str, Any]:
    ds = Dataset.load_from_disk(dataset_path)
    if len(ds) == 0:
        raise ValueError(f"No rows found in dataset at '{dataset_path}'")

    candidates = list(range(len(ds)))
    if sample_id is not None:
        candidates = [i for i in candidates if int(ds[i].get("sample_id", -1)) == int(sample_id)]
        if not candidates:
            raise ValueError(f"No rows with sample_id={sample_id} in '{dataset_path}'")

    idx = candidates[row_index] if row_index < len(candidates) else candidates[0]
    row = ds[idx]

    embedding = torch.tensor(row["embedding"], dtype=torch.float32)
    info: Dict[str, Any] = {
        "text": row.get("text", ""),
        "embedding": embedding,  # [num_compression_tokens, hidden]
        "num_compression_tokens": int(row.get("num_compression_tokens", embedding.shape[0])),
        "hidden_size": int(row.get("hidden_size", embedding.shape[1] if embedding.dim() == 2 else embedding.shape[-1])),
        "model_checkpoint": row.get("model_checkpoint", None),
        "sample_id": row.get("sample_id", None),
        "stage_index": row.get("stage_index", None),
    }
    return info


@torch.no_grad()
def generate_from_compression(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    compression_tokens: torch.Tensor,  # [1, C, D]
    max_new_tokens: int,
    num_return_sequences: int,
) -> list[str]:
    device = compression_tokens.device
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_id = tokenizer.eos_token_id

    # Prepare batch of prefixes
    if num_return_sequences > 1:
        compression_tokens = compression_tokens.expand(num_return_sequences, -1, -1)
    batch_size, prefix_len, hidden_size = compression_tokens.shape

    # Container for generated token ids
    generated_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)

    input_embeddings = model.get_input_embeddings()

    for _ in range(int(max_new_tokens)):
        if generated_ids.size(1) == 0:
            gen_embeds = torch.empty(batch_size, 0, hidden_size, device=device)
        else:
            gen_embeds = input_embeddings(generated_ids)

        inputs_embeds = torch.cat([compression_tokens, gen_embeds], dim=1)

        attn_prefix = torch.ones((batch_size, prefix_len), dtype=torch.long, device=device)
        attn_gen = torch.ones((batch_size, gen_embeds.size(1)), dtype=torch.long, device=device)
        attention_mask = torch.cat([attn_prefix, attn_gen], dim=1)

        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]

        next_tokens = torch.argmax(logits, dim=-1)

        # Optional early stop if eos is defined and sampled
        if eos_token_id is not None:
            # If a sequence already ended with eos, keep predicting eos for it
            if generated_ids.size(1) > 0:
                ended = generated_ids[:, -1].eq(eos_token_id)
                next_tokens = torch.where(ended, torch.full_like(next_tokens, eos_token_id), next_tokens)

        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)

        # Stop early if all sequences just produced eos and had eos previously
        if eos_token_id is not None and torch.all(next_tokens.eq(eos_token_id)):
            break

    texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from only compressed token embeddings")
    parser.add_argument(
        "--embedding_path", type=str, required=True, help="Path to compressed embedding dataset (load_from_disk)"
    )
    parser.add_argument("--sample_id", type=int, default=None, help="Optional sample_id filter if dataset has multiple rows")
    parser.add_argument("--row_index", type=int, default=0, help="Row index among filtered rows (default: 0)")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Optional HF model name; if omitted, taken from the dataset row",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of generated continuations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save generated samples as a text file",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(args.seed))
    torch.set_default_device(device)

    row = load_single_row(
        dataset_path=args.embedding_path,
        sample_id=args.sample_id,
        row_index=int(args.row_index),
    )

    embedding: torch.Tensor = row["embedding"].to(device)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    if embedding.dim() != 2:
        raise ValueError(f"Expected embedding of shape [C, D], got {tuple(embedding.shape)}")

    num_comp, hidden = embedding.shape
    model_name = args.model_checkpoint or row.get("model_checkpoint")
    if not model_name:
        raise ValueError("Model checkpoint is not provided and not found in the dataset row")

    print(f"Loaded embedding from '{args.embedding_path}'")
    print(f"  sample_id      : {row.get('sample_id')}")
    print(f"  stage_index    : {row.get('stage_index')}")
    print(f"  num_tokens (C) : {num_comp}")
    print(f"  hidden_size (D): {hidden}")
    print(f"  model_checkpoint: {model_name}")

    base_text = str(row.get("text", "") or "")
    if base_text.strip():
        preview = base_text.replace("\n", " ")[:200]
        print(f"Reference text (first 200 chars): {preview}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    compression_tokens = embedding.unsqueeze(0)
    generations = generate_from_compression(
        model=model,
        tokenizer=tokenizer,
        compression_tokens=compression_tokens,
        max_new_tokens=int(args.max_new_tokens),
        num_return_sequences=int(args.num_return_sequences),
    )

    print("\n=== Generated samples from compressed tokens ===")
    for i, text in enumerate(generations):
        print(f"\n--- Sample {i} ---")
        print(text)

    if args.output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for i, text in enumerate(generations):
                f.write(f"### Sample {i}\n{text}\n\n")
        print(f"\nSaved generations to: {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()

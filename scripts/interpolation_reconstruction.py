import argparse
import os

import matplotlib.pyplot as plt
import torch
from scripts.interpolation import (
    collate_stages_by_sample,
    embed_tokens,
    filter_records,
    load_progressive_dataset,
    pick_model_name,
    prepare_model,
    to_tensor_embedding,
    tokenize_text,
)


def main():
    parser = argparse.ArgumentParser(description="Interpolate compression embeddings and evaluate accuracies")
    parser.add_argument("--dataset_path1", type=str, required=True, help="Path to progressive_prefixes dataset")
    parser.add_argument("--dataset_path2", type=str, required=True, help="Path to progressive_prefixes dataset")
    parser.add_argument("--sample_id", type=int, default=None, help="Optional sample_id filter")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="HF model name; inferred if omitted")
    parser.add_argument("--num_points", type=int, default=100, help="Number of evaluation points along t  [0,1]")
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

        compression_tokens = (e0 + e1) / 2
        compression_tokens = compression_tokens.unsqueeze(0)

        attn_ct = torch.ones(
            (compression_tokens.size(0), compression_tokens.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        inputs_embeds_with_ct = torch.cat([compression_tokens, inputs_embeds], dim=1)
        attention_mask_with_ct = torch.cat([attn_ct, attention_mask], dim=1)
        outputs = model(inputs_embeds=inputs_embeds_with_ct, attention_mask=attention_mask_with_ct)
        preds = outputs.logits[:, 0:-1].argmax(dim=-1)
        correct_reconstruction_positions = preds == input_ids[:, :]
        print("Accuracy with interpolated prefix:", correct_reconstruction_positions.float().mean().item())

        plt.bar(range(correct_reconstruction_positions.shape[-1]), correct_reconstruction_positions.cpu()[0].float().numpy())
        figure_path = os.path.join("artifacts", "visualizations", f"correct_reconstruction_positions_{sid}.png")
        plt.title(f"Correct reconstruction positions for sample {sid}")
        plt.xlabel("Token position")
        plt.ylabel("Correct reconstruction")
        plt.savefig(figure_path)
        plt.close()
        print("Saved plot:", figure_path)

        print("Reconstructed words with interpolated prefix:")
        # Print all reconstructed words, make red color for terminal if it is incorrect
        for i in range(correct_reconstruction_positions.shape[-1]):
            if not correct_reconstruction_positions[0, i]:
                print(f"\033[91m{tok.decode(preds[0, i])}", end="\033[0m")
            else:
                print(tok.decode(preds[0, i]), end="")
        print()

        # Make model forward with no prefix
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        preds = outputs.logits[:, 0:-1].argmax(dim=-1)
        correct_reconstruction_positions = preds == input_ids[:, 1:]
        print("Accuracy with no prefix:", correct_reconstruction_positions.float().mean().item())

        plt.bar(range(correct_reconstruction_positions.shape[-1]), correct_reconstruction_positions.cpu()[0].float().numpy())
        figure_path = os.path.join("artifacts", "visualizations", f"correct_reconstruction_positions_no_prefix_{sid}.png")
        plt.title(f"Correct reconstruction positions for sample {sid} with no prefix")
        plt.xlabel("Token position")
        plt.ylabel("Correct reconstruction")
        plt.savefig(figure_path)
        plt.close()
        print("Saved plot:", figure_path)
        # Print all reconstructed words, make red color for terminal if it is incorrect
        print("Reconstructed words no prefix:")
        for i in range(correct_reconstruction_positions.shape[-1]):
            if not correct_reconstruction_positions[0, i]:
                print(f"\033[91m{tok.decode(preds[0, i])}", end="\033[0m")
            else:
                print(tok.decode(preds[0, i]), end="")
        print()

        # Compure prefictions for random compression token embedding
        compression_tokens = torch.rand_like(compression_tokens) * 100

        attn_ct = torch.ones(
            (compression_tokens.size(0), compression_tokens.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        inputs_embeds_with_ct = torch.cat([compression_tokens, inputs_embeds], dim=1)
        attention_mask_with_ct = torch.cat([attn_ct, attention_mask], dim=1)
        outputs = model(inputs_embeds=inputs_embeds_with_ct, attention_mask=attention_mask_with_ct)
        preds = outputs.logits[:, 0:-1].argmax(dim=-1)
        correct_reconstruction_positions = preds == input_ids[:, :]
        print("Accuracy with random compression token embedding:", correct_reconstruction_positions.float().mean().item())

        plt.bar(range(correct_reconstruction_positions.shape[-1]), correct_reconstruction_positions.cpu()[0].float().numpy())
        figure_path = os.path.join(
            "artifacts",
            "visualizations",
            f"correct_reconstruction_positions_random_compression_token_embedding_{sid}.png",
        )
        plt.title(f"Correct reconstruction positions for sample {sid} with random compression token embedding")
        plt.xlabel("Token position")
        plt.ylabel("Correct reconstruction")
        plt.savefig(figure_path)
        plt.close()
        print("Saved plot:", figure_path)

        print("Reconstructed words with random compression token embedding:")
        # Print all reconstructed words, make red color for terminal if it is incorrect
        for i in range(correct_reconstruction_positions.shape[-1]):
            if not correct_reconstruction_positions[0, i]:
                print(f"\033[91m{tok.decode(preds[0, i])}", end="\033[0m")
            else:
                print(tok.decode(preds[0, i]), end="")
        print()


if __name__ == "__main__":
    main()

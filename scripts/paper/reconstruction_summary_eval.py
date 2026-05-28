"""Compute Table 18 metrics for one compressed_prefixes directory.

For each saved sample we:
  1. Load the trained compression embedding and the saved input_ids.
  2. Greedy-decode N = len(input_ids) tokens from the compression embedding
     using ``compression_horizon.inference.generation.generate_from_compression``
     with EOS early-stop disabled (we need exactly N predictions for the
     position-wise mismatch counts).
  3. Compare against input_ids and report:
        final_conv        - mean teacher-forcing accuracy (from artifact)
        greedy_conv       - mean greedy-decoding accuracy
        mismatch@0/1/2    - fraction of samples whose greedy prediction differs
                            from ground truth at position 0 / 1 / 2

Output JSON is consumed by scripts/paper/tables/reconstruction_summary_table.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_horizon.inference.generation import calculate_logits, generate_from_compression
from compression_horizon.utils.launch import resolve_torch_dtype


def _load_prior_summary(path: str) -> dict | None:
    """Load a previously written eval JSON if it exists (for --skip_greedy merge)."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@torch.no_grad()
def _teacher_forced_matches(
    *,
    model: AutoModelForCausalLM,
    compression_tokens: torch.Tensor,  # [1, C, H]
    input_ids: torch.Tensor,  # [N]
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Per-position teacher-forced argmax match: model sees the GROUND-TRUTH prefix.

    Returns a [N] long tensor where entry k is 1 iff argmax of the logits that
    predict ``input_ids[k]`` (given the compression token + ground-truth tokens
    0..k-1) equals ``input_ids[k]``. This isolates "can the model predict
    position k from correct context" from greedy error-compounding.
    """
    num_compression_tokens = compression_tokens.shape[1]
    sequence_embeddings = model.get_input_embeddings()(input_ids.unsqueeze(0)).to(model_dtype)  # [1, N, H]
    attention_mask = torch.ones((1, input_ids.shape[0]), dtype=torch.long, device=device)
    logits = calculate_logits(
        model=model,
        compressed_embeddings=compression_tokens,
        sequence_embeddings=sequence_embeddings,
        attention_mask=attention_mask,
    )  # [1, C + N, vocab]
    # logits[:, C-1+k] predicts input_ids[k].
    pred = logits[0, num_compression_tokens - 1 : -1].argmax(dim=-1)  # [N]
    return (pred == input_ids).to(torch.long)


def _evaluate_one(
    *,
    row: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    model_dtype: torch.dtype,
    num_mismatch_positions: int = 3,
    skip_greedy: bool = False,
) -> dict:
    input_ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
    num_tokens = int(input_ids.shape[0])
    if num_tokens == 0:
        return {
            "sample_id": int(row.get("sample_id", -1)),
            "num_tokens": 0,
            "final_convergence": float(row.get("final_convergence", float("nan"))),
            "greedy_match_rate": None,
            "mismatch_at_position": [],
            "greedy_mismatch_at_position": [],
            "tf_mismatch_at_position": [],
        }

    embedding = torch.tensor(row["embedding"], dtype=torch.float32, device=device)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)  # [1, H]
    compression_tokens = embedding.unsqueeze(0).to(model_dtype)  # [1, C, H]

    k_max = min(num_mismatch_positions, num_tokens)

    # Greedy autoregressive decode (the expensive part: N sequential forwards).
    # Skipped when --skip_greedy to save time when only teacher-forced @k is needed.
    greedy_match_rate = None
    greedy_mismatch_at_position: list[int] = []
    predicted_first3: list[int] = []
    n_pred = 0
    if not skip_greedy:
        _texts, generated_ids = generate_from_compression(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=compression_tokens,
            max_new_tokens=num_tokens,
            num_return_sequences=1,
            return_generated_ids=True,
        )
        generated = generated_ids[0].to(device)
        # generate_from_compression aborts on EOS, so it may return < num_tokens
        # predictions. Pad the tail with EOS (or pad_token) so position-wise
        # mismatch counts at indices 0/1/2 stay well-defined and any post-EOS
        # gap counts as a mismatch against the ground-truth tokens.
        n_pred = int(generated.shape[0])
        if n_pred < num_tokens:
            fill_id = (
                tokenizer.eos_token_id
                if tokenizer.eos_token_id is not None
                else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            )
            pad = torch.full((num_tokens - n_pred,), int(fill_id), dtype=generated.dtype, device=device)
            generated = torch.cat([generated, pad], dim=0)
        greedy_matches = (generated == input_ids).to(torch.long)
        greedy_match_rate = float(greedy_matches.float().mean().item())
        greedy_mismatch_at_position = [int((1 - greedy_matches[k]).item()) for k in range(k_max)]
        predicted_first3 = generated[: min(3, num_tokens)].cpu().tolist()

    # Teacher-forced per-position matches (ground-truth prefix at each step): one forward.
    tf_matches = _teacher_forced_matches(
        model=model,
        compression_tokens=compression_tokens,
        input_ids=input_ids,
        device=device,
        model_dtype=model_dtype,
    )
    tf_mismatch_at_position = [int((1 - tf_matches[k]).item()) for k in range(k_max)]

    return {
        "sample_id": int(row.get("sample_id", -1)),
        "num_tokens": num_tokens,
        "num_generated": n_pred,
        "final_convergence": float(row.get("final_convergence", float("nan"))),
        "greedy_match_rate": greedy_match_rate,
        "tf_match_rate": float(tf_matches.float().mean().item()),
        # `mismatch_at_position` stays = greedy (back-compat with reconstruction_summary_table.py).
        "mismatch_at_position": greedy_mismatch_at_position[: min(3, k_max)],
        "greedy_mismatch_at_position": greedy_mismatch_at_position,
        "tf_mismatch_at_position": tf_mismatch_at_position,
        "predicted_first3": predicted_first3,
        "target_first3": input_ids[: min(3, num_tokens)].cpu().tolist(),
    }


def _mismatch_curve(valid: list[dict], field: str, num_positions: int) -> list[float | None]:
    """Per-position fraction of samples that mismatch (1.0 = all wrong)."""
    curve: list[float | None] = []
    for k in range(num_positions):
        miss = [r[field][k] for r in valid if len(r.get(field, [])) > k]
        curve.append((sum(miss) / len(miss)) if miss else None)
    return curve


def _aggregate(per_sample: list[dict], num_mismatch_positions: int) -> dict:
    valid = [r for r in per_sample if r["num_tokens"] > 0]
    n = len(valid)
    if n == 0:
        return {"num_samples": 0}

    final_conv = sum(r["final_convergence"] for r in valid) / n
    # greedy_match_rate may be None when --skip_greedy was used.
    greedy_vals = [r["greedy_match_rate"] for r in valid if r.get("greedy_match_rate") is not None]
    greedy_conv = (sum(greedy_vals) / len(greedy_vals)) if greedy_vals else None
    tf_conv = sum(r.get("tf_match_rate", float("nan")) for r in valid) / n

    # Back-compat scalar fields consumed by reconstruction_summary_table.py (greedy @0/1/2).
    mismatch = {}
    for k in (0, 1, 2):
        miss = [r["greedy_mismatch_at_position"][k] for r in valid if len(r["greedy_mismatch_at_position"]) > k]
        mismatch[f"mismatch_at_{k}"] = (sum(miss) / len(miss)) if miss else None

    greedy_curve = _mismatch_curve(valid, "greedy_mismatch_at_position", num_mismatch_positions)
    tf_curve = _mismatch_curve(valid, "tf_mismatch_at_position", num_mismatch_positions)

    return {
        "num_samples": n,
        "final_convergence_mean": final_conv,
        "greedy_match_rate_mean": greedy_conv,
        "tf_match_rate_mean": tf_conv,
        **mismatch,
        "num_mismatch_positions": num_mismatch_positions,
        "greedy_mismatch_curve": greedy_curve,
        "tf_mismatch_curve": tf_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Table 18 metrics for a single experiment")
    parser.add_argument(
        "--compressed_prefixes_path",
        required=True,
        help="Path to the compressed_prefixes/ directory saved by FullCrammingTrainer.",
    )
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        help="HF model name. If omitted, taken from the first dataset row.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype for evaluation (default: bfloat16, matching training).",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Where to write the aggregated metrics JSON. Default: <parent>/table_18_eval.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_mismatch_positions",
        type=int,
        default=3,
        help="How many leading positions to record per-position mismatch for "
        "(greedy AND teacher-forced). Use e.g. 25 to inspect the error curve.",
    )
    parser.add_argument(
        "--skip_greedy",
        action="store_true",
        help="Skip the (expensive) greedy autoregressive decode and compute only "
        "teacher-forced metrics. Greedy aggregate fields are preserved from a prior "
        "eval JSON at the output path if present, so the Greedy conv. column survives.",
    )
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))

    ds = Dataset.load_from_disk(args.compressed_prefixes_path)
    if len(ds) == 0:
        raise SystemExit(f"No rows in {args.compressed_prefixes_path}")

    model_checkpoint = args.model_checkpoint or ds[0].get("model_checkpoint")
    if not model_checkpoint:
        raise SystemExit("model_checkpoint not provided and missing from dataset rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_torch_dtype(args.dtype)
    print(f"Loading {model_checkpoint} ({model_dtype}) on {device}")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, dtype=model_dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    per_sample: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        if "input_ids" not in row or row["input_ids"] is None:
            raise SystemExit(
                f"Row {i} has no 'input_ids' field. Was training run BEFORE the camera-ready "
                "refactor that added input_ids to compressed_prefixes? Retrain the experiment."
            )
        result = _evaluate_one(
            row=row,
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_dtype=model_dtype,
            num_mismatch_positions=args.num_mismatch_positions,
            skip_greedy=args.skip_greedy,
        )
        greedy_str = "skip" if result["greedy_match_rate"] is None else f"{result['greedy_match_rate']:.4f}"
        print(
            f"[{i+1}/{len(ds)}] sample_id={result['sample_id']} "
            f"final={result['final_convergence']:.4f} greedy={greedy_str} "
            f"tf={result.get('tf_match_rate', float('nan')):.4f} "
            f"tf_mismatch@0/1/2={result['tf_mismatch_at_position']}"
        )
        per_sample.append(result)

    summary = _aggregate(per_sample, args.num_mismatch_positions)
    summary["compressed_prefixes_path"] = os.path.abspath(args.compressed_prefixes_path)
    summary["model_checkpoint"] = model_checkpoint
    summary["per_sample"] = per_sample

    out_path = args.output_json or str(Path(args.compressed_prefixes_path).parent / "table_18_eval.json")

    # When greedy was skipped, carry over greedy aggregates from a prior eval JSON
    # so the "Greedy conv." column (and legacy mismatch_at_k) is not lost.
    if args.skip_greedy:

        def _needs_fill(x) -> bool:
            # Missing if None, empty list, or a list whose entries are all None.
            if x is None or x == []:
                return True
            if isinstance(x, list) and all(v is None for v in x):
                return True
            return False

        prior = _load_prior_summary(out_path)
        if prior is not None:
            for key in ("greedy_match_rate_mean", "mismatch_at_0", "mismatch_at_1", "mismatch_at_2", "greedy_mismatch_curve"):
                if _needs_fill(summary.get(key)) and not _needs_fill(prior.get(key)):
                    summary[key] = prior[key]
            print(f"[skip_greedy] preserved greedy aggregates from prior {out_path}")
        else:
            print(f"[skip_greedy] WARNING: no prior JSON at {out_path}; greedy fields will be empty")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary to: {out_path}")
    scalar = {k: v for k, v in summary.items() if k not in ("per_sample", "greedy_mismatch_curve", "tf_mismatch_curve")}
    print(json.dumps(scalar, indent=2))

    # Per-position mismatch curves (fraction of samples wrong at each position).
    gc = summary.get("greedy_mismatch_curve") or []
    tc = summary.get("tf_mismatch_curve") or []
    if gc:
        print("\npos |  greedy_mismatch  teacher_forced_mismatch")
        print("----+-----------------------------------------")
        for k in range(len(gc)):
            g = f"{gc[k]*100:5.1f}%" if gc[k] is not None else "  —  "
            t = f"{tc[k]*100:5.1f}%" if k < len(tc) and tc[k] is not None else "  —  "
            print(f"{k:3d} |      {g}              {t}")


if __name__ == "__main__":
    main()

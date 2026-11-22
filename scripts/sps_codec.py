"""Certify a pretrained SPS stage, or recover its exact token IDs from a packet."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from datasets import load_from_disk
from scripts.pretrained_sps_experiments import experiment_configs
from scripts.pretrained_sps_worker import model_and_tokenizer

from compression_horizon.inference.sps_codec import certify, decompress


def load_stage(shard, sample_id):
    report = json.loads((shard / "single_sample_validation.json").read_text())
    exp = resolve_experiment(report["experiment"]["name"])
    if report["experiment"] != exp or report["embedding_role"] != "shared_predict":
        raise ValueError("Shard does not match the registered shared-prediction experiment")
    (sample,) = (r for r in report["samples"] if r["sample_id"] == sample_id)
    length = sample["horizon"]
    if not length or length != len(sample["target_ids"]):
        raise ValueError("Sample has no selected converged prefix")
    stages = load_from_disk(str(shard / "predict_prefixes"))
    (index,) = (
        i for i, (sid, n) in enumerate(zip(stages["sample_id"], stages["num_input_tokens"])) if sid == sample_id and n == length
    )
    model, _ = model_and_tokenizer(exp)
    model.cuda()
    z = torch.tensor(stages[index]["embedding"])[None]
    return exp, model, z, sample


@torch.inference_mode()
def first_error_probe(model, vector, sample):
    target = sample["target_ids"]
    observed = sample["generated_ids"]
    pos = next((i for i, (a, b) in enumerate(zip(target, observed)) if a != b), None)
    if pos is None:
        return {"first_mismatch": None, "generated_length": len(observed)}
    bos = model.config.bos_token_id
    prefix = [bos] + target[:pos]
    cases = {
        "growing": prefix,
        "fixed_bos_suffix": prefix + [bos] * (len(target) - pos),
        "teacher": [bos] + target,
    }
    scores = {}
    for case, ids in cases.items():
        x = torch.tensor([ids], device=model.device)
        logits = (
            model(input_ids=x, attention_mask=torch.ones_like(x), predict_embeddings=vector.to(model.device, model.dtype))
            .logits[0, pos]
            .float()
        )
        scores[case] = {
            "argmax": int(logits.argmax()),
            "target_minus_observed": float(logits[target[pos]] - logits[observed[pos]]),
        }
    return dict(first_mismatch=pos, target=target[pos], observed=observed[pos], scores=scores)


def resolve_experiment(name):
    for exp in experiment_configs:
        if exp["name"] == name and exp["arm"] != "ar":
            return exp
    raise ValueError(f"Unknown SPS experiment {name!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("certify")
    encode.add_argument("shard", type=Path)
    encode.add_argument("sample_id", type=int)
    encode.add_argument("packet", type=Path)
    probe = commands.add_parser("probe", help="compare forward shapes at the first archived greedy mismatch")
    probe.add_argument("shard", type=Path)
    probe.add_argument("sample_id", type=int)
    decode = commands.add_parser("decode")
    decode.add_argument("packet", type=Path)
    args = parser.parse_args()
    if args.command == "probe":
        _, model, z, sample = load_stage(args.shard, args.sample_id)
        print(json.dumps(first_error_probe(model, z, sample)))
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if args.command == "certify":
        exp, model, z, sample = load_stage(args.shard, args.sample_id)
        packet = certify(model, z, sample["target_ids"])
        packet["experiment"] = exp["name"]
        with NamedTemporaryFile(dir=args.packet.parent) as f:
            torch.save(packet, f)
            f.flush()
            os.link(f.name, args.packet)
    else:
        packet = torch.load(args.packet, weights_only=True, map_location="cpu")
        model, _ = model_and_tokenizer(resolve_experiment(packet["experiment"]))
        print(json.dumps(decompress(model.cuda(), packet)))


if __name__ == "__main__":
    main()

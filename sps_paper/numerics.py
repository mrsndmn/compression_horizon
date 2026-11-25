"""Export measured Qwen stability comparisons and render their appendix tables offline."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "data" / "numerics.json"
LENGTHS = (128, 512, 1024, 2048, 4096, 8192, 32768)


def export(root):
    long = root / "qwen_original_pg19_prefix_logits_32768"
    groups = defaultdict(list)
    with (long / "positions.csv").open() as src:
        for row in csv.DictReader(src):
            groups[int(row["length"])].append(row)
    prefix = {}
    for n in LENGTHS:
        rows = groups[n]
        if [int(r["position"]) for r in rows] != list(range(1, n + 1)):
            raise ValueError(f"Incomplete prefix comparison for length {n}")
        prefix[str(n)] = dict(
            flips=sum(r["top1"] != r["top1_at_32768"] for r in rows),
            mean_abs=sum(float(r["mean_abs_delta"]) for r in rows) / n,
            overlap=[sum(int(r[f"overlap{k}"]) for r in rows) / (n * k) for k in (3, 5, 10)],
        )
    with (long / "kv_summary.csv").open() as src:
        kv = list(csv.DictReader(src))
    fixed = json.loads((root / "qwen_fixed_attention/prefill.json").read_text())
    return dict(prefix=prefix, kv=kv, fixed=fixed)


def table(headers, rows):
    lines = [r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def render(data):
    kv = {(int(r["length"]), r["comparison"]): r for r in data["kv"]}
    rows = []
    for n in LENGTHS:
        p = data["prefix"][str(n)]
        for key, count in [("one_token_vs_no_cache", 32), ("chunk_vs_no_cache", n - 64), ("chunk_vs_32k_chunk", n - 64)]:
            if int(kv[n, key]["positions"]) != count:
                raise ValueError(f"Unexpected number of positions for {n} / {key}")
        rows.append(
            [
                f"{n:,}",
                p["flips"],
                f"{p['mean_abs']:.4f}",
                *(f"{100*x:.2f}" for x in p["overlap"]),
                *(kv[n, key]["top1_changes"] for key in ("one_token_vs_no_cache", "chunk_vs_no_cache", "chunk_vs_32k_chunk")),
            ]
        )
    length_table = table(["Length", "Flips", "MAE", "Top-3", "Top-5", "Top-10", "Step", "Split", "Split vs. 32k"], rows)
    fixed = data["fixed"]
    cfg = fixed["config"]
    if (cfg["length"], cfg["splits"], cfg["decode_steps"]) != (8192, [64, 65, 8191], 32):
        raise ValueError("Attention probe dimensions differ from the published comparisons")
    rows = []
    for mode, name in [("sdpa", "Default SDPA"), ("math", "Math SDPA"), ("fixed", "Fixed tiles")]:
        splits = {r["split"]: r["suffix"] for r in fixed["comparisons"] if r["mode"] == mode}
        if (
            any(splits[n]["positions"] != cfg["length"] - n for n in cfg["splits"])
            or fixed["one_token_steps"][mode]["positions"] != cfg["decode_steps"]
        ):
            raise ValueError(f"Unexpected number of scored positions for {mode}")
        rows.append(
            [
                name,
                *(f"{splits[n]['unequal_logits']:,} / {splits[n]['top1_changes']}" for n in fixed["config"]["splits"]),
                f"{fixed['one_token_steps'][mode]['unequal_logits']:,} / {fixed['one_token_steps'][mode]['top1_changes']}",
            ]
        )
    attention_table = table(["Attention", "Split at 64", "Split at 65", "Split at 8,191", "Last 32 steps"], rows)

    return {"numerics_attention": attention_table, "numerics_length": length_table}


def native_replay_table():
    rows = []
    for family, label in (("standard", "Standard"), ("sps", "SPS")):
        with (HERE / "data" / f"{family}_wide_cache.csv").open() as src:
            samples = list(csv.DictReader(src))
        if sorted(int(r["sample_id"]) for r in samples) != list(range(48)) or any(r["family"] != family for r in samples):
            raise ValueError(f"Incomplete or mislabeled {family} cohort")
        exact = []
        for key in ("teacher_exact", "uncached_exact", "cached_exact"):
            values = [r[key] for r in samples]
            if not set(values) <= {"True", "False"}:
                raise ValueError(f"Invalid {family} {key} flag")
            exact.append(f"{values.count('True')}/{len(samples)}")
        mean = sum(int(r["horizon"]) for r in samples) / len(samples)
        changed = sum(int(r["cache_vs_full_unequal"]) for r in samples)
        scored = sum(int(r["logits_scored"]) for r in samples)
        rows.append([label, f"{mean:.2f}", *exact, f"{changed:,} / {scored:,}"])
    return table(["Model", "Mean horizon", "Teacher", "Greedy", "Greedy + KV", "Unequal / scored logits"], rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-from", type=Path, help="Analysis artifact directory; omit for offline rendering")
    args = parser.parse_args()
    data = export(args.export_from) if args.export_from else json.loads(SNAPSHOT.read_text())
    tables = render(data)
    tables["numerics_native_replay"] = native_replay_table()
    if args.export_from:
        SNAPSHOT.write_text(json.dumps(data, indent=2) + "\n")
    for name, body in tables.items():
        (HERE / "tables" / (name + ".tex")).write_text(body)

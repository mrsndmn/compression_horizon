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
        assert [int(r["position"]) for r in rows] == list(range(1, n + 1))
        prefix[n] = dict(
            flips=sum(r["top1"] != r["top1_at_32768"] for r in rows),
            mean_abs=sum(float(r["mean_abs_delta"]) for r in rows) / n,
            overlap=[sum(int(r[f"overlap{k}"]) for r in rows) / (n * k) for k in (3, 5, 10)],
        )
    with (long / "kv_summary.csv").open() as src:
        kv = list(csv.DictReader(src))
    fixed = json.loads((root / "qwen_fixed_attention/prefill.json").read_text())
    SNAPSHOT.write_text(json.dumps(dict(prefix=prefix, kv=kv, fixed=fixed), indent=2) + "\n")


def table(name, headers, rows):
    lines = [r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    (HERE / "tables" / (name + ".tex")).write_text("\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n")


def render():
    data = json.loads(SNAPSHOT.read_text())
    kv = {(int(r["length"]), r["comparison"]): r for r in data["kv"]}
    rows = []
    for n in LENGTHS:
        p = data["prefix"][str(n)]
        rows.append(
            [
                f"{n:,}",
                p["flips"],
                f"{p['mean_abs']:.4f}",
                *(f"{100*x:.2f}" for x in p["overlap"]),
                *(kv[n, key]["top1_changes"] for key in ("one_token_vs_no_cache", "chunk_vs_no_cache", "chunk_vs_32k_chunk")),
            ]
        )
    table("numerics_length", ["Length", "Flips", "MAE", "Top-3", "Top-5", "Top-10", "Step", "Split", "Split vs. 32k"], rows)
    fixed = data["fixed"]
    rows = []
    for mode, name in [("sdpa", "Default SDPA"), ("math", "Math SDPA"), ("fixed", "Fixed tiles")]:
        splits = {r["split"]: r["suffix"] for r in fixed["comparisons"] if r["mode"] == mode}
        rows.append(
            [
                name,
                *(f"{splits[n]['unequal_logits']:,} / {splits[n]['top1_changes']}" for n in fixed["config"]["splits"]),
                f"{fixed['one_token_steps'][mode]['unequal_logits']:,} / {fixed['one_token_steps'][mode]['top1_changes']}",
            ]
        )
    table("numerics_attention", ["Attention", "Split at 64", "Split at 65", "Split at 8,191", "Last 32 steps"], rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-from", type=Path, help="Analysis artifact directory; omit for offline rendering")
    args = parser.parse_args()
    if args.export_from:
        export(args.export_from)
    render()

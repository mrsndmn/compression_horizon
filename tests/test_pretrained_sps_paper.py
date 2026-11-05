"""Portable pretrained evidence must retain pairing, stopping and replay semantics."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from sps_paper import pretrained as paper

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return paper.load(ROOT / "sps_paper/data")


def test_completed_bundle_reaudits_all_native_targets_and_retains_stopping_reasons(bundle):
    metrics = paper.derive(bundle)
    assert set(metrics["runs"]) == set(paper.NAMES)
    for run in metrics["runs"].values():
        assert run["n"] == run["teacher_exact"] == 48
        assert run["stopping_counts"] == {"total_budget": 48}
    assert metrics["runs"]["smol_ce"]["horizon"]["mean"] > metrics["runs"]["smol_ar"]["horizon"]["mean"]
    assert metrics["runs"]["qwen_ce"]["horizon"]["mean"] < metrics["runs"]["qwen_ar"]["horizon"]["mean"]
    assert (
        metrics["runs"]["smol_ce"]["greedy_initial_prefix"]["mean"]
        < metrics["runs"]["smol_ar"]["greedy_initial_prefix"]["mean"]
    )
    for family in paper.LABELS:
        for arm in ("ce", "kd"):
            a = metrics["runs"][family + "_" + arm]["samples"]
            b = metrics["runs"][family + "_ar"]["samples"]
            expected = sum(x["horizon"] - y["horizon"] for x, y in zip(a, b)) / len(a)
            assert metrics["paired"][f"{family}_{arm}_minus_ar"]["horizon"]["mean"] == expected


@pytest.mark.parametrize("fault", ["mask", "target", "document", "pairing", "uuid", "selection", "stage", "partial", "role"])
def test_compact_export_rejects_corruption_before_reporting(fault, bundle):
    data = copy.deepcopy(bundle)
    run = data["runs"]["qwen_ce"]
    if fault == "mask":
        data["families"]["qwen"]["targets"][0]["attention_mask"][0] = 0
    elif fault == "target":
        data["families"]["qwen"]["targets"][0]["input_ids"][0] += 1
    elif fault == "document":
        data["raw_documents"][0]["text"] += "changed"
    elif fault == "pairing":
        data["families"]["smol"]["manifest"]["document_hashes"].reverse()
    elif fault == "uuid":
        run["shards"][0]["job"]["result"]["job_name"] = data["runs"]["qwen_ar"]["shards"][0]["job"]["result"]["job_name"]
    elif fault == "selection":
        run["shards"][0]["report"]["samples"][0]["target_ids"][0] += 1
    elif fault == "stage":
        run["shards"][0]["stages"][0]["steps_taken"] = -1
    elif fault == "partial":
        run["shards"].pop()
    else:
        run["shards"][0]["report"]["embedding_role"] = "ordinary_memory"
    with pytest.raises(ValueError):
        paper.derive(data)


def test_checksum_is_checked_before_reading_compact_evidence(tmp_path):
    paper.json_write(tmp_path / "pretrained_manifest.json", dict(schema=1, files={"pretrained.json.gz": "0" * 64}))
    (tmp_path / "pretrained.json.gz").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="checksum"):
        paper.load(tmp_path)


def test_restore_materializes_exact_masks_and_ids_and_rejects_overwrite(bundle, tmp_path):
    from datasets import load_from_disk
    from scripts.pretrained_sps_worker import target_digest

    paper.restore_data(bundle, tmp_path)
    for family, source in bundle["families"].items():
        data = load_from_disk(str(tmp_path / "data" / family))
        assert target_digest(data) == source["manifest"]["targets_sha256"]
    assert paper.json_read(tmp_path / "data/raw.json") == bundle["raw_documents"]
    assert paper.json_read(tmp_path / "data/design.json") == bundle["data_design"]
    with pytest.raises(FileExistsError):
        paper.restore_data(bundle, tmp_path)


def test_render_from_foreign_workdir_needs_no_training_imports(tmp_path):
    script = ROOT / "sps_paper/pretrained.py"
    driver = """
import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','datasets','transformers','mls'}:
        raise AssertionError('training import during offline rendering: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=[sys.argv[1],'render','--output-root',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", driver, str(script), str(tmp_path)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("pretrained_horizons", "pretrained_greedy"):
        assert (tmp_path / f"figures/{name}.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "tables/pretrained_results.tex").is_file()
    assert (tmp_path / "data/pretrained_metrics.json").is_file()


def test_plan_covers_actual_shards_without_submitting_and_is_valid_shell(tmp_path):
    text = paper.plan()
    result = subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert text.count(" --preflight") == 2
    assert text.count(" --experiment ") == 50  # 48 shards plus family preflights.
    assert "git -C" in text and " archive " in text and "restore-data" in text
    assert "mls job" not in text and "--phase submit" not in text


@pytest.mark.parametrize("preflight", [False, True])
def test_replay_dispatches_archived_worker_with_only_paths_relocated(bundle, tmp_path, monkeypatch, preflight):
    from scripts import pretrained_sps_experiments as registry

    archive = tmp_path / "archive"
    scripts = archive / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "__init__.py").touch()
    (scripts / "pretrained_sps_experiments.py").write_text(
        "from pathlib import Path\nROOT=Path('/original/output')\nSOURCE_ROOT=Path('/original/inputs')\n"
        "experiment_configs=[dict(name='smol_ce',family='smol',source_path='/original/inputs/hf/model',checkpoint='/original/inputs/checkpoint.pt')]\n"
    )
    (scripts / "pretrained_sps_worker.py").write_text(
        "import json\nfrom scripts.pretrained_sps_experiments import ROOT,experiment_configs\n"
        "def train_shard(exp,offset,count,output):\n"
        "    (output/'called.json').write_text(json.dumps(dict(exp=exp,offset=offset,count=count,root=str(ROOT))))\n"
        "def preflight(family,output):\n"
        "    (output/'called.json').write_text(json.dumps(dict(family=family,exp=experiment_configs[0],root=str(ROOT))))\n"
    )
    monkeypatch.setattr(registry, "code_digest", lambda _: bundle["runs"]["smol_ce"]["summary"]["code_digest"])
    output_root = tmp_path / "new"
    (output_root / "data").mkdir(parents=True)
    (output_root / "data/completed").touch()
    inputs = tmp_path / "inputs"
    paper.replay_one(bundle, archive, output_root, inputs, "smol_ce", 0, preflight=preflight)
    call = json.loads((output_root / ("preflight" if preflight else "runs") / "smol_ce/0/called.json").read_text())
    assert call["root"] == str(output_root)
    assert call["exp"]["checkpoint"] == str(inputs / "checkpoint.pt")
    assert call["exp"]["source_path"] == str(inputs / "hf/model")
    if preflight:
        assert call["family"] == "smol"
    else:
        assert call["offset"] == 0 and call["count"] == len(bundle["runs"]["smol_ce"]["shards"][0]["report"]["samples"])
    with pytest.raises(ValueError, match="fresh"):
        paper.replay_one(bundle, archive, output_root, inputs, "smol_ce", 0, preflight=preflight)

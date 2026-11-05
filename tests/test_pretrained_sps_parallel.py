import json
import uuid
from types import SimpleNamespace

import pytest
from scripts import run_pretrained_sps_parallel as parallel
from scripts.pretrained_sps_experiments import write_json


@pytest.fixture
def configured(tmp_path, monkeypatch):
    base = parallel.base
    nfs1, nfs2 = tmp_path / "nfs1", tmp_path / "nfs2"
    nfs1.mkdir()
    nfs2.mkdir()
    root = nfs2 / "artifacts"
    root.symlink_to(nfs1, target_is_directory=True)
    monkeypatch.setattr(base, "ROOT", root)
    monkeypatch.setattr(parallel, "NFS2_ROOT", nfs2 / "artifacts_nfs2")
    monkeypatch.setattr(base, "code_digest", lambda _: "b" * 64)
    monkeypatch.setattr(parallel, "code_digest", lambda _: "b" * 64)
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 2**30))
    monkeypatch.setattr(
        base.subprocess,
        "check_output",
        lambda args, **kw: "c" * 40 if "rev-parse" in args else b"",
    )

    def stage_code(commit, stage):
        stage.mkdir(parents=True, exist_ok=True)
        script = stage / "scripts/run_pretrained_sps_parallel.py"
        script.parent.mkdir(exist_ok=True)
        script.write_bytes(parallel.Path(parallel.__file__).read_bytes())

    monkeypatch.setattr(base, "stage_code", stage_code)
    (root / "data").mkdir()
    (root / "data/completed").touch()
    for family in parallel.FAMILIES:
        path = base.output_path("preflight", family)
        write_json(path / "preflight.json", dict(code_digest="b" * 64))
        (path / "completed").touch()
    ar = base.output_path("runs", "smol_ar")
    ar.mkdir(parents=True)
    (ar / "ongoing").write_text("running AR must stay on NFS1")
    return root, nfs1, nfs2


class Client:
    def __init__(self, fail_at=None):
        self.payloads = []
        self.fail_at = fail_at

    def run_job(self, *, payload):
        self.payloads.append(payload)
        if len(self.payloads) == self.fail_at:
            raise TimeoutError("Unknown provider outcome")
        return dict(job_name="lm-mpi-job-" + str(uuid.UUID(int=len(self.payloads))))


def ar_active(n=8):
    return {f"CH: pretrained_sps_runs_smol_ar_{i * 6} bbbbbbbbbbbb" for i in range(n)}


def test_parallel_dry_has_no_filesystem_or_api_side_effects(configured, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["parallel", "--dry"])
    parallel.main()
    plans = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(plans) == 16 and {p["experiment"] for p in plans} == {
        "smol_ce",
        "smol_kd",
    }
    assert len({p["physical_output"] for p in plans}) == 16
    assert not parallel.NFS2_ROOT.exists()


def test_submit_sixteen_alongside_ar_routes_outputs_and_code_to_nfs2(configured):
    root, nfs1, nfs2 = configured
    assert parallel.require_ready() == "b" * 64
    client = Client()
    result = parallel.submit_with_client(client, dict(region="SR004"), ar_active(), "b" * 64)
    assert len(client.payloads) == 16
    assert {name: len(ids) for name, ids in result["experiments"].items()} == {
        "smol_ce": 8,
        "smol_kd": 8,
    }
    assert len({j for ids in result["experiments"].values() for j in ids}) == 16
    assert (parallel.base.output_path("runs", "smol_ar") / "ongoing").read_text() == "running AR must stay on NFS1"
    for name in parallel.NAMES:
        path = parallel.base.output_path("runs", name)
        assert path.resolve().is_relative_to(nfs2)
        assert not path.resolve().is_relative_to(nfs1)
    _, _, stage = parallel.base.payload("runs", "smol_ce")
    assert stage.resolve().is_relative_to(nfs2)
    active = ar_active() | {p["job_desc"].split(" #", 1)[0] for p in client.payloads}
    assert parallel.require_ready() == "b" * 64
    assert parallel.submit_with_client(client, dict(region="SR004"), active, "b" * 64) == result
    assert len(client.payloads) == 16


@pytest.mark.parametrize(
    "problem",
    [
        "full",
        "preflight",
        "foreign_directory",
        "foreign_link",
        "physical_data",
        "shard_link",
    ],
)
def test_unsafe_state_is_rejected_without_submission_or_relocation(configured, monkeypatch, problem):
    root, _, nfs2 = configured
    canonical = parallel.base.output_path("runs", "smol_ce").parent
    target = parallel.physical_path("smol_ce", "b" * 64)
    if problem == "full":
        monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    elif problem == "preflight":
        (parallel.base.output_path("preflight", "qwen") / "completed").unlink()
    elif problem == "foreign_directory":
        canonical.mkdir()
        (canonical / "keep").write_text("do not move")
    elif problem == "foreign_link":
        canonical.symlink_to(root / "data", target_is_directory=True)
    elif problem == "physical_data":
        target.mkdir(parents=True)
        (target / "keep").touch()
    else:
        target.mkdir(parents=True)
        canonical.symlink_to(target, target_is_directory=True)
        (target / "shard00").symlink_to(root / "data", target_is_directory=True)
    with pytest.raises(RuntimeError):
        parallel.require_ready()
    assert (parallel.base.output_path("runs", "smol_ar") / "ongoing").exists()
    if problem == "foreign_directory":
        assert (canonical / "keep").read_text() == "do not move"


def test_capacity_guard_accounts_for_other_models_before_any_submission(configured):
    client = Client()
    with pytest.raises(RuntimeError, match="concurrency limit"):
        parallel.submit_with_client(client, dict(region="SR004"), ar_active(9), "b" * 64)
    assert not client.payloads and not parallel.NFS2_ROOT.exists()


def test_uncertain_partial_submission_cannot_be_repeated(configured):
    client = Client(fail_at=2)
    with pytest.raises(TimeoutError):
        parallel.submit_with_client(client, dict(region="SR004"), ar_active(), "b" * 64)
    active = ar_active() | {client.payloads[0]["job_desc"].split(" #", 1)[0]}
    with pytest.raises(RuntimeError, match="uncertain"):
        parallel.submit_with_client(client, dict(region="SR004"), active, "b" * 64)
    assert len(client.payloads) == 2
    partial = json.loads((parallel.NFS2_ROOT / "latest-parallel.json").read_text())
    assert len(partial["experiments"]["smol_ce"]) == 1


def test_inactive_record_requires_reconciliation_instead_of_resubmission(configured):
    output = parallel.base.output_path("runs", "smol_ce")
    write_json(
        output / "job.json",
        dict(result=dict(job_name="lm-mpi-job-" + str(uuid.UUID(int=1)))),
    )
    client = Client()
    with pytest.raises(RuntimeError, match="not active"):
        parallel.submit_with_client(client, dict(region="SR004"), ar_active(), "b" * 64)
    assert not client.payloads


def test_parallel_lock_rejects_a_second_submitter_and_releases(configured):
    with parallel.submission_lock():
        with pytest.raises(RuntimeError, match="in progress"):
            with parallel.submission_lock():
                pytest.fail("Second submitter acquired lock")
    with parallel.submission_lock():
        pass

"""Background watcher/eval-driver for the Llama-3.1-8B transformer-depth ablation.

Llama-3.1-8B companion to ``scripts/jobs/watch_finetune_truncated.py`` (SmolLM2-1.7B),
structured like ``scripts/jobs/watch_width_ablation.py`` (single self-contained poll
loop that exits with a ``DONE.json`` summary so a background task notifies on
completion).

This ablation has ONE arm (finetuned only -- the user chose to skip raw-truncated
rows for Llama): per truncated depth N in {1,2,4} we

  * finetune ``Meta-Llama-3.1-8B-firstlast{N}`` -> ``…-firstlast{N}-ftw`` with the
    width/compression-head recipe (run_jobs_finetune_truncated_llama.py), then
  * evaluate it on progressive cramming (PG19, baseline init random0.02) via
    run_jobs_layer_ablation_ft_llama.py.

The watcher polls all three targets; as each finetune output becomes ready it
submits that target's progressive-eval job and retries a failed eval (saving
``mls job logs`` + cleaning the partial output dir) up to ``--max-retries``. It
EXITS once every target is terminal, writing ``DONE.json``. The full-depth
(32-layer) ``sl_4096_Meta-Llama-3.1-8B_ds_pg19_1k_limit_50_lr_0.1`` run already has
data and is the table's reference row -- it is reported but not waited on.

Training must already be submitted (run the finetune launcher first) OR pass
``--submit-train`` to have this watcher pre-build the shared packed-dataset cache
(once, on the login node) and submit the three finetune jobs at startup.

The watcher does NOT regenerate the paper table; extending ``tab:layer_ablation``
with the Llama rows + the appendix text is a deliberate follow-up performed after
this exits.

Usage:
    python scripts/jobs/watch_finetune_truncated_llama.py --plan            # print states, don't wait
    python scripts/jobs/watch_finetune_truncated_llama.py --submit-train    # prebuild + submit + watch + eval
    python scripts/jobs/watch_finetune_truncated_llama.py --max-retries 2 --poll 180
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_jobs_finetune_truncated_llama as FT  # noqa: E402
import run_jobs_layer_ablation_ft_llama as EV  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")
FT_LAUNCHER = os.path.join(PROJ, "scripts", "jobs", "run_jobs_finetune_truncated_llama.py")
STATE_DIR = os.path.join(PROJ, ".omc", "ablation_watch", "finetune_truncated_llama")
LOG_DIR = os.path.join(STATE_DIR, "logs")
WATCH_LOG = os.path.join(STATE_DIR, "watch.log")
DONE_JSON = os.path.join(STATE_DIR, "DONE.json")

ACTIVE = ("running", "pending", "completing")
FINAL_FAILED = {"failed", "stopped", "deleted", "terminated", "error", "cancelled", "canceled", "killed", "aborted"}

_client = None
_extra = None


# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(WATCH_LOG, "a") as fh:
        fh.write(line + "\n")


def _env() -> dict:
    e = os.environ.copy()
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        e.pop(k, None)
    e.setdefault("PYTHONPATH", os.path.join(PROJ, "src"))
    return e


def _mls(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([MLS_BIN, "job", *args], capture_output=capture, text=True, env=_env(), cwd=PROJ)


def mls_list(limit: int = 200) -> list[dict]:
    cp = _mls(["list", "-O", "json", "-l", str(limit)])
    if cp.returncode != 0:
        log(f"  WARN: mls job list failed: {cp.stderr.strip()[-200:]}")
        return []
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        for key in ("jobs", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return []
    return data if isinstance(data, list) else []


def status_for_desc(desc: str, jobs: list[dict]) -> tuple[str | None, str]:
    """Return (job_name, status) for the most relevant job matching ``desc`` (active preferred)."""
    matches = [j for j in jobs if str(j.get("job_desc", "")) == desc]
    if not matches:
        return None, "no-job"
    active = [j for j in matches if str(j.get("status", "")).lower() in ACTIVE]
    j = (active or matches)[0]
    return j.get("job_name"), str(j.get("status", "")).lower()


def save_logs(name: str, tail: int = 400) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    cp = _mls(["logs", name, "-t", str(tail)])
    path = os.path.join(LOG_DIR, f"{name}.log")
    with open(path, "w") as fh:
        fh.write(cp.stdout or "")
        if cp.stderr:
            fh.write("\n--- stderr ---\n" + cp.stderr)
    return path


def model_saved(out_dir: str) -> bool:
    """A finetune job is done once its dir holds a saved model (config + .safetensors)."""
    abs_dir = os.path.join(PROJ, out_dir)
    if not os.path.isfile(os.path.join(abs_dir, "config.json")):
        return False
    try:
        return any(f.endswith(".safetensors") for f in os.listdir(abs_dir))
    except OSError:
        return False


def eval_has_output(out_dir: str) -> bool:
    """A progressive-eval job has produced data once it wrote progressive_prefixes."""
    return os.path.isfile(os.path.join(PROJ, out_dir, "progressive_prefixes", "dataset_info.json"))


def _ensure_client():
    # Re-create the client on EVERY call so each MLS submission gets a fresh token.
    # The API access token expires after ~2h; caching it silently breaks long-running
    # watchers (submissions fail with error_code 20 "access_token expired"). The `mls`
    # CLI calls already re-auth per subprocess; only this in-process client needed the fix.
    global _client, _extra
    _client, _extra = EV.make_client()
    return _client, _extra


# --------------------------------------------------------------------------- #
def build_targets() -> list[dict]:
    """One entry per truncated depth (finetuned-only arm)."""
    targets: list[dict] = []
    for ck in FT.FINETUNE_CHECKPOINTS:
        ft = FT.finetuned_dir(ck)
        exp = next((e for e in EV.EXPERIMENTS if e["model_checkpoint"] == ft), None)
        if exp is None:
            log(f"  WARN: no eval experiment matches {ft}; skipping target")
            continue
        _, eval_suffix, eval_out = EV.render_job(exp)
        targets.append(
            {
                "label": os.path.basename(ck),
                "checkpoint": ck,
                "train_out": ft,
                "train_desc": FT.job_desc_for(os.path.basename(ck)),
                "eval_exp": exp,
                "eval_out": eval_out,
                "eval_desc": EV.job_desc_for(eval_suffix),
            }
        )
    return targets


def classify(target: dict, jobs: list[dict]) -> tuple[str, str | None]:
    """Return (state, relevant_job_name). States:
    eval_done | eval_running | eval_failed | train_ready | train_running | train_failed | train_pending
    """
    if eval_has_output(target["eval_out"]):
        return "eval_done", None

    ename, estatus = status_for_desc(target["eval_desc"], jobs)
    if ename and estatus in ACTIVE:
        return "eval_running", ename
    if ename and estatus in FINAL_FAILED:
        return "eval_failed", ename

    if model_saved(target["train_out"]):
        return "train_ready", None

    tname, tstatus = status_for_desc(target["train_desc"], jobs)
    if tname and tstatus in ACTIVE:
        return "train_running", tname
    if tname and tstatus in FINAL_FAILED:
        return "train_failed", tname
    return "train_pending", None


def submit_eval(target: dict, force: bool = False) -> None:
    """Submit the progressive-eval job for one ready target (idempotent)."""
    if force:
        abs_eval = os.path.join(PROJ, target["eval_out"])
        if os.path.isdir(abs_eval) and not eval_has_output(target["eval_out"]):
            log(f"  removing partial eval dir before retry: {target['eval_out']}")
            shutil.rmtree(abs_eval, ignore_errors=True)

    client, extra = _ensure_client()
    for attempt in range(3):
        result = EV.submit_experiment(target["eval_exp"], client, extra, force=force)
        name = (result or {}).get("job_name")
        if name:
            log(f"  submitted eval for {target['label']} -> {name}")
            return
        log(f"  eval submit for {target['label']} returned no job (attempt {attempt + 1}/3); retrying")
        time.sleep(5)
    log(f"  WARN: eval for {target['label']} not submitted after retries")


def submit_training() -> None:
    """Pre-build the shared packed-dataset cache (once) then submit the finetune jobs."""
    log("submit-train: prebuilding shared packed-dataset cache (login node, one-time; no-op if present) ...")
    cp = subprocess.run([PYTHON, FT_LAUNCHER, "--prebuild-only"], env=_env(), cwd=PROJ)
    if cp.returncode != 0:
        log(f"  WARN: dataset prebuild returned {cp.returncode}; finetune jobs may race the cache build")
    else:
        log("  dataset cache ready")

    log("submit-train: submitting missing causal-LM finetune jobs")
    cp = subprocess.run([PYTHON, FT_LAUNCHER], capture_output=True, text=True, env=_env(), cwd=PROJ)
    for line in (cp.stdout or "").splitlines():
        if any(k in line for k in ("job_name", "Would launch", "exists, skip", "already in queue")):
            log(f"  finetune: {line.strip()}")
    if cp.returncode != 0:
        log(f"  WARN: finetune submit returned {cp.returncode}: {(cp.stderr or '').strip()[-200:]}")


def write_done(targets: list[dict], states: dict[str, str], status: str) -> None:
    summary = {
        "status": status,
        "finished_at": datetime.now().isoformat(),
        "reference_out": EV.REFERENCE_OUT_DIR,
        "reference_has_output": eval_has_output(EV.REFERENCE_OUT_DIR),
        "targets": [
            {
                "label": t["label"],
                "state": states.get(t["label"], "?"),
                "eval_out": t["eval_out"],
                "has_output": eval_has_output(t["eval_out"]),
            }
            for t in targets
        ],
    }
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump(summary, open(DONE_JSON, "w"), indent=2)
    log(f"wrote completion summary -> {os.path.relpath(DONE_JSON, PROJ)}")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poll", type=int, default=180, help="Seconds between polls (default 180).")
    parser.add_argument("--max-retries", type=int, default=2, help="Eval resubmit attempts after a failure.")
    parser.add_argument(
        "--max-pending-polls",
        type=int,
        default=10,
        help="Polls a target may stay 'no training job' before the watcher gives up on it "
        "(absorbs scheduler lag right after submission; default 10).",
    )
    parser.add_argument("--plan", action="store_true", help="Print current target states once and exit.")
    parser.add_argument("--submit-train", action="store_true", help="Prebuild cache + submit finetune jobs at startup.")
    args = parser.parse_args()

    targets = build_targets()

    if args.plan:
        jobs = mls_list()
        log(f"PLAN -- Llama depth-ablation watcher ({len(targets)} targets, no waiting):")
        ref_ok = eval_has_output(EV.REFERENCE_OUT_DIR)
        log(f"  reference (full 32 layers): has_output={ref_ok}  {EV.REFERENCE_OUT_DIR}")
        for t in targets:
            state, name = classify(t, jobs)
            log(f"  [{state:13}] {t['label']}  (job={name})")
            log(f"                 train_out={t['train_out']}")
            log(f"                 eval_out={t['eval_out']}")
        return 0

    log(
        f"=== Llama depth-ablation watcher start (targets={len(targets)}, max_retries={args.max_retries}, poll={args.poll}s) ==="
    )
    if not eval_has_output(EV.REFERENCE_OUT_DIR):
        log(f"  NOTE: reference row has no data yet: {EV.REFERENCE_OUT_DIR}")
    if args.submit_train:
        submit_training()

    eval_attempts: dict[str, int] = {t["label"]: 0 for t in targets}
    pending_polls: dict[str, int] = {t["label"]: 0 for t in targets}
    gave_up: set[str] = set()
    seen_train_failed: set[str] = set()

    while True:
        jobs = mls_list()
        states: dict[str, str] = {}

        for t in targets:
            label = t["label"]
            if label in gave_up:
                states[label] = "gaveup"
                continue
            state, name = classify(t, jobs)
            states[label] = state
            if state != "train_pending":
                pending_polls[label] = 0

            if state == "train_ready":
                log(f"{label}: training ready -> submitting eval")
                submit_eval(t)
            elif state == "eval_failed":
                eval_attempts[label] += 1
                if eval_attempts[label] > args.max_retries:
                    log(f"{label}: eval gave up after {eval_attempts[label]} attempts (job={name}); see logs")
                    if name:
                        save_logs(name)
                    gave_up.add(label)
                    states[label] = "gaveup"
                else:
                    log(f"{label}: eval failed (job={name}); retry {eval_attempts[label]}/{args.max_retries}")
                    if name:
                        save_logs(name)
                    submit_eval(t, force=True)
            elif state == "train_failed" and label not in seen_train_failed:
                seen_train_failed.add(label)
                if name:
                    p = save_logs(name)
                    log(f"{label}: TRAINING failed (job={name}); logs -> {os.path.relpath(p, PROJ)}")
            elif state == "train_pending":
                pending_polls[label] += 1
                log(
                    f"{label}: no training job/checkpoint yet "
                    f"({pending_polls[label]}/{args.max_pending_polls} grace polls; "
                    f"submit training or use --submit-train)"
                )
                if pending_polls[label] > args.max_pending_polls:
                    log(f"{label}: still no training after grace window; giving up")
                    gave_up.add(label)
                    states[label] = "gaveup"

        n_done = sum(1 for s in states.values() if s == "eval_done")
        log("  states: " + ", ".join(f"{t['label'].split('-')[-1]}={states[t['label']]}" for t in targets))
        log(f"--- {n_done}/{len(targets)} eval_done ---")

        in_progress = any(
            states[t["label"]] in ("train_pending", "train_ready", "train_running", "eval_running")
            or (states[t["label"]] == "eval_failed" and eval_attempts[t["label"]] <= args.max_retries)
            for t in targets
        )
        all_done = n_done == len(targets)
        if all_done:
            log("All Llama depth-ablation evals have output. Watcher exiting (success).")
            write_done(targets, states, "complete")
            return 0
        if not in_progress:
            stuck = {t["label"]: states[t["label"]] for t in targets if states[t["label"]] != "eval_done"}
            log(f"No targets still in progress; exiting. Unfinished: {stuck}")
            write_done(targets, states, "incomplete")
            return 1

        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())

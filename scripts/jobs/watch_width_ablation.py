"""Background watcher/eval-driver for the model-width ablation (both arms).

The width ablation produces, per model width (SmolLM2 135M / 360M / 1.7B, all
first-4+last-4 = 8 layers), two recovered checkpoints to evaluate with progressive
cramming on PG19:

  * causal-LM arm: ``…-firstlast4-ftw`` (from run_jobs_finetune_width.py), evaluated
    with the baseline init ``embedding_init_method=random0.02`` via
    ``run_jobs_width_ablation_ft.py``;
  * Q-Former arm: ``ch_head_…`` (from run_jobs_compression_head_width.py), evaluated
    with ``embedding_init_method=compression_head_forward`` via
    ``run_jobs_compression_head_width.py --stage eval``.

This watcher polls all six (3 widths x 2 arms) targets. As each training output
becomes ready it submits that target's progressive-eval job; it retries a failed
eval (saving ``mls job logs`` + cleaning the partial output dir) up to
``--max-retries``. It EXITS once every target is terminal -- writing a machine
-readable ``DONE.json`` summary -- so running it as a background task notifies the
caller on completion (no in-watcher Telegram/sentinel polling). The watcher does
NOT regenerate the paper table; ``tab:width_ablation`` + the appendix section are a
deliberate follow-up step performed after this exits.

Training must already be submitted (run the two train launchers first) OR pass
``--submit-train`` to have this watcher submit any missing training jobs at startup.

Usage:
    python scripts/jobs/watch_width_ablation.py --plan                 # print states, don't wait
    python scripts/jobs/watch_width_ablation.py --submit-train         # submit training + watch + eval
    python scripts/jobs/watch_width_ablation.py --max-retries 2 --poll 180
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

import run_jobs_compression_head_width as CW  # noqa: E402
import run_jobs_finetune_width as FW  # noqa: E402
import run_jobs_width_ablation_ft as EW  # noqa: E402

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLS_BIN = os.environ.get("MLS_BIN", "/home/jovyan/.mlspace/envs/compression_horizon/bin/mls")
PYTHON = os.environ.get("WATCH_PYTHON", "/workspace-SR004.nfs2/d.tarasov/envs/compression_horizon/bin/python")
CW_LAUNCHER = os.path.join(PROJ, "scripts", "jobs", "run_jobs_compression_head_width.py")
FW_LAUNCHER = os.path.join(PROJ, "scripts", "jobs", "run_jobs_finetune_width.py")
STATE_DIR = os.path.join(PROJ, ".omc", "ablation_watch", "width_ablation")
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
    """A training job is done once its dir holds a saved model (config + .safetensors)."""
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
    global _client, _extra
    if _client is None:
        _client, _extra = FW.make_client()
    return _client, _extra


# --------------------------------------------------------------------------- #
def build_targets() -> list[dict]:
    """One entry per (width x arm) eval target."""
    targets: list[dict] = []

    # Causal-LM arm: -ftw checkpoints, baseline (random0.02) progressive eval.
    for ck in FW.FINETUNE_CHECKPOINTS:
        ft = FW.finetuned_dir(ck)
        exp = next((e for e in EW.EXPERIMENTS if e["model_checkpoint"] == ft), None)
        if exp is None:
            log(f"  WARN: no width-ft eval experiment matches {ft}; skipping target")
            continue
        _, _eval_suffix, eval_out = EW.render_job(exp)
        targets.append(
            {
                "arm": "causal_lm",
                "label": f"{os.path.basename(ck)} [causalLM]",
                "train_out": ft,
                "train_desc": FW.job_desc_for(os.path.basename(ck)),
                "eval_exp": exp,
                "eval_out": eval_out,
                "eval_desc": EW.job_desc_for(_eval_suffix),
            }
        )

    # Q-Former arm: ch_head_ checkpoints, compression_head_forward progressive eval.
    for exp in CW.EXPERIMENTS:
        _, ch_suffix, ch_out = CW.render_ch_job(exp)
        _, eval_suffix, eval_out = CW.render_eval_job(ch_out, ch_suffix)
        targets.append(
            {
                "arm": "ch_qformer",
                "label": f"{os.path.basename(ch_out)} [qformer]",
                "train_out": ch_out,
                "train_desc": CW.ch_job_desc(ch_suffix),
                "eval_exp": exp,
                "eval_out": eval_out,
                "eval_desc": CW.eval_job_desc(eval_suffix),
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
    """Submit the progressive-eval job for one ready target (idempotent per arm)."""
    if force:
        abs_eval = os.path.join(PROJ, target["eval_out"])
        if os.path.isdir(abs_eval) and not eval_has_output(target["eval_out"]):
            log(f"  removing partial eval dir before retry: {target['eval_out']}")
            shutil.rmtree(abs_eval, ignore_errors=True)

    if target["arm"] == "causal_lm":
        client, extra = _ensure_client()
        for attempt in range(3):
            result = EW.submit_experiment(target["eval_exp"], client, extra, force=force)
            name = (result or {}).get("job_name")
            if name:
                log(f"  submitted baseline eval for {target['label']} -> {name}")
                return
            log(f"  baseline eval submit for {target['label']} returned no job (attempt {attempt + 1}/3); retrying")
            time.sleep(5)
        log(f"  WARN: baseline eval for {target['label']} not submitted after retries")
    else:  # ch_qformer: reuse the idempotent launcher (submits all ready heads' evals)
        cp = subprocess.run([PYTHON, CW_LAUNCHER, "--stage", "eval"], capture_output=True, text=True, env=_env(), cwd=PROJ)
        for line in (cp.stdout or "").splitlines():
            if "job_name" in line or "exists, skip" in line or "Would launch" in line:
                log(f"  ch-eval-submit: {line.strip()}")
        if cp.returncode != 0:
            log(f"  WARN: ch eval submit returned {cp.returncode}: {(cp.stderr or '').strip()[-200:]}")


def submit_training() -> None:
    """Submit any missing training jobs for both arms (idempotent launchers)."""
    log("submit-train: submitting missing causal-LM + Q-Former training jobs")
    for desc, cmd in (
        ("causal-LM finetune", [PYTHON, FW_LAUNCHER]),
        ("Q-Former train", [PYTHON, CW_LAUNCHER, "--stage", "train"]),
    ):
        cp = subprocess.run(cmd, capture_output=True, text=True, env=_env(), cwd=PROJ)
        for line in (cp.stdout or "").splitlines():
            if any(k in line for k in ("job_name", "Would launch", "exists, skip", "already in queue")):
                log(f"  {desc}: {line.strip()}")
        if cp.returncode != 0:
            log(f"  WARN: {desc} submit returned {cp.returncode}: {(cp.stderr or '').strip()[-200:]}")


def write_done(targets: list[dict], states: dict[str, str], status: str) -> None:
    summary = {
        "status": status,
        "finished_at": datetime.now().isoformat(),
        "targets": [
            {
                "label": t["label"],
                "arm": t["arm"],
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
    parser.add_argument("--submit-train", action="store_true", help="Submit missing training jobs at startup.")
    args = parser.parse_args()

    targets = build_targets()

    if args.plan:
        jobs = mls_list()
        log(f"PLAN -- width-ablation watcher ({len(targets)} targets, no waiting):")
        for t in targets:
            state, name = classify(t, jobs)
            log(f"  [{state:13}] {t['label']}  (job={name})")
            log(f"                 train_out={t['train_out']}")
            log(f"                 eval_out={t['eval_out']}")
        return 0

    log(f"=== width-ablation watcher start (targets={len(targets)}, max_retries={args.max_retries}, poll={args.poll}s) ===")
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
                # No training job yet. This is normal briefly after submission (scheduler lag);
                # keep waiting up to a grace window, then give up so a background run still exits.
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

        # Progress summary.
        n_done = sum(1 for s in states.values() if s == "eval_done")
        log("  states: " + ", ".join(f"{t['label'].split()[-1]}={states[t['label']]}" for t in targets))
        log(f"--- {n_done}/{len(targets)} eval_done ---")

        # Terminal: no target is still making progress (train_pending counts as in-progress
        # until its grace window is exhausted, above).
        in_progress = any(
            states[t["label"]] in ("train_pending", "train_ready", "train_running", "eval_running")
            or (states[t["label"]] == "eval_failed" and eval_attempts[t["label"]] <= args.max_retries)
            for t in targets
        )
        all_done = n_done == len(targets)
        if all_done:
            log("All width-ablation evals have output. Watcher exiting (success).")
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

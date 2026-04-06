from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
UI_STATE_DIR = ROOT_DIR / "artifacts" / "ui"
JOB_STATE_PATH = UI_STATE_DIR / "jobs.json"
LOG_DIR = UI_STATE_DIR / "logs"


TRAIN_JOB = "training"
EVAL_LOCAL_JOB = "eval_local"
EVAL_GOLDEN_JOB = "eval_golden"
TRAINING_PRESETS = ("speed", "balanced", "quality")


LOCAL_REPORT_PATH = ROOT_DIR / "artifacts" / "local-evaluation" / "report.json"
GOLDEN_REPORT_PATH = ROOT_DIR / "artifacts" / "golden-evaluation" / "report.json"


@dataclass
class JobSpec:
    name: str
    command: list[str]
    log_path: Path
    report_path: Path | None = None


def ensure_state_dirs() -> None:
    UI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _default_state() -> dict[str, Any]:
    return {"jobs": {}}


def load_state() -> dict[str, Any]:
    ensure_state_dirs()
    if not JOB_STATE_PATH.exists():
        return _default_state()
    with JOB_STATE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state: dict[str, Any]) -> None:
    ensure_state_dirs()
    with JOB_STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _pid_is_running(pid: int) -> bool:
    # If this process launched the child, reap it when it has already exited.
    try:
        waited_pid, _ = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            return False
    except ChildProcessError:
        pass

    # On macOS, kill(pid, 0) can still succeed for zombie processes.
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False

        status = result.stdout.strip()
        if not status:
            return False
        if "Z" in status:
            return False
        return True
    except Exception:
        pass

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def refresh_jobs() -> dict[str, Any]:
    state = load_state()
    changed = False
    for job_name, job in state["jobs"].items():
        pid = job.get("pid")
        if job.get("status") != "running":
            continue

        report_path = job.get("report_path")
        if report_path:
            report_file = Path(report_path)
            if report_file.exists():
                finished_at = report_file.stat().st_mtime
                if finished_at >= job.get("started_at", 0):
                    job["status"] = "finished"
                    job["finished_at"] = finished_at
                    changed = True
                    continue

        if pid and not _pid_is_running(pid):
            job["status"] = "finished"
            job["finished_at"] = time.time()
            changed = True
    if changed:
        save_state(state)
    return state


def get_job(job_name: str) -> dict[str, Any] | None:
    return refresh_jobs()["jobs"].get(job_name)


def build_specs() -> dict[str, JobSpec]:
    return {
        TRAIN_JOB: JobSpec(
            name=TRAIN_JOB,
            command=["bash", "scripts/run_training_local.sh"],
            log_path=LOG_DIR / "training.log",
        ),
        EVAL_LOCAL_JOB: JobSpec(
            name=EVAL_LOCAL_JOB,
            command=["bash", "scripts/run_evaluation_local.sh"],
            log_path=LOG_DIR / "eval_local.log",
            report_path=LOCAL_REPORT_PATH,
        ),
        EVAL_GOLDEN_JOB: JobSpec(
            name=EVAL_GOLDEN_JOB,
            command=["bash", "scripts/run_evaluation_local.sh"],
            log_path=LOG_DIR / "eval_golden.log",
            report_path=GOLDEN_REPORT_PATH,
        ),
    }


def _env_for_job(job_name: str, env_overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(ROOT_DIR / ".cache" / "uv"))
    env.setdefault("HF_HOME", str(ROOT_DIR / ".cache" / "huggingface"))
    env.setdefault("TRANSFORMERS_CACHE", env["HF_HOME"])

    if job_name == EVAL_GOLDEN_JOB:
        env["EVAL_DATA_PATH"] = str(ROOT_DIR / "data" / "golden" / "golden.jsonl")
        env["EVAL_REPORT_PATH"] = str(GOLDEN_REPORT_PATH)
    elif job_name == EVAL_LOCAL_JOB:
        env.setdefault("EVAL_REPORT_PATH", str(LOCAL_REPORT_PATH))

    if env_overrides:
        env.update(env_overrides)

    return env


def start_job(job_name: str, env_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    specs = build_specs()
    spec = specs[job_name]
    state = refresh_jobs()
    existing = state["jobs"].get(job_name)
    if existing and existing.get("status") == "running":
        return existing

    ensure_state_dirs()
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n\n=== {job_name} started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    log_stream = spec.log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        spec.command,
        cwd=ROOT_DIR,
        env=_env_for_job(job_name, env_overrides=env_overrides),
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    record = {
        "name": job_name,
        "pid": process.pid,
        "status": "running",
        "command": spec.command,
        "log_path": str(spec.log_path),
        "report_path": str(spec.report_path) if spec.report_path else None,
        "env_overrides": env_overrides or {},
        "started_at": time.time(),
    }
    state["jobs"][job_name] = record
    save_state(state)
    return record


def stop_job(job_name: str) -> dict[str, Any] | None:
    state = refresh_jobs()
    job = state["jobs"].get(job_name)
    if not job or job.get("status") != "running":
        return job

    pid = job.get("pid")
    if pid:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass

    job["status"] = "stopped"
    job["finished_at"] = time.time()
    save_state(state)
    return job


def read_log(job_name: str, tail_lines: int = 80) -> str:
    specs = build_specs()
    log_path = specs[job_name].log_path
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-tail_lines:])


def read_report(report_path: Path) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_timestamp(timestamp: float | None) -> str:
    if not timestamp:
        return "n/a"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

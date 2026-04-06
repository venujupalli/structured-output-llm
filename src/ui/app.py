from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui.job_manager import (
    EVAL_GOLDEN_JOB,
    EVAL_LOCAL_JOB,
    GOLDEN_REPORT_PATH,
    LOCAL_REPORT_PATH,
    TRAIN_JOB,
    TRAINING_PRESETS,
    format_timestamp,
    get_job,
    read_log,
    read_report,
    refresh_jobs,
    start_job,
    stop_job,
)


st.set_page_config(page_title="Structured Output LLM", layout="wide")

PRESET_HELP = {
    "speed": "Fastest turnaround. 250 records, 1 epoch, shorter context.",
    "balanced": "Default preset. 500 records, 2 epochs, moderate context length.",
    "quality": "Longer run. 1000 records, 3 epochs, longer context.",
}


def render_job_controls(
    job_name: str,
    title: str,
    help_text: str,
    env_overrides: dict[str, str] | None = None,
) -> None:
    job = get_job(job_name) or {}
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    with col1:
        st.subheader(title)
        st.caption(help_text)
    with col2:
        status = job.get("status", "idle")
        st.metric("Status", status)
    with col3:
        if st.button(f"Start {title}", key=f"start-{job_name}", use_container_width=True):
            start_job(job_name, env_overrides=env_overrides)
            st.rerun()
    with col4:
        if st.button(f"Stop {title}", key=f"stop-{job_name}", use_container_width=True):
            stop_job(job_name)
            st.rerun()

    st.caption(
        f"Started: {format_timestamp(job.get('started_at'))} | "
        f"Finished: {format_timestamp(job.get('finished_at'))}"
    )

    log_text = read_log(job_name)
    st.text_area("Recent log output", value=log_text or "No log output yet.", height=220, key=f"log-{job_name}")


def render_metrics(title: str, report_path: Path) -> None:
    st.subheader(title)
    report = read_report(report_path)
    if not report:
        st.info(f"No report found at `{report_path}` yet.")
        return

    metrics = report.get("metrics", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Samples", metrics.get("total_samples", 0))
    m2.metric("JSON Parse", f"{metrics.get('json_parse_success_rate', 0.0):.1%}")
    m3.metric("Schema Valid", f"{metrics.get('schema_valid_rate', 0.0):.1%}")
    m4.metric("Required Fields", f"{metrics.get('required_field_completeness_rate', 0.0):.1%}")

    samples = report.get("samples", [])
    if not samples:
        return

    rows = []
    for idx, sample in enumerate(samples[:25], start=1):
        rows.append(
            {
                "sample": idx,
                "json_parse": sample.get("json_parse"),
                "schema_valid": sample.get("schema_valid"),
                "required_complete": sample.get("required_complete"),
                "error": sample.get("error"),
                "prediction_preview": str(sample.get("prediction", ""))[:160],
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Raw report JSON"):
        st.code(json.dumps(report, indent=2)[:20000], language="json")


refresh_jobs()

st.title("Structured Output LLM Dashboard")
st.caption("Trigger local LoRA training, evaluate the adapter, and compare against the golden set from one place.")

autorefresh = st.toggle("Auto refresh every 5 seconds", value=True)
if autorefresh:
    st.markdown(
        """
        <script>
        setTimeout(function() {
          window.parent.location.reload();
        }, 5000);
        </script>
        """,
        unsafe_allow_html=True,
    )

with st.container(border=True):
    training_job = get_job(TRAIN_JOB) or {}
    training_preset = st.selectbox(
        "Training preset",
        options=list(TRAINING_PRESETS),
        index=1,
        format_func=lambda preset: f"{preset} - {PRESET_HELP[preset]}",
    )
    active_preset = training_job.get("env_overrides", {}).get("TRAINING_PRESET", training_preset)
    st.caption(f"Active training preset for the next run: `{training_preset}`. Current job preset: `{active_preset}`.")
    render_job_controls(
        TRAIN_JOB,
        "Training",
        "Runs `scripts/run_training_local.sh` with the local config and the selected preset.",
        {"TRAINING_PRESET": training_preset},
    )

with st.container(border=True):
    render_job_controls(
        EVAL_LOCAL_JOB,
        "Local Evaluation",
        "Evaluates the current adapter against the default local validation set.",
    )

with st.container(border=True):
    render_job_controls(
        EVAL_GOLDEN_JOB,
        "Golden Evaluation",
        "Evaluates the current adapter against `data/golden/golden.jsonl`.",
    )

left, right = st.columns(2)
with left:
    render_metrics("Latest Local Evaluation Report", LOCAL_REPORT_PATH)
with right:
    render_metrics("Latest Golden Evaluation Report", GOLDEN_REPORT_PATH)

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install it first: https://docs.astral.sh/uv/"
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$ROOT_DIR/mlflow.db}"
export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-$ROOT_DIR/mlruns}"

mkdir -p "$HF_HOME" "$ROOT_DIR/artifacts/local-evaluation" "$MLFLOW_ARTIFACT_ROOT"

DEFAULT_MODEL_CONFIG="configs/model_config.local.yaml"
DEFAULT_TRAINING_CONFIG="configs/training_config.local.yaml"
DEFAULT_SCHEMA_CONFIG="configs/schema_config.yaml"
DEFAULT_ADAPTER_PATH="artifacts/local-model/adapter"
DEFAULT_REPORT_PATH="artifacts/local-evaluation/report.json"

CMD=(uv run python -m src.evaluation.evaluate
  --model-config "${MODEL_CONFIG_PATH:-$DEFAULT_MODEL_CONFIG}"
  --training-config "${TRAINING_CONFIG_PATH:-$DEFAULT_TRAINING_CONFIG}"
  --schema-config "${SCHEMA_CONFIG_PATH:-$DEFAULT_SCHEMA_CONFIG}"
  --output-report "${EVAL_REPORT_PATH:-$DEFAULT_REPORT_PATH}")

if [[ -n "${ADAPTER_PATH:-}" ]]; then
  CMD+=(--adapter-path "$ADAPTER_PATH")
else
  CMD+=(--adapter-path "$DEFAULT_ADAPTER_PATH")
fi

if [[ -n "${EVAL_DATA_PATH:-}" ]]; then
  CMD+=(--eval-data-path "$EVAL_DATA_PATH")
fi

echo "[local-eval] Using model config: ${MODEL_CONFIG_PATH:-$DEFAULT_MODEL_CONFIG}"
echo "[local-eval] Using training config: ${TRAINING_CONFIG_PATH:-$DEFAULT_TRAINING_CONFIG}"
echo "[local-eval] Using schema config: ${SCHEMA_CONFIG_PATH:-$DEFAULT_SCHEMA_CONFIG}"
echo "[local-eval] Adapter path: ${ADAPTER_PATH:-$DEFAULT_ADAPTER_PATH}"
echo "[local-eval] Report path: ${EVAL_REPORT_PATH:-$DEFAULT_REPORT_PATH}"
echo "[local-eval] HF cache: ${TRANSFORMERS_CACHE}"
echo "[local-eval] MLflow tracking URI: ${MLFLOW_TRACKING_URI}"
echo "[local-eval] MLflow artifact root: ${MLFLOW_ARTIFACT_ROOT}"
echo "[local-eval] Runtime: uv"

"${CMD[@]}"

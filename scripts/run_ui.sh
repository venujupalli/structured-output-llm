#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install it first: https://docs.astral.sh/uv/"
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$ROOT_DIR/mlflow.db}"
export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-$ROOT_DIR/mlruns}"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$ROOT_DIR/artifacts/ui/logs" "$MLFLOW_ARTIFACT_ROOT"

exec uv run streamlit run src/ui/app.py --server.headless true --server.port "${STREAMLIT_PORT:-8501}"

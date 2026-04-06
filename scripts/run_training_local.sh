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

mkdir -p "$HF_HOME" "$ROOT_DIR/artifacts/local-model"

TRAINING_PRESET="${TRAINING_PRESET:-balanced}"
PRESET_CONFIG_PATH=""

if [[ -z "${OVERRIDE_CONFIG_PATH:-}" ]]; then
  case "$TRAINING_PRESET" in
    speed)
      PRESET_CONFIG_PATH="configs/presets/training_speed.yaml"
      ;;
    balanced)
      PRESET_CONFIG_PATH="configs/presets/training_balanced.yaml"
      ;;
    quality)
      PRESET_CONFIG_PATH="configs/presets/training_quality.yaml"
      ;;
    none|"")
      PRESET_CONFIG_PATH=""
      ;;
    *)
      echo "Error: unknown TRAINING_PRESET '$TRAINING_PRESET'. Use speed, balanced, quality, or none."
      exit 1
      ;;
  esac
fi

CMD=(uv run python -m src.training.train
  --model-config "${MODEL_CONFIG_PATH:-configs/model_config.local.yaml}"
  --training-config "${TRAINING_CONFIG_PATH:-configs/training_config.local.yaml}")

if [[ -n "${OVERRIDE_CONFIG_PATH:-}" ]]; then
  CMD+=(--override-config "$OVERRIDE_CONFIG_PATH")
elif [[ -n "$PRESET_CONFIG_PATH" ]]; then
  CMD+=(--override-config "$PRESET_CONFIG_PATH")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  CMD+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "[local-train] Using model config: ${MODEL_CONFIG_PATH:-configs/model_config.local.yaml}"
echo "[local-train] Using training config: ${TRAINING_CONFIG_PATH:-configs/training_config.local.yaml}"
echo "[local-train] Training preset: ${TRAINING_PRESET}"
if [[ -n "${OVERRIDE_CONFIG_PATH:-}" ]]; then
  echo "[local-train] Override config: ${OVERRIDE_CONFIG_PATH}"
elif [[ -n "$PRESET_CONFIG_PATH" ]]; then
  echo "[local-train] Preset override config: ${PRESET_CONFIG_PATH}"
fi
echo "[local-train] HF cache: ${TRANSFORMERS_CACHE}"
echo "[local-train] Runtime: uv"

"${CMD[@]}"

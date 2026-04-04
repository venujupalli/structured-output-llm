#!/usr/bin/env bash
set -euo pipefail

CMD=(python3 -m src.training.train
  --model-config "${MODEL_CONFIG_PATH:-configs/model_config.yaml}"
  --training-config "${TRAINING_CONFIG_PATH:-configs/training_config.yaml}")

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  CMD+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${CMD[@]}"

#!/usr/bin/env bash
set -euo pipefail

python3 -m src.evaluation.evaluate \
  --model-config "${MODEL_CONFIG_PATH:-configs/model_config.yaml}" \
  --training-config "${TRAINING_CONFIG_PATH:-configs/training_config.yaml}" \
  --schema-config "${SCHEMA_CONFIG_PATH:-configs/schema_config.yaml}" \
  --output-report "${EVAL_REPORT_PATH:-artifacts/evaluation/report.json}"

#!/usr/bin/env bash
set -euo pipefail

python3 -m src.evaluation.evaluate \
  --model-config "${MODEL_CONFIG_PATH:-configs/model_config.yaml}" \
  --training-config "${TRAINING_CONFIG_PATH:-configs/training_config.yaml}" \
  --schema-config "${SCHEMA_CONFIG_PATH:-configs/schema_config.yaml}" \
  --adapter-path "${ADAPTER_PATH:-}" \
  --eval-data-path "${EVAL_DATA_PATH:-}" \
  --output-report "${EVAL_REPORT_PATH:-artifacts/evaluation/report.json}"

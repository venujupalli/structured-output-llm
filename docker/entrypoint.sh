#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-}"

case "$COMMAND" in
  train)
    shift
    exec python3 -m src.training.train "$@"
    ;;
  evaluate)
    shift
    exec python3 -m src.evaluation.evaluate "$@"
    ;;
  *)
    echo "Usage: entrypoint.sh [train|evaluate] <args>"
    exit 1
    ;;
esac

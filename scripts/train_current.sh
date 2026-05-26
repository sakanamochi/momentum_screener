#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=".venv/Scripts/python.exe"

"$python_bin" -m momentum_screener.cli run \
  --cache-only \
  --cache data/ohlcv_current.csv \
  --model-path models/momentum_nn_production.pt \
  --metrics-path outputs/metrics_production.json \
  --output outputs/candidates_current.csv \
  --train-end 2099-12-31 \
  --valid-end 2099-12-31 \
  --epochs 50 \
  --patience 8 \
  --min-events 50

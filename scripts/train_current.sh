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
  --gate-min-turnover-5d 100000000 \
  --gate-min-ret-5d -0.01 \
  --gate-min-turnover-ratio-1d-20d 1.05 \
  --gate-min-turnover-ratio-5d-20d 1.05 \
  --gate-min-close-ma25-ratio -0.01 \
  --label-mode barrier \
  --profit-barrier 0.15 \
  --stop-barrier -0.10 \
  --train-end 2099-12-31 \
  --valid-end 2099-12-31 \
  --epochs 50 \
  --patience 8 \
  --min-events 50

cp models/momentum_nn_production.pt models/momentum_nn_current.pt
cp outputs/metrics_production.json outputs/metrics_current.json

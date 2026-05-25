#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=".venv/Scripts/python.exe"

"$python_bin" -m momentum_screener.cli run \
  --ticker-csv config/listed_stocks.csv \
  --ticker-csv-code-column code \
  --no-sample-tickers \
  --refresh \
  --cache data/ohlcv_current.csv \
  --model-path models/momentum_nn_current.pt \
  --metrics-path outputs/metrics_current.json \
  --output outputs/candidates_current.csv \
  --gate-min-turnover-5d 50000000 \
  --gate-min-ret-5d -0.01 \
  --gate-min-turnover-ratio-1d-20d 1.05 \
  --gate-min-turnover-ratio-5d-20d 1.05 \
  --gate-min-close-ma25-ratio -0.01 \
  --epochs 50 \
  --patience 8 \
  --min-events 50


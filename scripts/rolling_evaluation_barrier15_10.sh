#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/Scripts/python.exe -m momentum_screener.cli rolling-eval \
  --cache-only \
  --cache data/ohlcv_current.csv \
  --metrics-path outputs/rolling_evaluation_barrier15_10.json \
  --output outputs/rolling_evaluation_barrier15_10.csv \
  --gate-min-turnover-5d 50000000 \
  --gate-min-ret-5d -0.01 \
  --gate-min-turnover-ratio-1d-20d 1.05 \
  --gate-min-turnover-ratio-5d-20d 1.05 \
  --gate-min-close-ma25-ratio -0.01 \
  --label-mode barrier \
  --profit-barrier 0.15 \
  --stop-barrier -0.10 \
  --epochs 50 \
  --patience 8 \
  --min-events 50

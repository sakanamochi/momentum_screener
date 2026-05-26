#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python_bin=".venv/Scripts/python.exe"
study="${1:-rolling_evaluation}"
out_dir="experiments/outputs/$study"
mkdir -p "$out_dir"

"$python_bin" -m momentum_screener.cli rolling-eval \
  --cache-only \
  --cache data/ohlcv_current.csv \
  --metrics-path "$out_dir/metrics.json" \
  --output "$out_dir/folds.csv" \
  --epochs 50 \
  --patience 8 \
  --min-events 50

echo "Wrote $out_dir"

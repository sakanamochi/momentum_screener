#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python_bin=".venv/Scripts/python.exe"
study="${1:-train_evaluation}"
out_dir="experiments/outputs/$study"
mkdir -p "$out_dir/models"

"$python_bin" -m momentum_screener.cli run \
  --cache-only \
  --cache data/ohlcv_current.csv \
  --model-path "$out_dir/models/momentum_nn.pt" \
  --metrics-path "$out_dir/metrics.json" \
  --output "$out_dir/candidates.csv" \
  --train-end 2023-12-31 \
  --valid-end 2024-12-31 \
  --epochs 50 \
  --patience 8 \
  --min-events 50

echo "Wrote $out_dir"

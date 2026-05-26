#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=".venv/Scripts/python.exe"

"$python_bin" -m momentum_screener.cli inspect-gate \
  --cache-only \
  --cache data/ohlcv_current.csv \
  --metrics-path outputs/gate_current.json

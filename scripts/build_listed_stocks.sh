#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=".venv/Scripts/python.exe"
issues_file="$(ls -t config/Issues_*.csv | head -n 1)"

"$python_bin" -m momentum_screener.cli build-listed-stocks "$issues_file" --output config/listed_stocks.csv


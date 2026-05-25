#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=".venv/Scripts/python.exe"

"$python_bin" scripts/screen_or_show.py --update-data


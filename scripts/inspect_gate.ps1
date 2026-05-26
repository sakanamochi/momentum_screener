$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = ".\.venv\Scripts\python.exe"

& $python -m momentum_screener.cli inspect-gate `
  --cache-only `
  --cache data\ohlcv_current.csv `
  --metrics-path outputs\gate_current.json

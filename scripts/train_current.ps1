$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = ".\.venv\Scripts\python.exe"
$cache = "data\ohlcv_current.csv"
$model = "models\momentum_nn_current.pt"
$metrics = "outputs\metrics_current.json"
$output = "outputs\candidates_current.csv"

& $python -m momentum_screener.cli run `
  --cache-only `
  --cache $cache `
  --model-path models\momentum_nn_production.pt `
  --metrics-path outputs\metrics_production.json `
  --output $output `
  --gate-min-turnover-5d 50000000 `
  --gate-min-ret-5d -0.01 `
  --gate-min-turnover-ratio-1d-20d 1.05 `
  --gate-min-turnover-ratio-5d-20d 1.05 `
  --gate-min-close-ma25-ratio -0.01 `
  --label-mode barrier `
  --profit-barrier 0.15 `
  --stop-barrier -0.10 `
  --train-end 2099-12-31 `
  --valid-end 2099-12-31 `
  --epochs 50 `
  --patience 8 `
  --min-events 50

Copy-Item -LiteralPath models\momentum_nn_production.pt -Destination models\momentum_nn_current.pt -Force
Copy-Item -LiteralPath outputs\metrics_production.json -Destination outputs\metrics_current.json -Force

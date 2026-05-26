$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

$python = ".\.venv\Scripts\python.exe"
$study = if ($args.Count -gt 0) { $args[0] } else { "rolling_evaluation" }
$outDir = "experiments\outputs\$study"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

& $python -m momentum_screener.cli rolling-eval `
  --cache-only `
  --cache data\ohlcv_current.csv `
  --metrics-path "$outDir\metrics.json" `
  --output "$outDir\folds.csv" `
  --epochs 50 `
  --patience 8 `
  --min-events 50

Write-Host "Wrote $outDir"

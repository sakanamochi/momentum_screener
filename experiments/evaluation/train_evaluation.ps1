$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

$python = ".\.venv\Scripts\python.exe"
$study = if ($args.Count -gt 0) { $args[0] } else { "train_evaluation" }
$outDir = "experiments\outputs\$study"
New-Item -ItemType Directory -Force -Path "$outDir\models" | Out-Null

& $python -m momentum_screener.cli run `
  --cache-only `
  --cache data\ohlcv_current.csv `
  --model-path "$outDir\models\momentum_nn.pt" `
  --metrics-path "$outDir\metrics.json" `
  --output "$outDir\candidates.csv" `
  --train-end 2023-12-31 `
  --valid-end 2024-12-31 `
  --epochs 50 `
  --patience 8 `
  --min-events 50

Write-Host "Wrote $outDir"

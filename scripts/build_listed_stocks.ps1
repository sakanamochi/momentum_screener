$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = ".\.venv\Scripts\python.exe"
$issues = Get-ChildItem config\Issues_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $issues) {
  throw "config\Issues_*.csv was not found."
}

& $python -m momentum_screener.cli build-listed-stocks $issues.FullName --output config\listed_stocks.csv


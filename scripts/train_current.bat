@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe -m momentum_screener.cli run ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --model-path models\momentum_nn_production.pt ^
  --metrics-path outputs\metrics_production.json ^
  --output outputs\candidates_current.csv ^
  --train-end 2099-12-31 ^
  --valid-end 2099-12-31 ^
  --epochs 50 ^
  --patience 8 ^
  --min-events 50

if errorlevel 1 (
  echo.
  echo Failed to train production model.
  pause
  exit /b 1
)

echo.
echo Wrote outputs\candidates_current.csv and outputs\metrics_production.json
start "" "outputs\candidates_current.csv"
pause

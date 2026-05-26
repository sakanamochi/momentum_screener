@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe -m momentum_screener.cli run ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --model-path models\momentum_nn_evaluation.pt ^
  --metrics-path outputs\metrics_evaluation.json ^
  --output outputs\candidates_evaluation.csv ^
  --train-end 2023-12-31 ^
  --valid-end 2024-12-31 ^
  --epochs 50 ^
  --patience 8 ^
  --min-events 50

if errorlevel 1 (
  echo.
  echo Failed to train evaluation model.
  pause
  exit /b 1
)

echo.
echo Wrote models\momentum_nn_evaluation.pt and outputs\metrics_evaluation.json
start "" "outputs\metrics_evaluation.json"
pause

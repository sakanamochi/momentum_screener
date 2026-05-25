@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe -m momentum_screener.cli run ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --model-path models\momentum_nn_evaluation.pt ^
  --metrics-path outputs\metrics_evaluation.json ^
  --output outputs\candidates_evaluation.csv ^
  --gate-min-turnover-5d 50000000 ^
  --gate-min-ret-5d -0.01 ^
  --gate-min-turnover-ratio-1d-20d 1.05 ^
  --gate-min-turnover-ratio-5d-20d 1.05 ^
  --gate-min-close-ma25-ratio -0.01 ^
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

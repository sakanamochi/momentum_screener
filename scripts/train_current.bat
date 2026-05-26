@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe -m momentum_screener.cli run ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --model-path models\momentum_nn_production.pt ^
  --metrics-path outputs\metrics_production.json ^
  --output outputs\candidates_current.csv ^
  --gate-min-turnover-5d 100000000 ^
  --gate-min-ret-5d -0.01 ^
  --gate-min-turnover-ratio-1d-20d 1.05 ^
  --gate-min-turnover-ratio-5d-20d 1.05 ^
  --gate-min-close-ma25-ratio -0.01 ^
  --label-mode barrier ^
  --profit-barrier 0.15 ^
  --stop-barrier -0.10 ^
  --train-end 2099-12-31 ^
  --valid-end 2099-12-31 ^
  --epochs 50 ^
  --patience 8 ^
  --min-events 50

if errorlevel 1 (
  echo.
  echo Failed to train current model.
  pause
  exit /b 1
)

echo.
copy /Y models\momentum_nn_production.pt models\momentum_nn_current.pt >nul
copy /Y outputs\metrics_production.json outputs\metrics_current.json >nul
echo Wrote outputs\candidates_current.csv and outputs\metrics_production.json
start "" "outputs\candidates_current.csv"
pause

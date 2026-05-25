@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe -m momentum_screener.cli screen ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --model-path models\momentum_nn_current.pt ^
  --output outputs\candidates_current.csv ^
  --gate-min-turnover-5d 50000000 ^
  --gate-min-ret-5d -0.01 ^
  --gate-min-turnover-ratio-1d-20d 1.05 ^
  --gate-min-turnover-ratio-5d-20d 1.05 ^
  --gate-min-close-ma25-ratio -0.01

if errorlevel 1 (
  echo.
  echo Failed to screen candidates.
  pause
  exit /b 1
)

echo.
echo Wrote outputs\candidates_current.csv
start "" "outputs\candidates_current.csv"
pause


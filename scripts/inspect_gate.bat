@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe -m momentum_screener.cli inspect-gate ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --metrics-path outputs\gate_current.json ^
  --gate-min-turnover-5d 100000000 ^
  --gate-min-ret-5d -0.01 ^
  --gate-min-turnover-ratio-1d-20d 1.05 ^
  --gate-min-turnover-ratio-5d-20d 1.05 ^
  --gate-min-close-ma25-ratio -0.01

if errorlevel 1 (
  echo.
  echo Failed to inspect gate.
  pause
  exit /b 1
)

echo.
echo Wrote outputs\gate_current.json
start "" "outputs\gate_current.json"
pause

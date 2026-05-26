@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe -m momentum_screener.cli inspect-gate ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --metrics-path outputs\gate_current.json

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

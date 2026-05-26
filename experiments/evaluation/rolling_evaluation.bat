@echo off
setlocal
cd /d "%~dp0\..\.."

if "%~1"=="" (
  set STUDY=rolling_evaluation
) else (
  set STUDY=%~1
)
set OUT_DIR=experiments\outputs\%STUDY%
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

call .venv\Scripts\python.exe -m momentum_screener.cli rolling-eval ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --metrics-path "%OUT_DIR%\metrics.json" ^
  --output "%OUT_DIR%\folds.csv" ^
  --epochs 50 ^
  --patience 8 ^
  --min-events 50

if errorlevel 1 (
  echo.
  echo Failed to run rolling evaluation.
  pause
  exit /b 1
)

echo.
echo Wrote %OUT_DIR%
start "" "%OUT_DIR%\folds.csv"
pause

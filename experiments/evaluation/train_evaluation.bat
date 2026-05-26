@echo off
setlocal
cd /d "%~dp0\..\.."

if "%~1"=="" (
  set STUDY=train_evaluation
) else (
  set STUDY=%~1
)
set OUT_DIR=experiments\outputs\%STUDY%
if not exist "%OUT_DIR%\models" mkdir "%OUT_DIR%\models"

call .venv\Scripts\python.exe -m momentum_screener.cli run ^
  --cache-only ^
  --cache data\ohlcv_current.csv ^
  --model-path "%OUT_DIR%\models\momentum_nn.pt" ^
  --metrics-path "%OUT_DIR%\metrics.json" ^
  --output "%OUT_DIR%\candidates.csv" ^
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
echo Wrote %OUT_DIR%
start "" "%OUT_DIR%\metrics.json"
pause

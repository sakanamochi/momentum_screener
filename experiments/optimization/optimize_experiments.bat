@echo off
setlocal
cd /d "%~dp0\..\.."

call .venv\Scripts\python.exe experiments\optimization\optimize_experiments.py ^
  --cache data\ohlcv_current.csv ^
  --output-dir outputs\optimization ^
  --search-size small ^
  --max-trials 60 ^
  --epochs 35 ^
  --patience 6 ^
  --min-events 50

if errorlevel 1 (
  echo.
  echo Failed to run optimization experiments.
  pause
  exit /b 1
)

echo.
echo Optimization experiments finished. Check outputs\optimization.
pause

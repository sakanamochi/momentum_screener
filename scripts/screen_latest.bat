@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe scripts\screen_or_show.py --update-data

if errorlevel 1 (
  echo.
  echo Failed to screen or show candidates.
  pause
  exit /b 1
)

pause

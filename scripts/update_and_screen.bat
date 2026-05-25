@echo off
setlocal
cd /d "%~dp0\.."

call .venv\Scripts\python.exe scripts\screen_or_show.py --update-data

if errorlevel 1 (
  echo.
  echo Failed to update data or screen candidates.
  pause
  exit /b 1
)

pause


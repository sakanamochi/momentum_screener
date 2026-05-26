@echo off
setlocal
cd /d "%~dp0\.."

if "%~1"=="" (
  set /p CODE=Enter stock code, e.g. 285A or 6976: 
) else (
  set CODE=%~1
)

if "%CODE%"=="" (
  echo No code supplied.
  pause
  exit /b 1
)

call .venv\Scripts\python.exe scripts\inspect_symbol_history.py "%CODE%"

if errorlevel 1 (
  echo.
  echo Failed to inspect symbol history.
  pause
  exit /b 1
)

pause

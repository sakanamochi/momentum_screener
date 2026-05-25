@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set "ISSUES="
for /f "delims=" %%F in ('dir /b /o-d config\Issues_*.csv 2^>nul') do (
  set "ISSUES=config\%%F"
  goto :found
)

echo config\Issues_*.csv was not found.
pause
exit /b 1

:found
call .venv\Scripts\python.exe -m momentum_screener.cli build-listed-stocks "%ISSUES%" --output config\listed_stocks.csv

if errorlevel 1 (
  echo.
  echo Failed to build listed stocks.
  pause
  exit /b 1
)

echo.
echo Wrote config\listed_stocks.csv
start "" "config\listed_stocks.csv"
pause


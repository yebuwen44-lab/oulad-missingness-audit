@echo off
setlocal
cd /d "%~dp0"

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\01_verify_frozen_assets.ps1"
if errorlevel 1 (
  echo.
  echo Verification failed. Please keep this window open and send the complete error message for review.
) else (
  echo.
  echo Verification completed successfully. Python is not required for this verification step.
)
pause

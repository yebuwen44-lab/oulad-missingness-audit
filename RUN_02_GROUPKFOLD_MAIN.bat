@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run RUN_01_CREATE_ENVIRONMENT.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe src\02_run_groupkfold_main.py
if errorlevel 1 goto :error
echo.
echo Five-fold analysis completed. Outputs are in results\rerun\groupkfold.
pause
exit /b 0
:error
echo.
echo Analysis failed. Keep this window open and send the error message for review.
pause
exit /b 1

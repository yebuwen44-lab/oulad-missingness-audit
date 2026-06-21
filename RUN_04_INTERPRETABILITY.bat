@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run RUN_01_CREATE_ENVIRONMENT.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe src\04_run_shap_explainability.py
if errorlevel 1 goto :error
.venv\Scripts\python.exe src\05_run_descriptive_missingness.py
if errorlevel 1 goto :error
echo.
echo Interpretability and descriptive analyses completed.
pause
exit /b 0
:error
echo.
echo Analysis failed. Keep this window open and send the error message for review.
pause
exit /b 1

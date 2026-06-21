@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYCMD="

if exist ".venv\Scripts\python.exe" (
  echo Existing virtual environment found.
  goto :install_packages
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 -c "import sys" >nul 2>nul
  if not errorlevel 1 set "PYCMD=py -3.13"
)

if not defined PYCMD (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYCMD=python"
  )
)

if not defined PYCMD (
  for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\ProgramData\miniconda3\python.exe"
  ) do (
    if exist "%%~P" if not defined PYCMD set "PYCMD=%%~P"
  )
)

if not defined PYCMD goto :python_missing

echo Using Python: %PYCMD%
%PYCMD% -m venv .venv
if errorlevel 1 goto :error

:install_packages
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Environment created successfully.
pause
exit /b 0

:python_missing
echo.
echo Python was not found on this computer.
echo The fast package verification does NOT require Python; use RUN_00_VERIFY_PACKAGE.bat.
echo Full experiment reruns require Python 3.13.
echo.
where winget >nul 2>nul
if errorlevel 1 goto :manual_install
choice /C YN /M "Install Python 3.13 automatically with Windows Package Manager"
if errorlevel 2 goto :manual_install
winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :manual_install
set "PYCMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PYCMD%" (
  echo Python was installed, but this window cannot locate it yet.
  echo Close this window, then double-click RUN_01_CREATE_ENVIRONMENT.bat again.
  pause
  exit /b 0
)
%PYCMD% -m venv .venv
if errorlevel 1 goto :error
goto :install_packages

:manual_install
echo.
echo Please install Python 3.13 from https://www.python.org/downloads/windows/
echo During installation, select "Add python.exe to PATH".
echo Then close this window and run RUN_01_CREATE_ENVIRONMENT.bat again.
pause
exit /b 1

:error
echo.
echo Environment creation failed. Keep this window open and send the complete error message for review.
pause
exit /b 1

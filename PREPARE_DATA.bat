@echo off
setlocal
call "%~dp0SETUP_ENV.bat"
"%ASSIGNMENT06_PYTHON%" tools\download_data.py
if errorlevel 1 exit /b 1
"%ASSIGNMENT06_PYTHON%" tools\prepare_data.py
if errorlevel 1 exit /b 1
"%ASSIGNMENT06_PYTHON%" tools\check_data.py
pause

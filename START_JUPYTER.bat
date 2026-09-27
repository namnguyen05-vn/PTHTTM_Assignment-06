@echo off
setlocal
call "%~dp0SETUP_ENV.bat"
"%ASSIGNMENT06_PYTHON%" -m ipykernel install --prefix "%~dp0runtime\jupyter" --name assignment06 --display-name "Python (Assignment 06)"
if errorlevel 1 exit /b 1
"%ASSIGNMENT06_PYTHON%" -m jupyterlab "%~dp0notebooks"

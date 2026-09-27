@echo off
setlocal
call "%~dp0SETUP_ENV.bat"
"%ASSIGNMENT06_PYTHON%" -m ipykernel install --prefix "%~dp0runtime\jupyter" --name assignment06 --display-name "Python (Assignment 06)"
if errorlevel 1 exit /b 1
"%ASSIGNMENT06_PYTHON%" tools/finalize.py
set "BUILD_STATUS=%ERRORLEVEL%"
if "%BUILD_STATUS%"=="0" echo Report generated. Visual review is still required.
pause
exit /b %BUILD_STATUS%

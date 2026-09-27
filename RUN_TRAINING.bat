@echo off
setlocal
call "%~dp0SETUP_ENV.bat"
"%ASSIGNMENT06_PYTHON%" run_experiments.py --resume
set "RUN_STATUS=%ERRORLEVEL%"
if "%RUN_STATUS%"=="0" echo All requested training runs are complete.
if "%RUN_STATUS%"=="75" echo Paused at a saved epoch. You may now sleep the computer.
if not "%RUN_STATUS%"=="0" if not "%RUN_STATUS%"=="75" echo Training stopped. Check the message above. Restart this file to resume the last saved epoch.
pause
exit /b %RUN_STATUS%

@echo off
cd /d "%~dp0"
if not defined ASSIGNMENT06_PYTHON (
  if exist "E:\PTHTTM\ASG_04\ASG_04_runtime\env\Scripts\python.exe" (
    set "ASSIGNMENT06_PYTHON=E:\PTHTTM\ASG_04\ASG_04_runtime\env\Scripts\python.exe"
  ) else (
    set "ASSIGNMENT06_PYTHON=python"
  )
)
if not defined ASG06_DATA_DIR for %%I in ("%~dp0..\ASG_06_data") do set "ASG06_DATA_DIR=%%~fI"
if not exist "%ASG06_DATA_DIR%\cache\tmp" mkdir "%ASG06_DATA_DIR%\cache\tmp"
set "TEMP=%ASG06_DATA_DIR%\cache\tmp"
set "TMP=%TEMP%"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PIP_CACHE_DIR=%ASG06_DATA_DIR%\cache\pip"
set "JUPYTER_CONFIG_DIR=%~dp0runtime\jupyter_config"
set "JUPYTER_RUNTIME_DIR=%~dp0runtime\jupyter_runtime"
set "JUPYTER_PATH=%~dp0runtime\jupyter\share\jupyter"
if not exist "%JUPYTER_RUNTIME_DIR%" mkdir "%JUPYTER_RUNTIME_DIR%"

@echo off
setlocal
cd /d "%~dp0"
echo pause>PAUSE_TRAINING
echo Pause requested. Wait for the training window to confirm the epoch is saved.
echo Do not sleep until the training process has stopped.
pause

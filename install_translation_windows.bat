@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" -Action Translation %*
set "MODEL_RESULT=%ERRORLEVEL%"
if not "%MODEL_RESULT%"=="0" echo Translation model preparation failed. Read the error above.
pause
exit /b %MODEL_RESULT%

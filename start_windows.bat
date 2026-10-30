@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if not exist "%~dp0work\temp" mkdir "%~dp0work\temp"
set "TEMP=%~dp0work\temp"
set "TMP=%~dp0work\temp"
if exist "%~dp0PrismInterview.exe" (
  "%~dp0PrismInterview.exe"
  goto done
)
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run install_windows.bat first, or start PrismInterview.exe from the portable package.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0run.py"
:done
set "APP_RESULT=%ERRORLEVEL%"
if not "%APP_RESULT%"=="0" (
  echo Application exited with an error. Code: %APP_RESULT%
  pause
)
exit /b %APP_RESULT%

@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo Prism Interview 0.4.1
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install.ps1" %*
set "INSTALL_RESULT=%ERRORLEVEL%"
echo.
if not "%INSTALL_RESULT%"=="0" echo Installation did not finish. Existing programs were not overwritten.
pause
exit /b %INSTALL_RESULT%

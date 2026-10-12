@echo off
rem Explicit platform launcher: enables the database and platform APIs.
rem Double-click this file, or pass options such as:
rem   start_platform.cmd -Port 9000
rem   start_platform.cmd -ResetDb
setlocal
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0run_platform.ps1" %*
echo.
echo Platform stopped. Press any key to close this window.
pause >nul

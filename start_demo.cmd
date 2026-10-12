@echo off
rem Double-click launcher for the Polymer-Agent closed-loop demo.
rem Starts offline replay and opens /auto. Install first: python -m pip install -e .
rem Power users can still pass args, e.g.:  start_demo.cmd -Port 9000 -Mode replay
setlocal
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0run_demo.ps1" %*
echo.
echo Server stopped. Press any key to close this window.
pause >nul

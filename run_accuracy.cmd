@echo off
setlocal
cd /d "%~dp0"
set "_MNE_FAKE_HOME_DIR=%CD%"
".venv-accuracy\Scripts\python.exe" motor_imagery_simple.py %*
exit /b %errorlevel%

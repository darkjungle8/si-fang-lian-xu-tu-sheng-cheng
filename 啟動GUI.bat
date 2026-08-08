@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%CD%"
python main.py
if errorlevel 1 pause

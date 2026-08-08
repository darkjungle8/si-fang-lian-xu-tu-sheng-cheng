@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%CD%"

set "PYEXE="
where py >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%I"
)
if not defined PYEXE set "PYEXE=python"

"%PYEXE%" -c "import tkinter" 2>nul
if errorlevel 1 (
  echo [錯誤] 這個 Python 沒有 tkinter：
  "%PYEXE%" -c "import sys; print(sys.executable)"
  echo 請改用官方 Python（含 tcl/tk），不要用 ComfyUI 的 python_embedded。
  pause
  exit /b 1
)

"%PYEXE%" main.py
if errorlevel 1 pause

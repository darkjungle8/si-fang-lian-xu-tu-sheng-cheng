@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%CD%"

REM Prefer a real Python with tkinter; avoid ComfyUI embedded python.
set "PYEXE="

where py >nul 2>&1
if not errorlevel 1 (
  for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%I"
)

if not defined PYEXE (
  for /f "delims=" %%I in ('where python 2^>nul') do (
    echo %%I | find /i "comfyui" >nul
    if errorlevel 1 (
      if not defined PYEXE set "PYEXE=%%I"
    )
  )
)

if not defined PYEXE set "PYEXE=python"

echo Using: %PYEXE%
"%PYEXE%" -c "import tkinter, sys; print('tkinter OK', sys.version)"
if errorlevel 1 (
  echo.
  echo [錯誤] 目前 Python 沒有 tkinter（常見於 ComfyUI 內建 Python）。
  echo 請安裝官方 Python 3.10+：https://www.python.org/downloads/windows/
  echo 安裝時務必勾選：
  echo   - Add python.exe to PATH
  echo   - tcl/tk and IDLE
  echo 裝完後重新開啟 CMD，再雙擊本腳本。
  pause
  exit /b 1
)

"%PYEXE%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [錯誤] 依賴安裝失敗
  pause
  exit /b 1
)

"%PYEXE%" main.py
if errorlevel 1 pause

@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%CD%"
set "VENV_DIR=%CD%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

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

echo 系統 Python: %PYEXE%
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

if not exist "%VENV_PY%" (
  echo.
  echo [資訊] 尚未找到虛擬環境，正在建立 .venv ...
  echo        套件只會裝進此資料夾，不會改動系統 Python。
  if exist "%VENV_DIR%" rd /s /q "%VENV_DIR%"
  "%PYEXE%" -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [錯誤] 建立虛擬環境失敗。請確認已安裝官方 Python 3.10+（含 venv 模組）。
    pause
    exit /b 1
  )
  echo [資訊] 虛擬環境已建立：%VENV_DIR%
)

echo Using venv: %VENV_PY%
"%VENV_PY%" -c "import tkinter, sys; print('venv tkinter OK', sys.version)"
if errorlevel 1 (
  echo [錯誤] 虛擬環境的 Python 沒有 tkinter，將重建 .venv
  rd /s /q "%VENV_DIR%"
  "%PYEXE%" -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [錯誤] 重建虛擬環境失敗。
    pause
    exit /b 1
  )
)

echo.
echo [資訊] 正在虛擬環境中安裝依賴（不會改到系統 Python）...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [錯誤] 依賴安裝失敗
  pause
  exit /b 1
)

"%VENV_PY%" main.py
if errorlevel 1 pause

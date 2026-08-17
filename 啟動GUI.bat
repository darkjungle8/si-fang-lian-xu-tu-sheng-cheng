@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%CD%"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [資訊] 尚未建立虛擬環境，改為一鍵建立 .venv、安裝依賴並啟動...
  echo        套件只會裝進 .venv，不會改動系統 Python。
  echo.
  call "%~dp0安裝依賴並啟動.bat"
  exit /b %ERRORLEVEL%
)

echo Using venv: %VENV_PY%
"%VENV_PY%" -c "import tkinter" 2>nul
if errorlevel 1 (
  echo [錯誤] 虛擬環境的 Python 沒有 tkinter：
  "%VENV_PY%" -c "import sys; print(sys.executable)"
  echo 請刪除 .venv 資料夾後，再雙擊「安裝依賴並啟動.bat」。
  pause
  exit /b 1
)

"%VENV_PY%" main.py
if errorlevel 1 pause

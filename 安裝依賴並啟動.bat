@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 四方連續圖 - 安裝依賴並啟動

echo ========================================
echo  專案目錄: %CD%
echo ========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo [錯誤] 找不到 python，請先安裝 Python 3.10+ 並勾選 Add to PATH
  pause
  exit /b 1
)

python --version
echo.

if not exist "app\gui.py" (
  echo [錯誤] 缺少 app\gui.py
  echo 請先在本目錄執行:
  echo   git pull origin main
  echo 或重新 clone:
  echo   git clone https://github.com/darkjungle8/si-fang-lian-xu-tu-sheng-cheng.git
  pause
  exit /b 1
)

echo [1/3] 升級 pip...
python -m pip install --upgrade pip
echo.
echo [2/3] 安裝 requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [錯誤] 依賴安裝失敗
  pause
  exit /b 1
)
echo.
echo [3/3] 驗證模組...
python -c "from app.gui import run_app; import PIL, numpy, customtkinter, cv2, openpyxl; print('OK: 依賴與 app.gui 皆可用')"
if errorlevel 1 (
  echo [錯誤] 驗證失敗
  pause
  exit /b 1
)

echo.
echo 啟動 GUI...
python main.py
if errorlevel 1 pause

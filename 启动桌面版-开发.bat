@echo off
rem Grain - 桌面版「开发模式」：改 .py 自动重启后端，改 static/ 自动刷新窗口
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul || (echo 未找到 Python，请先安装 Python 3.10+ 并加入 PATH & pause & exit /b 1)

python -c "import webview" 2>nul || (
  echo 未安装 pywebview：请运行  pip install pywebview
  pause
  exit /b 1
)

python scripts\fetch_assets.py
python desktop_pywebview.py --dev
pause

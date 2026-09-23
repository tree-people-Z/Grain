@echo off
rem Grain - 桌面版（pywebview 原生窗口 + Python 后端，无需 Rust）
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul || (echo 未找到 Python，请先安装 Python 3.10+ 并加入 PATH & pause & exit /b 1)

python -c "import webview" 2>nul || (
  echo 未安装 pywebview：请运行  pip install pywebview
  echo 或者改用 启动网页版.bat（浏览器版，无需此依赖）
  pause
  exit /b 1
)

python desktop_pywebview.py
pause

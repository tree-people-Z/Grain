@echo off
rem Grain - 网页版启动器
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul || (echo 未找到 Python，请先安装 Python 3.10+ & pause & exit /b 1)
start "" http://127.0.0.1:8770
python server.py --port 8770
pause

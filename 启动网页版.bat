@echo off
setlocal
rem Grain portable web launcher
cd /d "%~dp0"

rem Prefer the Python runtime bundled with this folder.
set "GRAIN_PYTHON=%~dp0runtime\python\python.exe"
if not exist "%GRAIN_PYTHON%" set "GRAIN_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%GRAIN_PYTHON%" (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [Grain] Bundled Python was not found.
    echo [Grain] Please keep the runtime\python folder, or install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
  )
  set "GRAIN_PYTHON=python"
)

if not exist "%~dp0static\index.html" (
  echo [Grain] static\index.html was not found. The package is incomplete.
  pause
  exit /b 1
)
if not exist "%~dp0runtime\ffmpeg\bin\ffmpeg.exe" echo [Grain] Warning: bundled ffmpeg is missing; media import may fail.
if not exist "%~dp0models\hf\hub" echo [Grain] Warning: bundled models are missing; auto detection may need internet access and a HuggingFace token.

"%GRAIN_PYTHON%" -c "import torch, pyannote.audio; assert torch.cuda.is_available()" >nul 2>nul
if errorlevel 1 echo [Grain] GPU unavailable in selected Python; auto detection may run on CPU.
rem Open the browser only after the server has had a moment to come up.
start "" cmd /c "timeout /t 2 /nobreak >nul & start "" http://127.0.0.1:8770"
"%GRAIN_PYTHON%" server.py --port 8770
pause

@echo off
setlocal EnableExtensions

set "ENV_NAME=neuralink"
set "PYTHON_VERSION=3.10"
set "OLLAMA_MODEL=gemma3:4b"
set "ENTRYPOINT=main.py"
set "PROJECT_DIR=%~dp0"

echo ==========================================
echo NeuroAgent First-Time Setup
echo ==========================================
echo.

REM Step 1 - Check Ollama
where ollama >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Ollama is not installed.
  echo Install it from: https://ollama.com/download
  pause
  exit /b 1
)

REM Check Conda availability
where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Conda is not installed or not available in PATH.
  echo Install Miniconda or Anaconda first.
  pause
  exit /b 1
)

REM Step 2 - Start Ollama server in background (best effort)
echo [INFO] Starting Ollama server...
start "" /MIN cmd /c "ollama serve >nul 2>&1"
timeout /t 3 /nobreak >nul

REM Step 3 - Pull the model
echo [INFO] Pulling Ollama model: %OLLAMA_MODEL%
ollama pull %OLLAMA_MODEL%
if errorlevel 1 (
  echo [ERROR] Failed to pull model %OLLAMA_MODEL%.
  pause
  exit /b 1
)

REM Step 4 - Create conda env if missing
echo [INFO] Checking conda environment: %ENV_NAME%
call conda run -n %ENV_NAME% python -V >nul 2>&1
if errorlevel 1 (
  echo [INFO] Creating conda environment %ENV_NAME% with Python %PYTHON_VERSION%...
  call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
  if errorlevel 1 (
    echo [ERROR] Failed to create conda environment.
    pause
    exit /b 1
  )
) else (
  echo [INFO] Conda environment %ENV_NAME% already exists.
)

REM Step 5 - Install Python dependencies
if exist "%PROJECT_DIR%requirements.txt" (
  echo [INFO] Upgrading pip...
  call conda run -n %ENV_NAME% python -m pip install --upgrade pip
  if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
  )

  echo [INFO] Installing dependencies from requirements.txt...
  call conda run -n %ENV_NAME% python -m pip install -r "%PROJECT_DIR%requirements.txt"
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
  )
) else (
  echo [WARN] requirements.txt not found in project root.
  echo [WARN] Skipping dependency installation.
)

REM Step 6 - Create launcher batch file
echo [INFO] Creating NeuroAgent.bat launcher...
(
  echo @echo off
  echo setlocal
  echo cd /d "%%~dp0"
  echo call conda run -n %ENV_NAME% python %ENTRYPOINT%
  echo pause
) > "%PROJECT_DIR%NeuroAgent.bat"

if not exist "%PROJECT_DIR%%ENTRYPOINT%" (
  echo [WARN] Entry point "%ENTRYPOINT%" was not found in project root.
  echo [WARN] Edit NeuroAgent.bat and update the python command if needed.
)

echo.
echo ==========================================
echo Setup complete.
echo Run NeuroAgent via: NeuroAgent.bat
echo ==========================================
echo.
echo Manual prerequisites:
echo - Ollama: https://ollama.com/download
echo - Miniconda or Anaconda
echo - CUDA drivers (for GPU support)
echo - OmniParser weights (YOLO + Florence2)
echo.
pause
exit /b 0

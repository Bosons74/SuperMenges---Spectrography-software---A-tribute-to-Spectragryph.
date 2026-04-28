@echo off
setlocal EnableExtensions
chcp 65001 >nul
title OpenSpectra Workbench V1.9 - Direct Launcher
cd /d "%~dp0"

echo.
echo ============================================================
echo   OpenSpectra Workbench V1.9 - Direct no-venv launcher
echo ============================================================
echo.

echo This fallback does NOT create .venv. It installs dependencies into your user Python environment.
echo Use it only if the normal launcher fails at ensurepip/venv creation.
echo.

set "PYTHON_CMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYTHON_CMD (
        py -%%V -c "import sys; print(sys.version)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=py -%%V"
    )
)
if not defined PYTHON_CMD (
    python -c "import sys; print(sys.version)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    echo [ERROR] Python was not found. Install Python 3.11 or 3.12 and tick Add python.exe to PATH.
    pause
    exit /b 1
)

echo Using:
%PYTHON_CMD% -c "import sys; print(sys.executable); print(sys.version)"
echo.

echo Installing dependencies with --user...
%PYTHON_CMD% -m pip install --user --upgrade pip setuptools wheel
%PYTHON_CMD% -m pip install --user -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo Launching OpenSpectra Workbench...
%PYTHON_CMD% -m openspectra_workbench.app
pause
endlocal

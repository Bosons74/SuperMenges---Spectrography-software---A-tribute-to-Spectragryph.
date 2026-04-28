@echo off
setlocal EnableExtensions
chcp 65001 >nul
title OpenSpectra Workbench V1.9 - Fixed Launcher

cd /d "%~dp0"

echo.
echo ============================================================
echo   OpenSpectra Workbench V1.9 - Fixed Windows launcher
echo ============================================================
echo.

echo [0/4] Selecting a compatible Python...
set "PYTHON_CMD="

for %%V in (3.12 3.11 3.10) do (
    if not defined PYTHON_CMD (
        py -%%V -c "import sys, venv, ensurepip; print(sys.version)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=py -%%V"
    )
)

if not defined PYTHON_CMD (
    python -c "import sys, venv, ensurepip; print(sys.version)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo [ERROR] No compatible Python found.
    echo.
    echo Install Python 3.11 or 3.12 from python.org.
    echo During installation, tick: Add python.exe to PATH
    echo Then close this window and run this launcher again.
    echo.
    pause
    exit /b 1
)

echo Using:
%PYTHON_CMD% -c "import sys; print(sys.executable); print(sys.version)"
echo.

if exist ".venv" (
    if not exist ".venv\Scripts\python.exe" (
        echo Removing incomplete .venv folder...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating local Python virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not create virtual environment.
        echo.
        echo Fix recommended:
        echo 1. Move this whole folder to C:\OpenSpectra_Workbench_V1.9
        echo 2. Delete the .venv folder if it exists
        echo 3. Run this file again
        echo.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Local virtual environment already exists.
)

echo [2/4] Upgrading pip / setuptools / wheel...
".venv\Scripts\python.exe" -m ensurepip --upgrade
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo.
    echo [ERROR] pip upgrade failed.
    echo Try deleting .venv and running again, preferably from C:\OpenSpectra_Workbench_V1.9
    echo.
    pause
    exit /b 1
)

echo [3/4] Installing / checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo Check your internet connection. If it still fails, install Python 3.11/3.12.
    echo.
    pause
    exit /b 1
)

echo [4/4] Launching OpenSpectra Workbench...
echo.
".venv\Scripts\python.exe" -m openspectra_workbench.app

if errorlevel 1 (
    echo.
    echo [ERROR] The application closed with an error.
    echo.
    pause
)
endlocal

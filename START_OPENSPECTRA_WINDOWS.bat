@echo off
setlocal
title OpenSpectra Workbench V1.9

cd /d "%~dp0"

echo.
echo ============================================================
echo   OpenSpectra Workbench V1.9 - Windows one-click launcher
echo ============================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON_CMD=py -3
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set PYTHON_CMD=python
    ) else (
        echo [ERROR] Python was not found.
        echo.
        echo Install Python 3.10 or newer from:
        echo https://www.python.org/downloads/windows/
        echo.
        echo IMPORTANT: tick "Add python.exe to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating local Python virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Local virtual environment already exists.
)

echo [2/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/4] Installing / checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo Check your internet connection, then run this launcher again.
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

@echo off
setlocal
title OpenSpectra Workbench V1.9 Fast Launcher
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m openspectra_workbench.app
) else (
    call START_OPENSPECTRA_WINDOWS.bat
)
endlocal

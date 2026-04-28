$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host ""
Write-Host "============================================================"
Write-Host "  OpenSpectra Workbench V1.9 - Windows PowerShell launcher"
Write-Host "============================================================"
Write-Host ""

$python = $null
try {
    py -3 --version | Out-Null
    $python = "py -3"
} catch {
    try {
        python --version | Out-Null
        $python = "python"
    } catch {
        Write-Host "[ERROR] Python was not found."
        Write-Host "Install Python 3.10+ and tick 'Add python.exe to PATH'."
        Read-Host "Press Enter to exit"
        exit 1
    }
}

if (!(Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[1/4] Creating local Python virtual environment..."
    Invoke-Expression "$python -m venv .venv"
} else {
    Write-Host "[1/4] Local virtual environment already exists."
}

Write-Host "[2/4] Upgrading pip..."
& ".venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "[3/4] Installing / checking dependencies..."
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "[4/4] Launching OpenSpectra Workbench..."
& ".venv\Scripts\python.exe" -m openspectra_workbench.app

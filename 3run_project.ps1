# Uruchamiaj w PowerShell (Windows). Każdy proces w osobnym oknie z własnym venv.

$ErrorActionPreference = 'Stop'

# Ścieżki
$root       = Convert-Path $PSScriptRoot
$apiVenv    = Join-Path $root '.api_venv'
$watusRoot  = Join-Path $root 'watus_project'
$watusVenv  = Join-Path $watusRoot '.watus_venv'

function Ensure-Venv($venvPath) {
    if (-not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
        python -m venv $venvPath
    }
}

function Start-PwshWindow([string]$command) {
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList '-NoExit','-NoProfile','-ExecutionPolicy','Bypass','-Command', $command `
        -WindowStyle Normal
}

# Tworzenie venv (jeśli brak)
Ensure-Venv $apiVenv
Ensure-Venv $watusVenv

# Komendy dla osobnych okien
$apiCmd = @"
cd '$root';
. '$apiVenv\Scripts\Activate.ps1';
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
"@.Trim()

$reporterCmd = @"
cd '$watusRoot';
. '$watusVenv\Scripts\Activate.ps1';
python reporter.py
"@.Trim()

$cameraCmd = @"
cd '$watusRoot';
. '$watusVenv\Scripts\Activate.ps1';
python camera_runner.py --jsonl './camera.jsonl' --device 0
"@.Trim()

$watusCmd = @"
cd '$watusRoot';
. '$watusVenv\Scripts\Activate.ps1';
python watus.py
"@.Trim()

# Start w osobnych terminalach
Start-PwshWindow $apiCmd
Start-PwshWindow $reporterCmd
Start-PwshWindow $cameraCmd
Start-PwshWindow $watusCmd
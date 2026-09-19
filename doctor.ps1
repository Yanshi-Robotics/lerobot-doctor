# LeRobot Doctor launcher for Windows 10/11 (PowerShell).
#   irm https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/v0.1.3/doctor.ps1 | iex
# or double-click doctor.bat from an unpacked zip.
#
# Three jobs only: make sure `uv` exists, make sure Python 3.12 exists, run lerobot_doctor.py.
$ErrorActionPreference = "Stop"

$DoctorTag = if ($env:DOCTOR_TAG) { $env:DOCTOR_TAG } else { "v0.1.3" }   # must equal the tag this file is published under
$Raw = "https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/$DoctorTag"
$HomeDir = Join-Path $HOME "lerobot-doctor"
New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null

# UTF-8 so the bilingual output and the check marks render
try { chcp 65001 | Out-Null } catch {}
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# 1. the script itself: next to this launcher (zip) or downloaded (one-liner)
$Here = if ($PSScriptRoot) { $PSScriptRoot } else { "" }
if ($Here -and (Test-Path (Join-Path $Here "lerobot_doctor.py"))) {
    $Script = Join-Path $Here "lerobot_doctor.py"
} else {
    $Script = Join-Path $HomeDir "lerobot_doctor.py"
    Write-Host "下载体检程序 | downloading lerobot_doctor.py ($DoctorTag)"
    Invoke-WebRequest -Uri "$Raw/lerobot_doctor.py" -OutFile $Script -UseBasicParsing
}

# 2. uv
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    $candidate = Join-Path $HOME ".local\bin\uv.exe"
    if (-not (Test-Path $candidate)) {
        Write-Host "安装 uv（Python 环境管理器，约 30 MB） | installing uv (~30 MB)"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    $env:Path = (Join-Path $HOME ".local\bin") + ";" + $env:Path
}

# 3. Python 3.12
Write-Host "准备 Python 3.12 | preparing Python 3.12"
uv python install 3.12 --quiet

# 4. run
uv run --python 3.12 --no-project $Script @args
exit $LASTEXITCODE

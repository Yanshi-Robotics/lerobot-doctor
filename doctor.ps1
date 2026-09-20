# LeRobot Doctor launcher for Windows 10/11 (PowerShell).
#   irm https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/main/doctor.ps1 | iex
# or double-click doctor.bat from an unpacked zip.
#
# Three jobs only: make sure `uv` exists, make sure Python 3.12 exists, run lerobot_doctor.py.
#
# Three rules this file lives by (each checked against a real PowerShell before it was written):
# 1. Pure ASCII. Windows PowerShell 5.1 reads a .ps1 without a BOM in the ANSI code page, which
#    turns Chinese literals into mojibake, and a BOM instead breaks `irm | iex` ("The term
#    '<BOM>...' is not recognized"). Chinese is therefore written as \u escapes; the readable
#    text lives in tests/test_doctor.py (PS1_ZH), which checks every escape against it.
# 2. No `exit` at the top level. Under `irm | iex` this file runs inside the user's own session,
#    so `exit` would close their window with everything on it. Only a file-mode start (doctor.bat,
#    right-click "Run with PowerShell") exits, and it waits for Enter first when the window would close.
# 3. `uv run` stays at the top level. There the program inherits the console: progress bars, the
#    Enter prompts, a real tty. Inside a function its output would become the function's return value.

function Get-Zh([string]$Escaped) { [regex]::Unescape($Escaped) }

function Initialize-LeRobotDoctor {
    # Steps 1-3. Returns the path of lerobot_doctor.py, or $null after saying what went wrong.
    # Whatever a function lets fall into the pipeline becomes its return value, so every command in
    # here either goes to the host (Write-Host, Out-Host) or is discarded ($null = ...).
    $ErrorActionPreference = "Stop"   # function scope: does not leak into the user's session under iex
    $DoctorTag = if ($env:DOCTOR_TAG) { $env:DOCTOR_TAG } else { "v0.1.5" }   # must equal the tag this file is published under
    $Raw = "https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/$DoctorTag"
    $Issues = "https://github.com/Yanshi-Robotics/lerobot-doctor/issues/new"
    $HomeDir = Join-Path $HOME "lerobot-doctor"
    $step = "start"
    try {
        # UTF-8 so the bilingual output and the check marks render
        try { chcp 65001 | Out-Null } catch {}
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $env:PYTHONIOENCODING = "utf-8"
        $env:PYTHONUTF8 = "1"
        # GitHub needs TLS 1.2; Windows PowerShell 5.1 on older Windows 10 builds does not offer it by default
        try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}
        $null = New-Item -ItemType Directory -Force -Path $HomeDir

        # 1. the script itself: next to this launcher (zip) or downloaded (one-liner)
        $step = "download lerobot_doctor.py ($DoctorTag)"
        $Here = if ($PSScriptRoot) { $PSScriptRoot } else { "" }
        if ($Here -and (Test-Path (Join-Path $Here "lerobot_doctor.py"))) {
            $Script = Join-Path $Here "lerobot_doctor.py"
        } else {
            $Script = Join-Path $HomeDir "lerobot_doctor.py"
            Write-Host ((Get-Zh '\u4e0b\u8f7d\u4f53\u68c0\u7a0b\u5e8f') + " | downloading lerobot_doctor.py ($DoctorTag)")   # zh: downloading the doctor program
            $null = Invoke-WebRequest -Uri "$Raw/lerobot_doctor.py" -OutFile $Script -UseBasicParsing
        }

        # 2. uv. Its installer runs in a child PowerShell: it calls `exit 1` on failure, which inside
        #    this session would close the user's window (rule 2 above).
        $step = "install uv"
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uv) {
            $candidate = Join-Path $HOME ".local\bin\uv.exe"
            if (-not (Test-Path $candidate)) {
                Write-Host ((Get-Zh '\u5b89\u88c5 uv\uff08Python \u73af\u5883\u7ba1\u7406\u5668\uff0c\u7ea6 30 MB\uff09') + " | installing uv (~30 MB)")   # zh: installing uv (Python environment manager, ~30 MB)
                powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" | Out-Host
                if ($LASTEXITCODE -ne 0) { throw "the uv installer exited with code $LASTEXITCODE (see its output above)" }
            }
            $env:Path = (Join-Path $HOME ".local\bin") + ";" + $env:Path
        }

        # 3. Python 3.12
        $step = "install Python 3.12"
        Write-Host ((Get-Zh '\u51c6\u5907 Python 3.12') + " | preparing Python 3.12")   # zh: preparing Python 3.12
        uv python install 3.12 --quiet | Out-Host
        return $Script
    } catch {
        Write-Host ""
        Write-Host ((Get-Zh '\u542f\u52a8\u5668\u51fa\u9519\u4e86\uff0c\u4f53\u68c0\u6ca1\u6709\u5f00\u59cb') + " | The launcher failed; the check did not start") -ForegroundColor Red   # zh: the launcher failed, the check did not start
        Write-Host ("  " + (Get-Zh '\u5361\u5728') + " | step : $step") -ForegroundColor Red   # zh: stuck at
        Write-Host ("  " + (Get-Zh '\u9519\u8bef') + " | error: " + $_.Exception.Message) -ForegroundColor Red   # zh: error
        Write-Host ("  " + (Get-Zh '\u6700\u5e38\u89c1\u662f\u7f51\u7edc\u95ee\u9898\uff1a\u91cd\u8bd5\u4e00\u6b21\uff1b\u4e0b\u8f7d\u6162\u53ef\u4ee5\u628a HF_ENDPOINT \u8bbe\u6210\u955c\u50cf') + " | Most often a network problem: try again; if downloads are slow, point HF_ENDPOINT at a mirror")   # zh: most often a network problem: try again; if downloads are slow, set HF_ENDPOINT to a mirror
        Write-Host ("  " + (Get-Zh '\u62a5 issue \u8bf7\u628a\u8fd9\u4e2a\u7a97\u53e3\u622a\u56fe\u8d34\u4e0a\u53bb') + " | to report it, paste a screenshot of this window at: $Issues")   # zh: to report an issue, paste a screenshot of this window
        return $null
    }
}

$launcherWasSet = $false
if (-not $env:DOCTOR_LAUNCHER) { $env:DOCTOR_LAUNCHER = "ps1"; $launcherWasSet = $true }   # tells the program a launcher is around it (doctor.bat sets "bat")
$code = 1
try {
    $Script = Initialize-LeRobotDoctor
    if ($Script) {
        # 4. run (rule 3). From here on the Python program handles its own errors: crash log + message.
        uv run --python 3.12 --no-project $Script @args
        $code = $LASTEXITCODE
        # 0 = done; 1 = a hard floor or the install failed (the program said so); 130 = Ctrl-C;
        # 70 = the program crashed and printed its own crash box. Anything else: uv never got it running.
        if ($code -notin 0, 1, 70, 130) {
            Write-Host ((Get-Zh 'uv \u6ca1\u80fd\u628a\u4f53\u68c0\u7a0b\u5e8f\u8dd1\u8d77\u6765\uff0c\u770b\u4e0a\u9762\u7684\u8f93\u51fa\uff08\u9000\u51fa\u7801 ' + $code + '\uff09') + " | uv could not start the program, see the output above (exit code $code)") -ForegroundColor Yellow   # zh: uv could not start the doctor program, see the output above (exit code N)
        }
    }
} finally {
    if ($launcherWasSet) { Remove-Item Env:DOCTOR_LAUNCHER -ErrorAction SilentlyContinue }
}
if ($PSScriptRoot) {
    # Started as a file (doctor.bat, or right-click "Run with PowerShell"): the window closes when this
    # script ends. doctor.bat pauses by itself; anyone else gets the pause here.
    if ($env:DOCTOR_LAUNCHER -ne "bat") { $null = Read-Host ((Get-Zh '\u6309\u56de\u8f66\u5173\u95ed\u7a97\u53e3') + " | Press Enter to close") }   # zh: press Enter to close the window
    exit $code
}
# Under `irm | iex` the user's own window stays open; nothing more to do.

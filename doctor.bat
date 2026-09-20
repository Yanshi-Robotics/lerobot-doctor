@echo off
rem LeRobot Doctor: double-click this file on Windows. It hands over to doctor.ps1.
chcp 65001 >nul
if not exist "%~dp0doctor.ps1" (
    echo.
    echo 旁边找不到 doctor.ps1。请先把整个 ZIP 解压，再双击解压出来的文件夹里的 doctor.bat。
    echo doctor.ps1 is not next to this file. Unzip the whole ZIP first, then double-click doctor.bat inside the unzipped folder.
    echo.
    echo 按任意键关闭窗口 ^| Press any key to close.
    pause >nul
    exit /b 1
)
set "DOCTOR_LAUNCHER=bat"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0doctor.ps1" %*
set RC=%ERRORLEVEL%
echo.
if "%RC%"=="0" (
    echo 体检结束。按任意键关闭窗口 ^| Done. Press any key to close.
) else (
    echo 体检没有正常结束（退出码 %RC%），看上面的输出。按任意键关闭窗口 ^| The check did not finish normally ^(exit code %RC%^), see the output above. Press any key to close.
)
pause >nul
exit /b %RC%

@echo off
rem LeRobot Doctor: double-click this file on Windows. It hands over to doctor.ps1.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0doctor.ps1" %*
echo.
echo 体检结束。按任意键关闭窗口 ^| Done. Press any key to close.
pause >nul

@echo off
set "PYW=%USERPROFILE%\.notetaker\venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
    echo Notetaker is not installed yet. Run:  powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    pause
    exit /b 1
)
start "" "%PYW%" "%~dp0notetaker.pyw"

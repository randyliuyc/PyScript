@echo off
rem Silent launcher for Windows Task Scheduler:
rem   - no browser, no pause (nobody is watching the console)
rem   - output goes to logs\page_server.log
rem The service itself reads host/port from settings.json.
setlocal
cd /d "%~dp0"

if not exist "logs" mkdir "logs"
set "LOG=logs\page_server.log"

rem Prefer the project venv, then the py launcher, then python on PATH.
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if errorlevel 1 (
        set "PY=python"
    ) else (
        set "PY=py -3"
    )
)

%PY% -c "import numpy, scipy" >nul 2>nul
if errorlevel 1 (
    echo [%date% %time%] Missing dependencies numpy / scipy >> "%LOG%"
    echo [%date% %time%] Install with: %PY% -m pip install numpy scipy >> "%LOG%"
    exit /b 1
)

rem Line-buffered output so the log shows the startup banner right away.
set "PYTHONUNBUFFERED=1"
echo. >> "%LOG%"
echo [%date% %time%] starting page_server %* >> "%LOG%"
%PY% page_server.py --no-open %* >> "%LOG%" 2>&1
echo [%date% %time%] page_server exited, code %errorlevel% >> "%LOG%"
endlocal

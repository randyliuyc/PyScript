@echo off
setlocal
title page_server
cd /d "%~dp0"

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

rem Fail early with a readable message instead of a traceback.
%PY% -c "import numpy, scipy" >nul 2>nul
if errorlevel 1 (
    echo Missing dependencies: numpy / scipy
    echo Install with:  %PY% -m pip install numpy scipy
    echo.
    pause
    exit /b 1
)

echo Starting page_server (host/port from settings.json; default 0.0.0.0:8765)
echo Press Ctrl+C to stop. Extra arguments are passed through, e.g.
echo     start_page_server.bat --port 8766 --no-open
echo.
%PY% page_server.py %*

echo.
echo page_server exited.
pause
endlocal

@echo off
setlocal enabledelayedexpansion

title Rotocraft Studio - FilmCraft AI Rotoscoping

echo =======================================================
echo   Rotocraft Studio: AI Rotoscoping for FilmCraft
echo =======================================================
echo.

cd /d "%~dp0"

set "PYTHON_EXE="

:: 1. Check if local venv exists
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "..\..\tools\rotocraft\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\..\tools\rotocraft\.venv\Scripts\python.exe"
)

:: 2. If no venv, try to find uv or python to set it up
if "%PYTHON_EXE%"=="" (
    echo [Rotocraft] Setting up environment for first run...
    where uv >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [Rotocraft] Creating virtual environment with uv...
        uv venv .venv --python 3.12
        uv pip install --python .venv\Scripts\python.exe -e .
        set "PYTHON_EXE=.venv\Scripts\python.exe"
    ) else (
        where py >nul 2>nul
        if %ERRORLEVEL% equ 0 (
            py -3 -m venv .venv
            .venv\Scripts\python.exe -m pip install -e .
            set "PYTHON_EXE=.venv\Scripts\python.exe"
        ) else (
            where python >nul 2>nul
            if %ERRORLEVEL% equ 0 (
                python -m venv .venv
                .venv\Scripts\python.exe -m pip install -e .
                set "PYTHON_EXE=.venv\Scripts\python.exe"
            )
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python 3.10+ was not found on your system.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [Rotocraft] Starting Rotocraft Studio on http://127.0.0.1:8765...
echo [Rotocraft] Your browser will open automatically.
echo Press Ctrl+C in this window to stop the studio.
echo.

"%PYTHON_EXE%" -m rotocraft.cli app --port 8765

pause
